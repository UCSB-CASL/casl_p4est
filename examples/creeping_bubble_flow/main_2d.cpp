// p4est library
#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_poisson_jump_faces_xgfm.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_poisson_jump_faces_xgfm.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;


static const string extra_info =
    "This main file tests the capability of the xgfm face jump solver to capture an interface jump in tangential stress when considering\n"
    "solenoidal vector fields, sampled at the faces. This is an application of most relevance for incompressible two-phase flows.\n"
    "Given the working hypotheses (solenoidal fields + symmetric stress tensor coupling velocity components across the interface),\n"
    "we cannot simply consider a manufactured solution. We consider the steady state solution of a spherical bubble driven by an \n"
    "interface-defined Marangoni force (only, no gravity). We consider the steady state solution of the Stokes flow.\n\n"
    "We consider a spherical bubble of radius r0, negative domain inside, positive domain outside. The fluid viscosities inside and\n"
    "outside the bubble are mu_m and mu_p, respectively. The surface tension is not uniform and has a nonzero background z-gradient,\n"
    "i.e., \\gamma = \\gamma_0*(1 + \\beta*z/r0). In a referential centered with the bubble, the Stokes flow solution is (in spherical\n"
    "coordinates):\n"
    "In the positive domain (outside the bubble),\n"
    "   u_r^+\t =  u_{\\infty}(1 - (r0/r)^3)\\cos(\\theta),\n"
    "   u_{\\theta}^+\t = -u_{\\infty}(1 + 0.5*(r0/r)^3)\\sin(\\theta),\n"
    "   u_{\\phi}^+\t = 0,\n"
    "   p^{+}\t = 0.\n"
    "In the negative domain (inside the bubble),\n"
    "   u_r^-\t =  1.5 u_{\\infty}((r/r0)^2 - 1)\\cos(\\theta),\n"
    "   u_{\\theta}^-\t = -1.5 u_{\\infty}(2(r/r0)^2 - 1)\\sin(\\theta),\n"
    "   u_{\\phi}^-\t = 0,\n"
    "   p^{-}\t = 2 gamma_0/r0 + 15 \\mu_m u_{\\infty}r cos(\\theta)/r0^2.\n"
    "where u_{\\infty} = 2\\gamma_0 \\beta/(3 (3\\mu_m + 2\\mu_p))"
    "\n"
    "We solve for -\\mu\\nabla^{2} \\vect{u} = -\\nabla p : we feed the face jump solver with the (known) pressure gradient as \n"
    "a right-hand side. No mass transfer across the interface is allowed, i.e.,\n"
    "   \\left[ \\vect{u}\\cdot \\vect{n} \\right] = 0.0,\n"
    "   \\left[ \\left(I - \\vect{n}\\vect{n}\\right) \\cdot \\mu \\left(\\nabla \\vect{u} + \\left(\\nabla \\vect{u}\\right)^{T} \\right) \\cdot \\vect{n} \\right] = -\\left(I - \\vect{n}\\vect{n}\\right) \\cdot \\nabla \\gamma, \n"
    "and we imposed Dirichlet boundary conditions on the boundaries of the computational domain (we are interested in the interface stress\n"
    "conditions most importantly, we do not want to pollute the analysis with border effects)."
    "The accuracy of the solution (and of its extrapolation) is checked in infinity norm therefater.\n"
    "Developer: Raphael Egan (Fall 2020)";

static const double default_length = 2.5;
static const double default_r0 = 0.5;
static const double uniform_band_factor = 0.15;

static const int default_lmin   = 3;
static const int default_lmax   = 5;
static const int default_ntree  = 1;
static const double default_mu_m    = 0.1;
static const double default_mu_p    = 0.5; // --> different mu_m and mu_p show the relevance of this analysis!
static const double default_gamma_0 = 0.075;
static const double default_beta    = -2.0/15.0;
static const bool default_voro_fly  = false;
static jump_solver_tag default_solver = xGFM;
static const uint default_ngrids    = 3;
static const bool default_save_vtk  = false;
static const bool default_print     = false;
static const bool default_subrefinement = false;
static const bool default_use_second_order_theta = false;
static const bool default_extrapolation = false;
const interpolation_method default_interp_method_phi = linear;
const double default_extrapolation_band_check = 3.0;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/creeping_bubble_flow";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/creeping_bubble_flow";
#else
const string default_work_folder = "/home/regan/workspace/projects/creeping_bubble_flow";
#endif

std::istream& operator>> (std::istream& is, jump_solver_tag& solver_to_test)
{
  std::string str;
  is >> str;

  std::vector<size_t> substr_found_at;
  case_insensitive_find_substr_in_str(str, "GFM", substr_found_at);
  for (size_t k = 0; k < substr_found_at.size(); ++k)
    if (substr_found_at[k] == 0 || !case_insenstive_char_compare(str[substr_found_at[k] - 1], 'x')) // make sure it's 'GFM' and not 'xGFM'
    {
      solver_to_test = GFM;
      return is;
    }

  case_insensitive_find_substr_in_str(str, "xGFM", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver_to_test = xGFM;
    return is;
  }

//  case_insensitive_find_substr_in_str(str, "FV", substr_found_at);
//  if(substr_found_at.size() > 0)
//    solvers_to_test.push_back(FV);
  return is;
}


class LEVEL_SET : public CF_DIM {
  const double xyz_c[P4EST_DIM];
  const double bubble_radius;
public:
  LEVEL_SET(const double *xyz_min, const double *xyz_max, const double& bubble_radius_)
    : xyz_c{DIM(0.5*(xyz_min[0] + xyz_max[0]), 0.5*(xyz_min[1] + xyz_max[1]), 0.5*(xyz_min[2] + xyz_max[2]))},
      bubble_radius(bubble_radius_)
  {
    lip = 1.2;
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    return ABSD(x - xyz_c[0], y - xyz_c[1], z - xyz_c[2]) - bubble_radius;
  }
  double operator()(const double *xyz) const
  {
    return operator()(DIM(xyz[0], xyz[1], xyz[2]));
  }
};

class EXACT_SOLUTION{
  double mu_minus, mu_plus;
  double gamma_0_beta;
  double r0;

  inline double u_infty() const
  {
    return 2.0*gamma_0_beta/(3.0*(3.0*mu_minus + 2.0*mu_plus));
  }

  inline double theta(const double *xyz) const
  {
    if(rr(xyz) < EPS*r0)
      return 0.0;

    return acos(xyz[2]/rr(xyz));
  }

  inline double phi(const double *xyz) const
  {
    if(rr(xyz) < EPS*r0 || (fabs(xyz[0]) < EPS*r0 && fabs(xyz[1]) < EPS*r0))
      return 0.0;

    return atan2(xyz[1], xyz[0]);
  }

  inline double rr(const double *xyz) const
  {
    return ABSD(xyz[0], xyz[1], xyz[2]);
  }

  inline double radial_solution_plus(const double *xyz) const
  {
    return u_infty()*(1.0 - pow(r0/rr(xyz), 3.0))*cos(theta(xyz));
  }

  inline double inclination_solution_plus(const double *xyz) const
  {
    return -u_infty()*(1.0 + 0.5*pow(r0/rr(xyz), 3.0))*sin(theta(xyz));
  }

  inline double radial_solution_minus(const double *xyz) const
  {
    return 1.5*u_infty()*(SQR(rr(xyz)/r0) - 1.0)*cos(theta(xyz));
  }

  inline double inclination_solution_minus(const double *xyz) const
  {
    return -1.5*u_infty()*(2.0*SQR(rr(xyz)/r0) - 1.0)*sin(theta(xyz));
  }

public:

  inline double solution(const char& sign, const u_char& comp, const double* xyz) const
  {
    const double ur     = (sign < 0 ? radial_solution_minus(xyz)      : radial_solution_plus(xyz));
    const double utheta = (sign < 0 ? inclination_solution_minus(xyz) : inclination_solution_plus(xyz));
    switch (comp) {
    case dir::x:
      return ur*sin(theta(xyz))*cos(phi(xyz)) + utheta*cos(theta(xyz))*cos(phi(xyz));
      break;
    case dir::y:
      return ur*sin(theta(xyz))*sin(phi(xyz)) + utheta*cos(theta(xyz))*sin(phi(xyz));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return ur*cos(theta(xyz)) - utheta*sin(theta(xyz));
      break;
#endif
    default:
      break;
    }
    return NAN;
  }

  inline double grad_p(const char& sign, const u_char& comp, const double*) const
  {
    if(sign > 0)
      return 0.0;

    switch (comp) {
    case dir::x:
    case dir::y:
      return 0.0;
      break;
#ifdef P4_TO_P8
    case dir::z:
      return 15.0*mu_minus*u_infty()/SQR(r0);
      break;
#endif
    default:
      break;
    }
    return NAN;
  }


  EXACT_SOLUTION() {}

  inline void set(const double& mu_minus_, const double& mu_plus_, const double& gamma_0_beta_, const double& bubble_radius)
  {
    mu_minus = mu_minus_;
    mu_plus = mu_plus_;
    gamma_0_beta = gamma_0_beta_;
    r0 = bubble_radius;
  }
} exact_solution;

struct BCWALLTYPE_U : WallBCDIM {
  BCWALLTYPE_U() {}
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
};

struct BCWALLTYPE_V : WallBCDIM {
  BCWALLTYPE_V() {}
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
};

#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBCDIM {
  BCWALLTYPE_W() {}
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
};
#endif

struct BCWALLVALUE_U : CF_DIM {
  BCWALLVALUE_U() {}
  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return exact_solution.solution(+1, dir::x, xyz);
  }
};

struct BCWALLVALUE_V : CF_DIM {
  BCWALLVALUE_V() {}
  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return exact_solution.solution(+1, dir::y, xyz);
  }
};

#ifdef P4_TO_P8
struct BCWALLVALUE_W : CF_DIM {
  BCWALLVALUE_W() {}
  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return exact_solution.solution(+1, dir::z, xyz);
  }
};
#endif

struct convergence_analyzer_for_jump_face_solver_t {
  my_p4est_poisson_jump_faces_t* jump_face_solver;
  const jump_solver_tag tag;
  std::vector<double> errors_in_solution[P4EST_DIM];
  std::vector<double> errors_in_extrapolated_solution_minus[P4EST_DIM];
  std::vector<double> errors_in_extrapolated_solution_plus[P4EST_DIM];
  Vec sharp_error[P4EST_DIM], extrapolation_error_minus[P4EST_DIM], extrapolation_error_plus[P4EST_DIM];
  double extrapolation_band_check_to_diag;

  void delete_and_nullify_face_sampled_errors_if_needed()
  {
    PetscErrorCode ierr;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = delete_and_nullify_vector(sharp_error[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(extrapolation_error_minus[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(extrapolation_error_plus[dim]); CHKERRXX(ierr);
    }
  }

  convergence_analyzer_for_jump_face_solver_t(const jump_solver_tag& tag_) : tag(tag_), sharp_error{DIM(NULL, NULL, NULL)},
    extrapolation_error_minus{DIM(NULL, NULL, NULL)}, extrapolation_error_plus{DIM(NULL, NULL, NULL)}, extrapolation_band_check_to_diag(default_extrapolation_band_check) { }

  void set_extrapolation_band_check(const double& desired_band_to_diag)
  {
    extrapolation_band_check_to_diag = desired_band_to_diag;
  }

  void measure_errors(const my_p4est_faces_t* faces)
  {
    PetscErrorCode ierr;
    const p4est_t* p4est = jump_face_solver->get_p4est();
    const my_p4est_interface_manager_t* interface_manager = jump_face_solver->get_interface_manager();
    const double band = extrapolation_band_check_to_diag*sqrt(SUMD(SQR(jump_face_solver->get_smallest_dxyz()[0]), SQR(jump_face_solver->get_smallest_dxyz()[1]), SQR(jump_face_solver->get_smallest_dxyz()[2])));


    const double *sharp_solution_p[P4EST_DIM];
    const double *extrapolated_solution_minus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    const double *extrapolated_solution_plus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    const Vec* sharp_solution = jump_face_solver->get_solution();
    const Vec* extrapolated_solution_minus  = jump_face_solver->get_extrapolated_solution_minus();
    const Vec* extrapolated_solution_plus   = jump_face_solver->get_extrapolated_solution_plus();

    delete_and_nullify_face_sampled_errors_if_needed();
    double *sharp_error_p[P4EST_DIM];
    double *extrapolation_error_minus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    double *extrapolation_error_plus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArrayRead(sharp_solution[dim], &sharp_solution_p[dim]); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est, faces, &sharp_error[dim], dim); CHKERRXX(ierr);
      ierr = VecGetArray(sharp_error[dim], &sharp_error_p[dim]); CHKERRXX(ierr);
      if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
         ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
      {
        ierr = VecGetArrayRead(extrapolated_solution_minus[dim], &extrapolated_solution_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecGetArrayRead(extrapolated_solution_plus[dim], &extrapolated_solution_plus_p[dim]); CHKERRXX(ierr);
        ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_error_minus[dim], dim); CHKERRXX(ierr);
        ierr = VecCreateGhostFaces(p4est, faces, &extrapolation_error_plus[dim], dim); CHKERRXX(ierr);
        ierr = VecGetArray(extrapolation_error_minus[dim], &extrapolation_error_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecGetArray(extrapolation_error_plus[dim], &extrapolation_error_plus_p[dim]); CHKERRXX(ierr);
      }
    }

    double err_n[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
    double err_extrapolation_minus[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
    double err_extrapolation_plus[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      for (size_t k = 0; k < faces->get_layer_size(dim); ++k) {
        const p4est_locidx_t f_idx    = faces->get_layer_face(dim, k);
        double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dim, xyz_face);

        const double phi_face = interface_manager->phi_at_point(xyz_face);
        sharp_error_p[dim][f_idx] = fabs(sharp_solution_p[dim][f_idx] - exact_solution.solution((phi_face <= 0 ? -1 : +1), dim, xyz_face));
        err_n[dim] = MAX(err_n[dim], sharp_error_p[dim][f_idx]);
        if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
           ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
        {
          extrapolation_error_minus_p[dim][f_idx] = extrapolation_error_plus_p[dim][f_idx] = 0.0;
          if(fabs(phi_face) < band)
          {
            if(phi_face <= 0.0)
            {
              extrapolation_error_plus_p[dim][f_idx] = fabs(extrapolated_solution_plus_p[dim][f_idx] - exact_solution.solution(+1, dim, xyz_face));
              err_extrapolation_plus[dim] = MAX(err_extrapolation_plus[dim], extrapolation_error_plus_p[dim][f_idx]);
            }
            else
            {
              extrapolation_error_minus_p[dim][f_idx] = fabs(extrapolated_solution_minus_p[dim][f_idx] - exact_solution.solution(-1, dim, xyz_face));
              err_extrapolation_minus[dim] = MAX(err_extrapolation_minus[dim], extrapolation_error_minus_p[dim][f_idx]);
            }
          }
        }
      }

      ierr = VecGhostUpdateBegin(sharp_error[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
         ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
      {
        ierr = VecGhostUpdateBegin(extrapolation_error_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(extrapolation_error_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      for (size_t k = 0; k < faces->get_local_size(dim); ++k) {
        const p4est_locidx_t f_idx    = faces->get_local_face(dim, k);
        double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dim, xyz_face);

        const double phi_face = interface_manager->phi_at_point(xyz_face);
        sharp_error_p[dim][f_idx] = fabs(sharp_solution_p[dim][f_idx] - exact_solution.solution((phi_face <= 0 ? -1 : +1), dim, xyz_face));
        err_n[dim] = MAX(err_n[dim], sharp_error_p[dim][f_idx]);
        if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
           ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
        {
          extrapolation_error_minus_p[dim][f_idx] = extrapolation_error_plus_p[dim][f_idx] = 0.0;
          if(fabs(phi_face) < band)
          {
            if(phi_face <= 0.0)
            {
              extrapolation_error_plus_p[dim][f_idx] = fabs(extrapolated_solution_plus_p[dim][f_idx] - exact_solution.solution(+1, dim, xyz_face));
              err_extrapolation_plus[dim] = MAX(err_extrapolation_plus[dim], extrapolation_error_plus_p[dim][f_idx]);
            }
            else
            {
              extrapolation_error_minus_p[dim][f_idx] = fabs(extrapolated_solution_minus_p[dim][f_idx] - exact_solution.solution(-1, dim, xyz_face));
              err_extrapolation_minus[dim] = MAX(err_extrapolation_minus[dim], extrapolation_error_minus_p[dim][f_idx]);
            }
          }
        }
      }

      ierr = VecGhostUpdateEnd(sharp_error[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
         ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
      {
        ierr = VecGhostUpdateEnd(extrapolation_error_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(extrapolation_error_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }

    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArrayRead(sharp_solution[dim], &sharp_solution_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(sharp_error[dim], &sharp_error_p[dim]); CHKERRXX(ierr);
      if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
         ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
      {
        ierr = VecRestoreArrayRead(extrapolated_solution_minus[dim], &extrapolated_solution_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(extrapolated_solution_plus[dim], &extrapolated_solution_plus_p[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArray(extrapolation_error_minus[dim], &extrapolation_error_minus_p[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArray(extrapolation_error_plus[dim], &extrapolation_error_plus_p[dim]); CHKERRXX(ierr);
      }
    }

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, err_n, P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
       ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
    {
      mpiret = MPI_Allreduce(MPI_IN_PLACE, err_extrapolation_minus, P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, err_extrapolation_plus, P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    }

    for(u_char dim = 0; dim < P4EST_DIM; ++dim) {
      errors_in_solution[dim].push_back(err_n[dim]);

      if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
         ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
      {
        errors_in_extrapolated_solution_minus[dim].push_back(err_extrapolation_minus[dim]);
        errors_in_extrapolated_solution_plus[dim].push_back(err_extrapolation_plus[dim]);
      }
    }

    P4EST_ASSERT(errors_in_solution[0].size() == errors_in_solution[1].size() ONLY3D( && errors_in_solution[0].size() == errors_in_solution[2].size()));

    char convergence_order_info[BUFSIZ] = "\0";
    const size_t iter_idx = errors_in_solution[0].size() - 1;

    ierr = PetscPrintf(p4est->mpicomm, "\nFor %s solver: \n", convert_to_string(tag).c_str()); CHKERRXX(ierr); // some spacing
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(iter_idx > 0)
        sprintf(convergence_order_info, ", order = %g", -log(errors_in_solution[dim][iter_idx]/errors_in_solution[dim][iter_idx - 1])/log(2.0));
      ierr = PetscPrintf(p4est->mpicomm, "Error on component %d :\t\t%.5e%s \n", int(dim), errors_in_solution[dim].back(), convergence_order_info); CHKERRXX(ierr);

      if(ANDD(extrapolated_solution_minus[0] != NULL, extrapolated_solution_minus[1] != NULL, extrapolated_solution_minus[2] != NULL) &&
         ANDD(extrapolated_solution_plus[0] != NULL, extrapolated_solution_plus[1] != NULL, extrapolated_solution_plus[2] != NULL))
      {
        if(iter_idx > 0)
          sprintf(convergence_order_info, ", order = %g", -log(errors_in_extrapolated_solution_minus[dim][iter_idx]/errors_in_extrapolated_solution_minus[dim][iter_idx - 1])/log(2.0));
        ierr = PetscPrintf(p4est->mpicomm, "Extrapolation error for minus solution component %d (within %.2g*diag, on faces):\t%.5e%s \n", int(dim), extrapolation_band_check_to_diag, errors_in_extrapolated_solution_minus[dim].back(), convergence_order_info); CHKERRXX(ierr);
        if(iter_idx > 0)
          sprintf(convergence_order_info, ", order = %g", -log(errors_in_extrapolated_solution_plus[dim][iter_idx]/errors_in_extrapolated_solution_plus[dim][iter_idx - 1])/log(2.0));
        ierr = PetscPrintf(p4est->mpicomm, "Extrapolation error for plus solution component %d (within %.2g*diag, on faces):\t%.5e%s \n", int(dim), extrapolation_band_check_to_diag, errors_in_extrapolated_solution_plus[dim].back(), convergence_order_info); CHKERRXX(ierr);
      }
    }
  }

  ~convergence_analyzer_for_jump_face_solver_t()
  {
    delete_and_nullify_face_sampled_errors_if_needed();
  }
};

void set_computational_grid_data(const mpi_environment_t &mpi, my_p4est_brick_t* brick, p4est_connectivity_t *connectivity, const splitting_criteria_cf_and_uniform_band_t* data,
                                 p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, Vec &phi,
                                 my_p4est_hierarchy_t* &hierarchy, my_p4est_node_neighbors_t* &ngbd_n, my_p4est_cell_neighbors_t* &ngbd_c, my_p4est_faces_t* &faces)
{
  if(p4est == NULL)
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*) data;

  for(int i = find_max_level(p4est); i < data->max_lvl; ++i) {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  if(ghost != NULL)
    p4est_ghost_destroy(ghost);
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est, ghost);

  if(nodes != NULL)
    p4est_nodes_destroy(nodes);
  nodes = my_p4est_nodes_new(p4est, ghost);

  if(hierarchy != NULL)
    hierarchy->update(p4est, ghost);
  else
    hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);

  if(ngbd_n != NULL)
    ngbd_n->update(hierarchy, nodes);
  else
  {
    ngbd_n = new my_p4est_node_neighbors_t(hierarchy, nodes);
    ngbd_n->init_neighbors();
  }

  PetscErrorCode ierr;
  if(phi != NULL){
    ierr = VecDestroy(phi); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *(data->phi), phi);
  my_p4est_level_set_t ls_coarse(ngbd_n);
  ls_coarse.reinitialize_2nd_order(phi);

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag(data->min_lvl, data->max_lvl, data->lip, data->uniform_band);
  p4est_t* new_p4est = p4est_copy(p4est, P4EST_FALSE);

  while(data_tag.refine_and_coarsen(new_p4est, nodes, phi_p))
  {
    my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
    interp_nodes.set_input(phi, linear);

    my_p4est_partition(new_p4est, P4EST_FALSE, NULL);
    p4est_ghost_t *new_ghost  = my_p4est_ghost_new(new_p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(new_p4est, new_ghost);
    p4est_nodes_t *new_nodes  = my_p4est_nodes_new(new_p4est, new_ghost);
    Vec new_phi;
    ierr = VecCreateGhostNodes(new_p4est, new_nodes, &new_phi); CHKERRXX(ierr);
    for(size_t nn = 0; nn < new_nodes->indep_nodes.elem_count; ++nn)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(nn, new_p4est, new_nodes, xyz);
      interp_nodes.add_point(nn, xyz);
    }
    interp_nodes.interpolate(new_phi);

    p4est_destroy(p4est); p4est = new_p4est; new_p4est = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_destroy(ghost); ghost = new_ghost;
    hierarchy->update(p4est, ghost);
    p4est_nodes_destroy(nodes); nodes = new_nodes;
    ngbd_n->update(hierarchy, nodes);

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecDestroy(phi); CHKERRXX(ierr); phi = new_phi;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  // uniform band and distance-based set but grid is not balanced, yet
  // make it graded (2-1 ratio):
  p4est_balance(new_p4est, P4EST_CONNECT_FULL, NULL);
  p4est_ghost_t *new_ghost  = my_p4est_ghost_new(new_p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(new_p4est, new_ghost);
  p4est_nodes_t *new_nodes  = my_p4est_nodes_new(new_p4est, new_ghost);
  Vec new_phi;
  ierr = VecCreateGhostNodes(new_p4est, new_nodes, &new_phi); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  interp_nodes.set_input(phi, linear);
  for(size_t nn = 0; nn < new_nodes->indep_nodes.elem_count; ++nn)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(nn, new_p4est, new_nodes, xyz);
    interp_nodes.add_point(nn, xyz);
  }
  interp_nodes.interpolate(new_phi);
  p4est_destroy(p4est); p4est = new_p4est;
  p4est_ghost_destroy(ghost); ghost = new_ghost;
  hierarchy->update(p4est, ghost);
  p4est_nodes_destroy(nodes); nodes = new_nodes;
  ngbd_n->update(hierarchy, nodes);
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = new_phi;

  if(ngbd_c != NULL)
    delete ngbd_c;
  ngbd_c = new my_p4est_cell_neighbors_t(hierarchy);

  if(faces != NULL)
    delete faces;
  faces = new my_p4est_faces_t(p4est, ghost, brick, ngbd_c);
}

void build_interface_capturing_grid_data(p4est_t* p4est_comp, my_p4est_brick_t *brick, const splitting_criteria_cf_t* subrefined_data,
                                         p4est_t* &subrefined_p4est, p4est_ghost_t* &subrefined_ghost, p4est_nodes_t* &subrefined_nodes, Vec &subrefined_phi,
                                         my_p4est_hierarchy_t* &subrefined_hierarchy, my_p4est_node_neighbors_t* &subrefined_ngbd_n)
{
  if(subrefined_p4est != NULL)
    p4est_destroy(subrefined_p4est);
  subrefined_p4est = p4est_copy(p4est_comp, P4EST_FALSE);
  subrefined_p4est->user_pointer = (void*) subrefined_data;
  my_p4est_refine(subrefined_p4est, P4EST_FALSE, refine_levelset_cf, NULL);

  if(subrefined_ghost != NULL)
    p4est_ghost_destroy(subrefined_ghost);
  subrefined_ghost = my_p4est_ghost_new(subrefined_p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(subrefined_p4est, subrefined_ghost);

  if(subrefined_nodes != NULL)
    p4est_nodes_destroy(subrefined_nodes);
  subrefined_nodes = my_p4est_nodes_new(subrefined_p4est, subrefined_ghost);

  if(subrefined_hierarchy != NULL)
    subrefined_hierarchy->update(subrefined_p4est, subrefined_ghost);
  else
    subrefined_hierarchy = new my_p4est_hierarchy_t(subrefined_p4est, subrefined_ghost, brick);

  if(subrefined_ngbd_n != NULL)
    subrefined_ngbd_n->update(subrefined_hierarchy, subrefined_nodes);
  else
  {
    subrefined_ngbd_n = new my_p4est_node_neighbors_t(subrefined_hierarchy, subrefined_nodes);
    subrefined_ngbd_n->init_neighbors();
  }

  PetscErrorCode ierr;
  if (subrefined_phi != NULL){
    ierr = VecDestroy(subrefined_phi); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(subrefined_p4est, subrefined_nodes, &subrefined_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(subrefined_p4est, subrefined_nodes, *(subrefined_data->phi), subrefined_phi);
  my_p4est_level_set_t ls(subrefined_ngbd_n);
  ls.reinitialize_2nd_order(subrefined_phi);

  const double *subrefined_phi_p;
  ierr = VecGetArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);
  splitting_criteria_tag_t data_tag(subrefined_data->min_lvl, subrefined_data->max_lvl, subrefined_data->lip);
  p4est_t *new_subrefined_p4est = p4est_copy(subrefined_p4est, P4EST_FALSE);

  while(data_tag.refine(new_subrefined_p4est, subrefined_nodes, subrefined_phi_p)) // not refine_and_coarsen, because we need the fine grid to be everywhere finer or as coarse as the coarse grid!
  {
    ierr = VecRestoreArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);
    my_p4est_interpolation_nodes_t interp_subrefined_nodes(subrefined_ngbd_n);
    interp_subrefined_nodes.set_input(subrefined_phi, linear);

    p4est_ghost_t *new_subrefined_ghost = my_p4est_ghost_new(new_subrefined_p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(new_subrefined_p4est, new_subrefined_ghost);
    p4est_nodes_t *new_subrefined_nodes  = my_p4est_nodes_new(new_subrefined_p4est, new_subrefined_ghost);
    Vec new_subrefined_phi;
    ierr = VecCreateGhostNodes(new_subrefined_p4est, new_subrefined_nodes, &new_subrefined_phi); CHKERRXX(ierr);
    for(size_t nn = 0; nn < new_subrefined_nodes->indep_nodes.elem_count; ++nn)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(nn, new_subrefined_p4est, new_subrefined_nodes, xyz);
      interp_subrefined_nodes.add_point(nn, xyz);
    }
    interp_subrefined_nodes.interpolate(new_subrefined_phi);


    p4est_destroy(subrefined_p4est); subrefined_p4est = new_subrefined_p4est; new_subrefined_p4est = p4est_copy(subrefined_p4est, P4EST_FALSE);
    p4est_ghost_destroy(subrefined_ghost); subrefined_ghost = new_subrefined_ghost;
    subrefined_hierarchy->update(subrefined_p4est, subrefined_ghost);
    p4est_nodes_destroy(subrefined_nodes); subrefined_nodes = new_subrefined_nodes;
    subrefined_ngbd_n->update(subrefined_hierarchy, subrefined_nodes);
    ls.update(subrefined_ngbd_n);

    ierr = VecDestroy(subrefined_phi); CHKERRXX(ierr); subrefined_phi = new_subrefined_phi;

    ierr = VecGetArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(subrefined_phi, &subrefined_phi_p); CHKERRXX(ierr);

  p4est_destroy(new_subrefined_p4est);
}

void sample_marangoni_force(const double& background_gradient_of_surf_tension, const p4est_nodes_t* nodes, Vec marangoni_force)
{
  double *marangoni_force_p;
  PetscErrorCode ierr;
  ierr = VecGetArray(marangoni_force, &marangoni_force_p); CHKERRXX(ierr);
  for (size_t k = 0; k < nodes->indep_nodes.elem_count; ++k) {
    marangoni_force_p[P4EST_DIM*k + 0] = 0.0;
    marangoni_force_p[P4EST_DIM*k + 1] = 0.0;
#ifdef P4_TO_P8
    marangoni_force_p[P4EST_DIM*k + 2] = -background_gradient_of_surf_tension;
#else
    (void) background_gradient_of_surf_tension;
#endif
  }
  ierr = VecRestoreArray(marangoni_force, &marangoni_force_p); CHKERRXX(ierr);
}

void sample_rhs(const my_p4est_faces_t* faces, Vec rhs_minus[P4EST_DIM], Vec rhs_plus[P4EST_DIM])
{
  PetscErrorCode ierr;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    double *rhs_minus_dir_p, *rhs_plus_dir_p;
    ierr = VecGetArray(rhs_minus[dir],  &rhs_minus_dir_p);  CHKERRXX(ierr);
    ierr = VecGetArray(rhs_plus[dir],   &rhs_plus_dir_p);   CHKERRXX(ierr);
    for (p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx) {
      double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz_face);
      rhs_minus_dir_p[f_idx]  = -exact_solution.grad_p(-1, dir, xyz_face);
      rhs_plus_dir_p[f_idx]   = -exact_solution.grad_p(+1, dir, xyz_face);
    }
    ierr = VecRestoreArray(rhs_minus[dir],  &rhs_minus_dir_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_plus[dir],   &rhs_plus_dir_p);   CHKERRXX(ierr);
  }
  return;
}

void get_sampled_exact_solution(Vec exact_msol_at_nodes, Vec exact_psol_at_nodes,
                                const p4est_t* p4est, const p4est_nodes_t* nodes)
{
  PetscErrorCode ierr;

  double *exact_msol_at_nodes_p, *exact_psol_at_nodes_p;
  ierr = VecGetArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  for(size_t node_idx = 0; node_idx < nodes->indep_nodes.elem_count; ++node_idx) {
    double xyz_node[P4EST_DIM];
    node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      exact_msol_at_nodes_p[P4EST_DIM*node_idx + dim] = exact_solution.solution(-1, dim, xyz_node);
      exact_psol_at_nodes_p[P4EST_DIM*node_idx + dim] = exact_solution.solution(+1, dim, xyz_node);
    }
  }
  ierr = VecRestoreArray(exact_psol_at_nodes, &exact_psol_at_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(exact_msol_at_nodes, &exact_msol_at_nodes_p); CHKERRXX(ierr);
}

void transfer_face_fields_to_cells(const my_p4est_faces_t* faces, std::vector<const Vec*> face_fields, std::vector<Vec> face_fields_on_cells)
{
  std::vector<const double*>  face_fields_p[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    face_fields_p[dim].resize(face_fields.size());
  }
  std::vector<double*>  face_fields_on_cells_p(face_fields.size());
  PetscErrorCode ierr;
  for (size_t k = 0; k < face_fields.size(); ++k) {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArrayRead(face_fields[k][dim], &face_fields_p[dim][k]); CHKERRXX(ierr);
    }
    ierr = VecGetArray(face_fields_on_cells[k], &face_fields_on_cells_p[k]); CHKERRXX(ierr);
  }

  for (size_t k = 0; k < faces->get_ngbd_c()->get_hierarchy()->get_layer_size(); ++k) {
    const p4est_locidx_t quad_idx = faces->get_ngbd_c()->get_hierarchy()->get_local_index_of_layer_quadrant(k);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(faces->q2f(quad_idx, 2*dim) != NO_VELOCITY)
      {
        for (size_t k = 0; k < face_fields.size(); ++k)
          face_fields_on_cells_p[k][P4EST_DIM*quad_idx + dim] = face_fields_p[dim][k][faces->q2f(quad_idx, 2*dim)];
      }
      else
      {
        set_of_neighboring_quadrants ngbd_cells;
        faces->get_ngbd_c()->find_neighbor_cells_of_cell(ngbd_cells, quad_idx, tree_index_of_quad(quad_idx, faces->get_p4est(), faces->get_ghost()), 2*dim);
        for (size_t k = 0; k < face_fields.size(); ++k)
          face_fields_on_cells_p[k][P4EST_DIM*quad_idx + dim] = 0.0;
        for (set_of_neighboring_quadrants::const_iterator it = ngbd_cells.begin(); it != ngbd_cells.end(); ++it)
        {
          P4EST_ASSERT(faces->q2f(it->p.piggy3.local_num, 2*dim + 1) != NO_VELOCITY);
          for (size_t k = 0; k < face_fields.size(); ++k)
            face_fields_on_cells_p[k][P4EST_DIM*quad_idx + dim] += face_fields_p[dim][k][faces->q2f(it->p.piggy3.local_num, 2*dim + 1)]/ngbd_cells.size();
        }
      }
    }
  }
  for (size_t k = 0; k < face_fields.size(); ++k) {
    ierr = VecGhostUpdateBegin(face_fields_on_cells[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  for (size_t k = 0; k < faces->get_ngbd_c()->get_hierarchy()->get_inner_size(); ++k) {
    const p4est_locidx_t quad_idx = faces->get_ngbd_c()->get_hierarchy()->get_local_index_of_inner_quadrant(k);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      if(faces->q2f(quad_idx, 2*dim) != NO_VELOCITY)
      {
        for (size_t k = 0; k < face_fields.size(); ++k)
          face_fields_on_cells_p[k][P4EST_DIM*quad_idx + dim] = face_fields_p[dim][k][faces->q2f(quad_idx, 2*dim)];
      }
      else
      {
        set_of_neighboring_quadrants ngbd_cells;
        faces->get_ngbd_c()->find_neighbor_cells_of_cell(ngbd_cells, quad_idx, tree_index_of_quad(quad_idx, faces->get_p4est(), faces->get_ghost()), 2*dim);
        for (size_t k = 0; k < face_fields.size(); ++k)
          face_fields_on_cells_p[k][P4EST_DIM*quad_idx + dim] = 0.0;
        for (set_of_neighboring_quadrants::const_iterator it = ngbd_cells.begin(); it != ngbd_cells.end(); ++it)
        {
          P4EST_ASSERT(faces->q2f(it->p.piggy3.local_num, 2*dim + 1) != NO_VELOCITY);
          for (size_t k = 0; k < face_fields.size(); ++k)
            face_fields_on_cells_p[k][P4EST_DIM*quad_idx + dim] += face_fields_p[dim][k][faces->q2f(it->p.piggy3.local_num, 2*dim + 1)]/ngbd_cells.size();
        }
      }
    }
  }
  for (size_t k = 0; k < face_fields.size(); ++k) {
    ierr = VecGhostUpdateEnd(face_fields_on_cells[k], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for (size_t k = 0; k < face_fields.size(); ++k) {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArrayRead(face_fields[k][dim], &face_fields_p[dim][k]); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(face_fields_on_cells[k], &face_fields_on_cells_p[k]); CHKERRXX(ierr);
  }
}

void save_VTK(const string out_dir, const int &iter, Vec exact_solution_minus, Vec exact_solution_plus,
              const convergence_analyzer_for_jump_face_solver_t& analyzer,
              const my_p4est_brick_t *brick)
{
  const my_p4est_interface_manager_t* interface_manager = analyzer.jump_face_solver->get_interface_manager();
  splitting_criteria_t* data = (splitting_criteria_t*) analyzer.jump_face_solver->get_p4est()->user_pointer;

  ostringstream command;
  command << "mkdir -p " << out_dir.c_str();
  int system_return = system(command.str().c_str()); (void) system_return;

  ostringstream oss_computational;
  oss_computational << out_dir << "/computational_grid_macromesh_" << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] ONLY3D(<< "x" << brick->nxyztrees[2])
      << "_lmin_" << data->min_lvl - iter << "_lmax_" << data->max_lvl - iter << "_iter_" << iter;

  const p4est_t* p4est = analyzer.jump_face_solver->get_p4est();
  const p4est_nodes_t* nodes = analyzer.jump_face_solver->get_nodes();
  const p4est_ghost_t* ghost = analyzer.jump_face_solver->get_ghost();

  std::vector<Vec_for_vtk_export_t> comp_node_scalar_fields;
  std::vector<Vec_for_vtk_export_t> comp_node_vector_fields;
  std::vector<Vec_for_vtk_export_t> comp_cell_scalar_fields;
  std::vector<Vec_for_vtk_export_t> comp_cell_vector_fields;
  std::vector<Vec_for_vtk_export_t>* interface_capturing_node_scalar_fields = (interface_manager->subcell_resolution() > 0 ? new std::vector<Vec_for_vtk_export_t> : &comp_node_scalar_fields);
  std::vector<Vec_for_vtk_export_t>* interface_capturing_node_vector_fields = (interface_manager->subcell_resolution() > 0 ? new std::vector<Vec_for_vtk_export_t> : &comp_node_vector_fields);
  std::vector<Vec_for_vtk_export_t>* interface_capturing_cell_scalar_fields = (interface_manager->subcell_resolution() > 0 ? new std::vector<Vec_for_vtk_export_t> : &comp_cell_scalar_fields);
  std::vector<Vec_for_vtk_export_t>* interface_capturing_cell_vector_fields = (interface_manager->subcell_resolution() > 0 ? new std::vector<Vec_for_vtk_export_t> : &comp_cell_vector_fields);

  // transfer face-sampled fields to cells for visualization purposes
  std::vector<const Vec *>  face_fields;
  std::vector<Vec>          face_fields_on_cells;
  PetscErrorCode ierr;
  Vec sharp_solution_cells, sharp_error_cells;
  Vec extrapolated_solution_minus_cells = NULL;
  Vec extrapolated_solution_plus_cells = NULL;
  Vec extrapolation_error_minus_cells = NULL;
  Vec extrapolation_error_plus_cells = NULL;

  face_fields.push_back(analyzer.jump_face_solver->get_solution());
  ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &sharp_solution_cells); CHKERRXX(ierr); face_fields_on_cells.push_back(sharp_solution_cells);
  face_fields.push_back(analyzer.sharp_error);
  ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &sharp_error_cells); CHKERRXX(ierr); face_fields_on_cells.push_back(sharp_error_cells);
  const Vec* extrapolation_minus = analyzer.jump_face_solver->get_extrapolated_solution_minus();
  if(ANDD(extrapolation_minus[0] != NULL, extrapolation_minus[1] != NULL, extrapolation_minus[2] != NULL))
  {
    face_fields.push_back(extrapolation_minus);
    ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &extrapolated_solution_minus_cells); CHKERRXX(ierr); face_fields_on_cells.push_back(extrapolated_solution_minus_cells);
  }
  const Vec* extrapolation_plus = analyzer.jump_face_solver->get_extrapolated_solution_plus();
  if(ANDD(extrapolation_plus[0] != NULL, extrapolation_plus[1] != NULL, extrapolation_plus[2] != NULL))
  {
    face_fields.push_back(extrapolation_plus);
    ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &extrapolated_solution_plus_cells); CHKERRXX(ierr); face_fields_on_cells.push_back(extrapolated_solution_plus_cells);
  }
  if(ANDD(analyzer.extrapolation_error_minus[0] != NULL, analyzer.extrapolation_error_minus[1] != NULL, analyzer.extrapolation_error_minus[2] != NULL))
  {
    face_fields.push_back(analyzer.extrapolation_error_minus);
    ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &extrapolation_error_minus_cells); CHKERRXX(ierr); face_fields_on_cells.push_back(extrapolation_error_minus_cells);
  }
  if(ANDD(analyzer.extrapolation_error_plus[0] != NULL, analyzer.extrapolation_error_plus[1] != NULL, analyzer.extrapolation_error_plus[2] != NULL))
  {
    face_fields.push_back(analyzer.extrapolation_error_plus);
    ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &extrapolation_error_plus_cells); CHKERRXX(ierr); face_fields_on_cells.push_back(extrapolation_error_plus_cells);
  }
  transfer_face_fields_to_cells(analyzer.jump_face_solver->get_faces(), face_fields, face_fields_on_cells);

  // on computational grid nodes
  comp_node_vector_fields.push_back(Vec_for_vtk_export_t(exact_solution_minus, "exact_solution_minus"));
  comp_node_vector_fields.push_back(Vec_for_vtk_export_t(exact_solution_plus, "exact_solution_plus"));
  comp_node_scalar_fields.push_back(Vec_for_vtk_export_t(interface_manager->get_phi_on_computational_nodes(), "phi"));
  comp_cell_vector_fields.push_back(Vec_for_vtk_export_t(sharp_solution_cells, "solution"));
  comp_cell_vector_fields.push_back(Vec_for_vtk_export_t(sharp_error_cells, "sharp_error"));
  if(extrapolated_solution_minus_cells != NULL)
    comp_cell_vector_fields.push_back(Vec_for_vtk_export_t(extrapolated_solution_minus_cells, "extrapolation_minus"));
  if(extrapolated_solution_plus_cells != NULL)
    comp_cell_vector_fields.push_back(Vec_for_vtk_export_t(extrapolated_solution_plus_cells, "extrapolation_plus"));
  if(extrapolation_error_minus_cells != NULL)
    comp_cell_vector_fields.push_back(Vec_for_vtk_export_t(extrapolation_error_minus_cells, "extrapolation_error_minus"));
  if(extrapolation_error_plus_cells != NULL)
    comp_cell_vector_fields.push_back(Vec_for_vtk_export_t(extrapolation_error_plus_cells, "extrapolation_error_plus"));
  // on interface-capturing grid nodes
  if(analyzer.jump_face_solver->get_jump_u_dot_n() != NULL)
    interface_capturing_node_scalar_fields->push_back(Vec_for_vtk_export_t(analyzer.jump_face_solver->get_jump_u_dot_n(), "jump_u_dot_n"));
  if(analyzer.jump_face_solver->get_jump_in_tangential_stress() != NULL)
    interface_capturing_node_vector_fields->push_back(Vec_for_vtk_export_t(analyzer.jump_face_solver->get_jump_in_tangential_stress(), "jump_tangential_stress"));
  Vec jump_flux_component[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  if(dynamic_cast<my_p4est_poisson_jump_faces_xgfm_t*>(analyzer.jump_face_solver) != NULL)
  {
    my_p4est_poisson_jump_faces_xgfm_t* xgfm_solver = dynamic_cast<my_p4est_poisson_jump_faces_xgfm_t*>(analyzer.jump_face_solver);
    if(xgfm_solver->get_validation_jump() != NULL)
      interface_capturing_node_vector_fields->push_back(Vec_for_vtk_export_t(xgfm_solver->get_validation_jump(), "jump_u_validation"));
    if(xgfm_solver->get_validation_jump_mu_grad_u() != NULL)
    {
      double *jump_flux_component_p[P4EST_DIM];
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecCreateGhostNodesBlock(interface_manager->get_interface_capturing_ngbd_n().get_p4est(), interface_manager->get_interface_capturing_ngbd_n().get_nodes(), P4EST_DIM, &jump_flux_component[dim]); CHKERRXX(ierr);
        ierr = VecGetArray(jump_flux_component[dim], &jump_flux_component_p[dim]); CHKERRXX(ierr);
      }
      const double *jump_flux_component_tensor_block_p;
      ierr = VecGetArrayRead(xgfm_solver->get_validation_jump_mu_grad_u(), &jump_flux_component_tensor_block_p); CHKERRXX(ierr);
      for (size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_nodes()->indep_nodes.elem_count; ++k) {
        for (u_char comp = 0; comp < P4EST_DIM; ++comp) {
          for (int der = 0; der < P4EST_DIM; ++der) {
            jump_flux_component_p[comp][P4EST_DIM*k + der] = jump_flux_component_tensor_block_p[SQR_P4EST_DIM*k + P4EST_DIM*comp + der];
          }
        }
      }
      ierr = VecRestoreArrayRead(xgfm_solver->get_validation_jump_mu_grad_u(), &jump_flux_component_tensor_block_p); CHKERRXX(ierr);
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecRestoreArray(jump_flux_component[dim], &jump_flux_component_p[dim]); CHKERRXX(ierr);
        interface_capturing_node_vector_fields->push_back(Vec_for_vtk_export_t(jump_flux_component[dim], "jump_validation_flux_" + to_string(dim)));
      }
    }
  }
  interface_capturing_node_vector_fields->push_back(Vec_for_vtk_export_t(interface_manager->get_grad_phi(), "grad_phi"));
  if(interface_manager->subcell_resolution() > 0)
    interface_capturing_node_scalar_fields->push_back(Vec_for_vtk_export_t(interface_manager->get_phi(), "phi"));

  my_p4est_vtk_write_all_general_lists(p4est, nodes, ghost, P4EST_TRUE, P4EST_TRUE, oss_computational.str().c_str(),
                                       (comp_node_scalar_fields.size() > 0 ? &comp_node_scalar_fields : NULL),
                                       (comp_node_vector_fields.size() > 0 ? &comp_node_vector_fields : NULL),
                                       (comp_cell_scalar_fields.size() > 0 ? &comp_cell_scalar_fields : NULL),
                                       (comp_cell_vector_fields.size() > 0 ? &comp_cell_vector_fields : NULL));

  comp_node_scalar_fields.clear(); comp_node_vector_fields.clear(); comp_cell_scalar_fields.clear(); comp_cell_vector_fields.clear();
  ierr = delete_and_nullify_vector(sharp_solution_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(extrapolated_solution_minus_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(extrapolated_solution_plus_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(sharp_error_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(extrapolation_error_minus_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(extrapolation_error_plus_cells); CHKERRXX(ierr);

  if(interface_manager->subcell_resolution() > 0)
  {
    std::ostringstream oss_interface_capturing;
    oss_interface_capturing << out_dir << "/interface_capturing_grid_macromesh_" << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] ONLY3D(<< "x" << brick->nxyztrees[2])
        << "_lmin_" << data->min_lvl - iter << "_lmax_" << data->max_lvl - iter << "_iter_" << iter;

    const p4est_t*        interface_capturing_p4est = interface_manager->get_interface_capturing_ngbd_n().get_p4est();
    const p4est_nodes_t*  interface_capturing_nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
    const p4est_ghost_t*  interface_capturing_ghost = interface_manager->get_interface_capturing_ngbd_n().get_ghost();
    my_p4est_vtk_write_all_general_lists(interface_capturing_p4est, interface_capturing_nodes, interface_capturing_ghost, P4EST_TRUE, P4EST_TRUE, oss_interface_capturing.str().c_str(),
                                         interface_capturing_node_scalar_fields, interface_capturing_node_vector_fields,
                                         interface_capturing_cell_scalar_fields, NULL);

    delete interface_capturing_node_scalar_fields;
    delete interface_capturing_node_vector_fields;
    delete interface_capturing_cell_scalar_fields;
    delete interface_capturing_cell_vector_fields;
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = delete_and_nullify_vector(jump_flux_component[dim]); CHKERRXX(ierr);
  }

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", out_dir.c_str());
  return;
}


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  // computational grid parameters
  cmd.add_option("lmin",            "first min level of the trees, default is " + to_string(default_lmin));
  cmd.add_option("lmax",            "first max level of the trees, default is " + to_string(default_lmax));
  cmd.add_option("ntree",           "number of trees in the macromesh along every dimension of the computational domain. Default value is " + to_string(default_ntree));
  cmd.add_option("uniform_band",    "number of grid cells of uniform grid layering the interface on either side (default is such that the finest level covers the region where the absolue value of phi is less than "  + to_string(100*uniform_band_factor) + "% of the bubble radius)");
  // method/inner procedure control
  cmd.add_option("voro_on_the_fly", "activates the calculation of Voronoi cells on the fly (default is " + string(default_voro_fly ? "done on the fly)" : "stored in memory"));
  cmd.add_option("solver",          "solver to be tested, possible choices are 'GFM', 'xGFM' [default is "  + convert_to_string(default_solver) + "].");
  // problem-related parameters
  cmd.add_option("radius",          "radius of the bubble. Default value is " + to_string(default_r0));
  cmd.add_option("length",          "side length of the computational domain. Default value is " + to_string(default_length));
  cmd.add_option("mu_minus",        "viscosity coefficient in the negative domain (default is " + to_string(default_mu_m) + string(")"));
  cmd.add_option("mu_plus",         "viscosity coefficient in the positive domain (default is " + to_string(default_mu_p) + string(")"));
  cmd.add_option("gamma_0",         "mean background surface tenstion (default is " + to_string(default_gamma_0) + string(")"));
  cmd.add_option("beta",            "nondimensional background gradient of surface tenstion profile, gamma = gamma_0*(1.0 + beta*z/bubble_radius). Default value is " + to_string(default_beta) + string(")"));
  cmd.add_option("subrefinement",   "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement.");
  cmd.add_option("second_order_ls", "activate second order interface localization if present. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("extrapolate",     "flag activating the extrapolation of the sharp solution from either side to the other. Default is " + string(default_extrapolation ? "with" : "without") + " extrapolation.");
  cmd.add_option("bandcheck",       "band check (in number of smallest diagonals) in which the extrapolation accuracy is checked. Default (if not specified) is " + to_string(default_extrapolation_band_check) + " smallest diagonals.");
  // exportation control
  cmd.add_option("save_vtk",        "saves vtk visualization files if present (default is " + string(default_save_vtk ? "" : "not ") + "saved)");
  cmd.add_option("work_dir",        "exportation directory, if not defined otherwise in the environment variable OUT_DIR. \n\
\tThis is required if saving vtk or summary files: work_dir/vtu for vtk files work_dir/summaries for summary files. Default is " + default_work_folder);
  cmd.add_option("ngrids",          "number of successively finer grids to consider (default is " + to_string(default_ngrids) + ")");
  cmd.add_option("print",           "prints results and final convergence results in a file in the root exportation directory if present (default is "  + string(default_print ? "with" : "without") + " exportation)");
  ostringstream oss; oss.str("");
  oss << default_interp_method_phi;
  cmd.add_option("phi_interp",      "interpolation method for the node-sampled levelset function. Default is " + oss.str());

  if(cmd.parse(argc, argv, extra_info))
    return 0;

#ifndef P4_TO_P8
  throw std::runtime_error("main for creeping bubble flow: has a valid use in 3D only, sorry!");
#endif

  const string root_export_dir = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR"))) + "/" + to_string(P4EST_DIM) + "D";

  const bool save_vtk = cmd.get<bool>("save_vtk", default_save_vtk);

  if(save_vtk && create_directory(root_export_dir.c_str(), mpi.rank(), mpi.comm()))
  {
    char error_msg[BUFSIZ];
    sprintf(error_msg, "main for creeping bubble flow: could not create the main exportation directory %s", root_export_dir.c_str());
    throw std::runtime_error(error_msg);
  }

  PetscErrorCode ierr;
  const int n_tree_xyz[P4EST_DIM]   = { DIM(cmd.get<int>("ntree", default_ntree), cmd.get<int>("ntree", default_ntree), cmd.get<int>("ntree", default_ntree))};
  const double side_length = cmd.get<double>("length", default_length);
  const double xyz_min [P4EST_DIM]  = { DIM(-0.5*side_length, -0.5*side_length, -0.5*side_length) };
  const double xyz_max [P4EST_DIM]  = { DIM( 0.5*side_length,  0.5*side_length,  0.5*side_length) };
  const double bubble_radius        = cmd.get<double>("radius", default_r0);
  if(bubble_radius > side_length/2)
    throw std::invalid_argument("main for creeping bubble flow: your domain is too small for the bubble radius");
  const int periodic[P4EST_DIM]     = { DIM(0, 0, 0) };
  const double mu_minus = cmd.get<double>("mu_minus", default_mu_m);
  const double mu_plus  = cmd.get<double>("mu_plus", default_mu_p);
  const double gamma_0  = cmd.get<double>("gamma_0", default_gamma_0);
  const double beta     = cmd.get<double>("beta", default_beta);
  exact_solution.set(mu_minus, mu_plus, gamma_0*beta, bubble_radius);
  LEVEL_SET levelset(xyz_min, xyz_max, bubble_radius);
  const jump_solver_tag solver_to_test = cmd.get<jump_solver_tag>("solver", xGFM);

  BoundaryConditionsDIM bc_v[P4EST_DIM];

  BCWALLTYPE_U wall_type_u; BCWALLVALUE_U wall_value_u; bc_v[0].setWallTypes(wall_type_u); bc_v[0].setWallValues(wall_value_u);
  BCWALLTYPE_V wall_type_v; BCWALLVALUE_V wall_value_v; bc_v[1].setWallTypes(wall_type_v); bc_v[1].setWallValues(wall_value_v);
#ifdef P4_TO_P8
  BCWALLTYPE_W wall_type_w; BCWALLVALUE_W wall_value_w; bc_v[2].setWallTypes(wall_type_w); bc_v[2].setWallValues(wall_value_w);
#endif

  int lmin = cmd.get<int>("lmin", default_lmin);
  int lmax = cmd.get<int>("lmax", default_lmax);
  const int ngrids = cmd.get<int>("ngrids", default_ngrids);
  const bool use_subrefinement = cmd.get<bool>("subrefinement", default_subrefinement);
  const bool use_second_order_theta = cmd.get<bool>("second_order_ls", default_use_second_order_theta);
  const interpolation_method phi_interp = cmd.get<interpolation_method>("phi_interp", default_interp_method_phi);
  const bool extrapolate_solution = cmd.get<bool>("extrapolate", default_extrapolation);
  const string vtk_out = root_export_dir + "/vtu/" + (use_subrefinement ? "subresolved" : "standard");

  my_p4est_brick_t brick;
  p4est_connectivity_t* connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min, xyz_max, &brick, periodic);
  splitting_criteria_cf_and_uniform_band_t *data = NULL; splitting_criteria_cf_t *subrefined_data        = NULL;
  p4est_t                       *p4est      = NULL, *subrefined_p4est       = NULL;
  p4est_nodes_t                 *nodes      = NULL, *subrefined_nodes       = NULL;
  p4est_ghost_t                 *ghost      = NULL, *subrefined_ghost       = NULL;
  my_p4est_hierarchy_t          *hierarchy  = NULL, *subrefined_hierarchy   = NULL;
  my_p4est_node_neighbors_t     *ngbd_n     = NULL, *subrefined_ngbd_n      = NULL;
  Vec                            phi        = NULL,  subrefined_phi         = NULL;
  my_p4est_cell_neighbors_t     *ngbd_c     = NULL;
  my_p4est_faces_t              *faces      = NULL;
  my_p4est_interface_manager_t  *interface_manager = NULL;
  convergence_analyzer_for_jump_face_solver_t analyzer(xGFM);
  if(cmd.contains("bandcheck"))
    analyzer.set_extrapolation_band_check(cmd.get<double>("bandcheck", default_extrapolation_band_check));

  for (int k_grid = 0; k_grid < ngrids; ++k_grid) {
    if(k_grid > 0)
    {
      lmin++;
      lmax++;
    }

    const double dxyzmin  = MAX(DIM((xyz_max[0] - xyz_min[0])/(double)n_tree_xyz[0], (xyz_max[1] - xyz_min[1])/(double)n_tree_xyz[1], (xyz_max[2] - xyz_min[2])/(double)n_tree_xyz[2]))/ (1 << lmax);
    const double uniform_band = cmd.get<double>("uniform_band", uniform_band_factor*bubble_radius/dxyzmin);
    /* build/updates the computational grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods, its cell neighborhoods
     * the REINITIALIZED levelset on the computational grid
     */
    data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &levelset, uniform_band/*, 2.8*/);
    set_computational_grid_data(mpi, &brick, connectivity, data,
                                p4est, ghost, nodes, phi, hierarchy, ngbd_n, ngbd_c, faces);

    const my_p4est_node_neighbors_t* interface_capturing_ngbd_n;  // no creation here, just a renamed pointer to streamline the logic
    Vec interface_capturing_phi;                                  // no creation here, just a renamed pointer to streamline the logic
    if(use_subrefinement)
    {
      /* build/updates the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
       * the REINITIALIZED levelset on the interface-capturing grid
       */
      subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, &levelset);
      build_interface_capturing_grid_data(p4est, &brick, subrefined_data,
                                          subrefined_p4est, subrefined_ghost, subrefined_nodes, subrefined_phi, subrefined_hierarchy, subrefined_ngbd_n);
      interface_capturing_ngbd_n  = subrefined_ngbd_n;
      interface_capturing_phi     = subrefined_phi;
    }
    else
    {
      interface_capturing_ngbd_n  = ngbd_n;
      interface_capturing_phi     = phi;
    }

    Vec interface_capturing_phi_xxyyzz = NULL;
    interface_manager = new my_p4est_interface_manager_t(faces, nodes, interface_capturing_ngbd_n);
    if(use_second_order_theta || (!use_subrefinement && phi_interp != linear)){
      ierr = VecCreateGhostNodesBlock(interface_capturing_ngbd_n->get_p4est(), interface_capturing_ngbd_n->get_nodes(), P4EST_DIM, &interface_capturing_phi_xxyyzz); CHKERRXX(ierr);
      interface_capturing_ngbd_n->second_derivatives_central(interface_capturing_phi, interface_capturing_phi_xxyyzz);
    }
    interface_manager->set_levelset(interface_capturing_phi, (use_subrefinement ? linear : phi_interp), interface_capturing_phi_xxyyzz, true); // last argument set to true cause we'll need the gradient of phi
    interface_manager->evaluate_FD_theta_with_quadratics(use_second_order_theta);
    if(use_subrefinement)
      interface_manager->set_under_resolved_levelset(phi);

    Vec jump_solution = NULL;
    Vec marangoni_force;
    ierr = VecCreateGhostNodesBlock(interface_capturing_ngbd_n->get_p4est(), interface_capturing_ngbd_n->get_nodes(), P4EST_DIM, &marangoni_force); CHKERRXX(ierr);
    sample_marangoni_force(gamma_0*beta/bubble_radius, interface_capturing_ngbd_n->get_nodes(), marangoni_force);

    Vec rhs_minus[P4EST_DIM], rhs_plus[P4EST_DIM];
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecCreateNoGhostFaces(p4est, faces, &rhs_minus[dim], dim); CHKERRXX(ierr);
      ierr = VecCreateNoGhostFaces(p4est, faces, &rhs_plus[dim], dim); CHKERRXX(ierr);
    }
    sample_rhs(faces, rhs_minus, rhs_plus);

    my_p4est_poisson_jump_faces_xgfm_t * jump_solver_faces = new my_p4est_poisson_jump_faces_xgfm_t(faces, nodes);
    jump_solver_faces->activate_xGFM_corrections(solver_to_test == xGFM);
    jump_solver_faces->set_interface(interface_manager);
    jump_solver_faces->set_bc(bc_v);
    jump_solver_faces->set_mus(mu_minus, mu_plus);
    jump_solver_faces->set_jumps(jump_solution, marangoni_force);
    jump_solver_faces->set_compute_partition_on_the_fly(default_voro_fly || cmd.contains("voro_on_the_fly"));
    jump_solver_faces->set_rhs(rhs_minus, rhs_plus);
    jump_solver_faces->solve_for_sharp_solution();

    if(extrapolate_solution)
    {
      jump_solver_faces->set_validity_of_interface_neighbors_for_extrapolation(true);
      jump_solver_faces->extrapolate_solution_from_either_side_to_the_other((int) ceil(10*analyzer.extrapolation_band_check_to_diag));
    }

    analyzer.jump_face_solver = jump_solver_faces;
    analyzer.measure_errors(faces);

    if(save_vtk)
    {
      Vec exact_solution_minus = NULL, exact_solution_plus = NULL; // to enable illustration of exact solution with wrap-by-scalar in paraview or to calculate the integral of the exact solution, numerically
      ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &exact_solution_minus);  CHKERRXX(ierr);
      ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &exact_solution_plus);   CHKERRXX(ierr);
      get_sampled_exact_solution(exact_solution_minus, exact_solution_plus, p4est, nodes);

      save_VTK(vtk_out, k_grid, exact_solution_minus, exact_solution_plus, analyzer, &brick);
      ierr = VecDestroy(exact_solution_minus); CHKERRXX(ierr);
      ierr = VecDestroy(exact_solution_plus); CHKERRXX(ierr);
    }

    ierr = delete_and_nullify_vector(phi); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(subrefined_phi); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(interface_capturing_phi_xxyyzz); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(jump_solution); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(marangoni_force); CHKERRXX(ierr);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = delete_and_nullify_vector(rhs_minus[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(rhs_plus[dim]); CHKERRXX(ierr);
    }

    delete data;
    if(subrefined_data != NULL)
      delete subrefined_data;

    delete interface_manager;
    delete jump_solver_faces;
  }

  if(p4est != NULL)
    p4est_destroy(p4est);
  if(subrefined_p4est != NULL)
    p4est_destroy(subrefined_p4est);
  if(nodes != NULL)
    p4est_nodes_destroy(nodes);
  if(subrefined_nodes)
    p4est_nodes_destroy(subrefined_nodes);
  if(ghost != NULL)
    p4est_ghost_destroy(ghost);
  if(subrefined_ghost != NULL)
    p4est_ghost_destroy(subrefined_ghost);
  if(hierarchy != NULL)
    delete  hierarchy;
  if(subrefined_hierarchy != NULL)
    delete subrefined_hierarchy;
  if(ngbd_n != NULL)
    delete ngbd_n;
  if(subrefined_ngbd_n != NULL)
    delete  subrefined_ngbd_n;
  ierr = delete_and_nullify_vector(phi); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(subrefined_phi); CHKERRXX(ierr);

  if(ngbd_c != NULL)
    delete ngbd_c;
  if(faces != NULL)
    delete faces;

  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
