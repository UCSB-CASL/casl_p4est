// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

enum shape_t {
  CIRCLE,
  FLOWER
};

const double xyz_m[P4EST_DIM]         = {DIM(-1.25, -1.25, -1.25)};
const double xyz_M[P4EST_DIM]         = {DIM(+1.25, +1.25, +1.25)};
const unsigned int default_lmin       = 3;
const unsigned int default_lmax       = 5;
const double default_r0               = 0.5;
const int default_nx                  = 1;
const int default_ny                  = 1;
#ifdef P4_TO_P8
const int default_nz                  = 1;
#endif
const std::string default_export_dir_root = "/home/regan/workspace/projects/two_phase_flow/face_extrapolation_" + std::to_string(P4EST_DIM) + "d";
const double default_mu_m             = 1.0;
const double default_mu_p             = 1.0;
const unsigned int default_ngrids     = 3;
static const shape_t default_shape    = CIRCLE;
const extrapolation_technique default_extrapolation_technique = PSEUDO_TIME;
const unsigned int default_extrapolation_niter = 20;
const unsigned int default_extrapolation_degree = 1;

class LEVEL_SET : public CF_DIM {
  shape_t shape;
  const double radius;
public:
  LEVEL_SET(const double& rad_) : radius(rad_) { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (shape) {
    case CIRCLE:
      return radius - sqrt(SUMD(SQR(x - (xyz_m[0] + xyz_M[0])/2.0), SQR(y - (xyz_m[1] + xyz_M[1])/2.0), SQR(z - (xyz_m[2] + xyz_M[2])/2.0)));
      break;
    case FLOWER:
    {
#ifdef P4_TO_P8
      double phi, theta;
      if(fabs(sqrt(SQR(x) + SQR(y)) < EPS*MAX(xyz_M[0] - xyz_m[0], xyz_M[1] - xyz_m[1])))
        phi = M_PI_2;
      else
        phi = acos(x/sqrt(SQR(x) + SQR(y))) + (y > 0.0 ? 0.0 : M_PI);
      theta = (sqrt(SQR(x) + SQR(y) + SQR(z)) > EPS*MAX(xyz_M[0] - xyz_m[0], xyz_M[1] - xyz_m[1], xyz_M[2] - xyz_m[2]) ? acos(z/sqrt(SQR(x) + SQR(y) + SQR(z))) : 0.0);
      return SQR(x) + SQR(y) + SQR(z) - SQR(0.7 + 0.2*(1.0 - 0.2*cos(6.0*phi))*(1.0 - cos(6.0*theta)));
#else
      double alpha = 0.02*sqrt(5.0);
      double tt;
      if(fabs(x  -alpha) < EPS*(xyz_M[0] - xyz_m[0]))
        tt = (y > alpha + EPS*(xyz_M[1] - xyz_m[1]) ? 0.5*M_PI : (y < alpha - EPS*(xyz_M[1] - xyz_m[1]) ? -0.5*M_PI : 0.0));
      else
        tt = (x > alpha + EPS*(xyz_M[0] - xyz_m[0]) ? atan((y - alpha)/(x - alpha)) : ( y >= alpha ? M_PI + atan((y - alpha)/(x - alpha)) : -M_PI + atan((y - alpha)/(x - alpha))));
      return SQR(x - alpha) + SQR(y - alpha) - SQR(0.5 + 0.2*sin(5*tt));
#endif
      break;
    }
    default:
      throw std::invalid_argument("main_test_viscosity: choose a valid level set.");
      break;
    }
  }
  double operator()(const double *xyz) const
  {
    return operator()( DIM(xyz[0], xyz[1], xyz[2]));
  }

  inline void set_shape(const shape_t &shape_in) { shape = shape_in; }
  inline shape_t get_shape() const { return  shape; }

};

class EXACT_SOLUTION{
  double mu_minus;
  double mu_plus;
public:
  inline double u_minus(const double *xyz) const
  {
    return MULTD(sin(xyz[0]), cos(xyz[1]), sin(xyz[2]));
  }
  inline double dx_u_minus(const double *xyz) const
  {
    return MULTD(cos(xyz[0]), cos(xyz[1]), sin(xyz[2]));
  }
  inline double dy_u_minus(const double *xyz) const
  {
    return MULTD(sin(xyz[0]), -sin(xyz[1]), sin(xyz[2]));
  }
#ifdef P4_TO_P8
  inline double dz_u_minus(const double *xyz) const
  {
    return MULTD(sin(xyz[0]), cos(xyz[1]), cos(xyz[2]));
  }
#endif
  inline double v_minus(const double *xyz) const
  {
    return (3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*log(1.0 + SQR(xyz[1]))ONLY3D(*atan(xyz[2]/2.5));
  }
  inline double dx_v_minus(const double *xyz) const
  {
    return (9.0*SQR(xyz[0]) - 2.5*pow(xyz[0], 4))*log(1.0 + SQR(xyz[1]))ONLY3D(*atan(xyz[2]/2.5));
  }
  inline double dy_v_minus(const double *xyz) const
  {
    return (3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*(2.0*xyz[1]/(1.0 + SQR(xyz[1])))ONLY3D(*atan(xyz[2]/2.5));
  }
#ifdef P4_TO_P8
  inline double dz_v_minus(const double *xyz) const
  {
    return (3.0*pow(xyz[0], 3.0) - 0.5*pow(xyz[0], 5.0))*log(1.0 + SQR(xyz[1]))ONLY3D(*(1.0/2.5)/(1.0 + SQR(xyz[2]/2.5)));
  }
#endif

#ifdef P4_TO_P8
  inline double w_minus(const double *xyz) const
  {
    return sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2]));
  }
  inline double dx_w_minus(const double *xyz) const
  {
    return (0.5*cos(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2])) + sin(0.5*(xyz[0] - xyz[2]))*(sin(xyz[1]) + sin(xyz[0] + xyz[1])*atan(xyz[2])));
  }
  inline double dy_w_minus(const double *xyz) const
  {
    return sin(0.5*(xyz[0] - xyz[2]))*(xyz[0]*cos(xyz[1]) + sin(xyz[0] + xyz[1])*atan(xyz[2]));
  }
  inline double dz_w_minus(const double *xyz) const
  {
    return (-0.5*cos(0.5*(xyz[0] - xyz[2]))*(xyz[0]*sin(xyz[1]) - cos(xyz[0] + xyz[1])*atan(xyz[2])) + sin(0.5*(xyz[0] - xyz[2]))*(-cos(xyz[0] + xyz[1])*(1.0/(1.0 + SQR(xyz[2])))));
  }
#endif

  inline double u_plus(const double *xyz) const
  {
    return (pow((xyz[1] - xyz[0])/2.5, 3.0) + cos(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*(xyz[2] - 2.0*xyz[0] + cos(3.0*xyz[2] - xyz[1]))));
  }
  inline double dx_u_plus(const double *xyz) const
  {
    return (-3.0*SQR((xyz[1] - xyz[0])/2.5)*(1.0/2.5) - 2.0*sin(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*(-2.0)));
  }
  inline double dy_u_plus(const double *xyz) const
  {
    return (3.0*SQR((xyz[1] - xyz[0])/2.5)*(1.0/2.5) + sin(2.0*xyz[0] - xyz[1]) ONLY3D( + (5.0/3.0)*sin(3.0*xyz[2] - xyz[1])));
  }
#ifdef P4_TO_P8
  inline double dz_u_plus(const double *xyz) const
  {
    return (5.0/3.0)*(1.0 - 3.0*sin(3.0*xyz[2] - xyz[1]));
  }
#endif

  inline double v_plus(const double *xyz) const
  {
    return (sin(3.0*xyz[0] - 2.0*xyz[1]) + log(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])) ONLY3D( + cos(1.7*xyz[2] - 0.3*xyz[0])));
  }
  inline double dt_v_plus(const double *xyz) const
  {
    return(sin(3.0*xyz[0] - 2.0*xyz[1]) + log(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])) ONLY3D( + cos(1.7*xyz[2] - 0.3*xyz[0])));
  }
  inline double dx_v_plus(const double *xyz) const
  {
    return (3.0*cos(3.0*xyz[0] - 2.0*xyz[1]) + (-2.0*1.2*(0.5*xyz[1] - 1.2*xyz[0]))/(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])) ONLY3D( + 0.3*sin(1.7*xyz[2] - 0.3*xyz[0])));
  }
  inline double dy_v_plus(const double *xyz) const
  {
    return (-2.0*cos(3.0*xyz[0] - 2.0*xyz[1]) + 2.0*0.5*(0.5*xyz[1] - 1.2*xyz[0])/(1.0 + SQR(0.5*xyz[1] - 1.2*xyz[0])));
  }
#ifdef P4_TO_P8
  inline double dz_v_plus(const double *xyz) const
  {
    return (- 1.7*sin(1.7*xyz[2] - 0.3*xyz[0]));
  }
  inline double w_plus(const double *xyz) const
  {
    return (0.1*xyz[0]*xyz[0]*xyz[0]*xyz[1] + 2.0*xyz[2]*cos(xyz[1]) - xyz[1]*sin(xyz[0] + xyz[2]));
  }
  inline double dt_w_plus(const double *xyz) const
  {
    return (0.1*xyz[0]*xyz[0]*xyz[0]*xyz[1] + 2.0*xyz[2]*cos(xyz[1]) - xyz[1]*sin(xyz[0] + xyz[2]));
  }
  inline double dx_w_plus(const double *xyz) const
  {
    return (0.3*xyz[0]*xyz[0]*xyz[1] - xyz[1]*cos(xyz[0] + xyz[2]));
  }
  inline double dy_w_plus(const double *xyz) const
  {
    return(0.1*xyz[0]*xyz[0]*xyz[0] - 2.0*xyz[2]*sin(xyz[1]) - sin(xyz[0] + xyz[2]));
  }
  inline double dz_w_plus(const double *xyz) const
  {
    return (2.0*cos(xyz[1]) - xyz[1]*cos(xyz[0] + xyz[2]));
  }
#endif

  inline double jump_in_solution(const unsigned char &dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return (u_plus(xyz) - u_minus(xyz));
      break;
    case dir::y:
      return (v_plus(xyz) - v_minus(xyz));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return (w_plus(xyz) - w_minus(xyz));
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::jump_in_solution: unknown cartesian direction");
    }
  }
  inline double jump_in_flux(const unsigned char &dir, const unsigned char &der, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_u_plus(xyz) - mu_minus*dx_u_minus(xyz));
        break;
      case dir::y:
        return (mu_plus*dy_u_plus(xyz) - mu_minus*dy_u_minus(xyz));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_u_plus(xyz) - mu_minus*dz_u_minus(xyz));
        break;
#endif
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
      }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_v_plus(xyz) - mu_minus*dx_v_minus(xyz));
        break;
      case dir::y:
        return (mu_plus*dy_v_plus(xyz) - mu_minus*dy_v_minus(xyz));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_v_plus(xyz) - mu_minus*dz_v_minus(xyz));
        break;
#endif
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
      }
      break;
#ifdef P4_TO_P8
    case dir::z:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_w_plus(xyz) - mu_minus*dx_w_minus(xyz));
        break;
      case dir::y:
        return (mu_plus*dy_w_plus(xyz) - mu_minus*dy_w_minus(xyz));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_w_plus(xyz) - mu_minus*dz_w_minus(xyz));
        break;
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
#endif
      }
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction");
    }
  }

  EXACT_SOLUTION() : mu_minus(1.0), mu_plus(1.0) {}

  inline void set_viscosities(const double& mu_m_, const double& mu_p_) { mu_minus = mu_m_; mu_plus = mu_p_; }

} exact_solution;

struct BCWALLTYPE : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLVALUE : CF_DIM {
  const LEVEL_SET &ls;
  const EXACT_SOLUTION &ex;
  const unsigned char dir;
  BCWALLVALUE(const LEVEL_SET &ls_, const EXACT_SOLUTION &ex_, const unsigned char& dir_) : ls(ls_), ex(ex_), dir(dir_) {}
  double operator()(const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return (ls(xyz) <= 0.0 ? ex.u_minus(xyz) : ex.u_plus(xyz));
      break;
    case dir::y:
      return (ls(xyz) <= 0.0 ? ex.v_minus(xyz) : ex.v_plus(xyz));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return (ls(xyz) <= 0.0 ? ex.w_minus(xyz) : ex.w_plus(xyz));
      break;
#endif
    default:
      throw std::runtime_error("I'm so tired...");
      break;
    }
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return operator()(xyz);
  }
};

struct wrapper_ex_to_cf : public CF_DIM
{
  const EXACT_SOLUTION& ex;
  const unsigned char dir;
  const unsigned char sign;
  wrapper_ex_to_cf(const EXACT_SOLUTION& ex_, const unsigned char& dir_, const unsigned char& sign_) :
    ex(ex_), dir(dir_), sign(sign_) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    switch (dir) {
    case dir::x:
      return (sign == 0 ? ex.u_minus(xyz) : ex.u_plus(xyz));
      break;
    case dir::y:
      return (sign == 0 ? ex.v_minus(xyz) : ex.v_plus(xyz));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return (sign == 0 ? ex.w_minus(xyz) : ex.w_plus(xyz));
      break;
#endif
    default:
      throw std::runtime_error("I'm so tired...");
      break;
    }
  }
};


//void evaluate_errors(my_p4est_two_phase_flows_t *solver, double error_vnp1_minus[P4EST_DIM], double error_vnp1_plus[P4EST_DIM], double *error_at_faces_minus_p[P4EST_DIM], double *error_at_faces_plus_p[P4EST_DIM])
//{
//  PetscErrorCode ierr;
//  Vec *vnp1_faces = solver->get_test_vnp1_faces();
//  const double *vnp1_faces_p;
//  double xyz_face[P4EST_DIM];
//  my_p4est_faces_t* faces = solver->get_faces();
//  my_p4est_interpolation_nodes_t* interp_phi = solver->get_interp_phi();
//  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
//    error_vnp1_minus[dir] = error_vnp1_plus[dir] = 0.0;
//    ierr = VecGetArrayRead(vnp1_faces[dir], &vnp1_faces_p); CHKERRXX(ierr);
//    for (p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx) {
//      faces->xyz_fr_f(f_idx, dir, xyz_face);
//      const double phi = (*interp_phi)(xyz_face);
//      switch (dir) {
//      case dir::x:
//        if(phi <= 0.0)
//        {
//          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.u_minus(xyz_face)));
//          error_at_faces_minus_p[0][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.u_minus(xyz_face));
//          error_at_faces_plus_p[0][f_idx] = 0.0;
//        }
//        else
//        {
//          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.u_plus(xyz_face)));
//          error_at_faces_minus_p[0][f_idx] = 0.0;
//          error_at_faces_plus_p[0][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.u_plus(xyz_face));
//        }
//        break;
//      case dir::y:
//        if(phi <= 0.0)
//        {
//          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.v_minus(xyz_face)));
//          error_at_faces_minus_p[1][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.v_minus(xyz_face));
//          error_at_faces_plus_p[1][f_idx] = 0.0;
//        }
//        else
//        {
//          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.v_plus(xyz_face)));
//          error_at_faces_minus_p[1][f_idx] = 0.0;
//          error_at_faces_plus_p[1][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.v_plus(xyz_face));
//        }
//        break;
//#ifdef P4_TO_P8
//      case dir::z:
//        if(phi <= 0.0)
//        {
//          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.w_minus(xyz_face)));
//          error_at_faces_minus_p[2][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.w_minus(xyz_face));
//          error_at_faces_plus_p[2][f_idx] = 0.0;
//        }
//        else
//        {
//          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.w_plus(xyz_face)));
//          error_at_faces_minus_p[2][f_idx] = 0.0;
//          error_at_faces_plus_p[2][f_idx] = fabs(vnp1_faces_p[f_idx] - exact_solution.w_plus(xyz_face));
//        }
//        break;
//#endif
//      default:
//        throw std::runtime_error("evaluate_errors: unknown cartesian direction");
//        break;
//      }
//    }
//    ierr = VecRestoreArrayRead(vnp1_faces[dir], &vnp1_faces_p); CHKERRXX(ierr);
//  }
//  int mpiret;
//  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vnp1_minus,  P4EST_DIM, MPI_DOUBLE, MPI_MAX, solver->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
//  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vnp1_plus,   P4EST_DIM, MPI_DOUBLE, MPI_MAX, solver->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
//  return;
//}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  // computational grid parameters
  cmd.add_option("lmin", "first min level of the trees, default is " + std::to_string(default_lmin));
  cmd.add_option("lmax", "first max level of the trees, default is " + std::to_string(default_lmax));
  cmd.add_option("nx", "number of trees in the x-direction. The default value is " + std::to_string(default_nx) + " (length of domain is 2.5)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is " + std::to_string(default_ny) + " (height of domain is 2.5)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is " + std::to_string(default_nz) + " (width of domain is 2.5)");
#endif
  cmd.add_option("r0", "r0 parameter in levelset (radius if circle). The default value is " + std::to_string(default_r0) + " (domain is 2.5 x 2.5 (x 2.5))");
  // physical parameters for the simulations
  cmd.add_option("mu_m",  "viscosity coefficient in the negative domain (default is " + std::to_string(default_mu_m) + std::string(")"));
  cmd.add_option("mu_p",  "viscosity coefficient in the positive domain (default is " + std::to_string(default_mu_p) + std::string(")"));
  cmd.add_option("shape", "the shape of the interface (0: circle, 1: flower), default shape is " + std::string(default_shape == CIRCLE ? "circle)" : "flower.)" ));
  cmd.add_option("extrapolation", "face-based extrapolation technique (0: pseudo-time, 1: explicit iterative), default is " + std::string(default_extrapolation_technique == PSEUDO_TIME ? "pseudo time" : "explicit iterative"));
  cmd.add_option("extrapolation_niter", "number of iterations used for the face-based extrapolation technique, default is " + std::to_string(default_extrapolation_niter));
  cmd.add_option("extrapolation_degree", "degree of the face-based extrapolation technique (0: constant, 1: extrapolate normal derivatives too), default is " + std::to_string(default_extrapolation_degree));
  // exportation control
  cmd.add_option("no_save_vtk",   "does not save vtk visualization files if present");
  cmd.add_option("ngrids",        "number of successively finer grids to consider (default is " + std::to_string(default_ngrids) + ")");
  cmd.add_option("export_dir",    "root directory for exportation, subfolders are created (default is "  + default_export_dir_root);

  std::string extra_info = "More details to come when I really have nothing better to do. If you really need it: reach out! (Raphael)";
  if(cmd.parse(argc, argv, extra_info))
    return 0;

  if(cmd.contains("shape") && cmd.get<int>("shape") != 0 && cmd.get<int>("shape") != 1)
    throw std::invalid_argument("Come on: choose a valid shape for the interface...");

  const int n_tree_xyz [P4EST_DIM] = {DIM(cmd.get<int>("nx", default_nx), cmd.get<int>("ny", default_ny), cmd.get<int>("nz", default_nz))};
  const std::string root_export_dir = cmd.get<std::string>("export_dir", default_export_dir_root);
  const bool save_vtk = !cmd.contains("no_save_vtk");
  const int periodic[P4EST_DIM]     = { DIM(0, 0, 0) };
  const unsigned int lmin = cmd.get<unsigned int>("lmin", default_lmin);
  const unsigned int lmax = cmd.get<unsigned int>("lmax", default_lmax);
  const double mu_m = cmd.get<double>("mu_m", default_mu_m);
  const double mu_p = cmd.get<double>("mu_p", default_mu_p);
  const extrapolation_technique extrapolation = (cmd.contains("extrapolation") ? (cmd.get<unsigned int>("extrapolation") == 0 ? PSEUDO_TIME : EXPLICIT_ITERATIVE) : default_extrapolation_technique);
  const unsigned int ext_nsteps               = cmd.get<unsigned int>("extrapolation_niter", default_extrapolation_niter);
  const unsigned int ext_degree               = cmd.get<unsigned int>("extrapolation_degree", default_extrapolation_degree);

  PetscErrorCode ierr;

  const double r0 = cmd.get<double>("r0", default_r0);
  LEVEL_SET ls(r0);
  ls.set_shape(cmd.contains("shape") ? (cmd.get<int>("shape") == 0 ? CIRCLE : FLOWER) : default_shape);
  exact_solution.set_viscosities(mu_m, mu_p);
  wrapper_ex_to_cf *solution_m[P4EST_DIM] = {DIM(new wrapper_ex_to_cf(exact_solution, dir::x, 0), new wrapper_ex_to_cf(exact_solution, dir::y, 0), new wrapper_ex_to_cf(exact_solution, dir::z, 0))};
  wrapper_ex_to_cf *solution_p[P4EST_DIM] = {DIM(new wrapper_ex_to_cf(exact_solution, dir::x, 1), new wrapper_ex_to_cf(exact_solution, dir::y, 1), new wrapper_ex_to_cf(exact_solution, dir::z, 1))};
  CF_DIM *solution_m_cf[P4EST_DIM] = {DIM(solution_m[0], solution_m[1], solution_m[2])};
  CF_DIM *solution_p_cf[P4EST_DIM] = {DIM(solution_p[0], solution_p[1], solution_p[2])};

  BCWALLTYPE bc_wtype;
  BCWALLVALUE bc_wvalue_u(ls, exact_solution, dir::x);
  BCWALLVALUE bc_wvalue_v(ls, exact_solution, dir::y);
#ifdef P4_TO_P8
  BCWALLVALUE bc_wvalue_w(ls, exact_solution, dir::z);
#endif
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  bc_v[0].setWallTypes(bc_wtype); bc_v[0].setWallValues(bc_wvalue_u);
  bc_v[1].setWallTypes(bc_wtype); bc_v[1].setWallValues(bc_wvalue_v);
#ifdef P4_TO_P8
  bc_v[2].setWallTypes(bc_wtype); bc_v[2].setWallValues(bc_wvalue_w);
#endif
  BoundaryConditionsDIM dummy_p;


  const unsigned int ngrids = cmd.get<unsigned int>("ngrids", default_ngrids);

  for (unsigned int k_grid = 0; k_grid < ngrids; ++k_grid) {

    my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
    my_p4est_brick_t* brick                           = NULL;
    p4est_connectivity_t *connectivity                = NULL;
    splitting_criteria_cf_and_uniform_band_t* data    = NULL;

    const double dxyzmin  = MAX(DIM((xyz_M[0] - xyz_m[0])/(double)n_tree_xyz[0], (xyz_M[1] - xyz_m[1])/(double)n_tree_xyz[1], (xyz_M[2] - xyz_m[2])/(double)n_tree_xyz[2])) / (1 << lmax);
    double uniform_band   = .15*r0;
    uniform_band         /= dxyzmin;

    brick = new my_p4est_brick_t;
    connectivity = my_p4est_brick_new(n_tree_xyz, xyz_m, xyz_M, brick, periodic);
    data  = new splitting_criteria_cf_and_uniform_band_t(lmin + k_grid, lmax + k_grid, &ls, uniform_band);

    p4est_t* p4est_comp = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est_comp->user_pointer = (void*) data;

    for(unsigned int l = 0; l < lmax + k_grid; ++l)
    {
      my_p4est_refine(p4est_comp, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
      my_p4est_partition(p4est_comp, P4EST_FALSE, NULL);
    }
    /* create the initial forest at time nm1 */
    p4est_balance(p4est_comp, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_comp, P4EST_FALSE, NULL);

    p4est_ghost_t *ghost_comp = my_p4est_ghost_new(p4est_comp, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_comp, ghost_comp);

    p4est_nodes_t *nodes_comp = my_p4est_nodes_new(p4est_comp, ghost_comp);
    my_p4est_hierarchy_t *hierarchy_comp = new my_p4est_hierarchy_t(p4est_comp, ghost_comp, brick);
    my_p4est_node_neighbors_t *ngbd_comp = new my_p4est_node_neighbors_t(hierarchy_comp, nodes_comp); ngbd_comp->init_neighbors();

    p4est_comp->user_pointer = (void*) data;
    my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_comp);
    my_p4est_faces_t *faces_comp = new my_p4est_faces_t(p4est_comp, ghost_comp, brick, ngbd_c, true);
    P4EST_ASSERT(faces_comp->finest_face_neighborhoods_are_valid());


    /* build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
     * the REINITIALIZED levelset on the interface-capturing grid
     */
    splitting_criteria_cf_t* data_fine = new splitting_criteria_cf_t(lmin + k_grid, lmax + k_grid + 1, &ls);
    p4est_t* p4est_fine = p4est_copy(p4est_comp, P4EST_FALSE);
    p4est_fine->user_pointer = (void*) data_fine;
    p4est_refine(p4est_fine, P4EST_FALSE, refine_levelset_cf, NULL);
    p4est_ghost_t* ghost_fine = my_p4est_ghost_new(p4est_fine, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_fine, ghost_fine);
    my_p4est_hierarchy_t* hierarchy_fine = new my_p4est_hierarchy_t(p4est_fine, ghost_fine, brick);
    p4est_nodes_t* nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);
    my_p4est_node_neighbors_t* ngbd_n_fine = new my_p4est_node_neighbors_t(hierarchy_fine, nodes_fine); ngbd_n_fine->init_neighbors();

    Vec fine_phi;
    ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &fine_phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_fine, nodes_fine, ls, fine_phi);
    if(ls.get_shape() == FLOWER)
    {
      my_p4est_level_set_t ls_fine(ngbd_n_fine);
      ls_fine.reinitialize_2nd_order(fine_phi);
    }

    two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_comp, ngbd_comp, faces_comp, ngbd_n_fine);
    bool second_order_phi = true;
    two_phase_flow_solver->set_phi(fine_phi, second_order_phi);
    two_phase_flow_solver->set_dynamic_viscosities(mu_m, mu_p);
    two_phase_flow_solver->set_densities(1.0, 1.0); // irrelevant for this test
    two_phase_flow_solver->set_uniform_bands(uniform_band, uniform_band);
    two_phase_flow_solver->set_bc(bc_v, &dummy_p);

    // initialize face fields
    two_phase_flow_solver->set_face_velocities_np1(solution_m_cf, solution_p_cf);
    const std::string export_dir = root_export_dir + std::string(ls.get_shape() == CIRCLE ? "/circle" : "/flower")
        + "/macromesh_" + std::to_string(n_tree_xyz[0]) + "_" + std::to_string(n_tree_xyz[1]) ONLY3D(+ "_" + std::to_string(n_tree_xyz[2]))
        + "/mu_m_" + std::to_string(mu_m) + "_mu_p_" + std::to_string(mu_p);
    if(save_vtk && create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
      throw std::runtime_error("two_phase_face_extrapolation: " + std::to_string(P4EST_DIM) + "d, could not create exportation directory " + export_dir);

    // data-to-provide-to-the-solver stuff
    Vec fine_jump_mu_grad_v, fine_jump_u;
    double *fine_jump_mu_grad_v_p, *fine_jump_u_p;
    ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, SQR_P4EST_DIM, &fine_jump_mu_grad_v); CHKERRXX(ierr);
    ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, P4EST_DIM, &fine_jump_u); CHKERRXX(ierr);
    ierr = VecGetArray(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
    ierr = VecGetArray(fine_jump_u, &fine_jump_u_p);     CHKERRXX(ierr);
    for (size_t fine_node_idx = 0; fine_node_idx < nodes_fine->indep_nodes.elem_count; ++fine_node_idx) {
      double xyz_node[P4EST_DIM]; node_xyz_fr_n(fine_node_idx, p4est_fine, nodes_fine, xyz_node);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      {
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          fine_jump_mu_grad_v_p[SQR_P4EST_DIM*fine_node_idx + P4EST_DIM*dir + der] = exact_solution.jump_in_flux(dir, der, xyz_node);
        fine_jump_u_p[P4EST_DIM*fine_node_idx + dir] = exact_solution.jump_in_solution(dir, xyz_node);
      }
    }
    ierr = VecRestoreArray(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(fine_jump_u, &fine_jump_u_p);     CHKERRXX(ierr);
    two_phase_flow_solver->set_fine_jump_mu_grad_v(fine_jump_mu_grad_v);
    two_phase_flow_solver->set_fine_jump_velocity(fine_jump_u);

    two_phase_flow_solver->extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(extrapolation, ext_nsteps, ext_degree);
    two_phase_flow_solver->compute_velocity_at_nodes();
    if(save_vtk)
      two_phase_flow_solver->save_vtk((export_dir + "/illustration_iter_" + std::to_string(k_grid)).c_str()); //  , true, (export_dir + "/fine_illustration_iter_" + std::to_string(k_grid)).c_str());

    delete two_phase_flow_solver;
    delete data;
    delete data_fine;
  }

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    delete solution_m[dir];
    delete solution_p[dir];
  }

  return 0;
}
