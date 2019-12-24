// System
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_vtk.h>
#endif


#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

static const double xmin = -1.25;
static const double xmax = +1.25;
static const double ymin = -1.25;
static const double ymax = +1.25;
#ifdef P4_TO_P8
static const double zmin = -1.25;
static const double zmax = +1.25;
#endif

static const double r0   = 0.5;

static const double rho       = 1.0;
static const double mu_minus  = 1.0;
static const double mu_plus   = 1.0;
static const bool implicit    = false;
static double tn;
static double dt;

class EXACT_SOLUTION{
public:
  double u_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(sin(x), cos(y), sin(z));
  }
  double dt_u_minus(DIM(double x, double y, double z))
  {
    return -sin(tn+(implicit?dt:0.0))*MULTD(sin(x), cos(y), sin(z));
  }
  double dx_u_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(cos(x), cos(y), sin(z));
  }
  double dy_u_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(sin(x), -sin(y), sin(z));
  }
#ifdef P4_TO_P8
  double dz_u_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(sin(x), cos(y), cos(z));
  }
#endif
  double laplace_u_minus(DIM(double x, double y, double z))
  {
    return -cos(tn+(implicit?dt:0.0))*((double) P4EST_DIM)*MULTD(sin(x), cos(y), sin(z));
  }
  double v_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5));
  }
  double dt_v_minus(DIM(double x, double y, double z))
  {
    return -sin(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5));
  }
  double dx_v_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(9.0*SQR(x)-2.5*pow(x, 4))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5));
  }
  double dy_v_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*(2.0*y/(1.0+SQR(y)))ONLY3D(*atan(z/2.5));
  }
#ifdef P4_TO_P8
  double dz_v_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))ONLY3D(*(1.0/2.5)/(1.0+SQR(z/2.5)));
  }
#endif
  double laplace_v_minus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*((18.0*x - 10*pow(x, 3.0))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5))
                                      + (3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*(2.0*(1.0-SQR(y))/(SQR(1.0+SQR(y))))ONLY3D(*atan(z/2.5))
                                  #ifdef P4_TO_P8
                                      + (3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))*(-2.0*z/(SQR(2.5)))/(SQR(1.0+SQR(z/2.5)))
                                  #endif
                                      );
  }

  double u_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(sin(x), cos(y), sin(z));
  }
  double dt_u_plus(DIM(double x, double y, double z))
  {
    return -sin(tn+(implicit?dt:0.0))*MULTD(sin(x), cos(y), sin(z));
  }
  double dx_u_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(cos(x), cos(y), sin(z));
  }
  double dy_u_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(sin(x), -sin(y), sin(z));
  }
#ifdef P4_TO_P8
  double dz_u_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*MULTD(sin(x), cos(y), cos(z));
  }
#endif
  double laplace_u_plus(DIM(double x, double y, double z))
  {
    return -cos(tn+(implicit?dt:0.0))*((double) P4EST_DIM)*MULTD(sin(x), cos(y), sin(z));
  }
  double v_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5));
  }
  double dt_v_plus(DIM(double x, double y, double z))
  {
    return -sin(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5));
  }
  double dx_v_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(9.0*SQR(x)-2.5*pow(x, 4))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5));
  }
  double dy_v_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*(2.0*y/(1.0+SQR(y)))ONLY3D(*atan(z/2.5));
  }
#ifdef P4_TO_P8
  double dz_v_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*(3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))ONLY3D(*(1.0/2.5)/(1.0+SQR(z/2.5)));
  }
#endif
  double laplace_v_plus(DIM(double x, double y, double z))
  {
    return cos(tn+(implicit?dt:0.0))*((18.0*x - 10*pow(x, 3.0))*log(1.0+SQR(y))ONLY3D(*atan(z/2.5))
                                      + (3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*(2.0*(1.0-SQR(y))/(SQR(1.0+SQR(y))))ONLY3D(*atan(z/2.5))
                                  #ifdef P4_TO_P8
                                      + (3.0*pow(x, 3.0)-0.5*pow(x, 5.0))*log(1.0+SQR(y))*(-2.0*z/(SQR(2.5)))/(SQR(1.0+SQR(z/2.5)))
                                  #endif
                                      );
  }

//  double u_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+exp(-SQR(tn+(implicit?dt:0.0)-0.5)))*(pow(y-x, 3.0) + cos(2.0*x-y) ONLY3D( + (5.0/3.0)*(z-2.0*x + cos(3.0*z-y))));
//  }
//  double dt_u_plus(DIM(double x, double y, double z))
//  {
//    return (exp(-SQR(tn+(implicit?dt:0.0)-0.5))*(1-2*(tn+(implicit?dt:0.0))))*(pow(y-x, 3.0) + cos(2.0*x-y) ONLY3D( + (5.0/3.0)*(z-2.0*x + cos(3.0*z-y))));
//  }
//  double dx_u_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+exp(-SQR(tn+(implicit?dt:0.0)-0.5)))*(-3.0*SQR(y-x) - 2.0*sin(2.0*x-y) ONLY3D( + (5.0/3.0)*(-2.0)));
//  }
//  double dy_u_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+exp(-SQR(tn+(implicit?dt:0.0)-0.5)))*(3.0*SQR(y-x) + sin(2.0*x-y) ONLY3D( + (5.0/3.0)*sin(3.0*z-y)));
//  }
//#ifdef P4_TO_P8
//  double dz_u_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+exp(-SQR(tn+(implicit?dt:0.0)-0.5)))*(pow(y-x, 3.0) + cos(2.0*x-y) ONLY3D( + (5.0/3.0)*(1.0 - 3.0*sin(3.0*z-y))));
//  }
//#endif
//  double laplace_u_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+exp(-SQR(tn+(implicit?dt:0.0)-0.5)))*(12.0*(y-x)-5.0*cos(2.0*x-y) ONLY3D( + (5.0/3.0)*(-10.0*cos(3.0*z-y))));
//  }

//  double v_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+0.3*cos(1.5*(tn+(implicit?dt:0.0))))*(sin(3.0*x-2.0*y) + log(1.0 + SQR(0.5*y-1.2*x)) ONLY3D( + cos(1.7*z-0.3*x)));
//  }
//  double dt_v_plus(DIM(double x, double y, double z))
//  {
//    return (-0.3*1.5*sin(1.5*(tn+(implicit?dt:0.0))))*(sin(3.0*x-2.0*y) + log(1.0 + SQR(0.5*y-1.2*x)) ONLY3D( + cos(1.7*z-0.3*x)));
//  }
//  double dx_v_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+0.3*cos(1.5*(tn+(implicit?dt:0.0))))*(3.0*cos(3.0*x-2.0*y) + (-2.0*1.2*(0.5*y-1.2*x))/(1.0 + SQR(0.5*y-1.2*x)) ONLY3D( + 0.3*sin(1.7*z-0.3*x)));
//  }
//  double dy_v_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+0.3*cos(1.5*(tn+(implicit?dt:0.0))))*(-2.0*cos(3.0*x-2.0*y) + (2.0*0.5*(0.5*y-1.2*x))/(1.0 + SQR(0.5*y-1.2*x)));
//  }
//#ifdef P4_TO_P8
//  double dz_v_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+0.3*cos(1.5*(tn+(implicit?dt:0.0))))*(- 1.7*sin(1.7*z-0.3*x));
//  }
//#endif
//  double laplace_v_plus(DIM(double x, double y, double z))
//  {
//    return (1.0+0.3*cos(1.5*(tn+(implicit?dt:0.0))))*((-SQR(3.0)-SQR(-2.0))*sin(3.0*x-2.0*y) +
//                                                      (2.0*(SQR(-1.2) + SQR(0.5))*(1.0-SQR(0.5*y - 1.2*x)))/(SQR(1.0+SQR(0.5*y-1.2*x)))
//                                                      ONLY3D(+ (-SQR(1.7) - SQR(0.3))*cos(1.7*z-0.3*x)));
//  }

  double source_term(const unsigned char &dir, DIM(const double &x, const double &y, const double &z), const my_p4est_interpolation_nodes_t &interp_phi)
  {
    switch (dir) {
    case dir::x:
      return ((interp_phi(DIM(x, y, z)) > 0.0) ? (dt_u_plus(DIM(x, y, z)) - mu_plus*laplace_u_plus(DIM(x, y, z))) : (dt_u_minus(DIM(x, y, z)) - mu_minus*laplace_u_minus(DIM(x, y, z))));
      break;
    case dir::y:
      return ((interp_phi(DIM(x, y, z)) > 0.0) ? (dt_v_plus(DIM(x, y, z)) - mu_plus*laplace_v_plus(DIM(x, y, z))) : (dt_v_minus(DIM(x, y, z)) - mu_minus*laplace_v_minus(DIM(x, y, z))));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return ((interp_phi(DIM(x, y, z)) > 0.0) ? (dt_w_plus(DIM(x, y, z)) - mu_plus*laplace_w_plus(DIM(x, y, z))) : (dt_w_minus(DIM(x, y, z)) - mu_minus*laplace_w_minus(DIM(x, y, z))));
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::source_term: unknown cartesian direction");
    }
  }

  double jump_in_solution(const unsigned char &dir, DIM(const double &x, const double &y, const double &z))
  {
    switch (dir) {
    case dir::x:
      return (u_plus(DIM(x, y, z)) - u_minus(DIM(x, y, z)));
      break;
    case dir::y:
      return (v_plus(DIM(x, y, z)) - v_minus(DIM(x, y, z)));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return (w_plus(DIM(x, y, z)) - w_minus(DIM(x, y, z)));
      break;
#endif
    default:
      throw std::invalid_argument("exact_solution::jump_in_solution: unknown cartesian direction");
    }
  }

  double jump_in_flux(const unsigned char &dir, const unsigned char &der, DIM(const double &x, const double &y, const double &z))
  {
    switch (dir) {
    case dir::x:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_u_plus(DIM(x, y, z)) - mu_minus*dx_u_minus(DIM(x, y, z)));
        break;
      case dir::y:
        return (mu_plus*dy_u_plus(DIM(x, y, z)) - mu_minus*dy_u_minus(DIM(x, y, z)));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_u_plus(DIM(x, y, z)) - mu_minus*dz_u_minus(DIM(x, y, z)));
        break;
#endif
      default:
        throw std::invalid_argument("exact_solution::jump_in_flux: unknown cartesian direction for derivatives");
      }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return (mu_plus*dx_v_plus(DIM(x, y, z)) - mu_minus*dx_v_minus(DIM(x, y, z)));
        break;
      case dir::y:
        return (mu_plus*dy_v_plus(DIM(x, y, z)) - mu_minus*dy_v_minus(DIM(x, y, z)));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_v_plus(DIM(x, y, z)) - mu_minus*dz_v_minus(DIM(x, y, z)));
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
        return (mu_plus*dx_w_plus(DIM(x, y, z)) - mu_minus*dx_w_minus(DIM(x, y, z)));
        break;
      case dir::y:
        return (mu_plus*dy_w_plus(DIM(x, y, z)) - mu_minus*dy_w_minus(DIM(x, y, z)));
        break;
#ifdef P4_TO_P8
      case dir::z:
        return (mu_plus*dz_w_plus(DIM(x, y, z)) - mu_minus*dz_w_minus(DIM(x, y, z)));
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
} exact_solution;

class LEVEL_SET: public CF_DIM {
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return r0 - sqrt(SUMD(SQR(x-(xmax+xmin)/2), SQR(y-(ymax+ymin)/2), SQR(z-(zmax+zmin)/2))) - 100.0*r0;
  }
} level_set;

class LEVEL_SET_MINUS: public CF_DIM {
public:
  LEVEL_SET_MINUS() { lip = 1.2; }
  double operator()(DIM(double, double, double)) const
  {
    return -1.0;
  }
} level_set_value;

struct BCWALLTYPE_U : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;


#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D {
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_w;
#endif

struct BCWALLVALUE_U : CF_DIM {
  double operator()(DIM(double x, double y, double z)) const
  {
    return exact_solution.u_minus(DIM(x, y, z));
//    return exact_solution.u_plus(DIM(x, y, z));
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_DIM {
  double operator()(DIM(double x, double y, double z)) const
  {
    return exact_solution.v_minus(DIM(x, y, z));
//    return exact_solution.v_plus(DIM(x, y, z));
  }
} bc_wall_value_v;

#ifdef P4_TO_P8
struct BCWALLVALUE_W : CF_3 {
  double operator()(double x, double y, double z)) const
  {
    return exact_solution.w_minus(DIM(x, y, z));
//    return exact_solution.w_plus(DIM(x, y, z));
  }
} bc_wall_value_w;
#endif

void evaluate_errors(my_p4est_two_phase_flows_t *solver, double error_vnp1_minus[P4EST_DIM], double error_vnp1_plus[P4EST_DIM])
{
  PetscErrorCode ierr;
  Vec *vnp1_faces = solver->get_vnp1_faces();
  const double *vnp1_faces_p;
  double xyz_face[P4EST_DIM];
  my_p4est_faces_t* faces = solver->get_faces();
  my_p4est_interpolation_nodes_t* interp_phi = solver->get_interp_phi();
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    error_vnp1_minus[dir] = error_vnp1_plus[dir] = 0.0;
    ierr = VecGetArrayRead(vnp1_faces[dir], &vnp1_faces_p); CHKERRXX(ierr);
    for (p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx) {
      faces->xyz_fr_f(f_idx, dir, xyz_face);
      const double phi = (*interp_phi)(xyz_face);
      switch (dir) {
      case dir::x:
        if(phi <= 0.0)
          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.u_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        else
          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.u_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        break;
      case dir::y:
        if(phi <= 0.0)
          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.v_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        else
          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.v_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        break;
#ifdef P4_TO_P8
      case dir::z:
        if(phi <= 0.0)
          error_vnp1_minus[dir] = MAX(error_vnp1_minus[dir],  fabs(vnp1_faces_p[f_idx] - exact_solution.w_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        else
          error_vnp1_plus[dir]  = MAX(error_vnp1_plus[dir],   fabs(vnp1_faces_p[f_idx] - exact_solution.w_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2]))));
        break;
#endif
      default:
        throw std::runtime_error("evaluate_errors: unknown cartesian direction");
        break;
      }
    }
    ierr = VecRestoreArrayRead(vnp1_faces[dir], &vnp1_faces_p); CHKERRXX(ierr);
  }
  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vnp1_minus,  P4EST_DIM, MPI_DOUBLE, MPI_MAX, solver->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_vnp1_plus,   P4EST_DIM, MPI_DOUBLE, MPI_MAX, solver->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
  return;
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;// computational grid parameters
  cmd.add_option("lmin", "min level of the trees, default is 2");
  cmd.add_option("lmax", "max level of the trees, default is 5");
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is 0.1");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced)");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is 1 (length of domain is 1)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is 1 (height of domain is 1)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is 1 (width of domain is 1)");
#endif
  // physical parameters for the simulations
  cmd.add_option("duration", "the duration of the simulation (tfinal-tstart). If not restarted, tstart = 0.0, default duration is 20.");
  cmd.add_option("vtk_dt", "time step between two vtk exportation, default duration is 0.1");
  cmd.add_option("final", "print only the final errors, if present");

  string extra_info = "More details to come";
  if(cmd.parse(argc, argv, extra_info))
    return 0;

  int lmin, lmax;
  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
  my_p4est_brick_t* brick                           = NULL;
  splitting_criteria_cf_and_uniform_band_t* data    = NULL;

  double uniform_band_m, uniform_band_p;
  int n_tree_xyz [P4EST_DIM];

  const string export_dir               = "/home/raphael/workspace/projects/two_phase_flow/check_viscosity/results_" + to_string(P4EST_DIM) + "d";
  const bool save_vtk                   = true;
  double vtk_dt                         = -1.0;
  if(save_vtk)
  {
    vtk_dt = cmd.get<double>("vtk_dt", +0.1);
    if(vtk_dt <= 0.0)
      throw std::invalid_argument("main_two_phase_flow_" + to_string(P4EST_DIM) + "d.cpp: the value of vtk_dt must be strictly positive.");
  }

  PetscErrorCode ierr;
  const double xyz_min [P4EST_DIM]  = { DIM(xmin, ymin, zmin) };
  const double xyz_max [P4EST_DIM]  = { DIM(xmax, ymax, zmax) };
  const double duration             = cmd.get<double>("duration", 1.0);
  const bool print_final_only       = cmd.contains("final");
  const int periodic[P4EST_DIM]     = { DIM(0, 0, 0) };

  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;

  bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
  bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
  bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif

  lmin                    = cmd.get<int>("lmin", 4+0);
  lmax                    = cmd.get<int>("lmax", 5+0);
  n_tree_xyz[0]           = cmd.get<int>("nx", 1);
  n_tree_xyz[1]           = cmd.get<int>("ny", 1);
#ifdef P4_TO_P8
  n_tree_xyz[2]           = cmd.get<int>("nz", 1);
#endif

  tn = 0.0;
  dt = 0.5*SQR(MIN(DIM(xmax-xmin, ymax-ymin, zmax-zmin))/((double) (1<<lmax)))/MIN(mu_minus, mu_plus);


  uniform_band_m = uniform_band_p = .15*r0;
  const double dxmin      = MAX(DIM((xmax-xmin)/(double)n_tree_xyz[0], (ymax-ymin)/(double)n_tree_xyz[1], (zmax-zmin)/(double)n_tree_xyz[2])) / (1<<lmax);
  uniform_band_m         /= dxmin;
  uniform_band_p         /= dxmin;
  uniform_band_m          = cmd.get<double>("uniform_band", uniform_band_m);
  uniform_band_p          = cmd.get<double>("uniform_band", uniform_band_p);


  p4est_connectivity_t *connectivity;
  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid);brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;

  connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min, xyz_max, brick, periodic);
  if(data != NULL){
    delete data; data = NULL; }
  P4EST_ASSERT(data == NULL);
  data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band_m);

  p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(int l=0; l<lmax; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);


  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1); ngbd_nm1->init_neighbors();

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = (void*) data;

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n); ngbd_n->init_neighbors();
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c, true);
  P4EST_ASSERT(faces_n->finest_face_neighborhoods_are_valid());

  /* build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
   * the REINITIALIZED levelset on the interface-capturing grid
   */
  splitting_criteria_cf_t* data_fine = new splitting_criteria_cf_t(lmin, lmax+1, &level_set);
  p4est_t* p4est_fine = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_fine->user_pointer = (void*) data_fine;
  p4est_refine(p4est_fine, P4EST_FALSE, refine_levelset_cf, NULL);
  p4est_ghost_t* ghost_fine = my_p4est_ghost_new(p4est_fine, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_fine, ghost_fine);
  my_p4est_hierarchy_t* hierarchy_fine = new my_p4est_hierarchy_t(p4est_fine, ghost_fine, brick);
  p4est_nodes_t* nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);
  my_p4est_node_neighbors_t* ngbd_n_fine = new my_p4est_node_neighbors_t(hierarchy_fine, nodes_fine); ngbd_n_fine->init_neighbors();


  Vec fine_phi;
  ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &fine_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_fine, nodes_fine, level_set_value, fine_phi);

  two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces_n, ngbd_n_fine);
  bool second_order_phi = true;
  two_phase_flow_solver->set_phi(fine_phi, second_order_phi);
  two_phase_flow_solver->set_dynamic_viscosities(mu_minus, mu_plus);
  two_phase_flow_solver->set_densities(rho, rho);
  two_phase_flow_solver->set_uniform_bands(uniform_band_m, uniform_band_p);

  // initialize face fields
  my_p4est_interpolation_nodes_t* interp_phi = two_phase_flow_solver->get_interp_phi();
  // time nm1
  tn = tn - (implicit? 2.0:1.0)*dt;
  Vec* vnm1_faces = two_phase_flow_solver->get_vnm1_faces();
  double *vnm1_faces_dir_p;
  double xyz_face[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(vnm1_faces[dir], &vnm1_faces_dir_p); CHKERRXX(ierr);
    for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[dir]; ++face_idx) {
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      switch (dir) {
      case dir::x:
        vnm1_faces_dir_p[face_idx] = (((*interp_phi)(xyz_face) <= 0.0)? exact_solution.u_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])) : exact_solution.u_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])));
        break;
      case dir::y:
        vnm1_faces_dir_p[face_idx] = (((*interp_phi)(xyz_face) <= 0.0)? exact_solution.v_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])) : exact_solution.v_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])));
        break;
#ifdef P4_TO_P8
      case dir::z:
        vnm1_faces_dir_p[face_idx] = (((*interp_phi)(xyz_face) <= 0.0)? exact_solution.w_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])) : exact_solution.w_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])));
        break;
#endif
      default:
        throw std::runtime_error("main_test_viscosity: unknown directon for vnm1");
        break;
      }
    }
    ierr = VecGhostUpdateBegin(vnm1_faces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vnm1_faces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnm1_faces[dir], &vnm1_faces_dir_p); CHKERRXX(ierr);
  }
  // time n
  tn += dt;
  Vec* vn_faces = two_phase_flow_solver->get_vn_faces();
  double *vn_faces_dir_p;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(vn_faces[dir], &vn_faces_dir_p); CHKERRXX(ierr);
    for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[dir]; ++face_idx) {
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      switch (dir) {
      case dir::x:
        vn_faces_dir_p[face_idx] = (((*interp_phi)(xyz_face) <= 0.0)? exact_solution.u_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])) : exact_solution.u_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])));
        break;
      case dir::y:
        vn_faces_dir_p[face_idx] = (((*interp_phi)(xyz_face) <= 0.0)? exact_solution.v_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])) : exact_solution.v_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])));
        break;
#ifdef P4_TO_P8
      case dir::z:
        vn_faces_dir_p[face_idx] = (((*interp_phi)(xyz_face) <= 0.0)? exact_solution.w_minus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])) : exact_solution.w_plus(DIM(xyz_face[0], xyz_face[1], xyz_face[2])));
        break;
#endif
      default:
        throw std::runtime_error("main_test_viscosity: unknown directon for vn");
        break;
      }
    }
    ierr = VecGhostUpdateBegin(vn_faces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vn_faces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_faces[dir], &vn_faces_dir_p); CHKERRXX(ierr);
  }
  if(implicit)
    tn += dt;

  two_phase_flow_solver->set_dt(dt, dt);
  two_phase_flow_solver->set_bc(bc_v, &bc_p);

  if(save_vtk && create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_two_phase_flow_3d: could not create exportation directory %s", export_dir.c_str());
#else
    sprintf(error_msg, "main_two_phase_flow_2d: could not create exportation directory %s", export_dir.c_str());
#endif
    throw std::runtime_error(error_msg);
  }

  int iter = 0, iter_vtk = -1;

  Vec rhs[P4EST_DIM], fine_jump_mu_grad_v;
  double *rhs_p, *fine_jump_mu_grad_v_p;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &rhs[dir], dir); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodesBlock(p4est_fine, nodes_fine, P4EST_DIM*P4EST_DIM, &fine_jump_mu_grad_v); CHKERRXX(ierr);
  two_phase_flow_solver->set_fine_jump_mu_grad_v(fine_jump_mu_grad_v);
  double xyz_node[P4EST_DIM];
  double error_vnp1_minus[P4EST_DIM];
  double error_vnp1_plus[P4EST_DIM];
  double max_error_vnp1_minus[P4EST_DIM];
  double max_error_vnp1_plus[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    max_error_vnp1_minus[dir] = max_error_vnp1_plus[dir] = 0.0;
  }

  while(tn+0.01*dt<duration)
  {
//    if(mpi.rank() == 0)
//      std::cout << "tn = " << tn << std::endl;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        faces_n->xyz_fr_f(f_idx, dir, xyz_face);
        rhs_p[f_idx] = exact_solution.source_term(dir, DIM(xyz_face[0], xyz_face[1], xyz_face[2]), (*two_phase_flow_solver->get_interp_phi()));
      }
      ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
    }

    ierr = VecGetArray(two_phase_flow_solver->get_fine_jump_mu_grad_v(), &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
    for (p4est_locidx_t fine_node_idx = 0; fine_node_idx < nodes_fine->num_owned_indeps; ++fine_node_idx) {
      node_xyz_fr_n(fine_node_idx, p4est_fine, nodes_fine, xyz_node);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          fine_jump_mu_grad_v_p[fine_node_idx+P4EST_DIM*dir+der] = exact_solution.jump_in_flux(dir, der, DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
    }
    ierr = VecRestoreArray(two_phase_flow_solver->get_fine_jump_mu_grad_v(), &fine_jump_mu_grad_v_p); CHKERRXX(ierr);

    two_phase_flow_solver->test_viscosity_explicit(rhs, tn);

    evaluate_errors(two_phase_flow_solver, error_vnp1_minus, error_vnp1_plus);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      max_error_vnp1_minus[dir] = MAX(max_error_vnp1_minus[dir],  error_vnp1_minus[dir] );
      max_error_vnp1_plus[dir]  = MAX(max_error_vnp1_plus[dir],   error_vnp1_plus[dir]  );
    }

    tn += dt;
//    if(!implicit)
//      two_phase_flow_solver->enforce_bc_v();

    two_phase_flow_solver->slide_face_fields();

    if(!print_final_only)
    {
      ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t number of leaves = %d\n",
                         iter, tn, 100*tn/duration, two_phase_flow_solver->get_p4est()->global_num_quadrants); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(mpi.comm(), "\t error u_minus = %.5e \t error v_minus = %.5e \t error w_minus = %.5e\n",
                         error_vnp1_minus[0], error_vnp1_minus[1], error_vnp1_minus[2]); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "\t error u_plus  = %.5e \t error v_plus  = %.5e \t error w_plus  = %.5e\n",
                         error_vnp1_plus[0], error_vnp1_plus[1], error_vnp1_plus[2]); CHKERRXX(ierr);
#else
      ierr = PetscPrintf(mpi.comm(), "\t error u_minus = %.5e \t error v_minus = %.5e\n",
                         error_vnp1_minus[0], error_vnp1_minus[1]); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "\t error u_plus  = %.5e \t error v_plus  = %.5e\n",
                         error_vnp1_plus[0], error_vnp1_plus[1]); CHKERRXX(ierr);
#endif
    }

    iter++;

//    break;
  }
#ifdef P4_TO_P8
  ierr = PetscPrintf(mpi.comm(), "\t MAX error u_minus = %.5e \t MAX error v_minus = %.5e \t MAX error w_minus = %.5e\n",
                     max_error_vnp1_minus[0], max_error_vnp1_minus[1], max_error_vnp1_minus[2]); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi.comm(), "\t MAX error u_plus  = %.5e \t MAX error v_plus  = %.5e \t MAX error w_plus  = %.5e\n",
                     max_error_vnp1_plus[0], max_error_vnp1_plus[1], max_error_vnp1_plus[2]); CHKERRXX(ierr);
#else
  ierr = PetscPrintf(mpi.comm(), "\t MAX error u_minus = %.5e \t MAX error v_minus = %.5e\n",
                     max_error_vnp1_minus[0], max_error_vnp1_minus[1]); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi.comm(), "\t MAX error u_plus  = %.5e \t MAX error v_plus  = %.5e\n",
                     max_error_vnp1_plus[0], max_error_vnp1_plus[1]); CHKERRXX(ierr);
#endif

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
  }

  Vec coarse_phi = NULL;
  two_phase_flow_solver->interpolate_linearly_from_fine_nodes_to_coarse_nodes(fine_phi, coarse_phi);

  const double *coarse_phi_p, *fine_phi_p;
  ierr = VecGetArrayRead(coarse_phi, &coarse_phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, (export_dir + "/illustration").c_str(),
                         VTK_NODE_SCALAR, "phi", coarse_phi_p);
  my_p4est_vtk_write_all(p4est_fine, nodes_fine, ghost_fine,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 0, (export_dir + "/fine_illustration").c_str(),
                         VTK_NODE_SCALAR, "phi", fine_phi_p);
  ierr = VecRestoreArrayRead(coarse_phi, &coarse_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  ierr = VecDestroy(coarse_phi); CHKERRXX(ierr);

  delete two_phase_flow_solver;
  delete data;
  delete data_fine;

  return 0;
}
