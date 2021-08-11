/* 
 * The surfactant solver usage, with verification and validation tests
 *
 * run the program with the -help flag to see the available options
 */

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

// casl_p4est Library
#include <src/Parser.h>
#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_surfactant.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_surfactant.h>
#endif

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------- EXTRA INFO (to be printed when -help is invoked) -------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

const std::string extra_info =
      std::string("This executable enables the user to run standard verification and validation tests of the surfactant\n"
                  "solver of the casl_p4est library.\n\n")
    + std::string("Developer: Fernando Temprano-Coleto (ftempranocoleto@ucsb.edu).\n\n");

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------- GLOBAL VARIABLES -------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

// Default inputs
const std::string default_export_dir     = "/home/temprano/Output/p4est_surfactant/tests";
const int default_time_integ             = 1;
const int default_lmin                   = 3;
const int default_lmax                   = 7;
const int default_nx                     = 1;
const int default_ny                     = 1;
#ifdef P4_TO_P8
const int default_nz                     = 1;
#endif
const double default_lip                 = 1.2;
const double default_CFL                 = 1.0;

// To be setup by the main function
int test_number;
double dmin = 1.0; // Smallest dimension of the problem. This parameter is changed later after the grid is defined.
double dt   = 1.0; // Smallest time increment of the problem. This parameter is changed later after the grid is defined.
double tn   = 0.0; // Time. We choose the start time at t=0.
double tf   = 1.0; // Final time of the simulation. This parameter is changed later after the case is selected.

double R = 1.0;
double D_s = 1.0;
double u_max = 1.0;

/*
 * [NOTE:] In 3D, we follow the ISO convention used in physics for spherical coordinates, i.e. phi in [0,2*pi) is the azimuthal angle
 *         and theta in [0, pi] is the polar angle. This extends to polar coordinates in 2D such that phi, and not theta, is used for
 *         the polar angle (which is also the azimuthal in 2D).
 */

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------ CARTESIAN TO SPHERICAL(POLAR) COORDINATES -----------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

struct r_from_xyz_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt( SUMD( SQR(x), SQR(y), SQR(z) ) );
  }
} rad;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct phi_from_xyz_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    ONLY3D( (void) z; )
    return atan2(y,x);
  }
} phi;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct theta_from_xyz_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    CODE2D( (void) x; (void) y; return PI/2.0; )
    CODE3D( return (rad(x,y,z)<EPS*dmin) ? PI/2.0 : acos(z/rad(x,y,z)); )
    /*[FERNANDO]: Setting theta as PI/2.0 is my own convention in the case of a zero radial coordinate.*/
  }
} theta;

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------ SPHERICAL(POLAR) TO CARTESIAN COORDINATES -----------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

struct x_from_sph_coord_t : public CF_DIM
{
  double operator()(DIM(double r_, double phi_, double theta_)) const
  {
    return MULTD(r_, cos(phi_), sin(theta_));
  }
} x_from_sph_coord;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct y_from_sph_coord_t : public CF_DIM
{
  double operator()(DIM(double r_, double phi_, double theta_)) const
  {
    return MULTD(r_, sin(phi_), sin(theta_));
  }
} y_from_sph_coord;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct z_from_sph_coord_t : public CF_DIM
{
  double operator()(DIM(double r_, double phi_, double theta_)) const
  {
    (void) phi_;
    CODE2D( (void) r_; return 0.0; )
    CODE3D( return r_*cos(theta_); )
  }
} z_from_sph_coord;

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------- GEOMETRY CLASSES -----------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

#ifdef P4_TO_P8 // TO-DO: Fix star class to avoid compiler macros, avoid default parameters and create it in main function
struct star : public CF_2
{
  double r0, alpha, beta;
  unsigned short m, n;

public:
  star(double r0_input=0.75, double alpha_input=0.6, double beta_input=4.0/15.0,
       unsigned short n_input=6, unsigned short m_input=6)
  {
    r0    = r0_input;
    alpha = alpha_input;
    beta  = beta_input;
    n     = n_input;
    m     = m_input;
  }
  double operator()(double phi, double theta) const
  {
    return r0*( 1 + beta*(1-alpha*cos((double)m*phi))*(1-cos((double)n*theta)) );
  }
  double d_theta(double phi, double theta) const
  {
    return r0*beta*n*(1-alpha*cos((double)m*phi))*sin((double)n*theta);
  }
  double d_phi(double phi, double theta) const
  {
    return r0*beta*alpha*m*sin((double)m*phi)*(1-cos((double)n*theta));
  }
} r_star;
#else
struct star : public CF_1
{
  double r0, alpha;
  unsigned short m;

public:
  star(double r0_input=0.75, double alpha_input=0.25, unsigned short m_input=7) // Adjust to input R instead of number
  {
    r0    = r0_input;
    alpha = alpha_input;
    m     = m_input;
  }
  double operator()(double phi) const
  {
    return r0*(1-alpha*sin((double)m*phi));
  }
  double d_phi(double phi) const
  {
    return -r0*alpha*m*cos((double)m*phi);
  }
} r_star(R,0.075);
#endif

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_geom_rad_t
{
  double operator()(CODIM1(double phi_, double theta_), double t_) const
  {
    switch(test_number)
    {
      case 0: return R; break;
      case 1: throw std::invalid_argument("Not ready."); break;
      case 2: return t_ + r_star(CODIM1(phi_, theta_)); break;
      case 3: throw std::invalid_argument("There is no available analytical solution for this test."); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} exact_geom_rad;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct level_set_t : public CF_DIM
{
  level_set_t(double t_input=0.0, double lip_input=1.2) {t = t_input; lip = lip_input;}
  double operator()(DIM(double x, double y, double z)) const
  {
    return exact_geom_rad(CODIM1(phi(DIM(x,y,z)), theta(DIM(x,y,z))), t) - rad(DIM(x,y,z));
  }
};

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------- EXACT (WHEN AVAILABLE) AND INITIAL SURFACE CONCENTRATION ---------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_Gamma_t
{
  double Gamma_0, epsilon;

public:
  exact_Gamma_t()
  {
    switch(test_number)
    {
      case 0:
        Gamma_0 = 1.0;
        CODE2D( epsilon = 0.5; )
        CODE3D( epsilon = 1.0/30.0; )
        break;
      case 1:
      case 2:
      case 3:
        throw std::invalid_argument("Not ready."); break;
      default:
        throw std::invalid_argument("Please choose a valid test.");
    }
  }
  double operator()(DIM(double x, double y, double z), double t) const
  {
    switch(test_number)
    {
      case 0:
        CODE2D( return Gamma_0*( 1.0 + epsilon*exp(-9.0*D_s*t/SQR(R))*sin(3.0*phi(DIM(x,y,z))) ); )
        CODE3D( return Gamma_0*( 1.0 + 15.0*epsilon*exp(-12.0*D_s*t/SQR(R))*pow(sin(theta(DIM(x,y,z))),3.0)*sin(3.0*phi(DIM(x,y,z)))); )
        break;
      case 1:
        throw std::invalid_argument("Not ready."); break;
      case 2:
        return          sqrt(SQR(  r_star(CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z)))))+SQR(r_star.d_phi(  CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z))))))
                       /sqrt(SQR(t+r_star(CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z)))))+SQR(r_star.d_phi(  CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z))))))
               ONLY3D( *sqrt(SQR(  r_star(CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z)))))+SQR(r_star.d_theta(CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z))))))
                       /sqrt(SQR(t+r_star(CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z)))))+SQR(r_star.d_theta(CODIM1(phi(DIM(x,y,z)),theta(DIM(x,y,z)))))) ); break;
      case 3:
        throw std::invalid_argument("There is no available analytical solution for this test."); break;
      default:
        throw std::invalid_argument("Please choose a valid test.");
    }
  }
  double integral(double t) const
  {
    switch(test_number)
    {
      case 0:
        (void) t;
        CODE2D( return 2*PI*R*Gamma_0; )
        CODE3D( return 4*PI*SQR(R)*Gamma_0; )
        break;
      case 1:
      case 2:
      case 3:
        throw std::invalid_argument("Not ready."); break;
      default:
        throw std::invalid_argument("Please choose a valid test.");
    }
  }
} exact_Gamma;

/*------------------------------------------------------------------------------------------------------------------------------------*/
struct initial_Gamma_nm1_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    return exact_Gamma(DIM(x,y,z),-dt);
  }
} initial_Gamma_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_Gamma_n_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    return exact_Gamma(DIM(x,y,z),0.0);
  }
} initial_Gamma_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------ PRESCRIBED AND INITIAL VELOCITY COMPONENTS ----------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_u_nm1_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(test_number)
    {
      case 0:
      case 1:
        (void) x; (void) y; ONLY3D((void) z;) return 0.0; break;
      case 2:
        return x/MAX(rad(DIM(x,y,z)),EPS*dmin); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_u_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_v_nm1_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(test_number)
    {
      case 0:
      case 1:
        (void) x; (void) y; ONLY3D((void) z;) return 0.0; break;
      case 2:
        return y/MAX(rad(DIM(x,y,z)),EPS*dmin); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_v_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_w_nm1_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(test_number)
    {
      case 0:
      case 1:
        (void) x; (void) y; ONLY3D((void) z;) return 0.0; break;
      case 2:
        return CODEDIM( 0.0, z/MAX(rad(DIM(x,y,z)),EPS*dmin) ); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_w_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_u_n_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(test_number)
    {
      case 0:
      case 1:
        (void) x; (void) y; ONLY3D((void) z;) return 0.0; break;
      case 2:
        return x/MAX(rad(DIM(x,y,z)),EPS*dmin); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_u_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_v_n_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(test_number)
    {
      case 0:
      case 1:
        (void) x; (void) y; ONLY3D((void) z;) return 0.0; break;
      case 2:
        return y/MAX(rad(DIM(x,y,z)),EPS*dmin); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_v_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_w_n_t : public CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(test_number)
    {
      case 0:
      case 1:
        (void) x; (void) y; ONLY3D((void) z;) return 0.0; break;
      case 2:
        return CODEDIM( 0.0, z/MAX(rad(DIM(x,y,z)),EPS*dmin) ); break;
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_w_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------ UTILITY FUNCTIONS -----------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

void print_banner(const mpi_environment_t& mpi)
{
  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(),
                     "\n"
                     "----------------------------------------------------------------------------------------------------\n"
                     "--------------------------------------== SURFACTANT SOLVER ==---------------------------------------\n"
                     "--------------------------------------                       ---------------------------------------\n"
                     "--------------------------------------       casl_p4est      ---------------------------------------\n"
                     "----------------------------------------------------------------------------------------------------\n");
  CHKERRXX(ierr);
}

const std::string my_bool_to_string(bool input)
{
  return input ? "true" : "false";
}

const std::string my_time_integrator_to_string(int input)
{
  switch(input)
  {
    case 0: return "IEEU1";
    case 1: return "SBDF2";
    case 2: return "CNLF2";
    case 3: return "MCNAB2";
    default: throw std::invalid_argument("my_time_integrator_to_string: Please choose a valid test.");
  }
}

const std::string my_adv_suffix_to_string(bool input)
{
  return input ? "-SL" : "-FV";
}

//void compute_and_save_errors_OLD(my_p4est_surfactant_t* solver,
//                                 char error_space_name[],
//                                 char error_time_name[],
//                                 int N_sampling=2500)
//{
//  // Sample exact and numerical solutions at points of the exact interface
//  my_p4est_interpolation_nodes_t interp_n(solver->get_ngbd_n());
//  interp_n.set_input(solver->get_Gamma_n(), linear);
//  std::vector<double> Gamma_num(N_sampling);
//  std::vector<double> Gamma_exact(N_sampling);
//  std::vector<double> xyz_intf[P4EST_DIM];
//  for (short dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    xyz_intf[dir].resize(N_sampling);
//  }

//  if(solver->get_p4est_n()->mpirank==0)
//  {
//    double xyz_tmp[P4EST_DIM];
//    P4EST_ASSERT(P4EST_DIM==2); // Not ready for 3D yet
//    double phi_k = 0.0;

//    for(p4est_locidx_t k=0; k<N_sampling; ++k)
//    {
//      phi_k = ((double)k)*2.0*PI / ((double)N_sampling);
//      xyz_tmp[0] = (/*tn +*/ r_star(phi_k))*cos(phi_k); // For 3D: need to add general functions x_from_rad, etc and call generally
//      xyz_tmp[1] = (/*tn +*/ r_star(phi_k))*sin(phi_k);

//      Gamma_exact[k] = exact_Gamma(xyz_tmp[0], xyz_tmp[1], tn);
//      interp_n.add_point(k, xyz_tmp);
//      for (short dir = 0; dir < P4EST_DIM; ++dir)
//        xyz_intf[dir][k] = xyz_tmp[dir];
//    }
//  }

//  double* data = Gamma_num.data();
//  interp_n.interpolate(data);
//  double integrated_Gamma = solver->get_integrated_Gamma_intf();

//  if(solver->get_p4est_n()->mpirank==0)
//  {
//    double l_inf_error = 0.0;
//    double phi_k = 0.0;

//    for(p4est_locidx_t k=0; k<N_sampling; ++k)
//    {
//      phi_k = ((double)k)*2.0*PI / ((double)N_sampling);
//      l_inf_error = MAX(l_inf_error, fabs(Gamma_num[k]-Gamma_exact[k]));

//      FILE* fp_errors;
//      if(k==0)
//      {
//        fp_errors = fopen(error_space_name, "w");
//        if(fp_errors==NULL)
//        {
//          char error_msg[1024];
//          sprintf(error_msg, "compute_and_save_errors: could not open file %s.", error_space_name);
//          throw std::invalid_argument(error_msg);
//        }
//        fprintf(fp_errors, "phi            \t Gamma_num          \t Gamma_exact\n");
//        fclose(fp_errors);
//      }
//      fp_errors = fopen(error_space_name, "a");
//      fprintf(fp_errors, "%.12f \t %.12e \t %.12e \n", phi_k, Gamma_num[k], Gamma_exact[k]);
//      fclose(fp_errors);
//    }

//    FILE* fp_errors_time;
//    if(tn==0.0)
//    {
//      fp_errors_time = fopen(error_time_name, "w");
//      if(fp_errors_time==NULL)
//      {
//        char error_msg[1024];
//        sprintf(error_msg, "compute_and_save_errors: could not open file %s.", error_time_name);
//        throw std::invalid_argument(error_msg);
//      }
//      fprintf(fp_errors_time, "time         \t l_inf_error_Gamma \t     integral_Gamma \n");
//      fclose(fp_errors_time);
//    }
//    fp_errors_time = fopen(error_time_name, "a");
//    fprintf(fp_errors_time, "%.12f \t %.12e \t %.12e \n", tn, l_inf_error, integrated_Gamma);
//    fclose(fp_errors_time);
//  }
//}

void compute_and_save_errors(int iter,
                             my_p4est_surfactant_t* solver,
                             char error_path[],
                             bool save_vtk)
{
  PetscErrorCode ierr;

  // Create paths to export files
  static char error_dat_name[1024];
  if(iter==0){
    sprintf(error_dat_name, "%s/errors_t.dat", error_path); }
  ierr = PetscPrintf(solver->get_p4est_n()->mpicomm, "  -> Updating error file %s\n", error_dat_name); CHKERRXX(ierr);

  // Sample error at nodes within a band around the interface
  Vec error = NULL;
  Vec exact_sol = NULL;
  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &error); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &exact_sol); CHKERRXX(ierr);
  double *error_p, *exact_sol_p;
  const double *phi_band_p, *Gamma_n_p;
  ierr = VecGetArray(exact_sol, &exact_sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(error, &error_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver->get_phi_band(), &phi_band_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solver->get_Gamma_n(), &Gamma_n_p); CHKERRXX(ierr);
  for(size_t n = 0; n < solver->get_nodes_n()->indep_nodes.elem_count; ++n)
  {
    if( phi_band_p[n] < 2.0*solver->get_band_width_distance() )
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, solver->get_p4est_n(), solver->get_nodes_n(), xyz);
      exact_sol_p[n] = exact_Gamma(DIM(xyz[0],xyz[1],xyz[2]),tn);
      error_p[n] = fabs( Gamma_n_p[n] - exact_sol_p[n] );
    }
  }
  ierr = VecRestoreArrayRead(solver->get_Gamma_n(), &Gamma_n_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solver->get_phi_band(), &phi_band_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error, &error_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(exact_sol, &exact_sol_p); CHKERRXX(ierr);

  // Compute several norms of the error
  double intf_length = interface_length(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi());
  double band_area   = area_in_negative_domain(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi_band());

  double err_l1_intf   = integrate_over_interface(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi(), error) / intf_length;
  double err_l1_band   = integrate_over_negative_domain(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi_band(), error) / band_area;
  double err_linf_intf = max_over_interface(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi(), error);
  double err_linf_band = max_over_negative_domain(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi_band(), error);

  // Compute integrated quantities (to track global conservation at the discrete level)
  double integral_Gamma_intf  = integrate_over_interface(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi(), solver->get_Gamma_n());
  double integral_Gamma_band  = integrate_over_negative_domain(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi_band(), solver->get_Gamma_n()) / solver->get_band_width_distance();
  double integral_Gamma_exact = exact_Gamma.integral(tn);

  if(solver->get_p4est_n()->mpirank==0)
  {
    // Print quantities to file
    FILE* fp_errors_time;
    if(tn==0.0)
    {
      fp_errors_time = fopen(error_dat_name, "w");
      if(fp_errors_time==NULL)
      {
        char error_msg[1024];
        sprintf(error_msg, "compute_and_save_errors: could not open file %s.", error_dat_name);
        throw std::invalid_argument(error_msg);
      }
      fprintf(fp_errors_time, "time             \t l1_err_Gamma_intf \t     l1_err_Gamma_band \t     linf_err_Gamma_intf \t linf_err_Gamma_band \t integral_Gamma_intf \t integral_Gamma_band \t integral_Gamma_exact \n");
      fclose(fp_errors_time);
    }
    fp_errors_time = fopen(error_dat_name, "a");
    fprintf(fp_errors_time, "%.14f \t %.14e \t %.14e \t %.14e \t %.14e \t %.14e \t %.14e \t %.14e \n", tn, err_l1_intf, err_l1_band, err_linf_intf, err_linf_band, integral_Gamma_intf, integral_Gamma_band, integral_Gamma_exact);
    fclose(fp_errors_time);
  }
  if(save_vtk)
  {
    char error_vtk_name[1024];
    sprintf(error_vtk_name, "%s/error_snapshot_%04d", error_path, iter);
    ierr = PetscPrintf(solver->get_p4est_n()->mpicomm, "  -> Saving vtk file     %s\n", error_vtk_name); CHKERRXX(ierr);

    const double *phi_p;

    ierr = VecGetArrayRead(solver->get_phi(), &phi_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(solver->get_phi_band(), &phi_band_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(solver->get_Gamma_n(), &Gamma_n_p); CHKERRXX(ierr);
    ierr = VecGetArray(error, &error_p); CHKERRXX(ierr);
    ierr = VecGetArray(exact_sol, &exact_sol_p); CHKERRXX(ierr);

    my_p4est_vtk_write_all_general(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_ghost_n(),
                                   P4EST_TRUE, P4EST_TRUE,
                                   5, // number of VTK_NODE_SCALAR
                                   0, // number of VTK_NODE_VECTOR_BY_COMPONENTS
                                   0, // number of VTK_NODE_VECTOR_BLOCK
                                   0, // number of VTK_CELL_SCALAR
                                   0, // number of VTK_CELL_VECTOR_BY_COMPONENTS
                                   0, // number of VTK_CELL_VECTOR_BLOCK
                                   error_vtk_name,
                                   VTK_NODE_SCALAR, "phi"         , phi_p,
                                   VTK_NODE_SCALAR, "phi_band"    , phi_band_p,
                                   VTK_NODE_SCALAR, "Gamma_num"   , Gamma_n_p,
                                   VTK_NODE_SCALAR, "Gamma_exact" , exact_sol_p,
                                   VTK_NODE_SCALAR, "error"       , error_p
                                   );

    ierr = VecRestoreArrayRead(solver->get_phi(), &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(solver->get_phi_band(), &phi_band_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(solver->get_Gamma_n(), &Gamma_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(error, &error_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(exact_sol, &exact_sol_p); CHKERRXX(ierr);

  }
  ierr = VecDestroy(error); CHKERRXX(ierr);
  ierr = VecDestroy(exact_sol); CHKERRXX(ierr);
}

void export_files_and_print_iteration_message(int iter, my_p4est_surfactant_t* solver,
                                              char out_path[], bool save_vtk, bool save_errors)
{
  static char vtk_path[1024], error_path[1024];/*, dat_error_time_name[1024];*/
  char vtk_name[1024];/*,  dat_error_space_name[1024];*/

  if(iter==0)
  {
    sprintf(vtk_path, "%s/vtu",  out_path);
    sprintf(error_path, "%s/errors",  out_path);
    //sprintf(dat_error_time_name, "%s/errors_t.dat", error_path);

    if(create_directory(out_path, solver->get_p4est_n()->mpirank, solver->get_p4est_n()->mpicomm))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_%dd: could not create exportation directory %s", P4EST_DIM, out_path);
      throw std::runtime_error(error_msg);
    }
    if(save_vtk && create_directory(vtk_path, solver->get_p4est_n()->mpirank, solver->get_p4est_n()->mpicomm))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_%dd: could not create exportation directory for vtk files %s", P4EST_DIM, vtk_path);
    }
    if(save_errors && create_directory(error_path, solver->get_p4est_n()->mpirank, solver->get_p4est_n()->mpicomm))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_%dd: could not create exportation directory for error files %s", P4EST_DIM, error_path);
    }
  }

  PetscErrorCode ierr;
  ierr = PetscPrintf(solver->get_p4est_n()->mpicomm,
                     "\nIteration #%04d:  tn = %.5e,  "
                     "percent done = %.1f%%,  "
                     "integral of Gamma = %.6e,  "
                     "number of leaves = %d\n",
                     iter,
                     tn,
                     100.0*tn/tf,
                     solver->get_integrated_Gamma_intf(),
                     solver->get_p4est_n()->global_num_quadrants); CHKERRXX(ierr);

  if(save_vtk)
  {
    sprintf(vtk_name, "%s/snapshot_%04d", vtk_path, iter);
    solver->save_vtk(vtk_name);
    ierr = PetscPrintf(solver->get_p4est_n()->mpicomm, "  -> Saving vtk file     %s\n", vtk_name); CHKERRXX(ierr);
  }
  if(save_errors)
    compute_and_save_errors(iter, solver, error_path, save_vtk);
}

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------- MAIN FUNCTION -------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char** argv) {
  
  // Setup parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // Options for the inputs
  cmdParser cmd;
  cmd.add_option("test_number",       "Specific test to be run. The options are:\n"
                                      "\t\t0) Surface diffusion on a circle (2D) or sphere (3D).\n"
                                      "\t\t1) Surface diffusion on an ellipse (2D) or oblate spheroid (3D).\n"
                                      "\t\t2) Surface advection on an expanding star.\n");
  cmd.add_option("time_integrator",   "Type of time integrator to be used in the simulation. The options are:\n"
                                      "\t\t    0) Implicit-explicit Euler, 1st order (IEEU1).\n"
                                      "\t\t    1) Semi-implicit Backward Differencing Formula, 2nd order (SBDF2).\n"
                                      "\t\t    2) Crank-Nicolson-Leapfrog, 2nd order (CNLF2).\n"
                                      "\t\t    3) Modified Crank-Nicolson-Adams-Bashforth, 2nd order (MCNAB2).\n"
                                      "\t\t    The default is " + my_time_integrator_to_string(default_time_integ) + ".\n");

  cmd.add_option("use_sl",            "Activates the use of a Semi-Lagrangian discretization for surface advection terms.\n");
  cmd.add_option("lmin",              "Minimum level of the forest trees, the default is " + std::to_string(default_lmin) + ".\n");
  cmd.add_option("lmax",              "Maximum level of the forest trees, the default is " + std::to_string(default_lmax) + ".\n");
  cmd.add_option("nx",                "Number of trees in the x-direction, the default is " + std::to_string(default_nx) + ".\n");
  cmd.add_option("ny",                "Number of trees in the y-direction, the default is " + std::to_string(default_ny) + ".\n");
#ifdef P4_TO_P8
  cmd.add_option("nz",                "Number of trees in the z-direction, the default is " + std::to_string(default_nz) + ".\n");
#endif
  cmd.add_option("lip",               "Lipschitz constant L used for grid refinement, the default is " + std::to_string(default_lip) + ".\n");
  cmd.add_option("export_dir",        "Complete path to the exportation directory of the outputs, the default is:\n"
                                      "\t       " + default_export_dir + "\n");
  cmd.add_option("save_vtk",          "Activates the exportation of results in vtk format.\n");
  cmd.add_option("save_errors",       "Activates the exportation of error norms. If save_vtk is also activated, vtk files with the errors are also generated.\n");
  cmd.add_option("CFL",               "CFL number such that dt=CFL*h/u_max or dt=CFL*h (if u_max=0), where h is the minimum grid\n"
                                      "\tsize in any dimension. The default is " + std::to_string(default_CFL) + ".\n");

  // Print banner on terminal
  print_banner(mpi);

  // Parse through script arguments arguments and terminate (printing options and extra_info) if "-help" is invoked
  if(cmd.parse(argc, argv, extra_info)) return 0;

  // Setup stopwatch to time the simulation
  parStopWatch watch;
  watch.start("Running example: surfactant");

  // Setup simulation parameters from inputs
  if(!cmd.contains("test_number")) throw std::invalid_argument("main(): the argument test_number MUST be provided by the user.");
  else                             test_number = cmd.get<int>("test_number", -1);
  int time_integ = cmd.get<int>("time_integrator", default_time_integ);
  const bool use_sl = cmd.contains("use_sl");
  const int lmin = cmd.get<int>("lmin", default_lmin);
  const int lmax = cmd.get<int>("lmax", default_lmax);
  const int nx = cmd.get<int>("nx", default_nx);
  const int ny = cmd.get<int>("ny", default_ny);
#ifdef P4_TO_P8
  const int nz = cmd.get<int>("nz", default_nz);
#endif
  const double lip = cmd.get<double>("lip", default_lip);
  std::string export_dir = cmd.get<std::string>("export_dir", default_export_dir);
  const bool save_vtk = cmd.contains("save_vtk");
  const bool save_errors = cmd.contains("save_errors");
  const double CFL = cmd.get<double>("CFL", default_CFL);

  // Setup simulation parameters depending on the test number
  double xmin;             // Minimum x coordinate of the brick
  double xmax;             // Maximum x coordinate of the brick
  double ymin;             // Minimum y coordinate of the brick
  double ymax;             // Maximum y coordinate of the brick
#ifdef P4_TO_P8
  double zmin;             // Minimum z coordinate of the brick
  double zmax;             // Maximum z coordinate of the brick
#endif
  std::string test_name;   // Test name for the output directory name

  switch(test_number)
  {
    case 0:
      CODE2D( D_s = 1.0/9.0; )
      CODE3D( D_s = 1.0/12.0; )
      xmin = -PI/2.0;
      xmax =  PI/2.0;
      ymin = -PI/2.0;
      ymax =  PI/2.0;
#ifdef P4_TO_P8
      zmin = -PI/2.0;
      zmax =  PI/2.0;
#endif
      tf = 1.0;
      u_max = 0.0;
      test_name = "surface_diffusion_sphere";
      break;
    case 1:
      xmin = -PI/2.0;
      xmax =  PI/2.0;
      ymin = -PI/2.0;
      ymax =  PI/2.0;
#ifdef P4_TO_P8
      zmin = -PI/2.0;
      zmax =  PI/2.0;
#endif
      tf = 1.0;
      u_max = 0.0;
      test_name = "surface_diffusion_star";
      break;
    case 2:
      xmin = -PI;
      xmax =  PI;
      ymin = -PI;
      ymax =  PI;
#ifdef P4_TO_P8
      zmin = -PI;
      zmax =  PI;
#endif
      tf = 1.5;
      test_name = "advection_expansion";
      break;
    case 3:
      throw std::invalid_argument("Not ready. Choose another test");
    default:
      throw std::invalid_argument("Please choose a valid test.");
  }

  // Set up other derived simulation parameters
#ifdef P4_TO_P8
  dmin = MIN((xmax-xmin),(ymax-ymin),(zmax-zmin))/pow(2.0,lmax);
  const int n_xyz      [P4EST_DIM] = {nx, ny, nz};
  const double xyz_min [P4EST_DIM] = {xmin, ymin, zmin};
  const double xyz_max [P4EST_DIM] = {xmax, ymax, zmax};
  const int periodicity[P4EST_DIM] = {0, 0, 0};
#else
  dmin = MIN((xmax-xmin),(ymax-ymin))/pow(2.0,lmax);
  const int n_xyz      [P4EST_DIM] = {nx, ny};
  const double xyz_min [P4EST_DIM] = {xmin, ymin};
  const double xyz_max [P4EST_DIM] = {xmax, ymax};
  const int periodicity[P4EST_DIM] = {0, 0};
#endif
  dt = (bool)u_max ? CFL*dmin/u_max : CFL*dmin;

  // Declare level-set functions
  level_set_t* ls_n   = new level_set_t(0.0, lip);
  level_set_t* ls_nm1 = new level_set_t(-dt, lip);

  // Create solver
  my_p4est_surfactant_t* surf = new my_p4est_surfactant_t(mpi, xyz_min, xyz_max, n_xyz, periodicity, lmin, lmax, ls_n, CFL);
  surf->set_D_s(D_s);

  // Turn off terms in the solver depending on the test number
  switch(test_number)
  {
    case 0:
      surf->set_no_surface_advection(true);
      break;
    case 1:
      surf->set_no_surface_diffusion(true);
      break;
    case 2:
      throw std::invalid_argument("Not ready. Choose another test");
    default:
      throw std::invalid_argument("Please choose a valid test.");
  }

  // Set up solver data
#ifdef P4_TO_P8
  CF_3 *vnm1[P4EST_DIM] = { &initial_u_nm1, &initial_v_nm1, &initial_w_nm1 };
  CF_3 *vn  [P4EST_DIM] = { &initial_u_n,   &initial_v_n,   &initial_w_n   };
#else
  CF_2 *vnm1[P4EST_DIM] = { &initial_u_nm1, &initial_v_nm1 };
  CF_2 *vn  [P4EST_DIM] = { &initial_u_n,   &initial_v_n   };
#endif
//  surf->set_velocities(vn, vnm1);
//  surf->compute_extended_velocities(ls_nm1, NULL, true, false);
  surf->set_Gamma(&initial_Gamma_nm1, &initial_Gamma_n);

  // Setup and create output directories
  char out_dir[1024];
  sprintf(out_dir, "%s/%dd/%s/%02d_%02d", export_dir.c_str(), (int)P4EST_DIM, test_name.c_str(), lmin, lmax);

  // Time evolution
  int iter = 0;
  export_files_and_print_iteration_message(iter, surf, out_dir, save_vtk, save_errors);

  while(tn+0.01*dt < tf)
  {
    if(tn+dt>tf)
    {
      dt = tf-tn;
      surf->set_dt_n(dt);
    }

    //surf->advect_interface_one_step();

    P4EST_ASSERT(tn >= 0.0);
      surf->compute_one_step_Gamma((time_integrator)time_integ);

    tn+=dt;
    iter++;

    surf->update_from_tn_to_tnp1();
    dt = surf->get_dt_n();
    export_files_and_print_iteration_message(iter, surf, out_dir, save_vtk, save_errors);
  }

  // Destroy the dynamically allocated classes
  delete surf;
  delete ls_n;
  delete ls_nm1;

  // Stop and read timer
  watch.stop();
  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(),"\n"); CHKERRXX(ierr);
  watch.read_duration();

  // Finish
  return 0;
}
