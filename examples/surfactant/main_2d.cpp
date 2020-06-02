/* 
 * The surfactant solver and its verification tests
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

// p4est Library
#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_surfactant.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_surfactant.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;


/*------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------- TEST OPTIONS --------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/
/*
 *  ********* 2D *********
 * 0 - Advection: expansion
 * 1 - Advection: vortex
 *
 *  ********* 3D *********
 * 0 - Advection: expansion
 * 1 - Advection: vortex
 */

int test_number;

double dmin = 1.0; // Smallest dimension of the problem. This parameter is changed later after the grid is defined.
double dt   = 1.0; // Smallest time increment of the problem. This parameter is changed later after the grid is defined.
double tn   = 0.0; // Time. We choose the origin at t=0 following the convention of Everyone et al.

#ifdef P4_TO_P8

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------- 3D FUNCTIONS --------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

/*
 * [NOTE:] In 3D, we follow the ISO convention used in physics for spherical coordinates, i.e. phi in [0,2*pi) is the azimuthal angle
 *         and theta in [0, pi] is the polar angle.
 */

struct r_from_xyz : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    return sqrt(SQR(x)+SQR(y)+SQR(z));
  }
} rad;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct phi_from_xyz : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    (void) z;
    return atan2(y,x);
  }
} phi;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct theta_from_xyz : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    if (rad(x,y,z) < EPS*dmin)
    {
      return 0.0; /*[FERNANDO]: my convention.*/
    }
    else
    {
      return acos(z/rad(x,y,z));
    }
  }
} theta;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct star : public CF_2
{
  double r0, alpha, beta;
  unsigned short n, m;

public:
  star(double r0_input=0.75, double alpha_input=4.0/15.0, double beta_input=0.6,
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
    return r0*( 1 + alpha*(1-beta*cos((double)m*phi))*(1-cos((double)n*theta)) );
  }
  double d_theta(double phi, double theta) const
  {
    return r0*alpha*n*(1-beta*cos((double)m*phi))*sin((double)n*theta);
  }
  double d_phi(double phi, double theta) const
  {
    return r0*alpha*beta*m*sin((double)m*phi)*(1-cos((double)n*theta));
  }
} r_star;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_level_set_t
{
  double operator()(double x, double y, double z, double t) const
  {
    switch(test_number)
    {
      case 0: return t + r_star(phi(x,y,z),theta(x,y,z)) - rad(x,y,z); // NOTE: This is NOT a signed distance function for any t
      case 1: throw std::invalid_argument("There is no available analytical solution for this test.");
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} exact_ls;

/*------------------------------------------------------------------------------------------------------------------------------------*/

class level_set_t : public CF_3
{
  double time;
public:
  level_set_t(double t_input=0.0) : time(t_input) {lip=1.2;}
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return exact_ls(x,y,z,time);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
};

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_u_nm1_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return x/MAX(rad(x,y,z),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_u_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_v_nm1_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return y/MAX(rad(x,y,z),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_v_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_w_nm1_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return z/MAX(rad(x,y,z),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_w_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_u_n_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return x/MAX(rad(x,y,z),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_u_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_v_n_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return y/MAX(rad(x,y,z),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_v_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_w_n_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: return z/MAX(rad(x,y,z),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_w_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_Gamma_t
{
  double operator()(double x, double y, double z, double t) const
  {
    switch(test_number)
    {
      case 0: return 0.5*sqrt(SQR(  r_star(phi(x,y,z),theta(x,y,z)))+SQR(r_star.d_theta(phi(x,y,z),theta(x,y,z))))*
                         sqrt(SQR(  r_star(phi(x,y,z),theta(x,y,z)))+SQR(r_star.d_phi(  phi(x,y,z),theta(x,y,z))))
                       /(sqrt(SQR(t+r_star(phi(x,y,z),theta(x,y,z)))+SQR(r_star.d_theta(phi(x,y,z),theta(x,y,z))))*
                         sqrt(SQR(t+r_star(phi(x,y,z),theta(x,y,z)))+SQR(r_star.d_phi(  phi(x,y,z),theta(x,y,z)))));
      case 1: throw std::invalid_argument("There is no available analytical solution for this test.");
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} exact_Gamma;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_Gamma_nm1_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: (void) x; (void) y; (void) z; return exact_Gamma(x,y,z,-dt);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_Gamma_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_Gamma_n_t : public CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
      case 0: (void) x; (void) y; (void) z; return exact_Gamma(x,y,z,0.0);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_Gamma_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

#else

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------- 2D FUNCTIONS --------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

struct r_from_xyz : public CF_2
{
  double operator()(double x, double y) const
  {
    return sqrt(SQR(x)+SQR(y));
  }
} rad;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct phi_from_xyz : public CF_2
{
  double operator()(double x, double y) const
  {
    return atan2(y,x);
  }
} phi;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct star : public CF_1
{
  double r0, alpha;
  unsigned short n;

public:
  star(double r0_input=0.75, double alpha_input=0.25, unsigned short n_input=7)
  {
    r0    = r0_input;
    alpha = alpha_input;
    n     = n_input;
  }
  double operator()(double phi) const
  {
    return r0*(1-alpha*sin((double)n*phi));
  }
  double d_phi(double phi) const
  {
    return -r0*alpha*n*cos((double)n*phi);
  }
} r_star(0.75,0.075);

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_level_set_t
{
  double operator()(double x, double y, double t) const
  {
    switch(test_number)
    {
      case 0: return t + r_star(phi(x,y)) - rad(x,y); // NOTE: This is NOT a signed distance function for any t
      case 1: throw std::invalid_argument("There is no available analytical solution for this test.");
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} exact_ls;

/*------------------------------------------------------------------------------------------------------------------------------------*/

class level_set_t : public CF_2
{
  double time;
public:
  level_set_t(double t_input=0.0) : time(t_input) {lip=1.2;}
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return exact_ls(x,y,time);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
};

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_u_nm1_t : public CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return x/MAX(rad(x,y),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_u_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_v_nm1_t : public CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return y/MAX(rad(x,y),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_v_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_u_n_t : public CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return x/MAX(rad(x,y),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_u_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_velocity_v_n_t : public CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return y/MAX(rad(x,y),EPS*dmin);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_v_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_Gamma_t
{
  double operator()(double x, double y, double t) const
  {
    switch(test_number)
    {
      case 0: return 0.5*sqrt(SQR(r_star(phi(x,y)))+SQR(r_star.d_phi(phi(x,y))))/sqrt(SQR(t+r_star(phi(x,y)))+SQR(r_star.d_phi(phi(x,y))));
      case 1: throw std::invalid_argument("There is no available analytical solution for this test.");
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} exact_Gamma;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_Gamma_nm1_t : public CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return exact_Gamma(x,y,-dt);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_Gamma_nm1;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct initial_Gamma_n_t : public CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return exact_Gamma(x,y,0.0);
      default: throw std::invalid_argument("Please choose a valid test.");
    }
  }
} initial_Gamma_n;

/*------------------------------------------------------------------------------------------------------------------------------------*/

#endif

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------ UTILITY FUNCTIONS -----------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/


void compute_and_save_errors(my_p4est_surfactant_t* solver,
                             char error_space_name[],
                             char error_time_name[],
                             int N_sampling=2500)
{
  // Sample exact and numerical solutions at points of the exact interface
  my_p4est_interpolation_nodes_t interp_n(solver->get_ngbd_n());
  interp_n.set_input(solver->get_Gamma_n(), linear);
  std::vector<double> Gamma_num(N_sampling);
  std::vector<double> Gamma_exact(N_sampling);
  std::vector<double> xyz_intf[P4EST_DIM];
  for (short dir = 0; dir < P4EST_DIM; ++dir)
  {
    xyz_intf[dir].resize(N_sampling);
  }

  if(solver->get_p4est_n()->mpirank==0)
  {
    double xyz_tmp[P4EST_DIM];
    P4EST_ASSERT(P4EST_DIM==2); // Not ready for 3D yet
    double phi_k = 0.0;

    for(p4est_locidx_t k=0; k<N_sampling; ++k)
    {
      phi_k = ((double)k)*2.0*PI / ((double)N_sampling);
      xyz_tmp[0] = (tn + r_star(phi_k))*cos(phi_k); // For 3D: need to add general functions x_from_rad, etc and call generally
      xyz_tmp[1] = (tn + r_star(phi_k))*sin(phi_k);

      Gamma_exact[k] = exact_Gamma(xyz_tmp[0], xyz_tmp[1], tn);
      interp_n.add_point(k, xyz_tmp);
      for (short dir = 0; dir < P4EST_DIM; ++dir)
        xyz_intf[dir][k] = xyz_tmp[dir];
    }
  }

  double* data = Gamma_num.data();
  interp_n.interpolate(data);
  double integrated_Gamma = solver->get_integrated_Gamma();

  if(solver->get_p4est_n()->mpirank==0)
  {
    double l_inf_error = 0.0;
    double phi_k = 0.0;

    for(p4est_locidx_t k=0; k<N_sampling; ++k)
    {
      phi_k = ((double)k)*2.0*PI / ((double)N_sampling);
      l_inf_error = MAX(l_inf_error, fabs(Gamma_num[k]-Gamma_exact[k]));

      FILE* fp_errors;
      if(k==0)
      {
        fp_errors = fopen(error_space_name, "w");
        if(fp_errors==NULL)
        {
          char error_msg[1024];
          sprintf(error_msg, "compute_and_save_errors: could not open file %s.", error_space_name);
          throw std::invalid_argument(error_msg);
        }
        fprintf(fp_errors, "phi            \t Gamma_num          \t Gamma_exact\n");
        fclose(fp_errors);
      }
      fp_errors = fopen(error_space_name, "a");
      fprintf(fp_errors, "%.12f \t %.12e \t %.12e \n", phi_k, Gamma_num[k], Gamma_exact[k]);
      fclose(fp_errors);
    }

    FILE* fp_errors_time;
    if(tn==0.0)
    {
      fp_errors_time = fopen(error_time_name, "w");
      if(fp_errors_time==NULL)
      {
        char error_msg[1024];
        sprintf(error_msg, "compute_and_save_errors: could not open file %s.", error_time_name);
        throw std::invalid_argument(error_msg);
      }
      fprintf(fp_errors_time, "time         \t l_inf_error_Gamma \t     integral_Gamma \n");
      fclose(fp_errors_time);
    }
    fp_errors_time = fopen(error_time_name, "a");
    fprintf(fp_errors_time, "%.12f \t %.12e \t %.12e \n", tn, l_inf_error, integrated_Gamma);
    fclose(fp_errors_time);
  }
}


//void compute_and_save_errors(my_p4est_surfactant_t* solver, char vtk_name[], char file_name[])
//{
//  PetscErrorCode ierr;

//  // Sample exact solution and exact ls
//  Vec Gamma_exact_tmp = NULL, Gamma_exact = NULL, phi_ex = NULL;
//  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &Gamma_exact_tmp); CHKERRXX(ierr);
//  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &phi_ex); CHKERRXX(ierr);
//  double *Gamma_exact_p, *phi_exact_p;
//  ierr = VecGetArray(Gamma_exact_tmp, &Gamma_exact_p);
//  ierr = VecGetArray(phi_ex, &phi_exact_p);
//  for(size_t n=0; n<solver->get_nodes_n()->indep_nodes.elem_count; ++n)
//  {
//    double xyz_tmp[P4EST_DIM];
//    node_xyz_fr_n(n, solver->get_p4est_n(), solver->get_nodes_n(), xyz_tmp);
//    Gamma_exact_p[n] = exact_Gamma(xyz_tmp[0],
//                                xyz_tmp[1],
//#ifdef P4_TO_P8
//                                xyz_tmp[2],
//#endif
//                                tn         );
//    phi_exact_p[n] = exact_ls(xyz_tmp[0],
//                           xyz_tmp[1],
//#ifdef P4_TO_P8
//                           xyz_tmp[2],
//#endif
//                           tn         );
//  }
//  ierr = VecRestoreArray(Gamma_exact_tmp, &Gamma_exact_p);
//  ierr = VecRestoreArray(phi_ex, &phi_exact_p);

//  my_p4est_level_set_t ls(solver->get_ngbd_n());
//  ls.reinitialize_2nd_order(phi_ex); // remember the exact phi is not a signed-distance function
//  ls.perturb_level_set_function(phi_ex, EPS*dmin);
//  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &Gamma_exact); CHKERRXX(ierr);
//  ls.extend_from_interface_to_whole_domain_TVD(phi_ex, Gamma_exact_tmp, Gamma_exact, 40);

//  // Compute errors
//  Vec err_Gamma = NULL, err_phi = NULL;
//  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &err_Gamma); CHKERRXX(ierr);
//  ierr = VecCreateGhostNodes(solver->get_p4est_n(), solver->get_nodes_n(), &err_phi); CHKERRXX(ierr);
//  double *err_Gamma_p, *err_phi_p;
//  ierr = VecGetArray(err_Gamma, &err_Gamma_p);
//  ierr = VecGetArray(err_phi, &err_phi_p);
//  const double* Gamma_exact_p_r;
//  const double* Gamma_num_p;
//  const double* phi_exact_p_r;
//  const double* phi_num_p;
//  ierr = VecGetArrayRead(Gamma_exact, &Gamma_exact_p_r);
//  ierr = VecGetArrayRead(solver->get_Gamma_n(), &Gamma_num_p);
//  ierr = VecGetArrayRead(phi_ex, &phi_exact_p_r);
//  ierr = VecGetArrayRead(solver->get_phi(), &phi_num_p);
//  for(size_t n=0; n<solver->get_nodes_n()->indep_nodes.elem_count; ++n)
//  {
//    err_Gamma_p[n] = fabs( Gamma_exact_p_r[n] - Gamma_num_p[n] );
//    err_phi_p[n]   = fabs( phi_exact_p_r[n]   - phi_num_p[n]   );
//  }
//  ierr = VecRestoreArray(err_Gamma, &err_Gamma_p);
//  ierr = VecRestoreArray(err_phi, &err_phi_p);
//  ierr = VecRestoreArrayRead(Gamma_exact, &Gamma_exact_p_r);
//  ierr = VecRestoreArrayRead(solver->get_Gamma_n(), &Gamma_num_p);
//  ierr = VecRestoreArrayRead(phi_ex, &phi_exact_p_r);
//  ierr = VecRestoreArrayRead(solver->get_phi(), &phi_num_p);

//  // Plot errors to vtk
//  unsigned short count_node_scalars = 0;
//  const double* phi_band_p;
//  const double* error_phi_p;
//  const double* error_Gamma_p;
//  ierr = VecGetArrayRead(solver->get_phi_band(), &phi_band_p);     ++count_node_scalars;
//  ierr = VecGetArrayRead(Gamma_exact, &Gamma_exact_p_r);             ++count_node_scalars;
//  ierr = VecGetArrayRead(phi_ex, &phi_exact_p_r);                    ++count_node_scalars;
//  ierr = VecGetArrayRead(err_phi, &error_phi_p);                   ++count_node_scalars;
//  ierr = VecGetArrayRead(err_Gamma, &error_Gamma_p);               ++count_node_scalars;
//  my_p4est_vtk_write_all_general(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_ghost_n(),
//                                 P4EST_TRUE, P4EST_TRUE,
//                                 count_node_scalars, // number of VTK_NODE_SCALAR
//                                 0,                  // number of VTK_NODE_VECTOR_BY_COMPONENTS
//                                 0,                  // number of VTK_NODE_VECTOR_BLOCK
//                                 0,                  // number of VTK_CELL_SCALAR
//                                 0,                  // number of VTK_CELL_VECTOR_BY_COMPONENTS
//                                 0,                  // number of VTK_CELL_VECTOR_BLOCK
//                                 vtk_name,
//                                 VTK_NODE_SCALAR, "phi_band",     phi_band_p,
//                                 VTK_NODE_SCALAR, "phi_exact",    phi_exact_p_r,
//                                 VTK_NODE_SCALAR, "Gamma_exact",  Gamma_exact_p_r,
//                                 VTK_NODE_SCALAR, "err_phi",      error_phi_p,
//                                 VTK_NODE_SCALAR, "err_Gamma",    error_Gamma_p);
//  ierr = VecRestoreArrayRead(solver->get_phi_band(), &phi_band_p);
//  ierr = VecRestoreArrayRead(Gamma_exact, &Gamma_exact_p_r);
//  ierr = VecRestoreArrayRead(phi_ex, &phi_exact_p_r);
//  ierr = VecRestoreArrayRead(err_phi, &error_phi_p);
//  ierr = VecRestoreArrayRead(err_Gamma, &error_Gamma_p);

//  // Compute error norms and export to file
//  double l_1_norm_err[2], l_inf_norm_err[2] = {0.0, 0.0};
//  l_1_norm_err[0] = integrate_over_negative_domain(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi_band(), err_Gamma)
//                    / area_in_negative_domain(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi_band());
//  l_1_norm_err[1] = integrate_over_interface(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi(), err_Gamma)
//                    / interface_length(solver->get_p4est_n(), solver->get_nodes_n(), solver->get_phi());
//  ierr = VecGetArrayRead(err_Gamma, &error_Gamma_p);
//  ierr = VecGetArrayRead(solver->get_phi_band(), &phi_band_p);
//  ierr = VecGetArrayRead(solver->get_phi(), &phi_num_p);
//  for(p4est_locidx_t n=0; n<solver->get_nodes_n()->num_owned_indeps; ++n)
//  {
//    if(phi_band_p[n] < solver->get_dxyz_max())
//    {
//      l_inf_norm_err[0] = MAX(l_inf_norm_err[0], error_Gamma_p[n]);
//    }
//    if(fabs(phi_num_p[n]) < solver->get_dxyz_max())
//    {
//      l_inf_norm_err[1] = MAX(l_inf_norm_err[1], error_Gamma_p[n]);
//    }
//  }
//  ierr = VecRestoreArrayRead(err_Gamma, &error_Gamma_p);
//  ierr = VecRestoreArrayRead(solver->get_phi_band(), &phi_band_p);
//  ierr = VecRestoreArrayRead(solver->get_phi(), &phi_num_p);
//  int mpiret = MPI_Allreduce(MPI_IN_PLACE, l_inf_norm_err, 2, MPI_DOUBLE, MPI_MAX, solver->get_p4est_n()->mpicomm); SC_CHECK_MPI(mpiret);
//  if(solver->get_p4est_n()->mpirank==0)
//  {
//    if(tn==0.0)
//    {
//      FILE* fp_errors;
//      fp_errors = fopen(file_name, "w");
//      if(fp_errors==NULL)
//      {
//        char error_msg[1024];
//        sprintf(error_msg, "compute_and_save_errors: could not open file %s.", file_name);
//        throw std::invalid_argument(error_msg);
//      }
//      fprintf(fp_errors, "tn     \t l1_band_error      \t linf_band_error    \t l1_intf_error        \t linf_intf_error\n");
//      fprintf(fp_errors, "%.4f \t %.12e \t %.12e \t %.12e \t %.12e \n", tn, l_1_norm_err[0], l_inf_norm_err[0], l_1_norm_err[1], l_inf_norm_err[1]);
//      fclose(fp_errors);
//    }
//    else
//    {
//      FILE* fp_errors;
//      fp_errors = fopen(file_name, "a");
//      fprintf(fp_errors, "%.4f \t %.12e \t %.12e \t %.12e \t %.12e \n", tn, l_1_norm_err[0], l_inf_norm_err[0], l_1_norm_err[1], l_inf_norm_err[1]);
//      fclose(fp_errors);
//    }
//  }
//}


/*------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------- MAIN FUNCTION -------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char** argv) {
  
  // Prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // Stopwatch
  parStopWatch w;
  w.start("Running example: surfactant");

  // Test number
  test_number = 0;

  // Save flags
  const bool save_vtk = true;
  const bool save_errors = true;

  // Domain parameters
  const double xmin = -PI;
  const double xmax =  PI;
  const double ymin = -PI;
  const double ymax =  PI;
#ifdef P4_TO_P8
  const double zmin = -PI;
  const double zmax =  PI;
#endif

  // Number of trees in the forest
  const int nx = 1;
  const int ny = 1;
#ifdef P4_TO_P8
  const int nz = 1;
#endif

  // Refinement levels
  const int lmin = 3;
  const int lmax = 7;
#ifdef P4_TO_P8
  dmin = MIN((xmax-xmin),(ymax-ymin),(zmax-zmin))/pow(2,lmax);
#else
  dmin = MIN((xmax-xmin),(ymax-ymin))/pow(2,lmax);
#endif

  // Time
  int integ = 3;
  bool use_SL = true;
  double CFL = 0.990;
  double u_max = 1.0;
  switch(test_number)
  {
    case 0:  u_max = 1.0; break;
    default: throw std::invalid_argument("Please choose a valid test.");
  }
  dt = CFL*dmin/u_max;

  // Domain size information
#ifdef P4_TO_P8
  const int n_xyz      [P4EST_DIM] = {nx, ny, nz};
  const double xyz_min [P4EST_DIM] = {xmin, ymin, zmin};
  const double xyz_max [P4EST_DIM] = {xmax, ymax, zmax};
  const int periodic   [P4EST_DIM] = {0, 0, 0};
#else
  const int n_xyz      [P4EST_DIM] = {nx, ny};
  const double xyz_min [P4EST_DIM] = {xmin, ymin};
  const double xyz_max [P4EST_DIM] = {xmax, ymax};
  const int periodic   [P4EST_DIM] = {0, 0};
#endif

  // Band parameters
  double band_width = 4.0;

  // Define brick from geometry of the problem
  my_p4est_brick_t brick;
  p4est_connectivity_t* conn  = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // Declare level-set functions
  level_set_t* ls_n   = new level_set_t(0.0);
  level_set_t* ls_nm1 = new level_set_t(-dt);

  // Create solver and set data
  my_p4est_surfactant_t* surf = new my_p4est_surfactant_t(mpi, &brick, conn, lmin, lmax, ls_n, NULL, false, band_width, CFL, 2.1);

#ifdef P4_TO_P8
  CF_3 *vnm1[P4EST_DIM] = { &initial_u_nm1, &initial_v_nm1, &initial_w_nm1 };
  CF_3 *vn  [P4EST_DIM] = { &initial_u_n,   &initial_v_n,   &initial_w_n   };
#else
  CF_2 *vnm1[P4EST_DIM] = { &initial_u_nm1, &initial_v_nm1 };
  CF_2 *vn  [P4EST_DIM] = { &initial_u_n,   &initial_v_n   };
#endif
  surf->set_velocities(vn, vnm1);
  surf->compute_extended_velocities(ls_nm1, NULL, true, false);
  surf->set_Gamma(&initial_Gamma_nm1, &initial_Gamma_n);
  surf->set_no_surface_diffusion(true);

  // Print vtk data
  char out_dir[1024], vtk_path[1024], vtk_name[1024], error_path[1024], dat_error_space_name[1024], dat_error_time_name[1024];
  string export_dir = "/home/temprano/Output/p4est_surfactant/tests";
  string test_name;
  string integrator_name;
  switch(integ){
    case(0): integrator_name = "IEEU1"; break;
    case(1): integrator_name = "SBDF2"; break;
    case(2): integrator_name = "CNLF2"; break;
    case(3): integrator_name = "MCNAB2"; break;
    default: throw std::invalid_argument("Please choose a valid time integrator.");
  }
  string advection_suffix;
  if(use_SL) advection_suffix = "-SL";
  else       advection_suffix = "-FV";

  switch(test_number)
  {
    case 0: test_name = "advection_expansion";     break;
    case 1: test_name = "advection_vortex";        break;
    default: throw std::invalid_argument("Please choose a valid test.");
  }
  sprintf(out_dir, "%s/%dd/%s/%s%s/%02d_%02d", export_dir.c_str(), (int)P4EST_DIM, test_name.c_str(), integrator_name.c_str(), advection_suffix.c_str(), lmin, lmax);
  sprintf(vtk_path, "%s/vtu",  out_dir);
  sprintf(error_path, "%s/errors",  out_dir);
  sprintf(dat_error_time_name, "%s/errors_t.dat", error_path);

  if(create_directory(out_dir, mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_3d: could not create exportation directory %s", out_dir);
#else
    sprintf(error_msg, "main_2d: could not create exportation directory %s", out_dir);
#endif
    throw std::runtime_error(error_msg);
  }
  if(save_vtk && create_directory(vtk_path, mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_3d: could not create exportation directory for vtk files %s", vtk_path);
#else
    sprintf(error_msg, "main_2d: could not create exportation directory for vtk files %s", vtk_path);
#endif
  }
  if(save_errors && create_directory(error_path, mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_3d: could not create exportation directory for error files %s", vtk_path);
#else
    sprintf(error_msg, "main_2d: could not create exportation directory for error files %s", vtk_path);
#endif
  }

  // Time evolution
  double tf = 2.0;
  int iter = 0;
  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(), "\nIteration #%04d:  tn = %.5e,  "
                                 "percent done = %.1f%%,  "
                                 "integral of Gamma = %.6e,  "
                                 "number of leaves = %d\n",
                     iter,
                     tn,
                     100.0*tn/tf,
                     surf->get_integrated_Gamma(),
                     surf->get_p4est_n()->global_num_quadrants); CHKERRXX(ierr);
  if(save_vtk)
  {
    sprintf(vtk_name, "%s/snapshot_%04d", vtk_path, iter);
    surf->save_vtk(vtk_name);
    ierr = PetscPrintf(mpi.comm(), "  -> Saving result vtk files in %s...\n",vtk_name); CHKERRXX(ierr);
  }
  if(save_errors)
  {
    sprintf(dat_error_space_name, "%s/error_snapshot_%04d.dat", error_path, iter);
    compute_and_save_errors(surf, dat_error_space_name, dat_error_time_name);
    ierr = PetscPrintf(mpi.comm(), "  -> Saving errors dat files in %s...\n", dat_error_space_name); CHKERRXX(ierr);
  }

  while(tn+0.01*dt < tf)
  {
    if(tn+dt>tf)
    {
      dt = tf-tn;
      surf->set_dt_n(dt);
    }

    surf->advect_interface_one_step();

    P4EST_ASSERT(tn >= 0.0);
//    if(tn > EPS*dt)
      surf->compute_one_step_Gamma((TimeIntegrator)integ, use_SL, true, iter);
//    else
//      surf->compute_one_step_Gamma((TimeIntegrator)0, use_SL, true, iter);

    tn+=dt;
    iter++;

    surf->update_from_tn_to_tnp1(vn);
    dt = surf->get_dt_n();
    ierr = PetscPrintf(mpi.comm(), "\nIteration #%04d:  tn = %.5e,  "
                                   "percent done = %.1f%%,  "
                                   "integral of Gamma = %.6e,  "
                                   "number of leaves = %d\n",
                       iter,
                       tn,
                       100.0*tn/tf,
                       surf->get_integrated_Gamma(),
                       surf->get_p4est_n()->global_num_quadrants); CHKERRXX(ierr);
    if(save_vtk)
    {
      sprintf(vtk_name, "%s/snapshot_%04d", vtk_path, iter);
      surf->save_vtk(vtk_name);
      ierr = PetscPrintf(mpi.comm(), "  -> Saving result vtk files in %s...\n",vtk_name); CHKERRXX(ierr);
    }
    if(save_errors)
    {
      sprintf(dat_error_space_name, "%s/error_snapshot_%04d.dat", error_path, iter);
      compute_and_save_errors(surf, dat_error_space_name, dat_error_time_name);
      ierr = PetscPrintf(mpi.comm(), "  -> Saving errors vtk files in %s...\n", dat_error_space_name); CHKERRXX(ierr);
    }
  }

  // Destroy the dynamically allocated classes
  delete surf;
  delete ls_n;
  delete ls_nm1;
  my_p4est_brick_destroy(conn, &brick);

  // Stop and print timer
  w.stop();
  ierr = PetscPrintf(mpi.comm(),"\n"); CHKERRXX(ierr);
  w.print_duration();

  // Finish
  return 0;
}
