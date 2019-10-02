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
/*------------------------------------------------------------- 3D -------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------*/

/*
 * NOTE: In 3D, we follow the ISO convention used in physics for spherical coordinates, i.e. phi in [0,2*pi) is the azimuthal angle and
 *       theta in [0, pi] is the polar angle.
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

struct star : public CF_3
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
  double operator()(double x, double y, double z) const
  {
    return r0*( 1 + alpha*(1-beta*cos((double)m*phi(x,y,z)))*(1-cos((double)n*theta(x,y,z))) );
  }
  double d_theta(double x, double y, double z) const
  {
    return r0*alpha*n*(1-beta*cos((double)m*phi(x,y,z)))*sin((double)n*theta(x,y,z));
  }
  double d_phi(double x, double y, double z) const
  {
    return r0*alpha*beta*m*sin((double)m*phi(x,y,z))*(1-cos((double)n*theta(x,y,z)));
  }
} r_star;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_level_set_t
{
  double operator()(double x, double y, double z, double t) const
  {
    switch(test_number)
    {
      case 0: return t + r_star(x,y,z) - rad(x,y,z); // NOTE: This is NOT a signed distance function for any t
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
      case 0: return 0.5*sqrt(SQR(  r_star(x,y,z))+SQR(r_star.d_theta(x,y,z)))*sqrt(SQR(  r_star(x,y,z))+SQR(r_star.d_phi(x,y,z)))
                       /(sqrt(SQR(t+r_star(x,y,z))+SQR(r_star.d_theta(x,y,z)))*sqrt(SQR(t+r_star(x,y,z))+SQR(r_star.d_phi(x,y,z))));
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
/*------------------------------------------------------------- 2D -------------------------------------------------------------------*/
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

struct star : public CF_2
{
  double r0, alpha;
  unsigned short n;

public:
  star(double r0_input=1.0, double alpha_input=0.25, unsigned short n_input=7)
  {
    r0    = r0_input;
    alpha = alpha_input;
    n     = n_input;
  }
  double operator()(double x, double y) const
  {
    return r0*(1-alpha*sin((double)n*phi(x,y)));
  }
  double d_phi(double x, double y) const
  {
    return -r0*alpha*n*cos((double)n*phi(x,y));
  }
} r_star;

/*------------------------------------------------------------------------------------------------------------------------------------*/

struct exact_level_set_t
{
  double operator()(double x, double y, double t) const
  {
    switch(test_number)
    {
      case 0: return t + r_star(x,y) - rad(x,y); // NOTE: This is NOT a signed distance function for any t
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
      case 0: return 0.5*sqrt(SQR(r_star(x,y))+SQR(r_star.d_phi(x,y)))/sqrt(SQR(t+r_star(x,y))+SQR(r_star.d_phi(x,y)));
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
  const int lmax = 8;
#ifdef P4_TO_P8
  dmin = MIN((xmax-xmin),(ymax-ymin),(zmax-zmin))/pow(2,lmax);
#else
  dmin = MIN((xmax-xmin),(ymax-ymin))/pow(2,lmax);
#endif

  // Time
  double CFL = 1.0;
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
  my_p4est_surfactant_t* surf = new my_p4est_surfactant_t(mpi, &brick, conn, lmin, lmax, ls_n, NULL, false, band_width, CFL, 1.2);

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
  char out_dir[1024], vtk_path[1024], vtk_name[1024];
  string export_dir = "/home/temprano/Output/p4est_surfactant/tests";
  string test_name;
  switch(test_number)
  {
    case 0: test_name = "advection_expansion";     break;
    case 1: test_name = "advection_vortex";        break;
    default: throw std::invalid_argument("Please choose a valid test.");
  }
  sprintf(out_dir, "%s/%dd/%s/%d_%d", export_dir.c_str(), (int)P4EST_DIM, test_name.c_str(), lmin, lmax);
  sprintf(vtk_path, "%s/vtu",  out_dir);

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

  // Time evolution
  double tf = 1.0;
  int iter = 0;
  if(save_vtk)
  {
    sprintf(vtk_name, "%s/snapshot_%d", vtk_path, iter);
    surf->save_vtk(vtk_name);
  }

  while(tn+0.01*dt < tf)
  {
    if(iter>0)
    {
      surf->update_from_tn_to_tnp1(vn);
      dt = surf->get_dt_n();

      if(tn+dt>tf)
      {
        dt = tf-tn;
        surf->set_dt_n(dt);
      }

      if(save_vtk)
      {
        sprintf(vtk_name, "%s/snapshot_%d", vtk_path, iter);
        surf->save_vtk(vtk_name);
      }
    }

    surf->advect_interface_one_step();
    surf->compute_one_step_Gamma();

    tn+=dt;
    iter++;

    PetscErrorCode ierr;
    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d:  tn = %.5e,  "
                                   "percent done = %.1f%%,  "
                                   "number of leaves = %d\n",
                       iter, tn, 100*tn/tf, surf->get_p4est_n()->global_num_quadrants);
    CHKERRXX(ierr);
  }

  // Stop and print timer
  w.stop();
  w.print_duration();

  // Destroy the dynamically allocated classes
  delete surf;
  delete ls_n;
  delete ls_nm1;
  my_p4est_brick_destroy(conn, &brick);

  // Finish
  return 0;
}
