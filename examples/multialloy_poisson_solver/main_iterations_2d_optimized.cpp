// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX
int lmin = 5;
int lmax = 5;
int nb_splits = 4;

double lip = 1.5;

using namespace std;

/* 0 - NiCu
 */
int alloy_type = 0;

double box_size = 4e-2;     //equivalent width (in x) of the box in cm - for plane convergence, 5e-3
double scaling = 1/box_size;

double xmin = 0;
double ymin = 0;
double zmin = 0;
double xmax = 1;
double ymax = 1;
double zmax = 1;
int n_xyz[] = {1, 1, 1};

double r_0 = 0.3;
double x_0 = 0.5+0.09;
double y_0 = 0.5-0.03;
double z_0 = 0.5+0.07;

const int num_comps = 2;
double rho;                  /* density                                    - kg.cm-3      */
double heat_capacity;        /* c, heat capacity                           - J.kg-1.K-1   */
double Tm;                   /* melting temperature                        - K            */
double G;                    /* thermal gradient                           - k.cm-1       */
double V;                    /* cooling velocity                           - cm.s-1       */
double latent_heat;          /* L, latent heat                             - J.cm-3       */
double thermal_conductivity; /* k, thermal conductivity                    - W.cm-1.K-1   */
double lambda;               /* thermal diffusivity                        - cm2.s-1      */

double ml[num_comps];         /* liquidus slope                             - K / at frac. */
double kp[num_comps];         /* partition coefficient                                     */
double c0[num_comps];         /* initial concentration                      - at frac.     */
double Dl[num_comps];         /* liquid concentration diffusion coefficient - cm2.s-1      */

double eps_c;                /* curvature undercooling coefficient         - cm.K         */
double eps_v;                /* kinetic undercooling coefficient           - s.K.cm-1     */
double eps_anisotropy;       /* anisotropy coefficient                                    */

double cfl_number = 0.1;
double dt;

void set_alloy_parameters()
{
  switch(alloy_type)
  {
  case 0:
    /* those are the default parameters for Ni-0.25831at%Cu-0.15at%Cu = Ni-0.40831at%Cu */
    rho                  = 8.88e-3;        /* kg.cm-3    */
    heat_capacity        = 0.46e3;         /* J.kg-1.K-1 */
    Tm                   = 1728;           /* K           */
    G                    = 4e2;            /* k.cm-1      */
    V                    = 0.01;           /* cm.s-1      */
    latent_heat          = 2350;           /* J.cm-3      */
    thermal_conductivity = 6.07e-1;        /* W.cm-1.K-1  */
//    thermal_conductivity = 100;        /* W.cm-1.K-1  */
    lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
//    eps_c                = 2.7207e-5;
//    eps_v                = 2.27e-2;
//    eps_v                = 1;
//    eps_c                = 1;
    eps_c                = 0;
    eps_v                = 0;
    eps_anisotropy       = 0.05;

    Dl[0] = 1e-5;
//    Dl[0] = 1;
    ml[0] =-357;
    c0[0] = 0.4;
    kp[0] = 0.86;

    Dl[1] = 1e-5;
//    Dl[1] = 1;
    ml[1] =-357;
    c0[1] = 0.4;
    kp[1] = 0.86;

    break;
  case 1:
      break;
  }
}

bool save_vtk = true;

/*
 * 0 - circle
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 -
 */
int test_number = 0;


double diag_add = 1;

class eps_v_cf_t : public CF_1
{
public:
  double operator()(double x) const
  {
    return eps_v*(1.0-15.0*eps_anisotropy*cos(4.0*x));
  }
} eps_v_cf;

class eps_c_cf_t : public CF_1
{
public:
  double operator()(double x) const
  {
    return eps_c*(1.0-15.0*eps_anisotropy*cos(4.0*x));
  }
} eps_c_cf;


#ifdef P4_TO_P8
#else

class zero_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 0;
  }
} zero_cf;

//------------------------------------------------------------
// Level-Set Function
//------------------------------------------------------------
class phi_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return r_0 - sqrt(SQR(x - x_0) + SQR(y - y_0));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_cf;

class phi_x_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return -(x-x_0)/sqrt(SQR(x - x_0) + SQR(y - y_0));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_x_cf;

class phi_y_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return -(y-y_0)/sqrt(SQR(x - x_0) + SQR(y - y_0));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_y_cf;

class phi_xx_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return -(y-y_0)*(y-y_0)/pow(SQR(x - x_0) + SQR(y - y_0), 1.5);
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_xx_cf;

class phi_xy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
        return (x-x_0)*(y-y_0)/pow(SQR(x - x_0) + SQR(y - y_0), 1.5);
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_xy_cf;

class phi_yy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
        return -(x-x_0)*(x-x_0)/pow(SQR(x - x_0) + SQR(y - y_0), 1.5);
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} phi_yy_cf;

class kappa_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return 1.1;
    return (SQR(phi_x_cf(x,y))*phi_yy_cf(x,y) - 2.*phi_x_cf(x,y)*phi_y_cf(x,y)*phi_xy_cf(x,y) + SQR(phi_y_cf(x,y))*phi_xx_cf(x,y))/pow(SQR(phi_x_cf(x,y))+SQR(phi_y_cf(x,y)), 1.5);
  }
} kappa_cf;

class theta_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double norm = sqrt(SQR(phi_x_cf(x,y)) + SQR(phi_y_cf(x,y)));
    return atan2(phi_y_cf(x,y)/norm, phi_x_cf(x,y)/norm);
  }
} theta_cf;



//------------------------------------------------------------
// Concentration 0
//------------------------------------------------------------
class c0_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*(1.1+sin(x)*cos(y));
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_exact;

class c0_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return 0.5*cos(x)*cos(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_x_exact;

class c0_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -0.5*sin(x)*sin(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_y_exact;

class c0_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -2.*0.5*sin(x)*cos(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c0_dd_exact;

//------------------------------------------------------------
// Concentration 1
//------------------------------------------------------------
class c1_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*sin(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_exact;

class c1_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0: return -sin(x)*sin(y);
      default: throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_x_exact;

class c1_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*cos(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_y_exact;

class c1_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -2.*cos(x)*sin(y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} c1_dd_exact;

//------------------------------------------------------------
// Temperature
//------------------------------------------------------------

class tm_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
//        return sin(x)*(y+y*y);
        return x*x + x*y + y*y;
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_exact;

class tm_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
//        return cos(x)*(y+y*y);
        return 2.*x + y;
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_x_exact;

class tm_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
//        return sin(x)*(1.+2.*y);
        return x + 2.*y;
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_y_exact;

class tm_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
//        return -sin(x)*(y+y*y) + 2.*sin(x);
        return 4.;
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tm_dd_exact;

class tp_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(y+y*y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_exact;

class tp_x_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return cos(x)*(y+y*y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_x_exact;

class tp_y_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return sin(x)*(1.+2.*y);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_y_exact;

class tp_dd_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
      case 0:
        return -sin(x)*(y+y*y) + 2.*sin(x);
      default:
        throw std::invalid_argument("Choose a valid test.");
    }
  }
} tp_dd_exact;

class t_exact_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if (phi_cf(x,y) < 0)  return tm_exact(x,y);
    else                  return tp_exact(x,y);
  }
} t_exact;

//------------------------------------------------------------
// right-hand-sides
//------------------------------------------------------------
class rhs_c0_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*c0_exact(x,y) - dt*Dl[0]*c0_dd_exact(x,y);
  }
} rhs_c0_cf;

class rhs_c1_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*c1_exact(x,y) - dt*Dl[1]*c1_dd_exact(x,y);
  }
} rhs_c1_cf;

class rhs_tm_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*tm_exact(x,y) - dt*lambda*tm_dd_exact(x,y);
  }
} rhs_tm_cf;

class rhs_tp_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return diag_add*tp_exact(x,y) - dt*lambda*tp_dd_exact(x,y);
  }
} rhs_tp_cf;

//------------------------------------------------------------
// jumps in t
//------------------------------------------------------------
class jump_t_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tm_exact(x,y) - tp_exact(x,y);
  }
} jump_t_cf;

class jump_tn_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return (tm_x_exact(x,y) - tp_x_exact(x,y))*nx + (tm_y_exact(x,y) - tp_y_exact(x,y))*ny;
  }
} jump_tn_cf;

//class jump_psi_tn_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return 1./ml[0]/lambda;
//  }
//} jump_psi_tn_cf;

class vn_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return (thermal_conductivity/latent_heat*((tm_x_exact(x,y) - tp_x_exact(x,y))*nx + (tm_y_exact(x,y) - tp_y_exact(x,y))*ny));
  }
} vn_cf;

//------------------------------------------------------------
// bc for c1
//------------------------------------------------------------
class c1_robin_coef_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return 1.0;
    return -(1.-kp[1])/Dl[1]*vn_cf(x,y);
  }
} c1_robin_coef_cf;

class c1_interface_val_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return c1_x_exact(x,y)*nx + c1_y_exact(x,y)*ny + c1_robin_coef_cf(x,y)*c1_exact(x,y);
  }
} c1_interface_val_cf;

//class psi_c1_interface_val_cf_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return -ml[1]/ml[0]/Dl[1];
//  }
//} psi_c1_interface_val_cf;


class bc_wall_type_t_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_t;

class bc_wall_type_c0_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_c0;

class bc_wall_type_c1_t : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_c1;





class c0_interface_val_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return c0_x_exact(x,y)*nx + c0_y_exact(x,y)*ny - (1.-kp[0])/Dl[0]*vn_cf(x,y)*c0_exact(x,y);
  }
} c0_interface_val_cf;



//class vn_from_c0_t : public CF_2
//{
//  CF_2 *c0, *dc0dn;
//public:
//  void set_input(CF_2& c0_, CF_2& dc0dn_) {c0 = &c0_; dc0dn = &dc0dn_;}
//  double operator()(double x, double y) const
//  {
//    return Dl[0]/(1.-kp[0])*((*dc0dn)(x,y) - c0_interface_val_cf(x,y))/(*c0)(x,y);
//  }
//} vn_from_c0;

//class c1_robin_from_c0_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return -(1.-kp[1])/Dl[1]*vn_from_c0(x,y);
//  }
//} c1_robin_from_c0;

//class tn_jump_from_c0_t : public CF_2
//{
//public:
//  double operator()(double x, double y) const
//  {
//    return latent_heat/thermal_conductivity*vn_from_c0(x,y);
//  }
//} tn_jump_from_c0;



class c0_guess_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 1.;
//    return c0_exact(x,y);
//    return c0_exact(x,y) + 0.1;
  }
} c0_guess;



class gibbs_thompson_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return tm_exact(x,y) - Tm - ml[0]*c0_exact(x,y) - ml[1]*c1_exact(x,y) - eps_c_cf(theta_cf(x,y))*kappa_cf(x,y) + eps_v_cf(theta_cf(x,y))*vn_cf(x,y);
  }
} gibbs_thompson;


#endif


int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
//  cmd.add_option("lmin", "min level of the tree");
//  cmd.add_option("lmax", "max level of the tree");
//  cmd.add_option("nb_splits", "number of recursive splits");
//  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
//  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
//  cmd.add_option("save_vtk", "save the p4est in vtk format");
//#ifdef P4_TO_P8
//  cmd.add_option("test", "choose a test.\n\
//                 0 - x+y+z\n\
//                 1 - x*x + y*y + z*z\n\
//                 2 - sin(x)*cos(y)*exp(z)");
//#else
//  cmd.add_option("test", "choose a test.\n\
//                 0 - x+y\n\
//                 1 - x*x + y*y\n\
//                 2 - sin(x)*cos(y)\n\
//                 3 - sin(x) + cos(y)");
//#endif
  cmd.parse(argc, argv);

//  lmin = cmd.get("lmin", lmin);
//  lmax = cmd.get("lmax", lmax);
//  nb_splits = cmd.get("nb_splits", nb_splits);
//  test_number = cmd.get("test", test_number);

//  bc_wtype = cmd.get("bc_wtype", bc_wtype);
//  bc_itype = cmd.get("bc_itype", bc_itype);

//  save_vtk = cmd.get("save_vtk", save_vtk);

  set_alloy_parameters();

  scaling = 1/box_size;
  rho                  /= (scaling*scaling*scaling);
  thermal_conductivity /= scaling;
  G                    /= scaling;
  V                    *= scaling;
  latent_heat          /= (scaling*scaling*scaling);
  eps_c                *= scaling;
  eps_v                /= scaling;
  lambda                = thermal_conductivity/(rho*heat_capacity);


  for (short i = 0; i < num_comps; ++i)
    Dl[i]              *= (scaling*scaling);


  dt = 1;


  parStopWatch w;
  w.start("total time");

  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};
  const int periodic[] = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_t_n,  err_t_nm1;
  double err_c0_n, err_c0_nm1;
  double err_c1_n, err_c1_nm1;
  double err_vn_n, err_vn_nm1;
  double err_kappa_n, err_kappa_nm1;
  double err_theta_n, err_theta_nm1;

//  double err_ex_n;
//  double err_ex_nm1;

  vector<double> h, e_v, e_t, e_c0, e_c1;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data_tmp(lmin, lmax, &phi_cf, lip);
    p4est->user_pointer = (void*)(&data_tmp);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    for (int i = 0; i < iter; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
//      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &phi_cf, lip);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);
    ngbd_n.init_neighbors();

    my_p4est_level_set_t ls(&ngbd_n);

    /* find dx and dy smallest */
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
    double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);

#ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
    double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double diag = sqrt(dx*dx + dy*dy);
#endif

    // compute dt
//    dt = cfl_number*diag/V;

    /* Initialize LSF */
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, phi_cf, phi);

    if (0) {
      double xyz[P4EST_DIM];
      double *phi_ptr;
      srand(mpi.rank());

      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);

        double d_phi = 0.001*dx*dx*dx*(double)(rand()%1000)/1000.;
//        double d_phi = 0.01*dx*dx*cos(2.*PI*data.max_lvl*xyz[0])*sin(2.*PI*data.max_lvl*xyz[1]);
        phi_ptr[n] += d_phi;
      }
      ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
      ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);
    }

    ls.reinitialize_1st_order_time_2nd_order_space(phi, 100);
//    ls.reinitialize_1st_order(phi, 100);
//    ls.reinitialize_2nd_order(phi, 100);
    ls.perturb_level_set_function(phi, EPS);

    Vec phi_dd[P4EST_DIM];

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_dd[dir]); CHKERRXX(ierr);
    }

    ngbd_n.second_derivatives_central(phi, phi_dd);

    // compute normal
    Vec normal[P4EST_DIM];
    Vec kappa; ierr = VecDuplicate(phi, &kappa); CHKERRXX(ierr);
    Vec theta; ierr = VecDuplicate(phi, &theta); CHKERRXX(ierr);

    {
      double *normal_p[P4EST_DIM];

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecCreateGhostNodes(p4est, nodes, &normal[dir]); CHKERRXX(ierr);
        ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
      }

      const double *phi_p;
      ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

      quad_neighbor_nodes_of_node_t qnnn;

      for(size_t i=0; i<ngbd_n.get_layer_size(); ++i)
      {
        p4est_locidx_t n = ngbd_n.get_layer_node(i);
        qnnn = ngbd_n.get_neighbors(n);
        normal_p[0][n] = qnnn.dx_central(phi_p);
        normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        normal_p[2][n] = qnnn.dz_central(phi_p);
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

//        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
//        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
//#ifdef P4_TO_P8
//        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
//#endif
      }

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecGhostUpdateBegin(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }

      for(size_t i=0; i<ngbd_n.get_local_size(); ++i)
      {
        p4est_locidx_t n = ngbd_n.get_local_node(i);
        qnnn = ngbd_n.get_neighbors(n);
        normal_p[0][n] = qnnn.dx_central(phi_p);
        normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        normal_p[2][n] = qnnn.dz_central(phi_p);
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
//        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

//        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
//        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
//#ifdef P4_TO_P8
//        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
//#endif
      }


      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecGhostUpdateEnd(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }

      // compute curvature and angle
      /* curvature */
      Vec kappa_tmp;
      ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);
      double *kappa_p;
      ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
      for(size_t i=0; i<ngbd_n.get_layer_size(); ++i)
      {
        p4est_locidx_t n = ngbd_n.get_layer_node(i);
        qnnn = ngbd_n.get_neighbors(n);
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    #ifdef P4_TO_P8
        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1/dxyz_max), -1/dxyz_max);
    #else
        if (is_node_Wall(p4est, ni))
        {
          kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]);
        } else {
          double phi_x = normal_p[0][n];
          double phi_y = normal_p[1][n];
//          ierr = PetscPrintf(p4est->mpicomm, "Here\n"); CHKERRXX(ierr);
          double phi_xx = qnnn.dxx_central(phi_p);
          double phi_yy = qnnn.dyy_central(phi_p);
          double phi_xy = .5*(qnnn.dx_central(normal_p[1])+qnnn.dy_central(normal_p[0]));
          kappa_p[n] = (phi_x*phi_x*phi_yy - 2.*phi_x*phi_y*phi_xy + phi_y*phi_y*phi_xx)/pow(phi_x*phi_x+phi_y*phi_y,3./2.);
//          kappa_p[n] = phi_yy + phi_xx;
        }
//        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1/dx), -1/dx);
    #endif
      }
      ierr = VecGhostUpdateBegin(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
      for(size_t i=0; i<ngbd_n.get_local_size(); ++i)
      {
        p4est_locidx_t n = ngbd_n.get_local_node(i);
        qnnn = ngbd_n.get_neighbors(n);
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    #ifdef P4_TO_P8
        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1/dxyz_max), -1/dxyz_max);
    #else
        if (is_node_Wall(p4est, ni))
        {
          kappa_p[n] = qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]);
        } else {
          double phi_x = normal_p[0][n];
          double phi_y = normal_p[1][n];
          double phi_xx = qnnn.dxx_central(phi_p);
          double phi_yy = qnnn.dyy_central(phi_p);
          double phi_xy = .5*(qnnn.dx_central(normal_p[1])+qnnn.dy_central(normal_p[0]));
          kappa_p[n] = (phi_x*phi_x*phi_yy - 2.*phi_x*phi_y*phi_xy + phi_y*phi_y*phi_xx)/pow(phi_x*phi_x+phi_y*phi_y,3./2.);
//          kappa_p[n] = phi_yy + phi_xx;
        }
//        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1/dx), -1/dx);
    #endif
      }
      ierr = VecGhostUpdateEnd(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
      ierr = VecRestoreArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
    #ifdef P4_TO_P8
        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
    #else
        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
    #endif

        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
    #ifdef P4_TO_P8
        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
    #endif
      }

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
      }

      my_p4est_level_set_t ls(&ngbd_n);
      ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
      ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);

//      ierr = VecDestroy(kappa); CHKERRXX(ierr);
//      kappa = kappa_tmp;

//      sample_cf_on_nodes(p4est, nodes, kappa_cf, kappa);

      /* angle between normal and direction of growth */

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
      }

    #ifdef P4_TO_P8
      Vec theta_xz_tmp; double *theta_xz_tmp_p;
      Vec theta_yz_tmp; double *theta_yz_tmp_p;
      ierr = VecDuplicate(phi, &theta_xz_tmp); CHKERRXX(ierr);
      ierr = VecDuplicate(phi, &theta_yz_tmp); CHKERRXX(ierr);
      ierr = VecGetArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
      ierr = VecGetArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
    #else
      Vec theta_tmp; double *theta_tmp_p;
      ierr = VecDuplicate(phi, &theta_tmp); CHKERRXX(ierr);
      ierr = VecGetArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
    #endif


      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      {
    #ifdef P4_TO_P8
        theta_xz_tmp_p[n] = atan2(normal_p[2][n], normal_p[0][n]);
        theta_yz_tmp_p[n] = atan2(normal_p[2][n], normal_p[1][n]);
    #else
        theta_tmp_p[n] = atan2(normal_p[1][n], normal_p[0][n]);
    #endif
      }

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
      }

    #ifdef P4_TO_P8
      ierr = VecRestoreArray(theta_xz_tmp, &theta_xz_tmp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(theta_yz_tmp, &theta_yz_tmp_p); CHKERRXX(ierr);
      ls.extend_from_interface_to_whole_domain_TVD(phi, theta_xz_tmp, theta_xz);
      ls.extend_from_interface_to_whole_domain_TVD(phi, theta_yz_tmp, theta_yz);
      ierr = VecDestroy(theta_xz_tmp); CHKERRXX(ierr);
      ierr = VecDestroy(theta_yz_tmp); CHKERRXX(ierr);
    #else
    //  ierr = VecRestoreArray(theta_tmp, &theta_tmp_p); CHKERRXX(ierr);
    //  ls.extend_from_interface_to_whole_domain_TVD(phi, theta_tmp, theta);
    //  ierr = VecDestroy(theta_tmp); CHKERRXX(ierr);
      ierr = VecDestroy(theta); CHKERRXX(ierr);
      theta = theta_tmp;
    #endif

    }


    /* Sample RHS */
    Vec rhs_c0; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_c0); CHKERRXX(ierr);
    Vec rhs_c1; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_c1); CHKERRXX(ierr);
    Vec rhs_tm; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_tm); CHKERRXX(ierr);
    Vec rhs_tp; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_tp); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, rhs_c0_cf, rhs_c0);
    sample_cf_on_nodes(p4est, nodes, rhs_c1_cf, rhs_c1);
    sample_cf_on_nodes(p4est, nodes, rhs_tm_cf, rhs_tm);
    sample_cf_on_nodes(p4est, nodes, rhs_tp_cf, rhs_tp);

    /* set boundary conditions */
#ifdef P4_TO_P8
    BoundaryConditions3D bc_t;
    BoundaryConditions3D bc_c0;
    BoundaryConditions3D bc_c1;
#else
    BoundaryConditions2D bc_t;
    BoundaryConditions2D bc_c0;
    BoundaryConditions2D bc_c1;
#endif

    bc_t.setWallTypes(bc_wall_type_t);
    bc_t.setWallValues(t_exact);

    bc_c0.setWallTypes(bc_wall_type_c0);
    bc_c0.setWallValues(c0_exact);

    bc_c1.setWallTypes(bc_wall_type_c1);
    bc_c1.setWallValues(c1_exact);

    my_p4est_poisson_nodes_multialloy_t solver_all_in_one(&ngbd_n);

    int pin_every_n_steps = 3;
    double bc_tolerance = 1.e-9;
    int max_iterations = 1000;

    solver_all_in_one.set_phi(phi, phi_dd, normal, kappa, theta, NULL);
    solver_all_in_one.set_parameters(dt, lambda, thermal_conductivity, latent_heat, Tm, Dl[0], kp[0], ml[0], Dl[1], kp[1], ml[1]);
    solver_all_in_one.set_bc(bc_t, bc_c0, bc_c1);
    solver_all_in_one.set_GT(gibbs_thompson);
    solver_all_in_one.set_undercoolings(eps_v_cf, eps_c_cf);
    solver_all_in_one.set_pin_every_n_steps(pin_every_n_steps);
    solver_all_in_one.set_tolerance(bc_tolerance, max_iterations);
    solver_all_in_one.set_rhs(rhs_tm, rhs_tp, rhs_c0, rhs_c1);

    solver_all_in_one.set_jump_t(jump_t_cf);
    solver_all_in_one.set_flux_c(c0_interface_val_cf, c1_interface_val_cf);
    solver_all_in_one.set_c0_guess(c0_guess);

    Vec sol_t;  ierr = VecCreateGhostNodes(p4est, nodes, &sol_t);  CHKERRXX(ierr);
    Vec sol_c0; ierr = VecCreateGhostNodes(p4est, nodes, &sol_c0); CHKERRXX(ierr);
    Vec sol_c1; ierr = VecCreateGhostNodes(p4est, nodes, &sol_c1); CHKERRXX(ierr);

    Vec sol_t_dd[P4EST_DIM];
    Vec sol_c0_dd[P4EST_DIM];
    Vec sol_c1_dd[P4EST_DIM];
    Vec bc_error;

    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecDuplicate(phi_dd[dim], &sol_t_dd[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_dd[dim], &sol_c0_dd[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_dd[dim], &sol_c1_dd[dim]); CHKERRXX(ierr);
    }

    ierr = VecDuplicate(phi, &bc_error); CHKERRXX(ierr);


    double bc_error_max = 0;
//    solver_all_in_one.solve(sol_t, sol_t_dd, sol_c0, sol_c0_dd, sol_c1, sol_c1_dd, bc_error_max, bc_error);
    solver_all_in_one.solve(sol_t, sol_c0, sol_c1, bc_error, bc_error_max);


    /* check the error */
    my_p4est_poisson_nodes_t *solver_c0 = solver_all_in_one.get_solver_c0();
    CF_2 *vn = solver_all_in_one.get_vn();

    double vn_error = 0;
    double kappa_error = 0;
    double theta_error = 0;
    double *kappa_p; ierr = VecGetArray(kappa, &kappa_p); CHKERRXX(ierr);
    double *theta_p; ierr = VecGetArray(theta, &theta_p); CHKERRXX(ierr);
    Vec err_vn;  ierr = VecDuplicate(sol_t,  &err_vn); CHKERRXX(ierr);
    Vec err_kappa;  ierr = VecDuplicate(sol_t,  &err_kappa); CHKERRXX(ierr);
    double *err_vn_p, *err_kappa_p;
    ierr = VecGetArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
    ierr = VecGetArray(err_vn,  &err_vn_p); CHKERRXX(ierr);

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      err_kappa_p[n] = 0;
      err_vn_p[n] = 0;
      for (short i = 0; i < solver_c0->pointwise_bc[n].size(); ++i)
      {
        double xyz[P4EST_DIM];
        solver_c0->get_xyz_interface_point(n, i, xyz);
        vn_error = MAX(vn_error, fabs(vn->value(xyz) - vn_cf.value(xyz)));
        kappa_error = MAX(kappa_error, fabs(kappa_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, kappa_p)));
        theta_error = MAX(theta_error, fabs(theta_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, theta_p)));

        err_kappa_p[n] = MAX(err_kappa_p[n], fabs(kappa_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, kappa_p)));
        err_vn_p[n] = MAX(err_vn_p[n], fabs(theta_cf.value(xyz) - solver_c0->interpolate_at_interface_point(n, i, theta_p)));
      }
    }
    ierr = VecRestoreArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_vn,  &err_vn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa, &kappa_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(theta, &theta_p); CHKERRXX(ierr);

    err_kappa_nm1 = err_kappa_n; err_kappa_n = kappa_error;
    err_theta_nm1 = err_theta_n; err_theta_n = theta_error;
    err_vn_nm1  = err_vn_n;  err_vn_n  = vn_error;
    err_t_nm1  = err_t_n;  err_t_n  = 0;
    err_c0_nm1 = err_c0_n; err_c0_n = 0;
    err_c1_nm1 = err_c1_n; err_c1_n = 0;

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    const double *sol_t_p, *sol_c0_p, *sol_c1_p;
    ierr = VecGetArrayRead(sol_t, &sol_t_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(sol_c0, &sol_c0_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(sol_c1, &sol_c1_p); CHKERRXX(ierr);

    Vec err_t_nodes;  ierr = VecDuplicate(sol_t,  &err_t_nodes); CHKERRXX(ierr);
    Vec err_c0_nodes; ierr = VecDuplicate(sol_c0, &err_c0_nodes); CHKERRXX(ierr);
    Vec err_c1_nodes; ierr = VecDuplicate(sol_c1, &err_c1_nodes); CHKERRXX(ierr);

    double *err_t_p, *err_c0_p, *err_c1_p;
    ierr = VecGetArray(err_t_nodes,  &err_t_p); CHKERRXX(ierr);
    ierr = VecGetArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
    ierr = VecGetArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

    double xyz[P4EST_DIM];
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);

      if(phi_p[n]<0)
      {
#ifdef P4_TO_P8
        err_t_p[n] =  fabs(sol_t_p[n]  - t_exact(xyz[0],xyz[1],xyz[2]));
        err_c0_p[n] = fabs(sol_c0_p[n] - c0_exact(xyz[0],xyz[1],xyz[2]));
        err_c1_p[n] = fabs(sol_c1_p[n] - c1_exact(xyz[0],xyz[1],xyz[2]));
#else
        err_t_p[n] =  fabs(sol_t_p[n]  - t_exact(xyz[0],xyz[1]));
        err_c0_p[n] = fabs(sol_c0_p[n] - c0_exact(xyz[0],xyz[1]));
        err_c1_p[n] = fabs(sol_c1_p[n] - c1_exact(xyz[0],xyz[1]));
#endif
      } else {

#ifdef P4_TO_P8
        err_t_p[n] = fabs(sol_t_p[n] - t_exact(xyz[0],xyz[1],xyz[2]));
#else
        err_t_p[n] = fabs(sol_t_p[n] - t_exact(xyz[0],xyz[1]));
#endif
        err_c0_p[n] = 0.;
        err_c1_p[n] = 0.;
      }

      err_t_n = MAX(err_t_n, err_t_p[n]);
      err_c0_n = MAX(err_c0_n, err_c0_p[n]);
      err_c1_n = MAX(err_c1_n, err_c1_p[n]);
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_t, &sol_t_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_c0, &sol_c0_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_c1, &sol_c1_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(err_t_nodes, &err_t_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_t_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_t_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_c0_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c0_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_c1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_c1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_kappa_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_theta_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_vn_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_t_n,      1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c0_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c1_n,     1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    ierr = PetscPrintf(p4est->mpicomm, "Error in kappa on nodes : %g, order = %g\n", err_kappa_n, log(err_kappa_nm1/err_kappa_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in theta on nodes : %g, order = %g\n", err_theta_n, log(err_theta_nm1/err_theta_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in vn on nodes : %g, order = %g\n", err_vn_n, log(err_vn_nm1/err_vn_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in T  on nodes : %g, order = %g\n", err_t_n,  log(err_t_nm1 /err_t_n )/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C0 on nodes : %g, order = %g\n", err_c0_n, log(err_c0_nm1/err_c0_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C1 on nodes : %g, order = %g\n", err_c1_n, log(err_c1_nm1/err_c1_n)/log(2)); CHKERRXX(ierr);

    h.push_back(dx);
    e_v.push_back(err_vn_n);
    e_t.push_back(err_t_n);
    e_c0.push_back(err_c0_n);
    e_c1.push_back(err_c1_n);

    //-------------------------------------------------------------------------------------------
    // Save output
    //-------------------------------------------------------------------------------------------
    if(save_vtk)
    {
      PetscErrorCode ierr;
      const char *out_dir = getenv("OUT_DIR");
      if (!out_dir) {
        out_dir = "out_dir";
        system((string("mkdir -p ")+out_dir).c_str());
      }

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/multiallo_poisson_solver_"
          << "proc"
          << p4est->mpisize << "_"
             << "brick"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
           #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
           #endif
//             "_levels=(" <<lmin << "," << lmax << ")" <<
             ".split" << iter;

      double *phi_p;
      double *sol_t_p, *err_t_p;
      double *sol_c0_p, *err_c0_p;
      double *sol_c1_p, *err_c1_p;

      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

      ierr = VecGetArray(sol_t,  &sol_t_p);  CHKERRXX(ierr);
      ierr = VecGetArray(sol_c0, &sol_c0_p); CHKERRXX(ierr);
      ierr = VecGetArray(sol_c1, &sol_c1_p); CHKERRXX(ierr);

      ierr = VecGetArray(err_t_nodes, &err_t_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

      ierr = VecGetArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
      ierr = VecGetArray(err_vn,  &err_vn_p); CHKERRXX(ierr);

      double *bc_error_p;

      ierr = VecGetArray(bc_error, &bc_error_p); CHKERRXX(ierr);

//      double *err_ex_p;
//      ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

      /* save the size of the leaves */
      Vec leaf_level;
      ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
      double *l_p;
      ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

      for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
          l_p[tree->quadrants_offset+q] = quad->level;
        }
      }

      for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
      {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
      }

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             10, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "sol_t",  sol_t_p,
                             VTK_POINT_DATA, "sol_c0", sol_c0_p,
                             VTK_POINT_DATA, "sol_c1", sol_c1_p,
                             VTK_POINT_DATA, "err_t",  err_t_p,
                             VTK_POINT_DATA, "err_c0", err_c0_p,
                             VTK_POINT_DATA, "err_c1", err_c1_p,
                             VTK_POINT_DATA, "bc_error", bc_error_p,
                             VTK_POINT_DATA, "kappa_error", err_kappa_p,
                             VTK_POINT_DATA, "vn_error", err_vn_p,
//                             VTK_POINT_DATA, "err_ex", err_ex_p,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol_t,  &sol_t_p);  CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_c0, &sol_c0_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_c1, &sol_c1_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(err_t_nodes, &err_t_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_c0_nodes, &err_c0_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_c1_nodes, &err_c1_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(err_kappa,  &err_kappa_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_vn,  &err_vn_p); CHKERRXX(ierr);

      ierr = VecRestoreArray(bc_error, &bc_error_p); CHKERRXX(ierr);

//      ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);


    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecDestroy(sol_t_dd[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_c0_dd[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_c1_dd[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(bc_error); CHKERRXX(ierr);

    ierr = VecDestroy(rhs_tm); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_tp); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_c0); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_c1); CHKERRXX(ierr);

    ierr = VecDestroy(sol_t); CHKERRXX(ierr);
    ierr = VecDestroy(sol_c0); CHKERRXX(ierr);
    ierr = VecDestroy(sol_c1); CHKERRXX(ierr);

    ierr = VecDestroy(err_t_nodes); CHKERRXX(ierr);
    ierr = VecDestroy(err_c0_nodes); CHKERRXX(ierr);
    ierr = VecDestroy(err_c1_nodes); CHKERRXX(ierr);

//    ierr = VecDestroy(err_ex); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  if (mpi.rank() == 0)
  {
    for (int i = 0; i < h.size(); ++i)
      std::cout << h[i] << " " << e_v[i] << " " << e_t[i] << " " << e_c0[i] << " " << e_c1[i] << "\n";
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
