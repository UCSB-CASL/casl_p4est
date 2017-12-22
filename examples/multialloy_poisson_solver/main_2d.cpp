
/*
 * Test the cell based p4est.
 * 1 - solve a poisson equation on an irregular domain (circle)
 * 2 - interpolate from faces to nodes
 * 3 - extrapolate faces over interface
 *
 * run the program with the -help flag to see the available options
 */

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
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX
int lmin = 5;
int lmax = 7;
int nb_splits = 6;

double lip = 3.5;

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

double velocity_tol = 1.e-8;

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
    lambda               = thermal_conductivity/(rho*heat_capacity); /* cm2.s-1  thermal diffusivity */
    eps_c                = 2.7207e-5;
    eps_v                = 2.27e-2;
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
        return sin(x)*cos(y);
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
        return cos(x)*cos(y);
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
        return -sin(x)*sin(y);
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
        return -2.*sin(x)*cos(y);
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

class vn_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double nx = phi_x_cf(x,y);
    double ny = phi_y_cf(x,y);
    double norm = sqrt(nx*nx+ny*ny);
    nx /= norm; ny /= norm;
    return thermal_conductivity/latent_heat*((tm_x_exact(x,y) - tp_x_exact(x,y))*nx + (tm_y_exact(x,y) - tp_y_exact(x,y))*ny);
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
    return 1.0;
    return -(1-kp[1])/Dl[1]*vn_cf(x,y);
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

class gibbs_thompson_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return t_exact(x,y) - ml[0]*c0_exact(x,y) - ml[0]*c1_exact(x,y);
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

  double err_ex_n;
  double err_ex_nm1;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &phi_cf, lip);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
//    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
//    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, phi_cf, phi);

    my_p4est_level_set_t ls(&ngbd_n);
//    ls.perturb_level_set_function(phi, EPS);

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
//    dt = cfl_number*dx/V;

    /* TEST THE NODES FUNCTIONS */
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
    bc_t.setInterfaceType(NOINTERFACE);
//    bc_t.setInterfaceValue(tp_exact);
//    bc_t.setRobinCoef(tm_exact);

    bc_c0.setWallTypes(bc_wall_type_c0);
    bc_c0.setWallValues(c0_exact);
    bc_c0.setInterfaceType(DIRICHLET);
    bc_c0.setInterfaceValue(c0_exact);

    bc_c1.setWallTypes(bc_wall_type_c1);
    bc_c1.setWallValues(c1_exact);
    bc_c1.setInterfaceType(ROBIN);
    bc_c1.setInterfaceValue(c1_interface_val_cf);
    bc_c1.setRobinCoef(c1_robin_coef_cf);

    Vec rhs_t;  ierr = VecCreateGhostNodes(p4est, nodes, &rhs_t); CHKERRXX(ierr);
    Vec rhs_tm; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_tm); CHKERRXX(ierr);
    Vec rhs_tp; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_tp); CHKERRXX(ierr);
    Vec rhs_c0; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_c0); CHKERRXX(ierr);
    Vec rhs_c1; ierr = VecCreateGhostNodes(p4est, nodes, &rhs_c1); CHKERRXX(ierr);

//    sample_cf_on_nodes(p4est, nodes, rhs_tp_cf, rhs_t);
    sample_cf_on_nodes(p4est, nodes, rhs_c0_cf, rhs_c0);
    sample_cf_on_nodes(p4est, nodes, rhs_c1_cf, rhs_c1);

    sample_cf_on_nodes(p4est, nodes, rhs_tm_cf, rhs_tm);
    sample_cf_on_nodes(p4est, nodes, rhs_tp_cf, rhs_tp);

    my_p4est_poisson_nodes_t solver_t(&ngbd_n);
    solver_t.set_phi(phi);
    solver_t.set_diagonal(diag_add);
    solver_t.set_mu(dt*lambda);
    solver_t.set_bc(bc_t);
    solver_t.set_use_refined_cube(false);
    solver_t.assemble_jump_rhs(rhs_t, jump_t_cf, jump_tn_cf, rhs_tm, rhs_tp);
    solver_t.set_rhs(rhs_t);
    solver_t.set_use_pointwise_dirichlet(false);

    my_p4est_poisson_nodes_t solver_c0(&ngbd_n);
    solver_c0.set_phi(phi);
    solver_c0.set_diagonal(diag_add);
    solver_c0.set_mu(dt*Dl[0]);
    solver_c0.set_bc(bc_c0);
    solver_c0.set_rhs(rhs_c0);
    solver_c0.set_use_refined_cube(true);
//    solver_c0.set_use_pointwise_dirichlet(false);

    my_p4est_poisson_nodes_t solver_c1(&ngbd_n);
    solver_c1.set_phi(phi);
    solver_c1.set_diagonal(diag_add);
    solver_c1.set_mu(dt*Dl[1]);
    solver_c1.set_bc(bc_c1);
    solver_c1.set_rhs(rhs_c1);
    solver_c1.set_use_refined_cube(false);

    Vec sol_t;  ierr = VecCreateGhostNodes(p4est, nodes, &sol_t);  CHKERRXX(ierr);
    Vec sol_c0; ierr = VecCreateGhostNodes(p4est, nodes, &sol_c0); CHKERRXX(ierr);
    Vec sol_c1; ierr = VecCreateGhostNodes(p4est, nodes, &sol_c1); CHKERRXX(ierr);

    solver_c0.set_use_pointwise_dirichlet(true);
    solver_c0.assemble_matrix(sol_c0);

    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      for (short i = 0; i < solver_c0.pointwise_bc[n].size(); ++i)
      {
        double xyz[P4EST_DIM];
        solver_c0.get_xyz_interface_point(n, i, xyz);
        double c0_gamma = (c0_exact)(xyz[0],xyz[1]);

        solver_c0.set_interface_point_value(n, i, c0_gamma);
      }
    }

//    solver_t.solve_t(sol_t, 0, KSPBCGS, PCHYPRE);
//    sample_cf_on_nodes(p4est, nodes, t_exact, sol_t);
    solver_t.solve(sol_t, 0, KSPBCGS, PCHYPRE);
//    solver_t.solve(sol_t, 0, KSPBCGS, PCSOR);
    solver_c0.solve(sol_c0, 0, KSPBCGS, PCHYPRE);
    solver_c1.solve(sol_c1, 0, KSPBCGS, PCHYPRE);

    /* check the error */
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
        err_t_p[n] =  fabs(sol_t_p[n]  - t_exact.value(xyz));
        err_c0_p[n] = fabs(sol_c0_p[n] - c0_exact.value(xyz));
        err_c1_p[n] = fabs(sol_c1_p[n] - c1_exact.value(xyz));
      } else {
        err_t_p[n] = fabs(sol_t_p[n] - t_exact.value(xyz));
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

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_t_n,  1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c0_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_c1_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    ierr = PetscPrintf(p4est->mpicomm, "Error in T  on nodes : %g, order = %g\n", err_t_n,  log(err_t_nm1 /err_t_n )/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C0 on nodes : %g, order = %g\n", err_c0_n, log(err_c0_nm1/err_c0_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error in C1 on nodes : %g, order = %g\n", err_c1_n, log(err_c1_nm1/err_c1_n)/log(2)); CHKERRXX(ierr);

    /* extrapolate the solution and check accuracy */
    double band = 2;

//    ls.extend_Over_Interface_TVD(phi, sol_c0,100);
    ls.extend_Over_Interface_TVD(phi, sol_c0, &solver_c0, 100);

    ierr = VecGetArrayRead(sol_c0, &sol_c0_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    Vec err_ex;
    ierr = VecDuplicate(sol_c0, &err_ex); CHKERRXX(ierr);
    double *err_ex_p;
    ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    err_ex_nm1 = err_ex_n;
    err_ex_n = 0;

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(phi_p[n]>0)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        if(phi_p[n]<band*diag)
          err_ex_p[n] = fabs(sol_c0_p[n] - c0_exact(x,y,z));
#else
        if(phi_p[n]<band*diag)
          err_ex_p[n] = fabs(sol_c0_p[n] - c0_exact(x,y));
#endif
        else
          err_ex_p[n] = 0;

        err_ex_n = MAX(err_ex_n, err_ex_p[n]);
      }
      else
        err_ex_p[n] = 0;
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_c0, &sol_c0_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Error extrapolation : %g, order = %g\n", err_ex_n, log(err_ex_nm1/err_ex_n)/log(2)); CHKERRXX(ierr);


    //-------------------------------------------------------------------------------------------
    // Save output
    //-------------------------------------------------------------------------------------------
    if(save_vtk)
    {
      PetscErrorCode ierr;
      const char *out_dir = getenv("OUT_DIR");
      if (!out_dir) {
        out_dir = "out_dir/vtu";
        system((string("mkdir -p ")+out_dir).c_str());
      }

      std::ostringstream oss;

      oss << out_dir
          << "/multiallo_poisson_solver_"
          << "proc="
          << p4est->mpisize << "_"
             << "brick="
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
           #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
           #endif
             "_levels=(" <<lmin << "," << lmax << ")" <<
             ".split=" << iter;

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
                             7, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "sol_t",  sol_t_p,
                             VTK_POINT_DATA, "sol_c0", sol_c0_p,
                             VTK_POINT_DATA, "sol_c1", sol_c1_p,
                             VTK_POINT_DATA, "err_t",  err_t_p,
                             VTK_POINT_DATA, "err_c0", err_c0_p,
                             VTK_POINT_DATA, "err_c1", err_c1_p,
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


//      ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

      PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    ierr = VecDestroy(rhs_t); CHKERRXX(ierr);
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

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
