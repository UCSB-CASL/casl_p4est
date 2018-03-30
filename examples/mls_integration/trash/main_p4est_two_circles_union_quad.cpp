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
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/simplex3_mls_vtk.h>
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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_integration_quad_mls.h>
#include <src/my_p4est_integration_refined_quad_mls.h>
#include <src/simplex2_mls_vtk.h>
#include <src/simplex2_quad_mls_vtk.h>
#endif

#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

/* discretization */
int lmin = 0;
int lmax = 3;
#ifdef P4_TO_P8
int nb_splits = 4;
#else
int nb_splits = 11;
#endif

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

bool save_vtk = true;

/* geometry */

double xmin = -1.00;
double xmax =  1.00;
double ymin = -1.00;
double ymax =  1.00;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

double r0 = 0.5;
double d = 0.35;

double theta = 0.579;
#ifdef P4_TO_P8
double phy = 0.312;
#endif

double cosT = cos(theta);
double sinT = sin(theta);
#ifdef P4_TO_P8
double cosP = cos(phy);
double sinP = sin(phy);
#endif

#ifdef P4_TO_P8
double xc_0 = -d*sinT*cosP; double yc_0 =  d*cosT*cosP; double zc_0 =  d*sinP;
double xc_1 =  d*sinT*cosP; double yc_1 = -d*cosT*cosP; double zc_1 = -d*sinP;
#else
double xc_0 = -d*sinT; double yc_0 =  d*cosT;
double xc_1 =  d*sinT; double yc_1 = -d*cosT;
#endif

#ifdef P4_TO_P8
class LS_CIRCLE_0: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)));
  }
} ls_circle_0;
#else
class LS_CIRCLE_0: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0)));
  }
} ls_circle_0;

class LS_CIRCLE_0_X: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_0) + SQR(y-yc_0));
    if (r < EPS)  return 0;
    else          return 1.0*(x-xc_0)/r;
  }
} ls_circle_0_x;

class LS_CIRCLE_0_Y: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_0) + SQR(y-yc_0));
    if (r < EPS)  return 0;
    else          return 1.0*(y-yc_0)/r;
  }
} ls_circle_0_y;

class LS_CIRCLE_0_XX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_0) + SQR(y-yc_0));
    if (r < EPS)  return 0;
    else          return 1.*(y-yc_0)*(y-yc_0)/r/r/r;
  }
} ls_circle_0_xx;

class LS_CIRCLE_0_XY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_0) + SQR(y-yc_0));
    if (r < EPS)  return 0;
    else          return -1.*(x-xc_0)*(y-yc_0)/r/r/r;
  }
} ls_circle_0_xy;

class LS_CIRCLE_0_YY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_0) + SQR(y-yc_0));
    if (r < EPS)  return 0;
    else          return 1.*(x-xc_0)*(x-xc_0)/r/r/r;
  }
} ls_circle_0_yy;
#endif


#ifdef P4_TO_P8
class LS_CIRCLE_1: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_1) + SQR(y-yc_1) + SQR(z-zc_1)));
  }
} ls_circle_1;
#else
class LS_CIRCLE_1: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_1) + SQR(y-yc_1)));
  }
} ls_circle_1;

class LS_CIRCLE_1_X: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_1) + SQR(y-yc_1));
    if (r < EPS)  return 0;
    else          return 1.0*(x-xc_1)/r;
  }
} ls_circle_1_x;

class LS_CIRCLE_1_Y: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_1) + SQR(y-yc_1));
    if (r < EPS)  return 0;
    else          return 1.0*(y-yc_1)/r;
  }
} ls_circle_1_y;

class LS_CIRCLE_1_XX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_1) + SQR(y-yc_1));
    if (r < EPS)  return 0;
    else          return 1.*(y-yc_1)*(y-yc_1)/r/r/r;
  }
} ls_circle_1_xx;

class LS_CIRCLE_1_XY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_1) + SQR(y-yc_1));
    if (r < EPS)  return 0;
    else          return -1.*(x-xc_1)*(y-yc_1)/r/r/r;
  }
} ls_circle_1_xy;

class LS_CIRCLE_1_YY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-xc_1) + SQR(y-yc_1));
    if (r < EPS)  return 0;
    else          return 1.*(x-xc_1)*(x-xc_1)/r/r/r;
  }
} ls_circle_1_yy;
#endif

#ifdef P4_TO_P8
class LS_REF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double a = min(ls_circle_0(x,y,z), ls_circle_1(x,y,z));
    if (a < 0) a = 0;
    return a;
  }
} ls_ref;
#else
class LS_REF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double a = min(ls_circle_0(x,y), ls_circle_1(x,y));
//    if (a < 0) a = 0;
    return a;
  }
} ls_ref;
#endif

#ifdef P4_TO_P8
class FUNC: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return 1.0;
  }
} func;
#else
class FUNC: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 1.0;
  }
} func;
#endif

#ifdef P4_TO_P8
class FUNC_R2: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return x*x+y*y+z*z;
  }
} func_r2;
#else
class FUNC_R2: public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return x*x+y*y;
//    return fabs(x);
    return sqrt(x*x+y*y);
  }
} func_r2;

class FUNC_X: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return x;
  }
} func_x;
#endif

class Exact {
public:
  double ID;
  double IB;
  double IDr2;
  double IBr2;
  vector<double> ISB, ISBr2;
  vector<double> IXr2;
  vector<int> IXc0, IXc1;
  double alpha;

  double n_subs = 2;
  double n_Xs = 1;

  bool provided = true;

  Exact()
  {
#ifdef P4_TO_P8
    alpha = acos(d/r0);
    /* the whole domain */
    ID = 2.0*4.0/3.0*PI*r0*r0*r0 - 2.0/3.0*PI*pow(r0-d,2.0)*(2.0*r0+d);
    IDr2 = 2.0*4.0/3.0*PI*r0*r0*r0*(1.5*0.4*r0*r0+d*d) -
           2.*(0.4*PI*pow(r0,5.)*(1-d/r0) - 0.1*PI*d*(pow(r0,4.)-pow(d,4.)) +
            1./3.*PI*pow(r0-d,2.)*(2.*r0+d)*(d*d-1.5*d*pow(r0+d,2.)/(2.*r0+d)));
    /* the whole boundary */
    IB = 2.0*(4.0*PI*r0*r0 - 2.*PI*r0*(r0-d));
    IBr2 = 2.0*(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d)-2.*PI*(1-d/r0)*r0*r0*(r0*r0+d*d-r0*d*(1+d/r0)));
    /* sub-boundaries */
    ISB.push_back(4.0*PI*r0*r0 - 2.*PI*r0*(r0-d));
    ISBr2.push_back(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d)-2.*PI*(1-d/r0)*r0*r0*(r0*r0+d*d-r0*d*(1+d/r0)));
    ISB.push_back(4.0*PI*r0*r0 - 2.*PI*r0*(r0-d));
    ISBr2.push_back(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d)-2.*PI*(1-d/r0)*r0*r0*(r0*r0+d*d-r0*d*(1+d/r0)));
    /* intersections */
    IXr2.push_back(2.*PI*pow(r0*r0-d*d,1.5));
    IXc0.push_back(0);
    IXc1.push_back(1);
#else
    // Auxiliary values
    alpha = acos(d/r0);
    double r_bar_A = 2.0*r0*sin(alpha)/alpha/3.0;
    double r_bar_B = r0*sin(alpha)/alpha;

    /* the whole domain */
    ID = 2.0*PI*r0*r0 - 2.0*(alpha*r0*r0-d*sqrt(r0*r0-d*d));
    IDr2 = 2.0*(0.5*PI*r0*r0*r0*r0 + PI*r0*r0*d*d) -
           (alpha*r0*r0*r0*r0 +
            2.0*alpha*r0*r0*d*(d-2.0*r_bar_A) -
            d*r0*r0*sqrt(r0*r0-d*d)/3.0);
    /* the whole boundary */
    IB = 2.0*2.0*(PI-alpha)*r0;
    IBr2 = 2.0*(2.0*PI*r0*(r0*r0+d*d) - 2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    /* sub-boundaries */
    ISB.push_back(2.0*(PI-alpha)*r0);
    ISBr2.push_back(2.0*PI*r0*(r0*r0+d*d) - 2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    ISB.push_back(2.0*(PI-alpha)*r0);
    ISBr2.push_back(2.0*PI*r0*(r0*r0+d*d) - 2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    /* intersections */
    IXr2.push_back(2.0*sqrt(r0*r0-d*d));
//    IXr2.push_back(2.0*sqrt(r0*r0-d*d)*cosT);
    IXc0.push_back(0);
    IXc1.push_back(1);
#endif
  }
} exact;

/* Vectors to store numerical results */
class Result
{
public:
  vector<double> ID, IB, IDr2, IBr2;
  vector< vector<double> > ISB, ISBr2, IXr2;
  Result()
  {
    for (int i = 0; i < exact.n_subs; i++)
    {
      ISB.push_back(vector<double>());
      ISBr2.push_back(vector<double>());
    }
    for (int i = 0; i < exact.n_Xs; i++)
    {
      IXr2.push_back(vector<double>());
    }
  }
};

class Geometry
{
public:
#ifdef P4_TO_P8
  vector<CF_3 *> LSF;
#else
  vector<CF_2 *> LSF, LSFxx, LSFxy, LSFyy, LSFx, LSFy;
#endif
  vector<action_t> action;
  vector<int> color;
  Geometry()
  {
    LSF.push_back(&ls_circle_1); action.push_back(INTERSECTION); color.push_back(1);
    LSFx.push_back(&ls_circle_1_x);
    LSFy.push_back(&ls_circle_1_y);
    LSFxx.push_back(&ls_circle_1_xx);
    LSFxy.push_back(&ls_circle_1_xy);
    LSFyy.push_back(&ls_circle_1_yy);
    LSF.push_back(&ls_circle_0); action.push_back(ADDITION); color.push_back(0);
    LSFx.push_back(&ls_circle_0_x);
    LSFy.push_back(&ls_circle_0_y);
    LSFxx.push_back(&ls_circle_0_xx);
    LSFxy.push_back(&ls_circle_0_xy);
    LSFyy.push_back(&ls_circle_0_yy);
  }
} geometry;


Result res_mlt;

vector<double> level, h;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz,
                                    xmin, xmax, ymin, ymax, zmin, zmax,
                                    &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny,
                                    xmin, xmax, ymin, ymax,
                                    &brick);
#endif

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", 0, lmax+iter); CHKERRXX(ierr);
//    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &ls_ref, 1.2);
    p4est->user_pointer = (void*)(&data);

    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    /* functions to integrate */
    Vec f_r2;
    ierr = VecCreateGhostNodes(p4est, nodes, &f_r2); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func_r2, f_r2);

    my_p4est_level_set_t ls(&ngbd_n);

    /* level-set functions */
    vector<Vec> phi_vec;
    vector<Vec> phix_vec;
    vector<Vec> phiy_vec;
    vector<Vec> phixx_vec;
    vector<Vec> phixy_vec;
    vector<Vec> phiyy_vec;

    for (int i = 0; i < geometry.LSF.size(); i++)
    {
      phi_vec.push_back(Vec());

      ierr = VecCreateGhostNodes(p4est, nodes, &phi_vec[i]); CHKERRXX(ierr);

      sample_cf_on_nodes(p4est, nodes, *geometry.LSF[i], phi_vec[i]);

      phix_vec.push_back(Vec());
      phiy_vec.push_back(Vec());

      ierr = VecCreateGhostNodes(p4est, nodes, &phix_vec.back()); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phiy_vec.back()); CHKERRXX(ierr);

      sample_cf_on_nodes(p4est, nodes, *geometry.LSFx[i], phix_vec.back());
      sample_cf_on_nodes(p4est, nodes, *geometry.LSFy[i], phiy_vec.back());

      phixx_vec.push_back(Vec());
      phixy_vec.push_back(Vec());
      phiyy_vec.push_back(Vec());

      ierr = VecCreateGhostNodes(p4est, nodes, &phixx_vec.back()); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phixy_vec.back()); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &phiyy_vec.back()); CHKERRXX(ierr);

      sample_cf_on_nodes(p4est, nodes, *geometry.LSFxx[i], phixx_vec.back());
      sample_cf_on_nodes(p4est, nodes, *geometry.LSFxy[i], phixy_vec.back());
      sample_cf_on_nodes(p4est, nodes, *geometry.LSFyy[i], phiyy_vec.back());
    }

//    my_p4est_integration_mls_t integration;
//    integration.set_p4est(p4est, nodes);
//    integration.set_phi(phi_vec, geometry.action, geometry.color);
//    integration.initialize();

//#ifdef P4_TO_P8
//    vector<simplex3_mls_t *> simplices;
//    int n_sps = NTETS;
//#else
//    vector<simplex2_mls_t *> simplices;
//    int n_sps = 2;
//#endif

//    for (int k = 0; k < integration.cubes.size(); k++)
//      if (integration.cubes[k].loc == FCE)
//        for (int l = 0; l < n_sps; l++)
//          simplices.push_back(&integration.cubes[k].simplex[l]);

//#ifdef P4_TO_P8
//    simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#else
//    simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#endif


    my_p4est_integration_quad_mls_t integration;
    integration.set_p4est(p4est, nodes);
//    integration.set_phi(phi_vec, phixx_vec, phixy_vec, phiyy_vec, geometry.action, geometry.color);
    integration.set_phi(phi_vec, phixx_vec, phixy_vec, phiyy_vec, phix_vec, phiy_vec, geometry.action, geometry.color);

//    integration.initialize();
//#ifdef P4_TO_P8
//    vector<simplex3_mls_t *> simplices;
//    int n_sps = NTETS;
//#else
//    vector<simplex2_quad_mls_t *> simplices;
//    int n_sps = 2;
//#endif

//    for (int k = 0; k < integration.cubes.size(); k++)
//      if (integration.cubes[k].loc == FCE)
//        for (int l = 0; l < n_sps; l++)
//          simplices.push_back(&integration.cubes[k].simplex[l]);

//#ifdef P4_TO_P8
//    simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#else
//    simplex2_quad_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
//#endif

//    my_p4est_integration_refined_quad_mls_t integration;
//    integration.set_p4est(p4est, nodes);
//    integration.set_refinement(8,8);
////    integration.set_phi(geometry.LSF, geometry.LSFxx, geometry.LSFxy, geometry.LSFyy, geometry.action, geometry.color);
//    integration.set_phi(geometry.LSF, geometry.LSFxx, geometry.LSFxy, geometry.LSFyy, geometry.LSFx, geometry.LSFy, geometry.action, geometry.color);


    /* Calculate and store results */
    if (exact.provided || iter < nb_splits-1)
    {
      level.push_back(lmax+iter);
      h.push_back((xmax-xmin)/pow(2.0,(double)(lmax+iter)));

      res_mlt.ID.push_back(integration.measure_of_domain    ());
//      res_mlt.IB.push_back(integration.measure_of_interface (-1));
      res_mlt.IB.push_back(0);

//      res_mlt.IDr2.push_back(integration.integrate_over_domain    (f_r2));
//      res_mlt.IBr2.push_back(integration.integrate_over_interface (f_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        res_mlt.ISB[i].push_back(integration.measure_of_interface(geometry.color[i]));
        res_mlt.IB.back() += res_mlt.ISB[i].back();
//        res_mlt.ISBr2[i].push_back(integration.integrate_over_interface(f_r2, geometry.color[i]));
      }

//      for (int i = 0; i < exact.n_Xs; i++)
//      {
//        res_mlt.IXr2[i].push_back(integration.integrate_over_intersection(f_r2, exact.IXc0[i], exact.IXc1[i]));
//      }
    }
    else if (iter == nb_splits-1)
    {
      exact.ID    = (integration.measure_of_domain        ());
      exact.IB    = (integration.measure_of_interface     (-1));
//      exact.IDr2  = (integration.integrate_over_domain    (f_r2));
//      exact.IBr2  = (integration.integrate_over_interface (f_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        exact.ISB.push_back(integration.measure_of_interface(geometry.color[i]));
//        exact.ISBr2.push_back(integration.integrate_over_interface(f_r2, geometry.color[i]));
      }

//      for (int i = 0; i < exact.n_Xs; i++)
//      {
//        exact.IXr2.push_back(integration.integrate_over_intersection(f_r2, exact.IXc0[i], exact.IXc1[i]));
//      }
    }

    ierr = VecDestroy(f_r2); CHKERRXX(ierr);

    for (int i = 0; i < phi_vec.size(); i++) {ierr = VecDestroy(phi_vec[i]); CHKERRXX(ierr);}
    for (int i = 0; i < phix_vec.size(); i++) {ierr = VecDestroy(phix_vec[i]); CHKERRXX(ierr);}
    for (int i = 0; i < phiy_vec.size(); i++) {ierr = VecDestroy(phiy_vec[i]); CHKERRXX(ierr);}
    for (int i = 0; i < phixx_vec.size(); i++) {ierr = VecDestroy(phixx_vec[i]); CHKERRXX(ierr);}
    for (int i = 0; i < phixy_vec.size(); i++) {ierr = VecDestroy(phixy_vec[i]); CHKERRXX(ierr);}
    for (int i = 0; i < phiyy_vec.size(); i++) {ierr = VecDestroy(phiyy_vec[i]); CHKERRXX(ierr);}
    phi_vec.clear();

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

//  std::vector<double> dom_3rd, ifc_3rd, level_3rd, h_3rd, IDr2_3rd, IBr2_3rd;

//  for (int i = 1; i < res_mlt.IB.size(); i++)
//  {
//    dom_3rd.push_back((4.0*res_mlt.ID[i]-res_mlt.ID[i-1])/3.0);
//    ifc_3rd.push_back((4.0*res_mlt.IB[i]-res_mlt.IB[i-1])/3.0);
//    IDr2_3rd.push_back((4.0*res_mlt.IDr2[i]-res_mlt.IDr2[i-1])/3.0);
//    IBr2_3rd.push_back((4.0*res_mlt.IBr2[i]-res_mlt.IBr2[i-1])/3.0);
//    h_3rd.push_back(h[i]);
//    level_3rd.push_back(level[i]);
//  }

  Gnuplot plot_ID;
  print_Table("Domain", exact.ID, level, h, "MLT", res_mlt.ID, 1, &plot_ID);
//  print_Table("Domain", exact.ID, level_3rd, h_3rd, "MLT (diff)", dom_3rd, 2, &plot_ID);

  Gnuplot plot_IB;
  print_Table("Interface", exact.IB, level, h, "MLT", res_mlt.IB, 1, &plot_IB);
//  print_Table("Interface", exact.IB, level_3rd, h_3rd, "MLT (diff)", ifc_3rd, 2, &plot_IB);

//  Gnuplot plot_IDr2;
//  print_Table("2nd moment of domain", exact.IDr2, level, h, "MLT", res_mlt.IDr2, 1, &plot_IDr2);
//  print_Table("2nd moment of domain", exact.IDr2, level_3rd, h_3rd, "MLT (diff)", IDr2_3rd, 2, &plot_IDr2);

//  Gnuplot plot_IBr2;
//  print_Table("2nd moment of interface", exact.IBr2, level, h, "MLT", res_mlt.IBr2, 1, &plot_IBr2);
//  print_Table("2nd moment of interface", exact.IBr2, level_3rd, h_3rd, "MLT (diff)", IBr2_3rd, 2, &plot_IBr2);

//  Gnuplot graph;
//  print_Table("3rd", exact.ID, level_3rd, h_3rd, "domain", dom_3rd, 1, &graph);
//  print_Table("3rd", exact.IB, level_3rd, h_3rd, "inface", ifc_3rd, 2, &graph);

  vector<Gnuplot *> plot_ISB;
  vector<Gnuplot *> plot_ISBr2;
  for (int i = 0; i < exact.n_subs; i++)
  {
    plot_ISB.push_back(new Gnuplot());
    print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "MLT", res_mlt.ISB[i], 1, plot_ISB[i]);

//    plot_ISBr2.push_back(new Gnuplot());
//    print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "MLT", res_mlt.ISBr2[i], 2, plot_ISBr2[i]);
  }

  vector<Gnuplot *> plot_IXr2;
//  for (int i = 0; i < exact.n_Xs; i++)
//  {
//    plot_IXr2.push_back(new Gnuplot());
//    print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IXr2[i], level, h, "MLT", res_mlt.IXr2[i], 2, plot_IXr2[i]);
//  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  std::cin.get();

  for (int i = 0; i < exact.n_subs; i++)
  {
    delete plot_ISB[i];
    delete plot_ISBr2[i];
  }

  for (int i = 0; i < exact.n_Xs; i++)
  {
    delete plot_IXr2[i];
  }

  return 0;
}
