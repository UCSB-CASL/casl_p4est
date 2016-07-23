
/*
 * Test the cell based multi level-set p4est.
 * Intersection of two circles
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
#include <src/my_p8est_poisson_nodes_mls.h>
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
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/simplex2_mls_vtk.h>
#endif

#include <src/point3.h>
#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

bool save_vtk = true;

double xmin = -1.0;
double xmax =  1.0;
double ymin = -1.0;
double ymax =  1.0;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

int lmin = 4;
int lmax = 4;
int nb_splits = 6;

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

double mu = 1.;
double diag_add = 0.;
int n_test = 1;

// GEOMETRY
#ifdef P4_TO_P8
double r0 = 0.957;
double r1 = 0.854;
double r2 = 0.631;
double r3 = -0.333;
double d = 0.234;
#else
double r0 = 0.587;
double r1 = 0.5945134;
//double r2 = -0.4315416;
double r2 = 0.397;
double r3 = -0.333;
//double d = 0.;
double d = 0.23410;
#endif

double theta = 0.6826;
#ifdef P4_TO_P8
double phy = 0.323;
#endif

double cosT = cos(theta);
double sinT = sin(theta);
#ifdef P4_TO_P8
double cosP = cos(phy);
double sinP = sin(phy);
#endif

#ifdef P4_TO_P8
double xc_0 = -1.*d*sinT*cosP; double yc_0 =  1.*d*cosT*cosP; double zc_0 =  1.*d*sinP;
double xc_1 =  1.*d*sinT*cosP; double yc_1 = -1.*d*cosT*cosP; double zc_1 = -1.*d*sinP;
double xc_2 =  2.*d*cosT*cosP; double yc_2 =  2.*d*sinT*cosP; double zc_2 =  2.*d*sinP;
//double xc_2 =  5.*d*sinT*cosP; double yc_2 = -2.*d*cosT*cosP; double zc_2 = -1.*d*sinP;
double xc_3 = -4.*d*sinT*cosP; double yc_3 =  2.*d*cosT*cosP; double zc_3 =  1.*d*sinP;
#else
double xc_0 = -1.*d*sinT; double yc_0 =  1.*d*cosT;
//double xc_0 = 0.13; double yc_0 =  0.11;
double xc_1 =  1.*d*sinT; double yc_1 = -1.*d*cosT;
double xc_2 =  2.*d*cosT; double yc_2 =  2.*d*sinT;
//double xc_2 =  0.14; double yc_2 =  0.11;
double xc_3 = -4.*d*sinT; double yc_3 =  2.*d*cosT;
#endif

#ifdef P4_TO_P8
class LEVEL_SET_0: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)));
  }
} level_set_0;
#else
class LEVEL_SET_0: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0)));
  }
} level_set_0;
#endif

#ifdef P4_TO_P8
class LEVEL_SET_1: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r1 - sqrt(SQR(x-xc_1) + SQR(y-yc_1)+SQR(z-zc_1)));
  }
} level_set_1;
#else
class LEVEL_SET_1: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r1 - sqrt(SQR(x-xc_1) + SQR(y-yc_1)));
  }
} level_set_1;
#endif

#ifdef P4_TO_P8
class LEVEL_SET_2: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return (r2 - sqrt(SQR(x-xc_2) + SQR(y-yc_2) + SQR(z-zc_2)));
  }
} level_set_2;
#else
class LEVEL_SET_2: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return (r2 - sqrt(SQR(x-xc_2) + SQR(y-yc_2)));
  }
} level_set_2;
#endif

#ifdef P4_TO_P8
class LEVEL_SET_3: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r3 - sqrt(SQR(x-xc_3) + SQR(y-yc_3) + SQR(z-zc_3)));
  }
} level_set_3;
#else
class LEVEL_SET_3: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r3 - sqrt(SQR(x-xc_3) + SQR(y-yc_3)));
  }
} level_set_3;
#endif

#ifdef P4_TO_P8
class LEVEL_SET_TOT: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return max(level_set_2(x,y,z),min(level_set_0(x,y,z),level_set_1(x,y,z)));
  }
} level_set_tot;
#else
class LEVEL_SET_TOT: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return max(level_set_2(x,y),min(level_set_0(x,y),level_set_1(x,y)));
  }
} level_set_tot;
#endif

#ifdef P4_TO_P8
class LEVEL_SET_REF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double a = max(level_set_2(x,y,z),min(level_set_0(x,y,z),level_set_1(x,y,z)));
    if (a > 0) return a;
    else return 0.0;
  }
} level_set_ref;
#else
class LEVEL_SET_REF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double a = max(level_set_2(x,y),min(level_set_0(x,y),level_set_1(x,y)));
    if (a > 0) return a;
    else return 0.0;
  }
} level_set_ref;
#endif


class Geometry {
public:
#ifdef P4_TO_P8
  std::vector<CF_3 *>   phi_cf;
#else
  std::vector<CF_2 *>   phi_cf;
#endif
  std::vector<action_t> action;
  std::vector<int>      color;

  Geometry()
  {
    phi_cf.push_back(&level_set_0); action.push_back(INTERSECTION); color.push_back(color.size());
    phi_cf.push_back(&level_set_1); action.push_back(ADDITION);     color.push_back(color.size());
    phi_cf.push_back(&level_set_2); action.push_back(INTERSECTION); color.push_back(color.size());
    phi_cf.push_back(&level_set_3); action.push_back(COLORATION);   color.push_back(color.size());
  }
} geometry;

// EXACT SOLUTION
#ifdef P4_TO_P8
class U_EXACT: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return (sin(x)+sin(y)+sin(z));
    case 1: return sin(x)*cos(y)*exp(z);
    }
  }
} u_exact;
#else
class U_EXACT: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return (sin(x)+sin(y));
    case 1: return (sin(x)*cos(y));
    }
  }
} u_exact;
#endif

// EXACT DERIVATIVES
#ifdef P4_TO_P8
class UX: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return cos(x);
    case 1: return cos(x)*cos(y)*exp(z);
    }
  }
} ux;
class UY: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return cos(y);
    case 1: return -sin(x)*sin(y)*exp(z);
    }
  }
} uy;
class UZ: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return cos(z);
    case 1: return sin(x)*cos(y)*exp(z);
    }
  }
} uz;
#else
class UX: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return cos(x);
    case 1: return cos(x)*cos(y);
    }
  }
} ux;
class UY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return cos(y);
    case 1: return -sin(x)*sin(y);
    }
  }
} uy;
class UXY: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return 0;
    case 1: return -cos(x)*sin(y);
    }
  }
} uxy;
#endif

// RHS
#ifdef P4_TO_P8
class RHS: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return mu*u_exact(x,y,z) + diag_add*u_exact(x,y,z);
    case 1: return mu*sin(x)*cos(y)*exp(z) + diag_add*u_exact(x,y,z);
    }
  }
} rhs_cf;
#else
class RHS: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return mu*u_exact(x,y) + diag_add*u_exact(x,y);
    case 1: return 2.0*mu*sin(x)*cos(y) + diag_add*u_exact(x,y);
    }
  }
} rhs_cf;
#endif

// BC COEFFICIENTS
#ifdef P4_TO_P8
class KAPPA_0 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return 1.0 + .0*sin(x)*cos(y)*cos(z);
  }
} kappa_0;

class KAPPA_1 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return .0 + 1.0*sin(y)*cos(x)*cos(z);
  }
} kappa_1;

class KAPPA_2 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return .0 + 1.0*sin(z)*cos(y)*cos(x);
  }
} kappa_2;

class KAPPA_3 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return .0 + 1.0*sin(z)*cos(y)*sin(x);
  }
} kappa_3;
#else
class KAPPA_0 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return .0 + 1.0*sin(x)*cos(y);
  }
} kappa_0;

class KAPPA_1 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return .0 + 1.0*cos(x)*sin(y);
  }
} kappa_1;

class KAPPA_2 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return .0 + 1.0*(sin(x)*sin(y));
  }
} kappa_2;

class KAPPA_3 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return .0 + 1.0*(cos(x)*cos(y));
  }
} kappa_3;
#endif

// BC VALUES
#ifdef P4_TO_P8
class BC_VALUE_0 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return  ((x-xc_0)*ux(x,y,z)+(y-yc_0)*uy(x,y,z)+(z-zc_0)*uz(x,y,z))/sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)) + kappa_0(x,y,z)*u_exact(x,y,z);
  }
} bc_value_0;

class BC_VALUE_1 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return ((x-xc_1)*ux(x,y,z) + (y-yc_1)*uy(x,y,z) + (z-zc_1)*uz(x,y,z))/sqrt(SQR(x-xc_1) + SQR(y-yc_1) + SQR(z-zc_1)) + kappa_1(x,y,z)*u_exact(x,y,z);
  }
} bc_value_1;

class BC_VALUE_2 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -((x-xc_2)*ux(x,y,z) + (y-yc_2)*uy(x,y,z) + (z-zc_2)*uz(x,y,z))/sqrt(SQR(x-xc_2) + SQR(y-yc_2) + SQR(z-zc_2)) + kappa_2(x,y,z)*u_exact(x,y,z);
  }
} bc_value_2;

class BC_VALUE_3 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return ((x-xc_0)*ux(x,y,z)+(y-yc_0)*uy(x,y,z)+(z-zc_0)*uz(x,y,z))/sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)) + kappa_3(x,y,z)*u_exact(x,y,z);
  }
} bc_value_3;
#else
class BC_VALUE_0 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return  ((x-xc_0)*ux(x,y)+(y-yc_0)*uy(x,y))/sqrt(SQR(x-xc_0) + SQR(y-yc_0)) + kappa_0(x,y)*u_exact(x,y);
  }
} bc_value_0;

class BC_VALUE_1 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return ((x-xc_1)*ux(x,y) + (y-yc_1)*uy(x,y))/sqrt(SQR(x-xc_1) + SQR(y-yc_1)) + kappa_1(x,y)*u_exact(x,y);
  }
} bc_value_1;

class BC_VALUE_2 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -((x-xc_2)*ux(x,y)+(y-yc_2)*uy(x,y))/sqrt(SQR(x-xc_2) + SQR(y-yc_2)) + kappa_2(x,y)*u_exact(x,y);
  }
} bc_value_2;

class BC_VALUE_3 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return ((x-xc_0)*ux(x,y)+(y-yc_0)*uy(x,y))/sqrt(SQR(x-xc_0) + SQR(y-yc_0)) + kappa_3(x,y)*u_exact(x,y);
  }
} bc_value_3;
#endif

class Problem {
public:

#ifdef P4_TO_P8
  std::vector<CF_3 *> bc_values;
  std::vector<CF_3 *> bc_coeffs;
#else
  std::vector<CF_2 *> bc_values;
  std::vector<CF_2 *> bc_coeffs;
#endif
  std::vector<BoundaryConditionType> bc_type;

  Problem()
  {
    bc_type.push_back(ROBIN); bc_values.push_back(&bc_value_0); bc_coeffs.push_back(&kappa_0);
    bc_type.push_back(ROBIN); bc_values.push_back(&bc_value_1); bc_coeffs.push_back(&kappa_1);
    bc_type.push_back(ROBIN); bc_values.push_back(&bc_value_2); bc_coeffs.push_back(&kappa_2);
    bc_type.push_back(ROBIN); bc_values.push_back(&bc_value_3); bc_coeffs.push_back(&kappa_3);
  }
} problem;


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err_nodes, Vec err_trunc, Vec err_grad,
              int compt)
{
  PetscErrorCode ierr;
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, OUTPUT_DIR);
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/nodes_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p, *sol_p, *err_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(err_nodes, &err_p); CHKERRXX(ierr);

  double *err_trunc_p, *err_grad_p;
  ierr = VecGetArray(err_trunc, &err_trunc_p); CHKERRXX(ierr);
  ierr = VecGetArray(err_grad,  &err_grad_p); CHKERRXX(ierr);

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
                         5, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", err_p,
                         VTK_POINT_DATA, "err_trunc", err_trunc_p,
                         VTK_POINT_DATA, "err_grad", err_grad_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_nodes, &err_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(err_trunc, &err_trunc_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_grad,  &err_grad_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}

//struct poisson_error_t
//{
//  Vec vec;
//  std::vector<double> max, L1;
//  double err, err_m1;

//  poisson_error_t () : vec(NULL), err(0.), err_m1(0.) {}
//};

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

//  cmdParser cmd;
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
//                 2 - sin(x)*cos(y)");
//#endif
//  cmd.parse(argc, argv);

//  cmd.print();

//  lmin = cmd.get("lmin", lmin);
//  lmax = cmd.get("lmax", lmax);
//  nb_splits = cmd.get("nb_splits", nb_splits);
//  test_number = cmd.get("test", test_number);

//  bc_wtype = cmd.get("bc_wtype", bc_wtype);
//  bc_itype = cmd.get("bc_itype", bc_itype);

//  save_vtk = cmd.get("save_vtk", save_vtk);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

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

  vector<double> level, h;

  vector<double> error_sl_arr, error_sl_l1_arr; double err_sl, err_sl_m1;
  vector<double> error_tr_arr, error_tr_l1_arr; double err_tr, err_tr_m1;
  vector<double> error_ux_arr, error_ux_l1_arr; double err_ux, err_ux_m1;
  vector<double> error_uy_arr, error_uy_l1_arr; double err_uy, err_uy_m1;
#ifdef P4_TO_P8
  vector<double> error_uz_arr, error_uz_l1_arr; double err_uz, err_uz_m1;
#endif
//  vector<double> error_uxy_arr, error_uxy_l1_arr; double err_uxy, err_uxy_m1;

  for(int iter=0; iter<nb_splits; ++iter)
  {
//    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", 0, lmax+iter); CHKERRXX(ierr);
//    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin, lmax+iter);
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

//    splitting_criteria_cf_t data(0, lmax+iter, &level_set_ref, 1.4);
//    splitting_criteria_cf_t data(lmin, lmax+iter, &level_set_tot, 1.4);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set_tot, 1.0);
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

    /* TEST THE NODES FUNCTIONS */


    // sample level-set functions
    std::vector<Vec> phi;
    for (int i = 0; i < geometry.phi_cf.size(); i++)
    {
      phi.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *geometry.phi_cf[i], phi.back());
    }

    std::vector<Vec> bc_values;
    for (int i = 0; i < problem.bc_values.size(); i++)
    {
      bc_values.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &bc_values.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *problem.bc_values[i], bc_values.back());
    }

    std::vector<Vec> bc_coeffs;
    for (int i = 0; i < problem.bc_coeffs.size(); i++)
    {
      bc_coeffs.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &bc_coeffs.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *problem.bc_coeffs[i], bc_coeffs.back());
    }

    Vec rhs;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);

    Vec u_exact_vec;
    ierr = VecCreateGhostNodes(p4est, nodes, &u_exact_vec); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, u_exact, u_exact_vec);

//    ierr = VecDestroy(rhs); CHKERRXX(ierr);
//    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);

    my_p4est_poisson_nodes_mls_t solver(&ngbd_n);
    solver.set_geometry(phi, geometry.action, geometry.color);
    solver.set_mu(mu);
    solver.set_rhs(rhs);
    solver.wall_value.set(u_exact_vec);
    solver.set_bc_type(problem.bc_type);
    solver.set_diag_add(diag_add);
    solver.set_bc_coeffs(bc_coeffs);
    solver.set_bc_values(bc_values);
    solver.set_use_taylor_correction(true);
    solver.compute_volumes();
    solver.set_keep_scalling(true);
    solver.set_kinks_treatment(true);
//    solver.set_cube_refinement(0);

    Vec sol; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);

    solver.solve(sol);

//    // extrapolation
//    Vec phi_tot;
//    ierr = VecCreateGhostNodes(p4est, nodes, &phi_tot); CHKERRXX(ierr);
//    sample_cf_on_nodes(p4est, nodes, level_set_tot, phi_tot);
//    ls.reinitialize_1st_order_time_2nd_order_space(phi_tot);
//    ls.extend_Over_Interface_TVD(phi_tot, sol, 20);
//    ierr = VecDestroy(phi_tot); CHKERRXX(ierr);

    // create Vec's for errors
    Vec error_sl; ierr = VecCreateGhostNodes(p4est, nodes, &error_sl); CHKERRXX(ierr);
    Vec error_tr; ierr = VecCreateGhostNodes(p4est, nodes, &error_tr); CHKERRXX(ierr);
    Vec error_ux; ierr = VecCreateGhostNodes(p4est, nodes, &error_ux); CHKERRXX(ierr);
    Vec error_uy; ierr = VecCreateGhostNodes(p4est, nodes, &error_uy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec error_uz; ierr = VecCreateGhostNodes(p4est, nodes, &error_uz); CHKERRXX(ierr);
#endif

//    Vec error_uxy; ierr = VecCreateGhostNodes(p4est, nodes, &error_uxy); CHKERRXX(ierr);

    // compute errors
    solver.compute_error_sl(u_exact, sol, error_sl);
    solver.compute_error_tr(u_exact, error_tr);
#ifdef P4_TO_P8
    solver.compute_error_gr(ux, uy, uz, sol, error_ux, error_uy, error_uz);
#else
    solver.compute_error_gr(ux, uy, sol, error_ux, error_uy);
#endif

//    solver.compute_error_xy(uxy, sol, error_uxy);

    // compute L-inf norm of errors
    err_sl_m1 = err_sl; err_sl = 0.;
    err_tr_m1 = err_tr; err_tr = 0.;
    err_ux_m1 = err_ux; err_ux = 0.;
    err_uy_m1 = err_uy; err_uy = 0.;
#ifdef P4_TO_P8
    err_uz_m1 = err_uz; err_uz = 0.;
#endif

//    err_uxy_m1 = err_uxy; err_uxy = 0.;

    VecMax(error_sl, NULL, &err_sl); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_sl, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    VecMax(error_tr, NULL, &err_tr); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_tr, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    VecMax(error_ux, NULL, &err_ux); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ux, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    VecMax(error_uy, NULL, &err_uy); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_uy, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
#ifdef P4_TO_P8
    VecMax(error_uz, NULL, &err_uz); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_uz, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
#endif

//    VecMax(error_uxy, NULL, &err_uxy); mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_uxy, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    // compute L1 errors
    my_p4est_integration_mls_t integrator;
    integrator.set_p4est(p4est, nodes);
#ifdef P4_TO_P8
    integrator.set_phi(phi, *solver.phi_xx, *solver.phi_yy, *solver.phi_zz, geometry.action, geometry.color);
#else
    integrator.set_phi(phi, *solver.phi_xx, *solver.phi_yy, geometry.action, geometry.color);
#endif
//    integrator.initialize();
    error_sl_l1_arr.push_back(integrator.integrate_everywhere(error_sl)/integrator.measure_of_domain());
    error_tr_l1_arr.push_back(integrator.integrate_everywhere(error_tr)/integrator.measure_of_domain());
    error_ux_l1_arr.push_back(integrator.integrate_everywhere(error_ux)/integrator.measure_of_domain());
    error_uy_l1_arr.push_back(integrator.integrate_everywhere(error_uy)/integrator.measure_of_domain());
#ifdef P4_TO_P8
    error_uz_l1_arr.push_back(integrator.integrate_everywhere(error_uz)/integrator.measure_of_domain());
#endif

    // Store error values
    level.push_back(lmin+iter);
    h.push_back(dx*pow(2.,(double) data.max_lvl - data.min_lvl));
    error_sl_arr.push_back(err_sl);
    error_tr_arr.push_back(err_tr);
    error_ux_arr.push_back(err_ux);
    error_uy_arr.push_back(err_uy);
#ifdef P4_TO_P8
    error_uz_arr.push_back(err_uz);
#endif

//    error_uxy_arr.push_back(err_uxy);

    // Print current errors
    ierr = PetscPrintf(p4est->mpicomm, "Error (sl): %g, order = %g\n", err_sl, log(err_sl_m1/err_sl)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error (tr): %g, order = %g\n", err_tr, log(err_tr_m1/err_tr)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error (ux): %g, order = %g\n", err_ux, log(err_ux_m1/err_ux)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Error (uy): %g, order = %g\n", err_uy, log(err_uy_m1/err_uy)/log(2)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = PetscPrintf(p4est->mpicomm, "Error (uz): %g, order = %g\n", err_uz, log(err_uz_m1/err_uz)/log(2)); CHKERRXX(ierr);
#endif

//    ierr = PetscPrintf(p4est->mpicomm, "Error (uxy): %g, order = %g\n", err_ux, log(err_uxy_m1/err_uxy)/log(2)); CHKERRXX(ierr);

    // if all NEUMANN boundary conditions, shift solution
//    if(solver.get_matrix_has_nullspace())
//    {
//      double avg_sol = integrate_over_interface(p4est, nodes, phi, sol)/area_in_negative_domain(p4est, nodes, phi);

//      double *sol_p;
//      ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

//      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//        sol_p[n] = sol_p[n] - avg_sol + avg_exa;

//      ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
//    }

    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &brick, solver.phi_eff, sol, error_sl, error_tr, error_ux, iter);
    }

    if (save_vtk)
    {
      integrator.initialize();
#ifdef P4_TO_P8
      vector<simplex3_mls_t *> simplices;
      int n_sps = NTETS;
#else
      vector<simplex2_mls_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < integrator.cubes.size(); k++)
        if (integrator.cubes[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&integrator.cubes[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#else
      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#endif
    }

    for (int i = 0; i < phi.size(); i++)
    {
      ierr = VecDestroy(phi[i]); CHKERRXX(ierr);
      ierr = VecDestroy(bc_values[i]); CHKERRXX(ierr);
      ierr = VecDestroy(bc_coeffs[i]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(sol); CHKERRXX(ierr);

    // destroy Vec's with errors
    ierr = VecDestroy(error_sl); CHKERRXX(ierr);
    ierr = VecDestroy(error_tr); CHKERRXX(ierr);
    ierr = VecDestroy(error_ux); CHKERRXX(ierr);
    ierr = VecDestroy(error_uy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(error_uz); CHKERRXX(ierr);
#endif

//    ierr = VecDestroy(error_uxy); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  if (mpi->mpirank == 0)
  {
    Gnuplot graph;

    print_Table("Error", 0.0, level, h, "err sl (max)", error_sl_arr,     1, &graph);
    print_Table("Error", 0.0, level, h, "err sl (L1)",  error_sl_l1_arr,  2, &graph);

    print_Table("Solution error", 0.0, level, h, "err tr (max)", error_tr_arr,     3, &graph);
//    print_Table("Error", 0.0, level, h, "err tr (L1)",  error_tr_l1_arr,  4, &graph);

    Gnuplot graph_grad;

    print_Table("Error", 0.0, level, h, "err ux (max)", error_ux_arr,     1, &graph_grad);
    print_Table("Error", 0.0, level, h, "err ux (L1)",  error_ux_l1_arr,  2, &graph_grad);

    print_Table("Error", 0.0, level, h, "err uy (max)", error_uy_arr,     3, &graph_grad);
    print_Table("Error in gradients", 0.0, level, h, "err uy (L1)",  error_uy_l1_arr,  4, &graph_grad);
#ifdef P4_TO_P8
    print_Table("Error", 0.0, level, h, "err uz (max)", error_uz_arr,     5, &graph_grad);
    print_Table("Error", 0.0, level, h, "err uz (L1)",  error_uz_l1_arr,  6, &graph_grad);
#endif

//    print_Table("Error", 0.0, level, h, "err uxy (max)", error_uxy_arr,     5, &graph_grad);

//  Gnuplot graph_geom;
//  print_Table("error", 2.*PI*r0, level, h, "ifc", ifc_measure, 1, &graph_geom);

//  Gnuplot graph_grad;
//  print_Table("error", 0.0, level, h, "error (ux)", error_ux, 1, &graph_grad);
//  print_Table("error", 0.0, level, h, "error (uy)", error_uy, 2, &graph_grad);

    cin.get();
  }

  return 0;
}
