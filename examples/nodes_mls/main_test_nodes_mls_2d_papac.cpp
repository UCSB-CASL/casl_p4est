
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
#include <src/simplex2_mls_vtk.h>
#endif

#include <src/point3.h>
#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

double xmin = -1.0;
double xmax =  1.0;
double ymin = -1.0;
double ymax =  1.0;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

using namespace std;

int lmin = 4;
int lmax = 4;
int nb_splits = 7;

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

bool save_vtk = true;


double mu = 1.0;
double add_diagonal = 0;
int n_test = 0;

BoundaryConditionType bc_itype_0 = ROBIN;
BoundaryConditionType bc_itype_1 = ROBIN;
BoundaryConditionType bc_wtype = DIRICHLET;

double scale = 1.0;

double r0 = .5;
double r1 = -0.5145134;
double r2 = -0.5315416;
double r3 = -0.414;
double d = 0.3410;

double b = 3.0;

double theta = 1.7826;
#ifdef P4_TO_P8
double phy = 0.523;
#endif

double cosT = cos(theta);
double sinT = sin(theta);
#ifdef P4_TO_P8
double cosP = cos(phy);
double sinP = sin(phy);
#endif

#ifdef P4_TO_P8
//double xc_0 = -d*sinT*cosP; double yc_0 =  d*cosT*cosP; double zc_0 =  d*sinP;
//double xc_1 =  d*sinT*cosP; double yc_1 = -d*cosT*cosP; double zc_1 = -d*sinP;
//double xc_2 =  2.*d*cosT*cosP; double yc_2 =  2.*d*sinT*cosP; double zc_2 =  2.*d*sinP;
//double xc_3 = -d*cosT*cosP; double yc_3 = -d*sinT*cosP; double zc_3 = -d*sinP;
double xc_0 = 0, yc_0 = 0, zc_0 = 0;
double xc_1 =  d*sinT*cosP; double yc_1 = -d*cosT*cosP; double zc_1 = -d*sinP;
double xc_2 = 0.85, yc_2 = 0, zc_2 = 0;
double xc_3 = -d*cosT*cosP; double yc_3 = -d*sinT*cosP; double zc_3 = -d*sinP;
#else
double xc_0 = -d*sinT; double yc_0 =  d*cosT;
double xc_1 =  d*sinT; double yc_1 = -d*cosT;
double xc_2 =  1.*d*cosT; double yc_2 =  1.*d*sinT;
double xc_3 = -d*cosT; double yc_3 = -d*sinT;
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
    double r = sqrt(SQR(x - 0.) + SQR(y - 0.));
    if(r<1.e-3) r = 1.e-3;
    return (r - r0 - (pow(y,5)+5.0*pow(x,4)*y-10.0*pow(x,2)*pow(y,3))/pow(r,5)/b);
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
class LEVEL_SET_TOT: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
//    return level_set_0(x,y,z);
//    return min(level_set_0(x,y,z),level_set_1(x,y,z));
//    return min(level_set_2(x,y,z),max(level_set_0(x,y,z),level_set_1(x,y,z)));
    return max(level_set_2(x,y,z),min(level_set_0(x,y,z),level_set_1(x,y,z)));
  }
} level_set_tot;
#else
class LEVEL_SET_TOT: public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return level_set_0(x,y);
//    return max(level_set_0(x,y),level_set_1(x,y));
//    return min(level_set_2(x,y),max(level_set_0(x,y),level_set_1(x,y)));
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
//    return level_set_0(x,y,z);
//    return min(level_set_0(x,y,z),level_set_1(x,y,z));
//    return min(level_set_2(x,y,z),max(level_set_0(x,y,z),level_set_1(x,y,z)));
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
//    double a = level_set_0(x,y);
//    double a = max(level_set_0(x,y),level_set_1(x,y));
//    return min(level_set_2(x,y),max(level_set_0(x,y),level_set_1(x,y)));
    double a = max(level_set_2(x,y),min(level_set_0(x,y),level_set_1(x,y)));
    if (a > 0) return a;
    else return 0.0;
  }
} level_set_ref;
#endif



#ifdef P4_TO_P8
class NO_INTERFACE_CF : public CF_3
{
public:
  double operator()(double, double, double) const
  {
    return -1;
  }
} no_interface_cf;
#else
class NO_INTERFACE_CF : public CF_2
{
public:
  double operator()(double, double) const
  {
    return -1;
  }
} no_interface_cf;
#endif

#ifdef P4_TO_P8
class U_EXACT: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return scale*(sin(x)+sin(y)+sin(z));
    case 1: return scale*sin(x)*cos(y)*exp(z);
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
    case 0: return exp(x*y);
    case 1: return scale*(sin(x)*cos(y));
    }
  }
} u_exact;
#endif

#ifdef P4_TO_P8
double ux(double x, double y, double z)
{
  switch (n_test){
  case 0: return scale*cos(x);
  case 1: return scale*cos(x)*cos(y)*exp(z);
  }
}
double uy(double x, double y, double z)
{
  switch (n_test){
  case 0: return scale*cos(y);
  case 1: return -scale*sin(x)*sin(y)*exp(z);
  }
}
double uz(double x, double y, double z)
{
  switch (n_test){
  case 0: return scale*cos(z);
  case 1: return scale*sin(x)*cos(y)*exp(z);
  }
}
#else
double ux(double x, double y)
{

  switch (n_test){
  case 0: return y*exp(x*y);
  case 1: return scale*cos(x)*cos(y);
  }
}
double uy(double x, double y)
{
  switch (n_test){
  case 0: return x*exp(x*y);
  case 1: return -scale*sin(x)*sin(y);
  }
}
#endif


#ifdef P4_TO_P8
double kappa_0(double x, double y, double z)
{
//  return -0.4*sin(x)*cos(y)*cos(z);
  return 1.0;
}
#else
double kappa_0(double x, double y)
{
  return 1.0;
//  return ((x-xc_0)+(y-yc_0))/r0;
}
#endif

#ifdef P4_TO_P8
double kappa_1(double x, double y, double z)
{
//  return 0.7*sin(y)*cos(x)*cos(z);
  return 0.85;
}
#else
double kappa_1(double x, double y)
{
  return 0.85;
//  return ((x-xc_1) + (y-yc_1))/r1;
}
#endif

#ifdef P4_TO_P8
double kappa_2(double x, double y, double z)
{
//  return -1.0*sin(z)*cos(y)*cos(x);
  return 0.9;
}
#else
double kappa_2(double x, double y)
{
  return 0.9;
}
#endif

#ifdef P4_TO_P8
class FORCE: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
    case 0: return mu*u_exact(x,y,z) + add_diagonal*u_exact(x,y,z);
    case 1: return mu*scale*sin(x)*cos(y)*exp(z) + add_diagonal*u_exact(x,y,z);
    }
  }
} force;
#else
class FORCE: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
    case 0: return mu*u_exact(x,y) + add_diagonal*u_exact(x,y);
    case 1: return scale*2.0*mu*sin(x)*cos(y) + add_diagonal*u_exact(x,y);
    }
  }
} force;
#endif

#ifdef P4_TO_P8
class BCINTERFACEVAL_0 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return  ((x-xc_0)*ux(x,y,z)+(y-yc_0)*uy(x,y,z)+(z-zc_0)*uz(x,y,z))/sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)) + kappa_0(x,y,z)*u_exact(x,y,z);
  }
} bc_interface_val_0;
#else
class BCINTERFACEVAL_0 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return  -((x-xc_0)*ux(x,y)+(y-yc_0)*uy(x,y))/sqrt(SQR(x-xc_0) + SQR(y-yc_0)) + kappa_0(x,y)*u_exact(x,y);
  }
} bc_interface_val_0;
#endif

#ifdef P4_TO_P8
class BCINTERFACEVAL_1 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
//    return ((x-xc_0)*ux(x,y,z) + (y-yc_0)*uy(x,y,z) + (z-zc_0)*uz(x,y,z))/sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)) + kappa_1(x,y,z)*u_exact(x,y,z);
    return ((x-xc_1)*ux(x,y,z) + (y-yc_1)*uy(x,y,z) + (z-zc_1)*uz(x,y,z))/sqrt(SQR(x-xc_1) + SQR(y-yc_1) + SQR(z-zc_1)) + kappa_1(x,y,z)*u_exact(x,y,z);
  }
} bc_interface_val_1;
#else
class BCINTERFACEVAL_1 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return ((x-xc_0)*ux(x,y) + (y-yc_0)*uy(x,y))/sqrt(SQR(x-xc_0) + SQR(y-yc_0)) + kappa_1(x,y)*u_exact(x,y);
    return ((x-xc_1)*ux(x,y) + (y-yc_1)*uy(x,y))/sqrt(SQR(x-xc_1) + SQR(y-yc_1)) + kappa_1(x,y)*u_exact(x,y);
  }
} bc_interface_val_1;
#endif

#ifdef P4_TO_P8
class BCINTERFACEVAL_2 : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -((x-xc_2)*ux(x,y,z) + (y-yc_2)*uy(x,y,z) + (z-zc_2)*uz(x,y,z))/sqrt(SQR(x-xc_2) + SQR(y-yc_2) + SQR(z-zc_2)) + kappa_2(x,y,z)*u_exact(x,y,z);
  }
} bc_interface_val_2;
#else
class BCINTERFACEVAL_2 : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -((x-xc_2)*ux(x,y)+(y-yc_2)*uy(x,y))/sqrt(SQR(x-xc_2) + SQR(y-yc_2)) + kappa_2(x,y)*u_exact(x,y);
  }
} bc_interface_val_2;
#endif

#ifdef P4_TO_P8
class BCWALLTYPE : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;
#else
class BCWALLTYPE : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;
#endif

#ifdef P4_TO_P8
class BCWALLVAL : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return u_exact(x,y,z);
  }
} bc_wall_val;
#else
class BCWALLVAL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return u_exact(x,y);
  }
} bc_wall_val;
#endif


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err_nodes, Vec err_ex,
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

  double *err_ex_p;
  ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

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
                         4, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", err_p,
                         VTK_POINT_DATA, "err_ex", err_ex_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_nodes, &err_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



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

  double err_n;
  double err_nm1;

  double err_ex_n;
  double err_ex_nm1;

  vector<double> level, h, error, error_in, error_all, error_tr;

  for(int iter=0; iter<nb_splits; ++iter)
  {
//    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", 0, lmax+iter); CHKERRXX(ierr);
//    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin, lmax+iter);
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

//    splitting_criteria_cf_t data(0, lmax+iter, &level_set_ref, 1.2);
//    splitting_criteria_cf_t data(lmin, lmax+iter, &level_set_tot, 1.4);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set_tot, 1.9);
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

    std::vector<Vec>        phi;
    std::vector<mls_opn_t>   action;
    std::vector<int>        color;

    phi.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set_0, phi.back());
    action.push_back(MLS_INTERSECTION);
    color.push_back(0);

    phi.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set_1, phi.back());
    action.push_back(MLS_ADDITION);
    color.push_back(1);

    phi.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set_2, phi.back());
    action.push_back(MLS_INTERSECTION);
    color.push_back(2);

    Vec phi_tot;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_tot); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set_tot, phi_tot);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.perturb_level_set_function(phi_tot, EPS);
//    ls.perturb_level_set_function(phi[1], EPS);

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
#ifdef P4_TO_P8
    std::vector<BoundaryConditions3D> bc;
#else
    std::vector<BoundaryConditions2D> bc;
#endif

#ifdef P4_TO_P8
    bc.push_back(BoundaryConditions3D());
#else
    bc.push_back(BoundaryConditions2D());
#endif
    bc.back().setWallTypes(bc_wall_type);
    bc.back().setWallValues(bc_wall_val);
    bc.back().setInterfaceType(bc_itype_0);
    bc.back().setInterfaceValue(bc_interface_val_0);

#ifdef P4_TO_P8
    bc.push_back(BoundaryConditions3D());
#else
    bc.push_back(BoundaryConditions2D());
#endif
    bc.back().setWallTypes(bc_wall_type);
    bc.back().setWallValues(bc_wall_val);
    bc.back().setInterfaceType(bc_itype_0);
    bc.back().setInterfaceValue(bc_interface_val_1);

#ifdef P4_TO_P8
    bc.push_back(BoundaryConditions3D());
#else
    bc.push_back(BoundaryConditions2D());
#endif
    bc.back().setWallTypes(bc_wall_type);
    bc.back().setWallValues(bc_wall_val);
    bc.back().setInterfaceType(bc_itype_0);
    bc.back().setInterfaceValue(bc_interface_val_2);

    Vec rhs;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
    double *rhs_p;
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      rhs_p[n] = force(x,y,z);
#else
      rhs_p[n] = force(x,y);
#endif
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

    std::vector<Vec> kappa;
    double *kappa_p;

    kappa.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa.back()); CHKERRXX(ierr);
    ierr = VecGetArray(kappa.back(), &kappa_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      kappa_p[n] = kappa_0(x,y,z);
#else
      kappa_p[n] = kappa_0(x,y);
#endif
    }

    ierr = VecRestoreArray(kappa.back(), &kappa_p); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(kappa.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (kappa.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    kappa.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa.back()); CHKERRXX(ierr);
    ierr = VecGetArray(kappa.back(), &kappa_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      kappa_p[n] = kappa_1(x,y,z);
#else
      kappa_p[n] = kappa_1(x,y);
#endif
    }

    ierr = VecRestoreArray(kappa.back(), &kappa_p); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(kappa.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (kappa.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    kappa.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa.back()); CHKERRXX(ierr);
    ierr = VecGetArray(kappa.back(), &kappa_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      kappa_p[n] = kappa_2(x,y,z);
#else
      kappa_p[n] = kappa_2(x,y);
#endif
    }

    ierr = VecRestoreArray(kappa.back(), &kappa_p); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(kappa.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (kappa.back(), INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

#ifdef P4_TO_P8
    vector<CF_3*> phi_cf;
#else
    vector<CF_2*> phi_cf;
#endif
    phi_cf.push_back(&level_set_0);
    phi_cf.push_back(&level_set_1);
    phi_cf.push_back(&level_set_2);

    my_p4est_poisson_nodes_mls_t solver(&ngbd_n);
    solver.set_action(action);
    solver.set_color(color);
    solver.set_phi(&phi);
    solver.set_diagonal(add_diagonal);
    solver.set_mu(mu);
    solver.set_bc(bc);
    solver.set_robin_coef(kappa);
    solver.set_rhs(rhs);
    solver.set_force(force);
    solver.set_phi_cf(phi_cf);

    Vec sol;
    ierr = VecDuplicate(rhs, &sol); CHKERRXX(ierr);

//    solver.construct_domain();

//#ifdef P4_TO_P8
//    vector<simplex3_mls_t *> simplices;
//    int n_sps = NTETS;
//#else
//    vector<simplex2_mls_t *> simplices;
//    int n_sps = 2;
//#endif

//    for (int k = 0; k < solver.cubes.size(); k++)
//      if (solver.cubes[k].loc == FCE)
//        for (int l = 0; l < n_sps; l++)
//          simplices.push_back(&solver.cubes[k].simplex[l]);

//#ifdef P4_TO_P8
//    simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR)+"/visual/", to_string(iter));
//#else
//    simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR)+"/visual/", to_string(iter));
//#endif

    solver.solve(sol);

//    /* if all NEUMANN boundary conditions, shift solution */
//    if(solver.get_matrix_has_nullspace())
//    {
//      double avg_sol = integrate_over_interface(p4est, nodes, phi, sol)/area_in_negative_domain(p4est, nodes, phi);

//      double *sol_p;
//      ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

//      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
//        sol_p[n] = sol_p[n] - avg_sol + avg_exa;

//      ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
//    }

    /* check the error */
    err_nm1 = err_n;
    err_n = 0;

    const double *phi_p;
    ierr = VecGetArrayRead(phi_tot, &phi_p); CHKERRXX(ierr);

    const double *sol_p;
    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);

    Vec err_nodes;
    ierr = VecDuplicate(sol, &err_nodes); CHKERRXX(ierr);
    double *err_p;
    ierr = VecGetArray(err_nodes, &err_p); CHKERRXX(ierr);

//    ls.reinitialize_1st_order_time_2nd_order_space(phi_tot);
//    ls.extend_Over_Interface_TVD(phi_tot, sol, 20);
    double err_in_n = 0;
    double err_all_n = 0;

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
//      if (phi_p[n]<0.0*diag)
      if (solver.is_calc(n))
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
        bool in, alt;
        solver.find_centroid(in, alt, x, y, n);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        err_p[n] = ABS(sol_p[n] - u_exact(x,y,z));
#else
        err_p[n] = ABS(sol_p[n] - u_exact(x,y));
#endif
        err_all_n = MAX(err_all_n, err_p[n]);
        if (phi_p[n] < 0.0) {err_n = MAX(err_n, err_p[n]);}
        if (solver.is_inside(n)) {err_in_n = MAX(err_in_n, err_p[n]);}
//        err_p[n] = u_exact(x,y);
      }
      else
        err_p[n] = 0;
    }
    ierr = VecRestoreArrayRead(phi_tot, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_nodes, &err_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_all_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_in_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Error on nodes : %g, order = %g\n", err_n, log(err_nm1/err_n)/log(2)); CHKERRXX(ierr);

    level.push_back(lmin+iter);
    h.push_back(dx);
    error.push_back(err_n);
    error_in.push_back(err_in_n);
    error_all.push_back(err_all_n);


    /* extrapolate the solution and check accuracy */
    double band = 4;
//    const double *phi_p;

//    ls.reinitialize_1st_order_time_2nd_order_space(phi_tot);

//    ls.extend_Over_Interface_TVD(phi_tot, sol, 100);

    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi_tot, &phi_p); CHKERRXX(ierr);

    Vec err_ex;
    ierr = VecDuplicate(sol, &err_ex); CHKERRXX(ierr);
    double *err_ex_p;
    ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    err_ex_nm1 = err_ex_n;
    err_ex_n = 0;

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(phi_p[n]>0)
//      if(!solver.is_inside(n))
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        if(phi_p[n]<band*diag)
          err_ex_p[n] = fabs(sol_p[n] - u_exact(x,y,z));
#else
        if(phi_p[n]<band*diag)
          err_ex_p[n] = fabs(sol_p[n] - u_exact(x,y));
#endif
        else
          err_ex_p[n] = 0;

        err_ex_n = MAX(err_ex_n, err_ex_p[n]);
      }
      else
        err_ex_p[n] = 0;

//      double x = node_x_fr_n(n, p4est, nodes);
//      double y = node_y_fr_n(n, p4est, nodes);
//#ifdef P4_TO_P8
//      double z = node_z_fr_n(n, p4est, nodes);
//      err_ex_p[n] = bc_interface_val_0(x,y,z);
//#endif
//      err_ex_p[n] = solver.node_volume(n);
    }

    ierr = VecRestoreArrayRead(phi_tot, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    err_ex_n = solver.calculate_trunc_error(u_exact);
    error_tr.push_back(err_ex_n);
    ierr = PetscPrintf(p4est->mpicomm, "Error extrapolation : %g, order = %g\n", err_ex_n, log(err_ex_nm1/err_ex_n)/log(2)); CHKERRXX(ierr);


    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &brick, solver.phi_eff_, sol, err_nodes, solver.trunc_error, iter);
    }

    for (int i = 0; i < phi.size(); i++)    {ierr = VecDestroy(phi[i]); CHKERRXX(ierr);}
    for (int i = 0; i < kappa.size(); i++)  {ierr = VecDestroy(kappa[i]); CHKERRXX(ierr);}

    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);

    ierr = VecDestroy(err_nodes); CHKERRXX(ierr);
//    ierr = VecDestroy(err_ex); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  if (mpi->mpirank == 0)
  {
  Gnuplot graph;
  print_Table("error", 0.0, level, h, "error (all)", error_all, 1, &graph);
  print_Table("error", 0.0, level, h, "error (inside)", error, 2, &graph);
  print_Table("error", 0.0, level, h, "error (interior)", error_in, 3, &graph);

  Gnuplot graph_tr;
  print_Table("error", 0.0, level, h, "error (truncation)", error_tr, 1, &graph_tr);
  cin.get();
  }

  return 0;
}
