// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

double xmin, xmax;
double ymin, ymax;
#ifdef P4_TO_P8
double zmin, zmax;
#endif

using namespace std;

/*
 * 0 - rotation
 * 1 - vortex
 */
int test_number = 0;

double tn;
double dt;

#ifdef P4_TO_P8
#else
struct level_set_t : CF_2
{
  level_set_t() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return sqrt(SQR(x+.75*sin(tn)) + SQR(y-.75*cos(tn))) - .15;
    case 1: return sqrt(SQR(x-.5) + SQR(y-.75)) - .15;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} level_set;

struct u_t : CF_2
{
  double operator()(double x, double y) const {
    switch(test_number)
    {
    case 0: return -y;
    case 1: return -SQR(sin(PI*x))*sin(2*PI*y);
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} u;

struct v_t : CF_2
{
  double operator()(double x, double y) const {
    switch(test_number)
    {
    case 0: return x;
    case 1: return SQR(sin(PI*y))*sin(2*PI*x);
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} v;
#endif

void save_VTK(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_brick_t *brick, Vec phi, int compt)
{
  PetscErrorCode ierr;

  std::ostringstream oss;
  const char* out_dir = getenv("OUT_DIR");
  if (out_dir)
    oss << out_dir << "/vtu";
  else
    oss << "out_dir/vtu";

  std::ostringstream command;
  command << "mkdir -p " << oss.str();
  system(command.str().c_str());

  struct stat st;
  if(stat(oss.str().data(),&st)!=0 || !S_ISDIR(st.st_mode))
  {
    ierr = PetscPrintf(p4est->mpicomm, "Trying to save files in ... %s\n", oss.str().data());
    throw std::invalid_argument("[ERROR]: the directory specified to export vtu images does not exist.");
  }

  oss << "/advection_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(  p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0,
                           oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = PetscPrintf(p4est->mpicomm, "Saved in ... %s\n", oss.str().data()); CHKERRXX(ierr);
}


int main (int argc, char* argv[])
{
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  PetscErrorCode ierr;
  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "number of trees in the x direction");
  cmd.add_option("ny", "number of trees in the y direction");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z direction");
#endif
  cmd.add_option("tf", "final time");
  cmd.add_option("save_vtk", "1 to export vtu images, 0 otherwise");
  cmd.add_option("save_every_n", "export images every n iterations");
  cmd.add_option("test", "the test to run. Available options are\
                 \t 0 - rotation\n\
                 \t 1 - vortex\n");
  cmd.parse(argc, argv);

  int lmin = cmd.get("lmin", 0);
  int lmax = cmd.get("lmax", 6);
  test_number = cmd.get("test", test_number);
  bool save_vtk = cmd.get("save_vtk", 0);
  int save_every_n = cmd.get("save_every_n", 1);

  splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  int nx, ny;
#ifdef P4_TO_P8
  int nz;
#endif
  double tf;

  switch(test_number)
  {
  case 0: nx=1; ny=1; xmin=-1; xmax= 1; ymin=-1; ymax= 1; tf=2*PI; break;
  case 1: nx=2; ny=2; xmin= 0; xmax= 1; ymin= 0; ymax= 1; tf=1; break;
  default: throw std::invalid_argument("[ERROR]: choose a valid test.");
  }

  nx = cmd.get("nx", nx);
  ny = cmd.get("ny", ny);
#ifdef P4_TO_P8
  nz = cmd.get("nz", nz);
#endif
  tf = cmd.get("tf", tf);

  // Create the connectivity object
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny, xmin, xmax, ymin, ymax, &brick);
#endif

  double dxyz_min[P4EST_DIM];
  dxyz_min[0] = (xmax-xmin)/nx/(1<<lmax);
  dxyz_min[1] = (ymax-ymin)/ny/(1<<lmax);
#ifdef P4_TO_P8
  dxyz_min[2] = (zmax-zmin)/nz/(1<<lmax);
#endif

#ifdef P4_TO_P8
  dt = 1*MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  dt = 1*MIN(dxyz_min[0], dxyz_min[1]);
#endif

  p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);
  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  /* Initialize the level-set function */
  tn = 0;
  Vec phi_n;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_n); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, level_set, phi_n);

  /* initialize the velocity field */
  const CF_2 *velo_cf[2] = { &u, &v };
  Vec velo_n[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(phi_n, &velo_n[dir]); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *velo_cf[dir], velo_n[dir]);
  }

  int iter = 0;

  save_VTK(p4est, nodes, &brick, phi_n, iter/save_every_n);
  iter++;

  while(tn+.1*dt<tf)
  {
//    ierr = PetscPrintf(p4est_n->mpicomm, "Iteration #%d, tn=%g\n", iter, tn);

    if(tn+dt>tf)
      dt = tf-tn;

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(velo_n[dir]); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_n, &velo_n[dir]); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *velo_cf[dir], velo_n[dir]);
    }

    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);
    sl.update_p4est(velo_n, dt, phi_n);
    //      sl.update_p4est(velo_cf, dt, phi_n);

    p4est_destroy(p4est); p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
    delete ngbd; ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
    ngbd->init_neighbors();

    my_p4est_level_set_t ls(ngbd);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_n);
//    ls.reinitialize_2nd_order(phi_n);

    if(save_vtk && iter % save_every_n == 0)
      save_VTK(p4est, nodes, &brick, phi_n, iter/save_every_n);

    tn += dt;
    iter++;
  }

  ierr = PetscPrintf(mpi->mpicomm, "Final time: tf=%g\n", tn); CHKERRXX(ierr);

  /* compute the error */
  const double *phi_p;
  ierr = VecGetArrayRead(phi_n, &phi_p); CHKERRXX(ierr);
  double err = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
    if(fabs(phi_p[n])<3*MAX(dxyz_min[0],dxyz_min[1],dxyz_min[2]))
       err = max(err, (phi_p[n]-level_set(x,y,z)));
#else
    if(fabs(phi_p[n])<3*MAX(dxyz_min[0],dxyz_min[1]))
       err = max(err, fabs(phi_p[n]-level_set(x,y)));
#endif
  }
  ierr = VecRestoreArrayRead(phi_n, &phi_p); CHKERRXX(ierr);

  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, mpi->mpicomm); SC_CHECK_MPI(mpiret);
  ierr = PetscPrintf(mpi->mpicomm, "Error : %g\n", err); CHKERRXX(ierr);

  ierr = VecDestroy(phi_n);   CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(velo_n  [dir]); CHKERRXX(ierr);
  }

  /* destroy the p4est and its connectivity structure */
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
