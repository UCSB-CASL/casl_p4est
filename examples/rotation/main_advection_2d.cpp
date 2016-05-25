#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>
#include <src/math.h>

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif


double xmin = 0;
double xmax = 1;
double ymin = 0;
double ymax = 1;
double zmin = 0;
double zmax = 1;

#ifdef P4_TO_P8
struct LEVEL_SET : CF_3
{
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y, double z) const
  {
    return sqrt(SQR(x-.35)+SQR(y-.35)+SQR(z-.35)) - .15;
  }
} level_set;

struct U0 : CF_3
{
  double operator()(double x, double y, double z) const
  {
    return 2*SQR(sin(PI*x))*sin(2*PI*y)*sin(2*PI*z);
  }
} u0;

struct V0 : CF_3
{
  double operator()(double x, double y, double z) const
  {
    return -SQR(sin(PI*y))*sin(2*PI*x)*sin(2*PI*z);
  }
} v0;

struct W0 : CF_3
{
  double operator()(double x, double y, double z) const
  {
    return -SQR(sin(PI*z))*sin(2*PI*x)*sin(2*PI*y);
  }
} w0;

struct U1 : CF_3
{
  double operator()(double x, double y, double z) const
  {
    return -2*SQR(sin(PI*x))*sin(2*PI*y)*sin(2*PI*z);
  }
} u1;

struct V1 : CF_3
{
  double operator()(double x, double y, double z) const
  {
    return SQR(sin(PI*y))*sin(2*PI*x)*sin(2*PI*z);
  }
} v1;

struct w1 : CF_3
{
  double operator()(double x, double y, double z) const
  {
    return SQR(sin(PI*z))*sin(2*PI*x)*sin(2*PI*y);
  }
} w1;
#else
struct LEVEL_SET : CF_2
{
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    return sqrt(SQR(x-.5)+SQR(y-.75)) - .15;
  }
} level_set;

struct U0 : CF_2
{
  double operator()(double x, double y) const
  {
    return -SQR(sin(PI*x))*sin(2*PI*y);
  }
} u0;

struct V0 : CF_2
{
  double operator()(double x, double y) const
  {
    return SQR(sin(PI*y))*sin(2*PI*x);
  }
} v0;

struct U1 : CF_2
{
  double operator()(double x, double y) const
  {
    return SQR(sin(PI*x))*sin(2*PI*y);
  }
} u1;

struct V1 : CF_2
{
  double operator()(double x, double y) const
  {
    return -SQR(sin(PI*y))*sin(2*PI*x);
  }
} v1;
#endif

void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, Vec phi, const char *name)
{
  PetscErrorCode ierr;

  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  PetscScalar *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
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

  const double *phi_p;

  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(p4est, nodes, ghost, P4EST_TRUE, P4EST_TRUE, 1, 1, name,
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_CELL_DATA, "leaf_level", l_p);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = PetscPrintf(p4est->mpicomm, "Saved vtu in %s\n", name); CHKERRXX(ierr);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);
}


int main(int argc, char ** argv)
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  ierr = PetscPrintf(mpi.comm(), "git hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG); CHKERRXX(ierr);

  cmdParser cmd;
  cmd.add_option("tf", "final time");
  cmd.add_option("lmin", "min level");
  cmd.add_option("lmax", "max level");
  cmd.add_option("save_vtk", "export vtk files");
  cmd.add_option("nb_splits", "number of split for convergence monitoring");
  cmd.add_option("nx", "number of trees in x direction");
  cmd.add_option("ny", "number of trees in y direction");
  cmd.add_option("nz", "number of trees in z direction");

  cmd.parse(argc, argv);
  cmd.print();

  int nx = cmd.get("nx", 1);
  int ny = cmd.get("ny", 1);
  int nz = cmd.get("nz", 1);

  int lmin = cmd.get("lmin", 0);
  int lmax = cmd.get("lmax", 5);
  int nb_splits = cmd.get("nb_splits", 1);
  double tf = cmd.get("tf", 1);
  bool save_vtk = cmd.get("save_vtk", 0);

  my_p4est_brick_t brick;
  p4est_connectivity_t *connectivity;

  int n_xyz [] = {nx, ny, nz};
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
  int periodic []   = {0, 0, 0};

  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  double err_nm1 = 0;
  double err_n   = 0;
  double mass_loss_nm1 = 0;
  double mass_loss_n   = 0;

#if defined(STAMPEDE) || defined(COMET)
    char *out_dir;
    out_dir = getenv("OUT_DIR");
#endif

  for(int repeat=0; repeat<nb_splits; ++repeat)
  {
    splitting_criteria_cf_t criteria(lmin, lmax+repeat, &level_set, 1.2);

    p4est_t *p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est->user_pointer = (void*) &criteria;

    for(int i=0; i<lmax+repeat; ++i)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
    my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy, nodes);

    Vec phi;
    const double *phi_p;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set, phi);

    double dxyz[P4EST_DIM];
    p4est_dxyz_min(p4est, dxyz);
    double dxyz_min = dxyz[0];
    for(int dir=1; dir<P4EST_DIM; ++dir)
    {
      dxyz_min = MIN(dxyz_min, dxyz[dir]);
    }
    double dt = 5*dxyz_min;
    double dtn = dt;

#ifdef P4_TO_P8
    const CF_3 *velo[] = {&u0, &v0, &w0};
#else
    const CF_2 *velo[] = {&u0, &v0};
#endif
    char name[1000];
    int iter = 0;

    if(save_vtk)
    {
#if defined(STAMPEDE) || defined(COMET)
#ifdef P4_TO_P8
      sprintf(name, "%s/vtu/step_vortex_%dx%dx%d_%d-%d_%05d", out_dir, nx, ny, nz, lmin, lmax+repeat, iter);
#else
      sprintf(name, "%s/vtu/step_vortex_%dx%d_%d-%d_%05d", out_dir, nx, ny, lmin, lmax+repeat, iter);
#endif
#else
#ifdef P4_TO_P8
      sprintf(name, "/home/guittet/code/Output/p4est_test/advection/3d_%05d", iter);
#else
      sprintf(name, "/home/guittet/code/Output/p4est_test/advection/2d_%05d", iter);
#endif
#endif
      save_VTK(p4est, ghost, nodes, phi, name);
    }

    double tn = 0;
#if defined(STAMPEDE) || defined(COMET)
#ifdef P4_TO_P8
    sprintf(name, "%s/vtu/vortex_%dx%dx%d_%d-%d_0", out_dir, nx, ny, nz, lmin, lmax+repeat);
#else
    sprintf(name, "%s/vtu/vortex_%dx%d_%d-%d_0", out_dir, nx, ny, lmin, lmax+repeat);
#endif
#else
#ifdef P4_TO_P8
    sprintf(name, "/home/guittet/code/Output/p4est_test/advection/vortex_%dx%dx%d_%d-%d_0", nx, ny, nz, lmin, lmax+repeat);
#else
    sprintf(name, "/home/guittet/code/Output/p4est_test/advection/vortex_%dx%d_%d-%d_0", nx, ny, lmin, lmax+repeat);
#endif
#endif
    save_VTK(p4est, ghost, nodes, phi, name);

    iter ++;

    while(tn<tf)
    {
      if(tn+dtn>tf) dtn = tf-tn;

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est->user_pointer = (void*) &criteria;
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n);
      sl.update_p4est(velo, dtn, phi);

      p4est_nodes_destroy(nodes); nodes = nodes_np1;
      p4est_ghost_destroy(ghost); ghost = ghost_np1;
      p4est_destroy(p4est); p4est = p4est_np1;
      delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
      delete ngbd_n; ngbd_n = new my_p4est_node_neighbors_t(hierarchy, nodes);

      my_p4est_level_set_t ls(ngbd_n);
      ls.reinitialize_1st_order_time_2nd_order_space(phi,40);

      if(save_vtk)
      {
#if defined(STAMPEDE) || defined(COMET)
#ifdef P4_TO_P8
        sprintf(name, "%s/vtu/step_vortex_%dx%dx%d_%d-%d_%05d", out_dir, nx, ny, nz, lmin, lmax+repeat, iter);
#else
        sprintf(name, "%s/vtu/step_vortex_%dx%d_%d-%d_%05d", out_dir, nx, ny, lmin, lmax+repeat, iter);
#endif
#else
#ifdef P4_TO_P8
        sprintf(name, "/home/guittet/code/Output/p4est_test/advection/3d_%05d", iter);
#else
        sprintf(name, "/home/guittet/code/Output/p4est_test/advection/2d_%05d", iter);
#endif
#endif
        save_VTK(p4est, ghost, nodes, phi, name);
      }

      tn += dtn;
//      ierr = PetscPrintf(p4est->mpicomm, "Iter #%d, tn = %e\n", iter, tn); CHKERRXX(ierr);
      iter++;
    }

    ierr = PetscPrintf(p4est->mpicomm, "Going back !\n"); CHKERRXX(ierr);

    velo[0] = &u1;
    velo[1] = &v1;
#ifdef P4_TO_P8
    velo[2] = &w1;
#endif

#if defined(STAMPEDE) || defined(COMET)
#ifdef P4_TO_P8
    sprintf(name, "%s/vtu/vortex_%dx%dx%d_%d-%d_1", out_dir, nx, ny, nz, lmin, lmax+repeat);
#else
    sprintf(name, "%s/vtu/vortex_%dx%d_%d-%d_1", out_dir, nx, ny, lmin, lmax+repeat);
#endif
#else
#ifdef P4_TO_P8
    sprintf(name, "/home/guittet/code/Output/p4est_test/advection/vortex_%dx%dx%d_%d-%d_1", nx, ny, nz, lmin, lmax+repeat);
#else
    sprintf(name, "/home/guittet/code/Output/p4est_test/advection/vortex_%dx%d_%d-%d_1", nx, ny, lmin, lmax+repeat);
#endif
#endif
    save_VTK(p4est, ghost, nodes, phi, name);

    dtn = dt;

    while(tn>0)
    {
      if(dtn>tn) dtn = tn;

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est->user_pointer = (void*) &criteria;
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n);
      sl.update_p4est(velo, dt, phi);

      p4est_nodes_destroy(nodes); nodes = nodes_np1;
      p4est_ghost_destroy(ghost); ghost = ghost_np1;
      p4est_destroy(p4est); p4est = p4est_np1;
      delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
      delete ngbd_n; ngbd_n = new my_p4est_node_neighbors_t(hierarchy, nodes);

      my_p4est_level_set_t ls(ngbd_n);
      ls.reinitialize_1st_order_time_2nd_order_space(phi,40);

      if(save_vtk)
      {
#if defined(STAMPEDE) || defined(COMET)
#ifdef P4_TO_P8
        sprintf(name, "%s/vtu/step_vortex_%dx%dx%d_%d-%d_%05d", out_dir, nx, ny, nz, lmin, lmax+repeat, iter);
#else
        sprintf(name, "%s/vtu/step_vortex_%dx%d_%d-%d_%05d", out_dir, nx, ny, lmin, lmax+repeat, iter);
#endif
#else
#ifdef P4_TO_P8
        sprintf(name, "/home/guittet/code/Output/p4est_test/advection/3d_%05d", iter);
#else
        sprintf(name, "/home/guittet/code/Output/p4est_test/advection/2d_%05d", iter);
#endif
#endif
        save_VTK(p4est, ghost, nodes, phi, name);
      }

      tn -= dtn;
//      ierr = PetscPrintf(p4est->mpicomm, "Iter #%d, tn = %e\n", iter, tn); CHKERRXX(ierr);
      iter++;
    }

#if defined(STAMPEDE) || defined(COMET)
#ifdef P4_TO_P8
    sprintf(name, "%s/vtu/vortex_%dx%dx%d_%d-%d_2", out_dir, nx, ny, nz, lmin, lmax+repeat);
#else
    sprintf(name, "%s/vtu/vortex_%dx%d_%d-%d_2", out_dir, nx, ny, lmin, lmax+repeat);
#endif
#else
#ifdef P4_TO_P8
    sprintf(name, "/home/guittet/code/Output/p4est_test/advection/vortex_%dx%dx%d_%d-%d_2", nx, ny, nz, lmin, lmax+repeat);
#else
    sprintf(name, "/home/guittet/code/Output/p4est_test/advection/vortex_%dx%d_%d-%d_2", nx, ny, lmin, lmax+repeat);
#endif
#endif
    save_VTK(p4est, ghost, nodes, phi, name);

    /* check error */
    err_nm1 = err_n;
    err_n = 0;
    double xyz[P4EST_DIM];

    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    double xyz_m[P4EST_DIM];

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(fabs(phi_p[n])<dxyz_min)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);
#ifdef P4_TO_P8
        double err = fabs(phi_p[n] - level_set(xyz[0],xyz[1],xyz[2]));
#else
        double err = fabs(phi_p[n] - level_set(xyz[0],xyz[1]));
#endif

        if(err_n<err)
        {
          xyz_m[0] = xyz[0];
          xyz_m[1] = xyz[1];
#ifdef P4_TO_P8
          xyz_m[2] = xyz[2];
#endif
          err_n = err;
        }
      }
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &xyz_m[0], P4EST_DIM, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    double vol = area_in_negative_domain(p4est, nodes, phi);
    double vol_exa = 4./3. * PI * .15*.15*.15;
    mass_loss_nm1 = mass_loss_n;
    mass_loss_n = fabs(vol-vol_exa)/vol_exa;

    ierr = PetscPrintf(p4est->mpicomm, "level %d/%d, error = %e, \torder = %e, \tmass loss = %e, \torder = %e\n",
                       lmin, lmax+repeat, err_n, log(err_nm1/err_n)/log(2),
                       mass_loss_n, log(mass_loss_nm1/mass_loss_n)/log(2)); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "error at (%e, %e, %e)\n", xyz_m[0], xyz_m[1], xyz_m[2]);
    ierr = VecDestroy(phi);

    delete ngbd_n;
    delete hierarchy;
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy(p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
