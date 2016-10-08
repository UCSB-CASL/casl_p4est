
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
/* provides u, ux, uy, uz, rhs_cf, problem */
#include "test_inside.cpp"
//#include "test_outside.cpp"
#define DISPLAY_ERROR


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

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

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

  /* create the p4est */
  my_p4est_brick_t brick;

  const int n_xyz[] = {nx, ny, nz};
  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};

  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);;
  p4est_ghost_t *ghost = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &LS, 1.2);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);
//  p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
//  my_p4est_ghost_expand(p4est, ghost);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);

  ngbd->init_neighbors();

  my_p4est_level_set_t ls(&ngbd_n);

  /* create and initialize geometry */
  std::vector<Vec> phi;

  for (int i = 0; i < num_surfaces; i++)
  {
    phi.push_back(Vec());
    ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
    phi_cf.set_num(i); sample_cf_on_nodes(p4est, nodes, phi_cf, phi.back());
    ls.reinitialize_1st_order_time_2nd_order_space(phi.back(),10);
  }

  /* create and initialize potentials */
  Vec mu_m;
  Vec mu_p;

  ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &mu_p); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, mu_m_cf, mu_m);
  sample_cf_on_nodes(p4est, nodes, mu_p_cf, mu_p);

  /* create and initialize density fields */
  Vec rho_a;
  Vec rho_b;

  ierr = VecCreateGhostNodes(p4est, nodes, &rho_a); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rho_b); CHKERRXX(ierr);

  ierr = VecSet(rho_a, 0.5); CHKERRXX(ierr);
  ierr = VecSet(rho_b, 0.5); CHKERRXX(ierr);

  /* create and initialize surface tension fields */
  std::vector<Vec> gamma_a();
  std::vector<Vec> gamma_b();

  for (int i = 0; i < num_surfaces; i++)
  {
    gamma_a.push_back(Vec());
    gamma_b.push_back(Vec());

    ierr = VecCreateGhostNodes(p4est, nodes, &gamma_a); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &gamma_b); CHKERRXX(ierr);

    gamma_a_cf.set_num(i); sample_cf_on_nodes(p4est, nodes, gamma_a_cf, gamma_a);
    gamma_b_cf.set_num(i); sample_cf_on_nodes(p4est, nodes, gamma_b_cf, gamma_b);
  }

  /* initialize solver */
  my_p4est_scft_t scft(&ngbd_n);

  scft.set_geometry(phi, action);
  scft.set_parameters(N, f, Xab, ns);
  scft.set_surface_tensions(gamma_a, gamma_b);
  scft.set_potentials(mu_a, mu_b);
  scft.set_densities(rho_a, rho_b);

  scft.initialize_bc_simple();
//  scft.initialize_bc_smart();

  scft.initialize_linear_system();

  while (1)
  {
    scft.solve_for_propogators();
    scft.calculate_densities();
    scft.update_potentials();

    if (0)
    {
      scft.initialize_bc_smart();
      scft.initialize_linear_system();
//      scft.subtract_singularity_from_pressure_field();
    }

    if (1)
    {
      scft.print_VTK();
    }
  }

  w.stop(); w.read_duration();

  return 0;
}
