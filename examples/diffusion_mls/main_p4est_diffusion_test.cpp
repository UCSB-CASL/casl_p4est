
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

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const int n_xyz[] = {nx, ny, nz};
  const double xyz_min[] = {xmin, ymin, zmin};
  const double xyz_max[] = {xmax, ymax, zmax};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

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
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    splitting_criteria_cf_t data(0, lmax+iter, &level_set_ref, 1.4);
//    splitting_criteria_cf_t data(lmin, lmax+iter, &level_set_tot, 1.4);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set_tot, 2.0);
    p4est->user_pointer = (void*)(&data);

    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
//    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    my_p4est_level_set_t ls(&ngbd_n);

    cout << "Grid has been constructed\n";

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
    for (int i = 0; i < problem.phi_cf.size(); i++)
    {
      phi.push_back(Vec());
      ierr = VecCreateGhostNodes(p4est, nodes, &phi.back()); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, *problem.phi_cf[i], phi.back());
      ls.reinitialize_1st_order_time_2nd_order_space(phi.back(),10);
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

    Vec mu;
    ierr = VecCreateGhostNodes(p4est, nodes, &mu); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_cf, mu);

    Vec diag_add;
    ierr = VecCreateGhostNodes(p4est, nodes, &diag_add); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, diag_add_cf, diag_add);

//    ierr = VecDestroy(rhs); CHKERRXX(ierr);
//    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);

    // Time discretization
    double time_min = 0.0;
    double time_max = 0.5;
    int nt = pow(2, lmin+iter);
    double dt = (time_max - time_min) / (double)(nt-1);

    double time = 0;
    u_exact.t = time;
    lap_u.t   = time;
    ux.t      = time;
    uy.t      = time;
#ifdef P4_TO_P8
    uz.t      = time;
#endif
    ut.t      = time;

    cout << "Starting a solver\n";
    my_p4est_poisson_nodes_mls_t solver(&ngbd_n);
    solver.set_geometry(phi, problem.action, problem.color);
    solver.set_mu(mu);
    solver.set_rhs(rhs);
    solver.wall_value.set(0.0);
    solver.set_bc_type(problem.bc_type);
    solver.set_diag_add(2.0/dt);
    solver.set_bc_coeffs(bc_coeffs);
    solver.set_bc_values(bc_values);
    solver.set_use_taylor_correction(true);
    solver.set_keep_scalling(true);
    solver.set_kinks_treatment(true);

    solver.compute_volumes();
//    solver.set_cube_refinement(0);

    // Initial condition
    u_exact.t = 0;
    Vec sol; ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, u_exact, sol);

    // Auxiliary vector for correcting RHS
    Vec add_to_rhs;
    ierr = VecCreateGhostNodes(p4est, nodes, &add_to_rhs); CHKERRXX(ierr);
    Vec rhs_old;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_old); CHKERRXX(ierr);

    solver.reusing_matrix = false;

    solver.setup_negative_variable_coeff_laplace_matrix_sym();

    for (int i_time = 1; i_time < nt; i_time++)
    {
      // Advance time
      time += dt;

      // Set correct time for analytical functions
      u_exact.t = time - 0.5*dt;
      lap_u.t   = time - 0.5*dt;
      ux.t      = time - 0.5*dt;
      uy.t      = time - 0.5*dt;
#ifdef P4_TO_P8
      uz.t      = time - 0.5*dt;
#endif
      ut.t      = time - 0.5*dt;

      // Calculate force at (t_n + dt/2)
      sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);
      solver.set_rhs(rhs);

      // Calculate BCs at (t_n + dt/2)
      for (int i = 0; i < problem.bc_values.size(); i++)
      {
        sample_cf_on_nodes(p4est, nodes, *problem.bc_values[i], bc_values.at(i));
      }
      solver.set_bc_values(bc_values);

      // Correct RHS
      ierr = VecPointwiseMult(add_to_rhs, solver.node_vol, sol);          CHKERRXX(ierr);
      ierr = VecPointwiseDivide(add_to_rhs, add_to_rhs, solver.scalling); CHKERRXX(ierr);
      if (i_time == 1)
      {
        ierr = VecScale(add_to_rhs, -4.0/dt);                               CHKERRXX(ierr);
        ierr = MatMultAdd(solver.A, sol, add_to_rhs, add_to_rhs);           CHKERRXX(ierr);
      }
      else
      {
        ierr = VecAYPX(add_to_rhs, -4.0/dt, rhs_old); CHKERRXX(ierr);
      }

      // Calculate RHS as for Poisson eqn
      solver.setup_negative_variable_coeff_laplace_rhsvec_sym();
      ierr = VecAXPBY(solver.rhs, -1.0, 2.0, add_to_rhs);                 CHKERRXX(ierr);
      VecCopy(solver.rhs, rhs_old);

      // Solve linear system
      solver.solve_linear_system(sol, false);
    }

    // Set time in analytical functions to the end time
    u_exact.t = time;
    ux.t = time;
    uy.t = time;
#ifdef P4_TO_P8
    uz.t = time;
#endif

    Vec u_exact_vec;
    ierr = VecCreateGhostNodes(p4est, nodes, &u_exact_vec); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, u_exact, u_exact_vec);

//    solver.solve(sol);
//    cin.get();

//    // extrapolation
//    Vec phi_shifted;
//    ierr = VecCreateGhostNodes(p4est, nodes, &phi_shifted); CHKERRXX(ierr);
//    ierr = VecDuplicate(solver.phi_eff, &phi_shifted); CHKERRXX(ierr);
//    ierr = VecCopy(solver.phi_eff, phi_shifted); CHKERRXX(ierr);
//    VecShift(phi_shifted, 0.*diag);
////    sample_cf_on_nodes(p4est, nodes, level_set_tot, phi_tot);
//    ls.reinitialize_1st_order_time_2nd_order_space(phi_shifted);
//    ls.extend_Over_Interface_TVD(phi_shifted, sol, 20);
//    ierr = VecDestroy(phi_shifted); CHKERRXX(ierr);

#ifdef DISPLAY_ERROR
    my_p4est_integration_mls_t integrator;
    integrator.set_p4est(p4est, nodes);
#ifdef P4_TO_P8
    integrator.set_phi(phi, *solver.phi_xx, *solver.phi_yy, *solver.phi_zz, problem.action, problem.color);
#else
    integrator.set_phi(phi, *solver.phi_xx, *solver.phi_yy, problem.action, problem.color);
#endif
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
//    cout << "Here!\n";
#ifdef P4_TO_P8
    solver.compute_error_gr(ux, uy, uz, sol, error_ux, error_uy, error_uz);
#else
    solver.compute_error_gr(ux, uy, sol, error_ux, error_uy);
#endif

//    ls.extend_Over_Interface_TVD(phi_shifted, error_ux, 20);
//    ls.extend_Over_Interface_TVD(phi_shifted, error_uy, 20);

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
//    integrator.initialize();
    double measure_of_dom = integrator.measure_of_domain();
    error_sl_l1_arr.push_back(integrator.integrate_everywhere(error_sl)/measure_of_dom);
    error_tr_l1_arr.push_back(integrator.integrate_everywhere(error_tr)/measure_of_dom);
    error_ux_l1_arr.push_back(integrator.integrate_everywhere(error_ux)/measure_of_dom);
    error_uy_l1_arr.push_back(integrator.integrate_everywhere(error_uy)/measure_of_dom);
#ifdef P4_TO_P8
    error_uz_l1_arr.push_back(integrator.integrate_everywhere(error_uz)/measure_of_dom);
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

    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &brick, solver.phi_eff, sol, error_sl, solver.rhs, error_ux, iter);
    }

    // destroy Vec's with errors
    ierr = VecDestroy(error_sl); CHKERRXX(ierr);
    ierr = VecDestroy(error_tr); CHKERRXX(ierr);
    ierr = VecDestroy(error_ux); CHKERRXX(ierr);
    ierr = VecDestroy(error_uy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(error_uz); CHKERRXX(ierr);
#endif
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



    for (int i = 0; i < phi.size(); i++)
    {
      ierr = VecDestroy(phi[i]);        CHKERRXX(ierr);
      ierr = VecDestroy(bc_values[i]);  CHKERRXX(ierr);
      ierr = VecDestroy(bc_coeffs[i]);  CHKERRXX(ierr);
    }

    ierr = VecDestroy(sol);         CHKERRXX(ierr);
    ierr = VecDestroy(mu);          CHKERRXX(ierr);
    ierr = VecDestroy(rhs);         CHKERRXX(ierr);
    ierr = VecDestroy(diag_add);    CHKERRXX(ierr);
    ierr = VecDestroy(u_exact_vec); CHKERRXX(ierr);
    ierr = VecDestroy(add_to_rhs); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_old); CHKERRXX(ierr);

//    ierr = VecDestroy(error_uxy); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

#ifdef DISPLAY_ERROR
  if (mpi.rank() == 0)
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
#endif

  return 0;
}
