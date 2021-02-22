// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>
#include <examples/two_phase_flows_validation/test_cases_for_two_phase_flows.h>

#undef MIN
#undef MAX

using namespace std;


const static string main_description =
    string("In this example, we test the fully sharp and fully implicit solver for incompressible two-phase flows\n")
    + string("The user can choose from several test cases (described in the list of possible 'test'), set various \n")
    + string("min/max levels of refinement, the number of trees long every Cartesian direction, in the macromesh.\n")
    + string("Results and illustration data can be saved in vtk format as well. \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), 2019-2020-2021-...-2523\n");

// test index
const int default_test = 0;
// grid-related
const int default_lmin = 4;
const int default_lmax = 6;
const double default_vorticity_threshold    = DBL_MAX;
const double default_uniform_band_in_dx_min = 5;
const int default_ntree[P4EST_DIM] = {DIM(1, 1, 1)};
// simulation-related:
const interpolation_method default_interp_method_phi = quadratic_non_oscillatory_continuous_v2;
const bool default_subrefinement = false;
const bool default_use_second_order_theta = (default_interp_method_phi == linear ? false : true); // relevant only if using (x)GFM cell solver
const int default_nviscous_subiter  = 5;
const int default_sl_order          = 2;
const int default_sl_order_itfc     = 2;
const double default_cfl_advection  = 0.85;
const double default_cfl_visco_capillary = 0.95;
const double default_cfl_capillary  = 0.95;
const jump_solver_tag default_cell_solver = FV;
const jump_solver_tag default_face_solver = xGFM;
const int default_n_reinit = 1;
const double default_vmax_abort = 100.0;
const double default_projection_threshold = 0.001;
const int default_niter = 5;
// exportation-related
const bool default_save_vtk = true;
const double default_vtk_dt         = 0.1;
const int default_vtk_idx_start     = 0;
const int default_save_nstates      = 0;
const double default_save_state_dt  = 0.25;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_flow/validation_tests/" + to_string(P4EST_DIM) + "D";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_flow/validation_tests/" + to_string(P4EST_DIM) + "D";
#elif defined(ABADDON)
const string default_work_folder = "/Users/raphael/workspace/projects/two_phase_flow/validation_tests/" + to_string(P4EST_DIM) + "D";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_flow/validation_tests/" + to_string(P4EST_DIM) + "D";
#endif

void get_pressure_errors(double& error_p_minus, double& error_p_plus, const my_p4est_two_phase_flows_t* solver, test_case_for_two_phase_flows_t* test_problem)
{
  const my_p4est_interface_manager_t* interface_manager = solver->get_interface_manager();
  const p4est_t*          computational_p4est = solver->get_p4est_n();
  const p4est_ghost_t*    computational_ghost = solver->get_ghost_n();
  double xyz[P4EST_DIM];
  Vec pressure_minus  = solver->get_pressure_minus();
  Vec pressure_plus   = solver->get_pressure_plus();
  const double *pressure_minus_p, *pressure_plus_p;
  PetscErrorCode ierr;
  int mpiret;
  error_p_minus = error_p_plus = 0.0; // initialization
  ierr= VecGetArrayRead(pressure_minus, &pressure_minus_p); CHKERRXX(ierr);
  ierr= VecGetArrayRead(pressure_plus, &pressure_plus_p); CHKERRXX(ierr);
  double floating_pressure = 0.0;
  if(test_problem->is_pressure_floating())
  {
    for (p4est_topidx_t tree_idx = computational_p4est->first_local_tree; tree_idx <= computational_p4est->last_local_tree; ++tree_idx)
    {
      const p4est_tree_t* tree = p4est_tree_array_index(computational_p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        quad_xyz_fr_q(quad_idx, tree_idx, computational_p4est, computational_ghost, xyz);
        const char sgn_cell = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
        if(sgn_cell < 0)
          floating_pressure += pressure_minus_p[quad_idx] - test_problem->pressure_minus(xyz);
        else
          floating_pressure += pressure_plus_p[quad_idx] -test_problem->pressure_plus(xyz);
      }
    }
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &floating_pressure, 1, MPI_DOUBLE, MPI_SUM, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
    floating_pressure /= computational_p4est->global_num_quadrants;
  }
  for (p4est_topidx_t tree_idx = computational_p4est->first_local_tree; tree_idx <= computational_p4est->last_local_tree; ++tree_idx)
  {
    const p4est_tree_t* tree = p4est_tree_array_index(computational_p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      quad_xyz_fr_q(quad_idx, tree_idx, computational_p4est, computational_ghost, xyz);
      const char sgn_cell = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
      if(sgn_cell < 0)
        error_p_minus = MAX(error_p_minus,  fabs(pressure_minus_p[quad_idx] - floating_pressure - test_problem->pressure_minus(xyz)));
      else
        error_p_plus  = MAX(error_p_plus,   fabs(pressure_plus_p[quad_idx] - floating_pressure - test_problem->pressure_plus(xyz)));
    }
  }
  ierr= VecRestoreArrayRead(pressure_minus, &pressure_minus_p); CHKERRXX(ierr);
  ierr= VecRestoreArrayRead(pressure_plus, &pressure_plus_p); CHKERRXX(ierr);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_p_minus, 1, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_p_plus,  1, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
  // done with pressure np1 at cells
  return;
}

void print_convergence_results(const string& export_dir, my_p4est_two_phase_flows_t* solver, test_case_for_two_phase_flows_t* test_problem)
{
  string filename = export_dir + "/sharp_error_analysis.dat";
  const my_p4est_interface_manager_t* interface_manager = solver->get_interface_manager();
  test_problem->set_time(solver->get_tnp1());

  const p4est_t*          computational_p4est = solver->get_p4est_n();
  const p4est_nodes_t*    computational_nodes = solver->get_nodes_n();
  const my_p4est_faces_t* computational_faces = solver->get_faces_n();
  double xyz[P4EST_DIM];

  PetscErrorCode ierr;
  int mpiret;
  double error_levelset = NAN;
  if(!test_problem->is_interface_static())
  {
    const p4est_nodes_t* interface_capturing_nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
    Vec exact_phi = NULL;
    double *exact_phi_p;
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(exact_phi);
    ierr = VecGetArray(exact_phi, &exact_phi_p); CHKERRXX(ierr);
    for(p4est_locidx_t node_idx = 0; node_idx < interface_capturing_nodes->num_owned_indeps; node_idx++)
    {
      node_xyz_fr_n(node_idx, interface_manager->get_interface_capturing_ngbd_n().get_p4est(), interface_capturing_nodes, xyz);
      exact_phi_p[node_idx] = test_problem->levelset_function(xyz);
    }
    ierr = VecRestoreArray(exact_phi, &exact_phi_p); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(exact_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(exact_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(test_problem->is_reinitialization_needed())
    {
      my_p4est_level_set_t ls(&interface_manager->get_interface_capturing_ngbd_n());
      ls.reinitialize_2nd_order(exact_phi);
    }

    error_levelset = 0.0;
    const double band = 3.0*ABSD(computational_faces->get_smallest_dxyz()[0], computational_faces->get_smallest_dxyz()[1], computational_faces->get_smallest_dxyz()[2]);
    const double *our_phi_read_p, *exact_phi_read_p;
    ierr = VecGetArrayRead(interface_manager->get_phi(), &our_phi_read_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(exact_phi, &exact_phi_read_p); CHKERRXX(ierr);
    for (p4est_locidx_t node_idx = 0; node_idx < interface_capturing_nodes->num_owned_indeps; ++node_idx) {
      if(fabs(exact_phi_read_p[node_idx]) > band)
        continue;
      error_levelset = MAX(error_levelset, fabs(our_phi_read_p[node_idx] - exact_phi_read_p[node_idx]));
    }
    ierr = VecRestoreArrayRead(exact_phi, &exact_phi_read_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(interface_manager->get_phi(), &our_phi_read_p); CHKERRXX(ierr);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, &error_levelset, 1, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = delete_and_nullify_vector(exact_phi); CHKERRXX(ierr);
  }
  // errors for velocity_np1 at nodes
  Vec vnp1_nodes_minus  = solver->get_vnp1_nodes_minus();
  Vec vnp1_nodes_plus   = solver->get_vnp1_nodes_plus();
  const double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
  ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  double error_v_minus_nodes[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
  double error_v_plus_nodes[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
  for (p4est_locidx_t node_idx = 0; node_idx < computational_nodes->num_owned_indeps; ++node_idx) {
    node_xyz_fr_n(node_idx, computational_p4est, computational_nodes, xyz);
    const char sgn_node = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
    if(sgn_node < 0)
    {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
        error_v_minus_nodes[dim] = MAX(error_v_minus_nodes[dim], fabs(vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim] - test_problem->velocity_minus(dim, xyz)));
    }
    else
    {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
        error_v_plus_nodes[dim] = MAX(error_v_plus_nodes[dim], fabs(vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim] - test_problem->velocity_plus(dim, xyz)));
    }
  }
  ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_v_minus_nodes, P4EST_DIM, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_v_plus_nodes,  P4EST_DIM, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
  // done with velocity_np1 at nodes

  // errors for velocity_np1 at faces
  const Vec* vnp1_face_minus  = solver->get_vnp1_face_minus();
  const Vec* vnp1_face_plus   = solver->get_vnp1_face_plus();
  const double *vnp1_face_minus_p[P4EST_DIM], *vnp1_face_plus_p[P4EST_DIM];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr= VecGetArrayRead(vnp1_face_minus[dim], &vnp1_face_minus_p[dim]); CHKERRXX(ierr);
    ierr= VecGetArrayRead(vnp1_face_plus[dim],  &vnp1_face_plus_p[dim]); CHKERRXX(ierr);
  }
  double error_v_minus_face[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
  double error_v_plus_face[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    for (p4est_locidx_t f_idx = 0; f_idx < computational_faces->num_local[dim]; ++f_idx) {
      computational_faces->xyz_fr_f(f_idx, dim, xyz);
      const char sgn_face = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
      if(sgn_face < 0)
        error_v_minus_face[dim] = MAX(error_v_minus_face[dim], fabs(vnp1_face_minus_p[dim][f_idx] - test_problem->velocity_minus(dim, xyz)));
      else
        error_v_plus_face[dim]  = MAX(error_v_plus_face[dim], fabs(vnp1_face_plus_p[dim][f_idx] - test_problem->velocity_plus(dim, xyz)));
    }
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr= VecRestoreArrayRead(vnp1_face_minus[dim], &vnp1_face_minus_p[dim]); CHKERRXX(ierr);
    ierr= VecRestoreArrayRead(vnp1_face_plus[dim],  &vnp1_face_plus_p[dim]); CHKERRXX(ierr);
  }
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_v_minus_face, P4EST_DIM, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, error_v_plus_face,  P4EST_DIM, MPI_DOUBLE, MPI_MAX, computational_p4est->mpicomm); SC_CHECK_MPI(mpiret);
  // done with velocity_np1 at faces

  // errors for pressure np1 at cells (2 ways)
  double error_p_minus, error_p_plus;
  get_pressure_errors(error_p_minus, error_p_plus, solver, test_problem);
  double error_finalized_p_minus, error_finalized_p_plus;
  solver->subtract_mu_div_star_from_pressure();
  get_pressure_errors(error_finalized_p_minus, error_finalized_p_plus, solver, test_problem);

  if(solver->get_rank() == 0)
  {
    FILE* fp_results = fopen(filename.c_str(), "w");
    if(fp_results == NULL)
      throw std::runtime_error("print_convergence_results: could not open file for output of cnovergence results.");
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    fprintf(fp_results, "---------------- MAX ERRORS IN FULL DOMAIN -------------------- \n");
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    if(!test_problem->is_interface_static())
    {
      fprintf(fp_results, "--------------------------------------------------------------- \n");
      fprintf(fp_results, "Error on the levelset at nodes of the interface-capturing grid: \n");
      fprintf(fp_results, "--------------------------------------------------------------- \n");
      fprintf(fp_results, "Error on \\phi = %g \n", error_levelset);
    }
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error for velocities sampled at computational nodes: \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error on u = %g \n", MAX(error_v_minus_nodes[0], error_v_plus_nodes[0]));
    fprintf(fp_results, "Error on v = %g \n", MAX(error_v_minus_nodes[1], error_v_plus_nodes[1]));
#ifdef P4_TO_P8
    fprintf(fp_results, "Error on w = %g \n", MAX(error_v_minus_nodes[2], error_v_plus_nodes[2]));
#endif
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error for velocities sampled at computational faces: \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error on u = %g \n", MAX(error_v_minus_face[0], error_v_plus_face[0]));
    fprintf(fp_results, "Error on v = %g \n", MAX(error_v_minus_face[1], error_v_plus_face[1]));
#ifdef P4_TO_P8
    fprintf(fp_results, "Error on w = %g \n", MAX(error_v_minus_face[2], error_v_plus_face[2]));
#endif
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error for pressure sampled at computational cells    \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error on p = %g \n", MAX(error_p_minus, error_p_plus));
    fprintf(fp_results, "Error on \"finalized\" p = %g \n", MAX(error_finalized_p_minus, error_finalized_p_plus));
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    fprintf(fp_results, "---------------- MAX ERRORS IN SUBDOMAINS --------------------- \n");
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    fprintf(fp_results, "--------------------------------------------------------------- \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error for velocities sampled at computational nodes: \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "In negative domain: \n");
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "Error on u = %g \n", error_v_minus_nodes[0]);
    fprintf(fp_results, "Error on v = %g \n", error_v_minus_nodes[1]);
#ifdef P4_TO_P8
    fprintf(fp_results, "Error on w = %g \n", error_v_minus_nodes[2]);
#endif
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "In positive domain: \n");
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "Error on u = %g \n", error_v_plus_nodes[0]);
    fprintf(fp_results, "Error on v = %g \n", error_v_plus_nodes[1]);
#ifdef P4_TO_P8
    fprintf(fp_results, "Error on w = %g \n", error_v_plus_nodes[2]);
#endif
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error for velocities sampled at computational faces: \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "In negative domain: \n");
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "Error on u = %g \n", error_v_minus_face[0]);
    fprintf(fp_results, "Error on v = %g \n", error_v_minus_face[1]);
#ifdef P4_TO_P8
    fprintf(fp_results, "Error on w = %g \n", error_v_minus_face[2]);
#endif
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "In positive domain: \n");
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "Error on u = %g \n", error_v_plus_face[0]);
    fprintf(fp_results, "Error on v = %g \n", error_v_plus_face[1]);
#ifdef P4_TO_P8
    fprintf(fp_results, "Error on w = %g \n", error_v_plus_face[2]);
#endif
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "Error for pressure sampled at computational cells    \n");
    fprintf(fp_results, "---------------------------------------------------- \n");
    fprintf(fp_results, "In negative domain: \n");
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "Error on p = %g \n", error_p_minus);
    fprintf(fp_results, "Error on \"finalized\" p = %g \n", error_finalized_p_minus);
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "In positive domain: \n");
    fprintf(fp_results, "------------------- \n");
    fprintf(fp_results, "Error on p = %g \n", error_p_plus);
    fprintf(fp_results, "Error on \"finalized\" p = %g \n", error_finalized_p_plus);
    fprintf(fp_results, "---------------------------------------------------- \n");
    fclose(fp_results);
  }
  return;
}

void export_error_visualization(const string& vtk_dir, const my_p4est_two_phase_flows_t* solver, test_case_for_two_phase_flows_t* test_problem)
{
  PetscErrorCode ierr;
  Vec error_v_minus_np1_nodes;
  Vec error_v_plus_np1_nodes;
  Vec error_v_sharp_np1_nodes;
  Vec error_pressure_minus;
  Vec error_pressure_plus;
  Vec error_sharp_pressure;
  const p4est_t* p4est = solver->get_p4est_n();
  const p4est_nodes_t* nodes = solver->get_nodes_n();
  const p4est_ghost_t* ghost = solver->get_ghost_n();
  const my_p4est_faces_t* faces = solver->get_faces_n();
  const my_p4est_interface_manager_t* interface_manager = solver->get_interface_manager();
  ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &error_v_minus_np1_nodes); CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &error_v_plus_np1_nodes); CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &error_v_sharp_np1_nodes); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est, ghost, &error_sharp_pressure); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est, ghost, &error_pressure_minus); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est, ghost, &error_pressure_plus); CHKERRXX(ierr);
  Vec vnp1_nodes_minus  = solver->get_vnp1_nodes_minus();
  Vec vnp1_nodes_plus   = solver->get_vnp1_nodes_plus();
  Vec pressure_minus    = solver->get_pressure_minus();
  Vec pressure_plus     = solver->get_pressure_plus();
  const double* vnp1_nodes_minus_p, *vnp1_nodes_plus_p, *pressure_minus_p, *pressure_plus_p;
  double *error_v_minus_np1_nodes_p, *error_v_plus_np1_nodes_p, *error_v_sharp_np1_nodes_p, *error_pressure_minus_p, *error_pressure_plus_p, *error_sharp_pressure_p;
  ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(pressure_minus,    &pressure_minus_p);   CHKERRXX(ierr);
  ierr = VecGetArrayRead(pressure_plus,     &pressure_plus_p);    CHKERRXX(ierr);
  ierr = VecGetArray(error_v_minus_np1_nodes, &error_v_minus_np1_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(error_v_plus_np1_nodes , &error_v_plus_np1_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(error_v_sharp_np1_nodes, &error_v_sharp_np1_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(error_sharp_pressure   , &error_sharp_pressure_p); CHKERRXX(ierr);
  ierr = VecGetArray(error_pressure_minus   , &error_pressure_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(error_pressure_plus    , &error_pressure_plus_p); CHKERRXX(ierr);
  double xyz[P4EST_DIM];
  test_problem->set_time(solver->get_tnp1());
  for (p4est_locidx_t node_idx = 0; node_idx < nodes->num_owned_indeps; ++node_idx) {
    node_xyz_fr_n(node_idx, p4est, nodes, xyz);
    char sgn_node = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      error_v_minus_np1_nodes_p[P4EST_DIM*node_idx + dim] = fabs(vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim] - test_problem->velocity_minus(dim, xyz));
      error_v_plus_np1_nodes_p[P4EST_DIM*node_idx + dim] = fabs(vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim] - test_problem->velocity_plus(dim, xyz));
      if(sgn_node < 0)
        error_v_sharp_np1_nodes_p[P4EST_DIM*node_idx + dim] = error_v_minus_np1_nodes_p[P4EST_DIM*node_idx + dim];
      else
        error_v_sharp_np1_nodes_p[P4EST_DIM*node_idx + dim] = error_v_plus_np1_nodes_p[P4EST_DIM*node_idx + dim];
    }
  }
  ierr = VecGhostUpdateBegin(error_v_minus_np1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(error_v_plus_np1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(error_v_sharp_np1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(error_v_minus_np1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(error_v_plus_np1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(error_v_sharp_np1_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  double floating_pressure = 0.0;
  if(test_problem->is_pressure_floating())
  {
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
      const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
        quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz);
        const char sgn_cell = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
        if(sgn_cell < 0)
          floating_pressure += pressure_minus_p[quad_idx] - test_problem->pressure_minus(xyz);
        else
          floating_pressure += pressure_plus_p[quad_idx] -test_problem->pressure_plus(xyz);
      }
    }
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &floating_pressure, 1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    floating_pressure /= p4est->global_num_quadrants;
  }
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx = tree->quadrants_offset + q;
      quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz);
      char sgn_quad = (interface_manager->phi_at_point(xyz) <= 0.0 ? -1 : +1);
      error_pressure_minus_p[quad_idx]  = fabs(pressure_minus_p[quad_idx] - floating_pressure - test_problem->pressure_minus(xyz));
      error_pressure_plus_p[quad_idx]   = fabs(pressure_plus_p[quad_idx] - floating_pressure - test_problem->pressure_plus(xyz));
      if(sgn_quad < 0)
        error_sharp_pressure_p[quad_idx] = error_pressure_minus_p[quad_idx];
      else
        error_sharp_pressure_p[quad_idx] = error_pressure_plus_p[quad_idx];
    }
  }
  ierr = VecGhostUpdateBegin(error_pressure_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(error_pressure_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(error_sharp_pressure, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(error_pressure_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(error_pressure_plus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(error_sharp_pressure, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  ierr = VecRestoreArray(error_v_minus_np1_nodes, &error_v_minus_np1_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error_v_plus_np1_nodes , &error_v_plus_np1_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error_v_sharp_np1_nodes, &error_v_sharp_np1_nodes_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error_sharp_pressure   , &error_sharp_pressure_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error_pressure_minus   , &error_pressure_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error_pressure_plus    , &error_pressure_plus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(pressure_minus,    &pressure_minus_p);   CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(pressure_plus,     &pressure_plus_p);    CHKERRXX(ierr);


  Vec error_v_minus_np1_faces[P4EST_DIM], error_v_minus_np1_on_cells;
  Vec error_v_plus_np1_faces[P4EST_DIM],  error_v_plus_np1_on_cells;
  const double *v_minus_np1_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  const double *v_plus_np1_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
  double *error_v_minus_np1_faces_p[P4EST_DIM]  = {DIM(NULL, NULL, NULL)};
  double *error_v_plus_np1_faces_p[P4EST_DIM]   = {DIM(NULL, NULL, NULL)};
  ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &error_v_minus_np1_on_cells);  CHKERRXX(ierr);
  ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &error_v_plus_np1_on_cells);   CHKERRXX(ierr);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostFaces(p4est, faces, &error_v_minus_np1_faces[dim], dim);     CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est, faces, &error_v_plus_np1_faces[dim], dim);      CHKERRXX(ierr);
    ierr = VecGetArray(error_v_minus_np1_faces[dim],  &error_v_minus_np1_faces_p[dim]);     CHKERRXX(ierr);
    ierr = VecGetArray(error_v_plus_np1_faces[dim],   &error_v_plus_np1_faces_p[dim]);      CHKERRXX(ierr);
    ierr = VecGetArrayRead(solver->get_vnp1_face_minus()[dim], &v_minus_np1_p[dim]);  CHKERRXX(ierr);
    ierr = VecGetArrayRead(solver->get_vnp1_face_plus()[dim], &v_plus_np1_p[dim]);    CHKERRXX(ierr);
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dim]; ++f_idx) {
      faces->xyz_fr_f(f_idx, dim, xyz);
      error_v_minus_np1_faces_p[dim][f_idx] = fabs(test_problem->velocity_minus(dim, xyz) - v_minus_np1_p[dim][f_idx]);
      error_v_plus_np1_faces_p[dim][f_idx]  = fabs(test_problem->velocity_plus(dim, xyz) - v_plus_np1_p[dim][f_idx]);
    }
  }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(solver->get_vnp1_face_minus()[dim], &v_minus_np1_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(solver->get_vnp1_face_plus()[dim], &v_plus_np1_p[dim]); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(error_v_minus_np1_faces[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(error_v_plus_np1_faces[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(error_v_minus_np1_faces[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(error_v_plus_np1_faces[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(error_v_minus_np1_faces[dim],  &error_v_minus_np1_faces_p[dim]);     CHKERRXX(ierr);
    ierr = VecRestoreArray(error_v_plus_np1_faces[dim],   &error_v_plus_np1_faces_p[dim]);      CHKERRXX(ierr);
  }
  std::vector<const Vec*> face_errors;          face_errors.push_back(error_v_minus_np1_faces);             face_errors.push_back(error_v_plus_np1_faces);
  std::vector<Vec>        face_errors_on_cells; face_errors_on_cells.push_back(error_v_minus_np1_on_cells); face_errors_on_cells.push_back(error_v_plus_np1_on_cells);
  solver->transfer_face_sampled_fields_to_cells(face_errors, face_errors_on_cells);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = delete_and_nullify_vector(error_v_minus_np1_faces[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(error_v_plus_np1_faces[dim]); CHKERRXX(ierr);
  }

  vector<Vec_for_vtk_export_t> node_scalar_fields;
  vector<Vec_for_vtk_export_t> node_vector_fields;
  vector<Vec_for_vtk_export_t> cell_scalar_fields;
  vector<Vec_for_vtk_export_t> cell_vector_fields;
  if(interface_manager->get_phi_on_computational_nodes() != NULL)
  {
    node_scalar_fields.push_back(Vec_for_vtk_export_t(interface_manager->get_phi_on_computational_nodes(), "phi"));
  }
  node_vector_fields.push_back(Vec_for_vtk_export_t(error_v_minus_np1_nodes, "error_v_minus_np1"));
  node_vector_fields.push_back(Vec_for_vtk_export_t(error_v_plus_np1_nodes, "error_v_plus_np1"));
  node_vector_fields.push_back(Vec_for_vtk_export_t(error_v_sharp_np1_nodes, "error_v_sharp_np1"));
  cell_scalar_fields.push_back(Vec_for_vtk_export_t(error_pressure_minus, "error_p_minus"));
  cell_scalar_fields.push_back(Vec_for_vtk_export_t(error_pressure_plus, "error_p_plus"));
  cell_scalar_fields.push_back(Vec_for_vtk_export_t(error_sharp_pressure, "error_p_sharp"));
  cell_vector_fields.push_back(Vec_for_vtk_export_t(error_v_minus_np1_on_cells, "error_v_minus"));
  cell_vector_fields.push_back(Vec_for_vtk_export_t(error_v_plus_np1_on_cells, "error_v_plus"));

  string file_name = vtk_dir + "/error_visualization";
  my_p4est_vtk_write_all_general_lists(p4est, nodes, ghost, P4EST_FALSE, P4EST_FALSE, file_name.c_str(),
                                       &node_scalar_fields, &node_vector_fields, &cell_scalar_fields, &cell_vector_fields);
  node_scalar_fields.clear();
  node_vector_fields.clear();
  cell_scalar_fields.clear();
  cell_vector_fields.clear();


  ierr = delete_and_nullify_vector(error_v_minus_np1_nodes); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_v_plus_np1_nodes); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_v_sharp_np1_nodes); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_pressure_minus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_pressure_plus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_sharp_pressure); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_v_minus_np1_on_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(error_v_plus_np1_on_cells); CHKERRXX(ierr);
  return;
}

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd, test_case_for_two_phase_flows_t& test_case,
                                my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity,
                                Vec& interface_force, Vec& nonconstant_surface_tension, Vec& mass_flux)
{
  PetscErrorCode ierr;
  const int lmin                        = cmd.get<int>    ("lmin",   default_lmin);
  const int lmax                        = cmd.get<int>    ("lmax",   default_lmax);
  const double vorticity_threshold      = cmd.get<double> ("thresh", default_vorticity_threshold);
  const int ntree_xyz[P4EST_DIM]        = {DIM(cmd.get<int>("ntree_x", default_ntree[0]),
                                           cmd.get<int>("ntree_y", default_ntree[1]),
                                           cmd.get<int>("ntree_z", default_ntree[2]))};
  const double xyz_min[P4EST_DIM]       = { DIM(test_case.get_domain().xyz_min[0],      test_case.get_domain().xyz_min[1],      test_case.get_domain().xyz_min[2])};
  const double xyz_max[P4EST_DIM]       = { DIM(test_case.get_domain().xyz_max[0],      test_case.get_domain().xyz_max[1],      test_case.get_domain().xyz_max[2])};
  const int periodic[P4EST_DIM]         = { DIM(test_case.get_domain().periodicity[0],  test_case.get_domain().periodicity[1],  test_case.get_domain().periodicity[2])};
  const double uniform_band_in_dxmin    = cmd.get<double> ("uniform_band",      default_uniform_band_in_dx_min);
  const double rho_plus                 = test_case.get_rho_plus();
  const double rho_minus                = test_case.get_rho_minus();
  const double mu_plus                  = test_case.get_mu_plus();
  const double mu_minus                 = test_case.get_mu_minus();
  const double surface_tension          = test_case.get_surface_tension();
  const bool use_second_order_theta     = cmd.get<bool>   ("second_order_ls",   default_use_second_order_theta);
  const int sl_order                    = cmd.get<int>    ("sl_order",          default_sl_order);
  const int sl_order_interface          = cmd.get<int>    ("sl_order_itfc",     default_sl_order_itfc);
  const int nviscous_subiter            = cmd.get<int>    ("nviscous_subiter",  default_nviscous_subiter);
  const double cfl_advection            = cmd.get<double> ("cfl_advection",     default_cfl_advection);
  const double cfl_visco_capillary      = cmd.get<double> ("cfl_visco_capillary",default_cfl_visco_capillary);
  const double cfl_capillary            = cmd.get<double> ("cfl_capillary",     default_cfl_capillary);
  const string root_export_folder       = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR")));
  const jump_solver_tag cell_solver_to_use  = cmd.get<jump_solver_tag>("cell_solver", default_cell_solver);
  const jump_solver_tag face_solver_to_use  = cmd.get<jump_solver_tag>("face_solver", default_face_solver);

  const interpolation_method phi_interp = cmd.get<interpolation_method>("phi_interp", default_interp_method_phi);
  const bool use_subrefinement          = cmd.get<bool>("subrefinement", default_subrefinement);

  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    if(brick->nxyz_to_treeid != NULL)
    {
      P4EST_FREE(brick->nxyz_to_treeid);
      brick->nxyz_to_treeid = NULL;
    }
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;
  if(connectivity != NULL)
  {
    p4est_connectivity_destroy(connectivity); connectivity = NULL;
  }
  connectivity = my_p4est_brick_new(ntree_xyz, xyz_min, xyz_max, brick, periodic);

  const double dx_min = MIN(DIM(test_case.get_domain().length()/ntree_xyz[0], test_case.get_domain().height()/ntree_xyz[1], test_case.get_domain().width()/ntree_xyz[2]))/((double) (1 << lmax));
  const double dt_0   = test_case.compute_dt_0(cfl_advection, cfl_visco_capillary, cfl_capillary, dx_min);
  test_case.set_time(dt_0); // we need to evaluate the levelset in np1!

  splitting_criteria_cf_and_uniform_band_t* data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, test_case.get_levelset(), uniform_band_in_dxmin);
  p4est_t                       *p4est_nm1      = NULL, *p4est_n      = NULL, *subrefined_p4est     = NULL;
  p4est_ghost_t                 *ghost_nm1      = NULL, *ghost_n      = NULL, *subrefined_ghost     = NULL;
  p4est_nodes_t                 *nodes_nm1      = NULL, *nodes_n      = NULL, *subrefined_nodes     = NULL;
  my_p4est_hierarchy_t          *hierarchy_nm1  = NULL, *hierarchy_n  = NULL, *subrefined_hierarchy = NULL;
  my_p4est_node_neighbors_t     *ngbd_nm1       = NULL, *ngbd_n       = NULL, *subrefined_ngbd_n    = NULL;
  Vec                                                    phi_np1      = NULL,  subrefined_phi_np1   = NULL;
  my_p4est_cell_neighbors_t                             *ngbd_c       = NULL;
  my_p4est_faces_t                                      *faces        = NULL;

  // we don't know the time step yet, we'll have to correct the definition of phi_np1 after the solver can actually compute the time step
  // (we use the same 'data' for n and nm1 grids, shouldn't be a big deal...)
  my_p4est_two_phase_flows_t::build_initial_computational_grids(mpi, brick, connectivity,
                                                                data, data,
                                                                p4est_nm1, ghost_nm1, nodes_nm1, hierarchy_nm1, ngbd_nm1,
                                                                p4est_n, ghost_n, nodes_n, hierarchy_n, ngbd_n,
                                                                ngbd_c, faces, phi_np1, test_case.is_reinitialization_needed());
  Vec interface_capturing_phi_np1 = phi_np1; // no creation here, just a renamed pointer to streamline the logic

  if(use_subrefinement)
  {
    // build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
    splitting_criteria_cf_t* subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, test_case.get_levelset());
    my_p4est_two_phase_flows_t::build_initial_interface_capturing_grid(p4est_n, brick, subrefined_data,
                                                                       subrefined_p4est, subrefined_ghost, subrefined_nodes, subrefined_hierarchy, subrefined_ngbd_n,
                                                                       subrefined_phi_np1);
    interface_capturing_phi_np1 = subrefined_phi_np1;
  }

  if(test_case.is_reinitialization_needed())
  {
    my_p4est_level_set_t ls((use_subrefinement ? subrefined_ngbd_n : ngbd_n));
    ls.reinitialize_2nd_order((use_subrefinement ? subrefined_phi_np1 : phi_np1));
    if(use_subrefinement)
    {
      my_p4est_interpolation_nodes_t interp_phi(subrefined_ngbd_n);
      interp_phi.set_input(subrefined_phi_np1, phi_interp);
      for (p4est_locidx_t node_idx = 0; node_idx < nodes_n->num_owned_indeps; ++node_idx) {
        double xyz_node[P4EST_DIM];
        node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
        interp_phi.add_point(node_idx, xyz_node);
      }
      interp_phi.interpolate(phi_np1);
      ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
    }
  }

  if(solver != NULL)
    delete solver;
  solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces, (use_subrefinement ? subrefined_ngbd_n : NULL));
  solver->set_phi_np1(interface_capturing_phi_np1, phi_interp, phi_np1);
  my_p4est_interface_manager_t* interface_manager = solver->get_interface_manager();

  solver->set_dynamic_viscosities(mu_minus, mu_plus);
  solver->set_densities(rho_minus, rho_plus);
  ierr = delete_and_nullify_vector(nonconstant_surface_tension);
  if(test_case.is_surface_tension_constant())
    solver->set_surface_tension(surface_tension);
  else
  {
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(nonconstant_surface_tension); CHKERRXX(ierr);
    test_case.sample_variable_surface_tension(interface_manager, nonconstant_surface_tension);
    solver->set_surface_tension(nonconstant_surface_tension);
  }
  solver->set_uniform_bands(uniform_band_in_dxmin, uniform_band_in_dxmin);
  solver->set_vorticity_split_threshold(vorticity_threshold);
  solver->set_cfls(cfl_advection, cfl_visco_capillary, cfl_capillary);
  solver->set_semi_lagrangian_order_advection(sl_order);
  solver->set_semi_lagrangian_order_interface(sl_order_interface);
  solver->set_n_viscous_subiterations(nviscous_subiter);
  solver->set_cell_jump_solver(cell_solver_to_use);
  solver->set_face_jump_solvers(face_solver_to_use);
  if(use_second_order_theta)
    solver->fetch_interface_points_with_second_order_accuracy();

  ierr = delete_and_nullify_vector(mass_flux); CHKERRXX(ierr);
  if(test_case.with_mass_flux())
  {
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(mass_flux); CHKERRXX(ierr);
    test_case.sample_mass_flux(interface_manager, mass_flux);
    solver->set_mass_flux(mass_flux);
  }

  ierr = delete_and_nullify_vector(interface_force); CHKERRXX(ierr);
  ierr = interface_manager->create_vector_on_interface_capturing_nodes(interface_force, P4EST_DIM); CHKERRXX(ierr);
  test_case.sample_interface_stress_source(interface_manager, interface_force);
  solver->set_interface_force(interface_force);

  solver->set_dt(dt_0, dt_0); // we manually initialize the time steps in this case

  // set node velocities at tnm1 and tn
  const CF_DIM *vminus_functors[P4EST_DIM], *vplus_functors[P4EST_DIM];
  test_case.get_velocity_functors(vminus_functors, vplus_functors);
  test_case.set_time(-dt_0); // to evaluate vnm1
  solver->set_node_velocities_nm1(vminus_functors, vplus_functors);
  test_case.set_time(0.0); // to evaluate vn
  solver->set_node_velocities_n(vminus_functors, vplus_functors);
  test_case.set_time(dt_0); // to get ready!


  ierr = PetscPrintf(solver->get_p4est_n()->mpicomm, "Successfully created and initialized the solver\n"); CHKERRXX(ierr);

  return;
}

void load_solver_from_state(const mpi_environment_t &mpi, const cmdParser &cmd, test_case_for_two_phase_flows_t& test_case,
                            my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity,
                            Vec& interface_force, Vec& nonconstant_surface_tension, Vec& mass_flux)
{
  const string backup_directory = cmd.get<string>("restart", "");
  if(!is_folder(backup_directory.c_str()))
    throw std::invalid_argument("load_solver_from_state: the restart path " + backup_directory + " is not an accessible directory.");

  if (solver != NULL) {
    delete solver; solver = NULL; }
  P4EST_ASSERT(solver == NULL);
  solver = new my_p4est_two_phase_flows_t(mpi, backup_directory.c_str());

  if (brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }

  P4EST_ASSERT(brick == NULL);
  brick = solver->get_brick();
  if(connectivity != NULL){
    p4est_connectivity_destroy(connectivity); connectivity = NULL; }
  connectivity = solver->get_connetivity();

  test_case.set_time(solver->get_tnp1());
  PetscErrorCode ierr;
  my_p4est_interface_manager_t *interface_manager = solver->get_interface_manager();

  ierr = delete_and_nullify_vector(nonconstant_surface_tension);
  if(!test_case.is_surface_tension_constant())
  {
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(nonconstant_surface_tension); CHKERRXX(ierr);
    test_case.sample_variable_surface_tension(interface_manager, nonconstant_surface_tension);
    solver->set_surface_tension(nonconstant_surface_tension);
  }
  ierr = delete_and_nullify_vector(mass_flux); CHKERRXX(ierr);
  if(test_case.with_mass_flux())
  {
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(mass_flux); CHKERRXX(ierr);
    test_case.sample_mass_flux(interface_manager, mass_flux);
    solver->set_mass_flux(mass_flux);
  }

  ierr = delete_and_nullify_vector(interface_force); CHKERRXX(ierr);
  ierr = interface_manager->create_vector_on_interface_capturing_nodes(interface_force, P4EST_DIM); CHKERRXX(ierr);
  test_case.sample_interface_stress_source(interface_manager, interface_force);
  solver->set_interface_force(interface_force);

  ierr = PetscPrintf(solver->get_p4est_n()->mpicomm, "Simulation restarted from state saved in %s\n", (cmd.get<std::string>("restart")).c_str()); CHKERRXX(ierr);

  return;
}

int vtk_index(const int& vtk_start, const my_p4est_two_phase_flows_t* solver, const double& vtk_dt)
{
  return vtk_start + int(floor(solver->get_progress_np1()/vtk_dt));
}
int backup_index(const my_p4est_two_phase_flows_t* solver, const double& backup_dt)
{
  return int(floor(solver->get_progress_n()/backup_dt));
}

string subdirectory_inteprolation_ls_name(my_p4est_two_phase_flows_t* solver)
{
  const interpolation_method interpolation_ls = solver->get_interface_manager()->get_interp_phi().get_interpolation_method();
  switch (interpolation_ls) {
  case linear:
    return "linear";
    break;
  case quadratic:
    return  "quadratic";
    break;
  case quadratic_non_oscillatory:
    return  "quadratic_non_oscillatory";
    break;
  case quadratic_non_oscillatory_continuous_v1:
    return "quadratic_non_oscillatory_v1";
    break;
  case quadratic_non_oscillatory_continuous_v2:
    return "quadratic_non_oscillatory_v2";
    break;
  default:
    throw std::runtime_error("main for two-phase flow validation::subdirectory_inteprolation_ls_name unknonw interpolation method for the levelset function");
    break;
  }
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  ostringstream streamObj;
  // test number
  cmd.add_option("test", "Test problem to choose. Available choices are (default test number is " + to_string(default_test) +"): \n" + list_of_test_problems_for_two_phase_flows.get_description_of_tests() + "\n");
  // computational grid parameters
  cmd.add_option("lmin", "min level of the trees, default is " + to_string(default_lmin));
  cmd.add_option("lmax", "max level of the trees, default is " + to_string(default_lmax));
  streamObj.str(""); streamObj << default_vorticity_threshold;
  cmd.add_option("thresh", "the vorticity-based threshold used for the refinement criteria, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_uniform_band_in_dx_min;
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx, default is " + streamObj.str());
  cmd.add_option("ntree_x", "number of trees in the macromesh, along the x-direction. The default value is " + to_string(default_ntree[0]));
  cmd.add_option("ntree_y", "number of trees in the macromesh, along the y-direction. The default value is " + to_string(default_ntree[1]));
#ifdef P4_TO_P8
  cmd.add_option("ntree_z", "number of trees in the macromesh, along the z-direction. The default value is " + to_string(default_ntree[2]));
#endif
  // method-related parameters
  cmd.add_option("second_order_ls", "flag activating second order F-D interface fetching if set to true or 1. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("sl_order", "the order for the semi lagrangian advection terms, either 1 or 2, default is " + to_string(default_sl_order));
  cmd.add_option("sl_order_itfc", "the order for the semi lagrangian interface advection, either 1 or 2, default is " + to_string(default_sl_order_itfc));
  cmd.add_option("nviscous_subiter", "the max number of subiterations for viscous solver, default is " + to_string(default_nviscous_subiter));
  streamObj.str(""); streamObj << default_cfl_advection;
  cmd.add_option("cfl_advection", "desired advection CFL number, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_cfl_visco_capillary;
  cmd.add_option("cfl_visco_capillary", "desired visco-capillary CFL number, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_cfl_capillary;
  cmd.add_option("cfl_capillary", "desired capillary-wave CFL number, default is " + streamObj.str());
  cmd.add_option("cell_solver", "cell-based solver to use for projection step and pressure guess, possible choices are 'GFM', 'xGFM' or 'FV'. Default is " + convert_to_string(default_cell_solver));
  cmd.add_option("face_solver", "face-based solver to use for viscosity step. Default is " + convert_to_string(default_face_solver));
  streamObj.str(""); streamObj << default_n_reinit;
  cmd.add_option("n_reinit", "number of solver iterations between two reinitializations of the levelset. Default is " + streamObj.str());
  streamObj.str(""); streamObj << default_vmax_abort;
  cmd.add_option("vmax_abort", "maximum velocity tolerated (the solver aborts if the local velocity exceeds this value at any point). Default is " + streamObj.str());
  streamObj.str(""); streamObj << default_projection_threshold;
  cmd.add_option("projection_threshold", "threshold for convergence of inner criterion (inner loop terminates if (max projection correction)/(max velocity component before projection) is below this value). Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_niter;
  cmd.add_option("niter", "max number of fix-point iterations for every time step. Default value is " + streamObj.str());
  // output-control parameters
  cmd.add_option("save_vtk", "flag activating  the exportation of vtk visualization files if set to true or 1. Default behavior is " + string(default_save_vtk ? "with" : "without") + " vtk exportation");
  streamObj.str(""); streamObj << default_vtk_dt;
  cmd.add_option("vtk_dt", "vtk_dt = time step between two vtk exportation (nondimendional), default is " + streamObj.str());
  streamObj.str(""); streamObj << default_vtk_idx_start;
  cmd.add_option("vtk_idx_start", "first desired index of exported vtk files, default is " + streamObj.str());
  cmd.add_option("work_dir", "root exportation directory, subfolders will be created therein (read from input if not defined otherwise in the environment variable OUT_DIR). \n\tThis is required for vtk files and for data files. Default is " + default_work_folder);
  streamObj.str(""); streamObj << default_interp_method_phi;
  cmd.add_option("phi_interp", "interpolation method for the node-sampled levelset function. Default is " + streamObj.str());
  cmd.add_option("subrefinement", "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");
  streamObj.str(""); streamObj << default_save_state_dt;
  cmd.add_option("save_state_dt", "if save_nstates > 0, the solver state is saved every save_state_dt*(D*mu_plus/gamma) time increments in backup_ subfolders. Default is " + streamObj.str());
  cmd.add_option("save_nstates",  "determines how many solver states must be memorized in backup_ folders (default is " + to_string(default_save_nstates) + ")");


  if(cmd.parse(argc, argv, main_description))
    return EXIT_SUCCESS;

  const string root_export_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR")));
  const int niter_reinit = cmd.get<int> ("n_reinit", default_n_reinit);

  PetscErrorCode ierr;
  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
  my_p4est_brick_t *brick                           = NULL;
  p4est_connectivity_t* connectivity                = NULL;

  const int test_number = cmd.get<int>("test", default_test);
  test_case_for_two_phase_flows_t *test_problem = list_of_test_problems_for_two_phase_flows[test_number];
  Vec non_constant_surface_tension  = NULL;
  Vec mass_flux                     = NULL;
  Vec interface_force               = NULL;
  if(cmd.contains("restart"))
    load_solver_from_state(mpi, cmd, *test_problem, two_phase_flow_solver, brick, connectivity, interface_force, non_constant_surface_tension, mass_flux);
  else
    create_solver_from_scratch(mpi, cmd, *test_problem, two_phase_flow_solver, brick, connectivity, interface_force, non_constant_surface_tension, mass_flux);

  two_phase_flow_solver->set_static_interface(test_problem->is_interface_static());
  two_phase_flow_solver->set_final_time(test_problem->get_final_time());

  CF_DIM *body_force_per_unit_mass_minus[P4EST_DIM], *body_force_per_unit_mass_plus[P4EST_DIM];
  test_problem->get_force_per_unit_mass_minus(body_force_per_unit_mass_minus);
  test_problem->get_force_per_unit_mass_plus(body_force_per_unit_mass_plus);
  two_phase_flow_solver->set_bc(test_problem->get_velocity_wall_bc(), &test_problem->get_pressure_wall_bc());
  two_phase_flow_solver->set_external_forces_per_unit_mass(body_force_per_unit_mass_minus, body_force_per_unit_mass_plus);

  splitting_criteria_t* data            = (splitting_criteria_t*) (two_phase_flow_solver->get_p4est_n()->user_pointer); // to delete it appropriately, eventually
  splitting_criteria_t* subrefined_data = (two_phase_flow_solver->get_fine_p4est_n() != NULL ? (splitting_criteria_t*) two_phase_flow_solver->get_fine_p4est_n()->user_pointer : NULL); // same, to delete it appropriately, eventually
  const bool save_vtk     = cmd.get<bool>   ("save_vtk",      default_save_vtk);
  const double vtk_dt     = cmd.get<double> ("vtk_dt",        default_vtk_dt);
  if(vtk_dt <= 0.0)
    throw std::invalid_argument("main for two-phase flow validation: the value of vtk_dt must be strictly positive.");
  const int save_nstates      = cmd.get<int>    ("save_nstates",  default_save_nstates);
  const double save_state_dt  = cmd.get<double> ("save_state_dt", default_save_state_dt);


  const string export_dir   = root_export_folder + (root_export_folder.back() == '/' ? "" : "/") + test_problem->get_name() + (subrefined_data != NULL ? "/subresolved/" : "/regular/") +
      subdirectory_inteprolation_ls_name(two_phase_flow_solver) + "/lmin_" + to_string(data->min_lvl) + "_lmax_" + to_string(data->max_lvl);
  const string vtk_dir      = export_dir + "/vtu";
  if(create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for two-phase flow validation: could not create exportation directory " + export_dir);
  if(save_vtk && create_directory(vtk_dir, mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for two-phase flow validation: could not create directory for visualization files, i.e., " + vtk_dir);

  const int vtk_start     = cmd.get<int>("vtk_idx_start", default_vtk_idx_start);
  const double vmax_abort = cmd.get<double>("vmax_abort", default_vmax_abort);
  const double projection_threshold = cmd.get<double>("projection_threshold", default_projection_threshold);
  const int n_fixpoint_iter_max = cmd.get<int>("niter", default_niter);
  int vtk_idx     = vtk_index(vtk_start, two_phase_flow_solver, vtk_dt) - 1; // -1 so that we do not miss the very first snapshot
  int backup_idx  = backup_index(two_phase_flow_solver, save_nstates);

  bool advance_solver = false;

  while(two_phase_flow_solver->get_tnp1() < test_problem->get_final_time() - 0.001*two_phase_flow_solver->get_dt_n()) // 0.005*dt_n as threshold for comparison of doubles...
  {
    if(advance_solver)
    {
      two_phase_flow_solver->update_from_tn_to_tnp1(niter_reinit);
      test_problem->set_time(two_phase_flow_solver->get_tnp1());
      ierr = delete_and_nullify_vector(interface_force); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(non_constant_surface_tension); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(mass_flux); CHKERRXX(ierr);
      my_p4est_interface_manager_t* interface_manager = two_phase_flow_solver->get_interface_manager();
      ierr = interface_manager->create_vector_on_interface_capturing_nodes(interface_force, P4EST_DIM); CHKERRXX(ierr);
      test_problem->sample_interface_stress_source(interface_manager, interface_force);
      two_phase_flow_solver->set_interface_force(interface_force);
      if(!test_problem->is_surface_tension_constant())
      {
        ierr = interface_manager->create_vector_on_interface_capturing_nodes(non_constant_surface_tension, 1); CHKERRXX(ierr);
        test_problem->sample_variable_surface_tension(interface_manager, non_constant_surface_tension);
        two_phase_flow_solver->set_surface_tension(non_constant_surface_tension);
      }
      if(test_problem->with_mass_flux())
      {
        ierr = interface_manager->create_vector_on_interface_capturing_nodes(mass_flux, 1); CHKERRXX(ierr);
        test_problem->sample_variable_surface_tension(interface_manager, mass_flux);
        two_phase_flow_solver->set_mass_flux(mass_flux);
      }
    }

    if(save_nstates > 0 && backup_index(two_phase_flow_solver, save_state_dt) != backup_idx)
    {
      backup_idx = backup_index(two_phase_flow_solver, save_state_dt);
      two_phase_flow_solver->save_state(export_dir.c_str(), save_nstates);
    }

    two_phase_flow_solver->solve_time_step(projection_threshold, n_fixpoint_iter_max);

    if(save_vtk && vtk_idx != vtk_index(vtk_start, two_phase_flow_solver, vtk_dt))
    {
      vtk_idx = vtk_index(vtk_start, two_phase_flow_solver, vtk_dt);
      two_phase_flow_solver->save_vtk(vtk_dir, vtk_idx, true);
    }

    if(two_phase_flow_solver->get_max_velocity() > vmax_abort)
    {
      if(save_vtk)
        two_phase_flow_solver->save_vtk(vtk_dir, ++vtk_idx, true);
      ierr = PetscPrintf(mpi.comm(), "The maximum velocity of %g exceeded the tolerated threshold of %g... \n", two_phase_flow_solver->get_max_velocity(), vmax_abort); CHKERRXX(ierr);

      delete two_phase_flow_solver;
      my_p4est_brick_destroy(connectivity, brick);
      delete brick;
      delete data;
      delete subrefined_data;

      ierr = delete_and_nullify_vector(interface_force); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(non_constant_surface_tension); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(mass_flux); CHKERRXX(ierr);

      return EXIT_FAILURE;
    }
    advance_solver = true;
  }
  ierr = PetscPrintf(mpi.comm(), "Gracefully finishing up now\n");
  if(save_vtk)
    export_error_visualization(vtk_dir, two_phase_flow_solver, test_problem);
  print_convergence_results(export_dir, two_phase_flow_solver, test_problem);

  delete two_phase_flow_solver;
  my_p4est_brick_destroy(connectivity, brick);
  delete brick;
  delete data;
  delete subrefined_data;

  ierr = delete_and_nullify_vector(interface_force); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(non_constant_surface_tension); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(mass_flux); CHKERRXX(ierr);

  return EXIT_SUCCESS;
}
