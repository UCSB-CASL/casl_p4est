#ifdef P4_TO_P8
#include <src/my_p8est_two_phase_flows.h>
#else
#include <src/my_p4est_two_phase_flows.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

const static string main_description =
    string("Shape oscillations of a bubble/droplet (as studied by Lamb, Prosperetti, Miller-Scriven, etc.)! \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), 2019-2020-2021-...-2523\n");

// problem setup: 1) set parameters
const double R0                       = 1.0;      // those are constant, not freely defined
const double rho_plus                 = 0.001;    // those are constant, not freely defined
const double mu_plus                  = 0.00002;  // those are constant, not freely defined
const double radius_epsilon           = 0.01;     // those are constant, not freely defined
// problem setup: 2) free quantities
const double default_domain[P4EST_DIM] = {DIM(8.0, 8.0, 8.0)};
const bool default_periodic[P4EST_DIM] = {DIM(false, false, false)};
const double default_ratio_rho  = 0.001;   // rho_plus/rho_minus
const double default_ratio_mu   = 0.001;   // mu_plus/mu_minus
const double default_Reynolds   = 35.5;    // Reynolds = sqrt(gamma*R0*rho_minus)/mu_minus
// grid-related
const int default_lmin = 4;
const int default_lmax = 7;
const double default_vorticity_threshold      = 0.02;
const double default_uniform_band_to_radius   = 0.2;
const int default_ntree[P4EST_DIM] = {DIM(1, 1, 1)};
// simulation-related:
const interpolation_method default_interp_method_phi = quadratic_non_oscillatory_continuous_v2;
const bool default_subrefinement = false;
const bool default_use_second_order_theta = (default_interp_method_phi == linear ? false : true);
const int default_nviscous_subiter  = 10;
const int default_niter             = 3;
const int default_sl_order          = 2;
const int default_sl_order_itfc     = 2;
const double default_cfl_advection        = 1.0;
const double default_cfl_visco_capillary  = 0.95;
const double default_cfl_capillary        = 0.95;
const jump_solver_tag default_cell_solver = FV;
const jump_solver_tag default_face_solver = xGFM;
const int default_n_reinit = 1;
#ifdef P4_TO_P8
const double default_t_end = 2.0;
#else
const double default_t_end = 4.0;
#endif
const double default_vmax_abort = 1000.0;
const double default_projection_threshold = 0.01;
// exportation-related
const bool default_save_vtk = true;
const double default_vtk_dt     = 0.05;
const int default_vtk_idx_start = 0;
const int default_save_nstates  = 0;
const double default_save_state_dt  = 0.1;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_flow/bubble_oscillations/" + to_string(P4EST_DIM) + "D";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_flow/bubble_oscillations/" + to_string(P4EST_DIM) + "D";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_flow/bubble_oscillations/" + to_string(P4EST_DIM) + "D";
#endif

class initial_level_set_t: public CF_DIM {
  inline double theta(DIM(double x, double y, double z)) const
  {
    const double rr = ABSD(x, y, z);
    if(rr > radius_epsilon*R0)
      return asin(sqrt(SQR(x) ONLY3D( + SQR(z)))/rr);
    else
       return 0.0;
  }
public:
  initial_level_set_t() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(SUMD(SQR(x), SQR(y), SQR(z))) - R0*(1.0 + radius_epsilon*0.5*(3*SQR(cos(theta(DIM(x, y, z)))) - 1));
  }
} initial_level_set;

void initialize_exportations(const double &tstart, const mpi_environment_t& mpi, const string& results_dir, string& datafile)
{
  datafile = results_dir + "/monitoring_results.dat";
  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(), "Saving raw results in ... %s\n", datafile.c_str()); CHKERRXX(ierr);

  if(mpi.rank() == 0)
  {
    if(!file_exists(datafile))
    {
      FILE* fp_results = fopen(datafile.c_str(), "w");
      if(fp_results == NULL)
        throw std::runtime_error("initialize_exportations: could not open file for output of raw results.");
      fprintf(fp_results, "%% tn | RR | R0 | a_2 (variable R) | a_2 (R = R0) | volume | dt/dt_visc | dt/dt_capillar \n");
      fclose(fp_results);
    }
    else
      truncate_exportation_file_up_to_tstart(tstart, datafile, 5);

    char liveplot[PATH_MAX];
    sprintf(liveplot, "%s/live_monitor.gnu", results_dir.c_str());
    if(!file_exists(liveplot))
    {
      FILE* fp_liveplot = fopen(liveplot, "w");
      if(fp_liveplot == NULL)
        throw std::runtime_error("initialize_exportations: could not open file for liveplot.");
      fprintf(fp_liveplot, "set term wxt 0 position 0,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"RR \" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot \"monitoring_results.dat\" using 1:2 title 'variable R' with lines lw 3, \\\n");
      fprintf(fp_liveplot, "\t \"monitoring_results.dat\" using 1:3 title 'R = R0'  with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 1 position 800,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"a_2 \" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot \"monitoring_results.dat\" using 1:4 title 'variable R' with lines lw 3, \\\n");
      fprintf(fp_liveplot, "\t \"monitoring_results.dat\" using 1:5 title 'R = R0' with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 2 position 1600,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Relative difference in bubble volume (in %%)\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "col = 6\n");
      fprintf(fp_liveplot, "row = 1\n");
      fprintf(fp_liveplot, "stats 'monitoring_results.dat' every ::row::row using col nooutput\n");
      fprintf(fp_liveplot, "original_volume = STATS_min\n");
      fprintf(fp_liveplot, "plot  \"monitoring_results.dat\" using 1:(100 * ($6 - original_volume)/original_volume) notitle with lines lw 3\n");
      fprintf(fp_liveplot, "reread");
      fclose(fp_liveplot);
    }
  }
  return;
}

void export_results(const double& tn, const double& RR, const double& RR0, const double& a_2_RR, const double& a_2_RR0, const double& bubble_volume,
                    const double& dt_to_dt_visc, const double& dt_to_dt_capillary, const string& datafile)
{
  FILE* fp_data = fopen(datafile.c_str(), "a");
  if(fp_data == NULL)
    throw std::invalid_argument("main for bubble oscillations: could not open file for output of raw results.");
  fprintf(fp_data, "%g %g %g %g %g %g %g %g \n", tn, RR, RR0, a_2_RR, a_2_RR0, bubble_volume, dt_to_dt_visc, dt_to_dt_capillary);
  fclose(fp_data);
}

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd,
                                my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity)
{
  const int lmin                        = cmd.get<int>    ("lmin",   default_lmin);
  const int lmax                        = cmd.get<int>    ("lmax",   default_lmax);
  const double vorticity_threshold      = cmd.get<double> ("thresh", default_vorticity_threshold);
  const int ntree_xyz[P4EST_DIM]        = {DIM(cmd.get<int>("ntree_x", default_ntree[0]),
                                           cmd.get<int>("ntree_y", default_ntree[1]),
                                           cmd.get<int>("ntree_z", default_ntree[2]))};
  const double domain_size[P4EST_DIM]   = {DIM(cmd.get<double>("length", default_domain[0]),
                                           cmd.get<double>("height", default_domain[1]),
                                           cmd.get<double>("width", default_domain[2]))};
  const double xyz_min[P4EST_DIM]       = { DIM(-0.5*domain_size[0], -0.5*domain_size[1], -0.5*domain_size[2]) };
  const double xyz_max[P4EST_DIM]       = { DIM( 0.5*domain_size[0],  0.5*domain_size[1],  0.5*domain_size[2]) };
  const int periodic[P4EST_DIM]         = { DIM(cmd.get<bool>("xperiodic", default_periodic[0]),
                                            cmd.get<bool>("yperiodic", default_periodic[1]),
                                            cmd.get<bool>("zperiodic", default_periodic[2]))};
  const double dxmin                    = MIN(DIM(domain_size[0]/ntree_xyz[0], domain_size[1]/ntree_xyz[1], domain_size[2]/ntree_xyz[2]))/((double) (1 << lmax));
  const double uniform_band_in_dxmin    = cmd.get<double> ("uniform_band",      default_uniform_band_to_radius*R0/dxmin);
  const double rho_minus                = rho_plus/cmd.get<double>("ratio_rho", default_ratio_rho);
  const double mu_minus                 = mu_plus/cmd.get<double> ("ratio_mu",  default_ratio_mu);
  const double Reynolds                 = cmd.get<double> ("Reynolds", default_Reynolds);
  const double surface_tension          = SQR(mu_minus*Reynolds)/(rho_minus*R0);
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

  if(R0 > 0.5*MIN(DIM(domain_size[0], domain_size[1], domain_size[2])))
    throw std::invalid_argument("main for bubble oscillations: create_solver_from_scratch: invalid bubble radius (the bubble is larger than the computational domain in some direction).");

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
  splitting_criteria_cf_and_uniform_band_t* data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &initial_level_set, uniform_band_in_dxmin);
  p4est_t                       *p4est_nm1      = NULL, *p4est_n      = NULL, *subrefined_p4est     = NULL;
  p4est_ghost_t                 *ghost_nm1      = NULL, *ghost_n      = NULL, *subrefined_ghost     = NULL;
  p4est_nodes_t                 *nodes_nm1      = NULL, *nodes_n      = NULL, *subrefined_nodes     = NULL;
  my_p4est_hierarchy_t          *hierarchy_nm1  = NULL, *hierarchy_n  = NULL, *subrefined_hierarchy = NULL;
  my_p4est_node_neighbors_t     *ngbd_nm1       = NULL, *ngbd_n       = NULL, *subrefined_ngbd_n    = NULL;
  Vec                                                    phi          = NULL,  subrefined_phi       = NULL;
  my_p4est_cell_neighbors_t                             *ngbd_c       = NULL;
  my_p4est_faces_t                                      *faces        = NULL;

  my_p4est_two_phase_flows_t::build_initial_computational_grids(mpi, brick, connectivity,
                                                                data, data,
                                                                p4est_nm1, ghost_nm1, nodes_nm1, hierarchy_nm1, ngbd_nm1,
                                                                p4est_n, ghost_n, nodes_n, hierarchy_n, ngbd_n,
                                                                ngbd_c, faces, phi, true);
  Vec interface_capturing_phi = phi; // no creation here, just a renamed pointer to streamline the logic

  if(use_subrefinement)
  {
    // build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
    splitting_criteria_cf_t* subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, &initial_level_set);
    my_p4est_two_phase_flows_t::build_initial_interface_capturing_grid(p4est_n, brick, subrefined_data,
                                                                       subrefined_p4est, subrefined_ghost, subrefined_nodes, subrefined_hierarchy, subrefined_ngbd_n,
                                                                       subrefined_phi);
    interface_capturing_phi = subrefined_phi;
  }

  if(solver != NULL)
    delete solver;
  solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces, (use_subrefinement ? subrefined_ngbd_n : NULL));
  solver->set_phi_np1(interface_capturing_phi, phi_interp, phi);
  solver->set_dynamic_viscosities(mu_minus, mu_plus);
  solver->set_densities(rho_minus, rho_plus);
  solver->set_surface_tension(surface_tension);
  solver->set_uniform_bands(uniform_band_in_dxmin, uniform_band_in_dxmin);
  solver->set_vorticity_split_threshold(vorticity_threshold);
  solver->set_cfls(cfl_advection, cfl_visco_capillary, cfl_capillary);
  solver->set_semi_lagrangian_order_advection(sl_order);
  solver->set_semi_lagrangian_order_interface(sl_order_interface);
  solver->set_n_viscous_subiterations(nviscous_subiter);
  solver->initialize_time_steps();

  solver->set_cell_jump_solver(cell_solver_to_use);
  solver->set_face_jump_solvers(face_solver_to_use);
  if(use_second_order_theta)
    solver->fetch_interface_points_with_second_order_accuracy();
  return;
}

void load_solver_from_state(const mpi_environment_t &mpi, const cmdParser &cmd,
                            my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity)
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

  PetscErrorCode ierr = PetscPrintf(solver->get_p4est_n()->mpicomm, "Simulation restarted from state saved in %s\n", (cmd.get<std::string>("restart")).c_str()); CHKERRXX(ierr);
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

void get_normal_mode_decomposition_and_volume(const my_p4est_two_phase_flows_t* solver, double& RR, double& a_2_variable_RR, double& a_2_constant_R0, double& volume_inside)
{
  const my_p4est_interface_manager_t* interface_manager = solver->get_interface_manager();
  const p4est_t* p4est = solver->get_p4est_n();
  const p4est_ghost_t* ghost = solver->get_ghost_n();
  const p4est_nodes_t* nodes = solver->get_nodes_n();
  const my_p4est_node_neighbors_t* ngbd_n = solver->get_ngbd_n();
  const my_p4est_poisson_jump_cells_t* cell_jump_solver = solver->get_cell_jump_solver();
  const double* tree_dimensions = solver->get_faces_n()->get_tree_dimensions();
  const double* dxyz_min = solver->get_faces_n()->get_smallest_dxyz();

  vector<double> data_for_all_reduce(7, 0.0);
  // --> 0 : volume inside the bubble;
  // --> 1 : surface integral of (1);
  // --> 2 : surface integral of (P2(cos(theta)));
  // --> 3 : surface integral of SQR(P2(cos(theta)));
  // --> 4 : surface integral of (distance to (DIM(0, 0, 0)));
  // --> 5 : surface integral of (P2(cos(theta))*(distance to (DIM(0, 0, 0))));

  const bool fetch_fv_in_cell_solver = (cell_jump_solver != NULL && dynamic_cast<const my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver) != NULL
      && dynamic_cast<const my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->are_fv_and_cf_known());

  my_p4est_finite_volume_t fv;
  const my_p4est_finite_volume_t* const_fv_ptr = NULL;

  PetscErrorCode ierr;
  Vec phi_np1_on_computational_nodes = interface_manager->get_phi_on_computational_nodes();
  bool destroy_phi_comp = false;
  if(phi_np1_on_computational_nodes == NULL)
  {
    double xyz[P4EST_DIM];
    double *phi_np1_on_computational_nodes_p;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1_on_computational_nodes); CHKERRXX(ierr);
    ierr = VecGetArray(phi_np1_on_computational_nodes, &phi_np1_on_computational_nodes_p); CHKERRXX(ierr);
    for(size_t k = 0; k < ngbd_n->get_layer_size(); k++)
    {
      p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz);
      phi_np1_on_computational_nodes_p[node_idx] = interface_manager->phi_at_point(xyz);
    }
    ierr = VecGhostUpdateBegin(phi_np1_on_computational_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t k = 0; k < ngbd_n->get_local_size(); k++)
    {
      p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz);
      phi_np1_on_computational_nodes_p[node_idx] = interface_manager->phi_at_point(xyz);
    }
    ierr = VecGhostUpdateEnd(phi_np1_on_computational_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_np1_on_computational_nodes, &phi_np1_on_computational_nodes_p); CHKERRXX(ierr);
    destroy_phi_comp = true;
  }
  const double *phi_np1_on_computational_nodes_p;
  ierr = VecGetArrayRead(phi_np1_on_computational_nodes, &phi_np1_on_computational_nodes_p); CHKERRXX(ierr);


  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    const p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      const p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      double xyz_quad[P4EST_DIM];
      const double *tree_xyz_min, *tree_xyz_max;
      const p4est_quadrant_t* quad;
      fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est, ghost);
      xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad);

      bool crossed_cell = false;
      for(u_char vv = 1; vv < P4EST_CHILDREN; vv++)
        crossed_cell = crossed_cell || ((phi_np1_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + 0]] <= 0.0) != (phi_np1_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + vv]] <= 0.0));

      if(!crossed_cell && quad->level == interface_manager->get_max_level_computational_grid())
        crossed_cell = crossed_cell || (fetch_fv_in_cell_solver ? (dynamic_cast<const my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->get_fv_map().find(quad_idx) != dynamic_cast<const my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->get_fv_map().end())
                                                                : interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx));

      if(!crossed_cell && phi_np1_on_computational_nodes_p[nodes->local_nodes[P4EST_CHILDREN*quad_idx + 0]] <= 0.0)
        data_for_all_reduce[0] += MULTD(tree_dimensions[0]/(1 << quad->level), tree_dimensions[1]/(1 << quad->level), tree_dimensions[2]/(1 << quad->level));
      else if(crossed_cell)
      {
        if(fetch_fv_in_cell_solver)
          const_fv_ptr =  &(dynamic_cast<const my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->get_fv_map().at(quad_idx));
        else
        {
          interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, &fv); // could be a redundant task but that's seriously the last of my concerns, right now
          const_fv_ptr = &fv;
        }

        data_for_all_reduce[0] += const_fv_ptr->volume_in_negative_domain();
        P4EST_ASSERT(const_fv_ptr->interfaces.size() <= 1);
        for (size_t k = 0; k < const_fv_ptr->interfaces.size(); ++k)
        {
          const double xyz_interface_quadrature[P4EST_DIM] = {DIM(xyz_quad[0] + const_fv_ptr->interfaces[k].centroid[0], xyz_quad[1] + const_fv_ptr->interfaces[k].centroid[1], xyz_quad[2] + const_fv_ptr->interfaces[k].centroid[2])};
          const double rr     = ABSD(xyz_interface_quadrature[0], xyz_interface_quadrature[1], xyz_interface_quadrature[2]);
          const double theta  = (rr >  0.1*ABSD(dxyz_min[0], dxyz_min[1], dxyz_min[2]) ? asin(sqrt(SQR(xyz_interface_quadrature[0]) ONLY3D( + SQR(xyz_interface_quadrature[2])))/rr) : 0.0);
          const double P2     = 0.5*(3.0*SQR(cos(theta)) - 1.0);
          data_for_all_reduce[1] += const_fv_ptr->interfaces[k].area;
          data_for_all_reduce[2] += const_fv_ptr->interfaces[k].area*P2;
          data_for_all_reduce[3] += const_fv_ptr->interfaces[k].area*SQR(P2);
          data_for_all_reduce[4] += const_fv_ptr->interfaces[k].area*rr;
          data_for_all_reduce[5] += const_fv_ptr->interfaces[k].area*rr*P2;
        }
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, data_for_all_reduce.data(), data_for_all_reduce.size(), MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  volume_inside = data_for_all_reduce[0];
  RR              = (data_for_all_reduce[3]*data_for_all_reduce[4] - data_for_all_reduce[2]*data_for_all_reduce[5])/(data_for_all_reduce[1]*data_for_all_reduce[3] - SQR(data_for_all_reduce[2]));
  a_2_variable_RR = (-data_for_all_reduce[2]*data_for_all_reduce[4] + data_for_all_reduce[1]*data_for_all_reduce[5])/(data_for_all_reduce[1]*data_for_all_reduce[3] - SQR(data_for_all_reduce[2]));
  a_2_constant_R0 = (data_for_all_reduce[5] - R0*data_for_all_reduce[2])/data_for_all_reduce[3];

  ierr = VecRestoreArrayRead(phi_np1_on_computational_nodes, &phi_np1_on_computational_nodes_p); CHKERRXX(ierr);

  if(destroy_phi_comp){
    ierr = delete_and_nullify_vector(phi_np1_on_computational_nodes); CHKERRXX(ierr);
  }

  return;
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  ostringstream streamObj;
  cmd.add_option("lmin", "min level of the trees, default is " + to_string(default_lmin));
  cmd.add_option("lmax", "max level of the trees, default is " + to_string(default_lmax));
  streamObj.str(""); streamObj << default_vorticity_threshold;
  cmd.add_option("thresh", "the vorticity-based threshold used for the refinement criteria, default is " + streamObj.str());
  streamObj.str(""); streamObj << 100.0*default_uniform_band_to_radius;
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is such that " + streamObj.str() + "% of the bubble radius is covered");
  cmd.add_option("ntree_x", "number of trees in the macromesh, along the x-direction. The default value is " + to_string(default_ntree[0]));
  cmd.add_option("ntree_y", "number of trees in the macromesh, along the y-direction. The default value is " + to_string(default_ntree[1]));
#ifdef P4_TO_P8
  cmd.add_option("ntree_z", "number of trees in the macromesh, along the z-direction. The default value is " + to_string(default_ntree[2]));
#endif
  streamObj.str(""); streamObj << default_domain[0];
  cmd.add_option("length",    "dimension of the computational domain along the x-direction. The default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_domain[1];
  cmd.add_option("height",    "dimension of the computational domain along the y-direction. The default value is " + streamObj.str());
#ifdef P4_TO_P8
  streamObj.str(""); streamObj << default_domain[2];
  cmd.add_option("width",    "dimension of the computational domain along the z-direction. The default value is " + streamObj.str());
#endif
  cmd.add_option("xperiodic", "flag activating periodicity along x, if set to true or 1, deactivating periodicity along x if set to false or 0. Default is " + string(default_periodic[0] ? "" : "not") + " x-periodic.");
  cmd.add_option("yperiodic", "flag activating periodicity along y, if set to true or 1, deactivating periodicity along y if set to false or 0. Default is " + string(default_periodic[1] ? "" : "not") + " y-periodic.");
#ifdef P4_TO_P8
  cmd.add_option("zperiodic", "flag activating periodicity along z, if set to true or 1, deactivating periodicity along z if set to false or 0. Default is " + string(default_periodic[2] ? "" : "not") + " z-periodic.");
#endif
  // physical parameters for the simulations
  streamObj.str(""); streamObj << default_ratio_rho;
  cmd.add_option("ratio_rho", "The ratio of mass densities (outer fluid to inner fluid). Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_ratio_mu;
  cmd.add_option("ratio_mu", "The ratio of dynamic viscosities (outer fluid to inner fluid). Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_Reynolds;
  cmd.add_option("Reynolds", "The desired Reynolds number Re = sqrt(gamma*R0*rho_plus)/mu_plus. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_t_end;
  cmd.add_option("t_end", "The final simulation time (in units of inviscid periods of oscillations). Default t_end is " + streamObj.str());
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
  cmd.add_option("vtk_dt", "vtk_dt = time step between two vtk exportation (in units of inviscid periods of oscillations), default is " + streamObj.str());
  streamObj.str(""); streamObj << default_vtk_idx_start;
  cmd.add_option("vtk_idx_start", "first desired index of exported vtk files, default is " + streamObj.str());
  cmd.add_option("work_dir", "root exportation directory, subfolders will be created therein (read from input if not defined otherwise in the environment variable OUT_DIR). \n\tThis is required for vtk files and for data files. Default is " + default_work_folder);
  streamObj.str(""); streamObj << default_interp_method_phi;
  cmd.add_option("phi_interp", "interpolation method for the node-sampled levelset function. Default is " + streamObj.str());
  cmd.add_option("subrefinement", "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");
  streamObj.str(""); streamObj << default_save_state_dt;
  cmd.add_option("save_state_dt", "time interval between two saved states (in units of inviscid periods of oscillations). Saved in backup_ subfolders. Default is " + streamObj.str());
  cmd.add_option("save_nstates",  "determines how many solver states must be memorized in backup_ folders (default is " + to_string(default_save_nstates) + ")");

  if(cmd.parse(argc, argv, main_description))
    return 0;

  const string root_export_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR")));
  const int niter_reinit = cmd.get<int> ("n_reinit", default_n_reinit);

  PetscErrorCode ierr;
  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
  my_p4est_brick_t *brick                           = NULL;
  p4est_connectivity_t* connectivity                = NULL;


  if(cmd.contains("restart"))
    load_solver_from_state(mpi, cmd, two_phase_flow_solver, brick, connectivity);
  else
    create_solver_from_scratch(mpi, cmd, two_phase_flow_solver, brick, connectivity);

  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p; bc_p.setWallTypes(neumann_cf); bc_p.setWallValues(zero_cf);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    bc_v[dim].setWallTypes(dirichlet_cf); bc_v[dim].setWallValues(zero_cf);
  }
  two_phase_flow_solver->set_bc(bc_v, &bc_p);

  splitting_criteria_t* data            = (splitting_criteria_t*) (two_phase_flow_solver->get_p4est_n()->user_pointer); // to delete it appropriately, eventually
  splitting_criteria_t* subrefined_data = (two_phase_flow_solver->get_fine_p4est_n() != NULL ? (splitting_criteria_t*) two_phase_flow_solver->get_fine_p4est_n()->user_pointer : NULL); // same, to delete it appropriately, eventually

  const int nn = 2;
#ifdef P4_TO_P8
  const double time_unit  = 2.0*M_PI/sqrt(two_phase_flow_solver->get_surface_tension()*(nn - 1)*nn*(nn + 1)*(nn + 2)/((nn*two_phase_flow_solver->get_rho_plus() + (nn + 1)*two_phase_flow_solver->get_rho_minus())*pow(R0, 3.0)));
#else
  const double time_unit  = 2.0*M_PI/sqrt(two_phase_flow_solver->get_surface_tension()*(nn - 1)*nn*nn/((two_phase_flow_solver->get_rho_plus() + two_phase_flow_solver->get_rho_minus())*pow(R0, 3.0)));
#endif
  const double t_end      = cmd.get<double> ("t_end",         default_t_end)*time_unit;
  two_phase_flow_solver->set_final_time(t_end);
  const bool save_vtk     = cmd.get<bool>   ("save_vtk",      default_save_vtk);
  const double vtk_dt     = cmd.get<double> ("vtk_dt",        default_vtk_dt)*time_unit;
  if(vtk_dt <= 0.0)
    throw std::invalid_argument("main for bubble oscillations: the value of vtk_dt must be strictly positive.");
  const int save_nstates      = cmd.get<int>    ("save_nstates",  default_save_nstates);
  const double save_state_dt  = cmd.get<double> ("save_state_dt", default_save_state_dt)*time_unit;

  const splitting_criteria_t* sp = (splitting_criteria_t*) two_phase_flow_solver->get_p4est_n()->user_pointer;
  streamObj.str("");
  streamObj << "mu_ratio_" << two_phase_flow_solver->get_mu_plus()/two_phase_flow_solver->get_mu_minus()
            << "_rho_ratio_" << two_phase_flow_solver->get_rho_plus()/two_phase_flow_solver->get_rho_minus()
            << "_Reynolds_" << sqrt(two_phase_flow_solver->get_surface_tension()*R0*two_phase_flow_solver->get_rho_minus())/two_phase_flow_solver->get_mu_minus();
  const string export_dir   = root_export_folder + (root_export_folder.back() == '/' ? "" : "/") + streamObj.str() + "/lmin_" + to_string(sp->min_lvl) + "_lmax_" + to_string(sp->max_lvl);
  const string vtk_dir      = export_dir + "/vtu";
  const string results_dir  = export_dir + "/results";
  if(create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for bubble oscillations: could not create exportation directory " + export_dir);
  if(create_directory(results_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for bubble oscillations: could not create directory for exportation of results, i.e., " + results_dir);
  if(save_vtk && create_directory(vtk_dir, mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for bubble oscillations: could not create directory for visualization files, i.e., " + vtk_dir);

  string datafile;
  initialize_exportations(two_phase_flow_solver->get_tn(), mpi, results_dir, datafile);

  const int vtk_start     = cmd.get<int>("vtk_idx_start", default_vtk_idx_start);
  const double vmax_abort = cmd.get<double>("vmax_abort", default_vmax_abort);
  const double projection_threshold = cmd.get<double>("projection_threshold", default_projection_threshold);
  const int n_fixpoint_iter_max = cmd.get<int>("niter", default_niter);
  int vtk_idx     = vtk_index(vtk_start, two_phase_flow_solver, vtk_dt) - 1; // -1 so that we do not miss the very first snapshot
  int backup_idx  = backup_index(two_phase_flow_solver, save_nstates);

  const double nu_minus = two_phase_flow_solver->get_mu_minus()/two_phase_flow_solver->get_rho_minus();
  const double nu_plus  = two_phase_flow_solver->get_mu_plus()/two_phase_flow_solver->get_rho_plus();
  const double dt_visc = 1.0/(MAX(nu_minus, nu_plus)*(SUMD(2.0/SQR((brick->xyz_max[0] - brick->xyz_min[0])/(brick->nxyztrees[0]*(1 << sp->max_lvl))), 2.0/SQR((brick->xyz_max[1] - brick->xyz_min[1])/(brick->nxyztrees[1]*(1 << sp->max_lvl))), 2.0/SQR((brick->xyz_max[2] - brick->xyz_min[2])/(brick->nxyztrees[2]*(1 << sp->max_lvl))))));
  bool advance_solver = false;

  while(two_phase_flow_solver->get_tn() < t_end)
  {
    if(advance_solver)
      two_phase_flow_solver->update_from_tn_to_tnp1(niter_reinit);

    if(save_nstates > 0 && backup_index(two_phase_flow_solver, save_state_dt) != backup_idx)
    {
      backup_idx = backup_index(two_phase_flow_solver, save_state_dt);
      two_phase_flow_solver->save_state(export_dir.c_str(), save_nstates);
    }

    two_phase_flow_solver->solve_time_step(projection_threshold, n_fixpoint_iter_max);

    if(save_vtk && vtk_idx != vtk_index(vtk_start, two_phase_flow_solver, vtk_dt))
    {
      vtk_idx = vtk_index(vtk_start, two_phase_flow_solver, vtk_dt);
      two_phase_flow_solver->save_vtk(vtk_dir, vtk_idx);
    }

    if(two_phase_flow_solver->get_max_velocity() > vmax_abort)
    {
      if(save_nstates > 0)
        two_phase_flow_solver->save_state(export_dir.c_str(), backup_index(two_phase_flow_solver, save_state_dt) + 1);
      if(save_vtk)
        two_phase_flow_solver->save_vtk(vtk_dir, ++vtk_idx, true);
      ierr = PetscPrintf(mpi.comm(), "The maximum velocity of %g exceeded the tolerated threshold of %g... \n", two_phase_flow_solver->get_max_velocity(), vmax_abort); CHKERRXX(ierr);
      break;
    }

    double RR, a_2_variable_RR, a_2_constant_RR, bubble_volume;
    get_normal_mode_decomposition_and_volume(two_phase_flow_solver, RR, a_2_variable_RR, a_2_constant_RR, bubble_volume);

    if(mpi.rank() == 0)
      export_results(two_phase_flow_solver->get_tnp1(),
                     RR,
                     R0,
                     a_2_variable_RR,
                     a_2_constant_RR,
                     bubble_volume,
                     two_phase_flow_solver->get_dt_n()/dt_visc,
                     two_phase_flow_solver->get_dt_n()/two_phase_flow_solver->get_capillary_dt(),
                     datafile);
    advance_solver = true;
  }
  ierr = PetscPrintf(mpi.comm(), "Gracefully finishing up now\n");

  delete two_phase_flow_solver;
  my_p4est_brick_destroy(connectivity, brick);
  delete brick;
  delete data;
  delete subrefined_data;

  return EXIT_SUCCESS;
}
