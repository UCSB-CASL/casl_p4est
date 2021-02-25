// p4est Library
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
    string("Small buoyant bubble test from section 4.1 of \"A boundary condition capturing method for multiphase incompressible flow\"! \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), 2019-2020-2021-...-2523\n");

// problem setup: 1) set parameters
const double initial_bubble_diameter  = 2.0/300.0;
const double rho_plus                 = 1000.0;
const double rho_minus                = 1.226;
const double mu_plus                  = 0.001137;
const double mu_minus                 = 0.0000178;
const double surface_tension          = 0.0728;
const double gravity                  = -9.81;
const double xyz_min_[P4EST_DIM]      = {DIM(-0.01, -0.01, -DBL_MAX)};
const double xyz_max_[P4EST_DIM]      = {DIM( 0.01,  0.02, +DBL_MAX)};
const int periodic[P4EST_DIM]         = {DIM(0, 0, 0)};
const int ntree_xyz[P4EST_DIM]            = {DIM(2, 3, INT_MAX)};
// grid-related
const int default_lmin = 2;
const int default_lmax = 6;
const double default_vorticity_threshold      = DBL_MAX;
const double default_uniform_band_to_radius   = 0.15;
// simulation-related:
const interpolation_method default_interp_method_phi = quadratic_non_oscillatory_continuous_v2;
const bool default_subrefinement = false;
const bool default_use_second_order_theta = (default_interp_method_phi == linear ? false : true); // relevant only if using (x)GFM cell solver
const int default_nviscous_subiter  = 5;
const int default_sl_order          = 2;
const int default_sl_order_itfc     = 2;
const double default_cfl_advection  = 1.0;
const double default_cfl_visco_capillary = 0.95;
const double default_cfl_capillary  = 0.95;
const jump_solver_tag default_cell_solver = FV;
const jump_solver_tag default_face_solver = xGFM;
const int default_n_reinit = 1;
const double default_t_end = 0.05;
const double default_vmax_abort = 1.0;
const double default_projection_threshold = 0.01;
const int default_niter = 5;
// exportation-related
const bool default_save_vtk         = true;
const double default_vtk_dt         = 0.001;
const int default_vtk_idx_start     = 0;
const int default_save_nstates      = 0;
const double default_save_state_dt  = 0.005;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_flow/small_buoyant_bubble_GFM";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_flow/small_buoyant_bubble_GFM";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_flow/small_buoyant_bubble_GFM";
#endif

class initial_level_set_t: public CF_DIM {
public:
  initial_level_set_t() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(SUMD(SQR(x), SQR(y), SQR(z))) - 0.5*initial_bubble_diameter;
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
      fprintf(fp_results, "%% tn | v_bubble_itfc_y | v_bubble_volume_y | circularity | volume | dt/dt_visc | dt/dt_capillar \n");
      fclose(fp_results);
    }
    else
      truncate_exportation_file_up_to_tstart(tstart, datafile, 7);

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
      fprintf(fp_liveplot, "set ylabel \"Rising-velocity (interface-average)\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot \"monitoring_results.dat\" using 1:2 notitle with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 1 position 800,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Rising-velocity (bubble-volume average) \" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot \"monitoring_results.dat\" using 1:3 notitle with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 2 position 1600,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Circularity \" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot \"monitoring_results.dat\" using 1:4 notitle with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 3 position 0,800 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Relative difference in bubble volume (in %%)\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "col = 5\n");
      fprintf(fp_liveplot, "row = 1\n");
      fprintf(fp_liveplot, "stats 'monitoring_results.dat' every ::row::row using col nooutput\n");
      fprintf(fp_liveplot, "original_volume = STATS_min\n");
      fprintf(fp_liveplot, "plot  \"monitoring_results.dat\" using 1:(100 * (original_volume - $5)/original_volume) notitle with lines lw 3\n");
      fprintf(fp_liveplot, "reread");
      fclose(fp_liveplot);
    }
  }
  return;
}

void export_results(const double& tn, const double& v_bubble_interface_y, const double& v_bubble_volume_y,
                    const double& circularity,
                    const double& bubble_volume,
                    const double& dt_to_dt_visc, const double& dt_to_dt_capillary, const string& datafile)
{
  FILE* fp_data = fopen(datafile.c_str(), "a");
  if(fp_data == NULL)
    throw std::invalid_argument("main for buoyant bubble: could not open file for output of raw results.");
  fprintf(fp_data, "%g %g %g %g %g %g %g \n", tn, v_bubble_interface_y, v_bubble_volume_y, circularity, bubble_volume, dt_to_dt_visc, dt_to_dt_capillary);
  fclose(fp_data);
}

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd,
                                my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity)
{
  const int lmin                        = cmd.get<int>    ("lmin",   default_lmin);
  const int lmax                        = cmd.get<int>    ("lmax",   default_lmax);
  const double vorticity_threshold      = cmd.get<double> ("thresh", default_vorticity_threshold);
  const double dxmin                    = MIN(DIM((xyz_max_[0] - xyz_min_[0])/ntree_xyz[0], (xyz_max_[1] - xyz_min_[1])/ntree_xyz[1], (xyz_max_[2] - xyz_min_[2])/ntree_xyz[2]))/((double) (1 << lmax));
  const double uniform_band_in_dxmin    = cmd.get<double> ("uniform_band",      default_uniform_band_to_radius*initial_bubble_diameter*0.5/dxmin);
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
  connectivity = my_p4est_brick_new(ntree_xyz, xyz_min_, xyz_max_, brick, periodic);
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
                                                                ngbd_c, faces, phi);
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
  // simulation time
  streamObj.str(""); streamObj << default_t_end;
  cmd.add_option("t_end", "The final simulation time. Default t_end is " + streamObj.str());
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
  cmd.add_option("vtk_dt", "vtk_dt = time step between two vtk exportation, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_vtk_idx_start;
  cmd.add_option("vtk_idx_start", "first desired index of exported vtk files, default is " + streamObj.str());
  cmd.add_option("work_dir", "root exportation directory, subfolders will be created therein (read from input if not defined otherwise in the environment variable OUT_DIR). \n\tThis is required for vtk files and for data files. Default is " + default_work_folder);
  streamObj.str(""); streamObj << default_interp_method_phi;
  cmd.add_option("phi_interp", "interpolation method for the node-sampled levelset function. Default is " + streamObj.str());
  cmd.add_option("subrefinement", "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");
  streamObj.str(""); streamObj << default_save_state_dt;
  cmd.add_option("save_state_dt", "if save_nstates > 0, the solver state is saved every save_state_dt time increments in backup_ subfolders. Default is " + streamObj.str());
  cmd.add_option("save_nstates",  "determines how many solver states must be memorized in backup_ folders (default is " + to_string(default_save_nstates) + ")");

#ifdef P4_TO_P8
  std::cerr << "This example is intrinsically two-dimensional; the main file would need adaptation for three-dimensional runs " << std::endl;
  return EXIT_FAILURE
#endif

  if(cmd.parse(argc, argv, main_description))
    return EXIT_SUCCESS;

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

  cf_const_t body_force_x(0.0);
  cf_const_t body_force_y(gravity);
#ifdef P4_TO_P8
  cf_const_t body_force_z(0.0);
#endif
  CF_DIM* body_force[P4EST_DIM] = {DIM(&body_force_x, &body_force_y, &body_force_z)};
  two_phase_flow_solver->set_external_forces_per_unit_mass(body_force);

  splitting_criteria_t* data            = (splitting_criteria_t*) (two_phase_flow_solver->get_p4est_n()->user_pointer); // to delete it appropriately, eventually
  splitting_criteria_t* subrefined_data = (two_phase_flow_solver->get_fine_p4est_n() != NULL ? (splitting_criteria_t*) two_phase_flow_solver->get_fine_p4est_n()->user_pointer : NULL); // same, to delete it appropriately, eventually

  const double t_end      = cmd.get<double> ("t_end",         default_t_end);
  two_phase_flow_solver->set_final_time(t_end);
  const bool save_vtk     = cmd.get<bool>   ("save_vtk",      default_save_vtk);
  const double vtk_dt     = cmd.get<double> ("vtk_dt",        default_vtk_dt);
  if(vtk_dt <= 0.0)
    throw std::invalid_argument("main for buoyant bubble: the value of vtk_dt must be strictly positive.");
  const int save_nstates      = cmd.get<int>    ("save_nstates",  default_save_nstates);
  const double save_state_dt  = cmd.get<double> ("save_state_dt", default_save_state_dt);

  const splitting_criteria_t* sp = (splitting_criteria_t*) two_phase_flow_solver->get_p4est_n()->user_pointer;
  streamObj.str("");
  const string export_dir   = root_export_folder + (root_export_folder.back() == '/' ? "" : "/") + "lmin_" + to_string(sp->min_lvl) + "_lmax_" + to_string(sp->max_lvl);
  const string vtk_dir      = export_dir + "/vtu";
  const string results_dir  = export_dir + "/results";
  if(create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for buoyant bubble: could not create exportation directory " + export_dir);
  if(create_directory(results_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for buoyant bubble: could not create directory for exportation of results, i.e., " + results_dir);
  if(save_vtk && create_directory(vtk_dir, mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for buoyant bubble: could not create directory for visualization files, i.e., " + vtk_dir);

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
      if(save_vtk)
        two_phase_flow_solver->save_vtk(vtk_dir, ++vtk_idx, true);
      ierr = PetscPrintf(mpi.comm(), "The maximum velocity of %g exceeded the tolerated threshold of %g... \n", two_phase_flow_solver->get_max_velocity(), vmax_abort); CHKERRXX(ierr);
      delete two_phase_flow_solver;
      my_p4est_brick_destroy(connectivity, brick);
      delete brick;
      delete data;
      delete subrefined_data;

      return EXIT_FAILURE;
    }

    double avg_itfc_velocity[P4EST_DIM], interface_length;
    double avg_velocity_in_bubble[P4EST_DIM], bubble_volume;
    two_phase_flow_solver->get_average_interface_velocity_and_interface_area(avg_itfc_velocity, interface_length);
    two_phase_flow_solver->get_volume_and_average_velocity_in_domain(-1, avg_velocity_in_bubble, bubble_volume);
    if(mpi.rank() == 0)
      export_results(two_phase_flow_solver->get_tnp1(),
                     avg_itfc_velocity[1],
          avg_velocity_in_bubble[1],
          (M_PI*initial_bubble_diameter)/interface_length,
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
