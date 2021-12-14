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
    string("Boiling multiscale, pretty pictures and so on... \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu)\n");

// problem setup: 1) set parameters
const double default_T_sat                        = 0;
const double default_T_bottom                     = 10;
const double default_latent_heat                  = -10000;
const double default_thermal_conductivity_plus    = 40;
const double default_specific_heat_capacity_plus  = 400;
const double default_rho_plus                     = 200;
const double default_mu_plus                      = 0.1;
const double default_surface_tension              = 0.1;
const double default_thermal_conductivity_minus   = 1;
const double default_specific_heat_capacity_minus = 200;
const double default_rho_minus                    = 5;
const double default_mu_minus                     = 0.005;
const double default_gravity                      = 9.81;
// problem setup: 2) free quantities
const double default_domain_to_lambda[P4EST_DIM] = {DIM(10, 10, 10)};
const bool default_periodic[P4EST_DIM] = {DIM(true, false, true)};
// grid-related
const int default_lmin = 6;
const int default_lmax = 9;
const double default_vorticity_threshold = 0.02;
const double default_uniform_band = 5;
const int default_ntree[P4EST_DIM]  = {DIM(1, 1, 1)};
// simulation-related:
const interpolation_method default_interp_method_phi = linear;
const bool default_subrefinement = false;
const bool default_use_second_order_theta = (default_interp_method_phi == linear ? false : true); // relevant only if using (x)GFM cell solver
const int default_nviscous_subiter  = 10;
const int default_niter             = 3;
const int default_sl_order          = 2;
const int default_sl_order_itfc     = 2;
const int default_sl_order_temp     = 2;
const double default_cfl_advection        = 1.0;
const double default_cfl_visco_capillary  = 0.95;
const double default_cfl_capillary        = 0.95;
const jump_solver_tag default_cell_solver = FV;
const jump_solver_tag default_face_solver = xGFM;
const int default_n_reinit = 1;
const double default_t_end = 1.5;
const double default_vmax_abort = 1000.0;
const double default_projection_threshold = 0.01;
// exportation-related
const bool default_save_vtk     = true;
const double default_vtk_dt     = 0.02;
const int default_vtk_idx_start = 0;
const int default_save_nstates  = 0;
const double default_save_state_dt = 0.02;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_flow/boiling_multiscale/" + to_string(P4EST_DIM) + "D";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_flow/boiling_multiscale/" + to_string(P4EST_DIM) + "D";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_flow/boiling_multiscale/" + to_string(P4EST_DIM) + "D";
#endif

class initial_level_set_t: public CF_DIM {
  const double taylor_wavelength;
public:
  initial_level_set_t(double taylor_wavelength_) : taylor_wavelength(taylor_wavelength_) { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
#ifdef P4_TO_P8
    return (y - (12 + 0.25*(1.0 + cos(2.0*M_PI*x/taylor_wavelength))*(1.0 + cos(2.0*M_PI*z/taylor_wavelength))
                 + 0.6*(1.0 + sin(2.0*M_PI*x/(2*taylor_wavelength)))*(1.0 + cos(2.0*M_PI*z/(4*taylor_wavelength)))
                 + 0.85*(1.0 + cos(2.0*M_PI*x/(4*taylor_wavelength)))*(1.0 + sin(2.0*M_PI*z/(2*taylor_wavelength))))*taylor_wavelength/128);
#else
    return (y - (10 + 0.3*cos(2.0*M_PI*x/taylor_wavelength) + 0.6*sin(2.0*M_PI*x/(2*taylor_wavelength)) + cos(2.0*M_PI*x/(4*taylor_wavelength)))*taylor_wavelength/128);
#endif
  }
};

class pressure_wall_bc_type_t : public WallBCDIM {
private:
  const double *xyz_min, *xyz_max;
public:
  pressure_wall_bc_type_t(const double* xyz_min_, const double* xyz_max_) : xyz_min(xyz_min_), xyz_max(xyz_max_) {}
  BoundaryConditionType operator()(DIM(double, double y, double)) const
  {
    bool top_wall   = fabs(y - xyz_max[1]) < EPS*(xyz_max[1] - xyz_min[1]);
    return (top_wall ? DIRICHLET : NEUMANN);
  }
};

class velocity_wall_bc_type_t : public WallBCDIM {
private:
  const double *xyz_min, *xyz_max;
public:
  velocity_wall_bc_type_t(const double* xyz_min_, const double* xyz_max_) : xyz_min(xyz_min_), xyz_max(xyz_max_) {}
  BoundaryConditionType operator()(DIM(double, double y, double)) const
  {
    bool top_wall   = fabs(y - xyz_max[1]) < EPS*(xyz_max[1] - xyz_min[1]);
    return top_wall ? NEUMANN : DIRICHLET;
  }
};

class temperature_wall_bc_type_t : public WallBCDIM {
private:
  const double *xyz_min, *xyz_max;
public:
  temperature_wall_bc_type_t(const double* xyz_min_, const double* xyz_max_) : xyz_min(xyz_min_), xyz_max(xyz_max_) {}
  BoundaryConditionType operator()(DIM(double, double y, double)) const
  {
    bool top_wall   = fabs(y - xyz_max[1]) < EPS*(xyz_max[1] - xyz_min[1]);
    return (top_wall ? NEUMANN : DIRICHLET);
  }
};

class temperature_wall_bc_value_t : public CF_DIM {
  const temperature_wall_bc_type_t bc_type;
  const double dirichlet_temperature;
public:
  temperature_wall_bc_value_t(const temperature_wall_bc_type_t& wall_type, double T_dirichlet) : bc_type(wall_type), dirichlet_temperature(T_dirichlet) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return (bc_type(DIM(x, y, z)) == DIRICHLET ? dirichlet_temperature : 0.0);
  }
};

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd,
                                my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity)
{
  const int lmin                        = cmd.get<int>    ("lmin",   default_lmin);
  const int lmax                        = cmd.get<int>    ("lmax",   default_lmax);
  const double vorticity_threshold      = cmd.get<double> ("thresh", default_vorticity_threshold);
  const int ntree_xyz[P4EST_DIM]        = {DIM(cmd.get<int>("ntree_x", default_ntree[0]),
                                           cmd.get<int>("ntree_y", default_ntree[1]),
                                           cmd.get<int>("ntree_z", default_ntree[2]))};
  const int periodic[P4EST_DIM]         = { DIM(cmd.get<bool>("xperiodic", default_periodic[0]),
                                            cmd.get<bool>("yperiodic", default_periodic[1]),
                                            cmd.get<bool>("zperiodic", default_periodic[2]))};
  const double uniform_band_in_dxmin    = cmd.get<double> ("uniform_band", default_uniform_band);
  const double rho_minus                = cmd.get<double>("rho_minus", default_rho_minus);
  const double mu_minus                 = cmd.get<double> ("mu_minus", default_mu_minus);
  const double rho_plus                 = cmd.get<double>("rho_plus", default_rho_plus);
  const double mu_plus                  = cmd.get<double> ("mu_plus", default_mu_plus);
  const double surface_tension          = cmd.get<double> ("surface_tension", default_surface_tension);

  const double T_sat                      = cmd.get<double>("saturation_temperature", default_T_sat);
  const double latent_heat                = cmd.get<double>("latent_heat", default_latent_heat);
  const double thermal_conductivity_minus = cmd.get<double>("thermal_conductivity_minus", default_thermal_conductivity_minus);
  const double thermal_conductivity_plus  = cmd.get<double>("thermal_conductivity_plus", default_thermal_conductivity_plus);
  const double specific_heat_capacity_minus = cmd.get<double>("specific_heat_capacity_minus", default_specific_heat_capacity_minus);
  const double specific_heat_capacity_plus  = cmd.get<double>("specific_heat_capacity_plus", default_specific_heat_capacity_plus);
  const double gravity                  = cmd.get<double>("gravity", default_gravity);
  const double taylor_wavelength        = 2.0*M_PI*sqrt(3.0*ONLY3D(2.0*)surface_tension/(gravity*(rho_plus - rho_minus)));
  const double domain_size[P4EST_DIM]   = {DIM(cmd.get<double>("length", default_domain_to_lambda[0])*taylor_wavelength,
                                           cmd.get<double>("height", default_domain_to_lambda[1])*taylor_wavelength,
                                           cmd.get<double>("width", default_domain_to_lambda[2])*taylor_wavelength)};
  const double xyz_min[P4EST_DIM]       = { DIM(-0.5*domain_size[0],              0, -0.5*domain_size[2]) };
  const double xyz_max[P4EST_DIM]       = { DIM( 0.5*domain_size[0], domain_size[1],  0.5*domain_size[2]) };

  const bool use_second_order_theta     = cmd.get<bool>   ("second_order_ls",   default_use_second_order_theta);
  const int sl_order                    = cmd.get<int>    ("sl_order",          default_sl_order);
  const int sl_order_interface          = cmd.get<int>    ("sl_order_itfc",     default_sl_order_itfc);
  const int sl_order_temperature        = cmd.get<int>    ("sl_order_temp",     default_sl_order_temp);
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
  initial_level_set_t ls(taylor_wavelength);
  splitting_criteria_cf_and_uniform_band_t* data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &ls, uniform_band_in_dxmin);
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
    splitting_criteria_cf_t* subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, &ls);
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
  solver->set_phase_change_parameters(latent_heat, T_sat);
  solver->set_thermal_conductivities(thermal_conductivity_minus, thermal_conductivity_plus);
  solver->set_specific_heats(specific_heat_capacity_minus, specific_heat_capacity_plus);
  solver->set_semi_lagrangian_order_temperature(sl_order_temperature);
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
  streamObj.str(""); streamObj << default_uniform_band;
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is " + streamObj.str());
  cmd.add_option("ntree_x", "number of trees in the macromesh, along the x-direction. The default value is " + to_string(default_ntree[0]));
  cmd.add_option("ntree_y", "number of trees in the macromesh, along the y-direction. The default value is " + to_string(default_ntree[1]));
#ifdef P4_TO_P8
  cmd.add_option("ntree_z", "number of trees in the macromesh, along the z-direction. The default value is " + to_string(default_ntree[2]));
#endif
  streamObj.str(""); streamObj << default_domain_to_lambda[0];
  cmd.add_option("length",    "dimension of the computational domain along the x-direction (in units of \\lambda). The default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_domain_to_lambda[1];
  cmd.add_option("height",    "dimension of the computational domain along the y-direction (in units of \\lambda). The default value is " + streamObj.str());
#ifdef P4_TO_P8
  streamObj.str(""); streamObj << default_domain_to_lambda[2];
  cmd.add_option("width",     "dimension of the computational domain along the z-direction (in units of \\lambda). The default value is " + streamObj.str());
#endif
  cmd.add_option("xperiodic", "flag activating periodicity along x, if set to true or 1, deactivating periodicity along x if set to false or 0. Default is " + string(default_periodic[0] ? "" : "not") + " x-periodic.");
  cmd.add_option("yperiodic", "flag activating periodicity along y, if set to true or 1, deactivating periodicity along y if set to false or 0. Default is " + string(default_periodic[1] ? "" : "not") + " y-periodic.");
#ifdef P4_TO_P8
  cmd.add_option("zperiodic", "flag activating periodicity along z, if set to true or 1, deactivating periodicity along z if set to false or 0. Default is " + string(default_periodic[2] ? "" : "not") + " z-periodic.");
#endif
  // physical parameters for the simulations
  streamObj.str(""); streamObj << default_surface_tension;
  cmd.add_option("surface_tension", "The surface tension. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_rho_minus;
  cmd.add_option("rho_minus", "The mass density in negative domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_rho_plus;
  cmd.add_option("rho_plus", "The mass density in positive domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_mu_minus;
  cmd.add_option("mu_minus", "The viscosity in negative domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_mu_plus;
  cmd.add_option("mu_plus", "The viscosity in positive domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_thermal_conductivity_minus;
  cmd.add_option("thermal_conductivity_minus", "The thermal conductivity in negative domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_thermal_conductivity_plus;
  cmd.add_option("thermal_conductivity_plus", "The thermal conductivity in positive domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_specific_heat_capacity_minus;
  cmd.add_option("specific_heat_capacity_minus", "The specific heat capacity in negative domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_specific_heat_capacity_plus;
  cmd.add_option("specific_heat_capacity_plus", "The specific heat capacity in positive domain. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_T_sat;
  cmd.add_option("saturation_temperature", "The saturation temperature. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_T_bottom;
  cmd.add_option("T_bottom", "The temperature of the bottom wall. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_latent_heat;
  cmd.add_option("latent_heat", "The latent heat. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_gravity;
  cmd.add_option("gravity", "The gravity acceleration. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_t_end;
  cmd.add_option("t_end", "The final simulation time. Default t_end is " + streamObj.str());
  // method-related parameters
  cmd.add_option("second_order_ls", "flag activating second order F-D interface fetching if set to true or 1. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("sl_order", "the order for the semi lagrangian advection terms, either 1 or 2, default is " + to_string(default_sl_order));
  cmd.add_option("sl_order_itfc", "the order for the semi lagrangian interface advection, either 1 or 2, default is " + to_string(default_sl_order_itfc));
  cmd.add_option("sl_order_temp", "the order for the semi lagrangian temperature advection, either 1 or 2, default is " + to_string(default_sl_order_temp));
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
  cmd.add_option("vtk_dt", "vtk_dt = time step between two vtk exportation, in units of D*mu_plus/gamma, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_vtk_idx_start;
  cmd.add_option("vtk_idx_start", "first desired index of exported vtk files, default is " + streamObj.str());
  cmd.add_option("work_dir", "root exportation directory, subfolders will be created therein (read from input if not defined otherwise in the environment variable OUT_DIR). \n\tThis is required for vtk files and for data files. Default is " + default_work_folder);
  streamObj.str(""); streamObj << default_interp_method_phi;
  cmd.add_option("phi_interp", "interpolation method for the node-sampled levelset function. Default is " + streamObj.str());
  cmd.add_option("subrefinement", "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");
  streamObj.str(""); streamObj << default_save_state_dt;
  cmd.add_option("save_state_dt", "time interval between two saved states, in units of D*mu_plus/gamma. Saved in backup_ subfolders. Default is " + streamObj.str());
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
  {
    try {
      cmd.get<double>("gravity");
      cmd.get<double>("T_bottom");
    } catch (runtime_error& e) {
      ierr = PetscFPrintf(mpi.comm(), stderr, "main for boiling_multiscale: I can't restart properly and safely without 'gravity' and 'T_bottom'. You must give me those."); CHKERRXX(ierr);
      return EXIT_FAILURE;
    }
    load_solver_from_state(mpi, cmd, two_phase_flow_solver, brick, connectivity);
  }
  else
    create_solver_from_scratch(mpi, cmd, two_phase_flow_solver, brick, connectivity);

  const double* xyz_min = two_phase_flow_solver->get_brick()->xyz_min;
  const double* xyz_max = two_phase_flow_solver->get_brick()->xyz_max;
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  pressure_wall_bc_type_t pressure_wall_type(xyz_min, xyz_max);
  velocity_wall_bc_type_t velocity_wall_type(xyz_min, xyz_max);
  BoundaryConditionsDIM bc_p; bc_p.setWallTypes(pressure_wall_type); bc_p.setWallValues(zero_cf);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    bc_v[dim].setWallTypes(velocity_wall_type); bc_v[dim].setWallValues(zero_cf);
  }
  temperature_wall_bc_type_t  temp_wall_type(xyz_min, xyz_max);
  temperature_wall_bc_value_t temp_wall_value(temp_wall_type, cmd.get<double>("T_bottom", default_T_bottom));
  BoundaryConditionsDIM bc_temp; bc_temp.setWallTypes(temp_wall_type); bc_temp.setWallValues(temp_wall_value);
  two_phase_flow_solver->set_bc(bc_v, &bc_p, &bc_temp);
  if(!cmd.contains("restart"))
    two_phase_flow_solver->initialize_temperature_fields();

  cf_const_t body_force_x(0.0);
  cf_const_t body_force_y(-cmd.get<double>("gravity", default_gravity));
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
    throw std::invalid_argument("main for boiling film: the value of vtk_dt must be strictly positive.");
  const int save_nstates      = cmd.get<int>    ("save_nstates",  default_save_nstates);
  const double save_state_dt  = cmd.get<double> ("save_state_dt", default_save_state_dt);

  const splitting_criteria_t* sp = (splitting_criteria_t*) two_phase_flow_solver->get_p4est_n()->user_pointer;

  const string export_dir   = root_export_folder + (root_export_folder.back() == '/' ? "" : "/") + "lmin_" + to_string(sp->min_lvl) + "_lmax_" + to_string(sp->max_lvl);
  const string vtk_dir      = export_dir + "/vtu";
  if(save_vtk && create_directory(vtk_dir, mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for boiling film: could not create directory for visualization files, i.e., " + vtk_dir);

  const int vtk_start     = cmd.get<int>("vtk_idx_start", default_vtk_idx_start);
  const double vmax_abort = cmd.get<double>("vmax_abort", default_vmax_abort);
  const double projection_threshold = cmd.get<double>("projection_threshold", default_projection_threshold);
  const int n_fixpoint_iter_max = cmd.get<int>("niter", default_niter);
  int vtk_idx     = vtk_index(vtk_start, two_phase_flow_solver, vtk_dt) - 1; // -1 so that we do not miss the very first snapshot
  int backup_idx  = backup_index(two_phase_flow_solver, save_nstates);


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

    two_phase_flow_solver->solve_for_mass_flux();
    two_phase_flow_solver->solve_time_step(projection_threshold, n_fixpoint_iter_max);

    if(save_vtk && vtk_idx != vtk_index(vtk_start, two_phase_flow_solver, vtk_dt))
    {
      vtk_idx = vtk_index(vtk_start, two_phase_flow_solver, vtk_dt);
      two_phase_flow_solver->save_vtk(vtk_dir, vtk_idx, true);
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

//    double negative_volune, bubble_volume;
//    two_phase_flow_solver->get_average_velocity_in_domain(-1, avg_bubble_velocity, &bubble_volume);
//    if(mpi.rank() == 0)
//      export_results(two_phase_flow_solver->get_tnp1()/time_unit,
//                     avg_bubble_velocity[1]*two_phase_flow_solver->get_rho_plus()*initial_bubble_diameter/two_phase_flow_solver->get_mu_plus(),
//          two_phase_flow_solver->get_rho_plus()*SQR(avg_bubble_velocity[1])*initial_bubble_diameter/two_phase_flow_solver->get_surface_tension(),
//          bubble_volume,
//          two_phase_flow_solver->get_dt_n()/dt_visc,
//          two_phase_flow_solver->get_dt_n()/two_phase_flow_solver->get_capillary_dt(),
//          datafile);

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
