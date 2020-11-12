// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_vtk.h>
#endif

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

const static string main_description =
    string("Buoyant bubble test! \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), 2019-2020\n");

std::istream& operator>> (std::istream& is, jump_solver_tag& solver)
{
  std::string str;
  is >> str;

  std::vector<size_t> substr_found_at;
  case_insensitive_find_substr_in_str(str, "xGFM", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver = xGFM;
    return is;
  }
  // xGFM not found, look for GFM
  case_insensitive_find_substr_in_str(str, "GFM", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver = GFM;
    return is;
  }
  // nor xGFM nor GFM found, look for FV
  case_insensitive_find_substr_in_str(str, "FV", substr_found_at);
  if(substr_found_at.size() > 0)
  {
    solver = FV;
    return is;
  }
  throw std::runtime_error("unkonwn poisson_jump_cell_solver");
  return is;
}

const int default_lmin = 2;
const int default_lmax = 6;
const int default_ntree[P4EST_DIM] = {DIM(1, 4, 1)};
const interpolation_method default_interp_method_phi = quadratic_non_oscillatory_continuous_v2;
const bool default_use_second_order_theta = false; // relevant only if using (x)GFM cell solver
const bool default_subrefinement = false;
// set 3 parameters: initial bubble diamater, outer mass density and outer viscosity
const double initial_bubble_diameter  = 1.0;
const double rho_plus                 = 1.0;
const double mu_plus                  = 1.0;
// other simulation parameters
const double default_domain[P4EST_DIM] = {DIM(10.0, 40.0, 10.0)};
const bool default_periodic[P4EST_DIM] = {DIM(false, false, false)};
// the 4 nondimensional setup parameters that may be set
const double default_ratio_rho  = 1000.0;
const double default_ratio_mu   = 100.0;
const double default_Eotvos     = 8.67; // case a) in Figure 1 from Bhaga and Weber, JFM (1981), vol. 105, pp. 61-85
const double default_Morton     = 711;  // case a) in Figure 1 from Bhaga and Weber, JFM (1981), vol. 105, pp. 61-85
// grid-related
const double default_vorticity_threshold      = DBL_MAX;
const double default_uniform_band_to_radius   = 0.15;
const double default_nondimensional_duration  = 10.0;
const double default_nondimensional_vtk_dt    = 0.5;
const double default_nondimensional_save_state_dt  = 0.1;
const int default_save_nstates      = 0;
const int default_sl_order          = 2;
const int default_sl_order_itfc     = 2;
const double default_cfl_advection  = 1.0;
const double default_cfl_visco_capillary = 0.95;
const double default_cfl_capillary = 0.95;
const jump_solver_tag default_projection = FV;
const int default_n_reinit = 1;
const bool default_save_vtk = true;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_flow/buoyant_bubble/" + to_string(P4EST_DIM) + "D";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_flow/buoyant_bubble/" + to_string(P4EST_DIM) + "D";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_flow/buoyant_bubble/" + to_string(P4EST_DIM) + "D";
#endif

class initial_level_set_t: public CF_DIM {
public:
  initial_level_set_t() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(SUMD(SQR(x), SQR(y), SQR(z))) - 0.5*initial_bubble_diameter;
  }
} initial_level_set;

class pressure_wall_bc_type_t : public WallBCDIM {
private:
  const double *xyz_min, *xyz_max;
public:
  pressure_wall_bc_type_t(const double* xyz_min_, const double* xyz_max_) : xyz_min(xyz_min_), xyz_max(xyz_max_) {}
  BoundaryConditionType operator()(DIM(double x, double y, double z)) const
  {
    bool top_wall   = fabs(y - xyz_max[1]) < EPS*(xyz_max[1] - xyz_min[1]);
    bool top_corner = top_wall && (fabs(x - xyz_min[0]) < EPS*(xyz_max[0] - xyz_min[0]) || fabs(x - xyz_max[0]) < EPS*(xyz_max[0] - xyz_min[0]))
        ONLY3D(&& (fabs(z - xyz_min[2]) < EPS*(xyz_max[2] - xyz_min[2]) || fabs(z - xyz_max[2]) < EPS*(xyz_max[2] - xyz_min[2])));
//    return (top_wall && !top_corner ? DIRICHLET : NEUMANN);
    return NEUMANN;
  }
};

class velocity_wall_bc_type_t : public WallBCDIM {
private:
  const double *xyz_min, *xyz_max;
public:
  velocity_wall_bc_type_t(const double* xyz_min_, const double* xyz_max_) : xyz_min(xyz_min_), xyz_max(xyz_max_) {}
  BoundaryConditionType operator()(DIM(double x, double y, double z)) const
  {
    bool top_wall   = fabs(y - xyz_max[1]) < EPS*(xyz_max[1] - xyz_min[1]);
    bool top_corner = top_wall && (fabs(x - xyz_min[0]) < EPS*(xyz_max[0] - xyz_min[0]) || fabs(x - xyz_max[0]) < EPS*(xyz_max[0] - xyz_min[0]))
        ONLY3D(&& (fabs(z - xyz_min[2]) < EPS*(xyz_max[2] - xyz_min[2]) || fabs(z - xyz_max[2]) < EPS*(xyz_max[2] - xyz_min[2])));
//    return (top_wall && !top_corner ? NEUMANN : DIRICHLET);
    return DIRICHLET;
  }
};

void build_computational_grids(const mpi_environment_t &mpi, my_p4est_brick_t *brick, p4est_connectivity_t* connectivity, const splitting_criteria_cf_and_uniform_band_t* data,
                               p4est_t* &p4est_nm1, p4est_ghost_t* &ghost_nm1, p4est_nodes_t* &nodes_nm1, my_p4est_hierarchy_t* &hierarchy_nm1, my_p4est_node_neighbors_t* &ngbd_nm1,
                               p4est_t* &p4est_n, p4est_ghost_t* &ghost_n, p4est_nodes_t* &nodes_n, my_p4est_hierarchy_t* &hierarchy_n, my_p4est_node_neighbors_t* &ngbd_n,
                               my_p4est_cell_neighbors_t* &ngbd_c, my_p4est_faces_t* &faces, Vec &phi)
{
  PetscErrorCode ierr;
  // clear inout computational grid data at time (n - 1), if needed
  if(p4est_nm1 != NULL)
    p4est_destroy(p4est_nm1);
  if(ghost_nm1 != NULL)
    p4est_ghost_destroy(ghost_nm1);
  if(nodes_nm1 != NULL)
    p4est_nodes_destroy(nodes_nm1);
  if(hierarchy_nm1 != NULL)
    delete hierarchy_nm1;
  if(ngbd_nm1 != NULL)
    delete ngbd_nm1;
  // clear inout computational grid data at time n, if needed
  if(p4est_n != NULL)
    p4est_destroy(p4est_n);
  if(ghost_n != NULL)
    p4est_ghost_destroy(ghost_n);
  if(nodes_n != NULL)
    p4est_nodes_destroy(nodes_n);
  if(hierarchy_n != NULL)
    delete hierarchy_n;
  if(ngbd_n != NULL)
    delete ngbd_n;
  if(ngbd_c != NULL)
    delete ngbd_c;
  if(faces != NULL)
    delete faces;
  if(phi != NULL){
    ierr = VecDestroy(phi); CHKERRXX(ierr); }

  // buil computational grid data at time (n - 1)
  p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(int l = 0; l < data->max_lvl; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);
  nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1); ngbd_nm1->init_neighbors();

  /* create the initial forest at time n */
  p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = (void*) data;
  ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);
  nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  ngbd_n  = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n); ngbd_n->init_neighbors();
  ngbd_c  = new my_p4est_cell_neighbors_t(hierarchy_n);
  faces   = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c, true);
  P4EST_ASSERT(faces->finest_face_neighborhoods_are_valid());

  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, initial_level_set, phi);
}

void build_interface_capturing_grid(p4est_t* p4est_n, my_p4est_brick_t* brick, const splitting_criteria_cf_t* subrefined_data,
                                    p4est_t* &subrefined_p4est, p4est_ghost_t* &subrefined_ghost, p4est_nodes_t* &subrefined_nodes,
                                    my_p4est_hierarchy_t* &subrefined_hierarchy, my_p4est_node_neighbors_t* &subrefined_ngbd_n, Vec &subrefined_phi)
{
  PetscErrorCode ierr;
  // clear inout data, if needed
  if(subrefined_p4est != NULL)
    p4est_destroy(subrefined_p4est);
  if(subrefined_ghost != NULL)
    p4est_ghost_destroy(subrefined_ghost);
  if(subrefined_nodes != NULL)
    p4est_nodes_destroy(subrefined_nodes);
  if(subrefined_hierarchy != NULL)
    delete subrefined_hierarchy;
  if(subrefined_ngbd_n != NULL)
    delete subrefined_ngbd_n;
  if(subrefined_phi != NULL) {
    ierr = VecDestroy(subrefined_phi); CHKERRXX(ierr); }

  subrefined_p4est = p4est_copy(p4est_n, P4EST_FALSE);
  subrefined_p4est->user_pointer = (void*) subrefined_data;
  while (find_max_level(subrefined_p4est) < (int8_t) subrefined_data->max_lvl) {
    p4est_refine(subrefined_p4est, P4EST_FALSE, refine_levelset_cf, NULL);
  }
  subrefined_ghost = my_p4est_ghost_new(subrefined_p4est, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(subrefined_p4est, subrefined_ghost);
  subrefined_hierarchy = new my_p4est_hierarchy_t(subrefined_p4est, subrefined_ghost, brick);
  subrefined_nodes = my_p4est_nodes_new(subrefined_p4est, subrefined_ghost);
  subrefined_ngbd_n = new my_p4est_node_neighbors_t(subrefined_hierarchy, subrefined_nodes); subrefined_ngbd_n->init_neighbors();

  ierr = VecCreateGhostNodes(subrefined_p4est, subrefined_nodes, &subrefined_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(subrefined_p4est, subrefined_nodes, initial_level_set, subrefined_phi);
  return;
}

void truncate_exportation_file_up_to_tstart(const double& tstart, const string &filename, const u_int& n_extra_values_exported_per_line)
{
  FILE* fp = fopen(filename.c_str(), "r+");
  char* read_line = NULL;
  size_t len = 0;
  ssize_t len_read;
  long size_to_keep = 0;
  if(((len_read = getline(&read_line, &len, fp)) != -1))
    size_to_keep += (long) len_read;
  else
    throw std::runtime_error("truncate_exportation_file_up_to_tstart: couldn't read the first header line of " + filename);
  string read_format = "%lg";
  for (u_int k = 0; k < n_extra_values_exported_per_line; ++k)
    read_format += " %*g";
  double time, time_nm1;
  double dt = 0.0;
  bool not_first_line = false;
  while ((len_read = getline(&read_line, &len, fp)) != -1) {
    if(not_first_line)
      time_nm1 = time;
    sscanf(read_line, read_format.c_str(), &time);
    if(not_first_line)
      dt = time - time_nm1;
    if(time <= tstart + 0.1*dt) // +0.1*dt to avoid roundoff errors when exporting the data
      size_to_keep += (long) len_read;
    else
      break;
    not_first_line = true;
  }
  fclose(fp);
  if(read_line)
    free(read_line);
  if(truncate(filename.c_str(), size_to_keep))
    throw std::runtime_error("truncate_exportation_file_up_to_tstart: couldn't truncate " + filename);
}

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
      fprintf(fp_results, "%% tn | Re | We | volume \n");
      fclose(fp_results);
    }
    else
      truncate_exportation_file_up_to_tstart(tstart, datafile, 3);

    char liveplot[PATH_MAX];
    sprintf(liveplot, "%s/live_monitor.gnu", results_dir.c_str());
    if(!file_exists(liveplot))
    {
      FILE* fp_liveplot = fopen(liveplot, "w");
      if(fp_liveplot == NULL)
        throw std::runtime_error("initialize_exportations: could not open file for liveplot.");
      fprintf(fp_liveplot, "set term wxt 0 position 0,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Nondimensional Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Rising-velocity Reynolds number \" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot \"monitoring_results.dat\" using 1:2 notitle with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 1 position 800,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Nondimensional Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Rising-velocity Weber number \" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "plot  \"monitoring_results.dat\" using 1:3 notitle with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 2 position 1600,50 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Nondimensional Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Relative difference in bubble volume (in %%)\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "col = 4\n");
      fprintf(fp_liveplot, "row = 1\n");
      fprintf(fp_liveplot, "stats 'monitoring_results.dat' every ::row::row using col nooutput\n");
      fprintf(fp_liveplot, "original_volume = STATS_min\n");
      fprintf(fp_liveplot, "plot  \"monitoring_results.dat\" using 1:(100 * ($4 - original_volume)/original_volume) with lines lw 3\n");
      fprintf(fp_liveplot, "pause 4\n");
      fprintf(fp_liveplot, "reread");
      fclose(fp_liveplot);
    }
  }
  return;
}

void export_results(const double& nondimensional_tn, const double& Re, const double& We, const double& bubble_volume, const string& datafile)
{
  FILE* fp_data = fopen(datafile.c_str(), "a");
  if(fp_data == NULL)
    throw std::invalid_argument("main for buoyant bubble: could not open file for output of raw results.");
  fprintf(fp_data, "%g %g %g %g \n", nondimensional_tn, Re, We, bubble_volume);
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
  const double xyz_min[P4EST_DIM]       = { DIM(-0.5*domain_size[0], -0.15*domain_size[1], -0.5*domain_size[2]) };
  const double xyz_max[P4EST_DIM]       = { DIM( 0.5*domain_size[0],  0.85*domain_size[1],  0.5*domain_size[2]) };
  const int periodic[P4EST_DIM]         = { DIM(cmd.get<bool>("xperiodic", default_periodic[0]),
                                            cmd.get<bool>("yperiodic", default_periodic[1]),
                                            cmd.get<bool>("zperiodic", default_periodic[2]))};
  const double dxmin                    = MIN(DIM(domain_size[0]/ntree_xyz[0], domain_size[1]/ntree_xyz[1], domain_size[2]/ntree_xyz[2]))/((double) (1 << lmax));
  const double uniform_band_in_dxmin    = cmd.get<double> ("uniform_band",      default_uniform_band_to_radius*initial_bubble_diameter*0.5/dxmin);
  const double rho_minus                = rho_plus/cmd.get<double>("ratio_rho", default_ratio_rho);
  const double mu_minus                 = mu_plus/cmd.get<double> ("ratio_mu",  default_ratio_mu);
  const double Morton                   = cmd.get<double> ("Morton", default_Morton);
  const double Eotvos                   = cmd.get<double> ("Eotvos", default_Eotvos);
  const double surface_tension          = (SQR(mu_plus)/(rho_plus*initial_bubble_diameter))*sqrt(Eotvos/Morton);
  const bool use_second_order_theta     = cmd.get<bool>   ("second_order_ls",   default_use_second_order_theta);
  const int sl_order                    = cmd.get<int>    ("sl_order",          default_sl_order);
  const int sl_order_interface          = cmd.get<int>    ("sl_order_itfc",     default_sl_order_itfc);
  const double cfl_advection            = cmd.get<double> ("cfl_advection",     default_cfl_advection);
  const double cfl_visco_capillary      = cmd.get<double> ("cfl_visco_capillary",default_cfl_visco_capillary);
  const double cfl_capillary            = cmd.get<double> ("cfl_capillary",     default_cfl_capillary);
  const string root_export_folder       = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR")));
  const jump_solver_tag projection_solver_to_use = cmd.get<jump_solver_tag>("projection", default_projection);


  const interpolation_method phi_interp = cmd.get<interpolation_method>("phi_interp", default_interp_method_phi);
  const bool use_subrefinement          = cmd.get<bool>("subrefinement", default_subrefinement);

  if(initial_bubble_diameter > MIN(DIM(domain_size[0], domain_size[1], domain_size[2])))
    throw std::invalid_argument("main for buoyant bubble: create_solver_from_scratch: invalid bubble radius (the bubble is larger than the computational domain in some direction).");

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

  build_computational_grids(mpi, brick, connectivity, data,
                            p4est_nm1, ghost_nm1, nodes_nm1, hierarchy_nm1, ngbd_nm1,
                            p4est_n, ghost_n, nodes_n, hierarchy_n, ngbd_n,
                            ngbd_c, faces, phi);
  Vec interface_capturing_phi = phi; // no creation here, just a renamed pointer to streamline the logic

  if(use_subrefinement)
  {
    // build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
    splitting_criteria_cf_t* subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, &initial_level_set);
    build_interface_capturing_grid(p4est_n, brick, subrefined_data,
                                   subrefined_p4est, subrefined_ghost, subrefined_nodes, subrefined_hierarchy, subrefined_ngbd_n, subrefined_phi);
    interface_capturing_phi = subrefined_phi;
  }

  CF_DIM *vnm1_minus[P4EST_DIM] = { DIM(&zero_cf, &zero_cf, &zero_cf) };
  CF_DIM *vnm1_plus[P4EST_DIM]  = { DIM(&zero_cf, &zero_cf, &zero_cf) };
  CF_DIM *vn_minus[P4EST_DIM]   = { DIM(&zero_cf, &zero_cf, &zero_cf) };
  CF_DIM *vn_plus[P4EST_DIM]    = { DIM(&zero_cf, &zero_cf, &zero_cf) };

  if(solver != NULL)
    delete solver;
  solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces, (use_subrefinement ? subrefined_ngbd_n : NULL));
  solver->set_phi(interface_capturing_phi, phi_interp, phi);
  solver->set_dynamic_viscosities(mu_minus, mu_plus);
  solver->set_densities(rho_minus, rho_plus);
  solver->set_surface_tension(surface_tension);
  solver->set_uniform_bands(uniform_band_in_dxmin, uniform_band_in_dxmin);
  solver->set_vorticity_split_threshold(vorticity_threshold);
  solver->set_cfls(cfl_advection, cfl_visco_capillary, cfl_capillary);
  solver->set_semi_lagrangian_order_advection(sl_order);
  solver->set_semi_lagrangian_order_interface(sl_order_interface);
  solver->set_node_velocities(vnm1_minus, vn_minus, vnm1_plus, vn_plus);

  solver->set_projection_solver(projection_solver_to_use);
  if(use_second_order_theta)
    solver->fetch_interface_points_with_second_order_accuracy();
  return;
}

void load_solver_from_state(const mpi_environment_t &mpi, const cmdParser &cmd, double& tn,
                            my_p4est_two_phase_flows_t* &solver, my_p4est_brick_t* &brick, p4est_connectivity_t* &connectivity)
{
  const string backup_directory = cmd.get<string>("restart", "");
  if(!is_folder(backup_directory.c_str()))
    throw std::invalid_argument("load_solver_from_state: the restart path " + backup_directory + " is not an accessible directory.");

  if (solver != NULL) {
    delete solver; solver = NULL; }
  P4EST_ASSERT(solver == NULL);
  solver = new my_p4est_two_phase_flows_t(mpi, backup_directory.c_str(), tn);

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
  streamObj.str(""); streamObj << default_Eotvos;
  cmd.add_option("Eotvos", "The desired Eotvos number Eo = rho_plus*SQR(initial_bubble_diameter)*gravity/surface_tension. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_Morton;
  cmd.add_option("Morton", "The desired Morton number Mo = gravity*pow(mu_plus, 4.0)/(rho_plus*pow(surface_tension, 3.0)). Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_nondimensional_duration;
  cmd.add_option("duration", "The overall duration of the simulation (nondimendional, i.e. in units of D*mu_plus/gamma). Default duration is " + streamObj.str());
  // method-related parameters
  cmd.add_option("second_order_ls", "flag activating second order F-D interface fetching if set to true or 1. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("sl_order", "the order for the semi lagrangian advection terms, either 1 or 2, default is " + to_string(default_sl_order));
  cmd.add_option("sl_order_itfc", "the order for the semi lagrangian interface advection, either 1 or 2, default is " + to_string(default_sl_order_itfc));
  streamObj.str(""); streamObj << default_cfl_advection;
  cmd.add_option("cfl_advection", "desired advection CFL number, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_cfl_visco_capillary;
  cmd.add_option("cfl_visco_capillary", "desired visco-capillary CFL number, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_cfl_capillary;
  cmd.add_option("cfl_capillary", "desired capillary-wave CFL number, default is " + streamObj.str());
  cmd.add_option("projection", "cell-based solver to use for projection step, possible choices are 'GFM', 'xGFM' or 'FV'. Default is " + convert_to_string(default_projection));
  streamObj.str(""); streamObj << default_n_reinit;
  cmd.add_option("n_reinit", "number of solver iterations between two reinitializations of the levelset. Default is " + streamObj.str());
  // output-control parameters
  cmd.add_option("save_vtk", "flag activating  the exportation of vtk visualization files if set to true or 1. Default behavior is " + string(default_save_vtk ? "with" : "without") + " vtk exportation");
  streamObj.str(""); streamObj << default_nondimensional_vtk_dt;
  cmd.add_option("vtk_dt", "vtk_dt = time step between two vtk exportation (nondimendional), default is " + streamObj.str());
  cmd.add_option("work_dir", "root exportation directory, subfolders will be created therein (read from input if not defined otherwise in the environment variable OUT_DIR). \n\tThis is required for vtk files and for data files. Default is " + default_work_folder);
  streamObj.str(""); streamObj << default_interp_method_phi;
  cmd.add_option("phi_interp", "interpolation method for the node-sampled levelset function. Default is " + streamObj.str());
  cmd.add_option("subrefinement", "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");
  streamObj.str(""); streamObj << default_nondimensional_save_state_dt;
  cmd.add_option("save_state_dt", "if save_nstates > 0, the solver state is saved every save_state_dt*(D*mu_plus/gamma) time increments in backup_ subfolders. Default is " + streamObj.str());
  cmd.add_option("save_nstates",  "determines how many solver states must be memorized in backup_ folders (default is " + to_string(default_save_nstates) + ")");

  if(cmd.parse(argc, argv, main_description))
    return 0;

  const string root_export_folder = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR")));
  const int niter_reinit = cmd.get<int> ("n_reinit", default_n_reinit);

  PetscErrorCode ierr;

  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
  my_p4est_brick_t *brick                           = NULL;
  p4est_connectivity_t* connectivity                = NULL;
  double tn, dt;

  if(cmd.contains("restart"))
  {
    load_solver_from_state(mpi, cmd, tn, two_phase_flow_solver, brick, connectivity);
    dt = two_phase_flow_solver->get_dt();
  }
  else
  {
    create_solver_from_scratch(mpi, cmd, two_phase_flow_solver, brick, connectivity);
    tn = 0.0;// no restart for now, so we assume we start from 0.0
    two_phase_flow_solver->compute_dt();
    dt = two_phase_flow_solver->get_dt();
    two_phase_flow_solver->set_dt(dt, dt);
  }

  const double* xyz_min = two_phase_flow_solver->get_brick()->xyz_min;
  const double* xyz_max = two_phase_flow_solver->get_brick()->xyz_max;
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  pressure_wall_bc_type_t pressure_wall_type(xyz_min, xyz_max);
  velocity_wall_bc_type_t velocity_wall_type(xyz_min, xyz_max);
  BoundaryConditionsDIM bc_p; bc_p.setWallTypes(pressure_wall_type); bc_p.setWallValues(zero_cf);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    bc_v[dim].setWallTypes(velocity_wall_type); bc_v[dim].setWallValues(zero_cf);
  }

  cf_const_t body_force_x(0.0);
  cf_const_t body_force_y(-cmd.get<double>("Morton", default_Morton)*two_phase_flow_solver->get_rho_plus()*pow(two_phase_flow_solver->get_surface_tension(), 3.0)/pow(two_phase_flow_solver->get_mu_plus(), 4.0));
#ifdef P4_TO_P8
  cf_const_t body_force_z(0.0);
#endif
  CF_DIM* body_force[P4EST_DIM] = {DIM(&body_force_x, &body_force_y, &body_force_z)};
  two_phase_flow_solver->set_external_forces_per_unit_mass(body_force);
  two_phase_flow_solver->set_bc(bc_v, &bc_p);

  splitting_criteria_t* data            = (splitting_criteria_t*) (two_phase_flow_solver->get_p4est_n()->user_pointer); // to delete it appropriately, eventually
  splitting_criteria_t* subrefined_data = (two_phase_flow_solver->get_fine_p4est_n() != NULL ? (splitting_criteria_t*) two_phase_flow_solver->get_fine_p4est_n()->user_pointer : NULL); // same, to delete it appropriately, eventually

  // make sure we're doing consistent stuff
//  if(!two_phase_flow_solver->viscosities_are_equal())
//    throw std::runtime_error("main for static bubble: this test is designed for mu_minus == mu_plus");


  const double time_unit  = two_phase_flow_solver->get_mu_plus()*initial_bubble_diameter/two_phase_flow_solver->get_surface_tension();
  const double duration   = cmd.get<double> ("duration",      default_nondimensional_duration)*time_unit;
  const double vtk_dt     = cmd.get<double> ("vtk_dt",        default_nondimensional_vtk_dt)*time_unit;
  const bool save_vtk     = cmd.get<bool>   ("save_vtk",      default_save_vtk);
  const int save_nstates  = cmd.get<int>    ("save_nstates",  default_save_nstates);
  const double save_state_dt = cmd.get<double> ("save_state_dt", default_nondimensional_save_state_dt)*time_unit;
  if(vtk_dt <= 0.0)
    throw std::invalid_argument("main for static bubble: the value of vtk_dt must be strictly positive.");
  if(save_vtk && !cmd.contains("retsart"))
  {
    dt = MIN(dt, vtk_dt);
    two_phase_flow_solver->set_dt(dt, dt);
  }

  const splitting_criteria_t* sp = (splitting_criteria_t*) two_phase_flow_solver->get_p4est_n()->user_pointer;
  streamObj.str("");
  streamObj << "mu_ratio_" << two_phase_flow_solver->get_mu_plus()/two_phase_flow_solver->get_mu_minus()
            << "_rho_ratio_" << two_phase_flow_solver->get_rho_plus()/two_phase_flow_solver->get_rho_minus()
            << "_Eotvos_" << two_phase_flow_solver->get_rho_plus()*SQR(initial_bubble_diameter)*fabs(body_force_y(0.0, 0.0))/two_phase_flow_solver->get_surface_tension()
            << "_Morton_" << fabs(body_force_y(0.0, 0.0))*pow(two_phase_flow_solver->get_mu_plus(), 4.0)/(two_phase_flow_solver->get_rho_plus()*pow(two_phase_flow_solver->get_surface_tension(), 3.0));
  const string export_dir   = root_export_folder + (root_export_folder.back() == '/' ? "" : "/") + streamObj.str() + "/lmin_" + to_string(sp->min_lvl) + "_lmax_" + to_string(sp->max_lvl);
  const string vtk_dir      = export_dir + "/vtu";
  const string results_dir  = export_dir + "/results";
  if(create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for buoyant bubble: could not create exportation directory " + export_dir);
  if(create_directory(results_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for buoyant bubble: could not create directory for exportation of results, i.e., " + results_dir);
  if(save_vtk && create_directory(vtk_dir, mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for buoyant bubble: could not create directory for visualization files, i.e., " + vtk_dir);

  string datafile;
  initialize_exportations(tn, mpi, results_dir, datafile);

  int iter = 0;
  int vtk_idx = (cmd.contains("restart") ? (int) floor(tn/vtk_dt) : -1);
  int backup_time_idx = (int) floor(tn/save_state_dt);
  if(mpi.rank() == 0 && !cmd.contains("restart"))
#ifdef P4_TO_P8
    export_results(tn, 0.0, 0.0, M_PI*pow(initial_bubble_diameter, 3.0)/6.0, datafile);
#else
    export_results(tn, 0.0, 0.0, M_PI*0.25*SQR(initial_bubble_diameter), datafile);
#endif

  while(tn + 0.01*dt < duration)
  {
    if(iter > 0)
    {
      two_phase_flow_solver->compute_dt();
      dt = two_phase_flow_solver->get_dt();
      two_phase_flow_solver->update_from_tn_to_tnp1(iter%niter_reinit == 0);
    }

    if(save_nstates > 0 && (int) floor(tn/save_state_dt) != backup_time_idx)
    {
      backup_time_idx = (int) floor(tn/save_state_dt);
      two_phase_flow_solver->save_state(export_dir.c_str(), tn, save_nstates);
    }

    two_phase_flow_solver->solve_viscosity();
    two_phase_flow_solver->solve_projection();
    two_phase_flow_solver->compute_velocities_at_nodes();

    two_phase_flow_solver->set_interface_velocity_np1(); // so that you can visualize it in the vtk files

    if(save_vtk && (int) floor((tn + dt)/vtk_dt) != vtk_idx)
    {
      vtk_idx = (int) floor((tn + dt)/vtk_dt);
      two_phase_flow_solver->save_vtk(vtk_dir, vtk_idx);
    }

    if(two_phase_flow_solver->get_max_velocity() > 100.0)
    {
      if(save_vtk)
        two_phase_flow_solver->save_vtk(vtk_dir, ++vtk_idx);
      ierr = PetscPrintf(mpi.comm(), "The simulation blew up... Maximum velocity found to be %g\n", two_phase_flow_solver->get_max_velocity()); CHKERRXX(ierr);
      break;
    }

    tn += dt;
    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n",
                       iter, tn, 100*tn/duration, two_phase_flow_solver->get_max_velocity(), two_phase_flow_solver->get_p4est_n()->global_num_quadrants); CHKERRXX(ierr);
    double avg_itfc_velocity[P4EST_DIM];
    const double bubble_volume = two_phase_flow_solver->volume_in_negative_domain();
    two_phase_flow_solver->get_average_interface_velocity(avg_itfc_velocity);
    if(mpi.rank() == 0)
      export_results(tn/time_unit, avg_itfc_velocity[1]*two_phase_flow_solver->get_rho_plus()*initial_bubble_diameter/two_phase_flow_solver->get_mu_plus(),
          two_phase_flow_solver->get_rho_plus()*SQR(avg_itfc_velocity[1])*initial_bubble_diameter/two_phase_flow_solver->get_surface_tension(),
          bubble_volume, datafile);
    iter++;
  }
  ierr = PetscPrintf(mpi.comm(), "Gracefully finishing up now\n");

  delete two_phase_flow_solver;
  my_p4est_brick_destroy(connectivity, brick);
  delete brick;
  delete data;
  delete subrefined_data;

  return 0;
}
