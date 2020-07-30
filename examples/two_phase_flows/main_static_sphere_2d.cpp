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
    string("In this example, we test the ability of the two-phase flow solver to capture a static bubble in a stable and accurate way. \n")
    + string("The static bubble is initialized at the center of a (square/cube) computational domain, in a quiescent environment, and \n")
    + string("no external force is considered. The two fluids have the same mass densities and viscosities, this test is a sanity check for\n")
    + string("the ability of the solver to capture stationary Laplace solutions. The user can choose between periodic boundary conditions or \n")
    + string("no-slip boundary condition on any of the borders of the computational domain. \n")
    + string("Developer: Raphael Egan (raphaelegan@ucsb.edu), 2019-2020\n");

std::istream& operator>> (std::istream& is, poisson_jump_cell_solver_tag& solver)
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

const int default_lmin = 4;
const int default_lmax = 4;
const int default_ntree = 1;
const interpolation_method default_interp_method_phi = linear;
const bool default_use_second_order_theta = false;
const bool default_subrefinement = false;
const double default_box_size = 2.5;
const bool default_periodic[P4EST_DIM] = {DIM(false, false, false)};
const double default_bubble_radius = 0.5;
const double default_vorticity_threshold = DBL_MAX;
const double default_uniform_band_to_radius = 0.15;
const double default_mass_density = 1.0;
const double default_viscosity = 1.0/12000.0;
const double default_surface_tension = 1.0/12000.0;
const double default_duration_nondimensional = 250.0;
const double default_vtk_dt_nondimensional = 1.0;
const int default_sl_order = 1;
const double default_cfl_advection = 1.0;
const double default_cfl_capillary = 0.5;
const poisson_jump_cell_solver_tag default_projection = GFM;
const bool default_save_vtk = true;

#if defined(STAMPEDE)
const string default_work_folder = "/scratch/04965/tg842642/two_phase_flow/static_bubble_" + to_string(P4EST_DIM) + "D";
#elif defined(POD_CLUSTER)
const string default_work_folder = "/scratch/regan/two_phase_flow/static_bubble_" + to_string(P4EST_DIM) + "D";
#else
const string default_work_folder = "/home/regan/workspace/projects/two_phase_flow/static_bubble_" + to_string(P4EST_DIM) + "D";
#endif

class LEVEL_SET: public CF_DIM {
  const double bubble_radius;
public:
  LEVEL_SET(const double& bubble_radius_) : bubble_radius(bubble_radius_) { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(SUMD(SQR(x), SQR(y), SQR(z))) - bubble_radius;
  }
};

struct BCWALLTYPE_P : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLTYPE_VELOCITY : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_walltype_velocity;

void build_computational_grids(const mpi_environment_t &mpi, my_p4est_brick_t *brick, p4est_connectivity_t* connectivity, const splitting_criteria_cf_and_uniform_band_t* data, const LEVEL_SET& level_set,
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
  sample_cf_on_nodes(p4est_n, nodes_n, level_set, phi);
}

void build_interface_capturing_grid(p4est_t* p4est_n, my_p4est_brick_t* brick, const splitting_criteria_cf_t* subrefined_data, const LEVEL_SET& level_set,
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
  sample_cf_on_nodes(subrefined_p4est, subrefined_nodes, level_set, subrefined_phi);
  return;
}

//void truncate_exportation_file_up_to_tstart(const string &filename, const unsigned int& n_extra_values_exported_per_line)
//{
//  FILE* fp = fopen(filename.c_str(), "r+");
//  char* read_line = NULL;
//  size_t len = 0;
//  ssize_t len_read;
//  long size_to_keep = 0;
//  if(((len_read = getline(&read_line, &len, fp)) != -1))
//    size_to_keep += (long) len_read;
//  else
//    throw std::runtime_error("truncate_exportation_file: couldn't read the first header line of " + filename);
//  string read_format = "%lg";
//  for (unsigned int k = 0; k < n_extra_values_exported_per_line; ++k)
//    read_format += " %*g";
//  double time, time_nm1;
//  double dt = 0.0;
//  bool not_first_line = false;
//  while ((len_read = getline(&read_line, &len, fp)) != -1) {
//    if(not_first_line)
//      time_nm1 = time;
//    sscanf(read_line, read_format.c_str(), &time);
//    if(not_first_line)
//      dt = time - time_nm1;
//    if(time <= tstart + 0.1*dt) // +0.1*dt to avoid roundoff errors when exporting the data
//      size_to_keep += (long) len_read;
//    else
//      break;
//    not_first_line = true;
//  }
//  fclose(fp);
//  if(read_line)
//    free(read_line);
//  if(truncate(filename.c_str(), size_to_keep))
//    throw std::runtime_error("simulation_setup::truncate_exportation_file: couldn't truncate " + filename);
//}

void initialize_exportations(const mpi_environment_t& mpi, const string& results_dir,
                             string& datafile_parasitic_current, string& datafile_volume_bubble)
{
  datafile_parasitic_current  = results_dir + "/magnitude_parasitic_current.dat";
  datafile_volume_bubble      = results_dir + "/volume_bubble.dat";
  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(), "Saving magnitude of parasitic currents in ... %s\n", datafile_parasitic_current.c_str()); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi.comm(), "Saving bubble volume in ... %s\n", datafile_volume_bubble.c_str()); CHKERRXX(ierr);

  if(mpi.rank() == 0)
  {
    FILE* fp_parasitic_currents = fopen(datafile_parasitic_current.c_str(), "w");
    if(fp_parasitic_currents == NULL)
      throw std::runtime_error("initialize_exportations: could not open file for output of magnitude of parasitic current.");
    fprintf(fp_parasitic_currents, "%% tn | nondimensional parasitic current \n");
    fclose(fp_parasitic_currents);

    FILE* fp_bubble_voume = fopen(datafile_volume_bubble.c_str(), "w");
    if(fp_bubble_voume == NULL)
      throw std::runtime_error("initialize_exportations: could not open file for output of bubble volume drift.");
    fprintf(fp_bubble_voume, "%% tn | volume \n");
    fclose(fp_bubble_voume);

    char liveplot[PATH_MAX];
    sprintf(liveplot, "%s/liveplot.gnu", results_dir.c_str());
    if(!file_exists(liveplot))
    {
      FILE* fp_liveplot = fopen(liveplot, "w");
      if(fp_liveplot == NULL)
        throw std::runtime_error("initialize_exportations: could not open file for liveplot.");
      fprintf(fp_liveplot, "set term wxt 0 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Nondimensional Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Nondimensional U\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set logscale y\n");
      fprintf(fp_liveplot, "plot \"magnitude_parasitic_current.dat\" using 1:2 title 'parasitic currents' with lines lw 3\n");
      fprintf(fp_liveplot, "set term wxt 1 noraise\n");
      fprintf(fp_liveplot, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set xlabel \"Nondimensional Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "set ylabel \"Error in bubble volume (%%)\" font \"Arial,14\"\n");
      fprintf(fp_liveplot, "unset logscale y\n");
      fprintf(fp_liveplot, "plot  \"volume_bubble.dat\" using 1:(100 * $2) title 'bubble volume error' with lines lw 3\n");
      fprintf(fp_liveplot, "pause 4\n");
      fprintf(fp_liveplot, "reread");
      fclose(fp_liveplot);
    }
  }
  return;
}

void export_results(const double& nondimensional_tn, const double& magnitude_nondimensional_parasitic_current, const double& error_bubble_volume,
                    const string& parasitic_current_datafile, const string& volume_datafile)
{
  FILE* fp_parasitic_current = fopen(parasitic_current_datafile.c_str(), "a");
  if(fp_parasitic_current == NULL)
    throw std::invalid_argument("main for static bubble: could not open file for output of parasitic current.");
  fprintf(fp_parasitic_current, "%g %g \n", nondimensional_tn, magnitude_nondimensional_parasitic_current);
  fclose(fp_parasitic_current);


  FILE* fp_volume = fopen(volume_datafile.c_str(), "a");
  if(fp_volume == NULL)
    throw std::invalid_argument("main for static bubble: could not open file for output of volume drift.");
  fprintf(fp_volume, "%g %g \n", nondimensional_tn, error_bubble_volume);
  fclose(fp_volume);
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;// computational grid parameters
  ostringstream streamObj;
  cmd.add_option("lmin", "min level of the trees, default is " + to_string(default_lmin));
  cmd.add_option("lmax", "max level of the trees, default is " + to_string(default_lmax));
  streamObj.str(""); streamObj << default_vorticity_threshold;
  cmd.add_option("thresh", "the vorticity-based threshold used for the refinement criteria, default is " + streamObj.str());
  streamObj.str(""); streamObj << 100.0*default_uniform_band_to_radius;
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is such that " + streamObj.str() + "% of the bubble radius is covered");
  cmd.add_option("ntree", "number of trees in the macromesh, in every cartesian direction. The default value is " + to_string(default_ntree));
  streamObj.str(""); streamObj << default_box_size;
  cmd.add_option("box_size", "side length of the computational domain. The default value is " + streamObj.str());
  cmd.add_option("xperiodic", "activates periodicity along x, if present. Default is " + string(default_periodic[0] ? "" : "not") + " x-periodic.");
  cmd.add_option("yperiodic", "activates periodicity along y, if present. Default is " + string(default_periodic[1] ? "" : "not") + " y-periodic.");
#ifdef P4_TO_P8
  cmd.add_option("zperiodic", "activates periodicity along z, if present. Default is " + string(default_periodic[2] ? "" : "not") + " z-periodic.");
#endif
  // physical parameters for the simulations
  streamObj.str(""); streamObj << default_bubble_radius;
  cmd.add_option("radius", "The (initial) radius of the bubble. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_mass_density;
  cmd.add_option("mass_density", "The mass density of the two fluids to be considered. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_viscosity;
  cmd.add_option("viscosity", "The viscosity of the two fluids to be considered. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_surface_tension;
  cmd.add_option("surface_tension", "The surface tension viscosity of the two fluids to be considered. Default value is " + streamObj.str());
  streamObj.str(""); streamObj << default_duration_nondimensional;
  cmd.add_option("duration", "The overall duration of the simulation in characteristic time units (D*mu/gamma). Default duration is " + streamObj.str());
  // method-related parameters
  cmd.add_option("second_order_ls", "activate second order F-D interface fetching if present. Default is " + string(default_use_second_order_theta ? "true" : "false"));
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 or 2, default is " + to_string(default_sl_order));
  streamObj.str(""); streamObj << default_cfl_advection;
  cmd.add_option("cfl_advection", "desired advection CFL number, default is " + streamObj.str());
  streamObj.str(""); streamObj << default_cfl_capillary;
  cmd.add_option("cfl_capillary", "desired capillary-wave CFL number, default is " + streamObj.str());
  cmd.add_option("projection", "cell-based solver to use for projection step, possible choices are 'GFM', 'xGFM' or 'FV'. Default is " + convert_to_string(default_projection));
  // output-control parameters
  cmd.add_option("save_vtk", "activates the exportatino of vtk visualization files if present. Default behavior is " + string(default_save_vtk ? "with" : "without") + " vtk exportation");
  streamObj.str(""); streamObj << default_vtk_dt_nondimensional;
  cmd.add_option("vtk_dt", "vtk_dt = time step between two vtk exportation in characteristic time units (D*mu/gamma), default is " + streamObj.str());
  cmd.add_option("work_dir", "root exportation directory, subfolders will be created therein (read from input if not defined otherwise in the environment variable OUT_DIR). \n\tThis is required for vtk files and for data files. Default is " + default_work_folder);
  streamObj.str(""); streamObj << default_interp_method_phi;
  cmd.add_option("phi_interp", "interpolation method for the node-sampled levelset function. Default is " + streamObj.str());
  cmd.add_option("subrefinement", "flag activating the usage of a subrefined interface-capturing grid if set to true or 1, deactivating if set to false or 0. Default is " + string(default_subrefinement ? "with" : "without") + " subrefinement");

  if(cmd.parse(argc, argv, main_description))
    return 0;

  const int lmin                        = cmd.get<int>    ("lmin",            default_lmin);
  const int lmax                        = cmd.get<int>    ("lmax",            default_lmax);
  const double vorticity_threshold      = cmd.get<double> ("thresh",          default_vorticity_threshold);
  const int ntree                       = cmd.get<int>    ("ntree",           default_ntree);
  const int ntree_xyz[P4EST_DIM]        = {DIM(ntree, ntree, ntree)};
  const double box_size                 = cmd.get<double> ("box_size",        default_box_size);
  const double xyz_min[P4EST_DIM]       = { DIM(-0.5*box_size, -0.5*box_size, -0.5*box_size) };
  const double xyz_max[P4EST_DIM]       = { DIM( 0.5*box_size,  0.5*box_size,  0.5*box_size) };
  const int periodic[P4EST_DIM]         = { DIM((default_periodic[0] || cmd.contains("xperiodic") ? 1 : 0),
                                            (default_periodic[1] || cmd.contains("yperiodic") ? 1 : 0),
                                            (default_periodic[2] || cmd.contains("zperiodic") ? 1 : 0))};
  const double dxmin                    = box_size/(((double) ntree)*((double) (1 << lmax)));
  const double bubble_radius            = cmd.get<double> ("radius",          default_bubble_radius);
  const double uniform_band_in_dxmin    = cmd.get<double> ("uniform_band",    default_uniform_band_to_radius*bubble_radius/dxmin);
  const double mass_density             = cmd.get<double> ("mass_density",    default_mass_density);
  const double viscosity                = cmd.get<double> ("viscosity",       default_viscosity);
  const double surface_tension          = cmd.get<double> ("surface_tension", default_surface_tension);
  const double characteristic_time_unit = (2.0*bubble_radius*viscosity/surface_tension);
  const double duration                 = cmd.get<double> ("duration",        default_duration_nondimensional)*characteristic_time_unit;
  const bool use_second_order_theta     = default_use_second_order_theta || cmd.contains("second_order_ls");
  const int sl_order                    = cmd.get<int>    ("sl_order",        default_sl_order);
  const double cfl_advection            = cmd.get<double> ("cfl_advection",   default_cfl_advection);
  const double cfl_capillary            = cmd.get<double> ("cfl_capillary",   default_cfl_capillary);
  const bool save_vtk                   = default_save_vtk || cmd.contains("save_vtk");
  const double vtk_dt                   = cmd.get<double> ("vtk_dt",          default_vtk_dt_nondimensional)*characteristic_time_unit;
  const string root_export_folder       = cmd.get<std::string>("work_dir", (getenv("OUT_DIR") == NULL ? default_work_folder : getenv("OUT_DIR")));
  const poisson_jump_cell_solver_tag projection_solver_to_use = cmd.get<poisson_jump_cell_solver_tag>("projection", default_projection);

  const interpolation_method phi_interp = cmd.get<interpolation_method>("phi_interp", default_interp_method_phi);
  const bool use_subrefinement          = cmd.get<bool>("subrefinement", default_subrefinement);

  if(2.0*bubble_radius > box_size)
    throw std::invalid_argument("main for static bubble: invalid bubble radius (the bubble is larger than the computational domain).");
  if(vtk_dt <= 0.0)
    throw std::invalid_argument("main for static bubble: the value of vtk_dt must be strictly positive.");
  const double inv_Oh_square = surface_tension*mass_density*2.0*bubble_radius/SQR(viscosity);
  streamObj.str(""); streamObj << inv_Oh_square;
  const string export_dir = root_export_folder + (root_export_folder.back() == '/' ? "" : "/") + "inv_Oh_squared_" + streamObj.str() + "/lmin_" + to_string(lmin) + "_lmax_" + to_string(lmax);
  const string vtk_dir    = export_dir + "/vtu";
  const string results_dir   = export_dir + "/results";
  if(create_directory(export_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for static bubble: could not create exportation directory " + export_dir);
  if(create_directory(results_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for static bubble: could not create exportation directory " + results_dir);
  if(save_vtk && create_directory(vtk_dir, mpi.rank(), mpi.comm()))
    throw std::runtime_error("main for static bubble: could not create directory for visualization files, i.e., " + vtk_dir);

  PetscErrorCode ierr;
  my_p4est_brick_t brick;
  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;

  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p; bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(zero_cf);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    bc_v[dim].setWallTypes(bc_walltype_velocity); bc_v[dim].setWallValues(zero_cf);
  }
  CF_DIM *external_force_per_unit_mass[P4EST_DIM] = { DIM(&zero_cf, &zero_cf, &zero_cf) };

  LEVEL_SET level_set(bubble_radius);
#ifdef P4_TO_P8
  const double exact_bubble_volume = 4.0*M_PI*pow(bubble_radius, 3.0)/3.0;
#else
  const double exact_bubble_volume = M_PI*SQR(bubble_radius);
#endif
  p4est_connectivity_t *connectivity = my_p4est_brick_new(ntree_xyz, xyz_min, xyz_max, &brick, periodic);
  splitting_criteria_cf_and_uniform_band_t* data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band_in_dxmin);

  splitting_criteria_cf_t* subrefined_data = NULL;
  p4est_t                       *p4est_nm1      = NULL, *p4est_n      = NULL, *subrefined_p4est     = NULL;
  p4est_ghost_t                 *ghost_nm1      = NULL, *ghost_n      = NULL, *subrefined_ghost     = NULL;
  p4est_nodes_t                 *nodes_nm1      = NULL, *nodes_n      = NULL, *subrefined_nodes     = NULL;
  my_p4est_hierarchy_t          *hierarchy_nm1  = NULL, *hierarchy_n  = NULL, *subrefined_hierarchy = NULL;
  my_p4est_node_neighbors_t     *ngbd_nm1       = NULL, *ngbd_n       = NULL, *subrefined_ngbd_n    = NULL;
  Vec                                                    phi          = NULL,  subrefined_phi       = NULL;
  my_p4est_cell_neighbors_t                             *ngbd_c       = NULL;
  my_p4est_faces_t                                      *faces        = NULL;

  build_computational_grids(mpi, &brick, connectivity, data, level_set,
                            p4est_nm1, ghost_nm1, nodes_nm1, hierarchy_nm1, ngbd_nm1,
                            p4est_n, ghost_n, nodes_n, hierarchy_n, ngbd_n,
                            ngbd_c, faces, phi);
  Vec interface_capturing_phi = phi; // no creation here, just a renamed pointer to streamline the logic

  if(use_subrefinement)
  {
    // build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
    subrefined_data = new splitting_criteria_cf_t(data->min_lvl, data->max_lvl + 1, &level_set);
    build_interface_capturing_grid(p4est_n, &brick, subrefined_data, level_set,
                                   subrefined_p4est, subrefined_ghost, subrefined_nodes, subrefined_hierarchy, subrefined_ngbd_n, subrefined_phi);
    interface_capturing_phi = subrefined_phi;
  }

  CF_DIM *vnm1_minus[P4EST_DIM] = { DIM(&zero_cf, &zero_cf, &zero_cf) };
  CF_DIM *vnm1_plus[P4EST_DIM]  = { DIM(&zero_cf, &zero_cf, &zero_cf) };
  CF_DIM *vn_minus[P4EST_DIM]   = { DIM(&zero_cf, &zero_cf, &zero_cf) };
  CF_DIM *vn_plus[P4EST_DIM]    = { DIM(&zero_cf, &zero_cf, &zero_cf) };

  two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces, (use_subrefinement ? subrefined_ngbd_n : NULL));
  two_phase_flow_solver->set_phi(interface_capturing_phi, phi_interp, phi);
  two_phase_flow_solver->set_dynamic_viscosities(viscosity, viscosity);
  two_phase_flow_solver->set_densities(mass_density, mass_density);
  two_phase_flow_solver->set_surface_tension(surface_tension);
  two_phase_flow_solver->set_uniform_bands(uniform_band_in_dxmin, uniform_band_in_dxmin);
  two_phase_flow_solver->set_vorticity_split_threshold(vorticity_threshold);
  two_phase_flow_solver->set_cfls(cfl_advection, cfl_capillary);
  two_phase_flow_solver->set_semi_lagrangian_order(sl_order);
  two_phase_flow_solver->set_node_velocities(vnm1_minus, vn_minus, vnm1_plus, vn_plus);

  double tn = 0.0; // no restart for now, so we assume we start from 0.0
  double dt = cfl_capillary*sqrt(2.0*mass_density*pow(dxmin, 3.0)/(M_PI*surface_tension));
  if(save_vtk)
    dt = MIN(dt, vtk_dt);
  two_phase_flow_solver->set_dt(dt, dt);
  two_phase_flow_solver->set_bc(bc_v, &bc_p);
  two_phase_flow_solver->set_external_forces_per_unit_mass(external_force_per_unit_mass);

  two_phase_flow_solver->set_projection_solver(projection_solver_to_use);
  if(use_second_order_theta)
    two_phase_flow_solver->fetch_interface_points_with_second_order_accuracy();

  string parasitic_current_datafile, volume_datafile;
  initialize_exportations(mpi, results_dir, parasitic_current_datafile, volume_datafile);
  int iter = 0, iter_vtk = -1, export_time_idx = -1;
  double max_nondimensional_velocity_overall = 0.0;
  double magnitude_nondimensional_parasitic_current = 0.0*viscosity/mass_density;
  double error_nondimensional_volume = (exact_bubble_volume - two_phase_flow_solver->volume_in_negative_domain())/exact_bubble_volume;
  if(mpi.rank() == 0)
    export_results(tn/characteristic_time_unit, magnitude_nondimensional_parasitic_current, error_nondimensional_volume, parasitic_current_datafile, volume_datafile);


  while(tn + 0.01*dt < duration)
  {
    if(iter > 0)
    {
      two_phase_flow_solver->compute_dt();
      dt = two_phase_flow_solver->get_dt();
      two_phase_flow_solver->update_from_tn_to_tnp1();
    }

    two_phase_flow_solver->solve_viscosity();
    two_phase_flow_solver->solve_projection();

    two_phase_flow_solver->compute_velocities_at_nodes();


    if((int) floor((tn + dt)/vtk_dt) != export_time_idx || two_phase_flow_solver->get_max_velocity() > 1.0)
    {
      export_time_idx = (int) floor((tn + dt)/vtk_dt);
      two_phase_flow_solver->save_vtk(vtk_dir, ++iter_vtk);
    }

    tn += dt;
    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n",
                       iter, tn, 100*tn/duration, two_phase_flow_solver->get_max_velocity(), two_phase_flow_solver->get_p4est_n()->global_num_quadrants); CHKERRXX(ierr);
    magnitude_nondimensional_parasitic_current = two_phase_flow_solver->get_max_velocity()*viscosity/surface_tension;
    error_nondimensional_volume = (exact_bubble_volume - two_phase_flow_solver->volume_in_negative_domain())/exact_bubble_volume;
    if(mpi.rank() == 0)
      export_results(tn/characteristic_time_unit, magnitude_nondimensional_parasitic_current, error_nondimensional_volume, parasitic_current_datafile, volume_datafile);
    max_nondimensional_velocity_overall = MAX(max_nondimensional_velocity_overall, magnitude_nondimensional_parasitic_current);

    iter++;
  }
  ierr = PetscPrintf(mpi.comm(), "Maximum value of parasitic current = %.5e\n", max_nondimensional_velocity_overall);

  delete two_phase_flow_solver;
  my_p4est_brick_destroy(connectivity, &brick);
  delete data;
  delete subrefined_data;

  return 0;
}
