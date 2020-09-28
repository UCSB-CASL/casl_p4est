/*
 * The navier stokes solver applied to the flow past a static sphere/cylinder
 *
 * run the program with the -help flag to see the available options
 */

// my_p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#else
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#endif

#include <src/Parser.h>
#include <iomanip>

#undef MIN
#undef MAX

using namespace std;

// --> extra info to be printed when -help is invoked
static const string extra_info =
      string("This program provides a general setup for simulating the flow past a static sphere in 3D, resp.\n")
    + string("past a static cylinder in 2D. The radius r0 of the sphere/cylinder is the reference length scale \n")
    + string("(i.e., set to 1.0), and the domain dimensions are length x width (x width) where length and width \n")
    + string("can be set by the user\n")
    + string("The static sphere/cylinder is located in the center of the domain's cross-section, 25%% of the domain's\n")
    + string("length away from the inflow boundary. The far-flow condition is (u, v (, w)) = (u0, 0 (,0)) where u0 is\n")
    + string("the reference velocity scale (i.e. set to 1.0). The initial conditon is a (u0, 0 (, 0)) uniform flow \n")
    + string("field (hence ignoring no-slip boundary conditions at the solid's surface at initialization).\n")
    + string("If activated by the user, smoke (passive scalar) is released into the domain from the inlet wall by a \n")
    + string("constant unit source located upstream from (and aligned with) the obstacle. \n")
    + string("The boundary conditions are:\n")
    + string("- Dirichlet (u_0, 0 (, 0)) for the velocity components on all walls but the outlet wall\n")
    + string("- homogeneous Neumann for the velocity components on the outlet wall \n")
    + string("- homogeneous Neumann for the pressure on all walls but the outlet wall \n")
    + string("- homogeneous Dirichlet for the pressure on the outlet wall \n")
    + string("- regular no-slip condition on the surface of the sphere (homogeneous Neumann for pressure and homogeneous \n")
    + string("Dirichlet for velocity components). \n\n")
    + string("By the definition of the above parameters and the Reynolds numbers, the Navier-Stokes solver is invoked \n")
    + string("with nondimensional inputs \n")
    + string("rho = 1.0, mu = 2.0*r0*u_0*rho/Re, body force per unit mass {0.0, 0.0 (, 0.0)}. \n")
    + string("Developers: cleaned up and restructured to new code features' and coding \"standards\" by Raphael Egan \n")
    + string("(raphaelegan@ucsb.edu) based on a general main file by Arthur Guittet");

#if defined(POD_CLUSTER)
const std::string default_export_dir  = "/scratch/regan/flow_past_" + string((P4EST_DIM == 3 ? "sphere" : "cylinder"));
#elif defined(STAMPEDE)
const std::string default_export_dir  = "/scratch/04965/tg842642/flow_past_" + string((P4EST_DIM == 3 ? "sphere" : "cylinder"));
#elif defined(LAPTOP)
const std::string default_export_dir  = "/home/raphael/workspace/projects/flow_past_" + string((P4EST_DIM == 3 ? "sphere" : "cylinder"));
#else
const std::string default_export_dir  = "/home/regan/workspace/projects/flow_past_" + string((P4EST_DIM == 3 ? "sphere" : "cylinder"));
#endif

const double u0 = 1.0; // reference velocity      (far-field velocity)
const double r0 = 1.0; // reference length scale  (radius of sphere/cylinder)

double Reynolds(const my_p4est_navier_stokes_t *ns) { return 2.0*ns->get_rho()*r0*u0/ns->get_mu(); }

const int default_lmin                        = 4;
const int default_lmax                        = 6;
const double default_thresh                   = 0.1;
const double default_uniform_band             = 0.5*r0;
const int default_ntree_streamwise            = 8;
const int default_ntree_spanwise              = 4;
#ifdef P4_TO_P8
const double default_length                   = 32.0;
const double default_width                    = 16.0;
#else
const double default_length                   = 64.0;
const double default_width                    = 32.0;
#endif
const double default_Re                       = 350.0;
const double default_smoke_threshold          = 0.5;

const double default_duration                 = 200.0;
const int default_sl_order                    = 2;
const double default_cfl                      = 1.0;
const double default_hodge_tol                = 1.0e-4;
const unsigned int default_n_hodge            = 10;
const hodge_control def_hodge_control         = uvw_components;
const unsigned int default_ngrid_update       = 1;
const std::string default_pc_cell             = "sor";
const std::string default_cell_solver         = "bicgstab";
const std::string default_pc_face             = "sor";
const unsigned int default_save_nstates       = 1;
#ifdef P4_TO_P8
const unsigned int default_nsurface_points    = 64;
#endif

struct domain_setup
{
  const my_p4est_brick_t *brick;
  double x_inlet() const { return brick->xyz_min[0]; }
  double x_outlet() const { return brick->xyz_max[0]; }
  double length() const { return brick->xyz_max[0] - brick->xyz_min[0]; }
  double width() const {
#ifdef P4_TO_P8
    P4EST_ASSERT(fabs((brick->xyz_max[2] - brick->xyz_min[2]) - (brick->xyz_max[1] - brick->xyz_min[1])) < EPS*MAX(brick->xyz_max[1] - brick->xyz_min[1], brick->xyz_max[2] - brick->xyz_min[2]));
#endif
    return brick->xyz_max[1] - brick->xyz_min[1];
  }
  double center(const unsigned char & dir) const { return 0.5*(brick->xyz_min[dir] + brick->xyz_max[dir]); }

  double xyz_sphere_center(const unsigned char & dir) const
  {
    return (dir == dir::x ? 0.75*x_inlet() + 0.25*x_outlet() : center(dir));
  }

  domain_setup() : brick(NULL){}
} domain;

struct simulation_setup
{
  double tstart;
  double dt;
  double tn;

  // inner convergence parameteres
  const double hodge_tol;
  const unsigned int niter_hodge_max;
  const hodge_control control_hodge;

  // simulation control
  int iter;
  const unsigned int steps_grid_update;
  const double duration;
  const bool use_adapted_dt;
  const bool with_smoke;
  const std::string des_pc_cell;
  const std::string des_solver_cell;
  const std::string des_pc_face;
  KSPType cell_solver_type;
  PCType pc_cell, pc_face;
  double Reynolds;

  // exportation
#ifdef P4_TO_P8
  const bool check_laminar_drag;
#endif
  const std::string export_dir_root;
  const bool save_vtk;
  double vtk_dt;
  const bool save_simulation_times;
  const bool save_forces;
  const bool save_timing;
  const bool save_hodge_convergence;
  const bool save_state;
#ifdef P4_TO_P8
  const bool save_surface_quantities;
  const size_t nsurface_points;
  const size_t nphi = 64;
#endif
  double dt_save_data;
  const unsigned int n_states;
  int export_vtk, save_data_idx;
  std::string export_dir, vtk_path, file_snapshot_to_time, file_forces, file_timings, file_hodge_convergence, file_surf_quantities;
  std::map<ns_task, double> global_computational_times;
  std::vector<double> hodge_convergence_checks;
  // for exportation of surface quantities
#ifdef P4_TO_P8
  std::vector<double> inclination_angle;              // inclination angles corresponding to the exported surface quantities, as measured with respect to the  x-axis
  std::vector< std::vector<double> > sampling_points; // nphi points uniformly distributed around circle cross sections for every inclination angle
  bool nm1_is_known, n_is_known;                      // flags to activate the calculations of time-integrals using trapezoidal rules
  double integration_tstart;
  std::vector<double> pressure_coefficient_nm1;
  std::vector<double> pressure_coefficient_n;
  std::vector<double> azimuthal_vorticity_nm1;
  std::vector<double> azimuthal_vorticity_n;
  std::vector<double> time_integrated_pressure_coefficient;
  std::vector<double> time_integrated_azimuthal_vorticity;
#endif

  simulation_setup(const mpi_environment_t&mpi, const cmdParser &cmd) :
    hodge_tol(cmd.get<double>("hodge_tol", default_hodge_tol)),
    niter_hodge_max(cmd.get<unsigned int>("niter_hodge", default_n_hodge)),
    control_hodge(cmd.get<hodge_control>("hodge_control", def_hodge_control)),
    steps_grid_update(cmd.get<unsigned int>("grid_update", default_ngrid_update)),
    duration(cmd.get<double>("duration", default_duration)),
    use_adapted_dt(cmd.contains("adapted_dt")),
    with_smoke(cmd.contains("smoke")),
    des_pc_cell(cmd.get<std::string>("pc_cell", default_pc_cell)),
    des_solver_cell(cmd.get<std::string>("cell_solver", default_cell_solver)),
    des_pc_face(cmd.get<std::string>("pc_face", default_pc_face)),
    Reynolds(cmd.get<double>("Re", default_Re)),
  #ifdef P4_TO_P8
    check_laminar_drag(cmd.contains("check_laminar_drag")),
  #endif
    export_dir_root(cmd.get<std::string>("export_folder", (getenv("OUT_DIR") == NULL ? default_export_dir : getenv("OUT_DIR")))),
    save_vtk(cmd.contains("save_vtk")),
    save_simulation_times(cmd.contains("save_simulation_time")),
    save_forces(cmd.contains("save_forces")),
    save_timing(cmd.contains("timing")),
    save_hodge_convergence(cmd.contains("track_subloop")),
    save_state(cmd.contains("save_state_dt")),
  #ifdef P4_TO_P8
    save_surface_quantities(cmd.contains("export_surface_quantities")),
    nsurface_points(cmd.get<unsigned int>("nsurface_points", default_nsurface_points)),
  #endif
    n_states(cmd.get<unsigned int>("save_nstates", default_save_nstates))
  {
    vtk_dt = -1.0;
    if (save_vtk)
    {
      if (!cmd.contains("vtk_dt"))
        throw std::runtime_error("simulation_setup::simulation_setup(): the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
      vtk_dt = cmd.get<double>("vtk_dt", -1.0);
      if (vtk_dt <= 0.0)
        throw std::invalid_argument("simulation_setup::simulation_setup(): the value of vtk_dt must be strictly positive.");
    }
    dt_save_data = -1.0;
    if (save_state)
    {
      dt_save_data = cmd.get<double>("save_state_dt");
      if (dt_save_data < 0.0)
        throw std::invalid_argument("simulation_setup::simulation_setup(): the value of save_state_dt must be strictly positive.");
    }

    if (des_pc_cell.compare("hypre") == 0)
      pc_cell = PCHYPRE;
    else if (des_pc_cell.compare("jacobi'") == 0)
      pc_cell = PCJACOBI;
    else
    {
      if (des_pc_cell.compare("sor") != 0 && mpi.rank() == 0)
        std::cerr << "The desired preconditioner for the cell-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
      pc_cell = PCSOR;
    }
    if (des_solver_cell.compare("cg") == 0)
      cell_solver_type = KSPCG;
    else
    {
      if (des_solver_cell.compare("bicgstab") != 0 && mpi.rank() == 0)
        std::cerr << "The desired Krylov solver for the cell-solver was either not allowed or not correctly understood. BiCGStab is used instead" << std::endl;
      cell_solver_type = KSPBCGS;
    }
    if (des_pc_face.compare("hypre") == 0)
      pc_face = PCHYPRE;
    else if (des_pc_face.compare("jacobi'") == 0)
      pc_face = PCJACOBI;
    else
    {
      if (des_pc_face.compare("sor") != 0 && mpi.rank() == 0)
        std::cerr << "The desired preconditioner for the face-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
      pc_face = PCSOR;
    }


    export_vtk = -1;
    iter = 0;
    // initialize those
    global_computational_times.clear();
    global_computational_times[grid_update] = 0.0;
    global_computational_times[viscous_step] = 0.0;
    global_computational_times[projection_step] = 0.0;
    global_computational_times[velocity_interpolation] = 0.0;
    hodge_convergence_checks.resize(niter_hodge_max, 0.0);


#ifdef P4_TO_P8
    if(save_surface_quantities)
    {
      nm1_is_known = n_is_known = false;
      integration_tstart = -DBL_MAX;
      inclination_angle.resize(nsurface_points);
      sampling_points.resize(nsurface_points); // to be fully initiated after the domain is set
      pressure_coefficient_nm1.resize(nsurface_points);
      pressure_coefficient_n.resize(nsurface_points);
      azimuthal_vorticity_nm1.resize(nsurface_points);
      azimuthal_vorticity_n.resize(nsurface_points);
      time_integrated_pressure_coefficient.resize(nsurface_points);
      time_integrated_azimuthal_vorticity.resize(nsurface_points);
    }
#endif
  }

  int running_save_data_idx() const { return (int) floor(tn/dt_save_data); }
  void update_save_data_idx()       { save_data_idx = running_save_data_idx(); }
  bool time_to_save_state() const   { return (save_state && running_save_data_idx() != save_data_idx); }

  int running_export_vtk() const  { return (int) floor(tn/vtk_dt); }
  void update_export_vtk()        { export_vtk = running_export_vtk(); }
  bool time_to_save_vtk() const   { return (save_vtk && running_export_vtk() != export_vtk); }

  void truncate_exportation_file_up_to_tstart(const string &filename, const unsigned int& n_extra_values_exported_per_line) const
  {
    FILE* fp = fopen(filename.c_str(), "r+");
    char* read_line = NULL;
    size_t len = 0;
    ssize_t len_read;
    long size_to_keep = 0;
    if(((len_read = getline(&read_line, &len, fp)) != -1))
      size_to_keep += (long) len_read;
    else
      throw std::runtime_error("simulation_setup::truncate_exportation_file: couldn't read the first header line of " + filename);
    string read_format = "%lg";
    for (unsigned int k = 0; k < n_extra_values_exported_per_line; ++k)
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
      throw std::runtime_error("simulation_setup::truncate_exportation_file: couldn't truncate " + filename);
  }

  bool set_dt_and_update_grid(my_p4est_navier_stokes_t *ns)
  {
    if(use_adapted_dt)
      ns->compute_adapted_dt();
    else
      ns->compute_dt();

    dt = ns->get_dt();

    if(tn + dt > tstart + duration)
    {
      dt = tstart + duration - tn;
      ns->set_dt(dt);
    }
    if(save_vtk && dt > vtk_dt)
    {
      dt = vtk_dt; // so that we don't miss snapshots...
      ns->set_dt(dt);
    }

    return ns->update_from_tn_to_tnp1(NULL, (iter%steps_grid_update != 0), false);
  }

  void export_forces(my_p4est_navier_stokes_t* ns)
  {
    double forces[P4EST_DIM];
    ns->compute_forces(forces);
    if(!ns->get_mpirank())
    {
      FILE* fp_forces = fopen(file_forces.c_str(), "a");
      if(fp_forces == NULL)
        throw std::invalid_argument("main_flow_past_sphere: could not open file for forces output.");
#ifdef P4_TO_P8
      fprintf(fp_forces, "%g %g %g %g\n", tn, forces[0]/(.5*M_PI*r0*r0*u0*u0*ns->get_rho()), forces[1]/(.5*M_PI*r0*r0*u0*u0*ns->get_rho()), forces[2]/(.5*M_PI*r0*r0*u0*u0*ns->get_rho()));
#else
      fprintf(fp_forces, "%g %g %g\n", tn, forces[0]/r0/u0/u0/ns->get_rho(), forces[1]/r0/u0/u0/ns->get_rho());
#endif
      fclose(fp_forces);
    }
  }

  void export_and_accumulate_timings(const my_p4est_navier_stokes_t* ns)
  {
    const std::map<ns_task, execution_time_accumulator>& timings = ns->get_timings();
    P4EST_ASSERT(timings.find(projection_step) != timings.end() && timings.find(viscous_step) != timings.end() && timings.find(velocity_interpolation) != timings.end()); // should *always* find those
    P4EST_ASSERT(timings.at(projection_step).read_counter() == timings.at(viscous_step).read_counter()); // the number of subiterations should match

    if(timings.find(grid_update) != timings.end())
      global_computational_times[grid_update]           += timings.at(grid_update).read_total_time();
    global_computational_times[viscous_step]            += timings.at(viscous_step).read_total_time();
    global_computational_times[projection_step]         += timings.at(projection_step).read_total_time();
    global_computational_times[velocity_interpolation]  += timings.at(velocity_interpolation).read_total_time();

    if(!ns->get_mpirank())
    {
      FILE* fp_timing = fopen(file_timings.c_str(), "a");
      if(fp_timing == NULL)
        throw std::invalid_argument("main_flow_past_sphere: could not open file for timings output.");
      fprintf(fp_timing, "%g %g %g %g %g %u %g %g\n",
              tn,
              (timings.find(grid_update) != timings.end() ? timings.at(grid_update).read_total_time() : 0.0),
              timings.at(viscous_step).read_total_time(),
              timings.at(projection_step).read_total_time(),
              timings.at(velocity_interpolation).read_total_time(),
              timings.at(projection_step).read_counter(),
              timings.at(viscous_step).read_fixed_point_extra_time(),
              timings.at(projection_step).read_fixed_point_extra_time());
      fclose(fp_timing);
    }
  }

  void export_hodge_convergence(const mpi_environment_t& mpi)
  {
    if(!mpi.rank())
    {
      FILE* fp_hodge = fopen(file_hodge_convergence.c_str(), "a");
      if(fp_hodge == NULL)
        throw std::invalid_argument("main_flow_past_sphere: could not open file for hodge convergence output.");

      fprintf(fp_hodge, "%g", tn);
      for (unsigned int k = 0; k < niter_hodge_max; ++k)
        fprintf(fp_hodge, " %g", hodge_convergence_checks[k]);
      fprintf(fp_hodge, "\n");
      fclose(fp_hodge);
    }
    for (unsigned int k = 0; k < niter_hodge_max; ++k)
      hodge_convergence_checks[k] = 0.0;
  }

#ifdef P4_TO_P8
  void set_sampling_points_for_surface_quantities()
  {
    for (size_t k = 0; k < nsurface_points; ++k)
    {
      inclination_angle[k] = M_PI/(2*nsurface_points) + k*(M_PI/nsurface_points);
      sampling_points[k].resize(P4EST_DIM*nphi);
      for (size_t nn = 0; nn < nphi; ++nn)
      {
        sampling_points[k][P4EST_DIM*nn + 0] = domain.xyz_sphere_center(0) - r0*cos(inclination_angle[k]);
        sampling_points[k][P4EST_DIM*nn + 1] = domain.xyz_sphere_center(1) + r0*sin(inclination_angle[k])*cos(2.0*M_PI*((double) nn)/((double) nphi));
        sampling_points[k][P4EST_DIM*nn + 2] = domain.xyz_sphere_center(2) + r0*sin(inclination_angle[k])*sin(2.0*M_PI*((double) nn)/((double) nphi));
      }
    }
  }

  void export_surface_quantities(const my_p4est_navier_stokes_t* ns)
  {
    // update integration_tstart if we haven't started integrating, yet
    if(!nm1_is_known || !n_is_known)
    {
      integration_tstart = tn - dt;
      // make sure the time-integrals are 0.0;
      for (size_t k = 0; k < nsurface_points; ++k) {
        time_integrated_azimuthal_vorticity[k] = 0.0;
        time_integrated_pressure_coefficient[k] = 0.0;
      }
    }
    // save tnm1 data, get tn:
    std::swap(pressure_coefficient_n, pressure_coefficient_nm1);
    std::swap(azimuthal_vorticity_n, azimuthal_vorticity_nm1);
    nm1_is_known = n_is_known;
    n_is_known = true;

    const my_p4est_cell_neighbors_t* ngbd_c = ns->get_ngbd_c();
    const my_p4est_node_neighbors_t* ngbd_n = ns->get_ngbd_n();
    Vec const* node_velocities_np1 = ns->get_node_velocities_np1(); // this is called after completion of the time step but before the next grid update...
    const double *node_velocities_np1_p[P4EST_DIM];
    Vec azimuthal_vorticity;
    double *azimuthal_vorticity_p;

    PetscErrorCode ierr;
    ierr = VecCreateGhostNodes(ngbd_n->get_p4est(), ngbd_n->get_nodes(), &azimuthal_vorticity); CHKERRXX(ierr);
    ierr = VecGetArray(azimuthal_vorticity, &azimuthal_vorticity_p); CHKERRXX(ierr);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArrayRead(node_velocities_np1[dim], &node_velocities_np1_p[dim]); CHKERRXX(ierr);
    }
    quad_neighbor_nodes_of_node_t qnnn;
    for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
      const p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
      ngbd_n->get_neighbors(node_idx, qnnn);
      double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, ngbd_n->get_p4est(), ngbd_n->get_nodes(), xyz_node);
      const double sqrt_y_sqr_plus_z_sqr = sqrt(SQR(xyz_node[1] - domain.xyz_sphere_center(1)) + SQR(xyz_node[2] - domain.xyz_sphere_center(2)));
      const double azimuthal_angle[P4EST_DIM] = {0.0,
                                                 -(xyz_node[2] - domain.xyz_sphere_center(2))/sqrt_y_sqr_plus_z_sqr,
                                                 (xyz_node[1] - domain.xyz_sphere_center(1))/sqrt_y_sqr_plus_z_sqr};
      const double vorticity[P4EST_DIM] = {qnnn.dy_central(node_velocities_np1_p[2]) - qnnn.dz_central(node_velocities_np1_p[1]),
                                           qnnn.dz_central(node_velocities_np1_p[0]) - qnnn.dx_central(node_velocities_np1_p[2]),
                                           qnnn.dx_central(node_velocities_np1_p[1]) - qnnn.dy_central(node_velocities_np1_p[0])};
      azimuthal_vorticity_p[node_idx] = SUMD(azimuthal_angle[0]*vorticity[0], azimuthal_angle[1]*vorticity[1], azimuthal_angle[2]*vorticity[2]);
    }
    ierr = VecGhostUpdateBegin(azimuthal_vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
      const p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
      ngbd_n->get_neighbors(node_idx, qnnn);
      double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, ngbd_n->get_p4est(), ngbd_n->get_nodes(), xyz_node);
      const double sqrt_y_sqr_plus_z_sqr = sqrt(SQR(xyz_node[1] - domain.xyz_sphere_center(1)) + SQR(xyz_node[2] - domain.xyz_sphere_center(2)));
      const double azimuthal_angle[P4EST_DIM] = {0.0,
                                                 -(xyz_node[2] - domain.xyz_sphere_center(2))/sqrt_y_sqr_plus_z_sqr,
                                                 (xyz_node[1] - domain.xyz_sphere_center(1))/sqrt_y_sqr_plus_z_sqr};
      const double vorticity[P4EST_DIM] = {qnnn.dy_central(node_velocities_np1_p[2]) - qnnn.dz_central(node_velocities_np1_p[1]),
                                           qnnn.dz_central(node_velocities_np1_p[0]) - qnnn.dx_central(node_velocities_np1_p[2]),
                                           qnnn.dx_central(node_velocities_np1_p[1]) - qnnn.dy_central(node_velocities_np1_p[0])};
      azimuthal_vorticity_p[node_idx] = SUMD(azimuthal_angle[0]*vorticity[0], azimuthal_angle[1]*vorticity[1], azimuthal_angle[2]*vorticity[2]);
    }
    ierr = VecGhostUpdateEnd(azimuthal_vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(azimuthal_vorticity, &azimuthal_vorticity_p); CHKERRXX(ierr);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArrayRead(node_velocities_np1[dim], &node_velocities_np1_p[dim]); CHKERRXX(ierr);
    }
    // all procs call for the same interpolated values, but whatever (I'm tired of all this)...
    my_p4est_interpolation_cells_t interp_pressure(ngbd_c, ngbd_n);
    my_p4est_interpolation_nodes_t interp_azimuthal_vorticity(ngbd_n);
    BoundaryConditionsDIM fake_bc; fake_bc.setInterfaceType(NOINTERFACE); // we call this after extensions etc --> fake BC to consider all cells in the interpolation
    interp_pressure.set_input(ns->get_pressure(), ns->get_phi(), &fake_bc);
    interp_azimuthal_vorticity.set_input(azimuthal_vorticity, linear);
    std::vector<double> interpolated_pressure(nsurface_points*nphi);
    std::vector<double> interpolated_azimuthal_vorticity(nsurface_points*nphi);
    for (size_t k = 0; k < nsurface_points; ++k)
      for (size_t nn = 0; nn < nphi; ++nn)
      {
        interp_pressure.add_point(nphi*k + nn, sampling_points[k].data() + P4EST_DIM*nn);
        interp_azimuthal_vorticity.add_point(nphi*k + nn, sampling_points[k].data() + P4EST_DIM*nn);
      }

    interp_pressure.interpolate(interpolated_pressure.data());
    interp_azimuthal_vorticity.interpolate(interpolated_azimuthal_vorticity.data());
    for (size_t k = 0; k < nsurface_points; ++k) {
      pressure_coefficient_n[k] = 0.0;
      azimuthal_vorticity_n[k] = 0.0;
      for (size_t nn = 0; nn < nphi; ++nn)
      {
        pressure_coefficient_n[k] += (interpolated_pressure[nphi*k + nn]/(0.5*ns->get_rho()*SQR(u0)))/nphi;
        azimuthal_vorticity_n[k]  += (interpolated_azimuthal_vorticity[nphi*k + nn]/(u0/(2.0*r0)))/nphi;
      }
      if(nm1_is_known && n_is_known)
      {
        time_integrated_azimuthal_vorticity[k]  += 0.5*dt*(azimuthal_vorticity_nm1[k]   + azimuthal_vorticity_n[k]);
        time_integrated_pressure_coefficient[k] += 0.5*dt*(pressure_coefficient_nm1[k]  + pressure_coefficient_n[k]);
      }
    }
    ierr = VecDestroy(azimuthal_vorticity); CHKERRXX(ierr);


    if(!ns->get_mpirank() && nm1_is_known && n_is_known)
    {
      FILE* fp_surface_quantities = fopen(file_surf_quantities.c_str(), "w");
      fprintf(fp_surface_quantities, "%% integration tstart = %g\n", integration_tstart);
      fprintf(fp_surface_quantities, "%% integration tend = %g\n", tn);
      fprintf(fp_surface_quantities, "%% theta | time-averaged C_p | time-averaged (nondimensional) azimuthal vorticity \n");
      for (size_t k = 0; k < nsurface_points; ++k)
        fprintf(fp_surface_quantities, "%g %g %g\n", inclination_angle[k], time_integrated_pressure_coefficient[k]/(tn - integration_tstart), time_integrated_azimuthal_vorticity[k]/(tn - integration_tstart));
      fclose(fp_surface_quantities);
    }
  }
#endif

  void print_averaged_timings(const mpi_environment_t &mpi) const
  {
    PetscErrorCode ierr;

    ierr = PetscPrintf(mpi.comm(), "Mean computational time spent on \n"); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " viscosity step: %.5e\n", global_computational_times.at(viscous_step)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " projection step: %.5e\n", global_computational_times.at(projection_step)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " computing velocities at nodes: %.5e\n", global_computational_times.at(velocity_interpolation)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " grid update: %.5e\n", global_computational_times.at(grid_update)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " full iteration (total): %.5e\n", (global_computational_times.at(viscous_step) + global_computational_times.at(projection_step) + global_computational_times.at(velocity_interpolation) + global_computational_times.at(grid_update))/iter); CHKERRXX(ierr);
  }

  void save_vtk_files(my_p4est_navier_stokes_t *ns)
  {
    update_export_vtk();
    ns->save_vtk((vtk_path + "/snapshot_" + to_string(export_vtk)).c_str(), true, u0, r0);

    if(save_simulation_times){
      PetscErrorCode ierr;
      PetscPrintf(ns->get_mpicomm(), "\n Gets to printing snapshot time info \n");
        //[Elyce:] Save the timing info -- aka what time in the simulation corresponds to what number snapshot
      FILE *fich;
      ierr = PetscFOpen(ns->get_mpicomm(), file_snapshot_to_time.c_str(), "a", &fich);CHKERRXX(ierr);
      PetscFPrintf(ns->get_mpicomm(), fich, "%e %d %e \n", tn, export_vtk, dt);
      ierr = PetscFClose(ns->get_mpicomm(), fich); CHKERRXX(ierr);
    }
  }
};

class INIT_SMOKE : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    bool active = x - (domain.x_inlet() < domain.length()/16.0);
    active = active && (sqrt(SQR(y - 0.5*domain.center(dir::y) ONLY3D(+ SQR(z - domain.center(dir::z)))))< 0.4*r0);
    return (active ? 1.0 : 0.0);
  }
} init_smoke;

class BC_SMOKE : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return (fabs(x - domain.x_inlet()) < EPS*domain.length() && sqrt(SQR(y - domain.center(dir::y)) ONLY3D(+ SQR(z - domain.center(dir::z)))) < .4*r0) ? 1.0 : 0.0;
  }
} bc_smoke;

class LEVEL_SET: public CF_DIM
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return r0 - sqrt(SUMD(SQR(x - domain.xyz_sphere_center(dir::x)), SQR(y - domain.xyz_sphere_center(dir::y)), SQR(z - domain.xyz_sphere_center(dir::z))));
  }
} level_set;

struct BCWALLTYPE_PRESSURE : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double x, double, double)) const
  {
    if(fabs(x - domain.x_outlet()) < EPS*domain.length())
      return DIRICHLET;
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLTYPE_VELOCITY : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double x, double y, double z)) const
  {
    // we exclude the "corners" of the domain, to enforce the usage of the Dirichlet boundary conditions for interpolation there
    if(ANDD(fabs(x - domain.x_outlet()) < EPS*domain.length(),
            fabs(y - domain.brick->xyz_min[1]) > EPS*domain.width() && fabs(y - domain.brick->xyz_max[1]) > EPS*domain.width(),
            fabs(z - domain.brick->xyz_min[2]) > EPS*domain.width() && fabs(z - domain.brick->xyz_max[2]) > EPS*domain.width()))
      return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type_velocity;

struct BCWALLVALUE_VELOCITY : CF_DIM
{
  const unsigned char dir;
  BCWALLVALUE_VELOCITY(const unsigned char &dir_) : dir(dir_) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    if(bc_wall_type_velocity(DIM(x, y, z)) == NEUMANN)
      return 0.0; // homogeneous Neumann if neumann (outflow)
    return (dir == dir::x ? u0 : 0.0);
  }
};

struct initial_velocity_t : CF_DIM
{
  const unsigned char dir;
  initial_velocity_t(const unsigned char &dir_) : dir(dir_) {}
  double operator()(DIM(double, double, double)) const
  {
    return (dir == dir::x ? u0 : 0.0);
  }
};

void initialize_force_output(simulation_setup & setup, const my_p4est_navier_stokes_t *ns)
{
  ostringstream filename;
  filename << std::fixed << std::setprecision(2);
  filename << "forces_" << ns->get_lmin() << "-" << ns->get_lmax() << "_split_threshold_" << ns->get_split_threshold()
           << "_cfl_" << ns->get_cfl() << "_sl_" << ns->get_sl_order() << ".dat";
  setup.file_forces = setup.export_dir + "/" + filename.str();
  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Saving forces in ... %s\n", setup.file_forces.c_str()); CHKERRXX(ierr);

  if(ns->get_mpirank() == 0)
  {
    if(!file_exists(setup.file_forces))
    {
      FILE* fp_forces = fopen(setup.file_forces.c_str(), "w");
      if(fp_forces == NULL)
        throw std::runtime_error("initialize_force_output: could not open file for force output.");
      fprintf(fp_forces, "%s", (string("%% tn | Cd_x | Cd_y") ONLY3D(+ string(" | Cd_z")) + string("\n")).c_str());
      fclose(fp_forces);
    }
    else
      setup.truncate_exportation_file_up_to_tstart(setup.file_forces, P4EST_DIM);

    char liveplot_forces[PATH_MAX];
    sprintf(liveplot_forces, "%s/live_forces.gnu", setup.export_dir.c_str());
    if(!file_exists(liveplot_forces))
    {
      FILE* fp_liveplot_forces = fopen(liveplot_forces, "w");
      if(fp_liveplot_forces == NULL)
        throw std::runtime_error("initialize_force_output: could not open file for force liveplot.");
      fprintf(fp_liveplot_forces, "set term wxt noraise\n");
      fprintf(fp_liveplot_forces, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "set ylabel \"Nondimensional force\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "plot");
      for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_liveplot_forces, "\t \"%s\" using 1:%d title '%c-component' with lines lw 3", filename.str().c_str(), ((int)dd + 2), (dd == 0 ? 'x' : (dd == 1 ? 'y' : 'z')));
        if(dd < P4EST_DIM-1)
          fprintf(fp_liveplot_forces, ",\\");
        fprintf(fp_liveplot_forces, "\n");
      }
      fprintf(fp_liveplot_forces, "pause 4\n");
      fprintf(fp_liveplot_forces, "reread");
      fclose(fp_liveplot_forces);
    }

    char tex_plot_forces[PATH_MAX];
    sprintf(tex_plot_forces, "%s/tex_forces.gnu", setup.export_dir.c_str());
    if(!file_exists(tex_plot_forces))
    {
      FILE *fp_tex_plot_forces = fopen(tex_plot_forces, "w");
      if(fp_tex_plot_forces == NULL)
        throw std::runtime_error("initialize_force_output: could not open file for force tex figure.");
      fprintf(fp_tex_plot_forces, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_forces, "set output 'force_history.tex'\n");
      fprintf(fp_tex_plot_forces, "set key top right Right \n");
      fprintf(fp_tex_plot_forces, "set xlabel \"$t$\"\n");
#ifdef P4_TO_P8
      fprintf(fp_tex_plot_forces, "set ylabel \"$\\\\mathbf{C}_{\\\\mathrm{D}} = \\\\frac{2}{\\\\rho \\\\pi r^{2}u_{0}^{2}}\\\\int_{\\\\Gamma}{ \\\\left( -p \\\\mathbf{I} + 2\\\\mu \\\\mathbf{D} \\\\right)\\\\cdot \\\\mathbf{n} \\\\, \\\\mathrm{d}\\\\Gamma}$ \" \n");
#else
      fprintf(fp_tex_plot_forces, "set ylabel \"$\\\\mathbf{C}_{\\\\mathrm{D}} = \\\\frac{1}{\\\\rho r u_{0}^{2}}\\\\int_{\\\\Gamma}{ \\\\left( -p \\\\mathbf{I} + 2\\\\mu \\\\mathbf{D} \\\\right)\\\\cdot \\\\mathbf{n} \\\\, \\\\mathrm{d}\\\\Gamma}$ \" \n");
#endif
      fprintf(fp_tex_plot_forces, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_tex_plot_forces, "\t \"%s\" using 1:%d title '$C_{\\mathrm{D}, %c}$' with lines lw 3", filename.str().c_str(), ((int) dd + 2), (dd == 0 ? 'x' : (dd == 1 ?'y':'z')));
        if(dd < P4EST_DIM - 1)
          fprintf(fp_tex_plot_forces, ",\\\n");
      }
      fclose(fp_tex_plot_forces);
    }

    char tex_forces_script[PATH_MAX];
    sprintf(tex_forces_script, "%s/plot_tex_forces.sh", setup.export_dir.c_str());
    if(!file_exists(tex_forces_script))
    {
      FILE *fp_tex_forces_script = fopen(tex_forces_script, "w");
      if(fp_tex_forces_script == NULL)
        throw std::runtime_error("initialize_force_output: could not open file for bash script plotting drag tex figure.");
      fprintf(fp_tex_forces_script, "#!/bin/sh\n");
      fprintf(fp_tex_forces_script, "gnuplot ./tex_forces.gnu\n");
      fprintf(fp_tex_forces_script, "latex ./force_history.tex\n");
      fprintf(fp_tex_forces_script, "dvipdf -dAutoRotatePages=/None ./force_history.dvi\n");
      fclose(fp_tex_forces_script);

      ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_forces_script;
      int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
    }
  }
}

void initialize_timing_output(simulation_setup & setup, const my_p4est_navier_stokes_t *ns)
{
  ostringstream filename;
  filename << std::fixed << std::setprecision(2);
  filename << "timing_" << ns->get_lmin() << "-" << ns->get_lmax() << "_split_threshold_" << ns->get_split_threshold()
           << "_cfl_" << ns->get_cfl() << "_sl_" << ns->get_sl_order() << ".dat";
  setup.file_timings = setup.export_dir + "/" + filename.str();
  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Saving timings per time step in ... %s\n", setup.file_timings.c_str()); CHKERRXX(ierr);

  if(ns->get_mpirank() == 0)
  {
    if(!file_exists(setup.file_timings))
    {
      FILE* fp_timing = fopen(setup.file_timings.c_str(), "w");
      if(fp_timing == NULL)
        throw std::runtime_error("initialize_timing_output: could not open file for timing output.");
      fprintf(fp_timing, "%s", (string("%% tn | grid update | viscosity step | projection setp | interpolate velocities || number of fixed-point iterations | extra work on projection | extra work on viscous step \n")).c_str());
      fclose(fp_timing);
    }
    else
      setup.truncate_exportation_file_up_to_tstart(setup.file_timings, 7);

    char liveplot_timings[PATH_MAX];
    sprintf(liveplot_timings, "%s/live_timings.gnu", setup.export_dir.c_str());
    if(!file_exists(liveplot_timings))
    {
      FILE* fp_liveplot_timings = fopen(liveplot_timings, "w");
      if(fp_liveplot_timings == NULL)
        throw std::runtime_error("initialize_timing_output: could not open file for timing liveplot.");
      fprintf(fp_liveplot_timings, "set term wxt noraise\n");
      fprintf(fp_liveplot_timings, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_timings, "set xlabel \"Simulation time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_timings, "set ylabel \"Computational effort per time step (in %%) \" font \"Arial,14\"\n");
      fprintf(fp_liveplot_timings, "plot \t \"%s\" using 1:(100*$2/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=0 title 'Grid update',\\\n", filename.str().c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Viscous step (approximate projection)',\\\n", filename.str().c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Projection step (approximate projection)',\\\n", filename.str().c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves below y2=100 title 'Interpolating velocities from faces to nodes',\\\n", filename.str().c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=100 title 'Extra viscous steps (fixed-point iteration(s))', \\\n", filename.str().c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) : (100*($2 + $3 + $4 + $5)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Extra projection steps (fixed-point iteration(s))' \n", filename.str().c_str());
      fprintf(fp_liveplot_timings, "pause 4\n");
      fprintf(fp_liveplot_timings, "reread");
      fclose(fp_liveplot_timings);
    }

    char tex_plot_timing[PATH_MAX];
    sprintf(tex_plot_timing, "%s/tex_timing.gnu", setup.export_dir.c_str());
    if(!file_exists(tex_plot_timing))
    {
      FILE *fp_tex_plot_timing = fopen(tex_plot_timing, "w");
      if(fp_tex_plot_timing == NULL)
        throw std::runtime_error("initialize_timing_output: could not open file for timing tex figure.");
      fprintf(fp_tex_plot_timing, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_timing, "set output 'timing_history.tex'\n");
      fprintf(fp_tex_plot_timing, "set key top right Right \n");
      fprintf(fp_tex_plot_timing, "set xlabel \"Simulation time $t$\"\n");
      fprintf(fp_tex_plot_timing, "set ylabel \"Computational effort per time step (in \\\\%%) \" \n");
      fprintf(fp_tex_plot_timing, "plot \t \"%s\" using 1:(100*$2/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=0 title 'Grid update',\\\n", filename.str().c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Viscous step (approximate projection)',\\\n", filename.str().c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Projection step (approximate projection)',\\\n", filename.str().c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves below y2=100 title 'Interpolating velocities from faces to nodes',\\\n", filename.str().c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=100 title 'Extra viscous steps (fixed-point iteration(s))', \\\n", filename.str().c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) : (100*($2 + $3 + $4 + $5)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Extra projection steps (fixed-point iteration(s))' \n", filename.str().c_str());
      fclose(fp_tex_plot_timing);
    }

    char tex_timing_script[PATH_MAX];
    sprintf(tex_timing_script, "%s/plot_tex_timing.sh", setup.export_dir.c_str());
    if(!file_exists(tex_timing_script))
    {
      FILE *fp_tex_timing_script = fopen(tex_timing_script, "w");
      if(fp_tex_timing_script == NULL)
        throw std::runtime_error("initialize_timing_output: could not open file for bash script plotting timing tex figure.");
      fprintf(fp_tex_timing_script, "#!/bin/sh\n");
      fprintf(fp_tex_timing_script, "gnuplot ./tex_timing.gnu\n");
      fprintf(fp_tex_timing_script, "latex ./timing_history.tex\n");
      fprintf(fp_tex_timing_script, "dvipdf -dAutoRotatePages=/None ./timing_history.dvi\n");
      fclose(fp_tex_timing_script);

      ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_timing_script;
      int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
    }
  }
}

void initialize_hodge_convergence_output(simulation_setup & setup, const my_p4est_navier_stokes_t *ns)
{
  ostringstream filename;
  filename << std::fixed << std::setprecision(2);
  filename << "hodge_convergence_" << ns->get_lmin() << "-" << ns->get_lmax() << "_split_threshold_" << ns->get_split_threshold()
           << "_cfl_" << ns->get_cfl() << "_sl_" << ns->get_sl_order() << ".dat";
  setup.file_hodge_convergence = setup.export_dir + "/" + filename.str();
  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Saving hodge convergence info per time step in ... %s\n", setup.file_hodge_convergence.c_str()); CHKERRXX(ierr);

  if(ns->get_mpirank() == 0)
  {
    if(!file_exists(setup.file_hodge_convergence))
    {
      FILE* fp_hodge = fopen(setup.file_hodge_convergence.c_str(), "w");
      if(fp_hodge == NULL)
        throw std::runtime_error("initialize_hodge_convergence_output: could not open file for hodge convergence output.");
      fprintf(fp_hodge, "%s", (string("%% tn | max is ") + to_string(setup.niter_hodge_max) + string(" convergence values\n")).c_str());
      fclose(fp_hodge);
    }
    else
      setup.truncate_exportation_file_up_to_tstart(setup.file_hodge_convergence, setup.niter_hodge_max);

    char liveplot_hodge[PATH_MAX];
    sprintf(liveplot_hodge, "%s/live_hodge.gnu", setup.export_dir.c_str());
    if(!file_exists(liveplot_hodge))
    {
      FILE* fp_liveplot_hodge = fopen(liveplot_hodge, "w");
      if(fp_liveplot_hodge == NULL)
        throw std::runtime_error("initialize_hodge_convergence_output: could not open file for hodge convergence liveplot.");
      fprintf(fp_liveplot_hodge, "set term wxt noraise\n");
      fprintf(fp_liveplot_hodge, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_hodge, "set xlabel \"Simulation time\" font \"Arial,14\"\n");
      ostringstream os; os << setup.control_hodge;
      fprintf(fp_liveplot_hodge, "set ylabel \"Convergence check for Hodge variable (%s)\" font \"Arial,14\"\n", os.str().c_str());
      fprintf(fp_liveplot_hodge, "set format y \"%%.2t*10^%%+03T \n");
      fprintf(fp_liveplot_hodge, "set logscale y \n");
      fprintf(fp_liveplot_hodge, "plot \t \"%s\" using 1:%u title 'subiteration %u' with lines lw 3", filename.str().c_str(), 2 + 0, 0);
      for (unsigned int k = 1; k < setup.niter_hodge_max; ++k)
        fprintf(fp_liveplot_hodge, ", \\\n\t \"%s\" using 1:%u title 'subiteration %u' with lines lw 3", filename.str().c_str(), 2 + k, k);
      fprintf(fp_liveplot_hodge, "\n");
      fprintf(fp_liveplot_hodge, "pause 4\n");
      fprintf(fp_liveplot_hodge, "reread");
      fclose(fp_liveplot_hodge);
    }

    char tex_plot_hodge[PATH_MAX];
    sprintf(tex_plot_hodge, "%s/tex_hodge.gnu", setup.export_dir.c_str());
    if(!file_exists(tex_plot_hodge))
    {
      FILE *fp_tex_plot_hodge = fopen(tex_plot_hodge, "w");
      if(fp_tex_plot_hodge == NULL)
        throw std::runtime_error("initialize_hodge_convergence_output: could not open file for hodge convergence tex figure.");
      fprintf(fp_tex_plot_hodge, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_hodge, "set output 'hodge_convergence_history.tex'\n");
      fprintf(fp_tex_plot_hodge, "set key top right Right \n");
      fprintf(fp_tex_plot_hodge, "set xlabel \"Simulation time $t$\"\n");
      ostringstream os; os << setup.control_hodge;
      fprintf(fp_tex_plot_hodge, "set ylabel \"Convergence check for Hodge variable \\n (%s)\" \n", os.str().c_str());
      fprintf(fp_tex_plot_hodge, "set format y '$10^{%%T}$' \n");
      fprintf(fp_tex_plot_hodge, "set logscale y \n");
      fprintf(fp_tex_plot_hodge, "plot \t \"%s\" using 1:%u title 'subiteration %u' with lines lw 3", setup.file_hodge_convergence.c_str(), 2 + 0, 0);
      for (unsigned int k = 1; k < setup.niter_hodge_max; ++k)
        fprintf(fp_tex_plot_hodge, ", \\\n\t \"%s\" using 1:%u title 'subiteration %u' with lines lw 3", setup.file_hodge_convergence.c_str(), 2 + k, k);
      fclose(fp_tex_plot_hodge);
    }

    char tex_hodge_script[PATH_MAX];
    sprintf(tex_hodge_script, "%s/plot_tex_hodge_convergence.sh", setup.export_dir.c_str());
    if(!file_exists(tex_hodge_script))
    {
      FILE *fp_tex_hodge_script = fopen(tex_hodge_script, "w");
      if(fp_tex_hodge_script == NULL)
        throw std::runtime_error("initialize_hodge_convergence_output: could not open file for bash script plotting hodge convergence tex figure.");
      fprintf(fp_tex_hodge_script, "#!/bin/sh\n");
      fprintf(fp_tex_hodge_script, "gnuplot ./tex_hodge.gnu\n");
      fprintf(fp_tex_hodge_script, "latex ./hodge_convergence_history.tex\n");
      fprintf(fp_tex_hodge_script, "dvipdf -dAutoRotatePages=/None ./hodge_convergence_history.dvi\n");
      fclose(fp_tex_hodge_script);

      ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_hodge_script;
      int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
    }
  }
}

#ifdef P4_TO_P8
void initialize_surface_quantities_output(simulation_setup & setup, const my_p4est_navier_stokes_t *ns)
{
  ostringstream filename;
  filename << std::fixed << std::setprecision(2);
  filename << "surf_quantities_" << ns->get_lmin() << "-" << ns->get_lmax() << "_split_threshold_" << ns->get_split_threshold()
           << "_cfl_" << ns->get_cfl() << "_sl_" << ns->get_sl_order() << ".dat";
  setup.file_surf_quantities = setup.export_dir + "/" + filename.str();
  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Saving surface quantities in ... %s\n", setup.file_surf_quantities.c_str()); CHKERRXX(ierr);

  if(ns->get_mpirank() == 0)
  {
    char liveplot_surf_quantities[PATH_MAX];
    sprintf(liveplot_surf_quantities, "%s/live_surf.gnu", setup.export_dir.c_str());
    if(!file_exists(liveplot_surf_quantities))
    {
      FILE* fp_liveplot_surf_quantities = fopen(liveplot_surf_quantities, "w");
      if(fp_liveplot_surf_quantities == NULL)
        throw std::runtime_error("initialize_surface_quantities_output: could not open file for liveplot of surface quantities.");
      fprintf(fp_liveplot_surf_quantities, "set term wxt 0 noraise\n");
      fprintf(fp_liveplot_surf_quantities, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_surf_quantities, "set xlabel \"Inclination angle\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_surf_quantities, "set ylabel \"Pressure coefficient\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_surf_quantities, "plot \t \"%s\" using 1:2 title 'pressure coefficient' with lines lw 3\n", filename.str().c_str());
      fprintf(fp_liveplot_surf_quantities, "set term wxt 1 noraise\n");
      fprintf(fp_liveplot_surf_quantities, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_surf_quantities, "set xlabel \"Inclination angle\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_surf_quantities, "set ylabel \"Non-dimensional azimuthal vorticity\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_surf_quantities, "plot \t \"%s\" using 1:3 title 'azimuthal vorticity' with lines lw 3\n", filename.str().c_str());
      fprintf(fp_liveplot_surf_quantities, "pause 4\n");
      fprintf(fp_liveplot_surf_quantities, "reread");
      fclose(fp_liveplot_surf_quantities);
    }
  }
}
#endif

void load_solver_from_state(const mpi_environment_t &mpi, const cmdParser &cmd, BoundaryConditionsDIM bc_v[P4EST_DIM], BoundaryConditionsDIM &bc_p, CF_DIM *external_forces[P4EST_DIM],
                            my_p4est_navier_stokes_t* &ns, my_p4est_brick_t* &brick, splitting_criteria_cf_and_uniform_band_t* &data, simulation_setup &setup)
{
  const string backup_directory = cmd.get<string>("restart", "");
  if(!is_folder(backup_directory.c_str()))
    throw std::invalid_argument("load_solver_from_state: the restart path " + backup_directory + " is not an accessible directory.");
  if (cmd.contains("ntree_streamwise") || cmd.contains("ntree_spanwise") || ORD(cmd.contains("length"), cmd.contains("height"), cmd.contains("width")))
    throw std::invalid_argument("load_solver_from_state: the length, height and width as well as the numbers of trees along x, y and z cannot be reset when restarting a simulation.");

  if (ns != NULL)
  {
    delete ns; ns = NULL;
  }
  P4EST_ASSERT(ns == NULL);
  ns                      = new my_p4est_navier_stokes_t(mpi, backup_directory.c_str(), setup.tstart);
  setup.dt                = ns->get_dt();
  p4est_t *p4est_n        = ns->get_p4est();
  p4est_t *p4est_nm1      = ns->get_p4est_nm1();
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    if (is_periodic(p4est_n, dir))
      throw std::invalid_argument("load_solver_from_state: the periodicity from the loaded state does not match the requirements.");

  if (brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }

  P4EST_ASSERT(brick == NULL);
  domain.brick = ns->get_brick();

  if (data != NULL) {
    delete data; data = NULL;
  }
  P4EST_ASSERT(data == NULL);

  data = new splitting_criteria_cf_and_uniform_band_t(cmd.get<int>("lmin", ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl),
                                                      cmd.get<int>("lmax", ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl),
                                                      &level_set,
                                                      cmd.get<int>("uniform_band", ns->get_uniform_band()),
                                                      ((splitting_criteria_t*) p4est_n->user_pointer)->lip);
  splitting_criteria_t* to_delete = (splitting_criteria_t*) p4est_n->user_pointer;
  bool fix_restarted_grid = (data->max_lvl != to_delete->max_lvl);
  delete to_delete;
  p4est_n->user_pointer   = (void*) data;
  p4est_nm1->user_pointer = (void*) data; // p4est_n and p4est_nm1 always point to the same splitting_criteria_t no need to delete the nm1 one, it's just been done

  P4EST_ASSERT(ANDD(ns->get_length_of_domain() > 4.0, ns->get_height_of_domain() > 2.0, ns->get_width_of_domain() > 2.0));
#ifdef P4_TO_P8
  P4EST_ASSERT(fabs(ns->get_height_of_domain() - ns->get_width_of_domain()) < EPS*MAX(ns->get_height_of_domain(), ns->get_width_of_domain()));
#endif

  if(cmd.contains("Re"))
  {
    if(fabs(cmd.get<double>("Re") - Reynolds(ns)) > 1e-6*MAX(cmd.get<double>("Re"), Reynolds(ns)))
      throw std::invalid_argument("load_solver_from_state: the Reynolds number cannot be reset when restarting a simulation.");
  }
  else
    setup.Reynolds = Reynolds(ns);

  ns->set_parameters(ns->get_mu(), ns->get_rho(), cmd.get<int>("sl_order", ns->get_sl_order()), data->uniform_band, cmd.get<double>("thresh", ns->get_split_threshold()), cmd.get<double>("cfl", ns->get_cfl()));


  if(!setup.with_smoke)
    ns->set_smoke(NULL, NULL, false, 1.0);
  else
  {
    Vec smoke = ns->get_smoke();
    if(smoke == NULL)
    {
      PetscErrorCode ierr = VecCreateGhostNodes(ns->get_p4est(), ns->get_nodes(), &smoke); CHKERRXX(ierr);
      sample_cf_on_nodes(ns->get_p4est(), ns->get_nodes(), init_smoke, smoke);
      // we do not refine with smoke even if not done beforehand and required as this would make
      // the restart very complicated (needs to reinterpolate data on the new grid, etc.)
    }
    bool refine_with_smoke;
    double smoke_thresh;
    if(cmd.contains("refine_with_smoke"))
    {
      refine_with_smoke   = true;
      smoke_thresh        = cmd.get("smoke_thresh", default_smoke_threshold);
    }
    else
    {
      refine_with_smoke   = ns->get_refine_with_smoke();
      smoke_thresh        = cmd.get("smoke_thresh", (refine_with_smoke ? ns->get_smoke_threshold() : default_smoke_threshold));
    }
    ns->set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }
  ns->set_bc(bc_v, &bc_p);
  ns->set_external_forces(external_forces);
  if(fix_restarted_grid)
    ns->refine_coarsen_grid_after_restart(&level_set, false);

  if(setup.save_vtk)
    setup.update_export_vtk(); // so that we don't overwrite visualization files that were possibly already exported...

  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Simulation restarted from state saved in %s\n", (cmd.get<std::string>("restart")).c_str()); CHKERRXX(ierr);
}

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd, BoundaryConditionsDIM bc_v[P4EST_DIM], BoundaryConditionsDIM &bc_p, CF_DIM *external_forces[P4EST_DIM],
                                my_p4est_navier_stokes_t* &ns, my_p4est_brick_t* &brick, splitting_criteria_cf_and_uniform_band_t* &data, simulation_setup &setup)
{
  // build the macromesh first
  const double length     = cmd.get<double>("length", default_length);
  if(length < 4.0*r0)
    throw std::invalid_argument("create_solver_from_scratch: the length of the domain must be at least 4.0 times the sphere/cylinder radius");
  const double width      = cmd.get<double>("width", default_width);
  if(width < 2.0*r0)
    throw std::invalid_argument("create_solver_from_scratch: the dimension of the domain in spanwise dircetion(s) must be at least twice the sphere/cylinder radius");
  const double xyz_min_[P4EST_DIM] = {DIM(0,       -0.5*width, -0.5*width)};
  const double xyz_max_[P4EST_DIM] = {DIM(length,  +0.5*width, +0.5*width)};
  const int n_tree_streamwise = cmd.get<int>("ntree_streamwise", default_ntree_streamwise);
  const int n_tree_spanwise   = cmd.get<int>("ntree_spanwise", default_ntree_spanwise);
  const int n_tree_xyz[P4EST_DIM] = {DIM(n_tree_streamwise, n_tree_spanwise, n_tree_spanwise)};
  p4est_connectivity_t *connectivity;
  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid);brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;
  const int periodic[P4EST_DIM] = {DIM(0, 0, 0)};
  connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min_, xyz_max_, brick, periodic);
  domain.brick = brick;

  if(data != NULL)
  {
    delete data; data = NULL;
  }
  P4EST_ASSERT(data == NULL);
  const int lmax = cmd.get<int>("lmax", default_lmax);
  const double default_ndx_band = default_uniform_band*((double) (1 << lmax))/MAX(domain.length()/brick->nxyztrees[0], domain.width()/brick->nxyztrees[1]);
  data  = new splitting_criteria_cf_and_uniform_band_t(cmd.get<int>("lmin", default_lmin),
                                                       lmax,
                                                       &level_set,
                                                       cmd.get<double>("uniform_band", default_ndx_band));

  p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(int l = 0; l < lmax; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = (void*) data;
  const bool refine_with_smoke  = cmd.contains("refine_with_smoke");
  const double smoke_thresh     = cmd.get<double>("smoke_thresh", default_smoke_threshold);
  if(setup.with_smoke && refine_with_smoke)
  {
    splitting_criteria_thresh_t crit_thresh(data->min_lvl, data->max_lvl, &init_smoke, smoke_thresh);
    p4est_n->user_pointer = (void*)&crit_thresh;
    my_p4est_refine(p4est_n, P4EST_TRUE, refine_levelset_thresh, NULL);
    p4est_balance(p4est_n, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
    p4est_n->user_pointer = (void*) data;
  }

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c);

  Vec phi;
  PetscErrorCode ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, level_set, phi);
  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_1st_order_time_2nd_order_space(phi);
  lsn.perturb_level_set_function(phi, EPS);


  ns = new my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n);
  ns->set_phi(phi);
  if(setup.with_smoke)
  {
    Vec smoke;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    ns->set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }
  ns->set_parameters(2.0*1.0*r0*u0/setup.Reynolds,
                     1.0,
                     cmd.get<int>("sl_order", default_sl_order),
                     data->uniform_band,
                     cmd.get<double>("thresh", default_thresh),
                     cmd.get<double>("cfl", default_cfl));

  CF_DIM *initial_velocity[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    initial_velocity[dim] = new initial_velocity_t(dim);
  ns->set_velocities(initial_velocity, initial_velocity);

  setup.tstart = 0.0; // no restart so we assume we start from 0.0
  const double dxmin = MIN(domain.length()/brick->nxyztrees[0], domain.width()/brick->nxyztrees[1])/((double) (1 << lmax));
  setup.dt = MIN(dxmin*ns->get_cfl()/u0, setup.duration);
  if(setup.save_vtk)
      setup.dt = MIN(setup.dt, setup.vtk_dt);
  ns->set_dt(setup.dt);
  ns->set_bc(bc_v, &bc_p);
  ns->set_external_forces(external_forces);

  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    delete  initial_velocity[dim];
}

void initialize_exportations_and_monitoring(const my_p4est_navier_stokes_t* ns, simulation_setup &setup)
{
  PetscErrorCode ierr;
  ostringstream oss;
  oss << std::scientific << std::setprecision(2);
  oss << "Parameters : Re = " << Reynolds(ns) << ", mu = " << ns->get_mu() << ", rho = " << ns->get_rho() << ", macromesh is "
      << ns->get_brick()->nxyztrees[0] << " X " << ns->get_brick()->nxyztrees[1] ONLY3D( << " X " << ns->get_brick()->nxyztrees[2]) << "\n";
  ierr = PetscPrintf(ns->get_mpicomm(), oss.str().c_str()); CHKERRXX(ierr);
  ierr = PetscPrintf(ns->get_mpicomm(), "cfl = %g, uniform_band = %g\n", ns->get_cfl(), ns->get_uniform_band());

  ostringstream subfolder;
  subfolder<< std::fixed << std::setprecision(2);
  subfolder << "Re_" << Reynolds(ns) <<  "/lmin_" << ns->get_lmin() << "_lmax_" << ns->get_lmax();
  setup.export_dir = setup.export_dir_root + "/" + subfolder.str();
  setup.vtk_path = setup.export_dir + "/vtu";
  if(setup.save_vtk || setup.save_forces || setup.save_state)
  {
    if(create_directory(setup.export_dir, ns->get_mpirank(), ns->get_mpicomm()))
      throw std::runtime_error("main_flow_past_sphere: could not create exportation directory " + setup.export_dir);

    if(setup.save_vtk && create_directory(setup.vtk_path, ns->get_mpirank(), ns->get_mpicomm()))
      throw std::runtime_error("main_flow_past_sphere: could not create exportation directory for vtk files " + setup.vtk_path);
  }

  // [Elyce:] Create file to hold snapshot and time info -- so we know what snapshot corresponds to what time in the simulation
  setup.file_snapshot_to_time = setup.export_dir + "/snapshot_to_time_info.dat";

  FILE *fich;
  if(setup.save_vtk && setup.save_simulation_times){
      //[Elyce:] Setup file to save the timing info -- aka what time in the simulation corresponds to what number snapshot
      ierr = PetscFOpen(ns->get_mpicomm(), setup.file_snapshot_to_time.c_str(), "w", &fich); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fich, "Time Snapshot No. dt \n"); CHKERRXX(ierr);
      ierr = PetscFClose(ns->get_mpicomm(), fich); CHKERRXX(ierr);
  }

  if(setup.save_forces)
    initialize_force_output(setup, ns);
  if(setup.save_timing)
    initialize_timing_output(setup, ns);
  if(setup.save_hodge_convergence)
    initialize_hodge_convergence_output(setup, ns);
#ifdef P4_TO_P8
  if(setup.save_surface_quantities)
    initialize_surface_quantities_output(setup, ns);
#endif
}

#ifdef P4_TO_P8
struct laminar_solution_velocity : CF_3
{
  const unsigned char dir;
  laminar_solution_velocity(const unsigned char dir_) : dir(dir_) {}
  double operator()(double x, double y, double z) const
  {
    const double r      = r0 - level_set(x, y, z);
    if(r < 0.5*r0) // we don't need it there...
      return 0.0;
    const double theta  = acos((x - domain.xyz_sphere_center(dir::x))/r);
    double phi_;
    if(sqrt(SQR(y - domain.xyz_sphere_center(dir::y)) + SQR(z - domain.xyz_sphere_center(dir::z))) < EPS*domain.width())
      phi_ = 0.0;
    else
      phi_ = atan2(z - domain.xyz_sphere_center(dir::z), y - domain.xyz_sphere_center(dir::y));
    const double ur     = u0*cos(theta)*(1.0 + 0.5*pow(r0/r, 3.0) - 1.5*r0/r);
    const double utheta = -u0*sin(theta)*(1.0 - 0.25*pow(r0/r, 3.0) - 0.75*r0/r);
    switch (dir) {
    case dir::x:
      return ur*cos(theta) - utheta*sin(theta);
      break;
    case dir::y:
      return (ur*sin(theta) + utheta*cos(theta))*cos(phi_);
      break;
    case dir::z:
      return (ur*sin(theta) + utheta*cos(theta))*sin(phi_);
      break;
    default:
      throw std::runtime_error("laminar_solution: unknown Cartesian direction");
      break;
    }
  }
};

struct laminar_solution_pressure : CF_3
{
  const my_p4est_navier_stokes_t *ns;
  laminar_solution_pressure(const my_p4est_navier_stokes_t *ns_) : ns(ns_) {}
  double operator()(double x, double y, double z) const
  {
    const double r      = r0 - level_set(x, y, z);
    const double theta  = acos((x - domain.xyz_sphere_center(dir::x))/r);
    return -1.5*ns->get_mu()*u0*r0*cos(theta)/pow(r, 3.0);
  }
};

void check_laminar_forces_and_print_errors(my_p4est_navier_stokes_t* ns)
{
  CF_DIM *laminar_flow_field[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    laminar_flow_field[dim] = new laminar_solution_velocity(dim);
  laminar_solution_pressure laminar_pressure(ns);

  ns->set_vnp1_nodes(laminar_flow_field);
  ns->set_pressure(laminar_pressure);
  double forces[P4EST_DIM];
  ns->compute_forces(forces);
  const double scaling = (.5*M_PI*r0*r0*u0*u0*ns->get_rho());

  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Errors on the calculation of laminar forces are %e, %e, %e\n", fabs(forces[0]/scaling - 24.0/Reynolds(ns)), fabs(forces[1]/scaling), fabs(forces[2]/scaling)); CHKERRXX(ierr);

  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    delete laminar_flow_field[dim];

  return;
}
#endif

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin",                  "min level of the trees, default is " + to_string(default_lmin));
  cmd.add_option("lmax",                  "max level of the trees, default is " + to_string(default_lmax));
  cmd.add_option("thresh",                "the threshold used for the refinement criteria, default is " + to_string(default_thresh));
  cmd.add_option("uniform_band",          "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is such that half the sphere radius is tesselated");
  cmd.add_option("ntree_streamwise",      "number of trees in the streamwise direction. The default value is " + to_string(default_ntree_streamwise));
  cmd.add_option("ntree_spanwise",        "number of trees in the spanwise direction(s). The default value is " + to_string(default_ntree_spanwise));
  cmd.add_option("smoke_thresh",          "threshold for smoke refinement, default is " + to_string(default_smoke_threshold));
  cmd.add_option("refine_with_smoke",     "refine the grid with the smoke density and threshold smoke_thresh if present");
  // physical parameters for the simulations
  cmd.add_option("length",                "the streamwise dimension of the domain, default is " + to_string(default_length));
  cmd.add_option("width",                 "the spanwise dimension(s) of the domain, default is " + to_string(default_width));
  cmd.add_option("duration",              "the duration of the simulation (tfinal - tstart). If not restarted, tstart = 0.0, default duration is " + to_string(default_duration));
  cmd.add_option("Re",                    "the Reynolds number = (2.0*r0*rho*u0)/mu, default is " + to_string(default_Re));
  cmd.add_option("adapted_dt",            "activates the calculation of dt based on the local cell sizes if present");
  cmd.add_option("smoke",                 "no smoke if option not present, with smoke if option present");
  // method-related parameters
  cmd.add_option("sl_order",              "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is " + to_string(default_sl_order));
  cmd.add_option("cfl",                   "dt = cfl * dx/vmax, default is " + to_string(default_cfl));
  cmd.add_option("hodge_tol",             "numerical tolerance used for the convergence criterion on the Hodge variable (or its gradient), at all time steps, default is " + to_string(default_hodge_tol));
  cmd.add_option("niter_hodge",           "max number of iterations for convergence of the Hodge variable, at all time steps, default is " + to_string(default_n_hodge));
  cmd.add_option("hodge_control",         "type of convergence check used for inner loops, i.e. convergence criterion on the Hodge variable. \n\
                 Possible values are 'u', 'v' (, 'w'), 'uvw' (for gradient components) or 'value' (for local values of Hodge), default is " + convert_to_string(def_hodge_control));
  cmd.add_option("grid_update",           "number of time steps between grid updates, default is " + to_string(default_ngrid_update));
  cmd.add_option("pc_cell",               "preconditioner for cell-solver: jacobi, sor or hypre, default is " + default_pc_cell);
  cmd.add_option("cell_solver",           "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is " + default_cell_solver);
  cmd.add_option("pc_face",               "preconditioner for face-solver: jacobi, sor or hypre, default is " + default_pc_face);
  // output-control parameters
#ifdef P4_TO_P8
  cmd.add_option("check_laminar_drag",    "calculates the forces applied onto the solid sphere by the laminar Stokes flow, prints the error and returns. (sanity check)");
#endif
  cmd.add_option("export_folder",         "exportation_folder if not defined otherwise in the environment variable OUT_DIR,\n\
                 subfolders will be created, default is " + default_export_dir);
  cmd.add_option("save_vtk",              "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt",                "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
  cmd.add_option("save_simulation_time",  "if present, saves the simulation times corresponding to snapshot indices in a file named snapshot_to_time_info.dat in the exportation directory (relevant only if 'save_vtk' is present)");
  cmd.add_option("save_forces",           "if defined, saves the forces onto the sphere/cylinder");
  cmd.add_option("save_state_dt",         "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates",          "determines how many solver states must be memorized in backup_ folders (default is " +to_string(default_save_nstates));
  cmd.add_option("timing",                "if defined, saves timing information in a file on disk (typically for scaling analysis).");
  cmd.add_option("track_subloop",         "if defined, saves the data corresponding to the inner loops for convergence of the hodge variable (saved in a file on disk).");
#ifdef P4_TO_P8
  cmd.add_option("export_surface_quantities",
                                          "if defined, export the pressure coefficient and azimuthal vorticity along the azimuthal angle at every time step (in a file on disk).");
  cmd.add_option("nsurface_points",       "number of surface points for exportation of surface quantities. Default is "  + to_string(default_nsurface_points));
#endif

  if (cmd.parse(argc, argv, extra_info))
    return 0;

  PetscErrorCode ierr;

  simulation_setup setup(mpi, cmd);

  my_p4est_navier_stokes_t* ns                    = NULL;
  my_p4est_brick_t* brick                         = NULL;
  splitting_criteria_cf_and_uniform_band_t* data  = NULL;

  CF_DIM *bc_wall_value[P4EST_DIM];
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    bc_wall_value[dim] = new BCWALLVALUE_VELOCITY(dim);
    bc_v[dim].setWallTypes(bc_wall_type_velocity);
    bc_v[dim].setWallValues(*bc_wall_value[dim]);
    bc_v[dim].setInterfaceType(DIRICHLET);
    bc_v[dim].setInterfaceValue(zero_cf);
  }
  bc_p.setWallTypes(bc_wall_type_p);
  bc_p.setWallValues(zero_cf); // homogeneous, whether Neumann or Dicihlet...
  bc_p.setInterfaceType(NEUMANN);
  bc_p.setInterfaceValue(zero_cf);
  CF_DIM *external_forces[P4EST_DIM] = {DIM(&zero_cf, &zero_cf, &zero_cf)};

  // create the solver, either loaded from saved state or built from scratch
  if (cmd.contains("restart"))
    load_solver_from_state(mpi, cmd, bc_v, bc_p, external_forces, ns, brick, data, setup);
  else
    create_solver_from_scratch(mpi, cmd, bc_v, bc_p, external_forces, ns, brick, data, setup);

#ifdef P4_TO_P8
  if(setup.save_surface_quantities)
    setup.set_sampling_points_for_surface_quantities();
  if(setup.check_laminar_drag)
  {
    check_laminar_forces_and_print_errors(ns);
    return 0;
  }
#endif

  initialize_exportations_and_monitoring(ns, setup);

  if(setup.save_timing)
    ns->activate_timer();

  setup.tn = setup.tstart;
  setup.update_save_data_idx(); // so that we don't save the very first one which was either already read from file, or the known initial condition...

  my_p4est_poisson_cells_t* cell_solver = NULL;
  my_p4est_poisson_faces_t* face_solver = NULL;
  Vec *dxyz_hodge_old = NULL, hold_old = NULL;
  if(setup.control_hodge != hodge_value)
    dxyz_hodge_old = new Vec[P4EST_DIM];

  while(setup.tn + 0.01*setup.dt < setup.tstart + setup.duration)
  {
    if(setup.iter > 0)
    {
      bool solvers_can_be_reused = setup.set_dt_and_update_grid(ns);

      if(cell_solver != NULL && !solvers_can_be_reused){
        delete  cell_solver; cell_solver = NULL; }
      if(face_solver != NULL && !solvers_can_be_reused){
        delete  face_solver; face_solver = NULL; }
    }

    if(setup.time_to_save_state())
    {
      setup.update_save_data_idx();
      ns->save_state(setup.export_dir.c_str(), setup.tn, setup.n_states);
    }

    double convergence_check_on_hodge = DBL_MAX;
    if(setup.control_hodge == hodge_value) {
      ierr = VecCreateNoGhostCells(ns->get_p4est(), &hold_old); CHKERRXX(ierr); }
    else
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir){
        ierr = VecCreateNoGhostFaces(ns->get_p4est(), ns->get_faces(), &dxyz_hodge_old[dir], dir); CHKERRXX(ierr); }

    unsigned int iter_hodge = 0;
    while(iter_hodge < setup.niter_hodge_max && convergence_check_on_hodge > setup.hodge_tol)
    {
      if(setup.control_hodge == hodge_value)
        ns->copy_hodge(hold_old, false);
      else
        ns->copy_dxyz_hodge(dxyz_hodge_old);

      ns->solve_viscosity(face_solver, (face_solver != NULL), KSPBCGS, setup.pc_face); // no other (good) choice than KSPBCGS for this one, symmetry is broken

      convergence_check_on_hodge = ns->solve_projection(cell_solver, (cell_solver != NULL), setup.cell_solver_type, setup.pc_cell, false, hold_old, dxyz_hodge_old, setup.control_hodge);

      if(setup.control_hodge == hodge_value){
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in Hodge = %e\n", iter_hodge, convergence_check_on_hodge); CHKERRXX(ierr);
      } else if(setup.control_hodge == uvw_components){
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in \\nabla Hodge = %e\n", iter_hodge, convergence_check_on_hodge); CHKERRXX(ierr);
      } else {
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in d(Hodge)/d%s = %e\n", iter_hodge, (setup.control_hodge == u_component ? "x" : (setup.control_hodge == v_component ? "y" : "z")), convergence_check_on_hodge); CHKERRXX(ierr);
      }

      setup.hodge_convergence_checks[iter_hodge] = convergence_check_on_hodge;
      iter_hodge++;
    }
    if(setup.control_hodge == hodge_value) {
      ierr = VecDestroy(hold_old); CHKERRXX(ierr); }
    else
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir){
        ierr = VecDestroy(dxyz_hodge_old[dir]); CHKERRXX(ierr); }

    ns->compute_velocity_at_nodes(setup.steps_grid_update > 1);

    ns->compute_pressure();

    setup.tn += setup.dt;

    if(setup.save_forces)
      setup.export_forces(ns);

    if(setup.save_timing)
      setup.export_and_accumulate_timings(ns);

    if(setup.save_hodge_convergence)
      setup.export_hodge_convergence(mpi);

#ifdef P4_TO_P8
    if(setup.save_surface_quantities)
      setup.export_surface_quantities(ns);
#endif

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", setup.iter, setup.tn, 100*(setup.tn - setup.tstart)/setup.duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(ns->get_max_L2_norm_u() > 200.0)
    {
      if(setup.save_vtk)
        ns->save_vtk((setup.vtk_path + "/snapshot_" + to_string(setup.export_vtk + 1)).c_str());
      std::cerr << "The simulation blew up..." << std::endl;
      break;
    }

    if(setup.time_to_save_vtk())
      setup.save_vtk_files(ns);

    setup.iter++;
  }

  if(setup.save_timing)
    setup.print_averaged_timings(mpi);

  if(dxyz_hodge_old != NULL)
    delete [] dxyz_hodge_old;

  if(cell_solver != NULL)
    delete cell_solver;
  if(face_solver != NULL)
    delete face_solver;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    delete bc_wall_value[dim];

  delete ns;
  // the connectivity is deleted within the above destructor...
  delete brick;
  delete data;

  return 0;
}
