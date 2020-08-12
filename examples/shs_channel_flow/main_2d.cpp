/*
 * The navier stokes solver applied for super-hydrophobic surfaces simulations
 *
 * run the program with the -help flag to see the available options
 */

// System
#if defined(JUPITER)
#include <mpich/mpi.h>
#else
#include <mpi.h>
#endif
#include <iterator>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_shs_channel.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_shs_channel.h>
#endif

#include <src/Parser.h>
#include <iomanip>

#undef MIN
#undef MAX

// --> extra info to be printed when -help is invoked
const std::string extra_info =
      std::string("This program provides a general setup for superhydrophobic channel flow simulations.\n\n")
    + std::string("It assumes no solid object and no passive scalar (i.e. no smoke) in the channel. If no dimension is set by the user, the \n")
    + std::string("channel is set to be 6 x 2 (x 3) by default. If the number of trees in the streamwise (spanwise) direction, i.e. nx (nz),\n")
    + std::string("is not provided by the user, it is set so that the aspect ratio of the computational cells is as close as possible to 1.\n")
    + std::string("This tool considers periodic boundary conditions in the streamwise (and spanwise) direction(s). \n\n")
    + std::string("Considering the following parameters,\n")
    + std::string("\t delta := half the height of the channel,\n")
    + std::string("\t f_x   := bulk force per unit mass driving the flow in the (positive) x direction,\n")
    + std::string("\t nu    := kinematic viscosity of the fluid (nu = mu/rho),\n")
    + std::string("\t U_b   := mean flow velocity across the channel (U_b = Q/(rho*2*delta(*width)) where Q is the mass flow rate in the channel),\n")
    + std::string("the user can require the simulation to\n")
    + std::string("1) either use a constant bulk force driving the flow by setting the value of Re_tau where Re_tau is defined as \n")
    + std::string("                                    Re_tau = delta*sqrt(delta*f_x)/nu \n")
    + std::string("\t In that case, the Navier-Stokes solver is invoked with the following inputs: \n")
    + std::string("\t\t rho = 1.0;\n")
    + std::string("\t\t f_x = 1.0/delta (so that u_tau = sqrt(f_x*tau) = 1.0 --> in fact, this nondimensionalizes velocities by u_tau);\n")
    + std::string("\t\t mu  = rho*u_tau*delta/Re_tau, to match the desired Re_tau;\n")
    + std::string("2) or enforce a constant mass flow by setting the value of Re_b where Re_b is defined as \n")
    + std::string("                                    Re_b = delta*U_b/nu \n")
    + std::string("\t In that case, the Navier-Stokes solver is invoked with the following inputs: \n")
    + std::string("\t\t rho = 1.0;\n")
    + std::string("\t\t f_x is calculated (and adapted) at every time-step to enforce Q = rho*2*delta(*width)\n")
    + std::string("\t\t\t (so that U_b = 1.0 --> in fact, this nondimensionalizes velocities by U_b);\n")
    + std::string("\t\t\t (f_x initialized to the laminar-case value);\n")
    + std::string("\t\t mu  = rho*U_b*delta/Re_b = delta/Re_b, to match the desired Re_b.\n")
    + std::string("The user can set either Re_tau or Re_b but not both!\n\n")
    + std::string("When restarting from a saved state, the user can change the simulation setup (i.e. desired Re_b or Re_tau) but the fluid\n")
    + std::string("parameters, the dimensions of the domain and macromesh will be identical to the ones loaded from the saved state. Therefore,\n")
    + std::string("\t\t rho = as loaded from the saved state;\n")
    + std::string("\t\t mu  = as loaded from the saved state;\n")
    + std::string("1) when restarting with a value of Re_tau, \n")
    + std::string("\t\t f_x = SQR(mu*Re_tau/(rho*delta))/delta = constant                                     [to match the desired Re_tau];\n")
    + std::string("2) when restarting with a value of Re_b\n")
    + std::string("\t\t f_x is calculated (and adapted) at every time-step to enforce Q = 2.0*mu*Re_b(*width) [to match the desired Re_b].\n")
    + std::string("\t\t [If desired, the initial value of f_x can be set with the input parameter 'initial_Re_tau'.]\n")
    + std::string("Developers: Raphael Egan (raphaelegan@ucsb.edu) with Fernando Temprano-Coleto's help for the analytical solutions.\n");

#if defined(POD_CLUSTER)
const std::string default_export_dir  = "/scratch/regan/superhydrophobic_channel/" + std::to_string(P4EST_DIM) + "D_channel";
#elif defined(STAMPEDE)
const std::string default_export_dir  = "/scratch/04965/tg842642/superhydrophobic_channel/" + std::to_string(P4EST_DIM) + "D_channel";
#elif defined(LAPTOP)
const std::string default_export_dir  = "/home/raphael/workspace/projects/superhydrophobic_channel/" + std::to_string(P4EST_DIM) + "D_channel";
#elif defined(JUPITER)
const std::string default_export_dir  = "/home/temprano/Output/p4est_ns_shs/" + std::to_string(P4EST_DIM) + "D_channel";
#elif defined(NEPTUNE)
const std::string default_export_dir  = "/home/hlevy/workspace/superhydrophobic_channel/" + std::to_string(P4EST_DIM) + "D_channel";
#else
const std::string default_export_dir  = "/home/regan/workspace/projects/superhydrophobic_channel/" + std::to_string(P4EST_DIM) + "D_channel";
#endif

const int default_lmin                        = 4;
const int default_lmax                        = 6;
const double default_thresh                   = 0.1;
const int default_wall_layer                  = 6;
const double default_lip                      = 1.2;
const int default_ntree_y                     = 2;
const double default_length                   = 6.0;
const double default_height                   = 2.0;
#ifdef P4_TO_P8
const double default_width                    = 3.0;
#endif
const double default_duration                 = 200.0;
const double def_white_noise_rms              = 0.00;
const double default_pitch_to_height          = 3.0/16.0;
const double default_gas_fraction             = 0.5;
const int default_sl_order                    = 2;
const double default_cfl                      = 1.0;
const double default_u_tol                    = 1.0e-6;
const unsigned int default_n_hodge            = 10;
const hodge_control def_hodge_control         = uvw_components;
const unsigned int default_grid_update        = 1;
const std::string default_pc_cell             = "sor";
const std::string default_cell_solver         = "bicgstab";
const std::string default_pc_face             = "sor";
const unsigned int default_save_nstates       = 1;
const unsigned int default_nexport_avg        = 100;
const int default_nterms                      = 2500;
const int periodicity[P4EST_DIM]              = {DIM(1, 0, 1)};
const std::string drag_output_format          = std::string("%g %g %g") ONLY3D(+ std::string(" %g")) + std::string("\n");

class external_force_per_unit_mass_t : public CF_DIM {
private:
  double forcing_term;
public:
  external_force_per_unit_mass_t(const double &forcing_term_) : forcing_term(forcing_term_) {}
  external_force_per_unit_mass_t(): external_force_per_unit_mass_t(0.0) {}
  double operator()(DIM(double, double, double)) const  { return forcing_term; }
  void update_term(const double &correction)            { forcing_term += correction; }
  double get_value() const                              { return forcing_term; }
  void set_value(const double &new_forcing_term)        { forcing_term = new_forcing_term; }
};

class mass_flow_controller_t
{
private:
  unsigned char flow_dir;
  bool forcing_is_on;
  double desired_bulk_velocity;
  double latest_mass_flow;
  double section;

public:
  mass_flow_controller_t(const char &flow_direction, const double& section_) :
    flow_dir(flow_direction), forcing_is_on(false), desired_bulk_velocity(-1.0), latest_mass_flow(0.0), section(section_) { P4EST_ASSERT(flow_dir < P4EST_DIM); }

  void activate_forcing(const double &desired_U_b)
  {
    P4EST_ASSERT(desired_U_b > 0.0);
    desired_bulk_velocity = desired_U_b;
    forcing_is_on = true;
  }

  double update_forcing_and_get_mean_streamwise_velocity_correction(my_p4est_navier_stokes_t* ns, external_force_per_unit_mass_t* external_acceleration[P4EST_DIM]) const
  {
    P4EST_ASSERT(forcing_is_on);
    double required_correction_to_hodge_derivative = ns->get_correction_in_hodge_derivative_for_enforcing_mass_flow(flow_dir, desired_bulk_velocity, &latest_mass_flow);
    external_acceleration[flow_dir]->update_term(-required_correction_to_hodge_derivative*ns->alpha()/ns->get_dt());
    return required_correction_to_hodge_derivative;
  }

  void evaluate_current_mass_flow(my_p4est_navier_stokes_t* ns)
  {
    ns->global_mass_flow_through_slice(dir::x, section, latest_mass_flow);
  }

  double read_latest_mass_flow() const    { return latest_mass_flow; }
  double targeted_bulk_velocity() const   { P4EST_ASSERT(forcing_is_on); return desired_bulk_velocity; }

  ~mass_flow_controller_t(){}
};

struct simulation_setup
{
  double tstart;
  double dt;
  double tn;

  // inner convergence parameteres
  const double u_tol;
  const unsigned int niter_hodge_max;
  const hodge_control control_hodge;

  // simulation control
  int iter;
  const unsigned int steps_grid_update;
  const double duration;
  const bool use_adapted_dt;
  const double white_noise_rms;
  const int nterms_in_series;
  const std::string des_pc_cell;
  const std::string des_solver_cell;
  const std::string des_pc_face;
  KSPType cell_solver_type;
  PCType pc_cell, pc_face;
  const flow_setting flow_condition;
  const double Reynolds; // either Re_tau or Re_b, depends on flow_condition here above!

  // exportation
  const std::string export_dir;
  const bool save_vtk;
  const bool save_timing;
  double vtk_dt;
  const bool save_drag;
  const bool do_accuracy_check;
  const bool save_state;
  double dt_save_data;
  const unsigned int n_states;
  const bool save_profiles;
  const unsigned int nexport_avg;
  int export_vtk, save_data_idx;
  std::string file_monitoring, vtk_path, file_drag, file_timings;
  bool accuracy_check_done;
  std::map<ns_task, double> global_computational_times;


  simulation_setup(const mpi_environment_t&mpi, const cmdParser &cmd) :
    u_tol(cmd.get<double>("u_tol", default_u_tol)),
    niter_hodge_max(cmd.get<unsigned int>("niter_hodge", default_n_hodge)),
    control_hodge(cmd.get<hodge_control>("hodge_control", def_hodge_control)),
    steps_grid_update(cmd.get<unsigned int>("grid_update", default_grid_update)),
    duration(cmd.get<double>("duration", default_duration)),
    use_adapted_dt(cmd.contains("adapted_dt")),
    white_noise_rms(cmd.get<double>("white_noise_rms", def_white_noise_rms)),
    nterms_in_series(cmd.get<int>("nterms", default_nterms)),
    des_pc_cell(cmd.get<std::string>("pc_cell", default_pc_cell)),
    des_solver_cell(cmd.get<std::string>("cell_solver", default_cell_solver)),
    des_pc_face(cmd.get<std::string>("pc_face", default_pc_face)),
    flow_condition((cmd.contains("Re_tau") ? constant_pressure_gradient : (cmd.contains("Re_b") ? constant_mass_flow : undefined_flow_condition))),
    Reynolds((cmd.contains("Re_tau") ? cmd.get<double>("Re_tau") : (cmd.contains("Re_b") ? cmd.get<double>("Re_b") : NAN))),
    export_dir(cmd.get<std::string>("export_folder", default_export_dir)),
    save_vtk(cmd.contains("save_vtk")),
    save_timing(cmd.contains("timing")),
    save_drag(cmd.contains("save_drag")),
    do_accuracy_check(cmd.contains("accuracy_check") && cmd.contains("restart")),
    save_state(cmd.contains("save_state_dt")),
    n_states(cmd.get<unsigned int>("save_nstates", default_save_nstates)),
    save_profiles(cmd.contains("save_mean_profiles")),
    nexport_avg(cmd.get<unsigned int>("nexport_avg", default_nexport_avg))
  {
    if(control_hodge == hodge_value)
      throw std::invalid_argument("simulation_setup::simulation_setup: this simulation setup does not allow for control on hodge value (because of singularities at no-slip to free-slip transition regions).");

    vtk_dt = -1.0;
    if (save_vtk)
    {
      if (!cmd.contains("vtk_dt") && !do_accuracy_check)
        throw std::runtime_error("simulation_setup::simulation_setup(): the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
      vtk_dt = cmd.get<double>("vtk_dt", -1.0);
      if (vtk_dt <= 0.0 && !do_accuracy_check)
        throw std::invalid_argument("simulation_setup::simulation_setup(): the value of vtk_dt must be strictly positive.");
    }
    dt_save_data = -1.0;
    if (save_state)
    {
      dt_save_data = cmd.get<double>("save_state_dt", -1.0);
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

    if (cmd.contains("Re_b") && cmd.contains("Re_tau")) // safeguard...
      throw std::invalid_argument("simulation_setup::simulation_setup(): forcing a constant bulk velocity AND a constant pressure gradient cannot be done: you have to choose only one!");
    if (flow_condition == undefined_flow_condition)
      throw std::invalid_argument("simulation_setup::simulation_setup(): you need to specify either the desired Re_tau or the desired Re_b...");


    export_vtk = -1;
    iter = 0;
    accuracy_check_done = false;
    // initialize those
    global_computational_times.clear();
    global_computational_times[grid_update] = 0.0;
    global_computational_times[viscous_step] = 0.0;
    global_computational_times[projection_step] = 0.0;
    global_computational_times[velocity_interpolation] = 0.0;
  }

  int running_save_data_idx() const { return (int) floor(tn/dt_save_data); }
  void update_save_data_idx()       { save_data_idx = running_save_data_idx(); }
  bool time_to_save_state() const   { return (save_state && running_save_data_idx() != save_data_idx); }

  bool done() const
  {
    return tn + 0.01*dt > tstart + duration || accuracy_check_done;
  }

  int running_export_vtk() const  { return (int) floor(tn/vtk_dt); }
  void update_export_vtk()        { export_vtk = running_export_vtk(); }
  bool time_to_save_vtk() const   { return (save_vtk && running_export_vtk() != export_vtk); }

  double max_tolerated_velocity(const mass_flow_controller_t* controller, external_force_per_unit_mass_t* external_acceleration[P4EST_DIM], const my_p4est_shs_channel_t& channel) const
  {
    return 5.0*(flow_condition == constant_mass_flow ? controller->targeted_bulk_velocity() : Reynolds*channel.canonical_u_tau(external_acceleration[0]->get_value()));
  }

  bool set_dt_and_update_grid(my_p4est_navier_stokes_t *ns)
  {
    if (use_adapted_dt)
      ns->compute_adapted_dt();
    else
      ns->compute_dt();
    dt = ns->get_dt();

    if (tn + dt > tstart + duration)
    {
      dt = tstart + duration - tn;
      ns->set_dt(dt);
    }

    if (save_vtk && dt > vtk_dt)
    {
      dt = vtk_dt; // so that we don't miss snapshots...
      ns->set_dt(dt);
    }

    return ns->update_from_tn_to_tnp1(NULL, (iter%steps_grid_update != 0), false);
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
        throw std::invalid_argument("export_and_accumulate_timings: could not open file for timings output.");
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

};

void truncate_exportation_file_up_to_tstart(const double &tstart, const std::string &filename, const bool &two_header_lines = false)
{
  FILE* fp = fopen(filename.c_str(), "r+");
  char* read_line = NULL;
  size_t len = 0;
  ssize_t len_read;
  long size_to_keep = 0;
  if (((len_read = getline(&read_line, &len, fp)) != -1))
    size_to_keep += (long) len_read;
  else
    throw std::runtime_error("simulation_setup::truncate_exportation_file_up_to_tstart: couldn't read the first header line of " + filename);
  if(two_header_lines)
  {
    if (((len_read = getline(&read_line, &len, fp)) != -1))
      size_to_keep += (long) len_read;
    else
      throw std::runtime_error("simulation_setup::truncate_exportation_file_up_to_tstart: couldn't read the second header line of " + filename);
  }
  double time;
  while ((len_read = getline(&read_line, &len, fp)) != -1) {
    sscanf(read_line, "%lg %*[^\n]", &time);
    if (time <= tstart - (1.0e-12)*pow(10.0, ceil(log10(tstart)))) // (1.0e-12)*pow(10.0, ceil(log10(tstart))) == given precision when exporting
      size_to_keep += (long) len_read;
    else
      break;
  }
  fclose(fp);
  if (read_line)
    free(read_line);
  if (truncate(filename.c_str(), size_to_keep))
    throw std::runtime_error("simulation_setup::truncate_exportation_file_up_to_tstart: couldn't truncate " + filename);
}

void initialize_velocity_profile_file(const std::string &filename, const my_p4est_navier_stokes_t* ns, const double &tstart)
{
  if (ns->get_mpirank() == 0)
  {
    if (!file_exists(filename))
    {
      FILE* fp_avg_profile = fopen(filename.c_str(), "w");
      if (fp_avg_profile == NULL)
        throw std::invalid_argument("initialize_velocity_profile_file: could not open file " + filename + ".");
      fprintf(fp_avg_profile, "%% __ | coordinates along y axis \n");
      fprintf(fp_avg_profile, "%% tn");
      for (int k = 0; k < ns->get_brick()->nxyztrees[1]*(1 << ns->get_lmax()); ++k)
        fprintf(fp_avg_profile, "  | %.12g", (-1.00 + (2.0/((double) (ns->get_brick()->nxyztrees[1]*(1 << ns->get_lmax()))))*(0.5 + k)));
      fprintf(fp_avg_profile, "\n");
      fprintf(fp_avg_profile, "%.12g", tstart);
      fclose(fp_avg_profile);
    }
    else
    {
      truncate_exportation_file_up_to_tstart(tstart, filename, true);

      FILE *fp_avg_profile = fopen(filename.c_str(), "a");
      fprintf(fp_avg_profile, "%.12g", tstart);
      fclose(fp_avg_profile);
    }
  }
}

void write_vector_to_binary_file(const std::vector<double> &myVector, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<char> osi{ ofs };
    const char* beginByte = (char*)&myVector[0];
    const char* endByte = (char*)&myVector.back() + sizeof(double);
    std::copy(beginByte, endByte, osi);
    ofs.flush();
    ofs.close();
}

void read_vector_from_file(const std::string &filename, std::vector<double> &to_return)
{
    std::vector<char> buffer{};
    std::ifstream ifs(filename, std::ios::in | std::ifstream::binary);
    std::istreambuf_iterator<char> iter(ifs);
    std::istreambuf_iterator<char> end{};
    std::copy(iter, end, std::back_inserter(buffer));
    ifs.close();
    to_return.resize(buffer.size() / sizeof(double));
    memcpy(&to_return[0], &buffer[0], buffer.size());
}

class velocity_profiler_t
{
  std::string               profile_path;
  std::string               file_slice_avg_velocity_profile;
  vector<std::string>       file_line_avg_velocity_profile;
  double                    t_slice_average;
  vector<double>            slice_averaged_profile;
  vector<double>            slice_averaged_profile_nm1;
  vector<double>            time_averaged_slice_averaged_profile;
  vector<double>            t_line_average;
  vector< vector<double> >  line_averaged_profiles;
  vector< vector<double> >  line_averaged_profiles_nm1;
  vector< vector<double> >  time_averaged_line_averaged_profiles;
  vector<unsigned int>      bin_index;
  int                       iter_export_profile;
  unsigned int              nbins;
  void set_number_of_bins(const unsigned int &nbins_)
  {
    nbins = nbins_;
    t_line_average.resize(nbins);
    line_averaged_profiles.resize(nbins);
    line_averaged_profiles_nm1.resize(nbins);
    time_averaged_line_averaged_profiles.resize(nbins);
  }

  void initialize_averaged_velocity_profiles(const my_p4est_navier_stokes_t* ns, const double &tstart, const my_p4est_shs_channel_t &channel)
  {
    PetscErrorCode ierr;
    file_slice_avg_velocity_profile = profile_path + "/slice_averaged_velocity_profile.dat";
    ierr = PetscPrintf(ns->get_mpicomm(), "Saving slice-averaged velocity profile in %s\n", file_slice_avg_velocity_profile.c_str()); CHKERRXX(ierr);
    initialize_velocity_profile_file(file_slice_avg_velocity_profile, ns, tstart);
  #ifdef P4_TO_P8
    const double smallest_traverse_length_scale = (channel.spanwise_grooves() ? channel.length()/(ns->get_brick()->nxyztrees[0]*(1 << ns->get_lmax())) : channel.width()/(ns->get_brick()->nxyztrees[2]*(1 << ns->get_lmax())));
  #else
    const double smallest_traverse_length_scale = channel.length()/(ns->get_brick()->nxyztrees[0]*(1 << ns->get_lmax()));
  #endif

    const unsigned int nb_cells_in_groove = (unsigned int) (channel.get_pitch()*channel.GF()/smallest_traverse_length_scale);
    const unsigned int nb_cells_in_ridge  = (unsigned int) (channel.get_pitch()*(1.0 - channel.GF())/smallest_traverse_length_scale);
    const unsigned int nb_cells_to_map = nb_cells_in_groove + nb_cells_in_ridge;
    P4EST_ASSERT(nb_cells_to_map == (unsigned int) (channel.get_pitch()/smallest_traverse_length_scale));
    bin_index.resize(nb_cells_to_map);
    const unsigned int nbins_in_groove = (nb_cells_in_groove+((nb_cells_in_groove%2==1)? 1:0))/2;
  #ifdef P4_TO_P8
    const unsigned int bin_offset = (channel.spanwise_grooves() ? 1 : 0);
  #else
    const unsigned int bin_offset = 1;
  #endif
    if (nb_cells_in_groove%2 == 0)
    {
      bin_index[bin_offset + nbins_in_groove - 1] = 0;
      bin_index[bin_offset + nbins_in_groove]     = 0;
    }
    else
      bin_index[bin_offset + nbins_in_groove - 1] = 0;
    for (unsigned int bin_idx = 1; bin_idx < nbins_in_groove; ++bin_idx)
    {
      if (nb_cells_in_groove%2 == 0)
      {
        bin_index[bin_offset + nbins_in_groove - 1 - bin_idx] = bin_idx;
        bin_index[bin_offset + nbins_in_groove + bin_idx]     = bin_idx;
      }
      else
      {
        bin_index[bin_offset + nbins_in_groove - 1 - bin_idx] = bin_idx;
        bin_index[bin_offset + nbins_in_groove - 1 + bin_idx] = bin_idx;
      }
    }
    const unsigned int nbins_in_ridge = (nb_cells_in_ridge + (nb_cells_in_ridge%2 == 1 ? 1:0))/2;
    set_number_of_bins(nbins_in_groove + nbins_in_ridge);
    if (nb_cells_in_ridge%2 == 0)
    {
      bin_index[(bin_offset + nb_cells_in_groove + nbins_in_ridge - 1)%nb_cells_to_map] = nbins - 1;
      bin_index[(bin_offset + nb_cells_in_groove+nbins_in_ridge)%nb_cells_to_map]       = nbins - 1;
    }
    else
      bin_index[(bin_offset + nb_cells_in_groove + nbins_in_ridge - 1)%nb_cells_to_map] = nbins - 1;
    for (unsigned int bin_idx = 1; bin_idx < nbins_in_ridge; ++bin_idx)
    {
      if (nb_cells_in_ridge%2 == 0)
      {
        bin_index[(bin_offset + nb_cells_in_groove + nbins_in_ridge - 1 - bin_idx)%nb_cells_to_map] = nbins - 1 - bin_idx;
        bin_index[(bin_offset + nb_cells_in_groove + nbins_in_ridge + bin_idx)%nb_cells_to_map]     = nbins - 1 - bin_idx;
      }
      else
      {
        bin_index[(bin_offset + nb_cells_in_groove + nbins_in_ridge - 1 - bin_idx)%nb_cells_to_map]  = nbins - 1 - bin_idx;
        bin_index[(bin_offset + nb_cells_in_groove + nbins_in_ridge - 1 + bin_idx)%nb_cells_to_map]  = nbins - 1 - bin_idx;
      }
    }
    file_line_avg_velocity_profile.resize(nbins);
    for (unsigned int bin_idx = 0; bin_idx < nbins; ++bin_idx) {
      file_line_avg_velocity_profile[bin_idx] = std::string(profile_path) + "/line_averaged_velocity_profile_index_" + std::to_string(bin_idx) + ".dat";
      ierr = PetscPrintf(ns->get_mpicomm(), "Saving line-averaged velocity profile in %s\n", file_line_avg_velocity_profile[bin_idx].c_str()); CHKERRXX(ierr);
      initialize_velocity_profile_file(file_line_avg_velocity_profile[bin_idx].c_str(), ns, tstart);
    }
    int mpiret = MPI_Barrier(ns->get_mpicomm()); SC_CHECK_MPI(mpiret);
  }

public:
  velocity_profiler_t(const cmdParser cmd, const my_p4est_navier_stokes_t* ns, const simulation_setup &setup, const my_p4est_shs_channel_t &channel)
  {
    profile_path = setup.export_dir +  "/profiles";
    if (create_directory(profile_path, ns->get_mpirank(), ns->get_mpicomm()))
      throw std::runtime_error("velocity_profiler_t::velocity_profiler_t(...): could not create exportation directory for velocity profiles " + profile_path);
    initialize_averaged_velocity_profiles(ns, setup.tstart, channel);
    iter_export_profile = 0;


    if (cmd.contains("restart"))
    {
      bool all_binary_files_nm1_are_there = true;
      if (ns->get_mpirank() == 0)
        all_binary_files_nm1_are_there = all_binary_files_nm1_are_there && file_exists(profile_path + "/slice_velocity_profile_nm1.bin");

      for (unsigned int bin_idx = 0; bin_idx < nbins; ++bin_idx)
        if ((unsigned int) ns->get_mpirank() == bin_idx%ns->get_mpisize())
          all_binary_files_nm1_are_there = all_binary_files_nm1_are_there && file_exists(profile_path + "/line_velocity_profile_nm1_index_" + std::to_string(bin_idx) + ".bin");

      int load_binary_files = (all_binary_files_nm1_are_there ? 1 : 0);
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &load_binary_files, 1, MPI_INT, MPI_LAND, ns->get_mpicomm()); SC_CHECK_MPI(mpiret);

      if (load_binary_files)
      {
        if (ns->get_mpirank() == 0)
        {
          read_vector_from_file(profile_path + "/slice_velocity_profile_nm1.bin", slice_averaged_profile_nm1);
          time_averaged_slice_averaged_profile.resize(slice_averaged_profile_nm1.size());
          t_slice_average = setup.tstart;
        }

        for (unsigned int bin_idx = 0; bin_idx < nbins; ++bin_idx)
          if ((unsigned int) ns->get_mpirank() == bin_idx%ns->get_mpisize())
          {
            read_vector_from_file(profile_path + "/line_velocity_profile_nm1_index_" + std::to_string(bin_idx)+ ".bin", line_averaged_profiles_nm1[bin_idx]);
            time_averaged_line_averaged_profiles[bin_idx].resize(line_averaged_profiles_nm1[bin_idx].size());
            t_line_average[bin_idx] = setup.tstart;
          }

        iter_export_profile = 1; // restarting so +1
      }
    }
  }

  void gather_and_dump_profiles(const simulation_setup &setup, my_p4est_navier_stokes_t* ns, const double& u_scaling ONLY3D(COMMA const bool& spanwise))
  {
    ns->get_slice_averaged_vnp1_profile(dir::x, dir::y, slice_averaged_profile, u_scaling);
    if (ns->get_mpirank() == 0)
    {
      if (iter_export_profile == 0)
      {
        t_slice_average = setup.tn;
        slice_averaged_profile_nm1.resize(slice_averaged_profile.size(), 0.0);
        time_averaged_slice_averaged_profile.resize(slice_averaged_profile.size(), 0.0);
      }
      else
      {
        for (unsigned int idx = 0; idx < slice_averaged_profile.size(); ++idx)
        {
          time_averaged_slice_averaged_profile[idx] += 0.5*setup.dt*(slice_averaged_profile_nm1[idx] + slice_averaged_profile[idx]);
          slice_averaged_profile_nm1[idx] = slice_averaged_profile[idx];
        }
      }

      if (iter_export_profile != 0 && (iter_export_profile%setup.nexport_avg == 0 || setup.time_to_save_state()))
      {
        // we export velocity profile data every nexport_avg iterations *OR* if the solver state is about to be exported at the beginning of next iteration
        // this second condition avoids truncation errors in the relevant data files when restarting the simulation from a saved solver state
        // In the latter case, we need to export nm1 profiles for it to be read in case of restart
        if (setup.time_to_save_state())
        {
          const std::string path_to_binary_slice_velocity_profile_nm1 = profile_path + "/slice_velocity_profile_nm1.bin";
          if (file_exists(path_to_binary_slice_velocity_profile_nm1))
            if (remove(path_to_binary_slice_velocity_profile_nm1.c_str()) != 0)
              throw std::runtime_error("main_shs_:: error when deleting file " + path_to_binary_slice_velocity_profile_nm1);
          write_vector_to_binary_file(slice_averaged_profile_nm1, path_to_binary_slice_velocity_profile_nm1);
          //            std::ofstream binary_slice_velocity_profile_nm1(path_to_binary_slice_velocity_profile_nm1, std::ios::out | std::ofstream::binary);
          //            std::copy(slice_averaged_profile_nm1.begin(), slice_averaged_profile_nm1.end(), std::ostreambuf_iterator<char>(binary_slice_velocity_profile_nm1));
          //            binary_slice_velocity_profile_nm1.flush();
          //            binary_slice_velocity_profile_nm1.close();
        }
        FILE* fp_velocity_profile = fopen(file_slice_avg_velocity_profile.c_str(), "a");
        if (fp_velocity_profile == NULL)
          throw std::invalid_argument("main_shs_" + std::to_string(P4EST_DIM) + "d: could not open file for slice-averaged velocity profile output.");
        for (int k = 0; k < ns->get_brick()->nxyztrees[1]*(1 << ns->get_lmax()); ++k)
        {
          fprintf(fp_velocity_profile, " %.12g", time_averaged_slice_averaged_profile.at(k)/(setup.tn - t_slice_average));
          time_averaged_slice_averaged_profile.at(k) = 0.0;
        }
        fprintf(fp_velocity_profile, "\n");
        // write the next start time
        fprintf(fp_velocity_profile, "%.12g", setup.tn);
        fclose(fp_velocity_profile);
        // reset these
        t_slice_average = setup.tn;
      }
    }
    ns->get_line_averaged_vnp1_profiles(DIM(dir::x, dir::y, spanwise ? dir::z : dir::x), bin_index, line_averaged_profiles, u_scaling);
    for (unsigned int bin_idx = 0; bin_idx < nbins; ++bin_idx) {
      if ((unsigned int) ns->get_mpirank() == bin_idx%ns->get_mpisize())
      {
        if (iter_export_profile == 0)
        {
          t_line_average[bin_idx] = setup.tn;
          line_averaged_profiles_nm1[bin_idx].resize(line_averaged_profiles[bin_idx].size(), 0.0);
          time_averaged_line_averaged_profiles[bin_idx].resize(line_averaged_profiles[bin_idx].size(), 0.0);
        }
        else
        {
          for (size_t idx = 0; idx < line_averaged_profiles[bin_idx].size(); ++idx)
          {
            time_averaged_line_averaged_profiles[bin_idx][idx] += 0.5*setup.dt*(line_averaged_profiles_nm1[bin_idx][idx] + line_averaged_profiles[bin_idx][idx]);
            line_averaged_profiles_nm1[bin_idx][idx] = line_averaged_profiles[bin_idx][idx];
          }
        }

        if (iter_export_profile != 0 && (iter_export_profile%setup.nexport_avg == 0 || (setup.time_to_save_state())))
        {
          // we export velocity profile data every nexport_avg iterations *OR* if the solver state is about to be exported at the beginning of next iteration
          // this second condition avoids truncation errors in the relevant data files when restarting the simulation from a saved solver state
          // In the latter case, we need to export nm1 profiles for it to be read in case of restart
          if (setup.time_to_save_state())
          {
            const std::string path_to_binary_line_velocity_profile_nm1 = profile_path + "/line_velocity_profile_nm1_index_" + std::to_string(bin_idx) + ".bin";
            if (file_exists(path_to_binary_line_velocity_profile_nm1))
              if (remove(path_to_binary_line_velocity_profile_nm1.c_str()) != 0)
                throw std::runtime_error("main_shs_:: error when deleting file " + std::string(path_to_binary_line_velocity_profile_nm1));
            write_vector_to_binary_file(line_averaged_profiles_nm1[bin_idx], path_to_binary_line_velocity_profile_nm1);
            //              std::ofstream binary_line_velocity_profile_nm1(path_to_binary_line_velocity_profile_nm1, std::ios::out | std::ofstream::binary);
            //              std::copy(line_averaged_profiles_nm1[bin_idx].begin(), line_averaged_profiles_nm1[bin_idx].end(), std::ostreambuf_iterator<char>(binary_line_velocity_profile_nm1));
            //              binary_line_velocity_profile_nm1.flush();
            //              binary_line_velocity_profile_nm1.close();
          }
          FILE* fp_velocity_profile = fopen(file_line_avg_velocity_profile[bin_idx].c_str(), "a");
          if (fp_velocity_profile == NULL)
            throw std::invalid_argument("main_shs_" + std::to_string(P4EST_DIM) + "d: could not open file for line-averaged velocity profile output.");
          for (int k = 0; k < ns->get_brick()->nxyztrees[1]*(1 << ns->get_lmax()); ++k)
          {
            fprintf(fp_velocity_profile, " %.12g", time_averaged_line_averaged_profiles[bin_idx].at(k)/(setup.tn - t_line_average[bin_idx]));
            time_averaged_line_averaged_profiles[bin_idx].at(k) = 0.0;
          }
          fprintf(fp_velocity_profile, "\n");
          // write the next start time
          fprintf(fp_velocity_profile, "%.12g", setup.tn);
          fclose(fp_velocity_profile);
          // reset these
          t_line_average[bin_idx] = setup.tn;
        }
      }
    }
    // if we exported velocity profile data because of an exported solver state but not because of a reached value of iter_export_profile, we reset its value to 0!
    if (iter_export_profile != 0 && setup.time_to_save_state() && iter_export_profile%setup.nexport_avg != 0)
      iter_export_profile = 0;

    iter_export_profile++;
  }

};

void initialize_monitoring(simulation_setup &setup, const my_p4est_navier_stokes_t* ns)
{
  setup.file_monitoring = setup.export_dir + "/flow_monitoring.dat";

  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Monitoring flow in ... %s\n", setup.file_monitoring.c_str()); CHKERRXX(ierr);
  if (ns->get_mpirank() == 0)
  {
    if (!file_exists(setup.file_monitoring))
    {
      FILE* fp_monitor = fopen(setup.file_monitoring.c_str(), "w");
      if (fp_monitor == NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for flow monitoring.");
      fprintf(fp_monitor, "%% tn | Re_tau | Re_b \n");
      fclose(fp_monitor);
    }
    else
      truncate_exportation_file_up_to_tstart(setup.tstart, setup.file_monitoring);

    const std::string liveplot_Re = setup.export_dir + "/live_monitor.gnu";
    if (!file_exists(liveplot_Re))
    {
      FILE *fp_liveplot_Re= fopen(liveplot_Re.c_str(), "w");
      if (fp_liveplot_Re == NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for Re liveplot.");
      fprintf(fp_liveplot_Re, "set term wxt noraise\n");
      fprintf(fp_liveplot_Re, "set key bottom right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_Re, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_Re, "set ylabel \"Re\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_Re, "plot");
      fprintf(fp_liveplot_Re, "\t \"flow_monitoring.dat\" using 1:2 title 'Re_{tau}' with lines lw 3,\\\n");
      fprintf(fp_liveplot_Re, "\t \"flow_monitoring.dat\" using 1:3 title 'Re_b' with lines lw 3\n");
      fprintf(fp_liveplot_Re, "pause 4\n");
      fprintf(fp_liveplot_Re, "reread");
      fclose(fp_liveplot_Re);
    }

    const std::string tex_plot_Re = setup.export_dir + "/tex_monitor.gnu";
    if (!file_exists(tex_plot_Re))
    {
      FILE *fp_tex_plot_Re = fopen(tex_plot_Re.c_str(), "w");
      if (fp_tex_plot_Re == NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for Re monitoring figure.");
      fprintf(fp_tex_plot_Re, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_Re, "set output 'monitor_history.tex'\n");
      fprintf(fp_tex_plot_Re, "set key bottom right Left \n");
      fprintf(fp_tex_plot_Re, "set xlabel \"$t$\"\n");
      fprintf(fp_tex_plot_Re, "set ylabel \"$\\\\mathrm{Re}$\" \n");
      fprintf(fp_tex_plot_Re, "plot");
      fprintf(fp_tex_plot_Re, "\t \"flow_monitoring.dat\" using 1:2 title '$\\mathrm{Re}_{\\tau}$' with lines lw 3,\\\n");
      fprintf(fp_tex_plot_Re, "\t \"flow_monitoring.dat\" using 1:3 title '$\\mathrm{Re}_{\\mathrm{b}}$' with lines lw 3");
      fclose(fp_tex_plot_Re);
    }

    const std::string tex_Re_script = setup.export_dir + "/plot_tex_monitor.sh";
    if (!file_exists(tex_Re_script))
    {
      FILE *fp_tex_monitor_script = fopen(tex_Re_script.c_str(), "w");
      if (fp_tex_monitor_script == NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for bash script plotting monitoring tex figure.");
      fprintf(fp_tex_monitor_script, "#!/bin/sh\n");
      fprintf(fp_tex_monitor_script, "gnuplot ./tex_monitor.gnu\n");
      fprintf(fp_tex_monitor_script, "latex ./monitor_history.tex\n");
      fprintf(fp_tex_monitor_script, "dvipdf -dAutoRotatePages=/None ./monitor_history.dvi\n");
      fclose(fp_tex_monitor_script);

      std::ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_Re_script;
      if (system(chmod_command.str().c_str()))
        throw std::runtime_error("initialize_monitoring: could not make the plot_tex_monitor.sh script executable");
    }
  }
}

void initialize_drag_force_output(simulation_setup &setup, const my_p4est_navier_stokes_t* ns)
{
  setup.file_drag = setup.export_dir + "/drag_monitoring.dat";

  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Saving drag in ... %s\n", setup.file_drag.c_str()); CHKERRXX(ierr);

  if (ns->get_mpirank() == 0)
  {
    if (!file_exists(setup.file_drag))
    {
      FILE* fp_drag = fopen(setup.file_drag.c_str(), "w");
      if (fp_drag == NULL)
        throw std::runtime_error("initialize_drag_force_output: could not open file for drag output.");
      fprintf(fp_drag, "%% __ | Normalized drag \n");
      fprintf(fp_drag, "%s", (std::string("%% tn | x-component | y-component") ONLY3D(+ std::string(" | z-component")) + std::string("\n")).c_str());
      fclose(fp_drag);
    }
    else
      truncate_exportation_file_up_to_tstart(setup.tstart, setup.file_drag, true);

    const std::string liveplot_drag = setup.export_dir + "/live_drag.gnu";
    if (!file_exists(liveplot_drag))
    {
      FILE* fp_liveplot_drag = fopen(liveplot_drag.c_str(), "w");
      if (fp_liveplot_drag == NULL)
        throw std::runtime_error("initialize_drag_force_output: could not open file for drage force liveplot.");
      fprintf(fp_liveplot_drag, "set term wxt noraise\n");
      fprintf(fp_liveplot_drag, "set key center right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_drag, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_drag, "set ylabel \"Nondimensional drag \" font \"Arial,14\"\n");
      fprintf(fp_liveplot_drag, "plot");
      for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_liveplot_drag, "\t \"drag_monitoring.dat\" using 1:%d title '%c-component' with lines lw 3",  (int) dd + 2, (dd == dir::x ? 'x' : ONLY3D(OPEN_PARENTHESIS dd == dir::y ?) 'y' ONLY3D(: 'z' CLOSE_PARENTHESIS)));
        if (dd < P4EST_DIM - 1)
          fprintf(fp_liveplot_drag, ",\\");
        fprintf(fp_liveplot_drag, "\n");
      }
      fprintf(fp_liveplot_drag, "pause 4\n");
      fprintf(fp_liveplot_drag, "reread");
      fclose(fp_liveplot_drag);
    }

    const std::string tex_plot_drag = setup.export_dir + "/tex_drag.gnu";
    if (!file_exists(tex_plot_drag))
    {
      FILE *fp_tex_plot_drag = fopen(tex_plot_drag.c_str(), "w");
      if (fp_tex_plot_drag == NULL)
        throw std::runtime_error("initialize_drag_foce_output: could not open file for drag force tex figure.");
      fprintf(fp_tex_plot_drag, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_drag, "set output 'drag_history.tex'\n");
      fprintf(fp_tex_plot_drag, "set key center right Left \n");
      fprintf(fp_tex_plot_drag, "set xlabel \"$t$\"\n");
      fprintf(fp_tex_plot_drag, "%s", (std::string("set ylabel \"Non-dimensional wall friction $\\\\mathbf{D} = \\\\frac{-\\\\hat{\\\\mathbf{D}}}{2\\\\rho U_{\\\\mathrm{b}}^2 L") ONLY3D(+ std::string(" W")) + std::string("}$ \" \n")).c_str());
      fprintf(fp_tex_plot_drag, "plot");
      for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_tex_plot_drag, "\t \"drag_monitoring.dat\" using 1:%d title '$D_{\\mathrm{%c}}$' with lines lw 3",  (int) dd + 2, (dd == dir::x ? 'x' : ONLY3D(OPEN_PARENTHESIS dd == dir::y ?) 'y' ONLY3D(: 'z' CLOSE_PARENTHESIS)));
        if (dd < P4EST_DIM - 1)
          fprintf(fp_tex_plot_drag, ",\\\n");
      }
      fclose(fp_tex_plot_drag);
    }

    const std::string tex_drag_script = setup.export_dir + "/plot_tex_drag.sh";
    if (!file_exists(tex_drag_script))
    {
      FILE *fp_tex_drag_script = fopen(tex_drag_script.c_str(), "w");
      if (fp_tex_drag_script == NULL)
        throw std::runtime_error("initialize_drag_force_output: could not open file for bash script plotting drag tex figure.");
      fprintf(fp_tex_drag_script, "#!/bin/sh\n");
      fprintf(fp_tex_drag_script, "gnuplot ./tex_drag.gnu\n");
      fprintf(fp_tex_drag_script, "latex ./drag_history.tex\n");
      fprintf(fp_tex_drag_script, "dvipdf -dAutoRotatePages=/None ./drag_history.dvi\n");
      fclose(fp_tex_drag_script);

      std::ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_drag_script;
      int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
    }
  }
}

void initialize_timing_output(simulation_setup & setup, const my_p4est_navier_stokes_t *ns)
{
  const std::string filename = "timing_monitoring.dat";
  setup.file_timings = setup.export_dir + "/" + filename;
  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Saving timings per time step in ... %s\n", setup.file_timings.c_str()); CHKERRXX(ierr);

  if(ns->get_mpirank() == 0)
  {
    if(!file_exists(setup.file_timings))
    {
      FILE* fp_timing = fopen(setup.file_timings.c_str(), "w");
      if(fp_timing == NULL)
        throw std::runtime_error("initialize_timing_output: could not open file for timing output.");
      fprintf(fp_timing, "%s", (std::string("%% tn | grid update | viscosity step | projection setp | interpolate velocities || number of fixed-point iterations | extra work on projection | extra work on viscous step \n")).c_str());
      fclose(fp_timing);
    }
    else
      truncate_exportation_file_up_to_tstart(setup.tstart, setup.file_timings);

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
      fprintf(fp_liveplot_timings, "plot \t \"%s\" using 1:(100*$2/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=0 title 'Grid update',\\\n", filename.c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Viscous step (approximate projection)',\\\n", filename.c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Projection step (approximate projection)',\\\n", filename.c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves below y2=100 title 'Interpolating velocities from faces to nodes',\\\n", filename.c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=100 title 'Extra viscous steps (fixed-point iteration(s))', \\\n", filename.c_str());
      fprintf(fp_liveplot_timings, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) : (100*($2 + $3 + $4 + $5)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Extra projection steps (fixed-point iteration(s))' \n", filename.c_str());
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
      fprintf(fp_tex_plot_timing, "plot \t \"%s\" using 1:(100*$2/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=0 title 'Grid update',\\\n", filename.c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Viscous step (approximate projection)',\\\n", filename.c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 - $7)/($2 + $3 + $4 + $5 - $7 - $8)):(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Projection step (approximate projection)',\\\n", filename.c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 + $4 - $7 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves below y2=100 title 'Interpolating velocities from faces to nodes',\\\n", filename.c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves above y1=100 title 'Extra viscous steps (fixed-point iteration(s))', \\\n", filename.c_str());
      fprintf(fp_tex_plot_timing, "\t \"%s\" using 1:(100*($2 + $3 + $4 + $5 - $8)/($2 + $3 + $4 + $5 - $7 - $8)) : (100*($2 + $3 + $4 + $5)/($2 + $3 + $4 + $5 - $7 - $8)) with filledcurves title 'Extra projection steps (fixed-point iteration(s))' \n", filename.c_str());
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

      std::ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_timing_script;
      int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
    }
  }
}

void load_solver_from_state(const mpi_environment_t &mpi, const cmdParser &cmd,
                            my_p4est_navier_stokes_t* &ns, my_p4est_brick_t* &brick, my_p4est_shs_channel_t &channel,
                            external_force_per_unit_mass_t* external_acceleration[P4EST_DIM], splitting_criteria_cf_and_uniform_band_t* &data,
                            mass_flow_controller_t* &controller, simulation_setup &setup)
{
  const std::string backup_directory = cmd.get<std::string>("restart", "");
  if (!is_folder(backup_directory.c_str()))
    throw std::invalid_argument("load_solver_from_state: the restart path " + backup_directory + " is not an accessible directory.");
  if (ORD(cmd.contains("nx"), cmd.contains("ny"), cmd.contains("nz")) || ORD(cmd.contains("length"), cmd.contains("height"), cmd.contains("width")))
    throw std::invalid_argument("load_solver_from_state: the length, height and width as well as the numbers of trees along x, y and z cannot be reset when restarting a simulation.");

  if (ns != NULL)
    delete  ns;

  ns                      = new my_p4est_navier_stokes_t(mpi, backup_directory.c_str(), setup.tstart);
  setup.dt                = ns->get_dt();
  p4est_t *p4est_n        = ns->get_p4est();
  p4est_t *p4est_nm1      = ns->get_p4est_nm1();
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    if (is_periodic(p4est_n, dir) != periodicity[dir] || is_periodic(p4est_nm1, dir) != periodicity[dir])
      throw std::invalid_argument("load_solver_from_state: the periodicity from the loaded state does not match the requirements.");

  if (brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick                   = ns->get_brick();
  channel.configure(brick, DIM(cmd.get<double>("pitch", default_pitch_to_height*ns->get_height_of_domain()),
                               cmd.get<double>("GF", default_gas_fraction),
                               cmd.contains("spanwise")), cmd.get<int>("lmax", ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl));

  if (controller != NULL)
    delete controller;
  controller = new mass_flow_controller_t(dir::x, brick->xyz_min[dir::x]);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if(external_acceleration[dir] != NULL)
      delete external_acceleration[dir];
    external_acceleration[dir] = new external_force_per_unit_mass_t;
  }
  if (setup.flow_condition == constant_mass_flow)
  {
    const double desired_U_b    = ns->get_nu()*setup.Reynolds/channel.delta();
    controller->activate_forcing(desired_U_b);
    if(cmd.contains("initial_Re_tau"))
      external_acceleration[dir::x]->set_value(channel.acceleration_for_canonical_u_tau(ns->get_nu()*cmd.get<double>("initial_Re_tau")/channel.delta()));
    else
      external_acceleration[dir::x]->set_value(channel.acceleration_for_constant_mass_flow(desired_U_b, setup.Reynolds, setup.nterms_in_series));
    P4EST_ASSERT(fabs(channel.Re_b(ns->get_rho()*MULTD(controller->targeted_bulk_velocity(), channel.height(), channel.width()), ns->get_rho(), ns->get_nu()) - setup.Reynolds) < setup.Reynolds*10.0*EPS);
  }
  else
  {
    const double desired_u_tau  = ns->get_nu()*setup.Reynolds/channel.delta();
    external_acceleration[dir::x]->set_value(channel.acceleration_for_canonical_u_tau(desired_u_tau));
    P4EST_ASSERT(fabs(channel.canonical_Re_tau(external_acceleration[0]->get_value(), ns->get_nu()) - setup.Reynolds) < setup.Reynolds*10.0*EPS);
  }

  double lip = ((splitting_criteria_t*) p4est_n->user_pointer)->lip;
  if (cmd.contains("lip")) // reset to match the desired value if given
    lip                   = channel.calculate_lip_for_ns_solver(cmd.get<double>("lip"));

  double uniform_band     = ns->get_uniform_band();
  if (cmd.contains("wall_layer")) // reset the uniform band to desired value if given
    uniform_band          = channel.calculate_uniform_band_for_ns_solver(cmd.get<unsigned int>("wall_layer"));

  if (data != NULL) {
    delete data; data = NULL;
  }
  P4EST_ASSERT(data == NULL);
  data = new splitting_criteria_cf_and_uniform_band_t(cmd.get<int>("lmin", ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl), channel.lmax(), &channel, uniform_band, lip);
  splitting_criteria_t* to_delete = (splitting_criteria_t*) p4est_n->user_pointer;
  bool fix_restarted_grid = (channel.lmax() != to_delete->max_lvl);
  delete to_delete;
  p4est_n->user_pointer   = (void*) data;
  p4est_nm1->user_pointer = (void*) data; // p4est_n and p4est_nm1 always point to the same splitting_criteria_t no need to delete the nm1 one, it's just been done
  ns->set_parameters(ns->get_mu(), ns->get_rho(), cmd.get<int>("sl_order", ns->get_sl_order()), uniform_band, cmd.get<double>("thresh", ns->get_split_threshold()), cmd.get<double>("cfl", ns->get_cfl()));

  ns->set_bc(channel.get_bc_on_velocity(), channel.get_bc_on_pressure());
  CF_DIM* tmp[P4EST_DIM] = {DIM(external_acceleration[0], external_acceleration[1], external_acceleration[2])};
  ns->set_external_forces_per_unit_mass(tmp);
  if (fix_restarted_grid)
    ns->refine_coarsen_grid_after_restart(&channel, false);

  if(setup.save_vtk)
    setup.update_export_vtk(); // so that we don't overwrite visualization files that were possibly already exported...

  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Simulation restarted from state saved in %s\n", (cmd.get<std::string>("restart")).c_str()); CHKERRXX(ierr);
}

p4est_connectivity_t* build_brick_and_get_connectivity(my_p4est_brick_t* &brick, const cmdParser &cmd)
{
  const double length     = cmd.get<double>("length", default_length);
  const double height     = cmd.get<double>("height", default_height);
#ifdef P4_TO_P8
  const double width      = cmd.get<double>("width",  default_width);
#endif
  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];
  int n_tree_xyz[P4EST_DIM];
  n_tree_xyz[0]           = cmd.get<int>("nx", (int) (default_ntree_y*length/height));  xyz_min[0] = -0.5*length; xyz_max[0] = 0.5*length;
  n_tree_xyz[1]           = cmd.get<int>("ny", (int) height);                           xyz_min[1] = -0.5*height; xyz_max[1] = 0.5*height;
#ifdef P4_TO_P8
  n_tree_xyz[2]           = cmd.get<int>("nz", (int) (default_ntree_y*width/height));   xyz_min[2] = -0.5*width;  xyz_max[2] = 0.5*width;
#endif
  if (brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;
  return my_p4est_brick_new(n_tree_xyz, xyz_min, xyz_max, brick, periodicity);
}

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd,
                                my_p4est_navier_stokes_t* &ns, my_p4est_brick_t* &brick, my_p4est_shs_channel_t &channel,
                                external_force_per_unit_mass_t* external_acceleration[P4EST_DIM], splitting_criteria_cf_and_uniform_band_t* &data,
                                mass_flow_controller_t* &controller, simulation_setup &setup)
{
  p4est_connectivity_t* connectivity = build_brick_and_get_connectivity(brick, cmd);
  channel.configure(brick, DIM(cmd.get<double>("pitch", default_pitch_to_height*(brick->xyz_max[1] - brick->xyz_min[1])), cmd.get<double>("GF", default_gas_fraction), cmd.contains("spanwise")), cmd.get<int>("lmax", default_lmax));
  // create grid at time nm1
  p4est_t *p4est_nm1        = NULL;
  p4est_ghost_t* ghost_nm1  = NULL;
  p4est_nodes_t* nodes_nm1  = NULL;
  channel.create_p4est_ghost_and_nodes(p4est_nm1, ghost_nm1, nodes_nm1, data, connectivity, mpi,
                                       cmd.get<int>("lmin", default_lmin), cmd.get<unsigned int>("wall_layer", default_wall_layer), cmd.get<double>("lip", default_lip));
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);
  /* create the initial forest at time n (copy of the former one) */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = p4est_nm1->user_pointer; // just to make sure

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);
  const double tree_dim[P4EST_DIM] = {DIM(channel.length()/brick->nxyztrees[0], channel.height()/brick->nxyztrees[1], channel.width()/brick->nxyztrees[2])};
  if(third_degree_ghost_are_required(tree_dim))
    my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n         = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c);

  Vec phi;
  PetscErrorCode ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, channel, phi);

  // create the solver
  ns = new my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n);
  ns->set_phi(phi);

  if (controller != NULL)
    delete controller;
  controller = new mass_flow_controller_t(dir::x, brick->xyz_min[dir::x]);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if(external_acceleration[dir] != NULL)
      delete external_acceleration[dir];
    external_acceleration[dir] = new external_force_per_unit_mass_t;
  }

  const double mass_density = 1.0;
  if (setup.flow_condition == constant_pressure_gradient)
  {
    const double desired_u_tau  = 1.0;
    const double viscosity      = mass_density*desired_u_tau*channel.delta()/setup.Reynolds;
    external_acceleration[dir::x]->set_value(channel.acceleration_for_canonical_u_tau(desired_u_tau));
    ns->set_parameters(viscosity, mass_density, cmd.get<int>("sl_order", default_sl_order), data->uniform_band, cmd.get<double>("thresh", default_thresh), cmd.get<double>("cfl", default_cfl));
    P4EST_ASSERT(fabs(channel.canonical_Re_tau(external_acceleration[0]->get_value(), ns->get_nu()) - setup.Reynolds) < setup.Reynolds*10.0*EPS);
  }
  else
  {
    const double desired_U_b    = 1.0;
    controller->activate_forcing(desired_U_b);
    external_acceleration[dir::x]->set_value(channel.acceleration_for_constant_mass_flow(desired_U_b, setup.Reynolds, setup.nterms_in_series));
    const double viscosity      = mass_density*desired_U_b*channel.delta()/setup.Reynolds;
    ns->set_parameters(viscosity, mass_density, cmd.get<int>("sl_order", default_sl_order), data->uniform_band, cmd.get<double>("thresh", default_thresh), cmd.get<double>("cfl", default_cfl));
    P4EST_ASSERT(fabs(channel.Re_b(ns->get_rho()*MULTD(controller->targeted_bulk_velocity(), channel.height(), channel.width()), ns->get_rho(), ns->get_nu()) - setup.Reynolds) < setup.Reynolds*10.0*EPS);
  }

  setup.tstart = 0.0; // no restart so we assume we start from 0.0
  channel.initialize_velocity(ns, setup.nterms_in_series, setup.flow_condition, setup.Reynolds, setup.white_noise_rms);
  ns->compute_dt(); // dt_n is now calculated to a correct value but dt_nm1 is still a bit shitty/not well-defined
  setup.dt = ns->get_dt(); // we get that value of dt_n into dt
  ns->set_dt(setup.dt, setup.dt); // we set dt_nm1 = dt_n = dt

  ns->set_bc(channel.get_bc_on_velocity(), channel.get_bc_on_pressure());
  CF_DIM* tmp[P4EST_DIM] = {DIM(external_acceleration[0], external_acceleration[1], external_acceleration[2])};
  ns->set_external_forces_per_unit_mass(tmp);
}

void check_accuracy_of_solution(my_p4est_navier_stokes_t* ns, my_p4est_shs_channel_t &channel, simulation_setup& setup)
{
  double my_errors[2*P4EST_DIM];
  for (unsigned char dir = 0; dir < 2*P4EST_DIM; ++dir)
    my_errors[dir] = 0.0;
  PetscErrorCode ierr;

  // calculate the analytical solution if not done, yet
  channel.solve_for_truncated_series(setup.nterms_in_series);
  channel.check_Reynolds(setup.flow_condition, setup.Reynolds);

  // Compute the face errors
  const Vec* v_faces; v_faces = ns->get_vnp1();
  const double *v_faces_ptr;

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(v_faces[dir], &v_faces_ptr); CHKERRXX(ierr);

    for (p4est_locidx_t f = 0; f < ns->get_faces()->num_local[dir]; ++f)
    {
      double xyz[P4EST_DIM]; ns->get_faces()->xyz_fr_f(f, dir, xyz);
      const double v_exact_tmp = channel.v_exact(dir, setup.flow_condition, setup.Reynolds, ns, xyz);
      my_errors[dir] = MAX(my_errors[dir], fabs(v_faces_ptr[f] - v_exact_tmp));
    }

    ierr = VecRestoreArrayRead(v_faces[dir], &v_faces_ptr); CHKERRXX(ierr);
  }

  // Compute the node errors

  const Vec* v_nodes; v_nodes = ns->get_velocity_np1();
  Vec v_exact_nodes[P4EST_DIM] = { DIM(NULL, NULL, NULL) };
  Vec error_nodes[P4EST_DIM]  = { DIM(NULL, NULL, NULL) };
  const double *v_nodes_p[P4EST_DIM];
  double *v_exact_nodes_p[P4EST_DIM], *error_nodes_p[P4EST_DIM];

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(v_nodes[dir], &v_nodes_p[dir]); CHKERRXX(ierr);
    if (setup.save_vtk)
    {
      ierr = VecCreateGhostNodes(ns->get_p4est(), ns->get_nodes(), &v_exact_nodes[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(ns->get_p4est(), ns->get_nodes(), &error_nodes[dir]); CHKERRXX(ierr);
      ierr = VecGetArray(error_nodes[dir], &error_nodes_p[dir]); CHKERRXX(ierr);
      ierr = VecGetArray(v_exact_nodes[dir], &v_exact_nodes_p[dir]); CHKERRXX(ierr);
    }
  }

  for (size_t n = 0; n < ns->get_nodes()->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM]; node_xyz_fr_n(n, ns->get_p4est(), ns->get_nodes(), xyz);
    double v_exact_tmp[P4EST_DIM];
    channel.v_exact(setup.flow_condition, setup.Reynolds, ns, xyz, v_exact_tmp);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      my_errors[P4EST_DIM + dir] = MAX(my_errors[P4EST_DIM + dir], fabs(v_nodes_p[dir][n] - v_exact_tmp[dir]));
      if (setup.save_vtk)
      {
        error_nodes_p[dir][n]    = fabs(v_nodes_p[dir][n] - v_exact_tmp[dir]);
        v_exact_nodes_p[dir][n]  = v_exact_tmp[dir];
      }
    }
  }

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(v_nodes[dir], &v_nodes_p[dir]); CHKERRXX(ierr);
    if (setup.save_vtk)
    {
      ierr = VecRestoreArray(error_nodes[dir], &error_nodes_p[dir]); CHKERRXX(ierr);
      ierr = VecRestoreArray(v_exact_nodes[dir], &v_exact_nodes_p[dir]); CHKERRXX(ierr);
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, my_errors, 2*P4EST_DIM, MPI_DOUBLE, MPI_MAX, ns->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);

  // Plot the errors and exact solution if the path is provided as an input
  if (setup.save_vtk)
  {
    if (create_directory(setup.vtk_path, ns->get_p4est()->mpirank, ns->get_p4est()->mpicomm))
      throw std::runtime_error("check_accuracy_of_solution: could not create exportation directory " + setup.vtk_path);

    const std::string vtk_name = setup.vtk_path + "/accuracy_check";
    const double *v_exact_vtk_p[P4EST_DIM], *error_vtk_p[P4EST_DIM];

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecGetArrayRead(error_nodes[dir], &error_vtk_p[dir]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(v_exact_nodes[dir], &v_exact_vtk_p[dir]); CHKERRXX(ierr);
    }

    my_p4est_vtk_write_all_general(ns->get_p4est(), ns->get_nodes(), ns->get_ghost(),
                                   P4EST_TRUE, P4EST_TRUE,
                                   0, 2, 0,
                                   0, 0, 0,
                                   vtk_name.c_str(),
                                   VTK_NODE_VECTOR_BY_COMPONENTS, "velocity", DIM(v_exact_vtk_p[0], v_exact_vtk_p[1], v_exact_vtk_p[2]),
        VTK_NODE_VECTOR_BY_COMPONENTS, "err_v", DIM(error_vtk_p[0], error_vtk_p[1], error_vtk_p[2]));

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecRestoreArrayRead(error_nodes[dir], &error_vtk_p[dir]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(v_exact_nodes[dir], &v_exact_vtk_p[dir]); CHKERRXX(ierr);
    }
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if (error_nodes[dir] != NULL){
      ierr = VecDestroy(error_nodes[dir]); CHKERRXX(ierr); }
    if (v_exact_nodes[dir] != NULL){
      ierr = VecDestroy(v_exact_nodes[dir]); CHKERRXX(ierr); }
  }

  ierr = PetscPrintf(ns->get_mpicomm(), "The face-error on u is %.6E\n", my_errors[0]); CHKERRXX(ierr);
  ierr = PetscPrintf(ns->get_mpicomm(), "The face-error on v is %.6E\n", my_errors[1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = PetscPrintf(ns->get_mpicomm(), "The face-error on w is %.6E\n", my_errors[2]); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(ns->get_mpicomm(), "The node-error on u is %.6E\n", my_errors[P4EST_DIM + 0]); CHKERRXX(ierr);
  ierr = PetscPrintf(ns->get_mpicomm(), "The node-error on v is %.6E\n", my_errors[P4EST_DIM + 1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = PetscPrintf(ns->get_mpicomm(), "The node-error on w is %.6E\n", my_errors[P4EST_DIM + 2]); CHKERRXX(ierr);
#endif
  setup.accuracy_check_done = true;
}

void initialize_exportations_and_monitoring(const my_p4est_navier_stokes_t* ns, const cmdParser &cmd, const my_p4est_shs_channel_t &channel, simulation_setup &setup,  velocity_profiler_t* &profiler)
{
  PetscErrorCode ierr;
  ierr = PetscPrintf(ns->get_mpicomm(), (std::string("Parameters : ") + (setup.flow_condition == constant_pressure_gradient ? std::string("Re_tau") : std::string("Re_b"))
                                         + std::string(" = %g, domain is %dx2") ONLY3D(+ std::string("x%d")) + std::string(" (delta units), P/delta = %g, GF = %g\n")).c_str(),
                     DIM(setup.Reynolds, (int) (channel.length()/channel.delta()), (int) (channel.width()/channel.delta())), channel.pitch_to_delta(), channel.GF()); CHKERRXX(ierr);

  ierr = PetscPrintf(ns->get_mpicomm(), "cfl = %g, wall layer = %u, rho = %g, mu = %g (1/mu = %g)\n", ns->get_cfl(), channel.ncells_layering_walls_from_ns_solver(ns->get_uniform_band()), ns->get_rho(), ns->get_mu(), 1.0/ns->get_mu());

  if (create_directory(setup.export_dir, ns->get_mpirank(), ns->get_mpicomm()))
    throw std::runtime_error("initialize_exportations_and_monitoring: could not create exportation directory " + setup.export_dir);
  // vtk exportation
  setup.vtk_path = setup.export_dir + "/vtu";
  if (setup.save_vtk && create_directory(setup.vtk_path, ns->get_mpirank(), ns->get_mpicomm()))
    throw std::runtime_error("initialize_exportations_and_monitoring: could not create exportation directory for vtk files " + setup.vtk_path);

  // drag exportation and simulation monitoring
  if (setup.save_drag)
    initialize_drag_force_output(setup, ns);
  if (setup.save_timing)
    initialize_timing_output(setup, ns);
  initialize_monitoring(setup, ns);

  // exportation of slice-averaged and line-averaged velocity profile(s)
  if (profiler != NULL)
    delete  profiler;
  profiler = NULL;

  if (setup.save_profiles)
    profiler = new velocity_profiler_t(cmd, ns, setup, channel);
}

bool monitor_simulation(const simulation_setup &setup, const mass_flow_controller_t* controller, my_p4est_navier_stokes_t* ns, external_force_per_unit_mass_t* external_acceleration[P4EST_DIM], const my_p4est_shs_channel_t &channel)
{
  if (ns->get_mpirank() == 0)
  {
    FILE* fp_monitor = fopen(setup.file_monitoring.c_str(), "a");
    if (fp_monitor == NULL)
      throw std::runtime_error("monitor_simulation: could not open monitoring file.");
    fprintf(fp_monitor, "%g %g %g\n", setup.tn,
            (external_acceleration[0]->get_value() > 0.0 ?  channel.canonical_Re_tau(external_acceleration[0]->get_value(), ns->get_nu()) : -1.0),
      channel.Re_b(controller->read_latest_mass_flow(), ns->get_rho(), ns->get_nu()));
    fclose(fp_monitor);
  }

  PetscErrorCode ierr;
  if (external_acceleration[0]->get_value() > 0.0){
    ierr = PetscPrintf(ns->get_mpicomm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d, \t Re_tau = %.2f, \t Re_b = %.2f\n",
                       setup.iter, setup.tn, 100*(setup.tn - setup.tstart)/setup.duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants,
                       channel.canonical_Re_tau(external_acceleration[0]->get_value(), ns->get_nu()),
                       channel.Re_b(controller->read_latest_mass_flow(), ns->get_rho(), ns->get_nu())); CHKERRXX(ierr); }
  else{
    ierr = PetscPrintf(ns->get_mpicomm(), "Iteration #%04d : driving bulk force is currently negative\n", setup.iter); CHKERRXX(ierr);
    ierr = PetscPrintf(ns->get_mpicomm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d, \t f_x = %.2f, \t Re_b = %.2f\n",
                       setup.iter, setup.tn, 100*(setup.tn - setup.tstart)/setup.duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants,
                       external_acceleration[0]->get_value(),
                       channel.Re_b(controller->read_latest_mass_flow(), ns->get_rho(), ns->get_nu())); CHKERRXX(ierr); }

  if (ns->get_max_L2_norm_u() > setup.max_tolerated_velocity(controller, external_acceleration, channel))
  {
    if (setup.save_vtk)
    {
      const std::string vtk_name = setup.vtk_path + "/snapshot_" + std::to_string(setup.export_vtk + 1);
      ns->save_vtk(vtk_name.c_str());
    }
    std::cerr << "The simulation blew up..." << std::endl;
    return true;
  }
  return false;
}

void export_drag(const simulation_setup &setup, const my_p4est_navier_stokes_t* ns, const my_p4est_shs_channel_t& channel, const mass_flow_controller_t* controller)
{
  double drag[P4EST_DIM];
  ns->get_noslip_wall_forces(drag);
  if (ns->get_mpirank() == 0)
  {
    const double U_b = channel.mean_u(controller->read_latest_mass_flow(), ns->get_rho());
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      drag[dir] /= (-2.0*SQR(U_b)*MULTD(ns->get_rho(), channel.length(), channel.width()));
    FILE* fp_drag = fopen(setup.file_drag.c_str(), "a");
    if (fp_drag == NULL)
      throw std::runtime_error("export_drag: could not open file for drag output.");
    fprintf(fp_drag, drag_output_format.c_str(), setup.tn, DIM(drag[0], drag[1], drag[2]));
    fclose(fp_drag);
  }
}

#ifdef P4EST_DEBUG
void check_voronoi_tesselation_and_print_warnings_if_wrong(const my_p4est_navier_stokes_t* ns, const my_p4est_poisson_faces_t* face_solver)
{
  double voro_global_volume[P4EST_DIM];
  face_solver->global_volume_of_voronoi_tesselation(voro_global_volume);
  // one should have EXACTLY the volume of the computational box for u and w components
  // and strictly less than the computational domain for v (because of face-wall alignment of Dirichlet boundary conditions --> the Voronoi cell is not even calculated there)
  const double expected_volume = MULTD(ns->get_length_of_domain(), ns->get_height_of_domain(), ns->get_width_of_domain());
  if(ns->get_mpirank() == 0 && fabs(voro_global_volume[0] - expected_volume) > 10.0*EPS*expected_volume)
    std::cerr << "The global volume of the Voronoi tesselation for faces of normal direction x is " << voro_global_volume[0] << " whereas it is expected to be " << expected_volume << " --> check the Voronoi tesselation!" << std::endl;
  if(ns->get_mpirank() == 0 && voro_global_volume[1] >= expected_volume)
    std::cerr << "The global volume of the Voronoi tesselation for faces of normal direction y is " << voro_global_volume[1] << " which is greater than the volume of the computational box (" << expected_volume << "): this is NOT NORMAL --> check the Voronoi tesselation!" << std::endl;
#ifdef P4_TO_P8
  if(ns->get_mpirank() == 0 && fabs(voro_global_volume[2] - expected_volume) > 10.0*EPS*expected_volume)
    std::cerr << "The global volume of the Voronoi tesselation for faces of normal direction z is " << voro_global_volume[2] << " whereas it is expected to be " << expected_volume << " --> check the Voronoi tesselation!" << std::endl;
#endif
}
#endif


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart",             "if present, this restarts the simulation from a saved state on disk \n\t(this must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin",                "min level of the trees, default is " + std::to_string(default_lmin) + " or value read from solver state if restarted.");
  cmd.add_option("lmax",                "max level of the trees, default is " + std::to_string(default_lmax) + " or value read from solver state if restarted.");
  cmd.add_option("thresh",              "the threshold used for the refinement criteria, default is " + std::to_string(default_thresh));
  cmd.add_option("wall_layer",          "number of finest cells desired to layer the channel walls, default is " + std::to_string(default_wall_layer) + " or value deduced from solver state if restarted.");
  cmd.add_option("lip",                 "Lipschitz constant L for grid refinement. The levelset is defined as the negative distance to the top/bottom wall in case of spanwise grooves or to the closest no-slip region with streamwise grooves. \n\tWarning: this application uses a modified criterion comparin the levelset value to L\\Delta y (as opposed to L*diag(C)). \n\tDefault value is " + std::to_string(default_lip) + (default_lip < 10.0 ? " (fyi: that's thin for turbulent cases)" : "") + " or value read from solver state if restarted.");
  cmd.add_option("nx",                  "number of trees in the x-direction. \n\tThe default value is " + std::to_string(default_ntree_y) + "*length/height  to ensure aspect ratio of cells = 1 when using default dimensions");
  cmd.add_option("ny",                  "number of trees in the y-direction. \n\tThe default value is " + std::to_string(default_ntree_y) + "                to ensure aspect ratio of cells = 1 when using default dimensions");
#ifdef P4_TO_P8
  cmd.add_option("nz",                  "number of trees in the z-direction. \n\tThe default value is " + std::to_string(default_ntree_y) + "*width/height   to ensure aspect ratio of cells = 1 when using default dimensions");
#endif
  // physical parameters for the simulations
  cmd.add_option("length",              "length of the channel (dimension in streamwise, x-direction), default is " + std::to_string(default_length));
  cmd.add_option("height",              "height of the channel (dimension in wall-normal, y-direction, ':= 2*delta'), default is " + std::to_string(default_height));
#ifdef P4_TO_P8
  cmd.add_option("width",               "width of the channel (dimension in spanwise, z-direction), default is " + std::to_string(default_width));
  cmd.add_option("spanwise",            "if present, the grooves and ridges are understood as spanwise, that is perpendicular to the flow. \n\tIf absent, the grooves are parallel to the flow.");
#endif
  cmd.add_option("duration",            "the duration of the simulation (tfinal - tstart). If not restarted, tstart = 0.0, default duration is " + std::to_string(default_duration));
  cmd.add_option("Re_tau",              "Reynolds number based on the wall-shear velocity (in canonical, standard channel of same size) and half the channel height, i.e. \n\t Re_tau = u_tau*delta/nu where u_tau = sqrt(f_x*delta), no default value.");
  cmd.add_option("Re_b",                "Reynolds number, based on the mean (bulk) velocity and half the channel height, i.e. \n\t Re_b = U_b*delta/nu, no default value.");
  cmd.add_option("initial_Re_tau",      "relevant ONLY in case of restart with constant mass flow rate (i.e. restart with Re_b defined by the user). \n\t If present (and defined), this input parameter is used to set the initial value of the bulk driving force f_x. \n\t If not defined, the value is calculated from the analytical laminar cases (which may be way off, expect discontinuities in the monitored Re_tau in such a case).");
  cmd.add_option("pitch",               "pitch, default = " + std::to_string(default_pitch_to_height) + "*height");
  cmd.add_option("GF",                  "gas fraction, default is " + std::to_string(default_gas_fraction));
  cmd.add_option("white_noise_rms",     "sets the intensity of white noise perturbation in initial velocity fields, relative to mean bulk velocity (useless when restarting); default value is " + std::to_string(def_white_noise_rms));
  // method-related parameters
  cmd.add_option("adapted_dt",          "activates the calculation of dt based on the local cell sizes if present");
  cmd.add_option("sl_order",            "the order for the semi-lagrangian method, either 1 (stable) or 2 (accurate), default is " + std::to_string(default_sl_order) + " or value read from solver state if restarted.");
  cmd.add_option("cfl",                 "dt = cfl * dx/vmax, default is " + std::to_string(default_cfl) + " or value read from solver state if restarted.");
  cmd.add_option("u_tol",               "relative numerical tolerance on the bulk velocity, default is " + std::to_string(default_u_tol));
  cmd.add_option("niter_hodge",         "max number of iterations for convergence of the projection step, at all time steps, default is " + std::to_string(default_n_hodge));
  cmd.add_option("hodge_control",       "type of convergence check used for inner loops, i.e. which velocity component. Possible values are 'u', 'v' (, 'w') or 'uvw', default is " + convert_to_string(def_hodge_control));
  cmd.add_option("grid_update",         "number of time steps between grid updates, default is " + std::to_string(default_grid_update));
  cmd.add_option("pc_cell",             "preconditioner for cell-solver: jacobi, sor or hypre, default is " + default_pc_cell);
  cmd.add_option("cell_solver",         "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is " + default_cell_solver);
  cmd.add_option("pc_face",             "preconditioner for face-solver: jacobi, sor or hypre, default is " + default_pc_face);
  cmd.add_option("nterms",              "number of terms used in the truncated series of the analytical solution (used for initialization and accuracy check is desired). Default is " + std::to_string(default_nterms));
  // output-control parameters
  cmd.add_option("export_folder",       "exportation folder (monitoring and drag files in there, velocity profiles, vtk and backup files in subfolder), will be created if inexistent;\n\t default is " + default_export_dir);
  cmd.add_option("save_vtk",            "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt",              "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
  cmd.add_option("save_drag",           "activates exportation of the total drag, non-dimensionalized by 2.0*rho*U_b^2*length (*width) (--> estimate of (Re_tau/Re_b)^2 at steady state)");
  cmd.add_option("save_state_dt",       "if present, this activates the 'save-state' feature. \n\tThe solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates",        "determines how many solver states must be memorized in backup_ folders, default is " + std::to_string(default_save_nstates));
  cmd.add_option("save_mean_profiles",  "computes and saves averaged streamwise-velocity profiles (makes sense only if the flow is fully-developed)");
  cmd.add_option("nexport_avg",         "number of iterations between two exportation of averaged velocity profiles, default is " + std::to_string(default_nexport_avg));
  cmd.add_option("timing",              "if defined, saves timing information (info for every major N-S task) in a file on disk.");
  cmd.add_option("accuracy_check",      "if present, prints information about accuracy with comparison to analytical solution \n\t(ONLY activated if restarted, supposedly after steady-state reached). \n\tIf save_vtk is activated as well, the errors are exported to the vtk path for visualization in space.");

  if (cmd.parse(argc, argv, extra_info))
    return 0;

  PetscErrorCode ierr;

  simulation_setup setup(mpi, cmd);

  my_p4est_navier_stokes_t* ns                    = NULL;
  my_p4est_brick_t* brick                         = NULL;
  splitting_criteria_cf_and_uniform_band_t* data  = NULL;
  mass_flow_controller_t *flow_controller         = NULL;
  external_force_per_unit_mass_t* external_acceleration[P4EST_DIM] = { DIM(NULL, NULL, NULL) };

  my_p4est_shs_channel_t channel(mpi);

  // create the solver, either loaded from saved state or built from scratch
  if (cmd.contains("restart"))
    load_solver_from_state(mpi, cmd, ns, brick, channel, external_acceleration, data, flow_controller, setup);
  else
    create_solver_from_scratch(mpi, cmd, ns, brick, channel, external_acceleration, data, flow_controller, setup);

  velocity_profiler_t *profiler = NULL;
  initialize_exportations_and_monitoring(ns, cmd, channel, setup, profiler);

  if(setup.save_timing)
    ns->activate_timer();

  setup.tn = setup.tstart;
  setup.update_save_data_idx(); // so that we don't save the very first one which was either already read from file, or the known initial condition...

  my_p4est_poisson_cells_t* cell_solver = NULL;
  my_p4est_poisson_faces_t* face_solver = NULL;
  Vec dxyz_hodge_old[P4EST_DIM];

  while (!setup.done())
  {
    if (setup.iter > 0)
    {
      if (flow_controller->read_latest_mass_flow() <= 0.0)
        std::runtime_error("main_shs_" + std::to_string(P4EST_DIM) + "d: something went wrong, the mass flow should be strictly positive and known to at this stage...");

      bool solvers_can_be_reused = setup.set_dt_and_update_grid(ns);
      if (cell_solver != NULL && !solvers_can_be_reused){
        delete cell_solver; cell_solver = NULL; }
      if (face_solver != NULL && !solvers_can_be_reused){
        delete face_solver; face_solver = NULL; }
    }

    if (setup.time_to_save_state())
    {
      setup.update_save_data_idx();
      ns->save_state(setup.export_dir.c_str(), setup.tn, setup.n_states);
    }

    double convergence_check_on_dxyz_hodge = DBL_MAX;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateNoGhostFaces(ns->get_p4est(), ns->get_faces(), &dxyz_hodge_old[dir], dir); CHKERRXX(ierr); }
    unsigned int iter_hodge = 0;
    while (iter_hodge < setup.niter_hodge_max && (convergence_check_on_dxyz_hodge > setup.u_tol*channel.mean_u(flow_controller->read_latest_mass_flow(), ns->get_rho())))
    {
      ns->copy_dxyz_hodge(dxyz_hodge_old);

      ns->solve_viscosity(face_solver, (face_solver != NULL), KSPBCGS, setup.pc_face);
#ifdef P4EST_DEBUG
      check_voronoi_tesselation_and_print_warnings_if_wrong(ns, face_solver);
#endif

      convergence_check_on_dxyz_hodge = ns->solve_projection(cell_solver, (cell_solver != NULL), setup.cell_solver_type, setup.pc_cell, false, NULL, dxyz_hodge_old, setup.control_hodge);

      flow_controller->evaluate_current_mass_flow(ns);
      if (setup.flow_condition == constant_mass_flow)
      {
        const double required_mean_correction_to_hodge_derivative = flow_controller->update_forcing_and_get_mean_streamwise_velocity_correction(ns, external_acceleration);
        if(setup.control_hodge == uvw_components || setup.control_hodge == u_component)
          convergence_check_on_dxyz_hodge = convergence_check_on_dxyz_hodge + fabs(required_mean_correction_to_hodge_derivative); // Yes, Sir, don't cut corners: you may have "converged" when comparing to data from previous time-step only but if the required correction is too big, you may have to do it again...
      }

      if(setup.control_hodge == uvw_components){
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in \\nabla Hodge = %e\n", iter_hodge, convergence_check_on_dxyz_hodge); CHKERRXX(ierr);
      } else {
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in d(Hodge)/d%s = %e\n", iter_hodge, (setup.control_hodge == u_component ? 'x' : (setup.control_hodge == v_component ? 'y' : 'z')), convergence_check_on_dxyz_hodge); CHKERRXX(ierr);
      }
      iter_hodge++;
    }
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecDestroy(dxyz_hodge_old[dir]); CHKERRXX(ierr); }

    ns->compute_velocity_at_nodes(setup.steps_grid_update > 1);

    ns->compute_pressure();

    setup.tn += setup.dt;

    // monitoring Re_tau and Re_b
    if (monitor_simulation(setup, flow_controller, ns, external_acceleration, channel))
      break;

    // DEBUG check that we correctly enforce de mass flow if that's what we want
    P4EST_ASSERT(setup.flow_condition != constant_mass_flow ||
        iter_hodge == setup.niter_hodge_max ||
        fabs(channel.Re_b(flow_controller->read_latest_mass_flow(), ns->get_rho(), ns->get_nu()) - flow_controller->targeted_bulk_velocity()*channel.delta()/ns->get_nu()) < setup.u_tol*flow_controller->targeted_bulk_velocity()*channel.delta()/ns->get_nu());

    // exporting drag if desired
    if (setup.save_drag)
      export_drag(setup, ns, channel, flow_controller);

    // exporting drag if desired
    if (setup.save_timing)
      setup.export_and_accumulate_timings(ns);

    // exporting velocity profiles if desired
    if (setup.save_profiles)
      profiler->gather_and_dump_profiles(setup, ns, (setup.flow_condition == constant_pressure_gradient ? channel.canonical_u_tau(external_acceleration[0]->get_value()) : flow_controller->targeted_bulk_velocity()) ONLY3D(COMMA channel.spanwise_grooves()));

    if (setup.time_to_save_vtk() && !setup.do_accuracy_check)
    {
      setup.update_export_vtk();
      const std::string vtk_name = setup.vtk_path + "/snapshot_" + std::to_string(setup.export_vtk);
      ns->save_vtk(vtk_name.c_str(), true, channel.mean_u(flow_controller->read_latest_mass_flow(), ns->get_rho()), channel.height()*0.5);
    }

    if (setup.do_accuracy_check)
      check_accuracy_of_solution(ns, channel, setup);

    setup.iter++;
  }


  if(setup.save_timing)
    setup.print_averaged_timings(mpi);

  if (cell_solver != NULL)
    delete  cell_solver;
  if (face_solver != NULL)
    delete face_solver;

  delete ns;        // deletes the navier-stokes solver
  // the brick and the connectivity are deleted within the above destructor...
  delete data;      // deletes the splitting criterion object
  if (flow_controller != NULL)
    delete flow_controller;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    if(external_acceleration[dir] != NULL)
      delete external_acceleration[dir];

  return 0;
}
