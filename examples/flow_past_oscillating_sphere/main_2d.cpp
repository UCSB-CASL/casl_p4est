/*
 * The navier stokes solver applied to the oscillating sphere problem
 *
 * run the program with the -help flag to see the available options
 */

// my_p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/Parser.h>
#include <iomanip>

#undef MIN
#undef MAX

using namespace std;

// --> extra info to be printed when -help is invoked
const std::string extra_info =
      string("This program provides a general setup for simulating the flow past an oscillating sphere in 3D, \n")
    + string("resp. cylinder in 2D, in a closed box. The domain is ([-0.5*side_length, 0.5*side_length])^{2 or 3). \n")
    + string("The radius of the sphere (resp. cylinder) is r and its center follows the kinematics \n")
    + string("\t\t x(t) = -X0*cos(2.0*M_PI*f0*t), y(t) = 0.0, z(t) = 0.0 \n")
    + string("(Note that one must have 2*(fabs(X0) + r) < side_length)\n")
    + string("The boundary conditions are:\n")
    + string("- Dirichlet (0, 0 (, 0)) for the velocity components on all walls (no slip); \n")
    + string("- no-slip condition, i.e. vx(t) = 2.0*M_PI*X0*f0*sin(2.0*M_PI*f0*t), vy(t) = 0.0, vz(t) = 0.0 on the solid's surface; \n")
    + string("- homogeneous Neumann for the pressure/hodge variable on all walls; \n")
    + string("- homogeneous Neumann for the pressure/hodge variable on the solid's surface. \n")
    + string("The flow starts from rest. This setup determines two non-dimensional numbers:\n")
    + string("1) Strouhal = (2*r*f0)/(2.0*M_PI*X0*f0) = r/(M_PI*X0);\n")
    + string("2) Reynolds = rho*(2.0*M_PI*X0*f0)*2.0*r)/mu = (rho*4.0*f0*r^2)/(St*mu);\n")
    + string("By the definition of the above parameters and nondimensional numbers, we set \n")
    + string("\t\t\t X0 = r/(M_PI*St), \n")
    + string("and \n")
    + string("\t\t\t mu = (4*rho*f0*r^2)/(St*Re).\n")
    + string("The user is free to specify St and Re and the solver's parameters will be set correspondingly, in a way that\n")
    + string("rho = 1.0 and f0 is 1.0 (reference time is one period of oscillation).")
    + string("Developer: cleaned up and restructured to new code features' and coding \"standards\" by Raphael Egan\n")
    + string("(raphaelegan@ucsb.edu) based on a general main file from Arthur Guittet (with corrections for solid's\n")
    + string("no-slip boundary conditions)");

#if defined(POD_CLUSTER)
const std::string default_export_dir  = "/scratch/regan/flow_past_oscillating_" + string(P4EST_DIM == 3 ? "sphere" : "cylinder");
#elif defined(STAMPEDE)
const std::string default_export_dir  = "/scratch/04965/tg842642/flow_past_oscillating_" + string(P4EST_DIM == 3 ? "sphere" : "cylinder");
#elif defined(LAPTOP)
const std::string default_export_dir  = "/home/raphael/workspace/projects/flow_past_oscillating_" + string(P4EST_DIM == 3 ? "sphere" : "cylinder");
#else
const std::string default_export_dir  = "/home/regan/workspace/projects/flow_past_oscillating_" + string(P4EST_DIM == 3 ? "sphere" : "cylinder");
#endif

const int default_lmin                        = 5;
const int default_lmax                        = 10;
const double default_thresh                   = 0.1;
const double default_uniform_band             = 4.0;
const int default_ntree                       = 1;
const double default_side_length              = 2.0;
const double default_radius                   = 0.1;
const double default_St                       = 4.0/M_PI;
const double default_Re                       = 300.0;

const double default_duration_to_period       = 3.0;
const double default_ntimesteps_per_period    = 200;
const int default_sl_order                    = 2;
const double default_hodge_tol                = 1.0e-3;
const unsigned int default_n_hodge            = 10;
const hodge_control def_hodge_control         = uvw_components;
const std::string default_pc_cell             = "sor";
const std::string default_cell_solver         = "bicgstab";
const std::string default_pc_face             = "sor";
const unsigned int default_save_nstates       = 1;


class LEVEL_SET: public CF_DIM
{
public:
  const double radius, X0, f0;
  LEVEL_SET(const double &radius_, const double &f0_, const double &Strouhal)
    : radius(radius_), X0(radius_/(M_PI*Strouhal)), f0(f0_)
  {
    lip = 1.2;
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    return radius - sqrt(SUMD(SQR(x + X0*cos(2.0*M_PI*f0*t)), SQR(y), SQR(z)));
  }
  double Strouhal() const { return radius/(M_PI*X0);                  }
  double vx_max() const   { return 2.0*M_PI*X0*f0;                    }
  double vx() const       { return 2.0*M_PI*X0*f0*sin(2.0*M_PI*f0*t); }
  void update_time(const double &tnp1) { t = tnp1; }
};

double Reynolds(const LEVEL_SET &ls, const my_p4est_navier_stokes_t *ns)
{
  return ls.vx_max()*(2.0*ls.radius)/ns->get_nu();
}

struct BCWALLTYPE_PRESSURE : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLTYPE_VELOCITY : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_velocity;

struct BCINTERFACE_VALUE_VELOCITY : CF_DIM
{
  const unsigned char dir;
  const LEVEL_SET *ls;
  BCINTERFACE_VALUE_VELOCITY(const unsigned char &dir_, const LEVEL_SET *ls_) : dir(dir_), ls(ls_) {}
  double operator()(DIM(double, double, double)) const
  {
    if(dir != dir::x)
      return 0.0;
    return ls->vx(); // no longer u0 as in the original main
  }
};

struct simulation_setup
{
  double tstart;
  double tn;

  // inner convergence parameteres
  const double hodge_tol;
  const unsigned int niter_hodge_max;
  const hodge_control control_hodge;

  // simulation control
  int iter;
  const double duration_to_period;
  const double dt_to_period;
  const std::string des_pc_cell;
  const std::string des_solver_cell;
  const std::string des_pc_face;
  KSPType cell_solver_type;
  PCType pc_cell, pc_face;
  double Reynolds;
  LEVEL_SET *ls;

  // exportation
  const std::string export_dir_root;
  const bool save_vtk;
  double vtk_dt_to_period;
  const bool save_forces;
  const bool save_timing;
  const bool save_hodge_convergence;
  const bool save_state;
  double dt_save_data_to_period;
  const unsigned int n_states;
  int export_vtk, save_data_idx;
  std::string export_dir, vtk_path, file_forces, file_timings, file_hodge_convergence;
  std::map<ns_task, double> global_computational_times;
  std::vector<double> hodge_convergence_checks;

  simulation_setup(const mpi_environment_t&mpi, const cmdParser &cmd) :
    hodge_tol(cmd.get<double>("hodge_tol", default_hodge_tol)),
    niter_hodge_max(cmd.get<unsigned int>("niter_hodge", default_n_hodge)),
    control_hodge(cmd.get<hodge_control>("hodge_control", def_hodge_control)),
    duration_to_period(cmd.get<double>("duration", default_duration_to_period)),
    dt_to_period(1.0/cmd.get<double>("nsteps", default_ntimesteps_per_period)),
    des_pc_cell(cmd.get<std::string>("pc_cell", default_pc_cell)),
    des_solver_cell(cmd.get<std::string>("cell_solver", default_cell_solver)),
    des_pc_face(cmd.get<std::string>("pc_face", default_pc_face)),
    Reynolds(cmd.get<double>("Re", default_Re)),
    export_dir_root(cmd.get<std::string>("export_folder", (getenv("OUT_DIR") == NULL ? default_export_dir : getenv("OUT_DIR")))),
    save_vtk(cmd.contains("save_vtk")),
    save_forces(cmd.contains("save_forces")),
    save_timing(cmd.contains("timing")),
    save_hodge_convergence(cmd.contains("track_subloop")),
    save_state(cmd.contains("save_state_dt")),
    n_states(cmd.get<unsigned int>("save_nstates", default_save_nstates))
  {
    vtk_dt_to_period = -1.0;
    if (save_vtk)
    {
      if (!cmd.contains("vtk_dt"))
        throw std::runtime_error("simulation_setup::simulation_setup(): the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
      vtk_dt_to_period = cmd.get<double>("vtk_dt", -1.0);
      if (vtk_dt_to_period <= 0.0)
        throw std::invalid_argument("simulation_setup::simulation_setup(): the value of vtk_dt must be strictly positive.");
    }
    dt_save_data_to_period = -1.0;
    if (save_state)
    {
      dt_save_data_to_period = cmd.get<double>("save_state_dt");
      if (dt_save_data_to_period < 0.0)
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

    ls = new LEVEL_SET(cmd.get<double>("radius", default_radius), 1.0, cmd.get<double>("St", default_St));
  }

  ~simulation_setup() { delete  ls; }

  int running_save_data_idx() const { return (int) floor(tn*ls->f0/dt_save_data_to_period); }
  void update_save_data_idx()       { save_data_idx = running_save_data_idx(); }
  bool time_to_save_state() const   { return (save_state && running_save_data_idx() != save_data_idx); }

  int running_export_vtk() const  { return (int) floor(tn*ls->f0/vtk_dt_to_period); }
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
    ns->set_dt(dt_to_period/ls->f0);

    if(tn + dt_to_period/ls->f0 > tstart + duration_to_period/ls->f0)
      ns->set_dt(tstart + duration_to_period/ls->f0 - tn);

    ls->update_time(tn + dt_to_period/ls->f0);

    return ns->update_from_tn_to_tnp1(ls, false, false);
  }

  bool done() const
  {
    return tn + 0.01*dt_to_period/ls->f0 > tstart + duration_to_period/ls->f0;
  }

  void export_forces(my_p4est_navier_stokes_t* ns)
  {
    double forces[P4EST_DIM];
    ns->compute_forces(forces);
    if(!ns->get_mpirank())
    {
      FILE* fp_forces = fopen(file_forces.c_str(), "a");
      if(fp_forces == NULL)
        throw std::invalid_argument("simulation_setup: could not open file for forces output.");
#ifdef P4_TO_P8
      const double scaling = .5*M_PI*SQR(ls->radius)*SQR(ls->vx_max())*ns->get_rho();
      fprintf(fp_forces, "%g %g %g %g\n", ls->f0*tn, forces[0]/scaling, forces[1]/scaling, forces[2]/scaling);
#else
      const double scaling = ls->radius*SQR(ls->vx_max())*ns->get_rho();
      fprintf(fp_forces, "%g %g %g\n", ls->f0*tn, forces[0]/scaling, forces[1]/scaling);
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
    ns->save_vtk((vtk_path + "/snapshot_" + to_string(export_vtk)).c_str(), true, ls->vx_max(), ls->radius);
  }
};

void initialize_force_output(simulation_setup & setup, const my_p4est_navier_stokes_t *ns)
{
  ostringstream filename;
  filename << std::fixed << std::setprecision(2);
  filename << "forces_" << ns->get_lmin() << "-" << ns->get_lmax() << "_split_threshold_" << ns->get_split_threshold()
           << "_nsteps_" << 1.0/setup.dt_to_period << "_sl_" << ns->get_sl_order() << ".dat";
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
      fprintf(fp_liveplot_hodge, "plot \t \"%s\" using 1:%u title 'subiteration %u' with lines lw 3", setup.file_hodge_convergence.c_str(), 2 + 0, 0);
      for (unsigned int k = 1; k < setup.niter_hodge_max; ++k)
        fprintf(fp_liveplot_hodge, ", \\\n\t \"%s\" using 1:%u title 'subiteration %u' with lines lw 3", setup.file_hodge_convergence.c_str(), 2 + k, k);
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

void load_solver_from_state(const mpi_environment_t &mpi, const cmdParser &cmd, BoundaryConditionsDIM bc_v[P4EST_DIM], BoundaryConditionsDIM &bc_p,
                            my_p4est_navier_stokes_t* &ns, my_p4est_brick_t* &brick, splitting_criteria_cf_and_uniform_band_t* &data, simulation_setup &setup)
{
  const string backup_directory = cmd.get<string>("restart", "");
  if(!is_folder(backup_directory.c_str()))
    throw std::invalid_argument("load_solver_from_state: the restart path " + backup_directory + " is not an accessible directory.");
  if (cmd.contains("ntree") || cmd.contains("side_length"))
    throw std::invalid_argument("load_solver_from_state: the side length as well as the numbers of trees in the macromesh cannot be reset when restarting a simulation.");

  if (ns != NULL)
  {
    delete ns; ns = NULL;
  }
  P4EST_ASSERT(ns == NULL);
  ns                      = new my_p4est_navier_stokes_t(mpi, backup_directory.c_str(), setup.tstart);

  setup.ls->update_time(setup.tstart + setup.dt_to_period/setup.ls->f0);
  ns->set_dt(setup.dt_to_period/setup.ls->f0);

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
  brick                   = ns->get_brick();


  if (data != NULL) {
    delete data; data = NULL;
  }
  P4EST_ASSERT(data == NULL);

  data = new splitting_criteria_cf_and_uniform_band_t(cmd.get<int>("lmin", ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl),
                                                      cmd.get<int>("lmax", ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl),
                                                      setup.ls,
                                                      cmd.get<int>("uniform_band", ns->get_uniform_band()),
                                                      ((splitting_criteria_t*) p4est_n->user_pointer)->lip);

  splitting_criteria_t* to_delete = (splitting_criteria_t*) p4est_n->user_pointer;
  bool fix_restarted_grid = (data->max_lvl != to_delete->max_lvl);
  delete to_delete;
  p4est_n->user_pointer   = (void*) data;
  p4est_nm1->user_pointer = (void*) data; // p4est_n and p4est_nm1 always point to the same splitting_criteria_t no need to delete the nm1 one, it's just been done


  if(cmd.contains("Re"))
  {
    if(fabs(cmd.get<double>("Re") - Reynolds(*setup.ls, ns)) > 1e-6*MAX(cmd.get<double>("Re"), Reynolds(*setup.ls, ns)))
      throw std::invalid_argument("load_solver_from_state: the Reynolds number cannot be reset when restarting a simulation.");
  }
  else
    setup.Reynolds = Reynolds(*setup.ls, ns);

  ns->set_parameters(ns->get_mu(), ns->get_rho(), cmd.get<int>("sl_order", ns->get_sl_order()), data->uniform_band,
                     cmd.get<double>("thresh", ns->get_split_threshold()), +1e7); // cfl is irrelevant in this case, we set all time steps externally

  // no smoke in this example
  ns->set_smoke(NULL, NULL, false, 1.0);

  ns->set_bc(bc_v, &bc_p);
  if(fix_restarted_grid)
    ns->refine_coarsen_grid_after_restart(setup.ls, false);

  if(setup.save_vtk)
    setup.update_export_vtk(); // so that we don't overwrite visualization files that were possibly already exported...

  PetscErrorCode ierr = PetscPrintf(ns->get_mpicomm(), "Simulation restarted from state saved in %s\n", (cmd.get<std::string>("restart")).c_str()); CHKERRXX(ierr);
}

void create_solver_from_scratch(const mpi_environment_t &mpi, const cmdParser &cmd, BoundaryConditionsDIM bc_v[P4EST_DIM], BoundaryConditionsDIM &bc_p,
                                my_p4est_navier_stokes_t* &ns, my_p4est_brick_t* &brick, splitting_criteria_cf_and_uniform_band_t* &data, simulation_setup &setup)
{
  // build the macromesh first
  const double side_length = cmd.get<double>("side_length", default_side_length);
  if(side_length < 2.0*(setup.ls->radius + setup.ls->X0))
    throw std::invalid_argument("create_solver_from_scratch: the domain is not big enough for your simulation parameters.");
  const double xyz_min_[P4EST_DIM] = {DIM(-0.5*side_length, -0.5*side_length, -0.5*side_length)};
  const double xyz_max_[P4EST_DIM] = {DIM(+0.5*side_length, +0.5*side_length, +0.5*side_length)};
  const int n_tree = cmd.get<int>("ntree", default_ntree);
  const int n_tree_xyz[P4EST_DIM] = {DIM(n_tree, n_tree, n_tree)};
  p4est_connectivity_t *connectivity;
  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;
  const int periodic[P4EST_DIM] = {DIM(0, 0, 0)};
  connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min_, xyz_max_, brick, periodic);


  setup.tstart = 0.0; // no restart, we start from 0.0
  setup.ls->update_time(setup.tstart + setup.dt_to_period/setup.ls->f0);

  if(data != NULL)
  {
    delete data; data = NULL;
  }
  P4EST_ASSERT(data == NULL);
  data  = new splitting_criteria_cf_and_uniform_band_t(cmd.get<int>("lmin", default_lmin),
                                                       cmd.get<int>("lmax", default_lmax),
                                                       setup.ls,
                                                       cmd.get<double>("uniform_band", default_uniform_band));


  p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(int l = 0; l < data->max_lvl; ++l)
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

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c);

  Vec phi;
  PetscErrorCode ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *setup.ls, phi);

  CF_DIM *vnm1[P4EST_DIM] = {DIM(&zero_cf, &zero_cf, &zero_cf)};
  CF_DIM *vn  [P4EST_DIM] = {DIM(&zero_cf, &zero_cf, &zero_cf)};

  ns = new my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n);
  ns->set_phi(phi);

  const double ns_rho = 1.0;
  ns->set_parameters(ns_rho*setup.ls->vx_max()*2.0*setup.ls->radius/setup.Reynolds,
                     ns_rho,
                     cmd.get<int>("sl_order", default_sl_order),
                     cmd.get<double>("uniform_band", default_uniform_band),
                     cmd.get<double>("thresh", default_thresh),
                     +1e7); // cfl is irrelevant in this case, we set all time steps externally
  ns->set_dt(setup.dt_to_period/setup.ls->f0, setup.dt_to_period/setup.ls->f0);
  ns->set_velocities(vnm1, vn);

  ns->set_bc(bc_v, &bc_p);
}

void initialize_exportations_and_monitoring(const my_p4est_navier_stokes_t* ns, simulation_setup &setup)
{
  PetscErrorCode ierr;
  ostringstream oss;
  oss << std::scientific << std::setprecision(2);
  oss << "Parameters : St = " << setup.ls->Strouhal() << ", Re = " << Reynolds(*setup.ls, ns) << ", f0 = " << setup.ls->f0
      << ", mu = " << ns->get_mu() << ", rho = " << ns->get_rho() << ", macromesh is " << ns->get_brick()->nxyztrees[0] << " X " << ns->get_brick()->nxyztrees[1] ONLY3D( << " X " << ns->get_brick()->nxyztrees[2]) << "\n";
  ierr = PetscPrintf(ns->get_mpicomm(), oss.str().c_str()); CHKERRXX(ierr);
  ierr = PetscPrintf(ns->get_mpicomm(), "dt = T/%g, uniform_band = %g\n", 1.0/setup.dt_to_period, ns->get_uniform_band());

  ostringstream subfolder;
  subfolder<< std::fixed << std::setprecision(2);
  subfolder << "Re_" << Reynolds(*setup.ls, ns) <<  "/lmin_" << ns->get_lmin() << "_lmax_" << ns->get_lmax();
  setup.export_dir = setup.export_dir_root + "/" + subfolder.str();
  setup.vtk_path = setup.export_dir + "/vtu";
  if(setup.save_vtk || setup.save_forces || setup.save_state)
  {
    if(create_directory(setup.export_dir, ns->get_mpirank(), ns->get_mpicomm()))
      throw std::runtime_error("main_flow_past_oscillating_sphere: could not create exportation directory " + setup.export_dir);

    if(setup.save_vtk && create_directory(setup.vtk_path, ns->get_mpirank(), ns->get_mpicomm()))
      throw std::runtime_error("main_flow_past_oscillating_sphere: could not create exportation directory for vtk files " + setup.vtk_path);
  }

  if(setup.save_forces)
    initialize_force_output(setup, ns);
  if(setup.save_timing)
    initialize_timing_output(setup, ns);
  if(setup.save_hodge_convergence)
    initialize_hodge_convergence_output(setup, ns);
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", string("if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved\n")
                 + string("\t IMPORTANT NOTE : the Strouhal number MUST be the same as originally used when restarting (the program is incapable of checking consistency w.r.t. Strouhal)."));
  // computational grid parameters
  cmd.add_option("lmin",          "min level of the trees, default is " + to_string(default_lmin));
  cmd.add_option("lmax",          "max level of the trees, default is " + to_string(default_lmax));
  cmd.add_option("thresh",        "the threshold used for the refinement criteria, default is " + to_string(default_thresh));
  cmd.add_option("uniform_band",  "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is " + to_string(100*default_uniform_band) + " or value read from restart");
  cmd.add_option("ntree",         "number of trees along every Cartesian direction. The default value is " + to_string(default_ntree));
  // physical parameters for the simulations
  cmd.add_option("duration",      "the duration of the simulation (tfinal - tstart), in units of oscillating periods. If not restarted, tstart = 0.0, default duration is " + to_string(default_duration_to_period) + " cycle(s).");
  cmd.add_option("St",            "the Strouhal number = r/(M_PI*X0). This option sets the amplitude of the motion as X0 = r/(PI*St). \n\t Too low values would lead to large excursion out of the domain and will not be accepted. Default value is " + to_string(default_St));
  cmd.add_option("Re",            "the Reynolds number = (2.0*M_PI*X0*f0)*2.0*r/nu, default value is " + to_string(default_Re));
  cmd.add_option("side_length",   "the side length of the box (computational domain), default value is " + to_string(default_side_length));
  cmd.add_option("radius",        "the radius of the sphere/cylinder, default value is " + to_string(default_radius));
  // method-related parameters
  cmd.add_option("sl_order",      "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is " + to_string(default_sl_order) + " or value read from restart");
  cmd.add_option("nsteps",        "number of time steps per oscillation period, default is " + to_string(default_ntimesteps_per_period));
  cmd.add_option("hodge_tol",     string("relative numerical tolerance used for the convergence criterion on the Hodge variable (or its gradient).\n")
                 + string("\tIf converging with respect to the value of the hodge variable, relative to 0.5*rho*SQR(2.0*M_PI*X0*f0)\n")
                 + string("\tIf converging with respect to the gradient of the hodge variable, relative to 2.0*M_PI*X0*f0\n")
                 + string("\tDefault value is ") + to_string(default_hodge_tol));
  cmd.add_option("niter_hodge",           "max number of iterations for convergence of the Hodge variable, at all time steps, default is " + to_string(default_n_hodge));
  cmd.add_option("hodge_control",         "type of convergence check used for inner loops, i.e. convergence criterion on the Hodge variable. \n\
                 Possible values are 'u', 'v' (, 'w'), 'uvw' (for gradient components) or 'value' (for local values of Hodge), default is " + convert_to_string(def_hodge_control));
  cmd.add_option("pc_cell",               "preconditioner for cell-solver: jacobi, sor or hypre, default is " + default_pc_cell);
  cmd.add_option("cell_solver",           "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is " + default_cell_solver);
  cmd.add_option("pc_face",               "preconditioner for face-solver: jacobi, sor or hypre, default is " + default_pc_face);
  // output-control parameters
  cmd.add_option("export_folder",         "exportation_folder if not defined otherwise in the environment variable OUT_DIR,\n\
                 subfolders will be created, default is " + default_export_dir);
  cmd.add_option("save_vtk",              "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt",                "export vtk files every vtk_dt (relative to oscillation cycle) time lapse (REQUIRED if save_vtk is activated)");
  cmd.add_option("save_forces",           "if defined, saves the forces onto the sphere/cylinder");
  cmd.add_option("save_state_dt",         "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates",          "determines how many solver states must be memorized in backup_ folders (default is " +to_string(default_save_nstates));
  cmd.add_option("timing",                "if defined, saves timing information in a file on disk (typically for scaling analysis).");
  cmd.add_option("track_subloop",         "if defined, saves the data corresponding to the inner loops for convergence of the hodge variable (saved in a file on disk).");

  if(cmd.parse(argc, argv, extra_info))
    return 0;

  PetscErrorCode ierr;

  simulation_setup setup(mpi, cmd);

  my_p4est_navier_stokes_t* ns                    = NULL;
  my_p4est_brick_t* brick                         = NULL;
  splitting_criteria_cf_and_uniform_band_t* data  = NULL;

  BCINTERFACE_VALUE_VELOCITY *bc_v_interface[P4EST_DIM];
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    bc_v[dim].setWallTypes(bc_wall_type_velocity);
    bc_v[dim].setWallValues(zero_cf);
    bc_v[dim].setInterfaceType(DIRICHLET);
    bc_v_interface[dim] = new BCINTERFACE_VALUE_VELOCITY(dim, setup.ls);
    bc_v[dim].setInterfaceValue(*bc_v_interface[dim]);
  }
  bc_p.setWallTypes(bc_wall_type_p);
  bc_p.setWallValues(zero_cf); // homogeneous, whether Neumann or Dicihlet...
  bc_p.setInterfaceType(NEUMANN);
  bc_p.setInterfaceValue(zero_cf);

  // create the solver, either loaded from saved state or built from scratch
  if (cmd.contains("restart"))
    load_solver_from_state(mpi, cmd, bc_v, bc_p, ns, brick, data, setup);
  else
    create_solver_from_scratch(mpi, cmd, bc_v, bc_p, ns, brick, data, setup);

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

  while(!setup.done())
  {
    if(setup.iter > 0)
    {
      bool solvers_can_be_reused = setup.set_dt_and_update_grid(ns);
      if(cell_solver!=NULL && (!solvers_can_be_reused)){
        delete  cell_solver; cell_solver = NULL; }
      if(face_solver!=NULL && (!solvers_can_be_reused)){
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

    while(iter_hodge < setup.niter_hodge_max && convergence_check_on_hodge > setup.hodge_tol*(setup.control_hodge ? 0.5*SQR(setup.ls->vx_max())*ns->get_dt() : setup.ls->vx_max()))
    {
      if(setup.control_hodge == hodge_value)
        ns->copy_hodge(hold_old, false);
      else
        ns->copy_dxyz_hodge(dxyz_hodge_old);

      ns->solve_viscosity(face_solver, (face_solver != NULL), KSPBCGS, setup.pc_face); // no other (good) choice than KSPBCGS for this one, symmetry is broken

      convergence_check_on_hodge = ns->solve_projection(cell_solver, (cell_solver != NULL), setup.cell_solver_type, setup.pc_cell, true, hold_old, dxyz_hodge_old, setup.control_hodge);

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

    ns->compute_velocity_at_nodes();
    ns->compute_pressure();

    setup.tn += setup.dt_to_period/setup.ls->f0;

    if(setup.save_forces)
      setup.export_forces(ns);

    if(setup.save_timing)
      setup.export_and_accumulate_timings(ns);

    if(setup.save_hodge_convergence)
      setup.export_hodge_convergence(mpi);

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", setup.iter, setup.tn, 100*(setup.tn - setup.tstart)/setup.duration_to_period/setup.ls->f0, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(ns->get_max_L2_norm_u() > 50.0*setup.ls->vx_max())
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

  if(dxyz_hodge_old != NULL)
    delete [] dxyz_hodge_old;

  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    delete bc_v_interface[dim];

  if(setup.save_timing)
    setup.print_averaged_timings(mpi);

  if(cell_solver!= NULL)
    delete cell_solver;
  if(face_solver!=NULL)
    delete face_solver;

  delete ns;
  // the brick and the connectivity are deleted within the above destructor...
  delete data;

  return 0;
}
