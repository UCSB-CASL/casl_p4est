/*
 * The navier stokes solver applied to the oscillating sphere problem
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_trajectory_of_point.h>
#endif


#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

// --> extra info to be printed when -help is invoked
#ifdef P4_TO_P8
const std::string extra_info = "\
    This program provides a general setup for simulating the flow past an oscillating sphere \n\
    in a closed box. The domain is [-1, 1]x[-1, 1]x[-1, 1]. The radius of the sphere is 0.1, the \n\
    sphere center follows the kinematics \n\
                  x(t) = -X0*cos(2.0*PI*f0*t), y(t) = 0.0, z(t) = 0.0 \n\
    The boundary conditions are:\n\
    - Dirichlet (0,0,0) for the velocity components on all walls (no slip); \n\
    - no-slip condition, i.e. vx(t)=2.0*PI*X0*f0*sin(2.0*PI*f0*t), vy(t) = 0.0, vz(t) = 0.0 on the sphere surface; \n\
    - homogeneous Neumann for the pressure on all walls; \n\
    - homogeneous Neumann for the pressure on the sphere interface. \n\
    The flow starts from rest. This setup determines two non-dimensional numbers:\n\
    1) Strouhal=(2*r*f0)/(2.0*PI*X0*f0)=r/(PI*X0);\n\
    2) Reynolds=rho*(2.0*PI*X0*f0)*2.0*r)/mu=(rho*4.0*f0*r^2)/(St*mu);\n\
    By the definition of the above parameters and nondimensional numbers, we set \n\
                                  X0=r/(PI*St) \n\
    rho is set to 1.0, f0 is set to 1.0, and \n\
                              mu=(4*rho*f0*r^2)/(St*Re). \n\
    Developer: Raphael Egan (raphaelegan@ucsb.edu) based on a general main file from Arthur Guittet (with corrections)";
#else
const std::string extra_info = "\
    This program provides a general setup for simulating the flow past an oscillating cylinder \n\
    in a closed box. The domain is [-1, 1]x[-1, 1]. The radius of the cylinder is 0.05, the \n\
    cylinder center follows the kinematics \n\
                      x(t) = -X0*cos(2.0*PI*f0*t), y(t) = 0.0 \n\
    The boundary conditions are:\n\
    - Dirichlet (0,0) for the velocity components on all walls (no slip); \n\
    - no-slip condition, i.e. vx(t)=2.0*PI*X0*f0*sin(2.0*PI*f0*t), vy(t) = 0.0 on the cylinder surface; \n\
    - homogeneous Neumann for the pressure on all walls; \n\
    - homogeneous Neumann for the pressure on the cylinder interface. \n\
    The flow starts from rest.  This setup determines two non-dimensional numbers:\n\
    1) Strouhal=(2*r*f0)/(2.0*PI*X0*f0)=r/(PI*X0);\n\
    2) Reynolds=rho*(2.0*PI*X0*f0)*2.0*r)/mu=(rho*4.0*f0*r^2)/(St*mu);\n\
    By the definition of the above parameters and nondimensional numbers, we set \n\
                                  X0=r/(PI*St) \n\
    rho is set to 1.0, f0 is set to 1.0, and \n\
                              mu=(4*rho*f0*r^2)/(St*Re). \n\
    Developer: Raphael Egan (raphaelegan@ucsb.edu) based on a general main file from Arthur Guittet (with corrections)";
#endif

const double xmin = -1.0;
const double xmax = +1.0;
const double ymin = -1.0;
const double ymax = +1.0;
#ifdef P4_TO_P8
const double zmin = -1.0;
const double zmax = +1.0;
#endif


#ifdef P4_TO_P8
const double r0 = 0.1;
#else
const double r0 = 0.05;
#endif

// mass density
const double rho  = 1.0; // set to 1.0 for simplicity (nondimensional inputs only)
// oscillation frequency
const double f0   = 1.0; // arbitrary value: the (nondimensional) results __should__ be frequency-independent!
// excursion of the sphere/cylinder motion
double X0; // to be set based on the desired Strouhal number

// current simulation time and (fixed) time step
// they are required for the evaluation of the levelset function and interface conditions
// we want to enforce field conditions at time tn + dt
double tn, dt;

class LEVEL_SET: public CF_DIM
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return r0 - sqrt(SUMD(SQR(x + X0*cos(2*PI*f0*(tn + dt))), SQR(y), SQR(z)));
  }
} level_set;

struct BCWALLTYPE_P : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;

#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_w;
#endif

struct BCWALLVALUE_U : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_wall_value_v;

#ifdef P4_TO_P8
struct BCWALLVALUE_W : CF_3
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_wall_value_w;
#endif

struct BCINTERFACE_VALUE_U : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 2.0*PI*f0*X0*sin(2*PI*f0*(tn + dt)); // no longer u0 as in the original main
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_interface_value_v;

#ifdef P4_TO_P8
struct BCINTERFACE_VALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} bc_interface_value_w;
#endif

struct initial_velocity_unm1_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_vnm1;

struct initial_velocity_v_n_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_vn;

#ifdef P4_TO_P8
struct initial_velocity_wnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} initial_velocity_wnm1;

struct initial_velocity_w_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} initial_velocity_wn;
#endif

void initialize_force_output(char* file_forces, const char *out_dir, const int& lmin, const int& lmax, const double& threshold_split_cell, const int& nsteps, const int& sl_order, const mpi_environment_t& mpi, const double& tstart)
{
  sprintf(file_forces, "%s/forces_%d-%d_split_threshold_%.2f_nsteps_%d_sl_%d.dat", out_dir, lmin, lmax, threshold_split_cell, nsteps, sl_order);
  PetscErrorCode ierr = PetscPrintf(mpi.comm(), "Saving forces in ... %s\n", file_forces); CHKERRXX(ierr);

  if(mpi.rank() == 0)
  {
    if(!file_exists(file_forces))
    {
      FILE* fp_forces = fopen(file_forces, "w");
      if(fp_forces==NULL)
        throw std::runtime_error("initialize_force_output: could not open file for force output.");
#ifdef P4_TO_P8
      fprintf(fp_forces, "%% tn/T | Cd_x | Cd_y | Cd_z\n");
#else
      fprintf(fp_forces, "%% tn/T | Cd_x | Cd_y\n");
#endif
      fclose(fp_forces);
    }
    else
    {
      FILE* fp_forces = fopen(file_forces, "r+");
      char* read_line = NULL;
      size_t len = 0;
      ssize_t len_read;
      long size_to_keep = 0;
      if(((len_read = getline(&read_line, &len, fp_forces)) != -1))
        size_to_keep += (long) len_read;
      else
        throw std::runtime_error("initialize_force_output: couldn't read the first header line of forces_...dat");
      double time, time_nm1;
      double dt = 0.0;
      bool not_first_line = false;
      while ((len_read = getline(&read_line, &len, fp_forces)) != -1) {
        if(not_first_line)
          time_nm1 = time;
#ifdef P4_TO_P8
        sscanf(read_line, "%lg %*g %*g %*g", &time);
#else
        sscanf(read_line, "%lg %*g %*g", &time);
#endif
        if(not_first_line)
          dt = time-time_nm1;
        if(time <= tstart+0.1*dt) // +0.1*dt to avoid roundoff errors when exporting the data
          size_to_keep += (long) len_read;
        else
          break;
        not_first_line=true;
      }
      fclose(fp_forces);
      if(read_line)
        free(read_line);
      if(truncate(file_forces, size_to_keep))
        throw std::runtime_error("initialize_force_output: couldn't truncate forces_...dat");
    }

    char liveplot_forces[PATH_MAX];
    sprintf(liveplot_forces, "%s/live_forces.gnu", out_dir);
    if(!file_exists(liveplot_forces))
    {
      FILE* fp_liveplot_forces = fopen(liveplot_forces, "w");
      if(fp_liveplot_forces==NULL)
        throw std::runtime_error("initialize_force_output: could not open file for force liveplot.");
      fprintf(fp_liveplot_forces, "set term wxt noraise\n");
      fprintf(fp_liveplot_forces, "set key top right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "set ylabel \"Nondimensional force\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_liveplot_forces, "\t \"forces_%d-%d_split_threshold_%.2f_nsteps_%d_sl_%d.dat\" using 1:%d title '%c-component' with lines lw 3", lmin, lmax, threshold_split_cell, nsteps, sl_order, ((int)dd+2), ((dd==0)?'x':((dd==1)?'y':'z')));
        if(dd < P4EST_DIM-1)
          fprintf(fp_liveplot_forces, ",\\");
        fprintf(fp_liveplot_forces, "\n");
      }
      fprintf(fp_liveplot_forces, "pause 4\n");
      fprintf(fp_liveplot_forces, "reread");
      fclose(fp_liveplot_forces);
    }

    char tex_plot_forces[PATH_MAX];
    sprintf(tex_plot_forces, "%s/tex_forces.gnu", out_dir);
    if(!file_exists(tex_plot_forces))
    {
      FILE *fp_tex_plot_forces = fopen(tex_plot_forces, "w");
      if(fp_tex_plot_forces==NULL)
        throw std::runtime_error("initialize_foce_output: could not open file for force tex figure.");
      fprintf(fp_tex_plot_forces, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_forces, "set output 'force_history.tex'\n");
      fprintf(fp_tex_plot_forces, "set key top right Right \n");
      fprintf(fp_tex_plot_forces, "set xlabel \"$t$\"\n");
#ifdef P4_TO_P8
      fprintf(fp_tex_plot_forces, "set ylabel \"$\\\\mathbf{C}_{\\\\mathrm{D}} = \\\\frac{2}{\\\\rho \\\\pi r^{2}\\\\left(2\\\\pi f_{0} X_{0}\\\\right)^{2}}\\\\int_{\\\\Gamma}{ \\\\left( -p \\\\mathbf{I} + 2\\\\mu \\\\mathbf{D} \\\\right)\\\\cdot \\\\mathbf{n} \\\\, \\\\mathrm{d}\\\\Gamma}$ \" \n");
#else
      fprintf(fp_tex_plot_forces, "set ylabel \"$\\\\mathbf{C}_{\\\\mathrm{D}} = \\\\frac{1}{\\\\rho r \\\\left(2\\\\pi f_{0} X_{0}\\\\right)^{2}}\\\\int_{\\\\Gamma}{ \\\\left( -p \\\\mathbf{I} + 2\\\\mu \\\\mathbf{D} \\\\right)\\\\cdot \\\\mathbf{n} \\\\, \\\\mathrm{d}\\\\Gamma}$ \" \n");
#endif
      fprintf(fp_tex_plot_forces, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_tex_plot_forces, "\t \"forces_%d-%d_split_threshold_%.2f_nsteps_%d_sl_%d.dat\" using 1:%d title '$C_{\\mathrm{D}, %c}$' with lines lw 3", lmin, lmax, threshold_split_cell, nsteps, sl_order, ((int) dd+2), ((dd==0)?'x':((dd==1)?'y':'z')));
        if(dd < P4EST_DIM-1)
          fprintf(fp_tex_plot_forces, ",\\\n");
      }
      fclose(fp_tex_plot_forces);
    }

    char tex_forces_script[PATH_MAX];
    sprintf(tex_forces_script, "%s/plot_tex_forces.sh", out_dir);
    if(!file_exists(tex_forces_script))
    {
      FILE *fp_tex_forces_script = fopen(tex_forces_script, "w");
      if(fp_tex_forces_script==NULL)
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


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin", "min level of the trees, default is " + to_string(8 - P4EST_DIM));
  cmd.add_option("lmax", "max level of the trees, default is " + to_string(13 - P4EST_DIM));
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is 0.1");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is 4.0!");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is 1 (length of domain is 2)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is 1 (height of domain is 2)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is 1 (width of domain is 2)");
#endif
  // physical parameters for the simulations
  cmd.add_option("duration", "the duration of the simulation (tfinal-tstart), in units of oscillating periods! If not restarted, tstart = 0.0, default duration is 3 cycles.");
  cmd.add_option("St", "the Strouhal number = r/(pi*X0). This option sets X0=r/(PI*St). The minimum value allowed is 0.04, default is 4/PI");
#ifdef P4_TO_P8
  cmd.add_option("Re", "the Reynolds number=(rho*(2.0*PI*X0*f0)*2.0*r)/mu=(rho*4.0*f0*r^2)/(St*mu) where rho = 1.0, f0 = 1, r = 0.1 (default is 300). This option sets mu.");
#else
  cmd.add_option("Re", "the Reynolds number=(rho*(2.0*PI*X0*f0)*2.0*r)/mu=(rho*4.0*f0*r^2)/(St*mu) where rho = 1.0, f0 = 1, r = 0.05 (default is 300). This option sets mu.");
#endif
  // method-related parameters
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is 2");
  cmd.add_option("nsteps", "number of time steps per oscillation period, default is 200.");
  cmd.add_option("hodge_tol", "numerical tolerance on the Hodge variable, at all time steps, relative to 0.5*rho*SQR(2*pi*X0*x0), default is 0.1");
  cmd.add_option("niter_hodge", "max number of iterations for convergence of the Hodge variable, at all time steps, default is 10");
  cmd.add_option("pc_cell", "preconditioner for cell-solver: jacobi, sor or hypre, default is sor.");
  cmd.add_option("cell_solver", "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is bicgstab.");
  cmd.add_option("pc_face", "preconditioner for face-solver: jacobi, sor or hypre, default is sor.");
  // output-control parameters
  cmd.add_option("export_folder", "exportation_folder");
  cmd.add_option("save_vtk", "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt", "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
  cmd.add_option("save_forces", "save the forces");
  cmd.add_option("save_state_dt", "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt/f0 time steps in backup_ subfolders.");
  cmd.add_option("save_nstates", "determines how many solver states must be memorized in backup_ folders (default is 1).");
  cmd.add_option("timing", "if defined, prints timing information (typically for scaling analysis).");

  if(cmd.parse(argc, argv, extra_info))
    return 0;

  int lmin, lmax;
  my_p4est_navier_stokes_t* ns                    = NULL;
  my_p4est_brick_t* brick                         = NULL;
  splitting_criteria_cf_and_uniform_band_t* data  = NULL;
  LEVEL_SET level_set;

  int sl_order, nsteps_per_period;
  double threshold_split_cell, uniform_band;
  int n_xyz [P4EST_DIM];

  const double hodge_tolerance          = cmd.get<double>("hodge_tol", 0.1);
  const unsigned int niter_hodge_max    = cmd.get<unsigned int>("niter_hodge", 10);
  const double duration                 = cmd.get<double>("duration", 3.0)*(1.0/f0);
#if defined(POD_CLUSTER)
  const string export_dir               = cmd.get<string>("export_folder", "/scratch/regan/oscillating_sphere");
#elif defined(STAMPEDE)
  const string export_dir               = cmd.get<string>("export_folder", "/work/04965/tg842642/stampede2/oscillating_sphere");
#elif defined(LAPTOP)
  const string export_dir               = cmd.get<string>("export_folder", "/home/raphael/workspace/projects/oscillating_sphere");
#else
  const string export_dir               = cmd.get<string>("export_folder", "/home/regan/workspace/projects/oscillating_sphere");
#endif
  const bool save_vtk                   = cmd.contains("save_vtk");
  const bool get_timing                 = cmd.contains("timing");
  double vtk_dt                         = -1.0;
  if(save_vtk)
  {
    if(!cmd.contains("vtk_dt"))
      throw std::runtime_error("main_oscillating_sphere_" + to_string(P4EST_DIM) +"d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
    vtk_dt = cmd.get<double>("vtk_dt", -1.0);
    if(vtk_dt <= 0.0)
      throw std::invalid_argument("main_oscillating_sphere_" + to_string(P4EST_DIM) + "d.cpp: the value of vtk_dt must be strictly positive.");
  }
  const bool save_forces                = cmd.contains("save_forces"); double forces[P4EST_DIM];
  const bool save_state                 = cmd.contains("save_state_dt"); double dt_save_data = -1.0;
  const unsigned int n_states           = cmd.get<unsigned int>("save_nstates", 1);
  if(save_state)
  {
    dt_save_data                        = cmd.get<double>("save_state_dt", -1.0)*(1.0/f0);
    if(dt_save_data < 0.0)
      throw std::invalid_argument("main_oscillating_sphere_" + to_string(P4EST_DIM) + "d.cpp: the value of save_state_dt must be strictly positive.");
  }

  const string des_pc_cell              = cmd.get<string>("pc_cell", "sor");
  const string des_solver_cell          = cmd.get<string>("cell_solver", "bicgstab");
  const string des_pc_face              = cmd.get<string>("pc_face", "sor");
  KSPType cell_solver_type;
  PCType pc_cell, pc_face;
  if (des_pc_cell.compare("hypre")==0)
    pc_cell = PCHYPRE;
  else if (des_pc_cell.compare("jacobi'")==0)
    pc_cell = PCJACOBI;
  else
  {
    if(des_pc_cell.compare("sor")!=0 && !mpi.rank())
      std::cerr << "The desired preconditioner for the cell-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
    pc_cell = PCSOR;
  }
  if(des_solver_cell.compare("cg")==0)
    cell_solver_type = KSPCG;
  else
  {
    if(des_solver_cell.compare("bicgstab")!=0 && !mpi.rank())
      std::cerr << "The desired Krylov solver for the cell-solver was either not allowed or not correctly understood. BiCGStab is used instead" << std::endl;
    cell_solver_type = KSPBCGS;
  }
  if (des_pc_face.compare("hypre")==0)
    pc_face = PCHYPRE;
  else if (des_pc_face.compare("jacobi'")==0)
    pc_face = PCJACOBI;
  else
  {
    if(des_pc_face.compare("sor")!=0 && !mpi.rank())
      std::cerr << "The desired preconditioner for the face-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
    pc_face = PCSOR;
  }


  PetscErrorCode ierr;
  double Re, St, mu;
  const double xyz_min[P4EST_DIM] = {DIM(xmin, ymin, zmin)};
  const double xyz_max[P4EST_DIM] = {DIM(xmax, ymax, zmax)};
  const int periodic  [P4EST_DIM] = {DIM(0, 0, 0)};

  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;

  bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
  bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
  bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

  bc_v[0].setInterfaceType(DIRICHLET); bc_v[0].setInterfaceValue(bc_interface_value_u);
  bc_v[1].setInterfaceType(DIRICHLET); bc_v[1].setInterfaceValue(bc_interface_value_v);
#ifdef P4_TO_P8
  bc_v[2].setInterfaceType(DIRICHLET); bc_v[2].setInterfaceValue(bc_interface_value_w);
#endif
  bc_p.setInterfaceType(NEUMANN); bc_p.setInterfaceValue(bc_interface_value_p);



  if(cmd.contains("restart"))
  {
    const string backup_directory = cmd.get<string>("restart", "");
    if(!is_folder(backup_directory.c_str()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_oscillating_sphere_3d: the restart path %s is not an accessible directory.", backup_directory.c_str());
#else
      sprintf(error_msg, "main_oscillating_sphere_2d: the restart path %s is not an accessible directory.", backup_directory.c_str());
#endif
      throw std::invalid_argument(error_msg);
    }
    if (ns != NULL)
    {
      delete ns; ns = NULL;
    }
    P4EST_ASSERT(ns == NULL);
    ns                      = new my_p4est_navier_stokes_t(mpi, backup_directory.c_str(), tn);
    nsteps_per_period       = cmd.get<int>("nsteps", 200);
    dt = 1.0/(((double) nsteps_per_period)*f0);

    ns->set_dt(dt);
    p4est_t *p4est_n        = ns->get_p4est();
    p4est_t *p4est_nm1      = ns->get_p4est_nm1();

    lmin                    = cmd.get<int>("lmin", ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl);
    lmax                    = cmd.get<int>("lmax", ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl);
    double lip              = ((splitting_criteria_t*) p4est_n->user_pointer)->lip;
    threshold_split_cell    = cmd.get<double>("thresh", ns->get_split_threshold());
#ifdef P4EST_ENABLE_DEBUG
    double length           = ns->get_length_of_domain();
    double height           = ns->get_height_of_domain();
#ifdef P4_TO_P8
    double width            = ns->get_width_of_domain();
#endif
    P4EST_ASSERT((fabs(length - 2.0) < 2.0*10.0*EPS) && (fabs(height - 2.0) < 2.0*10.0*EPS) && ONLY3D((fabs(width - 2.0) < 2.0*10.0*EPS) &&) (fabs(ns->get_rho() - 1.0) < 10.0*EPS));
#endif
    St                      = cmd.get<double>("St", 4.0/PI);
    X0                      = r0/(PI*St);
    if(cmd.contains("Re"))
    {
      Re                    = cmd.get<double>("Re");
      mu                    = (4.0*rho*f0*r0*r0)/(St*Re);
    }
    else
    {
      mu                    = ns->get_mu();
      Re                    = (4.0*rho*f0*r0*r0)/(St*mu);
    }

    if(brick != NULL && brick->nxyz_to_treeid != NULL)
    {
      P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
      delete brick; brick = NULL;
    }
    P4EST_ASSERT(brick == NULL);
    brick                   = ns->get_brick();
    n_xyz[0]                = brick->nxyztrees[0];
    n_xyz[1]                = brick->nxyztrees[1];
#ifdef P4_TO_P8
    n_xyz[2]                = brick->nxyztrees[2];
    P4EST_ASSERT((fabs(brick->xyz_min[2]-xyz_min[2]) < fabs(xyz_min[2])*10.0*EPS) && (fabs(brick->xyz_max[2]-xyz_max[2]) < fabs(xyz_max[2])*10.0*EPS));
#endif
    P4EST_ASSERT((fabs(brick->xyz_min[1]-xyz_min[1]) < fabs(xyz_min[1])*10.0*EPS) && (fabs(brick->xyz_max[1]-xyz_max[1]) < fabs(xyz_max[1])*10.0*EPS));
    P4EST_ASSERT((fabs(brick->xyz_min[0]) < 10.0*EPS) && (fabs(xyz_min[0]) < 10.0*EPS) && (fabs(brick->xyz_max[0]-xyz_max[0]) < fabs(xyz_max[0])*10.0*EPS));

    uniform_band            = cmd.get<double>("uniform_band", ns->get_uniform_band());
    sl_order                = cmd.get<int>("sl_order", ns->get_sl_order());

    if(data != NULL)
    {
      delete data; data = NULL;
    }
    P4EST_ASSERT(data == NULL);
    data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band, lip);
    splitting_criteria_t* to_delete = (splitting_criteria_t*) p4est_n->user_pointer;
    bool fix_restarted_grid = (lmax!=to_delete->max_lvl);
    delete to_delete;
    p4est_n->user_pointer   = (void*) data;
    p4est_nm1->user_pointer = (void*) data; // p4est_n and p4est_nm1 always point to the same splitting_criteria_t no need to delete the nm1 one, it's just been done
    ns->set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, -1.0); // cfl is irrelevant in this case

    ns->set_bc(bc_v, &bc_p);
    if(fix_restarted_grid)
      ns->refine_coarsen_grid_after_restart(&level_set, false);
  }
  else
  {
    tn                      = 0.0; // no restart so we assume we start from 0.0
    nsteps_per_period       = cmd.get<int>("nsteps", 200);
    dt                      = 1.0/(((double) nsteps_per_period)*f0);

    lmin                    = cmd.get<int>("lmin", (5 + (3-P4EST_DIM)));
    lmax                    = cmd.get<int>("lmax", (10 +(3-P4EST_DIM)));
    threshold_split_cell    = cmd.get<double>("thresh", 0.1);
    n_xyz[0]                = cmd.get<int>("nx", 1);
    n_xyz[1]                = cmd.get<int>("ny", 1);
#ifdef P4_TO_P8
    n_xyz[2]                = cmd.get<int>("nz", 1);
#endif
    uniform_band            = cmd.get<double>("uniform_band", 4.0);
    sl_order                = cmd.get<int>("sl_order", 2);

    St                      = cmd.get<double>("St", 4.0/PI);
    X0                      = r0/(PI*St);
    Re                      = cmd.get<double>("Re", 300.0);
    mu                      = (4.0*rho*f0*r0*r0)/(St*Re);

    p4est_connectivity_t *connectivity;
    if(brick != NULL && brick->nxyz_to_treeid != NULL)
    {
      P4EST_FREE(brick->nxyz_to_treeid);brick->nxyz_to_treeid = NULL;
      delete brick; brick = NULL;
    }
    P4EST_ASSERT(brick == NULL);
    brick = new my_p4est_brick_t;

    connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);
    if(data != NULL)
    {
      delete data; data = NULL;
    }
    P4EST_ASSERT(data == NULL);
    data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band);

    p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est_nm1->user_pointer = (void*) data;

    for(int l=0; l<lmax; ++l)
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
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, level_set, phi);

    CF_DIM *vnm1[P4EST_DIM] = {DIM(&initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1)};
    CF_DIM *vn  [P4EST_DIM] = {DIM(&initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn  )};

    ns = new my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n);
    ns->set_phi(phi);
    ns->set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, -1.0); // cfl is irrelevant in this case
    ns->set_dt(dt, dt);
    ns->set_velocities(vnm1, vn);

    ns->set_bc(bc_v, &bc_p);
  }


#ifdef P4_TO_P8
  ierr = PetscPrintf(mpi.comm(), "Parameters : St = %g, Re = %g, f0 = %g, mu = %g, rho = %g, grid is %dx%dx%d\n", St, Re, f0, mu, rho, n_xyz[0], n_xyz[1], n_xyz[2]); CHKERRXX(ierr);
#else
  ierr = PetscPrintf(mpi.comm(), "Parameters : St = %g, Re = %g, f0 = %g, mu = %g, rho = %g, grid is %dx%d\n", St, Re, f0, mu, rho, n_xyz[0], n_xyz[1]); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(mpi.comm(), "dt = T/%d, uniform_band = %g\n", nsteps_per_period, uniform_band);

  char out_dir[PATH_MAX], vtk_path[PATH_MAX], vtk_name[PATH_MAX];
  sprintf(out_dir, "%s/%dD/Re_%.2f/lmin_%d_lmax_%d", export_dir.c_str(), P4EST_DIM, Re, lmin, lmax);
  sprintf(vtk_path, "%s/vtu", out_dir);
  if(save_vtk || save_forces || save_state)
  {
    if(create_directory(out_dir, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_oscillating_sphere_%dd: could not create exportation directory %s", P4EST_DIM, out_dir);
      throw std::runtime_error(error_msg);
    }
    if(save_vtk && create_directory(vtk_path, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
      sprintf(error_msg, "main_oscillating_sphere_%dd: could not create exportation directory for vtk files %s", P4EST_DIM, vtk_path);
      throw std::runtime_error(error_msg);
    }
  }

  int iter = 0;
  int export_vtk = -1;
  int save_data_idx = (int) floor(tn/dt_save_data); // so that we don't save the very first one which was either already read from file, or the known initial condition...

  char file_forces[PATH_MAX];
  if(save_forces)
    initialize_force_output(file_forces, out_dir, lmin, lmax, threshold_split_cell, nsteps_per_period, sl_order, mpi, tn);

  parStopWatch watch, substep_watch;
  double mean_full_iteration_time = 0.0, mean_viscosity_step_time = 0.0, mean_projection_step_time = 0.0, mean_compute_velocity_at_nodes_time = 0.0, mean_update_time = 0.0;
  if(get_timing)
    watch.start("Total runtime");

  my_p4est_poisson_cells_t* cell_solver = NULL;
  my_p4est_poisson_faces_t* face_solver = NULL;

  const double tstart = tn;
  while(tn+0.01*dt<tstart+duration)
  {
    if(get_timing)
      substep_watch.start("");
    if(iter>0)
    {
      ns->set_dt(dt);

      if(tn+dt>tstart+duration)
      {
        dt = tstart+duration-tn;
        ns->set_dt(dt);
      }

      bool solvers_can_be_reused = ns->update_from_tn_to_tnp1(&level_set, false, false);
      if(cell_solver!=NULL && (!solvers_can_be_reused)){
        delete  cell_solver; cell_solver = NULL; }
      if(face_solver!=NULL && (!solvers_can_be_reused)){
        delete  face_solver; face_solver = NULL; }
    }
    if(get_timing)
    {
      substep_watch.stop();
      mean_update_time += substep_watch.read_duration();
    }
    if(save_state && ((int) floor(tn/dt_save_data)) != save_data_idx)
    {
      save_data_idx = ((int) floor(tn/dt_save_data));
      ns->save_state(out_dir, tn, n_states);
    }

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, ns->get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double corr_hodge = 1.0;
    unsigned int iter_hodge = 0;
    while(iter_hodge<niter_hodge_max && corr_hodge>hodge_tolerance*(0.5*SQR(2.0*PI*X0*f0)*dt))
    {
      hodge_new = ns->get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      if(get_timing)
        substep_watch.start("");
      ns->solve_viscosity(face_solver, (face_solver!=NULL), KSPBCGS, pc_face); // no other (good) choice than KSPBCGS for this one, symmetry is broken
      if(get_timing)
      {
        substep_watch.stop();
        mean_viscosity_step_time += substep_watch.read_duration();
        substep_watch.start("");
      }
      ns->solve_projection(cell_solver, (cell_solver!=NULL), cell_solver_type, pc_cell);
      if(get_timing)
      {
        substep_watch.stop();
        mean_projection_step_time += substep_watch.read_duration();
      }

      hodge_new = ns->get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      corr_hodge = 0.0;
      p4est_t *p4est = ns->get_p4est();
      my_p4est_interpolation_nodes_t *interp_phi = ns->get_interp_phi();
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
          double xyz[P4EST_DIM];
          quad_xyz_fr_q(quad_idx, tree_idx, p4est, ns->get_ghost(), xyz);
          if((*interp_phi)(xyz) < 0.0)
            corr_hodge = max(corr_hodge, fabs(ho[quad_idx]-hn[quad_idx]));
        }
      }
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &corr_hodge, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      ierr = VecRestoreArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(hodge_new, &hn); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi.comm(), "hodge iteration #%d, error = %e\n", iter_hodge, corr_hodge); CHKERRXX(ierr);
      iter_hodge++;
    }
    ierr = VecDestroy(hodge_old); CHKERRXX(ierr);
    if(get_timing)
      substep_watch.start("");
    ns->compute_velocity_at_nodes();
    if(get_timing)
    {
      substep_watch.stop();
      mean_compute_velocity_at_nodes_time += substep_watch.read_duration();
    }
    ns->compute_pressure();

    tn += dt;

    if(save_forces)
    {
      ns->compute_forces(forces);
      if(!mpi.rank())
      {
        FILE* fp_forces = fopen(file_forces, "a");
        if(fp_forces==NULL)
#ifdef P4_TO_P8
          throw std::invalid_argument("main_oscillating_sphere_3d: could not open file for forces output.");
        fprintf(fp_forces, "%g %g %g %g\n", f0*tn, forces[0]/(.5*PI*r0*r0*SQR(2.0*PI*X0*f0)*rho), forces[1]/(.5*PI*r0*r0*SQR(2.0*PI*X0*f0)*rho), forces[2]/(.5*PI*r0*r0*SQR(2.0*PI*X0*f0)*rho));
#else
          throw std::invalid_argument("main_oscillating_sphere_2d: could not open file for forces output.");
        fprintf(fp_forces, "%g %g %g\n", f0*tn, forces[0]/(r0*SQR(2.0*PI*X0*f0)*rho), forces[1]/(r0*SQR(2.0*PI*X0*f0)*rho));
#endif
        fclose(fp_forces);
      }
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*(tn-tstart)/duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(ns->get_max_L2_norm_u()>50.0*2.0*PI*X0*f0)
    {
      if(save_vtk)
      {
        sprintf(vtk_name, "%s/snapshot_%d", vtk_path, export_vtk+1);
        ns->save_vtk(vtk_name);
      }
      std::cerr << "The simulation blew up..." << std::endl;
      break;
    }

    if(save_vtk && ((int) floor(tn/vtk_dt)) != export_vtk)
    {
      export_vtk = ((int) floor(tn/vtk_dt));
      sprintf(vtk_name, "%s/snapshot_%d", vtk_path, export_vtk);
      ns->save_vtk(vtk_name);
    }

    iter++;
  }

  if(get_timing)
  {
    watch.stop();
    mean_full_iteration_time             = watch.read_duration()/((double) iter);
    mean_viscosity_step_time            /= ((double) iter);
    mean_projection_step_time           /= ((double) iter);
    mean_compute_velocity_at_nodes_time /= ((double) iter);
    mean_update_time                    /= ((double) iter);

    ierr = PetscPrintf(mpi.comm(), "Mean computational time spent on \n"); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " viscosity step: %.5e\n", mean_viscosity_step_time); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " projection step: %.5e\n", mean_projection_step_time); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " computing velocities at nodes: %.5e\n", mean_compute_velocity_at_nodes_time); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " grid update: %.5e\n", mean_update_time); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " full iteration (total): %.5e\n", mean_full_iteration_time); CHKERRXX(ierr);
  }

  if(cell_solver!= NULL)
    delete cell_solver;
  if(face_solver!=NULL)
    delete face_solver;

  delete ns;
  delete data;

  return 0;
}
