
/*
 * The navier stokes solver applied to the flow past a static sphere
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

double xmin = 0;
double xmax = 1;
double ymin = 0;
double ymax = 1;
#ifdef P4_TO_P8
double zmin = 0;
double zmax = 1;
#endif

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

/*
 * - karman street
 */

int test_number;

double mu;
double rho;
double tn;
double dt;
double u0;
double r0;


#ifdef P4_TO_P8

class INIT_SMOKE : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return (x<2 && sqrt(SQR(y) + SQR(z))<.4) ? 1 : 0;
  }
} init_smoke;

class BC_SMOKE : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return (fabs(x-xmin)<EPS && sqrt(SQR(y) + SQR(z))<.4) ? 1 : 0;
  }
} bc_smoke;

class LEVEL_SET: public CF_3
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y, double z) const
  {
    return r0 - sqrt(SQR(x-(xmax+xmin)/4) + SQR(y-(ymax+ymin)/2) + SQR(z-(zmax+zmin)/2));
  }
} level_set;

struct BCWALLTYPE_P : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS)
      return DIRICHLET;
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS)
      return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS)
      return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type_v;

struct BCWALLTYPE_W : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS)
      return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type_w;

struct BCWALLVALUE_U : CF_3
{
  double operator()(double x, double, double) const
  {
    if(fabs(x-xmax)<EPS)
      return 0;
    else return u0;
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_v;

struct BCWALLVALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_w;

struct BCINTERFACE_VALUE_U : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_v;

struct BCINTERFACE_VALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_interface_value_w;

struct initial_velocity_unm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return u0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return u0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_vnm1;

struct initial_velocity_v_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_vn;

struct initial_velocity_wnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_wnm1;

struct initial_velocity_w_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
} initial_velocity_wn;

struct external_force_u_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
};

struct external_force_v_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0;
  }
};

struct external_force_w_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
};

#else

class INIT_SMOKE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return (x<2 && y>-1 && y<1) ? 1 : 0;
  }
} init_smoke;


class BC_SMOKE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return(fabs(x-xmin)<EPS && y>-.5 && y<.5) ? 1 : 0;
  }
} bc_smoke;


class LEVEL_SET: public CF_2
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    return r0 - sqrt(SQR(x-(xmax+xmin)/4) + SQR(y-(ymax+ymin)/2));
  }
} level_set;

struct BCWALLTYPE_P : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS)
      return DIRICHLET;
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    return 0.0;
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    return 0.0;
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS)
      return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS)
      return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type_v;

struct BCWALLVALUE_U : CF_2
{
  double operator()(double x, double) const
  {
    if(fabs(x-xmax)<EPS)
      return 0;
    else
      return u0;
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_wall_value_v;

struct BCINTERFACE_VALUE_U : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_interface_value_v;

struct initial_velocity_unm1_t : CF_2
{
  double operator()(double, double) const
  {
    return u0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_2
{
  double operator()(double, double) const
  {
    return u0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} initial_velocity_vn;

struct external_force_u_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} external_force_u;

struct external_force_v_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
};

#endif


void initialize_force_output(char* file_forces, const char *out_dir, const int& lmin, const int& lmax, const double& threshold_split_cell, const double& cfl, const int& sl_order, const mpi_environment_t& mpi, const double& tstart)
{
  sprintf(file_forces, "%s/forces_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat", out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order);
  PetscErrorCode ierr = PetscPrintf(mpi.comm(), "Saving forces in ... %s\n", file_forces); CHKERRXX(ierr);

  if(mpi.rank() == 0)
  {
    if(!file_exists(file_forces))
    {
      FILE* fp_forces = fopen(file_forces, "w");
      if(fp_forces==NULL)
        throw std::runtime_error("initialize_force_output: could not open file for force output.");
#ifdef P4_TO_P8
      fprintf(fp_forces, "%% tn | Cd_x | Cd_y\n");
#else
      fprintf(fp_forces, "%% tn | Cd_x | Cd_y | Cd_z\n");
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
      fprintf(fp_liveplot_forces, "set ylabel \"Nondimensional force \" font \"Arial,14\"\n");
      fprintf(fp_liveplot_forces, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_liveplot_forces, "\t \"forces_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:%d title '%c-component' with lines lw 3", lmin, lmax, threshold_split_cell, cfl, sl_order, ((int)dd+2), ((dd==0)?'x':((dd==1)?'y':'z')));
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
      fprintf(fp_tex_plot_forces, "set ylabel \"$\\\\mathbf{C}_{\\\\mathrm{D}} = \\\\frac{2}{\\\\rho \\\\pi r^{2}u_{0}^{2}}\\\\int_{\\\\Gamma}{ \\\\left( -p \\\\mathbf{I} + 2\\\\mu \\\\mathbf{D} \\\\right)\\\\cdot \\\\mathbf{n} \\\\, \\\\mathrm{d}\\\\Gamma}$ \" \n");
#else
      fprintf(fp_tex_plot_forces, "set ylabel \"$\\\\mathbf{C}_{\\\\mathrm{D}} = \\\\frac{1}{\\\\rho r u_{0}^{2}}\\\\int_{\\\\Gamma}{ \\\\left( -p \\\\mathbf{I} + 2\\\\mu \\\\mathbf{D} \\\\right)\\\\cdot \\\\mathbf{n} \\\\, \\\\mathrm{d}\\\\Gamma}$ \" \n");
#endif
      fprintf(fp_tex_plot_forces, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_tex_plot_forces, "\t \"forces_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:%d title '$C_{\\mathrm{D}, %c}$' with lines lw 3", lmin, lmax, threshold_split_cell, cfl, sl_order, ((int) dd+2), ((dd==0)?'x':((dd==1)?'y':'z')));
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
//  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin", "min level of the trees, default is 3");
  cmd.add_option("lmax", "max level of the trees, default is 3");
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is 0.1");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is such that half the sphere radius is tesselated");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is 8 (length of domain is 32)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is 4 (height of domain is 16)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is 4 (width of domain is 16)");
#endif
  cmd.add_option("smoke_thresh", "threshold for smoke refinement, default is 0.5");
  cmd.add_option("refine_with_smoke", "refine the grid with the smoke density and threshold smoke_thresh if present");
  // physical parameters for the simulations
  cmd.add_option("duration", "the duration of the simulation (tfinal-tstart). If not restarted, tstart = 0.0, default duration is 10.");
#ifdef P4_TO_P8
  cmd.add_option("Re", "the Reynolds number = (rho*u0*D)/mu where D=2 is the sphere diameter (default is 350)");
#else
  cmd.add_option("Re", "the Reynolds number = (rho*u0*D)/mu where D=1 is the sphere diameter (default is 200)");
#endif
  cmd.add_option("adapted_dt", "activates the calculation of dt based on the local cell sizes if present");
  cmd.add_option("smoke", "no smoke if option not present, with smoke if option present");
  // method-related parameters
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is 2");
  cmd.add_option("cfl", "dt = cfl * dx/vmax, default is 1.");
  // output-control parameters
  cmd.add_option("export_folder", "exportation_folder");
  cmd.add_option("save_vtk", "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt", "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
  cmd.add_option("save_forces", "save the forces");
  cmd.add_option("save_state_dt", "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates", "determines how many solver states must be memorized in backup_ folders (default is 1).");


  cmd.parse(argc, argv);

  int sl_order                = cmd.get("sl_order", 2);
  int lmin                    = cmd.get("lmin", 3);
  int lmax                    = cmd.get("lmax", 3);
  double cfl                  = cmd.get("cfl", 1.);
  double threshold_split_cell = cmd.get("thresh", 0.1);
  bool save_vtk               = cmd.contains("save_vtk");
  bool save_state             = cmd.contains("save_state_dt"); double dt_save_data = -1.0;
  unsigned int n_states       = cmd.get<unsigned int>("save_nstates", 1);
  if(save_state)
  {
    dt_save_data              = cmd.get<double>("save_state_dt", -1.0);
    if(dt_save_data < 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_flow_past_sphere_3d.cpp: the value of save_state_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_flow_past_sphere_2d.cpp: the value of save_state_dt must be strictly positive.");
#endif
  }
  bool save_forces            = cmd.contains("save_forces");
  bool use_adapted_dt         = cmd.contains("adapted_dt");
  double vtk_dt               = -1.0;
  if(save_vtk)
  {
    if(!cmd.contains("vtk_dt"))
#ifdef P4_TO_P8
      throw std::runtime_error("main_flow_past_sphere_3d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
#else
      throw std::runtime_error("main_flow_past_sphere_2d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
#endif
    vtk_dt = cmd.get<double>("vtk_dt", -1.0);
    if(vtk_dt <= 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_flow_past_sphere_3d.cpp: the value of vtk_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_flow_past_sphere_2d.cpp: the value of vtk_dt must be strictly positive.");
#endif
  }

  bool is_smoke = cmd.contains("smoke");
  bool refine_with_smoke = cmd.contains("refine_with_smoke");
  double smoke_thresh = cmd.get("smoke_thresh", .5);
  double duration;
  double Re;

  rho   = 1;
  u0    = 1;
  nx    = cmd.get<int>("nx", 8);
  ny    = cmd.get<int>("ny", 4);
#ifdef P4_TO_P8
  nz    = cmd.get<int>("nz", 4);
#endif
  xmin  = +0.0;
  xmax  = +32.0;
  ymin  = -8.0;
  ymax  = +8.0;
#ifdef P4_TO_P8
  zmin  = -8.0;
  zmax  = +8.0;
  Re    = cmd.get("Re",350);
  r0    = 1;
#else
  Re    = cmd.get("Re", 200);
  r0    = 0.5 ;
#endif
  mu = 2*r0*rho*u0/Re;
  duration = cmd.get<double>("duration", 200);

  double uniform_band = .5*r0;
#ifdef P4_TO_P8
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny, (zmax-zmin)/(double)nz) / (1<<lmax);
#else
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny) / (1<<lmax);
#endif
  uniform_band /= dxmin;
  uniform_band = cmd.get<double>("uniform_band", uniform_band);

  PetscErrorCode ierr;
#ifdef P4_TO_P8
  ierr = PetscPrintf(mpi.comm(), "Parameters : mu = %g, rho = %g, grid is %dx%dx%d\n", mu, rho, nx, ny, nz); CHKERRXX(ierr);
#else
  ierr = PetscPrintf(mpi.comm(), "Parameters : Re = %g, mu = %g, rho = %g, grid is %dx%d\n", Re, mu, rho, nx, ny); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(mpi.comm(), "cfl = %g, uniform_band = %g\n", cfl, uniform_band);

  parStopWatch watch;
  watch.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

#ifdef P4_TO_P8
  int n_xyz [] = {nx, ny, nz};
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
  int periodic[] = {0, 0, 0};
#else
  int n_xyz [] = {nx, ny};
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
  int periodic[] = {0, 0};
#endif
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t *p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);

  p4est_nm1->user_pointer = (void*)&data;

  for(int l=0; l<lmax; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, &brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);

  if(refine_with_smoke)
  {
    splitting_criteria_thresh_t crit_thresh(lmin, lmax, &init_smoke, smoke_thresh);
    p4est_n->user_pointer = (void*)&crit_thresh;
    my_p4est_refine(p4est_n, P4EST_TRUE, refine_levelset_thresh, NULL);
    p4est_balance(p4est_n, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
    p4est_n->user_pointer = (void*)&data;
  }

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, &brick, ngbd_c);

  Vec phi;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, level_set, phi);

  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_1st_order_time_2nd_order_space(phi);
  lsn.perturb_level_set_function(phi, EPS);

#ifdef P4_TO_P8
  CF_3 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1 };
  CF_3 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn   };
#else
  CF_2 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1 };
  CF_2 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn   };
#endif

#ifdef P4_TO_P8
  BoundaryConditions3D bc_v[P4EST_DIM];
  BoundaryConditions3D bc_p;
#else
  BoundaryConditions2D bc_v[P4EST_DIM];
  BoundaryConditions2D bc_p;
#endif

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

  external_force_u_t *external_force_u=NULL;
  external_force_v_t *external_force_v=NULL;
#ifdef P4_TO_P8
  external_force_w_t *external_force_w=NULL;
#endif


  my_p4est_navier_stokes_t ns(ngbd_nm1, ngbd_n, faces_n);
  ns.set_phi(phi);
  ns.set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, cfl);
  dt = MIN(dxmin*cfl/u0, duration);
  if(save_vtk)
    dt = MIN(dt, vtk_dt);
  ns.set_dt(dt);
  ns.set_velocities(vnm1, vn);
  ns.set_bc(bc_v, &bc_p);

  if(is_smoke)
  {
    Vec smoke;
    ierr = VecDuplicate(phi, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    ns.set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }

  double tstart = 0.0;

  double forces[P4EST_DIM];
#if defined(POD_CLUSTER)
  string export_dir         = cmd.get<string>("export_folder", "/home/regan/flow_past_sphere");
#elif defined(STAMPEDE)
  string export_dir         = cmd.get<string>("export_folder", "/work/04965/tg842642/stampede2/flow_past_sphere");
#elif defined(LAPTOP)
  string export_dir         = cmd.get<string>("export_folder", "/home/raphael/workspace/projects/flow_past_sphere");
#else
  string export_dir         = cmd.get<string>("export_folder", "/home/regan/workspace/projects/flow_past_sphere");
#endif

  char out_dir[PATH_MAX];
  sprintf(out_dir, "%s/%dD/Re_%.2f/lmin_%d_lmax_%d", export_dir.c_str(), P4EST_DIM, Re, lmin, lmax);

  char vtk_path[PATH_MAX], vtk_name[PATH_MAX];
  sprintf(vtk_path, "%s/vtu/opt_2_reused_solvers_and_pchypre", out_dir);

  if(save_vtk || save_forces || save_state)
  {
    if(create_directory(out_dir, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_flow_past_sphere_3d: could not create exportation directory %s", out_dir);
#else
      sprintf(error_msg, "main_flow_past_sphere_2d: could not create exportation directory %s", out_dir);
#endif
      throw std::runtime_error(error_msg);
    }
    if(save_vtk && create_directory(vtk_path, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_flow_past_sphere_3d: could not create exportation directory for vtk files %s", vtk_path);
#else
      sprintf(error_msg, "main_flow_past_sphere_2d: could not create exportation directory for vtk files %s", vtk_path);
#endif
    }
  }



  char file_forces[PATH_MAX];
  if(save_forces)
    initialize_force_output(file_forces, out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart);

  my_p4est_poisson_cells_t* cell_solver = NULL;
  my_p4est_poisson_faces_t* face_solver = NULL;

  tn = tstart;
  int iter = 0;
  int export_vtk = -1;
  int save_data_idx = (int) floor(tstart/dt_save_data); // so that we don't save the very first one which was either already read from file, or the known initial condition...
  while(tn+0.01*dt<tstart+duration)
  {
    if(iter>0)
    {
      if(use_adapted_dt)
        ns.compute_adapted_dt();
      else
        ns.compute_dt();
      dt = ns.get_dt();

      if(tn+dt>tstart+duration)
      {
        dt = tstart+duration-tn;
        ns.set_dt(dt);
      }
      if(save_vtk && dt > vtk_dt)
      {
        dt = vtk_dt; // so that we don't miss snapshots...
        ns.set_dt(dt);
      }

      bool solvers_can_be_reused = ns.update_from_tn_to_tnp1(NULL, false, false);
      if(cell_solver!=NULL && !solvers_can_be_reused){
        delete  cell_solver; cell_solver = NULL; }
      if(face_solver!=NULL && !solvers_can_be_reused){
        delete  face_solver; face_solver = NULL; }
    }

    if(save_state && ((int) floor(tn/dt_save_data)) != save_data_idx)
    {
      save_data_idx = ((int) floor(tn/dt_save_data));
      ns.save_state(out_dir, tn, n_states);
    }

    if(external_force_u!=NULL) delete external_force_u;
    external_force_u = new external_force_u_t;

    if(external_force_v!=NULL) delete external_force_v;
    external_force_v = new external_force_v_t;


#ifdef P4_TO_P8
    if(external_force_w!=NULL) delete external_force_w;
    external_force_w = new external_force_w_t;
#endif


#ifdef P4_TO_P8
    CF_3 *external_forces[P4EST_DIM] = { external_force_u, external_force_v, external_force_w };
#else
    CF_2 *external_forces[P4EST_DIM] = { external_force_u, external_force_v };
#endif
    ns.set_external_forces(external_forces);

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, ns.get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double err_hodge = 1;
    int iter_hodge = 0;
    while(iter_hodge<10 && err_hodge>1e-3)
    {
      hodge_new = ns.get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      ns.solve_viscosity(face_solver, (face_solver!=NULL), KSPBCGS, PCHYPRE);
      ns.solve_projection(cell_solver, (cell_solver!=NULL), KSPCG, PCHYPRE);

      hodge_new = ns.get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      err_hodge = 0;
      p4est_t *p4est = ns.get_p4est();
      my_p4est_interpolation_nodes_t *interp_phi = ns.get_interp_phi();
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
          double xyz[P4EST_DIM];
          quad_xyz_fr_q(quad_idx, tree_idx, p4est, ns.get_ghost(), xyz);
#ifdef P4_TO_P8
          if((*interp_phi)(xyz[0],xyz[1],xyz[2])<0)
#else
          if((*interp_phi)(xyz[0],xyz[1])<0)
#endif
            err_hodge = max(err_hodge, fabs(ho[quad_idx]-hn[quad_idx]));
        }
      }
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      ierr = VecRestoreArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(hodge_new, &hn); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi.comm(), "hodge iteration #%d, error = %e\n", iter_hodge, err_hodge); CHKERRXX(ierr);
      iter_hodge++;
    }
    ierr = VecDestroy(hodge_old); CHKERRXX(ierr);
    ns.compute_velocity_at_nodes();
    ns.compute_pressure();

    tn += dt;

    if(save_forces)
    {
      ns.compute_forces(forces);
      FILE* fp_forces = fopen(file_forces, "a");
      if(fp_forces==NULL)
#ifdef P4_TO_P8
        throw std::invalid_argument("main_flow_past_sphere_3d: could not open file for forces output.");
      fprintf(fp_forces, "%g %g %g %g\n", tn, forces[0]/(.5*PI*r0*r0*u0*u0*rho), forces[1]/(.5*PI*r0*r0*u0*u0*rho), forces[2]/(.5*PI*r0*r0*u0*u0*rho));
#else
        throw std::invalid_argument("main_flow_past_sphere_2d: could not open file for forces output.");
      fprintf(fp_forces, "%g %g %g\n", tn, forces[0]/r0/u0/u0/rho, forces[1]/r0/u0/u0/rho);
#endif
      fclose(fp_forces);
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*(tn-tstart)/duration, ns.get_max_L2_norm_u(), ns.get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(ns.get_max_L2_norm_u()>200.0)
    {
      if(save_vtk)
      {
        sprintf(vtk_name, "%s/snapshot_%d", vtk_path, export_vtk+1);
        ns.save_vtk(vtk_name);
      }
      std::cerr << "The simulation blew up..." << std::endl;
      break;
    }

    if(save_vtk && ((int) floor(tn/vtk_dt)) != export_vtk)
    {
      export_vtk = ((int) floor(tn/vtk_dt));
      sprintf(vtk_name, "%s/snapshot_%d", vtk_path, export_vtk);
      ns.save_vtk(vtk_name);
    }

    iter++;
  }


  if(external_force_u!=NULL) delete external_force_u;
  if(external_force_v!=NULL) delete external_force_v;
#ifdef P4_TO_P8
  if(external_force_w!=NULL) delete external_force_w;
#endif

  watch.stop();
  watch.print_duration();

  return 0;
}
