
/*
 * The navier stokes solver applied for super-hydrophobic surfaces simulations
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

class detect_ridge
{
private:
  const double length;
#ifdef P4_TO_P8
  const double width;
  const bool streamwise;
#endif
  const double pitch;
  const double gas_frac;
  const double offset;
  double my_fmod(const double num, const double denom) const { return (num - floor(num/denom)*denom);}
public:
#ifdef P4_TO_P8
  detect_ridge(double len_, double width_, bool streamwise_, double pitch_, double gas_fraction_, const my_p4est_brick_t& brick_, int max_lvl)
    : length(len_), width(width_), streamwise(streamwise_), pitch(pitch_), gas_frac(gas_fraction_),
      offset(streamwise? (0.1*(brick_.xyz_max[2]-brick_.xyz_min[2])/((double) (brick_.nxyztrees[2]*(1<<max_lvl)))) : (0.5*(brick_.xyz_max[0]-brick_.xyz_min[0])/((double) (brick_.nxyztrees[0]*(1<<max_lvl))))){ }
  double normalized_z(const double& z) const
  {
    return my_fmod((z + 0.5*width), pitch);
  }
  bool operator () (double x, double, double z) const {
    if(streamwise)
      return ((offset >= normalized_z(z)) || (normalized_z(z) >= pitch*gas_frac - offset));
    else
      return (my_fmod((x + 0.5*length - offset), pitch)/pitch >= gas_frac);
  }
  double distance_to_ridge(double x, double y, double z) const
  {
    if(streamwise)
      return sqrt(SQR(MIN(1.0-y, y+1.0)) + ((this->operator()(x, y, z))? 0.0: SQR(MIN(normalized_z(z)-offset, pitch*gas_frac - offset - normalized_z(z)))));
    else
      return sqrt(SQR(MIN(1.0-y, y+1.0)) + ((this->operator()(x, y, z))? 0.0: SQR(MIN(my_fmod((x + 0.5*length - offset), pitch), gas_frac*pitch-my_fmod((x + 0.5*length - offset), pitch)))));
  }
#else
  detect_ridge(double len_, double pitch_, double gas_fraction_, const my_p4est_brick_t& brick_, int max_lvl)
    : length(len_), pitch(pitch_), gas_frac(gas_fraction_),
      offset(0.5*(brick_.xyz_max[0]-brick_.xyz_min[0])/((double) (brick_.nxyztrees[0]*(1<<max_lvl)))){ }
  bool operator () (double x, double) const {
    return (my_fmod((x + 0.5*length - offset), pitch)/pitch >= gas_frac);
  }
  double distance_to_ridge(double x, double y) const
  {
    return sqrt(SQR(MIN(1.0-y, y+1.0)) + ((this->operator()(x, y))? 0.0: SQR(MIN(my_fmod((x + 0.5*length - offset), pitch), gas_frac*pitch-my_fmod((x + 0.5*length - offset), pitch)))));
  }
#endif
};

#ifdef P4_TO_P8
class LEVEL_SET : public CF_3 {
#else
class LEVEL_SET : public CF_2 {
#endif
  int max_lvl;
  const detect_ridge* ridge_detector;
public:
  LEVEL_SET(int max_lvl_, detect_ridge* ridge_detector_) : max_lvl(max_lvl_), ridge_detector(ridge_detector_) { lip = 1.2; }
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
  {
    return -ridge_detector->distance_to_ridge(x, y, z)-pow(2.0, -max_lvl);
  }
#else
  double operator()(double x, double y) const
  {
    return -ridge_detector->distance_to_ridge(x, y)-pow(2.0, -max_lvl);
  }
#endif
};

#ifdef P4_TO_P8
struct BCWALLTYPE_P : WallBC3D {
  BoundaryConditionType operator()(double, double, double) const
#else
struct BCWALLTYPE_P : WallBC2D {
  BoundaryConditionType operator()(double, double) const
#endif
  {
    return NEUMANN;
  }
} bc_wall_type_p;

#ifdef P4_TO_P8
struct BCWALLVALUE_P : CF_3 {
  double operator()(double, double, double) const
#else
struct BCWALLVALUE_P : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} bc_wall_value_p;

#ifdef P4_TO_P8
class BCWALLTYPE_U : public WallBC3D {
#else
class BCWALLTYPE_U : public WallBC2D {
#endif
private:
  const detect_ridge* ridge_detector;
public:
  BCWALLTYPE_U(detect_ridge* ridge_detector_): ridge_detector(ridge_detector_){ }
#ifdef P4_TO_P8
  BoundaryConditionType operator()(double x, double y, double z) const
  {
    return (((*ridge_detector)(x, y, z))? DIRICHLET: NEUMANN);
  }
#else
  BoundaryConditionType operator()(double x, double y) const
  {
    return (((*ridge_detector)(x, y))? DIRICHLET: NEUMANN);
  }
#endif
};

#ifdef P4_TO_P8
struct BCWALLVALUE_U : CF_3 {
  double operator()(double, double, double) const
#else
struct BCWALLVALUE_U : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0; // always 0 value whether no slip or free slip
  }
} bc_wall_value_u;

#ifdef P4_TO_P8
struct BCWALLTYPE_V : WallBC3D {
  BoundaryConditionType operator()(double, double, double) const
#else
struct BCWALLTYPE_V : WallBC2D {
  BoundaryConditionType operator()(double, double) const
#endif
  {
    return DIRICHLET; // always homogeneous dirichlet : no penetration through the channel wall
  }
} bc_wall_type_v;

#ifdef P4_TO_P8
struct BCWALLVALUE_V : CF_3 {
  double operator()(double, double, double) const
#else
struct BCWALLVALUE_V : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0; // always homogeneous dirichlet : no penetration through the channel wall
  }
} bc_wall_value_v;

#ifdef P4_TO_P8
struct initial_velocity_unm1_t : CF_3 {
  double operator()(double, double, double) const
#else
struct initial_velocity_unm1_t : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_unm1;

#ifdef P4_TO_P8
struct initial_velocity_un_t : CF_3 {
  double operator()(double, double, double) const
#else
struct initial_velocity_un_t : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_un;

#ifdef P4_TO_P8
struct initial_velocity_vnm1_t : CF_3 {
  double operator()(double, double, double) const
#else
struct initial_velocity_vnm1_t : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_vnm1;

#ifdef P4_TO_P8
struct initial_velocity_vn_t : CF_3 {
  double operator()(double, double, double) const
#else
struct initial_velocity_vn_t : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_vn;

#ifdef P4_TO_P8
class external_force_u_t : public CF_3 {
#else
class external_force_u_t : public CF_2 {
#endif
private:
  double forcing_term;
public:
  external_force_u_t(const double& forcing_term_):forcing_term(forcing_term_) {}
  external_force_u_t(): external_force_u_t(1.0) {}
#ifdef P4_TO_P8
  double operator()(double, double, double) const
#else
  double operator()(double, double) const
#endif
  {
    return forcing_term;
  }
  void update_term(const double& correction)
  {
    forcing_term += correction;
  }
  double get_value() const {return forcing_term;}
};

#ifdef P4_TO_P8
struct external_force_v_t : CF_3 {
  double operator()(double, double, double) const
#else
struct external_force_v_t : CF_2 {
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
};

#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D
{
private:
  const detect_ridge* ridge_detector;
public:
  BCWALLTYPE_W(detect_ridge* ridge_detector_): ridge_detector(ridge_detector_){ }
  BoundaryConditionType operator()(double x, double y, double z) const
  {
    return (((*ridge_detector)(x, y, z))? DIRICHLET: NEUMANN);
  }
};

struct BCWALLVALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} bc_wall_value_w;

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

struct external_force_w_t : CF_3
{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
};
#endif

void initialize_monitoring(char* file_monitoring, const char *out_dir, const int& ntree_y, const int& lmin, const int& lmax, const double& threshold_split_cell, const double& cfl, const int& sl_order, const mpi_environment_t& mpi, const double& tstart)
{
  sprintf(file_monitoring, "%s/flow_monitoring_ny_%d_lmin_%d_lmax_%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat", out_dir, ntree_y, lmin, lmax, threshold_split_cell, cfl, sl_order);

  PetscErrorCode ierr = PetscPrintf(mpi.comm(), "Monitoring flow in ... %s\n", file_monitoring); CHKERRXX(ierr);
  if(mpi.rank() == 0)
  {
    if(!file_exists(file_monitoring))
    {
      FILE* fp_monitor = fopen(file_monitoring, "w");
      if(fp_monitor==NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for flow monitoring.");
      fprintf(fp_monitor, "%% tn | Re_tau | Re_b \n");
      fclose(fp_monitor);
    }
    else
    {
      FILE* fp_monitor = fopen(file_monitoring, "r+");
      char* read_line = NULL;
      size_t len = 0;
      ssize_t len_read;
      long size_to_keep = 0;
      if(((len_read = getline(&read_line, &len, fp_monitor)) != -1))
        size_to_keep += (long) len_read;
      else
        throw std::runtime_error("initialize_monitoring: couldn't read the first header line of mass_flow_...dat");
      double time, time_nm1;
      double dt = 0.0;
      bool not_first_line = false;
      char format_specifier[1024] = "%lg %*g %*g";
      while ((len_read = getline(&read_line, &len, fp_monitor)) != -1) {
        if(not_first_line)
          time_nm1 = time;
        sscanf(read_line, format_specifier, &time);
        if(not_first_line)
          dt = time - time_nm1;
        if(time <= tstart+0.1*dt) // +0.1*dt to avoid roundoff errors when exporting the data
          size_to_keep += (long) len_read;
        else
          break;
        not_first_line=true;
      }
      fclose(fp_monitor);
      if(read_line)
        free(read_line);
      if(truncate(file_monitoring, size_to_keep))
        throw std::runtime_error("initialize_monitoring: couldn't truncate flow_monitoring_...dat");
    }

    char liveplot_Re[PATH_MAX];
    sprintf(liveplot_Re, "%s/live_monitor.gnu", out_dir);
    if(!file_exists(liveplot_Re))
    {
      FILE *fp_liveplot_Re= fopen(liveplot_Re, "w");
      if(fp_liveplot_Re==NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for Re liveplot.");
      fprintf(fp_liveplot_Re, "set term wxt noraise\n");
      fprintf(fp_liveplot_Re, "set key bottom right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_Re, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_Re, "set ylabel \"Re\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_Re, "plot");
      fprintf(fp_liveplot_Re, "\t \"flow_monitoring_ny_%d_lmin_%d_lmax_%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:2 title 'Re_{tau}' with lines lw 3,\\\n", ntree_y, lmin, lmax, threshold_split_cell, cfl, sl_order);
      fprintf(fp_liveplot_Re, "\t \"flow_monitoring_ny_%d_lmin_%d_lmax_%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:3 title 'Re_b' with lines lw 3\n", ntree_y, lmin, lmax, threshold_split_cell, cfl, sl_order);
      fprintf(fp_liveplot_Re, "pause 4\n");
      fprintf(fp_liveplot_Re, "reread");
      fclose(fp_liveplot_Re);
    }

    char tex_plot_Re[PATH_MAX];
    sprintf(tex_plot_Re, "%s/tex_monitor.gnu", out_dir);
    if(!file_exists(tex_plot_Re))
    {
      FILE *fp_tex_plot_Re = fopen(tex_plot_Re, "w");
      if(fp_tex_plot_Re==NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for Re monitoring figure.");
      fprintf(fp_tex_plot_Re, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_Re, "set output 'monitor_history.tex'\n");
      fprintf(fp_tex_plot_Re, "set key bottom right Left \n");
      fprintf(fp_tex_plot_Re, "set xlabel \"$t$\"\n");
      fprintf(fp_tex_plot_Re, "set ylabel \"$\\\\mathrm{Re}$\" \n");
      fprintf(fp_tex_plot_Re, "plot");
      fprintf(fp_tex_plot_Re, "\t \"flow_monitoring_ny_%d_lmin_%d_lmax_%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:2 title '$Re_{tau}$' with lines lw 3,\\\n", ntree_y, lmin, lmax, threshold_split_cell, cfl, sl_order);
      fprintf(fp_tex_plot_Re, "\t \"flow_monitoring_ny_%d_lmin_%d_lmax_%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:3 title '$Re_{b}$' with lines lw 3", ntree_y, lmin, lmax, threshold_split_cell, cfl, sl_order);
      fclose(fp_tex_plot_Re);
    }

    char tex_Re_script[PATH_MAX];
    sprintf(tex_Re_script, "%s/plot_tex_monitor.sh", out_dir);
    if(!file_exists(tex_Re_script))
    {
      FILE *fp_tex_monitor_script = fopen(tex_Re_script, "w");
      if(fp_tex_monitor_script==NULL)
        throw std::runtime_error("initialize_monitoring: could not open file for bash script plotting monitoring tex figure.");
      fprintf(fp_tex_monitor_script, "#!/bin/sh\n");
      fprintf(fp_tex_monitor_script, "gnuplot ./tex_monitor.gnu\n");
      fprintf(fp_tex_monitor_script, "latex ./monitor_history.tex\n");
      fprintf(fp_tex_monitor_script, "dvipdf -dAutoRotatePages=/None ./monitor_history.dvi\n");
      fclose(fp_tex_monitor_script);

      ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_Re_script;
      if(system(chmod_command.str().c_str()))
        throw std::runtime_error("initialize_monitoring: could not make the plot_tex_monitor.sh script executable");
    }
  }
}

void initialize_drag_force_output(char* file_drag, const char *out_dir, const int& lmin, const int& lmax, const double& threshold_split_cell, const double& cfl, const int& sl_order, const mpi_environment_t& mpi, const double& tstart)
{
  sprintf(file_drag, "%s/drag_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat", out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order);
  PetscErrorCode ierr = PetscPrintf(mpi.comm(), "Saving drag in ... %s\n", file_drag); CHKERRXX(ierr);

  if(mpi.rank() == 0)
  {
    if(!file_exists(file_drag))
    {
      FILE* fp_drag = fopen(file_drag, "w");
      if(fp_drag==NULL)
        throw std::runtime_error("initialize_drag_force_output: could not open file for drag output.");
      fprintf(fp_drag, "%% __ | Normalized drag \n");
#ifdef P4_TO_P8
      fprintf(fp_drag, "%% tn | x-component | y-component | z-component\n");
#else
      fprintf(fp_drag, "%% tn | x-component | y-component\n");
#endif
      fclose(fp_drag);
    }
    else
    {
      FILE* fp_drag = fopen(file_drag, "r+");
      char* read_line = NULL;
      size_t len = 0;
      ssize_t len_read;
      long size_to_keep = 0;
      if(((len_read = getline(&read_line, &len, fp_drag)) != -1))
        size_to_keep += (long) len_read;
      else
        throw std::runtime_error("initialize_drag_force_output: couldn't read the first header line of drag_...dat");
      if(((len_read = getline(&read_line, &len, fp_drag)) != -1))
        size_to_keep += (long) len_read;
      else
        throw std::runtime_error("initialize_drag_force_output: couldn't read the second header line of drag_...dat");
      double time, time_nm1;
      double dt = 0.0;
      bool not_first_line = false;
      while ((len_read = getline(&read_line, &len, fp_drag)) != -1) {
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
      fclose(fp_drag);
      if(read_line)
        free(read_line);
      if(truncate(file_drag, size_to_keep))
        throw std::runtime_error("initialize_drag_force_output: couldn't truncate drag_...dat");
    }

    char liveplot_drag[PATH_MAX];
    sprintf(liveplot_drag, "%s/live_drag.gnu", out_dir);
    if(!file_exists(liveplot_drag))
    {
      FILE* fp_liveplot_drag = fopen(liveplot_drag, "w");
      if(fp_liveplot_drag==NULL)
        throw std::runtime_error("initialize_drag_force_output: could not open file for drage force liveplot.");
      fprintf(fp_liveplot_drag, "set term wxt noraise\n");
      fprintf(fp_liveplot_drag, "set key center right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_drag, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_drag, "set ylabel \"Nondimensional drag \" font \"Arial,14\"\n");
      fprintf(fp_liveplot_drag, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_liveplot_drag, "\t \"drag_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:%d title '%c-component' with lines lw 3", lmin, lmax, threshold_split_cell, cfl, sl_order, ((int)dd+2), ((dd==0)?'x':((dd==1)?'y':'z')));
        if(dd < P4EST_DIM-1)
          fprintf(fp_liveplot_drag, ",\\");
        fprintf(fp_liveplot_drag, "\n");
      }
      fprintf(fp_liveplot_drag, "pause 4\n");
      fprintf(fp_liveplot_drag, "reread");
      fclose(fp_liveplot_drag);
    }

    char tex_plot_drag[PATH_MAX];
    sprintf(tex_plot_drag, "%s/tex_drag.gnu", out_dir);
    if(!file_exists(tex_plot_drag))
    {
      FILE *fp_tex_plot_drag = fopen(tex_plot_drag, "w");
      if(fp_tex_plot_drag==NULL)
        throw std::runtime_error("initialize_drag_foce_output: could not open file for drag force tex figure.");
      fprintf(fp_tex_plot_drag, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_drag, "set output 'drag_history.tex'\n");
      fprintf(fp_tex_plot_drag, "set key center right Left \n");
      fprintf(fp_tex_plot_drag, "set xlabel \"$t$\"\n");
#ifdef P4_TO_P8
      fprintf(fp_tex_plot_drag, "set ylabel \"Non-dimensional wall friction $\\\\mathbf{D} = \\\\frac{\\\\hat{\\\\mathbf{D}}}{\\\\rho u_{\\\\tau}^{2} \\\\delta^{2}}$ \" \n");
#else
      fprintf(fp_tex_plot_drag, "set ylabel \"Non-dimensional wall friction $\\\\mathbf{D} = \\\\frac{\\\\hat{\\\\mathbf{D}}}{\\\\rho u_{\\\\tau}^{2} \\\\delta}$ \" \n");
#endif
      fprintf(fp_tex_plot_drag, "plot");
      for (short dd = 0; dd < P4EST_DIM; ++dd)
      {
        fprintf(fp_tex_plot_drag, "\t \"drag_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:%d title '$D_{%c}$' with lines lw 3", lmin, lmax, threshold_split_cell, cfl, sl_order, ((int) dd+2), ((dd==0)?'x':((dd==1)?'y':'z')));
        if(dd < P4EST_DIM-1)
          fprintf(fp_tex_plot_drag, ",\\\n");
      }
      fclose(fp_tex_plot_drag);
    }

    char tex_drag_script[PATH_MAX];
    sprintf(tex_drag_script, "%s/plot_tex_drag.sh", out_dir);
    if(!file_exists(tex_drag_script))
    {
      FILE *fp_tex_drag_script = fopen(tex_drag_script, "w");
      if(fp_tex_drag_script==NULL)
        throw std::runtime_error("initialize_drag_force_output: could not open file for bash script plotting drag tex figure.");
      fprintf(fp_tex_drag_script, "#!/bin/sh\n");
      fprintf(fp_tex_drag_script, "gnuplot ./tex_drag.gnu\n");
      fprintf(fp_tex_drag_script, "latex ./drag_history.tex\n");
      fprintf(fp_tex_drag_script, "dvipdf -dAutoRotatePages=/None ./drag_history.dvi\n");
      fclose(fp_tex_drag_script);

      ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_drag_script;
      int sys_return = system(chmod_command.str().c_str()); (void) sys_return;
    }
  }
}

void initialize_velocity_profile_file(const char* filename, const int& ntree_y, const int &lmax, const double& tstart,  const mpi_environment_t& mpi)
{
  if(mpi.rank() == 0)
  {
    if(!file_exists(filename))
    {
      FILE* fp_avg_profile = fopen(filename, "w");
      if(fp_avg_profile==NULL)
      {
        char error_msg[1024];
        sprintf(error_msg, "initialize_velocity_profile_file: could not open file %s.", filename);
        throw std::invalid_argument(error_msg);
      }
      fprintf(fp_avg_profile, "%% __ | coordinates along y axis \n");
      fprintf(fp_avg_profile, "%% tn");
      for (int k = 0; k < ntree_y*(1<<lmax); ++k)
        fprintf(fp_avg_profile, "  | %.12g", (-1.00+(2.0/((double) (ntree_y*(1<<lmax))))*(0.5+k)));
      fprintf(fp_avg_profile, "\n");
      fprintf(fp_avg_profile, "%.12g", tstart);
      fclose(fp_avg_profile);
    }
    else
    {
      FILE* fp_avg_profile = fopen(filename, "r+");
      char* read_line = NULL;
      size_t len = 0;
      ssize_t len_read;
      long size_to_keep = 0;
      if(((len_read = getline(&read_line, &len, fp_avg_profile)) != -1))
        size_to_keep += (long) len_read;
      else
      {
        char error_msg[1024];
        sprintf(error_msg, "initialize_velocity_profile_file: couldn't read the first header line of %s", filename);
        throw std::runtime_error(error_msg);
      }
      if(((len_read = getline(&read_line, &len, fp_avg_profile)) != -1))
        size_to_keep += (long) len_read;
      else
      {
        char error_msg[1024];
        sprintf(error_msg, "initialize_velocity_profile_file: couldn't read the second header line of %s", filename);
        throw std::runtime_error(error_msg);
      }
      double time, time_nm1;
      double dt = 0.0;
      bool not_first_line = false;
      while ((len_read = getline(&read_line, &len, fp_avg_profile)) != -1) {
        if(not_first_line)
          time_nm1 = time;
        sscanf(read_line, "%lg %*[^\n]", &time);
        if(not_first_line)
          dt = time-time_nm1;
        if(time <= tstart+0.1*dt) // +0.1*dt to avoid roundoff errors when exporting the data
          size_to_keep += (long) len_read;
        else
          break;
        not_first_line=true;
      }
      fclose(fp_avg_profile);
      if(read_line)
        free(read_line);
      if(truncate(filename, size_to_keep))
      {
        char error_msg[1024];
        sprintf(error_msg, "initialize_velocity_profile_file: couldn't truncate %s", filename);
        throw std::runtime_error(error_msg);
      }

      fp_avg_profile = fopen(filename, "a");
      fprintf(fp_avg_profile, "%.12g", tstart);
      fclose(fp_avg_profile);
    }
  }
}

void initialize_averaged_velocity_profiles(char* file_slice_avg_profile, const char *profile_dir, const int* ntree, const int& lmin, const int& lmax, const double& threshold_split_cell, const double& cfl, const int& sl_order, const mpi_environment_t& mpi, const double& tstart,
                                           vector<unsigned int>& bin_index, vector<string>& file_line_avg_profile, const double* dimensions_to_delta, const double& pitch_to_delta, const double& gas_fraction, unsigned int& number_of_bins, const bool& spanwise)
{
  PetscErrorCode ierr;
  sprintf(file_slice_avg_profile, "%s/slice_averaged_velocity_profile_ntreey_%d_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat", profile_dir, ntree[1], lmin, lmax, threshold_split_cell, cfl, sl_order);
  ierr = PetscPrintf(mpi.comm(), "Saving slice-averaged velocity profile in %s\n", file_slice_avg_profile); CHKERRXX(ierr);
  initialize_velocity_profile_file(file_slice_avg_profile, ntree[1], lmax, tstart, mpi);
  unsigned int nb_cells_in_groove = (unsigned int)((pitch_to_delta*gas_fraction)/((spanwise)? (dimensions_to_delta[0]/(ntree[0]*(1<<lmax))) : (dimensions_to_delta[2]/(ntree[2]*(1<<lmax)))));
  unsigned int nb_cells_in_ridge  = (unsigned int)((pitch_to_delta*(1.0-gas_fraction))/((spanwise)? (dimensions_to_delta[0]/(ntree[0]*(1<<lmax))) : (dimensions_to_delta[2]/(ntree[2]*(1<<lmax)))));
  P4EST_ASSERT(nb_cells_in_groove+nb_cells_in_ridge==(unsigned int)(pitch_to_delta/((spanwise)? (dimensions_to_delta[0]/(ntree[0]*(1<<lmax))) : (dimensions_to_delta[2]/(ntree[2]*(1<<lmax))))));
  bin_index.resize(nb_cells_in_groove+nb_cells_in_ridge);
  unsigned int nbins_in_groove = (nb_cells_in_groove+((nb_cells_in_groove%2==1)? 1:0))/2;
  if(nb_cells_in_groove%2 == 0)
  {
    bin_index[(spanwise?1:0)+nbins_in_groove-1] = 0;
    bin_index[(spanwise?1:0)+nbins_in_groove]   = 0;
  }
  else
    bin_index[(spanwise?1:0)+nbins_in_groove-1] = 0;
  for (unsigned int bin_idx = 1; bin_idx < nbins_in_groove; ++bin_idx)
  {
    if(nb_cells_in_groove%2 == 0)
    {
      bin_index[(spanwise?1:0)+nbins_in_groove-1-bin_idx] = bin_idx;
      bin_index[(spanwise?1:0)+nbins_in_groove+bin_idx]   = bin_idx;
    }
    else
    {
      bin_index[(spanwise?1:0)+nbins_in_groove-1-bin_idx] = bin_idx;
      bin_index[(spanwise?1:0)+nbins_in_groove-1+bin_idx] = bin_idx;
    }
  }
  unsigned int nbins_in_ridge = (nb_cells_in_ridge+((nb_cells_in_ridge%2==1)? 1:0))/2;
  if(nb_cells_in_ridge%2 == 0)
  {
    bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge-1)%bin_index.size()] = nbins_in_groove+nbins_in_ridge-1;
    bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge)%bin_index.size()]   = nbins_in_groove+nbins_in_ridge-1;
  }
  else
    bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge-1)%bin_index.size()] = nbins_in_groove+nbins_in_ridge-1;
  for (unsigned int bin_idx = 1; bin_idx < nbins_in_ridge; ++bin_idx)
  {
    if(nb_cells_in_ridge%2 == 0)
    {
      bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge-1-bin_idx)%bin_index.size()] = nbins_in_groove + nbins_in_ridge-1-bin_idx;
      bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge+bin_idx)%bin_index.size()]   = nbins_in_groove + nbins_in_ridge-1-bin_idx;
    }
    else
    {
      bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge-1-bin_idx)%bin_index.size()] = nbins_in_groove+nbins_in_ridge-1-bin_idx;
      bin_index[((spanwise?1:0)+nb_cells_in_groove+nbins_in_ridge-1+bin_idx)%bin_index.size()] = nbins_in_groove+nbins_in_ridge-1-bin_idx;
    }
  }
  number_of_bins = nbins_in_groove + nbins_in_ridge;
  file_line_avg_profile.resize(number_of_bins);
  for (unsigned int bin_idx = 0; bin_idx < number_of_bins; ++bin_idx) {
    char filename[PATH_MAX];
    sprintf(filename, "%s/line_averaged_velocity_profile_ntreey_%d_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d_index_%d.dat", profile_dir, ntree[1], lmin, lmax, threshold_split_cell, cfl, sl_order, bin_idx);
    file_line_avg_profile[bin_idx] = filename;
    ierr = PetscPrintf(mpi.comm(), "Saving line-averaged velocity profile in %s\n", file_line_avg_profile[bin_idx].c_str()); CHKERRXX(ierr);
    initialize_velocity_profile_file(file_line_avg_profile[bin_idx].c_str(), ntree[1], lmax, tstart, mpi);
  }
}

void check_pitch_and_gas_fraction(double length_to_delta, int ntree, int lmax, double pitch_to_delta, double gas_fraction)
{
  if(fabs(length_to_delta/pitch_to_delta - ((int) length_to_delta/pitch_to_delta)) > 1e-6)
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: the length of the domain in the direction transversal to the grooves MUST be a multiple of the pitch to satisfy periodicity.");
#else
    throw std::invalid_argument("main_shs_2d.cpp: the length of the domain in the direction transversal to the grooves MUST be a multiple of the pitch to satisfy periodicity.");
#endif

  double nb_finest_cell_in_groove =  pitch_to_delta*gas_fraction/(length_to_delta/((double) (ntree*(1<<lmax))));
  double nb_finest_cell_in_ridge  =  pitch_to_delta*(1.0-gas_fraction)/(length_to_delta/((double) (ntree*(1<<lmax))));

  if((fabs(nb_finest_cell_in_groove - ((int) nb_finest_cell_in_groove)) > 1e-6) || (fabs(nb_finest_cell_in_ridge - ((int) nb_finest_cell_in_ridge)) > 1e-6))
  {
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: the finest grid cells do not capture the groove and/or the ridge (subcell resolution for boundary condition would be required).");
#else
    throw std::invalid_argument("main_shs_2d.cpp: the finest grid cells do not capture the groove and/or the ridge (subcell resolution for boundary condition would be required).");
#endif
  }
}

double Re_tau(const external_force_u_t& force_per_unit_mass_x, const double& rho, const double& mu)
{
  return rho*1.0*sqrt(force_per_unit_mass_x.get_value()*1.0)/mu; // delta = 1.0
}

double Re_b(const double& mass_flow, const double &width, const double& rho, const double& mu)
{
  return rho*mass_flow*1.0/(mu*2.0*width); // delta = 1.0 --> height = 2.0
}

#ifdef P4_TO_P8
void check_accuracy_of_solution(my_p4est_navier_stokes_t* ns, const external_force_u_t& force_per_unit_mass_x, const double& pitch, const double& GF, const bool spanwise, double my_errors[])
#else
void check_accuracy_of_solution(my_p4est_navier_stokes_t* ns, const external_force_u_t& force_per_unit_mass_x, const double& pitch, const double& GF, double my_errors[])
#endif
{
  const double rho        = ns->get_rho();
  const double mu         = ns->get_mu();
  const double length     = ns->get_length_of_domain();
  const double height     = ns->get_height_of_domain();
#ifdef P4_TO_P8
  const double width      = ns->get_width_of_domain();
#endif
  const double abs_dp_dx  = force_per_unit_mass_x.get_value();
  double my_infty_norm_error_u = 0.0;
  double my_infty_norm_error_v = 0.0;
#ifdef P4_TO_P8
  double my_infty_norm_error_w = 0.0;
#endif
  // calculate the face errors here;

  my_errors[0] = my_infty_norm_error_u;
  my_errors[1] = my_infty_norm_error_v;
#ifdef P4_TO_P8
  my_errors[2] = my_infty_norm_error_w;
#endif

  my_infty_norm_error_u = 0.0;
  my_infty_norm_error_v = 0.0;
#ifdef P4_TO_P8
  my_infty_norm_error_w = 0.0;
#endif
  // calculate the node errors here;


  my_errors[P4EST_DIM+0] = my_infty_norm_error_u;
  my_errors[P4EST_DIM+1] = my_infty_norm_error_v;
#ifdef P4_TO_P8
  my_errors[P4EST_DIM+2] = my_infty_norm_error_w;
#endif
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, my_errors, 2*P4EST_DIM, MPI_DOUBLE, MPI_MAX, ns->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
}


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin", "min level of the trees, default is 4");
  cmd.add_option("lmax", "max level of the trees, default is 6");
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is 0.1");
  cmd.add_option("wall_layer", "number of finest cells desired to layer the channel walls, default is 6");
  cmd.add_option("lip", "Lipschitz constant L for grid refinement. The levelset is defined as the negative distance to the closest no-slip region. The criterion compares the levelset value to L\\Delta y. Default value is the standard 1.2 or value read from solver state if restarted (fyi: that's very thin for turbulent cases).");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is length to ensure aspect ratio of cells = 1");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is 2 (i.e. height) to ensure aspect ratio of cells = 1");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is width to ensure aspect ratio of cells = 1");
#endif
  // physical parameters for the simulations
  cmd.add_option("length", "length of the channel (dimension in streamwise, x-direction), in units of delta, default is 6.0");
#ifdef P4_TO_P8
  cmd.add_option("width", "width of the channel (dimension in spanwise, z-direction), in units of delta, default is 3.0");
  cmd.add_option("spanwise", "if present, the grooves and ridges are understood as spanwise, that is perpendicular to the flow. If absent, the grooves are parallel to the flow.");
#endif
  cmd.add_option("duration", "the duration of the simulation (tfinal-tstart). If not restarted, tstart = 0.0, default duration is 200.0.");
  cmd.add_option("Re_tau", "Reynolds number based on the wall-shear velocity and half the channel height, i.e. Re_tau = u_tau*delta/nu where u_tau = sqrt(-dp_dx*delta/rho), default is 180.0");
  cmd.add_option("Re_b", "Reynolds number, based on the mean (bulk) velocity and half the channel height, i.e. Re_b   = U_b*delta/nu, no default value. Can be used exclusively in case of restart and without Re_tau");
  cmd.add_option("pitch_to_delta", "P/delta ratio, default = 0.375");
  cmd.add_option("GF", "gas fraction, default is 0.5");
  cmd.add_option("adapted_dt", "activates the calculation of dt based on the local cell sizes if present");
  // method-related parameters
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is 2");
  cmd.add_option("cfl", "dt = cfl * dx/vmax, default is 1.00");
  cmd.add_option("Ub_tol", "relative numerical tolerance on the bulk velocity to be set (if restart and Re_b set), default is 1e-3");
  cmd.add_option("hodge_tol", "absolute numerical tolerance on the Hodge variable, at all time steps, default is 1e-3");
  cmd.add_option("niter_hodge", "max number of iterations for convergence of the Hodge variable, at all time steps, default is 10");
  cmd.add_option("grid_update", "number of time steps between grid updates, default is 1");
  cmd.add_option("pc_cell", "preconditioner for cell-solver: jacobi, sor or hypre, default is sor.");
  cmd.add_option("cell_solver", "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is bicgstab.");
  cmd.add_option("pc_face", "preconditioner for face-solver: jacobi, sor or hypre, default is sor.");
  // output-control parameters
  cmd.add_option("export_folder", "exportation_folder");
  cmd.add_option("save_vtk", "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt", "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
#ifdef P4_TO_P8
  cmd.add_option("save_drag", "activates exportation of the total drag (non-dimensionalized by rho*SQR(u_tau)*SQR(delta))");
#else
  cmd.add_option("save_drag", "activates exportation of the total drag (non-dimensionalized by rho*SQR(u_tau)*delta)");
#endif
  cmd.add_option("save_state_dt", "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates", "determines how many solver states must be memorized in backup_ folders (default is 1).");
  cmd.add_option("save_mean_profiles", "computes and saves averaged streamwise-velocity profiles (makes sense only if the flow is fully-developed)");
  cmd.add_option("nexport_avg", "number of iterations between two exportation of averaged velocity profiles (default is 100)");
  cmd.add_option("timing", "if defined, prints timing information (typically for scaling analysis).");
  cmd.add_option("accuracy_check", "if defined, prints information about accuracy with comparison to analytical solution (ONLY if restart after steady-state reached and with Re_tau enforced).");


  // --> extra info to be printed when -help is invoked
#ifdef P4_TO_P8
  const std::string extra_info = "\
      This program provides a general setup for superhydrophobic channel flow simulations.\n\
      It assumes no solid object and no passive scalar (i.e. smoke) in the channel. The height of the channel is set to \n\
      2*delta by default, the other channel dimensions are provided by the user in units of delta. If the numbers of \n\
      trees in the streamwise and spanwise directions (resp. input parameters nx and nz) are not provided by the user, \n\
      they are set in order to ensure aspect ratio of computational cells as close as possible to 1, i.e. each tree in \n\
      the forest is of size (as close as possible to) deltaXdeltaXdelta. \n\n\
      The set up builds upon the following non-dimensionalization ('_hat' for dimensional variables): \n\n\
      u = u_hat/u_tau, {x, y, z} = {x, y, z}_hat/delta, t = t_hat*u_tau/delta, p = p_hat/(rho*u_tau*u_tau), \n\n\
      where u_tau is the wall-friction velocity defined as sqrt(-dp_dx*delta/rho). \n\
      Therefore, the computational domain is [-0.5*length, 0.5*length]x[-1, 1]x[-0.5*width, 0.5*width]. \n\
      When started from scratch, the user can set Re_tau ONLY and the Navier-Stokes solver is then invoked with\n\
      nondimensional inputs: \n\
      rho = 1.0, mu = 1.0/Re_tau, body force per unit mass {1.0, 0.0, 0.0} (driving the flow), \n\
      and with periodic boundary conditions in the streamwise and spanwise directions. \n\
      When restarted from a saved state, the user can either\n\
      1) (re)set Re_tau: this resets only the viscosity of the fluid but keeps the body force per unit mass {1.0, 0.0, 0.0}\n\
      2) set Re_b: this leaves the viscosity unchanged (i.e. as read from the saved state) but adapts the body force per unit\n\
         mass dynamically in order to set the mean (bulk) velocity to the desired value that matches the desired bulk Reynolds\n\
      Developer: Raphael Egan (raphaelegan@ucsb.edu)";
#else
  const std::string extra_info = "\
      This program provides a general setup for superhydrophobic channel flow simulations.\n\
      It assumes no solid object and no passive scalar (i.e. smoke) in the channel. The height of the channel is set \n\
      to 2*delta by default, the other channel dimension is provided by the user in units of delta. If the number of \n\
      trees in the streamwise direction (input parameters nx) is not provided by the user, it is set in order to ensure \n\
      aspect ratio of computational cells as close as possible to 1, i.e. each tree in the forest is of size (as close \n\
      as possible to) deltaXdeltaXdelta. \n\n\
      The set up builds upon the following non-dimensionalization ('_hat' for dimensional variables): \n\n\
      u = u_hat/u_tau, {x, y} = {x, y}_hat/delta, t = t_hat*u_tau/delta, p = p_hat/(rho*u_tau*u_tau), \n\n\
      where u_tau is the wall-friction velocity defined as sqrt(-dp_dx*delta/rho). \n\
      Therefore, the computational domain is [-0.5*length, 0.5*length]x[-1, 1]. \n\
      When started from scratch, the user can set Re_tau ONLY and the Navier-Stokes solver is then invoked with\n\
      nondimensional inputs: \n\
      rho = 1.0, mu = 1.0/Re_tau, body force per unit mass {1.0, 0.0} (driving the flow), \n\
      and with periodic boundary conditions in the streamwise and directions. \n\
      When restarted from a saved state, the user can either\n\
      1) (re)set Re_tau: this resets only the viscosity of the fluid but keeps the body force per unit mass {1.0, 0.0}\n\
      2) set Re_b: this leaves the viscosity unchanged (i.e. as read from the saved state) but adapts the body force per unit\n\
          mass dynamically in order to set the mean (bulk) velocity to the desired value that matches the desired bulk Reynolds\n\
      Developer: Raphael Egan (raphaelegan@ucsb.edu)";
#endif
  cmd.parse(argc, argv, extra_info);

  double tstart;
  double dt;
  int lmin, lmax;
  my_p4est_navier_stokes_t* ns                    = NULL;
  my_p4est_brick_t* brick                         = NULL;
  splitting_criteria_cf_and_uniform_band_t* data  = NULL;
  LEVEL_SET* level_set                            = NULL;

  int sl_order;
  unsigned int wall_layer;
  double threshold_split_cell, uniform_band, cfl;
  double length;
  int ntree_x, ntree_y;
#ifdef P4_TO_P8
  double width;
  int ntree_z;
#endif
  int n_xyz [P4EST_DIM];

  const double Ub_tolerance             = cmd.get<double>("Ub_tol", 1e-3);
  const double hodge_tolerance          = cmd.get<double>("hodge_tol", 1e-3);
  const unsigned int niter_hodge_max    = cmd.get<unsigned int>("niter_hodge", 10);
  const unsigned int steps_grid_update  = cmd.get<unsigned int>("grid_update", 1);

  const double duration                 = cmd.get<double>("duration", 200.0);
  const double pitch_to_delta           = cmd.get<double>("pitch_to_delta", 1.0/4.0);
  const double gas_fraction             = cmd.get<double>("GF", 0.5);
#if defined(POD_CLUSTER)
  const string export_dir               = cmd.get<string>("export_folder", "/scratch/regan/superhydrophobic_channel");
#elif defined(STAMPEDE)
  const string export_dir               = cmd.get<string>("export_folder", "/work/04965/tg842642/stampede2/superhydrophobic_channel");
#elif defined(LAPTOP)
  const string export_dir               = cmd.get<string>("export_folder", "/home/raphael/workspace/projects/superhydrophobic_channel");
#else
  const string export_dir               = cmd.get<string>("export_folder", "/home/regan/workspace/projects/superhydrophobic_channel");
#endif
  const bool save_vtk                   = cmd.contains("save_vtk");
  const bool get_timing                 = cmd.contains("timing");
  double vtk_dt                         = -1.0;
  if(save_vtk)
  {
    if(!cmd.contains("vtk_dt"))
#ifdef P4_TO_P8
      throw std::runtime_error("main_shs_3d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
#else
      throw std::runtime_error("main_shs_2d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
#endif
    vtk_dt = cmd.get<double>("vtk_dt", -1.0);
    if(vtk_dt <= 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_shs_3d.cpp: the value of vtk_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_shs_2d.cpp: the value of vtk_dt must be strictly positive.");
#endif
  }
  const bool save_drag                  = cmd.contains("save_drag"); double drag[P4EST_DIM];
  const bool do_accuracy_check          = cmd.contains("accuracy_check") && cmd.contains("restart") && cmd.contains("Re_tau"); double my_accuracy_check_errors[2*P4EST_DIM];
  const bool save_state                 = cmd.contains("save_state_dt"); double dt_save_data = -1.0;
  const unsigned int n_states           = cmd.get<unsigned int>("save_nstates", 1);
  if(save_state)
  {
    dt_save_data                        = cmd.get<double>("save_state_dt", -1.0);
    if(dt_save_data < 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_shs_3d.cpp: the value of save_state_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_shs_2d.cpp: the value of save_state_dt must be strictly positive.");
#endif
  }
  const bool save_profiles              = cmd.contains("save_mean_profiles");
  const unsigned int nexport_avg        = cmd.get<unsigned int>("nexport_avg", 100);
  double                    t_slice_average;
  vector<double>            slice_averaged_profile;
  vector<double>            slice_averaged_profile_nm1;
  vector<double>            time_averaged_slice_averaged_profile;
  vector<double>            t_line_average;
  vector< vector<double> >  line_averaged_profiles;
  vector< vector<double> >  line_averaged_profiles_nm1;
  vector< vector<double> >  time_averaged_line_averaged_profiles;
  const bool use_adapted_dt             = cmd.contains("adapted_dt");
#ifdef P4_TO_P8
  const bool spanwise                   = cmd.contains("spanwise");
#endif

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

  if(cmd.contains("Re_b") && !cmd.contains("restart"))
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: forcing a constant bulk velocity, i.e. a constant mass flow, cannot be done if starting the simulation from scratch (no adequate initial condition): advance the simulation with constant pressure gradient first, fixing Re_tau; then restart it with Re_b.");
#else
    throw std::invalid_argument("main_shs_2d.cpp: forcing a constant bulk velocity, i.e. a constant mass flow, cannot be done if starting the simulation from scratch (no adequate initial condition): advance the simulation with constant pressure gradient first, fixing Re_tau; then restart it with Re_b.");
#endif
  if(cmd.contains("Re_b") && cmd.contains("Re_tau"))
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: forcing a constant bulk velocity AND a constant pressure gradient cannot be done: choose one!");
#else
    throw std::invalid_argument("main_shs_2d.cpp: forcing a constant bulk velocity AND a constant pressure gradient cannot be done: choose one!");
#endif
  if(cmd.contains("restart") && !cmd.contains("Re_b") && !cmd.contains("Re_tau"))
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: you need to specify either the desired Re_tau or the desired Re_b when restarting...");
#else
    throw std::invalid_argument("main_shs_2d.cpp: you need to specify either the desired Re_tau or the desired Re_b when restarting...");
#endif

  PetscErrorCode ierr;
  const double rho = 1.0;
  double mu;
  double bulk_velocity_to_set = 0.0; // irrelevant, except if enforcing mass flow after restart --> reset in that case
  double xyz_min [P4EST_DIM];
  double xyz_max [P4EST_DIM];
#ifdef P4_TO_P8
  const int periodic[] = {1, 0, 1};
#else
  const int periodic[] = {1, 0};
#endif

#ifdef P4_TO_P8
  BoundaryConditions3D bc_v[P4EST_DIM];
  BoundaryConditions3D bc_p;
#else
  BoundaryConditions2D bc_v[P4EST_DIM];
  BoundaryConditions2D bc_p;
#endif
  detect_ridge* ridge_detector = NULL;
  BCWALLTYPE_U* bc_wall_type_u = NULL;
#ifdef P4_TO_P8
  BCWALLTYPE_W* bc_wall_type_w = NULL;
#endif

  bc_v[0].setWallValues(bc_wall_value_u); // wall-type is simulation/restart-dependent, needs to be constructed later on
  bc_v[1].setWallValues(bc_wall_value_v); bc_v[1].setWallTypes(bc_wall_type_v);
#ifdef P4_TO_P8
  bc_v[2].setWallValues(bc_wall_value_w); // wall-type is simulation/restart-dependent, needs to be constructed later on
#endif
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

  external_force_u_t external_force_u;
  external_force_v_t external_force_v;
#ifdef P4_TO_P8
  external_force_w_t external_force_w;
  CF_3 *external_forces[P4EST_DIM] = { &external_force_u, &external_force_v, &external_force_w };
#else
  CF_2 *external_forces[P4EST_DIM] = { &external_force_u, &external_force_v };
#endif


  if(cmd.contains("restart"))
  {
    const string backup_directory = cmd.get<string>("restart", "");
    if(!is_folder(backup_directory.c_str()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_shs_3d: the restart path %s is not an accessible directory.", backup_directory.c_str());
#else
      sprintf(error_msg, "main_shs_2d: the restart path %s is not an accessible directory.", backup_directory.c_str());
#endif
      throw std::invalid_argument(error_msg);
    }
    if (ns != NULL)
    {
      delete ns; ns = NULL;
    }
    P4EST_ASSERT(ns == NULL);
    ns                      = new my_p4est_navier_stokes_t(mpi, backup_directory.c_str(), tstart);
    dt                      = ns->get_dt();
    p4est_t *p4est_n        = ns->get_p4est();
    p4est_t *p4est_nm1      = ns->get_p4est_nm1();

    lmin                    = cmd.get<int>("lmin", ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl);
    lmax                    = cmd.get<int>("lmax", ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl);
    double lip;
    if(!cmd.contains("lip"))
      lip                   = ((splitting_criteria_t*) p4est_n->user_pointer)->lip;
    threshold_split_cell    = cmd.get<double>("thresh", ns->get_split_threshold());
    length                  = ns->get_length_of_domain();
#ifdef P4EST_ENABLE_DEBUG
    double height           = ns->get_height_of_domain();
#endif
#ifdef P4_TO_P8
    width                   = ns->get_width_of_domain();
#endif
    P4EST_ASSERT((fabs(height-2.0) < 2.0*10.0*EPS) && (fabs(ns->get_rho() - 1.0) < 1.0*10.0*EPS));
    // if restarting with a specific Re_tau, adjust and reset the viscosity to match dp_dx = -1.0
    // otherwise, just read the solver's viscosity
    if(cmd.contains("Re_tau"))
      mu                    = rho*1.0*sqrt(external_force_u.get_value()*1.0)/(cmd.get<double>("Re_tau")); // delta = 1.0
    else
    {
      P4EST_ASSERT(cmd.contains("Re_b"));
      mu                    = ns->get_mu();
      // if restarting with a specific Re_b, set the desired bulk velocity to match it
      bulk_velocity_to_set  = mu*(cmd.get<double>("Re_b"))/(rho*1.0); // delta = 1.0
    }

    if(brick != NULL && brick->nxyz_to_treeid != NULL)
    {
      P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
      delete brick; brick = NULL;
    }
    P4EST_ASSERT(brick == NULL);
    brick                   = ns->get_brick();
    ntree_x                 = brick->nxyztrees[0];
    ntree_y                 = brick->nxyztrees[1];
#ifdef P4_TO_P8
    ntree_z                 = brick->nxyztrees[2];
    n_xyz[2]                = ntree_z;
    xyz_min[2]              = brick->xyz_min[2];
    xyz_max[2]              = brick->xyz_max[2];
#endif
    n_xyz[1]                = ntree_y;
    n_xyz[0]                = ntree_x;
    xyz_min[1]              = brick->xyz_min[1];
    xyz_min[0]              = brick->xyz_min[0];
    xyz_max[1]              = brick->xyz_max[1];
    xyz_max[0]              = brick->xyz_max[0];

    if(cmd.contains("lip"))
#ifdef P4_TO_P8
      lip                   = cmd.get<double>("lip")*(2.0/ntree_y)/sqrt(SQR(2.0/ntree_y)+SQR(length/ntree_x)+SQR(width/ntree_z));
#else
      lip                   = cmd.get<double>("lip")*(2.0/ntree_y)/sqrt(SQR(2.0/ntree_y)+SQR(length/ntree_x));
#endif

    if(cmd.contains("wall_layer"))
    {
      wall_layer            = cmd.get<unsigned int>("wall_layer");
#ifdef P4_TO_P8
      uniform_band          = ((double) wall_layer)*(2.0/((double) ntree_y))/MAX(length/((double) ntree_x), 2.0/((double) ntree_y), width/((double) ntree_z));
#else
      uniform_band          = ((double) wall_layer)*(2.0/((double) ntree_y))/MAX(length/((double) ntree_x), 2.0/((double) ntree_y));
#endif
    }
    else
    {
      uniform_band          = ns->get_uniform_band();
#ifdef P4_TO_P8
      wall_layer            = (unsigned int) (uniform_band*MAX(length/((double) ntree_x), 2.0/((double) ntree_y), width/((double) ntree_z))/(2.0/((double) ntree_y)));
#else
      wall_layer            = (unsigned int) (uniform_band*MAX(length/((double) ntree_x), 2.0/((double) ntree_y))/(2.0/((double) ntree_y)));
#endif
    }
#ifdef P4_TO_P8
    if(spanwise)
      check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
    else
      check_pitch_and_gas_fraction(width, ntree_z, lmax, pitch_to_delta, gas_fraction);
#else
    check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
#endif

    sl_order                = cmd.get<int>("sl_order", ns->get_sl_order());
    cfl                     = cmd.get<double>("cfl", ns->get_cfl());

    if(level_set != NULL)
    {
      delete level_set; level_set = NULL;
    }
    P4EST_ASSERT(level_set == NULL);
    if(ridge_detector!=NULL)
      delete ridge_detector;
#ifdef P4_TO_P8
    ridge_detector = new detect_ridge(length, width, !spanwise, pitch_to_delta, gas_fraction, *brick, lmax);
#else
    ridge_detector = new detect_ridge(length, pitch_to_delta, gas_fraction, *brick, lmax);
#endif
    level_set = new LEVEL_SET(lmax, ridge_detector);
    if(data != NULL)
    {
      delete data; data = NULL;
    }
    P4EST_ASSERT(data == NULL);
    data = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, level_set, uniform_band, lip);
    splitting_criteria_t* to_delete = (splitting_criteria_t*) p4est_n->user_pointer;
    bool fix_restarted_grid = (lmax!=to_delete->max_lvl);
    delete to_delete;
    p4est_n->user_pointer   = (void*) data;
    p4est_nm1->user_pointer = (void*) data; // p4est_n and p4est_nm1 always point to the same splitting_criteria_t no need to delete the nm1 one, it's just been done
    ns->set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, cfl);

    if(bc_wall_type_u!=NULL)
      delete  bc_wall_type_u;
#ifdef P4_TO_P8
    if(bc_wall_type_w!=NULL)
      delete bc_wall_type_w;
#endif
    bc_wall_type_u = new BCWALLTYPE_U(ridge_detector);
    bc_v[0].setWallTypes(*bc_wall_type_u);
#ifdef P4_TO_P8
    bc_wall_type_w = new BCWALLTYPE_W(ridge_detector);
    bc_v[2].setWallTypes(*bc_wall_type_w);
#endif
    ns->set_bc(bc_v, &bc_p);
    ns->set_external_forces(external_forces);
    if(fix_restarted_grid)
      ns->refine_coarsen_grid_after_restart(level_set, false);
  }
  else
  {
    lmin                    = cmd.get<int>("lmin", 4);
    lmax                    = cmd.get<int>("lmax", 6);
    threshold_split_cell    = cmd.get<double>("thresh", 0.1);
    length                  = cmd.get<double>("length", 6.0);
#ifdef P4_TO_P8
    width                   = cmd.get<double>("width", 3.0);
#endif
    mu                      = rho*1.0*sqrt(external_force_u.get_value()*1.0)/(cmd.get<double>("Re_tau", 180.0)); // delta = 1.0

    ntree_x                 = cmd.get<int>("nx", (int) length);
    ntree_y                 = cmd.get<int>("ny", 2);
#ifdef P4_TO_P8
    int ntree_z             = cmd.get<int>("nz", (int) width);
    n_xyz[2]                = ntree_z;
    xyz_min[2]              = -0.5*width;
    xyz_max[2]              = 0.5*width;
#endif
    n_xyz[1]                = ntree_y;
    n_xyz[0]                = ntree_x;
    xyz_min[1]              = -1.0;
    xyz_min[0]              = -0.5*length;
    xyz_max[1]              = +1.0;
    xyz_max[0]              = +0.5*length;
    wall_layer              = cmd.get<unsigned int>("wall_layer", 6);
#ifdef P4_TO_P8
    uniform_band            = ((double) wall_layer)*(2.0/((double) ntree_y))/MAX(length/((double) ntree_x), 2.0/((double) ntree_y), width/((double) ntree_z));
    if(spanwise)
      check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
    else
      check_pitch_and_gas_fraction(width, ntree_z, lmax, pitch_to_delta, gas_fraction);
#else
    uniform_band            = ((double) wall_layer)*(2.0/((double) ntree_y))/MAX(length/((double) ntree_x), 2.0/((double) ntree_y));
    check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
#endif
    sl_order                = cmd.get<int>("sl_order", 2);
    cfl                     = cmd.get<double>("cfl", 1.0);

    p4est_connectivity_t *connectivity;
    if(brick != NULL && brick->nxyz_to_treeid != NULL)
    {
      P4EST_FREE(brick->nxyz_to_treeid);brick->nxyz_to_treeid = NULL;
      delete brick; brick = NULL;
    }
    P4EST_ASSERT(brick == NULL);
    brick = new my_p4est_brick_t;

    connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);

    if(level_set != NULL)
    {
      delete level_set; level_set = NULL;
    }
    P4EST_ASSERT(level_set == NULL);
    if(ridge_detector!=NULL)
      delete ridge_detector;
#ifdef P4_TO_P8
    ridge_detector = new detect_ridge(length, width, !spanwise, pitch_to_delta, gas_fraction, *brick, lmax);
#else
    ridge_detector = new detect_ridge(length, pitch_to_delta, gas_fraction, *brick, lmax);
#endif
    level_set = new LEVEL_SET(lmax, ridge_detector);

    if(data != NULL)
    {
      delete data; data = NULL;
    }
    P4EST_ASSERT(data == NULL);
    // lip_const multiplies the cell-diagonal internally, but  we need only dy --> so scale it appropriately
#ifdef P4_TO_P8
    const double lip_const = cmd.get<double>("lip", 1.2)*(2.0/ntree_y)/sqrt(SQR(2.0/ntree_y)+SQR(length/ntree_x)+SQR(width/ntree_z));
#else
    const double lip_const = cmd.get<double>("lip", 1.2)*(2.0/ntree_y)/sqrt(SQR(2.0/ntree_y)+SQR(length/ntree_x));
#endif
    data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, level_set, uniform_band, lip_const);

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
    sample_cf_on_nodes(p4est_n, nodes_n, *level_set, phi);

#ifdef P4_TO_P8
    CF_3 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1 };
    CF_3 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn   };
#else
    CF_2 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1 };
    CF_2 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn   };
#endif

    ns = new my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n);
    ns->set_phi(phi);
    ns->set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, cfl);
    ns->set_velocities(vnm1, vn);

    tstart = 0.0; // no restart so we assume we start from 0.0
    // set the first time step: kinda arbitrary, don't really know what else I could do. I believe an inverse power of Re_tau should be used here instead...
    double min_dxyz[P4EST_DIM]; dxyz_min(p4est_n, min_dxyz);
#ifdef P4_TO_P8
    dt = MIN(min_dxyz[0], min_dxyz[1], min_dxyz[2])/Re_tau(external_force_u, ns->get_rho(), ns->get_mu()); // no problem using Re_tau() here, external_force_u returns 1.0 by default, by solver's initialization
#else
    dt = MIN(min_dxyz[0], min_dxyz[1])/Re_tau(external_force_u, ns->get_rho(), ns->get_mu()); // no problem using Re_tau() here, external_force_u returns 1.0 by default, by solver's initialization
#endif
    ns->set_dt(dt, dt);

    if(bc_wall_type_u!=NULL)
      delete  bc_wall_type_u;
#ifdef P4_TO_P8
    if(bc_wall_type_w!=NULL)
      delete bc_wall_type_w;
#endif
    bc_wall_type_u = new BCWALLTYPE_U(ridge_detector);
    bc_v[0].setWallTypes(*bc_wall_type_u);
#ifdef P4_TO_P8
    bc_wall_type_w = new BCWALLTYPE_W(ridge_detector);
    bc_v[2].setWallTypes(*bc_wall_type_w);
#endif

    ns->set_bc(bc_v, &bc_p);
    ns->set_external_forces(external_forces);
  }

  char out_dir[PATH_MAX], profile_path[PATH_MAX], vtk_path[PATH_MAX], vtk_name[PATH_MAX];
  if(cmd.contains("restart")){
    ierr = PetscPrintf(mpi.comm(), "Simulation restarted from state saved in %s\n", (cmd.get<string>("restart")).c_str()); CHKERRXX(ierr); }
  if(cmd.contains("Re_tau") || !cmd.contains("restart"))
  {
#ifdef P4_TO_P8
    ierr = PetscPrintf(mpi.comm(), "Parameters : Re_tau = %g, domain is %dx2x%d (delta units), P/delta = %g, GF = %g\n", cmd.get<double>("Re_tau"), (int) length, (int) width, pitch_to_delta, gas_fraction); CHKERRXX(ierr);
    sprintf(out_dir, "%s/%dX2X%d_channel/Re_tau_%.2f/%s/pitch_to_delta_%.3f/GF_%.2f/ny_%d_lmin_%d_lmax_%d", export_dir.c_str(), (int) length, (int) width, cmd.get<double>("Re_tau"), ((spanwise)? "spanwise": "streamwise"), pitch_to_delta, gas_fraction, n_xyz[1], lmin, lmax);
#else
    ierr = PetscPrintf(mpi.comm(), "Parameters : Re_tau = %g, domain is %dx2 (delta units), P/delta = %g, GF = %g\n", cmd.get<double>("Re_tau"), (int) length, pitch_to_delta, gas_fraction); CHKERRXX(ierr);
    sprintf(out_dir, "%s/%dX2_channel/Re_tau_%.2f/pitch_to_delta_%.3f/GF_%.2f/ny_%d_lmin_%d_lmax_%d", export_dir.c_str(), (int) length, cmd.get<double>("Re_tau"), pitch_to_delta, gas_fraction, n_xyz[1], lmin, lmax);
#endif
  }
  else
  {
#ifdef P4_TO_P8
    ierr = PetscPrintf(mpi.comm(), "Parameters : Re_b = %g, domain is %dx2x%d (delta units), P/delta = %g, GF = %g\n", cmd.get<double>("Re_b"), (int) length, (int) width, pitch_to_delta, gas_fraction); CHKERRXX(ierr);
    sprintf(out_dir, "%s/%dX2X%d_channel/Re_b_%.2f/%s/pitch_to_delta_%.3f/GF_%.2f/ny_%d_lmin_%d_lmax_%d", export_dir.c_str(), (int) length, (int) width, cmd.get<double>("Re_b"), ((spanwise)? "spanwise": "streamwise"), pitch_to_delta, gas_fraction, n_xyz[1], lmin, lmax);
#else
    ierr = PetscPrintf(mpi.comm(), "Parameters : Re_b = %g, domain is %dx2 (delta units), P/delta = %g, GF = %g\n", cmd.get<double>("Re_b"), (int) length, pitch_to_delta, gas_fraction); CHKERRXX(ierr);
    sprintf(out_dir, "%s/%dX2_channel/Re_b_%.2f/pitch_to_delta_%.3f/GF_%.2f/ny_%d_lmin_%d_lmax_%d", export_dir.c_str(), (int) length, cmd.get<double>("Re_b"), pitch_to_delta, gas_fraction, n_xyz[1], lmin, lmax);
#endif
  }
  ierr = PetscPrintf(mpi.comm(), "cfl = %g, wall layer = %u\n", cfl, wall_layer);

  if(create_directory(out_dir, mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_shs_3d: could not create exportation directory %s", out_dir);
#else
    sprintf(error_msg, "main_shs_2d: could not create exportation directory %s", out_dir);
#endif
    throw std::runtime_error(error_msg);
  }
  sprintf(vtk_path, "%s/vtu", out_dir);
  if(save_vtk && create_directory(vtk_path, mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_shs_3d: could not create exportation directory for vtk files %s", vtk_path);
#else
    sprintf(error_msg, "main_shs_2d: could not create exportation directory for vtk files %s", vtk_path);
#endif
    throw std::runtime_error(error_msg);
  }

  // for mass_flow_forcing
#ifdef P4_TO_P8
  const bool force_mass_flow[P4EST_DIM]         = {(cmd.contains("restart") && cmd.contains("Re_b")), false, false};
  const double desired_bulk_velocity[P4EST_DIM] = {bulk_velocity_to_set, 0.0, 0.0}; // 2nd and 3rd components are irrelevant
  double current_mass_flow[P4EST_DIM]           = {0.0,0.0};
  double mean_grad_hodge_correction[P4EST_DIM];
#else
  const bool force_mass_flow[P4EST_DIM]         = {(cmd.contains("restart") && cmd.contains("Re_b")), false};
  const double desired_bulk_velocity[P4EST_DIM] = {bulk_velocity_to_set, 0.0}; // 2nd component is irrelevant;
  double current_mass_flow[P4EST_DIM]           = {0.0,0.0};
  double mean_grad_hodge_correction[P4EST_DIM];
#endif

  int iter = 0;
  int export_vtk = -1;
  int save_data_idx = (int) floor(tstart/dt_save_data); // so that we don't save the very first one which was either already read from file, or the known initial condition...

  FILE *fp_velocity_profile;
  char file_drag[PATH_MAX], file_monitoring[PATH_MAX];

  if(save_drag)
    initialize_drag_force_output(file_drag, out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart);
  // initialize sections and mass flows through sections
  double section = -0.5*length;
  double mass_flow = -1.0;
  initialize_monitoring(file_monitoring, out_dir, ns->get_brick()->nxyztrees[1], lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart);

  sprintf(profile_path, "%s/profiles", out_dir);
  if(save_profiles && create_directory(profile_path, mpi.rank(), mpi.comm()))
  {
    char error_msg[1024];
#ifdef P4_TO_P8
    sprintf(error_msg, "main_shs_3d: could not create exportation directory for velocity profiles %s", profile_path);
#else
    sprintf(error_msg, "main_shs_2d: could not create exportation directory for velocity profiles %s", profile_path);
#endif
    throw std::runtime_error(error_msg);
  }
  char file_slice_avg_velocity_profile[PATH_MAX];
  vector<string> file_line_avg_velocity_profile;
  vector<unsigned int> bin_index;
  unsigned int nbins;
  if(save_profiles)
  {
#ifdef P4_TO_P8
    double domain_dimensions[P4EST_DIM] = {length, 2.0, width};
    initialize_averaged_velocity_profiles(file_slice_avg_velocity_profile, profile_path, n_xyz, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart,
                                          bin_index, file_line_avg_velocity_profile, domain_dimensions, pitch_to_delta, gas_fraction, nbins, spanwise);
#else
    double domain_dimensions[P4EST_DIM] = {length, 2.0};
    initialize_averaged_velocity_profiles(file_slice_avg_velocity_profile, profile_path, n_xyz, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart,
                                          bin_index, file_line_avg_velocity_profile, domain_dimensions, pitch_to_delta, gas_fraction, nbins, true);
#endif
    t_line_average.resize(nbins);
    line_averaged_profiles.resize(nbins);
    line_averaged_profiles_nm1.resize(nbins);
    time_averaged_line_averaged_profiles.resize(nbins);
  }

  parStopWatch watch, substep_watch;
  double mean_full_iteration_time = 0.0, mean_viscosity_step_time = 0.0, mean_projection_step_time = 0.0, mean_compute_velocity_at_nodes_time = 0.0, mean_update_time = 0.0;
  if(get_timing)
    watch.start("Total runtime");
  double tn = tstart;

  my_p4est_poisson_cells_t* cell_solver = NULL;
  my_p4est_poisson_faces_t* face_solver = NULL;

  bool accuracy_check_done = false;
  while(tn+0.01*dt<tstart+duration && !accuracy_check_done)
  {
    if(get_timing)
      substep_watch.start("");
    if(iter>0)
    {
      if(mass_flow < 0.0)
        std::runtime_error("main_shs_*d: something went wrong, the mass flow should be strictly positive and known to the solver at this stage...");
      // let's use 10% of the current average velocity for the min value considered for u_max in evaluating dt
      // (kinda arbitrary, but this parameter is not really relevant, we just want to avoid crazy big time step due to flow currently at rest or pretty much)
#ifdef P4_TO_P8
      double min_value_considered_for_umax = 0.1*mass_flow/(ns->get_height_of_domain()*ns->get_width_of_domain());
#else
      double min_value_considered_for_umax = 0.1*mass_flow/ns->get_height_of_domain();
#endif
      if(use_adapted_dt)
        ns->compute_adapted_dt(min_value_considered_for_umax);
      else
        ns->compute_dt(min_value_considered_for_umax);
      dt = ns->get_dt();

      if(tn+dt>tstart+duration)
      {
        dt = tstart+duration-tn;
        ns->set_dt(dt);
      }

      if(save_vtk && dt > vtk_dt)
      {
        dt = vtk_dt; // so that we don't miss snapshots...
        ns->set_dt(dt);
      }

      bool solvers_can_be_reused = ns->update_from_tn_to_tnp1(NULL, (iter%steps_grid_update!=0), false);
      if(cell_solver != NULL && !solvers_can_be_reused){
        delete cell_solver; cell_solver = NULL; }
      if(face_solver != NULL && !solvers_can_be_reused){
        delete face_solver; face_solver = NULL; }
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
    double constant_mass_flow_correction = 10.0*Ub_tolerance*MAX(fabs(desired_bulk_velocity[0]), 1.0);
    while((iter_hodge<niter_hodge_max && corr_hodge>hodge_tolerance) || (force_mass_flow[0] && (fabs(constant_mass_flow_correction) > Ub_tolerance*fabs(desired_bulk_velocity[0]))))
    {
      hodge_new = ns->get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      if(get_timing)
        substep_watch.start("");
      ns->solve_viscosity(face_solver, (face_solver!=NULL), KSPBCGS, pc_face);
      if(get_timing)
      {
        substep_watch.stop();
        mean_viscosity_step_time += substep_watch.read_duration();
        substep_watch.start("");
      }
      ns->solve_projection(cell_solver, (cell_solver!=NULL), cell_solver_type, pc_cell);
      if(force_mass_flow[0])
      {
        ns->global_mass_flow_through_slice(dir::x, section, mass_flow);
        current_mass_flow[0] = mass_flow;
        ns->enforce_mass_flow(force_mass_flow, desired_bulk_velocity, mean_grad_hodge_correction, current_mass_flow);
        constant_mass_flow_correction = mean_grad_hodge_correction[0];
        external_force_u.update_term(-constant_mass_flow_correction*ns->get_rho()*ns->alpha()/ns->get_dt());
      }

      if(get_timing)
      {
        substep_watch.stop();
        mean_projection_step_time += substep_watch.read_duration();
      }

      hodge_new = ns->get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      corr_hodge = 0;
      p4est_t *p4est = ns->get_p4est();
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
          corr_hodge = max(corr_hodge, fabs(ho[quad_idx]-hn[quad_idx]));
        }
      }
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &corr_hodge, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      ierr = VecRestoreArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(hodge_new, &hn); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi.comm(), "hodge iteration #%d, correction = %e\n", iter_hodge, corr_hodge); CHKERRXX(ierr);
      iter_hodge++;
    }
    ierr = VecDestroy(hodge_old); CHKERRXX(ierr);
    if(get_timing)
      substep_watch.start("");
    ns->compute_velocity_at_nodes((steps_grid_update > 1));
    if(get_timing)
    {
      substep_watch.stop();
      mean_compute_velocity_at_nodes_time += substep_watch.read_duration();
    }
    ns->compute_pressure();

    tn += dt;

    // monitoring Re_tau and Re_b
    ns->global_mass_flow_through_slice(dir::x, section, mass_flow);
    if(!mpi.rank())
    {
      FILE* fp_monitor = fopen(file_monitoring, "a");
      if(fp_monitor==NULL)
#ifdef P4_TO_P8
        throw std::runtime_error("main_shs_3d: could not open monitoring file.");
      if(external_force_u.get_value() > 0.0)
        fprintf(fp_monitor, "%g %g %g\n", tn, Re_tau(external_force_u, ns->get_rho(), ns->get_mu()),  Re_b(mass_flow, ns->get_width_of_domain(), ns->get_rho(), ns->get_mu()));
      else
        fprintf(fp_monitor, "%g %g %g\n", tn, -1.0,                                                   Re_b(mass_flow, ns->get_width_of_domain(), ns->get_rho(), ns->get_mu()));
#else
        throw std::runtime_error("main_shs_2d: could not open monitoring file.");
      if(external_force_u.get_value() > 0.0)
        fprintf(fp_monitor, "%g %g %g\n", tn, Re_tau(external_force_u, ns->get_rho(), ns->get_mu()),  Re_b(mass_flow, 1.0,                       ns->get_rho(), ns->get_mu()));
      else
        fprintf(fp_monitor, "%g %g %g\n", tn, -1.0,                                                   Re_b(mass_flow, 1.0,                       ns->get_rho(), ns->get_mu()));
#endif
      fclose(fp_monitor);
    }

    // exporting drag if desired
    if(save_drag)
    {
      ns->get_noslip_wall_forces(drag);
      if(!mpi.rank())
      {
        FILE* fp_drag = fopen(file_drag, "a");
        if(fp_drag==NULL)
#ifdef P4_TO_P8
          throw std::runtime_error("main_shs_3d: could not open file for drag output.");
        fprintf(fp_drag, "%g %g %g %g\n", tn, drag[0], drag[1], drag[2]);
#else
          throw std::runtime_error("main_shs_2d: could not open file for drag output.");
        fprintf(fp_drag, "%g %g %g\n", tn, drag[0], drag[1]);
#endif
        fclose(fp_drag);
      }
    }
    if (save_profiles)
    {
      ns->get_slice_averaged_vnp1_profile(dir::x, dir::y, slice_averaged_profile);
      if(!mpi.rank())
      {
        if(iter == 0)
        {
          t_slice_average = tn;
          slice_averaged_profile_nm1.resize(slice_averaged_profile.size(), 0.0);
          time_averaged_slice_averaged_profile.resize(slice_averaged_profile.size(), 0.0);
        }
        else
        {
          for (unsigned int idx = 0; idx < slice_averaged_profile.size(); ++idx)
          {
            time_averaged_slice_averaged_profile[idx] += 0.5*dt*(slice_averaged_profile_nm1[idx]+slice_averaged_profile[idx]);
            slice_averaged_profile_nm1[idx] = slice_averaged_profile[idx];
          }
        }

        if((iter!=0) && (iter%nexport_avg ==0))
        {
          fp_velocity_profile = fopen(file_slice_avg_velocity_profile, "a");
          if(fp_velocity_profile==NULL)
#ifdef P4_TO_P8
            throw std::invalid_argument("main_shs_3d: could not open file for slice-averaged velocity profile output.");
#else
            throw std::invalid_argument("main_shs_2d: could not open file for slice-averaged velocity profile output.");
#endif
          for (int k = 0; k < ntree_y*(1<<lmax); ++k)
          {
            fprintf(fp_velocity_profile, " %.12g", time_averaged_slice_averaged_profile.at(k)/(tn-t_slice_average));
            time_averaged_slice_averaged_profile.at(k) = 0.0;
          }
          fprintf(fp_velocity_profile, "\n");
          // write the next start time
          fprintf(fp_velocity_profile, "%.12g", tn);
          fclose(fp_velocity_profile);
          // reset these
          t_slice_average = tn;
        }
      }
#ifdef P4_TO_P8
      ns->get_line_averaged_vnp1_profiles(dir::x, dir::y, (spanwise? dir::z : dir::x), bin_index, line_averaged_profiles);
#else
      ns->get_line_averaged_vnp1_profiles(dir::x, dir::y, bin_index, line_averaged_profiles);
#endif
      for (unsigned int bin_idx = 0; bin_idx < nbins; ++bin_idx) {
        if(((unsigned int) mpi.rank()) == (bin_idx%mpi.size()))
        {
          if(iter == 0)
          {
            t_line_average[bin_idx] = tn;
            line_averaged_profiles_nm1[bin_idx].resize(line_averaged_profiles[bin_idx].size(), 0.0);
            time_averaged_line_averaged_profiles[bin_idx].resize(line_averaged_profiles[bin_idx].size(), 0.0);
          }
          else
          {
            for (unsigned int idx = 0; idx < line_averaged_profiles[bin_idx].size(); ++idx)
            {
              time_averaged_line_averaged_profiles[bin_idx][idx] += 0.5*dt*(line_averaged_profiles_nm1[bin_idx][idx]+line_averaged_profiles[bin_idx][idx]);
              line_averaged_profiles_nm1[bin_idx][idx] = line_averaged_profiles[bin_idx][idx];
            }
          }

          if((iter!=0) && (iter%nexport_avg ==0))
          {
            fp_velocity_profile = fopen(file_line_avg_velocity_profile[bin_idx].c_str(), "a");
            if(fp_velocity_profile==NULL)
#ifdef P4_TO_P8
              throw std::invalid_argument("main_shs_3d: could not open file for line-averaged velocity profile output.");
#else
              throw std::invalid_argument("main_shs_2d: could not open file for line-averaged velocity profile output.");
#endif
            for (int k = 0; k < ntree_y*(1<<lmax); ++k)
            {
              fprintf(fp_velocity_profile, " %.12g", time_averaged_line_averaged_profiles[bin_idx].at(k)/(tn-t_line_average[bin_idx]));
              time_averaged_line_averaged_profiles[bin_idx].at(k) = 0.0;
            }
            fprintf(fp_velocity_profile, "\n");
            // write the next start time
            fprintf(fp_velocity_profile, "%.12g", tn);
            fclose(fp_velocity_profile);
            // reset these
            t_line_average[bin_idx] = tn;
          }
        }
      }
    }

    if(external_force_u.get_value() > 0.0){
      ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d, \t Re_tau = %.2f, \t Re_b = %.2f\n",
                         iter, tn, 100*(tn - tstart)/duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants,
                         Re_tau(external_force_u, ns->get_rho(), ns->get_mu()),
                   #ifdef P4_TO_P8
                         Re_b(mass_flow, ns->get_width_of_domain(), ns->get_rho(), ns->get_mu())
                   #else
                         Re_b(mass_flow, 1.0,                       ns->get_rho(), ns->get_mu())
                   #endif
                         ); CHKERRXX(ierr);}
    else{
      ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : driving bulk force is currently negative\n", iter); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d, \t f_x = %.2f, \t Re_b = %.2f\n",
                         iter, tn, 100*(tn - tstart)/duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants,
                         external_force_u.get_value(),
                   #ifdef P4_TO_P8
                         Re_b(mass_flow, ns->get_width_of_domain(), ns->get_rho(), ns->get_mu())
                   #else
                         Re_b(mass_flow, 1.0,                       ns->get_rho(), ns->get_mu())
                   #endif
                         ); CHKERRXX(ierr);}

    if(ns->get_max_L2_norm_u()>5.0*(force_mass_flow[0]? desired_bulk_velocity[0] : Re_tau(external_force_u, ns->get_rho(), ns->get_mu())*sqrt(external_force_u.get_value()*1.0))) // delta = 1.0
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
      ns->save_vtk(vtk_name, true);
    }

    if(do_accuracy_check)
    {
#ifdef P4_TO_P8
      check_accuracy_of_solution(ns, external_force_u, pitch_to_delta, gas_fraction, spanwise, my_accuracy_check_errors);
#else
      check_accuracy_of_solution(ns, external_force_u, pitch_to_delta, gas_fraction, my_accuracy_check_errors);
#endif
      ierr = PetscPrintf(mpi.comm(), "The face-error on u is %.8f\n", my_accuracy_check_errors[0]); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "The face-error on v is %.8f\n", my_accuracy_check_errors[1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(mpi.comm(), "The face-error on w is %.8f\n", my_accuracy_check_errors[2]); CHKERRXX(ierr);
#endif
      ierr = PetscPrintf(mpi.comm(), "The node-error on u is %.8f\n", my_accuracy_check_errors[P4EST_DIM+0]); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "The node-error on v is %.8f\n", my_accuracy_check_errors[P4EST_DIM+1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = PetscPrintf(mpi.comm(), "The node-error on w is %.8f\n", my_accuracy_check_errors[P4EST_DIM+2]); CHKERRXX(ierr);
#endif
      accuracy_check_done = true;
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

  if(cell_solver!=NULL)
    delete  cell_solver;
  if(face_solver!=NULL)
    delete face_solver;

  delete ns;        // deletes the navier-stokes solver
  // the brick and the connectivity are deleted within the above destructor...
  delete data;      // deletes the splitting criterion object
  delete level_set; // deletes the levelset object
  if(ridge_detector!=NULL)
    delete ridge_detector;
  if(bc_wall_type_u!=NULL)
    delete bc_wall_type_u;
#ifdef P4_TO_P8
  if(bc_wall_type_w)
    delete bc_wall_type_w;
#endif

  return 0;
}
