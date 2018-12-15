
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

// we enforce the non-dimensional channel height to be from -1.0 to 1.0
const double ymin =-1.0;
const double ymax = +1.0;

class LEVEL_SET :
    #ifdef P4_TO_P8
    public CF_3
    #else
    public CF_2
    #endif
{
  int max_lvl;
public:
  LEVEL_SET(int max_lvl_) : max_lvl(max_lvl_) { lip = 1.2; }
  double operator()(double, double y
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return MAX(y-1.0-pow(2.0, -max_lvl), -y-1.0-pow(2.0, -max_lvl));
  }
};

struct BCWALLTYPE_P :
    #ifdef P4_TO_P8
    WallBC3D
    #else
    WallBC2D
    #endif
{
  BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                   , double
                                 #endif
                                   ) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} bc_wall_value_p;

class BCWALLTYPE_U :
    #ifdef P4_TO_P8
    public WallBC3D
    #else
    public WallBC2D
    #endif
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
  BCWALLTYPE_U(double len_, double width_, bool streamwise_, double pitch_, double gas_fraction_, const my_p4est_brick_t& brick_, int max_lvl)
    : length(len_), width(width_), streamwise(streamwise_), pitch(pitch_), gas_frac(gas_fraction_),
      offset(0.1*(brick_.xyz_max[(streamwise?2:0)]-brick_.xyz_min[(streamwise?2:0)])/((double) (brick_.nxyztrees[(streamwise?2:0)]*(1<<max_lvl)))){ }
#else
  BCWALLTYPE_U(double len_, double pitch_, double gas_fraction_, const my_p4est_brick_t& brick_, int max_lvl)
    : length(len_), pitch(pitch_), gas_frac(gas_fraction_),
      offset(0.5*(brick_.xyz_max[0]-brick_.xyz_min[0])/((double) (brick_.nxyztrees[0]*(1<<max_lvl)))){ }
#endif
#ifdef P4_TO_P8
  BoundaryConditionType operator()(double x, double, double z) const
  {
    if(streamwise)
    {
      double normalized_z = my_fmod((z + 0.5*width), pitch);
      return (((offset < normalized_z) && (normalized_z < pitch*gas_frac - offset)) ? NEUMANN : DIRICHLET);
    }
    else
    {
      double normalized_x = my_fmod((x + 0.5*length), pitch);
      return (((offset < normalized_x) && (normalized_x < pitch*gas_frac - offset)) ? NEUMANN : DIRICHLET);
    }
  }
#else
  BoundaryConditionType operator()(double x, double) const
  {
    return ((my_fmod((x + 0.5*length - offset), pitch)/pitch < gas_frac)? NEUMANN: DIRICHLET);
  }
#endif
};

struct BCWALLVALUE_U : // always 0 value whether no slip or free slip
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} bc_wall_value_u;

struct BCWALLTYPE_V : // always homogeneous dirichlet : no penetration through the channel wall
    #ifdef P4_TO_P8
    WallBC3D
    #else
    WallBC2D
    #endif
{
  BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                   , double
                                 #endif
                                   ) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;


struct BCWALLVALUE_V : // always homogeneous dirichlet : no penetration through the channel wall
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} bc_wall_value_v;


struct initial_velocity_unm1_t :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
} initial_velocity_vn;

struct external_force_u_t :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 1.0;
  }
};

struct external_force_v_t :
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  double operator()(double, double
                  #ifdef P4_TO_P8
                    , double
                  #endif
                    ) const
  {
    return 0.0;
  }
};

#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D
{
private:
  const double length;
  const double width;
  const bool streamwise;
  const double pitch;
  const double gas_frac;
  const double offset;
  double my_fmod(const double num, const double denom) const { return (num - floor(num/denom)*denom);}
public:
  BCWALLTYPE_W(double len_, double width_, bool streamwise_, double pitch_, double gas_fraction_, const my_p4est_brick_t& brick_, int max_lvl)
    : length(len_), width(width_), streamwise(streamwise_), pitch(pitch_), gas_frac(gas_fraction_),
      offset(0.1*(brick_.xyz_max[(streamwise?2:0)]-brick_.xyz_min[(streamwise?2:0)])/((double) (brick_.nxyztrees[(streamwise?2:0)]*(1<<max_lvl)))){ }
  BoundaryConditionType operator()(double x, double, double z) const
  {
    if(streamwise)
    {
      double normalized_z = my_fmod((z + 0.5*width), pitch);
      return (((offset < normalized_z) && (normalized_z < pitch*gas_frac - offset)) ? NEUMANN : DIRICHLET);
    }
    else
    {
      double normalized_x = my_fmod((x + 0.5*length), pitch);
      return (((offset < normalized_x) && (normalized_x < pitch*gas_frac - offset)) ? NEUMANN : DIRICHLET);
    }
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

#ifdef P4_TO_P8
void initialize_mass_flow_output(std::vector<double>& sections, std::vector<double>& mass_flows, char* file_mass_flow, const double& length, const double& width, const char *out_dir, const int& lmin, const int& lmax, const double& threshold_split_cell, const double& cfl, const int& sl_order, const mpi_environment_t& mpi, const double& tstart)
#else
void initialize_mass_flow_output(std::vector<double>& sections, std::vector<double>& mass_flows, char* file_mass_flow, const double& length, const char *out_dir, const int& lmin, const int& lmax, const double& threshold_split_cell, const double& cfl, const int& sl_order, const mpi_environment_t& mpi, const double& tstart)
#endif
{
  // initialize sections and mass flows through sections
  sections.resize(0);
  sections.push_back(-0.5*length);
  sections.push_back(-0.25*length);
  sections.push_back(0.0);
  sections.push_back(+0.25*length);
  mass_flows.resize(sections.size(), 0.0);

  sprintf(file_mass_flow, "%s/mass_flow_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat", out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order);

  PetscErrorCode ierr = PetscPrintf(mpi.comm(), "Saving mass flow in ... %s\n", file_mass_flow); CHKERRXX(ierr);
  if(mpi.rank() == 0)
  {
    if(!file_exists(file_mass_flow))
    {
      FILE* fp_mass_flow = fopen(file_mass_flow, "w");
      if(fp_mass_flow==NULL)
        throw std::runtime_error("initialize_mass_flow_output: could not open file for mass flow output.");
      fprintf(fp_mass_flow, "%% __ | Normalized mass flows \n");
      fprintf(fp_mass_flow, "%% tn | Inflow section | 0.25*length | Midway section | 0.75*length \n");
      fclose(fp_mass_flow);
    }
    else
    {
      FILE* fp_mass_flow= fopen(file_mass_flow, "r+");
      char* read_line = NULL;
      size_t len = 0;
      ssize_t len_read;
      long size_to_keep = 0;
      if(((len_read = getline(&read_line, &len, fp_mass_flow)) != -1))
        size_to_keep += (long) len_read;
      else
        throw std::runtime_error("initialize_mass_flow_output: couldn't read the first header line of mass_flow_...dat");
      if(((len_read = getline(&read_line, &len, fp_mass_flow)) != -1))
        size_to_keep += (long) len_read;
      else
        throw std::runtime_error("initialize_mass_flow_output: couldn't read the second header line of mass_flow_...dat");
      double time;
      char format_specifier[1024] = "%lg";
      for (size_t kk = 0; kk < sections.size(); ++kk)
        strcat(format_specifier, " %*g");
      while ((len_read = getline(&read_line, &len, fp_mass_flow)) != -1) {
        sscanf(read_line, format_specifier, &time);
        if(time <= tstart)
          size_to_keep += (long) len_read;
        else
          break;
      }
      fclose(fp_mass_flow);
      if(read_line)
        free(read_line);
      if(truncate(file_mass_flow, size_to_keep))
        throw std::runtime_error("initialize_mass_flow_output: couldn't truncate mass_flow_...dat");
    }

    char liveplot_mass[PATH_MAX];
    sprintf(liveplot_mass, "%s/live_mass_flow.gnu", out_dir);
    if(!file_exists(liveplot_mass))
    {
      FILE *fp_liveplot_mass = fopen(liveplot_mass, "w");
      if(fp_liveplot_mass==NULL)
        throw std::runtime_error("initialize_mass_flow_output: could not open file for mass flow liveplot.");
      fprintf(fp_liveplot_mass, "set term wxt noraise\n");
      fprintf(fp_liveplot_mass, "set key bottom right Left font \"Arial,14\"\n");
      fprintf(fp_liveplot_mass, "set xlabel \"Time\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_mass, "set ylabel \"Nondimensional mass flow\" font \"Arial,14\"\n");
      fprintf(fp_liveplot_mass, "plot");
      for (size_t k_section = 0; k_section < sections.size(); ++k_section)
      {
        fprintf(fp_liveplot_mass, "\t \"mass_flow_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:%d title 'x = %g' with lines lw 3", lmin, lmax, threshold_split_cell, cfl, sl_order, ((int)k_section+2), sections[k_section]);
        if(k_section < sections.size()-1)
          fprintf(fp_liveplot_mass, ",\\");
        fprintf(fp_liveplot_mass, "\n");
      }
      fprintf(fp_liveplot_mass, "pause 4\n");
      fprintf(fp_liveplot_mass, "reread");
      fclose(fp_liveplot_mass);
    }

    char tex_plot_mass[PATH_MAX];
    sprintf(tex_plot_mass, "%s/tex_mass_flow.gnu", out_dir);
    if(!file_exists(tex_plot_mass))
    {
      FILE *fp_tex_plot_mass = fopen(tex_plot_mass, "w");
      if(fp_tex_plot_mass==NULL)
        throw std::runtime_error("initialize_mass_flow_output: could not open file for mass flow tex figure.");
      fprintf(fp_tex_plot_mass, "set term epslatex color standalone\n");
      fprintf(fp_tex_plot_mass, "set output 'mass_flow_history.tex'\n");
      fprintf(fp_tex_plot_mass, "set key bottom right Left \n");
      fprintf(fp_tex_plot_mass, "set xlabel \"$t$\"\n");
#ifdef P4_TO_P8
      fprintf(fp_tex_plot_mass, "set ylabel \"$\\\\frac{1}{\\\\rho \\\\delta^{2} u_{\\\\tau}}\\\\int_{%g\\\\delta}^{%g\\\\delta}\\\\int_{-\\\\delta}^{\\\\delta} \\\\rho u \\\\,\\\\mathrm{d}y\\\\mathrm{d}z$\" \n", -0.5*width, width);
#else
      fprintf(fp_tex_plot_mass, "set ylabel \"$\\\\frac{1}{\\\\rho \\\\delta u_{\\\\tau}}\\\\int_{-\\\\delta}^{\\\\delta} \\\\rho u \\\\,\\\\mathrm{d}y$\" \n");
#endif
      fprintf(fp_tex_plot_mass, "plot");
      for (size_t k_section = 0; k_section < sections.size(); ++k_section)
      {
        fprintf(fp_tex_plot_mass, "\t \"mass_flow_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat\" using 1:%d title '$x = %g$' with lines lw 3", lmin, lmax, threshold_split_cell, cfl, sl_order, ((int)k_section+2), sections[k_section]);
        if(k_section < sections.size()-1)
          fprintf(fp_tex_plot_mass, ",\\\n");
      }
      fclose(fp_tex_plot_mass);
    }

    char tex_mass_flow_script[PATH_MAX];
    sprintf(tex_mass_flow_script, "%s/plot_tex_mass_flow.sh", out_dir);
    if(!file_exists(tex_mass_flow_script))
    {
      FILE *fp_tex_mass_flow_script = fopen(tex_mass_flow_script, "w");
      if(fp_tex_mass_flow_script==NULL)
        throw std::runtime_error("initialize_mass_flow_output: could not open file for bash script plotting mass flow tex figure.");
      fprintf(fp_tex_mass_flow_script, "#!/bin/sh\n");
      fprintf(fp_tex_mass_flow_script, "gnuplot ./tex_mass_flow.gnu\n");
      fprintf(fp_tex_mass_flow_script, "latex ./mass_flow_history.tex\n");
      fprintf(fp_tex_mass_flow_script, "dvipdf -dAutoRotatePages=/None ./mass_flow_history.dvi\n");
      fclose(fp_tex_mass_flow_script);

      ostringstream chmod_command;
      chmod_command << "chmod +x " << tex_mass_flow_script;
      if(system(chmod_command.str().c_str()))
        throw std::runtime_error("initialize_mass_flow_output: could not make the plot_tex_mass_flow.sh script executable");
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
      fprintf(fp_drag, "%% tn | x-component | y-component\n");
#else
      fprintf(fp_drag, "%% tn | x-component | y-component | z-component\n");
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
      double time;
      while ((len_read = getline(&read_line, &len, fp_drag)) != -1) {
#ifdef P4_TO_P8
        sscanf(read_line, "%lg %*g %*g %*g", &time);
#else
        sscanf(read_line, "%lg %*g %*g", &time);
#endif
        if(time <= tstart)
          size_to_keep += (long) len_read;
        else
          break;
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

void check_pitch_and_gas_fraction(double length_to_delta, int ntree, int lmax, double pitch_to_delta, double gas_fraction)
{
  if(fabs(length_to_delta/pitch_to_delta - ((int) length_to_delta/pitch_to_delta)) > 1e-6)
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: the length of the domain in the direction transversal to the grooves MUST be a multiple of pitch_to_delta to satisfy periodicity.");
#else
    throw std::invalid_argument("main_shs_2d.cpp: the length of the domain in the direction transversal to the grooves MUST be a multiple of pitch_to_delta to satisfy periodicity.");
#endif

  double nb_finest_cell_in_groove =  pitch_to_delta*gas_fraction/(length_to_delta/((double) (ntree*(1<<lmax))));
  double nb_finest_cell_in_ridge  =  pitch_to_delta*(1.0-gas_fraction)/(length_to_delta/((double) (ntree*(1<<lmax))));

  if((fabs(nb_finest_cell_in_groove - ((int) nb_finest_cell_in_groove)) > 1e-6) || (fabs(nb_finest_cell_in_ridge - ((int) nb_finest_cell_in_ridge)) > 1e-6))
#ifdef P4_TO_P8
    throw std::invalid_argument("main_shs_3d.cpp: the finest grid cells do not capture the groove and/or the ridge (subcell resolution for boundary condition would be required).");
#else
    throw std::invalid_argument("main_shs_2d.cpp: the finest grid cells do not capture the groove and/or the ridge (subcell resolution for boundary condition would be required).");
#endif
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is 0.1");
  cmd.add_option("wall_layer", "number of finest cells desired to layer the channel walls (a minimum of 2 is strictly enforced)");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is length to ensure aspect ratio of cells = 1");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is 2.0 (i.e. height) to ensure aspect ratio of cells = 1");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is width to ensure aspect ratio of cells = 1");
#endif
  // physical parameters for the simulations
  cmd.add_option("length", "length of the channel (dimension in streamwise, x-direction) , in units of delta (integer), default is 6");
#ifdef P4_TO_P8
  cmd.add_option("width", "width of the channel (dimension in spanwise, z-direction), in units of delta (integer), default is 3");
  cmd.add_option("streamwise", "if present, the grooves and ridges are understood as streamwise, that is parallel to the flow. If absent, the grooves are perpendicular to the flow.");
#endif
  cmd.add_option("duration", "the duration of the simulation (tfinal-tstart). If not restarted, tstart = 0.0, default duration is 10.");
  cmd.add_option("Re", "the Reynolds number based on wall-shear velocity and half the channel height (in a regular, i.e. not SH, channel), i.e. Re_tau = u_tau*delta/nu, default is 180.0;");
  cmd.add_option("pitch_to_delta", "P/delta ratio, default = 0.375");
  cmd.add_option("GF", "gas fraction, default is 0.5");
  cmd.add_option("adapted_dt", "activates the calculation of dt based on the local cell sizes if present");
  // method-related parameters
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is 2");
  cmd.add_option("cfl", "dt = cfl * dx/vmax, default is 0.75");
  // output-control parameters
  cmd.add_option("export_folder", "exportation_folder");
  cmd.add_option("save_vtk", "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt", "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
#ifdef P4_TO_P8
  cmd.add_option("save_mass_flow", "activates exportation of the streamwise mass flow (non-dimensionalized by rho*SQR(delta)*u_tau, calculated at inflow, 0.25*length, 0.5*length and 0.75*length)");
  cmd.add_option("save_drag", "activates exportation of the total drag (normalized by rho*SQR(u_tau)*SQR(delta))");
#else
  cmd.add_option("save_mass_flow", "activates exportation of the streamwise mass flow (non-dimensionalized by rho*delta*u_tau, calculated at inflow, 0.25*length, 0.5*length and 0.75*length)");
  cmd.add_option("save_drag", "activates exportation of the total drag (normalized by rho*SQR(u_tau)*delta)");
#endif
  cmd.add_option("save_state_dt", "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates", "determines how many solver states must be memorized in backup_ folders (default is 1).");
  cmd.add_option("save_mean_profile", "compute and save an averaged streamwise-velocity profile (makes sense only if the flow is fully-developed)");
  cmd.add_option("tstart_statistics", "time starting from which the statics can be computed (WARNING: default is 0)");


  // --> extra info to be printed when -help is invoked
  const std::string extra_info = "\
      This program provides a general setup for superhydrophobic channel flow simulations.\n\
      It assumes no solid object and no passive scalar (i.e. smoke) in the channel. The height of the channel is set to \n\
      2*delta by default, the other channel dimensions are provided by the user in units of delta. If the numbers \n\
      of trees in the streamwise and spanwise directions (resp. input parameters nx and nz) are not provided by the user, they \n\
      are set in order to ensure aspect ratio of computational cells as close as possible to 1, i.e. each tree in the forest is \n\
      of size (as close as possible to) deltaXdeltaXdelta. \n\n\
      The set up builds upon the following non-dimensionalization ('_hat' for dimensional variables): \n\n\
      u = u_hat/u_tau, {x, y, z} = {x, y, z}_hat/delta, t = t_hat*u_tau/delta, p = p_hat/(rho*u_tau*u_tau), \n\n\
      where u_tau is the wall-friction velocity in an equivalent regular channel (not superhydrophobic). \n\
      Therefore, the computational domain is [-0.5*length, 0.5*length]x[-1, 1]x[-0.5*width, 0.5*width]. \n\
      The Navier-Stokes solver is then invoked with nondimensional inputs \n\
      rho = 1.0, mu = 1.0/Re, body force per unit mass {1.0, 0.0, 0.0} (driving the flow), \n\
      and with periodic boundary conditions in the streamwise (and spanwise in 3D) directions. \n\
      Developer: Raphael Egan (raphaelegan@ucsb.edu)";
  cmd.parse(argc, argv, extra_info);

  double tstart;
  double dt;
  int lmin, lmax;
  my_p4est_navier_stokes_t* ns  = NULL;
  my_p4est_brick_t* brick       = NULL;
  splitting_criteria_cf_t* data = NULL;
  LEVEL_SET* level_set          = NULL;

  int sl_order;
  double threshold_split_cell, wall_layer, cfl;
  double length;
#ifdef P4_TO_P8
  double width;
#endif
  double wall_shear_Reynolds;
  int ntree_x, ntree_y;
#ifdef P4_TO_P8
  int ntree_z;
#endif
  int n_xyz [P4EST_DIM];
  double xyz_min [P4EST_DIM];
  double xyz_max [P4EST_DIM];



  double duration           = cmd.get<double>("duration", 1.0 /*250.0*/);
  double pitch_to_delta     = cmd.get<double>("pitch_to_delta", 1.0/4.0);
  double gas_fraction       = cmd.get<double>("GF", 0.5);
#if defined(POD_CLUSTER)
  string export_dir         = cmd.get<string>("export_folder", "/home/regan/superhydrophobic_channel");
#elif defined(STAMPEDE)
  string export_dir         = cmd.get<string>("export_folder", "/work/04965/tg842642/stampede2/superhydrophobic_channel");
#else
  string export_dir         = cmd.get<string>("export_folder", "/home/raphael/workspace/projects/superhydrophobic_channel");
#endif
  bool save_vtk             = cmd.contains("save_vtk");
  double vtk_dt = -1.0;
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
  bool save_drag            = cmd.contains("save_drag"); double drag[P4EST_DIM];
  bool save_mass_flow       = cmd.contains("save_mass_flow"); vector<double> mass_flows; vector<double> sections;
  bool save_state           = cmd.contains("save_state_dt"); double dt_save_data = -1.0;
  unsigned int n_states     = cmd.get<unsigned int>("save_nstates", 1);
  if(save_state)
  {
    dt_save_data            = cmd.get<double>("save_state_dt", -1.0);
    if(dt_save_data < 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_shs_3d.cpp: the value of save_state_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_shs_2d.cpp: the value of save_state_dt must be strictly positive.");
#endif
  }
  bool save_profile         = cmd.contains("save_mean_profile");
  double stat_start         = cmd.get<double>("tstart_statistics", 0.0);

  bool use_adapted_dt       = cmd.contains("adapted_dt");
#ifdef P4_TO_P8
  bool streamwise           = cmd.contains("streamwise");
#endif
  PetscErrorCode ierr;

  if(cmd.contains("restart"))
  {
    string backup_directory = cmd.get<string>("restart", "");
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

    lmin                    = ((splitting_criteria_t*) p4est_n->user_pointer)->min_lvl;
    lmax                    = ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl;
    double lip              = ((splitting_criteria_t*) p4est_n->user_pointer)->lip;
    threshold_split_cell    = ns->get_split_threshold();
    wall_layer              = ns->get_uniform_band();
    length                  = ns->get_length_of_domain();
#ifdef P4EST_ENABLE_DEBUG
    double height           = ns->get_height_of_domain();
#endif
#ifdef P4_TO_P8
    width                   = ns->get_width_of_domain();
#endif
    P4EST_ASSERT((fabs(height-2.0) < 1e-6) && (fabs(ns->get_rho() - 1.0) < 1e-6));
    wall_shear_Reynolds     = 1.0/ns->get_mu();

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
    xyz_max[2]              = brick->xyz_min[2];
#endif
    n_xyz[1]                = ntree_y;
    n_xyz[0]                = ntree_x;
    xyz_min[1]              = brick->xyz_min[1];
    xyz_min[0]              = brick->xyz_min[0];
    xyz_max[1]              = brick->xyz_min[1];
    xyz_max[0]              = brick->xyz_min[0];
#ifdef P4_TO_P8
    if(streamwise)
      check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
    else
      check_pitch_and_gas_fraction(width, ntree_z, lmax, pitch_to_delta, gas_fraction);
#else
    check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
#endif

    sl_order                = ns->get_sl_order();
    cfl                     = ns->get_cfl();

    if(level_set != NULL)
    {
      delete level_set; level_set = NULL;
    }
    P4EST_ASSERT(level_set == NULL);
    level_set = new LEVEL_SET(lmax);
    if(data != NULL)
    {
      delete data; data = NULL;
    }
    P4EST_ASSERT(data == NULL);
    data = new splitting_criteria_cf_t(lmin, lmax, level_set, lip);
    splitting_criteria_t* to_delete = (splitting_criteria_t*) p4est_n->user_pointer;
    delete to_delete;
    p4est_n->user_pointer   = (void*) data;
    p4est_nm1->user_pointer = (void*) data; // p4est_n and p4est_nm1 always point to the same splitting_criteria_t no need to delete the nm1 one, it's just been done
  }
  else
  {
    lmin                    = cmd.get<int>("lmin", 4);
    lmax                    = cmd.get<int>("lmax", 6);
    threshold_split_cell    = cmd.get<double>("thresh", 0.1);
    wall_layer              = cmd.get<double>("wall_layer", 8.0);
    length                  = cmd.get<double>("length", 6.0);
#ifdef P4_TO_P8
    width                   = cmd.get<double>("width", 3.0);
#endif
    wall_shear_Reynolds     = cmd.get<double>("Re", 60.0);

    int periodic[P4EST_DIM];
    ntree_x                 = cmd.get<int>("nx", (int) length);
    ntree_y                 = cmd.get<int>("ny", 2);
#ifdef P4_TO_P8
    int ntree_z             = cmd.get<int>("nz", (int) width);
    n_xyz[2]                = ntree_z;
    xyz_min[2]              = -0.5*width;
    xyz_max[2]              = 0.5*width;
    periodic[2]             = 1;
#endif
    n_xyz[1]                = ntree_y;
    n_xyz[0]                = ntree_x;
    xyz_min[1]              = -1.0;
    xyz_min[0]              = -0.5*length;
    xyz_max[1]              = +1.0;
    xyz_max[0]              = +0.5*length;
    periodic[1]             = 0;
    periodic[0]             = 1;
#ifdef P4_TO_P8
    if(streamwise)
      check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
    else
      check_pitch_and_gas_fraction(width, ntree_z, lmax, pitch_to_delta, gas_fraction);
#else
    check_pitch_and_gas_fraction(length, ntree_x, lmax, pitch_to_delta, gas_fraction);
#endif


    sl_order                = cmd.get<int>("sl_order", 2);
    cfl                     = cmd.get<double>("cfl", 0.75);

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
    level_set = new LEVEL_SET(lmax);

    if(data != NULL)
    {
      delete data; data = NULL;
    }
    P4EST_ASSERT(data == NULL);
    data  = new splitting_criteria_cf_t(lmin, lmax, level_set, 1.2);

    p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est_nm1->user_pointer = (void*) data;

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
    my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
    my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

    /* create the initial forest at time n */
    p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
    p4est_n->user_pointer = (void*) data;

    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);

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
    ns->set_parameters(1.0/wall_shear_Reynolds, 1.0, sl_order, wall_layer, threshold_split_cell, cfl);
    ns->set_velocities(vnm1, vn);

    tstart = 0.0; // no restart so we assume we start from 0.0
    // set he first time step: kinda arbitrary, don't really know what else I could do. I believe an inverse power of Re should be used here...
    double min_dxyz[P4EST_DIM]; dxyz_min(p4est_n, min_dxyz);
#ifdef P4_TO_P8
    dt = MIN(min_dxyz[0], min_dxyz[1], min_dxyz[2])/wall_shear_Reynolds;
#else
    dt = MIN(min_dxyz[0], min_dxyz[1])/wall_shear_Reynolds;
#endif
    ns->set_dt(dt);
  }


#ifdef P4_TO_P8
  BoundaryConditions3D bc_v[P4EST_DIM];
  BoundaryConditions3D bc_p;
  BCWALLTYPE_U bc_wall_type_u(length, width, streamwise, pitch_to_delta, gas_fraction, *brick, lmax);
#else
  BoundaryConditions2D bc_v[P4EST_DIM];
  BoundaryConditions2D bc_p;
  BCWALLTYPE_U bc_wall_type_u(length, pitch_to_delta, gas_fraction, *brick, lmax);
#endif
  bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
  bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
  BCWALLTYPE_W bc_wall_type_w(length, width, streamwise, pitch_to_delta, gas_fraction, *brick, lmax);
  bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

  external_force_u_t *external_force_u=NULL;
  external_force_v_t *external_force_v=NULL;
#ifdef P4_TO_P8
  external_force_w_t *external_force_w=NULL;
#endif
  ns->set_bc(bc_v, &bc_p);

  if(cmd.contains("restart"))
  {
    ns->update_dxyz_hodge();
  }

#ifdef P4_TO_P8
  ierr = PetscPrintf(mpi.comm(), "Parameters : Re_{tau, 0} = %g, domain is %dx2x%d (delta units), P/delta = %g, GF = %g\n", wall_shear_Reynolds, (int) length, (int) width, pitch_to_delta, gas_fraction); CHKERRXX(ierr);
#else
  ierr = PetscPrintf(mpi.comm(), "Parameters : Re_{tau, 0}  = %g, domain is %dx2 (delta units), P/delta = %g, GF = %g\n", wall_shear_Reynolds, (int) length, pitch_to_delta, gas_fraction); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(mpi.comm(), "cfl = %g, wall layer = %g\n", cfl, wall_layer);

  char out_dir[1024], vtk_path[1024], vtk_name[1024];
#ifdef P4_TO_P8
  sprintf(out_dir, "%s/%dX2X%d_channel/Retau_%d/%s/pitch_to_delta_%.3f/GF_%.2f/yplus_min_%.4f_yplus_max_%.4f", export_dir.c_str(), (int) length, (int) width, (int) wall_shear_Reynolds, ((streamwise)? "streamwise": "spanwise"), pitch_to_delta, gas_fraction, wall_shear_Reynolds/pow(2.0, lmax), wall_shear_Reynolds/pow(2.0, lmin));
#else
  sprintf(out_dir, "%s/%dX2_channel/Retau_%d/pitch_to_delta_%.3f/GF_%.2f/yplus_min_%.4f_yplus_max_%.4f", export_dir.c_str(), (int) length, (int) wall_shear_Reynolds, pitch_to_delta, gas_fraction, wall_shear_Reynolds/pow(2.0, lmax), wall_shear_Reynolds/pow(2.0, lmin));
#endif
  sprintf(vtk_path, "%s/vtu", out_dir);
  if(save_vtk || save_drag || save_mass_flow || save_profile || save_state)
  {
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
    if(save_vtk && create_directory(vtk_path, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_shs_3d: could not create exportation directory for vtk files %s", vtk_path);
#else
      sprintf(error_msg, "main_shs_2d: could not create exportation directory for vtk files %s", vtk_path);
#endif
    }
  }


  int iter = 0;
  int export_vtk = -1;
  int save_data_idx = (int) floor(tstart/dt_save_data); // so that we don't save the very first one which was either already read from file, or the known initial condition...

  FILE *fp_velocity_profile;
  char file_drag[PATH_MAX], file_mass_flow[PATH_MAX], file_velocity_profile[PATH_MAX];

  if(save_drag)
    initialize_drag_force_output(file_drag, out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart);
  if(save_mass_flow)
#ifdef P4_TO_P8
    initialize_mass_flow_output(sections, mass_flows, file_mass_flow, length, width, out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart);
#else
    initialize_mass_flow_output(sections, mass_flows, file_mass_flow, length, out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order, mpi, tstart);
#endif
  if(save_profile)
  {
    sprintf(file_velocity_profile, "%s/velocity_profile_%d-%d_split_threshold_%.2f_cfl_%.2f_sl_%d.dat", out_dir, lmin, lmax, threshold_split_cell, cfl, sl_order);

    ierr = PetscPrintf(mpi.comm(), "Saving averaged streamwise velocity profile in ... %s\n", file_velocity_profile); CHKERRXX(ierr);
    if(mpi.rank() == 0)
    {
      fp_velocity_profile = fopen(file_velocity_profile, "w");
      if(fp_velocity_profile==NULL)
#ifdef P4_TO_P8
        throw std::runtime_error("main_shs_3d: could not open file for averaged velocity profile output.");
#else
        throw std::runtime_error("main_shs_2d: could not open file for averaged velocity profile output.");
#endif
      fclose(fp_velocity_profile);
    }
  }


  parStopWatch watch;
  watch.start("Total runtime");
  double tn = tstart;

  while(tn+0.01*dt<tstart+duration)
  {
    if(iter>0)
    {
      if(use_adapted_dt)
        ns->compute_adapted_dt(0.1*wall_shear_Reynolds/pow(2.0, lmax)); // 0.1*y^{+}_min (assuming full resolution of viscous sublayer in regular channel)
      else
        ns->compute_dt(0.1*wall_shear_Reynolds/pow(2.0, lmax)); // 0.1*y^{+}_min (assuming full resolution of viscous sublayer in regular channel)
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

      ns->update_from_tn_to_tnp1(level_set, false, false);
    }
    if(save_state && ((int) floor(tn/dt_save_data)) != save_data_idx)
    {
      save_data_idx = ((int) floor(tn/dt_save_data));
      ns->save_state(out_dir, tn, n_states);
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
    ns->set_external_forces(external_forces);

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, ns->get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double err_hodge = 1;
    int iter_hodge = 0;
    while(iter_hodge<10 && err_hodge>1e-3)
    {
      hodge_new = ns->get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      ns->solve_viscosity();
      ns->solve_projection();

      hodge_new = ns->get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      err_hodge = 0;
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
    ns->compute_velocity_at_nodes();
    ns->compute_pressure();

    tn += dt;

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
    if(save_mass_flow)
    {
      ns->global_mass_flow_through_slice(dir::x, sections, mass_flows);
      if(!mpi.rank())
      {
        FILE* fp_mass_flow= fopen(file_mass_flow, "a");
        if(fp_mass_flow==NULL)
#ifdef P4_TO_P8
          throw std::runtime_error("main_shs_3d: could not open file for mass flow output.");
#else
          throw std::runtime_error("main_shs_2d: could not open file for mass flow output.");
#endif
        fprintf(fp_mass_flow, "%g", tn);
        for (size_t k_section = 0; k_section < mass_flows.size(); ++k_section)
          fprintf(fp_mass_flow, " %g", mass_flows[k_section]);
        fprintf(fp_mass_flow, "\n");
        fclose(fp_mass_flow);
      }
    }
    if (save_profile && (tn > stat_start))
    {
      // update averaged profile here
      if(!mpi.rank())
      {
        fp_velocity_profile = fopen(file_velocity_profile, "a");
        if(fp_velocity_profile==NULL)
#ifdef P4_TO_P8
          throw std::invalid_argument("main_shs_3d: could not open file for averaged velocity profile output.");
#else
          throw std::invalid_argument("main_shs_2d: could not open file for averaged velocity profile output.");
#endif
        //        fprintf(fp_velocity_profile, "%g %g\n", ys, averaged us);
        fclose(fp_velocity_profile);
      }
    }


    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*(tn - tstart)/duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(ns->get_max_L2_norm_u()>1000)
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

  if(external_force_u==NULL) delete external_force_u;
  if(external_force_v==NULL) delete external_force_v;
#ifdef P4_TO_P8
  if(external_force_w==NULL) delete external_force_w;
#endif

  watch.stop();
  watch.read_duration();

  delete ns;        // deletes the navier-stokes solver
  // the brick and the connectivity are deleted within the above destructor...
  // p4est_n and p4est_nm1 are deleted within the above destructor...
  delete data;      // deletes the splitting criterion object
  delete level_set; // deletes the levelset object

  return 0;
}
