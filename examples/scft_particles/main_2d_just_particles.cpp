/*
 * Title: scft_particles
 * Description:
 * Author: ivanabagaric
 * Date Created: 10-24-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_scft.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_shapes.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_shapes.h>
#endif
#include <src/Parser.h>
#include <src/casl_math.h>

#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>

#include <vector>

using namespace std;
static PetscErrorCode ierr;
static param_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<double> xmin (pl, -3, "xmin", "Box xmin");
param_t<double> ymin (pl, -3, "ymin", "Box ymin");
param_t<double> zmin (pl, -3, "zmin", "Box zmin");

param_t<double> xmax (pl, 3, "xmax", "Box xmax");
param_t<double> ymax (pl, 3, "ymax", "Box ymax");
param_t<double> zmax (pl, 3, "zmax", "Box zmax");

param_t<int> nx (pl, 1, "nx", "Number of trees in the x-direction");
param_t<int> ny (pl, 1, "ny", "Number of trees in the y-direction");
param_t<int> nz (pl, 1, "nz", "Number of trees in the z-direction");

param_t<bool> px (pl, 0, "px", "Periodicity in the x-direction (0/1)");
param_t<bool> py (pl, 0, "py", "Periodicity in the y-direction (0/1)");
param_t<bool> pz (pl, 0, "pz", "Periodicity in the z-direction (0/1)");

//-------------------------------------
// refinement parameters
//-------------------------------------
param_t<int> lmin  (pl, 6,   "lmin", "Min level of the tree");
param_t<int> lmax  (pl, 10,  "lmax", "Max level of the tree");
param_t<int> lip   (pl, 1.1, "lip" , "Refinement transition");
param_t<int> band  (pl, 2,   "band" , "Uniform grid band");

//-------------------------------------
// polymer parameters
//-------------------------------------
param_t<double> XN       (pl, 20,  "XN"      , "Interaction strength between A and B blocks");
param_t<double> gamma_a  (pl, 0, "gamma_a" , "Surface tension of A block");
param_t<double> gamma_b  (pl, 1, "gamma_b" , "Surface tension of B block");
param_t<int>    ns       (pl, 40,  "ns"      , "Discretization of polymer chain");
param_t<double> box_size (pl, 1,   "box_size", "Box size in units of Rg");
param_t<double> f        (pl, .5,  "f"       , "Fraction of polymer A");

//-------------------------------------
// scft parameters
//-------------------------------------
param_t<int>    seed                (pl, 5,     "seed", "Seed field for scft: 0 - zero, 1 - random, 2 - vertical stripes, 3 - horizontal stripes, 4 - diagonal stripes, 5 - dots");
param_t<double> scft_tol            (pl, 1.e-4, "scft_tol"           , "Tolerance for SCFT");
param_t<int>    max_scft_iterations (pl, 300,   "max_scft_iterations", "Maximum SCFT iterations");
param_t<int>    bc_adjust_min       (pl, 1,     "bc_adjust_min"      , "Minimun SCFT steps between adjusting BC");
param_t<bool>   smooth_pressure     (pl, 1,     "smooth_pressure"    , "Smooth pressure after first BC adjustment 0/1");
param_t<double> scft_bc_tol         (pl, 4.e-2, "scft_bc_tol"        , "Tolerance for adjusting BC");

//-------------------------------------
// particle dynamics parameters
//-------------------------------------
param_t<bool>   use_scft   (pl, 0,       "use_scft", "Turn on/off SCFT (0/1)");
param_t<bool>   minimize   (pl, 1,       "minimize", "Turn on/off energy minimization (0/1)");
param_t<int>    geometry   (pl, 1,       "geometry", "Initial placement of particles: 0 - one particle, 1 - ...");
param_t<int>    velocity   (pl, -1,       "velocity", "Predifined velocity in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<int>    rotation   (pl, 1,       "rotation", "Predifined rotation in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<double> CFL        (pl, 0.5*2,     "CFL", "CFL number");
param_t<double> time_limit (pl, DBL_MAX, "time_limit", "Time limit");
param_t<int>    step_limit (pl, 200,     "step_limit", "Step limit");

param_t<int>    pairwise_potential_type  (pl, 0, "pairwise_potential_type", "Type of pairwise potential: 0 - quadratic, 1 - 1/(e^x-1)");
param_t<double> pairwise_potential_mag   (pl, 1.0,   "pairwise_potential_mag", "Magnitude of pairwise potential");
param_t<double> pairwise_potential_width (pl, 20, "pairwise_potential_width", "Width of pairwise potential");


param_t<bool>   rerefine_at_start       (pl, 1,      "rerefine_at_start", "Reinitialze level-set function at the start 0/1)");
param_t<int>    contact_angle_extension (pl, 0,      "contact_angle_extension", "Method for extending level-set function into wall: 0 - constant angle (pl, 1 -  (pl, 2 - special");
param_t<int>    volume_corrections      (pl, 2,      "volume_corrections", "Number of volume correction after each move");

param_t<int> num_polymer_geometry (pl, 0, "num_polymer_geometry", "Initial polymer shape: 0 - drop (pl, 1 - film (pl, 2 - combination");
param_t<int> num_wall_geometry    (pl, 3, "num_wall_geometry", "Wall geometry: 0 - no wall (pl, 1 - wall (pl, 2 - well");
param_t<int> num_wall_pattern     (pl, 0, "num_wall_pattern", "Wall chemical pattern: 0 - no pattern");

// surface energies
param_t<int> wall_energy_type (pl, 1, "wall_energy_type", "Method for setting wall surface energy: 0 - explicitly (i.e. convert XN to angles) (pl, 1 - through contact angles (i.e. convert angles to XN)");

param_t<double> XN_air_avg (pl, 1.1, "XN_air_avg", "Polymer-air surface energy strength: average");
param_t<double> XN_air_del (pl, 1, "XN_air_del", "Polymer-air surface energy strength: difference");

param_t<double> angle_A_min (pl, 90, "angle_A_min", "Minimum contact angle for A-block");
param_t<double> angle_A_max (pl, 90, "angle_A_max", "Maximum contact angle for A-block");
param_t<double> angle_B_min (pl, 90, "angle_B_min", "Minimum contact angle for B-block");
param_t<double> angle_B_max (pl, 90, "angle_B_max", "Maximum contact angle for B-block");

param_t<double> XN_wall_A_min (pl, 5, "XN_wall_A_min", "Minimum Polymer-wall interaction strength for A-block");
param_t<double> XN_wall_A_max (pl, 5, "XN_wall_A_max", "Maximum Polymer-wall interaction strength for A-block");
param_t<double> XN_wall_B_min (pl, 8, "XN_wall_B_min", "Minimum Polymer-wall interaction strength for B-block");
param_t<double> XN_wall_B_max (pl, 8, "XN_wall_B_max", "Maximum Polymer-wall interaction strength for B-block");

// geometry parameters
param_t<double> drop_r      (pl, 0.611,  "drop_r", "");
param_t<double> drop_x      (pl, .0079,  "drop_x", "");
param_t<double> drop_y      (pl, -.013,  "drop_y", "");
param_t<double> drop_z      (pl, .0,     "drop_z", "");
param_t<double> drop_r0     (pl, 0.4,    "drop_r0", "");
param_t<double> drop_k      (pl, 5,      "drop_k", "");
param_t<double> drop_deform (pl, 0.0,    "drop_deform", "");

param_t<double> film_eps (pl, -0.0, "film_eps", ""); // curvature
param_t<double> film_nx  (pl, 0,    "film_nx", "");
param_t<double> film_ny  (pl, 1,    "film_ny", "");
param_t<double> film_nz  (pl, 0,    "film_nz", "");
param_t<double> film_x   (pl, .0,   "film_x", "");
param_t<double> film_y   (pl, .0,   "film_y", "");
param_t<double> film_z   (pl, .0,   "film_z", "");
param_t<double> film_perturb   (pl, .1,   "film_perturb", "");

param_t<double> wall_eps (pl, -0.0, "wall_eps", ""); // curvature
param_t<double> wall_nx  (pl, -0,   "wall_nx", "");
param_t<double> wall_ny  (pl, -1,   "wall_ny", "");
param_t<double> wall_nz  (pl, -0,   "wall_nz", "");
param_t<double> wall_x   (pl, .0,   "wall_x", "");
param_t<double> wall_y   (pl, -.2,  "wall_y", "");
param_t<double> wall_r   (pl, 2.5,   "wall_r", "");
param_t<double> wall_z   (pl, .0,   "wall_z", "");

param_t<double> well_x (pl, 0.00,   "well_x", "Well geometry: center");
param_t<double> well_z (pl, 0.0053, "well_z", "Well geometry: position");
param_t<double> well_h (pl, .50,    "well_h", "Well geometry: depth");
param_t<double> well_w (pl, 0.77,   "well_w", "Well geometry: width");
param_t<double> well_r (pl, 0.10,   "well_r", "Well geometry: corner smoothing");

void set_wall_surface_energies()
{
  switch (wall_energy_type()) {
    case 0:

      angle_A_min.val = 180.*acos( SIGN(XN_wall_A_max.val) * sqrt(XN_wall_A_max.val / (XN_air_avg.val+XN_air_del.val)) )/PI;
      angle_A_max.val = 180.*acos( SIGN(XN_wall_A_min.val) * sqrt(XN_wall_A_min.val / (XN_air_avg.val+XN_air_del.val)) )/PI;
      angle_B_min.val = 180.*acos( SIGN(XN_wall_B_max.val) * sqrt(XN_wall_B_max.val / (XN_air_avg.val-XN_air_del.val)) )/PI;
      angle_B_max.val = 180.*acos( SIGN(XN_wall_B_min.val) * sqrt(XN_wall_B_min.val / (XN_air_avg.val-XN_air_del.val)) )/PI;

      break;
    case 1:

      XN_wall_A_max.val = (XN_air_avg.val+XN_air_del.val)*SIGN(cos(angle_A_min.val*PI/180.))*SQR(cos(angle_A_min.val*PI/180.));
      XN_wall_A_min.val = (XN_air_avg.val+XN_air_del.val)*SIGN(cos(angle_A_max.val*PI/180.))*SQR(cos(angle_A_max.val*PI/180.));
      XN_wall_B_max.val = (XN_air_avg.val-XN_air_del.val)*SIGN(cos(angle_B_min.val*PI/180.))*SQR(cos(angle_B_min.val*PI/180.));
      XN_wall_B_min.val = (XN_air_avg.val-XN_air_del.val)*SIGN(cos(angle_B_max.val*PI/180.))*SQR(cos(angle_B_max.val*PI/180.));

      break;
    default:
      throw std::invalid_argument("Invalid method for setting wall surface energies");
  }
}

class gamma_Aa_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return SIGN(XN_air_avg()+XN_air_del())*sqrt(fabs(XN_air_avg()+XN_air_del()));
  }
} gamma_Aa_cf;

class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return SIGN(XN_air_avg()-XN_air_del())*sqrt(fabs(XN_air_avg()-XN_air_del()));
  }
} gamma_Ba_cf;

class gamma_Aw_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_wall_pattern())
    {
      case 0: return SIGN(XN_wall_A_min())*sqrt(fabs(XN_wall_A_min()));
      default: throw std::invalid_argument("Error: Invalid wall pattern number\n");
    }
  }
} gamma_Aw_cf;

class gamma_Bw_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_wall_pattern())
    {
      case 0: return SIGN(XN_wall_B_min())*sqrt(fabs(XN_wall_B_min()));
      default: throw std::invalid_argument("Error: Invalid wall pattern number\n");
    }
  }
} gamma_Bw_cf;

class gamma_aw_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return 0;
  }
} gamma_aw_cf;


/* geometry of interfaces */
class phi_wall_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_wall_geometry())
    {
      case 0: return -1;
      case 1:
        {
          double norm = ABSD(wall_nx(), wall_ny(), wall_nz());
          return - SUMD( (x-wall_x())*((x-wall_x())*wall_eps() - 2.*wall_nx() / norm),
                         (y-wall_y())*((y-wall_y())*wall_eps() - 2.*wall_ny() / norm),
                         (z-wall_z())*((z-wall_z())*wall_eps() - 2.*wall_nz() / norm) )
              / (  ABSD((x-wall_x())*wall_eps() - wall_nx() / norm,
                        (y-wall_y())*wall_eps() - wall_ny() / norm,
                        (z-wall_z())*wall_eps() - wall_nz() / norm)  + 1. );
        }
      case 2:
        {
          double phi_top   = well_z() - y;
          double phi_bot   = well_z()-well_h() - y;
          double phi_walls = MAX(x-well_x()-.5*well_w(), -(x-well_x())-.5*well_w());

          return smooth_max(phi_bot, smooth_min(phi_top, phi_walls, well_r()), well_r());
        }
      case 3:
      {
        return ABSD(x-wall_x(), y-wall_y(), z-wall_z()) - wall_r();
      }
      default:
        throw;
    }
  }
} phi_wall_cf;

class phi_infc_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (num_polymer_geometry())
    {
      case 0: return -1;
      case 1: return sqrt( SQR(x-drop_x()) + SQR(y-drop_y()) P8(+ SQR(z-drop_z())) )
            - drop_r()
            *(1.+drop_deform()*cos(drop_k()*atan2(x-drop_x(),y-drop_y()))
      #ifdef P4_TO_P8
              *(1.-cos(2.*acos((z-drop_z())/sqrt( SQR(x-drop_x()) + SQR(y-drop_y()) + SQR(z-drop_z()) + 1.e-12))))
      #endif
              );
      case 2:
      {
        double norm = ABSD(film_nx(), film_ny(), film_nz());
        return - SUMD( (x-film_x())*((x-film_x())*film_eps() - 2.*film_nx() / norm),
                       (y-film_y())*((y-film_y())*film_eps() - 2.*film_ny() / norm),
                       (z-film_z())*((z-film_z())*film_eps() - 2.*film_nz() / norm) )
            / (  ABSD((x-film_x())*film_eps() - film_nx() / norm,
                      (y-film_y())*film_eps() - film_ny() / norm,
                      (z-film_z())*film_eps() - film_nz() / norm)  + 1. )
             + film_perturb()*cos(PI*x*(lmax()-2));
      }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_infc_cf;


//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool> save_energy (pl, 1,  "save_energy", "Save effective energy into a file");
param_t<bool> save_vtk    (pl, 1,  "save_vtk", "Save vtk data");
param_t<int>  save_freq   (pl, 1, "save_freq", "Frequency of saving vtk data");

struct particle_t
{
  double xyz [P4EST_DIM];
  double axis[P4EST_DIM];
  double rot;

  CF_DIM *phi_cf;
  CF_DIM *phix_cf;
  CF_DIM *phiy_cf;
  CF_DIM *kappa_cf;

  CF_DIM *gA_cf;
  CF_DIM *gAx_cf;
  CF_DIM *gAy_cf;

  CF_DIM *gB_cf;
  CF_DIM *gBx_cf;
  CF_DIM *gBy_cf;

  particle_t()
  {
    foreach_dimension(dim)
    {
      xyz [dim] = 0;
      axis[dim] = 0;
    }

    rot      = 0;
    phi_cf   = NULL;
    phix_cf  = NULL;
    phiy_cf  = NULL;
    kappa_cf = NULL;
    gA_cf    = NULL;
    gAx_cf   = NULL;
    gAy_cf   = NULL;
    gB_cf    = NULL;
    gBx_cf   = NULL;
    gBy_cf   = NULL;
  }

  inline double sample_func(CF_DIM *cf, DIM(double x, double y, double z))
  {
    double X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot);
    double Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot);
    return (*cf)(DIM(X,Y,Z));
  }

  inline double sample_func_x(CF_DIM *cfx, CF_DIM *cfy, DIM(double x, double y, double z))
  {
    double X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot);
    double Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot);
    return (*cfx)(DIM(X,Y,Z))*cos(rot) + (*cfy)(DIM(X,Y,Z))*sin(rot);
  }

  inline double sample_func_y(CF_DIM *cfx, CF_DIM *cfy, DIM(double x, double y, double z))
  {
    double X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot);
    double Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot);
    return -(*cfx)(DIM(X,Y,Z))*sin(rot) + (*cfy)(DIM(X,Y,Z))*cos(rot);
  }

  inline double phi  (DIM(double x, double y, double z)) { return sample_func(phi_cf,   DIM(x,y,z)); }
  inline double kappa(DIM(double x, double y, double z)) { return sample_func(kappa_cf, DIM(x,y,z)); }
  inline double gA   (DIM(double x, double y, double z)) { return sample_func(gA_cf,    DIM(x,y,z)); }
  inline double gB   (DIM(double x, double y, double z)) { return sample_func(gB_cf,    DIM(x,y,z)); }

  inline double gAx  (DIM(double x, double y, double z)) { return sample_func_x(gAx_cf,  gAy_cf,  DIM(x,y,z)); }
  inline double gAy  (DIM(double x, double y, double z)) { return sample_func_y(gAx_cf,  gAy_cf,  DIM(x,y,z)); }
  inline double gBx  (DIM(double x, double y, double z)) { return sample_func_x(gBx_cf,  gBy_cf,  DIM(x,y,z)); }
  inline double gBy  (DIM(double x, double y, double z)) { return sample_func_y(gBx_cf,  gBy_cf,  DIM(x,y,z)); }
  inline double phix (DIM(double x, double y, double z)) { return sample_func_x(phix_cf, phiy_cf, DIM(x,y,z)); }
  inline double phiy (DIM(double x, double y, double z)) { return sample_func_y(phix_cf, phiy_cf, DIM(x,y,z)); }

};

class phi_true_cf_t : public CF_DIM
{
  particle_t *ptr;
public:
  phi_true_cf_t(particle_t *ptr) : ptr(ptr) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return ptr->phi(DIM(x,y,z));
  }
};

// this class constructs the 'analytical field' (level-set function for every (x,y) coordinate)
class phi_part_cf_t : public CF_DIM
{
private:
  std::vector<particle_t> *particles;

public:

  phi_part_cf_t(std::vector<particle_t> &particles)
    : particles(&particles) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    int np = particles->size();
    vector<double> phi_particles(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_particles[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_particles[n] = MAX(phi_particles[n],
                                   particles->at(n).phi(DIM(x+double(i)*(xmax()-xmin()),
                                                            y+double(j)*(ymax()-ymin()),
                                                            z+double(k)*(zmax()-zmin()))));
          }
    }

    double current_max = phi_particles[0];
    for(int n = 1; n < np; n++)
    {
      current_max = MAX(current_max, phi_particles[n]);
    }

    if (current_max != current_max) throw;

    return current_max;
  }
};


// this class constructs the 'particles_number' field (indicates for every (x,y) coordinate, which particle is the closest)
// it is needed for the energy minimization of multiple particles
class particles_number_cf_t : public CF_DIM
{
private:
  std::vector<particle_t> *particles;

public:

  particles_number_cf_t(std::vector<particle_t> &particles)
    : particles(&particles) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    int np = particles->size();
    vector<double> phi_particles(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_particles[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_particles[n] = MAX(phi_particles[n],
                                   particles->at(n).phi(DIM(x+double(i)*(xmax()-xmin()),
                                                            y+double(j)*(ymax()-ymin()),
                                                            z+double(k)*(zmax()-zmin()))));
          }
    }

    int particle_num = 0;
    for (int n = 1; n < np; n++)
    {
      if (phi_particles[particle_num] < phi_particles[n])
      {
        particle_num = n;
      }
    }

    return particle_num;
  }
};

int order = 2.;
double pairwise_potential(double r)
{
  switch (pairwise_potential_type())
  {
    case 0:
      if (r > 0) return 0;
      else return pairwise_potential_mag()*pow(-r/pairwise_potential_width(), order);
    case 2:
      if (r > 0) return 0;
      else return -pairwise_potential_mag()*r*r*r/pairwise_potential_width()/pairwise_potential_width()/pairwise_potential_width();
    case 1:
      if (r > 10.*pairwise_potential_width()) return 0;
    return pairwise_potential_mag()/(exp(r/pairwise_potential_width())-1.);
    default: throw;
  }
}


double pairwise_force(double r)
{
  switch (pairwise_potential_type())
  {
    case 0:
      if (r > 0) return 0;
      else return order*pairwise_potential_mag()*pow(-r/pairwise_potential_width(), order-1)/pairwise_potential_width();
    case 2:
      if (r > 0) return 0;
      else return -3.*pairwise_potential_mag()*r*r/pairwise_potential_width()/pairwise_potential_width()/pairwise_potential_width();
    case 1:
      if (r > 10.*pairwise_potential_width()) return 0;
    return -exp(r/pairwise_potential_width())*pairwise_potential_mag()/SQR(exp(r/pairwise_potential_width())-1.)/pairwise_potential_width();
    default: throw;
  }
}

class penalization_cf_t : public CF_DIM
{
private:
  std::vector<particle_t> *particles;

public:

  penalization_cf_t(std::vector<particle_t> &particles)
    : particles(&particles) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    int np = particles->size();
    vector<double> phi_particles(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_particles[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_particles[n] = MAX(phi_particles[n],
                                   particles->at(n).phi(DIM(x+double(i)*(xmax()-xmin()),
                                                            y+double(j)*(ymax()-ymin()),
                                                            z+double(k)*(zmax()-zmin()))));
          }
    }

    int particle_num = 0;
    for (int n = 1; n < np; n++)
    {
      if (phi_particles[particle_num] < phi_particles[n])
      {
        particle_num = n;
      }
    }

    double sum = 0;


    for (int n = 0; n < np; n++)
    {
      if (n != particle_num)
      {
        sum += pairwise_potential(fabs(phi_particles[n])-pairwise_potential_width());
      }
    }

    return sum;
  }
};


double bulk_lamellar_spacing = 1;


// this class constructs the exchange potential (mu_minus) for every (x,y), that describes the interaction between A&B (A, B are the monomer species)
class mu_minus_cf_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (seed())
    {
      case 0: return 0;
      case 1: return 0.01*XN()*(double)(rand()%1000)/1000.;
      case 2:
      {
        double nx = (xmax()-xmin())/bulk_lamellar_spacing; if (px()) nx = round(nx);
        return .5*XN()*cos(2.*PI*x/(xmax()-xmin())*nx);
      }
      case 3:
      {
        double ny = (ymax()-ymin())/bulk_lamellar_spacing;
        return .5*XN()*cos(2.*PI*y/(ymax()-ymin())*ny);
      }
      case 4:
      {
        double nx = .5*(xmax()-xmin())/bulk_lamellar_spacing; if (px()) nx = round(nx);
        double ny = .5*(ymax()-ymin())/bulk_lamellar_spacing; if (py()) ny = round(ny);
        return .5*XN()*cos(2.*PI*x/(xmax()-xmin())*nx + 2.*PI*y/(ymax()-ymin())*ny);
      }
      case 5:
      {
        double nx = .5*(xmax()-xmin())/bulk_lamellar_spacing; if (px()) nx = round(nx);
        double ny = .5*(ymax()-ymin())/bulk_lamellar_spacing; if (py()) ny = round(ny);
#ifdef P4_TO_P8
        double nz = .5*(zmax()-zmin())/bulk_lamellar_spacing; if (pz()) ny = round(nz);
#endif
        return .5*XN()*MULTD(cos(PI*x/(xmax()-xmin())*nx),
                             cos(PI*y/(ymax()-ymin())*ny),
                             cos(PI*z/(zmax()-zmin())*nz));
      }
    }
  }
};

class radial_gamma_cf_t : public CF_DIM
{
  double gamma_max;
  double gamma_min;
  double k;
  double theta;
  cf_value_type_t what;
public:
  radial_gamma_cf_t(cf_value_type_t what, double gamma_min, double gamma_max, double k, double theta = 0)
    : what(what), gamma_min(gamma_min), gamma_max(gamma_max), k(k), theta(theta) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    double t = atan2(y,x);
    double r = sqrt(x*x + y*y);

    switch (what)
    {
      case VAL: return gamma_min + (gamma_max-gamma_min)*0.5*(1.+cos(k*(t-theta)));
      case DDX: return -(gamma_max-gamma_min)*0.5*k*sin(k*(t-theta))*(-y/r/r);
      case DDY: return -(gamma_max-gamma_min)*0.5*k*sin(k*(t-theta))*( x/r/r);
      default: throw;
    }
  }
};

struct radial_gamma_t
{
  radial_gamma_cf_t val;
  radial_gamma_cf_t ddx;
  radial_gamma_cf_t ddy;

  radial_gamma_t(double gamma_min, double gamma_max, double k, double theta = 0)
    : val(VAL, gamma_min, gamma_max, k, theta),
      ddx(DDX, gamma_min, gamma_max, k, theta),
      ddy(DDY, gamma_min, gamma_max, k, theta) {}
};

void initalize_particles(std::vector<particle_t> &particles)
{
  particles.clear();
  particle_t p;
  switch (geometry())
  {
    case 0:
    {
      static radial_shaped_domain_t domain(0.6, DIM(0,0,0), -1, 3, 0.3, 0.0);

      static radial_gamma_t gA(0, gamma_a(), 1);
      static radial_gamma_t gB(0, gamma_b(), 1);

      p.phi_cf   = &domain.phi;
      p.phix_cf  = &domain.phi_x;
      p.phiy_cf  = &domain.phi_y;
      p.kappa_cf = &domain.phi_c;

      p.gA_cf  = &gA.val;
      p.gAx_cf = &gA.ddx;
      p.gAy_cf = &gA.ddy;

      p.gB_cf  = &gB.val;
      p.gBx_cf = &gB.ddx;
      p.gBy_cf = &gB.ddy;

      p.xyz[0] = -1.50; p.xyz[1] = -0.45; particles.push_back(p);
    }
      break;

    case 1:
    {
      static radial_shaped_domain_t domain(0.6, DIM(0,0,0), -1, 2, 0.1, 0.0);

      static radial_gamma_t gA(0, gamma_a(), 1);
      static radial_gamma_t gB(-gamma_b(), gamma_b(), 1);

      p.phi_cf   = &domain.phi;
      p.phix_cf  = &domain.phi_x;
      p.phiy_cf  = &domain.phi_y;
      p.kappa_cf = &domain.phi_c;

      p.gA_cf  = &gA.val;
      p.gAx_cf = &gA.ddx;
      p.gAy_cf = &gA.ddy;

      p.gB_cf  = &gB.val;
      p.gBx_cf = &gB.ddx;
      p.gBy_cf = &gB.ddy;

      p.xyz[0] = -1.30; p.xyz[1] = -1.50; particles.push_back(p);
      p.xyz[0] = +1.50; p.xyz[1] = -0.20; particles.push_back(p);
      p.xyz[0] = -0.90; p.xyz[1] = +0.75; particles.push_back(p);
    }
      break;

    default: throw;
  }
}

class vx_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (velocity())
    {
      case -1: return 0;
      case 0: return 1;
      case 1: return 0;
      case 2: return 1;
      case 3: return -sqrt(SQR(x)+SQR(y))*sin(atan2(y,x));
      default:
        throw;
    }
  }
} vx_cf;

class vy_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (velocity())
    {
      case -1: return 0;
      case 0: return 0;
      case 1: return 1;
      case 2: return 1;
      case 3: return sqrt(SQR(x)+SQR(y))*cos(atan2(y,x));
      default:
        throw;
    }
  }
} vy_cf;

class wz_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (rotation())
    {
      case 0: return  0;
      case 1: return  1;
      case 2: return -1;
      case 3: return sqrt(SQR(x)+SQR(y))*cos(atan2(y,x));
      default:
        throw;
    }
  }
} wz_cf;

class gamma_Aa_all_cf_t : public CF_DIM
{
  std::vector<particle_t> *particles;
  CF_DIM *number;
public:
  gamma_Aa_all_cf_t(std::vector<particle_t> &particles, CF_DIM &number)
    : particles(&particles), number(&number) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return particles->at(int((*number)(DIM(x, y, z)))).gA(DIM(x, y, z));
  }
};

class gamma_Ba_all_cf_t : public CF_DIM
{
  std::vector<particle_t> *particles;
  CF_DIM *number;
public:
  gamma_Ba_all_cf_t(std::vector<particle_t> &particles, CF_DIM &number)
    : particles(&particles), number(&number) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return particles->at(int((*number)(DIM(x, y, z)))).gB(DIM(x, y, z));
  }
};

class phi_eff_cf_t : public CF_DIM
{
  CF_DIM *phi_particles;
  CF_DIM *phi_wall;
  CF_DIM *phi_free;
public:
  phi_eff_cf_t(CF_DIM *phi_particles, CF_DIM *phi_wall, CF_DIM *phi_free)
    : phi_particles(phi_particles), phi_wall(phi_wall), phi_free(phi_free) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    return MAX((*phi_particles)(DIM(x,y,z)),
               (*phi_wall)(DIM(x,y,z)),
               (*phi_free)(DIM(x,y,z)));
  }
};


int main(int argc, char** argv)
{
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // parse command line arguments for parameters
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  // define some auxiliary variables
  double scaling = 1./box_size();
  double xyz_min [] = { DIM(xmin(), ymin(), zmin()) };
  double xyz_max [] = { DIM(xmax(), ymax(), zmax()) };
  int    nb_trees[] = { DIM(nx(), ny(), nz()) };
  int    periodic[] = { DIM(px(), py(), pz()) };

  pairwise_potential_width.val *= (xmax()-xmin())/pow(2.,lmax());

  bulk_lamellar_spacing = 2.*pow(8.*XN()/3./pow(PI,4.),1./6.)/box_size();

  // define the output directory
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
  else
  {
    if (mpi.rank() == 0)
    {
      std::ostringstream command;
      command << "mkdir -p " << out_dir;
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
    }
  }

  if (save_vtk() && mpi.rank() == 0)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  // make a separate file with the energies in every iteration (for Matlab plot)
  FILE *file_energy;
  char file_energy_name[1000];
  if (save_energy())
  {
    sprintf(file_energy_name, "%s/convergence.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_energy_name, "w", &file_energy); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_energy, "iteration "
                                                 "effective_energy "
                                                 "expected_dEdt "
                                                 "expected_dE "
                                                 "effective_dE"
                                                 "mu_minus integral\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
  }

  // stopwatch
  parStopWatch w;
  w.start("Running example: scft_particles");

  // ---------------------------------------------------------
  // initialize particles
  // ---------------------------------------------------------
  vector<particle_t> particles;
  initalize_particles(particles);
  int np = particles.size();

  // auxiliary grid variables to interpolate mu_minus from previous time step
  p4est_connectivity_t *connectivity_old;
  p4est_t *p4est_old;
  p4est_ghost_t *ghost_old;
  p4est_nodes_t *nodes_old;

  my_p4est_hierarchy_t *hierarchy_old;
  my_p4est_node_neighbors_t *ngbd_old;

  Vec mu_minus_old;
  Vec mu_plus_old;

  // create the mu_minus field
  mu_minus_cf_t mu_minus_cf;

  int step = 0;
  double t = 0;
  double energy = 0;
  double energy_previous = 0; //needed to calculate the energy difference between two iterations

  double rho_old = 1;

  vector< vector<double> > v_old(P4EST_DIM, vector<double> (np, 0));

  vector<double> w_old  (np, 0);

  vector<double> dt_old (np, 1);
  vector<double> dtw_old(np, 1);

  while (t < time_limit() && step < step_limit())
  {
    // create particle_level_set
    phi_part_cf_t phi_part_cf(particles);

    // create get_particle_num (creates a field, that shows number of closest particle)
    particles_number_cf_t particles_number_cf(particles);

    phi_eff_cf_t phi_eff_cf (&phi_part_cf, &phi_wall_cf, &phi_infc_cf);

    // ---------------------------------------------------------
    // create computational grid
    // ---------------------------------------------------------
    my_p4est_brick_t brick;
    p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
    p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin(), lmax(), &phi_eff_cf, lip(), band());
    data.set_refine_only_inside(true);
    p4est->user_pointer = (void*)(&data);

    // set P4EST_TRUE, because interface is moving (need to refine and partition the grid)
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
    my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
    ngbd->init_neighbors();

    // find the diagonal length of the smallest quadrant (needed to calculate the timestep delta_t)
    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    // ---------------------------------------------------------
    // allocate fields
    // ---------------------------------------------------------
    Vec     mu_minus;
    double *mu_minus_ptr;

    Vec     mu_minus_grad[P4EST_DIM];
    double *mu_minus_grad_ptr[P4EST_DIM];

    Vec     mu_plus;
    double *mu_plus_ptr;

    Vec     phi_part;
    double *phi_part_ptr;

    Vec     phi_wall;
    double *phi_wall_ptr;

    Vec     phi_free;
    double *phi_free_ptr;

    Vec     particles_number;
    double *particles_number_ptr;

    Vec     normal    [P4EST_DIM];
    double *normal_ptr[P4EST_DIM];

    Vec     normal_wall    [P4EST_DIM];
    double *normal_wall_ptr[P4EST_DIM];

    Vec     kappa;
    double *kappa_ptr;

    Vec     gamma_eff;
    double *gamma_eff_ptr;

    Vec     gamma_dif;
    double *gamma_dif_ptr;

    Vec     gamma_avg;
    double *gamma_avg_ptr;

    Vec     gamma_eff_grad    [P4EST_DIM];
    double *gamma_eff_grad_ptr[P4EST_DIM];

    Vec     gamma_dif_grad    [P4EST_DIM];
    double *gamma_dif_grad_ptr[P4EST_DIM];

    Vec     gamma_avg_grad    [P4EST_DIM];
    double *gamma_avg_grad_ptr[P4EST_DIM];

    Vec     velo;
    double *velo_ptr;

    Vec     Gx;
    double *Gx_ptr;

    Vec     Gy;
    double *Gy_ptr;

    Vec     Gxy;
    double *Gxy_ptr;

    ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &mu_plus);             CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &phi_part);            CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &phi_wall);            CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &phi_free);            CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &particles_number);    CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &kappa);               CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_eff);           CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_avg);           CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_dif);           CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &velo);                CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &Gx);                  CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &Gy);                  CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &Gxy);                 CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus_grad[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(mu_minus_grad[dim], &normal[dim]);         CHKERRXX(ierr);
      ierr = VecDuplicate(mu_minus_grad[dim], &normal_wall[dim]);    CHKERRXX(ierr);
      ierr = VecDuplicate(mu_minus_grad[dim], &gamma_eff_grad[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(mu_minus_grad[dim], &gamma_dif_grad[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(mu_minus_grad[dim], &gamma_avg_grad[dim]); CHKERRXX(ierr);
    }

    // ---------------------------------------------------------
    // initialize mu_minus field
    // ---------------------------------------------------------
    if (step == 0 || !use_scft())
    {
      sample_cf_on_nodes(p4est, nodes, mu_minus_cf, mu_minus);
    }
    else
    {
      // by interpolating the mu_minus field, the calculation is accelerated AND it 'stays on the same track' (since there are multiple solutions in scft)
      my_p4est_interpolation_nodes_t interp(ngbd_old);

      double xyz[P4EST_DIM];
      foreach_node(n, nodes)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }

      interp.set_input(mu_minus_old, linear);
      interp.interpolate(mu_minus);

      interp.set_input(mu_plus_old, linear);
      interp.interpolate(mu_plus);

      // destroy previous grid
      ierr = VecDestroy(mu_plus_old);  CHKERRXX(ierr);
      ierr = VecDestroy(mu_minus_old); CHKERRXX(ierr);

      delete hierarchy_old;
      delete ngbd_old;
      p4est_nodes_destroy(nodes_old);
      p4est_ghost_destroy(ghost_old);
      p4est_destroy      (p4est_old);
      p4est_connectivity_destroy(connectivity_old);
    }

    // ---------------------------------------------------------
    // sample particles geometry
    // ---------------------------------------------------------
    sample_cf_on_nodes(p4est, nodes, phi_part_cf, phi_part);
    sample_cf_on_nodes(p4est, nodes, particles_number_cf,    particles_number);
    sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);
    sample_cf_on_nodes(p4est, nodes, phi_infc_cf, phi_free);

//    my_p4est_level_set_t ls(ngbd);
//    ls.reinitialize_2nd_order(phi_part, 50);

    double rho = 1;

    // calculate mu_minus and velo
    if (use_scft())
    {
      my_p4est_scft_t scft(ngbd, ns());

      gamma_Aa_all_cf_t gamma_Aa_all(particles, particles_number_cf);
      gamma_Ba_all_cf_t gamma_Ba_all(particles, particles_number_cf);
      // set geometry
      scft.add_boundary(phi_part, MLS_INTERSECTION, gamma_Aa_all, gamma_Ba_all);

      scft.set_scalling(scaling);
      scft.set_polymer(f(), XN());
      scft.set_rho_avg(rho_old);

      // initialize potentials
      Vec mu_minus_scft = scft.get_mu_m();
      Vec mu_plus_scft  = scft.get_mu_p();
      VecCopyGhost(mu_minus, mu_minus_scft);
//      VecCopyGhost(mu_plus,  mu_plus_scft);

      // initialize diffusion solvers for propagators
      scft.initialize_solvers();
      scft.initialize_bc_smart(step != 0);

      // main loop for solving SCFT equations
      int    scft_iteration = 0;
      int    bc_iters       = 0;
      double scft_error     = 2.*scft_tol()+1.;
      double scft_energy    = 0;

      while ((scft_iteration < max_scft_iterations() && scft_error > scft_tol()) || scft_iteration < bc_adjust_min()+1)
      {
        // do an SCFT step
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();

        if (scft.get_exchange_force() < scft_bc_tol() && bc_iters >= bc_adjust_min())
        {
          scft.initialize_bc_smart();
          if (smooth_pressure())
          {
            scft.smooth_singularity_in_pressure_field();
            smooth_pressure.val = false;
          }
          bc_iters = 0;
        }
        ierr = PetscPrintf(mpi.comm(), "SCFT iteration no. %4d, Energy: %+6.5e, Pressure: %3.2e, Exchange: %3.2e, dE scft: %+3.2e, dE: %+3.2e%s\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), scft.get_energy()-scft_energy, scft.get_energy()-energy_previous, bc_iters == 0 ? " (new bc)" : ""); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        scft_iteration++;
        bc_iters++;
        scft_energy = scft.get_energy();
//        scft.save_VTK(scft_iteration);
      }

      // get energy
      energy = scft.get_energy();
      rho    = scft.get_rho_avg();
      rho_old = rho;

      scft.sync_and_extend();
      scft.compute_energy_shape_derivative(0, velo); //second part of dE/dt, which is derivative of integral of f dx

      VecCopyGhost(mu_minus_scft, mu_minus);
      VecCopyGhost(mu_plus_scft,  mu_plus);
    }
    else
    {
      VecSetGhost(velo, 0);
    }

    // calculate 'gamma_eff' by hand and not using 'sample_cf_on_nodes' (because mu_m is a vetor and not continuous function)
    ierr = VecGetArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    penalization_cf_t penalization(particles);

    foreach_node(n, nodes)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      int i = int(particles_number_ptr[n]);
      double ga = particles[i].gA(xyz[0], xyz[1]);
      double gb = particles[i].gB(xyz[0], xyz[1]);

      gamma_eff_ptr[n] = (0.5*(ga+gb)*rho + (ga-gb)*(mu_minus_ptr[n])/XN());// + penalization.value(xyz);
      gamma_avg_ptr[n] = (0.5*(ga+gb)*rho);
      gamma_dif_ptr[n] = ((ga-gb));
    }

    ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    if (!use_scft())
    {
      energy = integrate_over_interface(p4est, nodes, phi_part, gamma_eff);
      energy /= pow(scaling, P4EST_DIM-1.);
    }

    // ---------------------------------------------------------
    // calculate energy derivative with respect to particles' positions
    // ---------------------------------------------------------

    // compute normals, curvature and gradient of gamma_eff (needed for dE/dt formula)
    compute_normals(*ngbd, phi_wall, normal_wall);
    compute_normals_and_mean_curvature(*ngbd, phi_part, normal, kappa);
    ngbd->first_derivatives_central(gamma_eff, gamma_eff_grad);
    ngbd->first_derivatives_central(gamma_dif, gamma_dif_grad);
    ngbd->first_derivatives_central(gamma_avg, gamma_avg_grad);

    ngbd->first_derivatives_central(mu_minus, mu_minus_grad);

    // calculate G
    ierr = VecGetArray(Gx,               &Gx_ptr);               CHKERRXX(ierr);
    ierr = VecGetArray(Gy,               &Gy_ptr);               CHKERRXX(ierr);
    ierr = VecGetArray(Gxy,              &Gxy_ptr);              CHKERRXX(ierr);
    ierr = VecGetArray(velo,             &velo_ptr);             CHKERRXX(ierr);
    ierr = VecGetArray(mu_minus,         &mu_minus_ptr);         CHKERRXX(ierr);
    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecGetArray(mu_minus_grad[dim],  &mu_minus_grad_ptr[dim]);  CHKERRXX(ierr);
    }

    double factor = 1./pow(scaling, P4EST_DIM-1.);

    foreach_node(n, nodes)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      int i = int(particles_number_ptr[n]);

      double phix = particles[i].phix (DIM(xyz[0], xyz[1], xyz[2]));
      double phiy = particles[i].phiy (DIM(xyz[0], xyz[1], xyz[2]));
      double phic = particles[i].kappa(DIM(xyz[0], xyz[1], xyz[2]));

      double norm = ABSD(phix, phiy, phiz);
      EXECD(phix /= norm, phiy /= norm, phiz /= norm);

      double ga  = particles[i].gA (DIM(xyz[0], xyz[1], xyz[2]));
      double gax = particles[i].gAx(DIM(xyz[0], xyz[1], xyz[2]));
      double gay = particles[i].gAy(DIM(xyz[0], xyz[1], xyz[2]));

      double gb  = particles[i].gB (DIM(xyz[0], xyz[1], xyz[2]));
      double gbx = particles[i].gBx(DIM(xyz[0], xyz[1], xyz[2]));
      double gby = particles[i].gBy(DIM(xyz[0], xyz[1], xyz[2]));

      double g_eff  = (.5*(ga+gb)*rho + mu_minus_ptr[n]/XN()*(ga-gb));
      double gx_eff = (.5*(gax+gbx)*rho + mu_minus_ptr[n]/XN()*(gax-gbx) + mu_minus_grad_ptr[0][n]/XN()*(ga-gb));
      double gy_eff = (.5*(gay+gby)*rho + mu_minus_ptr[n]/XN()*(gay-gby) + mu_minus_grad_ptr[1][n]/XN()*(ga-gb));

      double g = SUMD(phix*gx_eff,
                      phiy*gy_eff,
                      phiz*gz_eff)*factor
          + g_eff*phic*factor
          + velo_ptr[n];

      Gx_ptr[n] = g*phix - (.5*(gax+gbx)*rho + mu_minus_ptr[n]/XN()*(gax-gbx))*factor;
      Gy_ptr[n] = g*phiy - (.5*(gay+gby)*rho + mu_minus_ptr[n]/XN()*(gay-gby))*factor;


      double delx = xyz[0]-particles[i].xyz[0];
      double dely = xyz[1]-particles[i].xyz[1];

      Gxy_ptr[n] = Gx_ptr[n]*dely - Gy_ptr[n]*delx;
    }

    ierr = VecRestoreArray(Gx,               &Gx_ptr);               CHKERRXX(ierr);
    ierr = VecRestoreArray(Gy,               &Gy_ptr);               CHKERRXX(ierr);
    ierr = VecRestoreArray(Gxy,              &Gxy_ptr);              CHKERRXX(ierr);
    ierr = VecRestoreArray(velo,             &velo_ptr);             CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus,         &mu_minus_ptr);         CHKERRXX(ierr);
    ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecRestoreArray(mu_minus_grad[dim],  &mu_minus_grad_ptr[dim]);  CHKERRXX(ierr);
    }

    // calculate
    vector< vector<double> > g(P4EST_DIM, vector<double> (np,0));

    vector<double> gw(np, 0);

    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_part, &phi_part_ptr); CHKERRXX(ierr);

    my_p4est_interpolation_nodes_local_t interp_local(ngbd);
    double proximity_crit = 50.*diag_min;
    energy = 0;
    foreach_local_node(n, nodes)
    {
      if (fabs(phi_part_ptr[n]) < proximity_crit)
      {
        int i = int(particles_number_ptr[n]);

        // get area and centroid of the interface
        double area_part = 0;
        double area_wall = 0;
        double area_free = 0;
        double xyz_part[P4EST_DIM];
        double xyz_wall[P4EST_DIM];
        double xyz_free[P4EST_DIM];

        phi_true_cf_t PHI(&particles[i]);
        vector<CF_DIM *> phi_cf(3);
        phi_cf[0] = &PHI;
        phi_cf[1] = &phi_wall_cf;
        phi_cf[2] = &phi_infc_cf;

        vector<mls_opn_t> opn(3, MLS_INT);
        my_p4est_finite_volume_t fv;
        construct_finite_volume(fv, n, p4est, nodes, phi_cf, opn, 1, 1);

        for (int j = 0; j < fv.interfaces.size(); ++j)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);

          switch (fv.interfaces[j].id)
          {
            case 0:
              xyz_part[0]  = xyz[0] + fv.interfaces[j].centroid[0];
              xyz_part[1]  = xyz[1] + fv.interfaces[j].centroid[1];
              area_part    = fv.interfaces[j].area;
              break;
            case 1:
              xyz_wall[0]  = xyz[0] + fv.interfaces[j].centroid[0];
              xyz_wall[1]  = xyz[1] + fv.interfaces[j].centroid[1];
              area_wall    = fv.interfaces[j].area;
              break;
            case 2:
              xyz_free[0]  = xyz[0] + fv.interfaces[j].centroid[0];
              xyz_free[1]  = xyz[1] + fv.interfaces[j].centroid[1];
              area_free    = fv.interfaces[j].area;
              break;
            default: throw;
          }
        }

        if (area_part != 0)
        {
          double phix = particles[i].phix (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          double phiy = particles[i].phiy (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));

          double norm = ABSD(phix, phiy, phiz);
          EXECD(phix /= norm, phiy /= norm, phiz /= norm);

          double dist = particles[i].phi (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));

          xyz_part[0] -= dist*phix/norm;
          xyz_part[1] -= dist*phiy/norm;

          phix = particles[i].phix (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          phiy = particles[i].phiy (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));

          norm = ABSD(phix, phiy, phiz);
          EXECD(phix /= norm, phiy /= norm, phiz /= norm);

          interp_local.initialize(n);

          interp_local.set_input(velo,             linear); double velo_val = interp_local.value(xyz_part);
          interp_local.set_input(mu_minus,         linear); double mu_m     = interp_local.value(xyz_part);
          interp_local.set_input(mu_minus_grad[0], linear); double mux_m    = interp_local.value(xyz_part);
          interp_local.set_input(mu_minus_grad[1], linear); double muy_m    = interp_local.value(xyz_part);

          double phic = particles[i].kappa(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));

          double ga  = particles[i].gA (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          double gax = particles[i].gAx(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          double gay = particles[i].gAy(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));

          double gb  = particles[i].gB (DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          double gbx = particles[i].gBx(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          double gby = particles[i].gBy(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));


          double g_eff  = (.5*(ga+gb)*rho + mu_m/XN()*(ga-gb));
          double gx_eff = (.5*(gax+gbx)*rho + mu_m/XN()*(gax-gbx) + mux_m/XN()*(ga-gb));
          double gy_eff = (.5*(gay+gby)*rho + mu_m/XN()*(gay-gby) + muy_m/XN()*(ga-gb));

          double G = SUMD(phix*gx_eff,
                          phiy*gy_eff,
                          phiz*gz_eff)*factor
                     + g_eff*phic*factor
                     + velo_val;

          // repulsion between particles, walls and free surface
          for (int j = 0; j < np; ++j)
          {
            if (j != i)
            {
              double dist_other = particles[j].phi(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
              if (fabs(dist_other) < pairwise_potential_width())
              {
                double phix_other = particles[j].phix(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
                double phiy_other = particles[j].phiy(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
                dist_other = fabs(dist_other)-pairwise_potential_width();
                G += phic*pairwise_potential(dist_other) + pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, FIX);

                double add_x = -pairwise_force(dist_other)*phix_other;
                double add_y = -pairwise_force(dist_other)*phiy_other;

                double delx = xyz_part[0]-particles[j].xyz[0];
                double dely = xyz_part[1]-particles[j].xyz[1];

                g[0][j] += add_x*area_part;
                g[1][j] += add_y*area_part;
                gw[j]   += (add_x*dely - add_y*delx)*area_part;
                energy  += pairwise_potential(dist_other)*area_part;

              }
            }
          }

          // wall repulsion
          double dist_other = phi_wall_cf(DIM(xyz_part[0], xyz_part[1], xyz_part[2]));
          if (fabs(dist_other) < pairwise_potential_width())
          {
            interp_local.set_input(normal_wall[0], linear); double phix_other = interp_local.value(xyz_part);
            interp_local.set_input(normal_wall[1], linear); double phiy_other = interp_local.value(xyz_part);
            dist_other = fabs(dist_other)-pairwise_potential_width();
            G += phic*pairwise_potential(dist_other) + pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, FIX);
          }

          double Gx = G*phix - (.5*(gax+gbx)*rho + mu_m/XN()*(gax-gbx))*factor;
          double Gy = G*phiy - (.5*(gay+gby)*rho + mu_m/XN()*(gay-gby))*factor;

          double delx = xyz_part[0]-particles[i].xyz[0];
          double dely = xyz_part[1]-particles[i].xyz[1];

          double Gxy = Gx*dely - Gy*delx;

          g[0][i] += Gx*area_part;
          g[1][i] += Gy*area_part;
          gw[i]   += Gxy*area_part;
          energy  += g_eff*area_part;

        }
      }
    }
    ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_part, &phi_part_ptr); CHKERRXX(ierr);

    ierr = MPI_Allreduce(MPI_IN_PLACE, g[0].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, g[1].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, gw.data(),   np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &energy,   1, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);



//        integrate_over_interface(np, g[0], p4est, nodes, phi_part, particles_number, Gx);
//        integrate_over_interface(np, g[1], p4est, nodes, phi_part, particles_number, Gy);
//        integrate_over_interface(np, gw,   p4est, nodes, phi_part, particles_number, Gxy);

    // ---------------------------------------------------------
    // select velocity and move particles
    // ---------------------------------------------------------
    vector< vector<double> > v(P4EST_DIM, vector<double> (np,0));
    vector<double> w(np, 0);

    if (minimize())
    {
      for(int i = 0; i < np; i++)
      {
        v[0][i] = -g[0][i];
        v[1][i] = -g[1][i];
        w[i] = -gw[i];
      }
    }
    else
    {
      for(int i = 0; i < np; i++)
      {
        v[0][i] = vx_cf(particles[i].xyz[0], particles[i].xyz[1]);
        v[1][i] = vy_cf(particles[i].xyz[0], particles[i].xyz[1]);
        w[i]    = wz_cf(particles[i].xyz[0], particles[i].xyz[1]);
      }
    }

    // find the maximum velocity (needed to calculate the timestep delta_t)

    vector<double> arm_max(np,0);

    double phi_thresh = 0;

    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_part, &phi_part_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      if (phi_part_ptr[n] > phi_thresh)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);

        int       i = int(particles_number_ptr[n]);
        double delx = xyz[0]-particles[i].xyz[0];
        double dely = xyz[1]-particles[i].xyz[1];

        arm_max[i] = MAX(arm_max[i], ABSD( dely,  delx, FIX THIS FOR 3D ));
      }
    }

    ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_part, &phi_part_ptr); CHKERRXX(ierr);

    ierr = MPI_Allreduce(MPI_IN_PLACE, arm_max.data(), np, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); CHKERRXX(ierr);

    double v_max  = 0;
    double wr_max = 0;
    for (int i = 0; i < np; ++i)
    {
//      v[0][i] = v[0][i];
//      v[1][i] = v[1][i];
//      w[i] = w[i];
      v_max  = MAX(v_max, ABSD(v[0][i], v[1][i], v[2][i]));
      wr_max = MAX(wr_max, fabs(w[i])*arm_max[i]);
    }

    // calculate timestep delta_t, such that it's small enough to capture 'everything'
    double delta_t  = diag_min*CFL()/MAX(v_max, 0.001);
    double delta_tw = diag_min*CFL()/MAX(wr_max,0.001);

    vector<double> dt(np);
    vector<double> dtw(np);

    // move particles
    for (int j = 0; j < np; ++j)
    {
      if (!minimize())
      {
        dt [j] = delta_t;
        dtw[j] = delta_tw;
      }
      else
      {
        double gxx = -(v[0][j] - v_old[0][j])/dt_old[j]/v_old[0][j];
        double gyy = -(v[1][j] - v_old[1][j])/dt_old[j]/v_old[1][j];
        double gxy = -(v[1][j] - v_old[1][j])/dt_old[j]/v_old[0][j];
        double gyx = -(v[0][j] - v_old[0][j])/dt_old[j]/v_old[1][j];

        double dt_tmp = -( g[0][j]*v[0][j] + g[1][j]*v[1][j] )/( gxx*v[0][j]*v[0][j] + .5*(gxy+gyx)*v[0][j]*v[1][j] + gyy*v[1][j]*v[1][j] );

        double dt_max = diag_min*CFL()/ABSD(v[0][j], v[1][j], v[2][j]);

        if (dt_tmp < 0 || dt_tmp > dt_max || step == 0) dt[j] = dt_max;
        else dt[j] = dt_tmp;

        double gww = -(w[j] - w_old[j])/dtw_old[j]/w_old[j];
        double dtw_tmp = - gw[j]/gww/w[j];

        double dtw_max = diag_min*CFL()/fabs(w[j]*arm_max[j]);

        if (dtw_tmp < 0 || dtw_tmp > dtw_max || step == 0) dtw[j] = dtw_max;
        else dtw[j] = dtw_tmp;

        dt[j] = dt_max;
        dtw[j] = dtw_max;
      }

      XCODE( particles[j].xyz[0] += v[0][j]*dt[j] );
      YCODE( particles[j].xyz[1] += v[1][j]*dt[j] );
      ZCODE( particles[j].xyz[2] += v[2][j]*dt[j] );

      particles[j].rot += w[j]*dtw[j];
    }

    v_old = v;
    dt_old = dt;
    w_old = w;
    dtw_old = dtw;

    if (px())
    {
      for (int j = 0; j < np; ++j)
      {
        if (particles[j].xyz[0] < xmin()) particles[j].xyz[0] += xmax()-xmin();
        if (particles[j].xyz[0] > xmax()) particles[j].xyz[0] -= xmax()-xmin();
      }
    }

    if (py())
    {
      for (int j = 0; j < np; ++j)
      {
        if (particles[j].xyz[1] < ymin()) particles[j].xyz[1] += ymax()-ymin();
        if (particles[j].xyz[1] > ymax()) particles[j].xyz[1] -= ymax()-ymin();
      }
    }

    // calculate the expected and effective changes in energy (compare step i with step i+1)
    // compute expected change in energy (for validation)
    double dE_dt_expected = 0;
    for (int i = 0; i < np; i++)
    {
      dE_dt_expected += SUMD(v[0][i]*g[0][i],
                             v[1][i]*g[1][i],
                             v[2][i]*g[2][i])*dt[i];

      dE_dt_expected += w[i]*gw[i]*dtw[i];
    }
    double expected_change_in_energy  = dE_dt_expected;
    double effective_change_in_energy = energy - energy_previous;

    ierr = PetscPrintf(mpi.comm(), "Iteration no. %4d: time = %3.2e, dt = %3.2e, vmax = %3.2e, E = %6.5e, dE = %+3.2e / %+3.2e \n", step, t, delta_t, v_max, energy, expected_change_in_energy, effective_change_in_energy); CHKERRXX(ierr);

    // ---------------------------------------------------------
    // save info into files
    // ---------------------------------------------------------

    double mu_minus_integral;
    mu_minus_integral = integrate_over_interface(p4est, nodes, phi_part, mu_minus);


    // write the energy in every iteration into the separate file
    if (save_energy())
    {
      ierr = PetscFOpen(mpi.comm(), file_energy_name, "a", &file_energy); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), file_energy, "%d %12.11e %3.2e %3.2e %3.2e %f\n", step, energy, dE_dt_expected, expected_change_in_energy, effective_change_in_energy, mu_minus_integral);
      ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
    }

    // save as vtk files
    if (save_vtk() && (step%save_freq()==0))
    {
      std::ostringstream oss;
      oss << out_dir
          << "/vtu/scft_"
          << p4est->mpisize
          << "_" << int(round(step/save_freq()));

      ierr = VecGetArray(phi_part, &phi_part_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             6, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_part_ptr,
                             VTK_POINT_DATA, "particles_number", particles_number_ptr,
                             VTK_POINT_DATA, "gamma_effective", gamma_eff_ptr,
                             VTK_POINT_DATA, "gamma_diff", gamma_dif_ptr,
                             VTK_POINT_DATA, "gamma_avg", gamma_avg_ptr,
                             VTK_POINT_DATA, "mu_minus", mu_minus_ptr);

      ierr = VecRestoreArray(phi_part, &phi_part_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
    }

    // ---------------------------------------------------------
    // clean-up memory
    // ---------------------------------------------------------
    ierr = VecDestroy(particles_number);    CHKERRXX(ierr);
    ierr = VecDestroy(kappa);               CHKERRXX(ierr);
    ierr = VecDestroy(gamma_eff);           CHKERRXX(ierr);
    ierr = VecDestroy(gamma_dif);           CHKERRXX(ierr);
    ierr = VecDestroy(gamma_avg);           CHKERRXX(ierr);
    ierr = VecDestroy(velo);                CHKERRXX(ierr);
    ierr = VecDestroy(Gx);                  CHKERRXX(ierr);
    ierr = VecDestroy(Gy);                  CHKERRXX(ierr);
    ierr = VecDestroy(Gxy);                 CHKERRXX(ierr);
    ierr = VecDestroy(phi_part); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecDestroy(gamma_eff_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(gamma_dif_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(gamma_avg_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }

    if (use_scft())
    {
      mu_minus_old = mu_minus;
      mu_plus_old  = mu_plus;

      connectivity_old = connectivity;
      p4est_old        = p4est;
      ghost_old        = ghost;
      nodes_old        = nodes;
      hierarchy_old    = hierarchy;
      ngbd_old         = ngbd;
    }
    else
    {
      ierr = VecDestroy(mu_plus); CHKERRXX(ierr);
      ierr = VecDestroy(mu_minus); CHKERRXX(ierr);

      delete hierarchy;
      delete ngbd;
      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);
      p4est_destroy      (p4est);
      p4est_connectivity_destroy(connectivity);
    }

    t += delta_t;
    step++;

    // store energy to compare with next step
    energy_previous = energy;
  }

  if (use_scft())
  {
    ierr = VecDestroy(mu_plus_old); CHKERRXX(ierr);
    ierr = VecDestroy(mu_minus_old); CHKERRXX(ierr);

    delete hierarchy_old;
    delete ngbd_old;
    p4est_nodes_destroy(nodes_old);
    p4est_ghost_destroy(ghost_old);
    p4est_destroy      (p4est_old);
    p4est_connectivity_destroy(connectivity_old);
  }

  // show total running time
  w.stop(); w.read_duration();

  return 0;
}
