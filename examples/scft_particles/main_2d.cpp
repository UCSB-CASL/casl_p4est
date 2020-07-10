
/*
 * Test the cell based multi level-set p4est.
 * Intersection of two circles
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_scft.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_scft.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/parameter_list.h>

#undef MIN
#undef MAX

using namespace std;
param_list_t pl;

/* TODO:
 * + saving run parameters
 * - saving temporal data
 * - predict change in energy
 * + use characteristic lengths for seeds
 * + level-set function for well
 * - level-set function for film+drop
 * - properly compute energy in the pure-curvature case
 * + compute limiting contact angles for A and B blocks
 * - define several examples
 */

// comptational domain
param_t<double> xmin (pl, -1, "xmin", "xmin");
param_t<double> ymin (pl, -1, "ymin", "ymin");
param_t<double> zmin (pl, -1, "zmin", "zmin");
param_t<double> xmax (pl,  1, "xmax", "xmax");
param_t<double> ymax (pl,  1, "ymax", "ymax");
param_t<double> zmax (pl,  1, "zmax", "zmax");

param_t<bool> px (pl, 0, "px", "periodicity in x-dimension 0/1");
param_t<bool> py (pl, 0, "py", "periodicity in y-dimension 0/1");
param_t<bool> pz (pl, 0, "pz", "periodicity in z-dimension 0/1");

param_t<int> nx (pl, 1, "nx", "number of trees in x-dimension");
param_t<int> ny (pl, 1, "ny", "number of trees in y-dimension");
param_t<int> nz (pl, 1, "nz", "number of trees in z-dimension");

// grid parameters
#ifdef P4_TO_P8
param_t<int> lmin (pl, 7, "lmin", "min level of trees");
param_t<int> lmax (pl, 7, "lmax", "max level of trees");
#else
param_t<int> lmin (pl, 7, "lmin", "min level of trees");
param_t<int> lmax (pl, 7, "lmax", "max level of trees");
#endif
param_t<double> lip (pl, 1.1, "lip", "Lipschitz constant");
param_t<int>    band  (pl, 2,   "band" , "Uniform grid band");
param_t<bool>   refine_only_inside (pl, 1, "refine_only_inside", "Refine only inside");


// advection parameters
param_t<double> cfl                     (pl, 1.0,    "cfl", "CFL number");
param_t<bool>   use_neumann             (pl, 1,      "use_neumann", "Impose contact angle use Neumann BC 0/1");
param_t<bool>   compute_exact           (pl, 0,      "compute_exact", "Compute exact final shape (only for pure-curvature) 0/1");
param_t<int>    contact_angle_extension (pl, 0,      "contact_angle_extension", "Method for extending level-set function into wall: 0 - constant angle, 1 - , 2 - special");
param_t<int>    volume_corrections      (pl, 2,      "volume_corrections", "Number of volume correction after each move");
param_t<int>    max_iterations          (pl, 1000,   "max_iterations", "Maximum number of advection steps");
param_t<double> tolerance               (pl, 1.0e-8, "tolerance", "Stopping criteria");

interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

// scft parameters
param_t<bool>   use_scft            (pl, 1,     "use_scft", "Turn on/off SCFT 0/1");
param_t<bool>   smooth_pressure     (pl, 1,     "smooth_pressure", "Smooth pressure after first BC adjustment 0/1");
param_t<int>    max_scft_iterations (pl, 100,   "max_scft_iterations", "Maximum SCFT iterations");
param_t<int>    bc_adjust_min       (pl, 1,     "bc_adjust_min", "Minimun SCFT steps between adjusting BC");
param_t<double> scft_tol            (pl, 1.e-4, "scft_tol", "Tolerance for SCFT");
param_t<double> scft_bc_tol         (pl, 1.e-3, "scft_bc_tol", "Tolerance for adjusting BC");

// polymer
param_t<double> box_size (pl, 2.34, "box_size", "Box size in units of Rg");
param_t<double> f        (pl, .5, "f", "Fraction of polymer A");
param_t<double> XN       (pl, 20, "XN", "Flory-Higgins interaction parameter");
param_t<int>    ns       (pl, 40, "ns", "Discretization of polymer chain");

// output parameters
param_t<bool> save_vtk        (pl, 1, "save_vtk", "");
param_t<bool> save_parameters (pl, 1, "save_parameters", "");
param_t<bool> save_data       (pl, 1, "save_data", "");
param_t<int>  save_every_dn   (pl, 1, "save_every_dn", ""); // for vtk

// problem setting

param_t<int>    geometry_ptcl (pl, 2, "geometry_ptcl", "Initial placement of particles: 0 - one particle, 1 - ...");
param_t<int>    geometry_free (pl, 0, "geometry_free", "Initial polymer shape: 0 - drop (pl, 1 - film (pl, 2 - combination");
param_t<int>    geometry_wall (pl, 0, "geometry_wall", "Wall geometry: 0 - no wall (pl, 1 - wall (pl, 2 - well");

param_t<bool>   minimize (pl, 0, "minimize", "Turn on/off energy minimization (0/1)");
param_t<int>    velocity (pl, 3, "velocity", "Predifined velocity in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<int>    rotation (pl, 1, "rotation", "Predifined rotation in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");

param_t<int>    wall_pattern (pl, 0, "wall_pattern", "Wall chemical pattern: 0 - no pattern");
param_t<int>    n_seed       (pl, 2, "n_seed", "Seed: 0 - zero, 1 - random, 2 - horizontal stripes, 3 - vertical stripes, 4 - dots");
param_t<int>    n_example    (pl, 0, "n_example", "Number of predefined example");

param_t<int>    pairwise_potential_type  (pl, 0, "pairwise_potential_type", "Type of pairwise potential: 0 - quadratic, 1 - 1/(e^x-1)");
param_t<double> pairwise_potential_mag   (pl, 1.0,   "pairwise_potential_mag", "Magnitude of pairwise potential");
param_t<double> pairwise_potential_width (pl, 5, "pairwise_potential_width", "Width of pairwise potential");

// surface energies
param_t<double> gamma_a  (pl, 0, "gamma_a" , "Surface tension of A block");
param_t<double> gamma_b  (pl, 2, "gamma_b" , "Surface tension of B block");

param_t<int> wall_energy_type (pl, 1, "wall_energy_type", "Method for setting wall surface energy: 0 - explicitly (i.e. convert XN to angles) (pl, 1 - through contact angles (i.e. convert angles to XN)");

param_t<double> XN_air_avg (pl, 2.0, "XN_air_avg", "Polymer-air surface energy strength: average");
param_t<double> XN_air_del (pl, 0.5, "XN_air_del", "Polymer-air surface energy strength: difference");

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
param_t<double> wall_y   (pl, -.3,  "wall_y", "");
param_t<double> wall_z   (pl, .0,   "wall_z", "");
param_t<double> wall_r   (pl, 2.5,   "wall_r", "");

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

void set_parameters()
{
  switch (n_example())
  {
    case 0:
      break;
    default:
      throw std::invalid_argument("Invalid exmaple number.\n");
  }
}

class gamma_Aa_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(XN_air_avg()+XN_air_del());
  }
} gamma_Aa_cf;

class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(XN_air_avg()-XN_air_del());
  }
} gamma_Ba_cf;

class gamma_Aw_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (wall_pattern())
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
    switch (wall_pattern())
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
    switch (geometry_wall())
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
    }
  }
} phi_wall_cf;

class phi_free_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (geometry_free())
    {
      case 0: return -10;
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
} phi_free_cf;

inline double lam_bulk_period()
{
  return 2.*pow(8.*XN()/3./pow(PI,4.),1./6.)/box_size();
}

class mu_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z) ) const
  {
    switch (n_seed())
    {
      case 0: return 0;
      case 1: return 0.01*XN()*(double)(rand()%1000)/1000.;
      case 2: {
        double nx = (xmax()-xmin())/lam_bulk_period(); if (px() == 1) nx = round(nx);
        return .5*XN()*cos(2.*PI*x/(xmax()-xmin())*nx);
      }
      case 3: {
        double ny = (ymax()-ymin())/lam_bulk_period(); if (py() == 1) ny = round(ny);
        return .5*XN()*cos(2.*PI*y/(ymax()-ymin())*ny);
      }
      case 4: {
        double nx = .5*(xmax()-xmin())/lam_bulk_period(); if (px() == 1) nx = round(nx);
        double ny = .5*(ymax()-ymin())/lam_bulk_period(); if (py() == 1) ny = round(ny);
#ifdef P4_TO_P8
        double nz = .5*(zmax()-zmin())/lam_bulk_period(); if (pz() == 1) nz = round(nz);
#endif
        return .5*XN()*MULTD( cos(2.*PI*x/(xmax()-xmin())*nx),
                              cos(2.*PI*y/(ymax()-ymin())*ny),
                              cos(2.*PI*z/(zmax()-zmin())*nz));
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} mu_cf;

inline void interpolate_between_grids(my_p4est_interpolation_nodes_t &interp, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, Vec &vec, Vec parent=NULL, interpolation_method method=quadratic_non_oscillatory_continuous_v2)
{
  PetscErrorCode ierr;
  Vec tmp;

  if (parent == NULL) {
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &tmp); CHKERRXX(ierr);
  } else {
    ierr = VecDuplicate(parent, &tmp); CHKERRXX(ierr);
  }

  interp.set_input(vec, method);
  interp.interpolate(tmp);

  ierr = VecDestroy(vec); CHKERRXX(ierr);
  vec = tmp;
}


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
class phi_ptcl_cf_t : public CF_DIM
{
private:
  std::vector<particle_t> *particles;

public:

  phi_ptcl_cf_t(std::vector<particle_t> &particles)
    : particles(&particles) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    int np = particles->size();
    vector<double> phi_ptcl(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_ptcl[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_ptcl[n] = MAX(phi_ptcl[n],
                                   particles->at(n).phi(DIM(x+double(i)*(xmax()-xmin()),
                                                            y+double(j)*(ymax()-ymin()),
                                                            z+double(k)*(zmax()-zmin()))));
          }
    }

    double current_max = -10;
    for(int n = 0; n < np; n++)
    {
      current_max = MAX(current_max, phi_ptcl[n]);
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
    vector<double> phi_ptcl(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_ptcl[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_ptcl[n] = MAX(phi_ptcl[n],
                                   particles->at(n).phi(DIM(x+double(i)*(xmax()-xmin()),
                                                            y+double(j)*(ymax()-ymin()),
                                                            z+double(k)*(zmax()-zmin()))));
          }
    }

    int particle_num = 0;
    for (int n = 1; n < np; n++)
    {
      if (phi_ptcl[particle_num] < phi_ptcl[n])
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
    vector<double> phi_ptcl(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_ptcl[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_ptcl[n] = MAX(phi_ptcl[n],
                                   particles->at(n).phi(DIM(x+double(i)*(xmax()-xmin()),
                                                            y+double(j)*(ymax()-ymin()),
                                                            z+double(k)*(zmax()-zmin()))));
          }
    }

    int particle_num = 0;
    for (int n = 1; n < np; n++)
    {
      if (phi_ptcl[particle_num] < phi_ptcl[n])
      {
        particle_num = n;
      }
    }

    double sum = 0;


    for (int n = 0; n < np; n++)
    {
      if (n != particle_num)
      {
        sum += pairwise_potential(fabs(phi_ptcl[n])-pairwise_potential_width());
      }
    }

    return sum;
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

void initalize_ptcl(std::vector<particle_t> &particles)
{
  particles.clear();
  particle_t p;
  switch (geometry_ptcl())
  {
    case 0: break;
    case 1:
    {
//      static radial_shaped_domain_t domain(0.2, DIM(0,0,0), -1, 3, 0.1, 0.0);
      static radial_shaped_domain_t domain(0.2, DIM(0,0,0), -1, 3, 0.0, 0.0);
//      static capsule_domain_t domain(0.2, DIM(0,0,0), 0.0, -1);

      static radial_gamma_t gA(gamma_a(), gamma_a(), 1);
      static radial_gamma_t gB(gamma_b(), gamma_b(), 1);

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

      p.xyz[0] = -0.27; p.xyz[1] = -0.23; particles.push_back(p);
    }
      break;

    case 2:
    {
      static radial_shaped_domain_t sphere(0.2, DIM(0,0,0), -1, 0, 0.0, 0.0);
      static radial_shaped_domain_t star(0.2, DIM(0,0,0), -1, 3, 0.3, 0.0);
      static capsule_domain_t capsule(0.1, DIM(0,0,0), 0.3, -1);

      static radial_gamma_t sphere_gA(0, gamma_a(), 1);
      static radial_gamma_t sphere_gB(-gamma_b(), gamma_b(), 1);

      static radial_gamma_t capsule_gA(0, gamma_a(), 0);
      static radial_gamma_t capsule_gB(-gamma_b(), gamma_b(), 0);

      static radial_gamma_t star_gA(0, gamma_a(), 3);
      static radial_gamma_t star_gB(-gamma_b(), gamma_b(), 3);

      p.phi_cf   = &sphere.phi;   p.gA_cf  = &sphere_gA.val; p.gB_cf  = &sphere_gB.val;
      p.phix_cf  = &sphere.phi_x; p.gAx_cf = &sphere_gA.ddx; p.gBx_cf = &sphere_gB.ddx;
      p.phiy_cf  = &sphere.phi_y; p.gAy_cf = &sphere_gA.ddy; p.gBy_cf = &sphere_gB.ddy;
      p.kappa_cf = &sphere.phi_c;
      p.xyz[0] = -0.30; p.xyz[1] = -0.30; particles.push_back(p);

      p.phi_cf   = &star.phi;   p.gA_cf  = &star_gA.val; p.gB_cf  = &star_gB.val;
      p.phix_cf  = &star.phi_x; p.gAx_cf = &star_gA.ddx; p.gBx_cf = &star_gB.ddx;
      p.phiy_cf  = &star.phi_y; p.gAy_cf = &star_gA.ddy; p.gBy_cf = &star_gB.ddy;
      p.kappa_cf = &star.phi_c;
      p.xyz[0] = +0.40; p.xyz[1] = -0.20; particles.push_back(p);

      p.phi_cf   = &capsule.phi;   p.gA_cf  = &capsule_gA.val; p.gB_cf  = &capsule_gB.val;
      p.phix_cf  = &capsule.phi_x; p.gAx_cf = &capsule_gA.ddx; p.gBx_cf = &capsule_gB.ddx;
      p.phiy_cf  = &capsule.phi_y; p.gAy_cf = &capsule_gA.ddy; p.gBy_cf = &capsule_gB.ddy;
      p.kappa_cf = &capsule.phi_c;
      p.xyz[0] = -0.20; p.xyz[1] = +0.25; particles.push_back(p);
    }
      break;

    case 3: // perturbed grid of spheres
    {
      static radial_shaped_domain_t sphere(0.05, DIM(0,0,0), -1, 0, 0.0, 0.0);
      static radial_gamma_t gA(0, gamma_a(), 1);
      static radial_gamma_t gB(0, gamma_b(), 1);
      int n = 10;
      int m = 10;

      p.phi_cf   = &sphere.phi;
      p.phix_cf  = &sphere.phi_x;
      p.phiy_cf  = &sphere.phi_y;
      p.kappa_cf = &sphere.phi_c;

      p.gA_cf  = &gA.val;
      p.gAx_cf = &gA.ddx;
      p.gAy_cf = &gA.ddy;

      p.gB_cf  = &gB.val;
      p.gBx_cf = &gB.ddx;
      p.gBy_cf = &gB.ddy;

      for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
        {
          p.xyz[0] = xmin() + double(i+1)*(xmax()-xmin())/double(n+1);
          p.xyz[1] = ymin() + double(j+1)*(ymax()-ymin())/double(m+1);
          particles.push_back(p);
        }
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

class phi_effe_cf_t : public CF_DIM
{
  CF_DIM *phi_ptcl;
  CF_DIM *phi_wall;
  CF_DIM *phi_free;
public:
  phi_effe_cf_t(CF_DIM &phi_ptcl, CF_DIM &phi_wall, CF_DIM &phi_free)
    : phi_ptcl(&phi_ptcl), phi_wall(&phi_wall), phi_free(&phi_free) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    return MIN( MAX((*phi_ptcl)(DIM(x,y,z)),
                    (*phi_wall)(DIM(x,y,z)),
                    (*phi_free)(DIM(x,y,z))), fabs((*phi_wall)(DIM(x,y,z))));
  }
};

PetscErrorCode ierr;
int main (int argc, char* argv[])
{
  // ------------------------------------------------------------------------------------------------------------------
  // initialize MPI
  // ------------------------------------------------------------------------------------------------------------------
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  srand(mpi.rank());

  // ------------------------------------------------------------------------------------------------------------------
  // parse command line arguments for parameters
  // ------------------------------------------------------------------------------------------------------------------
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  set_wall_surface_energies();

  double scaling = 1./box_size();
  pairwise_potential_width.val *= (xmax()-xmin())/pow(2.,lmax());

  // ------------------------------------------------------------------------------------------------------------------
  // prepare output directories
  // ------------------------------------------------------------------------------------------------------------------
  const char* out_dir = getenv("OUT_DIR");
  if (mpi.rank() == 0 && (save_vtk() || save_parameters() || save_data()))
  {
    if (!out_dir)
    {
      ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save visuals\n");
      return -1;
    }
    std::ostringstream command;
    command << "mkdir -p " << out_dir;
    int ret_sys = system(command.str().c_str());
    if (ret_sys < 0) throw std::invalid_argument("Could not create a directory");
  }

  if (mpi.rank() == 0 && save_vtk())
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys < 0) throw std::invalid_argument("Could not create a directory");
  }

  if (mpi.rank() == 0 && save_parameters()) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  FILE *file_conv;
  char file_conv_name[10000];
  if (save_data())
  {
    sprintf(file_conv_name, "%s/data.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_conv_name, "w", &file_conv); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_conv, "iteration "
                                               "energy "
                                               "energy_change_predicted "
                                               "energy_change_effective\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);
  }

  // ------------------------------------------------------------------------------------------------------------------
  // start a timer
  // ------------------------------------------------------------------------------------------------------------------
  parStopWatch w;
  w.start("total time");

  // ------------------------------------------------------------------------------------------------------------------
  // initialize particles
  // ------------------------------------------------------------------------------------------------------------------
  vector<particle_t> particles;
  initalize_ptcl(particles);
  int np = particles.size();

  phi_ptcl_cf_t phi_ptcl_cf(particles);
  particles_number_cf_t particles_number_cf(particles);

  phi_effe_cf_t phi_effe_cf(phi_ptcl_cf, phi_wall_cf, phi_free_cf);

  // ------------------------------------------------------------------------------------------------------------------
  // create initial grid
  // ------------------------------------------------------------------------------------------------------------------
  double xyz_min[]  = { DIM(xmin(), ymin(), zmin()) };
  double xyz_max[]  = { DIM(xmax(), ymax(), zmax()) };
  int    nb_trees[] = { DIM(nx(), ny(), nz()) };
  int    periodic[] = { DIM(px(), py(), pz()) };

  my_p4est_brick_t      brick;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t              *p4est        = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin(), lmax(), &phi_effe_cf, lip(), band());
  data.set_refine_only_inside(true);

  p4est->user_pointer = (void*)(&data);
  for (int i = 0; i < lmax(); ++i)
  {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  double dxyz[P4EST_DIM], h, diag;
  get_dxyz_min(p4est, dxyz, &h, &diag);

  // ------------------------------------------------------------------------------------------------------------------
  // create and allocate fields (mu_m is choosen to be the template for all other Vec's)
  // ------------------------------------------------------------------------------------------------------------------
  Vec     mu_m;
  double *mu_m_ptr;

  Vec     mu_p;
  double *mu_p_ptr;

  Vec     phi_free;
  double *phi_free_ptr;

  Vec     phi_wall;
  double *phi_wall_ptr;

  Vec     phi_ptcl;
  double *phi_ptcl_ptr;

  ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &mu_p);                CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &phi_free);            CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &phi_wall);            CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &phi_ptcl);            CHKERRXX(ierr);

  // ------------------------------------------------------------------------------------------------------------------
  // initialize fields
  // ------------------------------------------------------------------------------------------------------------------
  sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
  ierr = VecSetGhost(mu_p, 0); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_free_cf, phi_free);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);
  sample_cf_on_nodes(p4est, nodes, phi_ptcl_cf, phi_ptcl);

  my_p4est_level_set_t ls(ngbd);
  if (geometry_free() != 0) ls.reinitialize_2nd_order(phi_free);
  if (geometry_wall() != 0) ls.reinitialize_2nd_order(phi_wall);

  // ------------------------------------------------------------------------------------------------------------------
  // compute initial volume for volume-loss corrections
  // ------------------------------------------------------------------------------------------------------------------
  double volume = 0;

  std::vector<Vec> phi(3);
  std::vector<mls_opn_t> acn(3, MLS_INTERSECTION);
  std::vector<int> clr(3);

  phi[0] = phi_free; clr[0] = 0;
  phi[1] = phi_wall; clr[1] = 1;
  phi[2] = phi_ptcl; clr[2] = 2;

  my_p4est_integration_mls_t integration(p4est, nodes);
  integration.set_phi(phi, acn, clr);

  volume = integration.measure_of_domain();

  // ------------------------------------------------------------------------------------------------------------------
  // in case of constant contact angle and simple geometry we can compute analytically position and volume of the steady-state shape
  // ------------------------------------------------------------------------------------------------------------------
  double elev = 0;
  if (compute_exact())
  {
    double g_Aw = gamma_Aw_cf(DIM(0,0,0));
    double g_Bw = gamma_Bw_cf(DIM(0,0,0));
    double g_Aa = gamma_Aa_cf(DIM(0,0,0));
    double g_Ba = gamma_Ba_cf(DIM(0,0,0));
    double g_aw = gamma_aw_cf(DIM(0,0,0));

    double theta = acos( (.5*(g_Aw+g_Bw) - g_aw) / (.5*(g_Aa+g_Ba) ) );

    elev = (wall_eps()*drop_r0() - 2.*cos(PI-theta))*drop_r0()
        /(sqrt(SQR(wall_eps()*drop_r0()) + 1. - 2.*wall_eps()*drop_r0()*cos(PI-theta)) + 1.);

    double alpha = acos((elev + .5*(SQR(drop_r0()) + SQR(elev))*wall_eps())/drop_r0()/(1.+wall_eps()*elev));
#ifdef P4_TO_P8
    volume = 1./3.*PI*pow(drop_r0(), 3.)*(2.*(1.-cos(PI-alpha)) + cos(alpha)*SQR(sin(alpha)));
#else
    volume = SQR(drop_r0())*(PI - alpha + cos(alpha)*sin(alpha));
#endif

    if (wall_eps() != 0.)
    {
      double beta = acos(1.+.5*SQR(wall_eps())*(SQR(elev) - SQR(drop_r0()))/(1.+wall_eps()*elev));
#ifdef P4_TO_P8
      volume -= 1./3.*PI*(2.*(1.-cos(beta)) - cos(beta)*SQR(sin(beta)))/SQR(wall_eps())/wall_eps();
#else
      volume -= (beta - cos(beta)*sin(beta))/fabs(wall_eps())/wall_eps();
#endif
    }
  }

  double energy_old = 0;
  double rho_old    = 1;

  vector< vector<double> > v_old(P4EST_DIM, vector<double> (np,0));
  vector<double> w_old(np, 0);


  // ------------------------------------------------------------------------------------------------------------------
  // main loop
  // ------------------------------------------------------------------------------------------------------------------
  int iteration = 0;
  while (iteration < max_iterations())
  {
    // ------------------------------------------------------------------------------------------------------------------
    // refine and coarsen grid
    // ------------------------------------------------------------------------------------------------------------------
    Vec     phi_effe;
    double *phi_effe_ptr;

    Vec     phi_effe_tmp;
    double *phi_effe_tmp_ptr;

    ierr = VecDuplicate(mu_m, &phi_effe);     CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &phi_effe_tmp); CHKERRXX(ierr);

    ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);
      phi_effe_ptr[n] = MIN( MAX(phi_wall_ptr[n], phi_free_ptr[n], phi_ptcl_cf.value(xyz)), ABS(phi_wall_ptr[n]));
    }

    ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_effe, &phi_effe_ptr);  CHKERRXX(ierr);

    ierr = VecCopyGhost(phi_effe, phi_effe_tmp); CHKERRXX(ierr);

    p4est_t       *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    bool is_grid_changing = true;
    bool has_grid_changed = false;

    while (is_grid_changing)
    {
      ierr = VecGetArray(phi_effe_tmp, &phi_effe_tmp_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(data.min_lvl, data.max_lvl, data.lip, data.band);
      sp.set_refine_only_inside(data.refine_only_inside);

      is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_effe_tmp_ptr);

      ierr = VecRestoreArray(phi_effe_tmp, &phi_effe_ptr); CHKERRXX(ierr);

      if (is_grid_changing)
      {
        has_grid_changed = true;

        // repartition p4est
        my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

        // interpolate data between grids
        my_p4est_interpolation_nodes_t interp(ngbd);

        double xyz[P4EST_DIM];
        foreach_node(n, nodes_np1)
        {
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          interp.add_point(n, xyz);
        }

        ierr = VecDestroy(phi_effe_tmp); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_effe_tmp); CHKERRXX(ierr);

        interp.set_input(phi_effe, linear);
        interp.interpolate(phi_effe_tmp);
      }
    }

    ierr = VecDestroy(phi_effe_tmp); CHKERRXX(ierr);
    ierr = VecDestroy(phi_effe);     CHKERRXX(ierr);

    // interpolate data between grids
    if (has_grid_changed)
    {
      my_p4est_interpolation_nodes_t interp(ngbd);

      double xyz[P4EST_DIM];
      foreach_node(n, nodes_np1)
      {
        node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
        interp.add_point(n, xyz);
      }

      interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_m,     NULL, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_p,     mu_m, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_wall, mu_m, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_free, mu_m, interpolation_between_grids);

      ierr = VecDestroy(phi_ptcl); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_free, &phi_ptcl); CHKERRXX(ierr);
    }

    // delete old p4est
    p4est_destroy(p4est);       p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);

    if (iteration == 0)
    {
      ierr = VecDestroy(phi_free); CHKERRXX(ierr);
      ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
      ierr = VecDestroy(mu_p);     CHKERRXX(ierr);
      ierr = VecDestroy(mu_m);     CHKERRXX(ierr);

      ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
      ierr = VecDuplicate(mu_m, &mu_p);     CHKERRXX(ierr);
      ierr = VecDuplicate(mu_m, &phi_free); CHKERRXX(ierr);
      ierr = VecDuplicate(mu_m, &phi_wall); CHKERRXX(ierr);

      ierr = VecSetGhost(mu_p, 0); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, mu_cf,       mu_m);
      sample_cf_on_nodes(p4est, nodes, phi_free_cf, phi_free);
      sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);

      my_p4est_level_set_t ls(ngbd);
      if (geometry_free() != 0) ls.reinitialize_2nd_order(phi_free);
      if (geometry_wall() != 0) ls.reinitialize_2nd_order(phi_wall);
    }
    sample_cf_on_nodes(p4est, nodes, phi_ptcl_cf, phi_ptcl);

    // ------------------------------------------------------------------------------------------------------------------
    // allocate fields
    // ------------------------------------------------------------------------------------------------------------------
    Vec     normal_free    [P4EST_DIM];
    double *normal_free_ptr[P4EST_DIM];

    Vec     normal_wall    [P4EST_DIM];
    double *normal_wall_ptr[P4EST_DIM];

    Vec     mu_m_grad    [P4EST_DIM];
    double *mu_m_grad_ptr[P4EST_DIM];

    Vec     kappa;
    double *kappa_ptr;

    Vec     shape_grad_free;
    double *shape_grad_free_ptr;

    Vec     shape_grad_ptcl;
    double *shape_grad_ptcl_ptr;

    Vec     shape_grad_free_full;
    double *shape_grad_free_full_ptr;

    Vec     tmp;
    double *tmp_ptr;

    Vec     surf_tns;
    double *surf_tns_ptr;

    Vec     surf_tns_grad    [P4EST_DIM];
    double *surf_tns_grad_ptr[P4EST_DIM];

    Vec     cos_angle;
    double *cos_angle_ptr;

    Vec     velo_free;
    double *velo_free_ptr;

    Vec     velo_free_full;
    double *velo_free_full_ptr;

    Vec     particles_number;
    double *particles_number_ptr;

    Vec     integrand;
    double *integrand_ptr;

    foreach_dimension(dim)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &normal_free  [dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal_free[dim],    &normal_wall  [dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal_free[dim],    &mu_m_grad    [dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal_free[dim],    &surf_tns_grad[dim]); CHKERRXX(ierr);
    }

    ierr = VecDuplicate(mu_m, &phi_effe);             CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &kappa);                CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &shape_grad_free);      CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &shape_grad_free_full); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &shape_grad_ptcl);      CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &tmp);                  CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &surf_tns);             CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &cos_angle);            CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &velo_free);            CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &velo_free_full);       CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &particles_number);     CHKERRXX(ierr);
    ierr = VecDuplicate(mu_m, &integrand);            CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, particles_number_cf, particles_number);

    // ------------------------------------------------------------------------------------------------------------------
    // prepare integration tool
    // ------------------------------------------------------------------------------------------------------------------
    my_p4est_integration_mls_t integration(p4est, nodes);

    phi[0] = phi_free;
    phi[1] = phi_wall;
    phi[2] = phi_ptcl;

    integration.set_phi(phi, acn, clr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    // ------------------------------------------------------------------------------------------------------------------
    // compute effective level set function
    // ------------------------------------------------------------------------------------------------------------------
    ierr = VecPointwiseMaxGhost(phi_effe, phi_free, phi_wall); CHKERRXX(ierr);
    ierr = VecPointwiseMaxGhost(phi_effe, phi_effe, phi_ptcl); CHKERRXX(ierr);
//    ls.reinitialize_2nd_order(phi_effe);

    // ------------------------------------------------------------------------------------------------------------------
    // compute geometric information
    // ------------------------------------------------------------------------------------------------------------------
    compute_normals_and_mean_curvature(*ngbd, phi_free, normal_free, kappa);
    compute_normals(*ngbd, phi_wall, normal_wall);

    // ------------------------------------------------------------------------------------------------------------------
    // deform normal field
    // ------------------------------------------------------------------------------------------------------------------
    if (geometry_free() != 0 && geometry_wall() != 0)
    {
      double band_deform = 1.5*diag;
      double band_smooth = 7.0*diag;

      ierr = VecGetArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);
      foreach_dimension(dim)
      {
        ierr = VecGetArray(normal_free[dim], &normal_free_ptr[dim]); CHKERRXX(ierr);
        ierr = VecGetArray(normal_wall[dim], &normal_wall_ptr[dim]); CHKERRXX(ierr);
      }

      foreach_node(n, nodes)
      {
        double dot_product = SUMD(normal_free_ptr[0][n]*normal_wall_ptr[0][n],
            normal_free_ptr[1][n]*normal_wall_ptr[1][n],
            normal_free_ptr[2][n]*normal_wall_ptr[2][n]);

        EXECD(normal_wall_ptr[0][n] -= (1.-smoothstep(1, (fabs(phi_free_ptr[n]) - band_deform)/band_smooth))*dot_product*normal_free_ptr[0][n],
            normal_wall_ptr[1][n] -= (1.-smoothstep(1, (fabs(phi_free_ptr[n]) - band_deform)/band_smooth))*dot_product*normal_free_ptr[1][n],
            normal_wall_ptr[2][n] -= (1.-smoothstep(1, (fabs(phi_free_ptr[n]) - band_deform)/band_smooth))*dot_product*normal_free_ptr[2][n]);

        double norm = ABSD(normal_wall_ptr[0][n], normal_wall_ptr[1][n], normal_wall_ptr[2][n]);

        EXECD(normal_wall_ptr[0][n] /= norm,
            normal_wall_ptr[1][n] /= norm,
            normal_wall_ptr[2][n] /= norm);
      }

      ierr = VecRestoreArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);
      foreach_dimension(dim)
      {
        ierr = VecRestoreArray(normal_free[dim], &normal_free_ptr[dim]); CHKERRXX(ierr);
        ierr = VecRestoreArray(normal_wall[dim], &normal_wall_ptr[dim]); CHKERRXX(ierr);
      }

      // extension tangentially to the interface
      //    VecShiftGhost(phi_wall,  1.5*diag);
      //    ls.extend_Over_Interface_TVD_Full(phi_wall, kappa, 50, 0, 0, DBL_MAX, DBL_MAX, DBL_MAX, normal_wall);
      //    VecShiftGhost(phi_wall, -1.5*diag);

      ls.set_interpolation_on_interface(linear);
      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_free, kappa, phi_free, 20, phi_wall);
      ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
    }

    // ------------------------------------------------------------------------------------------------------------------
    // get density field and non-curvature shape gradient
    // ------------------------------------------------------------------------------------------------------------------
    double rho_avg = 1;
    double energy  = 0;

    bool adaptive = false;

    if (use_scft())
    {
      my_p4est_scft_t scft(ngbd, ns());

      // set geometry
      gamma_Aa_all_cf_t gamma_Aa_all(particles, particles_number_cf);
      gamma_Ba_all_cf_t gamma_Ba_all(particles, particles_number_cf);

      scft.add_boundary(phi_free, MLS_INTERSECTION, gamma_Aa_cf, gamma_Ba_cf);
      scft.add_boundary(phi_wall, MLS_INTERSECTION, gamma_Aw_cf, gamma_Bw_cf);
      scft.add_boundary(phi_ptcl, MLS_INTERSECTION, gamma_Aa_all, gamma_Ba_all);

      scft.set_scaling(scaling);
      scft.set_polymer(f(), XN());
      scft.set_rho_avg(rho_old);

      // initialize potentials
      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      ierr = VecCopyGhost(mu_m, mu_m_tmp); CHKERRXX(ierr);
//      ierr = VecCopyGhost(mu_p, mu_p_tmp); CHKERRXX(ierr);
      ierr = VecSetGhost(mu_p_tmp, 0); CHKERRXX(ierr);

      // initialize diffusion solvers for propagators
      scft.initialize_solvers();
      scft.initialize_bc_smart(iteration != 0);

      // main loop for solving SCFT equations
      int    scft_iteration = 0;
      int    bc_iters       = 0;
      double scft_error     = 2.*scft_tol()+1.;
      int    pressure_resets = 0;
      smooth_pressure.val = true;
      while ((scft_iteration < max_scft_iterations() && scft_error > scft_tol()) ||
             (scft_iteration < bc_adjust_min()+1))
      {
        for (int i=3;i--;) {
          scft.initialize_bc_smart(adaptive); adaptive = true;
          scft.solve_for_propogators();
          scft.calculate_densities();
//          scft.save_VTK(scft_iteration++);
          ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
                             scft_iteration,
                             scft.get_energy(),
                             scft.get_pressure_force(),
                             scft.get_exchange_force()); CHKERRXX(ierr);
        }
        // do an SCFT step
//        scft.solve_for_propogators();
//        scft.calculate_densities();
        scft.update_potentials();
        scft.save_VTK(scft_iteration);

//        if (scft.get_exchange_force() < scft_bc_tol() && bc_iters >= bc_adjust_min())
//        {
//          scft.initialize_bc_smart();
////          if (smooth_pressure())
////          {
////            scft.smooth_singularity_in_pressure_field();
////            smooth_pressure.val = false;
////          }
//          if (pressure_resets < 1)
//          {
//            scft.smooth_singularity_in_pressure_field();
//            pressure_resets++;

//            double pressure_force = scft.get_pressure_force();
//            while (scft.get_pressure_force() >= pressure_force) {
//              scft.solve_for_propogators();
//              scft.calculate_densities();
//              scft.update_potentials(false, true);

//              ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
//                                 scft_iteration,
//                                 scft.get_energy(),
//                                 scft.get_pressure_force(),
//                                 scft.get_exchange_force()); CHKERRXX(ierr);
//            }
//          }
//          bc_iters = 0;
//        }

        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e%s\n",
                           scft_iteration,
                           scft.get_energy(),
                           scft.get_pressure_force(),
                           scft.get_exchange_force(),
                           bc_iters == 0 ? " (bc)" : ""); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        scft_iteration++;
        bc_iters++;
      }

      energy  = scft.get_energy();
      rho_avg = scft.get_rho_avg();
      rho_old = rho_avg;

      scft.sync_and_extend();
      if (geometry_free() != 0)
      {
        scft.compute_energy_shape_derivative(0, shape_grad_free);
        ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_free, shape_grad_free, mu_m, 50, phi_wall);
      }
      else
      {
        ierr = VecSetGhost(shape_grad_free, 0); CHKERRXX(ierr);
      }

      if (geometry_ptcl() != 0)
      {
        scft.compute_energy_shape_derivative(2, shape_grad_ptcl);
        ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_ptcl, shape_grad_ptcl, mu_m, 50, phi_wall);
      }
      else
      {
        ierr = VecSetGhost(shape_grad_ptcl, 0); CHKERRXX(ierr);
      }

      ierr = VecScaleGhost(shape_grad_free, -1); CHKERRXX(ierr);

      ierr = VecCopyGhost(mu_m_tmp, mu_m); CHKERRXX(ierr);
      ierr = VecCopyGhost(mu_p_tmp, mu_p); CHKERRXX(ierr);
    }
    else
    {
      sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
      ierr = VecSetGhost(mu_p, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(shape_grad_free, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(shape_grad_ptcl, 0); CHKERRXX(ierr);
    }

    double energy_change_predicted = 0;
    // ------------------------------------------------------------------------------------------------------------------
    // shape derivative with respect to free surface position
    // ------------------------------------------------------------------------------------------------------------------
    double vmax = 0;
    double dt_free = DBL_MAX;
    if (geometry_free() != 0)
    {
      // ------------------------------------------------------------------------------------------------------------------
      // compute curvature part of shape gradient and contact angle
      // ------------------------------------------------------------------------------------------------------------------
      ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

      double xyz[P4EST_DIM];
      foreach_node(n, nodes)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);
        double g_Aw = gamma_Aw_cf.value(xyz);
        double g_Bw = gamma_Bw_cf.value(xyz);
        double g_Aa = gamma_Aa_cf.value(xyz);
        double g_Ba = gamma_Ba_cf.value(xyz);
        double g_aw = gamma_aw_cf.value(xyz);

        double ptcl_repulsion = 0;
        for (int j = 0; j < np; ++j)
        {
          double dist_other = particles[j].phi(DIM(xyz[0], xyz[1], xyz[2]));
          ptcl_repulsion += pairwise_potential(-dist_other-pairwise_potential_width());
        }

        surf_tns_ptr [n] = (.5*(g_Aa+g_Ba)*rho_avg + (g_Aa-g_Ba)*mu_m_ptr[n]/XN())/pow(scaling, P4EST_DIM-1) + ptcl_repulsion/pow(scaling, P4EST_DIM-1);
        cos_angle_ptr[n] = (.5*(g_Aw+g_Bw)*rho_avg + (g_Aw-g_Bw)*mu_m_ptr[n]/XN() - g_aw)
                           / (.5*(g_Aa+g_Ba) + (g_Aa-g_Ba)*mu_m_ptr[n]/XN());
      }

      ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

      if (!use_scft())
      {
        energy += integration.integrate_over_interface(0, surf_tns);

        if (geometry_wall() != 0)
        {
          Vec     surf_tns_wall;
          double *surf_tns_wall_ptr;

          ierr = VecDuplicate(mu_m, &surf_tns_wall); CHKERRXX(ierr);

          ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
          ierr = VecGetArray(surf_tns_wall, &surf_tns_wall_ptr); CHKERRXX(ierr);

          double xyz[P4EST_DIM];
          foreach_node(n, nodes)
          {
            node_xyz_fr_n(n, p4est, nodes, xyz);
            double g_Aw = gamma_Aw_cf.value(xyz);
            double g_Bw = gamma_Bw_cf.value(xyz);
            double g_aw = gamma_aw_cf.value(xyz);

            surf_tns_wall_ptr[n] = (.5*(g_Aw+g_Bw)*rho_avg + (g_Aw-g_Bw)*mu_m_ptr[n]/XN() - g_aw)/pow(scaling, P4EST_DIM-1);
          }

          ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
          ierr = VecRestoreArray(surf_tns_wall, &surf_tns_wall_ptr); CHKERRXX(ierr);

          energy += integration.integrate_over_interface(1, surf_tns_wall);

          ierr = VecDestroy(surf_tns_wall); CHKERRXX(ierr);
        }
      }

      ls.extend_Over_Interface_TVD_Full(phi_wall, cos_angle, 20, 0, 0, DBL_MAX, DBL_MAX, DBL_MAX, normal_wall);
      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_free, cos_angle, phi_free, 20, phi_wall);

      ls.extend_from_interface_to_whole_domain_TVD(phi_free, surf_tns, tmp);
      ierr = VecPointwiseMultGhost(shape_grad_free_full, kappa, tmp); CHKERRXX(ierr);

      ngbd->first_derivatives_central(surf_tns, surf_tns_grad);
      foreach_dimension(dim)
      {
        ierr = VecPointwiseMultGhost(surf_tns_grad[dim], surf_tns_grad[dim], normal_free[dim]); CHKERRXX(ierr);
        ls.extend_from_interface_to_whole_domain_TVD(phi_free, surf_tns_grad[dim], tmp);
        VecAXPBYGhost(shape_grad_free_full, 1, 1, tmp);
      }

      VecScaleGhost(shape_grad_free_full, -1);
      VecAXPBYGhost(shape_grad_free_full, 1, 1, shape_grad_free);

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_free, shape_grad_free_full, phi_free, 20, phi_wall);

      // ------------------------------------------------------------------------------------------------------------------
      // select free surface velocity
      // ------------------------------------------------------------------------------------------------------------------
      VecCopyGhost(shape_grad_free,      velo_free);
      VecCopyGhost(shape_grad_free_full, velo_free_full);

      double vn_avg = integration.integrate_over_interface(0, velo_free_full)/integration.measure_of_interface(0);

      VecShiftGhost(velo_free, -vn_avg);
      VecShiftGhost(velo_free_full, -vn_avg);

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_free, velo_free, phi_free, 20, phi_wall);

      // ------------------------------------------------------------------------------------------------------------------
      // compute time step for free surface
      // ------------------------------------------------------------------------------------------------------------------
      ierr = VecGetArray(velo_free, &velo_free_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo_free_full, &velo_free_full_ptr); CHKERRXX(ierr);

      quad_neighbor_nodes_of_node_t qnnn;
      foreach_local_node(n, nodes)
      {
        ngbd->get_neighbors(n, qnnn);

        double xyzn[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyzn);

        double s_p00 = fabs(qnnn.d_p00); double s_m00 = fabs(qnnn.d_m00);
        double s_0p0 = fabs(qnnn.d_0p0); double s_0m0 = fabs(qnnn.d_0m0);
#ifdef P4_TO_P8
        double s_00p = fabs(qnnn.d_00p); double s_00m = fabs(qnnn.d_00m);
#endif
#ifdef P4_TO_P8
        double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0), MIN(s_00p, s_00m));
#else
        double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0));
#endif

        dt_free = MIN(dt_free, cfl()*fabs(s_min/velo_free_ptr[n]));
        dt_free = MIN(dt_free, cfl()*fabs(s_min/velo_free_full_ptr[n]));
        vmax = MAX(vmax, fabs(velo_free_ptr[n]));
        vmax = MAX(vmax, fabs(velo_free_full_ptr[n]));
      }

      ierr = VecRestoreArray(velo_free, &velo_free_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_free_full, &velo_free_full_ptr); CHKERRXX(ierr);

      MPI_Allreduce(MPI_IN_PLACE, &dt_free, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);
      MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

      // ------------------------------------------------------------------------------------------------------------------
      // compute expected change in energy due to free surface
      // ------------------------------------------------------------------------------------------------------------------
      ierr = VecPointwiseMultGhost(integrand, velo_free_full, shape_grad_free_full); CHKERRXX(ierr);
      energy_change_predicted = -dt_free*integration.integrate_over_interface(0, integrand);
    }
    else
    {
      ierr = VecSetGhost(velo_free, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(velo_free_full, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(shape_grad_free, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(shape_grad_free_full, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(surf_tns, 0); CHKERRXX(ierr);
      ierr = VecSetGhost(cos_angle, 0); CHKERRXX(ierr);
    }

    // ------------------------------------------------------------------------------------------------------------------
    // compute derivatives with respect to particle positions
    // ------------------------------------------------------------------------------------------------------------------
    vector< vector<double> > g(P4EST_DIM, vector<double> (np,0));
    vector< vector<double> > v(P4EST_DIM, vector<double> (np,0));

    vector<double> gw(np, 0);
    vector<double> w(np, 0);

    vector<double> dt_ptcl_v(np, 1);
    vector<double> dt_ptcl_w(np, 1);

    if (geometry_ptcl() !=0)
    {
      ngbd->first_derivatives_central(mu_m, mu_m_grad);

      sample_cf_on_nodes(p4est, nodes, particles_number_cf, particles_number);

      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

      double factor  = 1./pow(scaling, P4EST_DIM-1.);

      my_p4est_interpolation_nodes_local_t interp_local(ngbd);
      double proximity_crit = 50.*diag;
      double surf_energy = 0;
      double repulsion_energy = 0;
      foreach_local_node(n, nodes)
      {
        if (fabs(phi_effe_ptr[n]) < proximity_crit)
        {
          int i = int(particles_number_ptr[n]);

          // get area and centroid of the interface
          double area_ptcl = 0;
          double area_wall = 0;
          double area_free = 0;
          double xyz_ptcl[P4EST_DIM];
          double xyz_wall[P4EST_DIM];
          double xyz_free[P4EST_DIM];

          interp_local.initialize(n);
          interp_local.set_input(phi_free, linear);

          phi_true_cf_t PHI(&particles[i]);
          vector<mls_opn_t> opn(3, MLS_INT);
          vector<CF_DIM *> phi_cf(3);
          phi_cf[0] = &PHI;
          phi_cf[1] = &phi_wall_cf;
          phi_cf[2] = &interp_local;

          my_p4est_finite_volume_t fv;
          construct_finite_volume(fv, n, p4est, nodes, phi_cf, opn, 1, 1);

          for (int j = 0; j < fv.interfaces.size(); ++j)
          {
            double xyz[P4EST_DIM];
            node_xyz_fr_n(n, p4est, nodes, xyz);

            switch (fv.interfaces[j].id)
            {
              case 0:
                xyz_ptcl[0]  = xyz[0] + fv.interfaces[j].centroid[0];
                xyz_ptcl[1]  = xyz[1] + fv.interfaces[j].centroid[1];
                area_ptcl    = fv.interfaces[j].area;
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

          if (area_ptcl != 0)
          {
            double phix = particles[i].phix (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double phiy = particles[i].phiy (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            double norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            double dist = particles[i].phi (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            xyz_ptcl[0] -= dist*phix/norm;
            xyz_ptcl[1] -= dist*phiy/norm;

            phix = particles[i].phix (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            phiy = particles[i].phiy (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            interp_local.set_input(shape_grad_ptcl, linear); double velo_val = interp_local.value(xyz_ptcl);
            interp_local.set_input(mu_m,            linear); double mu_m     = interp_local.value(xyz_ptcl);
            interp_local.set_input(mu_m_grad[0],    linear); double mux_m    = interp_local.value(xyz_ptcl);
            interp_local.set_input(mu_m_grad[1],    linear); double muy_m    = interp_local.value(xyz_ptcl);

            double phic = particles[i].kappa(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            double ga  = particles[i].gA (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double gax = particles[i].gAx(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double gay = particles[i].gAy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            double gb  = particles[i].gB (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double gbx = particles[i].gBx(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double gby = particles[i].gBy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));


            double g_eff  = (.5*(ga+gb)*rho_avg + mu_m/XN()*(ga-gb));
            double gx_eff = (.5*(gax+gbx)*rho_avg + mu_m/XN()*(gax-gbx) + mux_m/XN()*(ga-gb));
            double gy_eff = (.5*(gay+gby)*rho_avg + mu_m/XN()*(gay-gby) + muy_m/XN()*(ga-gb));

            double G = SUMD(phix*gx_eff,
                            phiy*gy_eff,
                            phiz*gz_eff)*factor
                       + g_eff*phic*factor
                       + velo_val;

            // repulsion between particles
            for (int j = 0; j < np; ++j)
            {
              if (j != i)
              {
                double dist_other = particles[j].phi(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
                if (-dist_other < pairwise_potential_width())
                {
                  double phix_other = particles[j].phix(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
                  double phiy_other = particles[j].phiy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
                  dist_other = -dist_other-pairwise_potential_width();
                  G += phic*pairwise_potential(dist_other)*factor + pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, FIX)*factor;

                  double add_x = -pairwise_force(dist_other)*phix_other*factor;
                  double add_y = -pairwise_force(dist_other)*phiy_other*factor;

                  double delx = xyz_ptcl[0]-particles[j].xyz[0];
                  double dely = xyz_ptcl[1]-particles[j].xyz[1];

                  g[0][j] += add_x*area_ptcl;
                  g[1][j] += add_y*area_ptcl;
                  gw[j]   += (add_x*dely - add_y*delx)*area_ptcl;
                  repulsion_energy  += pairwise_potential(dist_other)*area_ptcl*factor;
                }
              }
            }

            // wall repulsion
            double dist_other = phi_wall_cf(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            if (-dist_other < pairwise_potential_width())
            {
              interp_local.set_input(normal_wall[0], linear); double phix_other = interp_local.value(xyz_ptcl);
              interp_local.set_input(normal_wall[1], linear); double phiy_other = interp_local.value(xyz_ptcl);
              dist_other = -dist_other-pairwise_potential_width();
              G += phic*pairwise_potential(dist_other)*factor +
                   pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, FIX)*factor;
            }

            double Gx = G*phix - (.5*(gax+gbx)*rho_avg + mu_m/XN()*(gax-gbx))*factor;
            double Gy = G*phiy - (.5*(gay+gby)*rho_avg + mu_m/XN()*(gay-gby))*factor;

            double delx = xyz_ptcl[0]-particles[i].xyz[0];
            double dely = xyz_ptcl[1]-particles[i].xyz[1];

            double Gxy = Gx*dely - Gy*delx;

            g[0][i] += Gx*area_ptcl;
            g[1][i] += Gy*area_ptcl;
            gw[i]   += Gxy*area_ptcl;
            surf_energy  += g_eff*area_ptcl*factor;
          }

          if (area_wall != 0)
          {
            interp_local.set_input(normal_wall[0], linear); double phix = interp_local.value(xyz_wall);
            interp_local.set_input(normal_wall[1], linear); double phiy = interp_local.value(xyz_wall);

            double norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            interp_local.set_input(phi_wall, linear); double dist = interp_local.value(xyz_wall);

            xyz_wall[0] -= dist*phix/norm;
            xyz_wall[1] -= dist*phiy/norm;

//            interp_local.set_input(mu_m, linear); double mu_m = interp_local.value(xyz_wall);

//            double ga  = gamma_Aw_cf.value(xyz_wall);
//            double gb  = gamma_Bw_cf.value(xyz_wall);

//            double g_eff  = (.5*(ga+gb)*rho_avg + mu_m/XN()*(ga-gb));

//            surf_energy  += g_eff*area_wall*factor;

            // repulsion from walls
            for (int j = 0; j < np; ++j)
            {
              double dist_other = particles[j].phi(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2]));
              if (-dist_other < pairwise_potential_width())
              {
                double phix_other = particles[j].phix(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2]));
                double phiy_other = particles[j].phiy(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2]));
                dist_other = -dist_other-pairwise_potential_width();

                double add_x = -pairwise_force(dist_other)*phix_other*factor;
                double add_y = -pairwise_force(dist_other)*phiy_other*factor;

                double delx = xyz_wall[0]-particles[j].xyz[0];
                double dely = xyz_wall[1]-particles[j].xyz[1];

                g[0][j] += add_x*area_wall;
                g[1][j] += add_y*area_wall;
                gw[j]   += (add_x*dely - add_y*delx)*area_wall;
                repulsion_energy  += pairwise_potential(dist_other)*area_wall*factor;
              }
            }
          }

          if (area_free != 0)
          {
            interp_local.set_input(normal_free[0], linear); double phix = interp_local.value(xyz_free);
            interp_local.set_input(normal_free[1], linear); double phiy = interp_local.value(xyz_free);

            double norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            interp_local.set_input(phi_free, linear); double dist = interp_local.value(xyz_free);

            xyz_free[0] -= dist*phix/norm;
            xyz_free[1] -= dist*phiy/norm;

//            interp_local.set_input(mu_m, linear); double mu_m = interp_local.value(xyz_free);

//            double ga  = gamma_Aa_cf.value(xyz_free);
//            double gb  = gamma_Ba_cf.value(xyz_free);

//            double g_eff  = (.5*(ga+gb)*rho_avg + mu_m/XN()*(ga-gb));

//            surf_energy  += g_eff*area_free*factor;

            // repulsion from free interface
            for (int j = 0; j < np; ++j)
            {
              double dist_other = particles[j].phi(DIM(xyz_free[0], xyz_free[1], xyz_free[2]));
              if (-dist_other < pairwise_potential_width())
              {
                double phix_other = particles[j].phix(DIM(xyz_free[0], xyz_free[1], xyz_free[2]));
                double phiy_other = particles[j].phiy(DIM(xyz_free[0], xyz_free[1], xyz_free[2]));
                dist_other = -dist_other-pairwise_potential_width();

                double add_x = -pairwise_force(dist_other)*phix_other*factor;
                double add_y = -pairwise_force(dist_other)*phiy_other*factor;

                double delx = xyz_free[0]-particles[j].xyz[0];
                double dely = xyz_free[1]-particles[j].xyz[1];

                g[0][j] += add_x*area_free;
                g[1][j] += add_y*area_free;
                gw[j]   += (add_x*dely - add_y*delx)*area_free;
                repulsion_energy  += pairwise_potential(dist_other)*area_free*factor;
              }
            }
          }
        }
      }
      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

      ierr = MPI_Allreduce(MPI_IN_PLACE, g[0].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, g[1].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, gw.data(),   np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, &repulsion_energy, 1,  MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, &surf_energy,      1,  MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);

      energy += repulsion_energy;

      if (!use_scft()) energy += surf_energy;

      // ------------------------------------------------------------------------------------------------------------------
      // select particles' velocity
      // ------------------------------------------------------------------------------------------------------------------
      if (minimize())
      {
        for(int i = 0; i < np; i++)
        {
          switch (1)
          {
            case 0:
              v[0][i] = -g[0][i];
              v[1][i] = -g[1][i];
              w[i] = -gw[i];
            case 1:
              v[0][i] = .5*(v_old[0][i]-g[0][i]);
              v[1][i] = .5*(v_old[1][i]-g[1][i]);
              w[i]    = .5*(w_old[i]   -gw[i]);
          }
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
      v_old = v;
      w_old = w;

      // ------------------------------------------------------------------------------------------------------------------
      // compute time step for particles
      // ------------------------------------------------------------------------------------------------------------------
      vector<double> arm_max(np,0);

      double phi_thresh = 0;

      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);

      foreach_node(n, nodes)
      {
        if (phi_ptcl_ptr[n] > phi_thresh)
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
      ierr = VecRestoreArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);

      ierr = MPI_Allreduce(MPI_IN_PLACE, arm_max.data(), np, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); CHKERRXX(ierr);

      double v_max  = 0;
      double wr_max = 0;
      for (int i = 0; i < np; ++i)
      {
        v_max  = MAX(v_max, ABSD(v[0][i], v[1][i], v[2][i]));
        wr_max = MAX(wr_max, fabs(w[i])*arm_max[i]);
      }

      double delta_tv = diag*cfl()/MAX(v_max, 0.001);
      double delta_tw = diag*cfl()/MAX(wr_max,0.001);

      for (int j = 0; j < np; ++j)
      {
        if (!minimize())
        {
          dt_ptcl_v[j] = delta_tv;
          dt_ptcl_w[j] = delta_tw;
        }
        else
        {
          dt_ptcl_v[j] = diag*cfl()/ABSD(v[0][j], v[1][j], v[2][j]);
          dt_ptcl_w[j] = diag*cfl()/fabs(w[j]*arm_max[j]);
        }
      }

      // ------------------------------------------------------------------------------------------------------------------
      // compute expected change in energy due to particles
      // ------------------------------------------------------------------------------------------------------------------
      for (int i = 0; i < np; i++)
      {
        energy_change_predicted += SUMD(v[0][i]*g[0][i],
            v[1][i]*g[1][i],
            v[2][i]*g[2][i])*dt_ptcl_v[i];

        energy_change_predicted += w[i]*gw[i]*dt_ptcl_w[i];
      }
    }

    //    ierr = PetscPrintf(mpi.comm(), "Avg velo: %e, Time step: %e\n", vn_avg, dt);
    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e, Predicted: %e\n", energy, energy-energy_old, energy_change_predicted);

    // ------------------------------------------------------------------------------------------------------------------
    // save data
    // ------------------------------------------------------------------------------------------------------------------
    if (save_data())
    {
      ierr = PetscFOpen  (mpi.comm(), file_conv_name, "a", &file_conv); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), file_conv, "%d %e %e %e\n", (int) round(iteration), energy, energy_change_predicted, energy-energy_old); CHKERRXX(ierr);
      ierr = PetscFClose (mpi.comm(), file_conv); CHKERRXX(ierr);
    }

    if (save_vtk() && iteration%save_every_dn() == 0)
    {
      // file name
      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << mpi.size() << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
       #endif
             "." << (int) round(iteration/save_every_dn());

      // compute reference solution
      Vec     phi_exact;
      double *phi_exact_ptr;
      ierr = VecDuplicate(phi_free, &phi_exact); CHKERRXX(ierr);
      ierr = VecSetGhost(phi_exact, -1); CHKERRXX(ierr);

      if (compute_exact())
      {
        Vec XYZ[P4EST_DIM];
        double *xyz_ptr[P4EST_DIM];

        foreach_dimension(dim)
        {
          ierr = VecCreateGhostNodes(p4est, nodes, &XYZ[dim]); CHKERRXX(ierr);
          ierr = VecGetArray(XYZ[dim], &xyz_ptr[dim]); CHKERRXX(ierr);
        }

        double xyz[P4EST_DIM];
        foreach_node(n, nodes)
        {
          node_xyz_fr_n(n, p4est, nodes, xyz);
          foreach_dimension(dim) xyz_ptr[dim][n] = xyz[dim];
        }

        foreach_dimension(dim) {
          ierr = VecRestoreArray(XYZ[dim], &xyz_ptr[dim]); CHKERRXX(ierr);
        }

        double com[P4EST_DIM];
        double vol = integration.measure_of_domain();
        foreach_dimension(dim)
            com[dim] = integration.integrate_over_domain(XYZ[dim])/vol;

#ifdef P4_TO_P8
        double vec_d[P4EST_DIM] = { wall_x() - com[0], wall_y() - com[1], wall_z() - com[2] };
        double nw_norm = sqrt(SQR(wall_nx()) + SQR(wall_ny()) + SQR(wall_nz()));
        double d_norm2 = SQR(vec_d[0]) + SQR(vec_d[1]) + SQR(vec_d[2]);
        double dn0 = (vec_d[0]*wall_nx() + vec_d[1]*wall_ny() + vec_d[2]*wall_nz())/nw_norm;
#else
        double vec_d[P4EST_DIM] = { wall_x() - com[0], wall_y() - com[1] };
        double nw_norm = sqrt(SQR(wall_nx()) + SQR(wall_ny()));
        double d_norm2 = SQR(vec_d[0]) + SQR(vec_d[1]);
        double dn0 = (vec_d[0]*wall_nx() + vec_d[1]*wall_ny())/nw_norm;
#endif

        double del = (d_norm2*wall_eps() + 2.*dn0)
            / ( sqrt(1. + (d_norm2*wall_eps() + 2.*dn0)*wall_eps()) + 1. );

        double vec_n[P4EST_DIM];

        vec_n[0] = (wall_nx()/nw_norm + vec_d[0]*wall_eps())/(1.+wall_eps()*del);
        vec_n[1] = (wall_ny()/nw_norm + vec_d[1]*wall_eps())/(1.+wall_eps()*del);
#ifdef P4_TO_P8
        vec_n[2] = (wall_nz()/nw_norm + vec_d[2]*wall_eps())/(1.+wall_eps()*del);
        double norm = sqrt(SQR(vec_n[0]) + SQR(vec_n[1]) + SQR(vec_n[2]));
#else
        double norm = sqrt(SQR(vec_n[0]) + SQR(vec_n[1]));
#endif

        double xyz_c[P4EST_DIM];
        foreach_dimension(dim)
            xyz_c[dim] = com[dim] + (del-elev)*vec_n[dim]/norm;

        flower_shaped_domain_t exact(drop_r0(), DIM(xyz_c[0], xyz_c[1], xyz_c[2]));

        sample_cf_on_nodes(p4est, nodes, exact.phi, phi_exact);

        foreach_dimension(dim) {
          ierr = VecDestroy(XYZ[dim]); CHKERRXX(ierr);
        }
      }

      // compute leaf levels
      Vec     leaf_level;
      double *leaf_level_ptr;

      ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
      ierr = VecGetArray(leaf_level, &leaf_level_ptr); CHKERRXX(ierr);

      for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
          leaf_level_ptr[tree->quadrants_offset+q] = quad->level;
        }
      }

      for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
      {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        leaf_level_ptr[p4est->local_num_quadrants+q] = quad->level;
      }
      ierr = VecRestoreArray(leaf_level, &leaf_level_ptr); CHKERRXX(ierr);

      Vec     gamma_ptcl_avg;
      double *gamma_ptcl_avg_ptr;

      Vec     gamma_ptcl_dif;
      double *gamma_ptcl_dif_ptr;

      ierr = VecDuplicate(mu_m, &gamma_ptcl_avg); CHKERRXX(ierr);
      ierr = VecDuplicate(mu_m, &gamma_ptcl_dif); CHKERRXX(ierr);

      if (geometry_ptcl() != 0)
      {
        ierr = VecGetArray(gamma_ptcl_avg, &gamma_ptcl_avg_ptr); CHKERRXX(ierr);
        ierr = VecGetArray(gamma_ptcl_dif, &gamma_ptcl_dif_ptr); CHKERRXX(ierr);
        ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

        foreach_node(n, nodes)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);

          int i = int(particles_number_ptr[n]);

          double ga = particles[i].gA(xyz[0], xyz[1]);
          double gb = particles[i].gB(xyz[0], xyz[1]);

          gamma_ptcl_avg_ptr[n] = .5*(ga+gb);
          gamma_ptcl_dif_ptr[n] = .5*(ga-gb);
        }

        ierr = VecRestoreArray(gamma_ptcl_avg, &gamma_ptcl_avg_ptr); CHKERRXX(ierr);
        ierr = VecRestoreArray(gamma_ptcl_dif, &gamma_ptcl_dif_ptr); CHKERRXX(ierr);
        ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      }

      ierr = VecGetArray(leaf_level,       &leaf_level_ptr);       CHKERRXX(ierr);
      ierr = VecGetArray(phi_effe,         &phi_effe_ptr);         CHKERRXX(ierr);
      ierr = VecGetArray(phi_wall,         &phi_wall_ptr);         CHKERRXX(ierr);
      ierr = VecGetArray(phi_free,         &phi_free_ptr);         CHKERRXX(ierr);
      ierr = VecGetArray(phi_ptcl,         &phi_ptcl_ptr);         CHKERRXX(ierr);
      ierr = VecGetArray(phi_exact,        &phi_exact_ptr);        CHKERRXX(ierr);
      ierr = VecGetArray(kappa,            &kappa_ptr);            CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns,         &surf_tns_ptr);         CHKERRXX(ierr);
      ierr = VecGetArray(mu_m,             &mu_m_ptr);             CHKERRXX(ierr);
      ierr = VecGetArray(mu_p,             &mu_p_ptr);             CHKERRXX(ierr);
      ierr = VecGetArray(velo_free,        &velo_free_ptr);        CHKERRXX(ierr);
      ierr = VecGetArray(velo_free_full,   &velo_free_full_ptr);   CHKERRXX(ierr);
      ierr = VecGetArray(cos_angle,        &cos_angle_ptr);        CHKERRXX(ierr);
      ierr = VecGetArray(normal_wall[0],   &normal_wall_ptr[0]);   CHKERRXX(ierr);
      ierr = VecGetArray(normal_wall[1],   &normal_wall_ptr[1]);   CHKERRXX(ierr);
      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_ptcl_avg,   &gamma_ptcl_avg_ptr);   CHKERRXX(ierr);
      ierr = VecGetArray(gamma_ptcl_dif,   &gamma_ptcl_dif_ptr);   CHKERRXX(ierr);
      ierr = VecGetArray(shape_grad_free,  &shape_grad_free_ptr);  CHKERRXX(ierr);
      ierr = VecGetArray(shape_grad_ptcl,  &shape_grad_ptcl_ptr);  CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             19, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi",       phi_effe_ptr,
                             VTK_POINT_DATA, "phi_wall",  phi_wall_ptr,
                             VTK_POINT_DATA, "phi_free",  phi_free_ptr,
                             VTK_POINT_DATA, "phi_ptcl",  phi_ptcl_ptr,
                             VTK_POINT_DATA, "phi_exact", phi_exact_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_POINT_DATA, "surf_tns", surf_tns_ptr,
                             VTK_POINT_DATA, "mu_m", mu_m_ptr,
                             VTK_POINT_DATA, "mu_p", mu_p_ptr,
                             VTK_POINT_DATA, "velo_free", velo_free_ptr,
                             VTK_POINT_DATA, "velo_free_full", velo_free_full_ptr,
                             VTK_POINT_DATA, "cos", cos_angle_ptr,
                             VTK_POINT_DATA, "nx", normal_wall_ptr[0],
                             VTK_POINT_DATA, "ny", normal_wall_ptr[1],
                             VTK_POINT_DATA, "num", particles_number_ptr,
                             VTK_POINT_DATA, "gamma_avg", gamma_ptcl_avg_ptr,
                             VTK_POINT_DATA, "gamma_diff", gamma_ptcl_dif_ptr,
                             VTK_POINT_DATA, "shape_grad_free", shape_grad_free_ptr,
                             VTK_POINT_DATA, "shape_grad_ptcl", shape_grad_ptcl_ptr,
                             VTK_CELL_DATA , "leaf_level", leaf_level_ptr);

      ierr = VecRestoreArray(leaf_level,       &leaf_level_ptr);       CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_effe,         &phi_effe_ptr);         CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_wall,         &phi_wall_ptr);         CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_free,         &phi_free_ptr);         CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_ptcl,         &phi_ptcl_ptr);         CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_exact,        &phi_exact_ptr);        CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa,            &kappa_ptr);            CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns,         &surf_tns_ptr);         CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_m,             &mu_m_ptr);             CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_p,             &mu_p_ptr);             CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_free,        &velo_free_ptr);        CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_free_full,   &velo_free_full_ptr);   CHKERRXX(ierr);
      ierr = VecRestoreArray(cos_angle,        &cos_angle_ptr);        CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_wall[0],   &normal_wall_ptr[0]);   CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_wall[1],   &normal_wall_ptr[1]);   CHKERRXX(ierr);
      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_ptcl_avg,   &gamma_ptcl_avg_ptr);   CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_ptcl_dif,   &gamma_ptcl_dif_ptr);   CHKERRXX(ierr);
      ierr = VecRestoreArray(shape_grad_free,  &shape_grad_free_ptr);  CHKERRXX(ierr);
      ierr = VecRestoreArray(shape_grad_ptcl,  &shape_grad_ptcl_ptr);  CHKERRXX(ierr);

      ierr = VecDestroy(phi_exact);  CHKERRXX(ierr);
      ierr = VecDestroy(gamma_ptcl_avg);  CHKERRXX(ierr);
      ierr = VecDestroy(gamma_ptcl_dif);  CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

//      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    // ------------------------------------------------------------------------------------------------------------------
    // move particles
    // ------------------------------------------------------------------------------------------------------------------
    for (int j = 0; j < np; ++j)
    {
      XCODE( particles[j].xyz[0] += v[0][j]*dt_ptcl_v[j] );
      YCODE( particles[j].xyz[1] += v[1][j]*dt_ptcl_v[j] );
      ZCODE( particles[j].xyz[2] += v[2][j]*dt_ptcl_v[j] );

      particles[j].rot += w[j]*dt_ptcl_w[j];
    }

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

    // ------------------------------------------------------------------------------------------------------------------
    // advect interface and impose contact angle
    // ------------------------------------------------------------------------------------------------------------------
    if (geometry_free() != 0)
    {
      ls.set_use_neumann_for_contact_angle(use_neumann());
      ls.set_contact_angle_extension(contact_angle_extension());

      double correction_total = 0;

      int splits = 1;
      for (int i = 0; i < splits; ++i)
      {
        ls.advect_in_normal_direction_with_contact_angle(velo_free, surf_tns, cos_angle, phi_wall, phi_free, dt_free/double(splits));

        // cut off tails
        double *phi_wall_ptr;
        double *phi_free_ptr;

        ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
        ierr = VecGetArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

        foreach_node(n, nodes)
        {
          double transition = smoothstep(1, (phi_wall_ptr[n]/diag - 10.)/20.);
          //        phi_free_ptr[n] = smooth_max(phi_free_ptr[n], phi_wall_ptr[n] - 12.*diag, 4.*diag);
          phi_free_ptr[n] = (1.-transition)*phi_free_ptr[n] + transition*MAX(phi_free_ptr[n], phi_wall_ptr[n] - 10.*diag);
        }

        ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
        ierr = VecRestoreArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

        ls.reinitialize_2nd_order(phi_free);

        // correct for volume loss
        for (short i = 0; i < volume_corrections(); ++i)
        {
          double volume_cur = integration.measure_of_domain();
          double intf_len   = integration.measure_of_interface(0);
          double correction = (volume_cur-volume)/intf_len;

          VecShiftGhost(phi_free, correction);
          correction_total += correction;

          double volume_cur2 = integration.measure_of_domain();

          //        PetscPrintf(mpi.comm(), "Volume loss: %e, after correction: %e\n", (volume_cur-volume)/volume, (volume_cur2-volume)/volume);
        }
      }
    }

    foreach_dimension(dim)
    {
      ierr = VecDestroy(surf_tns_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(mu_m_grad    [dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal_wall  [dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal_free  [dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi_effe);             CHKERRXX(ierr);
    ierr = VecDestroy(kappa);                CHKERRXX(ierr);
    ierr = VecDestroy(shape_grad_free);      CHKERRXX(ierr);
    ierr = VecDestroy(shape_grad_free_full); CHKERRXX(ierr);
    ierr = VecDestroy(shape_grad_ptcl);      CHKERRXX(ierr);
    ierr = VecDestroy(tmp);                  CHKERRXX(ierr);
    ierr = VecDestroy(surf_tns);             CHKERRXX(ierr);
    ierr = VecDestroy(cos_angle);            CHKERRXX(ierr);
    ierr = VecDestroy(velo_free);            CHKERRXX(ierr);
    ierr = VecDestroy(velo_free_full);       CHKERRXX(ierr);
    ierr = VecDestroy(particles_number);     CHKERRXX(ierr);
    ierr = VecDestroy(integrand);            CHKERRXX(ierr);

    iteration++;
    energy_old = energy;
  }

  ierr = VecDestroy(phi_free); CHKERRXX(ierr);
  ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
  ierr = VecDestroy(phi_ptcl); CHKERRXX(ierr);
  ierr = VecDestroy(mu_p);     CHKERRXX(ierr);
  ierr = VecDestroy(mu_m);     CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
