
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
 */

// comptational domain
param_t<double> DIMPM( xmin (pl, -1, "xmin", "xmin"),
                       xmax (pl,  1, "xmax", "xmax"),
                       ymin (pl, -1, "ymin", "ymin"),
                       ymax (pl,  1, "ymax", "ymax"),
                       zmin (pl, -1, "zmin", "zmin"),
                       zmax (pl,  1, "zmax", "zmax") );

param_t<bool> DIM( px (pl, 0, "px", "periodicity in x-dimension 0/1"),
                   py (pl, 0, "py", "periodicity in y-dimension 0/1"),
                   pz (pl, 0, "pz", "periodicity in z-dimension 0/1") );

param_t<int> DIM( nx (pl, 1, "nx", "number of trees in x-dimension"),
                  ny (pl, 1, "ny", "number of trees in y-dimension"),
                  nz (pl, 1, "nz", "number of trees in z-dimension") );

// grid parameters
#ifdef P4_TO_P8
param_t<int> lmin (pl, 5, "lmin", "min level of trees");
param_t<int> lmax (pl, 5, "lmax", "max level of trees");
#else
param_t<int> lmin (pl, 6, "lmin", "min level of trees");
param_t<int> lmax (pl, 6, "lmax", "max level of trees");
#endif
param_t<double> lip (pl, 1.2, "lip", "Lipschitz constant");
param_t<int>    band  (pl, 2,   "band" , "Uniform grid band");
param_t<bool>   refine_only_inside (pl, 1, "refine_only_inside", "Refine only inside");

//param_t<double> resolution (pl, 0.1, "resolution", "desired mesh size in Rg");
//param_t<int>    ladd (pl, 0, "lmax", "max level of trees");

// advection parameters
param_t<double> cfl                     (pl, 1.5,    "cfl", "CFL number");
param_t<double> cfl_v_min               (pl, 0.1,   "cfl_v_min", "");
param_t<double> cfl_v_max               (pl, 0.75,   "cfl_v_max", "");
param_t<double> cfl_w_min               (pl, 0.1,   "cfl_w_min", "");
param_t<double> cfl_w_max               (pl, 0.75,   "cfl_w_max", "");
param_t<double> cfl_change_rate         (pl, 0.5,    "cfl_change_rate", "");
param_t<bool>   use_neumann             (pl, 1,      "use_neumann", "Impose contact angle use Neumann BC 0/1");
param_t<bool>   compute_exact           (pl, 0,      "compute_exact", "Compute exact final shape (only for pure-curvature) 0/1");
param_t<int>    contact_angle_extension (pl, 0,      "contact_angle_extension", "Method for extending level-set function into wall: 0 - constant angle, 1 - , 2 - special");
param_t<int>    volume_corrections      (pl, 2,      "volume_corrections", "Number of volume correction after each move");
param_t<int>    max_iterations          (pl, 500,   "max_iterations", "Maximum number of advection steps");
param_t<double> tolerance               (pl, 1.0e-8, "tolerance", "Stopping criteria");

interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

// scft parameters
param_t<bool>   use_scft               (pl, 1,   "use_scft", "Turn on/off SCFT 0/1");
param_t<int>    max_scft_iterations    (pl, 100, "max_scft_iterations", "Maximum SCFT iterations");
param_t<int>    num_scft_subiterations (pl, 2,   "num_scft_subiterations", "Maximum SCFT iterations");
param_t<double> scft_tol               (pl, 1.e-4, "scft_tol", "Tolerance for SCFT");
param_t<int>    num_pre_iterations     (pl, 100,   "num_pre_iterations", "Maximum SCFT iterations");
param_t<bool>   smart_bc               (pl, 1,   "smart_bc", "");

// polymer
param_t<double> box_size (pl, 10, "box_size", "Box size in units of Rg");
param_t<double> f        (pl, .45, "f", "Fraction of polymer A");
param_t<double> XN       (pl, 20, "XN", "Flory-Higgins interaction parameter");
param_t<int>    ns       (pl, 100, "ns", "Discretization of polymer chain");
param_t<bool>   grafted  (pl, 0, "grafted", "Switch between free and grafted polymer chains");

// output parameters
param_t<bool> save_vtk        (pl, 1, "save_vtk", "");
param_t<bool> save_parameters (pl, 1, "save_parameters", "");
param_t<bool> save_data       (pl, 1, "save_data", "");
param_t<int>  save_every_dn   (pl, 1, "save_every_dn", ""); // for vtk

// problem setting

param_t<int>    geometry_ptcl (pl, 0, "geometry_ptcl", "Initial placement of particles: 1 - one particle, 2 - ...");
param_t<int>    geometry_free (pl, 1, "geometry_free", "Initial polymer shape: 1 - drop, 2 - film, 3 - combination");
param_t<int>    geometry_wall (pl, 0, "geometry_wall", "Wall geometry: 0 - no wall, 1 - wall, 2 - well");

param_t<bool>   minimize  (pl, 1, "minimize", "Turn on/off energy minimization (0/1)");
param_t<int>    ptcl_velo (pl, 3, "ptcl_velo", "Predifined velocity field for particles in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<int>    ptcl_rotn (pl, 1, "ptcl_rotn", "Predifined rotation in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<int>    free_velo (pl, 1, "free_velo", "Predifined velocity field for free surface in case of minimize=0: 0 - nothing, 1 - oscillations");

param_t<int>    wall_pattern (pl, 0, "wall_pattern", "Wall chemical pattern: 0 - no pattern");
param_t<int>    n_seed       (pl, 1, "n_seed", "Seed: 0 - zero, "
                                               "1 - random, "
                                               "2 - vertical stripes, "
                                               "3 - horizontal stripes, "
                                               "4 - dots, "
                                               "5 - spheres");
param_t<int>    n_example    (pl, 30, "n_example", "Number of predefined example");

param_t<int>    pairwise_potential_type  (pl, 0, "pairwise_potential_type", "Type of pairwise potential: 0 - quadratic, 1 - 1/(e^x-1)");
param_t<double> pairwise_potential_mag   (pl, 2.0,   "pairwise_potential_mag", "Magnitude of pairwise potential");
param_t<double> pairwise_potential_width (pl, 3, "pairwise_potential_width", "Width of pairwise potential");

// surface energies
param_t<double> sqrtXN_free_avg     (pl, 1.0, "sqrtXN_free_avg",     "Polymer-air surface energy strength: average");
param_t<double> sqrtXN_free_dif     (pl, 0.5, "sqrtXN_free_dif",     "Polymer-air surface energy strength: difference");
param_t<double> sqrtXN_ptcl_dif_max (pl, 1.0, "sqrtXN_ptcl_dif_max", "Polymer-particle surface energy strength: difference");
param_t<double> sqrtXN_ptcl_dif_min (pl,-1.0, "sqrtXN_ptcl_dif_min", "Polymer-particle surface energy strength: difference");
param_t<double> sqrtXN_wall_dif_max (pl, 0.5, "sqrtXN_wall_dif_max", "Polymer-wall surface energy strength: difference");
param_t<double> sqrtXN_wall_dif_min (pl, 0.5, "sqrtXN_wall_dif_min", "Polymer-wall surface energy strength: difference");
param_t<double> sqrtXN_wall_avg     (pl, 1.0, "sqrtXN_wall_avg", "Polymer-wall surface energy strength: difference");


param_t<int> air_wall_energy_type (pl, 1, "air_energy_type", "Method for setting wall surface energy: "
                                                        "0 - based on A block, "
                                                        "1 - based on B block, "
                                                        "2 - based on weighted average of A and B blocks");

param_t<double> contact_angle (pl, 75, "contact_angle" , "Contact angle");

// geometry parameters
param_t<double> drop_r      (pl, 2.55111, "drop_r", "");
param_t<double> DIM(drop_x  (pl, .0079,   "drop_x", ""),
                    drop_y  (pl, -.013,   "drop_y", ""),
                    drop_z  (pl, .0,      "drop_z", "") );
param_t<double> drop_r0     (pl, 0.4,     "drop_r0", "");
param_t<double> drop_k      (pl, 5,       "drop_k", "");
param_t<double> drop_deform (pl, 0.0,     "drop_deform", "");

param_t<double> film_eps     (pl, -0.0, "film_eps", ""); // curvature
param_t<double> DIM( film_nx (pl, 0,    "film_nx", ""),
                     film_ny (pl, 1,    "film_ny", ""),
                     film_nz (pl, 0,    "film_nz", "") );
param_t<double> DIM( film_x  (pl, .0,   "film_x", ""),
                     film_y  (pl, 1.03,   "film_y", ""),
                     film_z  (pl, .0,   "film_z", "") );
param_t<double> film_perturb (pl, 1.e-4,   "film_perturb", "");

param_t<double> wall_eps     (pl, -0.0, "wall_eps", ""); // curvature
param_t<double> DIM( wall_nx (pl, -0,   "wall_nx", ""),
                     wall_ny (pl, -1,   "wall_ny", ""),
                     wall_nz (pl, -0,   "wall_nz", "") );
param_t<double> DIM( wall_x  (pl, .0,   "wall_x", ""),
                     wall_y  (pl, -.0003,  "wall_y", ""),
                     wall_z  (pl, .0,   "wall_z", "") );
param_t<double> wall_r       (pl, 2.5,   "wall_r", "");

param_t<double> DIM( well_x  (pl, 0.00,   "well_x", "Well geometry: center"),
                     well_y  (pl, 2.24, "well_y", "Well geometry: position"),
                     well_z  (pl, 0.00,   "well_z", "") );
param_t<double> well_h (pl, 3.50,    "well_h", "Well geometry: depth");
param_t<double> well_w (pl, 10.77,   "well_w", "Well geometry: width");
param_t<double> well_r (pl, 0.10,   "well_r", "Well geometry: corner smoothing");

param_t<int> num_cyls (pl, 5,   "num_cyls", "Number of cylinders in the well");

param_t<int> DIM( num_ptcl_x (pl, 4, "num_ptcl_x", ""),
                  num_ptcl_y (pl, 4, "num_ptcl_y", ""),
                  num_ptcl_z (pl, 1, "num_ptcl_z", "") );
param_t<double> initial_rotation (pl, -0.25*PI, "initial_rotation", "");
param_t<double> initial_rotation_rand (pl, -0.25*PI, "initial_rotation_rand", "");
param_t<double> rod_radius (pl, 0.1, "rod_radius", "");
param_t<double> rod_length (pl, 0.5, "rod_length", "");

param_t<bool> DIM( restrict_motion_x (pl, 0, "restrict_motion_x", ""),
                   restrict_motion_y (pl, 0, "restrict_motion_y", ""),
                   restrict_motion_z (pl, 0, "restrict_motion_z", "") );
param_t<bool> restrict_rotation (pl, 0, "restrict_rotation", "");

param_t<int> num_submotions (pl, 10, "num_submotions", "");

inline double lam_bulk_period()
{
  return 4.*pow(8.*XN()/3./pow(PI,4.),1./6.);
//  return 4.*pow(8.*XN()/3./pow(PI,4.),1./6.)/box_size();
}

void set_wall_surface_energies()
{
  switch (air_wall_energy_type()) {
    case 0: sqrtXN_wall_avg.val = -cos(contact_angle()*PI/180.)*(sqrtXN_free_avg() + .5*sqrtXN_free_dif()) - .5*sqrtXN_wall_dif_min(); break;
    case 1: sqrtXN_wall_avg.val = -cos(contact_angle()*PI/180.)*(sqrtXN_free_avg() - .5*sqrtXN_free_dif()) + .5*sqrtXN_wall_dif_min(); break;
    case 2: sqrtXN_wall_avg.val = -cos(contact_angle()*PI/180.)*(sqrtXN_free_avg()); break;
    default:
      throw std::invalid_argument("Invalid method for setting wall surface energies");
  }
}

void set_parameters()
{
  switch (n_example())
  {
    case 0: // periodic lamellar phase
      break;
    case 1: // periodic cylindrical phase
      break;
    case 2: // droplet
      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 0;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .3;
      XN.val       = 30;
      ns.val       = 60;

      break;
    case 3: // droplet on substrate with horizontal lamellae
    {
      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 1;

      n_seed.val = 3;

      xmin.val = -1; xmax.val = 1; px.val = 1; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 100;

      f.val   = .5;
      XN.val  = 30;
      ns.val  = 60;

      sqrtXN_free_avg.val     = 15.0;
      sqrtXN_free_dif.val     =  1.0;
      sqrtXN_wall_dif_max.val =  -1.0;
      sqrtXN_wall_dif_min.val =  -1.0;

      air_wall_energy_type.val = 0;
      contact_angle.val = 30;

      double h = 10;

      film_nx.val = 0;
      film_ny.val = 1;

      film_x.val = 0;
      film_y.val = 0.03;

      drop_r.val      = h/(1.-cos(contact_angle.val/180*PI));
      drop_x.val      = 0.02;
      drop_y.val      = film_y.val - drop_r.val + h;
      drop_r0.val     = 2.5;
      drop_k.val      = 5;
      drop_deform.val = 0;
    }

      break;
    case 4: // droplet on substrate with vertical lamellae
      break;
    case 5: // droplet on substrate with cylinders
      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 1;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .3;
      XN.val       = 30;
      ns.val       = 60;
      break;
    case 6: // film with cylinders in a groove
      geometry_ptcl.val = 0;
      geometry_free.val = 2;
      geometry_wall.val = 2;

      n_seed.val = 5;

      xmin.val = -1; xmax.val = 1; px.val = 1; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .3;
      XN.val       = 30;
      ns.val       = 60;
      well_x.val   = 0.00;
      well_y.val   = 5.03;
      well_h.val   = 10;
      well_w.val   = 3.5*5;
      well_r.val   = 0.5;

      film_y.val   = well_y.val-well_h.val + 3.5;

      sqrtXN_free_avg.val     =  5.0;
      sqrtXN_free_dif.val     =  2.0;
      sqrtXN_wall_dif_max.val =  2.0;
      sqrtXN_wall_dif_min.val =  2.0;

      air_wall_energy_type.val = 1;
      contact_angle.val = 30;


      num_cyls.val = 5;

      break;
    case 7: // film with lamellae in a groove
      geometry_ptcl.val = 0;
      geometry_free.val = 2;
      geometry_wall.val = 2;

      n_seed.val = 2;

      xmin.val = -1; xmax.val = 1; px.val = 1; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 15;
      f.val        = .5;
      XN.val       = 30;
      ns.val       = 60;
      well_x.val   = 0.00;
      well_y.val   = 5.03;
      well_h.val   = 10;
      well_w.val   = 3.3*6;
      well_r.val   = 0.5;

      film_y.val   = well_y.val-well_h.val + 4.5;

      sqrtXN_free_avg.val     = 10.0;
      sqrtXN_free_dif.val     =  0.0;
      sqrtXN_wall_dif_max.val =  0.0;
      sqrtXN_wall_dif_min.val =  0.0;

      air_wall_energy_type. val = 1;
      contact_angle.val = 120;

      num_cyls.val = 4;

      break;
    case 10: // one rod
      xmin.val = -1; xmax.val = 1; px.val = 1; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif
      box_size.val = .9;

      f.val        = .5;
      XN.val       = 30;
      ns.val       = 60;

      geometry_ptcl.val = 4;
      geometry_free.val = 0;
      geometry_wall.val = 0;

      initial_rotation.val = 0.25*PI;
      initial_rotation_rand.val = 0;
      rod_radius.val = 0.1;
      rod_length.val = 0.5;

      restrict_motion_x.val = 1;
      restrict_motion_y.val = 0;
//      restrict_motion_z.val = ;
      restrict_rotation.val = 0;
      num_submotions.val = 10;

      n_seed.val = 3;

      num_ptcl_x.val = 1;
      num_ptcl_y.val = 1;

      pairwise_potential_type.val = 0;
      pairwise_potential_mag.val = 0;
      pairwise_potential_width.val = 3;

      sqrtXN_ptcl_dif_max.val =  5;
      sqrtXN_ptcl_dif_min.val = -5;

      num_pre_iterations.val = 0;
      break;
    case 20: // flower-shaped simple scft test
      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 0;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .33;
      XN.val       = 20;
      ns.val       = 60;

      sqrtXN_free_avg.val     = 0.0;
      sqrtXN_free_dif.val     = 1.0;

      drop_r.val      = 5.;
      drop_x.val      = 0.02;
      drop_y.val      = 0.03;
      drop_k.val      = 5;
      drop_deform.val = 0.3;

      max_scft_iterations.val = 4000;
      num_scft_subiterations.val = 2;
      max_iterations.val = 1;
    break;
    case 21: // simple scft test
      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 4;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .33;
      XN.val       = 20;
      ns.val       = 60;

      sqrtXN_free_avg.val     = 0.0;
      sqrtXN_free_dif.val     = 1.0;

      wall_pattern.val = 1;
      sqrtXN_wall_avg.val = 0;
      sqrtXN_wall_dif_max.val =-1;
      sqrtXN_wall_dif_min.val =-1;

      drop_r.val      = 6.5;
      drop_x.val      = 0.01;
      drop_y.val      =-0.02;
      drop_k.val      = 15;
      drop_deform.val = 0.00;

      wall_x.val =  0.01;
      wall_y.val = -0.02;
      wall_eps.val = 0.0;
      wall_nx.val = 0;
      wall_ny.val = 1;

      well_w.val = 8;

      max_scft_iterations.val = 4000;
      num_scft_subiterations.val = 2;
      max_iterations.val = 1;
    break;
    case 22: // drop on corrugated surface simple scft test
      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 3;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .33;
      XN.val       = 20;
      ns.val       = 60;

      sqrtXN_free_avg.val     = 0.0;
      sqrtXN_free_dif.val     = 1.0;

      wall_pattern.val = 1;
      sqrtXN_wall_avg.val = 0;
      sqrtXN_wall_dif_max.val =-5;
      sqrtXN_wall_dif_min.val = 5;

      drop_r.val      = 6.5;
      drop_x.val      = 0.1;
      drop_y.val      = 2.1;
      drop_k.val      = 15;
      drop_deform.val = 0.00;

      wall_y.val = 0.01;
      wall_eps.val = 0.25;

      max_scft_iterations.val = 4000;
      num_scft_subiterations.val = 2;
      max_iterations.val = 1;
    break;
    case 23: // flower-shaped simple scft test, grafted
      grafted.val = 1;

      geometry_ptcl.val = 0;
      geometry_free.val = 1;
      geometry_wall.val = 0;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif

      box_size.val = 10;
      f.val        = .4;
      XN.val       = 20;
      ns.val       = 60;

      sqrtXN_free_avg.val     = 0.0;
      sqrtXN_free_dif.val     = 0.0;

      drop_r.val      = 5.;
      drop_x.val      = 0.02;
      drop_y.val      = 0.03;
      drop_k.val      = 5;
      drop_deform.val = 0.3;

      max_scft_iterations.val = 1000;
      num_scft_subiterations.val = 3;
      max_iterations.val = 1;
      n_seed.val = 3;

    break;
    case 30: // grafted rod
      grafted.val = 1;

      xmin.val = -1; xmax.val = 1; px.val = 0; nx.val = 1;
      ymin.val = -1; ymax.val = 1; py.val = 0; ny.val = 1;
#ifdef P4_TO_P8
      zmin.val = -1; zmax.val = 1; pz.val = 0; nz.val = 1;
#endif
      box_size.val = 5;

      f.val        = .37;
      XN.val       = 20;
      ns.val       = 60;

      geometry_ptcl.val = 4;
      geometry_free.val = 1;
      geometry_wall.val = 0;

      sqrtXN_free_avg.val     = 0.0;
      sqrtXN_free_dif.val     = 0.0;

      drop_r.val      = 2.0;
      drop_x.val      = 0.02;
      drop_y.val      = 0.03;
      drop_k.val      = 5;
      drop_deform.val = 0.1;

      initial_rotation.val = 0.25*PI;
      initial_rotation_rand.val = 0;
      rod_radius.val = 0.5;
      rod_length.val = 0.5;

      restrict_motion_x.val = 0;
      restrict_motion_y.val = 0;
//      restrict_motion_z.val = ;
      restrict_rotation.val = 0;
      num_submotions.val = 10;

      max_scft_iterations.val = 1000;
      num_scft_subiterations.val = 3;
      max_iterations.val = 1;
      n_seed.val = 3;

      num_ptcl_x.val = 1;
      num_ptcl_y.val = 1;

      pairwise_potential_type.val = 0;
      pairwise_potential_mag.val = 0;
      pairwise_potential_width.val = 3;

      sqrtXN_ptcl_dif_max.val = 0;
      sqrtXN_ptcl_dif_min.val = 0;

      num_pre_iterations.val = 0;
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
    return sqrtXN_free_avg()+.5*sqrtXN_free_dif();
  }
} gamma_Aa_cf;

class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrtXN_free_avg()-.5*sqrtXN_free_dif();
  }
} gamma_Ba_cf;

class gamma_Aw_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (wall_pattern())
    {
      case 0: return sqrtXN_wall_avg()+.5*sqrtXN_wall_dif_min();
      case 1: return sqrtXN_wall_avg()+.25*(sqrtXN_wall_dif_min() + sqrtXN_wall_dif_max())
            + .25*(sqrtXN_wall_dif_max()-sqrtXN_wall_dif_min())*cos(2.*PI*(x-wall_x())/lam_bulk_period());
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
      case 0: return sqrtXN_wall_avg()-.5*sqrtXN_wall_dif_min();
      case 1: return sqrtXN_wall_avg()-.25*(sqrtXN_wall_dif_min() + sqrtXN_wall_dif_max())
            - .25*(sqrtXN_wall_dif_max()-sqrtXN_wall_dif_min())*cos(2.*PI*(x-wall_x())/lam_bulk_period());
      default: throw std::invalid_argument("Error: Invalid wall pattern number\n");
    }
  }
} gamma_Bw_cf;

double gamma_Ap_max() { return MAX(0.0,  sqrtXN_ptcl_dif_max()); }
double gamma_Ap_min() { return MAX(0.0,  sqrtXN_ptcl_dif_min()); }
double gamma_Bp_max() { return MAX(0.0, -sqrtXN_ptcl_dif_max()); }
double gamma_Bp_min() { return MAX(0.0, -sqrtXN_ptcl_dif_min()); }

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
      case 1: // possibly curved wall
        {
          double norm = ABSD(wall_nx(), wall_ny(), wall_nz());
          return - SUMD( (x-wall_x())*((x-wall_x())*wall_eps() - 2.*wall_nx() / norm),
                         (y-wall_y())*((y-wall_y())*wall_eps() - 2.*wall_ny() / norm),
                         (z-wall_z())*((z-wall_z())*wall_eps() - 2.*wall_nz() / norm) )
              / (  ABSD((x-wall_x())*wall_eps() - wall_nx() / norm,
                        (y-wall_y())*wall_eps() - wall_ny() / norm,
                        (z-wall_z())*wall_eps() - wall_nz() / norm)  + 1. );
        }
      case 2: // well/groove
        {
          double phi_top   = well_y() - y;
          double phi_bot   = well_y()-well_h() - y;
          double phi_walls = MAX(x-well_x()-.5*well_w(), -(x-well_x())-.5*well_w());

          return smooth_max(phi_bot, smooth_min(phi_top, phi_walls, well_r()), well_r());
        }
      case 3: // corrugated wall
      {
        return -(y-wall_y()) + wall_eps()*cos(2.*PI*(x-wall_x())/lam_bulk_period());
      }
      case 4: // slit
        return fabs(SUMD((x-wall_x())*wall_nx(), (y-wall_y())*wall_ny(), (z-wall_z())*wall_nz()))
            / ABSD(wall_nx(), wall_ny(), wall_nz()) - .5*well_w();

//        double phi_walls = MAX(x-well_x()-.5*well_w(), -(x-well_x())-.5*well_w());
//        return MAX(
      default:
        throw;
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
      case 1: return ABSD(x-drop_x(), y-drop_y(), z-drop_z()) - drop_r()*(1.+drop_deform()*cos(drop_k()*atan2(x-drop_x(),y-drop_y()))
              ONLY3D(*(1.-cos(2.*acos((z-drop_z())/(ABSD(x-drop_x(), y-drop_y(), z-drop_z()) + 1.e-12))))));
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
      case 3: return MAX( cos((contact_angle()-90)*PI/180)*(x-drop_x()) + sin((contact_angle()-90)*PI/180)*(y-drop_y()-drop_r()),
                         -cos((contact_angle()-90)*PI/180)*(x-drop_x()) + sin((contact_angle()-90)*PI/180)*(y-drop_y()-drop_r()));
      case 4: return MIN( (y-wall_y()-.5*lam_bulk_period()),
                          MAX((y-wall_y()-1.*lam_bulk_period()), fabs(x-.5*(xmax()+xmin())) - .25*(xmax()-xmin())),
                          MAX((y-wall_y()-1.5*lam_bulk_period()), fabs(x-.5*(xmax()+xmin())) - .125*(xmax()-xmin())));
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_free_cf;


class mu_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z) ) const
  {
    switch (n_seed())
    {
      case 0: return 0;
      case 1: return 0.1*XN()*(2.*double(rand())/double(RAND_MAX)-1.);
      case 2: {
        double nx = (xmax()-xmin())/lam_bulk_period(); if (px() == 1) nx = round(nx);
        return .05*XN()*cos(2.*PI*x/(xmax()-xmin())*nx);
      }
      case 3: {
        double ny = (ymax()-ymin())/lam_bulk_period(); if (py() == 1) ny = round(ny);
        return .05*XN()*sin(2.*PI*y/(ymax()-ymin())*ny);
      }
      case 4: {
        double nx = (xmax()-xmin())/lam_bulk_period(); if (px() == 1) nx = round(nx);
        double ny = (ymax()-ymin())/lam_bulk_period(); if (py() == 1) ny = round(ny);
#ifdef P4_TO_P8
        double nz = (zmax()-zmin())/lam_bulk_period(); if (pz() == 1) nz = round(nz);
#endif
        return .5*XN()*MULTD( cos(2.*PI*x/(xmax()-xmin())*nx),
                              cos(2.*PI*y/(ymax()-ymin())*ny),
                              cos(2.*PI*z/(zmax()-zmin())*nz));
        }
      case 5:
      {
        double rc = 1;
        double yc = .5*(film_y.val+(well_y.val-well_h.val));
        double dx = well_w.val/double(num_cyls());

        if (num_cyls.val%2==0) x += dx/2.;

        double xc = round(x/dx)*dx;


        return .5*XN()*tanh((rc - ABS2(x-xc, y-yc))*sqrt(XN()));
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
  CF_DIM DIM( *phix_cf, *phiy_cf, *phiz_cf );
  CF_DIM *kappa_cf;

  CF_DIM *gA_cf, DIM( *gAx_cf, *gAy_cf, *gAz_cf);
  CF_DIM *gB_cf, DIM( *gBx_cf, *gBy_cf, *gBz_cf);

  particle_t()
  {
    foreach_dimension(dim) {
      xyz [dim] = 0;
      axis[dim] = 0;
    }

    rot      = 0;
    phi_cf   = NULL; EXECD( phix_cf  = NULL, phiy_cf  = NULL, phiz_cf  = NULL);
    kappa_cf = NULL;
    gA_cf    = NULL; EXECD( gAx_cf  = NULL, gAy_cf  = NULL, gAz_cf  = NULL);
    gB_cf    = NULL; EXECD( gBx_cf  = NULL, gBy_cf  = NULL, gBz_cf  = NULL);
  }

  inline void wrap_periodically(DIM(double &x, double &y, double &z))
  {
    if (px()) {
      if (x-xyz[0] >  .5*(xmax()-xmin())) x -= (xmax()-xmin());
      if (x-xyz[0] < -.5*(xmax()-xmin())) x += (xmax()-xmin());
    }

    if (py()) {
      if (y-xyz[1] >  .5*(ymax()-ymin())) y -= (ymax()-ymin());
      if (y-xyz[1] < -.5*(ymax()-ymin())) y += (ymax()-ymin());
    }
#ifdef P4_TO_P8
    if (pz()) {
      if (z-xyz[2] >  .5*(zmax()-zmin())) z -= (zmax()-zmin());
      if (z-xyz[2] < -.5*(zmax()-zmin())) z += (zmax()-zmin());
    }
#endif

  }

  inline double sample_func(CF_DIM *cf, DIM(double x, double y, double z))
  {
    wrap_periodically(DIM(x,y,z));
    double DIM( X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot),
                Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot),
                Z = z );
    return (*cf)(DIM(X,Y,Z));
  }

  inline double sample_func_x(DIM(CF_DIM *cfx, CF_DIM *cfy, CF_DIM *cfz), DIM(double x, double y, double z))
  {
    wrap_periodically(DIM(x,y,z));
    double DIM( X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot),
                Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot),
                Z = z );
    return (*cfx)(DIM(X,Y,Z))*cos(rot) + (*cfy)(DIM(X,Y,Z))*sin(rot);
  }

  inline double sample_func_y(DIM(CF_DIM *cfx, CF_DIM *cfy, CF_DIM *cfz), DIM(double x, double y, double z))
  {
    wrap_periodically(DIM(x,y,z));
    double DIM( X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot),
                Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot),
                Z = z );
    return -(*cfx)(DIM(X,Y,Z))*sin(rot) + (*cfy)(DIM(X,Y,Z))*cos(rot);
  }
#ifdef P4_TO_P8
  inline double sample_func_z(DIM(CF_DIM *cfx, CF_DIM *cfy, CF_DIM *cfz), DIM(double x, double y, double z))
  {
    wrap_periodically(DIM(x,y,z));
    double DIM( X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot),
                Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot),
                Z = z );
    return (*cfz)(DIM(X,Y,Z));
  }
#endif

  inline double phi  (DIM(double x, double y, double z)) { return sample_func(phi_cf,   DIM(x,y,z)); }
  inline double kappa(DIM(double x, double y, double z)) { return sample_func(kappa_cf, DIM(x,y,z)); }
  inline double gA   (DIM(double x, double y, double z)) { return sample_func(gA_cf,    DIM(x,y,z)); }
  inline double gB   (DIM(double x, double y, double z)) { return sample_func(gB_cf,    DIM(x,y,z)); }

  inline double gAx  (DIM(double x, double y, double z)) { return sample_func_x(DIM(gAx_cf,  gAy_cf,  gAz_cf ), DIM(x,y,z)); }
  inline double gBx  (DIM(double x, double y, double z)) { return sample_func_x(DIM(gBx_cf,  gBy_cf,  gBz_cf ), DIM(x,y,z)); }
  inline double phix (DIM(double x, double y, double z)) { return sample_func_x(DIM(phix_cf, phiy_cf, phiz_cf), DIM(x,y,z)); }

  inline double gAy  (DIM(double x, double y, double z)) { return sample_func_y(DIM(gAx_cf,  gAy_cf,  gAz_cf ), DIM(x,y,z)); }
  inline double gBy  (DIM(double x, double y, double z)) { return sample_func_y(DIM(gBx_cf,  gBy_cf,  gBz_cf ), DIM(x,y,z)); }
  inline double phiy (DIM(double x, double y, double z)) { return sample_func_y(DIM(phix_cf, phiy_cf, phiz_cf), DIM(x,y,z)); }
#ifdef P4_TO_P8
  inline double gAz  (DIM(double x, double y, double z)) { return sample_func_z(DIM(gAx_cf,  gAy_cf,  gAz_cf ), DIM(x,y,z)); }
  inline double gBz  (DIM(double x, double y, double z)) { return sample_func_z(DIM(gBx_cf,  gBy_cf,  gBz_cf ), DIM(x,y,z)); }
  inline double phiz (DIM(double x, double y, double z)) { return sample_func_z(DIM(phix_cf, phiy_cf, phiz_cf), DIM(x,y,z)); }
#endif
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

  double operator()( DIM( double x, double y, double z ) ) const
  {
    double phi_max = -10;
    for( size_t n = 0; n < particles->size(); n++ ) {
      double phi_cur = particles->at(n).phi( DIM( x,y,z ) );
      if( phi_cur > phi_max ) phi_max = phi_cur;
    }

    if( phi_max != phi_max ) throw;

    return phi_max;
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

  double operator()( DIM( double x, double y, double z ) ) const
  {
    int particle_num = 0;
    double phi_max = -10;
    for( size_t n = 0; n < particles->size(); n++ ) {
      double phi_cur = particles->at( n ).phi( DIM( x,y,z ) );
      if( phi_cur > phi_max ) {
        particle_num = n;
        phi_max = phi_cur;
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

class radial_gamma_cf_t : public CF_DIM // THIS WILL BEHAVE BAD IN 3D
{
  double gamma_max;
  double gamma_min;
  double k;
  double theta;
  cf_value_type_t what;
public:
  radial_gamma_cf_t(cf_value_type_t what, double gamma_min, double gamma_max, double k, double theta = 0)
    : gamma_max(gamma_max), gamma_min(gamma_min), k(k), theta(theta), what(what) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    double t = atan2(y,x);
    double r = sqrt(x*x + y*y);

    switch (what)
    {
      case VAL: return gamma_min + (gamma_max-gamma_min)*0.5*(1.+cos(k*(t-theta)));
      case DDX: return -(gamma_max-gamma_min)*0.5*k*sin(k*(t-theta))*(-y/r/r);
      case DDY: return -(gamma_max-gamma_min)*0.5*k*sin(k*(t-theta))*( x/r/r);
#ifdef P4_TO_P8
      case DDZ: return 0;
#endif
      default: throw;
    }
  }
};

struct radial_gamma_t
{
  radial_gamma_cf_t val, DIM( ddx, ddy, ddz );

  radial_gamma_t(double gamma_min, double gamma_max, double k, double theta = 0)
    : val(VAL, gamma_min, gamma_max, k, theta),
      DIM( ddx(DDX, gamma_min, gamma_max, k, theta),
           ddy(DDY, gamma_min, gamma_max, k, theta),
           ddz(DDY, gamma_min, gamma_max, k, theta) ) {}
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

      static radial_gamma_t gA(gamma_Ap_min(), gamma_Ap_max(), 0);
      static radial_gamma_t gB(gamma_Bp_min(), gamma_Bp_max(), 0);

      p.phi_cf   = &domain.phi;
      EXECD( p.phix_cf  = &domain.phi_x,
             p.phiy_cf  = &domain.phi_y,
             p.phiz_cf  = &domain.phi_z );
      p.kappa_cf = &domain.phi_c;

      p.gA_cf  = &gA.val; EXECD( p.gAx_cf = &gA.ddx, p.gAy_cf = &gA.ddy, p.gAz_cf = &gA.ddz);
      p.gB_cf  = &gB.val; EXECD( p.gBx_cf = &gB.ddx, p.gBy_cf = &gB.ddy, p.gBz_cf = &gB.ddz);

      EXECD( p.xyz[0] = -0.27, p.xyz[1] = -0.23, p.xyz[2] = 0 ); particles.push_back(p);
    }
      break;

    case 2:
    {
      static radial_shaped_domain_t sphere(0.2, DIM(0,0,0), -1, 0, 0.0, 0.0);
      static radial_shaped_domain_t star(0.2, DIM(0,0,0), -1, 3, 0.3, 0.0);
      static capsule_domain_t capsule(0.1, DIM(0,0,0), 0.3, -1);

      static radial_gamma_t sphere_gA(gamma_Ap_min(), gamma_Ap_max(), 1);
      static radial_gamma_t sphere_gB(gamma_Bp_min(), gamma_Bp_max(), 1);

      static radial_gamma_t capsule_gA(gamma_Ap_min(), gamma_Ap_max(), 0);
      static radial_gamma_t capsule_gB(gamma_Bp_min(), gamma_Bp_max(), 0);

      static radial_gamma_t star_gA(gamma_Ap_min(), gamma_Ap_max(), 3);
      static radial_gamma_t star_gB(gamma_Bp_min(), gamma_Bp_max(), 3);

      p.phi_cf   = &sphere.phi;   p.gA_cf  = &sphere_gA.val; p.gB_cf  = &sphere_gB.val;
      p.phix_cf  = &sphere.phi_x; p.gAx_cf = &sphere_gA.ddx; p.gBx_cf = &sphere_gB.ddx;
      p.phiy_cf  = &sphere.phi_y; p.gAy_cf = &sphere_gA.ddy; p.gBy_cf = &sphere_gB.ddy;
#ifdef P4_TO_P8
      p.phiz_cf  = &sphere.phi_z; p.gAz_cf = &sphere_gA.ddz; p.gBz_cf = &sphere_gB.ddz;
#endif
      p.kappa_cf = &sphere.phi_c;
      EXECD( p.xyz[0] = -0.30, p.xyz[1] = -0.30, p.xyz[2] = 0); particles.push_back(p);

      p.phi_cf   = &star.phi;   p.gA_cf  = &star_gA.val; p.gB_cf  = &star_gB.val;
      p.phix_cf  = &star.phi_x; p.gAx_cf = &star_gA.ddx; p.gBx_cf = &star_gB.ddx;
      p.phiy_cf  = &star.phi_y; p.gAy_cf = &star_gA.ddy; p.gBy_cf = &star_gB.ddy;
#ifdef P4_TO_P8
      p.phiz_cf  = &star.phi_z; p.gAz_cf = &star_gA.ddz; p.gBz_cf = &star_gB.ddz;
#endif
      p.kappa_cf = &star.phi_c;
      EXECD( p.xyz[0] = +0.40, p.xyz[1] = -0.20, p.xyz[2] = 0); particles.push_back(p);

      p.phi_cf   = &capsule.phi;   p.gA_cf  = &capsule_gA.val; p.gB_cf  = &capsule_gB.val;
      p.phix_cf  = &capsule.phi_x; p.gAx_cf = &capsule_gA.ddx; p.gBx_cf = &capsule_gB.ddx;
      p.phiy_cf  = &capsule.phi_y; p.gAy_cf = &capsule_gA.ddy; p.gBy_cf = &capsule_gB.ddy;
#ifdef P4_TO_P8
      p.phiz_cf  = &capsule.phi_z; p.gAz_cf = &capsule_gA.ddz; p.gBz_cf = &capsule_gB.ddz;
#endif
      p.kappa_cf = &capsule.phi_c;
      EXECD( p.xyz[0] = -0.20, p.xyz[1] = +0.25, p.xyz[2] = 0); particles.push_back(p);
    }
      break;

    case 3: // perturbed grid of spheres
    {
      static radial_shaped_domain_t domain(0.05, DIM(0,0,0), -1, 0, 0.0, 0.0);
      static radial_gamma_t gA(gamma_Ap_min(), gamma_Ap_max(), 1);
      static radial_gamma_t gB(gamma_Bp_min(), gamma_Bp_max(), 1);
      int DIM( n = num_ptcl_x.val,
               m = num_ptcl_y.val,
               l = num_ptcl_z.val );

      p.phi_cf   = &domain.phi;
      EXECD( p.phix_cf  = &domain.phi_x,
             p.phiy_cf  = &domain.phi_y,
             p.phiz_cf  = &domain.phi_z );
      p.kappa_cf = &domain.phi_c;

      p.gA_cf  = &gA.val; EXECD( p.gAx_cf = &gA.ddx, p.gAy_cf = &gA.ddy, p.gAz_cf = &gA.ddz);
      p.gB_cf  = &gB.val; EXECD( p.gBx_cf = &gB.ddx, p.gBy_cf = &gB.ddy, p.gBz_cf = &gB.ddz);

      double DIM( space_x = (nx.val == 1 ? (xmax()-xmin())/double(n) : (xmax()-xmin())/double(n)),
                  space_y = (ny.val == 1 ? (ymax()-ymin())/double(m) : (ymax()-ymin())/double(m)),
                  space_z = (nz.val == 1 ? (zmax()-zmin())/double(l) : (zmax()-zmin())/double(l)) );

      for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
          ONLY3D( for (int k = 0; k < l; ++k))
          {
            EXECD( p.xyz[0] = .5*(xmin()+xmax()) + double(i-n/2)*space_x + double(1-n%2)*.5*space_x,
                   p.xyz[1] = .5*(ymin()+ymax()) + double(j-m/2)*space_y + double(1-m%2)*.5*space_y,
                   p.xyz[2] = .5*(zmin()+zmax()) + double(k-l/2)*space_z + double(1-l%2)*.5*space_z );
            p.rot = initial_rotation.val;
            particles.push_back(p);
          }
    }
    break;

    case 4: // perturbed grid of capsules
    {
      static capsule_domain_t domain(rod_radius.val, DIM(0,0,0), rod_length.val, -1);
      static radial_gamma_t gA(gamma_Ap_min(), gamma_Ap_max(), 1);
      static radial_gamma_t gB(gamma_Bp_min(), gamma_Bp_max(), 1);
      int DIM( n = num_ptcl_x.val,
               m = num_ptcl_y.val,
               l = num_ptcl_z.val );

      p.phi_cf   = &domain.phi;
      EXECD( p.phix_cf  = &domain.phi_x,
             p.phiy_cf  = &domain.phi_y,
             p.phiz_cf  = &domain.phi_z );
      p.kappa_cf = &domain.phi_c;

      p.gA_cf  = &gA.val; EXECD( p.gAx_cf = &gA.ddx, p.gAy_cf = &gA.ddy, p.gAz_cf = &gA.ddz);
      p.gB_cf  = &gB.val; EXECD( p.gBx_cf = &gB.ddx, p.gBy_cf = &gB.ddy, p.gBz_cf = &gB.ddz);

      double DIM( space_x = (nx.val == 1 ? (xmax()-xmin())/double(n) : (xmax()-xmin())/double(n)),
                  space_y = (ny.val == 1 ? (ymax()-ymin())/double(m) : (ymax()-ymin())/double(m)),
                  space_z = (nz.val == 1 ? (zmax()-zmin())/double(l) : (zmax()-zmin())/double(l)) );

      srand(246246);

      for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
          ONLY3D( for (int k = 0; k < l; ++k))
          {
            EXECD( p.xyz[0] = .5*(xmin()+xmax()) + double(i-n/2)*space_x + double(1-n%2)*.5*space_x,
                   p.xyz[1] = .5*(ymin()+ymax()) + double(j-m/2)*space_y + double(1-m%2)*.5*space_y,
                   p.xyz[2] = .5*(zmin()+zmax()) + double(k-l/2)*space_z + double(1-l%2)*.5*space_z );
            p.rot = initial_rotation.val + initial_rotation_rand.val*double(rand())/double(RAND_MAX);
            particles.push_back(p);
          }
    }
    break;

    default: throw;
  }
}

class ptcl_velo_cf_t : public CF_DIM
{
  cf_value_type_t what;
public:
  ptcl_velo_cf_t(cf_value_type_t what) : what(what) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (ptcl_velo())
    {
      case -1: return 0;
      case  0:
        switch (what) {
          case V_X: return 1;
          case V_Y: return 0;
#ifdef P4_TO_P8
          case V_Z: return 0;
#endif
          default: throw;
        }
      case  1:
        switch (what) {
          case V_X: return 0;
          case V_Y: return 1;
#ifdef P4_TO_P8
          case V_Z: return 0;
#endif
          default: throw;
        }
      case  2:
        switch (what) {
          case V_X: return 1;
          case V_Y: return 1;
#ifdef P4_TO_P8
          case V_Z: return 0;
#endif
          default: throw;
        }
      case  3:
        switch (what) {
          case V_X: return -sqrt(SQR(x)+SQR(y))*sin(atan2(y,x));
          case V_Y: return  sqrt(SQR(x)+SQR(y))*cos(atan2(y,x));
#ifdef P4_TO_P8
          case V_Z: return 0;
#endif
          default: throw;
        }
      default:
        throw;
    }
  }
} DIM(ptcl_vx_cf(V_X), ptcl_vy_cf(V_Y), ptcl_vz_cf(V_Z));

class ptcl_wz_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (ptcl_rotn())
    {
      case 0: return  0;
      case 1: return  1;
      case 2: return -1;
      case 3: return sqrt(SQR(x)+SQR(y))*cos(atan2(y,x));
      default:
        throw;
    }
  }
} ptcl_wz_cf;

class free_velo_cf_t : public CF_DIM
{
  cf_value_type_t what;
public:
  free_velo_cf_t(cf_value_type_t what) : what(what) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (free_velo())
    {
      case 0: return 0;
      case 1:
        switch (what) {
          case V_X: return  x*cos(5.*t);
          case V_Y: return -y*cos(5.*t);
#ifdef P4_TO_P8
          case V_Z: return 0;
#endif
          default: throw;
        }
      default:
        throw;
    }
  }
} DIM(free_vx_cf(V_X), free_vy_cf(V_Y), free_vz_cf(V_Z));

class gamma_Ap_cf_t : public CF_DIM
{
  std::vector<particle_t> *particles;
  CF_DIM *number;
public:
  gamma_Ap_cf_t(std::vector<particle_t> &particles, CF_DIM &number)
    : particles(&particles), number(&number) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return particles->at(int((*number)(DIM(x, y, z)))).gA(DIM(x, y, z));
  }
};

class gamma_Bp_cf_t : public CF_DIM
{
  std::vector<particle_t> *particles;
  CF_DIM *number;
public:
  gamma_Bp_cf_t(std::vector<particle_t> &particles, CF_DIM &number)
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
  n_example.set_from_cmd(cmd);
  set_parameters();
  pl.set_from_cmd_all(cmd);

  set_wall_surface_energies();

  double scaling = 1.;
  pairwise_potential_width.val *= (xmax()-xmin())/pow(2.,lmax());
  EXECD(xmin.val *= box_size(), ymin.val *= box_size(), zmin.val *= box_size());
  EXECD(xmax.val *= box_size(), ymax.val *= box_size(), zmax.val *= box_size());

  lmin.val = lmin.val +round(log(box_size())/log(2.));
  lmax.val = lmax.val +round(log(box_size())/log(2.));

//  std::cout<< .5*lam_bulk_period() << " "; throw;

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
                                               "energy_change_effective "
                                               "ptcl_x "
                                               "ptcl_y "
                                               "ptcl_t\n"); CHKERRXX(ierr);
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

  gamma_Ap_cf_t gamma_Ap_cf(particles, particles_number_cf);
  gamma_Bp_cf_t gamma_Bp_cf(particles, particles_number_cf);

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

  phi_effe_cf_t phi_effe_cf(phi_ptcl_cf, phi_wall_cf, phi_free_cf);
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

  Vec     shape_grad;
  double *shape_grad_ptr;

  ierr = VecCreateGhostNodes(p4est, nodes, &mu_m); CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &mu_p);                CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &phi_free);            CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &phi_wall);            CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &phi_ptcl);            CHKERRXX(ierr);
  ierr = VecDuplicate(mu_m, &shape_grad);          CHKERRXX(ierr);

  // ------------------------------------------------------------------------------------------------------------------
  // initialize fields
  // ------------------------------------------------------------------------------------------------------------------
  sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
  ierr = VecSetGhost(mu_p, 0); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_free_cf, phi_free);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);
  sample_cf_on_nodes(p4est, nodes, phi_ptcl_cf, phi_ptcl);

  ierr = VecSetGhost(shape_grad, 0); CHKERRXX(ierr);

  my_p4est_level_set_t ls(ngbd);
  if (geometry_free() != 0) ls.reinitialize_1st_order(phi_free, 20);
  if (geometry_wall() != 0) ls.reinitialize_1st_order(phi_wall, 20);

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

  vector<double> cfl_v(np, cfl_v_max());
  vector<double> cfl_w(np, cfl_w_max());

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

  // ------------------------------------------------------------------------------------------------------------------
  // main loop
  // ------------------------------------------------------------------------------------------------------------------
  double energy_old = 0;
  double rho_old    = 1;
  vector< vector<double> > v_old(P4EST_DIM, vector<double> (np,0));
  vector<double> w_old(np, 0);
  vector< vector<double> > gv_old(P4EST_DIM, vector<double> (np,0));
  vector<double> gw_old(np, 0);

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

      interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_m,       NULL, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_p,       mu_m, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_wall,   mu_m, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_free,   mu_m, interpolation_between_grids);
      interpolate_between_grids(interp, p4est_np1, nodes_np1, shape_grad, mu_m, interpolation_between_grids);

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
      if (geometry_free() != 0) ls.reinitialize_1st_order_time_2nd_order_space(phi_free, 20);
      if (geometry_wall() != 0) ls.reinitialize_1st_order_time_2nd_order_space(phi_wall, 20);
    }

    // ------------------------------------------------------------------------------------------------------------------
    // allocate fields
    // ------------------------------------------------------------------------------------------------------------------
    Vec     normal_free    [P4EST_DIM];
    double *normal_free_ptr[P4EST_DIM];

    Vec     normal_wall    [P4EST_DIM];
    double *normal_wall_ptr[P4EST_DIM];

    Vec     mu_m_grad    [P4EST_DIM];
//    double *mu_m_grad_ptr[P4EST_DIM];

    Vec     kappa;
    double *kappa_ptr;

    Vec     shape_grad_free;
    double *shape_grad_free_ptr;

    Vec     shape_grad_ptcl;
    double *shape_grad_ptcl_ptr;

    Vec     shape_grad_free_full;
//    double *shape_grad_free_full_ptr;

    Vec     tmp;
//    double *tmp_ptr;

    Vec     surf_tns;
    double *surf_tns_ptr;

    Vec     surf_tns_grad    [P4EST_DIM];
//    double *surf_tns_grad_ptr[P4EST_DIM];

    Vec     cos_angle;
    double *cos_angle_ptr;

    Vec     velo_free;
    double *velo_free_ptr;

    Vec     velo_free_full;
    double *velo_free_full_ptr;

    Vec     particles_number;
    double *particles_number_ptr;

    Vec     integrand;
//    double *integrand_ptr;

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
    sample_cf_on_nodes(p4est, nodes, phi_ptcl_cf, phi_ptcl);

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
    double rho_avg = rho_old;
    double energy  = 0;

    bool adaptive = false;

//    if (iteration == 0 || iteration > num_pre_iterations.val)
    if (iteration % (num_pre_iterations.val+1) == 0)
    {
      if (use_scft())
      {
        my_p4est_scft_t scft(ngbd, ns());

        // set geometry
        scft.add_boundary(phi_free, MLS_INTERSECTION, gamma_Aa_cf, gamma_Ba_cf);
        scft.add_boundary(phi_wall, MLS_INTERSECTION, gamma_Aw_cf, gamma_Bw_cf);
        if (iteration != 0 || num_pre_iterations.val == 0) scft.add_boundary(phi_ptcl, MLS_INTERSECTION, gamma_Ap_cf, gamma_Bp_cf, grafted());

        scft.set_scaling(scaling);
        scft.set_polymer(f(), XN(), grafted());
        scft.set_rho_avg(rho_avg);

        // initialize potentials
        Vec mu_m_tmp = scft.get_mu_m();
        Vec mu_p_tmp = scft.get_mu_p();

        Vec rho_a_tmp = scft.get_rho_a();
        Vec rho_b_tmp = scft.get_rho_b();

        ierr = VecCopyGhost(mu_m, mu_m_tmp); CHKERRXX(ierr);
        //      ierr = VecCopyGhost(mu_p, mu_p_tmp); CHKERRXX(ierr);
        ierr = VecSetGhost(mu_p_tmp, 0); CHKERRXX(ierr);

        ierr = VecCopyGhost(mu_m, rho_a_tmp); CHKERRXX(ierr);
        ierr = VecCopyGhost(mu_m, rho_b_tmp); CHKERRXX(ierr);

        ierr = VecScaleGhost(rho_a_tmp,  1./XN()); CHKERRXX(ierr);
        ierr = VecScaleGhost(rho_b_tmp, -1./XN()); CHKERRXX(ierr);

        ierr = VecShiftGhost(rho_a_tmp, .5*rho_avg); CHKERRXX(ierr);
        ierr = VecShiftGhost(rho_b_tmp, .5*rho_avg); CHKERRXX(ierr);

        // initialize diffusion solvers for propagators
        scft.initialize_solvers();
        //      scft.initialize_bc_smart(iteration != 0);
        if (!smart_bc.val) {
          scft.initialize_bc_simple();
//          scft.initialize_bc_smart(false, 0);
        }

        // main loop for solving SCFT equations
        int    scft_iteration = 0;
        double scft_error     = 2.*scft_tol()+1.;
        while ((scft_iteration < max_scft_iterations() && scft_error > scft_tol())) {
          //      while ((scft_iteration < max_scft_iterations() && scft_error > scft_tol()) || (iteration == 0 && scft_iteration < 200)) {
          for (int i = 0; i < num_scft_subiterations(); i++) {
//            scft.save_VTK(scft_iteration*num_scft_subiterations()+i);
            //          scft.initialize_bc_smart(adaptive); adaptive = true;
            if (smart_bc.val) {
              scft.initialize_bc_smart(true);
            }

            scft.solve_for_propogators();
            scft.calculate_densities();
            ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
                               scft_iteration,
                               scft.get_energy(),
                               scft.get_pressure_force(),
                               scft.get_exchange_force()); CHKERRXX(ierr);
          }

          if (scft_iteration == 100) {
            ierr = VecSetGhost(mu_p_tmp, 0); CHKERRXX(ierr);
          }

          // do an SCFT step
//          scft.update_potentials(scft_iteration % 2 == 0, scft_iteration % 2 == 1);
          scft.update_potentials(1, 1);
          if (scft_iteration % 1 == 0) {
            for (int kk = 0; kk < scft.get_ns(); ++kk)
            {
              scft.save_VTK_q(kk);
            }
            scft.sync_and_extend();
            scft.save_VTK(scft_iteration);
//            return 0;
          }

          ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n",
                             scft_iteration,
                             scft.get_energy(),
                             scft.get_pressure_force(),
                             scft.get_exchange_force()); CHKERRXX(ierr);

          scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
          scft_iteration++;
        }

        energy  = scft.get_energy();
        rho_avg = scft.get_rho_avg();
        rho_old = rho_avg;

        scft.sync_and_extend();
        scft.compute_energy_shape_derivative(0, shape_grad);

        ierr = VecCopyGhost(mu_m_tmp, mu_m); CHKERRXX(ierr);
        ierr = VecCopyGhost(mu_p_tmp, mu_p); CHKERRXX(ierr);
      }
      else
      {
        sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
        ierr = VecSetGhost(mu_p, 0); CHKERRXX(ierr);
        ierr = VecSetGhost(shape_grad, 0);      CHKERRXX(ierr);
      }
    }

    if (geometry_free() != 0) {
      ls.extend_from_interface_to_whole_domain_TVD(phi_free, shape_grad, shape_grad_free, 50, phi_wall);
    } else {
      ierr = VecSetGhost(shape_grad_free, 0); CHKERRXX(ierr);
    }
    ierr = VecScaleGhost(shape_grad_free, -1); CHKERRXX(ierr);

    if (geometry_ptcl() != 0) {
      ls.extend_from_interface_to_whole_domain_TVD(phi_ptcl, shape_grad, shape_grad_ptcl, 50, phi_wall);
    } else {
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

//      ls.extend_Over_Interface_TVD_Full(phi_wall, cos_angle, 20, 0, 0, DBL_MAX, DBL_MAX, DBL_MAX, normal_wall);
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
      if (minimize()) {
        VecCopyGhost(shape_grad_free,      velo_free);
        VecCopyGhost(shape_grad_free_full, velo_free_full);
      } else {
        ierr = VecGetArray(velo_free, &velo_free_ptr); CHKERRXX(ierr);
        foreach_dimension(dim) {
          ierr = VecGetArray(normal_free[dim], &normal_free_ptr[dim]); CHKERRXX(ierr);
        }

        foreach_node(n, nodes) {
          node_xyz_fr_n(n, p4est, nodes, xyz);
          velo_free_ptr[n] = SUMD(normal_free_ptr[0][n]*free_vx_cf.value(xyz),
              normal_free_ptr[1][n]*free_vy_cf.value(xyz),
              normal_free_ptr[2][n]*free_vz_cf.value(xyz));
        }

        ierr = VecRestoreArray(velo_free, &velo_free_ptr); CHKERRXX(ierr);
        foreach_dimension(dim) {
          ierr = VecRestoreArray(normal_free[dim], &normal_free_ptr[dim]); CHKERRXX(ierr);
        }
        ierr = VecCopyGhost(velo_free, velo_free_full); CHKERRXX(ierr);
      }

      // ------------------------------------------------------------------------------------------------------------------
      // make sure velocity is mass conserving
      // ------------------------------------------------------------------------------------------------------------------
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
        // CHECK FOR PROXIMITY?

        ngbd->get_neighbors(n, qnnn);

        double xyzn[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyzn);

        double DIMPM( s_p00 = fabs(qnnn.d_p00), s_m00 = fabs(qnnn.d_m00),
                      s_0p0 = fabs(qnnn.d_0p0), s_0m0 = fabs(qnnn.d_0m0),
                      s_00p = fabs(qnnn.d_00p), s_00m = fabs(qnnn.d_00m) );

        double s_min = MIN(DIM(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0), MIN(s_00p, s_00m)));

        dt_free = MIN(dt_free, cfl()*fabs(s_min/velo_free_ptr[n]));
//        dt_free = MIN(dt_free, cfl()*fabs(s_min/velo_free_full_ptr[n]));
        vmax = MAX(vmax, fabs(velo_free_ptr[n]));
//        vmax = MAX(vmax, fabs(velo_free_full_ptr[n]));
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
    // save data
    // ------------------------------------------------------------------------------------------------------------------
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

        double vec_d[P4EST_DIM] = { DIM(wall_x() - com[0], wall_y() - com[1], wall_z() - com[2]) };
        double nw_norm = ABSD(wall_nx(), wall_ny(), wall_nz());
        double d_norm2 = SQRSUMD(vec_d[0], vec_d[1], vec_d[2]);
        double dn0 = SUMD(vec_d[0]*wall_nx(), vec_d[1]*wall_ny(), vec_d[2]*wall_nz())/nw_norm;

        double del = (d_norm2*wall_eps() + 2.*dn0)
            / ( sqrt(1. + (d_norm2*wall_eps() + 2.*dn0)*wall_eps()) + 1. );

        double vec_n[P4EST_DIM];

        EXECD( vec_n[0] = (wall_nx()/nw_norm + vec_d[0]*wall_eps())/(1.+wall_eps()*del),
               vec_n[1] = (wall_ny()/nw_norm + vec_d[1]*wall_eps())/(1.+wall_eps()*del),
               vec_n[2] = (wall_nz()/nw_norm + vec_d[2]*wall_eps())/(1.+wall_eps()*del) );
        double norm = ABSD(vec_n[0], vec_n[1], vec_n[2]);

        double xyz_c[P4EST_DIM];
        foreach_dimension(dim) {
            xyz_c[dim] = com[dim] + (del-elev)*vec_n[dim]/norm;
        }

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

          double ga = particles[i].gA(DIM(xyz[0], xyz[1], xyz[2]));
          double gb = particles[i].gB(DIM(xyz[0], xyz[1], xyz[2]));

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
    // compute derivatives with respect to particle positions
    // ------------------------------------------------------------------------------------------------------------------
    if (geometry_ptcl() !=0)
    {
      vector< vector<double> > gv(P4EST_DIM, vector<double> (np,0));
      vector<double> gw(np, 0);

      vector< vector<double> > gv_pp(P4EST_DIM, vector<double> (np,0));
      vector<double> gw_pp(np, 0);

      vector< vector<double> > v(P4EST_DIM, vector<double> (np,0));
      vector<double> w(np, 0);

      vector<double> dt_ptcl_v(np, 1);
      vector<double> dt_ptcl_w(np, 1);

      ngbd->first_derivatives_central(mu_m, mu_m_grad);

      sample_cf_on_nodes(p4est, nodes, particles_number_cf, particles_number);

      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

      double factor  = 1./pow(scaling, P4EST_DIM-1.);

      my_p4est_interpolation_nodes_local_t interp_local(ngbd);
      double proximity_crit = 5.*diag;
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

          for (size_t j = 0; j < fv.interfaces.size(); ++j)
          {
            double xyz[P4EST_DIM];
            node_xyz_fr_n(n, p4est, nodes, xyz);

            switch (fv.interfaces[j].id)
            {
              case 0:
                foreach_dimension(dim) {
                  xyz_ptcl[dim] = xyz[dim] + fv.interfaces[j].centroid[dim];
                }
                area_ptcl = fv.interfaces[j].area;
              break;
              case 1:
                foreach_dimension(dim) {
                  xyz_wall[dim] = xyz[dim] + fv.interfaces[j].centroid[dim];
                }
                area_wall = fv.interfaces[j].area;
              break;
              case 2:
                foreach_dimension(dim) {
                  xyz_free[dim] = xyz[dim] + fv.interfaces[j].centroid[dim];
                }
                area_free = fv.interfaces[j].area;
              break;
              default: throw;
            }
          }

          if (area_ptcl != 0)
          {
            double DIM( phix = particles[i].phix( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ),
                        phiy = particles[i].phiy( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ),
                        phiz = particles[i].phiz( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );

            double norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            double dist = particles[i].phi (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            XCODE( xyz_ptcl[0] -= dist*phix/norm );
            YCODE( xyz_ptcl[1] -= dist*phiy/norm );
            ZCODE( xyz_ptcl[2] -= dist*phiz/norm );

            XCODE( phix = particles[i].phix( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );
            YCODE( phiy = particles[i].phiy( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );
            ZCODE( phiz = particles[i].phiz( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );

            norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            interp_local.set_input(shape_grad_ptcl, linear); double velo_val = interp_local.value(xyz_ptcl);
            interp_local.set_input(mu_m,            linear); double mu_m     = interp_local.value(xyz_ptcl);
            interp_local.set_input(mu_m_grad[0],    linear); double mux_m    = interp_local.value(xyz_ptcl);
            interp_local.set_input(mu_m_grad[1],    linear); double muy_m    = interp_local.value(xyz_ptcl);
#ifdef P4_TO_P8
            interp_local.set_input(mu_m_grad[2],    linear); double muz_m    = interp_local.value(xyz_ptcl);
#endif

            double phic = particles[i].kappa(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

            double ga  = particles[i].gA (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double DIM( gax = particles[i].gAx(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                        gay = particles[i].gAy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                        gaz = particles[i].gAz(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])) );

            double gb  = particles[i].gB (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
            double DIM( gbx = particles[i].gBx(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                        gby = particles[i].gBy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                        gbz = particles[i].gBz(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])) );


            double g_eff  = (.5*(ga +gb )*rho_avg + mu_m/XN()*(ga -gb ));
            double DIM( gx_eff = (.5*(gax+gbx)*rho_avg + mu_m/XN()*(gax-gbx) + mux_m/XN()*(ga-gb)),
                        gy_eff = (.5*(gay+gby)*rho_avg + mu_m/XN()*(gay-gby) + muy_m/XN()*(ga-gb)),
                        gz_eff = (.5*(gaz+gbz)*rho_avg + mu_m/XN()*(gaz-gbz) + muz_m/XN()*(ga-gb)) );

            double G = SUMD(phix*gx_eff,
                            phiy*gy_eff,
                            phiz*gz_eff)*factor
                       + g_eff*phic*factor
                       + velo_val;

            double DIM( Gx = G*phix - (.5*(gax+gbx)*rho_avg + mu_m/XN()*(gax-gbx))*factor,
                        Gy = G*phiy - (.5*(gay+gby)*rho_avg + mu_m/XN()*(gay-gby))*factor,
                        Gz = G*phiz - (.5*(gaz+gbz)*rho_avg + mu_m/XN()*(gaz-gbz))*factor );

            double DIM( delx = xyz_ptcl[0]-particles[i].xyz[0],
                        dely = xyz_ptcl[1]-particles[i].xyz[1],
                        delz = xyz_ptcl[2]-particles[i].xyz[2] );

            // take into account periodicity
            if (px()) {
              if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
              if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
            }

            if (py()) {
              if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
              if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
            }
#ifdef P4_TO_P8
            if (pz()) {
              if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
              if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
            }
#endif

            double Gxy = Gx*dely - Gy*delx;

            EXECD( gv[0][i] += Gx*area_ptcl,
                   gv[1][i] += Gy*area_ptcl,
                   gv[2][i] += Gz*area_ptcl );
            gw[i]    += Gxy*area_ptcl;
            surf_energy  += g_eff*area_ptcl*factor;

            // take into account artificial repulsion between particles, walls and free surface
            double G_pp = 0;

            // repulsion between particles
            for (int j = 0; j < np; ++j)
            {
              if (j != i)
              {
                double dist_other = particles[j].phi(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

                if (-dist_other < pairwise_potential_width())
                {
                  double DIM( phix_other = particles[j].phix(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                              phiy_other = particles[j].phiy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                              phiz_other = particles[j].phiz(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])) );
                  dist_other = -dist_other-pairwise_potential_width();

                  G_pp += phic*pairwise_potential(dist_other)*factor
                          + pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, phiz_other*phiz)*factor;

                  double DIM( add_x = -pairwise_force(dist_other)*phix_other*factor,
                              add_y = -pairwise_force(dist_other)*phiy_other*factor,
                              add_z = -pairwise_force(dist_other)*phiz_other*factor );

                  double DIM( delx = xyz_ptcl[0]-particles[j].xyz[0],
                              dely = xyz_ptcl[1]-particles[j].xyz[1],
                              delz = xyz_ptcl[2]-particles[j].xyz[2] );

                  // take into account periodicity
                  if (px()) {
                    if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                    if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                  }

                  if (py()) {
                    if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                    if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                  }
#ifdef P4_TO_P8
                  if (pz()) {
                    if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                    if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                  }
#endif

                  EXECD( gv_pp[0][j] += add_x*area_ptcl,
                         gv_pp[1][j] += add_y*area_ptcl,
                         gv_pp[2][j] += add_z*area_ptcl );
                  gw_pp[j] += (add_x*dely - add_y*delx)*area_ptcl;
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
#ifdef P4_TO_P8
              interp_local.set_input(normal_wall[2], linear); double phiz_other = interp_local.value(xyz_ptcl);
#endif
              dist_other = -dist_other-pairwise_potential_width();
              G_pp += phic*pairwise_potential(dist_other)*factor +
                      pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, phiz_other*phiz)*factor;
            }

            XCODE( double G_ppx = G_pp*phix );
            YCODE( double G_ppy = G_pp*phiy );
            ZCODE( double G_ppz = G_pp*phiz );

            double G_ppxy = G_ppx*dely - G_ppy*delx;

            XCODE( gv_pp[0][i] += G_ppx*area_ptcl );
            YCODE( gv_pp[1][i] += G_ppy*area_ptcl );
            ZCODE( gv_pp[2][i] += G_ppz*area_ptcl );
            gw_pp[i]    += G_ppxy*area_ptcl;
          }

          // more repulsion from walls
          if (area_wall != 0)
          {
            interp_local.set_input(normal_wall[0], linear); double phix = interp_local.value(xyz_wall);
            interp_local.set_input(normal_wall[1], linear); double phiy = interp_local.value(xyz_wall);
#ifdef P4_TO_P8
            interp_local.set_input(normal_wall[2], linear); double phiz = interp_local.value(xyz_wall);
#endif
            double norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            interp_local.set_input(phi_wall, linear); double dist = interp_local.value(xyz_wall);

            XCODE( xyz_wall[0] -= dist*phix/norm );
            YCODE( xyz_wall[1] -= dist*phiy/norm );
            ZCODE( xyz_wall[2] -= dist*phiz/norm );

            for (int j = 0; j < np; ++j)
            {
              double dist_other = particles[j].phi(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2]));
              if (-dist_other < pairwise_potential_width())
              {
                XCODE( double phix_other = particles[j].phix(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2])) );
                YCODE( double phiy_other = particles[j].phiy(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2])) );
                ZCODE( double phiz_other = particles[j].phiz(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2])) );
                dist_other = -dist_other-pairwise_potential_width();

                XCODE( double add_x = -pairwise_force(dist_other)*phix_other*factor );
                YCODE( double add_y = -pairwise_force(dist_other)*phiy_other*factor );
                ZCODE( double add_z = -pairwise_force(dist_other)*phiz_other*factor );

                XCODE( double delx = xyz_wall[0]-particles[j].xyz[0] );
                YCODE( double dely = xyz_wall[1]-particles[j].xyz[1] );
                ZCODE( double delz = xyz_wall[2]-particles[j].xyz[2] );

                // take into account periodicity
                if (px()) {
                  if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                  if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                }

                if (py()) {
                  if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                  if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                }
#ifdef P4_TO_P8
                if (pz()) {
                  if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                  if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                }
#endif

                XCODE( gv_pp[0][j] += add_x*area_wall );
                YCODE( gv_pp[1][j] += add_y*area_wall );
                ZCODE( gv_pp[2][j] += add_z*area_wall );
                gw_pp[j] += (add_x*dely - add_y*delx)*area_wall;
                repulsion_energy  += pairwise_potential(dist_other)*area_wall*factor;
              }
            }
          }

          // more repulsion from free surface
          if (area_free != 0)
          {
            interp_local.set_input(normal_free[0], linear); double phix = interp_local.value(xyz_free);
            interp_local.set_input(normal_free[1], linear); double phiy = interp_local.value(xyz_free);
#ifdef P4_TO_P8
            interp_local.set_input(normal_free[2], linear); double phiz = interp_local.value(xyz_free);
#endif

            double norm = ABSD(phix, phiy, phiz);
            EXECD(phix /= norm, phiy /= norm, phiz /= norm);

            interp_local.set_input(phi_free, linear); double dist = interp_local.value(xyz_free);

            XCODE( xyz_free[0] -= dist*phix/norm );
            YCODE( xyz_free[1] -= dist*phiy/norm );
            ZCODE( xyz_free[2] -= dist*phiz/norm );

            // repulsion from free interface
            for (int j = 0; j < np; ++j)
            {
              double dist_other = particles[j].phi(DIM(xyz_free[0], xyz_free[1], xyz_free[2]));
              if (-dist_other < pairwise_potential_width())
              {
                XCODE( double phix_other = particles[j].phix(DIM(xyz_free[0], xyz_free[1], xyz_free[2])) );
                YCODE( double phiy_other = particles[j].phiy(DIM(xyz_free[0], xyz_free[1], xyz_free[2])) );
                ZCODE( double phiz_other = particles[j].phiz(DIM(xyz_free[0], xyz_free[1], xyz_free[2])) );
                dist_other = -dist_other-pairwise_potential_width();

                XCODE( double add_x = -pairwise_force(dist_other)*phix_other*factor );
                YCODE( double add_y = -pairwise_force(dist_other)*phiy_other*factor );
                ZCODE( double add_z = -pairwise_force(dist_other)*phiz_other*factor );

                XCODE( double delx = xyz_free[0]-particles[j].xyz[0] );
                YCODE( double dely = xyz_free[1]-particles[j].xyz[1] );
                ZCODE( double delz = xyz_free[2]-particles[j].xyz[2] );

                // take into account periodicity
                if (px()) {
                  if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                  if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                }

                if (py()) {
                  if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                  if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                }
#ifdef P4_TO_P8
                if (pz()) {
                  if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                  if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                }
#endif

                XCODE( gv_pp[0][j] += add_x*area_free );
                YCODE( gv_pp[1][j] += add_y*area_free );
                ZCODE( gv_pp[2][j] += add_z*area_free );
                gw_pp[j] += (add_x*dely - add_y*delx)*area_free;
                repulsion_energy += pairwise_potential(dist_other)*area_free*factor;
              }
            }
          }
        }
      }
      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

      foreach_dimension(dim) {
        ierr = MPI_Allreduce(MPI_IN_PLACE, gv   [dim].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
        ierr = MPI_Allreduce(MPI_IN_PLACE, gv_pp[dim].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      }
      ierr = MPI_Allreduce(MPI_IN_PLACE, gw.data(),    np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, gw_pp.data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);

      ierr = MPI_Allreduce(MPI_IN_PLACE, &repulsion_energy, 1,  MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, &surf_energy,      1,  MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);

      energy += repulsion_energy;

      if (!use_scft()) energy += surf_energy;

      // ------------------------------------------------------------------------------------------------------------------
      // select particles' velocity
      // ------------------------------------------------------------------------------------------------------------------
      if (minimize()) {
        for(int i = 0; i < np; i++) {
          EXECD( v[0][i] = (restrict_motion_x() ? 0 : -gv[0][i]-gv_pp[0][i]),
                 v[1][i] = (restrict_motion_y() ? 0 : -gv[1][i]-gv_pp[1][i]),
                 v[2][i] = (restrict_motion_z() ? 0 : -gv[2][i]-gv_pp[2][i]) );
          w[i] = (restrict_rotation() ? 0 : -gw[i]   -gw_pp[i]   );
        }
      } else {
        for(int i = 0; i < np; i++) {
          EXECD( v[0][i] = ptcl_vx_cf( DIM( particles[i].xyz[0], particles[i].xyz[1], particles[i].xyz[2] ) ),
                 v[1][i] = ptcl_vy_cf( DIM( particles[i].xyz[0], particles[i].xyz[1], particles[i].xyz[2] ) ),
                 v[2][i] = ptcl_vz_cf( DIM( particles[i].xyz[0], particles[i].xyz[1], particles[i].xyz[2] ) ) );
          w[i] = ptcl_wz_cf( DIM( particles[i].xyz[0], particles[i].xyz[1], particles[i].xyz[2] ) );
        }
      }

      // ------------------------------------------------------------------------------------------------------------------
      // compute time step for particles
      // ------------------------------------------------------------------------------------------------------------------
      vector<double> arm_max(np,0);

      double phi_thresh = 0;

      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);

      foreach_node(n, nodes) {
        if (phi_ptcl_ptr[n] > phi_thresh) {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);

          int       i = int(particles_number_ptr[n]);
          double DIM( delx = xyz[0]-particles[i].xyz[0],
                      dely = xyz[1]-particles[i].xyz[1],
                      delz = xyz[2]-particles[i].xyz[2] );

          if (px()) {
            if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
            if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
          }

          if (py()) {
            if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
            if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
          }
#ifdef P4_TO_P8
          if (pz()) {
            if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
            if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
          }
#endif
          arm_max[i] = MAX(arm_max[i], ABSD( dely, delx, delz ));
        }
      }

      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);

      ierr = MPI_Allreduce(MPI_IN_PLACE, arm_max.data(), np, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); CHKERRXX(ierr);

      double v_max  = 0;
      double wr_max = 0;
      for (int i = 0; i < np; ++i) {
        v_max  = MAX(v_max, ABSD(v[0][i], v[1][i], v[2][i]));
        wr_max = MAX(wr_max, fabs(w[i])*arm_max[i]);
      }

      double delta_tv = diag*cfl()/MAX(v_max, 0.0000001);
      double delta_tw = diag*cfl()/MAX(wr_max,0.0000001);

      for (int j = 0; j < np; ++j) {
        if (minimize()) {
          double cos_angle = SUMD((gv[0][j]+gv_pp[0][j])*gv_old[0][j],
                                  (gv[1][j]+gv_pp[1][j])*gv_old[1][j],
                                  (gv[2][j]+gv_pp[2][j])*gv_old[2][j])
              / MAX(ABSD(gv[0][j]+gv_pp[0][j],
                         gv[1][j]+gv_pp[1][j],
                         gv[2][j]+gv_pp[2][j]), 1.e-10)
              / MAX(ABSD(gv_old[0][j],
                         gv_old[1][j],
                         gv_old[2][j]), 1.e-10);

          if (cos_angle < 0)                  cfl_v[j] = MAX(cfl_v[j]*cfl_change_rate(), cfl_v_min());
          if ((gw[j]+gw_pp[j])*gw_old[j] < 0) cfl_w[j] = MAX(cfl_w[j]*cfl_change_rate(), cfl_w_min());

          dt_ptcl_v[j] = diag*cfl_v[j]/ABSD(v[0][j], v[1][j], v[2][j]);
          dt_ptcl_w[j] = diag*cfl_w[j]/fabs(w[j]*arm_max[j]);

          if (cos_angle > 0.0)                cfl_v[j] = MIN(cfl_v[j]/cfl_change_rate(), cfl_v_max());
          if ((gw[j]+gw_pp[j])*gw_old[j] > 0) cfl_w[j] = MIN(cfl_w[j]/cfl_change_rate(), cfl_w_max());
        } else {
          dt_ptcl_v[j] = delta_tv;
          dt_ptcl_w[j] = delta_tw;
        }
      }
      v_old = v;
      w_old = w;

      for (int j = 0; j < np; ++j) {
        EXECD( gv_old[0][j] = gv[0][j]+gv_pp[0][j],
               gv_old[1][j] = gv[1][j]+gv_pp[1][j],
               gv_old[2][j] = gv[2][j]+gv_pp[2][j] );
        gw_old[j]    = gw[j]+gw_pp[j];
      }
      // ------------------------------------------------------------------------------------------------------------------
      // compute expected change in energy due to particles
      // ------------------------------------------------------------------------------------------------------------------
      for (int i = 0; i < np; i++)
      {
        energy_change_predicted += SUMD(v[0][i]*(gv[0][i]+gv_pp[0][i]),
                                        v[1][i]*(gv[1][i]+gv_pp[1][i]),
                                        v[2][i]*(gv[2][i]+gv_pp[2][i]))*dt_ptcl_v[i];

        energy_change_predicted += w[i]*(gw[i]+gw_pp[i])*dt_ptcl_w[i];
      }

      // ------------------------------------------------------------------------------------------------------------------
      // move particles
      // ------------------------------------------------------------------------------------------------------------------
      vector<double> dt_ptcl_v_tmp(np, 1);
      vector<double> dt_ptcl_w_tmp(np, 1);

      dt_ptcl_v_tmp = dt_ptcl_v;
      dt_ptcl_w_tmp = dt_ptcl_w;

      for (int i = 0; i < num_submotions(); ++i) {
        for (int j = 0; j < np; ++j) {
          XCODE(particles[j].xyz[0] += v[0][j]*dt_ptcl_v_tmp[j]/num_submotions() );
          YCODE(particles[j].xyz[1] += v[1][j]*dt_ptcl_v_tmp[j]/num_submotions() );
          ZCODE(particles[j].xyz[2] += v[2][j]*dt_ptcl_v_tmp[j]/num_submotions() );

          particles[j].rot += w[j]*dt_ptcl_w_tmp[j]/num_submotions();
        }

        if (px()) {
          for (int j = 0; j < np; ++j) {
            if (particles[j].xyz[0] < xmin()) particles[j].xyz[0] += xmax()-xmin();
            if (particles[j].xyz[0] > xmax()) particles[j].xyz[0] -= xmax()-xmin();
          }
        }

        if (py()) {
          for (int j = 0; j < np; ++j) {
            if (particles[j].xyz[1] < ymin()) particles[j].xyz[1] += ymax()-ymin();
            if (particles[j].xyz[1] > ymax()) particles[j].xyz[1] -= ymax()-ymin();
          }
        }
#ifdef P4_TO_P8
        if (pz()) {
          for (int j = 0; j < np; ++j) {
            if (particles[j].xyz[2] < zmin()) particles[j].xyz[2] += zmax()-zmin();
            if (particles[j].xyz[2] > zmax()) particles[j].xyz[2] -= zmax()-zmin();
          }
        }
#endif

        if( minimize() ){

          vector< vector<double> > gv_tmp(P4EST_DIM, vector<double> (np,0));
          vector<double> gw_tmp(np, 0);

          sample_cf_on_nodes(p4est, nodes, particles_number_cf, particles_number);
          sample_cf_on_nodes(p4est, nodes, phi_ptcl_cf, phi_ptcl);
          sample_cf_on_nodes(p4est, nodes, phi_effe_cf, phi_effe);

          ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
          ierr = VecGetArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);
          ierr = VecGetArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

          double factor  = 1./pow(scaling, P4EST_DIM-1.);

          my_p4est_interpolation_nodes_local_t interp_local(ngbd);
          double proximity_crit = 5.*diag;
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

              for (size_t j = 0; j < fv.interfaces.size(); ++j)
              {
                double xyz[P4EST_DIM];
                node_xyz_fr_n(n, p4est, nodes, xyz);

                switch (fv.interfaces[j].id)
                {
                  case 0:
                    foreach_dimension(dim) {
                      xyz_ptcl[dim] = xyz[dim] + fv.interfaces[j].centroid[dim];
                    }
                    area_ptcl = fv.interfaces[j].area;
                  break;
                  case 1:
                    foreach_dimension(dim) {
                      xyz_wall[dim] = xyz[dim] + fv.interfaces[j].centroid[dim];
                    }
                    area_wall = fv.interfaces[j].area;
                  break;
                  case 2:
                    foreach_dimension(dim) {
                      xyz_free[dim] = xyz[dim] + fv.interfaces[j].centroid[dim];
                    }
                    area_free = fv.interfaces[j].area;
                  break;
                  default: throw;
                }
              }

              if (area_ptcl != 0)
              {
                double DIM( phix = particles[i].phix( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ),
                            phiy = particles[i].phiy( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ),
                            phiz = particles[i].phiz( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );

                double norm = ABSD(phix, phiy, phiz);
                EXECD(phix /= norm, phiy /= norm, phiz /= norm);

                double dist = particles[i].phi (DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

                XCODE( xyz_ptcl[0] -= dist*phix/norm );
                YCODE( xyz_ptcl[1] -= dist*phiy/norm );
                ZCODE( xyz_ptcl[2] -= dist*phiz/norm );

                XCODE( phix = particles[i].phix( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );
                YCODE( phiy = particles[i].phiy( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );
                ZCODE( phiz = particles[i].phiz( DIM( xyz_ptcl[0],xyz_ptcl[1],xyz_ptcl[2] ) ) );

                norm = ABSD(phix, phiy, phiz);
                EXECD(phix /= norm, phiy /= norm, phiz /= norm);

                double phic = particles[i].kappa(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

                double DIM( delx = xyz_ptcl[0]-particles[i].xyz[0],
                            dely = xyz_ptcl[1]-particles[i].xyz[1],
                            delz = xyz_ptcl[2]-particles[i].xyz[2] );

                // take into account periodicity
                if (px()) {
                  if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                  if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                }

                if (py()) {
                  if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                  if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                }
#ifdef P4_TO_P8
                if (pz()) {
                  if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                  if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                }
#endif

                // take into account artificial repulsion between particles, walls and free surface
                double G_pp = 0;

                // repulsion between particles
                for (int j = 0; j < np; ++j)
                {
                  if (j != i)
                  {
                    double dist_other = particles[j].phi(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));

                    if (-dist_other < pairwise_potential_width())
                    {
                      double DIM( phix_other = particles[j].phix(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                                  phiy_other = particles[j].phiy(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])),
                                  phiz_other = particles[j].phiz(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2])) );
                      dist_other = -dist_other-pairwise_potential_width();
                      G_pp += phic*pairwise_potential(dist_other)*factor + pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, phiz_other*phiz)*factor;

                      double DIM( add_x = -pairwise_force(dist_other)*phix_other*factor,
                                  add_y = -pairwise_force(dist_other)*phiy_other*factor,
                                  add_z = -pairwise_force(dist_other)*phiz_other*factor );

                      double DIM( delx = xyz_ptcl[0]-particles[j].xyz[0],
                                  dely = xyz_ptcl[1]-particles[j].xyz[1],
                                  delz = xyz_ptcl[2]-particles[j].xyz[2] );

                      // take into account periodicity
                      if (px()) {
                        if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                        if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                      }

                      if (py()) {
                        if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                        if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                      }
#ifdef P4_TO_P8
                      if (pz()) {
                        if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                        if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                      }
#endif

                      EXECD( gv_tmp[0][j] += add_x*area_ptcl,
                             gv_tmp[1][j] += add_y*area_ptcl,
                             gv_tmp[2][j] += add_z*area_ptcl );
                      gw_tmp[j] += (add_x*dely - add_y*delx)*area_ptcl;
                    }
                  }
                }

                // wall repulsion
                double dist_other = phi_wall_cf(DIM(xyz_ptcl[0], xyz_ptcl[1], xyz_ptcl[2]));
                if (-dist_other < pairwise_potential_width())
                {
                  interp_local.set_input(normal_wall[0], linear); double phix_other = interp_local.value(xyz_ptcl);
                  interp_local.set_input(normal_wall[1], linear); double phiy_other = interp_local.value(xyz_ptcl);
#ifdef P4_TO_P8
                  interp_local.set_input(normal_wall[2], linear); double phiz_other = interp_local.value(xyz_ptcl);
#endif
                  dist_other = -dist_other-pairwise_potential_width();
                  G_pp += phic*pairwise_potential(dist_other)*factor +
                          pairwise_force(dist_other)*SUMD(phix_other*phix, phiy_other*phiy, phiz_other*phiz)*factor;
                }

                XCODE( double G_ppx = G_pp*phix );
                YCODE( double G_ppy = G_pp*phiy );
                ZCODE( double G_ppz = G_pp*phiz );

                double G_ppxy = G_ppx*dely - G_ppy*delx;

                XCODE( gv_tmp[0][i] += G_ppx*area_ptcl );
                YCODE( gv_tmp[1][i] += G_ppy*area_ptcl );
                ZCODE( gv_tmp[2][i] += G_ppz*area_ptcl );
                gw_tmp[i] += G_ppxy*area_ptcl;
              }

              // more repulsion from walls
              if (area_wall != 0)
              {
                interp_local.set_input(normal_wall[0], linear); double phix = interp_local.value(xyz_wall);
                interp_local.set_input(normal_wall[1], linear); double phiy = interp_local.value(xyz_wall);
#ifdef P4_TO_P8
                interp_local.set_input(normal_wall[2], linear); double phiz = interp_local.value(xyz_wall);
#endif
                double norm = ABSD(phix, phiy, phiz);
                EXECD(phix /= norm, phiy /= norm, phiz /= norm);

                interp_local.set_input(phi_wall, linear); double dist = interp_local.value(xyz_wall);

                XCODE( xyz_wall[0] -= dist*phix/norm );
                YCODE( xyz_wall[1] -= dist*phiy/norm );
                ZCODE( xyz_wall[2] -= dist*phiz/norm );

                for (int j = 0; j < np; ++j)
                {
                  double dist_other = particles[j].phi(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2]));
                  if (-dist_other < pairwise_potential_width())
                  {
                    XCODE( double phix_other = particles[j].phix(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2])) );
                    YCODE( double phiy_other = particles[j].phiy(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2])) );
                    ZCODE( double phiz_other = particles[j].phiz(DIM(xyz_wall[0], xyz_wall[1], xyz_wall[2])) );
                    dist_other = -dist_other-pairwise_potential_width();

                    XCODE( double add_x = -pairwise_force(dist_other)*phix_other*factor );
                    YCODE( double add_y = -pairwise_force(dist_other)*phiy_other*factor );
                    ZCODE( double add_z = -pairwise_force(dist_other)*phiz_other*factor );

                    XCODE( double delx = xyz_wall[0]-particles[j].xyz[0] );
                    YCODE( double dely = xyz_wall[1]-particles[j].xyz[1] );
                    ZCODE( double delz = xyz_wall[2]-particles[j].xyz[2] );

                    // take into account periodicity
                    if (px()) {
                      if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                      if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                    }

                    if (py()) {
                      if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                      if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                    }
#ifdef P4_TO_P8
                    if (pz()) {
                      if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                      if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                    }
#endif

                    XCODE( gv_tmp[0][j] += add_x*area_wall );
                    YCODE( gv_tmp[1][j] += add_y*area_wall );
                    ZCODE( gv_tmp[2][j] += add_z*area_wall );
                    gw_tmp[j]    += (add_x*dely - add_y*delx)*area_wall;
                  }
                }
              }

              // more repulsion from free surface
              if (area_free != 0)
              {
                interp_local.set_input(normal_free[0], linear); double phix = interp_local.value(xyz_free);
                interp_local.set_input(normal_free[1], linear); double phiy = interp_local.value(xyz_free);
#ifdef P4_TO_P8
                interp_local.set_input(normal_free[2], linear); double phiz = interp_local.value(xyz_free);
#endif
                double norm = ABSD(phix, phiy, phiz);
                EXECD(phix /= norm, phiy /= norm, phiz /= norm);

                interp_local.set_input(phi_free, linear); double dist = interp_local.value(xyz_free);

                XCODE( xyz_free[0] -= dist*phix/norm );
                YCODE( xyz_free[1] -= dist*phiy/norm );
                ZCODE( xyz_free[2] -= dist*phiz/norm );

                // repulsion from free interface
                for (int j = 0; j < np; ++j)
                {
                  double dist_other = particles[j].phi(DIM(xyz_free[0], xyz_free[1], xyz_free[2]));
                  if (-dist_other < pairwise_potential_width())
                  {
                    XCODE( double phix_other = particles[j].phix(DIM(xyz_free[0], xyz_free[1], xyz_free[2])) );
                    YCODE( double phiy_other = particles[j].phiy(DIM(xyz_free[0], xyz_free[1], xyz_free[2])) );
                    ZCODE( double phiz_other = particles[j].phiz(DIM(xyz_free[0], xyz_free[1], xyz_free[2])) );
                    dist_other = -dist_other-pairwise_potential_width();

                    XCODE( double add_x = -pairwise_force(dist_other)*phix_other*factor );
                    YCODE( double add_y = -pairwise_force(dist_other)*phiy_other*factor );
                    ZCODE( double add_z = -pairwise_force(dist_other)*phiz_other*factor );

                    XCODE( double delx = xyz_free[0]-particles[j].xyz[0] );
                    YCODE( double dely = xyz_free[1]-particles[j].xyz[1] );
                    ZCODE( double delz = xyz_free[2]-particles[j].xyz[2] );

                    // take into account periodicity
                    if (px()) {
                      if (delx > .5*(xmax()-xmin())) delx -= (xmax()-xmin());
                      if (delx <-.5*(xmax()-xmin())) delx += (xmax()-xmin());
                    }

                    if (py()) {
                      if (dely > .5*(ymax()-ymin())) dely -= (ymax()-ymin());
                      if (dely <-.5*(ymax()-ymin())) dely += (ymax()-ymin());
                    }
#ifdef P4_TO_P8
                    if (pz()) {
                      if (delz > .5*(zmax()-zmin())) delz -= (zmax()-zmin());
                      if (delz <-.5*(zmax()-zmin())) delz += (zmax()-zmin());
                    }
#endif

                    XCODE( gv_tmp[0][j] += add_x*area_free );
                    YCODE( gv_tmp[1][j] += add_y*area_free );
                    ZCODE( gv_tmp[2][j] += add_z*area_free );
                    gw_tmp[j]    += (add_x*dely - add_y*delx)*area_free;
                  }
                }
              }
            }
          }
          ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
          ierr = VecRestoreArray(phi_ptcl, &phi_ptcl_ptr); CHKERRXX(ierr);
          ierr = VecRestoreArray(phi_effe, &phi_effe_ptr); CHKERRXX(ierr);

          foreach_dimension(dim) {
            ierr = MPI_Allreduce(MPI_IN_PLACE, gv_tmp[dim].data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);
          }
          ierr = MPI_Allreduce(MPI_IN_PLACE, gw_tmp.data(), np, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); CHKERRXX(ierr);

          for(int i = 0; i < np; i++) {
            EXECD( v[0][i] = (restrict_motion_x() ? 0 : -gv[0][i]-gv_tmp[0][i]),
                   v[1][i] = (restrict_motion_y() ? 0 : -gv[1][i]-gv_tmp[1][i]),
                   v[2][i] = (restrict_motion_z() ? 0 : -gv[2][i]-gv_tmp[2][i]) );
            w[i] = (restrict_rotation() ? 0 : -gw[i]-gw_tmp[i]   );
          }

          for (int j = 0; j < np; ++j) {
            dt_ptcl_v_tmp[j] = MIN(dt_ptcl_v_tmp[j], diag*cfl_v[j]/ABSD(v[0][j], v[1][j], v[2][j]));
            dt_ptcl_w_tmp[j] = MIN(dt_ptcl_w_tmp[j], diag*cfl_w[j]/fabs(w[j]*arm_max[j]));
//            dt_ptcl_v_tmp[j] = diag*cfl_v[j]/ABSD(v[0][j], v[1][j], v[2][j]);
//            dt_ptcl_w_tmp[j] = diag*cfl_w[j]/fabs(w[j]*arm_max[j]);
          }
        }
      }
    }

    // ------------------------------------------------------------------------------------------------------------------
    // save data
    // ------------------------------------------------------------------------------------------------------------------
    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e, Predicted: %e\n", energy, energy-energy_old, energy_change_predicted);

    double ptcl_x_avg = 0;
    double ptcl_y_avg = 0;
    double ptcl_t_avg = 0;
    for (int j = 0; j < np; ++j) {
      ptcl_x_avg += particles[j].xyz[0];
      ptcl_y_avg += particles[j].xyz[1];
      ptcl_t_avg += particles[j].rot;
    }

    if (np > 0) {
      ptcl_x_avg /= np;
      ptcl_y_avg /= np;
      ptcl_t_avg /= np;
    }

    if (save_data())
    {
      ierr = PetscFOpen  (mpi.comm(), file_conv_name, "a", &file_conv); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), file_conv, "%d %e %e %e %e %e %e\n",
                          (int) round(iteration),
                          energy,
                          energy_change_predicted,
                          energy-energy_old,
                          ptcl_x_avg,
                          ptcl_y_avg,
                          ptcl_t_avg); CHKERRXX(ierr);
      ierr = PetscFClose (mpi.comm(), file_conv); CHKERRXX(ierr);
    }

    // ------------------------------------------------------------------------------------------------------------------
    // advect interface and impose contact angle
    // ------------------------------------------------------------------------------------------------------------------
    if (geometry_free() != 0)
    {
      ls.set_use_neumann_for_contact_angle(use_neumann());
      ls.set_contact_angle_extension(contact_angle_extension());

      double correction_total = 0;

//      std::cout << dt_free;
      int splits = 1;
      for (int i = 0; i < splits; ++i)
      {
        if (minimize()) {
          ls.advect_in_normal_direction_with_contact_angle(velo_free, surf_tns, cos_angle, phi_wall, phi_free, dt_free/double(splits));
        } else {
          ls.advect_in_normal_direction(velo_free, phi_free, dt_free/double(splits));
          XCODE( free_vx_cf.t += dt_free/double(splits) );
          YCODE( free_vy_cf.t += dt_free/double(splits) );
          ZCODE( free_vz_cf.t += dt_free/double(splits) );
        }

        // cut off tails
        double *phi_wall_ptr;
        double *phi_free_ptr;

        ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
        ierr = VecGetArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

        foreach_node(n, nodes)
        {
          double transition = smoothstep(1, (phi_wall_ptr[n]/diag - 10.)/20.);
          // phi_free_ptr[n] = smooth_max(phi_free_ptr[n], phi_wall_ptr[n] - 12.*diag, 4.*diag);
          phi_free_ptr[n] = (1.-transition)*phi_free_ptr[n] + transition*MAX(phi_free_ptr[n], phi_wall_ptr[n] - 10.*diag);
        }

        ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
        ierr = VecRestoreArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

        ls.reinitialize_1st_order_time_2nd_order_space(phi_free, 20);

        // correct for volume loss
        for (short i = 0; i < volume_corrections(); ++i)
        {
          double volume_cur   = integration.measure_of_domain();
          double surface_area = integration.measure_of_interface(0);
          double correction   = (volume_cur-volume)/surface_area;

          VecShiftGhost(phi_free, correction);
          correction_total += correction;

//          double volume_cur2 = integration.measure_of_domain();

          // PetscPrintf(mpi.comm(), "Volume loss: %e, after correction: %e\n", (volume_cur-volume)/volume, (volume_cur2-volume)/volume);
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

  ierr = VecDestroy(shape_grad); CHKERRXX(ierr);
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
