
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
#include <src/my_p8est_tools_mls.h>
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
#include <src/my_p4est_tools_mls.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_scft.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

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
M_PARSER_DEFINE2(double, xmin, -1, "xmin")
M_PARSER_DEFINE2(double, ymin, -1, "ymin")
M_PARSER_DEFINE2(double, zmin, -1, "zmin")
M_PARSER_DEFINE2(double, xmax,  1, "xmax")
M_PARSER_DEFINE2(double, ymax,  1, "ymax")
M_PARSER_DEFINE2(double, zmax,  1, "zmax")

M_PARSER_DEFINE2(bool, px, 0, "periodicity in x-dimension 0/1")
M_PARSER_DEFINE2(bool, py, 0, "periodicity in y-dimension 0/1")
M_PARSER_DEFINE2(bool, pz, 0, "periodicity in z-dimension 0/1")

M_PARSER_DEFINE2(int, nx, 1, "number of trees in x-dimension")
M_PARSER_DEFINE2(int, ny, 1, "number of trees in y-dimension")
M_PARSER_DEFINE2(int, nz, 1, "number of trees in z-dimension")

// grid parameters
#ifdef P4_TO_P8
M_PARSER_DEFINE2(int, lmin, 8, "min level of trees")
M_PARSER_DEFINE2(int, lmax, 8, "max level of trees")
#else
M_PARSER_DEFINE2(int, lmin, 9, "min level of trees")
M_PARSER_DEFINE2(int, lmax, 9, "max level of trees")
#endif
M_PARSER_DEFINE2(double, lip, 0.8, "Lipschitz constant")

// advection parameters
M_PARSER_DEFINE2(double, cfl,                     0.15, "CFL number")
M_PARSER_DEFINE2(bool,   use_neumann,             1,    "Impose contact angle use Neumann BC 0/1")
M_PARSER_DEFINE2(bool,   compute_exact,           0,    "Compute exact final shape (only for pure-curvature) 0/1")
M_PARSER_DEFINE2(bool,   reinit_at_start,         1,    "Reinitialze level-set function at the start 0/1)")
M_PARSER_DEFINE2(int,    contact_angle_extension, 2,    "Method for extending level-set function into wall: 0 - constant angle, 1 - , 2 - special")
M_PARSER_DEFINE2(int,    volume_corrections,      2,    "Number of volume correction after each move")
M_PARSER_DEFINE2(int,    max_iterations,          1000, "Maximum number of advection steps")

interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

// scft parameters
M_PARSER_DEFINE2(bool,   use_scft,            0,     "Turn on/off SCFT 0/1")
M_PARSER_DEFINE2(bool,   smooth_pressure,     1,     "Smooth pressure after first BC adjustment 0/1")
M_PARSER_DEFINE2(int,    max_scft_iterations, 500,   "Maximum SCFT iterations")
M_PARSER_DEFINE2(int,    bc_adjust_min,       5,     "Minimun SCFT steps between adjusting BC")
M_PARSER_DEFINE2(double, scft_tol,            1.e-3, "Tolerance for SCFT")
M_PARSER_DEFINE2(double, scft_bc_tol,         1.e-2, "Tolerance for adjusting BC")

// polymer
M_PARSER_DEFINE2(double, box_size, 10, "Box size in units of Rg")
M_PARSER_DEFINE2(double, f,        0.5, "Fraction of polymer A")
M_PARSER_DEFINE2(double, XN,       20,  "Flory-Higgins interaction parameter")
M_PARSER_DEFINE2(int,    ns,       40,  "Discretization of polymer chain")

// output parameters
M_PARSER_DEFINE2(bool, save_vtk,        1, "")
M_PARSER_DEFINE2(bool, save_parameters, 1, "")
M_PARSER_DEFINE2(bool, save_data,       1, "")
M_PARSER_DEFINE2(int,  save_every_dn,   1, "") // for vtk

// problem setting
M_PARSER_DEFINE2(int, num_polymer_geometry, 0, "Initial polymer shape: 0 - drop, 1 - film, 2 - combination")
M_PARSER_DEFINE2(int, num_wall_geometry,    0, "Wall geometry: 0 - no wall, 1 - wall, 2 - well")
M_PARSER_DEFINE2(int, num_wall_pattern,     0, "Wall chemical pattern: 0 - no pattern")
M_PARSER_DEFINE2(int, num_seed,             2, "Seed: 0 - zero, 1 - random, 2 - horizontal stripes, 3 - vertical stripes, 4 - dots")
M_PARSER_DEFINE2(int, num_example,          0, "Number of predefined example")

// surface energies
M_PARSER_DEFINE2(int, wall_energy_type, 1, "Method for setting wall surface energy: 0 - explicitly (i.e. convert XN to angles), 1 - through contact angles (i.e. convert angles to XN)")

M_PARSER_DEFINE2(double, XN_air_avg, 20, "Polymer-air surface energy strength: average")
M_PARSER_DEFINE2(double, XN_air_del, 15, "Polymer-air surface energy strength: difference")

M_PARSER_DEFINE2(double, angle_A_min, 30,  "Minimum contact angle for A-block")
M_PARSER_DEFINE2(double, angle_A_max, 30,  "Maximum contact angle for A-block")
M_PARSER_DEFINE2(double, angle_B_min, 100, "Minimum contact angle for B-block")
M_PARSER_DEFINE2(double, angle_B_max, 100, "Maximum contact angle for B-block")

M_PARSER_DEFINE2(double, XN_wall_A_min, 5, "Minimum Polymer-wall interaction strength for A-block")
M_PARSER_DEFINE2(double, XN_wall_A_max, 5, "Maximum Polymer-wall interaction strength for A-block")
M_PARSER_DEFINE2(double, XN_wall_B_min, 8, "Minimum Polymer-wall interaction strength for B-block")
M_PARSER_DEFINE2(double, XN_wall_B_max, 8, "Maximum Polymer-wall interaction strength for B-block")

// geometry parameters
M_PARSER_DEFINE2(double, drop_r,      0.5, "")
M_PARSER_DEFINE2(double, drop_x,      .0, "")
M_PARSER_DEFINE2(double, drop_y,      .0, "")
M_PARSER_DEFINE2(double, drop_z,      .0, "")
M_PARSER_DEFINE2(double, drop_r0,     0.6, "")
M_PARSER_DEFINE2(double, drop_k,      5, "")
M_PARSER_DEFINE2(double, drop_deform, 0.05, "")

M_PARSER_DEFINE2(double, film_eps, -1.0, "") // curvature
M_PARSER_DEFINE2(double, film_nx,  0, "")
M_PARSER_DEFINE2(double, film_ny,  1, "")
M_PARSER_DEFINE2(double, film_nz,  0, "")
M_PARSER_DEFINE2(double, film_x,   .0, "")
M_PARSER_DEFINE2(double, film_y,   .0, "")
M_PARSER_DEFINE2(double, film_z,   .0, "")

M_PARSER_DEFINE2(double, wall_eps, -.5, "") // curvature
M_PARSER_DEFINE2(double, wall_nx,  -0, "")
M_PARSER_DEFINE2(double, wall_ny,  -1, "")
M_PARSER_DEFINE2(double, wall_nz,  -0, "")
M_PARSER_DEFINE2(double, wall_x,   .0, "")
M_PARSER_DEFINE2(double, wall_y,   -.5+0.01, "")
M_PARSER_DEFINE2(double, wall_z,   .0, "")

M_PARSER_DEFINE2(double, well_x, 0.00, "Well geometry: center")
M_PARSER_DEFINE2(double, well_z, 0.53, "Well geometry: position")
M_PARSER_DEFINE2(double, well_h, 1.00, "Well geometry: depth")
M_PARSER_DEFINE2(double, well_w, 0.77, "Well geometry: width")
M_PARSER_DEFINE2(double, well_r, 0.10, "Well geometry: corner smoothing")

void set_wall_surface_energies()
{
  switch (wall_energy_type) {
    case 0:

      angle_A_min = 180.*acos( SIGN(XN_wall_A_max) * sqrt(XN_wall_A_max / (XN_air_avg+XN_air_del)) )/PI;
      angle_A_max = 180.*acos( SIGN(XN_wall_A_min) * sqrt(XN_wall_A_min / (XN_air_avg+XN_air_del)) )/PI;
      angle_B_min = 180.*acos( SIGN(XN_wall_B_max) * sqrt(XN_wall_B_max / (XN_air_avg-XN_air_del)) )/PI;
      angle_B_max = 180.*acos( SIGN(XN_wall_B_min) * sqrt(XN_wall_B_min / (XN_air_avg-XN_air_del)) )/PI;

      break;
    case 1:

      XN_wall_A_max = (XN_air_avg+XN_air_del)*SIGN(cos(angle_A_min*PI/180.))*SQR(cos(angle_A_min*PI/180.));
      XN_wall_A_min = (XN_air_avg+XN_air_del)*SIGN(cos(angle_A_max*PI/180.))*SQR(cos(angle_A_max*PI/180.));
      XN_wall_B_max = (XN_air_avg-XN_air_del)*SIGN(cos(angle_B_min*PI/180.))*SQR(cos(angle_B_min*PI/180.));
      XN_wall_B_min = (XN_air_avg-XN_air_del)*SIGN(cos(angle_B_max*PI/180.))*SQR(cos(angle_B_max*PI/180.));

      break;
    default:
      throw std::invalid_argument("Invalid method for setting wall surface energies");
  }
}

void set_parameters()
{
  switch (num_example)
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
  double operator()(double x, double y P8C(double z)) const
  {
    return sqrt(XN_air_avg+XN_air_del);
  }
} gamma_Aa_cf;

class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    return sqrt(XN_air_avg-XN_air_del);
  }
} gamma_Ba_cf;

class gamma_Aw_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_wall_pattern)
    {
      case 0: return SIGN(XN_wall_A_min)*sqrt(fabs(XN_wall_A_min));
      default: throw std::invalid_argument("Error: Invalid wall pattern number\n");
    }
  }
} gamma_Aw_cf;

class gamma_Bw_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_wall_pattern)
    {
      case 0: return SIGN(XN_wall_B_min)*sqrt(fabs(XN_wall_B_min));
      default: throw std::invalid_argument("Error: Invalid wall pattern number\n");
    }
  }
} gamma_Bw_cf;

class gamma_aw_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    return 0;
  }
} gamma_aw_cf;


/* geometry of interfaces */
class phi_wall_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_wall_geometry)
    {
      case 0: return -1;
      case 1:
        {
          double norm = sqrt(SQR(wall_nx) + SQR(wall_ny) P8( + SQR(wall_nz) ) );
          return - ( (x-wall_x)*((x-wall_x)*wall_eps - 2.*wall_nx / norm) +
                     (y-wall_y)*((y-wall_y)*wall_eps - 2.*wall_ny / norm)
           #ifdef P4_TO_P8
                     + (z-wall_z)*((z-wall_z)*wall_eps - 2.*wall_nz / norm)
           #endif
                     )
              / ( sqrt( SQR((x-wall_x)*wall_eps - wall_nx / norm) +
                        SQR((y-wall_y)*wall_eps - wall_ny / norm)
              #ifdef P4_TO_P8
                        + SQR((z-wall_z)*wall_eps - wall_nz / norm)
              #endif
                        ) + 1. );
        }
      case 2:
        {
          double phi_top   = well_z - y;
          double phi_bot   = well_z-well_h - y;
          double phi_walls = MAX(x-well_x-.5*well_w, -(x-well_x)-.5*well_w);

          return smooth_max(phi_bot, smooth_min(phi_top, phi_walls, well_r), well_r);
        }
    }
  }
} phi_wall_cf;

class phi_intf_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z)) const
  {
    switch (num_polymer_geometry)
    {
      case 0: return sqrt( SQR(x-drop_x) + SQR(y-drop_y) P8(+ SQR(z-drop_z)) )
            - drop_r
            *(1.+drop_deform*cos(drop_k*atan2(x-drop_x,y-drop_y))
      #ifdef P4_TO_P8
              *(1.-cos(2.*acos((z-drop_z)/sqrt( SQR(x-drop_x) + SQR(y-drop_y) + SQR(z-drop_z) + 1.e-12))))
      #endif
              );
      case 1:
      {
        double norm = sqrt(SQR(film_nx) + SQR(film_ny) P8( + SQR(film_nz) ) );
        return - ( (x-film_x)*((x-film_x)*film_eps - 2.*film_nx / norm) +
                   (y-film_y)*((y-film_y)*film_eps - 2.*film_ny / norm)
               #ifdef P4_TO_P8
                 + (z-film_z)*((z-wall_z)*film_eps - 2.*film_nz / norm)
               #endif
                   )
            / ( sqrt( SQR((x-film_x)*film_eps - film_nx / norm) +
                      SQR((y-film_y)*film_eps - film_ny / norm)
                  #ifdef P4_TO_P8
                    + SQR((z-film_z)*film_eps - film_nz / norm)
                  #endif
                      ) + 1. );
      }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_intf_cf;

inline double lam_bulk_period()
{
  return 2.*pow(8.*XN/3./pow(PI,4.),1./6.)/box_size;
}

class mu_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z) ) const
  {
    switch (num_seed)
    {
      case 0: return 0;
      case 1: return 0.01*XN*(double)(rand()%1000)/1000.;
      case 2: {
          double nx = (px == 1) ? round((xmax-xmin)/lam_bulk_period()) : (xmax-xmin)/lam_bulk_period();
          return .5*XN*cos(2.*PI*x/(xmax-xmin)*nx);
        }
      case 3: {
          double ny = (py == 1) ? round((ymax-ymin)/lam_bulk_period()) : (ymax-ymin)/lam_bulk_period();
          return .5*XN*cos(2.*PI*y/(ymax-ymin)*ny);
        }
      case 4: {
          double nx = (px == 1) ? round(.5*(xmax-xmin)/lam_bulk_period()) : .5*(xmax-xmin)/lam_bulk_period();
          double ny = (py == 1) ? round(.5*(ymax-ymin)/lam_bulk_period()) : .5*(ymax-ymin)/lam_bulk_period();
#ifdef P4_TO_P8
          double nz = (pz == 1) ? round(.5*(zmax-zmin)/lam_bulk_period()) : .5*(zmax-zmin)/lam_bulk_period();
#endif
          return .5*XN*cos(2.*PI*x/(xmax-xmin)*nx)*
                       cos(2.*PI*y/(ymax-ymin)*ny)
                  P8C(*cos(2.*PI*z/(zmax-zmin)*nz));
        }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} mu_cf;

class phi_eff_cf_t : public CF_DIM
{
public:
  double operator()(double x, double y P8C(double z) ) const
  {
    return MIN( MAX(phi_intf_cf(x, y P8C(z)), phi_wall_cf(x, y P8C(z))), fabs(phi_wall_cf(x, y P8C(z))) );
  }
} phi_eff_cf;

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

void write_parameters(MPI_Comm mpicomm, const std::string &output);
void parse_cmd(int argc, char* argv[]);

PetscErrorCode ierr;

int main (int argc, char* argv[])
{
  /* initialize MPI */
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  /* create an output directory */
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save visuals\n");
    return -1;
  }

  std::ostringstream command;
  command << "mkdir -p " << out_dir << "/vtu";
  int ret_sys = system(command.str().c_str());
  if(ret_sys < 0)
    throw std::invalid_argument("Could not create directory");

  char name[10000];

  sprintf(name, "%s/parameters.dat", out_dir);

  parse_cmd(argc, argv);
  write_parameters(mpi.comm(), name);

  double scalling = 1./box_size;

#ifdef P4_TO_P8
  double xyz_min[] = { xmin, ymin, zmin };
  double xyz_max[] = { xmax, ymax, zmax };
  int nb_trees[] = { nx, ny, nz };
  int periodic[] = { px, py, pz };
#else
  double xyz_min[] = { xmin, ymin };
  double xyz_max[] = { xmax, ymax };
  int nb_trees[] = { nx, ny };
  int periodic[] = { px, py };
#endif

  /* create the p4est */
  my_p4est_brick_t brick;

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_eff_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_inside_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  double dxyz[P4EST_DIM], h, diag;
  get_dxyz_min(p4est, dxyz, h, diag);

  /* initialize geometry */
  Vec phi_intf; ierr = VecCreateGhostNodes(p4est, nodes, &phi_intf); CHKERRXX(ierr);
  Vec phi_wall; ierr = VecCreateGhostNodes(p4est, nodes, &phi_wall); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_intf_cf, phi_intf);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);

  /* reinitialize and refine-coarsen agan */
  if (reinit_at_start)
  {
    my_p4est_level_set_t ls(ngbd);

    ls.reinitialize_1st_order_time_2nd_order_space(phi_intf);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_wall);

    bool is_grid_changing = true;
    while (is_grid_changing) {

      std::vector<Vec> phi(2);
      std::vector<action_t> acn(2, INTERSECTION);
      std::vector<int> acn_int(2, 0);
      std::vector<int> clr(2);
      std::vector<bool> refine_always(2);

      phi[0] = phi_intf; clr[0] = 0; refine_always[0] = false;
      phi[1] = phi_wall; clr[1] = 1; refine_always[1] = true;

      Vec phi_eff;
      double *phi_eff_ptr;

      ierr = VecDuplicate(phi_intf, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(nodes, &phi, &acn_int, &refine_always, phi_eff);
      ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin, lmax, lip);
      sp.set_refine_only_inside(0);

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

      if (is_grid_changing)
      {
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

        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_wall, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_intf, NULL, interpolation_between_grids);

        // delete old p4est
        p4est_destroy(p4est);       p4est = p4est_np1;
        p4est_ghost_destroy(ghost); ghost = ghost_np1;
        p4est_nodes_destroy(nodes); nodes = nodes_np1;
        hierarchy->update(p4est, ghost);
        ngbd->update(hierarchy, nodes);
      }
    }

    sample_cf_on_nodes(p4est, nodes, phi_intf_cf, phi_intf);
    sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);

    my_p4est_level_set_t ls_new(ngbd);

    ls_new.reinitialize_1st_order_time_2nd_order_space(phi_intf);
    ls_new.reinitialize_1st_order_time_2nd_order_space(phi_wall);
  }

  /* initialize potentials */
  Vec mu_m; ierr = VecDuplicate(phi_intf, &mu_m); CHKERRXX(ierr);
  Vec mu_p; ierr = VecDuplicate(phi_intf, &mu_p); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
  VecSetGhost(mu_p, 0);

  /* compute initial volume for volume-loss corrections */
  double volume = 0;

  {
    std::vector<Vec> phi(2);
    std::vector<action_t> acn(2, INTERSECTION);
    std::vector<int> clr(2);

    phi[0] = phi_intf; clr[0] = 0;
    phi[1] = phi_wall; clr[1] = 1;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    volume = integration.measure_of_domain();
  }

  // in case of constant contact angle and simple geometry we can compute analytically position and volume of the steady-state shape
  double elev = 0;
  if (compute_exact)
  {
    double g_Aw = gamma_Aw_cf(0,0 P8C(0));
    double g_Bw = gamma_Bw_cf(0,0 P8C(0));
    double g_Aa = gamma_Aa_cf(0,0 P8C(0));
    double g_Ba = gamma_Ba_cf(0,0 P8C(0));
    double g_aw = gamma_aw_cf(0,0 P8C(0));

    double theta = acos( (.5*(g_Aw+g_Bw) - g_aw) / (.5*(g_Aa+g_Ba) ) );

    elev = (wall_eps*drop_r0 - 2.*cos(PI-theta))*drop_r0
        /(sqrt(SQR(wall_eps*drop_r0) + 1. - 2.*wall_eps*drop_r0*cos(PI-theta)) + 1.);

    double alpha = acos((elev + .5*(SQR(drop_r0) + SQR(elev))*wall_eps)/drop_r0/(1.+wall_eps*elev));
#ifdef P4_TO_P8
    volume = 1./3.*PI*pow(drop_r0, 3.)*(2.*(1.-cos(PI-alpha)) + cos(alpha)*SQR(sin(alpha)));
#else
    volume = SQR(drop_r0)*(PI - alpha + cos(alpha)*sin(alpha));
#endif

    if (wall_eps != 0.)
    {
      double beta = acos(1.+.5*SQR(wall_eps)*(SQR(elev) - SQR(drop_r0))/(1.+wall_eps*elev));
#ifdef P4_TO_P8
      volume -= 1./3.*PI*(2.*(1.-cos(beta)) - cos(beta)*SQR(sin(beta)))/SQR(wall_eps)/wall_eps;
#else
      volume -= (beta - cos(beta)*sin(beta))/fabs(wall_eps)/wall_eps;
#endif
    }
  }

  double energy = 0;
  double energy_old = 0;

  /* main loop */
  int iteration = 0;
  while (iteration < max_iterations)
  {
    std::vector<Vec> phi(2);
    std::vector<action_t> acn(2, INTERSECTION);
    std::vector<int> acn_int(2, 0);
    std::vector<int> clr(2);
    std::vector<bool> refine_always(2);

    phi[0] = phi_intf; clr[0] = 0; refine_always[0] = false;
    phi[1] = phi_wall; clr[1] = 1; refine_always[1] = true;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    /* normal and curvature */
    Vec normal[P4EST_DIM];
    Vec kappa;

    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);

    compute_normals_and_mean_curvature(*ngbd, phi_intf, normal, kappa);

    Vec kappa_tmp;
    ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);

    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, kappa, kappa_tmp);

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    kappa = kappa_tmp;

    /* get density field and non-curvature velocity */
    Vec velo;
    ierr = VecDuplicate(phi_intf, &velo); CHKERRXX(ierr);

    if (use_scft)
    {
      my_p4est_scft_t scft(ngbd);

      scft.set_scalling(scalling);
      scft.set_polymer(f, XN, ns);
      scft.set_geometry(phi, acn);

      /* initialize potentials */
      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);

      std::vector<CF_DIM *> gamma_a_cf(2, NULL);
      std::vector<CF_DIM *> gamma_b_cf(2, NULL);

      gamma_a_cf[0] = &gamma_Aa_cf; gamma_a_cf[1] = &gamma_Aw_cf;
      gamma_b_cf[0] = &gamma_Ba_cf; gamma_b_cf[1] = &gamma_Bw_cf;

      scft.set_surface_tensions(gamma_a_cf, gamma_b_cf, gamma_aw_cf);

      scft.initialize_linear_system();

      scft.initialize_bc_smart(iteration != 0);

      int scft_iteration = 0;
      double scft_error = 2.*scft_tol;
      int bc_iters = 0;
      while (scft_iteration < max_scft_iterations && scft_error > scft_tol || scft_iteration < bc_adjust_min+1)
      {
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();

        if (scft.get_exchange_force() < scft_bc_tol && bc_iters >= bc_adjust_min)
        {
          scft.initialize_bc_smart();
          if (smooth_pressure)
          {
            scft.smooth_singularity_in_pressure_field();
            smooth_pressure = false;
          }
          scft.recompute_matrices();
//          scft.initialize_linear_system();
          ierr = PetscPrintf(mpi.comm(), "Robin coefficients have been adjusted\n"); CHKERRXX(ierr);
          bc_iters = 0;
        }

        ierr = PetscPrintf(mpi.comm(), "%d Energy: %e; Pressure: %e; Exchange: %e\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force()); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        scft_iteration++;
        bc_iters++;
      }

      energy = scft.get_energy();

      scft.sync_and_extend();

      scft.compute_energy_shape_derivative(0, velo);

      VecScaleGhost(velo, -1);

      VecCopyGhost(mu_m_tmp, mu_m);
      VecCopyGhost(mu_p_tmp, mu_p);
    }
    else
    {
      sample_cf_on_nodes(p4est, nodes, mu_cf, mu_m);
      VecSetGhost(mu_p, 0);
      VecSetGhost(velo, 0);

      // compute energy
      Vec mu_wall;
      Vec mu_intf;

      ierr = VecDuplicate(phi_intf, &mu_wall); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_intf, &mu_intf); CHKERRXX(ierr);

      ls.extend_from_interface_to_whole_domain_TVD(phi_wall, mu_m, mu_wall, 20);
      ls.extend_from_interface_to_whole_domain_TVD(phi_intf, mu_m, mu_intf, 20);

//      energy  = .5*(gamma_Aa+gamma_Ba)*integration.measure_of_interface(0) + .5*(gamma_Aa-gamma_Ba)*integration.integrate_over_interface(0, mu_intf);
//      energy += .5*(gamma_Aw+gamma_Bw)*integration.measure_of_interface(1) + .5*(gamma_Aw-gamma_Bw)*integration.integrate_over_interface(1, mu_wall);

      ierr = VecDestroy(mu_wall); CHKERRXX(ierr);
      ierr = VecDestroy(mu_intf); CHKERRXX(ierr);
    }

    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e\n", energy, energy-energy_old);
    energy_old = energy;

    /* compute velocity and contact angle */
    Vec surf_tns;
    Vec cos_angle;

    ierr = VecDuplicate(phi_intf, &surf_tns); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_intf, &cos_angle); CHKERRXX(ierr);

    double *mu_ptr;
    double *surf_tns_ptr;
    double *cos_angle_ptr;

    ierr = VecGetArray(mu_m, &mu_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

    double xyz[P4EST_DIM];
    foreach_node(n, nodes)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);
      double g_Aw = gamma_Aw_cf.value(xyz)*scalling;
      double g_Bw = gamma_Bw_cf.value(xyz)*scalling;
      double g_Aa = gamma_Aa_cf.value(xyz)*scalling;
      double g_Ba = gamma_Ba_cf.value(xyz)*scalling;
      double g_aw = gamma_aw_cf.value(xyz)*scalling;

      surf_tns_ptr[n] = (.5*(g_Aa+g_Ba) + (g_Aa-g_Ba)*mu_ptr[n]/XN)/volume;
      cos_angle_ptr[n] = (.5*(g_Aw+g_Bw) + (g_Aw-g_Bw)*mu_ptr[n]/XN - g_aw)
          / (.5*(g_Aa+g_Ba) + (g_Aa-g_Ba)*mu_ptr[n]/XN);
    }

    ierr = VecRestoreArray(mu_m, &mu_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

    // average velocity
    Vec velo_full;
    Vec surf_tns_d[P4EST_DIM];
    Vec tmp;

    ierr = VecDuplicate(phi_intf, &velo_full); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_intf, &tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &surf_tns_d[dim]); CHKERRXX(ierr); }

    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, surf_tns, tmp);
    VecPointwiseMultGhost(velo_full, kappa, tmp);

    ngbd->first_derivatives_central(surf_tns, surf_tns_d);
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(surf_tns_d[dim], surf_tns_d[dim], normal[dim]);
      ls.extend_from_interface_to_whole_domain_TVD(phi_intf, surf_tns_d[dim], tmp);
      VecAXPBYGhost(velo_full, 1, 1, tmp);
    }

    VecScaleGhost(velo_full, -1);
    VecAXPBYGhost(velo_full, 1, 1, velo);

    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_intf, velo_full, phi_intf);

    double vn_avg = integration.integrate_over_interface(0, velo_full)/integration.measure_of_interface(0);

    ierr = VecDestroy(tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDestroy(surf_tns_d[dim]); CHKERRXX(ierr); }

    VecShiftGhost(velo, -vn_avg);
//    VecSetGhost(vn, 0);

    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_intf, velo, phi_intf);

    /* compute time step dt */
    double dt_local = DBL_MAX;
    double dt;

    double *velo_ptr;

    ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);

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

      /* choose CFL = 0.8 ... just for fun! */
      dt_local = MIN(dt_local, cfl*fabs(s_min/velo_ptr[n]));
    }

    ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);

    MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

    ierr = PetscPrintf(mpi.comm(), "Avg velo: %e, Time step: %e\n", vn_avg, dt);

//    Vec surf_tns_tmp;
//    ierr = VecDuplicate(phi_intf, &surf_tns_tmp); CHKERRXX(ierr);
//    ls.extend_from_interface_to_whole_domain_TVD(phi_intf, surf_tns, surf_tns_tmp, 20);
//    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
//    surf_tns = surf_tns_tmp;

    /* save data */
    if (save_vtk && iteration%save_every_dn == 0)
    {
      // Effective phi
      Vec phi_eff;
      ierr = VecDuplicate(phi_intf, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(nodes, &phi, &acn_int, NULL, phi_eff);

      // compute reference solution
      Vec phi_exact;
      ierr = VecDuplicate(phi_intf, &phi_exact); CHKERRXX(ierr);
      VecSetGhost(phi_exact, -1);

      if (compute_exact)
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
        double vec_d[P4EST_DIM] = { wall_x - com[0], wall_y - com[1], wall_z - com[2] };
        double nw_norm = sqrt(SQR(wall_nx) + SQR(wall_ny) + SQR(wall_nz));
        double d_norm2 = SQR(vec_d[0]) + SQR(vec_d[1]) + SQR(vec_d[2]);
        double dn0 = (vec_d[0]*wall_nx + vec_d[1]*wall_ny + vec_d[2]*wall_nz)/nw_norm;
#else
        double vec_d[P4EST_DIM] = { wall_x - com[0], wall_y - com[1] };
        double nw_norm = sqrt(SQR(wall_nx) + SQR(wall_ny));
        double d_norm2 = SQR(vec_d[0]) + SQR(vec_d[1]);
        double dn0 = (vec_d[0]*wall_nx + vec_d[1]*wall_ny)/nw_norm;
#endif

        double del = (d_norm2*wall_eps + 2.*dn0)
            / ( sqrt(1. + (d_norm2*wall_eps + 2.*dn0)*wall_eps) + 1. );

        double vec_n[P4EST_DIM];

        vec_n[0] = (wall_nx/nw_norm + vec_d[0]*wall_eps)/(1.+wall_eps*del);
        vec_n[1] = (wall_ny/nw_norm + vec_d[1]*wall_eps)/(1.+wall_eps*del);
#ifdef P4_TO_P8
        vec_n[2] = (wall_nz/nw_norm + vec_d[2]*wall_eps)/(1.+wall_eps*del);
        double norm = sqrt(SQR(vec_n[0]) + SQR(vec_n[1]) + SQR(vec_n[2]));
#else
        double norm = sqrt(SQR(vec_n[0]) + SQR(vec_n[1]));
#endif

        double xyz_c[P4EST_DIM];
        foreach_dimension(dim)
            xyz_c[dim] = com[dim] + (del-elev)*vec_n[dim]/norm;

#ifdef P4_TO_P8
        flower_shaped_domain_t exact(drop_r0, xyz_c[0], xyz_c[1], xyz_c[2]);
#else
        flower_shaped_domain_t exact(drop_r0, xyz_c[0], xyz_c[1]);
#endif

        sample_cf_on_nodes(p4est, nodes, exact.phi, phi_exact);

        foreach_dimension(dim) {
          ierr = VecDestroy(XYZ[dim]); CHKERRXX(ierr);
        }
      }

      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << mpi.size() << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
       #endif
             "." << (int) round(iteration/save_every_dn);

      PetscPrintf(mpi.comm(), "VTK is being saved in %s\n", oss.str().c_str());

      /* save the size of the leaves */
      Vec leaf_level;
      ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
      double *l_p;
      ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

      for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
          l_p[tree->quadrants_offset+q] = quad->level;
        }
      }

      for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
      {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
      }

      double *phi_ptr;
      double *phi_wall_ptr;
      double *phi_intf_ptr;
      double *phi_exact_ptr;
      double *kappa_ptr;
      double *surf_tns_ptr;
      double *mu_m_ptr;
      double *mu_p_ptr;
      double *velo_ptr;
      double *velo_full_ptr;

      ierr = VecGetArray(phi_eff, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_exact, &phi_exact_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             10, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "phi_wall", phi_wall_ptr,
                             VTK_POINT_DATA, "phi_intf", phi_intf_ptr,
                             VTK_POINT_DATA, "phi_exact", phi_exact_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_POINT_DATA, "surf_tns", surf_tns_ptr,
                             VTK_POINT_DATA, "mu_m", mu_m_ptr,
                             VTK_POINT_DATA, "mu_p", mu_p_ptr,
                             VTK_POINT_DATA, "velo", velo_ptr,
                             VTK_POINT_DATA, "velo_full", velo_full_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_eff, &phi_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_exact, &phi_exact_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      ierr = VecDestroy(phi_exact); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    /* advect interface and impose contact angle */
    ls.set_use_neumann_for_contact_angle(use_neumann);
    ls.set_contact_angle_extension(contact_angle_extension);
    ls.advect_in_normal_direction_with_contact_angle(velo, surf_tns, cos_angle, phi_wall, phi_intf, dt);

    ls.reinitialize_1st_order_time_2nd_order_space(phi_intf, 50);

    ierr = VecDestroy(velo); CHKERRXX(ierr);
    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
    ierr = VecDestroy(cos_angle); CHKERRXX(ierr);
    ierr = VecDestroy(velo_full); CHKERRXX(ierr);

    /* correct for volume loss */
    for (short i = 0; i < volume_corrections; ++i)
    {
      double volume_cur = integration.measure_of_domain();
      double intf_len   = integration.measure_of_interface(0);
      double correction = (volume_cur-volume)/intf_len;

      VecShiftGhost(phi_intf, correction);

      double volume_cur2 = integration.measure_of_domain();

      PetscPrintf(mpi.comm(), "Volume loss: %e, after correction: %e\n", (volume_cur-volume)/volume, (volume_cur2-volume)/volume);
    }

    /* refine and coarsen grid */
    {
      Vec phi_eff;
      double *phi_eff_ptr;

      ierr = VecDuplicate(phi_intf, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(nodes, &phi, &acn_int, &refine_always, phi_eff);
      ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin, lmax, lip);
      sp.set_refine_only_inside(0);

      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      bool is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

      if (is_grid_changing)
      {
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

        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_wall, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_intf, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_m, phi_intf, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_p, phi_intf, interpolation_between_grids);

        // delete old p4est
        p4est_destroy(p4est);       p4est = p4est_np1;
        p4est_ghost_destroy(ghost); ghost = ghost_np1;
        p4est_nodes_destroy(nodes); nodes = nodes_np1;
        hierarchy->update(p4est, ghost);
        ngbd->update(hierarchy, nodes);
      }
    }
    iteration++;

    foreach_dimension(dim)
    {
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(kappa); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi_intf); CHKERRXX(ierr);
  ierr = VecDestroy(phi_wall); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}



void parse_cmd(int argc, char *argv[])
{
  /* parse command line arguments */
  cmdParser cmd;
  M_PARSER_START
  {
    M_PARSER_ADD_OPTION(cmd, int, num_example, 0, "Number of predefined example")

    M_PARSER_STAGE_1 { set_parameters(); }

    // comptational domain
    M_PARSER_ADD_OPTION(cmd, double, xmin, -1, "xmin")
    M_PARSER_ADD_OPTION(cmd, double, ymin, -1, "ymin")
    M_PARSER_ADD_OPTION(cmd, double, zmin, -1, "zmin")
    M_PARSER_ADD_OPTION(cmd, double, xmax,  1, "xmax")
    M_PARSER_ADD_OPTION(cmd, double, ymax,  1, "ymax")
    M_PARSER_ADD_OPTION(cmd, double, zmax,  1, "zmax")

    M_PARSER_ADD_OPTION(cmd, bool, px, 0, "periodicity in x-dimension 0/1")
    M_PARSER_ADD_OPTION(cmd, bool, py, 0, "periodicity in y-dimension 0/1")
    M_PARSER_ADD_OPTION(cmd, bool, pz, 0, "periodicity in z-dimension 0/1")

    M_PARSER_ADD_OPTION(cmd, int, nx, 1, "number of trees in x-dimension")
    M_PARSER_ADD_OPTION(cmd, int, ny, 1, "number of trees in y-dimension")
    M_PARSER_ADD_OPTION(cmd, int, nz, 1, "number of trees in z-dimension")

    // grid parameters
    #ifdef P4_TO_P8
    M_PARSER_ADD_OPTION(cmd, int, lmin, 8, "min level of trees")
    M_PARSER_ADD_OPTION(cmd, int, lmax, 8, "max level of trees")
    #else
    M_PARSER_ADD_OPTION(cmd, int, lmin, 8, "min level of trees")
    M_PARSER_ADD_OPTION(cmd, int, lmax, 8, "max level of trees")
    #endif
    M_PARSER_ADD_OPTION(cmd, double, lip, 1.5, "Lipschitz constant")

    // advection parameters
    M_PARSER_ADD_OPTION(cmd, double, cfl,                     0.15, "CFL number")
    M_PARSER_ADD_OPTION(cmd, bool,   use_neumann,             1,    "Impose contact angle use Neumann BC 0/1")
    M_PARSER_ADD_OPTION(cmd, bool,   compute_exact,           0,    "Compute exact final shape (only for pure-curvature) 0/1")
    M_PARSER_ADD_OPTION(cmd, bool,   reinit_at_start,         1,    "Reinitialze level-set function at the start 0/1)")
    M_PARSER_ADD_OPTION(cmd, int,    contact_angle_extension, 2,    "Method for extending level-set function into wall: 0 - constant angle, 1 - , 2 - special")
    M_PARSER_ADD_OPTION(cmd, int,    volume_corrections,      2,    "Number of volume correction after each move")
    M_PARSER_ADD_OPTION(cmd, int,    max_iterations,          1000, "Maximum number of advection steps")

    // scft parameters
    M_PARSER_ADD_OPTION(cmd, bool,   use_scft,            0,     "Turn on/off SCFT 0/1")
    M_PARSER_ADD_OPTION(cmd, bool,   smooth_pressure,     1,     "Smooth pressure after first BC adjustment 0/1")
    M_PARSER_ADD_OPTION(cmd, int,    max_scft_iterations, 500,   "Maximum SCFT iterations")
    M_PARSER_ADD_OPTION(cmd, int,    bc_adjust_min,       5,     "Minimun SCFT steps between adjusting BC")
    M_PARSER_ADD_OPTION(cmd, double, scft_tol,            1.e-3, "Tolerance for SCFT")
    M_PARSER_ADD_OPTION(cmd, double, scft_bc_tol,         1.e-2, "Tolerance for adjusting BC")

    // polymer
    M_PARSER_ADD_OPTION(cmd, double, box_size, 10, "Box size in units of Rg")
    M_PARSER_ADD_OPTION(cmd, double, f,        0.5, "Fraction of polymer A")
    M_PARSER_ADD_OPTION(cmd, double, XN,       20,  "Flory-Higgins interaction parameter")
    M_PARSER_ADD_OPTION(cmd, int,    ns,       40,  "Discretization of polymer chain")

    // output parameters
    M_PARSER_ADD_OPTION(cmd, bool, save_vtk,        1, "")
    M_PARSER_ADD_OPTION(cmd, bool, save_parameters, 1, "")
    M_PARSER_ADD_OPTION(cmd, bool, save_data,       1, "")
    M_PARSER_ADD_OPTION(cmd, int,  save_every_dn,   1, "") // for vtk

    // problem setting
    M_PARSER_ADD_OPTION(cmd, int, num_polymer_geometry, 0, "Initial polymer shape: 0 - drop, 1 - film, 2 - combination")
    M_PARSER_ADD_OPTION(cmd, int, num_wall_geometry,    0, "Wall geometry: 0 - no wall, 1 - wall, 2 - well")
    M_PARSER_ADD_OPTION(cmd, int, num_wall_pattern,     0, "Wall chemical pattern: 0 - no pattern")
    M_PARSER_ADD_OPTION(cmd, int, num_seed,             2, "Seed: 0 - zero, 1 - random, 2 - horizontal stripes, 3 - vertical stripes, 4 - dots")

    // surface energies
    M_PARSER_ADD_OPTION(cmd, int, wall_energy_type, 1, "Method for setting wall surface energy: 0 - explicitly (i.e. convert XN to angles), 1 - through contact angles (i.e. convert angles to XN)")

    M_PARSER_ADD_OPTION(cmd, double, XN_air_avg, 20, "Polymer-air surface energy strength: average")
    M_PARSER_ADD_OPTION(cmd, double, XN_air_del, 15, "Polymer-air surface energy strength: difference")

    M_PARSER_ADD_OPTION(cmd, double, angle_A_min, 30,  "Minimum contact angle for A-block")
    M_PARSER_ADD_OPTION(cmd, double, angle_A_max, 30,  "Maximum contact angle for A-block")
    M_PARSER_ADD_OPTION(cmd, double, angle_B_min, 100, "Minimum contact angle for B-block")
    M_PARSER_ADD_OPTION(cmd, double, angle_B_max, 100, "Maximum contact angle for B-block")

    M_PARSER_ADD_OPTION(cmd, double, XN_wall_A_min, 5, "Minimum Polymer-wall interaction strength for A-block")
    M_PARSER_ADD_OPTION(cmd, double, XN_wall_A_max, 5, "Maximum Polymer-wall interaction strength for A-block")
    M_PARSER_ADD_OPTION(cmd, double, XN_wall_B_min, 8, "Minimum Polymer-wall interaction strength for B-block")
    M_PARSER_ADD_OPTION(cmd, double, XN_wall_B_max, 8, "Maximum Polymer-wall interaction strength for B-block")

    // geometry parameters
    M_PARSER_ADD_OPTION(cmd, double, drop_r,      0.5, "")
    M_PARSER_ADD_OPTION(cmd, double, drop_x,      .0, "")
    M_PARSER_ADD_OPTION(cmd, double, drop_y,      .0, "")
    M_PARSER_ADD_OPTION(cmd, double, drop_z,      .0, "")
    M_PARSER_ADD_OPTION(cmd, double, drop_r0,     0.6, "")
    M_PARSER_ADD_OPTION(cmd, double, drop_k,      10, "")
    M_PARSER_ADD_OPTION(cmd, double, drop_deform, 0.2, "")

    M_PARSER_ADD_OPTION(cmd, double, film_eps, -1.0, "") // curvature
    M_PARSER_ADD_OPTION(cmd, double, film_nx,  0, "")
    M_PARSER_ADD_OPTION(cmd, double, film_ny,  1, "")
    M_PARSER_ADD_OPTION(cmd, double, film_nz,  0, "")
    M_PARSER_ADD_OPTION(cmd, double, film_x,   .0, "")
    M_PARSER_ADD_OPTION(cmd, double, film_y,   .0, "")
    M_PARSER_ADD_OPTION(cmd, double, film_z,   .0, "")

    M_PARSER_ADD_OPTION(cmd, double, wall_eps, -.5, "") // curvature
    M_PARSER_ADD_OPTION(cmd, double, wall_nx,  -0, "")
    M_PARSER_ADD_OPTION(cmd, double, wall_ny,  -1, "")
    M_PARSER_ADD_OPTION(cmd, double, wall_nz,  -0, "")
    M_PARSER_ADD_OPTION(cmd, double, wall_x,   .0, "")
    M_PARSER_ADD_OPTION(cmd, double, wall_y,   -.5+0.01, "")
    M_PARSER_ADD_OPTION(cmd, double, wall_z,   .0, "")

    M_PARSER_ADD_OPTION(cmd, double, well_x, 0.00, "Well geometry: center")
    M_PARSER_ADD_OPTION(cmd, double, well_z, 0.53, "Well geometry: position")
    M_PARSER_ADD_OPTION(cmd, double, well_h, 1.00, "Well geometry: depth")
    M_PARSER_ADD_OPTION(cmd, double, well_w, 0.77, "Well geometry: width")
    M_PARSER_ADD_OPTION(cmd, double, well_r, 0.10, "Well geometry: corner smoothing")

    M_PARSER_PARSE(cmd, argc, argv);
  }

  set_wall_surface_energies();
}

void write_parameters(MPI_Comm mpicomm, const std::string &output)
{
  /* save parameters */
  FILE *fich;
  ierr = PetscFOpen(mpicomm, output.c_str(), "w", &fich); CHKERRXX(ierr);

  // comptational domain
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, xmin, -1, "xmin");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, ymin, -1, "ymin");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, zmin, -1, "zmin");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, xmax,  1, "xmax");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, ymax,  1, "ymax");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, zmax,  1, "zmax");
  PetscFPrintf(mpicomm, fich, "\n");

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool, px, 0, "periodicity in x-dimension 0/1");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool, py, 0, "periodicity in y-dimension 0/1");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool, pz, 0, "periodicity in z-dimension 0/1");
  PetscFPrintf(mpicomm, fich, "\n");

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, nx, 1, "number of trees in x-dimension");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, ny, 1, "number of trees in y-dimension");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, nz, 1, "number of trees in z-dimension");
  PetscFPrintf(mpicomm, fich, "\n");

  // grid parameters
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, lmin, 8, "min level of trees");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, lmax, 8, "max level of trees");
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, lip, 1.5, "Lipschitz constant");
  PetscFPrintf(mpicomm, fich, "\n");

  // advection parameters
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, cfl,                     0.15, "CFL number")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool,   use_neumann,             1,    "Impose contact angle use Neumann BC 0/1")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool,   compute_exact,           0,    "Compute exact final shape (only for pure-curvature) 0/1")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool,   reinit_at_start,         1,    "Reinitialze level-set function at the start 0/1)")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,    contact_angle_extension, 2,    "Method for extending level-set function into wall: 0 - constant angle, 1 - , 2 - special")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,    volume_corrections,      2,    "Number of volume correction after each move")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,    max_iterations,          1000, "Maximum number of advection steps")
  PetscFPrintf(mpicomm, fich, "\n");

  // scft parameters
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool,   use_scft,            0,     "Turn on/off SCFT 0/1")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool,   smooth_pressure,     1,     "Smooth pressure after first BC adjustment 0/1")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,    max_scft_iterations, 500,   "Maximum SCFT iterations")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,    bc_adjust_min,       5,     "Minimun SCFT steps between adjusting BC")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, scft_tol,            1.e-3, "Tolerance for SCFT")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, scft_bc_tol,         1.e-2, "Tolerance for adjusting BC")
  PetscFPrintf(mpicomm, fich, "\n");

  // polymer
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, box_size, 10, "Box size in units of Rg")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, f,        0.5, "Fraction of polymer A")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN,       20,  "Flory-Higgins interaction parameter")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,    ns,       40,  "Discretization of polymer chain")
  PetscFPrintf(mpicomm, fich, "\n");

  // output parameters
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool, save_vtk,        1, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool, save_parameters, 1, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, bool, save_data,       1, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int,  save_every_dn,   1, "") // for vtk
  PetscFPrintf(mpicomm, fich, "\n");

  // problem setting
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, num_polymer_geometry, 0, "Initial polymer shape: 0 - drop, 1 - film, 2 - combination")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, num_wall_geometry,    0, "Wall geometry: 0 - no wall, 1 - wall, 2 - well")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, num_wall_pattern,     0, "Wall chemical pattern: 0 - no pattern")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, num_seed,             2, "Seed: 0 - zero, 1 - random, 2 - horizontal stripes, 3 - vertical stripes, 4 - dots")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, num_example,          0, "Number of predefined example")
      PetscFPrintf(mpicomm, fich, "\n");

  // surface energies
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, int, wall_energy_type, 1, "Method for setting wall surface energy: 0 - explicitly (i.e. convert XN to angles), 1 - through contact angles (i.e. convert angles to XN)")

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN_air_avg, 20, "Polymer-air surface energy strength: average")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN_air_del, 15, "Polymer-air surface energy strength: difference")

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, angle_A_min, 30,  "Minimum contact angle for A-block")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, angle_A_max, 30,  "Maximum contact angle for A-block")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, angle_B_min, 100, "Minimum contact angle for B-block")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, angle_B_max, 100, "Maximum contact angle for B-block")

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN_wall_A_min, 5, "Minimum Polymer-wall interaction strength for A-block")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN_wall_A_max, 5, "Maximum Polymer-wall interaction strength for A-block")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN_wall_B_min, 8, "Minimum Polymer-wall interaction strength for B-block")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, XN_wall_B_max, 8, "Maximum Polymer-wall interaction strength for B-block")
      PetscFPrintf(mpicomm, fich, "\n");

  // geometry parameters
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_r,      0.5, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_x,      .0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_y,      .0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_z,      .0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_r0,     0.6, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_k,      10, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, drop_deform, 0.2, "")
      PetscFPrintf(mpicomm, fich, "\n");

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_eps, -1.0, "") // curvature
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_nx,  0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_ny,  1, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_nz,  0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_x,   .0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_y,   .0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, film_z,   .0, "")
      PetscFPrintf(mpicomm, fich, "\n");

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_eps, -.5, "") // curvature
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_nx,  -0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_ny,  -1, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_nz,  -0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_x,   .0, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_y,   -.5+0.01, "")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, wall_z,   .0, "")
      PetscFPrintf(mpicomm, fich, "\n");

  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, well_x, 0.00, "Well geometry: center")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, well_z, 0.53, "Well geometry: position")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, well_h, 1.00, "Well geometry: depth")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, well_w, 0.77, "Well geometry: width")
  M_PARSER_WRITE_VARIABLE(mpicomm, fich, double, well_r, 0.10, "Well geometry: corner smoothing")
      PetscFPrintf(mpicomm, fich, "\n");

  ierr = PetscFClose(mpicomm, fich); CHKERRXX(ierr);
}
