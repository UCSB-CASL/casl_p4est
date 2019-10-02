
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
parameter_list_t pl;

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
static double xmin = -1; ADD_TO_LIST(pl, xmin, "xmin");
static double ymin = -1; ADD_TO_LIST(pl, ymin, "ymin");
static double zmin = -1; ADD_TO_LIST(pl, zmin, "zmin");
static double xmax =  1; ADD_TO_LIST(pl, xmax, "xmax");
static double ymax =  1; ADD_TO_LIST(pl, ymax, "ymax");
static double zmax =  1; ADD_TO_LIST(pl, zmax, "zmax");

static bool px = 1; ADD_TO_LIST(pl, px, "periodicity in x-dimension 0/1");
static bool py = 0; ADD_TO_LIST(pl, py, "periodicity in y-dimension 0/1");
static bool pz = 0; ADD_TO_LIST(pl, pz, "periodicity in z-dimension 0/1");

static int nx = 1; ADD_TO_LIST(pl, nx, "number of trees in x-dimension");
static int ny = 1; ADD_TO_LIST(pl, ny, "number of trees in y-dimension");
static int nz = 1; ADD_TO_LIST(pl, nz, "number of trees in z-dimension");

// grid parameters
#ifdef P4_TO_P8
static int lmin = 8; ADD_TO_LIST(pl, lmin, "min level of trees");
static int lmax = 8; ADD_TO_LIST(pl, lmax, "max level of trees");
#else
static int lmin = 8; ADD_TO_LIST(pl, lmin, "min level of trees");
static int lmax = 8; ADD_TO_LIST(pl, lmax, "max level of trees");
#endif
static double lip = 1.5; ADD_TO_LIST(pl, lip, "Lipschitz constant");
static bool refine_only_inside = 1; ADD_TO_LIST(pl, refine_only_inside, "Refine only inside");


// advection parameters
static double cfl                     = 0.5;    ADD_TO_LIST(pl, cfl, "CFL number");
static bool   use_neumann             = 1;      ADD_TO_LIST(pl, use_neumann, "Impose contact angle use Neumann BC 0/1");
static bool   compute_exact           = 0;      ADD_TO_LIST(pl, compute_exact, "Compute exact final shape (only for pure-curvature) 0/1");
static bool   rerefine_at_start       = 1;      ADD_TO_LIST(pl, rerefine_at_start, "Reinitialze level-set function at the start 0/1)");
static int    contact_angle_extension = 0;      ADD_TO_LIST(pl, contact_angle_extension, "Method for extending level-set function into wall: 0 - constant angle = 1 -  = 2 - special");
static int    volume_corrections      = 2;      ADD_TO_LIST(pl, volume_corrections, "Number of volume correction after each move");
static int    max_iterations          = 10000;   ADD_TO_LIST(pl, max_iterations, "Maximum number of advection steps");
static double tolerance               = 1.0e-8; ADD_TO_LIST(pl, tolerance, "Stopping criteria");

static interpolation_method interpolation_between_grids = quadratic_non_oscillatory_continuous_v2;

// scft parameters
static bool   use_scft            = 1;     ADD_TO_LIST(pl, use_scft           , "Turn on/off SCFT 0/1");
static bool   smooth_pressure     = 1;     ADD_TO_LIST(pl, smooth_pressure    , "Smooth pressure after first BC adjustment 0/1");
static int    max_scft_iterations = 300;   ADD_TO_LIST(pl, max_scft_iterations, "Maximum SCFT iterations");
static int    bc_adjust_min       = 5;     ADD_TO_LIST(pl, bc_adjust_min      , "Minimun SCFT steps between adjusting BC");
static double scft_tol            = 4.e-3; ADD_TO_LIST(pl, scft_tol           , "Tolerance for SCFT");
static double scft_bc_tol         = 4.e-2; ADD_TO_LIST(pl, scft_bc_tol        , "Tolerance for adjusting BC");

// polymer
static double box_size = 10; ADD_TO_LIST(pl, box_size, "Box size in units of Rg");
static double f        = .5; ADD_TO_LIST(pl, f       , "Fraction of polymer A");
static double XN       = 20; ADD_TO_LIST(pl, XN      , "Flory-Higgins interaction parameter");
static int    ns       = 40; ADD_TO_LIST(pl, ns      , "Discretization of polymer chain");

// output parameters
static bool save_vtk        = 1; ADD_TO_LIST(pl, save_vtk       , "");
static bool save_parameters = 1; ADD_TO_LIST(pl, save_parameters, "");
static bool save_data       = 1; ADD_TO_LIST(pl, save_data      , "");
static int  save_every_dn   = 1; ADD_TO_LIST(pl, save_every_dn  , ""); // for vtk

// problem setting
static int num_polymer_geometry = 1; ADD_TO_LIST(pl, num_polymer_geometry, "Initial polymer shape: 0 - drop = 1 - film = 2 - combination");
static int num_wall_geometry    = 1; ADD_TO_LIST(pl, num_wall_geometry   , "Wall geometry: 0 - no wall = 1 - wall = 2 - well");
static int num_wall_pattern     = 0; ADD_TO_LIST(pl, num_wall_pattern    , "Wall chemical pattern: 0 - no pattern");
static int num_seed             = 3; ADD_TO_LIST(pl, num_seed            , "Seed: 0 - zero = 1 - random = 2 - horizontal stripes = 3 - vertical stripes = 4 - dots");
static int num_example          = 0; ADD_TO_LIST(pl, num_example         , "Number of predefined example");

// surface energies
static int wall_energy_type = 1; ADD_TO_LIST(pl, wall_energy_type, "Method for setting wall surface energy: 0 - explicitly (i.e. convert XN to angles) = 1 - through contact angles (i.e. convert angles to XN)");

static double XN_air_avg = 5;    ADD_TO_LIST(pl, XN_air_avg, "Polymer-air surface energy strength: average");
static double XN_air_del = 0; ADD_TO_LIST(pl, XN_air_del, "Polymer-air surface energy strength: difference");

static double angle_A_min = 90; ADD_TO_LIST(pl, angle_A_min, "Minimum contact angle for A-block");
static double angle_A_max = 90; ADD_TO_LIST(pl, angle_A_max, "Maximum contact angle for A-block");
static double angle_B_min = 90; ADD_TO_LIST(pl, angle_B_min, "Minimum contact angle for B-block");
static double angle_B_max = 90; ADD_TO_LIST(pl, angle_B_max, "Maximum contact angle for B-block");

static double XN_wall_A_min = 5; ADD_TO_LIST(pl, XN_wall_A_min, "Minimum Polymer-wall interaction strength for A-block");
static double XN_wall_A_max = 5; ADD_TO_LIST(pl, XN_wall_A_max, "Maximum Polymer-wall interaction strength for A-block");
static double XN_wall_B_min = 8; ADD_TO_LIST(pl, XN_wall_B_min, "Minimum Polymer-wall interaction strength for B-block");
static double XN_wall_B_max = 8; ADD_TO_LIST(pl, XN_wall_B_max, "Maximum Polymer-wall interaction strength for B-block");

// geometry parameters
static double drop_r      = 0.611;  ADD_TO_LIST(pl, drop_r     , "");
static double drop_x      = .0079;  ADD_TO_LIST(pl, drop_x     , "");
static double drop_y      = -.013; ADD_TO_LIST(pl, drop_y     , "");
static double drop_z      = .0;     ADD_TO_LIST(pl, drop_z     , "");
static double drop_r0     = 0.4;    ADD_TO_LIST(pl, drop_r0    , "");
static double drop_k      = 5;      ADD_TO_LIST(pl, drop_k     , "");
static double drop_deform = 0.0;    ADD_TO_LIST(pl, drop_deform, "");

static double film_eps = -0.0; ADD_TO_LIST(pl, film_eps, ""); // curvature
static double film_nx  = 0;    ADD_TO_LIST(pl, film_nx , "");
static double film_ny  = 1;    ADD_TO_LIST(pl, film_ny , "");
static double film_nz  = 0;    ADD_TO_LIST(pl, film_nz , "");
static double film_x   = .0;   ADD_TO_LIST(pl, film_x  , "");
static double film_y   = .0;   ADD_TO_LIST(pl, film_y  , "");
static double film_z   = .0;   ADD_TO_LIST(pl, film_z  , "");
static double film_perturb   = .1;   ADD_TO_LIST(pl, film_perturb  , "");

static double wall_eps = -0.0; ADD_TO_LIST(pl, wall_eps, ""); // curvature
static double wall_nx  = -0;   ADD_TO_LIST(pl, wall_nx , "");
static double wall_ny  = -1;   ADD_TO_LIST(pl, wall_ny , "");
static double wall_nz  = -0;   ADD_TO_LIST(pl, wall_nz , "");
static double wall_x   = .0;   ADD_TO_LIST(pl, wall_x  , "");
static double wall_y   = -.2; ADD_TO_LIST(pl, wall_y  , "");
static double wall_z   = .0;   ADD_TO_LIST(pl, wall_z  , "");

static double well_x = 0.00;   ADD_TO_LIST(pl, well_x, "Well geometry: center");
static double well_z = 0.0053; ADD_TO_LIST(pl, well_z, "Well geometry: position");
static double well_h = .50;    ADD_TO_LIST(pl, well_h, "Well geometry: depth");
static double well_w = 0.77;   ADD_TO_LIST(pl, well_w, "Well geometry: width");
static double well_r = 0.10;   ADD_TO_LIST(pl, well_r, "Well geometry: corner smoothing");

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
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(XN_air_avg+XN_air_del);
  }
} gamma_Aa_cf;

class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return sqrt(XN_air_avg-XN_air_del);
  }
} gamma_Ba_cf;

class gamma_Aw_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
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
  double operator()(DIM(double x, double y, double z)) const
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

class phi_infc_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
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
                      ) + 1. ) + film_perturb*cos(PI*x*(lmax-2));
      }
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_infc_cf;

inline double lam_bulk_period()
{
  return 2.*pow(8.*XN/3./pow(PI,4.),1./6.)/box_size;
}

class mu_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z) ) const
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
  double operator()(DIM(double x, double y, double z) ) const
  {
    return MIN( MAX(phi_infc_cf(DIM(x,y,z)), phi_wall_cf(DIM(x,y,z))), fabs(phi_wall_cf(DIM(x,y,z))) );
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

PetscErrorCode ierr;
int main (int argc, char* argv[])
{
  /* initialize MPI */
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  srand(mpi.rank());

  /* parse command line arguments for parameters */
  cmdParser cmd;
  pl.initialize_parser(cmd);
//  cmd.show_help();
  cmd.parse(argc, argv);
//  geometry = cmd.get("geometry", geometry);
//  set_geometry();
  pl.get_all(cmd);
//  if (mpi.rank() == 0) pl.print_all();

  set_wall_surface_energies();

  /* prepare output directories */
  const char* out_dir = getenv("OUT_DIR");
  if (mpi.rank() == 0 && (save_vtk || save_parameters || save_data))
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

  if (mpi.rank() == 0 && save_vtk)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys < 0) throw std::invalid_argument("Could not create a directory");
  }

  if (mpi.rank() == 0 && save_parameters) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  FILE *file_conv;
  char file_conv_name[10000];
  if (save_data)
  {
    sprintf(file_conv_name, "%s/data.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_conv_name, "w", &file_conv); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_conv, "iteration "
                                               "total_time "
                                               "iteration_time "
                                               "energy "
                                               "pressure_force "
                                               "exchange_force "
                                               "bc_adjusted\n"); CHKERRXX(ierr);
//    ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);
  }

  /* start a timer */
  parStopWatch w;
  w.start("total time");

  /* create the p4est */
  double xyz_min[]  = { DIM(xmin, ymin, zmin) };
  double xyz_max[]  = { DIM(xmax, ymax, zmax) };
  int    nb_trees[] = { DIM(nx, ny, nz) };
  int    periodic[] = { DIM(px, py, pz) };

  my_p4est_brick_t      brick;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t              *p4est        = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_eff_cf, lip);
  data.set_refine_only_inside(true);

  p4est->user_pointer = (void*)(&data);
  for (int i = 0; i < lmax; ++i)
  {
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
  }

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  /* reinitialize and refine-coarsen agan */
  if (rerefine_at_start)
  {
    my_p4est_level_set_t ls(ngbd);

    Vec phi_wall;
    Vec phi_infc;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_wall); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_infc); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);
    sample_cf_on_nodes(p4est, nodes, phi_infc_cf, phi_infc);
    ls.reinitialize_1st_order(phi_wall, 100);
    ls.reinitialize_1st_order(phi_infc, 100);

    Vec phi_eff;
    ierr = VecDuplicate(phi_wall, &phi_eff); CHKERRXX(ierr);

    double *phi_wall_ptr; ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
    double *phi_infc_ptr; ierr = VecGetArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
    double *phi_eff_ptr;  ierr = VecGetArray(phi_eff,  &phi_eff_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      phi_eff_ptr[n] = MIN( MAX(phi_wall_ptr[n], phi_infc_ptr[n]), ABS(phi_wall_ptr[n]));
    }

    ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_eff,  &phi_eff_ptr); CHKERRXX(ierr);

    ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
    ierr = VecDestroy(phi_infc); CHKERRXX(ierr);

    p4est_t       *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    Vec phi_eff_np1;
    ierr = VecDuplicate(phi_eff, &phi_eff_np1); CHKERRXX(ierr);
    VecCopyGhost(phi_eff, phi_eff_np1);

    bool is_grid_changing = true;
    while (is_grid_changing)
    {
      double *phi_eff_ptr;
      ierr = VecGetArray(phi_eff_np1, &phi_eff_ptr); CHKERRXX(ierr);

      splitting_criteria_tag_t sp(lmin, lmax, lip);
      sp.set_refine_only_inside(refine_only_inside);
      is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

      ierr = VecRestoreArray(phi_eff_np1, &phi_eff_ptr); CHKERRXX(ierr);

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

        ierr = VecDestroy(phi_eff_np1); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_eff_np1); CHKERRXX(ierr);

        interp.set_input(phi_eff, linear);
        interp.interpolate(phi_eff_np1);

      }
    }

    ierr = VecDestroy(phi_eff); CHKERRXX(ierr);
    ierr = VecDestroy(phi_eff_np1); CHKERRXX(ierr);

    // delete old p4est
    p4est_destroy(p4est);       p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);
  }

  double dxyz[P4EST_DIM], h, diag;
  get_dxyz_min(p4est, dxyz, h, diag);

  /* initialize geometry */
  Vec phi_infc; ierr = VecCreateGhostNodes(p4est, nodes, &phi_infc); CHKERRXX(ierr);
  Vec phi_wall; ierr = VecCreateGhostNodes(p4est, nodes, &phi_wall); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_infc_cf, phi_infc);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);

  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_2nd_order(phi_infc);
  ls.reinitialize_2nd_order(phi_wall);

  /* initialize potentials */
  Vec mu_m; ierr = VecDuplicate(phi_infc, &mu_m); CHKERRXX(ierr);
  Vec mu_p; ierr = VecDuplicate(phi_infc, &mu_p); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, mu_cf,   mu_m);
  VecSetGhost(mu_p, 0);

  /* compute initial volume for volume-loss corrections */
  double volume = 0;

  {
    std::vector<Vec> phi(2);
    std::vector<mls_opn_t> acn(2, MLS_INTERSECTION);
    std::vector<int> clr(2);

    phi[0] = phi_infc; clr[0] = 0;
    phi[1] = phi_wall; clr[1] = 1;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    volume = integration.measure_of_domain();
  }

  // in case of constant contact angle and simple geometry we can compute analytically position and volume of the steady-state shape
  double elev = 0;
  if (compute_exact)
  {
    double g_Aw = gamma_Aw_cf(DIM(0,0,0));
    double g_Bw = gamma_Bw_cf(DIM(0,0,0));
    double g_Aa = gamma_Aa_cf(DIM(0,0,0));
    double g_Ba = gamma_Ba_cf(DIM(0,0,0));
    double g_aw = gamma_aw_cf(DIM(0,0,0));

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
    std::vector<mls_opn_t> acn(2, MLS_INTERSECTION);
    std::vector<int> acn_int(2, 0);
    std::vector<int> clr(2);
    std::vector<bool> refine_always(2);

    phi[0] = phi_infc; clr[0] = 0; refine_always[0] = false;
    phi[1] = phi_wall; clr[1] = 1; refine_always[1] = true;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    Vec phi_eff;
    ierr = VecDuplicate(phi_infc, &phi_eff); CHKERRXX(ierr);

    VecPointwiseMaxGhost(phi_eff, phi_infc, phi_wall);
    ls.reinitialize_2nd_order(phi_eff);

    /* normal and curvature */
    Vec normal[P4EST_DIM];
    Vec kappa;

    foreach_dimension(dim) { ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);

    compute_normals_and_mean_curvature(*ngbd, phi_infc, normal, kappa);

//    double *kappa_ptr;
//    double *phi_wall_ptr;
//    double *phi_infc_ptr;

//    ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);
//    ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
//    ierr = VecGetArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);

//    foreach_node(n, nodes)
//    {
//      double dist = MAX(phi_wall_ptr[n], fabs(phi_infc_ptr[n]));

//      kappa_ptr[n] *= 1.-smoothstep(1, (dist/diag - 10.)/20.);
//    }

//    ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
//    ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
//    ierr = VecRestoreArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);

    /* vector field for extrapolation tangentially to the interface */
    // compute normal to the wall
    Vec normal_wall[P4EST_DIM];
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &normal_wall[dim]); CHKERRXX(ierr); }

    compute_normals(*ngbd, phi_wall, normal_wall);

    // deform normal field
    double band_deform = 1.5*diag;
    double band_smooth = 7.0*diag;

    double *normal_ptr[P4EST_DIM];
    double *normal_wall_ptr[P4EST_DIM];
    double *phi_infc_ptr;

    ierr = VecGetArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
    foreach_dimension(dim)
    {
      ierr = VecGetArray(normal[dim], &normal_ptr[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(normal_wall[dim], &normal_wall_ptr[dim]); CHKERRXX(ierr);
    }

    foreach_node(n, nodes)
    {
      double dot_product = normal_ptr[0][n]*normal_wall_ptr[0][n] +
                           normal_ptr[1][n]*normal_wall_ptr[1][n];

      normal_wall_ptr[0][n] -= (1.-smoothstep(1, (fabs(phi_infc_ptr[n]) - band_deform)/band_smooth))*dot_product*normal_ptr[0][n];
      normal_wall_ptr[1][n] -= (1.-smoothstep(1, (fabs(phi_infc_ptr[n]) - band_deform)/band_smooth))*dot_product*normal_ptr[1][n];

      double norm = sqrt(SQR(normal_wall_ptr[0][n]) + SQR(normal_wall_ptr[1][n]));

      normal_wall_ptr[0][n] /= norm;
      normal_wall_ptr[1][n] /= norm;
    }

    ierr = VecRestoreArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
    foreach_dimension(dim)
    {
      ierr = VecRestoreArray(normal[dim], &normal_ptr[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_wall[dim], &normal_wall_ptr[dim]); CHKERRXX(ierr);
    }

    // extension tangentially to the interface
//    VecShiftGhost(phi_wall,  1.5*diag);
//    ls.extend_Over_Interface_TVD_Full(phi_wall, kappa, 50, 0, 0, DBL_MAX, DBL_MAX, DBL_MAX, normal_wall);
//    VecShiftGhost(phi_wall, -1.5*diag);

    ls.set_interpolation_on_interface(linear);
    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_infc, kappa, phi_infc, 20, phi_wall);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);


    /* get density field and non-curvature velocity */
    Vec velo;
    ierr = VecDuplicate(phi_infc, &velo); CHKERRXX(ierr);

    if (use_scft)
    {
      my_p4est_scft_t scft(ngbd, ns);

      /* set geometry */
      scft.add_boundary(phi_infc, MLS_INTERSECTION, gamma_Aa_cf, gamma_Ba_cf);
      scft.add_boundary(phi_wall, MLS_INTERSECTION, gamma_Aw_cf, gamma_Bw_cf);

      scft.set_scalling(1./box_size);
      scft.set_polymer(f, XN);

      /* initialize potentials */
      Vec mu_m_tmp = scft.get_mu_m();
      Vec mu_p_tmp = scft.get_mu_p();

      VecCopyGhost(mu_m, mu_m_tmp);
      VecCopyGhost(mu_p, mu_p_tmp);

      /* initialize diffusion solvers for propagators */
      scft.initialize_solvers();
      scft.initialize_bc_smart(iteration != 0);

      /* main loop for solving SCFT equations */
      int    scft_iteration = 0;
      int    bc_iters       = 0;
      double scft_error     = 2.*scft_tol+1.;
      while (scft_iteration < max_scft_iterations && scft_error > scft_tol || scft_iteration < bc_adjust_min+1)
      {
        // do an SCFT step
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

      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_infc, velo, phi_infc, 20, phi_wall);

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

      ierr = VecDuplicate(phi_infc, &mu_wall); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_infc, &mu_intf); CHKERRXX(ierr);

//      ls.extend_Over_Interface_TVD_Full(phi_eff, mu_m, 20, 2);
//      ls.extend_Over_Interface_TVD_Full(phi_wall, mu_m, 20, 2, 0, DBL_MAX, DBL_MAX, DBL_MAX, normal_wall);
//      ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_infc, mu_m, phi_infc, 20, phi_wall);

//      energy  = .5*(gamma_Aa+gamma_Ba)*integration.measure_of_interface(0) + .5*(gamma_Aa-gamma_Ba)*integration.integrate_over_interface(0, mu_intf);
//      energy += .5*(gamma_Aw+gamma_Bw)*integration.measure_of_interface(1) + .5*(gamma_Aw-gamma_Bw)*integration.integrate_over_interface(1, mu_wall);

      ierr = VecDestroy(mu_wall); CHKERRXX(ierr);
      ierr = VecDestroy(mu_intf); CHKERRXX(ierr);
    }

    /* compute velocity and contact angle */
    Vec surf_tns;
    Vec cos_angle;

    ierr = VecDuplicate(phi_infc, &surf_tns); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_infc, &cos_angle); CHKERRXX(ierr);

    double *mu_ptr;
    double *surf_tns_ptr;
    double *cos_angle_ptr;

    ierr = VecGetArray(mu_m, &mu_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(cos_angle, &cos_angle_ptr); CHKERRXX(ierr);

    double scalling = 1./box_size;

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

    ls.extend_Over_Interface_TVD_Full(phi_wall, cos_angle, 20, 0, 0, DBL_MAX, DBL_MAX, DBL_MAX, normal_wall);
    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_infc, cos_angle, phi_infc, 20, phi_wall);

    // average velocity
    Vec velo_full;
    Vec surf_tns_d[P4EST_DIM];
    Vec tmp;

    ierr = VecDuplicate(phi_infc, &velo_full); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_infc, &tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDuplicate(normal[dim], &surf_tns_d[dim]); CHKERRXX(ierr); }

    ls.extend_from_interface_to_whole_domain_TVD(phi_infc, surf_tns, tmp);
    VecPointwiseMultGhost(velo_full, kappa, tmp);

    ngbd->first_derivatives_central(surf_tns, surf_tns_d);
    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(surf_tns_d[dim], surf_tns_d[dim], normal[dim]);
      ls.extend_from_interface_to_whole_domain_TVD(phi_infc, surf_tns_d[dim], tmp);
      VecAXPBYGhost(velo_full, 1, 1, tmp);
    }

    VecScaleGhost(velo_full, -1);
    VecAXPBYGhost(velo_full, 1, 1, velo);

    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_infc, velo_full, phi_infc, 20, phi_wall);

    double vn_avg = integration.integrate_over_interface(0, velo_full)/integration.measure_of_interface(0);

    ierr = VecDestroy(tmp); CHKERRXX(ierr);
    foreach_dimension(dim) { ierr = VecDestroy(surf_tns_d[dim]); CHKERRXX(ierr); }

    VecShiftGhost(velo, -vn_avg);
    VecShiftGhost(velo_full, -vn_avg);

    ls.extend_from_interface_to_whole_domain_TVD_in_place(phi_infc, velo, phi_infc, 20, phi_wall);

    /* compute time step dt */
    double dt_local = DBL_MAX;
    double vmax = 0;
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

      dt_local = MIN(dt_local, cfl*fabs(s_min/velo_ptr[n]));
      vmax = MAX(vmax, fabs(velo_ptr[n]));
    }

    ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);

    MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);
    MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

//    VecSetGhost(cos_angle, 0);


//    Vec surf_tns_tmp;
//    ierr = VecDuplicate(phi_infc, &surf_tns_tmp); CHKERRXX(ierr);
//    ls.extend_from_interface_to_whole_domain_TVD(phi_infc, surf_tns, surf_tns_tmp, 20);
//    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
//    surf_tns = surf_tns_tmp;

    double energy_change_predicted;

    Vec integrand;
    ierr = VecDuplicate(phi_infc, &integrand); CHKERRXX(ierr);

    VecPointwiseMultGhost(integrand, velo_full, velo_full); CHKERRXX(ierr);

    energy_change_predicted = -dt*integration.integrate_over_interface(0, integrand);

    ierr = VecDestroy(integrand); CHKERRXX(ierr);

    if (save_data && iteration%save_every_dn == 0)
    {
      ierr = PetscFPrintf(mpi.comm(), file_conv, "%d %e %e %e %e %e\n", (int) round(iteration/save_every_dn), dt, energy, energy-energy_old, energy_change_predicted, vmax); CHKERRXX(ierr);
    }

    ierr = PetscPrintf(mpi.comm(), "Avg velo: %e, Time step: %e\n", vn_avg, dt);
    ierr = PetscPrintf(mpi.comm(), "Energy: %e, Change: %e, Predicted: %e\n", energy, energy-energy_old, energy_change_predicted);
    energy_old = energy;

    /* save data */
    if (save_vtk && iteration%save_every_dn == 0)
    {
      // Effective phi
      Vec phi_eff;
      ierr = VecDuplicate(phi_infc, &phi_eff); CHKERRXX(ierr);
      compute_phi_eff(phi_eff, nodes, phi, acn);

      // compute reference solution
      Vec phi_exact;
      ierr = VecDuplicate(phi_infc, &phi_exact); CHKERRXX(ierr);
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
      double *phi_infc_ptr;
      double *phi_exact_ptr;
      double *kappa_ptr;
      double *surf_tns_ptr;
      double *mu_m_ptr;
      double *mu_p_ptr;
      double *velo_ptr;
      double *velo_full_ptr;
      double *cos_ptr;
      double *nx;
      double *ny;

      ierr = VecGetArray(phi_eff, &phi_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_exact, &phi_exact_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(cos_angle, &cos_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(normal_wall[0], &nx); CHKERRXX(ierr);
      ierr = VecGetArray(normal_wall[1], &ny); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             13, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_ptr,
                             VTK_POINT_DATA, "phi_wall", phi_wall_ptr,
                             VTK_POINT_DATA, "phi_infc", phi_infc_ptr,
                             VTK_POINT_DATA, "phi_exact", phi_exact_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_POINT_DATA, "surf_tns", surf_tns_ptr,
                             VTK_POINT_DATA, "mu_m", mu_m_ptr,
                             VTK_POINT_DATA, "mu_p", mu_p_ptr,
                             VTK_POINT_DATA, "velo", velo_ptr,
                             VTK_POINT_DATA, "velo_full", velo_full_ptr,
                             VTK_POINT_DATA, "cos", cos_ptr,
                             VTK_POINT_DATA, "nx", nx,
                             VTK_POINT_DATA, "ny", ny,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_eff, &phi_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_exact, &phi_exact_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(surf_tns, &surf_tns_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_m, &mu_m_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_p, &mu_p_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo, &velo_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo_full, &velo_full_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(cos_angle, &cos_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_wall[0], &nx); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_wall[1], &ny); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      ierr = VecDestroy(phi_exact); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    /* advect interface and impose contact angle */
    ls.set_use_neumann_for_contact_angle(use_neumann);
    ls.set_contact_angle_extension(contact_angle_extension);

    int splits = 1;
    for (int i = 0; i < splits; ++i)
    {
      ls.advect_in_normal_direction_with_contact_angle(velo, surf_tns, cos_angle, phi_wall, phi_infc, dt/double(splits));

      // cut off tails
      double *phi_wall_ptr;
      double *phi_infc_ptr;

      ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);

      foreach_node(n, nodes)
      {
        double transition = smoothstep(1, (phi_wall_ptr[n]/diag - 10.)/20.);
//        phi_infc_ptr[n] = smooth_max(phi_infc_ptr[n], phi_wall_ptr[n] - 12.*diag, 4.*diag);
        phi_infc_ptr[n] = (1.-transition)*phi_infc_ptr[n] + transition*MAX(phi_infc_ptr[n], phi_wall_ptr[n] - 10.*diag);
      }

      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);

      ls.reinitialize_2nd_order(phi_infc);
      /* correct for volume loss */
      for (short i = 0; i < volume_corrections; ++i)
      {
        double volume_cur = integration.measure_of_domain();
        double intf_len   = integration.measure_of_interface(0);
        double correction = (volume_cur-volume)/intf_len;

        VecShiftGhost(phi_infc, correction);

        double volume_cur2 = integration.measure_of_domain();

        PetscPrintf(mpi.comm(), "Volume loss: %e, after correction: %e\n", (volume_cur-volume)/volume, (volume_cur2-volume)/volume);
      }
    }

    ierr = VecDestroy(velo); CHKERRXX(ierr);
    ierr = VecDestroy(surf_tns); CHKERRXX(ierr);
    ierr = VecDestroy(cos_angle); CHKERRXX(ierr);
    ierr = VecDestroy(velo_full); CHKERRXX(ierr);


    /* refine and coarsen grid */
    {
      Vec phi_eff;
      ierr = VecDuplicate(phi_wall, &phi_eff); CHKERRXX(ierr);

      double *phi_wall_ptr; ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      double *phi_infc_ptr; ierr = VecGetArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
      double *phi_eff_ptr;  ierr = VecGetArray(phi_eff,  &phi_eff_ptr); CHKERRXX(ierr);

      foreach_node(n, nodes)
      {
        phi_eff_ptr[n] = MIN( MAX(phi_wall_ptr[n], phi_infc_ptr[n]), ABS(phi_wall_ptr[n]));
      }

      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_infc, &phi_infc_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_eff,  &phi_eff_ptr); CHKERRXX(ierr);

      p4est_t       *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      Vec phi_eff_np1;
      ierr = VecDuplicate(phi_eff, &phi_eff_np1); CHKERRXX(ierr);
      VecCopyGhost(phi_eff, phi_eff_np1);

      bool is_grid_changing = true;
      bool grid_changed = false;
      while (is_grid_changing)
      {
        double *phi_eff_ptr;
        ierr = VecGetArray(phi_eff_np1, &phi_eff_ptr); CHKERRXX(ierr);

        splitting_criteria_tag_t sp(lmin, lmax, lip);
        sp.set_refine_only_inside(refine_only_inside);
        is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

        ierr = VecRestoreArray(phi_eff_np1, &phi_eff_ptr); CHKERRXX(ierr);

        if (is_grid_changing)
        {
          grid_changed = true;
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

          ierr = VecDestroy(phi_eff_np1); CHKERRXX(ierr);
          ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_eff_np1); CHKERRXX(ierr);

          interp.set_input(phi_eff, linear);
          interp.interpolate(phi_eff_np1);
        }
      }

      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);
      ierr = VecDestroy(phi_eff_np1); CHKERRXX(ierr);

      if (grid_changed)
      {
        // interpolate data between grids
        my_p4est_interpolation_nodes_t interp(ngbd);

        double xyz[P4EST_DIM];
        foreach_node(n, nodes_np1)
        {
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          interp.add_point(n, xyz);
        }

        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_wall, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_infc, NULL, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_m, phi_infc, interpolation_between_grids);
        interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_p, phi_infc, interpolation_between_grids);
      }

      // delete old p4est
      p4est_destroy(p4est);       p4est = p4est_np1;
      p4est_ghost_destroy(ghost); ghost = ghost_np1;
      p4est_nodes_destroy(nodes); nodes = nodes_np1;
      hierarchy->update(p4est, ghost);
      ngbd->update(hierarchy, nodes);
    }
//    {
//      Vec phi_eff;
//      double *phi_eff_ptr;

//      ierr = VecDuplicate(phi_infc, &phi_eff); CHKERRXX(ierr);
//      compute_phi_eff(phi_eff, nodes, phi, acn);
//      ierr = VecGetArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);

//      splitting_criteria_tag_t sp(lmin, lmax, lip);
//      sp.set_refine_only_inside(refine_only_inside);

//      p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
//      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
//      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

//      bool is_grid_changing = sp.refine_and_coarsen(p4est_np1, nodes_np1, phi_eff_ptr);

//      ierr = VecRestoreArray(phi_eff, &phi_eff_ptr); CHKERRXX(ierr);
//      ierr = VecDestroy(phi_eff); CHKERRXX(ierr);

//      if (is_grid_changing)
//      {
//        // repartition p4est
//        my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

//        // reset nodes, ghost, and phi
//        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
//        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

//        // interpolate data between grids
//        my_p4est_interpolation_nodes_t interp(ngbd);

//        double xyz[P4EST_DIM];
//        foreach_node(n, nodes_np1)
//        {
//          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
//          interp.add_point(n, xyz);
//        }

//        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_wall, NULL, interpolation_between_grids);
//        interpolate_between_grids(interp, p4est_np1, nodes_np1, phi_infc, NULL, interpolation_between_grids);
//        interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_m, phi_infc, interpolation_between_grids);
//        interpolate_between_grids(interp, p4est_np1, nodes_np1, mu_p, phi_infc, interpolation_between_grids);

//        // delete old p4est
//        p4est_destroy(p4est);       p4est = p4est_np1;
//        p4est_ghost_destroy(ghost); ghost = ghost_np1;
//        p4est_nodes_destroy(nodes); nodes = nodes_np1;
//        hierarchy->update(p4est, ghost);
//        ngbd->update(hierarchy, nodes);
//      }
//    }
    iteration++;

    foreach_dimension(dim)
    {
      ierr = VecDestroy(normal_wall[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }

    ierr = VecDestroy(kappa); CHKERRXX(ierr);

    ierr = VecDestroy(phi_eff); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi_infc); CHKERRXX(ierr);
  ierr = VecDestroy(phi_wall); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  ierr = PetscFClose(mpi.comm(), file_conv); CHKERRXX(ierr);

  return 0;
}
