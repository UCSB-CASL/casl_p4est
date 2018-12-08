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
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_biofilm.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_biofilm.h>
#endif

// TODO:
// - add log variables
// - sort out jump solver

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

#define ADD_OPTION(i, var, description) \
  i == 0 ? cmd.add_option(#var, description) : (void) (var = cmd.get(#var, var));

#define ADD_OPTION2(i, var, name, description) \
  i == 0 ? cmd.add_option(name, description) : (void) (var = cmd.get(name, var));

using namespace std;

// grid parameters
int lmin = 6;
int lmax = 11;
double lip = 2.5;

double xmin = 0, xmax = 1; int nx = 1; bool px = 0;
double ymin = 0, ymax = 1; int ny = 1; bool py = 1;
double zmin = 0, zmax = 1; int nz = 1; bool pz = 1;

// model options
int  velocity_type; /* 0 - using concentration (not implemented), 1 - using pressure (Darcy) */
bool steady_state;  /* assume steady state profile for concentration or not */

double box_size; /* lateral dimensions of simulation box      - m         */
double Df;       /* diffusivity of nutrients in air           - m^2/s     */
double Db;       /* diffusivity of nutrients in biofilm       - m^2/s     */
double Da;       /* diffusivity of nutrients in agar          - m^2/s     */
double sigma;    /* surface tension of air/film interface     - N/m       */
double rho;      /* density of biofilm                        - kg/m^3    */
double lambda;   /* mobility of biofilm                       - m^4/(N*s) */
double A;        /* maximum uptake rate                       - kg/m^3    */
double Kc;       /* half-saturation constant                  - kg/m^3    */
double gam;      /* biofilm yield per nutrient mass           -           */
double C0f;      /* initial nutrient concentration in air     - kg/m^3    */
double C0b;      /* initial nutrient concentration in biofilm - kg/m^3    */
double C0a;      /* initial nutrient concentration in agar    - kg/m^3    */

BoundaryConditionType bc_agar; /* BC type (on computatoinal domain boundary) for nutrients in agar    */
BoundaryConditionType bc_free; /* BC type (on computatoinal domain boundary) for nutrients in biofilm */
BoundaryConditionType bc_biof; /* BC type (on computatoinal domain boundary) for nutrients in air     */

int nb_geometry; /* initial geometry:
                  * 0 - planar
                  * 1 - sphere
                  * 2 - three spheres
                  * 3 - pipe
                  * 4 - planar + bump
                  * 5 - corrugated agar
                  * 6 - porous media (grains)
                  * 7 - porous media (cavities)
                  */
double h_agar;   /* characteristic size of agar      - m */
double h_biof;   /* characteristic size of biofilm   - m */

// specifically for porous media examples
int    grain_num;        /* number of grains or cavities    */
double grain_dispersity; /* grains/cavities size dispersion */
double grain_smoothing;  /* smoothing of initial geometry   */

// time discretization
int    advection_scheme = 0;   // 0 - 1st order, 1 - 2nd order
int    time_scheme      = 0;   // 0 - Euler (1st order), 1 - BDF2 (2nd order)
double cfl_number       = 0.1; //

// solving non-linear diffusion equation
int    iteration_scheme = 1;      // iterative scheme : 0 - simple fixed-point, 1 - linearized fixed-point
int    max_iterations   = 7;      // max iterations
double tolerance        = 1.e-8; // tolerance

// general poisson solver parameters
bool use_sc_scheme         = 1;
bool use_taylor_correction = 1;
int  integration_order     = 2;

// output parameters
bool   save_data  = 1; // save scalar characteristics
bool   save_vtk   = 1; // save spatial data
int    save_type  = 0; // 0 - every dn iterations, 1 - every dl of growth, 2 - every dt of time
int    save_every_dn = 1;
double save_every_dl = 0.01;
double save_every_dt = 0.1;

// simulation run parameters
int    limit_iter   = 10000;
double limit_time   = DBL_MAX;
double limit_length = 1.8;
double init_perturb = 0.00001;

// pre-defined cases
int nb_experiment = 1;
/* 0 - (biofilm + agar + water), planar, transient
 * 1 - (biofilm + agar + water), planar, steady-state
 * 2 - (biofilm + agar), planar with a bump, transient
 * 3 - (biofilm + water), spherical, steady-state
 * 4 - (biofilm + water), pipe, transient
 * 5 - (biofilm + water), porous (cavities), transient
 * 6 - (biofilm + water + agar), porous (grains), transient
 */

void setup_experiment()
{
  // common parameters
  velocity_type = 1;
  box_size = 1;
  sigma = 0.01;
  rho = 1;
  lambda = 1.0;
  A = 100;
  Kc = 100;
  gam = 0.0;
  C0a = 1;
  C0b = 0.01;
  C0f = 1;

  switch(nb_experiment)
  {
    case 0:
      {
        steady_state = 0;

        Da = 0.001;
        Db = 0.01;
        Df = 0.1;

        bc_agar = NEUMANN;
        bc_free = NEUMANN;
        bc_biof = NEUMANN;

        nb_geometry = 0;
        h_agar      = 0.2;
        h_biof      = 0.015;
        break;
      }
    case 1:
      {
        steady_state = 1;

        Da = 0.5e-5;
        Db = 0.5e-5;
        Df = 2.5e-5;

        bc_agar = NEUMANN;
        bc_free = DIRICHLET;
        bc_biof = NEUMANN;

        nb_geometry = 0;
        h_agar      = -0.2;
        h_biof      = 0.115;
        break;
      }
    case 2:
      {
        steady_state = 0;

        Da = 0.001;
        Db = 0.01;
        Df = 0;

        bc_agar = NEUMANN;
        bc_free = NEUMANN;
        bc_biof = NEUMANN;

        nb_geometry = 4;
        h_agar      = 0.4;
        h_biof      = 0.015;
        break;
      }
    case 3:
      {
        steady_state = 0;

        Da = 0;
        Db = 0.01;
        Df = 0.1;

        bc_agar = NEUMANN;
        bc_free = DIRICHLET;
        bc_biof = NEUMANN;

        nb_geometry = 1;
        h_agar      = 0.025;
        h_biof      = 0.015;
        break;
      }
    case 4:
      {
        steady_state = 0;

        Da = 0;
        Db = 0.01;
        Df = 0.1;

        bc_agar = NEUMANN;
        bc_free = DIRICHLET;
        bc_biof = NEUMANN;

        nb_geometry = 3;
        h_agar      = 0.4;
        h_biof      = 0.015;
        break;
      }
    case 5:
      {
        steady_state = 0;

        Da = 0;
        Db = 0.01;
        Df = 0.1;

        bc_agar = NEUMANN;
        bc_free = NEUMANN;
        bc_biof = NEUMANN;

        nb_geometry = 7;
        h_agar      = 0.03;
        h_biof      = 0.015;

        grain_num        = 100;
        grain_dispersity = 2;
        grain_smoothing  = 0.01;
        break;
      }
    case 6:
      {
        steady_state = 0;

        Da = 0.001;
        Db = 0.01;
        Df = 0.1;

        bc_agar = NEUMANN;
        bc_free = NEUMANN;
        bc_biof = NEUMANN;

        nb_geometry = 6;
        h_agar      = 0.011;
        h_biof      = 0.015;

        grain_num        = 50;
        grain_dispersity = 1.5;
        grain_smoothing  = 0.01;
        break;
      }
  }
}


void set_periodicity()
{
  switch (nb_geometry)
  {
    case 0: px = 0; py = 1; pz = 1; break;
    case 1: px = 0; py = 0; pz = 0; break;
    case 2: px = 0; py = 0; pz = 0; break;
    case 3: px = 0; py = 0; pz = 1; break;
    case 4: px = 0; py = 1; pz = 1; break;
    case 5: px = 0; py = 1; pz = 1; break;
    case 6: px = 1; py = 1; pz = 1; break;
    case 7: px = 1; py = 1; pz = 1; break;
    default: throw std::invalid_argument("[ERROR]: Wrong type of initial geometry");
  }
}

#ifdef P4_TO_P8
class phi_agar_cf_t : public CF_DIM {
public:
  double operator()(double x, double y, double z) const { return -(x-0.1); }
} phi_agar_cf;

class phi_free_cf_t : public CF_3 {
public:
  double operator()(double x, double y, double z) const { return  (x-0.2); }
} phi_free_cf;

class phi_biof_cf_t : public CF_3 {
public:
  double operator()(double x, double y, double z) const { return MAX(phi_agar_cf(x,y,z), phi_free_cf(x,y,z)); }
} phi_biof_cf;

class bc_wall_type_t : public WallBC3D {
public:
  BoundaryConditionType operator()( double, double, double ) const
  {
    return NEUMANN;
  }
} bc_wall_type;

class bc_wall_value_t : public CF_3 {
public:
  double operator()(double , double, double ) const { return 0; }
} bc_wall_value;

class initial_concentration_free_t : public CF_3 {
public:
  double operator()(double , double, double ) const {
    return C0f;
  }
} initial_concentration_free;

class initial_concentration_agar_t : public CF_3 {
public:
  double operator()(double , double, double ) const {
    return C0a;
  }
} initial_concentration_agar;

class initial_concentration_biof_t : public CF_3 {
public:
  double operator()(double , double, double ) const {
    return C0b;
  }
} initial_concentration_biof;
#else
class phi_agar_cf_t : public CF_2 {
public:
  double operator()(double x, double y) const
  {
    switch (nb_geometry)
    {
      case 0: return -(x-h_agar);
      case 1: return h_agar - sqrt( SQR(x-0.5*(xmax+xmin)) + SQR(y-0.5*(ymax+ymin)));
      case 2: return MAX(h_agar - sqrt( SQR(x-(xmin + .3*(xmax-xmin))) + SQR(y-(ymin + .5*(ymax-ymin)))),
                         h_agar - sqrt( SQR(x-(xmin + .5*(xmax-xmin))) + SQR(y-(ymin + .6*(ymax-ymin)))),
                         h_agar - sqrt( SQR(x-(xmin + .6*(xmax-xmin))) + SQR(y-(ymin + .4*(ymax-ymin)))));
      case 3: return -(h_agar - sqrt(SQR(x-.5*(xmax+xmin)) + SQR(y-.5*(ymax+ymin))));
      case 4: return -(x-h_agar);
      case 5: return -(x-h_agar) + 0.02*cos(2.*PI*10*(y-ymin)/(ymax-ymin));
      case 6:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num; ++i)
          {
            double R = h_agar * (1. + grain_dispersity*(((double) rand() / (double) RAND_MAX)));
            double X = xmin + ((double) rand() / (double) RAND_MAX) *(xmax-xmin);
            double Y = ymin + ((double) rand() / (double) RAND_MAX) *(ymax-ymin);

            int nx = round((X-x)/(xmax-xmin));
            int ny = round((Y-y)/(ymax-ymin));

            double dist = R - sqrt( SQR(x-X + nx*(xmax-xmin)) + SQR(y-Y + ny*(ymax-ymin)) );

            sum = MIN(sum, -dist);
          }

          return -sum;
        }
      case 7:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num; ++i)
          {
            double R = h_agar * (1. + grain_dispersity*(((double) rand() / (double) RAND_MAX)));
            double X = xmin + ((double) rand() / (double) RAND_MAX) *(xmax-xmin);
            double Y = ymin + ((double) rand() / (double) RAND_MAX) *(ymax-ymin);

            int nx = round((X-x)/(xmax-xmin));
            int ny = round((Y-y)/(ymax-ymin));

            double dist = R - sqrt( SQR(x-X + nx*(xmax-xmin)) + SQR(y-Y + ny*(ymax-ymin)) );

            sum = MIN(sum, -dist);
          }

          return sum;
        }
      default: throw std::invalid_argument("[ERROR]: Wrong type of initial geometry");
    }
  }
} phi_agar_cf;

class phi_free_cf_t : public CF_2 {
public:
  double operator()(double x, double y) const
  {
    switch (nb_geometry)
    {
      case 0: return (x- MAX(0., h_agar) - h_biof);
      case 1: return -(MAX(0., h_agar) + h_biof - sqrt( SQR(x-0.5*(xmax+xmin)) + SQR(y-0.5*(ymax+ymin))));
      case 2: return -MAX(MAX(0., h_agar) + h_biof - sqrt( SQR(x-(xmin + .3*(xmax-xmin))) + SQR(y-(ymin + .5*(ymax-ymin)))),
                          MAX(0., h_agar) + h_biof - sqrt( SQR(x-(xmin + .5*(xmax-xmin))) + SQR(y-(ymin + .6*(ymax-ymin)))),
                          MAX(0., h_agar) + h_biof - sqrt( SQR(x-(xmin + .6*(xmax-xmin))) + SQR(y-(ymin + .4*(ymax-ymin)))));
      case 3: return ((MAX(0., h_agar) - h_biof) - sqrt(SQR(x-.5*(xmax+xmin)) + SQR(y-.5*(ymax+ymin))));
      case 4:
        {
          double plane = (x-(MAX(0., h_agar) + h_biof));
          double bump = -(.1*(xmax-xmin) - sqrt( SQR(x-(MAX(0., h_agar) + h_biof)) + SQR(y-0.5*(ymax+ymin))));
          return .5*(plane+bump - sqrt(SQR(plane-bump) + .001*(xmax-xmin)));
        }
      case 5: return (x-(MAX(0., h_agar) + h_biof));
      case 6:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num; ++i)
          {
            double R = (MAX(0., h_agar) + h_biof) * (1. + grain_dispersity*(((double) rand() / (double) RAND_MAX)));
            double X = xmin + ((double) rand() / (double) RAND_MAX) *(xmax-xmin);
            double Y = ymin + ((double) rand() / (double) RAND_MAX) *(ymax-ymin);

            int nx = round((X-x)/(xmax-xmin));
            int ny = round((Y-y)/(ymax-ymin));

            double dist = R - sqrt( SQR(x-X + nx*(xmax-xmin)) + SQR(y-Y + ny*(ymax-ymin)) );

            sum = MIN(sum, -dist);
          }

          return sum;
        }
      case 7:
        {
          srand(0);

          double sum = 10;
          for (int i = 0; i < grain_num; ++i)
          {
            double R = (MAX(0., h_agar) - h_biof) * (1. + grain_dispersity*(((double) rand() / (double) RAND_MAX)));
            double X = xmin + ((double) rand() / (double) RAND_MAX) *(xmax-xmin);
            double Y = ymin + ((double) rand() / (double) RAND_MAX) *(ymax-ymin);

            int nx = round((X-x)/(xmax-xmin));
            int ny = round((Y-y)/(ymax-ymin));

            double dist = R - sqrt( SQR(x-X + nx*(xmax-xmin)) + SQR(y-Y + ny*(ymax-ymin)) );

            sum = MIN(sum, -dist);
          }

          return -sum;
        }
      default: throw std::invalid_argument("[ERROR]: Wrong type of initial geometry");
    }
  }
} phi_free_cf;

class phi_biof_cf_t : public CF_2 {
public:
  double operator()(double x, double y) const { return MAX(phi_agar_cf(x,y), phi_free_cf(x,y)); }
} phi_biof_cf;

class bc_wall_type_t : public WallBC2D {
public:
  BoundaryConditionType operator()(double x, double y) const
  {
    double pa = phi_agar_cf(x,y);
    double pf = phi_free_cf(x,y);
    if (pa < 0 && pf < 0) return bc_biof;
    else if (pa > 0 && pf > 0) throw;
    else if (pa > 0)      return bc_agar;
    else if (pf > 0)      return bc_free;
  }
} bc_wall_type;

class bc_wall_value_t : public CF_2 {
public:
  double operator()(double x, double y) const
  {
    double pa = phi_agar_cf(x,y);
    double pf = phi_free_cf(x,y);
    if (pa < 0 && pf < 0) return bc_biof == NEUMANN ? 0 : C0b;
    else if (pa > 0 && pf > 0) throw;
    else if (pa > 0)      return bc_agar == NEUMANN ? 0 : C0a;
    else if (pf > 0)      return bc_free == NEUMANN ? 0 : C0f;
  }
} bc_wall_value;

class initial_concentration_free_t : public CF_2 {
public:
  double operator()(double , double ) const {
    return C0f;
  }
} initial_concentration_free;

class initial_concentration_agar_t : public CF_2 {
public:
  double operator()(double , double ) const {
    return C0a;
  }
} initial_concentration_agar;

class initial_concentration_biof_t : public CF_2 {
public:
  double operator()(double , double ) const {
    return C0b;
  }
} initial_concentration_biof;
#endif

class f_cf_t : public CF_1 {
public:
  double operator()(double c) const { return A*c/(Kc+c); }
} f_cf;

class fc_cf_t : public CF_1 {
public:
  double operator()(double c) const { return A/(Kc+c) - A*c/pow(Kc+c, 2.); }
} fc_cf;

#ifdef P4_TO_P8
class zero_cf_t : public CF_3 {
public:
  double operator()(double, double, double) const { return 0; }
} zero_cf;
#else
class zero_cf_t : public CF_2{
public:
  double operator()(double, double) const { return 0; }
} zero_cf;
#endif

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;

  for (short i = 0; i < 2; ++i)
  {
    // grid parameters
    ADD_OPTION(i, lmin, "min level of the tree");
    ADD_OPTION(i, lmax, "max level of the tree");
    ADD_OPTION(i, lip,  "Lipschitz constant");

    ADD_OPTION(i, nx, "number of blox in x-dimension");
    ADD_OPTION(i, ny, "number of blox in y-dimension");
#ifdef P4_TO_P8
    ADD_OPTION(i, nz, "number of blox in z-dimension");
#endif

    ADD_OPTION(i, xmin, "xmin"); ADD_OPTION(i, xmax, "xmax");
    ADD_OPTION(i, ymin, "ymin"); ADD_OPTION(i, ymax, "ymax");
#ifdef P4_TO_P8
    ADD_OPTION(i, zmin, "zmin"); ADD_OPTION(i, zmax, "zmax");
#endif

    // TODO: add rest of parameters

    if (i == 0) cmd.parse(argc, argv);
  }
  setup_experiment();
  set_periodicity();

  // scale computational box
  double scaling = 1/box_size;

  rho /= (scaling*scaling*scaling);

  Da *= (scaling*scaling);
  Db *= (scaling*scaling);
  Df *= (scaling*scaling);

  sigma /= (scaling);

  A /= (scaling*scaling*scaling);
  Kc /= (scaling*scaling*scaling);

  C0a /= (scaling*scaling*scaling);
  C0b /= (scaling*scaling*scaling);
  C0f /= (scaling*scaling*scaling);

  lambda *= (scaling*scaling*scaling*scaling);

  parStopWatch w1;
  w1.start("total time");

  /* create the p4est */
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  double xyz_min [] = { xmin, ymin, zmin };
  double xyz_max [] = { xmax, ymax, zmax };
  int periodic   [] = { px, py, pz };
  int nxyz       [] = { nx, ny, nz };
#else
  double xyz_min [] = { xmin, ymin };
  double xyz_max [] = { xmax, ymax };
  int periodic   [] = { px, py };
  int nxyz       [] = { nx, ny };
#endif

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nxyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_biof_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin_tree = p4est->connectivity->vertices[3*vm + 0];
  double ymin_tree = p4est->connectivity->vertices[3*vm + 1];
  double xmax_tree = p4est->connectivity->vertices[3*vp + 0];
  double ymax_tree = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax_tree-xmin_tree) / pow(2., (double) data.max_lvl);
  double dy = (ymax_tree-ymin_tree) / pow(2., (double) data.max_lvl);
#ifdef P4_TO_P8
  double zmin_tree = p4est->connectivity->vertices[3*vm + 2];
  double zmax_tree = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax_tree-zmin_tree) / pow(2.,(double) data.max_lvl);
#endif

//  double dt_max = MIN( 1.e8*dx*dx/MAX(Da, Db, Df), 10000./A);
  double dt_max = 1.5e-8*pow(2., 3.*(11.-lmax));

  /* initial geometry */
  Vec phi_free; ierr = VecCreateGhostNodes(p4est, nodes, &phi_free); CHKERRXX(ierr);
  Vec phi_agar; ierr = VecCreateGhostNodes(p4est, nodes, &phi_agar); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_free_cf, phi_free);
  sample_cf_on_nodes(p4est, nodes, phi_agar_cf, phi_agar);

  // initial air-biofilm interface perturbation
  {
    double *phi_free_ptr;
    ierr = VecGetArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

    srand(mpi.rank());

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      phi_free_ptr[n] += init_perturb*dx*(double)(rand()%1000)/1000.;
    }

    ierr = VecRestoreArray(phi_free, &phi_free_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(phi_free, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (phi_free, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if (nb_geometry == 6)
  {
    shift_ghosted_vec(phi_free, -grain_smoothing);
    shift_ghosted_vec(phi_agar, +grain_smoothing);
  }

  if (nb_geometry == 7)
  {
    shift_ghosted_vec(phi_free, +grain_smoothing);
    shift_ghosted_vec(phi_agar, -grain_smoothing);
  }

  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi_free, 100);
  ls.reinitialize_1st_order_time_2nd_order_space(phi_agar, 100);

  if (nb_geometry == 6)
  {
    shift_ghosted_vec(phi_free, +grain_smoothing);
    shift_ghosted_vec(phi_agar, -grain_smoothing);
    copy_ghosted_vec(phi_free, phi_agar);
    invert_phi(nodes, phi_agar);
    shift_ghosted_vec(phi_agar, -h_biof);
  }

  if (nb_geometry == 7)
  {
    shift_ghosted_vec(phi_free, -grain_smoothing);
    shift_ghosted_vec(phi_agar, +grain_smoothing);
    if (h_agar > 0)
    {
      copy_ghosted_vec(phi_agar, phi_free);
      invert_phi(nodes, phi_free);
      shift_ghosted_vec(phi_free, -h_biof);
    } else {
      set_ghosted_vec(phi_agar, -1);
    }
  }

  /* initial concentration */
  Vec Ca; ierr = VecDuplicate(phi_free, &Ca); CHKERRXX(ierr);
  Vec Cb; ierr = VecDuplicate(phi_free, &Cb); CHKERRXX(ierr);
  Vec Cf; ierr = VecDuplicate(phi_free, &Cf); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, initial_concentration_agar, Ca);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_biof, Cb);
  sample_cf_on_nodes(p4est, nodes, initial_concentration_free, Cf);

  /* initialize the solver */
  my_p4est_biofilm_t biofilm_solver(ngbd);

  // model parameters
  biofilm_solver.set_velocity_type(velocity_type);
  biofilm_solver.set_parameters   (Da, Db, Df, sigma, rho, lambda, gam, scaling);
  biofilm_solver.set_kinetics     (f_cf, fc_cf);
  biofilm_solver.set_bc           (bc_wall_type, bc_wall_value);
  biofilm_solver.set_steady_state (steady_state);

  // time discretization parameters
  biofilm_solver.set_advection_scheme(advection_scheme);
  biofilm_solver.set_time_scheme     (time_scheme);
  biofilm_solver.set_dt_max          (dt_max);
  biofilm_solver.set_cfl             (cfl_number);

  // parameters for solving non-linear equation
  biofilm_solver.set_iteration_scheme(iteration_scheme);
  biofilm_solver.set_max_iterations  (max_iterations);
  biofilm_solver.set_tolerance       (tolerance);

  // general poisson solver parameter
  biofilm_solver.set_use_sc_scheme(use_sc_scheme);
  biofilm_solver.set_use_taylor_correction(use_taylor_correction);
  biofilm_solver.set_integration_order(integration_order);

  // initial geometry and concentrations
  biofilm_solver.set_phi(phi_free, phi_agar);
  biofilm_solver.set_concentration(Ca, Cb, Cf);

  ierr = VecDestroy(phi_free); CHKERRXX(ierr);
  ierr = VecDestroy(phi_agar); CHKERRXX(ierr);

  ierr = VecDestroy(Ca); CHKERRXX(ierr);
  ierr = VecDestroy(Cb); CHKERRXX(ierr);
  ierr = VecDestroy(Cf); CHKERRXX(ierr);

  // loop over time
  double tn = 0;
  int iteration = 0;

  FILE *fich;
  char name[10000];

  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
#ifdef P4_TO_P8
  sprintf(name, "%s/data_%dx%dx%d_box_%g_level_%d-%d.dat", out_dir, nxyz[0], nxyz[1], nxyz[2], box_size, lmin, lmax);
#else
  sprintf(name, "%s/data_%dx%d_box_%g_level_%d-%d.dat", out_dir, nxyz[0], nxyz[1], box_size, lmin, lmax);
#endif

  if(save_data)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "time average_interface_velocity max_interface_velocity interface_length biofilm_area time_elapsed iteration\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  bool keep_going = true;

  int vtk_idx = 0;

  double total_growth = 0;
  double base = 0.1;

  biofilm_solver.update_grid();
  while(keep_going)
  {
    if (tn + biofilm_solver.get_dt() > limit_time) { biofilm_solver.set_dt_max(limit_time-tn); keep_going = false; }

    tn += biofilm_solver.get_dt();

    biofilm_solver.one_step();

    // compute how far the air-biofilm interface has advanced
    {
      total_growth = base;

      p4est = biofilm_solver.get_p4est();
      nodes = biofilm_solver.get_nodes();
      phi_free = biofilm_solver.get_phi_free();

      const double *phi_free_ptr;
      ierr = VecGetArrayRead(phi_free, &phi_free_ptr); CHKERRXX(ierr);
      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if (phi_free_ptr[n] < 0)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);
          total_growth = MAX(total_growth, xyz[0]);
        }
      }
      ierr = VecRestoreArrayRead(phi_free, &phi_free_ptr); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &total_growth, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      total_growth -= base;
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration %d, growth %e, time %e\n", iteration, total_growth, tn); CHKERRXX(ierr);

    // determine to save or not
    bool save_now =
        (save_type == 0 && iteration    >= vtk_idx*save_every_dn) ||
        (save_type == 1 && total_growth >= vtk_idx*save_every_dl) ||
        (save_type == 2 && tn           >= vtk_idx*save_every_dt);

    // save velocity, area of interface and volume of biofilm
    if(save_data && save_now)
    {
      p4est = biofilm_solver.get_p4est();
      nodes = biofilm_solver.get_nodes();
      phi_free = biofilm_solver.get_phi_free();
      Vec vn = biofilm_solver.get_vn();
      Vec phi_biof = biofilm_solver.get_phi_biof();

      Vec ones;
      ierr = VecDuplicate(phi_free, &ones); CHKERRXX(ierr);
      set_ghosted_vec(ones, 1);

      // calculate the length of the interface and solid phase area
      double interface_area = integrate_over_interface(p4est, nodes, phi_free, ones);
      double biofilm_volume = integrate_over_negative_domain(p4est, nodes, phi_biof, ones);

#ifdef P4_TO_P8
      interface_area /= (scaling*scaling);
      biofilm_volume /= (scaling*scaling*scaling);
#else
      interface_area /= (scaling);
      biofilm_volume /= (scaling*scaling);
#endif

      double avg_velo = integrate_over_interface(p4est, nodes, phi_free, vn) / interface_area;

      ierr = VecDestroy(ones); CHKERRXX(ierr);

      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      double time_elapsed = w1.read_duration_current();

      PetscFPrintf(mpi.comm(), fich, "%e %e %e %e %e %e %d\n", tn, avg_velo/scaling, biofilm_solver.get_vn_max()/scaling, interface_area, biofilm_volume, time_elapsed, iteration);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
      ierr = PetscPrintf(mpi.comm(), "saved data in %s\n", name); CHKERRXX(ierr);
    }

    keep_going = keep_going && (iteration < limit_iter) && (total_growth < limit_length);

    // save field data
    if(save_vtk && save_now)
    {
      biofilm_solver.save_VTK(vtk_idx);
    }

    biofilm_solver.compute_dt();
    biofilm_solver.update_grid();

    iteration++;

    if (save_now) vtk_idx++;
  }

  /* destroy the p4est and its connectivity structure */
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}
