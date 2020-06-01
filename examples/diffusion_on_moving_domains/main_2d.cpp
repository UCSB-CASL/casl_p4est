/*
 * Title: diffusion_on_moving_domains
 * Description:
 * Author: dbochkov
 * Date Created: 04-16-2020
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_level_set.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_level_set.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>
#include <iomanip>
#include <mpich/mpi.h>

using namespace std;


param_list_t pl;

param_t<int> num_trees_x(pl, 1, "num_trees_x", "");
param_t<int> num_trees_y(pl, 1, "num_trees_y", "");
param_t<int> num_trees_z(pl, 1, "num_trees_z", "");

param_t<double> xmin (pl, -1, "xmin", "");
param_t<double> xmax (pl,  1, "xmax", "");

param_t<double> ymin (pl, -1, "xmin", "");
param_t<double> ymax (pl,  1, "xmax", "");
#ifdef P4_TO_P8
param_t<double> zmin (pl, -1, "xmin", "");
param_t<double> zmax (pl,  1, "xmax", "");
#endif

param_t<int>    lmin (pl, 5, "lmin", "");
param_t<int>    lmax (pl, 5, "lmax", "");
param_t<double> band (pl, 2, "band", "");
param_t<double> lip  (pl, 1.0, "lip", "");

param_t<int>    num_splits (pl, 5, "num_splits", "");
param_t<int>    add_splits (pl, 3, "add_splits", "");

param_t<double> diffusivity  (pl, 0.1, "diffusivity", "");
param_t<double> time_range   (pl, 1, "time_range", "");
param_t<int>    test_solution(pl, 0, "test_solution", "");
param_t<int>    test_geometry(pl, 5, "test_geometry", "");
param_t<int>    motion_type  (pl, 3, "motion_type", "0 - steady, "
                                                    "1 - parallel to x, "
                                                    "2 - parallel to y, "
                                                    "3 - diagonally, "
                                                    "4 - in a circle");

param_t<double> num_spins     (pl, 0.5, "num_spins", "");
param_t<double> radii_ratio   (pl, 0.7, "radii_ratio", "");

param_t<double> cfl           (pl, 0.8, "cfl", "");
param_t<int>    order_in_time (pl, 2, "order_in_time", "");

param_t<double> xc    (pl, 0.01, "xc", "");
param_t<double> yc    (pl, 0.51, "yc", "");
param_t<double> angle (pl, 0.00, "angle", "");
param_t<double> scale (pl, 0.25, "scale", "");

param_t<bool>   save_vtk         (pl, 0, "save_vtk", "");
param_t<int>    save_vtk_freq    (pl, 1, "save_vtk_freq", "");
param_t<bool>   save_convergence (pl, 1, "save_convergence", "");
param_t<bool>   save_parameters  (pl, 1, "save_parameters", "");

param_t<int>    num_extension_iters (pl, 50, "num_extension_iters", "");
param_t<int>    extension_region    (pl, 0, "extension_region", "0 - phi > 0, "
                                                                "1 - mask > 0");
param_t<int>    extension_order     (pl, 2, "extension_order", "");
param_t<int>    extension_type      (pl, 2, "extension_type", "0 - Analytic, "
                                                              "1 - Aslam's method, "
                                                              "2 - Cartesian derivatives based");

param_t<BoundaryConditionType> bc_type (pl, NEUMANN, "bc_type", "");
param_t<BoundaryConditionType> wc_type (pl, NEUMANN, "bc_type", "");

int num_phis()
{
  switch (test_geometry.val) {
    case 0: return 0; // no geometry
    case 1: return 1; // sphere
    case 2: return 1; // star-shaped
    case 3: return 2; // difference of two spheres
    case 4: return 2; // intersection of two spheres
    case 5: return 2; // union of two spheres
    case 6: return 2; // rectangle
    default: throw;
  }
}

int num_time_layers() { return order_in_time.val; }

mls_opn_t phi_opn(int idx)
{
  switch (test_geometry.val) {
    case 0:
      switch (idx) {
        default: throw;
      }
    case 1:
      switch (idx) {
        case 0: return MLS_INT;
        default: throw;
      }
    case 2:
      switch (idx) {
        case 0: return MLS_INT;
        default: throw;
      }
    case 3:
      switch (idx) {
        case 0: return MLS_INT;
        case 1: return MLS_ADD;
        default: throw;
      }
    case 4:
      switch (idx) {
        case 0: return MLS_INT;
        case 1: return MLS_INT;
        default: throw;
      }
    case 5:
      switch (idx) {
        case 0: return MLS_INT;
        case 1: return MLS_INT;
        default: throw;
      }
    case 6:
      switch (idx) {
        case 0: return MLS_INT;
        case 1: return MLS_ADD;
        default: throw;
      }
    default: throw;
  }
}

class phi_cf_t : public CF_DIM
{
public:
  int             idx;
  cf_value_type_t what;
  phi_cf_t(cf_value_type_t what, int idx) : idx(idx), what(what) {}

  double phi_value(int idx_, cf_value_type_t what_, DIM(double x, double y, double z)) const
  {
    switch (test_geometry.val) {
      case 0:
        switch (idx_) {
          default: throw;
        }
      case 1: {
        static radial_shaped_domain_t circle(0.999, DIM(0,0,0), -1);
        switch (idx_) {
          case 0:
            switch (what_) {
              case VAL: return circle.phi(DIM(x,y,z));
              case DDX: return circle.phi_x(DIM(x,y,z));
              case DDY: return circle.phi_y(DIM(x,y,z));
              default: throw;
            }
          default: throw;
        }
      }
      case 2: {
        static radial_shaped_domain_t flower(0.999, DIM(0,0,0), -1, 3, 0.3, 0);
        switch (idx_) {
          case 0:
            switch (what_) {
              case VAL: return flower.phi(DIM(x,y,z));
              case DDX: return flower.phi_x(DIM(x,y,z));
              case DDY: return flower.phi_y(DIM(x,y,z));
              default: throw;
            }
          default: throw;
        }
      }
      case 3: {
        static double a = 0.999/sqrt(1.+SQR(radii_ratio.val));
        static double b = 0.999*SQR(radii_ratio.val)/sqrt(1.+SQR(radii_ratio.val));
        static radial_shaped_domain_t circle_one(0.999, DIM(0,0,0), -1);
        static radial_shaped_domain_t circle_two(0.999*radii_ratio.val, DIM(-(a+b),0,0), 1);
        switch (idx_) {
          case 0:
            switch (what_) {
              case VAL: return circle_one.phi(DIM(x,y,z));
              case DDX: return circle_one.phi_x(DIM(x,y,z));
              case DDY: return circle_one.phi_y(DIM(x,y,z));
              default: throw;
            }
          case 1:
            switch (what_) {
              case VAL: return circle_two.phi(DIM(x,y,z));
              case DDX: return circle_two.phi_x(DIM(x,y,z));
              case DDY: return circle_two.phi_y(DIM(x,y,z));
              default: throw;
            }
          default: throw;
        }
      }
      case 4: {
        static double d = 0.999*sqrt(1.+SQR(radii_ratio.val));
        static radial_shaped_domain_t circle_one(0.999, DIM(-.5*0.999*(1-radii_ratio.val) - .5*d,0,0), 1);
        static radial_shaped_domain_t circle_two(0.999*radii_ratio.val, DIM(-.5*0.999*(1-radii_ratio.val) + .5*d,0,0), 1);
        switch (idx_) {
          case 0:
            switch (what_) {
              case VAL: return circle_one.phi(DIM(x,y,z));
              case DDX: return circle_one.phi_x(DIM(x,y,z));
              case DDY: return circle_one.phi_y(DIM(x,y,z));
              default: throw;
            }
          case 1:
            switch (what_) {
              case VAL: return circle_two.phi(DIM(x,y,z));
              case DDX: return circle_two.phi_x(DIM(x,y,z));
              case DDY: return circle_two.phi_y(DIM(x,y,z));
              default: throw;
            }
          default: throw;
        }
      }
      case 5: {
        static double a = 0.999/sqrt(1.+SQR(radii_ratio.val));
        static double b = 0.999*SQR(radii_ratio.val)/sqrt(1.+SQR(radii_ratio.val));
        static radial_shaped_domain_t circle_one(0.999, DIM(-.5*(a+b),0,0), -1);
        static radial_shaped_domain_t circle_two(0.999*radii_ratio.val, DIM(.5*(a+b),0,0), -1);
        switch (idx_) {
          case 0:
            switch (what_) {
              case VAL: return circle_one.phi(DIM(x,y,z));
              case DDX: return circle_one.phi_x(DIM(x,y,z));
              case DDY: return circle_one.phi_y(DIM(x,y,z));
              default: throw;
            }
          case 1:
            switch (what_) {
              case VAL: return circle_two.phi(DIM(x,y,z));
              case DDX: return circle_two.phi_x(DIM(x,y,z));
              case DDY: return circle_two.phi_y(DIM(x,y,z));
              default: throw;
            }
          default: throw;
        }
      }
      case 6: {
        switch (idx_) {
          case 0:
            switch (what_) {
              case VAL: return 0.999-fabs(x);
              case DDX: return -(x)/(fabs(x)+EPS);
              case DDY: return 0;
              default: throw;
            }
          case 1:
            switch (what_) {
              case VAL: return 0.999*radii_ratio.val-fabs(y);
              case DDX: return 0;
              case DDY: return -(y)/(fabs(y)+EPS);
              default: throw;
            }
          default: throw;
        }
      }
    }
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    double cos_a = cos(angle.val);
    double sin_a = sin(angle.val);
    double X = ((x-xc.val)*cos_a - (y-yc.val)*sin_a)/scale.val;
    double Y = ((x-xc.val)*sin_a + (y-yc.val)*cos_a)/scale.val;
    switch (what) {
      case VAL: return scale.val*  phi_value(idx, VAL, DIM(X,Y,z));
      case DDX: return scale.val*( phi_value(idx, DDX, DIM(X,Y,z))*cos_a + phi_value(idx, DDY, DIM(X,Y,z))*sin_a);
      case DDY: return scale.val*(-phi_value(idx, DDX, DIM(X,Y,z))*sin_a + phi_value(idx, DDY, DIM(X,Y,z))*cos_a);
      default: throw;
    }
  }
};

phi_cf_t phi_cf_all  [] = { phi_cf_t(VAL, 0),  phi_cf_t(VAL, 1) };
phi_cf_t phi_x_cf_all[] = { phi_cf_t(DDX, 0),  phi_cf_t(DDX, 1) };
phi_cf_t phi_y_cf_all[] = { phi_cf_t(DDY, 0),  phi_cf_t(DDY, 1) };

// TEST SOLUTIONS
double aij[4][4] = {{-0.5, -0.1, -0.5, 0.6}, {-0.6, -0.5, -0.1, -0.1}, {0.2, -0.2, -0.2, 0.4}, {0.1, -0.9, 0.8, 0.4}};

double omx(double i) { return PI*i/(xmax.val-xmin.val); }
double omy(double j) { return PI*j/(ymax.val-ymin.val); }
double sigma(double i, double j) { return diffusivity.val*(SQR(omx(i))+SQR(omy(j))); }

class sol_cf_t : public CF_DIM
{
public:
  cf_value_type_t what;
  sol_cf_t(cf_value_type_t what) : what(what) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (test_solution.val) {
      case 0:
      {
        double result = 0;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            switch (what) {
#ifdef P4_TO_P8
#else
              case VAL: result += aij[i][j]*cos(omx(i)*(x-xmin.val))*cos(omy(j)*(y-ymin.val))*exp(-sigma(i,j)*t); break;
              case DDT: result += 0; break;
              case DDX: result += aij[i][j]*sin(omx(i)*(x-xmin.val))*cos(omy(j)*(y-ymin.val))*exp(-sigma(i,j)*t)*(-omx(i)); break;
              case DDY: result += aij[i][j]*cos(omx(i)*(x-xmin.val))*sin(omy(j)*(y-ymin.val))*exp(-sigma(i,j)*t)*(-omy(j)); break;
              case LAP: result += 0; break;
#endif
              default: throw;
            }
          }
        }
        return result;
      }
      default:
        throw std::invalid_argument("Unknown test function\n");
    }
  }
};

sol_cf_t sol_cf(VAL);
sol_cf_t sol_l_cf(LAP);
sol_cf_t sol_t_cf(DDT);
sol_cf_t sol_x_cf(DDX);
sol_cf_t sol_y_cf(DDY);
#ifdef P4_TO_P8
sol_cf_t sol_z_cf(DDZ);
#endif

// RHS
struct: CF_DIM
{
  double operator()(DIM(double x, double y, double z)) const {
    return sol_t_cf(DIM(x,y,z)) - diffusivity.val*sol_l_cf(DIM(x,y,z));
  }
} rhs_cf;

// BC VALUES
class bc_value_cf_t : public CF_DIM
{
  int idx;
public:
  bc_value_cf_t(int idx) : idx(idx) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (bc_type.val) {
      case DIRICHLET: return sol_cf(DIM(x,y,z));
      case NEUMANN: {
          double DIM( nx = phi_x_cf_all[idx](DIM(x,y,z)),
                      ny = phi_y_cf_all[idx](DIM(x,y,z)),
                      nz = phi_z_cf_all[idx](DIM(x,y,z)) );

          return diffusivity.val*SUMD(nx*sol_x_cf(DIM(x,y,z)),
                                      ny*sol_y_cf(DIM(x,y,z)),
                                      nz*sol_z_cf(DIM(x,y,z)))/ABSD(nx, ny, nz);
        }
      default: throw;
    }
  }
};

bc_value_cf_t bc_value_cf_all[] = { bc_value_cf_t(0),
                                    bc_value_cf_t(1)};

// BC VALUES ON WALLS
class wc_value_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (wc_type.val) {
      case DIRICHLET: return sol_cf(DIM(x,y,z));
      case NEUMANN:
        if (x == xmin.val) return -sol_x_cf(DIM(x,y,z));
        if (x == xmax.val) return  sol_x_cf(DIM(x,y,z));

        if (y == ymin.val) return -sol_y_cf(DIM(x,y,z));
        if (y == ymax.val) return  sol_y_cf(DIM(x,y,z));
#ifdef P4_TO_P8
        if (z == zmin.val) return -sol_z_cf(DIM(x,y,z));
        if (z == zmax.val) return  sol_z_cf(DIM(x,y,z));
#endif
        throw;
      default: throw;
    }
  }
} wc_value_cf;

// MOTION
double trajectory(cf_value_type_t what, double t)
{
  switch (motion_type.val) {
    case 0:
      switch (what) {
        case _X_: return 0.02;
        case _Y_: return 0.01;
        case V_X: return 0;
        case V_Y: return 0;
        default: throw;
      }
    case 1:
      switch (what) {
        case _X_: return -0.51 + t/time_range.val;
        case _Y_: return  0.01;
        case V_X: return  1./time_range.val;
        case V_Y: return  0;
        default: throw;
      }
    case 2:
      switch (what) {
        case _X_: return  0.01;
        case _Y_: return -0.51 + t/time_range.val;
        case V_X: return  0;
        case V_Y: return  1./time_range.val;
        default: throw;
      }
    case 3:
      switch (what) {
        case _X_: return -0.51 + t/time_range.val;
        case _Y_: return  0.52 - t/time_range.val;
        case V_X: return  1./time_range.val;
        case V_Y: return -1./time_range.val;
        default: throw;
      }
    case 4:
      switch (what) {
        case _X_: return  0.01 + 0.5*sin(2.*PI*t/time_range.val);
        case _Y_: return  0.02 + 0.5*cos(2.*PI*t/time_range.val);
        case V_X: return  PI/time_range.val*cos(2.*PI*t/time_range.val);
        case V_Y: return -PI/time_range.val*sin(2.*PI*t/time_range.val);
        default: throw;
      }
    default: throw;
  }
}

double spin(cf_value_type_t what, double t)
{
  switch (what) {
    case VAL: return num_spins.val*2.*PI*t /time_range.val;
    case DDT: return num_spins.val*2.*PI*1./time_range.val;
    default: throw;
  }

}


int main(int argc, char** argv)
{
  PetscErrorCode ierr;

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // parse command line arguments
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  // stopwatch
  parStopWatch w;
  w.start("Running example: diffusion_on_moving_domains");

  // prepare output

  // prepare stuff for output
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
  else
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir;
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR directory");
  }

  FILE *fich;
  char name[10000];
  sprintf(name, "%s/convergence.dat", out_dir);

  if (save_convergence.val)
  {
    ierr = PetscFOpen(mpi.comm(), name, "w", &fich); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), fich, "lmin "
                                          "lmax "
                                          "h "
                                          "sol_error "
                                          "grad_error\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
  }

  if (mpi.rank() == 0 && save_parameters.val) {
    std::ostringstream file;
    file << out_dir << "/parameters.dat";
    pl.save_all(file.str().c_str());
  }

  if (save_vtk.val)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  // define effective level set function
  mls_eff_cf_t phi_eff_cf;

  for (int i = 0; i < num_phis(); ++i) {
    phi_eff_cf.add_domain(phi_cf_all[i], phi_opn(i));
  }

  for (int split = 0; split < num_splits.val; ++split)
  {
    for (int sub_split = 0; sub_split < pow(2, add_splits.val); ++sub_split)
    {

      if (split == num_splits.val - 1 && sub_split > 0) break;

    FILE *fich_current;
    char name_current[10000];
    sprintf(name_current, "%s/error_split_%2d.dat", out_dir, split);

    if (save_convergence.val)
    {
      ierr = PetscFOpen(mpi.comm(), name_current, "w", &fich_current); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich_current, "step "
                                                    "time "
                                                    "dt "
                                                    "sol_error "
                                                    "grad_error\n"); CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(), fich_current); CHKERRXX(ierr);
    }

    int initial_division = (pow(2, add_splits.val) + sub_split);
    int lvl_decrease = floor(log(double(initial_division))/log(2.));

    while (true) {
      if (initial_division % 2 == 0) {
        initial_division = initial_division / 2;
        --lvl_decrease;
      } else {
        break;
      }
    }

    // domain size information
    const int n_xyz[]      = { DIM(num_trees_x.val*initial_division, num_trees_y.val*initial_division, num_trees_z.val*initial_division) };
    const double xyz_min[] = { DIM(xmin.val, ymin.val, zmin.val) };
    const double xyz_max[] = { DIM(xmax.val, ymax.val, zmax.val)};
    const int periodic[]   = { 0,  0,  0};
    splitting_criteria_cf_t sp(lmin.val+split-lvl_decrease, lmax.val+split-lvl_decrease, &phi_eff_cf, lip.val, band.val);

    double dx_eff = (xmax.val-xmin.val)/double(n_xyz[0])/pow(2., sp.max_lvl);
    double lmax_eff = sp.max_lvl + log(initial_division)/log(2.);
    double lmin_eff = sp.min_lvl + log(initial_division)/log(2.);

    // p4est variables
    p4est_t                   *p4est;
    p4est_nodes_t             *nodes;
    p4est_ghost_t             *ghost;
    p4est_connectivity_t      *conn;
    my_p4est_brick_t          *brick;
    my_p4est_hierarchy_t      *hierarchy;
    my_p4est_node_neighbors_t *ngbd;

    vec_and_ptr_t       phi_eff;
    vec_and_ptr_array_t phi(num_phis());
    vec_and_ptr_t       vn;
    vec_and_ptr_t       rhs;
    vec_and_ptr_dim_t   normal;
    vec_and_ptr_array_t sol(num_time_layers());
    vec_and_ptr_t       sol_exact;
    vec_and_ptr_t       sol_error;
    vec_and_ptr_t       grad_error;

    std::vector<double> dt(num_time_layers(), 0);

    double time           = 0;
    int    step           = 0;
    bool   keep_going     = true;
    double sol_error_max  = 0;
    double grad_error_max = 0;

    int done = 0;

    ierr = PetscPrintf(mpi.comm(), "Grid %2.3f / %2.3f, h %1.2e -> ",
                       lmin_eff, lmax_eff, dx_eff); CHKERRXX(ierr);
    while (keep_going)
    {
      // ------------------------------------
      // update location
      // ------------------------------------
      xc.val = trajectory(_X_, time);
      yc.val = trajectory(_Y_, time);
      angle.val = spin(VAL, time);

      // ------------------------------------
      // update grid
      // ------------------------------------
      if (lmin.val != lmax.val || step == 0) {

        p4est_t                   *tmp_p4est     = p4est;
        p4est_nodes_t             *tmp_nodes     = nodes;
        p4est_ghost_t             *tmp_ghost     = ghost;
        p4est_connectivity_t      *tmp_conn      = conn;
        my_p4est_brick_t          *tmp_brick     = brick;
        my_p4est_hierarchy_t      *tmp_hierarchy = hierarchy;
        my_p4est_node_neighbors_t *tmp_ngbd      = ngbd;
        vec_and_ptr_array_t tmp_sol(num_time_layers());
        tmp_sol.vec = sol.vec;

        // create grid
        brick = new my_p4est_brick_t();
        conn  = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);
        p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

        // refine grid
        p4est->user_pointer = &sp;
        my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

        // partition grid
        my_p4est_partition(p4est, P4EST_TRUE, NULL);

        // create ghost layer
        ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

        // create node structure
        nodes = my_p4est_nodes_new(p4est, ghost);

        // create hierarchy and neighborhood information
        hierarchy = new my_p4est_hierarchy_t(p4est, ghost, brick);
        ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
        ngbd->init_neighbors();

        if (step != 0)
        {
          phi_eff.destroy();
          phi.destroy();
          normal.destroy();
          vn.destroy();
          rhs.destroy();
          sol_error.destroy();
          sol_exact.destroy();
          grad_error.destroy();
        }

        phi_eff.create(p4est, nodes);
        phi.create(p4est, nodes);
        normal.create(p4est, nodes);
        sol.create(phi_eff.vec);
        vn.create(phi_eff.vec);
        rhs.create(phi_eff.vec);
        sol_error.create(phi_eff.vec);
        sol_exact.create(phi_eff.vec);
        grad_error.create(phi_eff.vec);

        if (step != 0)
        {
          // transfer data
          my_p4est_interpolation_nodes_t interp(tmp_ngbd);
          interp.add_all_nodes(p4est, nodes);

          for (int i = 0; i < num_time_layers(); ++i) {
            interp.set_input(tmp_sol.vec[i], quadratic_non_oscillatory_continuous_v2);
            interp.interpolate(sol.vec[i]);
          }

          // destroy old data structures
          tmp_sol.destroy();
          delete tmp_ngbd;
          delete tmp_hierarchy;
          p4est_nodes_destroy(tmp_nodes);
          p4est_ghost_destroy(tmp_ghost);
          p4est_destroy      (tmp_p4est);
          my_p4est_brick_destroy(tmp_conn, tmp_brick);
          delete tmp_brick;
        }
      }

      // ------------------------------------
      // sample geometry
      // ------------------------------------
      sample_cf_on_nodes(p4est, nodes, phi_eff_cf, phi_eff.vec);

      for (int i = 0; i < num_phis(); ++i) {
        sample_cf_on_nodes(p4est, nodes, phi_cf_all[i], phi.vec[i]);
      }

      if (num_phis() > 0) {
        normal.get_array();
        foreach_node(n, nodes) {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);

          int idx = phi_eff_cf.get_idx(DIM(xyz[0], xyz[1], xyz[2]));
          XCODE( normal.ptr[0][n] = phi_x_cf_all[idx].value(xyz) );
          YCODE( normal.ptr[1][n] = phi_y_cf_all[idx].value(xyz) );
          ZCODE( normal.ptr[2][n] = phi_z_cf_all[idx].value(xyz) );

          double norm = ABSD(normal.ptr[0][n], normal.ptr[1][n], normal.ptr[2][n]);

          XCODE( normal.ptr[0][n] /= norm );
          YCODE( normal.ptr[1][n] /= norm );
          ZCODE( normal.ptr[2][n] /= norm );
        }
        normal.restore_array();
      }

      // ------------------------------------
      // solve pde
      // ------------------------------------
      sol_cf.t = time;
      sol_x_cf.t = time;
      sol_y_cf.t = time;

      sample_cf_on_nodes(p4est, nodes, sol_cf, sol_exact.vec);

      if (step < num_time_layers()) {

        for (int i = 1; i < num_time_layers(); ++i) {
          VecCopyGhost(sol.vec[num_time_layers()-i-1], sol.vec[num_time_layers()-i]);
        }

        sample_cf_on_nodes(p4est, nodes, sol_cf, sol.vec[0]);
      } else {

        std::vector<double> time_coeffs;
        variable_step_BDF_implicit(order_in_time.val, dt, time_coeffs);

        sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs.vec);

        for (int i = 0; i < num_time_layers(); ++i) {
          VecAXPBYGhost(rhs.vec, -time_coeffs[i+1], 1., sol.vec[i]);
        }

        VecScaleGhost(rhs.vec, 1./dt[0]);

        for (int i = 1; i < num_time_layers(); ++i) {
          VecCopyGhost(sol.vec[num_time_layers()-i-1], sol.vec[num_time_layers()-i]);
        }

        my_p4est_poisson_nodes_mls_t solver(ngbd);
        solver.set_diag(time_coeffs[0]/dt[0]);
        solver.set_wc(wc_type.val, wc_value_cf);
        solver.set_mu(diffusivity.val);
        solver.set_rhs(rhs.vec);
        solver.set_fv_scheme(1);
        solver.set_kink_treatment(1);
        solver.set_cube_refinement(1);
        solver.set_integration_order(2);
        solver.set_use_taylor_correction(1);

        for (int i = 0; i < num_phis(); ++i) {
          solver.add_boundary(phi_opn(i), phi.vec[i], DIM(NULL, NULL, NULL), bc_type.val, bc_value_cf_all[i], zero_cf);
        }

        solver.solve(sol.vec[0], true);

        my_p4est_level_set_t ls(ngbd);
        vec_and_ptr_t mask;
        switch (extension_region.val) {
          case 0: mask.vec = phi_eff.vec; break;
          case 1: mask.vec = solver.get_mask(); break;
          default: throw; //TODO
        }

        if (num_phis() > 0) {
          switch (extension_type.val) {
            case 0:
              mask.get_array();
              sol.get_array();
              sol_exact.get_array();
              foreach_node(n, nodes) {
                if (mask.ptr[n] > 0) {
                  sol.ptr[0][n] = sol_exact.ptr[n];
                }
              }
              mask.restore_array();
              sol.restore_array();
              sol_exact.restore_array();
            break;
            case 1:
              ls.extend_Over_Interface_TVD(phi_eff.vec, sol.vec[0], num_extension_iters.val, extension_order.val, 0, -5.*p4est_diag_min(p4est), 5.*p4est_diag_min(p4est), DBL_MAX, NULL, mask.vec);
            break;
            case 2:
              ls.extend_Over_Interface_TVD_Full(phi_eff.vec, sol.vec[0], num_extension_iters.val,  extension_order.val, 0, -5.*p4est_diag_min(p4est), 5.*p4est_diag_min(p4est), DBL_MAX, NULL, mask.vec);
            break;
            default: throw; //TODO
          }
        }

        // ------------------------------------
        // compute error
        // ------------------------------------
        mask.get_array();
        sol_error.get_array();
        sol.get_array();
        sol_exact.get_array();
        grad_error.get_array();

        foreach_local_node(n, nodes) {
          if (mask.ptr[n] < 0) {
            sol_error.ptr[n] = fabs(sol.ptr[0][n]-sol_exact.ptr[n]);

            const quad_neighbor_nodes_of_node_t &qnnn = (*ngbd)[n];

            if (qnnn.is_stencil_in_negative_domain(mask.ptr)) {
              double xyz[P4EST_DIM];
              node_xyz_fr_n(n, p4est, nodes, xyz);

              double grad[P4EST_DIM];
              qnnn.gradient(sol.ptr[0], grad);

              grad_error.ptr[n] = ABSD(grad[0] - sol_x_cf.value(xyz), grad[1] - sol_y_cf.value(xyz), FIX FOR 3D);
            }
          } else {
            sol_error.ptr[n] = 0;
            grad_error.ptr[n] = 0;
          }
        }

        ierr = VecGhostUpdateBegin(sol_error.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd  (sol_error.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        ierr = VecGhostUpdateBegin(grad_error.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd  (grad_error.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        mask.restore_array();
        sol_error.restore_array();
        sol.restore_array();
        sol_exact.restore_array();
        grad_error.restore_array();
      }

      double local_max_errors[2];
      ierr = VecMax(sol_error.vec, NULL, &local_max_errors[0]); CHKERRXX(ierr);
      ierr = VecMax(grad_error.vec, NULL, &local_max_errors[1]); CHKERRXX(ierr);
      ierr = MPI_Allreduce(MPI_IN_PLACE, local_max_errors, 2, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(ierr);

      sol_error_max  = MAX(sol_error_max,  local_max_errors[0]);
      grad_error_max = MAX(grad_error_max, local_max_errors[1]);

//      ierr = PetscPrintf(mpi.comm(), "Step: %5d, dt: %1.2e, sol error: %1.3e, grad error: %1.3e\n", step, dt[0], local_max_errors[0], local_max_errors[1]); CHKERRXX(ierr);

      if (save_convergence.val)
      {
        ierr = PetscFOpen(mpi.comm(), name_current, "a", &fich_current); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fich_current, "%d ", step); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fich_current, "%1.2e ", time); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fich_current, "%1.2e ", dt); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fich_current, "%1.2e ", local_max_errors[0]); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(), fich_current, "%1.2e\n", local_max_errors[1]); CHKERRXX(ierr);
        ierr = PetscFClose(mpi.comm(), fich_current); CHKERRXX(ierr);
      }

      // ------------------------------------
      // compute time step
      // ------------------------------------
      double vn_max = 0;
      double vx = trajectory(V_X, time);
      double vy = trajectory(V_Y, time);
      double wz = spin(DDT, time);

      vn.get_array();
      phi_eff.get_array();
      normal.get_array();

      double dist_close = 2.*p4est_diag_min(p4est);

      foreach_local_node(n, nodes) {
        if (phi_eff.ptr[n] > 0 && phi_eff.ptr[n] < dist_close) {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est, nodes, xyz);

          double del_x = xyz[0] - xc.val;
          double del_y = xyz[1] - yc.val;

          double vx_full = vx + del_y*wz;
          double vy_full = vy - del_x*wz;

          vn.ptr[n] = SUMD(vx_full*normal.ptr[0][n], vy_full*normal.ptr[1][n], vz_full*normal.ptr[2][n]);

          vn_max = MAX(vn_max, fabs(vn.ptr[n]));
        }
      }

      vn.restore_array();
      phi_eff.restore_array();
      normal.restore_array();

      ierr = MPI_Allreduce(MPI_IN_PLACE, &vn_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(ierr);

      for (int i = 1; i < num_time_layers(); ++i) {
        dt[num_time_layers()-i] = dt[num_time_layers()-i-1];
      }

      dt[0] = MIN(time_range.val/pow(2.,lmax.val), cfl.val*dx_eff/(vn_max+EPS));

      if (time+dt[0] >= time_range.val) {
        dt[0] = time_range.val - time;
        keep_going = false;
      }

      time += dt[0];
      step++;

      // ------------------------------------
      // save data
      // ------------------------------------
      if (save_vtk.val && (step % save_vtk_freq.val == 0)) {
        const char *out_dir = getenv("OUT_DIR");
        if (!out_dir) out_dir = ".";
        else if (mpi.rank() == 0)
        {
          std::ostringstream command;
          command << "mkdir -p " << out_dir << "/vtu";
          int ret_sys = system(command.str().c_str());
          if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
        }

        std::ostringstream oss;
        oss << out_dir
            << "/vtu/diffusion" << CODEDIM("_2d", "_3d")
            << "_nprocs_" << p4est->mpisize
            << "_split_" << std::setfill('0') << std::setw(3)<< split
            << "_step_" << std::setfill('0') << std::setw(6)<< step/save_vtk_freq.val;

        vn.get_array();
        sol.get_array();
        phi_eff.get_array();
        sol_error.get_array();
        grad_error.get_array();
        normal.get_array();
        sol_exact.get_array();

        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               8, 0, oss.str().c_str(),
                               VTK_POINT_DATA, "phi", phi_eff.ptr,
                               VTK_POINT_DATA, "nx", normal.ptr[0],
            VTK_POINT_DATA, "ny", normal.ptr[1],
            VTK_POINT_DATA, "vn", vn.ptr,
            VTK_POINT_DATA, "sol", sol.ptr[0],
            VTK_POINT_DATA, "sol_error", sol_error.ptr,
            VTK_POINT_DATA, "sol_exact", sol_exact.ptr,
            VTK_POINT_DATA, "grad_error", grad_error.ptr);

        vn.restore_array();
        sol.restore_array();
        phi_eff.restore_array();
        sol_error.restore_array();
        grad_error.restore_array();
        normal.restore_array();
        sol_exact.restore_array();
      }

      if (10.*time/time_range.val >= done) {
        ierr = PetscPrintf(mpi.comm(), "."); CHKERRXX(ierr);
        done++;
      }
    }

    ierr = PetscPrintf(mpi.comm(), " -> sol error %1.2e, grad error %1.2e, steps %d\n",
                       sol_error_max, grad_error_max, step); CHKERRXX(ierr);

    if (save_convergence.val)
    {
      ierr = PetscFOpen(mpi.comm(), name, "a", &fich); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "%f ", lmin_eff); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "%f ", lmax_eff); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "%e ", dx_eff); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "%1.2e ", sol_error_max); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(), fich, "%1.2e\n", grad_error_max); CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(), fich); CHKERRXX(ierr);
    }

    // ------------------------------------
    // destroy the structures
    // ------------------------------------
    phi_eff.destroy();
    phi.destroy();
    normal.destroy();
    vn.destroy();
    rhs.destroy();
    sol_error.destroy();
    sol_exact.destroy();
    sol.destroy();
    grad_error.destroy();

    delete ngbd;
    delete hierarchy;
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(conn, brick);
    delete brick;

    }
  }

  w.stop(); w.read_duration();
}

