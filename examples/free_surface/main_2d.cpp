
/*
 * The navier-stokes free surface flow solver
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
#include <src/my_p8est_ns_free_surface.h>
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_ns_free_surface.h>
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

double xmin;
double xmax;
double ymin;
double ymax;
#ifdef P4_TO_P8
double zmin;
double zmax;
#endif

int nx;
int ny;
#ifdef P4_TO_P8
int nz;
#endif

/*
 *  ********* 2D *********
 * 0
 *  ********* 3D *********
 *
 */

int test_number;

double mu;
double rho;
double surf_tension;
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
    switch(test_number)
    {
    case 0:
      return (sqrt(SQR(x-0.25) + SQR(y-0.25) + SQR(z-0.25))<0.1) ? 1 : 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} init_smoke;


class BC_SMOKE : public CF_3
{
public:
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_smoke;


class LEVEL_SET: public CF_3
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return 0.15 - sqrt(SQR(x-(0.5 + 0.25*cos(2.0*M_PI*(tn+dt)/2.0))) + SQR(y-(0.5 + 0.25*sin(2.0*M_PI*(tn+dt)/2.0))) + SQR(z - 0.3));
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} level_set;

class INITIAL_FS: public CF_3
{
public:
  INITIAL_FS() { lip = 1.2; }
  double operator()(double, double, double z) const
  {
    switch(test_number)
    {
    case 0: return z - 0.7;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_fs;

class GLOBAL_INIT_LS : public CF_3
{
private:
  const LEVEL_SET *solid_ls;
  const INITIAL_FS *initial_fs_ls;
public:
  GLOBAL_INIT_LS(LEVEL_SET& solid_fs_, INITIAL_FS& initial_fs_ls_) : solid_ls(&solid_fs_), initial_fs_ls(&initial_fs_ls_) {}
  double operator()(double x, double y, double z) const
  {
    return ((((*solid_ls)(x, y, z) > 0.0) && ((*initial_fs_ls)(x, y, z) > 0.0))? MIN((*solid_ls)(x, y, z), (*initial_fs_ls)(x, y, z)) : MAX((*solid_ls)(x, y, z), (*initial_fs_ls)(x, y, z)));
  }
};

struct BCWALLTYPE_P : WallBC3D
{
  BoundaryConditionType operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0: return NEUMANN;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC3D
{
  BoundaryConditionType operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC3D
{
  BoundaryConditionType operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_v;

struct BCWALLTYPE_W : WallBC3D
{
  BoundaryConditionType operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_w;

struct BCWALLVALUE_U : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_v;

struct BCWALLVALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_w;

struct BCINTERFACE_VALUE_U : CF_3
{
  double operator()(double x, double y, double) const
  {
    switch(test_number)
    {
    case 0:
    {
      double radius = sqrt(SQR(x - 0.5) + SQR(y - 0.5));
      return -M_PI*radius*sin(2.0*M_PI*(tn+dt)/2.0);
    }
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_3
{
  double operator()(double x, double y, double) const
  {
    switch(test_number)
    {
    case 0:
    {
      double radius = sqrt(SQR(x - 0.5) + SQR(y - 0.5));
      return M_PI*radius*cos(2.0*M_PI*(tn+dt)/2.0);
    }
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_v;

struct BCINTERFACE_VALUE_W : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_w;

struct initial_velocity_unm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vn;

struct initial_velocity_wnm1_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_wnm1;

struct initial_velocity_wn_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_wn;

struct external_force_u_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

struct external_force_v_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

struct external_force_w_t : CF_3
{
  double operator()(double, double, double) const
  {
    switch(test_number)
    {
    case 0:
      return -9.81*rho;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

#else

class INIT_SMOKE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return (sqrt(SQR(x-0.25) + SQR(y-0.25))<0.1) ? 1 : 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} init_smoke;


class BC_SMOKE : public CF_2
{
public:
  double operator()(double , double ) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_smoke;


class LEVEL_SET: public CF_2
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return -1;
    case 1: return 0.15 - sqrt(SQR(x-(0.5 + 0.1*cos(2.0*M_PI*(tn+dt)))) + SQR(y- 0.3));
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} level_set;

class INITIAL_FS: public CF_2
{
public:
  INITIAL_FS() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return y - (0.7 + 0.05*cos(2.0*M_PI*x)); // increasing the amplitude will make the simulation blow up (unphysical conditions on the side walls...)
    case 1: return y - 0.7;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_fs;

class GLOBAL_INIT_LS : public CF_2
{
private:
  const LEVEL_SET *solid_ls;
  const INITIAL_FS *initial_fs_ls;
public:
  GLOBAL_INIT_LS(LEVEL_SET& solid_fs_, INITIAL_FS& initial_fs_ls_) : solid_ls(&solid_fs_), initial_fs_ls(&initial_fs_ls_) {}
  double operator()(double x, double y) const
  {
    return ((((*solid_ls)(x, y) > 0.0) && ((*initial_fs_ls)(x, y) > 0.0))? MIN((*solid_ls)(x, y), (*initial_fs_ls)(x, y)) : MAX((*solid_ls)(x, y), (*initial_fs_ls)(x, y)));
  }
};

struct BCWALLTYPE_P : WallBC2D
{
  BoundaryConditionType operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return NEUMANN;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC2D
{
  BoundaryConditionType operator()(double, double y) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return (((fabs(y) < EPS) || (fabs(y - 1.0) < EPS))? DIRICHLET : NEUMANN);
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_v;

struct BCWALLVALUE_U : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_v;

struct BCINTERFACE_VALUE_U : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0: return 0.0;
    case 1: return -0.2*M_PI*sin(2.0*M_PI*(tn+dt));
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_v;

struct initial_velocity_unm1_t : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vn;

struct external_force_u_t : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return 0.0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

struct external_force_v_t : CF_2
{
  double operator()(double, double) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
      return -9.81*rho;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

#endif


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nx", "the number of trees in the x direction");
  cmd.add_option("ny", "the number of trees in the y direction");
#ifdef P4_TO_P8
  cmd.add_option("nz", "the number of trees in the z direction");
#endif
  cmd.add_option("tf", "the final time");
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate)");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx");
  cmd.add_option("n_times_dt", "dt = n_times_dt * dx/vmax");
  cmd.add_option("thresh", "the threshold used for the refinement criteria");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("save_every_delta_t", "export images every delta_t simulation time");
  cmd.add_option("save_forces", "save the forces");
  cmd.add_option("smoke", "0 - no smoke, 1 - with smoke");
  cmd.add_option("smoke_thresh", "threshold for smoke refinement");
  cmd.add_option("refine_with_smoke", "refine the grid with the smoke density and threshold smoke_thresh");
  cmd.add_option("out_dir", "exportation directory");
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
                 0: [0, 1]X[0, 1]X[0, 1] computational box, periodic x and y; \n\
                    with gravity toward negative z \n\
                    free surface initially at z = 0.7 \n\
                    fluid initially at rest \n\
                    solid sphere of radius 0.15, intially centered at (0.75, 0.5, 0.3) \n\
                    rotating around the z-axiz centered in x = 0.5 and y = 0.5 (frequency = 0.5 Hz) \n\
                    no-slip condition on the sphere, no-slip wall on bottom wall z = 0 \n");
#else
  cmd.add_option("test", "choose a test.\n\
                 0: [0, 1]X[0, 1] computational box - \n\
                    with gravity toward negative y \n\
                    free surface initially at y = 0.7 + 0.05*cos(2.0*M_PI*x) \n\
                    fluid initially at rest \n\
                    no immersed solid \n\
                    no-slip wall on bottom wall y = 0 \n\
                    no-penetration on side walls x= 0 or x = 1 => u = 0 \n\
                    free-slip on the side walls  x= 0 or x = 1 => dv/dn = 0 \n\
                 1: [0, 1]X[0, 1] computational box - \n\
                    with gravity toward negative y \n\
                    free surface initially at y = 0.7 \n\
                    fluid initially at rest \n\
                    immersed disk of radius 0.15 and oscillating (xc, yc) = (0.5+0.1*cos(2.0*M_PI*t), 0.3) \n\
                    no-slip wall on bottom wall y = 0 \n\
                    periodic boundary conditions along x.\n");
#endif
  cmd.parse(argc, argv);

  int sl_order = cmd.get("sl_order", 2);
  int lmin = cmd.get("lmin", 4);
  int lmax = cmd.get("lmax", 6);
  double n_times_dt = cmd.get("n_times_dt", 1.0);
  double threshold_split_cell = cmd.get("thresh", 0.1);
  bool save_vtk = cmd.contains("save_vtk");
  bool save_forces = cmd.get("save_forces", 1);
  double save_every_dt = cmd.get("save_every_delta_t", 1.0/24.0);
  test_number = cmd.get("test", 1);

  bool with_smoke = cmd.get("smoke", 1);
  bool refine_with_smoke = cmd.get("refine_with_smoke", 0);
  double smoke_thresh = cmd.get("smoke_thresh", .5);
  double uniform_band = cmd.get("uniform_band", 3.0);

  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

  double tf;

  switch(test_number)
  {
#ifdef P4_TO_P8
  case 0: nx=cmd.get("nx", 1); ny=cmd.get("ny", 1); nz=cmd.get("nz", 1); xmin=0.0; xmax=1.0; ymin=0; ymax=1.0; zmin=0.0; zmax=1.0; mu=8.9e-4; rho=1000.0; surf_tension=72.86e-3; u0=0.4*M_PI; r0=0.15; tf=cmd.get("tf", 10.0);  break;
#else
  case 0: nx=cmd.get("nx", 1); ny=cmd.get("ny", 1); xmin=0.0; xmax=1.0; ymin=0.0; ymax=1.0; mu=8.9e-4; rho=1000.0; surf_tension=72.86e-3; u0=1.0; /*arbitrary estimate...*/ r0=-2.0; tf=cmd.get("tf", 10.0);  break;
  case 1: nx=cmd.get("nx", 1); ny=cmd.get("ny", 1); xmin=0.0; xmax=1.0; ymin=0.0; ymax=1.0; mu=8.9e-4; rho=1000.0; surf_tension=72.86e-3; u0=0.2*M_PI; r0=0.15; tf=cmd.get("tf", 20.0);  break;
#endif
  default: throw std::invalid_argument("choose a valid test.");
  }


#ifdef P4_TO_P8
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny, (zmax-zmin)/(double)nz) / (1<<lmax);
#else
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny) / (1<<lmax);
#endif

  PetscErrorCode ierr;
#ifdef P4_TO_P8
#else
  ierr = PetscPrintf(mpi.comm(), "Parameters : mu = %g, rho = %g, grid is %dx%d\n", mu, rho, nx, ny); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(mpi.comm(), "n_times_dt = %g, uniform_band = %g\n", n_times_dt, uniform_band);

  parStopWatch watch;
  watch.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

#ifdef P4_TO_P8
  int n_xyz [] = {nx, ny, nz};
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
  int periodic[P4EST_DIM];
  switch (test_number) {
  case 0:
    periodic[0] = 1;
    periodic[1] = 1;
    periodic[2] = 0;
    break;
  default:
    throw std::invalid_argument("free_surface_3d::main(): choose a valid test [periodicity].");
    break;
  }
#else
  int n_xyz [] = {nx, ny};
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
  int periodic[P4EST_DIM];
  switch (test_number) {
  case 0:
    periodic[0] = 0;
    periodic[1] = 0;
    break;
  case 1:
    periodic[0] = 1;
    periodic[1] = 0;
    break;
  default:
    throw std::invalid_argument("free_surface_2d::main(): choose a valid test [periodicity].");
    break;
  }
#endif
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t *p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  GLOBAL_INIT_LS global_init_ls(level_set, initial_fs);

  splitting_criteria_cf_t data(lmin, lmax, &global_init_ls, 1.2);

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

  if(refine_with_smoke==1)
  {
    splitting_criteria_thresh_t crit_thresh(lmin, lmax, &init_smoke, smoke_thresh);
    p4est_n->user_pointer = (void*)&crit_thresh;
    my_p4est_refine(p4est_n, P4EST_TRUE, refine_levelset_thresh, NULL);
    p4est_balance(p4est_n, P4EST_CONNECT_FULL, NULL);
  }

  p4est_n->user_pointer = (void*)&data;
  my_p4est_partition(p4est_n, P4EST_FALSE, NULL);

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
  Vec fs_phi;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &fs_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, initial_fs, fs_phi);

  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_2nd_order(phi);
  lsn.reinitialize_2nd_order(fs_phi);
  lsn.perturb_level_set_function(phi, EPS);
  lsn.perturb_level_set_function(fs_phi, EPS);

#ifdef P4_TO_P8
  CF_3 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1 , &initial_velocity_wnm1 };
  CF_3 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn   , &initial_velocity_wn};
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

  my_p4est_ns_free_surface_t free_surface_solver(ngbd_nm1, ngbd_n, faces_n);
  free_surface_solver.set_phis(phi, fs_phi);
  free_surface_solver.set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, n_times_dt, surf_tension);

  switch (test_number) {
#ifdef P4_TO_P8
  case 0:
    free_surface_solver.set_dt(MIN(dxmin*n_times_dt/u0, 0.05*2.0), MIN(dxmin*n_times_dt/u0, 0.05*2.0)); // MIN(max_cfl_solid_velocity, 0.05*period)
    break;
#else
  case 0:
    free_surface_solver.set_dt(dxmin*n_times_dt/u0, dxmin*n_times_dt/u0);
    break;
  case 1:
    free_surface_solver.set_dt(MIN(dxmin*n_times_dt/u0, 0.05), MIN(dxmin*n_times_dt/u0, 0.05)); // MIN(max_cfl_solid_velocity, 0.05*period)
    break;
#endif
  default:
#ifdef P4_TO_P8
    throw std::invalid_argument("free_surface_3d::main(): choose a valid test [set_dt].");
#else
    throw std::invalid_argument("free_surface_2d::main(): choose a valid test [set_dt].");
#endif
    break;
  }
  dt = free_surface_solver.get_dt();
  free_surface_solver.set_velocities(vnm1, vn);
  free_surface_solver.set_bc(bc_v, &bc_p);

  if(with_smoke)
  {
    Vec smoke;
    ierr = VecDuplicate(phi, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    free_surface_solver.set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }

  tn = 0;
  int iter = 0;
  int export_n = -1;

  char vtk_name[1000];
  double forces[P4EST_DIM];

  FILE *fp_forces;
  char file_forces[1000];

#if defined(POD_CLUSTER)
  string out_dir = cmd.get<string>("out_dir", "/home/regan/free_surface");
#elif defined(STAMPEDE)
  string out_dir = cmd.get<string>("out_dir", "/work/04965/tg842642/stampede2/free_surface");
#else
  string out_dir = cmd.get<string>("out_dir", "/home/regan/workspace/projects/free_surface");
#endif

  string sub_folder_path = out_dir + "/" + to_string(P4EST_DIM) + "d/test_" + to_string(test_number);
  string vtk_path = sub_folder_path + "/vtu";
  if (mpi.rank()==0 && (save_forces || save_vtk))
  {
    ostringstream command;
    command << "mkdir -p " << sub_folder_path;
    cout << "Creating a folder in " << sub_folder_path << endl;
    int sys_return = system(command.str().c_str()); (void) sys_return;
    if(save_vtk)
    {
      ostringstream vtk_command;
      vtk_command << "mkdir -p " << vtk_path;
      cout << "Creating folder for vtk's in " << vtk_path << endl;
      sys_return = system(vtk_command.str().c_str()); (void) sys_return;
    }
  }

  if(save_forces &&
   #ifdef P4_TO_P8
     (test_number==0)
   #else
     (test_number==1)
   #endif
     )
  {
#ifdef P4_TO_P8
    sprintf(file_forces, "%s/forces_%d-%d_%dx%dx%d_thresh_%g_ntimesdt_%g_sl_%d.dat", sub_folder_path.c_str(), lmin, lmax, nx, ny, nz, threshold_split_cell, n_times_dt, sl_order);
#else
    sprintf(file_forces, "%s/forces_%d-%d_%dx%d_thresh_%g_ntimesdt_%g_sl_%d.dat", sub_folder_path.c_str(), lmin, lmax, nx, ny, threshold_split_cell, n_times_dt, sl_order);
#endif

    ierr = PetscPrintf(mpi.comm(), "Saving forces in ... %s\n", file_forces); CHKERRXX(ierr);
    if(mpi.rank()==0)
    {
      fp_forces = fopen(file_forces, "w");
      if(fp_forces==NULL)
#ifdef P4_TO_P8
        throw std::invalid_argument("[ERROR]: free_surface_3d::main(): could not open file for forces output.");
      fprintf(fp_forces, "%% tn | Cd_x | Cd_y | Cd_z\n");
#else
        throw std::invalid_argument("[ERROR]: free_surface_2d::main(): could not open file for forces output.");
      fprintf(fp_forces, "%% tn | Cd_x | Cd_y\n");
#endif
      fclose(fp_forces);
    }
  }

  while(tn+0.01*dt<tf)
  {
    if(iter>0)
    {
      free_surface_solver.compute_dt(free_surface_solver.get_max_L2_norm_u());

      dt = free_surface_solver.get_dt();

      if(tn+dt>tf)
      {
        dt = tf-tn;
        free_surface_solver.set_dt(dt);
      }

      if(save_vtk && dt > save_every_dt)
      {
        dt = save_every_dt; // so that we don't miss frames...
        free_surface_solver.set_dt(dt);
      }

      free_surface_solver.update_from_tn_to_tnp1(&level_set, ((sl_order == 2) && (iter > 1)));
    }

    if(external_force_u==NULL) delete external_force_u;
    external_force_u = new external_force_u_t;

    if(external_force_v==NULL) delete external_force_v;
    external_force_v = new external_force_v_t;

#ifdef P4_TO_P8
    if(external_force_w==NULL) delete external_force_w;
    external_force_w = new external_force_w_t;
#endif


#ifdef P4_TO_P8
    CF_3 *external_forces[P4EST_DIM] = { external_force_u, external_force_v, external_force_w };
#else
    CF_2 *external_forces[P4EST_DIM] = { external_force_u, external_force_v };
#endif
    free_surface_solver.set_external_forces(external_forces);

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, free_surface_solver.get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double err_hodge = 1;
    int iter_hodge = 0;
    while(iter_hodge< 10 && err_hodge>1e-3)
    {
      hodge_new = free_surface_solver.get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      free_surface_solver.solve_viscosity();
      free_surface_solver.solve_projection();

      hodge_new = free_surface_solver.get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      err_hodge = 0;
      p4est_t *p4est = free_surface_solver.get_p4est();
      my_p4est_interpolation_nodes_t *interp_global_phi = free_surface_solver.get_interp_global_phi();
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
          double xyz[P4EST_DIM];
          quad_xyz_fr_q(quad_idx, tree_idx, p4est, free_surface_solver.get_ghost(), xyz);
#ifdef P4_TO_P8
          if((*interp_global_phi)(xyz[0],xyz[1],xyz[2])<1.5*dxmin)
#else
          if((*interp_global_phi)(xyz[0],xyz[1])<1.5*dxmin)
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
    free_surface_solver.compute_velocity_at_nodes();
    free_surface_solver.compute_pressure();

    tn += dt;


    if(save_forces &&
     #ifdef P4_TO_P8
       (test_number==0)
     #else
       (test_number==1)
     #endif
       )
    {
      free_surface_solver.compute_forces(forces);
      if(mpi.rank()==0)
      {
        fp_forces = fopen(file_forces, "a");
        if(fp_forces==NULL)
#ifdef P4_TO_P8
          throw std::invalid_argument("[ERROR]: free_surface_3d::main(): could not open file for forces output.");
        fprintf(fp_forces, "%g %g %g %g\n", tn, forces[0]/r0/r0/u0/u0/rho, forces[1]/r0/r0/u0/u0/rho, forces[2]/r0/r0/u0/u0/rho);
#else
          throw std::invalid_argument("[ERROR]: free_surface_2d::main(): could not open file for forces output.");
        fprintf(fp_forces, "%g %g %g\n", tn, forces[0]/r0/u0/u0/rho, forces[1]/r0/u0/u0/rho);
#endif
        fclose(fp_forces);
      }
    }

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*tn/tf, free_surface_solver.get_max_L2_norm_u(), free_surface_solver.get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(free_surface_solver.get_max_L2_norm_u()>100.0) {
      if(save_vtk)
      {
        sprintf(vtk_name, "%s/with_time_%d", vtk_path.c_str(), export_n+1);
        free_surface_solver.save_vtk(vtk_name);
      }
      std::cerr << "I think I've blown up..." << std::endl;
      break;
    }

    if(save_vtk && ((int) floor(tn/save_every_dt)) != export_n)
    {
      export_n = ((int) floor(tn/save_every_dt));
      sprintf(vtk_name, "%s/with_time_%d", vtk_path.c_str(), export_n);
      free_surface_solver.save_vtk(vtk_name);
    }
    iter++;
  }

  if(external_force_u==NULL) delete external_force_u;
  if(external_force_v==NULL) delete external_force_v;
#ifdef P4_TO_P8
  if(external_force_w==NULL) delete external_force_w;
#endif

  watch.stop();
  watch.print_duration();

  return 0;
}
