// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <p8est_communication.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/point3.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_communication.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/point2.h>
#endif

#include <src/ipm_logging.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/math.h>
#include <mpi.h>

using namespace std;

int nx, ny, nz;

#ifdef P4_TO_P8
class tube: public CF_3 {
  double tx, ty, tz, r, y0, z0, w1, w2, o;
public:
  tube(double tx_, double ty_, double tz_, double r_, double y0_, double z0_, double w1_, double w2_, int o_)
    : tx(tx_), ty(ty_), tz(tz_), r(r_), y0(y0_), z0(z0_), w1(w1_), w2(w2_), o(o_)
  {lip = 1.2;}
  double operator ()(double x, double y, double z) const {

    x -= 0.5*nx;
    y -= 0.5*ny;
    z -= 0.5*nz;

    double xp, yp, zp;
    if (o == 0){
      // rotate about z-axis
      xp =  cos(M_PI/180.*tz)*x + sin(M_PI/180.*tz)*y;
      yp = -sin(M_PI/180.*tz)*x + cos(M_PI/180.*tz)*y;
      zp =  z;
      x = xp; y = yp; z = zp;

      // rotate about y-axis
      xp =  cos(M_PI/180.*ty)*x - sin(M_PI/180.*ty)*z;
      yp =  y;
      zp =  sin(M_PI/180.*ty)*x + cos(M_PI/180.*ty)*z;
      x = xp; y = yp; z = zp;

      // rotate about x-axis
      xp = x;
      yp =  cos(M_PI/180.*tx)*y + sin(M_PI/180.*tx)*z;
      zp = -sin(M_PI/180.*tx)*y + cos(M_PI/180.*tx)*z;
      x = xp; y = yp; z = zp;
    } else if (o == 1) {
      // rotate about z-axis
      xp =  cos(M_PI/180.*tz)*x + sin(M_PI/180.*tz)*y;
      yp = -sin(M_PI/180.*tz)*x + cos(M_PI/180.*tz)*y;
      zp =  z;
      x = xp; y = yp; z = zp;

      // rotate about x-axis
      xp = x;
      yp =  cos(M_PI/180.*tx)*y + sin(M_PI/180.*tx)*z;
      zp = -sin(M_PI/180.*tx)*y + cos(M_PI/180.*tx)*z;
      x = xp; y = yp; z = zp;

      // rotate about y-axis
      xp =  cos(M_PI/180.*ty)*x - sin(M_PI/180.*ty)*z;
      yp =  y;
      zp =  sin(M_PI/180.*ty)*x + cos(M_PI/180.*ty)*z;
      x = xp; y = yp; z = zp;

    } else if (o == 2){
      // rotate about y-axis
      xp =  cos(M_PI/180.*ty)*x - sin(M_PI/180.*ty)*z;
      yp =  y;
      zp =  sin(M_PI/180.*ty)*x + cos(M_PI/180.*ty)*z;
      x = xp; y = yp; z = zp;

      // rotate about z-axis
      xp =  cos(M_PI/180.*tz)*x + sin(M_PI/180.*tz)*y;
      yp = -sin(M_PI/180.*tz)*x + cos(M_PI/180.*tz)*y;
      zp =  z;
      x = xp; y = yp; z = zp;

      // rotate about x-axis
      xp = x;
      yp =  cos(M_PI/180.*tx)*y + sin(M_PI/180.*tx)*z;
      zp = -sin(M_PI/180.*tx)*y + cos(M_PI/180.*tx)*z;
      x = xp; y = yp; z = zp;
    } else {
      // rotate about y-axis
      xp =  cos(M_PI/180.*ty)*x - sin(M_PI/180.*ty)*z;
      yp =  y;
      zp =  sin(M_PI/180.*ty)*x + cos(M_PI/180.*ty)*z;
      x = xp; y = yp; z = zp;

      // rotate about x-axis
      xp = x;
      yp =  cos(M_PI/180.*tx)*y + sin(M_PI/180.*tx)*z;
      zp = -sin(M_PI/180.*tx)*y + cos(M_PI/180.*tx)*z;
      x = xp; y = yp; z = zp;

      // rotate about z-axis
      xp =  cos(M_PI/180.*tz)*x + sin(M_PI/180.*tz)*y;
      yp = -sin(M_PI/180.*tz)*x + cos(M_PI/180.*tz)*y;
      zp =  z;
      x = xp; y = yp; z = zp;
    }

    return r*(1.0 - sqrt(SQR((y-y0)/(r*(1+0.15*sin(w1*M_PI*x)))) + SQR((z-z0)/(r*(1+0.15*sin(w2*M_PI*x))))));

  }
};

class Noodels:public CF_3{
  std::vector<tube> c;
public:
  Noodels(int n, double rmin, double rmax){
    lip = 1.2;
    c.reserve(n);
    for (int i=0; i<n; i++){
      double r  = ranged_rand(rmin, rmax);
      double tx = ranged_rand(0., 180.);
      double ty = ranged_rand(0., 180.);
      double tz = ranged_rand(0., 180.);
      double y  = ranged_rand(-0.5*ny, 0.5*ny);
      double z  = ranged_rand(-0.5*nz, 0.5*nz);
      double w1 = ranged_rand(3., 5.);
      double w2 = ranged_rand(3., 5.);
      int o     = rand() % 4;
      c.push_back(tube(tx, ty, tz, r, y, z, w1, w2, o));
    }
  }
  double operator ()(double x, double y, double z) const {
    double phi = -DBL_MAX;

    for (size_t i =0; i<c.size(); i++){
      const tube& cyl = c[i];
      phi = MAX(phi, cyl(x,y,z));
    }

    return phi;
  }
};

class velocity_t {
  std::vector<double> x0, y0, z0, a;
public:
  velocity_t(int n): x0(n), y0(n), z0(n), a(n) {
    for (int i=0; i<n; i++){
      x0[i] = ranged_rand(0.1*nx, 0.9*nx);
      y0[i] = ranged_rand(0.1*ny, 0.9*ny);
      z0[i] = ranged_rand(0.1*nz, 0.9*nz);
      a[i]  = 1.0;//ranged_rand(-1.5, 1.5);
    }
  }
  double vx(double x, double y, double z) const {
    double v = 0;
    for (size_t i=0; i<x0.size(); i++)
      v += 2.*a[i]*SQR(sin(M_PI*(x-x0[i])))*sin(2.*M_PI*(y-y0[i]))*sin(2.*M_PI*(z-z0[i]));
    return v/x0.size();
  }

  double vy(double x, double y, double z) const {
    double v = 0;
    for (size_t i=0; i<x0.size(); i++)
      v += -a[i]*SQR(sin(M_PI*(y-y0[i])))*sin(2.*M_PI*(x-x0[i]))*sin(2.*M_PI*(z-z0[i]));
    return v/x0.size();
  }

  double vz(double x, double y, double z) const {
    double v = 0;
    for (size_t i=0; i<x0.size(); i++)
      v += -a[i]*SQR(sin(M_PI*(z-z0[i])))*sin(2.*M_PI*(x-x0[i]))*sin(2.*M_PI*(y-y0[i]));
    return v/x0.size();
  }
};

class constant_cf: public CF_3{
  double c;
public:
  explicit constant_cf(double c_): c(c_) {}
  double operator()(double /* x */, double /* y */, double /* z */) const {
    return c;
  }
};
#else
class tube: public CF_2 {
  double tz, r, y0, w1, o;
public:
  tube(double tz_, double r_, double y0_, double w1_)
    : tz(tz_), r(r_), y0(y0_), w1(w1_)
  {lip = 1.2;}
  double operator ()(double x, double y) const {
    x -= 0.5;
    y -= 0.5;

    double xp, yp;
    // rotate about z-axis
    xp =  cos(M_PI/180.*tz)*x + sin(M_PI/180.*tz)*y;
    yp = -sin(M_PI/180.*tz)*x + cos(M_PI/180.*tz)*y;
    x = xp; y = yp;

    return r*(1+0.3*sin(w1*M_PI*x)) - ABS((y-y0));
  }
};

class Noodels:public CF_2{
  std::vector<tube> c;
public:
  Noodels(int n, double rmin, double rmax){
    lip = 1.1;
    c.reserve(n);
    for (int i=0; i<n; i++){
      double r = ranged_rand(rmin, rmax);
      double t = ranged_rand(0., 180.);
      double y = ranged_rand(0.25, 0.75);
      double w = ranged_rand(1., 2.);

      c.push_back(tube(t,r,y,w));
    }
  }
  double operator ()(double x, double y) const {
    double phi = -DBL_MAX;
    for (size_t i =0; i<c.size(); i++){
      const tube& cyl = c[i];
      phi = MAX(phi, cyl(x,y));
    }

    return phi;
  }
};

class vx_t: public CF_2 {
public:
  double operator()(double x, double y) const {
    return -SQR(sin(M_PI*x))*sin(2.*M_PI*y);
  }
} vx_cf;

class vy_t: public CF_2 {
public:
  double operator()(double x,double y) const {
    return SQR(sin(M_PI*y))*sin(2.*M_PI*x);
  }
} vy_cf;

class constant_cf: public CF_2{
  double c;
public:
  explicit constant_cf(double c_): c(c_) {}
  double operator()(double /* x */, double /* y */) const {
    return c;
  }
};
#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

template <typename T> 
inline int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

std::string output_dir;
double alpha, beta;

void motion_under_curvature1(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax);
void motion_under_curvature2(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax);
void motion_under_curvature3(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax);
void motion_normal_direction(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax);
void construct_grid_with_reinitializatrion1(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi);
void construct_grid_with_reinitializatrion2(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi);
void construct_grid_with_reinitializatrion3(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi);


#ifdef P4_TO_P8
class Interface:public CF_3{
public:
  double operator ()(double x, double y, double z) const {
    return 0.15 - sqrt(SQR(x-0.5) + SQR(y - 0.5) + SQR(z - 0.5));
  }
} sphere;

class Box: public CF_3{
  // CF_2 interface
public:
  double operator ()(double x, double y, double z) const {
    double f1 = fabs(x/nx - 0.5) - 0.48;
    double f2 = fabs(y/ny - 0.5) - 0.48;
    double f3 = fabs(z/nz - 0.5) - 0.48;

    return MAX(f1, f2, f3);
  }
} box;
#else
class Interface:public CF_2{
public:
  double operator ()(double x, double y) const {
    return 0.15 - sqrt(SQR(x-0.5) + SQR(y - 0.5));
  }
} sphere;

class Box: public CF_2{
  // CF_2 interface
public:
  double operator ()(double x, double y) const {
    double f1 = fabs(x/nx - 0.5) - 0.4;
    double f2 = fabs(y/ny - 0.5) - 0.4;

    return MAX(f1, f2);
  }
} box;
#endif


int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    PetscErrorCode      ierr;

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("itmax", "maximum number of iterations when creating random tree for strong scaling");
    cmd.add_option("seed", "seed for the RNG");
    cmd.add_option("output-dir", "address of the output directory for all I/O");
		cmd.add_option("rmin", "min radius of tubes");
		cmd.add_option("rmax", "max radius of tubes");
    cmd.add_option("nx", "# of blocks in x direction");
    cmd.add_option("ny", "# of blocks in y direction");
    cmd.add_option("nz", "# of blocks in z direction");
    cmd.add_option("alpha", "veclocity of normal direction");
    cmd.add_option("beta", "strength of curvature");
    cmd.add_option("count", "number of elements");
    cmd.add_option("Da_max", "maximum value of the Damkohler #");
    cmd.add_option("Da_min", "minimum value of the Damkohler #");
		cmd.parse(argc, argv);
    cmd.print();

    output_dir       = cmd.get<std::string>("output-dir");
    alpha = cmd.get("alpha", 1);
    beta  = cmd.get("beta", 0.02);
    const int lmin   = cmd.get("lmin", 3);
    const int lmax   = cmd.get("lmax", 10);
    const int itmax  = cmd.get("itmax", 3);
    const int seed   = cmd.get("seed", 0);
    const int count  = cmd.get("count", 250);
    const double Da_max = cmd.get("Da_max", 1.0e+3);
    const double Da_min = cmd.get("Da_min", 1.0e-3);
    nx = cmd.get("nx", 1);
    ny = cmd.get("ny", 1);
    nz = cmd.get("nz", 1);
    srand(seed);

    parStopWatch w1;//(parStopWatch::all_timings);
    parStopWatch w2;//(parStopWatch::all_timings);
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Print the SHA1 of the current commit
    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

    // print basic information
    PetscPrintf(mpi->mpicomm, "mpisize = %d\n", mpi->mpisize);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(nx, ny, nz, brick);
#else
    connectivity = my_p4est_brick_new(nx, ny, brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    Noodels noodles(count, cmd.get("rmin", 0.005), cmd.get("rmax", 0.01));
    splitting_criteria_cf_t sp(lmin, lmax, &noodles, 1.2);
    p4est->user_pointer = &sp;
    w2.stop(); w2.read_duration();

    p4est_ghost_t *ghost = NULL;
    p4est_nodes_t *nodes = NULL;
    Vec phi = NULL;
    double *phi_p = NULL; 

    // make the level-set signed distance
    w2.start("grid construction");
    construct_grid_with_reinitializatrion1(p4est, ghost, nodes, brick, phi);
    w2.stop(); w2.read_duration();    

    p4est_gloidx_t num_nodes = 0;
    for (int r =0; r<p4est->mpisize; r++)
      num_nodes += nodes->global_owned_indeps[r];

    PetscPrintf(p4est->mpicomm, "%% Initial grid info:\n global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);

    w2.start("advecting");
    {
      double dx = (double)P4EST_QUADRANT_LEN(sp.max_lvl)/(double)P4EST_ROOT_LEN;

      const static velocity_t vel(5);

      struct:CF_3{
        double operator()(double x, double y, double z) const {
          return vel.vx(x, y, z);
        }
      } vx_cf;

      struct:CF_3{
        double operator()(double x, double y, double z) const {
          return vel.vy(x, y, z);
        }
      } vy_cf;

      struct:CF_3{
        double operator()(double x, double y, double z) const {
          return vel.vz(x, y, z);
        }
      } vz_cf;

      for (int it = 0; it<itmax; it++){
        my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
        my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
        SemiLagrangian sl(&p4est, &nodes, &ghost, brick, &ngbd);
        sl.update_p4est_second_order(vx_cf, vy_cf, vz_cf, 10*dx, phi);
      }
    }
    w2.stop(); w2.read_duration();

    w2.start("smoothing things");
//    motion_normal_direction(p4est, ghost, nodes, brick, phi, itmax);
    motion_under_curvature3(p4est, ghost, nodes, brick, phi, itmax);
    w2.stop(); w2.read_duration();

    // write some statistics
    Vec ones, ones_l;
    ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(ones, &ones_l); CHKERRXX(ierr);
    ierr = VecSet(ones_l, 1.0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ones, &ones_l); CHKERRXX(ierr);

    double v_p =integrate_over_negative_domain(p4est, nodes, phi, ones);
    double a_p =integrate_over_interface(p4est, nodes, phi, ones);
    double h_p = v_p/a_p;
    double porosity = v_p/nx/ny/nz;
    PetscPrintf(p4est->mpicomm, "Porosity = %% %2.2f \t h_p = %e\n", 100.*porosity, h_p);
    ierr = VecDestroy(ones); CHKERRXX(ierr);

    std::ostringstream parname, topname, ngbname;
    parname << output_dir + "/" + "partition" << ".dat";
    topname << output_dir + "/" + "topology" << ".dat";
    ngbname << output_dir + "/" + "neighbors" << ".dat";

    write_comm_stats(p4est, ghost, nodes, parname.str().c_str(), topname.str().c_str(), ngbname.str().c_str());

    // compute Da for the domain
    struct Da_cf_t:public CF_3 {
      Da_cf_t(double da_min, double da_max): r(13), y(13), z(13) {
        double z_ [] = {0.5, 1.5, 2.5, 1.0, 2.0, 0.5, 1.5, 2.5, 1.0, 2.0, 0.5, 1.5, 2.5};
        double y_ [] = {0.5, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 2.5};
        for (int i=0; i<13; i++){
          y[i] = y_[i] + ranged_rand(-0.1, 0.1);
          z[i] = z_[i] + ranged_rand(-0.1, 0.1);
          r[i] = ranged_rand(0.05, 0.25);
        }

        this->da_min = da_min;
        this->da_max = da_max;
      }

      double operator()(double x_, double y_, double z_) const {
        (void) x_;

        double f = r[0] - sqrt(SQR(y_-y[0]) + SQR(z_-z[0]));
        for (int i=1; i<13; i++)
          f = MAX(f, r[i] - sqrt(SQR(y_-y[i]) + SQR(z_-z[i])));
        
        return da_min + da_max*0.5*(1 + erf(10*f));
      }
      
    private:
      std::vector<double> r, y, z;
      double da_min, da_max;
    };

    Da_cf_t Da_cf(Da_min, Da_max);

    Vec Da;
    double *Da_p;
    ierr = VecDuplicate(phi, &Da); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, Da_cf, Da);

    {
      ostringstream oss;
      oss << output_dir + "/Da";

      ierr = VecGetArray(Da, &Da_p); CHKERRXX(ierr);
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_FALSE,
                             2, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "Da", Da_p);
      ierr = VecRestoreArray(Da, &Da_p); CHKERRXX(ierr);      
    }

    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    ngbd.init_neighbors();

    // construct the solver
    struct:WallBC3D {
      BoundaryConditionType operator()(double x, double y, double z) const {
        (void) y;
        (void) z;

        if (x < EPS)
          return DIRICHLET;
        else
          return NEUMANN;
      }
    } wall_bc_type;

    struct:CF_3{
      double operator()(double x, double y, double z) const {
        (void) y;
        (void) z;

        if (x < EPS)
          return 1.0;
        else
          return 0.0;
      }
    } wall_bc_value;

    struct:CF_3 {
      double operator()(double x, double y, double z) const {
        (void) x;
        (void) y;
        (void) z;

        return 0;
      }
    } interface_bc_value;

    BoundaryConditions3D bc;
    bc.setInterfaceType(ROBIN);
    bc.setInterfaceValue(interface_bc_value);
    bc.setWallTypes(wall_bc_type);
    bc.setWallValues(wall_bc_value);

    double dt = 0.001;
    Vec rhs, con;
    double *rhs_p, *con_p;
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &con); CHKERRXX(ierr);

    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecGetArray(con, &con_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      rhs_p[i] = 0;
      con_p[i] = 0;
    }

    PoissonSolverNodeBase solver(&ngbd);
    solver.set_bc(bc);
    solver.set_robin_coef(Da);
    solver.set_diagonal(1.0/dt);
    solver.set_phi(phi);
    solver.set_rhs(rhs);

    int tc = 0;
    double tf = 1;
    my_p4est_level_set ls(&ngbd);
    for (double t = 0; t<tf; t += dt, tc++) {
      w2.start("solving system");
      solver.solve(con, true);
      w2.stop(); w2.read_duration();

      w2.start("extension");
      ls.extend_Over_Interface_TVD(phi, con);
      w2.stop(); w2.read_duration();

      for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
        rhs_p[i] = con_p[i] / dt;

      // save the output
      w2.start("saving vtk");
      ostringstream oss;
      oss << output_dir + "/solution." << tc;
      PetscPrintf(mpi->mpicomm, "%s\n", oss.str().c_str());
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_FALSE,
                             2, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "con", con_p);
      w2.stop(); w2.read_duration();
    }

    ierr = VecRestoreArray(con, &con_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    // free memory
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(Da); CHKERRXX(ierr);
    ierr = VecDestroy(con); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);

    p4est_destroy(p4est);
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

void motion_under_curvature1(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax)
{
  PetscErrorCode ierr;
  const splitting_criteria_cf_t *sp = (const splitting_criteria_cf_t*)p4est->user_pointer;
  parStopWatch w;

  double beta = 0.02; // solving for phi_t - beta kappa |grad phi| = 0
  double dx = (double)P4EST_QUADRANT_LEN(sp->max_lvl)/(double)P4EST_ROOT_LEN;
  double d_tau = dx;


  Vec phi_x, phi_y, norm_grad_phi;
  double *phi_x_p, *phi_y_p, *norm_grad_phi_p;
#ifdef P4_TO_P8
  Vec phi_z;
  double *phi_z_p;
#endif
  Vec rhs;
  Vec phi_np1;

  double *phi_p, *rhs_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  std::ostringstream oss; oss << output_dir + "/curvature_" << p4est->mpisize <<
                                 "p" << ".0";

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_FALSE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  struct:WallBC3D{
    BoundaryConditionType operator()(double /* x */, double /* y */, double /* z */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_3{
    double operator ()(double /* x */, double /* y */, double /* z */) const {return 0;}
  } zero_func;
#else
  struct:WallBC2D{
    BoundaryConditionType operator()(double /* x */, double /* y */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_2{
    double operator ()(double /* x */, double /* y */) const {return 0;}
  } zero_func;
#endif

  for(int iter = 0; iter < itmax; iter++)
  {
    w.start("preparing rhs");
    // create hierarchy and node neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, myb);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);

    ierr = VecDuplicate(phi, &phi_x); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_y); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecDuplicate(phi, &phi_z); CHKERRXX(ierr);
  #endif
    ierr = VecDuplicate(phi, &norm_grad_phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecGetArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    /* first compute grad phi */
    // 1- layer nodes
    for(size_t ni = 0; ni<ngbd.get_layer_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_layer_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 2- begin nonblocking update
    ierr = VecGhostUpdateBegin(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // 3- local nodes
    for(size_t ni = 0; ni<ngbd.get_local_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_local_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 4- finish nonblocking update
    ierr = VecGhostUpdateEnd(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* prepare right hand side */
    for(p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n)
    {
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      rhs_p[n] = phi_p[n];
      if(norm_grad_phi_p[n]>EPS){
#ifdef P4_TO_P8
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) + phi_z_p[n]*qnnn.dz_central(norm_grad_phi_p) );
#else
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) );
#endif
      }
    }

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    // remove unecessary arrays
    ierr = VecDestroy(phi_x); CHKERRXX(ierr);
    ierr = VecDestroy(phi_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_z); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(norm_grad_phi); CHKERRXX(ierr);
    w.stop(); w.read_duration();

    /* solve the system */
    w.start("solving the system");
    {
#ifdef P4_TO_P8
      BoundaryConditions3D bc;
#else
      BoundaryConditions2D bc;
#endif
      bc.setWallTypes(wall_bc_neumann);
      bc.setWallValues(zero_func);

      VecSet(phi, -1);

      PoissonSolverNodeBase solver(&ngbd);
      solver.set_bc(bc);
      solver.set_rhs(rhs);
      solver.set_phi(phi);
      solver.set_diagonal(1.0);
      solver.set_mu(d_tau*beta);
      solver.solve(phi_np1);
    }
    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD);       
    w.stop(); w.read_duration();

    my_p4est_level_set ls(&ngbd);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_np1, 10);

    /* construct a new grid */
    w.start("update grid");
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_np1->connectivity = p4est->connectivity;
    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, myb, &ngbd);

    phi_interp.set_input_parameters(phi_np1, quadratic);

    splitting_criteria_cf_t sp_np1(sp->min_lvl, sp->max_lvl, &phi_interp, sp->lip);
    p4est_np1->user_pointer = &sp_np1;

    my_p4est_coarsen(p4est_np1, P4EST_FALSE, coarsen_levelset_cf, NULL);
    my_p4est_refine(p4est_np1, P4EST_FALSE, refine_levelset_cf, NULL);

    // partition the new forest and create new nodes and ghost structures
    my_p4est_partition(p4est_np1, NULL);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // update the level-set value by interpolating from the old grid
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    for (size_t n = 0; n<nodes_np1->indep_nodes.elem_count; n++) {
      p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
      p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

      p4est_topidx_t* t2v = p4est->connectivity->tree_to_vertex;
      double *t2c = p4est->connectivity->vertices;
      p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree

      double xyz [] =
      {
        node_x_fr_i(indep_node) + t2c[3 * tr_mm + 0],
        node_y_fr_j(indep_node) + t2c[3 * tr_mm + 1]
  #ifdef P4_TO_P8
        ,
        node_z_fr_k(indep_node) + t2c[3 * tr_mm + 2]
  #endif
      };

      phi_interp.add_point_to_buffer(n, xyz);
    }
    phi_interp.interpolate(phi_p);

    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_destroy(p4est); p4est = p4est_np1;

    std::ostringstream oss; oss << output_dir + "/curvature_" << p4est->mpisize <<
                                   "p" << "." << iter+1;

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w.stop(); w.read_duration();
  }
}

void motion_under_curvature2(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax)
{
  PetscErrorCode ierr;
  const splitting_criteria_cf_t *sp = (const splitting_criteria_cf_t*)p4est->user_pointer;
  parStopWatch w;

  double beta = 0.025; // solving for phi_t - beta kappa |grad phi| = 0
  double dx = (double)P4EST_QUADRANT_LEN(sp->max_lvl)/(double)P4EST_ROOT_LEN;
  double d_tau = dx;


  Vec phi_x, phi_y, norm_grad_phi;
  double *phi_x_p, *phi_y_p, *norm_grad_phi_p;
#ifdef P4_TO_P8
  Vec phi_z;
  double *phi_z_p;
#endif
  Vec rhs;

  double *phi_p, *rhs_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  std::ostringstream oss; oss << output_dir + "/curvature_" << p4est->mpisize <<
                                 "p" << ".0";

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_FALSE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  struct:WallBC3D{
    BoundaryConditionType operator()(double /* x */, double /* y */, double /* z */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_3{
    double operator ()(double /* x */, double /* y */, double /* z */) const {return 0;}
  } zero_func;
#else
  struct:WallBC2D{
    BoundaryConditionType operator()(double /* x */, double /* y */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_2{
    double operator ()(double /* x */, double /* y */) const {return 0;}
  } zero_func;
#endif

  // create hierarchy and node neighbors
  my_p4est_hierarchy_t hierarchy(p4est, ghost, myb);
  my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);

  for(int iter = 0; iter < itmax; iter++)
  {
    w.start("preparing rhs");
    ierr = VecDuplicate(phi, &phi_x); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_y); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecDuplicate(phi, &phi_z); CHKERRXX(ierr);
  #endif
    ierr = VecDuplicate(phi, &norm_grad_phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);

    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecGetArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    /* first compute grad phi */
    // 1- layer nodes
    for(size_t ni = 0; ni<ngbd.get_layer_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_layer_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 2- begin nonblocking update
    ierr = VecGhostUpdateBegin(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // 3- local nodes
    for(size_t ni = 0; ni<ngbd.get_local_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_local_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 4- finish nonblocking update
    ierr = VecGhostUpdateEnd(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* prepare right hand side */
    for(p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n)
    {
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      rhs_p[n] = phi_p[n];
      if(norm_grad_phi_p[n]>EPS){
#ifdef P4_TO_P8
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) + phi_z_p[n]*qnnn.dz_central(norm_grad_phi_p) );
#else
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) );
#endif
      }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    // remove unecessary arrays
    ierr = VecDestroy(phi_x); CHKERRXX(ierr);
    ierr = VecDestroy(phi_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_z); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(norm_grad_phi); CHKERRXX(ierr);
    w.stop(); w.read_duration();

    /* solve the system */
    w.start("solving the system");
    {
#ifdef P4_TO_P8
      BoundaryConditions3D bc;
#else
      BoundaryConditions2D bc;
#endif
      bc.setWallTypes(wall_bc_neumann);
      bc.setWallValues(zero_func);

      VecSet(phi, -1);

      PoissonSolverNodeBase solver(&ngbd);
      solver.set_bc(bc);
      solver.set_rhs(rhs);
      solver.set_phi(phi);
      solver.set_diagonal(1.0);
      solver.set_mu(d_tau*beta);
      solver.solve(phi);
    }
    ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);
    w.stop(); w.read_duration();


    std::ostringstream oss; oss << output_dir + "/curvature_" << p4est->mpisize <<
                                   "p" << "." << iter+1;

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w.stop(); w.read_duration();
  }
}


void motion_under_curvature3(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax)
{
  PetscErrorCode ierr;
  const splitting_criteria_cf_t *sp = (const splitting_criteria_cf_t*)p4est->user_pointer;
  parStopWatch w;

  constant_cf vn(-alpha);
  double dx = (double)P4EST_QUADRANT_LEN(sp->max_lvl)/(double)P4EST_ROOT_LEN;
  double d_tau = dx;

  Vec phi_x, phi_y, norm_grad_phi;
  double *phi_x_p, *phi_y_p, *norm_grad_phi_p;
#ifdef P4_TO_P8
  Vec phi_z;
  double *phi_z_p;
#endif
  Vec rhs;
  Vec phi_np1;

  double *phi_p, *rhs_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  std::ostringstream oss; oss << output_dir + "/curvature.0";

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_FALSE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  struct:WallBC3D{
    BoundaryConditionType operator()(double /* x */, double /* y */, double /* z */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_3{
    double operator ()(double /* x */, double /* y */, double /* z */) const {return 0;}
  } zero_func;
#else
  struct:WallBC2D{
    BoundaryConditionType operator()(double /* x */, double /* y */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_2{
    double operator ()(double /* x */, double /* y */) const {return 0;}
  } zero_func;
#endif

  // clip to a small box
	// if (false)
  {
    std::vector<double> phi_box(nodes->indep_nodes.elem_count);
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, box, phi_box);
    for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
      phi_p[i] = -MAX(-phi_p[i], phi_box[i]);
    }
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  }

  for(int iter = 0; iter < itmax; iter++)
  {
    w.start("advect in normal direction");
    // create hierarchy and node neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, myb);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    my_p4est_level_set ls(&ngbd);
    ls.advect_in_normal_direction(vn, phi);
    w.stop(); w.read_duration();

    w.start("preparing rhs");
    ierr = VecDuplicate(phi, &phi_x); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_y); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecDuplicate(phi, &phi_z); CHKERRXX(ierr);
  #endif
    ierr = VecDuplicate(phi, &norm_grad_phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecGetArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    /* first compute grad phi */
    // 1- layer nodes
    for(size_t ni = 0; ni<ngbd.get_layer_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_layer_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 2- begin nonblocking update
    ierr = VecGhostUpdateBegin(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // 3- local nodes
    for(size_t ni = 0; ni<ngbd.get_local_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_local_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 4- finish nonblocking update
    ierr = VecGhostUpdateEnd(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* prepare right hand side */
    for(p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n)
    {
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      rhs_p[n] = phi_p[n];
      if(norm_grad_phi_p[n]>EPS){
#ifdef P4_TO_P8
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) + phi_z_p[n]*qnnn.dz_central(norm_grad_phi_p) );
#else
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) );
#endif
      }
    }

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    // remove unecessary arrays
    ierr = VecDestroy(phi_x); CHKERRXX(ierr);
    ierr = VecDestroy(phi_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_z); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(norm_grad_phi); CHKERRXX(ierr);
    w.stop(); w.read_duration();

    /* solve the system */
    w.start("solving the system");
    {
#ifdef P4_TO_P8
      BoundaryConditions3D bbc;
#else
      BoundaryConditions2D bbc;
#endif
      bbc.setWallTypes(wall_bc_neumann);
      bbc.setWallValues(zero_func);

      VecSet(phi, -1);

      PoissonSolverNodeBase solver(&ngbd);
      solver.set_bc(bbc);
      solver.set_rhs(rhs);
      solver.set_phi(phi);
      solver.set_diagonal(1.0);
      solver.set_mu(d_tau*beta);
//      solver.set_tolerances(1e-6);
      solver.solve(phi_np1);
    }
    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD);
    w.stop(); w.read_duration();

    ls.reinitialize_1st_order_time_2nd_order_space(phi_np1, 10);

//    /* construct a new grid */
    w.start("update grid");
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_np1->connectivity = p4est->connectivity;
    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, myb, &ngbd);

    phi_interp.set_input_parameters(phi_np1, quadratic);

    splitting_criteria_cf_t sp_np1(sp->min_lvl, sp->max_lvl, &phi_interp, 0.5*sp->lip);
    p4est_np1->user_pointer = &sp_np1;

    my_p4est_coarsen(p4est_np1, P4EST_FALSE, coarsen_levelset_cf, NULL);
    my_p4est_refine(p4est_np1, P4EST_FALSE, refine_levelset_cf, NULL);

    // partition the new forest and create new nodes and ghost structures
    my_p4est_partition(p4est_np1, NULL);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // update the level-set value by interpolating from the old grid
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    for (size_t n = 0; n<nodes_np1->indep_nodes.elem_count; n++) {
      p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
      p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

      p4est_topidx_t* t2v = p4est->connectivity->tree_to_vertex;
      double *t2c = p4est->connectivity->vertices;
      p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree

      double xyz [] =
      {
        node_x_fr_i(indep_node) + t2c[3 * tr_mm + 0],
        node_y_fr_j(indep_node) + t2c[3 * tr_mm + 1]
  #ifdef P4_TO_P8
        ,
        node_z_fr_k(indep_node) + t2c[3 * tr_mm + 2]
  #endif
      };

      phi_interp.add_point_to_buffer(n, xyz);
    }
    phi_interp.interpolate(phi_p);

    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_destroy(p4est); p4est = p4est_np1;

    std::ostringstream oss; oss << output_dir + "/curvature." << iter+1;

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w.stop(); w.read_duration();
  }

  p4est->user_pointer = (void*)(sp);
}


void construct_grid_with_reinitializatrion1(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi)
{
  splitting_criteria_cf_t *sp = (splitting_criteria_cf_t*)p4est->user_pointer;
  PetscErrorCode ierr;
  parStopWatch w;

  // Now refine the tree
  w.start("initial grid");
  for (int n=0; n<sp->max_lvl; n++){
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, NULL);
  }

  // Create the ghost structure
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  nodes = my_p4est_nodes_new(p4est, ghost);

  // create level-set
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *sp->phi, phi);

  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
  ngbd.init_neighbors();

  my_p4est_level_set ls(&ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
  w.stop(); w.read_duration();

  // recreate the grid
  Vec phi_xx, phi_yy;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec phi_zz;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz); CHKERRXX(ierr);
  ngbd.second_derivatives_central(phi, phi_xx, phi_yy, phi_zz);
#else
  ngbd.second_derivatives_central(phi, phi_xx, phi_yy);
#endif

  p4est_t *p4est_tmp = my_p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, NULL);
  p4est_ghost_t *ghost_tmp = NULL;
  p4est_nodes_t *nodes_tmp = NULL;

  Vec phi_tmp;
  for (int l=0; l<=sp->max_lvl; l++){
    my_p4est_partition(p4est_tmp, NULL);

    std::ostringstream oss;
    oss << "partial refinement of " << l << "/" << sp->max_lvl;
    w.start(oss.str());

    ghost_tmp = my_p4est_ghost_new(p4est_tmp, P4EST_CONNECT_FULL);
    nodes_tmp = my_p4est_nodes_new(p4est_tmp, ghost_tmp);

    InterpolatingFunctionNodeBase interp(p4est, nodes, ghost, brick, &ngbd);
#ifdef P4_TO_P8
    interp.set_input_parameters(phi, quadratic_non_oscillatory, phi_xx, phi_yy, phi_zz);
#else
    interp.set_input_parameters(phi, quadratic_non_oscillatory, phi_xx, phi_yy);
#endif
    double *phi_tmp_p;
    ierr = VecCreateGhostNodes(p4est_tmp, nodes_tmp, &phi_tmp); CHKERRXX(ierr);
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    // interpolate form old grid
    for (size_t n = 0; n<nodes_tmp->indep_nodes.elem_count; n++){
      p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_tmp->indep_nodes, n);
      p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

      p4est_topidx_t* t2v = p4est_tmp->connectivity->tree_to_vertex;
      double *t2c = p4est_tmp->connectivity->vertices;
      p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree

      double xyz [] =
      {
        node_x_fr_i(indep_node) + t2c[3 * tr_mm + 0],
        node_y_fr_j(indep_node) + t2c[3 * tr_mm + 1]
  #ifdef P4_TO_P8
        ,
        node_z_fr_k(indep_node) + t2c[3 * tr_mm + 2]
  #endif
      };

      interp.add_point_to_buffer(n, xyz);
    }
    interp.interpolate(phi_tmp_p);

    if(l == sp->max_lvl)
      break;

    // mark the cells for refinement
    splitting_criteria_marker_t markers(p4est_tmp, sp->min_lvl, sp->max_lvl, 1.2);
    p4est_locidx_t *q2n = nodes_tmp->local_nodes;

    for (p4est_topidx_t tr_it = p4est_tmp->first_local_tree; tr_it<= p4est_tmp->last_local_tree; tr_it++){
      p4est_tree_t *tree = (p4est_tree_t *)sc_array_index(p4est_tmp->trees, tr_it);
      for (size_t q = 0; q<tree->quadrants.elem_count; q++){
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
        p4est_locidx_t qu_idx = q + tree->quadrants_offset;
        double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

        double f[P4EST_CHILDREN];
        for (short i = 0; i<P4EST_CHILDREN; i++){
          f[i] = phi_tmp_p[q2n[P4EST_CHILDREN*qu_idx + i]];
          if (fabs(f[i]) <= 0.5*markers.lip*dx){
            markers[qu_idx] = P4EST_TRUE;
            continue;
          }
        }

#ifdef P4_TO_P8
        if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
            f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
        if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
        {
          markers[qu_idx] = P4EST_TRUE;
          continue;
        }
      }
    }

    // refine p4est
    p4est_tmp->user_pointer = &markers;
    my_p4est_refine(p4est_tmp, P4EST_FALSE, refine_marked_quadrants, NULL);

    p4est_nodes_destroy(nodes_tmp);
    p4est_ghost_destroy(ghost_tmp);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
    ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);

    w.stop(); w.read_duration();
  }

  p4est_destroy(p4est);
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);

  ierr = VecDestroy(phi_xx); CHKERRXX(ierr);
  ierr = VecDestroy(phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(phi_zz); CHKERRXX(ierr);
#endif

  p4est = p4est_tmp; p4est->user_pointer = sp;
  ghost = ghost_tmp;
  nodes = nodes_tmp;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_tmp;
}

void construct_grid_with_reinitializatrion2(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi)
{
  PetscErrorCode ierr;
  parStopWatch w;
  splitting_criteria_cf_t *sp = (splitting_criteria_cf_t*)p4est->user_pointer;

  for (int l=0; l<sp->min_lvl; l++)
    my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);

  for (int l=sp->min_lvl; l<=sp->max_lvl; l++){
    my_p4est_partition(p4est, NULL);

    std::ostringstream oss;
    oss << "partial refinement of " << l << "/" << sp->max_lvl;
    w.start(oss.str());

    if (ghost) p4est_ghost_destroy(ghost);
    if (nodes) p4est_nodes_destroy(nodes);
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    ngbd.init_neighbors();

    // sample level-set
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *sp->phi, phi);

    // reinitialize level-set function on this grid
    my_p4est_level_set ls(&ngbd);
    ls.reinitialize_1st_order_time_2nd_order_space(phi);

    if (l == sp->max_lvl)
      break;

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    // mark the cells for refinement
    splitting_criteria_marker_t markers(p4est, sp->min_lvl, sp->max_lvl, 1.2);
    p4est_locidx_t *q2n = nodes->local_nodes;

    for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it<= p4est->last_local_tree; tr_it++){
      p4est_tree_t *tree = (p4est_tree_t *)sc_array_index(p4est->trees, tr_it);
      for (size_t q = 0; q<tree->quadrants.elem_count; q++){
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
        p4est_locidx_t qu_idx = q + tree->quadrants_offset;
        double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

        double f[P4EST_CHILDREN];
        for (short i = 0; i<P4EST_CHILDREN; i++){
          f[i] = phi_p[q2n[P4EST_CHILDREN*qu_idx + i]];
          if (fabs(f[i]) <= 0.5*markers.lip*dx){
            markers[qu_idx] = P4EST_TRUE;
            continue;
          }
        }

#ifdef P4_TO_P8
        if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
            f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
        if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
        {
          markers[qu_idx] = P4EST_TRUE;
          continue;
        }
      }
    }
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    // new refinement
    p4est->user_pointer = &markers;
    my_p4est_refine(p4est, P4EST_FALSE, refine_marked_quadrants, NULL);
    w.stop(); w.read_duration();
  }

  p4est->user_pointer = sp;
}

void construct_grid_with_reinitializatrion3(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi)
{
  PetscErrorCode ierr;
  parStopWatch w;
  splitting_criteria_cf_t *sp = (splitting_criteria_cf_t*)p4est->user_pointer;

  for (int l=0; l<sp->max_lvl; l++){
    std::ostringstream oss;
    oss << "partial refinement of " << l+1 << "/" << sp->max_lvl;
    w.start(oss.str());
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, NULL);
    w.stop(); w.read_duration();
  }

  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
  ngbd.init_neighbors();

  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *sp->phi, phi);

  my_p4est_level_set ls(&ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
}

#ifdef P4_TO_P8
void motion_normal_direction(p8est_t *&p4est, p8est_ghost_t *&ghost, p8est_nodes_t *&nodes, my_p4est_brick_t *brick, const CF_3& cf, Vec &phi)
#else
void motion_normal_direction(p8est_t *&p4est, p8est_ghost_t *&ghost, p8est_nodes_t *&nodes, my_p4est_brick_t *brick, const CF_2& cf, Vec &phi)
#endif
{
  parStopWatch w;
  splitting_criteria_t *sp = (splitting_criteria_t*)p4est->user_pointer;
  p4est_connectivity_t *connectivity = p4est->connectivity;

  w.start("advecting in normal direction");
  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
  node_neighbors.init_neighbors();

  my_p4est_level_set level_set(&node_neighbors);

  level_set.advect_in_normal_direction(cf, phi);
  w.stop(); w.read_duration();

  w.start("reinitialization");
  level_set.reinitialize_1st_order_time_2nd_order_space(phi, 10);
  w.stop(); w.read_duration();

  w.start("grid adjustment");
  InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, brick, &node_neighbors);
  phi_interp.set_input_parameters(phi, quadratic_non_oscillatory);

  // refine and coarsen new p4est
  p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
  splitting_criteria_cf_t sp_cf(sp->min_lvl, sp->max_lvl, &phi_interp, sp->lip);
  p4est_np1->user_pointer = &sp_cf;

  my_p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset_cf, NULL);
  my_p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition
  my_p4est_partition(p4est_np1, NULL);

  // recompute ghost and nodes
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  // transfer solution to the new grid
  Vec phi_np1;
  PetscErrorCode ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; n++)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

    double xyz [] =
    {
      node_x_fr_i(node) + tree_xmin,
      node_y_fr_j(node) + tree_ymin
#ifdef P4_TO_P8
      ,
      node_z_fr_k(node) + tree_zmin
#endif
    };

    phi_interp.add_point_to_buffer(n, xyz);
  }
  phi_interp.interpolate(phi_np1);

  // get rid of old stuff and replace them with new
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;
  p4est_destroy(p4est); p4est = p4est_np1; p4est->connectivity = connectivity; p4est->user_pointer = sp;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  w.stop();; w.read_duration();
}

void motion_normal_direction(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t *myb, Vec &phi, int itmax)
{
  constant_cf vn_f(-1.0);
  constant_cf vn_b( 1.0);
  PetscErrorCode ierr;

  // forward motion
  std::ostringstream oss; oss << output_dir + "/normal_" << p4est->mpisize <<
                                 "p" << ".0";

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_FALSE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  for (int it = 0; it<itmax; it++){
    motion_normal_direction(p4est, ghost, nodes, myb, vn_f, phi);

    std::ostringstream oss; oss << output_dir + "/normal_" << p4est->mpisize <<
                                   "p" << "." << it+1;

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  }

  // backward motion
  for (int it = itmax; it<2*itmax; it++){
    motion_normal_direction(p4est, ghost, nodes, myb, vn_b, phi);

    std::ostringstream oss; oss << output_dir + "/normal_" << p4est->mpisize <<
                                   "p" << "." << it+1;

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  }
}
