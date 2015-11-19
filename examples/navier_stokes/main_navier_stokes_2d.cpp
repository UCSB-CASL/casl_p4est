
/*
 * The navier stokes solver
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

// casl library
#include <lib/arrays/ArrayV.h>
#include <lib/amr/QuadTree.h>
#include <lib/tools/QuadraticInterpolationOnQuadTree.h>

#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

double xmin = 0;
double xmax = 1;
double ymin = 0;
double ymax = 1;
#ifdef P4_TO_P8
double zmin = 0;
double zmax = 1;
#endif

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

/*
 *  ********* 2D *********
 * 0 - analytic vortex without time dependence
 * 1 - analytic vortex with time dependence
 * 2 - driven cavity
 * 3 - driven cavity with a hole in the middle
 * 4 - karman street
 * 5 - oscillating cylinder
 * 6 - naca
 * 7 - smoke drop
 * 8 - smoke on hill
 * 9 - rotating cylinder
 *
 *  ********* 3D *********
 * 0 - analytic vortex without time dependence
 * 1 - analytic vortex with time dependence
 * 2 - karman
 * 3 - smoke drop
 * 4 - smoke hill
 * 5 - oscillating sphere
 * 6 - smoke city
 */

int test_number;

double mu;
double rho;
double tn;
double dt;
double u0;
double r0;

/* for oscillating cylinder */
double KC;
double X0;
double f0;

#ifdef P4_TO_P8

class INIT_SMOKE : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0:
    case 1: return (sqrt(SQR(x-PI) + SQR(y-PI) + SQR(z-PI))<.2) ? 1 : 0;
    case 2: return (x<2 && sqrt(SQR(y) + SQR(z))<.4) ? 1 : 0;
    case 3: return (sqrt(SQR(x-.5) + +SQR(y-.5) + SQR(z-.75))<.1) ? 1 : 0;
    case 4: return (0.01<x && x<.4 && z<.9 && -.5<y && y<.5 && x+z>.51) ? 1 : 0;
    case 5: return sqrt(x*x + y*y + z*z)<.05 ? 1 : 0;
    case 6: return (x<.1 && -.5<y && y<.5 && .2<z && z<.5) ? 1 : 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} init_smoke;

class BC_SMOKE : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0:
    case 1: return 0;
    case 2: return (fabs(x-xmin)<EPS && sqrt(SQR(y) + SQR(z))<.4) ? 1 : 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return (x<.1 && -.5<y && y<.5 && .2<z && z<.4) ? 1 : 0;
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
    double d;
    double xc[] = {1.5, 1.9 , 1.6 , 2.0 , 1.3,   1.25, 2.2 , 1.65, 0.62, 0.8, 0.7, 0.9, 1.05}; /* building x center */
    double yc[] = {0.5,-0.3 , 0.01, 0.4 ,-0.6,  -0.1 ,-0.05,-0.5 ,-0.45, 0.2, 0.5,-0.2, 0.3}; /* building y center */
    double zc[] = {0.602, 0.812 , 0.642 , 1.065, 0.8052,   0.751, 0.6832, 0.7212, 0.96, 0.67, 0.86, 0.94, 1.11}; /* building z center */
    double lc[] = {0.1, 0.08, 0.09, 0.092, 0.085, 0.11, 0.08, 0.078, 0.095, 0.07, 0.06, 0.08, 0.072}; /* building half width */
    switch(test_number)
    {
    case 0: return cos(x)*cos(y)*cos(z) + .4;
    case 1: return cos(x)*cos(y)*cos(z) + .4;
    case 2: return r0 - sqrt(SQR(x-(xmax+xmin)/4) + SQR(y-(ymax+ymin)/2) + SQR(z-(zmax+zmin)/2));
    case 3: return -1;
    case 4:
      d = .5-(x+z);
      for(int i=0; i<8; ++i)
        d = MAX(d, MIN(-z+zc[i], MIN(x-(xc[i]-lc[i]),-x+(xc[i]+lc[i])), MIN(y-(yc[i]-lc[i]),-y+(yc[i]+lc[i]))));
//      d = MAX(d, -x+(xmin+0.01), x-(xmax-0.01));
//      d = MAX(d, -y+(ymin+0.01), y-(ymax-0.01));
//      d = MAX(d, -z+(zmin+0.01), z-(zmax-0.01));
      return d;
    case 5: return r0 - sqrt(SQR(x-X0*(1-cos(2*PI*f0*(tn+dt)))+X0) + SQR(y) + SQR(z));
    case 6:
      d = -100;
      for(int i=0; i<13; ++i)
        d = MAX(d, MIN(-z+zc[i], MIN(x-(xc[i]-lc[i]),-x+(xc[i]+lc[i])), MIN(y-(yc[i]-lc[i]),-y+(yc[i]+lc[i]))));
      return d;
    default: throw std::invalid_argument("Choose a valid test.");
    }
  }
} level_set;

struct BCWALLTYPE_P : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    switch(test_number)
    {
    case 0: return NEUMANN;
    case 1: return NEUMANN;
    case 2: if(fabs(x-xmax)<EPS) return DIRICHLET; return NEUMANN;
    case 3: return NEUMANN;
    case 4: return NEUMANN;
    case 5: return NEUMANN;
    case 6: if(fabs(x-xmax)<EPS) return DIRICHLET; return NEUMANN;
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
    case 1: return 0;
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
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
    case 0: return 0;
    case 1: return 0;
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_p;

struct BCWALLTYPE_U : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    switch(test_number)
    {
    case 0: return DIRICHLET;
    case 1: return DIRICHLET;
    case 2: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 3: return DIRICHLET;
    case 4: return DIRICHLET;
    case 5: return DIRICHLET;
    case 6: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    switch(test_number)
    {
    case 0: return DIRICHLET;
    case 1: return DIRICHLET;
    case 2: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 3: return DIRICHLET;
    case 4: return DIRICHLET;
    case 5: return DIRICHLET;
    case 6: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_v;

struct BCWALLTYPE_W : WallBC3D
{
  BoundaryConditionType operator()(double x, double, double) const
  {
    switch(test_number)
    {
    case 0: return DIRICHLET;
    case 1: return DIRICHLET;
    case 2: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 3: return DIRICHLET;
    case 4: return DIRICHLET;
    case 5: return DIRICHLET;
    case 6: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_w;

struct BCWALLVALUE_U : CF_3
{
  double operator()(double x, double, double) const
  {
    switch(test_number)
    {
    case 0: return 0;
    case 1: return 0;
    case 2: if(fabs(x-xmax)<EPS) return 0; else return u0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: if(fabs(x-xmax)<EPS) return 0; else return u0;
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
    case 0: return 0;
    case 1: return 0;
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
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
    case 0: return 0;
    case 1: return 0;
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_w;

struct BCINTERFACE_VALUE_U : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return cos(x)*sin(y)*sin(z);
    case 1: return cos(x)*sin(y)*sin(z)*cos(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return u0*sin(2*PI*f0*tn);
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y)*sin(z);
    case 1: return sin(x)*cos(y)*sin(z)*cos(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_v;

struct BCINTERFACE_VALUE_W : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return -2*sin(x)*sin(y)*cos(z);
    case 1: return -2*sin(x)*sin(y)*cos(z)*cos(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_w;

struct initial_velocity_unm1_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return cos(x)*sin(y)*sin(z);
    case 1: return cos(x)*sin(y)*sin(z)*cos(-dt);
    case 2: return u0;
    case 3: return 0;
    case 4: return 0;
    case 5: return u0*sin(2*PI*f0*(tn-2*dt));
    case 6: return u0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return cos(x)*sin(y)*sin(z);
    case 1: return cos(x)*sin(y)*sin(z)*cos(0);
    case 2: return u0;
    case 3: return 0;
    case 4: return 0;
    case 5: return u0*sin(2*PI*f0*(tn-dt));
    case 6: return u0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y)*sin(z);
    case 1: return sin(x)*cos(y)*sin(z)*cos(-dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vnm1;

struct initial_velocity_v_n_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y)*sin(z);
    case 1: return sin(x)*cos(y)*sin(z)*cos(0);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vn;

struct initial_velocity_wnm1_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return -2*sin(x)*sin(y)*cos(z);
    case 1: return -2*sin(x)*sin(y)*cos(z)*cos(-dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_wnm1;

struct initial_velocity_w_n_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return -2*sin(x)*sin(y)*cos(z);
    case 1: return -2*sin(x)*sin(y)*cos(z)*cos(0);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_wn;

struct external_force_u_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return (-  rho*cos(x)*sin(y)*sin(z) * sin(x)*sin(y)*sin(z)
                    +  rho*sin(x)*cos(y)*sin(z) * cos(x)*cos(y)*sin(z)
                    -2*rho*sin(x)*sin(y)*cos(z) * cos(x)*sin(y)*cos(z)
                    +3*mu *cos(x)*sin(y)*sin(z) );
    case 1: return (-  rho*cos(x)*sin(y)*sin(z)*sin(tn+dt)
                    -  rho*cos(x)*sin(y)*sin(z)*cos(tn+dt) * sin(x)*sin(y)*sin(z)*cos(tn+dt)
                    +  rho*sin(x)*cos(y)*sin(z)*cos(tn+dt) * cos(x)*cos(y)*sin(z)*cos(tn+dt)
                    -2*rho*sin(x)*sin(y)*cos(z)*cos(tn+dt) * cos(x)*sin(y)*cos(z)*cos(tn+dt)
                    +3*mu *cos(x)*sin(y)*sin(z)*cos(tn+dt));
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

struct external_force_v_t : CF_3
{
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return (   rho*cos(x)*sin(y)*sin(z) * cos(x)*cos(y)*sin(z)
                    -  rho*sin(x)*cos(y)*sin(z) * sin(x)*sin(y)*sin(z)
                    -2*rho*sin(x)*sin(y)*cos(z) * sin(x)*cos(y)*cos(z)
                    +3*mu *sin(x)*cos(y)*sin(z) );
    case 1: return (-  rho*sin(x)*cos(y)*sin(z)*sin(tn+dt)
                    +  rho*cos(x)*sin(y)*sin(z)*cos(tn+dt) * cos(x)*cos(y)*sin(z)*cos(tn+dt)
                    -  rho*sin(x)*cos(y)*sin(z)*cos(tn+dt) * sin(x)*sin(y)*sin(z)*cos(tn+dt)
                    -2*rho*sin(x)*sin(y)*cos(z)*cos(tn+dt) * sin(x)*cos(y)*cos(z)*cos(tn+dt)
                    +3*mu *sin(x)*cos(y)*sin(z)*cos(tn+dt));
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

struct external_force_w_t : CF_3
{
private:
  my_p4est_interpolation_nodes_t *interp;
public:
  external_force_w_t(my_p4est_node_neighbors_t *ngbd_n, Vec smoke)
  {
    interp = new my_p4est_interpolation_nodes_t(ngbd_n);
    interp->set_input(smoke, linear);
  }
  ~external_force_w_t()
  {
    delete interp;
  }
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0: return (-  rho*cos(x)*sin(y)*sin(z) * 2*cos(x)*sin(y)*cos(z)
                    -  rho*sin(x)*cos(y)*sin(z) * 2*sin(x)*cos(y)*cos(z)
                    -2*rho*sin(x)*sin(y)*cos(z) * 2*sin(x)*sin(y)*sin(z)
                    -6*mu *sin(x)*sin(y)*cos(z) );
    case 1: return ( 2*rho*sin(x)*sin(y)*cos(z)*sin(tn+dt)
                    -  rho*cos(x)*sin(y)*sin(z)*cos(tn+dt) * 2*cos(x)*sin(y)*cos(z)*cos(tn+dt)
                    -  rho*sin(x)*cos(y)*sin(z)*cos(tn+dt) * 2*sin(x)*cos(y)*cos(z)*cos(tn+dt)
                    -2*rho*sin(x)*sin(y)*cos(z)*cos(tn+dt) * 2*sin(x)*sin(y)*sin(z)*cos(tn+dt)
                    -6*mu *sin(x)*sin(y)*cos(z)*cos(tn+dt));
    case 2: return 0;
    case 3: return -(*interp)(x,y,z)*9.81;
    case 4: return -(rho+10*(*interp)(x,y,z))*9.81;
    case 5: return 0;
    case 6: return 0;
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
    case 1: return (sqrt(SQR(x-PI/2) + SQR(y-3*PI/4))<.2) ? 1 : 0;
    case 2:
    case 3: return (sqrt(SQR(x-.5) + SQR(y-.85))<.05) ? 1 : 0;
    case 4: return (x<2 && y>-1 && y<1) ? 1 : 0;
    case 5: return (sqrt(SQR(x) + SQR(y))<.05) ? 1 : 0;
    case 6: return (x<2 && y>-.5 && y<.5) ? 1 : 0;
    case 7: return (sqrt(SQR(x-.5) + SQR(y-.75))<.1) ? 1 : 0;
    case 8: return (x>0.01 && x<.4 && y<.9 && x+y>.5) ? 1 : 0;
    case 9: return (sqrt(x*x+y*y)<.1) ? 1 : 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} init_smoke;


class BC_SMOKE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0:
    case 1:
    case 2:
    case 3:
    case 5: return 0;
    case 4:
    case 6: return(fabs(x-xmin)<EPS && y>-.5 && y<.5) ? 1 : 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_smoke;

class NACA_SAMPLED : public CF_2
{
private:
  vector<double> sample;
  unsigned int N;
  double naca_length;
  double naca_number;
  double naca_angle;
  double x_edge;
public:
  NACA_SAMPLED(double number, double length, double angle)
  {
    N = 10000;
    sample.resize(N);
    lip = 1.2;
    naca_length = length;
    naca_number = number;
    naca_angle = angle;
    x_edge = 8;

    double dx = naca_length/(double)(N-1);

    for(unsigned int i=0; i<N; ++i)
    {
      double x = i*dx;
      double r = x/naca_length;
      sample[i] = naca_number/100/.2 * naca_length * (.2969*sqrt(r) - .126*r -.3516*r*r + .2843*r*r*r - .1015*r*r*r*r);
    }
  }

  double operator()(double x, double y) const
  {
    x -= x_edge;

    /* apply the rotation */
    x -= naca_length/2;
    double theta = PI*naca_angle/180;
    double x_rot = x*cos(theta) - y*sin(theta);
    double y_rot = x*sin(theta) + y*cos(theta);
    x = x_rot + naca_length/2;
    y = y_rot;

    double dx = naca_length/(double) (N-1);
    double dist = DBL_MAX;
    for(unsigned int i=0; i<N; ++i)
    {
      double xt = i*dx;
      double yt = sample[i];
      dist = MIN(dist, sqrt(SQR(x-xt)+SQR(y-yt)), sqrt(SQR(x-xt)+SQR(y+yt)));
    }

    if(x<0 || x>naca_length) return -dist;

    double r = x/naca_length;
    double yt = naca_number/100/.2 * naca_length * (.2969*sqrt(r) - .126*r -.3516*r*r + .2843*r*r*r - .1015*r*r*r*r);
    if(-yt<y && y<yt) return  dist;
    else              return -dist;
  }
};


class NACA_CASL : public CF_2
{
private:
  CASL::QuadTree tr;
  CASL::ArrayV<double> phi;
  CASL::QuadraticInterpolationOnQuadTree interp;
  double x_edge;
  double naca_length;
  double angle;
public:
  NACA_CASL(MPI_Comm comm, int number, double length, double angle)
  {
    (void) length;
    char path_to_naca[1000];
#if defined(STAMPEDE)
    sprintf(path_to_naca, "/work/02673/guittet/code/data/casl_naca/naca_%04d_lmax15_sampling500000", number);
#elif defined(COMET)
    sprintf(path_to_naca, "/home/guittet/code/data/casl_naca/naca_%04d_lmax15_sampling500000", number);
#else
    sprintf(path_to_naca, "/home/guittet/code/data/casl_naca/naca_%04d_lmax15_sampling500000", number);
#endif

    PetscErrorCode ierr;
    ierr = PetscPrintf(comm, "importing the naca model for file: %s\n", path_to_naca); CHKERRXX(ierr);

    lip = 1.2;
    char name[1000];
    sprintf(name, "%s_tree.dat", path_to_naca);
    tr.load(name);
    sprintf(name, "%s_phi.dat", path_to_naca);
    phi.load(name);

    interp.set(tr,phi);
    this->angle = angle;
    this->x_edge = 8;
    this->naca_length = 4;
  }
  double operator()(double x, double y) const
  {
    double theta = PI*angle/180;
    x -= x_edge;
    x -= naca_length/2;

    double x_ = x*cos(theta) - y*sin(theta);
    double y_ = x*sin(theta) + y*cos(theta);
    x_ += naca_length/2;
    x_ += x_edge;
    return interp(x_,y_);
  }
};

CF_2 *naca;


class LEVEL_SET: public CF_2
{
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return -sin(x)*sin(y) + .2;
    case 1: return -sin(x)*sin(y) + .2;
    case 2: return -1;
    case 3: return r0 - sqrt(SQR(x-(xmax+xmin)/2) + SQR(y-(ymax+ymin)/2));
    case 4: return r0 - sqrt(SQR(x-(xmax+xmin)/4) + SQR(y-(ymax+ymin)/2));
    case 5: return r0 - sqrt(SQR(x-X0*(1-cos(2*PI*f0*(tn+dt)))+X0) + SQR(y));
    case 6: return (*naca)(x,y);
    case 7: return -1;
    case 8: return .5-(x+y);
    case 9: return r0 - sqrt(SQR(x-.5*cos(tn+dt)) + SQR(y-.5*sin(tn+dt)));
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} level_set;

struct BCWALLTYPE_P : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    switch(test_number)
    {
    case 0: return NEUMANN;
    case 1: return NEUMANN;
    case 2: return NEUMANN;
    case 3: return NEUMANN;
    case 4: if(fabs(x-xmax)<EPS) return DIRICHLET; return NEUMANN;
    case 5: return NEUMANN;
    case 6: if(fabs(x-xmax)<EPS) return DIRICHLET; return NEUMANN;
    case 7: return NEUMANN;
    case 8: return NEUMANN;
    case 9: return NEUMANN;
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
    case 0: return 0;
    case 1: return 0;
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
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
    case 0: return 0;
    case 1: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
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
    case 0: return DIRICHLET;
    case 1: return DIRICHLET;
    case 2: return DIRICHLET;
    case 3: return DIRICHLET;
    case 4: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 5: return DIRICHLET;
    case 6: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 7: return DIRICHLET;
    case 8: return DIRICHLET;
    case 9: return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC2D
{
  BoundaryConditionType operator()(double x, double) const
  {
    switch(test_number)
    {
    case 0: return DIRICHLET;
    case 1: return DIRICHLET;
    case 2: return DIRICHLET;
    case 3: return DIRICHLET;
    case 4: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 5: return DIRICHLET;
    case 6: if(fabs(x-xmax)<EPS) return NEUMANN; return DIRICHLET;
    case 7: return DIRICHLET;
    case 8: return DIRICHLET;
    case 9: return DIRICHLET;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_type_v;

struct BCWALLVALUE_U : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y);
    case 1: return sin(x)*cos(y)*cos(tn+dt);
    case 2: if(fabs(y-ymax)<EPS) return u0; else return 0;
    case 3: if(fabs(y-ymax)<EPS) return u0; else return 0;
    case 4: if(fabs(x-xmax)<EPS) return 0; else return u0;
    case 5: return 0;
    case 6: if(fabs(x-xmax)<EPS) return 0; else return u0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return -cos(x)*sin(y);
    case 1: return -cos(x)*sin(y)*cos(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_wall_value_v;

struct BCINTERFACE_VALUE_U : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y);
    case 1: return sin(x)*cos(y)*cos(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return u0*sin(2*PI*f0*tn);
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return -y;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return -cos(x)*sin(y);
    case 1: return -cos(x)*sin(y)*cos(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return x;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} bc_interface_value_v;

struct initial_velocity_unm1_t : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y);
    case 1: return sin(x)*cos(y)*cos(-dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return u0;
    case 5: return u0*sin(2*PI*f0*(tn-2*dt));
    case 6: return u0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return sin(x)*cos(y);
    case 1: return sin(x)*cos(y)*cos(0);
    case 2: return 0;
    case 3: return 0;
    case 4: return u0;
    case 5: return u0*sin(2*PI*f0*(tn-dt));
    case 6: return u0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return -cos(x)*sin(y);
    case 1: return -cos(x)*sin(y)*cos(-dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return -cos(x)*sin(y);
    case 1: return -cos(x)*sin(y)*cos(0);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} initial_velocity_vn;

struct external_force_u_t : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return rho*sin(x)*cos(x) + 2*mu*sin(x)*cos(y);
    case 1: return rho*sin(x)*cos(x)*SQR(cos(tn+dt)) + 2*mu*sin(x)*cos(y)*cos(tn+dt) - rho*sin(x)*cos(y)*sin(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return 0;
    case 8: return 0;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
} external_force_u;

struct external_force_v_t : CF_2
{
private:
  my_p4est_interpolation_nodes_t *interp;
public:
  external_force_v_t(my_p4est_node_neighbors_t *ngbd_n, Vec smoke)
  {
    interp = new my_p4est_interpolation_nodes_t(ngbd_n);
    interp->set_input(smoke, linear);
  }
  ~external_force_v_t()
  {
    delete interp;
  }

  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0: return rho*sin(y)*cos(y) - 2*mu*cos(x)*sin(y);
    case 1: return rho*sin(y)*cos(y)*SQR(cos(tn+dt)) - 2*mu*cos(x)*sin(y)*cos(tn+dt) + rho*cos(x)*sin(y)*sin(tn+dt);
    case 2: return 0;
    case 3: return 0;
    case 4: return 0;
    case 5: return 0;
    case 6: return 0;
    case 7: return -(*interp)(x,y)*9.81;
    case 8: return -(rho+1000*(*interp)(x,y))*9.81;
    case 9: return 0;
    default: throw std::invalid_argument("choose a valid test.");
    }
  }
};

#endif



void check_error_analytic_vortex(mpi_context_t *mpi, my_p4est_navier_stokes_t *ns)
{
  PetscErrorCode ierr;
  int mpiret;

  const p4est_t *p4est = ns->get_p4est();
  p4est_ghost_t *ghost = ns->get_ghost();
  p4est_nodes_t *nodes = ns->get_nodes();
  const Vec *v = ns->get_velocity_np1();

  const double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  double err_v[P4EST_DIM]; for(int dir=0; dir<P4EST_DIM; ++dir) err_v[dir]=0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz);

#ifdef P4_TO_P8
    if(level_set(xyz[0], xyz[1], xyz[2])<0)
#else
    if(level_set(xyz[0], xyz[1])<0)
#endif
    {
      double v_ex;

#ifdef P4_TO_P8
      if(test_number==0) v_ex = cos(xyz[0])*sin(xyz[1])*sin(xyz[2]);
      else               v_ex = cos(xyz[0])*sin(xyz[1])*sin(xyz[2])*cos(tn);
      err_v[0] = MAX(err_v[0], fabs(v_p[0][n]-v_ex));

      if(test_number==0) v_ex = sin(xyz[0])*cos(xyz[1])*sin(xyz[2]);
      else               v_ex = sin(xyz[0])*cos(xyz[1])*sin(xyz[2])*cos(tn);
      err_v[1] = MAX(err_v[1], fabs(v_p[1][n]-v_ex));

      if(test_number==0) v_ex = -2*sin(xyz[0])*sin(xyz[1])*cos(xyz[2]);
      else               v_ex = -2*sin(xyz[0])*sin(xyz[1])*cos(xyz[2])*cos(tn);
      err_v[2] = MAX(err_v[2], fabs(v_p[2][n]-v_ex));
#else
      if(test_number==0) v_ex = sin(xyz[0])*cos(xyz[1]);
      else               v_ex = sin(xyz[0])*cos(xyz[1])*cos(tn);
      err_v[0] = MAX(err_v[0], fabs(v_p[0][n]-v_ex));

      if(test_number==0) v_ex = -cos(xyz[0])*sin(xyz[1]);
      else               v_ex = -cos(xyz[0])*sin(xyz[1])*cos(tn);
      err_v[1] = MAX(err_v[1], fabs(v_p[1][n]-v_ex));
#endif
    }
  }

  mpiret = MPI_Allreduce(MPI_IN_PLACE, err_v, P4EST_DIM, MPI_DOUBLE, MPI_MAX, mpi->mpicomm); SC_CHECK_MPI(mpiret);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi->mpicomm, "Error on velocity in direction %d : %.5e\n", dir, err_v[dir]); CHKERRXX(ierr);
  }

  Vec hodge = ns->get_hodge();
  double err_h = 0;
  const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;
      double xyz[P4EST_DIM];
      quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost ,xyz);
#ifdef P4_TO_P8
      if(level_set(xyz[0], xyz[1], xyz[2])<0)
#else
      if(level_set(xyz[0], xyz[1])<0)
#endif
        err_h = MAX(err_h, fabs(hodge_p[quad_idx]));
    }
  }
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_h, 1, MPI_DOUBLE, MPI_MAX, mpi->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = PetscPrintf(mpi->mpicomm, "Error on hodge : %.5e\n", err_h); CHKERRXX(ierr);
}


#ifndef P4_TO_P8
void check_velocity_cavity(mpi_context_t *mpi, my_p4est_navier_stokes_t *ns, double Re, double n_times_dt)
{
  PetscErrorCode ierr;

  const my_p4est_node_neighbors_t *ngbd_n = ns->get_ngbd_n();
  const Vec *vn = ns->get_velocity_np1();
  const p4est_t *p4est = ns->get_p4est();
  Vec phi = ns->get_phi();

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  my_p4est_interpolation_nodes_t interp0(ngbd_n);
  my_p4est_interpolation_nodes_t interp1(ngbd_n);
  my_p4est_interpolation_nodes_t interp0_phi(ngbd_n);
  my_p4est_interpolation_nodes_t interp1_phi(ngbd_n);
  int N = 200;
  for(int i=0; i<=N; ++i)
  {
    double xyz0[] = { (double)i/(double)N, .5 };
    interp0.add_point(i, xyz0);
    interp0_phi.add_point(i, xyz0);

    double xyz1[] = { .5, (double)i/(double)N };
    interp1.add_point(i, xyz1);
    interp1_phi.add_point(i, xyz1);
  }

  std::vector<double> v0(N+1);
  interp0.set_input(vn[1], quadratic);
  interp0.interpolate(v0.data());

  std::vector<double> v1(N+1);
  interp1.set_input(vn[0], quadratic);
  interp1.interpolate(v1.data());

  std::vector<double> phi0(N+1);
  interp1.set_input(phi, quadratic);
  interp1.interpolate(phi0.data());

  std::vector<double> phi1(N+1);
  interp1.set_input(phi, quadratic);
  interp1.interpolate(phi1.data());

  if(!mpi->mpirank)
  {
    FILE* fp;
    char name[1000];
#if defined(STAMPEDE) || defined(COMET)
    char *out_dir;
    out_dir = getenv("OUT_DIR");
    if     (test_number==2) sprintf(name, "%s/velo_driven_cavity.dat", out_dir);
    else if(test_number==3) sprintf(name, "%s/velo_driven_cavity_hole.dat", out_dir);
#else
    if     (test_number==2) sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/driven_cavity/cavity_velocity_%d-%d_%dx%d_Re%g_ntimesdt%g.dat", data->min_lvl, data->max_lvl, nx, ny, Re, n_times_dt);
    else if(test_number==3) sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/driven_cavity_hole/cavity_hole_velocity_%d-%d_%dx%d_Re%g_ntimesdt%g.dat", data->min_lvl, data->max_lvl, nx, ny, Re, n_times_dt);
#endif
    fp = fopen(name, "w");

    if(fp==NULL)
      throw std::invalid_argument("check_forces_cavity: could not open file.");

    ierr = PetscFPrintf(mpi->mpicomm, fp, "%% x/y \t vx \t vy\n"); CHKERRXX(ierr);
    for(int i=0; i<=N; ++i)
    {
      ierr = PetscFPrintf(mpi->mpicomm, fp, "%g, %g, %g\n", (double)i/(double)N, phi0[i]<0 ? v0[i] : 0, phi1[i]<0 ? v1[i] : 0); CHKERRXX(ierr);
    }

    fclose(fp);
  }
}
#endif



int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "additional splits to lmin and lmax");
  cmd.add_option("nx", "the number of trees in the x direction");
  cmd.add_option("ny", "the number of trees in the y direction");
#ifdef P4_TO_P8
  cmd.add_option("nz", "the number of trees in the z direction");
#endif
  cmd.add_option("tf", "the final time");
  cmd.add_option("Re", "the reynolds number");
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate)");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx");
  cmd.add_option("n_times_dt", "dx = n_times_dt * dx/vmax");
  cmd.add_option("thresh", "the threshold used for the refinement criteria");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("save_every_n", "export images every n iterations");
  cmd.add_option("smoke", "0 - no smoke, 1 - with smoke");
  cmd.add_option("smoke_thresh", "threshold for smoke refinement");
  cmd.add_option("refine_with_smoke", "refine the grid with the smoke density and threshold smoke_thresh");
#ifndef P4_TO_P8
  cmd.add_option("naca_angle", "angle of the naca airfoil for test 6");
  cmd.add_option("naca_number", "number of the naca, see naca reference online. Default is 0015");
#endif
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
                 0 - analytic vortex without time dependence\n\
                 1 - analytic vortex with time dependence\n\
                 2 - karman\n\
                 3 - smoke drop\n\
                 4 - smoke on hill\n\
                 5 - oscillating sphere\n\
                 6 - smoke city\n");
#else
  cmd.add_option("test", "choose a test.\n\
                 0 - analytic vortex\n\
                 1 - analytic vortex with time dependence\n\
                 2 - driven cavity\n\
                 3 - driven cavity with hole\n\
                 4 - karman street\n\
                 5 - oscillating cylinder\n\
                 6 - naca airfoil\n\
                 7 - smoke drop\n\
                 8 - smoke on hill\n\
                 9 - rotating cylinder\n");
#endif
  cmd.parse(argc, argv);

  cmd.print();

  int sl_order = cmd.get("sl_order", 2);
  int nb_splits = cmd.get("nb_splits", 0);
  int lmin = cmd.get("lmin", 3) + nb_splits;
  int lmax = cmd.get("lmax", 3) + nb_splits;
  double n_times_dt = cmd.get("n_times_dt", 1.);
  double threshold_split_cell = cmd.get("thresh", 0.1);
  bool save_vtk = cmd.get("save_vtk", false);
  int save_every_n = cmd.get("save_every_n", 1);
  test_number = cmd.get("test", 1);

  bool is_smoke = cmd.get("smoke", 0);
  bool refine_with_smoke = cmd.get("refine_with_smoke", 0);
  double smoke_thresh = cmd.get("smoke_thresh", .5);

#ifndef P4_TO_P8
  double naca_length = 4;
#endif

  double Re;
  rho = 1;
  u0 = 0;
  double tf;

  switch(test_number)
  {
#ifdef P4_TO_P8
  case 0: nx=1; ny=1; nz=1; xmin=PI/2; xmax=3*PI/2; ymin=PI/2; ymax=3*PI/2; zmin=PI/2; zmax=3*PI/2; Re=0; mu = rho = 1; u0 = 1; tf=cmd.get("tf",PI/3); break;
  case 1: nx=1; ny=1; nz=1; xmin=PI/2; xmax=3*PI/2; ymin=PI/2; ymax=3*PI/2; zmin=PI/2; zmax=3*PI/2; Re=0; mu = rho = 1; u0 = 1; tf=cmd.get("tf",PI/3); break;
  case 2: nx=8; ny=4; nz=4; xmin=   0; xmax=    32; ymin=  -8; ymax=     8; zmin=  -8; zmax=     8; Re=cmd.get("Re",350); r0=1; u0=1; rho=1; mu=2*r0*rho*u0/Re; tf=cmd.get("tf",200); break;
  case 3: nx=2; ny=2; nz=2; xmin=   0; xmax=     1; ymin=   0; ymax=     1; zmin   =0; zmax=     1; Re=cmd.get("Re",5000); u0=rho=1; mu=rho*u0*(zmax-zmin)/Re; tf=cmd.get("tf",5); break;
  case 4: nx=3; ny=2; nz=1; xmin=   0; xmax=     3; ymin=  -1; ymax=     1; zmin=   0; zmax=     1; Re=cmd.get("Re",5000); u0=rho=1; mu=rho*u0*(xmax-xmin)/Re; tf=cmd.get("tf",5); u0=500; break;
  case 5: nx=1; ny=1; nz=1; xmin=  -1; xmax=     1; ymin=  -1; ymax=     1; zmin=  -1; zmax=     1; Re=cmd.get("Re", 80); r0=0.1; u0=Re/(2*r0); X0=0.125*2*r0; mu=rho=1; f0=u0*1.5/(2*r0); tf=cmd.get("tf", 3/f0); break;
  case 6: nx=6; ny=4; nz=2; xmin=   0; xmax=     6; ymin=  -2; ymax=     2; zmin=   0; zmax=     2; Re=cmd.get("Re",5000); u0=rho=1; mu=rho*u0*(xmax-xmin)/Re; tf=cmd.get("tf",200); break;
#else
  case 0: nx=1; ny=1; xmin = 0; xmax = PI; ymin = 0; ymax = PI; Re = 0; mu = rho = 1; u0 = 1; tf = cmd.get("tf", PI/3);  break;
  case 1: nx=1; ny=1; xmin = 0; xmax = PI; ymin = 0; ymax = PI; Re = 0; mu = rho = 1; u0 = 1; tf = cmd.get("tf", PI/3);  break;
  case 2: nx=2; ny=2; xmin = 0; xmax =  1; ymin = 0; ymax =  1; Re = cmd.get("Re", 1000); u0 = 1; rho = 1; mu = rho*u0*(xmax-xmin)/Re; tf = cmd.get("tf", 37);    break;
  case 3: nx=2; ny=2; xmin = 0; xmax =  1; ymin = 0; ymax =  1; Re = cmd.get("Re", 1000); r0 = 0.25; u0 = 1; rho = 1; mu = rho*u0*1/Re; tf = cmd.get("tf", 37);    break;
  case 4: nx=8; ny=4; xmin = 0; xmax = 32; ymin =-8; ymax =  8; Re = cmd.get("Re", 200);  r0 = 0.5 ; u0 = 1; rho = 1; mu = 2*r0*rho*u0/Re; tf = cmd.get("tf", 200);   break;
  case 5: nx=2; ny=2; xmin =-1; xmax =  1; ymin =-1; ymax =  1; Re = cmd.get("Re", 100);  r0 = 0.05; KC = 5; X0 = .7957*2*r0; Re = 100; u0 = Re/(2*r0); mu = rho = 1; f0 = u0/(KC*2*r0); tf = cmd.get("tf", 3/f0); break;
  case 6: nx=8; ny=4; xmin = 0; xmax = 32; ymin =-8; ymax =  8; Re = cmd.get("Re", 5300); u0 = 1; rho = 1; mu = rho*u0*naca_length/Re; tf = cmd.get("tf", 200); break;
  case 7: nx=2; ny=2; xmin = 0; xmax =  1; ymin = 0; ymax =  1; Re = cmd.get("Re", 5000); u0 = 1; rho = 1; mu = rho*u0*(xmax-xmin)/Re; tf = cmd.get("tf", 5); break;
  case 8: nx=6; ny=2; xmin = 0; xmax =  3; ymin = 0; ymax =  1; Re = cmd.get("Re", 5000); u0 = 1; rho = 1; mu = rho*u0*(xmax-xmin)/Re; tf = cmd.get("tf", 5); u0=100; break;
  case 9: nx=2; ny=2; xmin =-1; xmax =  1; ymin =-1; ymax =  1; Re = cmd.get("Re", 5000); u0 = 1; r0 = .2; rho = 1; mu = 2*r0*rho*u0/Re; tf = cmd.get("tf", 500); break;
#endif
  default: throw std::invalid_argument("choose a valid test.");
  }

#ifndef P4_TO_P8
  double naca_angle = cmd.get("naca_angle", 15.);
  int naca_number = cmd.get("naca_number", 12);
  if(test_number==6)
  {
//    naca = new NACA_SAMPLED(naca_number, naca_length, naca_angle);
    naca = new NACA_CASL(mpi->mpicomm, naca_number, naca_length, naca_angle);
  }
#endif

  tf = cmd.get("tf", tf);
  nx = cmd.get("nx", nx);
  ny = cmd.get("ny", ny);
#ifdef P4_TO_P8
  nz = cmd.get("nz", nz);
#endif

//  double uniform_band = 3*r0/sqrt(Re);
  double uniform_band = 0;
#ifdef P4_TO_P8
  if(test_number==2)
#else
  if(test_number==4)
#endif
    uniform_band = .5*r0;

#ifdef P4_TO_P8
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny, (zmax-zmin)/(double)nz) / (1<<lmax);
#else
  double dxmin = MAX((xmax-xmin)/(double)nx, (ymax-ymin)/(double)ny) / (1<<lmax);
#endif
  uniform_band = cmd.get("uniform_band", uniform_band);
  uniform_band /= dxmin;

#ifdef P4_TO_P8
  ierr = PetscPrintf(mpi->mpicomm, "Parameters : mu = %g, rho = %g, grid is %dx%dx%d\n", mu, rho, nx, ny, nz); CHKERRXX(ierr);
#else
  ierr = PetscPrintf(mpi->mpicomm, "Parameters : Re = %g, mu = %g, rho = %g, grid is %dx%d\n", Re, mu, rho, nx, ny); CHKERRXX(ierr);
#endif
  ierr = PetscPrintf(mpi->mpicomm, "n_times_dt = %g, uniform_band = %g\n", n_times_dt, uniform_band);

  parStopWatch watch;
  watch.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny, xmin, xmax, ymin, ymax, &brick);
#endif

  p4est_t *p4est_nm1 = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);

  p4est_nm1->user_pointer = (void*)&data;
  if(test_number==0 || test_number==1)
  {
    data.min_lvl = lmin-nb_splits;
    data.max_lvl = lmax-nb_splits;
    for(int l=0; l<lmax-nb_splits; ++l)
    {
      my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
    }

    /* create the initial forest at time nm1 */
    p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

    for(int l=0; l<nb_splits; ++l)
    {
      my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_every_cell, NULL);
      my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
    }
    data.min_lvl = lmin;
    data.max_lvl = lmax;
  }
  else
  {
    for(int l=0; l<lmax; ++l)
    {
      my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
    }

    /* create the initial forest at time nm1 */
    p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }

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

  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_1st_order_time_2nd_order_space(phi);
  lsn.perturb_level_set_function(phi, EPS);

#ifdef P4_TO_P8
  CF_3 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1 };
  CF_3 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn   };
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

#ifndef P4_TO_P8
  if(test_number!=2)
#endif
  {
    bc_v[0].setInterfaceType(DIRICHLET); bc_v[0].setInterfaceValue(bc_interface_value_u);
    bc_v[1].setInterfaceType(DIRICHLET); bc_v[1].setInterfaceValue(bc_interface_value_v);
#ifdef P4_TO_P8
    bc_v[2].setInterfaceType(DIRICHLET); bc_v[2].setInterfaceValue(bc_interface_value_w);
#endif
    bc_p.setInterfaceType(NEUMANN); bc_p.setInterfaceValue(bc_interface_value_p);
  }

  external_force_u_t *external_force_u;
  external_force_v_t *external_force_v;
#ifdef P4_TO_P8
  external_force_w_t *external_force_w;
#endif


  my_p4est_navier_stokes_t ns(ngbd_nm1, ngbd_n, faces_n);
  ns.set_phi(phi);
  ns.set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, n_times_dt);
  if(test_number==5) ns.set_dt(.005*1/f0);
  else               ns.set_dt(dxmin*n_times_dt/u0, dxmin*n_times_dt/u0);
  dt = ns.get_dt();
  ns.set_velocities(vnm1, vn);
  ns.set_bc(bc_v, &bc_p);

  if(is_smoke)
  {
    Vec smoke;
    ierr = VecDuplicate(phi, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    ns.set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }

  tn = 0;
  int iter = 0;

  char name[1000];
  double forces[P4EST_DIM];

  FILE *fp_forces;
  char file_forces[1000];

#if defined(STAMPEDE) || defined(COMET)
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#endif

#ifdef P4_TO_P8
  if(test_number==2 || test_number==5)
  {
#if defined(STAMPEDE) || defined(COMET)
    if     (test_number==2) sprintf(file_forces, "%s/forces_karman_%d-%d_%dx%dx%d_Re_%g_thresh_%g_ntimesdt_%g.dat", out_dir, lmin, lmax, nx, ny, nz, Re, threshold_split_cell, n_times_dt);
    else if(test_number==5) sprintf(file_forces, "%s/forces_oscillating_sphere_%d-%d_%dx%dx%d_thresh_%g_Re_%g.dat", out_dir, lmin, lmax, nx, ny, nz, threshold_split_cell, Re);
#else
    if     (test_number==2) sprintf(file_forces, "/home/guittet/code/Output/p4est_navier_stokes/3d/karman/forces_%d-%d_%dx%dx%d_Re_%g.dat", lmin, lmax, nx, ny, nz, Re);
    else if(test_number==5) sprintf(file_forces, "/home/guittet/code/Output/p4est_navier_stokes/3d/oscillating_sphere/forces_%d-%d_%dx%dx%d.dat", lmin, lmax, nx, ny, nz);
#endif

    ierr = PetscPrintf(mpi->mpicomm, "Saving forces in ... %s\n", file_forces); CHKERRXX(ierr);
    if(!mpi->mpirank)
    {
      fp_forces = fopen(file_forces, "w");
      if(fp_forces==NULL)
        throw std::invalid_argument("[ERROR]: could not open file for forces output.");
      if(test_number==5) fprintf(fp_forces, "%% tn | Cd_x | Cd_y | Cd_z | Cp_x | Cp_y | Cp_z\n");
      else               fprintf(fp_forces, "%% tn | Cd_x | Cd_y | Cd_z\n");
      fclose(fp_forces);
    }
  }
#else
  if(test_number==4 || test_number==5 || test_number==6)
  {
#if defined(STAMPEDE) || defined(COMET)
    if     (test_number==4) sprintf(file_forces, "%s/forces_karman_%d-%d_%dx%d_Re_%g_thresh_%g_ntimesdt_%g.dat", out_dir, lmin, lmax, nx, ny, Re, threshold_split_cell, n_times_dt);
    else if(test_number==5) sprintf(file_forces, "%s/forces_oscillating_cylinder_%d-%d_%dx%d_Re_%g_thresh_%g_ntimesdt_%g.dat", out_dir, lmin, lmax, nx, ny, Re, threshold_split_cell, n_times_dt);
    else if(test_number==6) sprintf(file_forces, "%s/forces_naca_%04d_angle_%g_level_%d-%d_macro_%dx%d_Re_%g_thresh_%g_ntimesdt_%g.dat", out_dir, naca_number, naca_angle, lmin, lmax, nx, ny, Re, threshold_split_cell, n_times_dt);
#else
    if     (test_number==4) sprintf(file_forces, "/home/guittet/code/Output/p4est_navier_stokes/2d/karman/forces_%d-%d_%dx%d_Re_%g.dat", lmin, lmax, nx, ny, Re);
    else if(test_number==5) sprintf(file_forces, "/home/guittet/code/Output/p4est_navier_stokes/2d/oscillating_cylinder/forces_%d-%d_%dx%d_Re_%g.dat", lmin, lmax, nx, ny, Re);
    else if(test_number==6) sprintf(file_forces, "/home/guittet/code/Output/p4est_navier_stokes/2d/naca/forces_naca_%04d_angle_%g_level_%d-%d_macro_%dx%d_Re_%g_thresh_%g_ntimesdt_%g.dat", naca_number, naca_angle, lmin, lmax, nx, ny, Re, threshold_split_cell, n_times_dt);
#endif

    ierr = PetscPrintf(mpi->mpicomm, "Saving forces in ... %s\n", file_forces); CHKERRXX(ierr);
    if(!mpi->mpirank)
    {
      fp_forces = fopen(file_forces, "w");
      if(fp_forces==NULL)
        throw std::invalid_argument("[ERROR]: could not open file for forces output.");
      fprintf(fp_forces, "%% tn | Cd_x | Cd_y\n");
      fclose(fp_forces);
    }
  }
#endif

  while(tn+0.01*dt<tf)
  {
    if(iter>0)
    {
      ns.compute_dt();

      if(test_number==5)
        ns.set_dt(.005*1/f0);
//      if(test_number==5 && iter<4)
//        ns.set_dt(dxmin*n_times_dt/u0);

      dt = ns.get_dt();

      if(tn+dt>tf)
      {
        dt = tf-tn;
        ns.set_dt(dt);
      }

      if(test_number==0 || test_number==1)
        ns.update_from_tn_to_tnp1(&level_set, true);
      else
        ns.update_from_tn_to_tnp1(&level_set);
    }

    if(external_force_u==NULL) delete external_force_u;
    external_force_u = new external_force_u_t;

    if(external_force_v==NULL) delete external_force_v;
#ifdef P4_TO_P8
    external_force_v = new external_force_v_t;
#else
    external_force_v = new external_force_v_t(ns.get_ngbd_n(), ns.get_smoke());
#endif

#ifdef P4_TO_P8
    if(external_force_w==NULL) delete external_force_w;
    external_force_w = new external_force_w_t(ns.get_ngbd_n(), ns.get_smoke());
#endif


#ifdef P4_TO_P8
    CF_3 *external_forces[P4EST_DIM] = { external_force_u, external_force_v, external_force_w };
#else
    CF_2 *external_forces[P4EST_DIM] = { external_force_u, external_force_v };
#endif
    ns.set_external_forces(external_forces);

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, ns.get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double err_hodge = 1;
    int iter_hodge = 0;
    while(iter_hodge<10 && err_hodge>1e-3)
    {
      hodge_new = ns.get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      ns.solve_viscosity();
      ns.solve_projection();

      hodge_new = ns.get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      err_hodge = 0;
      p4est_t *p4est = ns.get_p4est();
      my_p4est_interpolation_nodes_t *interp_phi = ns.get_interp_phi();
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
          double xyz[P4EST_DIM];
          quad_xyz_fr_q(quad_idx, tree_idx, p4est, ns.get_ghost(), xyz);
#ifdef P4_TO_P8
          if((*interp_phi)(xyz[0],xyz[1],xyz[2])<0)
#else
          if((*interp_phi)(xyz[0],xyz[1])<0)
#endif
            err_hodge = max(err_hodge, fabs(ho[quad_idx]-hn[quad_idx]));
        }
      }
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge, 1, MPI_DOUBLE, MPI_MAX, mpi->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = VecRestoreArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(hodge_new, &hn); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi->mpicomm, "hodge iteration #%d, error = %e\n", iter_hodge, err_hodge); CHKERRXX(ierr);
      iter_hodge++;
    }
    ierr = VecDestroy(hodge_old); CHKERRXX(ierr);
    ns.compute_velocity_at_nodes();
    ns.compute_pressure();

    tn += dt;

#ifdef P4_TO_P8
    if(test_number==2 || test_number==5)
#else
    if(test_number==4 || test_number==5 || test_number==6)
#endif
    {
      ns.compute_forces(forces);
      if(!mpi->mpirank)
      {
        fp_forces = fopen(file_forces, "a");
        if(fp_forces==NULL)
          throw std::invalid_argument("[ERROR]: could not open file for forces output.");
#ifdef P4_TO_P8
				fprintf(fp_forces, "%g %g %g %g\n", tn, forces[0]/(.5*PI*r0*r0*u0*u0*rho), forces[1]/(.5*PI*r0*r0*u0*u0*rho), forces[2]/(.5*PI*r0*r0*u0*u0*rho));
#else
				if(test_number==4 || test_number==5)
	        fprintf(fp_forces, "%g %g %g\n", tn, forces[0]/r0/u0/u0/rho, forces[1]/r0/u0/u0/rho);
				if(test_number==6)
	        fprintf(fp_forces, "%g %g %g\n", tn, forces[0]/(naca_length*0.5*rho*u0*u0), forces[1]/(naca_length*0.5*rho*u0*u0));
#endif
        fclose(fp_forces);
      }
    }

    ierr = PetscPrintf(mpi->mpicomm, "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*tn/tf, ns.get_max_L2_norm_u(), ns.get_p4est()->global_num_quadrants); CHKERRXX(ierr);

#ifdef P4_TO_P8
    if((test_number==0 || test_number==1 || test_number==2) && ns.get_max_L2_norm_u()>10) break;
		if(test_number==4 && ns.get_max_L2_norm_u()>200) break;
#else
    if(test_number!=7 && test_number!=8 && test_number!=5 && ns.get_max_L2_norm_u()>100) break;
#endif

    if(save_vtk && iter%save_every_n==0)
    {
#if defined(STAMPEDE) || defined(COMET)
      sprintf(name, "%s/vtu/%05d_", out_dir, iter/save_every_n);
#else

      switch(test_number)
      {
#ifdef P4_TO_P8
      case 0: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/analytic_vortex/without_time_%d", iter/save_every_n); break;
      case 1: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/analytic_vortex/with_time_%d", iter/save_every_n); break;
      case 2: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/karman/karman_%d", iter/save_every_n); break;
      case 3: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/smoke_drop/smoke_drop_%d", iter/save_every_n); break;
      case 4: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/smoke_hill/smoke_hill_%d", iter/save_every_n); break;
      case 5: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/oscillating_sphere/oscillating_sphere_%d", iter/save_every_n); break;
      case 6: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/3d/vtu/smoke_city/smoke_city_%d", iter/save_every_n); break;
#else
      case 0: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/analytic_vortex/without_time_%d", iter/save_every_n); break;
      case 1: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/analytic_vortex/with_time_%d", iter/save_every_n); break;
      case 2: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/driven_cavity/cavity_%d", iter/save_every_n); break;
      case 3: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/driven_cavity_with_hole/hole_%d", iter/save_every_n); break;
      case 4: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/karman/karman_%d", iter/save_every_n); break;
      case 5: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/oscillating_cylinder/oscillating_%d", iter/save_every_n); break;
      case 6: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/naca/naca_%d", iter/save_every_n); break;
      case 7: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/smoke_drop/smoke_drop_%d", iter/save_every_n); break;
      case 8: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/smoke_hill/smoke_hill_%d", iter/save_every_n); break;
      case 9: sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/2d/vtu/rotating_cylinder/rotating_cylinder_%d", iter/save_every_n); break;
#endif
      default: throw std::invalid_argument("choose a valid test.");
      }
#endif

      ns.save_vtk(name);

#ifndef P4_TO_P8
      if(test_number==2 || test_number==3)
        check_velocity_cavity(mpi, &ns, Re, n_times_dt);
#endif
    }

    iter++;
  }

#ifdef P4_TO_P8
  if(test_number==0 || test_number==1)
#else
  if(test_number==0 || test_number==1)
#endif
    check_error_analytic_vortex(mpi, &ns);

#ifndef P4_TO_P8
  if(test_number==6)
    delete naca;
#endif

  if(external_force_u==NULL) delete external_force_u;
  if(external_force_v==NULL) delete external_force_v;
#ifdef P4_TO_P8
  if(external_force_w==NULL) delete external_force_w;
#endif

  watch.stop();
  watch.read_duration();

  return 0;
}
