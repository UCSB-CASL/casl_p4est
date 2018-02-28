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
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

int lmin = 4;
int lmax = 4;
int nb_splits = 4;

int nx = 1;
int ny = 1;
int nz = 1;

bool save_vtk = false;
bool save_voro = false;
bool save_stats = false;
bool check_partition = true;
//bool check_partition = false;

const int wave_number = 1;
const double mu_value = 4.3;
const double mu_ratio = 10.0;

struct domain{
  double xmin, xmax, ymin, ymax, zmin, zmax;
  domain(const double xmin_, const double xmax_, const double ymin_, const double ymax_, const double zmin_, const double zmax_):
    xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_)
  {}
};

domain centered_ones(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
double tiny_value = 0.000006487;
domain centered_tiny(-tiny_value, tiny_value, -tiny_value, tiny_value, -tiny_value, tiny_value);
double big_value = 86463.6134;
domain centered_big(-big_value, big_value, -big_value, big_value, -big_value, big_value);
domain shifted(512.0, 512.0+24.0, -9.13, -9.13+24.0, 73.84, 73.84+24.0);

//domain omega = centered_ones;
//domain omega = centered_tiny;
//domain omega = centered_big;
domain omega = shifted;

/*
 * 0 - circle
 */
int level_set_type = 0;

int test_number = 6;
/*
 *  ********* 2D *********
 * 0 - u_m=1+log(r/r0), u_p=1, mu_m=mu_p=1
 * 1 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=mu_p=mu_value, BC dirichlet
 * 2 - u_m=u_p=sin(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=mu_p=mu_value, BC neumann
 * 3 - u_m=exp((x-0.5*(xmin+xmax))/(xmax-xmin)), u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=SQR((y-ymin)/(ymax-ymin))*ln((x-0.5*(xmax+xmin))/(xmax-xmin)+2)+4, mu_p=exp(-(y-0.5*(ymin+ymax))/(ymax-ymin))   article example 4.4
 * 4 - u_m=up=sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin))*log((y-ymin)/(ymax-ymin) + 1.2), mu_m=mu_p=mu_value, BC periodic in x, dirichlet in y
 * 5 - u_m=exp(-SQR(sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin))))*(SQR((y-ymin)/(ymax-ymin))+atan(3.0*(y-0.5*(ymin+ymax))/(ymax-ymin))),
 *   - u_p= 1.0-pow((x-0.5*(xmin+xmax))/(xmax-xmin), 3.0)-SQR((y-0.5*(ymin+ymax))/(ymax-ymin))
 *   - mu_m= 2.0+0.3*cos(2.0*PI*(x-xmin)/(xmax-xmin))
 *   - mu_p=mu_value
 *   - BC periodic in x, dirichlet in y
 * 6 - u_m=up=SQR(sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin)))*SQR(cos(2.0*PI*2.0*wave_number*(y-ymin)/(ymax-ymin)))
 *   - mu_m=mu_p/mu_ratio=mu_value
 *   - fully periodic
 *
 *  ********* 3D *********
 * 0 - u_m=exp((z-zmin)/(xmax-zmin)), u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=mu_p=1.0
 * 1 - u_m=exp((z-zmin)/(xmax-zmin)), u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=SQR((y-ymin)/(ymax-ymin))*log((x-xmin)/(xmax-xmin)+2)+4, mu_p=exp(-(z-zmin)/(zmax-zmin))   article example 4.6
 * 2 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin))*exp((z-zmin)/(zmax-zmin)), mu_m=mu_p=1.45, BC dirichlet
 * 3 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin))*exp((z-zmin)/(zmax-zmin)), mu_m=mu_p=exp((x-xmin)/(xmax-xmin))*ln((y-ymin)/(ymax-ymin)+(z-zmin)/(zmax-zmin)+2), BC dirichlet
 * 4 - u_m=((y-0.5*(ymin+ymax))/(ymax-ymin))*((z-0.5*(zmin+zmax))/(zmax-zmin))*sin(x/(xmax-xmin)), u_p=((x-0.5*(xmin+xmax))/(xmax-xmin))*SQR((y-0.5*(ymin+ymax))/(ymax-ymin))+pow((z-0.5*(zmin+zmax))/(zmax-zmin), 3.0), mu_m=SQR((y-0.5*(ymin+ymax))/(ymax-ymin))+5, mu_p=exp((x-0.5*(xmin+xmax))/(xmax-xmin)+(z-0.5*(zmin+zmax))/(zmax-zmin))    article example 4.7
 * 5 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin))*exp((z-zmin)/(zmax-zmin)), mu_m=SQR((y-0.5*(ymin+ymax))/(ymax-ymin))+5, mu_p=exp((x-0.5*(xmin+xmax))/(xmax-xmin)+(z-0.5*(zmin+zmax))/(zmax-zmin))
 * 6 - u_m=exp(-SQR(sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin))))*(SQR((y-ymin)/(ymax-ymin))*atan(3.0*(z-0.5*(zmin+zmax))/(zmax-zmin))),
 *   - u_p= 1.0-pow((x-0.5*(xmin+xmax))/(xmax-xmin), 3.0)-SQR((y-0.5*(ymin+ymax))/(ymax-ymin)) + ((z-0.5*(zmin+zmax))/(zmax-zmin))
 *   - mu_m= 2.0+0.3*cos(2.0*PI*(x-xmin)/(xmax-xmin))
 *   - mu_p=mu_value
 *   - BC periodic in x, dirichlet in y, z
 */

#ifdef P4_TO_P8
double r0 = (double) MIN(omega.xmax-omega.xmin,omega.ymax-omega.ymin,omega.zmax-omega.zmin) / 4.0;
#else
double r0 = (double) MIN(omega.xmax-omega.xmin,omega.ymax-omega.ymin) / 4.0;
#endif



#ifdef P4_TO_P8

class LEVEL_SET: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(level_set_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - (omega.xmin+omega.xmax)/2) + SQR(y - (omega.ymin+omega.ymax)/2) + SQR(z - (omega.zmin+omega.zmax)/2));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class ONE: public CF_3
{
public:
  double operator()(double , double, double ) const
  {
    return -1;
  }
} one;

double phi_x(double x, double y, double z)
{
  switch(level_set_type)
  {
  case 0:
    return -(x-(omega.xmin+omega.xmax)/2)/sqrt(SQR(x-(omega.xmin+omega.xmax)/2)+SQR(y-(omega.ymin+omega.ymax)/2)+SQR(z-(omega.zmin+omega.zmax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

double phi_y(double x, double y, double z)
{
  switch(level_set_type)
  {
  case 0:
    return -(y-(omega.ymin+omega.ymax)/2)/sqrt(SQR(x-(omega.xmin+omega.xmax)/2)+SQR(y-(omega.ymin+omega.ymax)/2)+SQR(z-(omega.zmin+omega.zmax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

double phi_z(double x, double y, double z)
{
  switch(level_set_type)
  {
  case 0:
    return -(z-(omega.zmin+omega.zmax)/2)/sqrt(SQR(x-(omega.xmin+omega.xmax)/2)+SQR(y-(omega.ymin+omega.ymax)/2)+SQR(z-(omega.zmin+omega.zmax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

class MU_M: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
      return SQR((y-omega.ymin)/(omega.ymax-omega.ymin))*log((x-omega.xmin)/(omega.xmax-omega.xmin)+2.0)+4.0;
    case 2:
      return 1.45;
    case 3:
      return exp((x-omega.xmin)/(omega.xmax-omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+(z-omega.zmin)/(omega.zmax-omega.zmin)+2.0);
    case 4:
    case 5:
      return SQR((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))+5.0;
    case 6:
      return (2.0 + 0.3*cos(2.0*PI*(x-omega.xmin)/(omega.xmax-omega.xmin)));
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_m;

class MU_P: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(test_number)
    {
    case 0:
    case 2:
    case 3:
      return mu_m(x, y, z);
    case 1:
      return exp(-(z-omega.zmin)/(omega.zmax - omega.zmin));
    case 4:
    case 5:
      return exp((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin)+(z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin));
    case 6:
      return mu_value;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_p;

double u_m(double x, double y, double z)
{
  switch(test_number)
  {
  case 0:
  case 1:
    return exp((z-omega.zmin)/(omega.zmax - omega.zmin));
  case 2:
  case 3:
  case 5:
    return cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin));
  case 4:
    return ((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))*((z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))*sin(x/(omega.xmax-omega.xmin));
  case 6:
    return (exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*SQR((y-omega.ymin)/(omega.ymax-omega.ymin))*atan(3.0*(z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin)));
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

double u_p(double x, double y, double z)
{
  switch(test_number)
  {
  case 0:
  case 1:
    return cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax - omega.ymin));
  case 2:
  case 3:
  case 5:
    return u_m(x,y,z);
  case 4:
    return ((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin))*SQR((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))+pow((z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin), 3.0);
  case 6:
    return (1.0 - pow((x - 0.5*(omega.xmin + omega.xmax))/(omega.xmax - omega.xmin), 3.0) - SQR((y - 0.5*(omega.ymin + omega.ymax))/(omega.ymax - omega.ymin)) + ((z - 0.5*(omega.zmin + omega.zmax))/(omega.zmax - omega.zmin)));
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

class U_JUMP: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return u_p(x,y,z) - u_m(x,y,z);
  }
} u_jump;

double u_exact(double x, double y, double z)
{
  if(level_set(x,y,z)>0) return u_p(x,y,z);
  else                   return u_m(x,y,z);
}

double grad_u_m(double x, double y, double z)
{
  double ux, uy, uz;
  switch(test_number)
  {
  case 0:
    ux = 0;
    uy = 0;
    uz = exp((z-omega.zmin)/(omega.zmax - omega.zmin))*(1.0/(omega.zmax - omega.zmin));
    break;
  case 1:
    ux = 0;
    uy = 0;
    uz = exp((z-omega.zmin)/(omega.zmax - omega.zmin))*(1.0/(omega.zmax - omega.zmin));
    break;
  case 2:
  case 3:
  case 5:
    ux = -(1.0/(omega.xmax-omega.xmin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin));
    uy =  (1.0/(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin));
    uz =  (1.0/(omega.zmax-omega.zmin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin));
    break;
  case 4:
    ux = (1.0/(omega.xmax-omega.xmin))*((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))*((z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))*cos(x/(omega.xmax-omega.xmin));
    uy = (1.0/(omega.ymax-omega.ymin))*((z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))*sin(x/(omega.xmax-omega.xmin));
    uz = (1.0/(omega.zmax-omega.zmin))*((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin));
    break;
  case 6:
    ux = -(2.0*PI*((double) wave_number)/(omega.xmax - omega.xmin))*sin(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*u_m(x,y,z);
    uy = (2.0/(omega.ymax-omega.ymin))*((y-omega.ymin)/(omega.ymax-omega.ymin))*exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*atan(3.0*(z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin));
    uz = exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*SQR((y-omega.ymin)/(omega.ymax-omega.ymin))*(1.0/(1.0 + SQR(3.0*(z - 0.5*(omega.zmin + omega.zmax))/(omega.zmax - omega.zmin))))*(3.0/(omega.zmax - omega.zmin));
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y,z);
  double phiy = phi_y(x,y,z);
  double phiz = phi_z(x,y,z);

  return ux*phix + uy*phiy + uz*phiz;
}

double grad_u_p(double x, double y, double z)
{
  double ux, uy, uz;
  switch(test_number)
  {
  case 0:
    ux = -sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.xmax - omega.xmin))*(1.0/(omega.xmax - omega.xmin));
    uy =  cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.xmax - omega.xmin))*(1.0/(omega.ymax - omega.ymin));
    uz = 0;
    break;
  case 1:
    ux = -sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.xmax - omega.xmin))*(1.0/(omega.xmax - omega.xmin));
    uy =  cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.xmax - omega.xmin))*(1.0/(omega.ymax - omega.ymin));
    uz = 0;
    break;
  case 2:
  case 3:
  case 5:
    return grad_u_m(x,y,z);
  case 4:
    ux = (1.0/(omega.xmax-omega.xmin))*SQR((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin));
    uy = (2.0/(omega.ymax-omega.ymin))*((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin))*((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin));
    uz = (3.0/(omega.zmax-omega.zmin))*SQR((z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin));
    break;
  case 6:
    ux = -3.0*SQR((x - 0.5*(omega.xmin + omega.xmax))/(omega.xmax - omega.xmin))*(1.0/(omega.xmax - omega.xmin));
    uy = -2.0*((y - 0.5*(omega.ymin + omega.ymax))/(omega.ymax - omega.ymin))*(1.0/(omega.ymax - omega.ymin));
    uz = +1.0/(omega.zmax - omega.zmin);
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y,z);
  double phiy = phi_y(x,y,z);
  double phiz = phi_z(x,y,z);

  return ux*phix + uy*phiy + uz*phiz;
}


class MU_GRAD_U_JUMP: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return mu_p(x,y,z)*grad_u_p(x,y,z) - mu_m(x,y,z)*grad_u_m(x,y,z);
  }
} mu_grad_u_jump;

class BC_WALL_TYPE : public WallBC3D
{
public:
  BoundaryConditionType operator() (double , double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type;


class BC_WALL_VALUE : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return level_set(x,y,z)<0 ? u_m(x,y,z) : u_p(x,y,z);
  }
} bc_wall_value;

#else

class LEVEL_SET: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(level_set_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - (omega.xmin+omega.xmax)/2) + SQR(y - (omega.ymin+omega.ymax)/2));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class ONE: public CF_2
{
public:
  double operator()(double , double ) const
  {
    return -1;
  }
} one;

double phi_x(double x, double y)
{
  switch(level_set_type)
  {
  case 0:
    return -(x-(omega.xmin+omega.xmax)/2)/sqrt(SQR(x-(omega.xmin+omega.xmax)/2)+SQR(y-(omega.ymin+omega.ymax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

double phi_y(double x, double y)
{
  switch(level_set_type)
  {
  case 0:
    return -(y-(omega.ymin+omega.ymax)/2)/sqrt(SQR(x-(omega.xmin+omega.xmax)/2)+SQR(y-(omega.ymin+omega.ymax)/2));
  default:
    throw std::invalid_argument("Choose a valid level set.");
  }
}

class MU_M: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
    case 2:
    case 4:
    case 6:
      return mu_value;
    case 3:
      return SQR((y-omega.ymin)/(omega.ymax-omega.ymin))*log((x-0.5*(omega.xmax+omega.xmin))/(omega.xmax-omega.xmin)+2.0)+4.0;
    case 5:
      return 2.0+0.3*cos(2.0*PI*(x-omega.xmin)/(omega.xmax-omega.xmin));
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_m;

class MU_P: public CF_2
{
public:
  double operator()(double , double y) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
    case 2:
    case 4:
    case 5:
      return mu_value;
    case 3:
      return exp(-(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin));
    case 6:
      return mu_ratio*mu_value;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} mu_p;

struct U_M : CF_2
{
  double operator()(double x, double y) const
  {
    double r = sqrt(SQR(x-(omega.xmin+omega.xmax)/2) + SQR(y-(omega.ymin+omega.ymax)/2));
    switch(test_number)
    {
    case 0:
      return 1+log(r/r0);
    case 1:
      return cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    case 2:
      return sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    case 3:
      return exp((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin));
    case 4:
      return sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+1.2);
    case 5:
      return exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*(SQR((y-omega.ymin)/(omega.ymax-omega.ymin))+atan(3.0*(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin)));
    case 6:
      return SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin)))*SQR(cos(2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin)));
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} u_m;

struct U_P : CF_2
{
  double operator()(double x, double y) const
  {
    switch(test_number)
    {
    case 0:
      return 1;
    case 1:
    case 2:
    case 4:
    case 6:
      return u_m(x,y);
    case 3:
      return cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    case 5:
      return 1.0-pow((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin), 3.0)-SQR((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin));
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
  }
} u_p;

class U_JUMP: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return u_p(x,y) - u_m(x,y);
  }
} u_jump;

double u_exact(double x, double y)
{
  if(level_set(x,y)>0) return u_p(x,y);
  else                 return u_m(x,y);
}

double grad_u_m(double x, double y)
{
  double ux, uy;
  switch(test_number)
  {
  case 0:
    ux = (x-0.5*(omega.xmin+omega.xmax))/(SQR(x-0.5*(omega.xmin+omega.xmax))+SQR(y-0.5*(omega.ymin+omega.ymax)));
    uy = (y-0.5*(omega.ymin+omega.ymax))/(SQR(x-0.5*(omega.xmin+omega.xmax))+SQR(y-0.5*(omega.ymin+omega.ymax)));
    break;
  case 1:
    ux = -(1.0/(omega.xmax-omega.xmin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    uy =  (1.0/(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
    break;
  case 2:
    ux =  (1.0/(omega.xmax-omega.xmin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    uy =  (1.0/(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
    break;
  case 3:
    ux = (1.0/(omega.xmax-omega.xmin))*exp((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin));
    uy = 0;
    break;
  case 4:
    ux = (2.0*PI*wave_number/(omega.xmax-omega.xmin))*cos(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+1.2);;
    uy = sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*(1.0/(y-omega.ymin+1.2*(omega.ymax - omega.ymin)));
    break;
  case 5:
    ux = exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*(SQR((y-omega.ymin)/(omega.ymax-omega.ymin))+atan(3.0*(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin)))*(-sin(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*(2.0*PI*wave_number/(omega.xmax-omega.xmin)));
    uy = exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*(2.0*(y-omega.ymin)/SQR(omega.ymax-omega.ymin) + (1.0/(1.0+SQR(3.0*(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))))*(3.0/(omega.ymax-omega.ymin)));
    break;
  case 6:
    ux = (2.0*PI*wave_number/(omega.xmax-omega.xmin))*sin(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*SQR(cos(2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin)));
    uy = -(2.0*PI*2.0*wave_number/(omega.ymax-omega.ymin))*SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin)))*sin(2.0*2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin));
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y);
  double phiy = phi_y(x,y);

  return ux*phix + uy*phiy;
}

double grad_u_p(double x, double y)
{
  double ux, uy;
  switch(test_number)
  {
  case 0:
    ux = 0;
    uy = 0;
    break;
  case 1:
    ux = -(1.0/(omega.xmax-omega.xmin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    uy =  (1.0/(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
    break;
  case 2:
    ux =  (1.0/(omega.xmax-omega.xmin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    uy =  (1.0/(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
    break;
  case 3:
    ux = -(1.0/(omega.xmax-omega.xmin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
    uy =  (1.0/(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
    break;
  case 4:
    ux = (2.0*PI*wave_number/(omega.xmax-omega.xmin))*cos(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+1.2);;
    uy = sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*(1.0/(y-omega.ymin+1.2*(omega.ymax - omega.ymin)));
    break;
  case 5:
    ux = -3.0*SQR((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin))*(1.0/(omega.xmax-omega.xmin));
    uy = -2.0*((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))*(1.0/(omega.ymax-omega.ymin));
    break;
  case 6:
    ux = (2.0*PI*wave_number/(omega.xmax-omega.xmin))*sin(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*SQR(cos(2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin)));
    uy = -(2.0*PI*2.0*wave_number/(omega.ymax-omega.ymin))*SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin)))*sin(2.0*2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin));
    break;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }

  double phix = phi_x(x,y);
  double phiy = phi_y(x,y);

  return ux*phix + uy*phiy;
}

class MU_GRAD_U_JUMP: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return mu_p(x,y)*grad_u_p(x,y) - mu_m(x,y)*grad_u_m(x,y);
  }
} mu_grad_u_jump;

class BC_WALL_TYPE : public WallBC2D
{
public:
  BoundaryConditionType operator() (double, double) const
  {
    if(test_number==2) return NEUMANN;
    return DIRICHLET;
  }
} bc_wall_type;

class BC_WALL_VALUE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(bc_wall_type(x,y)==DIRICHLET)
      return level_set(x,y)<0 ? u_m(x,y) : u_p(x,y);
    else
    {
      if(ABS(x-omega.xmin)<EPS*(omega.xmax - omega.xmin)) return -(1.0/(omega.xmax-omega.xmin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
      if(ABS(x-omega.xmax)<EPS*(omega.xmax - omega.xmin)) return  (1.0/(omega.xmax-omega.xmin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
      if(ABS(y-omega.ymin)<EPS*(omega.ymax - omega.ymin)) return -(1.0/(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
      return  (1.0/(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin));
    }
  }
} bc_wall_value;

#endif



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err,
              int compt)
{
  PetscErrorCode ierr;
  string output = "/home/egan/workspace/projects/jump_solver/output";
  const char *out_dir = output.c_str();
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save visuals\n"); CHKERRXX(ierr);
    return;
  }

  std::ostringstream oss;

  oss << out_dir << "/jump_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p, *sol_p, *err_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);

  Vec mu;
  ierr = VecDuplicate(phi, &mu); CHKERRXX(ierr);
  double *mu_p_;
  ierr = VecGetArray(mu, &mu_p_); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
    mu_p_[n] = phi_p[n]<0 ? mu_m(x,y,z) : mu_p(x,y,z);
#else
    mu_p_[n] = phi_p[n]<0 ? mu_m(x,y) : mu_p(x,y);
#endif
  }

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

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         4, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "mu", mu_p_,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", err_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(mu, &mu_p_); CHKERRXX(ierr);
  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(mu); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



void shift_Neumann_Solution(p4est_t *p4est, p4est_nodes_t *nodes, Vec sol)
{
  PetscErrorCode ierr;

  ierr = PetscPrintf(p4est->mpicomm, "Shifting all neumann solution\n");

  double ex_int = 0;
  switch(test_number)
  {
  case 2:
    ex_int = 4.0*(omega.xmax-omega.xmin)*(omega.ymax-omega.ymin)*SQR(sin(0.5))*sin(0.5*(omega.xmin+omega.xmax)/(omega.xmax-omega.xmin))*sin(0.5*(omega.ymin+omega.ymax)/(omega.ymax-omega.ymin));
    break;
  case 6:
    ex_int = 0.5*0.5*(omega.xmax-omega.xmin)*(omega.ymax-omega.ymin);
    break;
  default:
    ex_int = 0;
  }

  Vec ones;
  ierr = VecDuplicate(sol, &ones); CHKERRXX(ierr);
  ierr = VecSet(ones, -1); CHKERRXX(ierr);
  double sol_int = integrate_over_negative_domain(p4est, nodes, ones, sol);

  double *sol_p;
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    sol_p[n] += (ex_int - sol_int)/((omega.xmax-omega.xmin)*(omega.ymax-omega.ymin));

  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
}



void solve_Poisson_Jump( p4est_t *p4est, p4est_nodes_t *nodes,
                         my_p4est_node_neighbors_t *ngbd_n, my_p4est_cell_neighbors_t *ngbd_c,
                         Vec phi, Vec sol)
{
  PetscErrorCode ierr;

  Vec rhs_m, rhs_p;
  Vec mu_m_, mu_p_;
  Vec u_jump_;
  Vec mu_grad_u_jump_;

  ierr = VecDuplicate(phi, &rhs_m); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs_p); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &mu_m_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &mu_p_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &u_jump_); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &mu_grad_u_jump_); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, mu_m, mu_m_);
  sample_cf_on_nodes(p4est, nodes, mu_p, mu_p_);
  sample_cf_on_nodes(p4est, nodes, u_jump, u_jump_);
  sample_cf_on_nodes(p4est, nodes, mu_grad_u_jump, mu_grad_u_jump_);

  double *rhs_m_p, *rhs_p_p;
  ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
    switch(test_number)
    {
    case 0:
    case 1:
      rhs_m_p[n] = -mu_m(x,y,z)*exp((z-omega.zmin)/(omega.zmax - omega.zmin))*SQR(1.0/(omega.zmax-omega.zmin));
      rhs_p_p[n] = mu_p(x,y,z)*cos(x/(omega.xmax - omega.xmin))*sin(y/(omega.ymax - omega.ymin))*(SQR(1.0/(omega.xmax-omega.xmin)) + SQR(1.0/(omega.ymax-omega.ymin)));
      break;
    case 2:
      rhs_m_p[n] = mu_m(x,y,z)*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin))*(1.0/SQR(omega.xmax-omega.xmin)+1.0/SQR(omega.ymax-omega.ymin)-1.0/SQR(omega.zmax-omega.zmin));
      rhs_p_p[n] = mu_p(x,y,z)*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin))*(1.0/SQR(omega.xmax-omega.xmin)+1.0/SQR(omega.ymax-omega.ymin)-1.0/SQR(omega.zmax-omega.zmin));
      break;
    case 3:
      rhs_m_p[n] = rhs_p_p[n] =
          + (1.0/SQR(omega.xmax-omega.xmin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin))*exp((x-omega.xmin)/(omega.xmax-omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+(z-omega.zmin)/(omega.zmax-omega.zmin)+2.0)
          - (1.0/SQR(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin))*exp((x-omega.xmin)/(omega.xmax-omega.xmin))*(1.0/((y-omega.ymin)/(omega.ymax-omega.ymin)+(z-omega.zmin)/(omega.zmax-omega.zmin)+2.0))
          - (1.0/SQR(omega.zmax-omega.zmin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin))*exp((x-omega.xmin)/(omega.xmax-omega.xmin))*(1.0/((y-omega.ymin)/(omega.ymax-omega.ymin)+(z-omega.zmin)/(omega.zmax-omega.zmin)+2.0))
          - mu_m(x,y,z)*u_m(x,y,z)*(-1.0/SQR(omega.xmax-omega.xmin)-1.0/SQR(omega.ymax-omega.ymin)+1.0/SQR(omega.zmax-omega.zmin));
      break;
    case 4:
      rhs_m_p[n] = ((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))*((z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))*sin(x/(omega.xmax-omega.xmin))*(mu_m(x,y,z)/SQR(omega.xmax-omega.xmin)-2.0/SQR(omega.ymax-omega.ymin));
      rhs_p_p[n] = -exp((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin) + (z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))*(SQR((y-0.5*(omega.ymin+omega.ymax))/((omega.xmax-omega.xmin)*(omega.ymax-omega.ymin))) + 3.0*SQR((z-0.5*(omega.zmin+omega.zmax))/SQR(omega.zmax-omega.zmin)))
          -mu_p(x,y,z)*(2.0*(x-0.5*(omega.xmin + omega.xmax))/((omega.xmax-omega.xmin)*SQR(omega.ymax-omega.ymin)) +6.0*(z-0.5*(omega.zmin+omega.zmax))/(pow(omega.zmax-omega.zmin, 3.0)));
      break;
    case 5:
      rhs_m_p[n] = mu_m(x,y,z)*u_m(x,y,z)*(1.0/SQR(omega.xmax - omega.xmin) + 1.0/SQR(omega.ymax - omega.ymin) - 1.0/SQR(omega.zmax-omega.zmin))
          - (2.0/SQR(omega.ymax - omega.ymin))*((y-0.5*(omega.ymin+omega.ymax))/(omega.ymax - omega.ymin))*cos(x/(omega.xmax - omega.xmin))*cos(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin));
      rhs_p_p[n] = mu_p(x,y,z)*u_p(x,y,z)*(1.0/SQR(omega.xmax - omega.xmin) + 1.0/SQR(omega.ymax - omega.ymin) - 1.0/SQR(omega.zmax-omega.zmin))
          +(1.0/SQR(omega.xmax-omega.xmin))*mu_p(x,y,z)*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin))
          -(1.0/SQR(omega.zmax-omega.zmin))*mu_p(x,y,z)*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin))*exp((z-omega.zmin)/(omega.zmax-omega.zmin));
      break;
    case 6:
      rhs_m_p[n] = -0.3*SQR(2.0*PI/(omega.xmax - omega.xmin))*((double) wave_number)*sin(2.0*PI*(x-omega.xmin)/(omega.xmax - omega.xmin))*sin(2.0*2.0*PI*wave_number*(x- omega.xmin)/(omega.xmax - omega.xmin))*u_m(x,y,z)
          -mu_m(x,y,z)*u_m(x,y,z)*SQR(2.0*PI*((double) wave_number)/(omega.xmax - omega.xmin))*(SQR(sin(2.0*2.0*PI*wave_number*(x - omega.xmin)/(omega.xmax - omega.xmin))) - 2.0*cos(2.0*2.0*PI*wave_number*(x - omega.xmin)/(omega.xmax - omega.xmin)))
          -mu_m(x,y,z)*2.0*SQR(1.0/(omega.ymax - omega.ymin))*exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*atan(3.0*(z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))
          +mu_m(x,y,z)*SQR(3.0/(omega.zmax - omega.zmin))*2.0*(3.0*(z - 0.5*(omega.zmin + omega.zmax))/(omega.zmax-omega.zmin))*exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*SQR((y-omega.ymin)/(omega.ymax-omega.ymin))*(1.0/SQR(1.0 + SQR(3.0*(z-0.5*(omega.zmin+omega.zmax))/(omega.zmax-omega.zmin))));
      rhs_p_p[n] = -mu_p(x,y,z)*(-6.0*SQR(1.0/(omega.xmax - omega.xmin))*((x - 0.5*(omega.xmin + omega.xmax))/(omega.xmax - omega.xmin)) - 2.0*SQR(1.0/(omega.ymax - omega.ymin)));
      break;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
#else
    switch(test_number)
    {
    case 0:
      rhs_m_p[n] = 0;
      rhs_p_p[n] = 0;
      break;
    case 1:
      rhs_m_p[n] = mu_m(x,y)*(1.0/SQR(omega.xmax-omega.xmin) + 1.0/SQR(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
      rhs_p_p[n] = mu_p(x,y)*(1.0/SQR(omega.xmax-omega.xmin) + 1.0/SQR(omega.ymax-omega.ymin))*cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
      break;
    case 2:
      rhs_m_p[n] = mu_m(x,y)*(1.0/SQR(omega.xmax-omega.xmin) + 1.0/SQR(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
      rhs_p_p[n] = mu_p(x,y)*(1.0/SQR(omega.xmax-omega.xmin) + 1.0/SQR(omega.ymax-omega.ymin))*sin(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin));
      break;
    case 3:
      rhs_m_p[n] = -SQR((y-omega.ymin)/(omega.ymax-omega.ymin))*(1.0/(x-0.5*(omega.xmin+omega.xmax)+2.0*(omega.xmax-omega.xmin)))*(1.0/(omega.xmax-omega.xmin))*exp((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin)) - mu_m(x, y)*SQR(1.0/(omega.xmax-omega.xmin))*exp((x-0.5*(omega.xmin+omega.xmax))/(omega.xmax-omega.xmin));
      rhs_p_p[n] = (1.0/(omega.ymax-omega.ymin))*exp(-(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin))*(cos(x/(omega.xmax-omega.xmin))*cos(y/(omega.ymax-omega.ymin))*(1.0/(omega.ymax-omega.ymin))) + mu_p(x, y)*(cos(x/(omega.xmax-omega.xmin))*sin(y/(omega.ymax-omega.ymin)))*(SQR(1.0/(omega.xmax-omega.xmin)) + SQR(1.0/(omega.ymax-omega.ymin)));
      break;
    case 4:
      rhs_m_p[n] = mu_m(x, y)*(SQR(2.0*PI*wave_number/(omega.xmax - omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+1.2) + SQR(1.0/(y-omega.ymin+1.2*(omega.ymax - omega.ymin))))*sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin));
      rhs_p_p[n] = mu_p(x, y)*(SQR(2.0*PI*wave_number/(omega.xmax - omega.xmin))*log((y-omega.ymin)/(omega.ymax-omega.ymin)+1.2) + SQR(1.0/(y-omega.ymin+1.2*(omega.ymax - omega.ymin))))*sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin));
      break;
    case 5:
      rhs_m_p[n] = +0.3*(2.0*PI/(omega.xmax-omega.xmin))*sin(2.0*PI*(x-omega.xmin)/(omega.xmax-omega.xmin))*(SQR((y-omega.ymin)/(omega.ymax-omega.ymin))+atan(3.0*(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin)))*exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*(-sin(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*(2.0*PI*wave_number/(omega.xmax-omega.xmin)))
          -mu_m(x, y)*(SQR((y-omega.ymin)/(omega.ymax-omega.ymin))+atan(3.0*(y-0.5*(omega.ymin+omega.ymax))/(omega.ymax-omega.ymin)))*exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*(SQR(-sin(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*(2.0*PI*wave_number/(omega.xmax-omega.xmin))) - 2.0*SQR(2.0*PI*wave_number/(omega.xmax-omega.xmin))*cos(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin)))
          -mu_m(x, y)*exp(-SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))))*(2.0/SQR(omega.ymax-omega.ymin)-54.0*(omega.ymax-omega.ymin)*(y-0.5*(omega.ymin+omega.ymax))/SQR(SQR(omega.ymax-omega.ymin)+SQR(3*(y-0.5*(omega.ymin+omega.ymax)))));
      rhs_p_p[n] = mu_p(x, y)*(6.0*(x-0.5*(omega.xmin+omega.xmax))/pow((omega.xmax-omega.xmin), 3.0) + 2.0/SQR(omega.ymax-omega.ymin));
      break;
    case 6:
      rhs_m_p[n] = -mu_m(x, y)*(2.0*SQR(2.0*PI*wave_number/(omega.xmax-omega.xmin))*cos(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*SQR(cos(2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin))) - 2.0*SQR(2.0*PI*2.0*wave_number/(omega.ymax-omega.ymin))*SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin)))*cos(2.0*2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin)));
      rhs_p_p[n] = -mu_p(x, y)*(2.0*SQR(2.0*PI*wave_number/(omega.xmax-omega.xmin))*cos(2.0*2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin))*SQR(cos(2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin))) - 2.0*SQR(2.0*PI*2.0*wave_number/(omega.ymax-omega.ymin))*SQR(sin(2.0*PI*wave_number*(x-omega.xmin)/(omega.xmax-omega.xmin)))*cos(2.0*2.0*PI*2.0*wave_number*(y-omega.ymin)/(omega.ymax-omega.ymin)));
      break;
    default:
      throw std::invalid_argument("Choose a valid test.");
    }
#endif
  }

  ierr = VecRestoreArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  my_p4est_poisson_jump_nodes_voronoi_t solver(ngbd_n, ngbd_c);
  solver.set_phi(phi);
  solver.set_bc(bc);
  solver.set_mu(mu_m_, mu_p_);
  solver.set_u_jump(u_jump_);
  solver.set_mu_grad_u_jump(mu_grad_u_jump_);
  solver.set_rhs(rhs_m, rhs_p);

  solver.solve(sol);
  //  solver.compute_voronoi_points();
  if(check_partition)
    solver.check_voronoi_partition();
  //  if(p4est->mpirank==0)
  //  solver.compute_voronoi_mesh();
  //  solver.setup_negative_laplace_matrix();
  //  solver.setup_negative_laplace_rhsvec();
//  sample_cf_on_nodes(p4est, nodes, u_m, sol);

  char out_path[1000];
  char *out_dir = "/home/egan/workspace/projects/jump_solver/output";
//  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
  }
  else
  {
    if(save_stats)
    {
      sprintf(out_path, "%s/stats.dat", out_dir);
      solver.write_stats(out_path);
    }

    if(save_voro)
    {
      snprintf(out_path,1000, "%s/voronoi", out_dir);
      solver.print_voronoi_VTK(out_path);
    }
  }

  if(solver.get_matrix_has_nullspace())
    shift_Neumann_Solution(p4est, nodes, sol);

  ierr = VecDestroy(rhs_m); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_p); CHKERRXX(ierr);
  ierr = VecDestroy(mu_m_); CHKERRXX(ierr);
  ierr = VecDestroy(mu_p_); CHKERRXX(ierr);
  ierr = VecDestroy(u_jump_); CHKERRXX(ierr);
  ierr = VecDestroy(mu_grad_u_jump_); CHKERRXX(ierr);
}



int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("save_vtk", "1 to save vtu files, 0 otherwise");
  cmd.add_option("save_voro", "1 to save voronoi partition, 0 otherwise");
  cmd.add_option("save_stats", "1 to save statistics about the voronoi partition, 0 otherwise");
  cmd.add_option("check_partition", "1 to check if the voronoi partition is symmetric, 0 otherwise");
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
                 0 - u_m=exp((z-zmin)/(xmax-zmin)), u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=mu_p=1.0\n\
                 1 - u_m=exp((z-zmin)/(xmax-zmin)), u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=SQR((y-ymin)/(ymax-ymin))*log((x-xmin)/(xmax-xmin)+2)+4, mu_p=exp(-(z-zmin)/(zmax-zmin)) article example 4.6 \n\
                 2 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin))*exp((z-zmin)/(zmax-zmin)), mu_m=mu_p=1.45, BC dirichlet\n\
                 3 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin))*exp((z-zmin)/(zmax-zmin)), mu_m=mu_p=exp((x-xmin)/(xmax-xmin))*ln((y-ymin)/(ymax-ymin)+(z-zmin)/(zmax-zmin)+2), BC dirichlet\n\
                 4 - u_m=((y-0.5*(ymin+ymax))/(ymax-ymin))*((z-0.5*(zmin+zmax))/(zmax-zmin))*sin(x/(xmax-xmin)), u_p=((x-0.5*(xmin+xmax))/(xmax-xmin))*SQR((y-0.5*(ymin+ymax))/(ymax-ymin))+pow((z-0.5*(zmin+zmax))/(zmax-zmin), 3.0), mu_m=SQR((y-0.5*(ymin+ymax))/(ymax-ymin))+5, mu_p=exp((x-0.5*(xmin+xmax))/(xmax-xmin)+(z-0.5*(zmin+zmax))/(xmax-xmin))    article example 4.7");
#else
  cmd.add_option("test", "choose a test.\n\
                  0 - u_m=1+log(r/r0), r = sqrt(SQR(x-0.5*(xmax+xmin)) + SQR(y-0.5*(ymax+ymin)))), r0 = MIN(xmax-xmin,ymax-ymin/4), u_p=1, mu_m=mu_p=1\n\
                  1 - u_m=u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=mu_p=constant, BC dirichlet\n\
                  2 - u_m=u_p=sin(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=mu_p=constant, BC neumann\n\
                  3 - u_m=exp((x-0.5*(xmin+xmax))/(xmax-xmin)), u_p=cos(x/(xmax-xmin))*sin(y/(ymax-ymin)), mu_m=SQR((y-ymin)/(ymax-ymin))*ln((x-0.5*(xmax+xmin))/(xmax-xmin)+2)+4, mu_p=exp(-(y-0.5*(ymin+ymax))/(ymax-ymin))   article example 4.4\n\
                  4 - u_m=up=sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin))*log((y-ymin)/(ymax-ymin) + 1.2), mu_m=mu_p=constant, BC periodic in x, dirichlet in y\n\
                  5 - u_m=exp(-SQR(sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin))))*(SQR((y-ymin)/(ymax-ymin))+atan(3.0*(y-0.5*(ymin+ymax))/(ymax-ymin))),\n\
                    - u_p= 1.0-pow((x-0.5*(xmin+xmax))/(xmax-xmin), 3.0)-SQR((y-0.5*(ymin+ymax))/(ymax-ymin))\n\
                    - mu_m= 2.0+0.3*cos(2.0*PI*(x-xmin)/(xmax-xmin))\n\
                    - mu_p=constant\n\
                    - BC periodic in x, dirichlet in y\n\
                  6 - u_m=up=SQR(sin(2.0*PI*wave_number*(x-xmin)/(xmax-xmin)))*SQR(cos(2.0*PI*2.0*wave_number*(y-ymin)/(ymax-ymin)))\n\
                    - mu_m=mu_p/mu_ratio=constant\n\
                    - fully periodic");
#endif
  cmd.parse(argc, argv);

  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  test_number = cmd.get("test", test_number);
  save_vtk = cmd.get("save_vtk", save_vtk);
  save_voro= cmd.get("save_voro", save_voro);
  save_stats = cmd.get("save_stats", save_stats);
  check_partition = cmd.get("check_partition", check_partition);

  parStopWatch w;
  w.start("total time");

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

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  const int n_xyz []       = {nx, ny, nz};
  const double xyz_min []  = {omega.xmin, omega.ymin, omega.zmin};
  const double xyz_max []  = {omega.xmax, omega.ymax, omega.zmax};
  int p_x, p_y, p_z;
#ifdef P4_TO_P8
  switch (test_number) {
  case 6:
    p_x=1;
    p_y=p_z=0;
    break;
  default:
    p_x=p_y=p_z=0;
    break;
  }
#else
  switch (test_number) {
  case 4:
  case 5:
    p_x=1;
    p_y=p_z=0;
    break;
  case 6:
    p_x=p_y=1;
    p_z=0;
    break;
  default:
    p_x=p_y=p_z=0;
    break;
  }
#endif
  const int periodic []    = {p_x, p_y, p_z};

  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_n   = 0;
  double err_nm1 = 0;

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    //    srand(1);
    //    splitting_criteria_random_t data(4, 6, 1000, 10000);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set, 1.2);
    p4est->user_pointer = (void*)(&data);

    for(int i=0; i<lmax+iter; ++i)
    {
      //    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
      my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    if(p4est->mpirank==0)
    {
      p4est_gloidx_t nb_nodes = 0;
      for(int r=0; r<p4est->mpisize; ++r)
        nb_nodes += nodes->global_owned_indeps[r];
      ierr = PetscPrintf(p4est->mpicomm, "number of nodes : %d\n", nb_nodes); CHKERRXX(ierr);
    }

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);

    my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set, phi);
    // bousouf
//    sample_cf_on_nodes(p4est, nodes, one, phi);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.perturb_level_set_function(phi, EPS);

    Vec sol;
    ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

    solve_Poisson_Jump(p4est, nodes, &ngbd_n, &ngbd_c, phi, sol);

    /* compute the error on the tree*/
    Vec err;
    ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
    double *err_p, *sol_p;
    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    err_nm1 = err_n;
    err_n = 0;
    double x_err=-1, y_err=-1;
    double domain_diag = SQR(omega.xmax - omega.xmin) + SQR(omega.ymax - omega.ymin);
#ifdef P4_TO_P8
    double z_err=-1;
    domain_diag += SQR(omega.zmax - omega.zmin);
#endif
    domain_diag = sqrt(domain_diag);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      err_p[n] = fabs(u_exact(x,y,z) - sol_p[n]);
      double level_set_value = level_set(x, y, z);
#else
      err_p[n] = fabs(u_exact(x,y) - sol_p[n]);
      double level_set_value = level_set(x, y);
#endif
      if((err_p[n]>err_n) && (fabs(level_set_value) > EPS*domain_diag))
      {
        x_err = x;
        y_err = y;
#ifdef P4_TO_P8
        z_err = z;
#endif
      }
      err_n = max(err_n, err_p[n]);
    }

    MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
    PetscPrintf(p4est->mpicomm, "Iter %d : %g, \t order : %g\n", iter, err_n, log(err_nm1/err_n)/log(2));
#ifdef P4_TO_P8
    PetscPrintf(p4est->mpicomm, "error at %g, %g, %g, qh = %g, dist_interface = %g\n", x_err, y_err, z_err, (double) 2/pow(2,lmax+iter+1), fabs(level_set(x_err, y_err, z_err)));
#else
    PetscPrintf(p4est->mpicomm, "error at %g, %g, qh = %g, dist_interface = %g\n", x_err, y_err, (double) 2/pow(2,lmax+iter+1), fabs(level_set(x_err, y_err)));
#endif

    ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);

    if(save_vtk)
      save_VTK(p4est, ghost, nodes, &brick, phi, sol, err, iter);

    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);
    ierr = VecDestroy(err); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
