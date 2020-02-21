/*
 * Title: scft_particles
 * Description:
 * Author: ivanabagaric
 * Date Created: 10-24-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_scft.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_shapes.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_shapes.h>
#endif
#include <src/Parser.h>
#include <src/casl_math.h>

#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>

#include <vector>

using namespace std;
static PetscErrorCode ierr;
static param_list_t pl;

//-------------------------------------
// computational domain parameters
//-------------------------------------
param_t<double> xmin (pl, -3, "xmin", "Box xmin");
param_t<double> ymin (pl, -3, "ymin", "Box ymin");
param_t<double> zmin (pl, -3, "zmin", "Box zmin");

param_t<double> xmax (pl, 3, "xmax", "Box xmax");
param_t<double> ymax (pl, 3, "ymax", "Box ymax");
param_t<double> zmax (pl, 3, "zmax", "Box zmax");

param_t<int> nx (pl, 1, "nx", "Number of trees in the x-direction");
param_t<int> ny (pl, 1, "ny", "Number of trees in the y-direction");
param_t<int> nz (pl, 1, "nz", "Number of trees in the z-direction");

param_t<bool> px (pl, 0, "px", "Periodicity in the x-direction (0/1)");
param_t<bool> py (pl, 0, "py", "Periodicity in the y-direction (0/1)");
param_t<bool> pz (pl, 0, "pz", "Periodicity in the z-direction (0/1)");

//-------------------------------------
// refinement parameters
//-------------------------------------
param_t<int> lmin  (pl, 6,   "lmin", "Min level of the tree");
param_t<int> lmax  (pl, 7,  "lmax", "Max level of the tree");
param_t<int> lip   (pl, 1.1, "lip" , "Refinement transition");
param_t<int> band  (pl, 2,   "band" , "Uniform grid band");

//-------------------------------------
// polymer parameters
//-------------------------------------
param_t<double> XN       (pl, 20,  "XN"      , "Interaction strength between A and B blocks");
param_t<double> gamma_a  (pl, 0, "gamma_a" , "Surface tension of A block");
param_t<double> gamma_b  (pl, 1, "gamma_b" , "Surface tension of B block");
param_t<int>    ns       (pl, 40,  "ns"      , "Discretization of polymer chain");
param_t<double> box_size (pl, 1,   "box_size", "Box size in units of Rg");
param_t<double> f        (pl, .5,  "f"       , "Fraction of polymer A");

//-------------------------------------
// scft parameters
//-------------------------------------
param_t<int>    seed                (pl, 2,     "seed", "Seed field for scft: 0 - zero, 1 - random, 2 - vertical stripes, 3 - horizontal stripes, 4 - diagonal stripes, 5 - dots");
param_t<double> scft_tol            (pl, 1.e-4, "scft_tol"           , "Tolerance for SCFT");
param_t<int>    max_scft_iterations (pl, 300,   "max_scft_iterations", "Maximum SCFT iterations");
param_t<int>    bc_adjust_min       (pl, 1,     "bc_adjust_min"      , "Minimun SCFT steps between adjusting BC");
param_t<bool>   smooth_pressure     (pl, 1,     "smooth_pressure"    , "Smooth pressure after first BC adjustment 0/1");
param_t<double> scft_bc_tol         (pl, 4.e-2, "scft_bc_tol"        , "Tolerance for adjusting BC");

//-------------------------------------
// particle dynamics parameters
//-------------------------------------
param_t<bool>   use_scft   (pl, 0,       "use_scft", "Turn on/off SCFT (0/1)");
param_t<bool>   minimize   (pl, 0,       "minimize", "Turn on/off energy minimization (0/1)");
param_t<int>    geometry   (pl, 0,       "geometry", "Initial placement of particles: 0 - one particle, 1 - ...");
param_t<int>    velocity   (pl, 0,       "velocity", "Predifined velocity in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<int>    rotation   (pl, 0,       "rotation", "Predifined rotation in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<double> CFL        (pl, 0.5,     "CFL", "CFL number");
param_t<double> time_limit (pl, DBL_MAX, "time_limit", "Time limit");
param_t<int>    step_limit (pl, 200,     "step_limit", "Step limit");

param_t<int>    pairwise_potential_type  (pl, 0, "pairwise_potential_type", "Type of pairwise potential: 0 - quadratic, 1 - 1/(e^x-1)");
param_t<double> pairwise_potential_mag   (pl, 1.0,   "pairwise_potential_mag", "Magnitude of pairwise potential");
param_t<double> pairwise_potential_width (pl, 10, "pairwise_potential_width", "Width of pairwise potential");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool> save_energy (pl, 1,  "save_energy", "Save effective energy into a file");
param_t<bool> save_vtk    (pl, 1,  "save_vtk", "Save vtk data");
param_t<int>  save_freq   (pl, 1, "save_freq", "Frequency of saving vtk data");

struct particle_t
{
  double xyz [P4EST_DIM];
  double axis[P4EST_DIM];
  double rot;
  CF_DIM *phi;
  CF_DIM *gA;
  CF_DIM *gB;

  particle_t()
  {
    foreach_dimension(dim)
    {
      xyz [dim] = 0;
      axis[dim] = 0;
    }

    rot = 0;
    phi = NULL;
    gA  = NULL;
    gB  = NULL;
  }

//  ~particle_t()
//  {
//    if (phi != NULL) { delete phi; phi = NULL; }
//    if (gA  != NULL) { delete gA;  gA  = NULL; }
//    if (gB  != NULL) { delete gB;  gB  = NULL; }
//  }

  double phi_value(DIM(double x, double y, double z))
  {
    double X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot);
    double Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot);

    return (*phi)(DIM(X,Y,Z));
  }

  double gamma_A_value(DIM(double x, double y, double z))
  {
    double X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot);
    double Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot);

    return (*gA)(DIM(X,Y,Z));
  }

  double gamma_B_value(DIM(double x, double y, double z))
  {
    double X = (x-xyz[0])*cos(rot)-(y-xyz[1])*sin(rot);
    double Y = (x-xyz[0])*sin(rot)+(y-xyz[1])*cos(rot);

    return (*gB)(DIM(X,Y,Z));
  }

};

// this class constructs the 'analytical field' (level-set function for every (x,y) coordinate)
class particles_level_set_cf_t : public CF_DIM
{
private:
  std::vector<particle_t> *particles;

public:

  particles_level_set_cf_t(std::vector<particle_t> &particles)
    : particles(&particles) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    int np = particles->size();
    vector<double> phi_particles(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_particles[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_particles[n] = MAX(phi_particles[n],
                                   particles->at(n).phi_value(DIM(x+double(i)*(xmax()-xmin()),
                                                                  y+double(j)*(ymax()-ymin()),
                                                                  z+double(k)*(zmax()-zmin()))));
          }
    }

    double current_max = phi_particles[0];
    for(int n = 1; n < np; n++)
    {
      current_max = MAX(current_max, phi_particles[n]);
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
    vector<double> phi_particles(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_particles[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_particles[n] = MAX(phi_particles[n],
                                   particles->at(n).phi_value(DIM(x+double(i)*(xmax()-xmin()),
                                                                  y+double(j)*(ymax()-ymin()),
                                                                  z+double(k)*(zmax()-zmin()))));
          }
    }

    int particle_num = 0;
    for (int n = 1; n < np; n++)
    {
      if (phi_particles[particle_num] < phi_particles[n])
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
    vector<double> phi_particles(np);

    int i_min = 0, i_max = 0;
    int j_min = 0, j_max = 0;
    int k_min = 0, k_max = 0;

    for (int n = 0; n < np; ++n)
    {
      phi_particles[n] = -100;
      XCODE( if (px()) { x < particles->at(n).xyz[0] ? i_max = 1 : i_min = -1; } );
      YCODE( if (py()) { y < particles->at(n).xyz[1] ? j_max = 1 : j_min = -1; } );
      ZCODE( if (pz()) { z < particles->at(n).xyz[2] ? k_max = 1 : k_min = -1; } );

      for (int k = k_min; k <= k_max; ++k)
        for (int j = j_min; j <= j_max; ++j)
          for (int i = i_min; i <= i_max; ++i)
          {
            phi_particles[n] = MAX(phi_particles[n],
                                   particles->at(n).phi_value(DIM(x+double(i)*(xmax()-xmin()),
                                                                  y+double(j)*(ymax()-ymin()),
                                                                  z+double(k)*(zmax()-zmin()))));
          }
    }

    int particle_num = 0;
    for (int n = 1; n < np; n++)
    {
      if (phi_particles[particle_num] < phi_particles[n])
      {
        particle_num = n;
      }
    }

    double sum = 0;


    for (int n = 0; n < np; n++)
    {
      if (n != particle_num)
      {
        sum += pairwise_potential(fabs(phi_particles[n])-pairwise_potential_width());
      }
    }

    return sum;
  }
};


double bulk_lamellar_spacing = 1;


// this class constructs the exchange potential (mu_minus) for every (x,y), that describes the interaction between A&B (A, B are the monomer species)
class mu_minus_cf_t : public CF_DIM {
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (seed())
    {
      case 0: return 0;
      case 1: return 0.01*XN()*(double)(rand()%1000)/1000.;
      case 2:
      {
        double nx = (xmax()-xmin())/bulk_lamellar_spacing; if (px()) nx = round(nx);
        return .5*XN()*cos(2.*PI*x/(xmax()-xmin())*nx);
      }
      case 3:
      {
        double ny = (ymax()-ymin())/bulk_lamellar_spacing;
        return .5*XN()*cos(2.*PI*y/(ymax()-ymin())*ny);
      }
      case 4:
      {
        double nx = .5*(xmax()-xmin())/bulk_lamellar_spacing; if (px()) nx = round(nx);
        double ny = .5*(ymax()-ymin())/bulk_lamellar_spacing; if (py()) ny = round(ny);
        return .5*XN()*cos(2.*PI*x/(xmax()-xmin())*nx + 2.*PI*y/(ymax()-ymin())*ny);
      }
      case 5:
      {
        double nx = .5*(xmax()-xmin())/bulk_lamellar_spacing; if (px()) nx = round(nx);
        double ny = .5*(ymax()-ymin())/bulk_lamellar_spacing; if (py()) ny = round(ny);
#ifdef P4_TO_P8
        double nz = .5*(zmax()-zmin())/bulk_lamellar_spacing; if (pz()) ny = round(nz);
#endif
        return .5*XN()*MULTD(cos(PI*x/(xmax()-xmin())*nx),
                             cos(PI*y/(ymax()-ymin())*ny),
                             cos(PI*z/(zmax()-zmin())*nz));
      }
    }
  }
};

class radial_gamma_cf_t : public CF_DIM
{
  double gamma_max;
  double gamma_min;
  double k;
public:
  radial_gamma_cf_t(double gamma_min, double gamma_max, double k)
    : gamma_min(gamma_min), gamma_max(gamma_max), k(k) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    double t = atan2(y,x);
    return gamma_min + (gamma_max-gamma_min)*0.5*(1.+cos(k*t));
  }
};

void initalize_particles(std::vector<particle_t> &particles)
{
  particles.clear();
  particle_t p;
  switch (geometry())
  {
    case 0:
    {

//            static radial_phi_t sphere(VAL, 0.6, DIM(0,0,0), -1);
      static flower_phi_t sphere(0.6, DIM(0,0,0), 0.0, -1);

      static radial_gamma_cf_t gA(gamma_a(), 0, 0);
      static radial_gamma_cf_t gB(0, gamma_b(), 0);

      p.xyz[0] = -1.50;
      p.xyz[1] = -0.45;
      p.phi = &sphere;
      p.gA  = &gA;
      p.gB  = &gB;

      particles.push_back(p);
    }
      break;

    case 1:
    {

      static flower_phi_t  sphere(0.5, DIM(0,0,0), 0.0, -1);
      static flower_phi_t  star(0.5, DIM(0,0,0), 0.1, -1);
      static capsule_phi_t rod(VAL, 0.25, 0,0, 1, -1);

      static radial_gamma_cf_t gA(gamma_a(), 0, 1);
      static radial_gamma_cf_t gB(0, gamma_b(), 1);

      p.xyz[0] = -1.50; p.xyz[1] = -0.45;

      p.gA  = &gA;
      p.gB  = &gB;


      p.phi = &sphere; p.xyz[0] = -1.30; p.xyz[1] = -1.50; particles.push_back(p);
      p.phi = &star;   p.xyz[0] = +1.50; p.xyz[1] = -0.20; particles.push_back(p);
      p.phi = &rod;    p.xyz[0] = -0.90; p.xyz[1] = +0.75; particles.push_back(p);
//      p.xyz[0] = +2.20; p.xyz[1] = -2.90; particles.push_back(p);
    }
      break;

//    case 2:

//      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-3.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-4.10); yc.push_back(+1.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.60); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+1.50); yc.push_back(+4.10); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.60); xc.push_back(+4.80); yc.push_back(-3.60); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+5.00); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+4.90); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+1.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-0.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-5.10); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-5.00); yc.push_back(+5.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-0.20); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.60); xc.push_back(+5.30); yc.push_back(+0.90); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(+3.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+5.20); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-5.00); yc.push_back(-1.80); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-3.50); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+1.20); yc.push_back(-5.40); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-5.30); yc.push_back(+2.70); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-0.20); yc.push_back(-1.00); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 3:

//      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-3.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-4.10); yc.push_back(+1.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.40); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+1.50); yc.push_back(+4.10); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.40); xc.push_back(+4.80); yc.push_back(-3.60); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+5.00); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+4.90); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+1.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-0.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-5.10); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-5.00); yc.push_back(+5.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-0.20); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.40); xc.push_back(+5.30); yc.push_back(+0.90); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(+3.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+5.20); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-5.00); yc.push_back(-1.80); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-3.50); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+1.20); yc.push_back(-5.40); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-5.30); yc.push_back(+2.70); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-0.20); yc.push_back(-1.00); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 4:

//      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-3.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-4.10); yc.push_back(+1.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.20); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+1.50); yc.push_back(+4.10); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.20); xc.push_back(+4.80); yc.push_back(-3.60); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+5.00); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+4.90); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+1.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-0.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-5.10); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-5.00); yc.push_back(+5.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-0.20); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.20); xc.push_back(+5.30); yc.push_back(+0.90); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(+3.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+5.20); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-5.00); yc.push_back(-1.80); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-3.50); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+1.20); yc.push_back(-5.40); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-5.30); yc.push_back(+2.70); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-0.20); yc.push_back(-1.00); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 5:

//      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-3.75); yc.push_back(+2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-5.10); yc.push_back(-0.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.60); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+1.50); yc.push_back(+4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 6:

//      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-3.75); yc.push_back(+2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-5.10); yc.push_back(-0.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.40); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+1.50); yc.push_back(+4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;


//  case 7:

//      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-3.75); yc.push_back(+2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-5.10); yc.push_back(-0.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      radius.push_back(+0.20); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+1.50); yc.push_back(+4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 8:

//      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-3.75); yc.push_back(+4.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(-5.10); yc.push_back(-4.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.60); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 9:

//      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-3.75); yc.push_back(+4.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(-5.10); yc.push_back(-4.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.40); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

//  case 10:

//      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-3.75); yc.push_back(+4.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(-5.10); yc.push_back(-4.50); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
//      radius.push_back(+0.20); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

//      np = radius.size();

//    break;

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
    return particles->at(int((*number)(DIM(x, y, z)))).gamma_A_value(DIM(x, y, z));
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
    return particles->at(int((*number)(DIM(x, y, z)))).gamma_B_value(DIM(x, y, z));
  }
};


int main(int argc, char** argv)
{
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // parse command line arguments for parameters
  cmdParser cmd;
  pl.initialize_parser(cmd);
  cmd.parse(argc, argv);
  pl.set_from_cmd_all(cmd);

  // define some auxiliary variables
  double scaling = 1./box_size();
  double xyz_min [] = { DIM(xmin(), ymin(), zmin()) };
  double xyz_max [] = { DIM(xmax(), ymax(), zmax()) };
  int    nb_trees[] = { DIM(nx(), ny(), nz()) };
  int    periodic[] = { DIM(px(), py(), pz()) };

  pairwise_potential_width.val *= (xmax()-xmin())/pow(2.,lmax());

  bulk_lamellar_spacing = 2.*pow(8.*XN()/3./pow(PI,4.),1./6.)/box_size();

  // define the output directory
  const char *out_dir = getenv("OUT_DIR");
  if (!out_dir) out_dir = ".";
  else
  {
    if (mpi.rank() == 0)
    {
      std::ostringstream command;
      command << "mkdir -p " << out_dir;
      int ret_sys = system(command.str().c_str());
      if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR directory");
    }
  }

  if (save_vtk() && mpi.rank() == 0)
  {
    std::ostringstream command;
    command << "mkdir -p " << out_dir << "/vtu";
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0) throw std::invalid_argument("could not create OUT_DIR/vtu directory");
  }

  // make a separate file with the energies in every iteration (for Matlab plot)
  FILE *file_energy;
  char file_energy_name[1000];
  if (save_energy())
  {
    sprintf(file_energy_name, "%s/convergence.dat", out_dir);

    ierr = PetscFOpen(mpi.comm(), file_energy_name, "w", &file_energy); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(), file_energy, "iteration "
                                                 "effective_energy "
                                                 "expected_dEdt "
                                                 "expected_dE "
                                                 "effective_dE"
                                                 "mu_minus integral\n"); CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
  }

  // stopwatch
  parStopWatch w;
  w.start("Running example: scft_particles");

  // ---------------------------------------------------------
  // initialize particles
  // ---------------------------------------------------------
  vector<particle_t> particles;
  initalize_particles(particles);
  int np = particles.size();

  // auxiliary grid variables to interpolate mu_minus from previous time step
  p4est_connectivity_t *connectivity_old;
  p4est_t *p4est_old;
  p4est_ghost_t *ghost_old;
  p4est_nodes_t *nodes_old;

  my_p4est_hierarchy_t *hierarchy_old;
  my_p4est_node_neighbors_t *ngbd_old;

  Vec mu_minus_old;
  Vec mu_plus_old;

  // create the mu_minus field
  mu_minus_cf_t mu_minus_cf;

  int step = 0;
  double t = 0;
  double energy = 0;
  double energy_previous = 0; //needed to calculate the energy difference between two iterations

  double rho_old = 1;

  vector< vector<double> > v_old(P4EST_DIM, vector<double> (np, 0));

  vector<double> w_old  (np, 0);

  vector<double> dt_old (np, 1);
  vector<double> dtw_old(np, 1);

  while (t < time_limit() && step < step_limit())
  {
    // create particle_level_set
    particles_level_set_cf_t particles_level_set_cf(particles);

    // create get_particle_num (creates a field, that shows number of closest particle)
    particles_number_cf_t particles_number_cf(particles);

    // ---------------------------------------------------------
    // create computational grid
    // ---------------------------------------------------------
    my_p4est_brick_t brick;
    p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
    p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin(), lmax(), &particles_level_set_cf, lip(), band());
    data.set_refine_only_inside(true);
    p4est->user_pointer = (void*)(&data);

    // set P4EST_TRUE, because interface is moving (need to refine and partition the grid)
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
    my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
    ngbd->init_neighbors();

    // ---------------------------------------------------------
    // allocate fields
    // ---------------------------------------------------------
    Vec     mu_minus;
    double *mu_minus_ptr;

    Vec     mu_plus;
    double *mu_plus_ptr;

    Vec     particles_level_set;
    double *particles_level_set_ptr;

    Vec     particles_number;
    double *particles_number_ptr;

    Vec     normal    [P4EST_DIM];
    double *normal_ptr[P4EST_DIM];

    Vec     kappa;
    double *kappa_ptr;

    Vec     gamma_eff;
    double *gamma_eff_ptr;

    Vec     gamma_dif;
    double *gamma_dif_ptr;

    Vec     gamma_avg;
    double *gamma_avg_ptr;

    Vec     gamma_eff_grad    [P4EST_DIM];
    double *gamma_eff_grad_ptr[P4EST_DIM];

    Vec     gamma_dif_grad    [P4EST_DIM];
    double *gamma_dif_grad_ptr[P4EST_DIM];

    Vec     gamma_avg_grad    [P4EST_DIM];
    double *gamma_avg_grad_ptr[P4EST_DIM];

    Vec     velo;
    double *velo_ptr;

    Vec     Gx;
    double *Gx_ptr;

    Vec     Gy;
    double *Gy_ptr;

    Vec     Gxy;
    double *Gxy_ptr;

    ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &mu_plus);             CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &particles_level_set); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &particles_number);    CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &kappa);               CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_eff);           CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_avg);           CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_dif);           CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &velo);                CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &Gx);                  CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &Gy);                  CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &Gxy);                 CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal[dim], &gamma_eff_grad[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal[dim], &gamma_dif_grad[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal[dim], &gamma_avg_grad[dim]); CHKERRXX(ierr);
    }

    // ---------------------------------------------------------
    // initialize mu_minus field
    // ---------------------------------------------------------
    if (step == 0 || !use_scft())
    {
      sample_cf_on_nodes(p4est, nodes, mu_minus_cf, mu_minus);
    }
    else
    {
      // by interpolating the mu_minus field, the calculation is accelerated AND it 'stays on the same track' (since there are multiple solutions in scft)
      my_p4est_interpolation_nodes_t interp(ngbd_old);

      double xyz[P4EST_DIM];
      foreach_node(n, nodes)
      {
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
      }

      interp.set_input(mu_minus_old, linear);
      interp.interpolate(mu_minus);

      interp.set_input(mu_plus_old, linear);
      interp.interpolate(mu_plus);

      // destroy previous grid
      ierr = VecDestroy(mu_plus_old);  CHKERRXX(ierr);
      ierr = VecDestroy(mu_minus_old); CHKERRXX(ierr);

      delete hierarchy_old;
      delete ngbd_old;
      p4est_nodes_destroy(nodes_old);
      p4est_ghost_destroy(ghost_old);
      p4est_destroy      (p4est_old);
      p4est_connectivity_destroy(connectivity_old);
    }

    // ---------------------------------------------------------
    // sample particles geometry
    // ---------------------------------------------------------
    sample_cf_on_nodes(p4est, nodes, particles_level_set_cf, particles_level_set);
    sample_cf_on_nodes(p4est, nodes, particles_number_cf,    particles_number);

//    my_p4est_level_set_t ls(ngbd);
//    ls.reinitialize_2nd_order(particles_level_set, 50);

    double rho = 1;

    // calculate mu_minus and velo
    if (use_scft())
    {
      my_p4est_scft_t scft(ngbd, ns());

      gamma_Aa_all_cf_t gamma_Aa_all(particles, particles_number_cf);
      gamma_Ba_all_cf_t gamma_Ba_all(particles, particles_number_cf);
      // set geometry
      scft.add_boundary(particles_level_set, MLS_INTERSECTION, gamma_Aa_all, gamma_Ba_all);

      scft.set_scalling(scaling);
      scft.set_polymer(f(), XN());
      scft.set_rho_avg(rho_old);

      // initialize potentials
      Vec mu_minus_scft = scft.get_mu_m();
      Vec mu_plus_scft  = scft.get_mu_p();
      VecCopyGhost(mu_minus, mu_minus_scft);
//      VecCopyGhost(mu_plus,  mu_plus_scft);

      // initialize diffusion solvers for propagators
      scft.initialize_solvers();
      scft.initialize_bc_smart(step != 0);

      // main loop for solving SCFT equations
      int    scft_iteration = 0;
      int    bc_iters       = 0;
      double scft_error     = 2.*scft_tol()+1.;
      double scft_energy    = 0;

      while ((scft_iteration < max_scft_iterations() && scft_error > scft_tol()) || scft_iteration < bc_adjust_min()+1)
      {
        // do an SCFT step
        scft.solve_for_propogators();
        scft.calculate_densities();
        scft.update_potentials();

        if (scft.get_exchange_force() < scft_bc_tol() && bc_iters >= bc_adjust_min())
        {
          scft.initialize_bc_smart();
          if (smooth_pressure())
          {
            scft.smooth_singularity_in_pressure_field();
            smooth_pressure.val = false;
          }
          bc_iters = 0;
        }
        ierr = PetscPrintf(mpi.comm(), "SCFT iteration no. %4d, Energy: %+6.5e, Pressure: %3.2e, Exchange: %3.2e, dE scft: %+3.2e, dE: %+3.2e%s\n", scft_iteration, scft.get_energy(), scft.get_pressure_force(), scft.get_exchange_force(), scft.get_energy()-scft_energy, scft.get_energy()-energy_previous, bc_iters == 0 ? " (new bc)" : ""); CHKERRXX(ierr);

        scft_error = MAX(fabs(scft.get_pressure_force()), fabs(scft.get_exchange_force()));
        scft_iteration++;
        bc_iters++;
        scft_energy = scft.get_energy();
//        scft.save_VTK(scft_iteration);
      }

      // get energy
      energy = scft.get_energy();
      rho    = scft.get_rho_avg();
      rho_old = rho;

      scft.sync_and_extend();
      scft.compute_energy_shape_derivative(0, velo); //second part of dE/dt, which is derivative of integral of f dx

      VecCopyGhost(mu_minus_scft, mu_minus);
      VecCopyGhost(mu_plus_scft,  mu_plus);
    }
    else
    {
      VecSetGhost(velo, 0);
    }

    // calculate 'gamma_eff' by hand and not using 'sample_cf_on_nodes' (because mu_m is a vetor and not continuous function)
    ierr = VecGetArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    penalization_cf_t penalization(particles);

    foreach_node(n, nodes)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      int i = int(particles_number_ptr[n]);
      double ga = particles[i].gamma_A_value(xyz[0], xyz[1]);
      double gb = particles[i].gamma_B_value(xyz[0], xyz[1]);

      gamma_eff_ptr[n] = (0.5*(ga+gb)*rho + (ga-gb)*(mu_minus_ptr[n])/XN()) + penalization.value(xyz);
      gamma_avg_ptr[n] = (0.5*(ga+gb)*rho);
      gamma_dif_ptr[n] = ((ga-gb));
    }

    ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    if (!use_scft())
    {
      energy = integrate_over_interface(p4est, nodes, particles_level_set, gamma_eff);
      energy /= pow(scaling, P4EST_DIM-1.);
    }

    // ---------------------------------------------------------
    // calculate energy derivative with respect to particles' positions
    // ---------------------------------------------------------

    // compute normals, curvature and gradient of gamma_eff (needed for dE/dt formula)
    compute_normals_and_mean_curvature(*ngbd, particles_level_set, normal, kappa);
    ngbd->first_derivatives_central(gamma_eff, gamma_eff_grad);
    ngbd->first_derivatives_central(gamma_dif, gamma_dif_grad);
    ngbd->first_derivatives_central(gamma_avg, gamma_avg_grad);

    // calculate G
    ierr = VecGetArray(Gx,               &Gx_ptr);               CHKERRXX(ierr);
    ierr = VecGetArray(Gy,               &Gy_ptr);               CHKERRXX(ierr);
    ierr = VecGetArray(Gxy,              &Gxy_ptr);              CHKERRXX(ierr);
    ierr = VecGetArray(velo,             &velo_ptr);             CHKERRXX(ierr);
    ierr = VecGetArray(kappa,            &kappa_ptr);            CHKERRXX(ierr);
    ierr = VecGetArray(gamma_eff,        &gamma_eff_ptr);        CHKERRXX(ierr);
    ierr = VecGetArray(mu_minus,         &mu_minus_ptr);         CHKERRXX(ierr);
    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecGetArray(normal[dim],         &normal_ptr[dim]);         CHKERRXX(ierr);
      ierr = VecGetArray(gamma_eff_grad[dim], &gamma_eff_grad_ptr[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_dif_grad[dim], &gamma_dif_grad_ptr[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_avg_grad[dim], &gamma_avg_grad_ptr[dim]); CHKERRXX(ierr);
    }

    double factor = 1./pow(scaling, P4EST_DIM-1.);

    foreach_node(n, nodes)
    {
      double g = SUMD(normal_ptr[0][n]*gamma_eff_grad_ptr[0][n],
                      normal_ptr[1][n]*gamma_eff_grad_ptr[1][n],
                      normal_ptr[2][n]*gamma_eff_grad_ptr[2][n])*factor
          + gamma_eff_ptr[n]*kappa_ptr[n]*factor
          + velo_ptr[n];

      Gx_ptr[n] = g*normal_ptr[0][n] - (gamma_avg_grad_ptr[0][n] + (mu_minus_ptr[n])/XN()*gamma_dif_grad_ptr[0][n]);
      Gy_ptr[n] = g*normal_ptr[1][n] - (gamma_avg_grad_ptr[1][n] + (mu_minus_ptr[n])/XN()*gamma_dif_grad_ptr[1][n]);

      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      int       i = int(particles_number_ptr[n]);
      double delx = xyz[0]-particles[i].xyz[0];
      double dely = xyz[1]-particles[i].xyz[1];

      Gxy_ptr[n] = Gx_ptr[n]*dely - Gy_ptr[n]*delx;
    }

    ierr = VecRestoreArray(Gx,               &Gx_ptr);               CHKERRXX(ierr);
    ierr = VecRestoreArray(Gy,               &Gy_ptr);               CHKERRXX(ierr);
    ierr = VecRestoreArray(Gxy,              &Gxy_ptr);              CHKERRXX(ierr);
    ierr = VecRestoreArray(velo,             &velo_ptr);             CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa,            &kappa_ptr);            CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_eff,        &gamma_eff_ptr);        CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus,         &mu_minus_ptr);         CHKERRXX(ierr);
    ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecRestoreArray(normal[dim],         &normal_ptr[dim]);         CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_eff_grad[dim], &gamma_eff_grad_ptr[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_dif_grad[dim], &gamma_dif_grad_ptr[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_avg_grad[dim], &gamma_avg_grad_ptr[dim]); CHKERRXX(ierr);
    }

    // calculate
    vector< vector<double> > g(P4EST_DIM, vector<double> (np,0));

    vector<double> gw(np, 0);

    integrate_over_interface(np, g[0], p4est, nodes, particles_level_set, particles_number, Gx);
    integrate_over_interface(np, g[1], p4est, nodes, particles_level_set, particles_number, Gy);
    integrate_over_interface(np, gw,   p4est, nodes, particles_level_set, particles_number, Gxy);

    // additional terms due to particle-particle interactions

//    // calculate q_x and q_y (additional velocity terms; take into account what happens when particles come close to each other)
//    vector< vector<double> > q(P4EST_DIM, vector<double> (np,0));

//    double energy_term = 0;

//    for (int i = 0; i < np; i++)
//    {
//      for (int j = 0; j < np; j++)
//      {
//        if (i != j)
//        {
//          XCODE( double dx = particles[i].xyz[0]-particles[j].xyz[0] );
//          YCODE( double dy = particles[i].xyz[1]-particles[j].xyz[1] );
//          ZCODE( double dz = particles[i].xyz[2]-particles[j].xyz[2] );

//          double dist =  sqrt(SUMD(SQR(dx), SQR(dy), SQR(dz)))-0;

//          double force = pairwise_force(dist);

//          if (force != force) throw;

//          XCODE( q[0][i] += (dx/dist)*force ); // u = a*exp(-r^2/(2*c^2)) => use derivative of u for multiplication
//          YCODE( q[1][i] += (dy/dist)*force );
//          ZCODE( q[2][i] += (dy/dist)*force );

//          energy_term = energy_term + pairwise_potential(dist);
//        }
//      }
//    }

//    // add correction term to the energy (when particles come too close to each other, 'push' them away)
//    energy += energy_term;

//    for (int i = 0; i < np; ++i)
//    {
//      foreach_dimension(dim) g[dim][i] += q[dim][i];
//    }

    // ---------------------------------------------------------
    // select velocity and move particles
    // ---------------------------------------------------------
    vector< vector<double> > v(P4EST_DIM, vector<double> (np,0));
    vector<double> w(np, 0);

    if (minimize())
    {
      for(int i = 0; i < np; i++)
      {
        foreach_dimension(dim) v[dim][i] = -g[dim][i];
        w[i] = -gw[i];
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

    // find the maximum velocity (needed to calculate the timestep delta_t)

    vector<double> arm_max(np,0);

    double phi_thresh = 0;

    ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(particles_level_set, &particles_level_set_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      if (particles_level_set_ptr[n] > phi_thresh)
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
    ierr = VecRestoreArray(particles_level_set, &particles_level_set_ptr); CHKERRXX(ierr);

    ierr = MPI_Allreduce(MPI_IN_PLACE, arm_max.data(), np, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); CHKERRXX(ierr);

    double v_max  = 0;
    double wr_max = 0;
    for (int i = 0; i < np; ++i)
    {
      v[0][i] = v[0][i];
      v[1][i] = v[1][i];
      w[i] = w[i];
      v_max  = MAX(v_max, ABSD(v[0][i], v[1][i], v[2][i]));
      wr_max = MAX(wr_max, fabs(w[i])*arm_max[i]);
    }

    // find the diagonal length of the smallest quadrant (needed to calculate the timestep delta_t)
    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    // calculate timestep delta_t, such that it's small enough to capture 'everything'
    double delta_t  = diag_min*CFL()/MAX(v_max, 0.001);
    double delta_tw = diag_min*CFL()/MAX(wr_max,0.001);

    vector<double> dt(np);
    vector<double> dtw(np);

    // move particles
    for (int j = 0; j < np; ++j)
    {
      if (!minimize())
      {
        dt [j] = delta_t;
        dtw[j] = delta_tw;
      }
      else
      {
        double gxx = -(v[0][j] - v_old[0][j])/dt_old[j]/v_old[0][j];
        double gyy = -(v[1][j] - v_old[1][j])/dt_old[j]/v_old[1][j];
        double gxy = -(v[1][j] - v_old[1][j])/dt_old[j]/v_old[0][j];
        double gyx = -(v[0][j] - v_old[0][j])/dt_old[j]/v_old[1][j];

        double dt_tmp = -( g[0][j]*v[0][j] + g[1][j]*v[1][j] )/( gxx*v[0][j]*v[0][j] + .5*(gxy+gyx)*v[0][j]*v[1][j] + gyy*v[1][j]*v[1][j] );

        double dt_max = diag_min*CFL()/ABSD(v[0][j], v[1][j], v[2][j]);

        if (dt_tmp < 0 || dt_tmp > dt_max || step == 0) dt[j] = dt_max;
        else dt[j] = dt_tmp;

        double gww = -(w[j] - w_old[j])/dtw_old[j]/w_old[j];
        double dtw_tmp = - gw[j]/gww/w[j];

        double dtw_max = diag_min*CFL()/fabs(w[j]*arm_max[j]);

        if (dtw_tmp < 0 || dtw_tmp > dtw_max || step == 0) dtw[j] = dtw_max;
        else dtw[j] = dtw_tmp;

        dt[j] = dt_max;
        dtw[j] = dtw_max;
      }
      XCODE( particles[j].xyz[0] += v[0][j]*dt[j] );
      YCODE( particles[j].xyz[1] += v[1][j]*dt[j] );
      ZCODE( particles[j].xyz[2] += v[2][j]*dt[j] );

      particles[j].rot += w[j]*dtw[j];
    }

    v_old = v;
    dt_old = dt;
    w_old = w;
    dtw_old = dtw;

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

    // calculate the expected and effective changes in energy (compare step i with step i+1)
    // compute expected change in energy (for validation)
    double dE_dt_expected = 0;
    for (int i = 0; i < np; i++)
    {
      dE_dt_expected += SUMD(v[0][i]*g[0][i],
                             v[1][i]*g[1][i],
                             v[2][i]*g[2][i])*dt[i];
      dE_dt_expected += w[i]*gw[i]*dtw[i];
    }
    double expected_change_in_energy  = dE_dt_expected;
    double effective_change_in_energy = energy - energy_previous;

    ierr = PetscPrintf(mpi.comm(), "Iteration no. %4d: time = %3.2e, dt = %3.2e, vmax = %3.2e, E = %6.5e, dE = %+3.2e / %+3.2e, dE_dt_expected= %6.5e \n", step, t, delta_t, v_max, energy, expected_change_in_energy, effective_change_in_energy, dE_dt_expected); CHKERRXX(ierr);

    // ---------------------------------------------------------
    // save info into files
    // ---------------------------------------------------------

    double mu_minus_integral;
    mu_minus_integral = integrate_over_interface(p4est, nodes, particles_level_set, mu_minus);


    // write the energy in every iteration into the separate file
    if (save_energy())
    {
      ierr = PetscFOpen(mpi.comm(), file_energy_name, "a", &file_energy); CHKERRXX(ierr);
      PetscFPrintf(mpi.comm(), file_energy, "%d %12.11e %3.2e %3.2e %3.2e %f\n", step, energy, dE_dt_expected, expected_change_in_energy, effective_change_in_energy, mu_minus_integral);
      ierr = PetscFClose(mpi.comm(), file_energy); CHKERRXX(ierr);
    }

    // save as vtk files
    if (save_vtk() && (step%save_freq()==0))
    {
      std::ostringstream oss;
      oss << out_dir
          << "/vtu/scft_"
          << p4est->mpisize
          << "_" << int(round(step/save_freq()));

      ierr = VecGetArray(particles_level_set, &particles_level_set_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             6, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", particles_level_set_ptr,
                             VTK_POINT_DATA, "particles_number", particles_number_ptr,
                             VTK_POINT_DATA, "gamma_effective", gamma_eff_ptr,
                             VTK_POINT_DATA, "gamma_diff", gamma_dif_ptr,
                             VTK_POINT_DATA, "gamma_avg", gamma_avg_ptr,
                             VTK_POINT_DATA, "mu_minus", mu_minus_ptr);

      ierr = VecRestoreArray(particles_level_set, &particles_level_set_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_dif, &gamma_dif_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_avg, &gamma_avg_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
    }

    // ---------------------------------------------------------
    // clean-up memory
    // ---------------------------------------------------------
    ierr = VecDestroy(particles_number);    CHKERRXX(ierr);
    ierr = VecDestroy(kappa);               CHKERRXX(ierr);
    ierr = VecDestroy(gamma_eff);           CHKERRXX(ierr);
    ierr = VecDestroy(gamma_dif);           CHKERRXX(ierr);
    ierr = VecDestroy(gamma_avg);           CHKERRXX(ierr);
    ierr = VecDestroy(velo);                CHKERRXX(ierr);
    ierr = VecDestroy(Gx);                  CHKERRXX(ierr);
    ierr = VecDestroy(Gy);                  CHKERRXX(ierr);
    ierr = VecDestroy(Gxy);                 CHKERRXX(ierr);
    ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecDestroy(gamma_eff_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(gamma_dif_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(gamma_avg_grad[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }

    if (use_scft())
    {
      mu_minus_old = mu_minus;
      mu_plus_old  = mu_plus;

      connectivity_old = connectivity;
      p4est_old        = p4est;
      ghost_old        = ghost;
      nodes_old        = nodes;
      hierarchy_old    = hierarchy;
      ngbd_old         = ngbd;
    }
    else
    {
      ierr = VecDestroy(mu_plus); CHKERRXX(ierr);
      ierr = VecDestroy(mu_minus); CHKERRXX(ierr);

      delete hierarchy;
      delete ngbd;
      p4est_nodes_destroy(nodes);
      p4est_ghost_destroy(ghost);
      p4est_destroy      (p4est);
      p4est_connectivity_destroy(connectivity);
    }

    t += delta_t;
    step++;

    // store energy to compare with next step
    energy_previous = energy;
  }

  if (use_scft())
  {
    ierr = VecDestroy(mu_plus_old); CHKERRXX(ierr);
    ierr = VecDestroy(mu_minus_old); CHKERRXX(ierr);

    delete hierarchy_old;
    delete ngbd_old;
    p4est_nodes_destroy(nodes_old);
    p4est_ghost_destroy(ghost_old);
    p4est_destroy      (p4est_old);
    p4est_connectivity_destroy(connectivity_old);
  }

  // show total running time
  w.stop(); w.read_duration();

  return 0;
}
