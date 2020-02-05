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
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif
#include <src/my_p4est_level_set.h>
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
param_t<double> xmin (pl, -6, "xmin", "Box xmin");
param_t<double> ymin (pl, -6, "ymin", "Box ymin");
param_t<double> zmin (pl, -6, "zmin", "Box zmin");

param_t<double> xmax (pl, 6, "xmax", "Box xmax");
param_t<double> ymax (pl, 6, "ymax", "Box ymax");
param_t<double> zmax (pl, 6, "zmax", "Box zmax");

param_t<int> nx (pl, 1, "nx", "Number of trees in the x-direction");
param_t<int> ny (pl, 1, "ny", "Number of trees in the y-direction");
param_t<int> nz (pl, 1, "nz", "Number of trees in the z-direction");

param_t<bool> px (pl, 1, "px", "Periodicity in the x-direction (0/1)");
param_t<bool> py (pl, 1, "py", "Periodicity in the y-direction (0/1)");
param_t<bool> pz (pl, 1, "pz", "Periodicity in the z-direction (0/1)");

//-------------------------------------
// refinement parameters
//-------------------------------------
param_t<int> lmin (pl, 7, "lmin", "Min level of the tree");
param_t<int> lmax (pl, 9, "lmax", "Max level of the tree");
param_t<int> lip  (pl, 2, "lip" , "Refinement transition");

//-------------------------------------
// polymer parameters
//-------------------------------------
param_t<double> XN       (pl, 20,  "XN"      , "Interaction strength between A and B blocks");
param_t<double> gamma_a  (pl, 1, "gamma_a" , "Surface tension of A block");
param_t<double> gamma_b  (pl, 0, "gamma_b" , "Surface tension of B block");
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
param_t<bool>   use_scft   (pl, 1,       "use_scft", "Turn on/off SCFT (0/1)");
param_t<bool>   minimize   (pl, 1,       "minimize", "Turn on/off energy minimization (0/1)");
param_t<int>    geometry   (pl, 6,       "geometry", "Initial placement of particles: 0 - one particle, 1 - ...");
param_t<int>    velocity   (pl, 3,       "velocity", "Predifined velocity in case of minimize=0: 0 - along x-axis, 1 - along y-axis, 2 - diagonally, 3 - circular");
param_t<double> CFL        (pl, 1,     "CFL", "CFL number");
param_t<double> time_limit (pl, DBL_MAX, "time_limit", "Time limit");
param_t<int>    step_limit (pl, 200,     "step_limit", "Step limit");

param_t<double> pairwise_potential_mag   (pl, 0.3, "pairwise_potential_mag", "Magnitude of pairwise potential");
param_t<double> pairwise_potential_width (pl, 0.01, "pairwise_potential_width", "Width of pairwise potential");

//-------------------------------------
// output parameters
//-------------------------------------
param_t<bool> save_energy (pl, 1,  "save_energy", "Save effective energy into a file");
param_t<bool> save_vtk    (pl, 1,  "save_vtk", "Save vtk data");
param_t<int>  save_freq   (pl, 10, "save_freq", "Frequency of saving vtk data");

// this class constructs the 'analytical field' (level-set function for every (x,y) coordinate)
class particles_level_set_cf_t : public CF_DIM
{
private:
  int np;
  vector<double> R;
  vector<double> xc;
  vector<double> yc;
  vector<double> zc;

public:

  particles_level_set_cf_t(int np_, vector<double> R_, DIM(vector<double> xc_, vector<double> yc_, vector<double> zc_))
    : np(np_), R(R_), DIM(xc(xc_), yc(yc_), zc(zc_)) {}

  void change_coordinates_of_particle(vector<double> xc_new_, vector<double> yc_new_, vector<double> zc_new_){
    xc = xc_new_;
    yc = yc_new_;
    zc = zc_new_;
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    vector<double> phi_particles(np);
    vector<double> distances(4, 0); // vector with 4 rows, all of the entities are 0
    double xc_mirror = 0;
    double yc_mirror = 0;
    double delta_x = 0;
    double delta_y =0;
    double distance_min = 0;

    // if we have periodicity in all directions, we assume an INFINITE DOMAIN
    if (px()==1 && py()==1 && pz()==1){
      for (int j = 0; j < np; ++j)
      {
        delta_x = xc[j] - x;
        delta_y = yc[j] - y;

        // decide where the mirror x and y coordinates are depending on distance of xc/yc to the node (x,y)
        if(delta_x > 0){
          xc_mirror = xc[j] - (xmax()-xmin());
        }
        else
          xc_mirror = xc[j] + (xmax()-xmin());

        if(delta_y > 0){
          yc_mirror = yc[j] - (xmax()-xmin());
        }
        else
          yc_mirror = yc[j] + (xmax()-xmin());
        // the initial particle and the three mirrored particles are at: (xc, yc), (xc_mirror, yc), (xc, yc_mirror) and (xc_mirror, yc_mirror)

        // calculate the distance from all 4 particles to node (x,y)
        distances[0] = sqrt((SQR(x-xc[j]) + SQR(y-yc[j])));
        distances[1] = sqrt((SQR(x-xc_mirror) + SQR(y-yc[j])));
        distances[2] = sqrt((SQR(x-xc[j]) + SQR(y-yc_mirror)));
        distances[3] = sqrt((SQR(x-xc_mirror) + SQR(y-yc_mirror)));

        // find the particle which has the minimum distance to the node (x,y)
        distance_min = distances[0];
        for(int k = 1; k < distances.size(); k++){
          if (distance_min > distances[k]){
            distance_min = distances[k];
          }
        }
        // calculate the level-set function which includes the new, mirrored particles
        phi_particles[j] =  R[j] - distance_min;
      }
    }

    else{
      for (int j = 0; j < np; ++j)
      {
        phi_particles[j] =  R[j] - sqrt( SUMD(SQR(x-xc[j]),
                                              SQR(y-yc[j]),
                                              SQR(z-zc[j])) );
      }
    }

    double current_max = phi_particles[0];
    for(int k = 1; k < np; k++)
    {
      current_max = MAX(current_max, phi_particles[k]);
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
  int np;
  vector<double> R;
  vector<double> xc;
  vector<double> yc;
  vector<double> zc;

public:
  particles_number_cf_t(int np_, vector<double> R_, DIM(vector<double> xc_, vector<double> yc_, vector<double> zc_))
    : np(np_), R(R_), DIM(xc(xc_), yc(yc_), zc(zc_)) {}

  double operator()(DIM(double x, double y, double z)) const
  {
    vector<double> phi_particles(np);
    vector<double> distances(4, 0); // vector with 4 rows, all of the entities are 0
    double xc_mirror = 0;
    double yc_mirror = 0;
    double delta_x = 0;
    double delta_y =0;
    double distance_min = 0;

    // if we have periodicity in all directions, we assume an INFINITE DOMAIN
    if (px()==1 && py()==1 && pz()==1){
      for (int j = 0; j < np; ++j)
      {
        delta_x = xc[j] - x;
        delta_y = yc[j] - y;

        // decide where the mirror x and y coordinates are depending on distance of xc/yc to the node (x,y)
        if(delta_x > 0){
          xc_mirror = xc[j] - (xmax()-xmin());
        }
        else
          xc_mirror = xc[j] + (xmax()-xmin());

        if(delta_y > 0){
          yc_mirror = yc[j] - (xmax()-xmin());
        }
        else
          yc_mirror = yc[j] + (xmax()-xmin());
        // the initial particle and the three mirrored particles are at: (xc, yc), (xc_mirror, yc), (xc, yc_mirror) and (xc_mirror, yc_mirror)

        // calculate the distance from all 4 particles to node (x,y)
        distances[0] = sqrt((SQR(x-xc[j]) + SQR(y-yc[j])));
        distances[1] = sqrt((SQR(x-xc_mirror) + SQR(y-yc[j])));
        distances[2] = sqrt((SQR(x-xc[j]) + SQR(y-yc_mirror)));
        distances[3] = sqrt((SQR(x-xc_mirror) + SQR(y-yc_mirror)));

        // find the particle which has the minimum distance to the node (x,y)
        distance_min = distances[0];
        for(int k = 1; k < distances.size(); k++){
          if (distance_min > distances[k]){
            distance_min = distances[k];
          }
        }
        // calculate the level-set function which includes the new, mirrored particles
        phi_particles[j] =  R[j] - distance_min;
      }
    }

    else{
      for (int j = 0; j < np; ++j)
      {
        phi_particles[j] =  R[j] - sqrt( SUMD(SQR(x-xc[j]),
                                              SQR(y-yc[j]),
                                              SQR(z-zc[j])) );
      }
    }

    int particle_num = 0;
    for (int k = 1; k < np; k++)
    {
      if (phi_particles[particle_num] < phi_particles[k])
      {
        particle_num = k;
      }
    }

    return particle_num;
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
        return .5*XN()*MULTD(cos(2.*PI*x/(xmax()-xmin())*nx),
                             cos(2.*PI*y/(ymax()-ymin())*ny),
                             cos(2.*PI*z/(zmax()-zmin())*nz));
      }
    }
  }
};

class gamma_Aa_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return gamma_a();
  }
} gamma_Aa;


class gamma_Ba_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    return gamma_b();
  }
} gamma_Bb;

void initalize_particles(int &np, vector<double> &radius, DIM( vector<double> &xc, vector<double> &yc, vector<double> &zc))
{
  switch (geometry())
  {
    case 0:

      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(-0.45); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

      break;

    case 1:

      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(-1.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.50); xc.push_back(+1.50); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-0.90); yc.push_back(+0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(2.20); yc.push_back(-2.90); CODE3D( zc.push_back(+0.00) ); //place particle at boundary to see if the mirroring works

      np = radius.size();

      break;

    case 2:

      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-3.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-4.10); yc.push_back(+1.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.60); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+1.50); yc.push_back(+4.10); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.60); xc.push_back(+4.80); yc.push_back(-3.60); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+5.00); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+4.90); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+1.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-0.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-5.10); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-5.00); yc.push_back(+5.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-0.20); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.60); xc.push_back(+5.30); yc.push_back(+0.90); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(+3.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+5.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-5.00); yc.push_back(-1.80); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-3.50); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+1.20); yc.push_back(-5.40); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-5.30); yc.push_back(+2.70); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-0.20); yc.push_back(-1.00); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 3:

      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-3.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-4.10); yc.push_back(+1.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.40); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+1.50); yc.push_back(+4.10); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.40); xc.push_back(+4.80); yc.push_back(-3.60); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+5.00); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+4.90); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+1.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-0.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-5.10); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-5.00); yc.push_back(+5.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-0.20); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.40); xc.push_back(+5.30); yc.push_back(+0.90); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(+3.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+5.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-5.00); yc.push_back(-1.80); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-3.50); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+1.20); yc.push_back(-5.40); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-5.30); yc.push_back(+2.70); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-0.20); yc.push_back(-1.00); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 4:

      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-3.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-4.10); yc.push_back(+1.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.20); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+1.50); yc.push_back(+4.10); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.20); xc.push_back(+4.80); yc.push_back(-3.60); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+5.00); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+4.90); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+1.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-0.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-5.10); yc.push_back(-0.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-5.00); yc.push_back(+5.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-0.20); yc.push_back(+3.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.20); xc.push_back(+5.30); yc.push_back(+0.90); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(+3.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+5.20); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-5.00); yc.push_back(-1.80); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-3.50); yc.push_back(-5.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+1.20); yc.push_back(-5.40); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-5.30); yc.push_back(+2.70); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-0.20); yc.push_back(-1.00); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 5:

      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-3.75); yc.push_back(+2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-5.10); yc.push_back(-0.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.60); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+1.50); yc.push_back(+4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 6:

      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-3.75); yc.push_back(+2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-5.10); yc.push_back(-0.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.40); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+1.50); yc.push_back(+4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;


  case 7:

      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-3.75); yc.push_back(+2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-5.10); yc.push_back(-0.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      radius.push_back(+0.20); xc.push_back(-3.50); yc.push_back(+4.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+1.50); yc.push_back(+4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+4.50); yc.push_back(+5.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+0.00); yc.push_back(-4.15); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-4.80); yc.push_back(-4.00); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-0.75); yc.push_back(-2.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(-4.85); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 8:

      radius.push_back(+0.60); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-3.75); yc.push_back(+4.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(-5.10); yc.push_back(-4.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.60); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 9:

      radius.push_back(+0.40); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-3.75); yc.push_back(+4.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(-5.10); yc.push_back(-4.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.40); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

  case 10:

      radius.push_back(+0.20); xc.push_back(-2.50); yc.push_back(-1.45); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(-4.30); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-1.50); yc.push_back(+1.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-3.75); yc.push_back(+4.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+2.50); yc.push_back(+2.35); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(-5.10); yc.push_back(-4.50); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+4.75); yc.push_back(-0.75); CODE3D( zc.push_back(+0.00) );
      radius.push_back(+0.20); xc.push_back(+0.50); yc.push_back(+0.35); CODE3D( zc.push_back(+0.00) );

      np = radius.size();

    break;

    default: throw;
  }
}

double pairwise_potential(double r)
{
  if (r > 10.*pairwise_potential_width()) return 0;
  return pairwise_potential_mag()/(exp(r/pairwise_potential_width())-1.);
}

double pairwise_force(double r)
{
  if (r > 10.*pairwise_potential_width()) return 0;
  return -exp(r/pairwise_potential_width())*pairwise_potential_mag()/SQR(exp(r/pairwise_potential_width())-1.)/pairwise_potential_width();
}

class vx_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (velocity())
    {
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
      case 0: return 0;
      case 1: return 1;
      case 2: return 1;
      case 3: return sqrt(SQR(x)+SQR(y))*cos(atan2(y,x));
      default:
        throw;
    }
  }
} vy_cf;


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
  int np;
  vector<double> radius;
  vector<double> DIM(xc, yc, zc);

  initalize_particles(np, radius, DIM(xc, yc, zc));

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

  while (t < time_limit() && step < step_limit())
  {
    // create particle_level_set
    particles_level_set_cf_t particles_level_set_cf(np, radius, DIM(xc, yc, zc));

    // create get_particle_num (creates a field, that shows number of closest particle)
    particles_number_cf_t particles_number_cf(np, radius, DIM(xc, yc, zc));

    // ---------------------------------------------------------
    // create computational grid
    // ---------------------------------------------------------
    my_p4est_brick_t brick;
    p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
    p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin(), lmax(), &particles_level_set_cf, lip());
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

    Vec     gamma_d    [P4EST_DIM];
    double *gamma_d_ptr[P4EST_DIM];

    Vec     velo;
    double *velo_ptr;

    Vec     G;
    double *G_ptr;

    Vec     integrand;
    double *integrand_ptr;

    ierr = VecCreateGhostNodes(p4est, nodes, &mu_minus); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &mu_plus); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &particles_level_set); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &particles_number); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &kappa); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &gamma_eff); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &velo); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &G); CHKERRXX(ierr);
    ierr = VecDuplicate(mu_minus, &integrand); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr);
      ierr = VecDuplicate(normal[dim], &gamma_d[dim]); CHKERRXX(ierr);
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

    double rho = 1;

    // calculate mu_minus and velo
    if (use_scft())
    {
      my_p4est_scft_t scft(ngbd, ns());

      // set geometry
      scft.add_boundary(particles_level_set, MLS_INTERSECTION, gamma_Aa, gamma_Bb);

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
    ierr = VecGetArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      gamma_eff_ptr[n] = (0.5*(gamma_a()+gamma_b())*rho + (gamma_a()-gamma_b())*(mu_minus_ptr[n])/XN());
    }

    ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);

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
    ngbd->first_derivatives_central(gamma_eff, gamma_d);

    // calculate G
    ierr = VecGetArray(G,         &G_ptr);         CHKERRXX(ierr);
    ierr = VecGetArray(velo,      &velo_ptr);      CHKERRXX(ierr);
    ierr = VecGetArray(kappa,     &kappa_ptr);     CHKERRXX(ierr);
    ierr = VecGetArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecGetArray(normal [dim], &normal_ptr [dim]); CHKERRXX(ierr);
      ierr = VecGetArray(gamma_d[dim], &gamma_d_ptr[dim]); CHKERRXX(ierr);
    }

    double factor = 1./pow(scaling, P4EST_DIM-1.);

    foreach_node(n, nodes)
    {
      G_ptr[n] = SUMD(normal_ptr[0][n]*gamma_d_ptr[0][n],
                      normal_ptr[1][n]*gamma_d_ptr[1][n],
                      normal_ptr[2][n]*gamma_d_ptr[2][n])*factor
          + gamma_eff_ptr[n]*kappa_ptr[n]*factor
          + velo_ptr[n];
    }

    ierr = VecRestoreArray(G,         &G_ptr);         CHKERRXX(ierr);
    ierr = VecRestoreArray(velo,      &velo_ptr);      CHKERRXX(ierr);
    ierr = VecRestoreArray(kappa,     &kappa_ptr);     CHKERRXX(ierr);
    ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecRestoreArray(normal [dim], &normal_ptr [dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_d[dim], &gamma_d_ptr[dim]); CHKERRXX(ierr);
    }

    // calculate
    vector< vector<double> > g(P4EST_DIM, vector<double> (np,0));

    foreach_dimension(dim)
    {
      VecPointwiseMultGhost(integrand, G, normal[dim]);
      integrate_over_interface(np, g[dim], p4est, nodes, particles_level_set, particles_number, integrand);
    }

    ierr = VecDestroy(integrand); CHKERRXX(ierr);

    // additional terms due to particle-particle interactions

    // calculate q_x and q_y (additional velocity terms; take into account what happens when particles come close to each other)
    vector< vector<double> > q(P4EST_DIM, vector<double> (np,0));

    double energy_term = 0;

    for (int i = 0; i < np; i++)
    {
      for (int j = 0; j < np; j++)
      {
        if (i != j)
        {
          XCODE( double dx = xc[i]-xc[j] );
          YCODE( double dy = yc[i]-yc[j] );
          ZCODE( double dz = zc[i]-zc[j] );

          double dist =  sqrt(SUMD(SQR(dx), SQR(dy), SQR(dz))) - radius[i] - radius[j];

          double force = pairwise_force(dist);

          if (force != force) throw;

          XCODE( q[0][i] += (dx/dist)*force ); // u = a*exp(-r^2/(2*c^2)) => use derivative of u for multiplication
          YCODE( q[1][i] += (dy/dist)*force );
          ZCODE( q[2][i] += (dy/dist)*force );

          energy_term = energy_term + pairwise_potential(dist);
        }
      }
    }

    // add correction term to the energy (when particles come too close to each other, 'push' them away)
    energy += energy_term;

    for (int i = 0; i < np; ++i)
    {
      foreach_dimension(dim) g[dim][i] += q[dim][i];
    }

    // ---------------------------------------------------------
    // select velocity and move particles
    // ---------------------------------------------------------
    vector< vector<double> > v(P4EST_DIM, vector<double> (np,0));

    if (minimize())
    {
      for(int i = 0; i < np; i++)
      {
        foreach_dimension(dim) v[dim][i] = -g[dim][i];
      }
    }
    else
    {
      for(int i = 0; i < np; i++)
      {
        v[0][i] = vx_cf(xc[i], yc[i]);
        v[1][i] = vy_cf(xc[i], yc[i]);
      }
    }

    // find the maximum velocity (needed to calculate the timestep delta_t)
    double v_max = 0;
    for (int i = 0; i < np; ++i)
    {
      double v_abs = sqrt( SUMD(SQR(v[0][i]), SQR(v[1][i]), SQR(v[2][i])) );
      if (v_max < v_abs) v_max = v_abs;
    }
//    if(v_max < pow(10, -3)){
//      v_max = pow(10, -3);
//    }

    // find the diagonal length of the smallest quadrant (needed to calculate the timestep delta_t)
    double dxyz[P4EST_DIM]; // dimensions of the smallest quadrants
    double dxyz_min;        // minimum side length of the smallest quadrants
    double diag_min;        // diagonal length of the smallest quadrants
    get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);

    // calculate timestep delta_t, such that it's small enough to capture 'everything'
    double delta_t = diag_min*CFL()/v_max;



    // how to move particles if we chose a periodic domain
    if (px()==1 && py()==1 && pz()==1){
      for (int j = 0; j < np; ++j)
      {
        if(xc[j] > xmax()){
          xc[j] = xc[j] - (xmax()-xmin());
        }
        if (xc[j] < xmin()){
          xc[j] = xc[j] + (xmax()-xmin());
        }

        if(yc[j] < ymin()){
          yc[j] = yc[j] + (ymax()-ymin());
        }
        if (yc[j] > ymax()){
          yc[j] = yc[j] - (ymax()-ymin());
        }

        XCODE( xc[j] = xc[j] + v[0][j]*delta_t );
        YCODE( yc[j] = yc[j] + v[1][j]*delta_t );
        ZCODE( zc[j] = zc[j] + v[2][j]*delta_t );
      }

    }

    else{

      // move particles
      for (int j = 0; j < np; ++j)
      {
        XCODE( xc[j] = xc[j] + v[0][j]*delta_t );
        YCODE( yc[j] = yc[j] + v[1][j]*delta_t );
        ZCODE( zc[j] = zc[j] + v[2][j]*delta_t );
      }
    }

    // calculate the expected and effective changes in energy (compare step i with step i+1)
    // compute expected change in energy (for validation)
    double dE_dt_expected = 0;
    for (int i = 0; i < np; i++)
    {
      dE_dt_expected += SUMD(v[0][i]*g[0][i],
                             v[1][i]*g[1][i],
                             v[2][i]*g[2][i]);
    }
    double expected_change_in_energy  = dE_dt_expected*delta_t;
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
      ierr = VecGetArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(G, &G_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             5, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", particles_level_set_ptr,
                             VTK_POINT_DATA, "particles_number", particles_number_ptr,
                             VTK_POINT_DATA, "gamma_effective", gamma_eff_ptr,
                             VTK_POINT_DATA, "mu_minus", mu_minus_ptr,
                             VTK_POINT_DATA, "G", G_ptr);

      ierr = VecRestoreArray(particles_level_set, &particles_level_set_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(particles_number, &particles_number_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(gamma_eff, &gamma_eff_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(mu_minus, &mu_minus_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(G, &G_ptr); CHKERRXX(ierr);
    }

    // ---------------------------------------------------------
    // clean-up memory
    // ---------------------------------------------------------
    ierr = VecDestroy(particles_number);     CHKERRXX(ierr);
    ierr = VecDestroy(kappa);               CHKERRXX(ierr);
    ierr = VecDestroy(gamma_eff);           CHKERRXX(ierr);
    ierr = VecDestroy(velo);                CHKERRXX(ierr);
    ierr = VecDestroy(G);                   CHKERRXX(ierr);
    ierr = VecDestroy(integrand);           CHKERRXX(ierr);
    ierr = VecDestroy(particles_level_set); CHKERRXX(ierr);

    foreach_dimension(dim)
    {
      ierr = VecDestroy(gamma_d[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal [dim]); CHKERRXX(ierr);
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
