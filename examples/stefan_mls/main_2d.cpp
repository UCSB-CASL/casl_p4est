/*
 * Title: stefan_mls
 * Description:
 * Author: Elyce
 * Date Created: 11-27-2019
 */

//#define P4_TO_P8

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>

#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>

#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>

#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>

#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>


using namespace std;

const static std::string main_description =
     "In this example, we illustrate the procedure and methods used in the CASL p4est \n"
     "library to solve the Stefan problem. This example builds on fundamental tools \n"
     "of the CASL library such as interpolation, extrapolation of fields, adaptive grid \n"
     "refinement, reinitialization and advection of a level set function, and solution of \n"
     "a Poisson problem. \n"
     "Two examples are included in each 2D and 3D. One solves the Frank Sphere problem, a \n"
     "known analytical solution to the Stefan problem, and computes errors of the solution. \n"
     "The other models the melting of ice in a tub of water. Note: the ice melting example \n"
     "is more for illustrative purposes of a physical example, and has not been rigorously validated. \n"
     "\n"
     "For more information, please see the Solving_a_Stefan_problem_with_CASL_p4est.pdf file included \n"
     "in the example folder. \n "
    "Developer: Elyce Bayat (ebayat@ucsb.edu), December 2019.\n"
     "\n\n\n"
     "Note: There are 3 environment variables to set depending on the data you wish to save. They are: \n"
     "(1) OUT_DIR_VTK -- the directory where you wish to save the vtk files of your simulation \n"
     "(2) OUT_DIR_ERR -- the directory where you wish to save files with the frank sphere error information at each timestep \n"
     "(3) OUT_DIR_MEM -- the directory where you wish to save files with data regarding memory usage of the code (this is more for debugging)\n";
// ------------------------
// Define parameters:
// ------------------------
parameter_list_t pl;

// Examples:
DEFINE_PARAMETER(pl,int,example_,0,"Example number. 0 = Frank sphere, 1 = Ice melting. (default: 0)");

// Define the numeric label for each type of example to make implementation a bit more clear
enum{
    FRANK_SPHERE = 0,
    ICE_MELT = 1
};

// Save settings:
DEFINE_PARAMETER(pl,bool,save_vtk,1,"Save to vtk? (default: true)");
DEFINE_PARAMETER(pl,bool,save_frank_sphere_errors,0,"Save frank sphere errors? (default: false, but running frank sphere example will set it to true)");
DEFINE_PARAMETER(pl,int,save_every_iter,1,"Save every n (provided value) number of iterations (default:1)");


// Output settings:
DEFINE_PARAMETER(pl,bool,print_checkpoints,0,"Boolean value for whether or not you want to print simulation information -- mostly for debugging. (default: 0)");
// Solution methods:
DEFINE_PARAMETER(pl,int,method_,1,"Timestepping method. 1 = Backward Euler, 2= Crank-Nicholson. (default: 1)");

// Grid parameters
DEFINE_PARAMETER(pl,int,lmin,5,"Minimum level of refinement (default: 4)");
DEFINE_PARAMETER(pl,int,lmax,7,"Maximum level of refinement (default: 6)");
DEFINE_PARAMETER(pl,double,lip,1.5,"Lipschitz constant (default: 1.5)");
DEFINE_PARAMETER(pl,int,num_splits,0,"Number of splits -- used for convergence analysis (default: 0)");

// Problem geometry:
DEFINE_PARAMETER(pl,double,xmin,0.0,"Minimum x-coordinate of the grid (default: 0.0)");
DEFINE_PARAMETER(pl,double,xmax,1.0,"Maximum x-coordinate of the grid (default: 1.0)");
DEFINE_PARAMETER(pl,double,ymin,0.0,"Minimum y-coordinate of the grid (default: 0.0)");
DEFINE_PARAMETER(pl,double,ymax,1.0,"Maximum y-coordinate of the grid (default: 1.0)");
DEFINE_PARAMETER(pl,double,zmin,0.0,"Minimum z-coordinate of the grid (default: 0.0)");
DEFINE_PARAMETER(pl,double,zmax,1.0,"Maximum z-coordinate of the grid (default: 1.0)");

DEFINE_PARAMETER(pl,int,px,0,"Periodicity in x? Default: false");
DEFINE_PARAMETER(pl,int,py,0,"Periodicity in y? Default: false");
DEFINE_PARAMETER(pl,int,pz,0,"Periodicity in z? Default: false");

DEFINE_PARAMETER(pl,int,nx,0,"Number of trees in x-direction (default: 1)");
DEFINE_PARAMETER(pl,int,ny,0,"Number of trees in x-direction (default: 1)");
DEFINE_PARAMETER(pl,int,nz,0,"Number of trees in x-direction (default: 1)");

DEFINE_PARAMETER(pl,double,scaling,1.0,"The desired scaling between your physical problem and computational domain. ie.) physical_length_scale*scaling = computational_length_scale (default: 1.0 - aka, no scaling necessary)");

// Simulation time:
DEFINE_PARAMETER(pl,double,tstart,0.0,"Simulation start time (default: 0.0)");
DEFINE_PARAMETER(pl,double,tfinal,1.0,"Simulation end time (default: 1.0)");

// Solution stability:
DEFINE_PARAMETER(pl,double,cfl,0.5,"CFL number to enforce for timestepping (default: 0.25)");
DEFINE_PARAMETER(pl,double,v_interface_max_allowed,500.0,"Maximum interfacial velocity allowed -- will abort if interface value exceeds this (default: 500.0");
DEFINE_PARAMETER(pl,double,dt_max_allowed,1.0,"Maximum allowable timestep -- if timestep exceeds this, it will be set to this value instead (default 1.0)");

// For debugging:
DEFINE_PARAMETER(pl,bool,check_memory_usage,0,"Boolean on whether you want to check your memory usage or not (default is 0)");
// ---------------------------------------
// Auxiliary global variables:
// ---------------------------------------
// For timestepping:
double tn; // the current simulation time
bool keep_going = true;

// For frank sphere:
double s0;
double T_inf;

// For ice cube:
double r0;
double Twall;
double Tinterface;
double Tice_init;

// For keeping track of interfacial velocity:
double v_interface_max_norm;

// For surface tension: (used to apply some interfacial BC's in temperature)
double sigma;
// -----------------------------------------
// Auxiliary functions for initializing the problem:
// -----------------------------------------
void set_geometry(){
  switch(example_){
  case FRANK_SPHERE:
         CODE2D(xmin = -5.0;ymin = -5.0; zmin = 0.0;
                xmax = 5.0; ymax = 5.0; zmax = 0.0;)
         CODE3D(xmin = -3.0;ymin = -3.0; zmin = -3.0;
                xmax = 3.0; ymax = 3.0; zmax = 3.0;)

         nx = 1; ny = 1; CODE2D(nz = 0); CODE3D(nz = 1);
         px = 0; py = 0; pz = 0;

         CODE2D(s0 = 0.6286;) // 1.5621
         CODE3D(s0 = 0.5653;) // 2.0760
         CODE2D(T_inf = -0.2); // -0.5
         CODE3D(T_inf = -0.1);

         Tinterface = 0.0;
         Twall = T_inf;
         scaling = 1.;

      break;
  case ICE_MELT:
      CODE2D(xmin = -0.8; ymin = -0.8; zmin = 0.0;
             xmax = 0.8; ymax = 0.8; zmax = 0.0);
       CODE3D(xmin = -0.8; ymin = -0.8; zmin = -0.8;
       xmax = 0.8; ymax = 0.8; zmax = 0.8);
       nx = 1; ny = 1; CODE2D(nz = 0); CODE3D(nz = 1);
       px = 0; py = 0; pz = 0;

      // Scaling -> phyiscal_length_scale*scaling = computational_length_scale.
      // Scaling has units of 1/L where L is the physical length scale
      // Scaling = computational size/physical size
      double r_physical = 0.02; // 2 cm --> 0.02 m
      scaling = 1.5/0.15;
      r0 = r_physical*scaling; // Computational radius -- not physical size

      Tice_init = 263.0; // [K] initial temp of ice out of freezer
      Tinterface = 273.0; // [K] -- freezing temp of water
      Twall = 298.0; // [K] -- a bit under boiling temp of water
      break;

    }
}


// ---------------------------------------
// Time-stepping:
// ---------------------------------------
double dt; // Global variable which holds the current timestep value
void simulation_time_info(){
  switch(example_){
    case FRANK_SPHERE:
      tstart = 1.0;
      tfinal = 1.3;

      dt_max_allowed = 0.05;

      break;
    case ICE_MELT:
      tstart = 0.0;
      tfinal = 90.*60.; // approx 90 minutes
      dt_max_allowed = 20.0;
      break;
    }
  tn = tstart;
}
// ---------------------------------------
// Physical properties:
// ---------------------------------------
double alpha_s;
double alpha_l;
void set_diffusivities(){
  switch(example_){
    case FRANK_SPHERE:
      alpha_s = 1.0;
      alpha_l = 1.0;
      break;
    case ICE_MELT:
      alpha_s = (1.1820e-6); //ice - [m^2]/s
      alpha_l = (1.4547e-7); //water- [m^2]/s
      break;
    }
}

double k_s; // Thermal conductivity of the solid
double k_l; // Thermal conductivity of the liquid
double L; // Latent heat of fusion
double rho_l; // Liquid density

void set_conductivities(){
  switch(example_){
    case FRANK_SPHERE:
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      break;

    case ICE_MELT:
      k_s = 2.3;//2.2; // W/[m*K]
      k_l = 0.65; // W/[m*K]
      L = 334.e3; //J/kg
      rho_l = 1000.0; // kg/m^3
      sigma = 9.e-6;//30.e-3 // J/m^2 // surface tension of water
      break;

    }
}

// ---------------------------------------
// Other parameters:
// ---------------------------------------
DEFINE_PARAMETER(pl,bool,check_temperature_values,0,"Boolean for whether or not to check temperature values at various points during the simulation. Mainly for debugging. (default: 0)");
// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
// Auxiliary Frank sphere functions -- Functions necessary for evaluating the analytical solution of the Frank sphere problem, to validate results for example 1
// --------------------------------------------------------------------------------------------------------------
double s(double r, double t){
  //std::cout<<"Time being used to compute s is: " << t << "\n"<< std::endl;
  return r/sqrt(t);
}

// Exponential integral function : taken from existing code in examples/stefan/main_2d.cpp
double E1(double x)
{
  const double EULER=0.5772156649;
  const int    MAXIT=100;
  const double FPMIN=1.0e-20;

  int i,ii;
  double a,b,c,d,del,fact,h,psi,ans=0;

  int n   =1;
  int nm1 =0;

  if (x > 1.0)
  {        /* Lentz's algorithm */
    b=x+n;
    c=1.0/FPMIN;
    d=1.0/b;
    h=d;
    for (i=1;i<=MAXIT;i++)
    {
      a = -i*(nm1+i);
      b += 2.0;
      d=1.0/(a*d+b);    /* Denominators cannot be zero */
      c=b+a/c;
      del=c*d;
      h *= del;
      if (fabs(del-1.0) < EPS)
      {
        ans=h*exp(-x);
        return ans;
      }
    }
    //printf("Continued fraction failed in expint\n");
  }
  else
  {
    ans = (nm1!=0 ? 1.0/nm1 : -log(x)-EULER);    /* Set first term */
    fact=1.0;
    for (i=1;i<=MAXIT;i++)
    {
      fact *= -x/i;
      if (i != nm1) del = -fact/(i-nm1);
      else
      {
        psi = -EULER;  /* Compute psi(n) */
        for (ii=1;ii<=nm1;ii++) psi += 1.0/ii;
        del=fact*(-log(x)+psi);
      }
      ans += del;
      if (fabs(del) < fabs(ans)*EPS) return ans;
    }
    printf("series failed in expint\n");
    printf("x value used was : %0.5f",x);

  }

  return ans;
}


double F(double s){

  double z = SQR(s)/4.0;
  CODE2D(return E1(z);)
  CODE3D(return (1./s)*exp(-1.*z) - 0.5*sqrt(PI)*erfc(0.5*s));
}

double dF(double s){
  double z = SQR(s)/4.0;
  return -0.5*s*exp(z)/z;
}

double frank_sphere_solution_t(double s){

  if (s<s0) return 0;
  else      return T_inf*(1.0 - F(s)/F(s0));


}

// --------------------------------------------------------------------------------------------------------------
// Level Set Function:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
     switch (example_){
      case FRANK_SPHERE:
        return s0 - sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));

      case ICE_MELT:
         return r0 - sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));

      default: throw std::invalid_argument("You must choose an example type\n");
      }
  }
} level_set;


// --------------------------------------------------------------------------------------------------------------
// INTERFACIAL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_temp;
// Auxiliary function which initializes the interface BC type -- call this function before setting interface bc in the solver
void interface_bc(){
  switch(example_){
    case FRANK_SPHERE:
      interface_bc_type_temp = DIRICHLET;
      break;
    case ICE_MELT:
      interface_bc_type_temp = DIRICHLET;
      break;
    }
}

// A class that evaluates the interfacial BC value -- dependent on the current LSF geometry
class BC_interface_value: public CF_DIM{
private:
  // Have interpolation objects for case with surface tension included in boundary condition: can interpolate the curvature in a timestep to the interface points while applying the boundary condition
  my_p4est_interpolation_nodes_t kappa_interp;

public:
  BC_interface_value(my_p4est_node_neighbors_t *ngbd, vec_and_ptr_t kappa): kappa_interp(ngbd)
  {
    // Set the curvature and normal inputs to be interpolated when the BC object is constructed:
    kappa_interp.set_input(kappa.vec,linear);
  }
  double operator()(DIM(double x, double y,double z)) const
  {
    switch(example_){
      case FRANK_SPHERE: // Frank sphere case, no surface tension
        return Tinterface;
    case ICE_MELT: // Water case, has surface tension effects
        //return Tinterface;
        return Tinterface*(1. + sigma*kappa_interp(DIM(x,y,z))/L);
      }
  }
};

class BC_interface_coeff: public CF_DIM{
public:
  double operator()(DIM(double x, double y,double z)) const
  { switch(example_){
      case FRANK_SPHERE: return 1.0;
      case ICE_MELT: return 1.0;
      }
  }
}bc_interface_coeff;


// --------------------------------------------------------------------------------------------------------------
// Wall functions -- these evaluate to true or false depending on if the location is on the wall --  they just add coding simplicity
// --------------------------------------------------------------------------------------------------------------
struct XLOWER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return ((fabs(x - xmin) < EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS)); // front x wall, excluding the top and bottom corner points in y
  }
} xlower_wall;

struct XUPPER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return ((fabs(x - xmax) < EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS)); // back x wall, excluding the top and bottom corner points in y
  }
} xupper_wall;

struct YLOWER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (fabs(y - ymin) < EPS);
  }
} ylower_wall;

struct YUPPER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (fabs(y - ymax) < EPS);
  }
} yupper_wall;

struct ZLOWER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    CODE3D(return (fabs(z - zmin) < EPS));
  }
} zlower_wall;

struct ZUPPER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    CODE3D(return (fabs(z - zmax) < EPS));
  }
} zupper_wall;
// --------------------------------------------------------------------------------------------------------------
// Wall temperature boundary conditions
// --------------------------------------------------------------------------------------------------------------
class WALL_BC_TYPE_TEMP: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE: return DIRICHLET;
      case ICE_MELT: return DIRICHLET;
      default: break;
      }
  }
} wall_bc_type_temp;

class WALL_BC_VALUE_TEMP: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:{
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) CODE3D(|| zlower_wall(DIM(x,y,z)) || zupper_wall(DIM(x,y,z)))){
            if (level_set(DIM(x,y,z)) < EPS){
                double r = sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));
                double sval = s(r,tn+dt);
                return frank_sphere_solution_t(sval);
              }
            else{
                return Tinterface;
              }
          }
        break;
       }
    case ICE_MELT: {
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) CODE3D(|| zlower_wall(DIM(x,y,z)) || zupper_wall(DIM(x,y,z)))){
            if (level_set(DIM(x,y,z)) < EPS){
                return Twall;
              }
            else{
                return Tinterface;
              }
          } // end of "if on wall"
        break;
    }// end of ICE_MELT case

    default: break;
    } // end of switch case
  }
} wall_bc_value_temp;

// --------------------------------------------------------------------------------------------------------------
// TEMPERATURE INITIAL CONDITION/ANALYTICAL SOLUTION -- can be used for both
// --------------------------------------------------------------------------------------------------------------
class TEMP_ANALYTICAL_IN_TIME: public CF_DIM // This class is used as the initial condition, and for frank sphere is also used to check error
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    double m;
    double r;
    double sval;
    double Tsloped;
    switch(example_){
    case FRANK_SPHERE:{
        r = sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));
        sval = s(r,tn);
        return frank_sphere_solution_t(sval);
        }
    case ICE_MELT:{
        r = sqrt(SQR(x) + SQR(y) CODE3D(+ SQR(z)));
        m = (Twall - Tinterface)/(level_set(DIM(xmax,ymax,zmax)) - r0);
        if (level_set(DIM(x,y,z))<0){
            return Twall;
        }
        else{
            return Tice_init;
        }

    }

    }
  }
}temp_current_time;



// --------------------------------------------------------------------------------------------------------------
// Functions for checking the values of interest during the solution process
// --------------------------------------------------------------------------------------------------------------
// For checking temperature values: (NOTE: called when check_temperature_values flag is on)
void check_T_values(vec_and_ptr_t phi, vec_and_ptr_t T, p4est_nodes_t* nodes, p4est_t* p4est) {
  T.get_array();
  phi.get_array();

  double avg_T = 0.0;
  double max_T = 0.;
  double min_T = 1.e10;
  double min_mag_T = 1.e10;

  int pts_avg = 0;

  bool in_domain;

  // Loop over each node, check if node is in the subdomain we are considering. If so, compute average,max, and min values for the domain
  foreach_local_node(n,nodes){
    // Check if the node is in the domain we are checking:
    (phi.ptr[n] < EPS) ? in_domain = true : in_domain = false;

    // Compute required values:
    if (in_domain){
      avg_T+=T.ptr[n];
      pts_avg ++;

      max_T = max(max_T,T.ptr[n]);
      min_T = min(min_T,T.ptr[n]);

      min_mag_T = min(min_mag_T,fabs(T.ptr[n]));
      }
  }

  // Use MPI_Allreduce to get the values on a global scale, not just on one process:
  double global_avg_T = 0.0;
  double global_max_T = 0.;
  double global_min_T = 1.e10;
  double global_min_mag_T = 1.e10;

  int global_pts_avg = 0;

  MPI_Allreduce(&avg_T,&global_avg_T,1,MPI_DOUBLE,MPI_SUM,p4est->mpicomm);
  MPI_Allreduce(&pts_avg,&global_pts_avg,1,MPI_INT,MPI_SUM,p4est->mpicomm);

  global_avg_T/=global_pts_avg;

  MPI_Allreduce(&max_T,&global_max_T,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
  MPI_Allreduce(&min_T,&global_min_T,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);
  MPI_Allreduce(&min_mag_T,&global_min_mag_T,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);

  PetscPrintf(p4est->mpicomm,"\n");
  PetscPrintf(p4est->mpicomm,"Average temperature: %0.2f \n",global_avg_T);
  PetscPrintf(p4est->mpicomm,"Maximum temperature: %0.2f \n",global_max_T);
  PetscPrintf(p4est->mpicomm,"Minimum temperature: %0.2f \n",global_min_T);
  PetscPrintf(p4est->mpicomm,"Minimum temperature magnitude: %0.2f \n",global_min_mag_T);

//  if(global_max_T>400. || global_min_T<100.0) SC_ABORT("temp values are geting unreasonable");
  T.restore_array();
  phi.restore_array();
}

// For checking temperature derivative values: (NOTE: not called in main currently)
void check_T_d_values(vec_and_ptr_t phi, vec_and_ptr_dim_t dT, p4est_nodes_t* nodes, p4est_t* p4est, bool get_location){
  dT.get_array();
  phi.get_array();

  foreach_dimension(d){
    double avg_dT = 0.0;
    double max_dT = 0.;
    double min_dT = 1.e10;
    double min_mag_dT = 1.e10;

    int pts_avg = 0;

    foreach_local_node(n,nodes){
      avg_dT+=dT.ptr[d][n];
      pts_avg ++;

      max_dT = max(max_dT,dT.ptr[d][n]);
      min_dT = min(min_dT,dT.ptr[d][n]);

      min_mag_dT = min(min_mag_dT,fabs(dT.ptr[d][n]));
    }
    double global_avg_dT = 0.0;
    double global_max_dT = 0.;
    double global_min_dT = 1.e10;
    double global_min_mag_dT = 1.e10;

    int global_pts_avg = 0;

    MPI_Allreduce(&avg_dT,&global_avg_dT,1,MPI_DOUBLE,MPI_SUM,p4est->mpicomm);
    MPI_Allreduce(&pts_avg,&global_pts_avg,1,MPI_INT,MPI_SUM,p4est->mpicomm);

    global_avg_dT/=global_pts_avg;

    MPI_Allreduce(&max_dT,&global_max_dT,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
    MPI_Allreduce(&min_dT,&global_min_dT,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);
    MPI_Allreduce(&min_mag_dT,&global_min_mag_dT,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);

    PetscPrintf(p4est->mpicomm,"\n");
    if(d==0) PetscPrintf(p4est->mpicomm,"In x direction: \n{\n");
    else PetscPrintf(p4est->mpicomm,"In y direction: \n{\n");

    PetscPrintf(p4est->mpicomm,"Average: %0.2f \n",global_avg_dT);
    PetscPrintf(p4est->mpicomm,"Maximum: %0.2f \n",global_max_dT);
    PetscPrintf(p4est->mpicomm,"Minimum: %0.2f \n",global_min_dT);
    PetscPrintf(p4est->mpicomm,"Minimum magnitude: %0.2f \n",global_min_mag_dT);

    // See what the location is of the min and max values:
    if (get_location){
        double xyz_min[P4EST_DIM];
        double xyz_max[P4EST_DIM];
        double rmin;
        double rmax;
        foreach_local_node(n,nodes){
          if ((dT.ptr[d][n] <= global_max_dT + EPS) && (dT.ptr[d][n] >= global_max_dT - EPS)){
              node_xyz_fr_n(n,p4est,nodes,xyz_max);
              rmax = sqrt(SQR(xyz_max[0]) + SQR(xyz_max[1]));
              PetscPrintf(p4est->mpicomm,"Maximum value occurs at (%0.2f, %0.2f), on r = %0.2f \n",xyz_max[0],xyz_max[1],rmax);

            }

          if ((dT.ptr[d][n] <= global_min_dT + EPS) && (dT.ptr[d][n] >= global_min_dT - EPS)){
              node_xyz_fr_n(n,p4est,nodes,xyz_min);
              rmin = sqrt(SQR(xyz_min[0]) + SQR(xyz_min[1]));

              PetscPrintf(p4est->mpicomm,"Minimum value occurs at (%0.2f, %0.2f), on r = %0.2f \n",xyz_min[0],xyz_min[1],rmin);

            }
        }}
}
  dT.restore_array();
  phi.restore_array();
}

// For checking interfacial velocity values: (NOTE: not called in main currently)
void check_vel_values(vec_and_ptr_t phi, vec_and_ptr_dim_t vel, p4est_nodes_t* nodes, p4est_t* p4est, bool get_location,double dxyz_close_to_interface){
  vel.get_array();
  phi.get_array();

  // Check directional info:
  foreach_dimension(d){
    double avg_vel = 0.0;
    double max_vel = 0.;
    double min_vel = 1.e10;
    double min_mag_vel = 1.e10;

    int pts_avg = 0;

    foreach_local_node(n,nodes){
      if(fabs(phi.ptr[n]) < dxyz_close_to_interface){
        avg_vel+=vel.ptr[d][n];
        pts_avg ++;

        max_vel = max(max_vel,vel.ptr[d][n]);
        min_vel = min(min_vel,vel.ptr[d][n]);

        min_mag_vel = min(min_mag_vel,fabs(vel.ptr[d][n]));
      }
    }
    double global_avg_vel = 0.0;
    double global_max_vel = 0.;
    double global_min_vel = 1.e10;
    double global_min_mag_vel = 1.e10;

    int global_pts_avg = 0;

    MPI_Allreduce(&avg_vel,&global_avg_vel,1,MPI_DOUBLE,MPI_SUM,p4est->mpicomm);
    MPI_Allreduce(&pts_avg,&global_pts_avg,1,MPI_INT,MPI_SUM,p4est->mpicomm);

    global_avg_vel/=global_pts_avg;

    MPI_Allreduce(&max_vel,&global_max_vel,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
    MPI_Allreduce(&min_vel,&global_min_vel,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);
    MPI_Allreduce(&min_mag_vel,&global_min_mag_vel,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);

    PetscPrintf(p4est->mpicomm,"\n");
    if(d==0) PetscPrintf(p4est->mpicomm,"In x direction: \n{\n");
    else PetscPrintf(p4est->mpicomm,"In y direction: \n{\n");

    PetscPrintf(p4est->mpicomm,"Average: %0.2e \n",global_avg_vel);
    PetscPrintf(p4est->mpicomm,"Maximum: %0.2e \n",global_max_vel);
    PetscPrintf(p4est->mpicomm,"Minimum: %0.2e \n",global_min_vel);
    PetscPrintf(p4est->mpicomm,"Minimum magnitude: %0.2e \n",global_min_mag_vel);

    // See what the location is of the min and max values:
    if (get_location){
        double xyz_min[P4EST_DIM];
        double xyz_max[P4EST_DIM];
        double rmin;
        double rmax;
        foreach_local_node(n,nodes){
          if ((vel.ptr[d][n] <= global_max_vel + EPS) && (vel.ptr[d][n] >= global_max_vel - EPS)){
              node_xyz_fr_n(n,p4est,nodes,xyz_max);
              rmax = sqrt(SQR(xyz_max[0]) + SQR(xyz_max[1]));
              PetscPrintf(p4est->mpicomm,"Maximum value occurs at (%0.2f, %0.2f), on r = %0.2f \n",xyz_max[0],xyz_max[1],rmax);

            }

          if ((vel.ptr[d][n] <= global_min_vel + EPS) && (vel.ptr[d][n] >= global_min_vel - EPS)){
              node_xyz_fr_n(n,p4est,nodes,xyz_min);
              rmin = sqrt(SQR(xyz_min[0]) + SQR(xyz_min[1]));

              PetscPrintf(p4est->mpicomm,"Minimum value occurs at (%0.2f, %0.2f), on r = %0.2f \n",xyz_min[0],xyz_min[1],rmin);

            }
        }}



}

  // Now check the norms:
  double avg_vel = 0.0;
  double max_vel = 0.;
  double min_vel = 1.e10;

  int pts_avg = 0;

  foreach_local_node(n,nodes){
    if(fabs(phi.ptr[n])<dxyz_close_to_interface){
      double local_vel = sqrt(SQR(vel.ptr[0][n]) + SQR(vel.ptr[1][n]));
      avg_vel+=local_vel;
      pts_avg ++;

      max_vel = max(max_vel,local_vel);
      min_vel = min(min_vel,local_vel);
    }
  }

  double global_avg_vel = 0.0;
  double global_max_vel = 0.;
  double global_min_vel = 1.e10;

  int global_pts_avg = 0;

  MPI_Allreduce(&avg_vel,&global_avg_vel,1,MPI_DOUBLE,MPI_SUM,p4est->mpicomm);
  MPI_Allreduce(&pts_avg,&global_pts_avg,1,MPI_INT,MPI_SUM,p4est->mpicomm);

  global_avg_vel/=global_pts_avg;

  MPI_Allreduce(&max_vel,&global_max_vel,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
  MPI_Allreduce(&min_vel,&global_min_vel,1,MPI_DOUBLE,MPI_MIN,p4est->mpicomm);

  PetscPrintf(p4est->mpicomm,"2 norm info: \n{\n");

  PetscPrintf(p4est->mpicomm,"Average: %0.2e \n",global_avg_vel);
  PetscPrintf(p4est->mpicomm,"Maximum: %0.2e \n",global_max_vel);
  PetscPrintf(p4est->mpicomm,"Minimum: %0.2e \n",global_min_vel);

  vel.restore_array();
  phi.restore_array();
}

// --------------------------------------------------------------------------------------------------------------
// Function for checking the error in the Frank sphere solution:
// --------------------------------------------------------------------------------------------------------------
void check_frank_sphere_error(vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t phi, vec_and_ptr_dim_t v_interface, p4est_t *p4est, p4est_nodes_t *nodes, double dxyz_close_to_interface,char *name, FILE *fich,int tstep){
  PetscErrorCode ierr;

  T_l.get_array(); T_s.get_array();
  phi.get_array(); v_interface.get_array();

  double T_l_error = 0.0;
  double T_s_error = 0.0;
  double phi_error = 0.0;
  double v_int_error = 0.0;


  double Linf_Tl = 0.0;
  double Linf_Ts = 0.0;
  double Linf_phi = 0.0;
  double Linf_v_int = 0.0;

  double r;
  double s;
  double vel;

  // NOTE: We have just solved for time (n+1), so we compare computed solution to the analytical solution at time = (tn + dt)
  // Calculate interfacial velocity error with previously computed max_v_norm: (No need to Allreduce, it has been done already)
  double v_exact = s0/(2.0*sqrt(tn+dt));
  v_int_error = 0.0;

  //PetscPrintf(mpi.comm(),"Exact solution of velocity is: %0.2f",v_exact);
  // Now loop through nodes to compare errors between LSF and Temperature profiles:
  double xyz[P4EST_DIM];
  foreach_local_node(n,nodes){
    node_xyz_fr_n(n,p4est,nodes,xyz);

    r = sqrt(SQR(xyz[0]) + SQR(xyz[1]) CODE3D(+ SQR(xyz[2])));
    s = r/sqrt(tn+dt);

    double phi_exact = s0*sqrt(tn+dt) - r;
    double T_exact = frank_sphere_solution_t(s);


    // Error on phi and v_int:
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){
      // Errors on phi:
      phi_error = fabs(phi.ptr[n] - phi_exact);
      Linf_phi = max(Linf_phi,phi_error); // CHECK THIS -- NOT ENTIRELY SURE THIS IS CORRECT

      // Errors on v_int:
      vel = sqrt(SQR(v_interface.ptr[0][n])+ SQR(v_interface.ptr[1][n]) CODE3D(+ SQR(v_interface.ptr[2][n])));
      v_int_error = fabs(vel - v_exact);
      Linf_v_int = max(Linf_v_int,v_int_error);
      }

    // Check error in the negative subdomain (T_liquid) (Domain = Omega_minus)
    if(phi.ptr[n]<EPS){
        T_l_error  = fabs(T_l.ptr[n] - T_exact);
        Linf_Tl = max(Linf_Tl,T_l_error);
      }
    if (phi.ptr[n]>EPS){
        T_s_error  = fabs(T_s.ptr[n] - T_exact);
        Linf_Ts = max(Linf_Ts,T_s_error);

      }
  }

  double global_Linf_errors[] = {0.0, 0.0, 0.0,0.0,0.0};
  double local_Linf_errors[] = {Linf_phi,Linf_Tl,Linf_Ts,Linf_v_int};


  // Now get the global maximum errors:
  MPI_Barrier(p4est->mpicomm);
  int mpiret = MPI_Allreduce(local_Linf_errors,global_Linf_errors,4,MPI_DOUBLE,MPI_MAX,p4est->mpicomm); SC_CHECK_MPI(mpiret);


  // Print Errors to application output:
  PetscPrintf(p4est->mpicomm,"\n----------------\n Errors on frank sphere: \n --------------- \n");
  PetscPrintf(p4est->mpicomm,"dxyz close to interface: %0.3e \n",dxyz_close_to_interface);

  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"Number of grid points used: %d \n \n", num_nodes);


  PetscPrintf(p4est->mpicomm," Linf on phi: %0.3e \n Linf on T_l: %0.3e \n Linf on T_s: %0.3e \n Linf on v_int: %0.3e \n", global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3]);

  // Print errors to file:
  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  PetscFPrintf(p4est->mpicomm,fich,"%e %e %d %e %e %e %e %d %e \n",tn + dt,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],num_nodes,dxyz_close_to_interface);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);

  T_l.restore_array();
  T_s.restore_array();
  phi.restore_array();
  v_interface.restore_array();
}

// --------------------------------------------------------------------------------------------------------------
// Auxiliary functions for solving the problem:
// --------------------------------------------------------------------------------------------------------------
// For ice melt case, check if the ice is all melted yet
bool check_ice_melted(vec_and_ptr_t phi, double time, p4est_nodes_t* nodes,p4est_t* p4est){
  int still_solid_present = 0;
  int global_still_solid_present;// = false;

  // Check if LSF is positive anywhere to see if solid is still present:
  phi.get_array();
  foreach_local_node(n,nodes){
    if (phi.ptr[n]>EPS) still_solid_present = 1;
  }
  phi.restore_array();

  //printf("Process %d has still solid present = %s \n",p4est->mpirank,still_solid_present ? "true": "false");

  // Get the global outcome:
  int mpi_check;

  mpi_check = MPI_Allreduce(&still_solid_present,&global_still_solid_present,1,MPI_INT,MPI_LOR,p4est->mpicomm);


  SC_CHECK_MPI(mpi_check);
  MPI_Barrier(p4est->mpicomm);
  if (!global_still_solid_present){ // If no more solid, then ice has melted
      PetscPrintf(p4est->mpicomm,"\n \n Ice has entirely melted as of t = %0.3e seconds, or %0.2f minutes \n \n ",time,time/60.);
    }
return global_still_solid_present;
}

void setup_rhs(vec_and_ptr_t T_l, vec_and_ptr_t T_s,vec_and_ptr_t rhs_Tl, vec_and_ptr_t rhs_Ts, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd){

  // In building RHS, we have two options: (1) Backward Euler 1st order approx, (2) Crank Nicholson 2nd order approx
  // (1) dT/dt = (T(n+1) - T(n)/dt) --> which is a backward euler 1st order approximation (since the RHS is discretized spatially at T(n+1))
  // (2) dT/dt = alpha*laplace(T) ~ (T(n+1) - T(n)/dt) = (1/2)*(laplace(T(n)) + laplace(T(n+1)) )  ,
  //                              in which case we need the second derivatives of the temperature field at time n
  // Where alpha is the thermal diffusivity of the phase



    // Get derivatives of temperature fields if we are using Crank Nicholson:
    vec_and_ptr_dim_t T_l_dd;
    vec_and_ptr_dim_t T_s_dd;
    if(method_ ==2){
        T_s_dd.create(p4est,nodes);
        ngbd->second_derivatives_central(T_s.vec,T_s_dd.vec);
        T_s_dd.get_array();

        T_l_dd.create(p4est,nodes);
        ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec);
        T_l_dd.get_array();
      }

  // Get Ts arrays:
  T_s.get_array();
  rhs_Ts.get_array();

  // Get Tl arrays:
  rhs_Tl.get_array();
  T_l.get_array();


  foreach_node(n,nodes){
    // First, assemble system for Ts depending on case:
    if(method_ == 2){ // Crank Nicholson
        rhs_Ts.ptr[n] = 2.*T_s.ptr[n]/dt + alpha_s*(T_s_dd.ptr[0][n] + T_s_dd.ptr[1][n] CODE3D(+ T_s_dd.ptr[2][n]));
        rhs_Tl.ptr[n] = 2.*T_l.ptr[n]/dt + alpha_l*(T_l_dd.ptr[0][n] + T_l_dd.ptr[1][n] CODE3D(+ T_l_dd.ptr[2][n]));
      }
    else{ // Backward Euler
        rhs_Ts.ptr[n] = T_s.ptr[n]/dt;
        rhs_Tl.ptr[n] = T_l.ptr[n]/dt;
      }
  }// end of loop over nodes

  // Restore Ts arrays:
  T_s.restore_array();
  rhs_Ts.restore_array();

  // Restore Tl arrays:
  rhs_Tl.restore_array();
  T_l.restore_array();

  if(method_ ==2){
      T_s_dd.restore_array();
      T_s_dd.destroy();

      T_l_dd.restore_array();
      T_l_dd.destroy();
    }
}

void interpolate_values_onto_new_grid(vec_and_ptr_t T_l, vec_and_ptr_t T_l_new,
                                      vec_and_ptr_t T_s, vec_and_ptr_t T_s_new,
                                      vec_and_ptr_dim_t v_interface, vec_and_ptr_dim_t v_interface_new,
                                      p4est_nodes_t *nodes_new_grid, p4est_t *p4est_new,
                                      my_p4est_node_neighbors_t *ngbd_old_grid,interpolation_method interp_method){

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_old_grid);

  // Create an array of the vectors for faster interpolation -- interpolate all fields at once:
  unsigned int num_fields =2 + P4EST_DIM;
  Vec all_fields_old[num_fields];
  Vec all_fields_new[num_fields];

  // Set existing vectors as elements of the array of vectors:
  all_fields_old[0] = T_l.vec;
  all_fields_old[1] = T_s.vec;
  all_fields_old[2] = v_interface.vec[0];
  all_fields_old[3] = v_interface.vec[1];
  CODE3D(all_fields_old[4] = v_interface.vec[2]);


  all_fields_new[0] = T_l_new.vec;
  all_fields_new[1] = T_s_new.vec;
  all_fields_new[2] = v_interface_new.vec[0];
  all_fields_new[3] = v_interface_new.vec[1];
  CODE3D(all_fields_new[4] = v_interface_new.vec[2]);

  // Set the fields as input to the interpolator:
  interp_nodes.set_input(all_fields_old,interp_method,num_fields);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes_new_grid){
    node_xyz_fr_n(n,p4est_new,nodes_new_grid,xyz);
    interp_nodes.add_point(n,xyz);
  }

  // Interpolate the fields all at once:
  interp_nodes.interpolate(all_fields_new);
} // end of interpolate_values_onto_new_grid


void compute_interfacial_velocity(vec_and_ptr_dim_t T_l_d, vec_and_ptr_dim_t T_s_d, vec_and_ptr_dim_t jump, vec_and_ptr_dim_t v_interface, vec_and_ptr_t phi, my_p4est_node_neighbors_t *ngbd){

  // Get arrays:
  jump.get_array();
  T_l_d.get_array();
  T_s_d.get_array();

  // First, compute jump in the layer nodes:
  for(size_t i=0; i<ngbd->get_layer_size();i++){
    p4est_locidx_t n = ngbd->get_layer_node(i);

    foreach_dimension(d){
        jump.ptr[d][n] = (k_s*T_s_d.ptr[d][n] -k_l*T_l_d.ptr[d][n])/(L*rho_l);
    }
   }

  // Begin updating the ghost values of the layer nodes:
  foreach_dimension(d){
    VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
  }

  // Compute the jump in the local nodes:
  for(size_t i = 0; i<ngbd->get_local_size();i++){
      p4est_locidx_t n = ngbd->get_local_node(i);

      foreach_dimension(d){
          jump.ptr[d][n] = (k_s*T_s_d.ptr[d][n] -k_l*T_l_d.ptr[d][n])/(L*rho_l);
      }
    }

  // Finish updating the ghost values of the layer nodes:
  foreach_dimension(d){
    VecGhostUpdateEnd(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
  }

  // Restore arrays:
  jump.restore_array();
  T_l_d.restore_array();
  T_s_d.restore_array();

  my_p4est_level_set_t ls(ngbd);

  // Extend the interfacial velocity to the whole domain for advection of the LSF:
  foreach_dimension(d){
     ls.extend_from_interface_to_whole_domain_TVD(phi.vec,jump.vec[d],v_interface.vec[d],20);
  }

}

void compute_timestep(vec_and_ptr_dim_t v_interface, vec_and_ptr_t phi, double dxyz_close_to_interface, double dxyz_smallest[P4EST_DIM],p4est_nodes_t *nodes, p4est_t *p4est){

  // Check the values of v_interface locally:
  v_interface.get_array();
  phi.get_array();
  double max_v_norm = 0.0;
  foreach_local_node(n,nodes){
    if (fabs(phi.ptr[n]) < dxyz_close_to_interface){
        max_v_norm = max(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n]) CODE3D(+ SQR(v_interface.ptr[2][n]))));
      }
  }
  v_interface.restore_array();
  phi.restore_array();

  // Get the maximum v norm across all the processors:
  MPI_Barrier(p4est->mpicomm);
  double global_max_vnorm = 0.0;
  int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
  SC_CHECK_MPI(mpi_ret);
  PetscPrintf(p4est->mpicomm,"\n Computed interfacial velocity and timestep: \n");
  PetscPrintf(p4est->mpicomm,"\n -- Max v norm: %0.2g \n", global_max_vnorm);
  if(scaling>(1.0 + EPS) || scaling<(1.0 - EPS))PetscPrintf(p4est->mpicomm,"\n -- Physically, this corresponds to: %0.2g [m/s] \n \n",global_max_vnorm/scaling);

  // Compute new timestep:
  dt = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/min(global_max_vnorm,1.0);
  PetscPrintf(p4est->mpicomm,"-- Computed timestep: %0.3e \n",dt);

  dt = min(dt,dt_max_allowed);

  // Report computed timestep and minimum grid size:
  PetscPrintf(p4est->mpicomm," -- Used timestep: %0.3e \n",dt);
  if(print_checkpoints)PetscPrintf(p4est->mpicomm,"dxyz close to interface : %0.3e \n } \n \n  ",dxyz_close_to_interface);

  v_interface_max_norm = global_max_vnorm;
}

void compute_curvature(vec_and_ptr_t phi,vec_and_ptr_dim_t normal,vec_and_ptr_t curvature, my_p4est_node_neighbors_t *ngbd,my_p4est_level_set_t LS){

  vec_and_ptr_t curvature_tmp;
  curvature_tmp.create(curvature.vec);

  // Get arrays needed:
  curvature_tmp.get_array();
  normal.get_array();
  // Define the qnnn object to help compute the derivatives of the normal:
  quad_neighbor_nodes_of_node_t qnnn;

  // Compute curvature on layer nodes:
  for(size_t i = 0; i<ngbd->get_layer_size(); i++){
      p4est_locidx_t n = ngbd->get_layer_node(i);
      ngbd->get_neighbors(n,qnnn);
      curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]) CODE3D(+ qnnn.dz_central(normal.ptr[2]));
    }

  // Begin ghost update:
  VecGhostUpdateBegin(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);

  // Compute curvature on local nodes:
  for(size_t i = 0; i<ngbd->get_local_size(); i++){
      p4est_locidx_t n = ngbd->get_local_node(i);
      ngbd->get_neighbors(n,qnnn);
      curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]) CODE3D(+ qnnn.dz_central(normal.ptr[2]));
    }

  // End ghost update:
  VecGhostUpdateEnd(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);

  // Restore arrays needed:
  curvature_tmp.restore_array();
  normal.restore_array();

  // Now go ahead and extend the curvature values to the whole domain -- Will be used to apply the pointwise Dirichlet condition, dependent on curvature
  LS.extend_from_interface_to_whole_domain_TVD(phi.vec,curvature_tmp.vec,curvature.vec,20);

  // Destroy temp now:
  curvature_tmp.destroy();

}

// --------------------------------------------------------------------------------------------------------------
// Function for saving to VTK for visualization in paraview:
// --------------------------------------------------------------------------------------------------------------
void save_stefan_fields(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi,vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int, vec_and_ptr_t T_error, vec_and_ptr_t T_ana, char* filename ){
  // Things we want to save:
  /*
   * LSF
   * LSF2 for ex 2
   * Tl
   * Ts
   * v_interface
   * */

    // First, need to scale the fields appropriately:

    // Scale velocities:
    foreach_dimension(d){
      VecScaleGhost(v_int.vec[d],1./scaling);
    }

    // Get arrays:
    phi.get_array();
    Tl.get_array(); Ts.get_array();
    v_int.get_array();
    if(example_ == FRANK_SPHERE){
        T_error.get_array();
        T_ana.get_array();
    }


    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    if(example_ == FRANK_SPHERE){
        point_names = {"phi","Tl","Ts","v_int_x","v_int_y","T_error" ,"T_analytical",ZCODE("v_int_z")};
        point_data = {phi.ptr,Tl.ptr,Ts.ptr,v_int.ptr[0],v_int.ptr[1],T_error.ptr,T_ana.ptr,ZCODE(v_int.ptr[2])};
    }
    else{
        point_names = {"phi","Tl","Ts","v_int_x","v_int_y",ZCODE("v_int_z")};
        point_data = {phi.ptr,Tl.ptr,Ts.ptr,v_int.ptr[0],v_int.ptr[1],ZCODE(v_int.ptr[2])};
    }

    std::vector<std::string> cell_names;
    std::vector<double*> cell_data;

    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


    // Restore arrays:

    phi.restore_array();
    Tl.restore_array(); Ts.restore_array();
    v_int.restore_array();
    if(example_ == FRANK_SPHERE){
        T_error.restore_array();
        T_ana.restore_array();
    }


    // Scale things back:
    foreach_dimension(d){
      VecScaleGhost(v_int.vec[d],scaling);
    }
}

// --------------------------------------------------------------------------------------------------------------
// BEGIN MAIN OPERATION:
// --------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;

  cmdParser cmd;

  // Parse inputs:
  pl.initialize_parser(cmd);
  if (cmd.parse(argc, argv, main_description)) return 0;

  cmd.parse(argc,argv);

  pl.get_all(cmd);


  // stopwatch
  parStopWatch w;
  w.start("Running example: multialloy_with_fluids");

  // Loop through all grid resolutions (if we are studying multiple):
  for(int grid_res_iter=0; grid_res_iter<=num_splits; grid_res_iter++){
      PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

   // -----------------------------------------------
    // Set up grid structure and partition:
    // -----------------------------------------------
    // p4est variables
    p4est_t*              p4est;
    p4est_nodes_t*        nodes;
    p4est_ghost_t*        ghost;
    p4est_connectivity_t* conn;
    my_p4est_brick_t      brick;

    p4est_t               *p4est_np1;
    p4est_nodes_t         *nodes_np1;
    p4est_ghost_t         *ghost_np1;

    // Initialize geometry:
    set_geometry();

    const int n_xyz[]      = { nx,  ny,  nz};
    const double xyz_min[] = {xmin, ymin, zmin};
    const double xyz_max[] = {xmax,  ymax,  zmax};
    const int periodic[]   = { px,  py,  pz};

    // Initialize simulation time info:
    simulation_time_info();

    // -----------------------------------------------
    // Set properties for the Poisson node problem:
    // -----------------------------------------------
    int cube_refinement = 1; // We can set this to 1 since we are only considering Dirichlet BC's, and this only comes into play for Neumann/Robin
    interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

    // Initialize physical properties:
    set_diffusivities();
    set_conductivities();

    // Initialize interface BC:
    interface_bc();

    // -----------------------------------------------
    // Scale the problem appropriately:
    // -----------------------------------------------
    rho_l/=(scaling*scaling*scaling);
    k_s/=scaling;
    k_l/=scaling;

    if(example_==ICE_MELT){
        sigma/=scaling;
    }

    alpha_l*=(scaling*scaling);
    alpha_s*=(scaling*scaling);

    if(print_checkpoints){
        PetscPrintf(mpi.comm(),"Scaled values are: "
                               "ks = %0.3g \n"
                               "kl = %0.3g \n"
                               "alpha_s = %0.3g \n"
                               "alpha_l = %0.3g \n"
                               "L = %0.3g \n"
                               "rho_l = %0.3g \n",k_s,k_l,alpha_s,alpha_l,L,rho_l);
    }


    // -----------------------------------------------
    // Create the grid:
    // -----------------------------------------------
    // Create the brick and connectivity:
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

    // Create the forest
    p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

    // Refine based on distance to a level-set
    splitting_criteria_cf_t sp(lmin + grid_res_iter, lmax + grid_res_iter, &level_set,lip);
    p4est->user_pointer = &sp; // Save the user pointer to the forest splitting criteria --> so we can use it for refinement

    // Refine the forest according to the splitting criteria:
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    // Partition the forest, do not allow for coarsening
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    // create ghost layer
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL); // same

    // create node structure
    nodes = my_p4est_nodes_new(p4est, ghost); //same

    // Create hierarchy
    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);

    // Get neighbors
    my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
    ngbd->init_neighbors();

    // -----------------------------------------------
    // Initialize the Level Set function(s):
    // -----------------------------------------------
    // Initialize a vector to hold the LSF:
    vec_and_ptr_t phi;
    phi.create(p4est,nodes);
    sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);

    // Initialize a vector to hold the LSF for solid domain: -- This will be assigned within the loop as the negative of phi
    vec_and_ptr_t phi_solid;

    // Vectors to hold 2nd derivatives of LSF's
    vec_and_ptr_dim_t phi_dd;
    vec_and_ptr_dim_t phi_solid_dd;

    // -----------------------------------------------
    // Initialize the interfacial velocity field (used for Stefan problem)
    // -----------------------------------------------
    vec_and_ptr_dim_t v_interface(p4est,nodes); // Vector for interfacial velocity
    vec_and_ptr_dim_t v_interface_new; // Auxiliary vector to temporarily ho

    for (int dir = 0; dir<P4EST_DIM;dir++){
        sample_cf_on_nodes(p4est,nodes,zero_cf,v_interface.vec[dir]);
      }



    // -----------------------------------------------
    // Initialize the fields relevant to the Poisson problem:
    // -----------------------------------------------
    // Vectors for T_liquid:
    vec_and_ptr_t T_l_n;
    vec_and_ptr_t rhs_Tl;

    // Sample the initial condition on the grid:
    T_l_n.create(p4est,nodes);
    sample_cf_on_nodes(p4est,nodes,temp_current_time,T_l_n.vec);

    // Vectors for T_solid:
    vec_and_ptr_t T_s_n;
    vec_and_ptr_t rhs_Ts;

    // Sample the initial condition on the grid:
    T_s_n.create(p4est,nodes);
    sample_cf_on_nodes(p4est,nodes,temp_current_time,T_s_n.vec);

    // Auxiliary vectors to hold T values on new grid (for interpolation purposes)
    vec_and_ptr_t T_l_new;
    vec_and_ptr_t T_s_new;

    // Vectors to hold first derivatives of T
    vec_and_ptr_dim_t T_l_d;
    vec_and_ptr_dim_t T_s_d;

    // Vectors to hold first derivatives of T
    vec_and_ptr_dim_t T_l_dd;
    vec_and_ptr_dim_t T_s_dd;

    // Vectors to hold solution for T_np1 temporarily within each timestep
    vec_and_ptr_t T_l_np1;
    vec_and_ptr_t T_s_np1;

    // Vectors to hold the normals of each domain:
    vec_and_ptr_dim_t liquid_normals;
    vec_and_ptr_dim_t solid_normals;
    vec_and_ptr_dim_t cyl_normals;

    // -----------------------------------------------
    // Initialize variables for extension bands across interface and etc:
    // -----------------------------------------------
    double dxyz_smallest[P4EST_DIM];
    double dxyz_close_to_interface;

    double min_volume_;
    double extension_band_use_;
    double extension_band_extend_;
    double extension_band_check_;

    // -----------------------------------------------
    // Initialize the output file for vtk:
    // -----------------------------------------------
    int out_idx = 0;
    char outdir[1000];
    const char* outdir_vtk = getenv("OUT_DIR_VTK");
    if(!outdir_vtk && save_vtk){
        throw std::invalid_argument("You need to set the environment variable OUT_DIR_VTK to save vtk files\n");
    }
    //sprintf(outdir,"%s/lmin_%d_lmax_%d_snapshot_%d",outdir_vtk,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);

    // -----------------------------------------------
    // Initialize file to output error and convergence information:
    // -----------------------------------------------
    // For checking error for Frank Sphere analytical solution:
    FILE *fich;
    char name[1000];
    if (example_ == FRANK_SPHERE){
      const char* outdir_err = getenv("OUT_DIR_ERR");
      if(!outdir_err && save_frank_sphere_errors){
          throw std::invalid_argument("You need to set the environment variable OUT_DIR_ERR to save vtk files\n");
      }

      CODE2D(sprintf(name,"%s/frank_sphere_error_2d_lmin_%d_lmax_%d_method_%d.dat",outdir_err,lmin+grid_res_iter,lmax + grid_res_iter,method_);)
      CODE3D(sprintf(name,"%s/frank_sphere_error_3d_lmin_%d_lmax_%d_method_%d.dat",outdir_err,lmin+grid_res_iter,lmax + grid_res_iter,method_);)

      ierr = PetscFOpen(mpi.comm(),name,"w",&fich); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(),fich,"time " "timestep " "iteration " "phi_error " "T_l_error " "T_s_error " "v_int_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(),fich); CHKERRXX(ierr);
      }

    // For checking memory usage
    FILE *fich_mem;
    char name_mem[1000];
    if(check_memory_usage){
        const char* outdir_mem = getenv("OUT_DIR_MEM");
        if(!outdir_mem && check_memory_usage){
            throw std::invalid_argument("You need to set the environment variable OUT_DIR_MEM to save memory usage files\n");
        }


          CODE2D(sprintf(name_mem,"%s/memory_usage_stefan_2d_lmin_%d_lmax_%d.dat",outdir_mem,lmin,lmax));
          CODE3D(sprintf(name_mem,"%s/memory_usage_stefan_3d_lmin_%d_lmax_%d.dat",outdir_mem,lmin,lmax));

          ierr = PetscFOpen(mpi.comm(),name_mem,"w",&fich_mem); CHKERRXX(ierr);

          ierr = PetscFPrintf(mpi.comm(),fich_mem,"time " "timestep " "iteration " "check1 check2 check3 check4 check5 check6 check7 check8 check9 check10 check11 check12 check13 check14 \n");CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);
    }


    // -----------------------------------------------
    // Initialize the needed solvers for the Temperature problem
    // -----------------------------------------------
    my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
    my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

    // -----------------------------------------------
    // Begin stepping through time
    // -----------------------------------------------
    int tstep = 0;

    PetscPrintf(mpi.comm(),"Gets to here ");
    for (tn;tn<tfinal; tn+=dt){
        tstep++;
        if (!keep_going) break;

        // Get current memory usage:
        PetscLogDouble mem1;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem1);


        // --------------------------------------------------------------------------------------------------------------
        // Print iteration information:
        // --------------------------------------------------------------------------------------------------------------

        PetscPrintf(mpi.comm(),"\n -------------------------------------------\n");
        ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3g , Timestep: %0.3e, Percent Done : %0.2f % \n ------------------------------------------- \n",tstep,tn,dt,((tn-tstart)/(tfinal - tstart))*100.0);

        // --------------------------------------------------------------------------------------------------------------
        // Define variables needed to specify how to extend across the interface:
        // --------------------------------------------------------------------------------------------------------------
        // Get smallest grid size:
        dxyz_min(p4est,dxyz_smallest);

        dxyz_close_to_interface = 1.0*max(dxyz_smallest[0],dxyz_smallest[1]);
        min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);

        // Extension band inputs : set to be some multiple of the smallest lengthscale on the grid
        extension_band_use_    = 16.*pow(min_volume_, 1./ double(P4EST_DIM)); //8
        extension_band_extend_ = 20.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
        extension_band_check_  = 12.*pow(min_volume_, 1./ double(P4EST_DIM)); // 6


        // If first iteration, perturb the LSF(s):
        my_p4est_level_set_t ls(ngbd);
        if(tstep<1){
            ls.perturb_level_set_function(phi.vec,EPS);
          }

        // --------------------------------------------------------------------------------------------------------------
        // STEP 1.1: Extend Fields Across Interface:
        // --------------------------------------------------------------------------------------------------------------
        // Define LSF for the solid domain (as just the negative of the liquid one):

        if(print_checkpoints)PetscPrintf(mpi.comm(),"Preparing to extend fields across interface \n");
        phi_solid.create(p4est,nodes);
        VecScaleGhost(phi.vec,-1.0);
        VecCopyGhost(phi.vec,phi_solid.vec);
        VecScaleGhost(phi.vec,-1.0);

        if(check_temperature_values){
          // Check Temperature values:
          PetscPrintf(mpi.comm(),"\n Checking temperature values before field extension: \n [ ");
          PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
          check_T_values(phi,T_l_n,nodes,p4est);
          PetscPrintf(mpi.comm(),"\nIn solid domain: ");
          check_T_values(phi_solid,T_s_n,nodes,p4est);
          PetscPrintf(mpi.comm()," ] \n");
          }

        // Get second derivatives of both LSFs:
        phi_dd.create(p4est,nodes);
        phi_solid_dd.create(p4est,nodes);

        ngbd->second_derivatives_central(phi.vec,phi_dd.vec);
        ngbd->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

        // Compute normals for each domain:
        liquid_normals.create(p4est,nodes);
        compute_normals(*ngbd,phi.vec,liquid_normals.vec);

        solid_normals.create(p4est,nodes);
        compute_normals(*ngbd,phi_solid.vec,solid_normals.vec);

        // Extend Temperature Fields across the interface:
        ls.extend_Over_Interface_TVD_Full(phi.vec, T_l_n.vec, 50, 2, 1.e-9, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normals.vec, NULL, NULL, false, NULL, NULL);
        ls.extend_Over_Interface_TVD_Full(phi_solid.vec, T_s_n.vec, 50, 2, 1.e-9, extension_band_use_, extension_band_extend_, extension_band_check_, solid_normals.vec, NULL, NULL, false, NULL, NULL);

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Successfully extended fields across interface \n");

        // Delete data for normals since it is no longer needed:
        liquid_normals.destroy();
        solid_normals.destroy();

        if (check_temperature_values){
          // Check Temperature values:
          PetscPrintf(mpi.comm(),"\n Checking temperature values after field extension: \n [ ");
          PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
          check_T_values(phi,T_l_n,nodes,p4est);
          PetscPrintf(mpi.comm(),"\nIn solid domain: ");
          check_T_values(phi_solid,T_s_n,nodes,p4est);
          PetscPrintf(mpi.comm()," ] \n");
          }

        PetscLogDouble mem2;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem2);
        // --------------------------------------------------------------------------------------------------------------
        // SAVING DATA: Save data every specified amout of timesteps: -- Do this after values are extended across interface (rather than before) to make visualization nicer
        // --------------------------------------------------------------------------------------------------------------
        if(save_vtk && (tstep%save_every_iter ==0)){
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Preparing to save vtk ... \n");

          // Prepare filename:
          char output[1000];
          out_idx++;
          sprintf(output,"%s/lmin_%d_lmax_%d_snapshot%d",outdir_vtk,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);

          // Evaluate the error for visualization (in Frank Sphere Case):
          vec_and_ptr_t T_ana;
          vec_and_ptr_t T_error;


          if(example_ == FRANK_SPHERE){
              T_ana.create(p4est,nodes);
              T_error.create(p4est,nodes);
              sample_cf_on_nodes(p4est,nodes,temp_current_time,T_ana.vec);

              T_error.get_array();T_ana.get_array();T_l_n.get_array();T_s_n.get_array();phi.get_array();
              foreach_node(n,nodes){
                if(phi.ptr[n]<=0.){
                  T_error.ptr[n] = fabs(T_l_n.ptr[n] - T_ana.ptr[n]);
                  }
                else{
                    T_error.ptr[n] = fabs(T_s_n.ptr[n] - T_ana.ptr[n]);
                  }

              }
              T_error.restore_array();
              T_ana.restore_array();
              T_l_n.restore_array();
              T_s_n.restore_array();
              phi.restore_array();


              save_stefan_fields(p4est,nodes,ghost,phi,T_l_n,T_s_n,v_interface,T_error,T_ana,output);
              T_ana.destroy();
              T_error.destroy();
          }
          else{
              save_stefan_fields(p4est,nodes,ghost,phi,T_l_n,T_s_n,v_interface,T_error,T_ana,output);
          }


          if(print_checkpoints) PetscPrintf(mpi.comm(),"Successfully saved to vtk \n");
          }

        PetscLogDouble mem3;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem3);

        // --------------------------------------------------------------------------------------------------------------
        // STEP 1.2: Compute the jump in flux across the interface to use to advance the LSF:
        // --------------------------------------------------------------------------------------------------------------
        // Get the first derivatives to compute the jump
        T_l_d.create(p4est,nodes); T_s_d.create(T_l_d.vec);
        ngbd->first_derivatives_central(T_l_n.vec,T_l_d.vec);
        ngbd->first_derivatives_central(T_s_n.vec,T_s_d.vec);

        // Create vector to hold the jump values:
        vec_and_ptr_dim_t jump;
        jump.create(p4est,nodes);
        v_interface.destroy();
        v_interface.create(p4est,nodes);

        // Call the compute_velocity_function:
        compute_interfacial_velocity(T_l_d,T_s_d,jump,v_interface,phi,ngbd);

        // Destroy values once no longer needed:
        T_l_d.destroy();
        T_s_d.destroy();
        jump.destroy();


        PetscLogDouble mem4;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem4);
        // --------------------------------------------------------------------------------------------------------------
        // Compute the timestep -- determined by velocity at the interface:
        // --------------------------------------------------------------------------------------------------------------
        // Call function to compute the timestep, provided the current interfacial velocity and some information about the grid:
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing timestep ...\n");
        compute_timestep(v_interface, phi, dxyz_close_to_interface, dxyz_smallest,nodes,p4est); // this function modifies the global variable dt

        // Adjust the timestep if we are near the end of our simulation, to get the proper end time:
        if(tn + dt > tfinal){
            dt = tfinal - tn;
          }
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Timestep successfully computed \n");

        PetscLogDouble mem5;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem5);
        // --------------------------------------------------------------------------------------------------------------
        // STEP 2.1/STEP 2.2: Advance the LSF/Update the grid :
        // --------------------------------------------------------------------------------------------------------------
        // We advect the LSF and update the grid according to the LSF

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Making copies of the grid \n");
        // Make a copy of the grid objects for the next timestep:
        p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
        ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);

        // Expand the ghost layer for navier stokes:
        nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Copied grid successfully \n");

        // Create the semi-lagrangian object and do the advection:
        my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Advecting LSF and updating grid ...\n");

        // Advect the LSF and update the grid under the v_interface field:
        sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec);

        // Destroy old derivative values now that they are no longer needed (they were needed for field extension, LSF advection and grid update):
        phi_dd.destroy();

        // Get the new neighbors:
        my_p4est_hierarchy_t *hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
        my_p4est_node_neighbors_t *ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

        // Initialize the neigbors:
        ngbd_np1->init_neighbors();

        if(print_checkpoints) PetscPrintf(mpi.comm(),"LSF advected and grid update successful \n");

        PetscLogDouble mem6;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem6);
        // --------------------------------------------------------------------------------------------------------------
        // STEP 2.3: Reinitialize the LSF on the new grid
        // --------------------------------------------------------------------------------------------------------------

        my_p4est_level_set_t ls_new(ngbd_np1);
        ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec, 50);
        ls_new.perturb_level_set_function(phi.vec,EPS);

        PetscLogDouble mem7;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem7);
        // --------------------------------------------------------------------------------------------------------------
        // STEP 2.4: Interpolate Values onto New Grid:
        // -------------------------------------------------------------------------------------------------------------
        // Create vectors to hold new values:

        T_l_new.create(p4est_np1,nodes_np1);
        T_s_new.create(T_l_new.vec);
        v_interface_new.create(p4est_np1,nodes_np1);

        // Interpolate things to the new grid:
        interpolate_values_onto_new_grid(T_l_n,T_l_new,
                                         T_s_n, T_s_new,
                                         v_interface, v_interface_new,
                                         nodes_np1, p4est_np1,
                                         ngbd, interp_bw_grids);

        //---------------------
        // Copy new data over:
        //---------------------
        // First, recreate vectors on the new grid:
        T_l_n.destroy(); T_s_n.destroy();
        T_l_n.create(p4est_np1,nodes_np1); T_s_n.create(T_l_n.vec);

        v_interface.destroy();
        v_interface.create(p4est_np1,nodes_np1);

        // Next, transfer new values to the original objects:
        VecCopyGhost(T_l_new.vec,T_l_n.vec);
        VecCopyGhost(T_s_new.vec,T_s_n.vec);

        foreach_dimension(d){
          VecCopyGhost(v_interface_new.vec[d],v_interface.vec[d]);
        }

        // Delete the "new value" objects since they are no longer needed:
        T_l_new.destroy(); T_s_new.destroy();
        v_interface_new.destroy();

        // Get the new solid LSF:
        phi_solid.destroy();
        phi_solid.create(p4est_np1,nodes_np1);
        VecCopyGhost(phi.vec,phi_solid.vec);
        VecScaleGhost(phi_solid.vec,-1.0);

        PetscLogDouble mem8;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem8);
        // --------------------------------------------------------------------------------------------------------------
        // Compute the normal and curvature of the interface -- curvature is used in some of the interfacial boundary condition(s) on temperature
        // --------------------------------------------------------------------------------------------------------------
        // Create vectors needed to compute normal and curvature:
        vec_and_ptr_dim_t normal;
        vec_and_ptr_t curvature;  // This one will hold curvature extended from interface to whole domain

        normal.create(p4est_np1,nodes_np1);
        curvature.create(p4est_np1,nodes_np1);

        // Compute normals on the interface:
        compute_normals(*ngbd_np1,phi.vec,normal.vec);

        // Compute curvature on the interface:
        my_p4est_level_set_t ls_new_new(ngbd_np1);
        compute_curvature(phi,normal,curvature,ngbd_np1,ls_new_new);

        // Destroy normals since they are no longer needed:
        normal.destroy();

        PetscLogDouble mem9;
        if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem9);
        // --------------------------------------------------------------------------------------------------------------
        // STEP 3: Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
        // --------------------------------------------------------------------------------------------------------------
        // Get most updated derivatives of the LSF's (on current grid) -- Solver uses these:
        // ------------------------------------------------------------

          if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Poisson problem ... \n");
          phi_solid_dd.create(p4est_np1,nodes_np1);
          ngbd_np1->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

          phi_dd.create(p4est_np1,nodes_np1);
          ngbd_np1->second_derivatives_central(phi.vec,phi_dd.vec);


          // Do quick optional check of values after interpolation: --> don't check till now bc we need phi_cylinder on new grid for ex 2
          if (check_temperature_values){
            // Check Temperature values:
            PetscPrintf(mpi.comm(),"\n Checking temperature values after interpolating onto new grid: \n [ ");
            PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
            check_T_values(phi,T_l_n,nodes_np1,p4est_np1);
            PetscPrintf(mpi.comm(),"\nIn solid domain: ");
            check_T_values(phi_solid,T_s_n,nodes_np1,p4est_np1);
            PetscPrintf(mpi.comm()," ] \n");
            }

          // ------------------------------------------------------------
          // STEP 3.1: Setup RHS:
          // ------------------------------------------------------------
          // Create arrays to hold the RHS:
          rhs_Tl.create(p4est_np1,nodes_np1);
          rhs_Ts.create(p4est_np1,nodes_np1);

          // Set up the RHS:
          setup_rhs(T_l_n,T_s_n,
                    rhs_Tl,rhs_Ts,
                    p4est_np1,nodes_np1,ngbd_np1);

          PetscLogDouble mem10;
          if(check_memory_usage)PetscMemoryGetCurrentUsage(&mem10);
          // ------------------------------------------------------------
          // STEP 3.1(cont.):Setup the solvers:
          // ------------------------------------------------------------
          // Now, set up the solver(s):
          solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd_np1);
          solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd_np1);

          BC_interface_value bc_interface_val(ngbd_np1,curvature);

          // Add the appropriate interfaces and interfacial boundary conditions:
          solver_Tl->add_boundary(MLS_INTERSECTION,phi.vec,DIM(phi_dd.vec[0],phi_dd.vec[1],phi_dd.vec[2]),interface_bc_type_temp,bc_interface_val,bc_interface_coeff);
          solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid.vec,DIM(phi_solid_dd.vec[0],phi_solid_dd.vec[1],phi_solid_dd.vec[2]),interface_bc_type_temp,bc_interface_val,bc_interface_coeff);


          // Set diagonal for Tl:
          if(method_ == 2){
              solver_Tl->set_diag(2./dt);
              solver_Ts->set_diag(2./dt);
          }
          else if(method_ ==1){
              solver_Tl->set_diag(1./dt);
              solver_Ts->set_diag(1./dt);
          }
          else{
              throw std::invalid_argument("Error setting the diagonal. You must select a time-stepping method!\n");
          }

          // Set diffusivities and RHS for each subdomain
          solver_Tl->set_mu(alpha_l);
          solver_Tl->set_rhs(rhs_Tl.vec);

          solver_Ts->set_mu(alpha_s);
          solver_Ts->set_rhs(rhs_Ts.vec);

          // Set some other solver properties:
          solver_Tl->set_integration_order(1); // For Neumann/Robin
          solver_Tl->set_use_sc_scheme(0); // For Neumann/Robin
          solver_Tl->set_cube_refinement(cube_refinement); // For Neumann/Robin
          solver_Tl->set_store_finite_volumes(0);

          solver_Ts->set_integration_order(1);
          solver_Ts->set_use_sc_scheme(0);
          solver_Ts->set_cube_refinement(cube_refinement);
          solver_Ts->set_store_finite_volumes(0);

          // Set the wall BC:
          solver_Tl ->set_wc(wall_bc_type_temp,wall_bc_value_temp);
          solver_Ts ->set_wc(wall_bc_type_temp,wall_bc_value_temp);


          PetscLogDouble mem11;
          if(check_memory_usage) PetscMemoryGetCurrentUsage(&mem11);
          // ---------------------------------------------------------------------------
          // STEP 3.2: Get the solution to your Poisson equation!
          // ---------------------------------------------------------------------------
          // Preassemble the linear system
          solver_Tl->preassemble_linear_system();
          solver_Ts->preassemble_linear_system();

          // Create vector to hold the solution:
          T_l_np1.create(p4est_np1,nodes_np1);
          T_s_np1.create(T_l_np1.vec);

          // Solve the system:
          solver_Tl->solve(T_l_np1.vec);
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Solved Tl \n");

          solver_Ts->solve(T_s_np1.vec);
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Solved Ts \n");

          // Destroy the T_n values now and update them with the solution for the next timestep:
          T_l_n.destroy(); T_s_n.destroy();
          T_l_n.create(p4est_np1,nodes_np1); T_s_n.create(T_l_n.vec);

          VecCopyGhost(T_l_np1.vec,T_l_n.vec);
          VecCopyGhost(T_s_np1.vec,T_s_n.vec);

          // Destroy solvers once done:
          delete solver_Tl;
          delete solver_Ts;

          // Destroy np1 vectors now that theyre not needed:
          T_l_np1.destroy(); T_s_np1.destroy();

          phi_dd.destroy(); phi_solid_dd.destroy();
          curvature.destroy();


          // Destroy rhs vectors now that no longer in use:
          rhs_Tl.destroy();
          rhs_Ts.destroy();
          if (check_temperature_values){
            // Check Temperature values:
            PetscPrintf(mpi.comm(),"\n Checking temperature values after acquiring solution: \n [ ");
            PetscPrintf(mpi.comm(),"\n In fluid domain: ");
            check_T_values(phi,T_l_n,nodes_np1,p4est_np1);
            PetscPrintf(mpi.comm(),"\n In solid domain: ");
            check_T_values(phi_solid,T_s_n,nodes_np1,p4est_np1);
            PetscPrintf(mpi.comm()," ] \n");
            }

          phi_solid.destroy();

          PetscLogDouble mem12;
          PetscMemoryGetCurrentUsage(&mem12);
          // ------------------------------------------------------------
          // Some example specific operations for the Poisson problem:
          // ------------------------------------------------------------
          // Check error on the Frank sphere, if relevant:
          if(example_ == FRANK_SPHERE){
              check_frank_sphere_error(T_l_n, T_s_n, phi, v_interface, p4est_np1, nodes_np1, dxyz_close_to_interface,name,fich,tstep);
            }
          if(example_ == ICE_MELT){
              keep_going = check_ice_melted(phi,tn + dt,nodes_np1,p4est_np1);
          }

        PetscLogDouble mem13;
        PetscMemoryGetCurrentUsage(&mem13);

        // --------------------------------------------------------------------------------------------------------------
        // Delete the old grid:
        // --------------------------------------------------------------------------------------------------------------
        // Delete the old grid and update with the new one:

        p4est_destroy(p4est); p4est = p4est_np1;
        p4est_ghost_destroy(ghost); ghost = ghost_np1;
        p4est_nodes_destroy(nodes); nodes = nodes_np1;

        delete hierarchy; hierarchy = hierarchy_np1;
        delete ngbd; ngbd = ngbd_np1;


        // Get current memory usage and print out all memory usage checkpoints:
        PetscLogDouble mem14;
        PetscMemoryGetCurrentUsage(&mem14);
        if(check_memory_usage) PetscPrintf(mpi.comm(),"Memory used %g \n\n",mem14);


        if(check_memory_usage){
              ierr = PetscFOpen(mpi.comm(),name_mem,"a",&fich_mem); CHKERRXX(ierr);
              ierr = PetscFPrintf(mpi.comm(),fich_mem,"%g %g %d %g %g %g %g %g %g %g %g %g %g %g %g %g %g \n",tn,dt,tstep,mem1,mem2,mem3,mem4,mem5,mem6,mem7,mem8,mem9,mem10,mem11,mem12,mem13,mem14);CHKERRXX(ierr);
              ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);
        }

      } // <-- End of for loop through time

    phi.destroy();
    T_l_n.destroy();
    T_s_n.destroy();


    // destroy the structures
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
    my_p4est_brick_destroy(conn, &brick);

}//end of loop through grid resolutions
  w.stop(); w.read_duration();
}
