/*
 * Title: multialloy_with_fluids
 * Description:
 * Author: Elyce
 * Date Created: 08-06-2019
 */


#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>


#include <src/my_p4est_semi_lagrangian.h>


#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_multialloy.h>




#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8esT_l_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>

#include <src/my_p8esT_l_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>
#include <src/parameter_list.h>


using namespace std;
parameter_list_t pl;

// ---------------------------------------
// Examples to run:
// ---------------------------------------
// Define the numeric label for each type of example to make implementation a bit more clear
enum{
  FRANK_SPHERE = 0,
  NS_GIBOU_EXAMPLE = 1,
  NS_LLNL_EXAMPLE = 2,
  COUPLED_PROBLEM_EXAMPLE = 3,
  ICE_AROUND_CYLINDER = 4,
  FLOW_PAST_CYLINDER = 5,

};

enum{LIQUID_DOMAIN=0, SOLID_DOMAIN=1};
DEFINE_PARAMETER(pl,int,example_,3,"example number: \n"
                                   "0 - Frank Sphere (Stefan only) \n"
                                   "1 - NS Gibou example (Navier Stokes only) \n"
                                   "2 - work in progress \n"
                                   "3 - Coupled problem example for verification \n"
                                   "4 - Ice solidifying around a cooled cylinder \n"
                                   "5 - Flow past a cylinder (WIP) (Navier Stokes only)\n"
                                   "default: 4");

// ---------------------------------------
// Save options:
// ---------------------------------------
DEFINE_PARAMETER(pl,bool,save_stefan,false,"Save stefan ?");
DEFINE_PARAMETER(pl,bool,save_navier_stokes,false,"Save navier stokes?");
DEFINE_PARAMETER(pl,bool,save_coupled_fields,true,"Save the coupled problem?");

DEFINE_PARAMETER(pl,bool,save_to_vtk,true,"We save vtk files using a given dt increment if this is set to true \n");
DEFINE_PARAMETER(pl,bool,save_using_dt,true,"We save vtk files using a given dt increment if this is set to true \n");
DEFINE_PARAMETER(pl,bool,save_using_iter,false,"We save every prescribed number of iterations if this is set to true \n");

DEFINE_PARAMETER(pl,int,save_every_iter,1,"Saves vtk every n number of iterations (default is 1)");
DEFINE_PARAMETER(pl,int,save_state_every_iter,10000,"Saves simulation state every n number of iterations (default is 500)");
DEFINE_PARAMETER(pl,int,num_save_states,20,"Number of save states we keep on file (default is 20)");

DEFINE_PARAMETER(pl,int,check_mem_every_iter,1000,"Checks memory usage every n number of iterations (default is 1000)");

DEFINE_PARAMETER(pl,double,save_every_dt,0.1,"Saves vtk every dt amount of time (default is .1)");


DEFINE_PARAMETER(pl,int,timing_every_n,100,"Print timing info every n iterations (default 100)");
DEFINE_PARAMETER(pl,bool,print_checkpoints,false,"Print checkpoints throughout script for debugging? ");
DEFINE_PARAMETER(pl,bool,mem_checkpoints,false,"checks various memory checkpoints for mem usage");
DEFINE_PARAMETER(pl,double,mem_safety_limit,60.e9,"Memory upper liFmit before closing the program -- in bytes");
DEFINE_PARAMETER(pl,bool,save_fluid_forces,false,"Saves fluid forces if true (default: false) \n");

// Load options
DEFINE_PARAMETER(pl,bool,loading_from_previous_state,false,"");

// ---------------------------------------
// Solution options:
// ---------------------------------------
DEFINE_PARAMETER(pl,bool,solve_stefan,false,"Solve stefan ?");
DEFINE_PARAMETER(pl,bool,solve_navier_stokes,false,"Solve navier stokes?");
DEFINE_PARAMETER(pl,bool,solve_coupled,true,"Solve the coupled problem?");
DEFINE_PARAMETER(pl,bool,do_advection,1,"Boolean flag whether or not to do advection (default : 1)");
DEFINE_PARAMETER(pl,double,Re_overwrite,-100.0,"overwrite the examples set Reynolds number");
DEFINE_PARAMETER(pl,double,duration_overwrite,-100.0,"overwrite the duration");

DEFINE_PARAMETER(pl,bool,use_uniform_band,true,"Boolean whether or not to refine using a uniform band");

void select_solvers(){
  switch(example_){
    case FRANK_SPHERE:
      save_stefan = true;
      solve_stefan = true;
      save_navier_stokes = false;
      solve_navier_stokes = false;
      save_coupled_fields = false;
      do_advection = false;
      break;
    case ICE_AROUND_CYLINDER:
      save_stefan = false;
      solve_stefan = true;
      solve_navier_stokes = true;
      save_navier_stokes = false;//false;
      save_coupled_fields = true;//true;
      do_advection=true;
      break;
    case NS_GIBOU_EXAMPLE:
      save_stefan = false; solve_stefan = false;
      save_navier_stokes = true; solve_navier_stokes = true;
      save_coupled_fields = false;

      break;
    case FLOW_PAST_CYLINDER:
      save_stefan = false; solve_stefan = false;
      save_navier_stokes = true; solve_navier_stokes = true;
      save_coupled_fields = false;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      save_stefan = false; solve_stefan = true;
      save_navier_stokes = false; solve_navier_stokes = true;
      save_coupled_fields = true;
      break;

    }
  if(save_using_dt && save_using_iter){
      throw std::invalid_argument("You have selected to save using dt and using iteration, you need to select only one \n");
    }
}

DEFINE_PARAMETER(pl,int,advection_sl_order,2,"Integer for advection solution order (can choose 1 or 2) for the fluid temperature field(default:1) -- note: this also sets the NS solution order");
DEFINE_PARAMETER(pl,int,NS_advection_sl_order,2,"Integer for advection solution order (can choose 1 or 2) for the fluid velocity fields (default:1)");

DEFINE_PARAMETER(pl,double,hodge_tolerance,1.e-3,"Tolerance on hodge for error convergence (default:1.e-3)");


DEFINE_PARAMETER(pl,double,cfl,0.5,"CFL number (default:0.5)");
DEFINE_PARAMETER(pl,bool,force_interfacial_velocity_to_zero,false,"Force the interfacial velocity to zero? ");
DEFINE_PARAMETER(pl,double,vorticity_threshold,0.05,"Threshold to refine vorticity by, default is 0.1 \n");
DEFINE_PARAMETER(pl,double,gradT_threshold,1.e-4,"Threshold to refine the nondimensionalized temperature gradient by \n (default: 0.99)");

int stop_flag = -1;
// ---------------------------------------
// Geometry options:
// ---------------------------------------
DEFINE_PARAMETER(pl,double,xmin,0.,"Minimum dimension in x (default: 0)");
DEFINE_PARAMETER(pl,double,xmax,1.,"Maximum dimension in x (default: 0)");

DEFINE_PARAMETER(pl,double,ymin,0.,"Minimum dimension in y (default: 0)");
DEFINE_PARAMETER(pl,double,ymax,1.,"Maximum dimension in y (default: 1)");

DEFINE_PARAMETER(pl,int,nx,1,"Number of trees in x (default:1)");
DEFINE_PARAMETER(pl,int,ny,1,"Number of trees in y (default:1)");

DEFINE_PARAMETER(pl,int,px,0,"Periodicity in x (default false)");
DEFINE_PARAMETER(pl,int,py,0,"Periodicity in y (default false)");

DEFINE_PARAMETER(pl,double,uniform_band,4.,"Uniform band (default:4.)");

// For level set:
double r0;

// For frank sphere:
double s0;
double T_inf;

// For solution of temperature fields:
double Twall;
double Tinterface;
double back_wall_temp_flux;
double deltaT;

double theta_wall;
double theta_interface;
double theta_cyl;

// For solidifying ice problem:
double r_cyl; // non dim
double T_cyl; // dimensional
double d_cyl; // dimensional

// For tracking allowable temperature values:
double T_max_allowable;
double T_max_allowable_err = 1.0e-7;

// Keeping track of maxes and mins:
double T_l_max, T_l_min, T_s_max, T_s_min;

// For surface tension: (used to apply some interfacial BC's in temperature)
double sigma;

// For the coupled test case where we have to swtich sign:
double coupled_test_sign;
bool vel_has_switched;
void coupled_test_switch_sign(){coupled_test_sign*=-1.;}

double x0_lsf;
double y0_lsf;

unsigned int num_fields_interp = 0;

void set_geometry(){
  switch(example_){
  case FRANK_SPHERE: {
    // Frank sphere
    // Grid size
    xmin = -5.0; xmax = 5.0; //5.0;
    ymin = -5.0; ymax = 5.0;

    // Number of trees
    nx = 2;
    ny = 2;

    // Periodicity
    px = 0; py = 0;

    // Uniform band
    uniform_band=4.;

    // Problem geometry:
    s0 = 0.628649269043202;
    r0 = s0; // for consistency, and for setting up NS problem (if wanted)

    // Necessary boundary condition info:
    Twall = -0.2;    T_inf = Twall;

    Tinterface = 0.0;

    break;
    }

    case FLOW_PAST_CYLINDER: // intentionally waterfalls into same settings as ice around cylinder

    case ICE_AROUND_CYLINDER:{ // Ice layer growing around a constant temperature cooled cylinder

      // Domain size:
      xmin = 0.0; xmax = 20.0;/*32.0;*/
      ymin = 0.0; ymax = 10.0;/*16.0;*/

      // Number of trees:
      nx =8;/*8;*/
      ny =4;/*4;*/

      // Periodicity:
      px = 0;
      py = 1;

      // Problem geometry:
      r_cyl = 0.5;     // Computational radius of the cylinder
      /*r0 = r_cyl*1.17;*/ // Computational radius of initial ice height // should set r0 = r_cyl*1.0572 to get height_init = 1mm, matching the experiments (or at least matching the Okada model)
      //r0 = r_cyl*1.0572;
      r0 = r_cyl*1.10;
//      r0 = r_cyl*1.7;
      d_cyl = 35.e-3;  // Physical diameter of cylinder [m] -- value used for some nondimensionalization

      // Boundary condition info:
      Twall = 276.;    // Physical wall temp [K]
      Tinterface = 273.0; // Physical interface temp [K]
      T_cyl = 260.;   // Physical cylinder temp [K]

      back_wall_temp_flux = 0.0; // Flux in temp on back wall (non dim) (?) TO-DO: check this

      deltaT = Twall - T_cyl; // Characteristic Delta T [K] -- used for some non dimensionalization

      theta_wall = 1.0; // Non dim temp at wall
      theta_cyl = 0.0; // Non dim temp at cylinder

      theta_interface = (Tinterface - T_cyl)/(deltaT); // Non dim temp at interface

      T_max_allowable=Twall; // Max allowable value for T

      break;}

    case NS_GIBOU_EXAMPLE: {// Navier Stokes Validation case from Gibou 2015
      // Domain size:
      xmin = 0.0; xmax = PI;
      ymin = 0.0; ymax = PI;

      // Number of trees:
      nx = 2; ny = 2;
      px = 0; py = 0;

      // Radius of the level set function:
      r0 = 0.20;
      break;}

    case COUPLED_PROBLEM_EXAMPLE:{
      // Domain size:
      xmin = -PI/2.; xmax = 3.*PI/2.;
      ymin = -PI; ymax = PI;

      x0_lsf = PI/2; y0_lsf = PI/4.;

      // Number of trees:
      nx = 2; ny = 2;
      px = 0; py = 0;

      // Radius of the level set function:
      r0 = PI/3.;

      uniform_band=4.;

      break;}
    }
  // set number of interpolation fields:
  num_fields_interp = 0;
  if(solve_stefan){
    num_fields_interp+=4; // Tl, Ts, vint_x, vint_y
  }
  if(solve_navier_stokes){
    num_fields_interp+=2; // vNS_x, vNS_y
  }
  }

double v_interface_max_norm; // For keeping track of the interfacial velocity maximum norm

// ---------------------------------------
// Grid refinement:
// ---------------------------------------
DEFINE_PARAMETER(pl,int,lmin,3,"Minimum level of refinement");
DEFINE_PARAMETER(pl,int,lmax,8,"Maximum level of refinement");
DEFINE_PARAMETER(pl,double,lip,1.75,"Lipschitz coefficient");
DEFINE_PARAMETER(pl,int,method_,1,"Solver in time for solid domain, and for fluid if no advection. 1 - Backward Euler, 2 - Crank Nicholson");
DEFINE_PARAMETER(pl,int,num_splits,0,"Number of splits -- used for convergence tests");
DEFINE_PARAMETER(pl,bool,refine_by_ucomponent,false,"Flag for whether or not to refine by a backflow condition for the fluid velocity");
DEFINE_PARAMETER(pl,bool,refine_by_d2T,true,"Flag for whether or not to refine by the nondimensionalized temperature gradient");

// ---------------------------------------
// Non dimensional groups:
// ---------------------------------------
DEFINE_PARAMETER(pl,double,Re,300.,"Reynolds number - default is 300 \n");
DEFINE_PARAMETER(pl,double,Pr,0.,"Prandtl number - computed from mu_l, alpha_l, rho_l \n");
DEFINE_PARAMETER(pl,double,Pe,0.,"Peclet number - computed from Re and Pr \n");
DEFINE_PARAMETER(pl,double,St,0.,"Stefan number - computed from cp_s, deltaT, L \n");

// ---------------------------------------
// Physical properties:
// ---------------------------------------
double alpha_s;
double alpha_l;
double k_s;
double k_l;
double L; // Latent heat of fusion
double rho_l;
double rho_s;
double cp_s;
double mu_l;

void set_physical_properties(){
  switch(example_){
    case FRANK_SPHERE:
      alpha_s = 1.0;
      alpha_l = 1.0;
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      rho_s = 1.0;
      break;

    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER:
      alpha_s = (1.1e-6); //ice - [m^2]/s
      alpha_l = (1.3e-7); //water- [m^2]/s  // 6-16-2020: changed from 1.5 e-7 to 1.3e-7 to be more consistent with Okada paper peclet number --> Pe ~ 13, before was doing Pe ~ 11
      k_s = 2.22; // W/[m*K]
      k_l = 0.608; // W/[m*K]
      L = 334.e3;  // J/kg
      rho_l = 1000.0;// kg/m^3
      sigma = (2.10e-10); // [m]
      rho_s = 920.; //[kg/m^3]
      mu_l = 1.7106e-3;//1.793e-3;  // Viscosity of water , [Pa s]
      cp_s = k_s/(alpha_s*rho_s); // Specific heat of solid  []
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      alpha_s = 1.0;
      alpha_l = 1.0;
      k_s = 1.;
      k_l = 1.;
      L = 1.;
      rho_l = 1.;
      rho_s = 1.0;
      mu_l = 1.0;
      break;
  case NS_GIBOU_EXAMPLE:
      rho_l = 1.0;
      mu_l = 1.0;
      Re = 1.0;
    }
}

//-----------------------------------------
// Properties to set if you are solving NS
// ----------------------------------------
double pressure_prescribed_flux;
double pressure_prescribed_value;
double u0;
double v0;

double Re_u; // reynolds number in x direction
double Re_v; // reynolds number in y direction


double outflow_u;
double outflow_v;
double hodge_percentage_of_max_u;
int hodge_max_it = 30;
double T_l_IC_band = 2.0;
bool ramp_T_l_IC_space = false;
double dt_NS;

double hodge_global_error;

double NS_norm = 0.0; // To keep track of the NS norm

double u_inf; // physical value of freestream velocity
void set_NS_info(){
  pressure_prescribed_flux = 0.0; // For the Neumann condition on the two x walls and lower y wall
  pressure_prescribed_value = 0.0; // For the Dirichlet condition on the back y wall

  dt_NS = 1.e-3; // initial dt for NS

  // Note: fluid velocity is set via Re and u0,v0 --> v0 = 0 is equivalent to single direction flow, u0=1, v0=1 means both directions will flow at Re (TO-DO:make this more clear)
  switch(example_){
    case FRANK_SPHERE:throw std::invalid_argument("NS isnt setup for this example");
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER:
      Re = 201.;
      u0 = 1; // computational freestream velocity
      v0 = 0;
      break;
    case NS_GIBOU_EXAMPLE:
      Re = 1.0;

      u0 = 1.0;
      v0 = 1.0;

      u_inf=1.0;
      uniform_band = 4.;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      Re = 1.0;
      u0 = 1.0;
      v0 = 1.0;
      uniform_band = 4.;
      break;
    }

  outflow_u = 0.0;
  outflow_v = 0.0;

  hodge_percentage_of_max_u = 1.e-3;
}

void set_nondimensional_groups(){
    if ((example_==ICE_AROUND_CYLINDER) || (example_==FLOW_PAST_CYLINDER)){
        if(Re_overwrite>0.) Re = Re_overwrite;
        u_inf=Re*mu_l/rho_l/d_cyl;
        Pr = mu_l/(alpha_l*rho_l);
        Pe = Re*Pr;
        St = cp_s*deltaT/L;
    }
    else{
      Pe = 1.;
      St = 1.;
      Pr = 1.;
    }
}

// ---------------------------------------
// Time-stepping:
// ---------------------------------------
double tfinal;
double dt_max_allowed;
bool keep_going = true;

double tn;
double tstart;
double dt;
double dt_nm1;
int tstep;
DEFINE_PARAMETER(pl,double,t_ramp,0.25,"Time at which NS boundary conditions are ramped up to the freestream value \n");
DEFINE_PARAMETER(pl,bool,ramp_bcs,true,"Boolean option to ramp the BC's for the ice over cylinder case \n");

void simulation_time_info(){
  switch(example_){
    case FRANK_SPHERE:
      tfinal = 1.30;
      dt_max_allowed = 1.e-4;/*0.1;*/
      tstart = 1.0;
      break;
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER: // ice solidifying around isothermally cooled cylinder
      tfinal = (40.*60.)*(u_inf/d_cyl); // 40 minutes
      if(duration_overwrite>0.) tfinal = (duration_overwrite*60.)*(u_inf/d_cyl);
      dt_max_allowed = save_every_dt;
      tstart = 0.0;
      dt = 1.e-5;
      break;

    case NS_GIBOU_EXAMPLE:
      tfinal = PI/3.;
      dt_max_allowed = 1.e-2;
      tstart = 0.0;
      dt = 1.e-3;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      tfinal = PI/2.;
      dt_max_allowed = 1.0e-1;
      tstart = 0.0;
      dt = 1.e-3;
      break;

    }
  dt_nm1 = dt;
}

// ---------------------------------------
// Other parameters:
// ---------------------------------------
double v_int_max_allowed = 10.0;

bool move_interface_with_v_external = false;

bool check_temperature_values = false; // Whether or not you want to print out temperature value averages during various steps of the solution process -- for debugging

DEFINE_PARAMETER(pl,bool,check_avg_values,false,"Check average temp vals per timestep or not (default:false)");
bool check_derivative_values = false;// Whether or not you want to print out temperature derivative value averages during various steps of the solution process -- for debugging

bool check_interfacial_velocity = true; // Whether or not you want to print out interfacial velocity value averages during various steps of the solution process -- for debugging

bool save_temperature_derivative_fields = false; // saving temperature derivative fields to vtk or not

// Variables used for advection:
double advection_alpha_coeff= 0.0;
double advection_beta_coeff =0.0;

// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
// Frank sphere functions -- Functions necessary for evaluating the analytical solution of the Frank sphere problem, to validate results for example 1
// --------------------------------------------------------------------------------------------------------------
double s(double r, double t){
  //std::cout<<"Time being used to compute s is: " << t << "\n"<< std::endl;
  return r/sqrt(t);
}

// Error function : taken from existing code in examples/stefan/main_2d.cpp
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
  return E1(z);
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
// Functions for validating the Navier-Stokes solver:
// --------------------------------------------------------------------------------------------------------------
// Re-doing the NS validation case:
struct velocity_component: CF_DIM
{
  const unsigned char dir;
  const double k_NS=1.0;
  velocity_component(const unsigned char& dir_) : dir(dir_){
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  double v(DIM(double x, double y, double z)) const{ // gives vel components without the time component
    switch(dir){
    case dir::x:
      return sin(x)*cos(y);
    case dir::y:
      return -1.0*cos(x)*sin(y);
    default:
      throw std::runtime_error("analytical solution velocity: unknown cartesian direction \n");
    }
  }
  double dv_d(const unsigned char& dirr,DIM(double x, double y, double z)) const{
    switch(dir){
    case dir::x:
      switch(dirr){
      case dir::x: //du_dx (without time component)
        return cos(x)*cos(y);
      case dir::y: // du_dy (without time component)
        return -sin(x)*sin(y);
      }
    case dir::y:
      switch(dirr){
      case dir::x: // dv_dx ("")
        return sin(x)*sin(y);
      case dir::y: // dv_dy ("")
        return -cos(x)*cos(y);
      }
    }
  }

  double operator()(DIM(double x, double y, double z)) const{ // Returns the velocity field
    return cos(t*k_NS)*v(DIM(x,y,z));
  }
  double _d(const unsigned char& dirr, DIM(double x, double y, double z)){ // Returns spatial derivatives of velocity field in given cartesian direction
    return cos(t*k_NS)*dv_d(dirr,DIM(x,y,z));
  }
  double laplace(DIM(double x, double y, double z)){
    return -P4EST_DIM*cos(t*k_NS)*v(DIM(x,y,z));
  }
  double dv_dt(DIM(double x, double y, double z)){
    return -sin(k_NS*t)*v(DIM(x,y,z));
  }
};

struct external_force_per_unit_volume_component : CF_DIM{
  const unsigned char dir;
  velocity_component** velocity_field;
  external_force_per_unit_volume_component(const unsigned char& dir_, velocity_component** analytical_soln):dir(dir_),velocity_field(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const{ // returns the forcing term in a given direction
    return velocity_field[dir]->dv_dt(DIM(x,y,z)) +
        SUMD((*velocity_field[0])(DIM(x,y,z))*velocity_field[dir]->_d(dir::x,DIM(x,y,z)),
        (*velocity_field[1])(DIM(x,y,z))*velocity_field[dir]->_d(dir::y,DIM(x,y,z)),
        (*velocity_field[2])(DIM(x,y,z))*velocity_field[dir]->_d(dir::z,DIM(x,y,z))) -
        velocity_field[dir]->laplace(DIM(x,y,z));
  }
};
//------------------------------------------------------------------------
// For coupled problem validation:
// -----------------------------------------------------------------------
struct temperature_field: CF_DIM
{
  const unsigned char dom; //dom signifies which domain--> domain liq = 0, domain solid =1
  const double k_NS=1.0;

  const double n = 1.0;
  const double m = 2.0;
  const double x0 = PI/4;
  const double y0 = 0.;

  temperature_field(const unsigned char& dom_) : dom(dom_){
    P4EST_ASSERT(dom>=0 && dom<=1);
  }

  double T(DIM(double x, double y, double z)) const{
    switch(dom){
    case LIQUID_DOMAIN:
      return sin(x)*sin(y)*(x + cos(t)*cos(x)*cos(y));
    case SOLID_DOMAIN:
      return cos(x)*cos(y)*(cos(t)*sin(x)*sin(y) - 1.);

    default:
      throw std::runtime_error("analytical solution temperature: unknown domain \n");
    }
  }
  double operator()(DIM(double x, double y, double z)) const{ // Returns the velocity field
    return T(DIM(x,y,z));
    }
  double dT_d(const unsigned char& dir,DIM(double x, double y, double z)){
    switch(dom){
    case LIQUID_DOMAIN:
      switch(dir){
      case dir::x:
        return cos(x)*sin(y)*(x + cos(t)*cos(x)*cos(y)) - sin(x)*sin(y)*(cos(t)*cos(y)*sin(x) - 1.);
      case dir::y:
        return cos(y)*sin(x)*(x + cos(t)*cos(x)*cos(y)) - cos(t)*cos(x)*sin(x)*SQR(sin(y));

      default:
        throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
      }
    case SOLID_DOMAIN:
      switch(dir){
      case dir::x:
        return cos(t)*SQR(cos(x))*cos(y)*sin(y) - cos(y)*sin(x)*(cos(t)*sin(x)*sin(y) - 1.);
      case dir::y:
        return cos(t)*cos(x)*SQR(cos(y))*sin(x) - cos(x)*sin(y)*(cos(t)*sin(x)*sin(y) - 1.);

      default:
        throw std::runtime_error("dT_dd of analytical temperature field: unrecognized Cartesian direction \n");
      }
    default:
      throw std::runtime_error("dT_dd of analytical temperature field: unrecognized domain \n");

    }
  }

  double dT_dt(DIM(double x, double y, double z)){
    switch(dom){
    case LIQUID_DOMAIN:
      return -cos(x)*cos(y)*sin(t)*sin(x)*sin(y);
    case SOLID_DOMAIN:
      return -cos(x)*cos(y)*sin(t)*sin(x)*sin(y);

    default:
      throw std::runtime_error("dT_dt in analytical temperature: unrecognized domain \n");
    }
  }

  double laplace(DIM(double x, double y, double z)){
    switch(dom){
    case LIQUID_DOMAIN:
      return -2.*sin(y)*(x*sin(x) - cos(x) + 4.*cos(t)*cos(x)*cos(y)*sin(x));

//      return -2.*SQR(n)*cos(n*x)*sin(n*y)*(m*cos(m*y)*sin(m*x)*cos(t) - 1.) -
//          2.*pow(n,3.)*sin(n*x)*sin(n*y)*(x + cos(m*x)*cos(m*y)*cos(t)) -
//          2.*m*SQR(n)*cos(m*x)*cos(n*y)*sin(m*y)*sin(n*x)*cos(t) -
//          2.*SQR(m)*n*cos(m*x)*cos(m*y)*sin(n*x)*sin(n*y)*cos(t);


//      return -2.*sin(y)*(x*sin(x) - cos(x) + 4.*cos(t)*cos(x)*cos(y)*sin(x));
    case SOLID_DOMAIN:

//      return -1.0*2.*SQR(m)*n*cos(m*x)*cos(m*y)*(sin(n*x)*sin(n*y)*cos(t) - 1.) -
//          2.*pow(n,3.)*cos(m*x)*cos(m*y)*sin(n*x)*sin(n*y)*cos(t) -
//          2.*m*SQR(n)*cos(m*x)*cos(n*y)*sin(m*y)*sin(n*x)*cos(t) -
//          2.*m*SQR(n)*cos(m*y)*cos(n*x)*sin(m*x)*sin(n*y)*cos(t);

      return -2.*cos(x)*cos(y)*(4.*cos(t)*sin(x)*sin(y) - 1.);
    default:
      throw std::runtime_error("laplace for analytical temperature field: unrecognized domain \n");
    }
  }
};

struct interfacial_velocity : CF_DIM{ // will yield analytical solution to interfacial velocity in a given cartesian direction (not including the multiplication by the normal, which will have to be done outside of this struct)

public:

  const unsigned char dir;
  temperature_field** temperature_;


  interfacial_velocity(const unsigned char &dir_,temperature_field** analytical_soln):dir(dir_),temperature_(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const{
    return (temperature_[SOLID_DOMAIN]->dT_d(dir,x,y) - temperature_[LIQUID_DOMAIN]->dT_d(dir,x,y))*coupled_test_sign;

  }
};

struct external_heat_source: CF_DIM{
  const unsigned char dom;
  temperature_field** temperature_;
  velocity_component** velocity_;

  external_heat_source(const unsigned char &dom_,temperature_field** analytical_T,velocity_component** analytical_v):dom(dom_),temperature_(analytical_T),velocity_(analytical_v){
    P4EST_ASSERT(dom>=0 && dom<=1);
  }

  double operator()(DIM(double x, double y, double z)) const {
    double advective_term;
    switch(dom){
    case LIQUID_DOMAIN:
      advective_term= (*velocity_[dir::x])(DIM(x,y,z))*temperature_[LIQUID_DOMAIN]->dT_d(dir::x,x,y) + (*velocity_[dir::y])(DIM(x,y,z))*temperature_[LIQUID_DOMAIN]->dT_d(dir::y,x,y);
      break;
    case SOLID_DOMAIN:
      advective_term= 0.;
      break;
    default:
      throw std::runtime_error("external heat source : advective term : unrecognized domain \n");
    }

    return temperature_[dom]->dT_dt(DIM(x,y,z)) + advective_term - temperature_[dom]->laplace(DIM(x,y,z));
  }
};




// --------------------------------------------------------------------------------------------------------------
// Level set functions:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch (example_){
      case FRANK_SPHERE:
        return s0 - sqrt(SQR(x) + SQR(y));
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case NS_GIBOU_EXAMPLE:
        return r0 - sin(x)*sin(y);
      case COUPLED_PROBLEM_EXAMPLE:
        return r0 - sqrt(SQR(x - x0_lsf) + SQR(y - y0_lsf));
      default: throw std::invalid_argument("You must choose an example type\n");
      }
  }
} level_set;


// This one is for the inner cylinder in example 2
struct MINI_LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("This option may not be used for the particular example being called");
      case ICE_AROUND_CYLINDER: return r_cyl - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case NS_GIBOU_EXAMPLE: throw std::invalid_argument("This option may not be used for the particular example being called");
      }
  }
} mini_level_set;
// Function for ramping the boundary conditions:
double ramp_BC(double initial,double goal_value){
  if(tn<t_ramp){
    return initial + ((goal_value - initial)/(t_ramp - tstart))*(tn - tstart);
    }
  else {
      return goal_value;
    }

}

// --------------------------------------------------------------------------------------------------------------
// INTERFACIAL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_temp;
void interface_bc(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE:
      interface_bc_type_temp = DIRICHLET;
      break;
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER:
      interface_bc_type_temp = DIRICHLET; 
      break;
    case COUPLED_PROBLEM_EXAMPLE:
      interface_bc_type_temp = DIRICHLET;
    }
}

BoundaryConditionType inner_interface_bc_type_temp;
void inner_interface_bc(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("This option may not be used for the particular example being called");
    case ICE_AROUND_CYLINDER:
      inner_interface_bc_type_temp = DIRICHLET;
      break;
    }
}

class BC_INTERFACE_VALUE_TEMP: public CF_DIM{ // TO CHECK -- changed how interp is initialized
private:
  // Have interpolation objects for case with surface tension included in boundary condition: can interpolate the curvature in a timestep to the interface points while applying the boundary condition
  my_p4est_node_neighbors_t* ngbd;
  my_p4est_interpolation_nodes_t* kappa_interp;
  temperature_field** temperature_;
  unsigned const char dom;

public:
  BC_INTERFACE_VALUE_TEMP(my_p4est_node_neighbors_t *ngbd_=NULL,Vec kappa = NULL, temperature_field** analytical_T=NULL, unsigned const char& dom_=NULL): ngbd(ngbd_),temperature_(analytical_T),dom(dom_)
  {
    if(ngbd!=NULL){
      kappa_interp = new my_p4est_interpolation_nodes_t(ngbd);
      kappa_interp->set_input(kappa,linear);
    }
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:{ // Frank sphere case, no surface tension
          return Tinterface; // TO-DO : CHANGE THIS TO ANALYTICAL SOLN
        }
      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder, with surface tension -- MAY ADD COMPLEXITY TO THIS LATER ON
        if(ramp_bcs){
            return ramp_BC(theta_wall,theta_interface*(1. - (sigma/d_cyl)*(*kappa_interp)(x,y)));
          }
        else return theta_interface*(1. - (sigma/d_cyl)*(*kappa_interp)(x,y));

      case COUPLED_PROBLEM_EXAMPLE:
        return temperature_[dom]->T(DIM(x,y,z));
    default:
      throw std::runtime_error("BC INTERFACE VALUE TEMP: unrecognized example \n");
      }
  }
  void clear(){
    kappa_interp->clear();
  };
  void set(my_p4est_node_neighbors_t *ngbd_,Vec kappa){
    if(ngbd_!=NULL){
      ngbd = ngbd_;
      kappa_interp = new my_p4est_interpolation_nodes_t(ngbd);
      kappa_interp->set_input(kappa,linear);
    }
  }
};

class BC_interface_coeff: public CF_DIM{
public:
  double operator()(double x, double y) const
  { switch(example_){
      case FRANK_SPHERE: return 1.0;
      case ICE_AROUND_CYLINDER: return 1.0;
      case COUPLED_PROBLEM_EXAMPLE:
        return 1.0;
      }
  }
}bc_interface_coeff;

class BC_interface_value_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER:
        if(ramp_bcs){
            return ramp_BC(theta_wall,theta_cyl);
          }
        else return theta_cyl;
      }
  }
}bc_interface_val_inner;

class BC_interface_coeff_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER:
        return 1.0;
      }
  }
}bc_interface_coeff_inner;

// --------------------------------------------------------------------------------------------------------------
// Wall functions -- these evaluate to true or false depending on if the location is on the wall --  they just add coding simplicity
// --------------------------------------------------------------------------------------------------------------
bool xlower_wall(DIM(double x, double y, doublze z)){
  // Front x wall, excluding the top and bottom corner points in y
  return ((fabs(x - xmin) <= EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS));
};
bool xupper_wall(DIM(double x, double y, double z)){
  // back x wall, excluding the top and bottom corner points in y
  return ((fabs(x - xmax) <= EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS));

}

bool ylower_wall(DIM(double x, double y, double z)){
  return (fabs(y - ymin) <= EPS);
}
bool yupper_wall(DIM(double x, double y, double z)){
  return (fabs(y - ymax) <= EPS);

}

// --------------------------------------------------------------------------------------------------------------
// WALL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
double temp_three_wall_dirichlet_val(DIM(double x, double y, double z)){
  if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
        return theta_wall;
      }
  else {
      return back_wall_temp_flux;
    }
}

double temp_three_wall_neumann_val(DIM(double x, double y, double z)){
  if (xupper_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
      return back_wall_temp_flux;}
  else {
      return theta_wall;
    }
}

BoundaryConditionType temp_three_wall_dirichlet_type(DIM(double x, double y, double z)){
  if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
      return DIRICHLET;}
  else {
      return NEUMANN;
    }
}

BoundaryConditionType temp_three_wall_neumann_type(DIM(double x, double y, double z)){
  if (xupper_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
      return NEUMANN;}
  else {
      return DIRICHLET;
    }
}

class BC_WALL_TYPE_TEMP: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE: return DIRICHLET;
      case ICE_AROUND_CYLINDER:
          return temp_three_wall_dirichlet_type(DIM(x,y,z));
      case COUPLED_PROBLEM_EXAMPLE:
        if(xlower_wall(x,y) || xupper_wall(x,y)){
            return DIRICHLET;
          }
        else if(ylower_wall(x,y) || yupper_wall(x,y)){
            return DIRICHLET;
          }
        break;
      default:
        throw std::runtime_error("WALL BC TYPE TEMP: unrecognized example \n");
        }
  }
}bc_wall_type_temp;


class BC_WALL_VALUE_TEMP: public CF_DIM
{
public:
  const unsigned char dom;
  temperature_field** temperature_;
  BC_WALL_VALUE_TEMP(const unsigned char& dom_, temperature_field** analytical_soln=NULL): dom(dom_),temperature_(analytical_soln){
    P4EST_ASSERT(dom>=0 && dom<=1);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:{
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z))){
            if (level_set(DIM(x,y,z)) < EPS){
                double r;
                double sval;
                r = sqrt(SQR(x) + SQR(y));
                sval = r/sqrt(tn + dt);
                return frank_sphere_solution_t(sval);
              }
            else{
                return Tinterface;
              }
          }
        break;       }
      case ICE_AROUND_CYLINDER:{
          return temp_three_wall_dirichlet_val(DIM(x,y,z));
        }
      case COUPLED_PROBLEM_EXAMPLE:{
          if(xlower_wall(x,y) || xupper_wall(x,y) || ylower_wall(x,y) || yupper_wall(x,y)){
              return temperature_[dom]->T(DIM(x,y,z));;
            }
          break;
        }
      default:
        throw std::runtime_error("WALL BC TYPE TEMP: unrecognized example \n");
      }
  }
};

// --------------------------------------------------------------------------------------------------------------
// TEMPERATURE INITIAL CONDITION
// --------------------------------------------------------------------------------------------------------------
class INITIAL_TEMP: public CF_DIM
{
public:
  const unsigned char dom;
  temperature_field** temperature_;
  INITIAL_TEMP(const unsigned char &dom_,temperature_field** analytical_T=NULL):dom(dom_),temperature_(analytical_T){}
  double operator() (DIM(double x, double y, double z)) const
  {
    double r;
    double sval;
    switch(example_){
      case FRANK_SPHERE:{
        r = sqrt(SQR(x) + SQR(y));
        sval = s(r,t);
        return frank_sphere_solution_t(sval); // Initial distribution is the analytical solution of Frank Sphere problem at t = tstart
      }
      case ICE_AROUND_CYLINDER:{
        switch(dom){
          case LIQUID_DOMAIN:{
            return theta_wall;
          }
          case SOLID_DOMAIN:{
            if(ramp_bcs){
              return theta_interface;
            }
            else{
              return theta_wall;
            }
          }
          default:{
            throw std::runtime_error("Initial condition for temperature: unrecognized domain \n");
          }
        }
      }
      case COUPLED_PROBLEM_EXAMPLE:{
          return temperature_[dom]->T(DIM(x,y,z));
      }
    }
  }
};

// --------------------------------------------------------------------------------------------------------------
// VELOCITY BOUNDARY CONDITION -- for velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
class BC_WALL_VALUE_VELOCITY: public CF_DIM
{
public:
  const unsigned char dir;
  velocity_component** velocity_field;

  BC_WALL_VALUE_VELOCITY(const unsigned char& dir_, velocity_component** analytical_soln=NULL):dir(dir_),velocity_field(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      //------------------------------------------------------------------
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");

      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            if(ramp_bcs){
              switch(dir){
              case dir::x:
                return ramp_BC(0.,u0);
              case dir::y:
                return ramp_BC(0.,v0);
              default:
                throw std::runtime_error("WALL BC VELOCITY: unrecognized Cartesian direction \n");
              }
            } // end of ramp BC case
            else{
              switch(dir){
              case dir::x:
                return u0;
              case dir::y:
                return v0;
              default:
                throw std::runtime_error("WALL BC VELOCITY: unrecognized Cartesian direction \n");
              }
            }
          }
        else if(xupper_wall(DIM(x,y,z))){ // Neumann condition on back wall
          switch(dir){
          case dir::x:
            return outflow_u;
          case dir::y:
            return outflow_v;
          default:
            throw std::runtime_error("WALL BC VELOCITY: unrecognized Cartesian direction \n");
          }
          }
        break;
        }
      case NS_GIBOU_EXAMPLE:{
        return (*velocity_field[dir])(DIM(x,y,z));
        }
      case COUPLED_PROBLEM_EXAMPLE:{
        return (*velocity_field[dir])(DIM(x,y,z));
        }
    default:
      throw std::runtime_error("WALL BC VALUE VELOCITY: unrecognized example \n");
    }
  }
};

class BC_WALL_TYPE_VELOCITY: public WallBCDIM
{
public:
  const unsigned char dir;
  BC_WALL_TYPE_VELOCITY(const unsigned char& dir_):dir(dir_){
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET; // free stream
          }
        else if (xupper_wall(DIM(x,y,z))){
            return NEUMANN; // presribed outflow
          }
        break;
        }
      case NS_GIBOU_EXAMPLE:
        return DIRICHLET;
      case COUPLED_PROBLEM_EXAMPLE:{
        return DIRICHLET;
        }
      default:
        throw std::runtime_error("WALL BC TYPE VELOCITY: unrecognized example \n");
      }
  }
};

// --------------------------------------------------------------------------------------------------------------
// VELOCITY INTERFACIAL CONDITION -- for velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
// Interfacial condition for the u component:
BoundaryConditionType interface_bc_type_velocity[P4EST_DIM];
void BC_INTERFACE_TYPE_VELOCITY(const unsigned char& dir){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER:
      interface_bc_type_velocity[dir] = DIRICHLET;
      break;
    case NS_GIBOU_EXAMPLE:
      interface_bc_type_velocity[dir] = DIRICHLET;
      break;
    case COUPLED_PROBLEM_EXAMPLE:{
      interface_bc_type_velocity[dir] = DIRICHLET;
      break;
      }
    }
}


// Interfacial condition:
class BC_interface_value_velocity: public CF_DIM{
private:
  my_p4est_node_neighbors_t* ngbd;
  my_p4est_interpolation_nodes_t* v_interface_interp;

public:
  const unsigned char dir;
  velocity_component** velocity_field;
  BC_interface_value_velocity(const unsigned char& dir_, my_p4est_node_neighbors_t* ngbd_=NULL, Vec v_interface=NULL,velocity_component** analyical_soln=NULL): ngbd(ngbd_),dir(dir_),velocity_field(analyical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
    if((ngbd_!=NULL) && (v_interface!=NULL)){
      v_interface_interp = new my_p4est_interpolation_nodes_t(ngbd);
      v_interface_interp->set_input(v_interface,linear);
    }
  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:{ // Ice solidifying around a cylinder
          if(!solve_stefan) return 0.;
          else{
            return (*v_interface_interp)(x,y); // No slip on the interface
          }
      }
      case NS_GIBOU_EXAMPLE:
        return (*velocity_field[dir])(x,y);
      case COUPLED_PROBLEM_EXAMPLE:
        return (*velocity_field[dir])(x,y);
    default:
      throw std::runtime_error("BC INTERFACE VALUE VELOCITY: unrecognized example ");
      }
  }
  void clear(){
    v_interface_interp->clear();
  }
  void set(my_p4est_node_neighbors_t* ngbd_, Vec v_interface){
    if((ngbd_!=NULL) && (v_interface!=NULL)){
      ngbd = ngbd_;
      v_interface_interp = new my_p4est_interpolation_nodes_t(ngbd);
      v_interface_interp->set_input(v_interface,linear);
    }
  }
};

// --------------------------------------------------------------------------------------------------------------
// VELOCITY INITIAL CONDITION -- for velocity vector = (u,v,w), in Navier-Stokes problem
// --------------------------------------------------------------------------------------------------------------
struct INITIAL_VELOCITY : CF_DIM
{
  const unsigned char dir;
  velocity_component** velocity_field;

  INITIAL_VELOCITY(const unsigned char& dir_,velocity_component** analytical_soln=NULL):dir(dir_), velocity_field(analytical_soln){
    P4EST_ASSERT(dir<P4EST_DIM);
  }

  double operator() (DIM(double x, double y,double z)) const{
    switch(example_){
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:
        if(ramp_bcs) return 0.;
        else{
          switch(dir){
          case dir::x:
            return u0;
          case dir::y:
            return v0;
          default:
            throw std::runtime_error("Vel_initial error: unrecognized cartesian direction \n");
          }
        }
      case COUPLED_PROBLEM_EXAMPLE:
        switch(dir){
        case dir::x:
          return (*velocity_field[0])(x,y);
        case dir::y:
          return (*velocity_field[1])(x,y);
        default:
          throw std::runtime_error("Vel_initial error: unrecognized cartesian direction \n");
        }
      case NS_GIBOU_EXAMPLE:
        switch(dir){
        case dir::x:
          return (*velocity_field[0])(x,y);
        case dir::y:
          return (*velocity_field[1])(x,y);
        default:
          throw std::runtime_error("Vel_initial error: unrecognized cartesian direction \n");
        }
      default:
        throw std::runtime_error("vel initial: unrecognized example_ being run \n");
      }
  }
} ;

// --------------------------------------------------------------------------------------------------------------
// PRESSURE BOUNDARY CONDITIONS
// --------------------------------------------------------------------------------------------------------------
class BC_WALL_TYPE_PRESSURE: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes solution is not "
                                                     "compatible with this example, please choose another \n");
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER:{
          if(fabs(x - xmax)<EPS){
              return DIRICHLET;
            }
          else{
              return NEUMANN;
            }
        }
      case NS_GIBOU_EXAMPLE: {
            return NEUMANN;
        }
      case COUPLED_PROBLEM_EXAMPLE:{
            return NEUMANN;
//        return DIRICHLET;
        }
      default:
        throw std::runtime_error("WALL BC TYPE PRESSURE: unrecognized example \n");
        }
  }
}/*bc_wall_type_pressure*/;

class BC_WALL_VALUE_PRESSURE: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not "
                                    "compatible with this example, please choose another \n");
      case FLOW_PAST_CYLINDER:

      case ICE_AROUND_CYLINDER:{ // coupled problem
        return 0.0;}

      case NS_GIBOU_EXAMPLE: {// benchmark NS case
        return 0.0;}

      case COUPLED_PROBLEM_EXAMPLE:{
            return 0.0;
        }
      default:
        throw std::runtime_error("WALL BC VALUE PRESSURE: unrecognized example \n");
        }
  }

} /*bc_wall_value_pressure*/;

static BoundaryConditionType interface_bc_type_pressure;
void interface_bc_pressure(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");

    case FLOW_PAST_CYLINDER:
    case ICE_AROUND_CYLINDER:
      interface_bc_type_pressure = NEUMANN;
      break;
    case NS_GIBOU_EXAMPLE:
      interface_bc_type_pressure = NEUMANN;
      break;
    case COUPLED_PROBLEM_EXAMPLE:
      interface_bc_type_pressure = NEUMANN;
//    interface_bc_type_pressure = DIRICHLET;

      break;
    }
}

class BC_INTERFACE_VALUE_PRESSURE: public CF_DIM{
public:
  double operator()(DIM(double x, double y,double z)) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case FLOW_PAST_CYLINDER:
      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder
        return 0.0;
      case NS_GIBOU_EXAMPLE: // Benchmark NS
        return 0.0;
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
      default:
        throw std::runtime_error("INTERFACE BC VAL PRESSURE: unrecognized example \n");
      }
  }
} /*bc_interface_value_pressure*/;

// --------------------------------------------------------------------------------------------------------------
// Functions for checking the values of interest during the solution process
// --------------------------------------------------------------------------------------------------------------
void check_T_values(vec_and_ptr_t phi, vec_and_ptr_t T, p4est_nodes* nodes, p4est_t* p4est, int example, vec_and_ptr_t phi_cyl,bool check_for_reasonable_values, bool update_Tl_values, bool update_Ts_values) {
  T.get_array();
  phi.get_array();
  if (example_ == ICE_AROUND_CYLINDER) phi_cyl.get_array();

  double avg_T = 0.0;
  double max_T = 0.;
  double min_T = 1.e10;
  double min_mag_T = 1.e10;

  int pts_avg = 0;
  bool in_domain;

  // Make sure phi_cyl is defined if we are running an example where it is required:
  if(example_ ==ICE_AROUND_CYLINDER && phi_cyl.ptr == NULL){
      throw std::invalid_argument("You must provide a phi_cylinder vector to run the ice cylnder example \n");
    }

  // Loop over each node, check if node is in the subdomain we are considering. If so, compute average,max, and min values for the domain
  foreach_local_node(n,nodes){
    in_domain = false;
    // Check if the node is in the domain we are checking:
    if (example_ ==ICE_AROUND_CYLINDER ){
        ((phi.ptr[n] < 0.) && (phi_cyl.ptr[n] < 0.)) ? in_domain = true : in_domain = false;
      }
    else{
        (phi.ptr[n] < 0.) ? in_domain = true : in_domain = false;
      }
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

  PetscPrintf(p4est->mpicomm,"\n"
                             "Average: %0.4f  "
                             "Max: %0.4f  "
                             "Min: %0.4f  "
                             "Min magnitude: %0.4f \n",global_avg_T,global_max_T,global_min_T,global_min_mag_T);
  if(update_Tl_values) {T_l_max = global_max_T; T_l_min = global_min_T;}
  if(update_Ts_values){T_s_max = global_max_T; T_s_min = global_min_T;}

  T.restore_array();
  phi.restore_array();
  if(example_ ==ICE_AROUND_CYLINDER) phi_cyl.restore_array();

}

void check_T_d_values(vec_and_ptr_t phi, vec_and_ptr_dim_t dT, p4est_nodes* nodes, p4est_t* p4est, bool get_location){
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

void check_vel_values(vec_and_ptr_t phi, vec_and_ptr_dim_t vel, p4est_nodes* nodes, p4est_t* p4est, bool get_location,double dxyz_close_to_interface){
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
void save_stefan_test_case(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t phi, vec_and_ptr_dim_t v_interface,  double dxyz_close_to_interface,bool are_we_saving_vtk,char* filename_vtk,char *name, FILE *fich){
  PetscErrorCode ierr;

  vec_and_ptr_t T_ana,phi_ana, v_interface_ana;
  T_ana.create(p4est,nodes); phi_ana.create(p4est,nodes);v_interface_ana.create(p4est,nodes);

  vec_and_ptr_t T_l_err,T_s_err,phi_err,v_interface_err;
  T_l_err.create(p4est,nodes); T_s_err.create(p4est,nodes); phi_err.create(p4est,nodes); v_interface_err.create(p4est,nodes);

  T_l.get_array(); T_s.get_array();
  phi.get_array(); v_interface.get_array();

  T_ana.get_array(); v_interface_ana.get_array();phi_ana.get_array();
  T_l_err.get_array();T_s_err.get_array();v_interface_err.get_array();phi_err.get_array();

  double Linf_Tl = 0.0;
  double Linf_Ts = 0.0;
  double Linf_phi = 0.0;
  double Linf_v_int = 0.0;

  double r;
  double sval;
  double vel;

  // NOTE: We have just solved for time (n+1), so we compare computed solution to the analytical solution at time = (tn + dt)
//  double v_exact = s0/(2.0*sqrt(tn/*+dt*/));

  //PetscPrintf(mpi.comm(),"Exact solution of velocity is: %0.2f",v_exact);
  // Now loop through nodes to compare errors between LSF and Temperature profiles:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes){
    node_xyz_fr_n(n,p4est,nodes,xyz);

    r = sqrt(SQR(xyz[0]) + SQR(xyz[1]));
    sval = r/sqrt(tn+dt);

    phi_ana.ptr[n] = s0*sqrt(tn+dt) - r;

    T_ana.ptr[n] = frank_sphere_solution_t(sval);

    v_interface_ana.ptr[n] = s0/(2.0*sqrt(tn+dt));

    // Error on phi and v_int:
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){

      // Errors on phi:
      phi_err.ptr[n] = fabs(phi.ptr[n] - phi_ana.ptr[n]);

      Linf_phi = max(Linf_phi,phi_err.ptr[n]); // CHECK THIS -- NOT ENTIRELY SURE THIS IS CORRECT

      // Errors on v_int:
      vel = sqrt(SQR(v_interface.ptr[0][n])+ SQR(v_interface.ptr[1][n]));
      v_interface_err.ptr[n] = fabs(vel - v_interface_ana.ptr[n]);
      Linf_v_int = max(Linf_v_int,v_interface_err.ptr[n]);
      }

    // Check error in the negative subdomain (T_liquid) (Domain = Omega_minus)
    if(phi.ptr[n]<0.){
        T_l_err.ptr[n]  = fabs(T_l.ptr[n] - T_ana.ptr[n]);
        Linf_Tl = max(Linf_Tl,T_l_err.ptr[n]);
      }
    if (phi.ptr[n]>0.){
        T_s_err.ptr[n]  = fabs(T_s.ptr[n] - T_ana.ptr[n]);
        Linf_Ts = max(Linf_Ts,T_s_err.ptr[n]);
      }
  }

  double global_Linf_errors[] = {0.0, 0.0, 0.0,0.0,0.0};
  double local_Linf_errors[] = {Linf_phi,Linf_Tl,Linf_Ts,Linf_v_int};


  // Now get the global maximum errors:
  int mpiret = MPI_Allreduce(local_Linf_errors,global_Linf_errors,4,MPI_DOUBLE,MPI_MAX,p4est->mpicomm); SC_CHECK_MPI(mpiret);


  // Print Errors to application output:
  PetscPrintf(p4est->mpicomm,"\n----------------\n Errors on frank sphere: \n --------------- \n");
  PetscPrintf(p4est->mpicomm,"dxyz close to interface: %0.2f \n",dxyz_close_to_interface);

  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"Number of grid points used: %d \n \n", num_nodes);


  PetscPrintf(p4est->mpicomm," Linf on phi: %0.4f \n Linf on T_l: %0.4f \n Linf on T_s: %0.4f \n Linf on v_int: %0.4f \n", global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3]);

  // Print errors to file:
  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  PetscFPrintf(p4est->mpicomm,fich,"%e %e %d %e %e %e %e %d %e \n",tn + dt,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],num_nodes,dxyz_close_to_interface);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);


  // If we are saving this timestep, output the results to vtk:
  if(are_we_saving_vtk){
      std::vector<std::string> point_names;
      std::vector<double*> point_data;

      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","phi_ana","T_ana","v_interface_vec_ana","phi_err","T_l_err","T_s_err","v_interface_vec_err"};
      point_data = {phi.ptr,T_l.ptr,T_s.ptr,v_interface.ptr[0],v_interface.ptr[1],phi_ana.ptr,T_ana.ptr,v_interface_ana.ptr,phi_err.ptr,T_l_err.ptr,T_s_err.ptr,v_interface_err.ptr};


      std::vector<std::string> cell_names = {};
      std::vector<double*> cell_data = {};

      //my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);
      my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);

    }


  T_l.restore_array();
  T_s.restore_array();
  phi.restore_array();
  v_interface.restore_array();

  T_ana.restore_array();v_interface_ana.restore_array();phi_ana.restore_array();
  T_l_err.restore_array();T_s_err.restore_array();v_interface_err.restore_array();phi_err.restore_array();

  T_ana.destroy();v_interface_ana.destroy();phi_ana.destroy();
  T_l_err.destroy();T_s_err.destroy();v_interface_err.destroy();phi_err.destroy();
}

//void check_coupled_problem_error(vec_and_ptr_t phi,vec_and_ptr_dim_t v_n, vec_and_ptr_t p, vec_and_ptr_t Tl, p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd, double dxyz_close_to_interface, char *name, FILE *fich, int tstep){
//  PetscErrorCode ierr;

//  double u_error = 0.0;
//  double v_error = 0.0;
//  double P_error = 0.0;
//  double T_error = 0.0;

//  double L_inf_u = 0.0;
//  double L_inf_v = 0.0;
//  double L_inf_P = 0.0;
//  double L_inf_T = 0.0;


//  // Get arrays:
//  v_n.get_array();
//  p.get_array();
//  phi.get_array();
//  Tl.get_array();

//  // Get local errors in negative subdomain:
//  double xyz[P4EST_DIM];
//  double x;
//  double y;
//  foreach_local_node(n,nodes){
//    if(phi.ptr[n] < 0.){
//        node_xyz_fr_n(n,p4est,nodes,xyz);

//        x = xyz[0]; y = xyz[1];

//        u_error = fabs(v_n.ptr[0][n] - u_ana_tn(x,y));
//        v_error = fabs(v_n.ptr[1][n] - v_ana_tn(x,y));
//        P_error = fabs(p.ptr[n] - p_ana_tn(x,y));
//        T_error = fabs(Tl.ptr[n] - T_ana_tn(x,y));

//        L_inf_u = max(L_inf_u,u_error);
//        L_inf_v = max(L_inf_v,v_error);
//        L_inf_P = max(L_inf_P,P_error);
//        L_inf_T = max(L_inf_T,T_error);

//      }
//  }

////  // Loop over each quadrant in each tree, check the error in hodge
////  double xyz_c[P4EST_DIM];
////  double x_c; double y_c;
////  my_p4est_interpolation_nodes_t interp_phi(ngbd);
////  interp_phi.set_input(phi.vec,linear);
////  foreach_tree(tr,p4est){
////    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tr);

////    foreach_local_quad(q,tree){
////      // Get the global index of the quadrant:
////      p4est_locidx_t quad_idx = tree->quadrants_offset + q;

////      // Get xyz location of the quad center so we can interpolate phi there and check which domain we are in:
////      quad_xyz_fr_q(quad_idx,tr,p4est,ghost,xyz_c);
////      x_c = xyz_c[0]; y_c = xyz_c[1];

////      // Get the error in the negative subdomain:
////      if(interp_phi(x_c,y_c) < 0){
////          P_error = fabs(p.ptr[quad_idx] - p_analytical(x_c,y_c));
////        }

////    }
////  }


//  // NEED TO GRAB PRESSURE ERROR AT QUADS, NEED TO CHANGE PRESSURE TO VEC_AND_PTR_CELLS
//  // Restore arrays
//  v_n.restore_array();
//  p.restore_array();
//  phi.restore_array();
//  Tl.restore_array();

//  // Get the global errors:
//  double local_Linf_errors[4] = {L_inf_u,L_inf_v,L_inf_P, L_inf_T};
//  double global_Linf_errors[4] = {0.0,0.0,0.0,0.0};

//  int mpi_err;

//  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,4,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

//  // Print errors to application output:
//  int num_nodes = nodes->indep_nodes.elem_count;
//  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n "
//                             "Errors on Coupled Problem Example "
//                             "\n -------------------------------------\n "
//                             "Linf on u: %0.4e \n"
//                             "Linf on v: %0.4e \n"
//                             "Linf on P: %0.4e \n"
//                             "Linf on Tl: %0.4e \n"
//                             "Number grid points used: %d \n"
//                             "dxyz close to interface : %0.4f \n",
//                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],
//                              num_nodes,dxyz_close_to_interface);



//  // Print errors to file:

//  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
//  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
//  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);



//}

void check_ice_cylinder_v_and_radius(vec_and_ptr_t phi,p4est_t* p4est,p4est_nodes_t* nodes,double dxyz_close,char *name,FILE *fich){

  std::vector<double> theta;
  std::vector<double> delta_r;

  phi.get_array();
  double xyz[P4EST_DIM];
  double max_theta =-200.0;
  foreach_local_node(n,nodes){
    if(fabs(phi.ptr[n])<0.5*dxyz_close){
        node_xyz_fr_n(n,p4est,nodes,xyz);
        // Note: we are using coordinates shifted such that the origin is defined as the center of the cylinder

        // Note: this x and y must be defined in the same way as it is in the Level set function
        double x = xyz[0] - (xmax)/4.0;
        double y = xyz[1] - (ymax)/2.0;


        double r = sqrt(SQR(x) + SQR(y));
        double Theta = atan2(y,x);


        double dr = r - r_cyl;

//        printf("(x,y) = (%0.3f, %0.3f), (xnew,ynew) = (%0.3f, %0.3f), r = %.2f, dr = %.2f, theta = %.2f \n",xyz[0],xyz[1],x,y,r,dr,Theta*180./PI);


        delta_r.push_back(dr);
        theta.push_back(Theta);
        max_theta = max(max_theta,Theta);
      }
  }

  phi.restore_array();
  PetscErrorCode ierr;

  unsigned long local_theta_size = 0.0;
  local_theta_size+=theta.size();
  unsigned long global_theta_size = 0;
  MPI_Allreduce(&local_theta_size,&global_theta_size,1,MPI_INT,MPI_SUM,p4est->mpicomm);

  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"\n%0.4e %0.4e %d ",tn, v_interface_max_norm,global_theta_size);CHKERRXX(ierr);

  for (unsigned long i = 0; i<theta.size();i++){
      ierr = PetscSynchronizedFPrintf(p4est->mpicomm,fich,"%0.4e %0.4e ",theta[i],delta_r[i]);CHKERRXX(ierr);
    }

  ierr= PetscSynchronizedFlush(p4est->mpicomm,fich);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);

//  theta.clear();
//  delta_r.clear();
//  theta.shrink_to_fit();
//  delta_r.shrink_to_fit();
  std::vector<double>().swap(delta_r);
  std::vector<double>().swap(theta);


}
// --------------------------------------------------------------------------------------------------------------
// FUNCTIONS FOR SOLVING THE PROBLEM:
// --------------------------------------------------------------------------------------------------------------
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
  if (!global_still_solid_present){ // If no more solid, then ice has melted
      PetscPrintf(p4est->mpicomm,"\n \n Ice has entirely melted as of t = %0.3e \n \n ",time);
    }
return global_still_solid_present;
}

void setup_rhs(vec_and_ptr_t phi,vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t rhs_Tl, vec_and_ptr_t rhs_Ts,vec_and_ptr_t T_l_backtrace, vec_and_ptr_t T_l_backtrace_nm1, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd, external_heat_source** external_heat_source_term=NULL){

  // In building RHS, if we are doing advection, we have two options:
  // (1) 1st order -- approx is (dT/dt + u dot grad(T)) ~ (T(n+1) - Td(n))/dt --> so we add Td/dt to the RHS
  // (2) 2nd order -- approx is (dT/dt + u dot grad(T)) ~ alpha*(T(n+1) - Td(n))/dt + beta*(Td(n) - Td(n-1))/dt_nm1
  //                       --> so we add Td(n)*(alpha/dt - beta/dt_nm1) + Td(n-1)*(beta/dt_nm1) to the RHS
  //               -- where alpha and beta are weights of the two timesteps
  // See Semi-Lagrangian backtrace advection schemes for more details

  // If we are not doing advection, then we have:
  // (1) dT/dt = (T(n+1) - T(n)/dt) --> which is a backward euler 1st order approximation (since the RHS is discretized spatially at T(n+1))
  // (2) dT/dt = alpha*laplace(T) ~ (T(n+1) - T(n)/dt) = (1/2)*(laplace(T(n)) + laplace(T(n+1)) )  ,
  //                              in which case we need the second derivatives of the temperature field at time n


  // Establish forcing terms if applicable:
  vec_and_ptr_t forcing_term_liquid;
  vec_and_ptr_t forcing_term_solid;

  if(example_ == COUPLED_PROBLEM_EXAMPLE){
    forcing_term_liquid.create(p4est,nodes);
    forcing_term_solid.create(p4est,nodes);

    sample_cf_on_nodes(p4est,nodes,*external_heat_source_term[LIQUID_DOMAIN],forcing_term_liquid.vec);
    sample_cf_on_nodes(p4est,nodes,*external_heat_source_term[SOLID_DOMAIN],forcing_term_solid.vec);
  }

  // Get derivatives of temperature fields if we are using Crank Nicholson:
  vec_and_ptr_dim_t T_l_dd;
  vec_and_ptr_dim_t T_s_dd;
  if(method_ ==2){
      T_s_dd.create(p4est,nodes);
      ngbd->second_derivatives_central(T_s.vec,T_s_dd.vec[0],T_s_dd.vec[1]);
      T_s_dd.get_array();
      if(!do_advection) {
          T_l_dd.create(p4est,nodes);
          ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec[0],T_l_dd.vec[1]);
          T_l_dd.get_array();
        }
    }

  // Prep coefficients if we are doing 2nd order advection:
  if(do_advection && advection_sl_order==2){
      advection_alpha_coeff = (2.*dt + dt_nm1)/(dt + dt_nm1);
      advection_beta_coeff = (-1.*dt)/(dt + dt_nm1);
    }

  // Get arrays:
  // Get Ts arrays:
  T_s.get_array();
  rhs_Ts.get_array();

  // Get Tl arrays:
  rhs_Tl.get_array();
  if(do_advection){
      T_l_backtrace.get_array();
      if(advection_sl_order ==2) T_l_backtrace_nm1.get_array();
    }
  else{
      T_l.get_array();
    }

  if(example_ == COUPLED_PROBLEM_EXAMPLE){
    forcing_term_liquid.get_array();
    forcing_term_solid.get_array();
  }

  phi.get_array();
  foreach_local_node(n,nodes){
    // First, assemble system for Ts depending on case:
    if(method_ == 2){ // Crank Nicholson
        rhs_Ts.ptr[n] = 2.*T_s.ptr[n]/dt + alpha_s*(T_s_dd.ptr[0][n] + T_s_dd.ptr[1][n]);
      }
    else{ // Backward Euler
        rhs_Ts.ptr[n] = T_s.ptr[n]/dt;
      }

    // Now for Tl depending on case:
    if(do_advection){
      if(advection_sl_order ==2){
        rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]*((advection_alpha_coeff/dt) - (advection_beta_coeff/dt_nm1)) + T_l_backtrace_nm1.ptr[n]*(advection_beta_coeff/dt_nm1);
        }
      else{
        rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]/dt;
        }
     }
    else{
      if(method_ ==2){//Crank Nicholson
        rhs_Tl.ptr[n] = 2.*T_l.ptr[n]/dt + alpha_l*(T_l_dd.ptr[0][n] + T_l_dd.ptr[1][n]);
        }
      else{ // Backward Euler
        rhs_Tl.ptr[n] = T_l.ptr[n]/dt;
        }
      }
    if(example_ == COUPLED_PROBLEM_EXAMPLE){
      // Add forcing terms:
      rhs_Tl.ptr[n]+=forcing_term_liquid.ptr[n];
      rhs_Ts.ptr[n]+=forcing_term_solid.ptr[n];
    }

  }// end of loop over nodes

  // Restore arrays:
  phi.restore_array();

  T_s.restore_array();
  rhs_Ts.restore_array();

  rhs_Tl.restore_array();
  if(do_advection){
      T_l_backtrace.restore_array();
      if(advection_sl_order==2) T_l_backtrace_nm1.restore_array();
    }
  else{
      T_l.restore_array();
    }
  if(method_ ==2){
      T_s_dd.restore_array();
      T_s_dd.destroy();
      if(!do_advection){
          T_l_dd.restore_array();
          T_l_dd.destroy();
        }
    }

  if(example_ == COUPLED_PROBLEM_EXAMPLE){
    forcing_term_liquid.restore_array();
    forcing_term_solid.restore_array();

    // Destroy these if they were created
    forcing_term_liquid.destroy();
    forcing_term_solid.destroy();
  }
}

void do_backtrace(vec_and_ptr_t T_l,vec_and_ptr_t T_l_nm1,
                  vec_and_ptr_t T_l_backtrace,vec_and_ptr_t T_l_backtrace_nm1,
                  vec_and_ptr_dim_t v, vec_and_ptr_dim_t v_nm1,
                  p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t* ngbd,
                  p4est_t* p4est_nm1, p4est_nodes_t* nodes_nm1, my_p4est_node_neighbors_t* ngbd_nm1){
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Beginning to do backtrace \n");
  PetscErrorCode ierr;
  // Initialize objects we will use in this function:
  // PETSC Vectors for second derivatives
  vec_and_ptr_dim_t T_l_dd, T_l_dd_nm1;
  Vec v_dd[P4EST_DIM][P4EST_DIM];
  Vec v_dd_nm1[P4EST_DIM][P4EST_DIM];

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];
  vector <double> xyz_d_nm1[P4EST_DIM];

  // Interpolators
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd); /*= NULL;*/
//  SL_backtrace_interp = new my_p4est_interpolation_nodes_t(ngbd);

  my_p4est_interpolation_nodes_t SL_backtrace_interp_nm1(ngbd_nm1);/* = NULL;*/
//  SL_backtrace_interp_nm1 = new my_p4est_interpolation_nodes_t(ngbd_nm1);

  // Get the relevant second derivatives
  T_l_dd.create(p4est,nodes);
  ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec);

  if(advection_sl_order==2) {
      T_l_dd_nm1.create(p4est_nm1,nodes_nm1);
      ngbd_nm1->second_derivatives_central(T_l_nm1.vec,T_l_dd_nm1.vec);
    }

  foreach_dimension(d){
    foreach_dimension(dd){
      ierr = VecCreateGhostNodes(p4est, nodes, &v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2){
          ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &v_dd_nm1[d][dd]); CHKERRXX(ierr);
        }
    }
  }

  // v_dd[k] is the second derivative of the velocity components n along cartesian direction k
  // v_dd_nm1[k] is the second derivative of the velocity components nm1 along cartesian direction k

  ngbd->second_derivatives_central(v.vec,v_dd[0],v_dd[1],P4EST_DIM);
  if(advection_sl_order ==2){
      ngbd_nm1->second_derivatives_central(v_nm1.vec, DIM(v_dd_nm1[0], v_dd_nm1[1], v_dd_nm1[2]), P4EST_DIM);
    }

  // Do the Semi-Lagrangian backtrace:
  if(advection_sl_order ==2){
      trajectory_from_np1_to_nm1(p4est,nodes,ngbd_nm1,ngbd,v_nm1.vec,v_dd_nm1,v.vec,v_dd,dt_nm1,dt,xyz_d_nm1,xyz_d);
      if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Completes backtrace trajectory \n");
    }
  else{
      trajectory_from_np1_to_n(p4est,nodes,ngbd,dt,v.vec,v_dd,xyz_d);
    }

  // Add backtrace points to the interpolator(s):
  foreach_local_node(n,nodes){
    double xyz_temp[P4EST_DIM];
    double xyz_temp_nm1[P4EST_DIM];

    foreach_dimension(d){
      xyz_temp[d] = xyz_d[d][n];

      if(advection_sl_order ==2){
          xyz_temp_nm1[d] = xyz_d_nm1[d][n];
        }
    } // end of "for each dimension"

    SL_backtrace_interp.add_point(n,xyz_temp);
    if(advection_sl_order ==2 ) SL_backtrace_interp_nm1.add_point(n,xyz_temp_nm1);
  } // end of loop over local nodes

  // Interpolate the Temperature data to back-traced points:
  SL_backtrace_interp.set_input(T_l.vec,T_l_dd.vec[0],T_l_dd.vec[1],quadratic_non_oscillatory_continuous_v2);
  SL_backtrace_interp.interpolate(T_l_backtrace.vec);

  if(advection_sl_order ==2){
      SL_backtrace_interp_nm1.set_input(T_l_nm1.vec,T_l_dd_nm1.vec[0],T_l_dd_nm1.vec[1], quadratic_non_oscillatory_continuous_v2);
      SL_backtrace_interp_nm1.interpolate(T_l_backtrace_nm1.vec);
    }

  // Destroy velocity derivatives now that not needed:
  foreach_dimension(d){
    foreach_dimension(dd)
    {
      ierr = VecDestroy(v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2) ierr = VecDestroy(v_dd_nm1[d][dd]); CHKERRXX(ierr);
    }
  }

  // Destroy temperature derivatives
  T_l_dd.destroy();
  if(advection_sl_order==2) {
      T_l_dd_nm1.destroy();
    }

  // Clear interp points:
  xyz_d->clear();xyz_d->shrink_to_fit();
  xyz_d_nm1->clear();xyz_d_nm1->shrink_to_fit();

  // Clear and delete interpolators:
  SL_backtrace_interp.clear();
  SL_backtrace_interp_nm1.clear();

  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Completes backtrace \n");
}

void interpolate_values_onto_new_grid(Vec *T_l, Vec *T_s, Vec v_interface[P4EST_DIM],
                                      Vec v_external[P4EST_DIM],
                                      p4est_nodes_t *nodes_new_grid, p4est_t *p4est_new,
                                      my_p4est_node_neighbors_t *ngbd_old_grid,interpolation_method interp_method/*,
                                      Vec *all_fields_old=NULL, Vec *all_fields_new=NULL*/){
  // Need neighbors of old grid to create interpolation object
  // Need nodes of new grid to get the points that we must interpolate to

  Vec all_fields_old[num_fields_interp];
  Vec all_fields_new[num_fields_interp];

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_old_grid);
//  my_p4est_interpolation_nodes_t* interp_nodes = NULL;
//  interp_nodes = new my_p4est_interpolation_nodes_t(ngbd_old_grid);

  // Set existing vectors as elements of the array of vectors: --------------------------
  unsigned int i = 0;
  if(solve_stefan){

      all_fields_old[i++] = *T_l; // Now, all_fields_old[0] and T_l both point to same object (where old T_l vec sits)
      all_fields_old[i++] = *T_s;

      foreach_dimension(d){
        all_fields_old[i++] = v_interface[d];
      }
    }
  if(solve_navier_stokes){
      foreach_dimension(d){
        all_fields_old[i++] = v_external[d];
      }
    }
  P4EST_ASSERT(i == num_fields_interp);

  // Create the array of vectors to hold the new values: ------------------------------
  PetscErrorCode ierr;
  for(unsigned int j = 0;j<num_fields_interp;j++){
    ierr = VecCreateGhostNodes(p4est_new,nodes_new_grid,&all_fields_new[j]);CHKERRXX(ierr);
    }

  // Do interpolation:--------------------------------------------
  interp_nodes.set_input(all_fields_old,interp_method,num_fields_interp);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes_new_grid){
    node_xyz_fr_n(n,p4est_new,nodes_new_grid,xyz);
    interp_nodes.add_point(n,xyz);
  }

  interp_nodes.interpolate(all_fields_new);
  interp_nodes.clear();

  // Destroy the old fields no longer in use:------------------------
  for(unsigned int k=0;k<num_fields_interp;k++){
    ierr = VecDestroy(all_fields_old[k]);CHKERRXX(ierr); // Destroy objects where the old vectors were
  }
  // Slide the newly interpolated fields to back to their passed objects
  i = 0;
  if(solve_stefan){
      *T_l = all_fields_new[i++]; // Now, T_l points to (new T_l vec)
      *T_s = all_fields_new[i++];

      foreach_dimension(d){
        v_interface[d] = all_fields_new[i++];
      }
    }
  if(solve_navier_stokes){
      foreach_dimension(d){
        v_external[d] = all_fields_new[i++];
      }
    }
  P4EST_ASSERT(i==num_fields_interp);
} // end of interpolate_values_onto_new_grid


void compute_interfacial_velocity(vec_and_ptr_dim_t T_l_d, vec_and_ptr_dim_t T_s_d, vec_and_ptr_dim_t jump, vec_and_ptr_dim_t v_interface, vec_and_ptr_t phi, my_p4est_node_neighbors_t *ngbd, double dxyz_close_to_interface){

  if(!force_interfacial_velocity_to_zero){  // Get arrays:
      jump.get_array();
      T_l_d.get_array();
      T_s_d.get_array();
      phi.get_array();

      // First, compute jump in the layer nodes:
      for(size_t i=0; i<ngbd->get_layer_size();i++){
        p4est_locidx_t n = ngbd->get_layer_node(i);

        if(fabs(phi.ptr[n])<2.*uniform_band*dxyz_close_to_interface/*true*/){
            foreach_dimension(d){
                if(example_ == ICE_AROUND_CYLINDER){ // for this example, we solve nondimensionalized problem
                    jump.ptr[d][n] = (St/Pe)*(rho_l/rho_s)*( (k_s/k_l)*T_s_d.ptr[d][n] - T_l_d.ptr[d][n]);
                }
                else{
                    jump.ptr[d][n] = (k_s*T_s_d.ptr[d][n] -k_l*T_l_d.ptr[d][n])/(L*rho_s);
                }
            } // end of loop over dimensions
        }
       }

      // Begin updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }

      // Compute the jump in the local nodes:
      for(size_t i = 0; i<ngbd->get_local_size();i++){
          p4est_locidx_t n = ngbd->get_local_node(i);
          if(/*true*/fabs(phi.ptr[n])<2.*uniform_band*dxyz_close_to_interface){

              foreach_dimension(d){
                  if(example_ == ICE_AROUND_CYLINDER){ // for this example, we solve nondimensionalized problem
                      jump.ptr[d][n] = (St/Pe)*(rho_l/rho_s)*( (k_s/k_l)*T_s_d.ptr[d][n] - T_l_d.ptr[d][n]);
                  }
                  else {
                      jump.ptr[d][n] = (k_s*T_s_d.ptr[d][n] -k_l*T_l_d.ptr[d][n])/(L*rho_s);
                  }
              } // end over loop on dimensions
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


  if(force_interfacial_velocity_to_zero){
        foreach_dimension(d){
            VecScaleGhost(v_interface.vec[d],0.0);
        }
  }
}

void compute_timestep(vec_and_ptr_dim_t v_interface, vec_and_ptr_t phi, double dxyz_close_to_interface, double dxyz_smallest[P4EST_DIM],p4est_nodes_t *nodes, p4est_t *p4est){

  double max_v_norm = 0.0;
  double global_max_vnorm = 0.0;

  if(example_ == COUPLED_PROBLEM_EXAMPLE){
    global_max_vnorm = PI; // known analytically

  }
  else {
    // Check the values of v_interface locally:
    v_interface.get_array();
    phi.get_array();
    foreach_local_node(n,nodes){
      if (fabs(phi.ptr[n]) < uniform_band*dxyz_close_to_interface){
        max_v_norm = max(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n])));
      }
    }
    v_interface.restore_array();
    phi.restore_array();

    // Get the maximum v norm across all the processors:
    int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
    SC_CHECK_MPI(mpi_ret);
  }

  // Compute new timestep:
  double dt_computed;
  dt_computed = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/global_max_vnorm;//min(global_max_vnorm,1.0);
  dt = min(dt_computed,dt_max_allowed);

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

void prepare_refinement_fields(vec_and_ptr_t phi, vec_and_ptr_t vorticity, vec_and_ptr_t vorticity_refine, vec_and_ptr_dim_t T_l_dd, my_p4est_node_neighbors_t* ngbd){
  PetscErrorCode ierr;

  // Get relevant arrays:
  vorticity.get_array();
  vorticity_refine.get_array();
  if(refine_by_d2T) {T_l_dd.get_array();}
  phi.get_array();

  // Compute proper refinement fields on layer nodes:
  for(size_t i = 0; i<ngbd->get_layer_size(); i++){
      p4est_locidx_t n = ngbd->get_layer_node(i);
      if(phi.ptr[n] < 0.){
          vorticity_refine.ptr[n] = vorticity.ptr[n];
        }
      else{
          vorticity_refine.ptr[n] = 0.0;

          if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
              foreach_dimension(d){
                T_l_dd.ptr[d][n]=0.;
              }
            }
        }
    } // end of loop over layer nodes

  // Begin updating the ghost values:
  ierr = VecGhostUpdateBegin(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateBegin(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  //Compute proper refinement fields on local nodes:
  for(size_t i = 0; i<ngbd->get_local_size(); i++){
      p4est_locidx_t n = ngbd->get_local_node(i);
      if(phi.ptr[n] < 0.){
          vorticity_refine.ptr[n] = vorticity.ptr[n];
        }
      else{
          vorticity_refine.ptr[n] = 0.0;
          if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
              foreach_dimension(d){
                T_l_dd.ptr[d][n]=0.;
              }
            }
        }
    } // end of loop over local nodes

  // Finish updating the ghost values:
  ierr = VecGhostUpdateEnd(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateEnd(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  // Restore appropriate arrays:
  if(refine_by_d2T) {T_l_dd.restore_array();}
  vorticity.restore_array();
  vorticity_refine.restore_array();
  phi.restore_array();
}


void poisson_step(Vec phi, Vec phi_solid,
                  Vec phi_dd[P4EST_DIM], Vec phi_solid_dd[P4EST_DIM],
                  Vec* T_l, Vec* T_s,
                  Vec rhs_Tl, Vec rhs_Ts,
                  BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2],
                  BC_WALL_VALUE_TEMP* bc_wall_value_temp[2],
                  my_p4est_node_neighbors_t* ngbd,
                  int cube_refinement,
                  Vec phi_cylinder=NULL, Vec phi_cylinder_dd[P4EST_DIM]=NULL ){
  my_p4est_poisson_nodes_mls_t* solver_Tl=NULL;
  my_p4est_poisson_nodes_mls_t* solver_Ts = NULL;

  // Create solvers:
  solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd);
  solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd);

  // Add the appropriate interfaces and interfacial boundary conditions:
  solver_Tl->add_boundary(MLS_INTERSECTION,phi,phi_dd[0],phi_dd[1],
      interface_bc_type_temp,*bc_interface_val_temp[LIQUID_DOMAIN],bc_interface_coeff);

  solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid,phi_solid_dd[0],phi_solid_dd[1],
      interface_bc_type_temp,*bc_interface_val_temp[SOLID_DOMAIN],bc_interface_coeff);

  if(example_ == ICE_AROUND_CYLINDER){
    solver_Ts->add_boundary(MLS_INTERSECTION,phi_cylinder,phi_cylinder_dd[0],phi_cylinder_dd[1],
        inner_interface_bc_type_temp,bc_interface_val_inner,bc_interface_coeff_inner);
    }

  // Set diagonal for Tl:
  if(do_advection){ // Cases with advection use semi lagrangian advection discretization in time
      if(advection_sl_order ==2){ // 2nd order semi lagrangian (BDF2 coefficients)
          solver_Tl->set_diag(advection_alpha_coeff/dt);
        }
      else{ // 1st order semi lagrangian (Backward Euler but with backtrace)
          solver_Tl->set_diag(1./dt);
        }
    }
  else{ // Cases with no temperature advection
      if(method_ ==2){ // Crank Nicholson
          solver_Tl->set_diag(2./dt);
        }
      else{ // Backward Euler
         solver_Tl->set_diag(1./dt);
        }
    }
  // Set diagonal for Ts:
  if(method_ == 2){ // Crank Nicholson
      solver_Ts->set_diag(2./dt);
    }
  else{ // Backward Euler
      solver_Ts->set_diag(1./dt);
    }

  if(solve_navier_stokes) solver_Tl->set_mu(1./Pe);
  else solver_Tl->set_mu(alpha_l);

  solver_Tl->set_rhs(rhs_Tl);

  if(solve_navier_stokes) solver_Ts->set_mu(1./Pe);
  else solver_Ts->set_mu(alpha_s);
  solver_Ts->set_rhs(rhs_Ts);

  // Set some other solver properties:
  solver_Tl->set_integration_order(1);
  solver_Tl->set_use_sc_scheme(0);
  solver_Tl->set_cube_refinement(cube_refinement);
  solver_Tl->set_store_finite_volumes(0);

  solver_Ts->set_integration_order(1);
  solver_Ts->set_use_sc_scheme(0);
  solver_Ts->set_cube_refinement(cube_refinement);
  solver_Ts->set_store_finite_volumes(0);

  // Set the wall BC and RHS:
  solver_Tl->set_wc(bc_wall_type_temp,*bc_wall_value_temp[LIQUID_DOMAIN]);
  solver_Ts->set_wc(bc_wall_type_temp,*bc_wall_value_temp[SOLID_DOMAIN]);


  // Preassemble the linear system
  solver_Tl->preassemble_linear_system();
  solver_Ts->preassemble_linear_system();

  // Solve the system:
  solver_Tl->solve(*T_l,false,true,KSPBCGS,PCHYPRE);
  solver_Ts->solve(*T_s,false,true,KSPBCGS,PCHYPRE);


  // Delete solvers:
  delete solver_Tl;
  delete solver_Ts;

}


void navier_stokes_step(my_p4est_navier_stokes_t* ns,
                        p4est_t* p4est_np1,p4est_nodes_t* nodes_np1,
                        Vec v_n[P4EST_DIM], Vec v_nm1[P4EST_DIM], Vec vorticity,Vec press_nodes,
                        KSPType face_solver_type, PCType pc_face,
                        KSPType cell_solver_type, PCType pc_cell,
                        my_p4est_faces_t* faces_np1, bool compute_pressure_,
                        char* name_fluid_forces=NULL, FILE* fich_fluid_forces=NULL){
  PetscErrorCode ierr;

  my_p4est_poisson_faces_t* face_solver;
  my_p4est_poisson_cells_t* cell_solver;
  Vec dxyz_hodge_old[P4EST_DIM];

  int mpi_comm = p4est_np1->mpicomm;

  // Create vector to store old dxyz hodge:
//  faces_np1 = ns->get_faces();
  for (unsigned char d=0;d<P4EST_DIM;d++){
    ierr = VecCreateNoGhostFaces(p4est_np1,faces_np1,&dxyz_hodge_old[d],d); CHKERRXX(ierr);
  }

  if (ramp_bcs && (tn<t_ramp)) hodge_tolerance = u0*hodge_percentage_of_max_u;
  else hodge_tolerance = NS_norm*hodge_percentage_of_max_u;
  PetscPrintf(mpi_comm,"Hodge tolerance is %e \n",hodge_tolerance);

  int hodge_iteration = 0;
  double convergence_check_on_dxyz_hodge = DBL_MAX;

  face_solver = NULL;
  cell_solver = NULL;

  while(hodge_iteration<hodge_max_it && convergence_check_on_dxyz_hodge>hodge_tolerance){
    ns->copy_dxyz_hodge(dxyz_hodge_old);
    ns->solve_viscosity(face_solver,(face_solver!=NULL),face_solver_type,pc_face);
    convergence_check_on_dxyz_hodge=
        ns->solve_projection(cell_solver,(cell_solver!=NULL),cell_solver_type,pc_cell,
                             false,NULL,dxyz_hodge_old,uvw_components);
    ierr= PetscPrintf(mpi_comm,"Hodge iteration : %d, hodge error: %0.3e \n",hodge_iteration,convergence_check_on_dxyz_hodge);CHKERRXX(ierr);
    hodge_iteration++;
  }

  for (unsigned char d=0;d<P4EST_DIM;d++){
    ierr = VecDestroy(dxyz_hodge_old[d]); CHKERRXX(ierr);
  }

  // Delete solvers:
  delete face_solver;
  delete cell_solver;

  // Compute velocity at the nodes
  ns->compute_velocity_at_nodes();

  // Set this timestep's "v_n" to be the "v_nm1" for the next timestep
  ns->copy_velocity_n(v_nm1);


  // Now set this step's "v_np1" to be "v_n" for the next timestep -- v_n for next step will be sampled at this grid for now, but will be interpolated onto new grid for next step in beginning of next step
  ns->copy_velocity_np1(v_n);

  // Compute the pressure
  if(compute_pressure_){
    ns->compute_pressure(); // note: only compute pressure at nodes when we are saving to VTK (or evaluating some errors)
    ns->compute_pressure_at_nodes(&press_nodes);
  }


  // Get the computed values of vorticity
  ns->copy_vorticity(vorticity);

  // Compute forces (if we are doing that)
  if(save_fluid_forces){
    double forces[P4EST_DIM];
    ns->compute_forces(forces);
    PetscPrintf(mpi_comm,"tn = %g, fx = %g, fy = %g \n",tn+dt,forces[0],forces[1]);
    ierr = PetscFOpen(mpi_comm,name_fluid_forces,"a",&fich_fluid_forces); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi_comm,fich_fluid_forces,"%g %g %g \n",tn+dt,forces[0],forces[1]);CHKERRXX(ierr);
    ierr = PetscFClose(mpi_comm,fich_fluid_forces); CHKERRXX(ierr);
    PetscPrintf(mpi_comm,"forces saved \n");

  }

  // Check the L2 norm of u to make sure nothing is blowing up
  NS_norm = ns->get_max_L2_norm_u();
  PetscPrintf(mpi_comm,"\n Max NS velocity norm: \n"
                         " - Computational value: %0.3e  "
                         " - Physical value: %0.3e [m/s]  "
                         " - Physical value: %0.3e [mm/s] \n \n",NS_norm,NS_norm*u_inf,NS_norm*u_inf*1000.);

  // Stop simulation if things are blowing up
  if(NS_norm>100.0){
      std::cerr<<"The simulation blew up \n"<<std::endl;
      SC_ABORT("Navier Stokes velocity blew up \n");
    }

  // Compute the corresponding timestep:
  ns->compute_dt();
  dt_NS = ns->get_dt();

}

void initialize_ns_solver(my_p4est_navier_stokes_t* &ns,
                          p4est_t* p4est_np1,p4est_ghost_t* ghost_np1,
                          my_p4est_node_neighbors_t* ngbd_np1, my_p4est_node_neighbors_t* ngbd_n,
                          my_p4est_hierarchy_t* hierarchy_np1, my_p4est_brick_t* brick,
                          Vec phi, Vec v_n_NS[P4EST_DIM],Vec v_nm1_NS[P4EST_DIM],
                          my_p4est_faces_t* &faces_np1, my_p4est_cell_neighbors_t* &ngbd_c_np1){

  // Create the initial neigbhors and faces (after first step, NS grid update will handle this internally)
  ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);
  faces_np1 = new my_p4est_faces_t(p4est_np1,ghost_np1,brick,ngbd_c_np1);

  // Create the solver
  ns = new my_p4est_navier_stokes_t(ngbd_n,ngbd_np1,faces_np1);

  // Set the LSF:
  ns->set_phi(phi);
  ns->set_dt(dt_nm1,dt);
  ns->set_velocities(v_nm1_NS,v_n_NS);

  PetscPrintf(p4est_np1->mpicomm,"CFL: %0.2f, rho : %0.2f, mu : %0.3e \n",cfl,rho_l,mu_l);
  ns->set_parameters((1./Re),1.0,NS_advection_sl_order,uniform_band,vorticity_threshold,cfl);


}

bool are_we_saving_vtk(double tstep_, double tn_,bool is_load_step, int& out_idx, bool get_new_outidx){
  bool out = false;
  if(save_to_vtk){
    if(save_using_dt){
        out= (((int) floor(tn_/save_every_dt) ) !=out_idx) && (!is_load_step);
        if(get_new_outidx){
          out_idx = ((int) floor(tn_/save_every_dt) );
        }
      }
    else if (save_using_iter){
        out = (( (int) floor(tstep_/save_every_iter) ) !=out_idx) && (!is_load_step);
        if(get_new_outidx) {
          out_idx = ((int) floor(tn_/save_every_iter) );
        }
      }
  }
  return out;
}
// --------------------------------------------------------------------------------------------------------------
// FUNCTIONS FOR SAVING TO VTK:
// --------------------------------------------------------------------------------------------------------------
void save_everything(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_node_neighbors_t* ngbd,vec_and_ptr_t phi, vec_and_ptr_t phi_2, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int,vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, char* filename){
// Things we want to save:
/*
 * LSF
 * LSF2 for ex 2
 * Tl
 * Ts
 * v_interface
 * v NS
 * pressure
 * vorticity
 * smoke -TAKEN OUT
 * */

  // Get arrays:
  phi.get_array();
  if(example_ == ICE_AROUND_CYLINDER) phi_2.get_array();
  Tl.get_array(); Ts.get_array();
  v_int.get_array(); v_NS.get_array();
  press.get_array(); vorticity.get_array();


  // Save data:
  std::vector<std::string> point_names;
  std::vector<double*> point_data;

  if (example_ == ICE_AROUND_CYLINDER) {
      point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
      point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
  }

  else{
      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
      point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
    }

  std::vector<std::string> cell_names = {};
  std::vector<double*> cell_data = {};

  my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);

  point_names.clear();point_data.clear();
  cell_names.clear(); cell_data.clear();

  // Restore arrays:
  phi.restore_array();
  if(example_ == ICE_AROUND_CYLINDER) phi_2.restore_array();

  Tl.restore_array(); Ts.restore_array();
  v_int.restore_array(); v_NS.restore_array();
  press.restore_array(); vorticity.restore_array();

//  Tl_dd.restore_array();
//  Tl_dd.destroy();

}

void save_stefan_fields(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_t phi_2, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int, char* filename ){
  // Things we want to save:
  /*
   * LSF
   * LSF2 for ex 2
   * Tl
   * Ts
   * v_interface
   * */

    // Get arrays:
    phi.get_array();
    if(example_ == ICE_AROUND_CYLINDER) phi_2.get_array();

    Tl.get_array(); Ts.get_array();

    v_int.get_array();


    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    if (example_ == ICE_AROUND_CYLINDER) {
        point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y"};
        point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1]};
      }

    else{
        point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y"};
        point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1]};
      }

    std::vector<std::string> cell_names;
    std::vector<double*> cell_data;

    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);



    // Clear the vectors:
    cell_names.clear(); cell_data.clear();
    point_names.clear();point_data.clear();

    // Restore arrays:
    phi.restore_array();
    if(example_ == ICE_AROUND_CYLINDER) phi_2.restore_array();

    Tl.restore_array(); Ts.restore_array();

    v_int.restore_array();

}
void save_navier_stokes_fields(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, char* filename){
  // Things we want to save:
  /*
   * LSF
   * v NS
   * pressure
   * vorticity
   * smoke - REMOVED
   * */
    // First, need to scale the fields appropriately:
    // Scale velocities:

    // Get arrays:
    phi.get_array();
    v_NS.get_array();

    press.get_array(); vorticity.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    point_names = {"phi","u","v","vorticity","pressure"};
    point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};


    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

//    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);
    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


    point_names.clear(); point_data.clear();
    cell_names.clear(); cell_data.clear();


    // Restore arrays:

    phi.restore_array();
    v_NS.restore_array();

    press.restore_array(); vorticity.restore_array();

} // end of save_navier_stokes_fields

void save_navier_stokes_test_case(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, double dxyz_close_to_interface,bool are_we_saving_vtk,char* filename_vtk, char* filename_err_output, FILE* fich){

  // Save NS analytical to compare:
  vec_and_ptr_dim_t vn_analytical;
  vec_and_ptr_t pn_analytical;

  vn_analytical.create(p4est,nodes);
  pn_analytical.create(p4est,nodes);

  velocity_component* analytical_soln_comp[P4EST_DIM];
  for(unsigned char d=0;d<P4EST_DIM;++d){
    analytical_soln_comp[d] = new velocity_component(d);
    analytical_soln_comp[d]->t = tn;
  }
  CF_DIM *analytical_soln[P4EST_DIM] = {DIM(analytical_soln_comp[0],analytical_soln_comp[1],analytical_soln_comp[2])};

  foreach_dimension(d){
    sample_cf_on_nodes(p4est,nodes,*analytical_soln[d],vn_analytical.vec[d]);
  }
  sample_cf_on_nodes(p4est,nodes,zero_cf,pn_analytical.vec);

  // Get errors:
  vec_and_ptr_dim_t vn_error;
  vec_and_ptr_t press_error;

  vn_error.create(p4est,nodes);
  press_error.create(p4est,nodes);

  double L_inf_u = 0., L_inf_v = 0.,L_inf_P = 0.;

  vn_analytical.get_array(); vn_error.get_array(); v_NS.get_array();
  pn_analytical.get_array(); press_error.get_array(); press.get_array();

  phi.get_array();
  foreach_node(n,nodes){
    if(phi.ptr[n]<0.){
      press_error.ptr[n] = fabs(press.ptr[n] - pn_analytical.ptr[n]);
      vn_error.ptr[0][n] = fabs(v_NS.ptr[0][n] - vn_analytical.ptr[0][n]);
      vn_error.ptr[1][n] = fabs(v_NS.ptr[1][n] - vn_analytical.ptr[1][n]);

      L_inf_u = max(L_inf_u,vn_error.ptr[0][n]);
      L_inf_v = max(L_inf_v,vn_error.ptr[1][n]);
      L_inf_P = max(L_inf_P,press_error.ptr[n]);
    }

  }

  // Get the global errors:
  double local_Linf_errors[3] = {L_inf_u,L_inf_v,L_inf_P};
  double global_Linf_errors[3] = {0.0,0.0,0.0};

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,3,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n"
                             "Errors on NS Validation "
                             "\n -------------------------------------\n"
                             "Linf on u: %0.3e \n"
                             "Linf on v: %0.3e \n"
                             "Linf on P: %0.3e \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.3e \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:
  PetscErrorCode ierr;
  ierr = PetscFOpen(p4est->mpicomm,filename_err_output,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],hodge_global_error,num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);


  if(are_we_saving_vtk){
    vorticity.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    point_names = {"phi","u","v","vorticity","pressure","u_ana","v_ana","P_ana","u_err","v_err","P_err"};
    point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,vn_analytical.ptr[0],vn_analytical.ptr[1],
                  pn_analytical.ptr,vn_error.ptr[0],vn_error.ptr[1],press_error.ptr};


    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

//    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);
    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);

    point_names.clear(); point_data.clear();
    cell_names.clear(); cell_data.clear();

    vorticity.restore_array();
  }


  v_NS.restore_array();press.restore_array();
  vn_analytical.restore_array(); pn_analytical.restore_array();
  vn_error.restore_array(); press_error.restore_array(); phi.restore_array();

  vn_analytical.destroy();
  pn_analytical.destroy();

  vn_error.destroy();
  press_error.destroy();
}

void save_coupled_test_case(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_interface ,vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, double dxyz_close_to_interface,bool are_we_saving_vtk,char* filename_vtk, char* filename_err_output, FILE* fich){

  // Save analytical fields to compare:
  vec_and_ptr_dim_t vn_analytical;
  vec_and_ptr_t pn_analytical;
  vec_and_ptr_t Tl_analytical;
  vec_and_ptr_t Ts_analytical;
  vec_and_ptr_dim_t v_interface_analytical;
  vec_and_ptr_t phi_analytical; // Only for last timestep, we see how well phi is recovered after deforming

  vn_analytical.create(p4est,nodes);
  pn_analytical.create(p4est,nodes);
  Tl_analytical.create(p4est,nodes);
  Ts_analytical.create(p4est,nodes);
  v_interface_analytical.create(p4est,nodes);

  velocity_component* analytical_soln_velNS[P4EST_DIM];
  temperature_field* analytical_soln_temp[2];
  interfacial_velocity* analytical_soln_velINT[P4EST_DIM];

  for(unsigned char d=0;d<2;++d){
    analytical_soln_temp[d] = new temperature_field(d);
    analytical_soln_temp[d]->t = tn;
  }

  for(unsigned char d=0;d<P4EST_DIM;++d){
    analytical_soln_velNS[d] = new velocity_component(d);
    analytical_soln_velNS[d]->t = tn;

    analytical_soln_velINT[d] = new interfacial_velocity(d,analytical_soln_temp);
  }
  CF_DIM *analytical_soln_velNS_cf[P4EST_DIM] = {DIM(analytical_soln_velNS[0],analytical_soln_velNS[1],analytical_soln_velNS[2])};
  CF_DIM *analytical_soln_velINT_cf[P4EST_DIM] = {DIM(analytical_soln_velINT[0],analytical_soln_velINT[1],analytical_soln_velINT[2])};
  CF_DIM *analytical_soln_temp_cf[2] = {analytical_soln_temp[LIQUID_DOMAIN],analytical_soln_temp[SOLID_DOMAIN]};

  foreach_dimension(d){
    sample_cf_on_nodes(p4est,nodes,*analytical_soln_velNS_cf[d],vn_analytical.vec[d]);
    sample_cf_on_nodes(p4est,nodes,*analytical_soln_velINT_cf[d],v_interface_analytical.vec[d]);
  }
  sample_cf_on_nodes(p4est,nodes,*analytical_soln_temp_cf[LIQUID_DOMAIN],Tl_analytical.vec);
  sample_cf_on_nodes(p4est,nodes,*analytical_soln_temp_cf[SOLID_DOMAIN],Ts_analytical.vec);
  sample_cf_on_nodes(p4est,nodes,zero_cf,pn_analytical.vec);


  // Get errors:
  vec_and_ptr_dim_t vn_error;
  vec_and_ptr_t press_error;
  vec_and_ptr_t Tl_error;
  vec_and_ptr_t Ts_error;
  vec_and_ptr_t v_int_error; // Only use magnitude for v_int error, not component by component
  vec_and_ptr_t phi_error; // only for last timestep

  vn_error.create(p4est,nodes);
  press_error.create(p4est,nodes);
  Tl_error.create(p4est,nodes);
  Ts_error.create(p4est,nodes);
  v_int_error.create(p4est,nodes);

  double L_inf_u = 0., L_inf_v = 0.,L_inf_P = 0.,L_inf_Tl = 0., L_inf_Ts = 0., L_inf_vint = 0.,L_inf_phi=0.;

  vn_analytical.get_array(); vn_error.get_array(); v_NS.get_array();
  pn_analytical.get_array(); press_error.get_array(); press.get_array();

  Tl_analytical.get_array(); Tl_error.get_array(); Tl.get_array();
  Ts_analytical.get_array(); Ts_error.get_array(); Ts.get_array();
  v_interface_analytical.get_array();v_int_error.get_array(); v_interface.get_array();

  if((tn+dt)>=tfinal){
    phi_analytical.create(p4est,nodes); phi_error.create(p4est,nodes);
    sample_cf_on_nodes(p4est,nodes,level_set,phi_analytical.vec);

    phi_analytical.get_array();
    phi_error.get_array();

  }

  phi.get_array();
  foreach_node(n,nodes){
    if(phi.ptr[n]<0.){
      press_error.ptr[n] = fabs(press.ptr[n] - pn_analytical.ptr[n]);
      vn_error.ptr[0][n] = fabs(v_NS.ptr[0][n] - vn_analytical.ptr[0][n]);
      vn_error.ptr[1][n] = fabs(v_NS.ptr[1][n] - vn_analytical.ptr[1][n]);

      Tl_error.ptr[n] = fabs(Tl.ptr[n] - Tl_analytical.ptr[n]);

      L_inf_u = max(L_inf_u,vn_error.ptr[0][n]);
      L_inf_v = max(L_inf_v,vn_error.ptr[1][n]);
      L_inf_P = max(L_inf_P,press_error.ptr[n]);
      L_inf_Tl = max(L_inf_Tl,Tl_error.ptr[n]);
    }
    else{
      Ts_error.ptr[n] = fabs(Ts.ptr[n] - Ts_analytical.ptr[n]);

      L_inf_Ts = max(L_inf_Ts,Ts_error.ptr[n]);
    }

    // Check error in v_int and phi only in a uniform band around the interface
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){
      v_int_error.ptr[n] = fabs(sqrt(SUMD(SQR(v_interface.ptr[0][n]),SQR(v_interface.ptr[1][n]),SQR(v_interface.ptr[2][n]))) -
          sqrt(SUMD(SQR(v_interface_analytical.ptr[0][n]),SQR(v_interface_analytical.ptr[1][n]),SQR(v_interface_analytical.ptr[2][n]))));
      L_inf_vint = max(L_inf_vint,v_int_error.ptr[n]);

      if((tn+dt)>=tfinal){ // Check phi error only at the final time
        phi_error.ptr[n] = fabs(phi.ptr[n] - phi_analytical.ptr[n]);
        L_inf_phi = max(L_inf_phi,phi_error.ptr[n]);
      }
    }


  }

  // Get the global errors:
  double local_Linf_errors[7] = {L_inf_u,L_inf_v,L_inf_P,L_inf_Tl,L_inf_Ts,L_inf_vint,L_inf_phi};
  double global_Linf_errors[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,7,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM,p4est->mpicomm);

  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n"
                             "Errors on Coupled Validation "
                             "\n -------------------------------------\n"
                             "Linf on u: %0.3e \n"
                             "Linf on v: %0.3e \n"
                             "Linf on P: %0.3e \n"
                             "Linf on Tl: %0.3e \n"
                             "Linf on Ts: %0.3e \n"
                             "Linf on v_int: %0.3e \n"
                             "Linf on phi: %0.3e (only relevant for last timestep)\n \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.3e \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                              global_Linf_errors[3],global_Linf_errors[4],global_Linf_errors[5],global_Linf_errors[6],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:
  PetscErrorCode ierr;
  ierr = PetscFOpen(p4est->mpicomm,filename_err_output,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d "
                                          "%g %g %g "
                                          "%g %g %g "
                                          "%g "
                                          "%d %g \n",tn,dt,tstep,
                                                     global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                                                     global_Linf_errors[3],global_Linf_errors[4],global_Linf_errors[5],
                                                     global_Linf_errors[6],
                                                     num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);


  if(are_we_saving_vtk){
    vorticity.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    if((tn+dt)>=tfinal){
      point_names = {"phi","u","v","vorticity","pressure","Tl","Ts","v_int_x","v_int_y",
                     "phi_ana","u_ana","v_ana","P_ana","Tl_ana","Ts_ana","v_int_x_ana","v_int_y_ana",
                     "phi_err","u_err","v_err","P_err","Tl_err","Ts_err","v_int_err"};
      point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,Tl.ptr,Ts.ptr,v_interface.ptr[0],v_interface.ptr[1],
                    phi_analytical.ptr,vn_analytical.ptr[0],vn_analytical.ptr[1],pn_analytical.ptr, Tl_analytical.ptr,Ts_analytical.ptr,v_interface_analytical.ptr[0],v_interface_analytical.ptr[1],
                    phi_error.ptr,vn_error.ptr[0],vn_error.ptr[1],press_error.ptr,Tl_error.ptr,Ts_error.ptr,v_int_error.ptr};

    }
    else{
      point_names = {"phi","u","v","vorticity","pressure","Tl","Ts","v_int_x","v_int_y",
                     "u_ana","v_ana","P_ana","Tl_ana","Ts_ana","v_int_x_ana","v_int_y_ana",
                     "u_err","v_err","P_err","Tl_err","Ts_err","v_int_err"};
      point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,Tl.ptr,Ts.ptr,v_interface.ptr[0],v_interface.ptr[1],
                    vn_analytical.ptr[0],vn_analytical.ptr[1],pn_analytical.ptr, Tl_analytical.ptr,Ts_analytical.ptr,v_interface_analytical.ptr[0],v_interface_analytical.ptr[1],
                    vn_error.ptr[0],vn_error.ptr[1],press_error.ptr,Tl_error.ptr,Ts_error.ptr,v_int_error.ptr};
    }


    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

    my_p4est_vtk_write_all_lists(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename_vtk,point_data,point_names,cell_data,cell_names);

    point_names.clear(); point_data.clear();
    cell_names.clear(); cell_data.clear();

    vorticity.restore_array();
  }


  // Restore arrays:
  vn_analytical.restore_array(); vn_error.restore_array(); v_NS.restore_array();
  pn_analytical.restore_array(); press_error.restore_array(); press.restore_array();

  Tl_analytical.restore_array(); Tl_error.restore_array(); Tl.restore_array();
  Ts_analytical.restore_array(); Ts_error.restore_array(); Ts.restore_array();
  v_interface_analytical.restore_array();v_int_error.restore_array(); v_interface.restore_array();


  phi.restore_array();

  // Destroy arrays:
  vn_analytical.destroy(); vn_error.destroy();
  pn_analytical.destroy(); press_error.destroy();

  Tl_analytical.destroy(); Tl_error.destroy();
  Ts_analytical.destroy(); Ts_error.destroy();
  v_interface_analytical.destroy();v_int_error.destroy();

  // Handle phi error checking last if it was done:
  if((tn+dt)>=tfinal){
    phi_error.restore_array();
    phi_error.destroy();
    phi_analytical.restore_array();
    phi_analytical.destroy();
  }


}

// --------------------------------------------------------------------------------------------------------------
// FUNCTIONS FOr SAVING OR LOADING SIMULATION STATE:
// --------------------------------------------------------------------------------------------------------------

void fill_or_load_double_parameters(save_or_load flag,PetscInt num,splitting_criteria_t* sp, PetscReal *data){
  size_t idx=0;
  switch(flag){
    case SAVE:{
        data[idx++] = tn;
        data[idx++] = dt;
        data[idx++] = dt_nm1;
        data[idx++] = k_l;
        data[idx++] = k_s;
        data[idx++] = alpha_l;
        data[idx++] = alpha_s;
        data[idx++] = rho_l;
        data[idx++] = rho_s;
        data[idx++] = mu_l;
        data[idx++] = L;
        data[idx++] = cfl;
        data[idx++] = uniform_band;
        data[idx++] = sp->lip;
        data[idx++] = NS_norm;
        break;
      }
    case LOAD:{
        tn = data[idx++];
        dt = data[idx++];
        // Note: since these parameters depend on advection sl order, need to load integers first before doubles
        dt_nm1 = data[idx++];
        k_l = data[idx++];
        k_s = data[idx++];
        alpha_l = data[idx++];
        alpha_s = data[idx++];
        rho_l = data[idx++];
        rho_s = data[idx++];
        mu_l = data[idx++];
        L = data[idx++];
        cfl = data[idx++];
        uniform_band= data[idx++];
        sp->lip = data[idx++];
        NS_norm = data[idx++];
      }

    }
  P4EST_ASSERT(idx == num);
};

void fill_or_load_integer_parameters(save_or_load flag, PetscInt num, splitting_criteria_t* sp,PetscInt *data){
  size_t idx=0;
  switch(flag){
    case SAVE:{
        data[idx++] = advection_sl_order;
        data[idx++] = save_every_iter;
        data[idx++] = tstep;
        data[idx++] = sp->min_lvl;
        data[idx++] = sp->max_lvl;
        break;
      }
    case LOAD:{
        advection_sl_order = data[idx++];
        save_every_iter = data[idx++];
        tstep = data[idx++];
        sp->min_lvl=data[idx++];
        sp->max_lvl=data[idx++];
      }

    }
  P4EST_ASSERT(idx == num);
};
void save_or_load_parameters(const char* filename, splitting_criteria_t* sp,save_or_load flag, const mpi_environment_t* mpi=NULL){
  PetscErrorCode ierr;

  // Double parameters we need to save:
  // - tn, dt, dt_nm1 (if 2nd order), k_l, k_s, alpha_l, alpha_s, rho_l, rho_s, mu_l, L, cfl, uniform_band, scaling, data->lip
  PetscInt num_doubles = 15;
  PetscReal double_parameters[num_doubles];

  // Integer parameters we need to save:
  // - current lmin, current lmax, advection_sl_order, save_every_iter, tstep, data->min_lvl, data->max_lvl
  PetscInt num_integers = 5;
  PetscInt integer_parameters[num_integers];

  int fd;
  char diskfilename[PATH_MAX];

  switch(flag){
    case SAVE:{
        if(mpi->rank() ==0){

            // Save the integer parameters to a file
            sprintf(diskfilename,"%s_integers",filename);
            fill_or_load_integer_parameters(flag,num_integers,sp,integer_parameters);
            ierr = PetscBinaryOpen(diskfilename,FILE_MODE_WRITE,&fd); CHKERRXX(ierr);
            ierr = PetscBinaryWrite(fd, integer_parameters, num_integers, PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

            // Save the double parameters to a file:

            sprintf(diskfilename, "%s_doubles", filename);
            fill_or_load_double_parameters(flag,num_doubles,sp, double_parameters);

            ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
            ierr = PetscBinaryWrite(fd, double_parameters, num_doubles, PETSC_DOUBLE, PETSC_TRUE); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);


          }
        break;
      }
    case LOAD: {
        // First, load the integer parameters:
        sprintf(diskfilename, "%s_integers", filename);
        if(!file_exists(diskfilename))
          throw std::invalid_argument("The file storing the solver's integer parameters could not be found");
        if(mpi->rank()==0){
            ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
            ierr = PetscBinaryRead(fd, integer_parameters, num_integers, PETSC_INT); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
          }
        int mpiret = MPI_Bcast(integer_parameters, num_integers, MPI_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
        fill_or_load_integer_parameters(flag,num_integers,sp, integer_parameters);

        // Now, load the double parameters:
        sprintf(diskfilename, "%s_doubles", filename);
        if(!file_exists(diskfilename))
          throw std::invalid_argument("The file storing the solver's double parameters could not be found");
        if(mpi->rank() == 0)
        {
          ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
          ierr = PetscBinaryRead(fd, double_parameters, num_doubles, PETSC_DOUBLE); CHKERRXX(ierr);
          ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

        }
        mpiret = MPI_Bcast(double_parameters, num_doubles, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
        fill_or_load_double_parameters(flag,num_doubles,sp, double_parameters);
        break;
      }
    default:
      throw std::runtime_error("Unkown flag values were used when load/saving parameters \n");


    }

}

void save_state(mpi_environment_t &mpi,const char* path_to_directory,unsigned int n_saved,
                splitting_criteria_cf_and_uniform_band_t* sp, p4est_t* p4est, p4est_nodes_t* nodes,
                Vec phi, Vec T_l_n,Vec T_l_nm1, Vec T_s_n,
                Vec v_NS[P4EST_DIM],Vec v_NS_nm1[P4EST_DIM],Vec vorticity){
  PetscErrorCode ierr;

  if(!file_exists(path_to_directory)){
    create_directory(path_to_directory,p4est->mpirank,p4est->mpicomm);
  }
  if(!is_folder(path_to_directory)){
      if(!create_directory(path_to_directory, p4est->mpirank, p4est->mpicomm))
      {
        char error_msg[1024];
        sprintf(error_msg, "save_state: the path %s is invalid and the directory could not be created", path_to_directory);
        throw std::invalid_argument(error_msg);
      }
    }

  unsigned int backup_idx = 0;

  if(mpi.rank() ==0){
      unsigned int n_backup_subfolders = 0;

      // Get the current number of backups already present:
      // (Delete extra ones that exist for whatever reason)
      std::vector<std::string> subfolders; subfolders.resize(0);
      get_subdirectories_in(path_to_directory,subfolders);

      for(size_t idx =0; idx<subfolders.size(); ++idx){
          if(!subfolders[idx].compare(0,7,"backup_")){
              unsigned int backup_idx;
              sscanf(subfolders[idx].c_str(), "backup_%d", &backup_idx);

              if(backup_idx >= n_saved)
              {
                char full_path[PATH_MAX];
                sprintf(full_path, "%s/%s", path_to_directory, subfolders[idx].c_str());
                delete_directory(full_path, p4est->mpirank, p4est->mpicomm, true);
              }
              else
                n_backup_subfolders++;
            }
        }

      // check that they are successively indexed if less than the max number
      if(n_backup_subfolders < n_saved)
      {
        backup_idx = 0;
        for (unsigned int idx = 0; idx < n_backup_subfolders; ++idx) {
          char expected_dir[PATH_MAX];
          sprintf(expected_dir, "%s/backup_%d", path_to_directory, (int) idx);

          if(!is_folder(expected_dir))
            break; // well, it's a mess in there, but I can't really do any better...
          backup_idx++;
        }
      }

      // Slide the names of the backup folders in time:
      if ((n_saved > 1) && (n_backup_subfolders == n_saved))
      {
        char full_path_zeroth_index[PATH_MAX];
        sprintf(full_path_zeroth_index, "%s/backup_0", path_to_directory);
        // delete the 0th
        delete_directory(full_path_zeroth_index, p4est->mpirank, p4est->mpicomm, true);
        // shift the others
        for (size_t idx = 1; idx < n_saved; ++idx) {
          char old_name[PATH_MAX], new_name[PATH_MAX];
          sprintf(old_name, "%s/backup_%d", path_to_directory, (int) idx);
          sprintf(new_name, "%s/backup_%d", path_to_directory, (int) (idx-1));
          rename(old_name, new_name);
        }
        backup_idx = n_saved-1;
      }

      subfolders.clear();

    } // end of operations only on rank 0

    int mpiret = MPI_Bcast(&backup_idx, 1, MPI_INT, 0, p4est->mpicomm); SC_CHECK_MPI(mpiret);// acts as a MPI_Barrier, too

    char path_to_folder[PATH_MAX];
    sprintf(path_to_folder, "%s/backup_%d", path_to_directory, (int) backup_idx);
    create_directory(path_to_folder, p4est->mpirank, p4est->mpicomm);

    char filename[PATH_MAX];

    // save the solver parameters
    sprintf(filename, "%s/solver_parameters", path_to_folder);
    save_or_load_parameters(filename,sp, SAVE,&mpi);

    // Save the p4est and corresponding data:
    if(solve_coupled){
        if(advection_sl_order==2){
            my_p4est_save_forest_and_data(path_to_folder,p4est,nodes,
                                          "p4est",7,
                                          "phi",1,&phi,
                                          "T_l_n",1, &T_l_n,
                                          "T_l_nm1",1, &T_l_nm1,
                                          "T_s_n",1,&T_s_n,
                                          "v_NS_n",P4EST_DIM,v_NS,
                                          "v_NS_nm1",P4EST_DIM,v_NS_nm1,
                                          "vorticity",1,&vorticity);
        }
        else{
          my_p4est_save_forest_and_data(path_to_folder,p4est,nodes,
                                        "p4est",6,
                                        "phi",1,&phi,
                                        "T_l_n",1, &T_l_n,
                                        "T_s_n",1,&T_s_n,
                                        "v_NS_n",P4EST_DIM,v_NS,
                                        "v_NS_nm1",P4EST_DIM,v_NS_nm1,
                                        "vorticity",1,&vorticity);
        }
    }
    else if (solve_navier_stokes && !solve_stefan){
            my_p4est_save_forest_and_data(path_to_folder,p4est,nodes,
                                          "p4est",4,
                                          "phi",1,&phi,
                                          "v_NS_n",P4EST_DIM,v_NS,
                                          "v_NS_nm1",P4EST_DIM,v_NS_nm1,
                                          "vorticity",1,&vorticity);
    }
    ierr = PetscPrintf(p4est->mpicomm,"Saved solver state in ... %s \n",path_to_folder);CHKERRXX(ierr);
}

void load_state(const mpi_environment_t& mpi, const char* path_to_folder,
                splitting_criteria_cf_and_uniform_band_t* sp, p4est_t* &p4est, p4est_nodes_t* &nodes,
                p4est_ghost_t* &ghost,p4est_connectivity* &conn,
                Vec *phi,Vec *T_l_n, Vec *T_l_nm1, Vec *T_s_n,
                Vec v_NS[P4EST_DIM],Vec v_NS_nm1[P4EST_DIM],Vec *vorticity){

  char filename[PATH_MAX];
  if(!is_folder(path_to_folder)) throw std::invalid_argument("Load state: path to directory is invalid \n");

  // First load the general solver parameters -- integers and doubles
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename,sp,LOAD,&mpi);

  // Load p4est_n and corresponding objections
  PetscPrintf(mpi.comm(),"About to try and load forest and data , adv order = %d \n",advection_sl_order);
  if(solve_coupled){
      if(advection_sl_order==2){
          my_p4est_load_forest_and_data(mpi.comm(),path_to_folder,p4est,conn,P4EST_TRUE,ghost,nodes,
                                        "p4est",7,
                                        "phi",NODE_DATA,1,phi,
                                        "T_l_n",NODE_DATA,1,T_l_n,
                                        "T_l_nm1",NODE_DATA,1,T_l_nm1,
                                        "T_s_n",NODE_DATA,1,T_s_n,
                                        "v_NS_n",NODE_DATA,P4EST_DIM,v_NS,
                                        "v_NS_nm1",NODE_DATA,P4EST_DIM,v_NS_nm1,
                                        "vorticity",NODE_DATA,1,vorticity);
      }
      else{
          my_p4est_load_forest_and_data(mpi.comm(),path_to_folder,p4est,conn,P4EST_TRUE,ghost,nodes,
                                        "p4est",6,
                                        "phi",NODE_DATA,1,phi,
                                        "T_l_n",NODE_DATA,1,T_l_n,
                                        "T_s_n",NODE_DATA,1,T_s_n,
                                        "v_NS_n",NODE_DATA,P4EST_DIM,v_NS,
                                        "v_NS_nm1",NODE_DATA,P4EST_DIM,v_NS_nm1,
                                        "vorticity",NODE_DATA,1,vorticity);
      }
  }
  else if (solve_navier_stokes && !solve_stefan){
          my_p4est_load_forest_and_data(mpi.comm(),path_to_folder,p4est,conn,P4EST_TRUE,ghost,nodes,
                                        "p4est",4,
                                        "phi",NODE_DATA,1,phi,
                                        "v_NS_n",NODE_DATA,P4EST_DIM,v_NS,
                                        "v_NS_nm1",NODE_DATA,P4EST_DIM,v_NS_nm1,
                                        "vorticity",NODE_DATA,P4EST_DIM,vorticity);
  }

  P4EST_ASSERT(find_max_level(p4est) == sp->max_lvl);

  // Update the user pointer:
  splitting_criteria_cf_and_uniform_band_t* sp_new = new splitting_criteria_cf_and_uniform_band_t(*sp);
  p4est->user_pointer = (void*) sp_new;

  PetscPrintf(mpi.comm(),"Loads forest and data \n");
}


// --------------------------------------------------------------------------------------------------------------
// BEGIN MAIN OPERATION:
// --------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;
  PetscViewer viewer;
  int mpi_ret; // Check mpi issues

//  PetscMemorySetGetMaximumUsage();
  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc,argv);

  pl.get_all(cmd);
  select_solvers();
  solve_coupled = solve_navier_stokes && solve_stefan;


  PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d, method = %d \n",lmin,lmax,method_);
  PetscPrintf(mpi.comm(),"Number of mpi tasks: %d \n",mpi.size());
  PetscPrintf(mpi.comm(),"Stefan = %d, NS = %d, Coupled = %d \n",solve_stefan,solve_navier_stokes,solve_coupled);

  // -----------------------------------------------
  // Declare all needed variables:
  // -----------------------------------------------
  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;
  my_p4est_hierarchy_t* hierarchy;
  my_p4est_node_neighbors_t* ngbd;

  p4est_t               *p4est_np1;
  p4est_nodes_t         *nodes_np1;
  p4est_ghost_t         *ghost_np1;
  my_p4est_hierarchy_t* hierarchy_np1;
  my_p4est_node_neighbors_t* ngbd_np1;

  // Level set function(s):---------------------------
  vec_and_ptr_t phi;
  vec_and_ptr_t phi_solid; // LSF for solid domain: -- This will be assigned within the loop as the negative of phi
  vec_and_ptr_t phi_cylinder;   // LSF for the inner cylinder, if applicable (example ICE_OVER_CYLINDER)

  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_cylinder_dd;

  // Interface geometry:------------------------------
  vec_and_ptr_dim_t normal;
  vec_and_ptr_t curvature;

  vec_and_ptr_dim_t liquid_normals;
  vec_and_ptr_dim_t solid_normals;
  vec_and_ptr_dim_t cyl_normals;

  // Poisson problem:---------------------------------
  int cube_refinement = 1;
//  my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
//  my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

  vec_and_ptr_t T_l_n;
  vec_and_ptr_t T_l_nm1;
  vec_and_ptr_t T_l_backtrace;
  vec_and_ptr_t T_l_backtrace_nm1;
  vec_and_ptr_t rhs_Tl;

  vec_and_ptr_t T_s_n;
  vec_and_ptr_t rhs_Ts;

  // Vectors to hold first derivatives of T
  vec_and_ptr_dim_t T_l_d;
  vec_and_ptr_dim_t T_s_d;

  vec_and_ptr_dim_t T_l_dd;
  // Stefan problem:------------------------------------
  vec_and_ptr_dim_t v_interface;;
  vec_and_ptr_dim_t jump;


  // Navier-Stokes problem:-----------------------------
  my_p4est_navier_stokes_t* ns = NULL;
  my_p4est_poisson_cells_t* cell_solver; // TO-DO: These may be unnecessary now
  my_p4est_poisson_faces_t* face_solver;

  PCType pc_face = PCSOR;
  KSPType face_solver_type = KSPBCGS;
  PCType pc_cell = PCSOR;
  KSPType cell_solver_type = KSPBCGS;

  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t press_nodes;

  Vec dxyz_hodge_old[P4EST_DIM];

  my_p4est_cell_neighbors_t *ngbd_c_np1 = NULL;
  my_p4est_faces_t *faces_np1 = NULL;

  // Poisson boundary conditions:
  temperature_field* analytical_T[2];
  external_heat_source* external_heat_source_T[2];

  BC_INTERFACE_VALUE_TEMP* bc_interface_val_temp[2];
  BC_WALL_VALUE_TEMP* bc_wall_value_temp[2];

  // Navier-Stokes boundary conditions: -----------------
  BoundaryConditions2D bc_velocity[P4EST_DIM];
  BoundaryConditions2D bc_pressure;

  BC_interface_value_velocity* bc_interface_value_velocity[P4EST_DIM];
  BC_WALL_VALUE_VELOCITY* bc_wall_value_velocity[P4EST_DIM];
  BC_WALL_TYPE_VELOCITY* bc_wall_type_velocity[P4EST_DIM];

  // Note: Pressure BC objects take no arguments, don't need to be initialized

  external_force_per_unit_volume_component* external_force_components[P4EST_DIM];

  // Coupled/NS boundary conditions:
  velocity_component* analytical_soln_v[P4EST_DIM];

  // Interp method: -------------------------------------
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // Variables for extension band and grid size: ---------

  double dxyz_smallest[P4EST_DIM];
  double dxyz_close_to_interface;

  double min_volume_;
  double extension_band_use_;
  double extension_band_extend_;
  double extension_band_check_;

  // stopwatch
  parStopWatch w;
  w.start("Running example: multialloy_with_fluids");
  // -----------------------------------------------

  for(int grid_res_iter=0;grid_res_iter<=num_splits;grid_res_iter++){
    // Make sure your flags are set to solve at least one of the problems:
    if(!solve_stefan && !solve_navier_stokes){
        throw std::invalid_argument("Woops, you haven't set options to solve either type of physical problem. \n"
                                    "You must at least set solve_stefan OR solve_navier_stokes to true. ");
      }

    // -----------------------------------------------
    // Set up initial grid:
    // -----------------------------------------------
    // domain size information
    set_geometry();
    const int n_xyz[]      = { nx,  ny,  0};
    const double xyz_min[] = {xmin, ymin, 0};
    const double xyz_max[] = {xmax,  ymax,  0};
    const int periodic[]   = { px,  py,  0};

    // Set physical properties:
    set_physical_properties();
    // -----------------------------------------------
    // Set properties for the Navier - Stokes problem (if applicable):
    // -----------------------------------------------
    if(solve_navier_stokes){
        set_NS_info();
        set_nondimensional_groups();
        PetscPrintf(mpi.comm(),"\n Nondim groups are: \n"
                               "Re = %f \n"
                               "Pr = %f \n"
                               "Pe = %f \n"
                               "St = %f \n"
                               "And we have: \n"
                               "u_inf = %0.3e [m/s]\n", Re, Pr, Pe, St,u_inf);

      }

    // Get the simulation time info (it is example dependent): -- Must be set after non dim groups
    simulation_time_info();
    if(solve_navier_stokes)PetscPrintf(mpi.comm(),"Sim time: %0.2f [min] = %0.2f [nondim]\n",tfinal*d_cyl/(60.*u_inf),tfinal);

    // -----------------------------------------------
    // Create the grid:
    // -----------------------------------------------
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Creating the grid ... \n");

    int load_tstep=-1;

    splitting_criteria_cf_and_uniform_band_t sp(lmin+grid_res_iter,lmax+grid_res_iter,&level_set,uniform_band);
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);
    double t_original_start = tstart;

    if(!loading_from_previous_state){
      // Create the p4est at time n:
      p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
      p4est->user_pointer = &sp;

      for(unsigned int l=0;l<lmax+grid_res_iter;l++){
        my_p4est_refine(p4est,P4EST_FALSE,refine_levelset_cf,NULL);
        my_p4est_partition(p4est,P4EST_FALSE,NULL);
      }
      p4est_balance(p4est,P4EST_CONNECT_FULL,NULL);
      my_p4est_partition(p4est,P4EST_FALSE,NULL);

      ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(p4est,ghost);
      nodes = my_p4est_nodes_new(p4est, ghost); //same

      hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
      ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
      ngbd->init_neighbors();

      // Create the p4est at time np1:(this will be modified but is useful for initializing solvers):
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      p4est_np1->user_pointer = &sp;
      my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);

      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(p4est_np1,ghost_np1);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Get the new neighbors:
      hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
      ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

      // Initialize the neigbors:
      ngbd_np1->init_neighbors();
    }
    else{
      p4est=NULL;
      conn=NULL;
      p4est=NULL; ghost=NULL;nodes=NULL;
      hierarchy=NULL;ngbd=NULL;

      phi.vec=NULL;
      T_l_n.vec=NULL; T_l_nm1.vec=NULL;
      T_s_n.vec=NULL;
      foreach_dimension(d){
        v_n.vec[d]=NULL;
        v_nm1.vec[d]=NULL;

      }
      vorticity.vec=NULL;

      const char* load_path = getenv("LOAD_STATE_PATH");
      if(!load_path){
          throw std::invalid_argument("You need to set the  directory for the desired load state");
        }
      PetscPrintf(mpi.comm(),"Load dir is:  %s \n",load_path);

      load_state(mpi,load_path,&sp,p4est,nodes,ghost,conn,
                 &phi.vec,&T_l_n.vec,&T_l_nm1.vec,&T_s_n.vec,
                 v_n.vec,v_nm1.vec,&vorticity.vec);

      PetscPrintf(mpi.comm(),"State was loaded successfully from %s \n",load_path);

      // Update the neigborhood and hierarchy:
      if(hierarchy!=NULL) {
        delete hierarchy;
        }
      if(ngbd!=NULL) {delete ngbd;}

      hierarchy = new my_p4est_hierarchy_t(p4est,ghost,&brick);
      ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
      ngbd->init_neighbors();

      // Create the p4est at time np1 (this will be modified but is useful for initializing solvers):
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      my_p4est_ghost_expand(p4est_np1,ghost_np1);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Get the new neighbors:
      hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
      ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

      // Initialize the neigbors:
      ngbd_np1->init_neighbors();

      // Initialize pressure vector (if navier stokes)
      if(solve_navier_stokes) press_nodes.create(p4est,nodes);

      load_tstep =tstep;
      tstart=tn;
      }

    if(!loading_from_previous_state)tstep = 0;

    PetscPrintf(mpi.comm(),"\nLoading from previous state? %s \n"
                           "Starting timestep = %d \n"
                           "Save state every iter = %d \n"
                           "Save to vtk? %s \n"
                           "Save using %s \n"
                           "Save every dt = %0.2f\n"
                           "Save every iter = %d \n",loading_from_previous_state?"Yes":"No",tstep,save_state_every_iter,save_to_vtk?"Yes":"No",save_using_dt? "dt" :"iter",save_every_dt,save_every_iter);


    // Initialize output file numbering:
    int out_idx = -1;

    // ------------------------------------------------------------
    // Initialize relevant fields:
    // ------------------------------------------------------------
    // Only initialize if we are NOT loading from a previous state
    if(!loading_from_previous_state){
      // LSF:
      if(print_checkpoints) PetscPrintf(mpi.comm(),"Initializing the level set function (s) ... \n");
      phi.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);

      // Temperature fields:
      INITIAL_TEMP *T_init_cf[2];
      temperature_field* analytical_temp[2];
      if(solve_stefan){
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Initializing the temperature fields (s) ... \n");

        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          coupled_test_sign = 1.;
          vel_has_switched=false;

          for(unsigned char d=0;d<2;++d){
            analytical_temp[d]= new temperature_field(d);
            analytical_temp[d]->t = tstart;
          }
          for(unsigned char d=0;d<2;++d){
            T_init_cf[d]= new INITIAL_TEMP(d,analytical_temp);
          }

        }
        else{
          for(unsigned char d=0;d<2;++d){
            T_init_cf[d] = new INITIAL_TEMP(d);
            T_init_cf[d]->t = tstart;
          }
        }

        T_l_n.create(p4est,nodes);
        sample_cf_on_nodes(p4est,nodes,*T_init_cf[LIQUID_DOMAIN],T_l_n.vec);

        T_s_n.create(p4est,nodes);
        sample_cf_on_nodes(p4est,nodes,*T_init_cf[SOLID_DOMAIN],T_s_n.vec);

        if(do_advection && advection_sl_order ==2){
          T_l_nm1.create(p4est,nodes);
          sample_cf_on_nodes(p4est,nodes,*T_init_cf[LIQUID_DOMAIN],T_l_nm1.vec);
        }

        v_interface.create(p4est,nodes);
        foreach_dimension(d){
          sample_cf_on_nodes(p4est,nodes,zero_cf,v_interface.vec[d]);
        }

        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          for(unsigned char d=0;d<2;++d){
            delete analytical_temp[d];
            delete T_init_cf[d];
          }
        }
      }

      // Navier-Stokes fields:
      if(print_checkpoints) PetscPrintf(mpi.comm(),"Initializing the Navier-Stokes fields (s) ... \n");

      INITIAL_VELOCITY *v_init_cf[P4EST_DIM];
      velocity_component* analytical_soln[P4EST_DIM];

      if(solve_navier_stokes){
        if(example_ == NS_GIBOU_EXAMPLE || example_ == COUPLED_PROBLEM_EXAMPLE)
        {
          for(unsigned char d=0;d<P4EST_DIM;++d){
            analytical_soln[d] = new velocity_component(d);
            analytical_soln[d]->t = tstart;
          }
        }
        for(unsigned char d=0;d<P4EST_DIM;++d){
          if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
            v_init_cf[d] = new INITIAL_VELOCITY(d,analytical_soln);
            v_init_cf[d]->t = tstart;
          }
          else {
            v_init_cf[d] = new INITIAL_VELOCITY(d);
          }
        }

        v_n.create(p4est,nodes);
        v_nm1.create(p4est,nodes);
        vorticity.create(p4est,nodes);
        press_nodes.create(p4est,nodes);

        foreach_dimension(d){
          sample_cf_on_nodes(p4est,nodes,*v_init_cf[d],v_n.vec[d]);
          sample_cf_on_nodes(p4est,nodes,*v_init_cf[d],v_nm1.vec[d]);
        }
        sample_cf_on_nodes(p4est,nodes,zero_cf,vorticity.vec);
        sample_cf_on_nodes(p4est,nodes,zero_cf,press_nodes.vec);
      }

      for(unsigned char d=0;d<P4EST_DIM;d++){
        if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
          delete analytical_soln[d];
        }
        delete v_init_cf[d];
      }

      NS_norm = max(u0,v0);
    } // end of (if not loading from previous state)

    // ------------------------------------------------------------
    // Initialize relevant boundary condition objects:
    // ------------------------------------------------------------
    // For NS or coupled case:
    // Create analytical velocity field for each Cartesian direction if needed:
    if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
      for(unsigned char d=0;d<P4EST_DIM;d++){
        analytical_soln_v[d] = new velocity_component(d);
        analytical_soln_v[d]->t = tn+dt;
      }
    }

    // For temperature problem:
    if(solve_stefan){
      // Create analytical temperature field for each domain if needed:
      for(unsigned char d=0;d<2;++d){
        if(example_ == COUPLED_PROBLEM_EXAMPLE){ // TO-DO: make all incrementing consistent
          analytical_T[d] = new temperature_field(d);
          analytical_T[d]->t = tn+dt;
        }
      }
      // Create necessary RHS forcing terms and BC's
      for(unsigned char d=0;d<2;++d){
        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          external_heat_source_T[d] = new external_heat_source(d,analytical_T,analytical_soln_v);
          external_heat_source_T[d]->t = tn+dt;
          bc_interface_val_temp[d] = new BC_INTERFACE_VALUE_TEMP(NULL,NULL,analytical_T,d);
          bc_wall_value_temp[d] = new BC_WALL_VALUE_TEMP(d,analytical_T);
        }
        else{
          bc_interface_val_temp[d] = new BC_INTERFACE_VALUE_TEMP(); // will set proper objects later, can be null on initialization
          bc_wall_value_temp[d] = new BC_WALL_VALUE_TEMP(d);
        }
      }
    }

    // For NS problem:
    BC_INTERFACE_VALUE_PRESSURE bc_interface_value_pressure;
    BC_WALL_VALUE_PRESSURE bc_wall_value_pressure;
    BC_WALL_TYPE_PRESSURE bc_wall_type_pressure;

    if(solve_navier_stokes){
      for(unsigned char d=0;d<P4EST_DIM;d++){
        // Set the BC types:
        BC_INTERFACE_TYPE_VELOCITY(d);
        bc_wall_type_velocity[d] = new BC_WALL_TYPE_VELOCITY(d);

        // Set the BC values (and potential forcing terms) depending on what we are running:
        if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
          // Interface conditions values:
          bc_interface_value_velocity[d] = new BC_interface_value_velocity(d,NULL,NULL,analytical_soln_v);
          bc_interface_value_velocity[d]->t = tn+dt;

          // Wall conditions values:
          bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d,analytical_soln_v);
          bc_wall_value_velocity[d]->t = tn+dt;

          // External forcing terms:
          external_force_components[d] = new external_force_per_unit_volume_component(d,analytical_soln_v);
          external_force_components[d]->t = tn+dt;
        }
        else{
          // Interface condition values:
          bc_interface_value_velocity[d] = new BC_interface_value_velocity(d,NULL,NULL); // initialize null for now, will add relevant neighbors and vector as required later on

          // Wall condition values:
          bc_wall_value_velocity[d] = new BC_WALL_VALUE_VELOCITY(d);
        }
      }
      interface_bc_pressure(); // sets the interfacial bc type for pressure
    }


    // ------------------------------------------------------------
    // Initialize relevant solvers:
    // ------------------------------------------------------------
    // First, initialize the Navier-Stokes solver with the grid:
    vec_and_ptr_dim_t v_n_NS, v_nm1_NS;
//    vec_and_ptr_dim_t v_n_NS(p4est,nodes);
//    vec_and_ptr_dim_t v_nm1_NS(p4est,nodes); // fields for NS solver to own
    if(solve_navier_stokes){
//      PetscPrintf(mpi.comm(),"Initializing the navier stokes solver ... \n");

//      // Initialize faces and cell neighbors: (will be updated in time loop)
//      ngbd_c_n = new my_p4est_cell_neighbors_t(hierarchy);
//      faces_n = new my_p4est_faces_t(p4est,ghost,&brick,ngbd_c_n); // not sure we need to do this

//      // Initialize the NS solver:
//      ns = new my_p4est_navier_stokes_t(ngbd,ngbd,faces_n);

//      // Set the LSF:
//      ns->set_phi(phi.vec);
//      ns->set_dt(dt_nm1,dt);

      // Initialize the velocity objects that the NS solver will own:
/*      foreach_dimension(d){
        ierr = VecCopyGhost(v_n.vec[d],v_n_NS.vec[d]);
        ierr = VecCopyGhost(v_nm1.vec[d],v_nm1_NS.vec[d]);
      }*//*
      ns->set_velocities(v_nm1_NS.vec,v_n_NS.vec);

      // These get passed into the NS solver to handle, and NS solver will handle deleting them

      PetscPrintf(mpi.comm(),"CFL: %0.2f, rho : %0.2f, mu : %0.3e \n",cfl,rho_l,mu_l);

      ns->set_parameters((1./Re),1.0,NS_advection_sl_order,uniform_band,vorticity_threshold,cfl);

      // Set the initial boundary conditions: (for the first timestep) -- need to do this to do proper interpolation to new grid

      for(unsigned char d=0;d<P4EST_DIM;d++){
        if(example_ == ICE_AROUND_CYLINDER){
          bc_interface_value_velocity[d]->set(ngbd_np1,v_interface.vec[d]);
        }
        bc_velocity[d].setInterfaceType(interface_bc_type_velocity[d]);
        bc_velocity[d].setInterfaceValue(*bc_interface_value_velocity[d]);
        bc_velocity[d].setWallValues(*bc_wall_value_velocity[d]);
        bc_velocity[d].setWallTypes(*bc_wall_type_velocity[d]);
      }

      // Set pressure conditions:
      bc_pressure.setInterfaceType(interface_bc_type_pressure);
      bc_pressure.setInterfaceValue(bc_interface_value_pressure);
      bc_pressure.setWallTypes(bc_wall_type_pressure);
      bc_pressure.setWallValues(bc_wall_value_pressure);

      // Set the boundary conditions:
      ns->set_bc(bc_velocity,&bc_pressure);

      CF_DIM *external_forces[P4EST_DIM]=
      {DIM(external_force_components[0],external_force_components[1],external_force_components[2])};
      if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
        ns->set_external_forces(external_forces);
      }

      if(!loading_from_previous_state){
        // Do an initial NS step -- this gets the grid and objects all initialized in a way consistent with how our setup is
        navier_stokes_step(ns,p4est,nodes,
                           v_n.vec,v_nm1.vec,vorticity.vec,press_nodes.vec,
                           face_solver_type,pc_face,cell_solver_type,pc_cell,
                           NULL,NULL,NULL);
      }*/
    }


    // -----------------------------------------------
    // Initialize files to output various data of interest:
    // -----------------------------------------------
    if(print_checkpoints)PetscPrintf(mpi.comm(),"Initializing output files ... \n");
    FILE *fich_stefan_errors;
    char name_stefan_errors[1000];

    FILE *fich_NS_errors;
    char name_NS_errors[1000];

    FILE *fich_coupled_errors;
    char name_coupled_errors[1000];

    FILE *fich_fluid_forces;
    char name_fluid_forces[1000];

    switch(example_){
      case FRANK_SPHERE:{
        // Output file for Frank Sphere errors:
        const char* out_dir_err_stefan = getenv("OUT_DIR_ERR_stefan");
        if(!out_dir_err_stefan){
            throw std::invalid_argument("You need to set the environment variable OUT_DIR_ERR_stefan to save stefan errors");
          }
        sprintf(name_stefan_errors,"%s/frank_sphere_error_lmin_%d_lmax_%d_method_%d.dat",
                out_dir_err_stefan,lmin+grid_res_iter,lmax+grid_res_iter,method_);

        ierr = PetscFOpen(mpi.comm(),name_stefan_errors,"w",&fich_stefan_errors); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(),fich_stefan_errors,"time " "timestep " "iteration "
                                                          "phi_error " "T_l_error " "T_s_error "
                                                          "v_int_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
        ierr = PetscFClose(mpi.comm(),fich_stefan_errors); CHKERRXX(ierr);
        break;
        }
      case NS_GIBOU_EXAMPLE:{
          // Output file for NS test case errors:
          const char* out_dir_err_NS = getenv("OUT_DIR_ERR_NS");
          if(!out_dir_err_NS){
              throw std::invalid_argument("You need to set the environment variable OUT_DIR_ERR_NS to save Navier Stokes errors");
            }
          sprintf(name_NS_errors,"%s/navier_stokes_error_lmin_%d_lmax_%d_advection_order_%d.dat",
                  out_dir_err_NS,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

          ierr = PetscFOpen(mpi.comm(),name_NS_errors,"w",&fich_NS_errors); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_NS_errors,"time " "timestep " "iteration " "u_error "
                                                        "v_error " "P_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_NS_errors); CHKERRXX(ierr);

          break;
        }
      case COUPLED_PROBLEM_EXAMPLE:{
          // Output file for coupled problem test case:
          const char* out_dir_err_coupled = getenv("OUT_DIR_ERR_coupled");
          sprintf(name_coupled_errors,"%s/coupled_error_lmin_%d_lmax_%d_method_%d_advection_order_%d.dat",
                  out_dir_err_coupled,lmin+grid_res_iter,lmax + grid_res_iter,method_,advection_sl_order);

          ierr = PetscFOpen(mpi.comm(),name_coupled_errors,"w",&fich_coupled_errors); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_coupled_errors,"time " "timestep " "iteration "
                                                             "u_error " "v_error " "P_error "
                                                             "Tl_error " "Ts_error " "vint_error" "phi_error "
                                                             "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_coupled_errors); CHKERRXX(ierr);
          break;
        }
      case ICE_AROUND_CYLINDER:{
        if(save_fluid_forces){
          // Output file for NS test case errors:
          const char* out_dir_fluid_forces = getenv("OUT_DIR_NS_FORCES");
          if(!out_dir_fluid_forces){
              throw std::invalid_argument("You need to set the environment variable OUT_DIR_NS_FORCES to save fluid forces");
            }
          sprintf(name_fluid_forces,"%s/fluid_forces_Re_%0.2f_lmin_%d_lmax_%d_advection_order_%d.dat",
                  out_dir_fluid_forces,Re,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

          ierr = PetscFOpen(mpi.comm(),name_fluid_forces,"w",&fich_fluid_forces); CHKERRXX(ierr);
          ierr = PetscFPrintf(mpi.comm(),fich_fluid_forces,"time fx fy \n");CHKERRXX(ierr);
          ierr = PetscFClose(mpi.comm(),fich_fluid_forces); CHKERRXX(ierr);

        }
          break;
        }
    case FLOW_PAST_CYLINDER:{
      if(save_fluid_forces){
        // Output file for NS test case errors:
        const char* out_dir_fluid_forces = getenv("OUT_DIR_NS_FORCES");
        if(!out_dir_fluid_forces){
            throw std::invalid_argument("You need to set the environment variable OUT_DIR_NS_FORCES to save fluid forces");
          }
        sprintf(name_fluid_forces,"%s/fluid_forces_Re_%0.2f_lmin_%d_lmax_%d_advection_order_%d.dat",
                out_dir_fluid_forces,Re,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

        ierr = PetscFOpen(mpi.comm(),name_fluid_forces,"w",&fich_fluid_forces); CHKERRXX(ierr);
        ierr = PetscFPrintf(mpi.comm(),fich_fluid_forces,"time fx fy \n");CHKERRXX(ierr);
        ierr = PetscFClose(mpi.comm(),fich_fluid_forces); CHKERRXX(ierr);
        break;

      }
    }
      default:{
        break;
        }
      }

    // ------------------------------------------------------------
    // Begin stepping through time
    // ------------------------------------------------------------
    for (tn=tstart;tn<tfinal; tn+=dt, tstep++){
      // ------------------------------------------------------------
      // Print iteration information:
      // ------------------------------------------------------------
      ierr = PetscPrintf(mpi.comm(),"\n -------------------------------------------\n"
                                    "Iteration %d , Time: %0.2f [nondim] = Time: %0.3e [nondim] = %0.2f [sec] = %0.2f [min],"
                                    " Timestep: %0.3e [nondim] = %0.1g [sec],"
                                    " Percent Done : %0.2f %"
                                    " \n ------------------------------------------- \n",
                                    tstep,tn,tn,tn*(d_cyl/u_inf),tn*(d_cyl/u_inf)/60.,
                                    dt, dt*(d_cyl/u_inf),
                                    ((tn-t_original_start)/(tfinal-t_original_start))*100.0);

      if(tstep%timing_every_n == 0) {
        PetscPrintf(mpi.comm(),"Current time info : \n");
        w.read_duration_current();
      }
      if(solve_stefan){
          if(v_interface_max_norm>v_int_max_allowed){
              PetscPrintf(mpi.comm(),"Interfacial velocity has exceeded its max allowable value \n"
                                     "Max allowed is : %g \n",v_int_max_allowed);
              MPI_Abort(mpi.comm(),1);
            }
        }
      // ------------------------------------------------------------
      // Define some variables needed to specify how to extend across the interface:
      // ------------------------------------------------------------
        // Get smallest grid size:
        dxyz_min(p4est,dxyz_smallest);
        dxyz_close_to_interface = 1.2*max(dxyz_smallest[0],dxyz_smallest[1]);
        min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);
        extension_band_use_    = (8.)*pow(min_volume_, 1./ double(P4EST_DIM)); //8
        extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
        extension_band_check_  = (6.)*pow(min_volume_, 1./ double(P4EST_DIM)); // 6

        if((tstep ==0) && example_ == ICE_AROUND_CYLINDER && solve_coupled){
            double delta_r = r0 - r_cyl;
            PetscPrintf(mpi.comm(),"The uniform band is %0.2f\n",uniform_band);
//            if(delta_r<4.*dxyz_close_to_interface ){
//                PetscPrintf(mpi.comm()," Your initial delta_r is %0.3e, and it must be at least %0.3e \n",delta_r,4.*dxyz_close_to_interface);
//                SC_ABORT("Your initial delta_r is too small \n");
//              }
          }

        // -------------------------------
        // If first iteration, perturb the LSF(s):
        // -------------------------------
        my_p4est_level_set_t ls(ngbd);
        if(tstep<1){
            // Perturb the LSF on the first iteration
            ls.perturb_level_set_function(phi.vec,EPS);
          }

        // ------------------------------------------------------------
        // Extend Fields Across Interface (if solving Stefan):
        // -- Note: we do not extend NS velocity fields bc NS solver handles that internally
        // ------------------------------------------------------------
        if(solve_stefan){
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning field extension \n");
          // -------------------------------
          // Create all fields for this procedure:
          // -------------------------------
          phi_solid.create(p4est,nodes);
          liquid_normals.create(p4est,nodes);
          solid_normals.create(p4est,nodes);

          // -------------------------------
          // Get the solid LSF:
          // -------------------------------
          VecCopyGhost(phi.vec,phi_solid.vec);
          VecScaleGhost(phi_solid.vec,-1.0);

          // -------------------------------
          // Compute normals for each domain:
          // -------------------------------

          compute_normals(*ngbd,phi.vec,liquid_normals.vec);

          foreach_dimension(d){
            VecCopyGhost(liquid_normals.vec[d],solid_normals.vec[d]);
            VecScaleGhost(solid_normals.vec[d],-1.0);
          }

          // -------------------------------
          // Extend Temperature Fields across the interface:
          // -------------------------------
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling extension over phi \n");
          ls.extend_Over_Interface_TVD_Full(phi.vec, T_l_n.vec,
                                            50, 2, 1.e-15,
                                            extension_band_use_, extension_band_extend_,
                                            extension_band_check_,
                                            liquid_normals.vec, NULL,
                                            NULL, false, NULL,NULL);

          ls.extend_Over_Interface_TVD_Full(phi_solid.vec, T_s_n.vec,
                                            50, 2, 1.e-15,
                                            extension_band_use_, extension_band_extend_,
                                            extension_band_check_,
                                            solid_normals.vec, NULL,
                                            NULL, false, NULL, NULL);

          if(example_ == ICE_AROUND_CYLINDER){
            phi_cylinder.create(p4est,nodes);
            cyl_normals.create(p4est,nodes);

            sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
            compute_normals(*ngbd,phi_cylinder.vec,cyl_normals.vec);

            if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling extension over phi_cylinder \n");
            ls.extend_Over_Interface_TVD_Full(phi_cylinder.vec, T_s_n.vec,
                                              50, 2, 1.e-15,
                                              0.5*extension_band_use_, 0.5*extension_band_extend_, 0.5*extension_band_check_,
                                              cyl_normals.vec, NULL, NULL,
                                              false, NULL, NULL);

            cyl_normals.destroy();
            phi_cylinder.destroy();
          }

          // -------------------------------
          // Delete fields now:
          // -------------------------------
          liquid_normals.destroy();
          solid_normals.destroy();
          phi_solid.destroy();

        } // end of "if solve stefan"
        // --------------------------------------------------------------------------------------------------------------
        // Save simulation state every specified number of iterations
        // --------------------------------------------------------------------------------------------------------------
        if(tstep>0 && ((tstep%save_state_every_iter)==0) && tstep!=load_tstep){
            char output[1000];
            const char* out_dir_coupled = getenv("OUT_DIR_SAVE_STATE");
            if(!out_dir_coupled){
                throw std::invalid_argument("You need to set the output directory for save states: OUT_DIR_SAVE_STATE");
              }
            sprintf(output,
                    "%s/save_states_output_lmin_%d_lmax_%d_advection_order_%d_example_%d",
                    out_dir_coupled,
                    lmin+grid_res_iter,lmax+grid_res_iter,
                    advection_sl_order,example_);

            save_state(mpi,output,num_save_states,
                       &sp,p4est,nodes,
                       phi.vec,T_l_n.vec,T_l_nm1.vec,T_s_n.vec,
                       v_n.vec,v_nm1.vec,vorticity.vec);

            PetscPrintf(mpi.comm(),"Simulation state was saved . \n");
          }

        // --------------------------------------------------------------------------------------------------------------
        // Saving to VTK: either every specified number of iterations, or every specified dt:
        // Note: we do this after extension of fields to make visualization nicer
        // --------------------------------------------------------------------------------------------------------------
        bool are_we_saving = false;
        are_we_saving = are_we_saving_vtk(tstep,tn,tstep==load_tstep,out_idx,true);

        // Save to VTK if we are saving this timestep:
        if(are_we_saving){
          PetscPrintf(mpi.comm(),"Saving to vtk... \n");

          char output[1000];
          if(save_coupled_fields){
              const char* out_dir_coupled = getenv("OUT_DIR_VTK_coupled");
              if(!out_dir_coupled){
                  throw std::invalid_argument("You need to set the output directory for coupled VTK: OUT_DIR_VTK_coupled");
                }

             if(example_ !=COUPLED_PROBLEM_EXAMPLE){
                // Create the cylinder just for visualization purposes, then destroy after saving
                phi_cylinder.create(p4est,nodes);
                sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);

                sprintf(output,"%s/snapshot_example_%d_lmin_%d_lmax_%d_outidx_%d",out_dir_coupled,example_,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
                save_everything(p4est,nodes,ghost,ngbd,phi,phi_cylinder,T_l_n,T_s_n,v_interface,v_n,press_nodes,vorticity,output);

                phi_cylinder.destroy();
              }
          }
          if(save_navier_stokes){
              const char* out_dir_ns = getenv("OUT_DIR_VTK_NS");
              if(example_ != NS_GIBOU_EXAMPLE){
                sprintf(output,"%s/snapshot_example_%d_lmin_%d_lmax_%d_outidx_%d",out_dir_ns,example_,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
                save_navier_stokes_fields(p4est,nodes,ghost,phi,v_n,press_nodes,vorticity,output);
              }
            }
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes saving to VTK \n");

        } // end of if "are we saving"

        // Check errors on NS validation case if relevant, save errors to vtk if we are saving this timestep

        if(example_ == NS_GIBOU_EXAMPLE){
            const char* out_dir_ns = getenv("OUT_DIR_VTK_NS");
            char output[1000];
            PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

            sprintf(output,"%s/snapshot_NS_Gibou_test_lmin_%d_lmax_%d_outidx_%d",out_dir_ns,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
            PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

            if(tstep>0){
              // In typical saving, only compute pressure nodes when we save to vtk. For this example, save pressure nodes every time so we can check the error
              press_nodes.destroy();press_nodes.create(p4est,nodes);
              PetscPrintf(mpi.comm(),"Computed pressure at nodes \n");
              ns->compute_pressure_at_nodes(&press_nodes.vec);
            }

            save_navier_stokes_test_case(p4est,nodes,ghost,phi,v_n,press_nodes,vorticity,dxyz_close_to_interface,are_we_saving,output,name_NS_errors,fich_NS_errors);
          }
        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          const char* out_dir_coupled = getenv("OUT_DIR_VTK_coupled");

          char output[1000];
          PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

          sprintf(output,"%s/snapshot_coupled_test_lmin_%d_lmax_%d_outidx_%d",out_dir_coupled,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);

          if(tstep>0){
            save_coupled_test_case(p4est,nodes,ghost,phi,T_l_n,T_s_n,v_interface,v_n,press_nodes,vorticity,dxyz_close_to_interface,are_we_saving,output,name_coupled_errors,fich_coupled_errors); // Don't check first timestep bc have not computed velocity yet
          }
        }

        // --------------------------------------------------------------------------------------------------------------
        // Compute the interfacial velocity and timestep (Stefan):
        // --------------------------------------------------------------------------------------------------------------
        dt_nm1 = dt;
        char stefan_timestep[1000];
        if(solve_stefan){
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing interfacial velocity ... \n");
          // Get the first derivatives to compute the jump
          T_l_d.create(p4est,nodes); T_s_d.create(T_l_d.vec);
          ngbd->first_derivatives_central(T_l_n.vec,T_l_d.vec);
          ngbd->first_derivatives_central(T_s_n.vec,T_s_d.vec);

          // Create vector to hold the jump values:
          jump.create(p4est,nodes);
          v_interface.destroy();
          v_interface.create(p4est,nodes);

          compute_interfacial_velocity(T_l_d,T_s_d,jump,v_interface,phi,ngbd,dxyz_close_to_interface);

          // Scale v_interface computed by appropriate sign if we are doing the coupled test case:
          if(example_ == COUPLED_PROBLEM_EXAMPLE){
            foreach_dimension(d){
             VecScaleGhost(v_interface.vec[d],coupled_test_sign);
            }
          }

          // Destroy values once no longer needed:
          T_l_d.destroy();
          T_s_d.destroy();
          jump.destroy();

          // Compute timestep:
          compute_timestep(v_interface, phi, dxyz_close_to_interface, dxyz_smallest,nodes,p4est); // this function modifies the variable dt

          sprintf(stefan_timestep,"Computed interfacial velocity: \n"
                                  " - Computational : %0.3e  - Physical : %0.3e [m/s]  - Physical : %0.3e  [mm/s] \n",v_interface_max_norm,v_interface_max_norm*u_inf,v_interface_max_norm*u_inf*1000.);
          }
        // Take NS timestep into account if relevant:
        if(solve_navier_stokes){
            // Determine the timestep depending on timestep restrictions from both NS solver and from the Stefan problem
            if(solve_stefan){
                if(tstep==load_tstep){dt_NS=dt_nm1;} // TO-DO: not sure this logic is 100% correct, what about NS only case?
                dt = min(dt,dt_NS);
              }
            else{
                // If we are only solving Navier Stokes
                dt = dt_NS;
              }
          }
        PetscPrintf(mpi.comm(),"\n"
                               "%s \n"
                               "Computed timestep: \n"
                               " - dt used: %0.3e -dt_NS : %0.3e  -dxyz close to interface : %0.3e \n \n",solve_stefan?stefan_timestep:"",dt,dt_NS,dxyz_close_to_interface);

        // Clip the timestep if we are near the end of our simulation, to get the proper end time:
        bool is_last_step = false;
        if(tn + dt > tfinal){
            dt = tfinal - tn;
            is_last_step=true;
          }

        // Clip time and switch vel direction for coupled problem example:
        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          if((tn+dt>=tfinal/2.0) && !vel_has_switched){
            dt = (tfinal/2.0) - tn;
            coupled_test_switch_sign();
            vel_has_switched=true;
          }
        }

        // --------------------------------------------------------------------------------------------------------------
        // Advance the LSF/Update the grid :
        // --------------------------------------------------------------------------------------------------------------
        /* In Coupled case: advect the LSF and update the grid according to vorticity, d2T/dd2, and phi
         * In Stefan case:  advect the LSF and update the grid according to phi
         * In NS case:      update the grid according to phi (no advection)
        */
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Updating grid ... \n");

        // -------------------------------
        // Create the semi-lagrangian object and do the advection:
        // -------------------------------

        my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

        // -------------------------------
        // Prepare refinement tool necessary information
        // -------------------------------

        bool use_block = false;
        bool expand_ghost_layer = true;
        double threshold = vorticity_threshold; // originally was set to 0.

        std::vector<compare_option_t> compare_opn;
        std::vector<compare_diagonal_option_t> diag_opn;
        std::vector<double> criteria;

        PetscInt num_fields = 0;
        if(solve_navier_stokes) num_fields+=1;// for vorticity
        if(refine_by_d2T)num_fields+=2; // for second derivatives of temperature

        // Create array of fields we wish to refine by, to pass to the refinement tools
        Vec fields_[num_fields];

        // Create vectors for our refinement fields:
        if(solve_navier_stokes && (num_fields!=0)){
          // Only use values of vorticity and d2T in the positive subdomain for refinement:
          vorticity_refine.create(p4est,nodes);

          if(refine_by_d2T){
              T_l_dd.create(p4est,nodes);
              ngbd->second_derivatives_central(T_l_n.vec,T_l_dd.vec);
            }

          // Prepare refinement fields:
          prepare_refinement_fields(phi,vorticity,vorticity_refine,T_l_dd,ngbd);

          // Add our refinement fields to the array:
          PetscInt fields_idx = 0;
          fields_[fields_idx++] = vorticity_refine.vec;
          if(refine_by_d2T){
              fields_[fields_idx++] = T_l_dd.vec[0];
              fields_[fields_idx++] = T_l_dd.vec[1];
            }

          P4EST_ASSERT(fields_idx ==num_fields);

          // Add our instructions:
          // Coarsening instructions: (for vorticity)
          compare_opn.push_back(LESS_THAN);
          diag_opn.push_back(DIVIDE_BY);
          criteria.push_back(threshold*NS_norm/2.);

          // Refining instructions: (for vorticity)
          compare_opn.push_back(GREATER_THAN);
          diag_opn.push_back(DIVIDE_BY);
          criteria.push_back(threshold*NS_norm);

          if(refine_by_d2T){
            double dTheta = (theta_wall - theta_interface)/(min(dxyz_smallest[0],dxyz_smallest[1])); // max dTheta in liquid subdomain

            // Coarsening instructions: (for dT/dx)
            compare_opn.push_back(SIGN_CHANGE);
            diag_opn.push_back(DIVIDE_BY);
            criteria.push_back(dTheta*gradT_threshold); // did 0.1* () for the coarsen if no sign change OR below threshold case

            // Refining instructions: (for dT/dx)
            compare_opn.push_back(SIGN_CHANGE);
            diag_opn.push_back(DIVIDE_BY);
            criteria.push_back(dTheta*gradT_threshold);

            // Coarsening instructions: (for dT/dy)
            compare_opn.push_back(SIGN_CHANGE);
            diag_opn.push_back(DIVIDE_BY);
            criteria.push_back(dTheta*gradT_threshold*0.1);

            // Refining instructions: (for dT/dy)
            compare_opn.push_back(SIGN_CHANGE);
            diag_opn.push_back(DIVIDE_BY);
            criteria.push_back(dTheta*gradT_threshold); // doesnt get used
            }
          } // end of "if solve navier stokes and num_fields!=0"

        // -------------------------------
        // Call grid advection and update:
        // -------------------------------

        if(solve_stefan){
          // Create second derivatives for phi in the case that we are using update_p4est:
          if(solve_stefan){
            phi_dd.create(p4est,nodes);
            ngbd->second_derivatives_central(phi.vec,phi_dd.vec);

            // Get inner cylinder LSF if needed
            if(example_ == ICE_AROUND_CYLINDER){
              phi_cylinder.create(p4est,nodes); // create to refine around, then will destroy
              sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
              }
            }
          // Call advection and refinement
          sl.update_p4est(v_interface.vec, dt,
                        phi.vec, phi_dd.vec, (example_==ICE_AROUND_CYLINDER) ? phi_cylinder.vec: NULL,
                        num_fields ,use_block ,true,
                        uniform_band,uniform_band*(1.5),
                        fields_ ,NULL,
                        criteria,compare_opn,diag_opn,
                        expand_ghost_layer);

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Grid update completed \n");

        // Destroy 2nd derivatives of LSF now that not needed
        phi_dd.destroy();

        // Destroy cylinder LSF if it was created, now that it is not needed:
        if(example_==ICE_AROUND_CYLINDER){ phi_cylinder.destroy();}
        }
        else {
            // NS only case --> no advection --> do grid update iteration manually:
            splitting_criteria_tag_t sp_NS(sp.min_lvl,sp.max_lvl,sp.lip);

            // Create a new vector which will hold the updated values of the fields -- since we will interpolate with each grid iteration
            Vec fields_new_[num_fields];
            if(num_fields!=0)
              {
                for(unsigned int k = 0;k<num_fields; k++){
                    ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
                    ierr = VecCopyGhost(fields_[k],fields_new_[k]);
                  }
              }

            // Create a vector which will hold the updated values of the LSF:
            vec_and_ptr_t phi_new;
            phi_new.create(p4est,nodes);
            ierr = VecCopyGhost(phi.vec,phi_new.vec);

            bool is_grid_changing = true;
            int no_grid_changes = 0;
            bool last_grid_balance = false;
            while(is_grid_changing){
                if(!last_grid_balance){
                    is_grid_changing = sp_NS.refine_and_coarsen(p4est_np1,nodes_np1,phi_new.vec,num_fields,use_block,true,uniform_band,uniform_band*1.5,fields_new_,NULL,criteria,compare_opn,diag_opn);

                    if(no_grid_changes>0 && !is_grid_changing){
                        last_grid_balance = true; // if the grid isn't changing anymore but it has changed, we need to do one more special interp of fields and balancing of the grid
                      }
                  }

                if(is_grid_changing || last_grid_balance){
                    no_grid_changes++;
                    PetscPrintf(mpi.comm(),"NS grid changed %d times \n",no_grid_changes);
                    if(last_grid_balance){
                        p4est_balance(p4est_np1,P4EST_CONNECT_FULL,NULL);
                        PetscPrintf(mpi.comm(),"Does last grid balance \n");
                      }

                    my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);
                    p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
                    my_p4est_ghost_expand(p4est_np1,ghost_np1);
                    p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);

                    // Destroy fields_new and create it on the new grid:
                    if(num_fields!=0){
                        for(unsigned int k = 0; k<num_fields; k++){
                            ierr = VecDestroy(fields_new_[k]);
                            ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
                        }
                    }
                    phi_new.destroy();
                    phi_new.create(p4est_np1,nodes_np1);

                    // Interpolate fields onto new grid:
                    my_p4est_interpolation_nodes_t interp_refine_and_coarsen(ngbd);
                    double xyz_interp[P4EST_DIM];
                    foreach_node(n,nodes_np1){
                      node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz_interp);
                      interp_refine_and_coarsen.add_point(n,xyz_interp);
                    }
                    if(num_fields!=0){
                        interp_refine_and_coarsen.set_input(fields_,quadratic_non_oscillatory_continuous_v2,num_fields);
                        // Interpolate fields
                        interp_refine_and_coarsen.interpolate(fields_new_);
                      }
                    interp_refine_and_coarsen.set_input(phi.vec,quadratic_non_oscillatory_continuous_v2);
                    interp_refine_and_coarsen.interpolate(phi_new.vec);

                    if(last_grid_balance){
                        last_grid_balance = false;
                      }
                  } // End of if grid is changing

                // Do last balancing of the grid, and final interp of phi:
                if(no_grid_changes>10) {PetscPrintf(mpi.comm(),"NS grid did not converge!\n"); break;}
              } // end of while grid is changing

            // Update the LSF accordingly:
            phi.destroy();
            phi.create(p4est_np1,nodes_np1);
            ierr = VecCopyGhost(phi_new.vec,phi.vec);

            // Destroy the vectors we created for refine and coarsen:
            for(unsigned int k = 0;k<num_fields; k++){
                ierr = VecDestroy(fields_new_[k]);
              }
            phi_new.destroy();
          } // end of if only navier stokes

        // -------------------------------
        // Destroy refinement fields now that they're not in use:
        // -------------------------------
        if(solve_navier_stokes){
            vorticity_refine.destroy();
            if(refine_by_d2T){T_l_dd.destroy();}
          }
        // -------------------------------
        // Clear up the memory from the std vectors holding refinement info:
        // -------------------------------

        compare_opn.clear(); diag_opn.clear(); criteria.clear();
        compare_opn.shrink_to_fit(); diag_opn.shrink_to_fit(); criteria.shrink_to_fit();

        // -------------------------------
        // Update hierarchy and neighbors to match new updated grid:
        // -------------------------------

        hierarchy_np1->update(p4est_np1,ghost_np1);
        ngbd_np1->update(hierarchy_np1,nodes_np1);

        // Initialize the neigbors:
        ngbd_np1->init_neighbors();

        // -------------------------------
        // Reinitialize the LSF on the new grid (if it has been advected):
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Reinitializing LSF... \n");

        my_p4est_level_set_t ls_new(ngbd_np1);
        if(solve_stefan)ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec);
        if(solve_navier_stokes && !solve_stefan){
            // If only solving Navier-Stokes, only need to do this once, not every single timestep
            if(tstep==0){ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec);
              }
          }
        // Perturb LSF if this is first timestep:
        if(tstep==0){ls_new.perturb_level_set_function(phi.vec,EPS);}

        // --------------------------------------------------------------------------------------------------------------
        // Interpolate Values onto New Grid:
        // -------------------------------------------------------------------------------------------------------------

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Interpolating fields to new grid ... \n");

        interpolate_values_onto_new_grid(&T_l_n.vec,&T_s_n.vec,
                                         v_interface.vec,v_n.vec,
                                         nodes_np1,p4est_np1,ngbd,interp_bw_grids);


        // -------------------------------------------------------------------------------------------------------------
        // Setup RHS and BC objects for both Poisson and NS (keep it all in one place):
        // -------------------------------------------------------------------------------------------------------------
        // Update analytical velocity for coupled problem example:
        // -------------------------------
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting up appropriate boundary conditions... \n");

        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          for(unsigned char d=0;d<P4EST_DIM;d++){
            analytical_soln_v[d]->t = tn+dt;
          }
        }
        // -------------------------------
        // Update BC objects for stefan problem:
        // -------------------------------

        if(solve_stefan){
          if(example_ == COUPLED_PROBLEM_EXAMPLE){
            for(unsigned char d=0;d<2;d++){
              analytical_T[d]->t = tn+dt;
              bc_interface_val_temp[d]->t = tn+dt;
              bc_wall_value_temp[d]->t = tn+dt;
              external_heat_source_T[d]->t = tn+dt;
            }
          }
        } // If not, we use the curvature and neighbors, but we have to wait till curvature is computed in Poisson step to apply this, so it is applied later
        // -------------------------------
        // Update BC and RHS objects for navier-stokes problem:
        // -------------------------------

        if(solve_navier_stokes){
          // Setup velocity conditions
          for(unsigned char d=0;d<P4EST_DIM;d++){
            if(example_ == ICE_AROUND_CYLINDER){
              bc_interface_value_velocity[d]->set(ngbd_np1,v_interface.vec[d]);
            }
            bc_interface_value_velocity[d]->t = tn+dt;
            bc_wall_value_velocity[d]->t = tn+dt;

            bc_velocity[d].setInterfaceType(interface_bc_type_velocity[d]);
            bc_velocity[d].setInterfaceValue(*bc_interface_value_velocity[d]);
            bc_velocity[d].setWallValues(*bc_wall_value_velocity[d]);
            bc_velocity[d].setWallTypes(*bc_wall_type_velocity[d]);
          }
          // Setup pressure conditions:
          bc_pressure.setInterfaceType(interface_bc_type_pressure);
          bc_pressure.setInterfaceValue(bc_interface_value_pressure);
          bc_pressure.setWallTypes(bc_wall_type_pressure);
          bc_pressure.setWallValues(bc_wall_value_pressure);

          // Set external_forces if applicable
          if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
            foreach_dimension(d){
              external_force_components[d]->t = tn+dt;
            }
          }
        }
        // ------------------------------------------------------------
        // Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
        // ------------------------------------------------------------

        if(solve_stefan){ // mostly memory safe (may have tiniest leak TO-DO)
          // Create all vectors that will be used
          // strictly for the stefan step
          // (aka created and destroyed in stefan step)
          // -------------------------------

          // Solid LSF:
          phi_solid.create(p4est_np1,nodes_np1);

          //Curvature and normal for BC's and setting up solver:
          normal.create(p4est_np1,nodes_np1);
          curvature.create(p4est_np1,nodes_np1);

          // Second derivatives of LSF's (for solver):
          phi_solid_dd.create(p4est_np1,nodes_np1);
          phi_dd.create(p4est_np1,nodes_np1);

          if(example_ == ICE_AROUND_CYLINDER){
            phi_cylinder.create(p4est_np1,nodes_np1);
            phi_cylinder_dd.create(p4est_np1,nodes_np1);
          }
          if(do_advection){
            T_l_backtrace.create(p4est_np1,nodes_np1);
            if(advection_sl_order ==2){
                T_l_backtrace_nm1.create(p4est_np1,nodes_np1);
              }

          }
          // Create arrays to hold the RHS:
          rhs_Tl.create(p4est_np1,nodes_np1);
          rhs_Ts.create(p4est_np1,nodes_np1);

          // -------------------------------
          // Compute the normal and curvature of the interface
          //-- curvature is used in some of the interfacial boundary condition(s) on temperature
          // -------------------------------

          if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing normal and curvature ... \n");

          // Compute normals on the interface:
          compute_normals(*ngbd_np1,phi.vec,normal.vec);

          // Compute curvature on the interface:
          my_p4est_level_set_t ls_new_new(ngbd_np1);
          compute_curvature(phi,normal,curvature,ngbd_np1,ls_new_new); // TO-DO: don't need to do this for coupled problem example

          // Feed the curvature computed to the interfacial boundary condition:
          if(example_ !=COUPLED_PROBLEM_EXAMPLE){
            for(unsigned char d=0;d<2;d++){
              bc_interface_val_temp[d]->set(ngbd_np1,curvature.vec);
            }
          }
          // -------------------------------
          // Get most updated derivatives of the LSF's (on current grid)
          // -------------------------------
          if(print_checkpoints)PetscPrintf(mpi.comm(),"Beginning Poisson problem ... \n");

          // Get the new solid LSF:
          VecScaleGhost(phi.vec,-1.0);
          VecCopyGhost(phi.vec,phi_solid.vec);
          VecScaleGhost(phi.vec,-1.0);

          // Get derivatives of liquid and solid LSF's
          if (print_checkpoints) PetscPrintf(mpi.comm(),"New solid LSF acquired \n");
          ngbd_np1->second_derivatives_central(phi.vec,phi_dd.vec);
          ngbd_np1->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

          // Get inner LSF and derivatives if required:
          if(example_ ==ICE_AROUND_CYLINDER){
              sample_cf_on_nodes(p4est_np1,nodes_np1,mini_level_set,phi_cylinder.vec);
              ngbd_np1->second_derivatives_central(phi_cylinder.vec,phi_cylinder_dd.vec);
            }

          // -------------------------------
          // Compute advection terms (if applicable):
          // -------------------------------
          if (do_advection){
              if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing advection terms ... \n");
              do_backtrace(T_l_n,T_l_nm1,
                           T_l_backtrace,T_l_backtrace_nm1,
                           v_n,v_nm1,
                           p4est_np1,nodes_np1,ngbd_np1,
                           p4est,nodes,ngbd);
              // Do backtrace with v_n --> navier-stokes fluid velocity
          } // end of do_advection if statement

          // -------------------------------
          // Set up the RHS for Poisson step:
          // -------------------------------
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting up RHS for Poisson problem ... \n");

          setup_rhs(phi,T_l_n,T_s_n,
                    rhs_Tl,rhs_Ts,
                    T_l_backtrace,T_l_backtrace_nm1,
                    p4est_np1,nodes_np1,ngbd_np1,external_heat_source_T);

          // -------------------------------
          // Execute the Poisson step:
          // -------------------------------

          // Slide Temp fields:
          if(do_advection && advection_sl_order==2){
            T_l_nm1.destroy();
            T_l_nm1.create(p4est_np1,nodes_np1);
            ierr = VecCopyGhost(T_l_n.vec,T_l_nm1.vec);CHKERRXX(ierr);
          }
          // Solve Poisson problem:
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Poisson problem solution step... \n");

          poisson_step(phi.vec,phi_solid.vec,
                       phi_dd.vec,phi_solid_dd.vec,
                       &T_l_n.vec,&T_s_n.vec,
                       rhs_Tl.vec,rhs_Ts.vec,
                       bc_interface_val_temp,bc_wall_value_temp,
                       ngbd_np1,cube_refinement,
                       (example_==ICE_AROUND_CYLINDER)? phi_cylinder.vec:NULL,
                       (example_==ICE_AROUND_CYLINDER)? phi_cylinder_dd.vec:NULL);
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Poisson step completed ... \n");


          // -------------------------------
          // Destroy all vectors
          // that were used strictly for the
          // stefan step (aka created and destroyed in stefan step)
          // -------------------------------
          // Solid LSF:
          phi_solid.destroy();

          // Curvature and normal for BC's and setting up solver:
          normal.destroy();
          curvature.destroy();

          // Second derivatives of LSF's (for solver):
          phi_solid_dd.destroy();
          phi_dd.destroy();

          if(example_ == ICE_AROUND_CYLINDER){
            phi_cylinder.destroy();
            phi_cylinder_dd.destroy();
          }
          if(do_advection){
            T_l_backtrace.destroy();
            if(advection_sl_order ==2){
                T_l_backtrace_nm1.destroy();
              }
          }
          // Destroy arrays to hold the RHS:
          rhs_Tl.destroy();
          rhs_Ts.destroy();

          // -------------------------------
          // Clear interfacial BC if needed
          // -------------------------------
          if(example_ != COUPLED_PROBLEM_EXAMPLE){
            for(unsigned char d=0;d<2;++d){
              bc_interface_val_temp[d]->clear();
            }
          }
          // -------------------------------
          // Check Frank Sphere error if relevant
          // -------------------------------
          // Check error on the Frank sphere, if relevant:
          if(example_ == FRANK_SPHERE){
              const char* out_dir_stefan = getenv("OUT_DIR_VTK_stefan");

              char output[1000];

              sprintf(output,"%s/snapshot_Frank_Sphere_test_lmin_%d_lmax_%d_outidx_%d",out_dir_stefan,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
              PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin+grid_res_iter,lmax+grid_res_iter);

              save_stefan_test_case(p4est_np1,nodes_np1,ghost_np1,T_l_n, T_s_n, phi, v_interface, dxyz_close_to_interface,are_we_saving,output,name_stefan_errors,fich_stefan_errors);
            }
        } // end of "if solve stefan"

        // --------------------------------------------------------------------------------------------------------------
        // Navier-Stokes Problem: Setup and solve a NS problem in the liquid subdomain
        // --------------------------------------------------------------------------------------------------------------
        if (solve_navier_stokes){

          // -------------------------------
          // Update the grid (or initialize the solver)
          // -------------------------------
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling the Navier-Stokes grid update... \n");
          if((tstep<1) || (tstep==load_tstep)){
            PetscPrintf(mpi.comm(),"Initializing Navier-Stokes solver \n");
            v_n_NS.create(p4est_np1,nodes_np1);
            v_nm1_NS.create(p4est,nodes);

            foreach_dimension(d){
              ierr = VecCopyGhost(v_nm1.vec[d],v_nm1_NS.vec[d]); CHKERRXX(ierr);
              ierr = VecCopyGhost(v_n.vec[d],v_n_NS.vec[d]); CHKERRXX(ierr);
            }

            initialize_ns_solver(ns,p4est_np1,ghost_np1,ngbd_np1,ngbd,
                                 hierarchy_np1,&brick,
                                 phi.vec,v_n_NS.vec,v_nm1_NS.vec,
                                 faces_np1,ngbd_c_np1);

            PetscPrintf(mpi.comm(),"Initialized \n");

          }
          else{
            ns->update_from_tn_to_tnp1_grid_external(phi.vec,
                                                     p4est_np1,nodes_np1,ghost_np1,
                                                     ngbd_np1,
                                                     faces_np1,ngbd_c_np1,
                                                     hierarchy_np1);
          }

          // NOTE: we update NS grid first, THEN set new BCs and forces. This is because the update grid interpolation of the hodge variable
          // requires knowledge of the boundary conditions from that same timestep (the previous one, in our case)
          // -------------------------------
          // Set the timestep: // change to include both timesteps (dtnm1,dtn)
          // -------------------------------
          if(advection_sl_order ==2){
              ns->set_dt(dt_nm1,dt);
            }
          else{
              ns->set_dt(dt);
            }
          // -------------------------------
          // Set BC's and external forces if relevant
          // -------------------------------
          // Set the boundary conditions:
          ns->set_bc(bc_velocity,&bc_pressure);

          // Set the RHS:
          if((example_ == NS_GIBOU_EXAMPLE) || (example_ == COUPLED_PROBLEM_EXAMPLE)){
            CF_DIM *external_forces[P4EST_DIM]=
            {DIM(external_force_components[0],external_force_components[1],external_force_components[2])};
            ns->set_external_forces(external_forces);
          }

          // -------------------------------
          // Prepare vectors to receive solution for np1 timestep:
          // -------------------------------

          v_n.destroy();v_n.create(p4est_np1,nodes_np1);
          v_nm1.destroy();v_nm1.create(p4est_np1,nodes_np1);
          vorticity.destroy();vorticity.create(p4est_np1,nodes_np1);
          press_nodes.destroy();press_nodes.create(p4est_np1,nodes_np1);

          // -------------------------------
          // Solve the Navier-Stokes problem:
          // -------------------------------
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Navier-Stokes solution step... \n");

          bool compute_pressure_to_save = false;
          compute_pressure_to_save = are_we_saving_vtk(tstep + 1,tn + dt, false,out_idx,false);
          PetscPrintf(mpi.comm(),"Compute pressure to save? %s \n",compute_pressure_to_save?"Yes":"No");
          // Check if we are going to be saving to vtk for the next timestep... if so, we will compute pressure at nodes for saving

          navier_stokes_step(ns,p4est_np1,nodes_np1,
                             v_n.vec,v_nm1.vec,vorticity.vec,press_nodes.vec,
                             face_solver_type,pc_face,cell_solver_type,pc_cell,
                             faces_np1, compute_pressure_to_save,
                             save_fluid_forces? name_fluid_forces:NULL,
                             save_fluid_forces? fich_fluid_forces:NULL);

          // -------------------------------
          // Update timestep info as needed
          // -------------------------------
          if(dt_NS>dt_max_allowed) dt_NS = dt_max_allowed;

          // -------------------------------
          // Clear out the interfacial BC for the next timestep, if needed
          // -------------------------------
          if(example_ == ICE_AROUND_CYLINDER){
            for(unsigned char d=0;d<P4EST_DIM;d++){
              bc_interface_value_velocity[d]->clear();
              }
          }

          if(print_checkpoints) PetscPrintf(mpi.comm(),"Completed Navier-Stokes step \n");
      } // End of "if solve navier stokes"

        // --------------------------------------------------------------------------------------------------------------
        // Delete the old grid:
        // --------------------------------------------------------------------------------------------------------------
        // Destroy p4est at n and slide grids:
        // -------------------------------

        if(!is_last_step){
          p4est_destroy(p4est);
          p4est_ghost_destroy(ghost);
          p4est_nodes_destroy(nodes);
          delete ngbd;
          delete hierarchy;

          p4est = p4est_np1;
          ghost = ghost_np1;
          nodes = nodes_np1;

          hierarchy = hierarchy_np1;
          ngbd = ngbd_np1;

          if(solve_navier_stokes){
            ns->nullify_p4est_nm1(); // the nm1 grid has just been destroyed, but pointer within NS has not been updated, so it needs to be nullified (p4est_nm1 in NS == p4est in main)
          }
          // -------------------------------
          // Create the new p4est at time np1:
          // -------------------------------

          p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
          ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
          my_p4est_ghost_expand(p4est_np1,ghost_np1);
          nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

          // Get the new neighbors: // TO-DO : no need to do this here, is there ?
          hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
          ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

          // Initialize the neigbors:
          ngbd_np1->init_neighbors();
        }

        // -------------------------------
        // Do a memory safety check as user specified
        // -------------------------------
        PetscLogDouble mem_safety_check;

        if((tstep%check_mem_every_iter)==0){
          MPI_Barrier(mpi.comm());
          PetscMemoryGetCurrentUsage(&mem_safety_check);

          int no = nodes->num_owned_indeps;
          MPI_Allreduce(MPI_IN_PLACE,&no,1,MPI_INT,MPI_SUM,mpi.comm());

          MPI_Allreduce(MPI_IN_PLACE,&mem_safety_check,1,MPI_DOUBLE,MPI_SUM,mpi.comm());

          PetscPrintf(mpi.comm(),"\n"
                                 "Memory safety check:\n"
                                 " - Current memory usage is : %0.9e GB \n"
                                 " - Number of grid nodes is: %d \n"
                                 " - Percent of safety limit: %0.2f % \n \n \n",
                      mem_safety_check*1.e-9,
                      no,
                      (mem_safety_check)/(mem_safety_limit)*100.0);

          // Output file for NS test case errors:
          const char* out_dir_fluid_forces = getenv("OUT_DIR_NS_FORCES");
          if(!out_dir_fluid_forces){
              throw std::invalid_argument("You need to set the environment variable OUT_DIR_NS_FORCES to save fluid forces");
            }
          FILE* fich_mem;
          char name_mem[1000];
          sprintf(name_mem,"%s/memory_check_Re_%0.2f_lmin_%d_lmax_%d_advection_order_%d.dat",
                  out_dir_fluid_forces,Re,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order);

          if(tstep==0){
            ierr = PetscFOpen(mpi.comm(),name_mem,"w",&fich_mem); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(),fich_mem,"tstep mem vtk_bool\n"
                                                    "%d %g %d \n",tstep,mem_safety_check,are_we_saving);CHKERRXX(ierr);
            ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);

          }
          else{
            ierr = PetscFOpen(mpi.comm(),name_mem,"a",&fich_mem); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(),fich_mem,"%d %g %d \n",tstep,mem_safety_check,are_we_saving);CHKERRXX(ierr);
            ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);

          }
        }
      } // <-- End of for loop through time

    PetscPrintf(mpi.comm(),"Time loop exited \n");

    // Final destructions: TO-DO: need to revisit these, make sure they're done correctly
    if(solve_stefan){
      T_l_n.destroy();
      T_s_n.destroy();
      v_interface.destroy();

      if(advection_sl_order==2) T_l_nm1.destroy();

      // Destroy relevant BC and RHS info:
      for(unsigned char d=0;d<2;++d){
        if(example_ == COUPLED_PROBLEM_EXAMPLE){
          delete analytical_T[d];
          delete external_heat_source_T[d];
        }
        else{ // case where we used curvature, want to clear interpolator before destroying
          bc_interface_val_temp[d]->clear();
        }
        delete bc_interface_val_temp[d];
        delete bc_wall_value_temp[d];
      }
      if(!solve_navier_stokes){
        phi.destroy();

        // destroy the structures leftover (in non NS case)
        p4est_nodes_destroy(nodes);
        p4est_ghost_destroy(ghost);
        p4est_destroy      (p4est);

        p4est_nodes_destroy(nodes_np1);
        p4est_ghost_destroy(ghost_np1);
        p4est_destroy(p4est_np1);

        my_p4est_brick_destroy(conn, &brick);
        delete hierarchy;
        delete ngbd;

        delete hierarchy_np1;
        delete ngbd_np1;
      }
    }

    if(solve_navier_stokes){
      v_n.destroy();
      v_nm1.destroy();

      // NS takes care of destroying v_NS_n and v_NS_nm1
      vorticity.destroy();
      press_nodes.destroy();

      for(unsigned char d=0;d<P4EST_DIM;d++){
        if((example_ == COUPLED_PROBLEM_EXAMPLE) || (example_ == NS_GIBOU_EXAMPLE)){
          delete analytical_soln_v[d];
          delete external_force_components[d];
        }

        delete bc_interface_value_velocity[d];
        delete bc_wall_value_velocity[d];
        delete bc_wall_type_velocity[d];
      }

      delete ns;
    }
  }// end of loop through number of splits

  MPI_Barrier(mpi.comm());
  w.stop(); w.read_duration();
  return 0;
}

