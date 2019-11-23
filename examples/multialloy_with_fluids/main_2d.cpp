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
int example_ = 3;
// Define the numeric label for each type of example to make implementation a bit more clear
enum{
  FRANK_SPHERE = 0,
  NS_GIBOU_EXAMPLE = 1,
  NS_LLNL_EXAMPLE = 2,
  COUPLED_PROBLEM_EXAMPLE = 3,
  ICE_AROUND_CYLINDER = 4,
  FLOW_PAST_CYLINDER = 5,

};

bool elyce_laptop = false; // Set to true if working on laptop --> changes the output path

// ---------------------------------------
// Save options:
// ---------------------------------------
DEFINE_PARAMETER(pl,bool,save_stefan,false,"Save stefan ?");
DEFINE_PARAMETER(pl,bool,save_navier_stokes,true,"Save navier stokes?");
DEFINE_PARAMETER(pl,bool,save_coupled_fields,false,"Save the coupled problem?");


// ---------------------------------------
// Solution options:
// ---------------------------------------
//bool solve_stefan = true;
//bool solve_navier_stokes = true;
//bool solve_coupled = solve_stefan && solve_navier_stokes;

DEFINE_PARAMETER(pl,bool,solve_stefan,false,"Solve stefan ?");
DEFINE_PARAMETER(pl,bool,solve_navier_stokes,true,"Solve navier stokes?");
DEFINE_PARAMETER(pl,bool,solve_coupled,false,"Solve the coupled problem?");
void select_solvers(){
  switch(example_){
    case FRANK_SPHERE:
      save_stefan = true;
      solve_stefan = true;
      save_navier_stokes = false;
      solve_navier_stokes = false;
      save_coupled_fields = false;
      break;
    case ICE_AROUND_CYLINDER:
      save_stefan = false;
      solve_stefan = true;
      solve_navier_stokes = true;
      save_navier_stokes = false;
      save_coupled_fields = true;
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
}

bool do_advection = true;
int advection_sl_order = 2;
bool solve_smoke = false;
double cfl = 0.5;

// ---------------------------------------
// Geometry options:
// ---------------------------------------
double xmin; double xmax;
double ymin; double ymax;

int nx, ny;       // number trees in each direction
int px, py;       // periodicity in each direction
double box_size;  // equivalent physical width (in x) in meters of the computational domain -- used for scaling the problem
double scaling;   // for scaling the problem

// For frank sphere:
double s0;
double T_inf;

// For ice cube:
double r0;
double Twall;
double Tinterface;
double back_wall_temp_flux;

// For solidifying ice problem:
double r_cyl;
double T_cyl;

// For surface tension: (used to apply some interfacial BC's in temperature)
double sigma;

double Tmax_allowed = 300.;


void set_geometry(){
  switch(example_){
    case FRANK_SPHERE: // Frank sphere
      xmin = -5.0; xmax = 5.0; //5.0;
      ymin = -5.0; ymax = 5.0;
      box_size = 1.0;
      nx = 1;
      ny = 1;
      px = 0; py = 1;


      s0 = 1.56;
      r0 = s0; // for consistency, and for setting up NS problem (if wanted)
      T_inf = -0.5;
      Twall = -0.5;
      Tinterface = 0.0;
      break;

    case ICE_AROUND_CYLINDER: // Ice layer growing around a constant temperature cooled cylinder
      xmin = 0.0; xmax = 10.0;
      ymin = 0.0; ymax = 5.0;

      nx = 2;
      ny = 1;

      px = 0;
      py = 1;

      box_size = 0.1;// Equivalent width [in meters]
      r0 = 1.55; //0.045 0.25 [in meters]
      r_cyl = 1.5; // [in meters]
      Twall = 298.0; Tinterface = 273.0;
      T_cyl = 273.0 - 70.0;
      back_wall_temp_flux = 0.0;

      sigma = 1.e-2; // Some artificial surface tension for the Dirichlet temperature BC including some curvature effects
      break;

    case NS_GIBOU_EXAMPLE: // Navier Stokes Validation case from Gibou 2015
      xmin = 0.0; xmax = PI;
      ymin = 0.0; ymax = PI;

      nx = 1; ny = 1;
      px = 0; py = 0;

      box_size = 1.0;
      r0 = 0.10;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      xmin = 0.0; xmax = PI;
      ymin = 0.0; ymax = PI;

      nx = 1; ny = 1;
      px = 0; py = 0;

      box_size = 1.0;
      r0 = 0.2;
      break;

    case FLOW_PAST_CYLINDER:
      xmin = 0.0; xmax = 5.0;
      ymin = 0.0; ymax = 5.0;

      nx = 2; ny = 2;
      px = 0; py = 1;

      box_size = 0.1;
      r0 = 0.5;
      break;

    }

  scaling = 1./box_size;
//  r0*=scaling;
//  r_cyl*=scaling;
}

double v_interface_max_norm; // For keeping track of the interfacial velocity maximum norm

// ---------------------------------------
// Grid refinement:
// ---------------------------------------
DEFINE_PARAMETER(pl,int,lmin,4,"Minimum level of refinement");
DEFINE_PARAMETER(pl,int,lmax,7,"Maximum level of refinement");
DEFINE_PARAMETER(pl,double,lip,1.75,"Lipschitz coefficient");
DEFINE_PARAMETER(pl,int,method_,1,"Solver in time for solid domain, and for fluid if no advection. 1 - Backward Euler, 2 - Crank Nicholson");

// ---------------------------------------
// Time-stepping:
// ---------------------------------------
double tfinal;
double delta_t;
double dt_max_allowed;
bool keep_going = true;

double tn;
double dt;
double dt_nm1;


void simulation_time_info(){
  switch(example_){
//    case 0:
//      tfinal = 3.6e3; // corresponds to 1 hour -- 3600 seconds
//      delta_t = 1.e-1;
//      dt_max_allowed = 1.e2;
//      tn = 0.0;

//      break;
    case FRANK_SPHERE:
      tfinal = 1.25;
      delta_t = 0.01;
      dt_max_allowed = 0.05;
      tn = 1.0;
      break;
    case ICE_AROUND_CYLINDER: // ice solidifying around isothermally cooled cylinder
      tfinal = 25.0;
      delta_t = 1.e-3;
      dt_max_allowed = 1.0;
      tn = 0.0;
      break;

    case NS_GIBOU_EXAMPLE:
      tfinal = PI/3.;
      delta_t = 1.0e-2;
      dt_max_allowed = 1.0e-1;
      tn = 0.0;
      break;

    case FLOW_PAST_CYLINDER:
      tfinal = 50.0;
      delta_t =1.e-2;
      dt_max_allowed = 1.0;
      tn = 0.0;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      tfinal = 0.2;
      delta_t = 1.0e-2;
      dt_max_allowed = 1.0e-1;
      tn = 0.0;
      break;

    }
}
// ---------------------------------------
// Physical properties:
// ---------------------------------------
double alpha_s;
double alpha_l;

void set_diffusivities(){
  switch(example_){
//    case 0:
//      alpha_s = (1.1820e-6); //ice - [m^2]/s
//      alpha_l = (1.4547e-7); //water- [m^2]/s
//      break;
    case FRANK_SPHERE:
      alpha_s = 1.0;
      alpha_l = 1.0;
      break;

    case ICE_AROUND_CYLINDER:
      alpha_s = (1.1820e-6); //ice - [m^2]/s
      alpha_l = (1.4547e-7); //water- [m^2]/s
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      alpha_s = 1.0;
      alpha_l = 1.0;
      break;
    }
}


double k_s;
double k_l;
double L; // Latent heat of fusion
double rho_l;

void set_conductivities(){
  switch(example_){
//    case 0:
//       k_s = 2.22; // W/[m*K]
//       k_l = 0.608; // W/[m*K]
//       L = 334.e3;  // J/kg
//       rho_l = 1000.0; // kg/m^3
//      break;
    case FRANK_SPHERE:
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      break;

    case ICE_AROUND_CYLINDER:
      k_s = 2.22; // W/[m*K]
      k_l = 0.608; // W/[m*K]
      L = 334.e3;  // J/kg
      rho_l = 1000.0; // kg/m^3
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      k_s = 1.;
      k_l = 1.;
      L = 1.;
      rho_l = 1.;
      break;
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
double mu_l;
double hodge_percentage_of_max_u;
double uniform_band;
double dt_NS;

double NS_norm = 0.0; // To keep track of the NS norm
void set_NS_info(){
  pressure_prescribed_flux = 0.0; // For the Neumann condition on the two x walls and lower y wall
  pressure_prescribed_value = 0.0; // For the Dirichlet condition on the back y wall

  uniform_band = 5.0;
  dt_NS = 1.e-2; // initial dt for NS
  switch(example_){
    case FRANK_SPHERE:throw std::invalid_argument("NS isnt setup for this example");
    case ICE_AROUND_CYLINDER:
      Re_u = 350.;
      Re_v = 0.;
      mu_l = 8.9e-4;  // Viscosity of water , [Pa s]
      break;
    case NS_GIBOU_EXAMPLE:
      Re_u = 1.0;
      Re_v = 1.0;
      mu_l = 1.0;
      rho_l = 1.0;

      u0 = 1.0;
      v0 = 1.0;
      break;

    case FLOW_PAST_CYLINDER:
      Re_u = 500.0;
      Re_v = 0.0;
      mu_l = 8.9e-4;
      rho_l = 1000.0;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      Re_u = 1.0;
      Re_v = 1.0;
      mu_l = 1.0;
      rho_l = 1.0;
      u0 = 1.0;
      v0 = 1.0;
      break;
    }

  outflow_u = 0.0;
  outflow_v = 0.0;

  hodge_percentage_of_max_u = 1.e-3;

  // WAY OF SETTING VELOCITIES FOR NS SOLVER NEEDS TO BE FIXED -- THIS IS JUST A TEMPORARY WAY
//  if(Re_u != 0.0){
//      u0 = 1.0;
//      mu_l = u0*rho_l*(2.*r0)/Re_u;
//    }
//  else{
//      u0 = 0.0;
//    }

//  if(Re_v != 0.0){
//      v0 = 1.0;
//      mu_l = v0*rho_l*(2.*r0)/Re_v;
//    }
//  else{
//      v0 = 0.0;
//    }
//  u0 = Re_u*mu_l/(rho_l*2.*r0);
//  v0 = Re_v*mu_l/(rho_l*2.*r0);
}

// ---------------------------------------
// Other parameters:
// ---------------------------------------
double v_int_max_allowed = 250.0;

bool move_interface_with_v_external = false;

bool check_temperature_values = false; // Whether or not you want to print out temperature value averages during various steps of the solution process -- for debugging

bool check_derivative_values = false;// Whether or not you want to print out temperature derivative value averages during various steps of the solution process -- for debugging

bool check_interfacial_velocity = true; // Whether or not you want to print out interfacial velocity value averages during various steps of the solution process -- for debugging

bool save_temperature_derivative_fields = false; // saving temperature derivative fields to vtk or not

bool force_interfacial_velocity_to_zero = true;
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
/*
double theta(double x, double omega){
  return x - omega;
}

double omega(double t){
  return 1. + sin(2.*PI*SQR(t));
}

double domega_dt(double t){
  return 4.*PI*t*cos(2.*PI*SQR(t));
}

class u_analytical_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn);
//    double Theta = theta(x,Omega);
//    return cos(2.*PI*Theta)*(3.*SQR(y) - 2.*y);
    return cos(tn)*sin(x)*cos(y);

  }
}u_analytical;

class v_analytical_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn);
//    double Theta = theta(x,Omega);

//    return 2.*PI*sin(2.*PI*Theta)*SQR(y)*(y-1);
    return -1.*cos(tn)*cos(x)*sin(y);

  }
}v_analytical;

class p_analytical_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn);
//    double d_Omega_dt = domega_dt(tn);
//    double Theta = theta(x,Omega);

//    double part1 = -1.*d_Omega_dt*(1./(2.*PI))*sin(2.*PI*Theta)*(sin(2.*PI*y) - 2.*PI*y + PI);

//    double part2 = -1.*(mu_l/rho_l)*cos(2*PI*Theta)*(-2.*sin(2.*PI*y) + 2.*PI*y - PI);

//    return part1 + part2;
    return 0.0;

  }
}p_analytical;

class dp_dx_analytical_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    double Omega = omega(tn);
    double d_Omega_dt = domega_dt(tn);
    double Theta = theta(x,Omega);

    double part1 = -1.*cos(2.*PI*Theta)*d_Omega_dt*(PI + sin(2.*PI*y) - 2.*PI*y);

    double part2 =  -2.*PI*sin(2.*PI*Theta)*(PI + 2.*sin(2.*PI*y) - 2.*PI*y);

    return part1 + part2;

  }
}dp_dx_analytical;

class dp_dy_analytical_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    double Omega = omega(tn);
    double d_Omega_dt = domega_dt(tn);
    double Theta = theta(x,Omega);

    double part1 = sin(2.*PI*Theta)*(2.*PI - 2.*PI*cos(2.*PI*y))*(d_Omega_dt/(2.*PI));

    double part2 = -1.*cos(2.*PI*Theta)*(2.*PI - 4.*PI*cos(2.*PI*y));

    return part1 + part2;

  }
}dp_dy_analytical;

class u_analytical_bc_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn + dt);
//    double Theta = theta(x,Omega);
//    return cos(2.*PI*Theta)*(3.*SQR(y) - 2.*y);
    return cos(tn+dt)*sin(x)*cos(y);
  }
}u_analytical_bc;

class v_analytical_bc_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn + dt);
//    double Theta = theta(x,Omega);

//    return 2.*PI*sin(2.*PI*Theta)*SQR(y)*(y-1);
    return -1.*cos(tn+dt)*cos(x)*sin(y);
  }
}v_analytical_bc;

class p_analytical_bc_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn + dt);
//    double d_Omega_dt = domega_dt(tn + dt);
//    double Theta = theta(x,Omega);

//    double part1 = -1.*d_Omega_dt*(1./(2.*PI))*sin(2.*PI*Theta)*(sin(2.*PI*y) - 2.*PI*y + PI);

//    double part2 = -1.*(mu_l/rho_l)*cos(2*PI*Theta)*(-2.*sin(2.*PI*y) + 2.*PI*y - PI);

//    return part1 + part2;
    return 0.0;

  }
}p_analytical_bc;

class dp_dx_analytical_bc_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    double Omega = omega(tn);
    double d_Omega_dt = domega_dt(tn + dt);
    double Theta = theta(x,Omega);

    double part1 = -1.*cos(2.*PI*Theta)*d_Omega_dt*(PI + sin(2.*PI*y) - 2.*PI*y);

    double part2 =  -2.*PI*sin(2.*PI*Theta)*(PI + 2.*sin(2.*PI*y) - 2.*PI*y);

    return part1 + part2;

  }
}dp_dx_analytical_bc;

class dp_dy_analytical_bc_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn + dt);
//    double d_Omega_dt = domega_dt(tn + dt);
//    double Theta = theta(x,Omega);

//    double part1 = sin(2.*PI*Theta)*(2.*PI - 2.*PI*cos(2.*PI*y))*(d_Omega_dt/(2.*PI));

//    double part2 = -1.*cos(2.*PI*Theta)*(2.*PI - 4.*PI*cos(2.*PI*y));

//    return part1 + part2;

  }
}dp_dy_analytical_bc;

class external_force_x_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn);
//    double d_Omega_dt = domega_dt(tn);
//    double Theta = theta(x,Omega);

//    double part1 = 4.*SQR(PI)*y*(3.*y -2) - d_Omega_dt*(PI - 2.*PI*y + sin(2.*PI*y)) - 6.;

//    double part2 = -1.*(PI + 2.*sin(2.*PI*y) - 2*PI*y) + y*(3*y - 2)*d_Omega_dt;
//    double part3 = -1.*SQR(3.*y - 2.) + (6.*y - 2.)*(y - 1.);

//    return cos(2.*PI*Theta)*part1 + sin(2.*PI*Theta)*part2 + sin(4.*PI*Theta)*part3;

    return -1.*rho_l*sin(tn)*sin(x)*cos(y) + rho_l*SQR(cos(tn))*sin(x)*cos(x) + 2.*mu_l*cos(tn)*sin(x)*cos(y);
  }
}external_force_x;

class external_force_y_t: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
//    double Omega = omega(tn);
//    double d_Omega_dt = domega_dt(tn);
//    double Theta = theta(x,Omega);

//    double part1 = sin(2.*PI*Theta)*(2.*SQR(PI*y) - 1.) + PI*y*(y*(3.*y -2.) - cos(2.*PI*Theta)*d_Omega_dt);
//    double part2 = -1.*d_Omega_dt*(cos(2.*PI*y) - 1.) - 8.*PI*y;
//    double part3 = 2.*cos(2.*PI*y) - 1.;

//    return 4.*PI*(y - 1.)*part1 + sin(2.*PI*Theta)*part2 + 2.*PI*cos(2.*PI*Theta)*part3;

    return rho_l*sin(t)*cos(x)*sin(y) + rho_l*SQR(cos(tn))*sin(y)*cos(y) - 2.*mu_l*cos(tn)*cos(x)*sin(y);
  }
}external_force_y;
*/

double u_analytical(double x, double y, double t){
  return cos(t)*sin(x)*cos(y);
}
class u_analytical_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return u_analytical(x,y,tn);
  }
}u_ana_tn;

class u_analytical_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return u_analytical(x,y,tn+dt);
  }
}u_ana_tnp1;

double v_analytical(double x, double y, double t){
  return -1.*cos(t)*cos(x)*sin(y);
}
class v_analytical_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return v_analytical(x,y,tn);
  }
}v_ana_tn;

class v_analytical_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return v_analytical(x,y,tn+dt);
  }
}v_ana_tnp1;

double p_analytical(double x, double y, double t){
  return 0.0;
}
class p_analytical_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return p_analytical(x,y,tn);
  }
}p_ana_tn;
class p_analytical_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return p_analytical(x,y,tn+dt);
  }
}p_ana_tnp1;
double external_force_x(double x, double y, double t){
  return -1.*rho_l*sin(t)*sin(x)*cos(y) + rho_l*SQR(cos(t))*sin(x)*cos(x) + 2.*mu_l*cos(t)*sin(x)*cos(y);

}
class fx_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return external_force_x(x,y,tn);
  }
}fx_ext_tn;
class fx_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return external_force_x(x,y,tn+dt);
  }
}fx_ext_tnp1;


double external_force_y(double x, double y, double t){
  return rho_l*sin(t)*cos(x)*sin(y) + rho_l*SQR(cos(t))*sin(y)*cos(y) - 2.*mu_l*cos(t)*cos(x)*sin(y);
}
class fy_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return external_force_y(x,y,tn);
  }
}fy_ext_tn;
class fy_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return external_force_y(x,y,tn+dt);
  }
}fy_ext_tnp1;


// For coupled problem validation:
double T_analytical(double x, double y, double t){
  double n = 2.0; double p = 2.0;
  return cos(pow(t,p))*sin(n*x)*cos(n*y);;
}
class T_analytical_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return T_analytical(x,y,tn);
  }
}T_ana_tn;

class T_analytical_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return T_analytical(x,y,tn+dt);
  }
}T_ana_tnp1;

double g_analytical(double x, double y, double t){
  double n = 2.0; double p = 2.0;
  double part1 = sin(pow(t,p))*(-p*sin(n*x)*cos(n*y));
  double part2 = cos(pow(t,p))*cos(t)*(n*sin(x)*cos(y))*(cos(n*x)*cos(n*y) + sin(n*x)*sin(n*y));
  return part1 + part2;
}
class g_analytical_tn: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return g_analytical(x,y,tn);
  }
}g_ana_tn;

class g_analytical_tnp1: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    return g_analytical(x,y,tn+dt);
  }
}g_ana_tnp1;

// --------------------------------------------------------------------------------------------------------------
// LEVEL SET FUNCTIONS:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch (example_){
//      case 0:
//        return r0 - sqrt(SQR(x - (xmax/2.0)) + SQR(y - (ymax/2.0)));
      case FRANK_SPHERE:
        return s0 - sqrt(SQR(x) + SQR(y));
      case ICE_AROUND_CYLINDER:
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case NS_GIBOU_EXAMPLE:
        return sin(x)*sin(y) - 0.5 ;
      case FLOW_PAST_CYLINDER:
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case COUPLED_PROBLEM_EXAMPLE:
        return sin(x)*sin(y) - 0.5;

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

// --------------------------------------------------------------------------------------------------------------
// INTERFACIAL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_temp;
void interface_bc(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE:
      interface_bc_type_temp = DIRICHLET;
      break;
    case ICE_AROUND_CYLINDER:
      interface_bc_type_temp = DIRICHLET; 
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

class BC_interface_value: public CF_DIM{
private:
  // Have interpolation objects for case with surface tension included in boundary condition: can interpolate the curvature in a timestep to the interface points while applying the boundary condition
  my_p4est_interpolation_nodes_t kappa_interp;
  my_p4est_interpolation_nodes_t nx_interp;
  my_p4est_interpolation_nodes_t ny_interp;

public:
  BC_interface_value(my_p4est_node_neighbors_t *ngbd, vec_and_ptr_dim_t normal, vec_and_ptr_t kappa): kappa_interp(ngbd), nx_interp(ngbd), ny_interp(ngbd)
  {
    // Set the curvature and normal inputs to be interpolated when the BC object is constructed:
    kappa_interp.set_input(kappa.vec,linear);
    nx_interp.set_input(normal.vec[0],linear);
    ny_interp.set_input(normal.vec[1],linear);
  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: // Frank sphere case, no surface tension
         return Tinterface;
      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder, with surface tension -- MAY ADD COMPLEXITY TO THIS LATER ON
        return Tinterface + (1.*sigma)*kappa_interp(x,y);
      case COUPLED_PROBLEM_EXAMPLE:
        return Tinterface;

      }

  }
};

class BC_interface_coeff: public CF_DIM{
public:
  double operator()(double x, double y) const
  { switch(example_){
      case FRANK_SPHERE: return 1.0; // maybe this should be 0?
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
        return T_cyl;
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

struct NS_sides : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return ((xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z))) && (!ylower_wall(DIM(x,y,z))) && !yupper_wall(DIM(x,y,z)));
  }
} ns_sides;

struct NS_top_bottom : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return ((ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z))));
  }
} ns_top_bottom;


// --------------------------------------------------------------------------------------------------------------
// WALL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
class WALL_BC_TYPE_TEMP: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE: return DIRICHLET;
      case ICE_AROUND_CYLINDER:
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET;
          }
        else if (xupper_wall(DIM(x,y,z))){
//            return DIRICHLET;
            return NEUMANN;
          }
        break;
      case COUPLED_PROBLEM_EXAMPLE:
        if(xlower_wall(x,y) || xupper_wall(x,y)){
            return NEUMANN;
          }
        else if(ylower_wall(x,y) || yupper_wall(x,y)){
            return DIRICHLET;
          }
        break;
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
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z))){
            if (level_set(DIM(x,y,z)) < EPS){
                return Twall;
              }
            else{
                return Tinterface;
              }
          }
        break;       }
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            if (level_set(DIM(x,y,z)) < EPS){
                return Twall;
              }
            else{
                return Tinterface;
                }
          }
        else if(xupper_wall(DIM(x,y,z))){ // Neumann condition on back wall
//            return Twall;
            return back_wall_temp_flux; // Neumann back wall flux;
          }
        break;
        }
      case COUPLED_PROBLEM_EXAMPLE:{
          if(xlower_wall(x,y) || xupper_wall(x,y)){
              return 0.0;
            }
          else if(ylower_wall(x,y) || yupper_wall(x,y)){
              return 0.0;
            }
          break;
        }
      }
  }
} wall_bc_value_temp;

// --------------------------------------------------------------------------------------------------------------
// TEMPERATURE INITIAL CONDITION
// --------------------------------------------------------------------------------------------------------------
class INITIAL_CONDITION_TEMP: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    double m;
    double r;
    double sval;
    double Tsloped;
    if (level_set(DIM(x,y,z)) > EPS){ // In the solid subdomain
        switch(example_){
          case FRANK_SPHERE:{
            r = sqrt(SQR(x) + SQR(y));
            sval = s(r,tn);
            return frank_sphere_solution_t(sval); // Initial distribution is the analytical solution of Frank Sphere problem at t = 0
          }
          case ICE_AROUND_CYLINDER:{
            return Tinterface;
            }
          case COUPLED_PROBLEM_EXAMPLE:{
              return T_ana_tn(x,y);
            }
          }
      }
    else{// In the fluid subdomain:
        switch(example_){
          case FRANK_SPHERE: {// Analytical solution to frank sphere as initial condition
            r = sqrt(SQR(x) + SQR(y));
            sval = s(r,tn);
            return frank_sphere_solution_t(sval);}
          case ICE_AROUND_CYLINDER:{
            m = (Twall - Tinterface)/(level_set(DIM(xmin,ymin,z)));
            Tsloped = Tinterface + m*level_set(DIM(x,y,z));
            if(Tsloped<Twall) return Tsloped;
            else return Twall;
            }
          case COUPLED_PROBLEM_EXAMPLE:
            {return T_ana_tn(x,y);}
          }
      }
  }
}IC_temp;
// --------------------------------------------------------------------------------------------------------------
// SMOKE BC and IC's -- if you want to pass smoke, a passive scalar, through the domain
// --------------------------------------------------------------------------------------------------------------
class bc_smoke_type: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
      return DIRICHLET;
  }
} bc_type_smoke;

class BC_smoke_value: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    if(xlower_wall(DIM(x,y,z)) && (y<ymax*3./4.) && (y>ymin + 1.*ymax/4.)){
        return 1.0;
      }
    else{
        return 0.0;
      }
  }
}bc_smoke_value;

class BC_smoke_coeff: public CF_DIM{
public:
  double operator()(double x, double y) const {
    return 1.0;

  }
}bc_smoke_coeff;

class IC_SMOKE: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return bc_smoke_value(DIM(x,y,z));

  }
}ic_smoke;

// --------------------------------------------------------------------------------------------------------------
// Prescribed external velocity fields -- These are used in the case where you want to advect the temperature by some externally imposed velocity field
// --------------------------------------------------------------------------------------------------------------
struct u_advance : CF_DIM
{ double operator() (double x, double y) const{
  return 4.e-4;
  }

} u_adv;

struct v_advance: CF_DIM{
  double operator()(double x, double y) const
  {
    return 0.0;
  }
} v_adv;


// --------------------------------------------------------------------------------------------------------------
// VELOCITY BOUNDARY CONDITION -- for velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
// Wall boundary conditions on u:

class WALL_BC_TYPE_VELOCITY_U: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("This option may not be used for the particular example being called");
      case ICE_AROUND_CYLINDER: {// water solidifying around a cylinder
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET; // Free stream
          }
        else if (xupper_wall(DIM(x,y,z))){
            return NEUMANN; // presribed outflow
          }
        break;
        }
      case NS_GIBOU_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return DIRICHLET;
          }
        break;}
      case COUPLED_PROBLEM_EXAMPLE:{
          if(ns_sides(x,y)){
              return NEUMANN;
            }
          else if(ns_top_bottom(x,y)){
              return DIRICHLET;
            }
        }
      } // end of switch case
  }
} wall_bc_type_velocity_u;

class WALL_BC_VALUE_VELOCITY_U: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return u0; //Free stream velocity
          }

        else if(xupper_wall(DIM(x,y,z))){ // Homogenous Dirichlet condition on back wall
            return outflow_u;
          }
        break;
        }
      case NS_GIBOU_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return u_ana_tnp1(x,y);
          }
        break;}
      case COUPLED_PROBLEM_EXAMPLE:{
          return 0.0;
        }
      }
  }
} wall_bc_value_velocity_u;

// Wall boundary conditions on v:

class WALL_BC_TYPE_VELOCITY_V: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n"); // ice cube melting
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET; // free stream
          }
        else if (xupper_wall(DIM(x,y,z))){
            return NEUMANN; // presribed outflow
          }
        break;}
      case NS_GIBOU_EXAMPLE:
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return DIRICHLET;
          }
        break;
      case COUPLED_PROBLEM_EXAMPLE:{
          if(ns_sides(x,y)){
              return NEUMANN;
            }
          else if(ns_top_bottom(x,y)){
              return DIRICHLET;
            }
        }
      }
  }
} wall_bc_type_velocity_v;

class WALL_BC_VALUE_VELOCITY_V: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return v0; // Free stream
          }
        else if(xupper_wall(DIM(x,y,z))){ // prescribed outflow
            return outflow_v;
          }
        break;}

      case NS_GIBOU_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
          return v_ana_tnp1(x,y);
          }
        break;
        }
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
      }
  }
} wall_bc_value_velocity_v;

// --------------------------------------------------------------------------------------------------------------
// VELOCITY INTERFACIAL CONDITION -- for velocity vector = (u,v,w)
// --------------------------------------------------------------------------------------------------------------
// Interfacial condition for the u component:
BoundaryConditionType interface_bc_type_velocity_u;
void interface_bc_velocity_u(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case ICE_AROUND_CYLINDER:
    case NS_GIBOU_EXAMPLE:
      interface_bc_type_velocity_u = DIRICHLET;
      break;
    case COUPLED_PROBLEM_EXAMPLE:{
        interface_bc_type_velocity_u = DIRICHLET;
      }

    }
}

// Interfacial condition for the u component:
class BC_interface_value_velocity_u: public CF_DIM{
private:
  my_p4est_interpolation_nodes_t v_interface_interp;

public:
  BC_interface_value_velocity_u(my_p4est_node_neighbors_t *ngbd,vec_and_ptr_dim_t v_interface): v_interface_interp(ngbd){
    // Set up the interpolation of the interfacial velocity x component:
    v_interface_interp.set_input(v_interface.vec[0],linear);
  }

  void update(my_p4est_node_neighbors_t *ngbd_new,vec_and_ptr_dim_t v_interface_new){
    v_interface_interp.update_neighbors(ngbd_new);
    v_interface_interp.set_input(v_interface_new.vec[0],linear);
  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case ICE_AROUND_CYLINDER:{ // Ice solidifying around a cylinder
        try {
          v_interface_interp(x,y);
          //std::cout<<"Returning vx = " << v_interface_interp(x,y) << " at ( " << x << ", "<< y << ") \n"<<std::endl;

        } catch (std::exception &e) {
          std::cout<<"Was unable to interpolate the x direction boundary condition properly \n"<<std::endl;
          std::cout<<" Point is: ( "<< x << ", "<< y << ") \n"<<std::endl;
          throw std::invalid_argument("Interpolation failed at this point \n ");

        }
        return v_interface_interp(x,y); // No slip on the interface  -- Thus is equal to the x component of the interfacial velocity
      }
      case NS_GIBOU_EXAMPLE:
         return u_ana_tnp1(x,y);
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
      }
  }
};


BoundaryConditionType interface_bc_type_velocity_v;
void interface_bc_velocity_v(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case ICE_AROUND_CYLINDER:
    case NS_GIBOU_EXAMPLE:
      interface_bc_type_velocity_v = DIRICHLET;
      break;
    case COUPLED_PROBLEM_EXAMPLE:{
        interface_bc_type_velocity_v = DIRICHLET;
      }
    }
}

class BC_interface_value_velocity_v: public CF_DIM{
private:
  my_p4est_interpolation_nodes_t v_interface_interp;

public:
  BC_interface_value_velocity_v(my_p4est_node_neighbors_t *ngbd,vec_and_ptr_dim_t v_interface): v_interface_interp(ngbd){
    // Set up the interpolation of the interfacial velocity for the y component:
    v_interface_interp.set_input(v_interface.vec[1],linear);
  }
  void update(my_p4est_node_neighbors_t *ngbd_new,vec_and_ptr_dim_t v_interface_new){
    v_interface_interp.update_neighbors(ngbd_new);
    v_interface_interp.set_input(v_interface_new.vec[1],linear);
  }
  double operator()(double x, double y) const
  {
    double v;
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case ICE_AROUND_CYLINDER:{ // Set velocity at interface = interfacial velocity computed by Stefan problem
        try {
          //std::cout<<"Calling interpolation of the boundary condition at ( " << x << ", "<<y<< ")\n"<<std::endl;
          v = v_interface_interp(x,y);

        } catch (std::exception &e) {
          std::cout<<"Was unable to interpolate the y direction boundary condition properly \n"<<std::endl;
          std::cout<<" Point is: ( "<< x << ", "<< y << ") \n"<<std::endl;
          throw;

        }
        return v_interface_interp(x,y); // No slip on the interface  -- Thus is equal to the y component of the interfacial velocity
      }
      case NS_GIBOU_EXAMPLE:
        return v_ana_tnp1(x,y);
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
      }
  }
};
// --------------------------------------------------------------------------------------------------------------
// VELOCITY INITIAL CONDITION -- for velocity vector = (u,v,w), in Navier-Stokes problem
// --------------------------------------------------------------------------------------------------------------
struct u_initial : CF_DIM
{ double operator() (double x, double y) const{
    switch(example_){
      case ICE_AROUND_CYLINDER:
        return u0;
      case COUPLED_PROBLEM_EXAMPLE:
        return u_ana_tn(x,y);
      case NS_GIBOU_EXAMPLE:
        return u_ana_tn(x,y);
      }  }

} u_initial;

struct v_initial: CF_DIM{
  double operator()(double x, double y) const
  {
    switch(example_){
      case ICE_AROUND_CYLINDER:
        return v0;
      case COUPLED_PROBLEM_EXAMPLE:
        return v_ana_tn(x,y);
      case NS_GIBOU_EXAMPLE:
        return v_ana_tn(x,y);
      }
  }
} v_initial;

// --------------------------------------------------------------------------------------------------------------
// PRESSURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
class WALL_BC_TYPE_PRESSURE: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes solution is not "
                                                     "compatible with this example, please choose another \n");
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return NEUMANN;
          }
        else if (yupper_wall(DIM(x,y,z))){
            return DIRICHLET;
          }
        break;}
      case NS_GIBOU_EXAMPLE: {
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return DIRICHLET;
          }
        break;}
      case COUPLED_PROBLEM_EXAMPLE:{
          if(ns_sides(x,y)){
              return DIRICHLET;
            }
          else if(ns_top_bottom(x,y)){
              return NEUMANN;
            }
        }
      }
  }
} wall_bc_type_pressure;

class WALL_BC_VALUE_PRESSURE: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not "
                                    "compatible with this example, please choose another \n");
      case ICE_AROUND_CYLINDER:{ // coupled problem
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return pressure_prescribed_flux; // Neumann BC in pressure on all walls but back y wall
          }
        else if(yupper_wall(DIM(x,y,z))){ // Dirichlet condition on back wall (usually homogeneous, but could be nonhomogeneous)
            return pressure_prescribed_value;
          }
        break;}
      case NS_GIBOU_EXAMPLE: {// benchmark NS case
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return p_ana_tnp1(x,y);
          }
        break;}
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0; // either homogeneous dirichlet or neumann
      }
  }
} wall_bc_value_pressure;

// vvv Used for NS LLNL validation case
class BC_wall_value_pressure_using_normals: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    double nx = 0.0; double ny = 0.0;

    // Get appropriate normals depending on which wall we are on:
    if(ylower_wall(DIM(x,y,z)) && (!xlower_wall(DIM(x,y,z)) && !xupper_wall(DIM(x,y,z)))){
        nx = 0.0; ny = -1.0;
      }
    else if(yupper_wall(DIM(x,y,z)) && (!xlower_wall(DIM(x,y,z)) && !xupper_wall(DIM(x,y,z)))){
        nx = 0.0; ny = 1.0;
      }
    else if(xlower_wall(DIM(x,y,z))){
        nx = -1.0; ny = 0.0;
      }
    else if (xupper_wall(DIM(x,y,z))){
        nx = 1.0; ny = 0.0;
      }
    if(ns_sides(x,y) || ns_top_bottom(x,y)){
      return 0.0;
      }
  }
};
// --------------------------------------------------------------------------------------------------------------
// PRESSURE INTERFACIAL CONDITION
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_pressure;
void interface_bc_pressure(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    case ICE_AROUND_CYLINDER:
      interface_bc_type_pressure = NEUMANN;
      break;
    case NS_GIBOU_EXAMPLE:
      interface_bc_type_pressure = NEUMANN;
      break;
    case COUPLED_PROBLEM_EXAMPLE:
      interface_bc_type_pressure = NEUMANN;
      break;

    }
}

class BC_interface_value_pressure: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder
        return 0.0;
      case NS_GIBOU_EXAMPLE: // Benchmark NS
        return 0.0;
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
      }
  }
}interface_bc_value_pressure;

class BC_interface_value_pressure_using_normals: public CF_DIM{
private:
  // Have interpolation objects for case with surface tension included in boundary condition: can interpolate the curvature in a timestep to the interface points while applying the boundary condition
  my_p4est_interpolation_nodes_t nx_interp;
  my_p4est_interpolation_nodes_t ny_interp;

public:
  BC_interface_value_pressure_using_normals(my_p4est_node_neighbors_t *ngbd, vec_and_ptr_dim_t normal): nx_interp(ngbd), ny_interp(ngbd)
  {
    // Set the curvature and normal inputs to be interpolated when the BC object is constructed:
    nx_interp.set_input(normal.vec[0],linear);
    ny_interp.set_input(normal.vec[1],linear);
  }
  double operator()(double x, double y) const
  {
//    return dp_dx_analytical_bc(x,y)*nx_interp(x,y) + dp_dy_analytical_bc(x,y)*ny_interp(x,y);
//    return p_analytical_bc(x,y);
    return 0.0;
  }
};

// --------------------------------------------------------------------------------------------------------------
// Functions for checking the values of interest during the solution process
// --------------------------------------------------------------------------------------------------------------
void check_T_values(vec_and_ptr_t phi, vec_and_ptr_t T, p4est_nodes* nodes, p4est_t* p4est, int example, vec_and_ptr_t phi_cyl) {
  T.get_array();
  phi.get_array();
  if (example_ == 2) phi_cyl.get_array();

  double avg_T = 0.0;
  double max_T = 0.;
  double min_T = 1.e10;
  double min_mag_T = 1.e10;

  int pts_avg = 0;

  bool in_domain;

  // Make sure phi_cyl is defined if we are running an example where it is required:
  if(example_ ==ICE_AROUND_CYLINDER && phi_cyl.ptr == NULL){
      throw std::invalid_argument("You must provide a phi_cylinder vector to run example 2 \n");
    }

  // Loop over each node, check if node is in the subdomain we are considering. If so, compute average,max, and min values for the domain
  foreach_local_node(n,nodes){

    in_domain = false;
    // Check if the node is in the domain we are checking:
    if (example_ ==ICE_AROUND_CYLINDER ){
        ((phi.ptr[n] < EPS) && (phi_cyl.ptr[n] < EPS)) ? in_domain = true : in_domain = false;
      }
    else{
        (phi.ptr[n] < EPS) ? in_domain = true : in_domain = false;
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

  PetscPrintf(p4est->mpicomm,"\n");
  PetscPrintf(p4est->mpicomm,"Average temperature: %0.2f \n",global_avg_T);
  PetscPrintf(p4est->mpicomm,"Maximum temperature: %0.2f \n",global_max_T);
  PetscPrintf(p4est->mpicomm,"Minimum temperature: %0.2f \n",global_min_T);
  PetscPrintf(p4est->mpicomm,"Minimum temperature magnitude: %0.2f \n",global_min_mag_T);

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

    r = sqrt(SQR(xyz[0]) + SQR(xyz[1]));
    s = r/sqrt(tn+dt);

    double phi_exact = s0*sqrt(tn+dt) - r;
    double T_exact = frank_sphere_solution_t(s);


    // Error on phi and v_int:
    if(fabs(phi.ptr[n]) < dxyz_close_to_interface){
      // Errors on phi:
      phi_error = fabs(phi.ptr[n] - phi_exact);
      Linf_phi = max(Linf_phi,phi_error); // CHECK THIS -- NOT ENTIRELY SURE THIS IS CORRECT

      // Errors on v_int:
      vel = sqrt(SQR(v_interface.ptr[0][n])+ SQR(v_interface.ptr[1][n]));
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
  PetscPrintf(p4est->mpicomm,"dxyz close to interface: %0.2f \n",dxyz_close_to_interface);

  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"Number of grid points used: %d \n \n", num_nodes);


  PetscPrintf(p4est->mpicomm," Linf on phi: %0.4f \n Linf on T_l: %0.4f \n Linf on T_s: %0.4f \n Linf on v_int: %0.4f \n", global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3]);

  // Print errors to file:
  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  PetscFPrintf(p4est->mpicomm,fich,"%e %e %d %e %e %e %e %d %e \n",tn + dt,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],num_nodes,dxyz_close_to_interface);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);

  T_l.restore_array();
  T_s.restore_array();
  phi.restore_array();
  v_interface.restore_array();
}

void check_NS_LLNL_benchmark_error(vec_and_ptr_t phi,vec_and_ptr_dim_t v_n, vec_and_ptr_t p, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_node_neighbors_t *ngbd, double dxyz_close_to_interface, char *name, FILE *fich, int tstep){
  PetscErrorCode ierr;

  double u_error = 0.0;
  double v_error = 0.0;
  double P_error = 0.0;

  double L_inf_u = 0.0;
  double L_inf_v = 0.0;
  double L_inf_P = 0.0;


  // Get arrays:
  v_n.get_array();
  p.get_array();
  phi.get_array();

  // Get local errors in negative subdomain:
  double xyz[P4EST_DIM];
  double x;
  double y;
  foreach_local_node(n,nodes){
    if(phi.ptr[n] < 0.){
        node_xyz_fr_n(n,p4est,nodes,xyz);

        x = xyz[0]; y = xyz[1];

        u_error = fabs(v_n.ptr[0][n] - u_ana_tn(x,y));
        v_error = fabs(v_n.ptr[1][n] - v_ana_tn(x,y));
        P_error = fabs(p.ptr[n] - p_ana_tn(x,y));

        L_inf_u = max(L_inf_u,u_error);
        L_inf_v = max(L_inf_v,v_error);
        L_inf_P = max(L_inf_P,P_error);

      }
  }

//  // Loop over each quadrant in each tree, check the error in hodge
//  double xyz_c[P4EST_DIM];
//  double x_c; double y_c;
//  my_p4est_interpolation_nodes_t interp_phi(ngbd);
//  interp_phi.set_input(phi.vec,linear);
//  foreach_tree(tr,p4est){
//    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tr);

//    foreach_local_quad(q,tree){
//      // Get the global index of the quadrant:
//      p4est_locidx_t quad_idx = tree->quadrants_offset + q;

//      // Get xyz location of the quad center so we can interpolate phi there and check which domain we are in:
//      quad_xyz_fr_q(quad_idx,tr,p4est,ghost,xyz_c);
//      x_c = xyz_c[0]; y_c = xyz_c[1];

//      // Get the error in the negative subdomain:
//      if(interp_phi(x_c,y_c) < 0){
//          P_error = fabs(p.ptr[quad_idx] - p_analytical(x_c,y_c));
//        }

//    }
//  }


  // NEED TO GRAB PRESSURE ERROR AT QUADS, NEED TO CHANGE PRESSURE TO VEC_AND_PTR_CELLS
  // Restore arrays
  v_n.restore_array();
  p.restore_array();
  phi.restore_array();

  // Get the global errors:
  double local_Linf_errors[3] = {L_inf_u,L_inf_v,L_inf_P};
  double global_Linf_errors[3] = {0.0,0.0,0.0};

  MPI_Barrier(p4est->mpicomm);
  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,3,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n "
                             "Errors on LLNL NS Benchmark "
                             "\n -------------------------------------\n "
                             "Linf on u: %0.4f \n"
                             "Linf on v: %0.4f \n"
                             "Linf on P: %0.4f \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.4f \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:

  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);



}

void check_coupled_problem_error(vec_and_ptr_t phi,vec_and_ptr_dim_t v_n, vec_and_ptr_t p, vec_and_ptr_t Tl, p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd, double dxyz_close_to_interface, char *name, FILE *fich, int tstep){
  PetscErrorCode ierr;

  double u_error = 0.0;
  double v_error = 0.0;
  double P_error = 0.0;
  double T_error = 0.0;

  double L_inf_u = 0.0;
  double L_inf_v = 0.0;
  double L_inf_P = 0.0;
  double L_inf_T = 0.0;


  // Get arrays:
  v_n.get_array();
  p.get_array();
  phi.get_array();
  Tl.get_array();

  // Get local errors in negative subdomain:
  double xyz[P4EST_DIM];
  double x;
  double y;
  foreach_local_node(n,nodes){
    if(phi.ptr[n] < 0.){
        node_xyz_fr_n(n,p4est,nodes,xyz);

        x = xyz[0]; y = xyz[1];

        u_error = fabs(v_n.ptr[0][n] - u_ana_tn(x,y));
        v_error = fabs(v_n.ptr[1][n] - v_ana_tn(x,y));
        P_error = fabs(p.ptr[n] - p_ana_tn(x,y));
        T_error = fabs(Tl.ptr[n] - T_ana_tn(x,y));

        L_inf_u = max(L_inf_u,u_error);
        L_inf_v = max(L_inf_v,v_error);
        L_inf_P = max(L_inf_P,P_error);
        L_inf_T = max(L_inf_T,T_error);

      }
  }

//  // Loop over each quadrant in each tree, check the error in hodge
//  double xyz_c[P4EST_DIM];
//  double x_c; double y_c;
//  my_p4est_interpolation_nodes_t interp_phi(ngbd);
//  interp_phi.set_input(phi.vec,linear);
//  foreach_tree(tr,p4est){
//    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees,tr);

//    foreach_local_quad(q,tree){
//      // Get the global index of the quadrant:
//      p4est_locidx_t quad_idx = tree->quadrants_offset + q;

//      // Get xyz location of the quad center so we can interpolate phi there and check which domain we are in:
//      quad_xyz_fr_q(quad_idx,tr,p4est,ghost,xyz_c);
//      x_c = xyz_c[0]; y_c = xyz_c[1];

//      // Get the error in the negative subdomain:
//      if(interp_phi(x_c,y_c) < 0){
//          P_error = fabs(p.ptr[quad_idx] - p_analytical(x_c,y_c));
//        }

//    }
//  }


  // NEED TO GRAB PRESSURE ERROR AT QUADS, NEED TO CHANGE PRESSURE TO VEC_AND_PTR_CELLS
  // Restore arrays
  v_n.restore_array();
  p.restore_array();
  phi.restore_array();
  Tl.restore_array();

  // Get the global errors:
  double local_Linf_errors[4] = {L_inf_u,L_inf_v,L_inf_P, L_inf_T};
  double global_Linf_errors[4] = {0.0,0.0,0.0,0.0};

  MPI_Barrier(p4est->mpicomm);
  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,4,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n "
                             "Errors on LLNL NS Benchmark "
                             "\n -------------------------------------\n "
                             "Linf on u: %0.4f \n"
                             "Linf on v: %0.4f \n"
                             "Linf on P: %0.4f \n"
                             "Linf on Tl: %0.4f \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.4f \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:

//  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
//  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
//  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);



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

  MPI_Allreduce(&still_solid_present,&global_still_solid_present,1,MPI_INT,MPI_LOR,p4est->mpicomm);


  SC_CHECK_MPI(mpi_check);
  if (!global_still_solid_present){ // If no more solid, then ice has melted
      PetscPrintf(p4est->mpicomm,"\n \n Ice has entirely melted as of t = %0.3e \n \n ",time);
    }
return global_still_solid_present;
}

void setup_rhs(vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t smoke, vec_and_ptr_t rhs_Tl, vec_and_ptr_t rhs_Ts, vec_and_ptr_t rhs_smoke, vec_and_ptr_t T_l_backtrace, vec_and_ptr_t smoke_backtrace, vec_and_ptr_t T_l_backtrace_nm1, vec_and_ptr_t smoke_backtrace_nm1, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd){

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

  // Get Ts arrays:
  T_s.get_array();
  rhs_Ts.get_array();

  // Get Tl arrays:
  rhs_Tl.get_array();
  if(do_advection){
      T_l_backtrace.get_array();
    }
  else{
      T_l.get_array();
    }

  // Get smoke arrays:
  if(solve_smoke){
      rhs_smoke.get_array();
      if(do_advection){
          smoke_backtrace.get_array();
        }
      else{
          smoke.get_array();
        }
    }


  foreach_node(n,nodes){
    // First, assemble system for Ts depending on case:
    if(method_ == 2){ // Crank Nicholson
        rhs_Ts.ptr[n] = 2.*T_s.ptr[n]/dt + alpha_s*(T_s_dd.ptr[0][n] + T_s_dd.ptr[1][n]);
      }
    else{ // Backward Euler
        rhs_Ts.ptr[n] = T_s.ptr[n]/dt;
      }

    // Now for Tl depending on case:
    if(do_advection){
        rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]/dt;
     }
    else{
        if(method_ ==2){//Crank Nicholson
            rhs_Tl.ptr[n] = 2.*T_l.ptr[n]/dt + alpha_l*(T_l_dd.ptr[0][n] + T_l_dd.ptr[1][n]);
          }
        else{ // Backward Euler
            rhs_Tl.ptr[n] = T_l.ptr[n]/dt;
          }
      }

    // Now for smoke:
    if(solve_smoke){
        if(do_advection){
            rhs_smoke.ptr[n] = smoke_backtrace.ptr[n]/dt;
          }
        else{
            rhs_smoke.ptr[n] = smoke.ptr[n]/dt;
          }
      }


//    rhs_smoke.ptr[n] = smoke_backtrace.ptr[n];
  }// end of loop over nodes

  // Restore Ts arrays:
  T_s.restore_array();
  rhs_Ts.restore_array();

  // Restore Tl arrays:
  rhs_Tl.restore_array();
  if(do_advection){
      T_l_backtrace.restore_array();
    }
  else{
      T_l.restore_array();
    }
  // Restore smoke arrays:
  if(solve_smoke){
      rhs_smoke.restore_array();
      if(do_advection){
          smoke_backtrace.restore_array();
        }
      else{
          smoke.restore_array();
        }
    }

  if(method_ ==2){
      T_s_dd.restore_array();
      T_s_dd.destroy();
      if(!do_advection){
          T_l_dd.restore_array();
          T_l_dd.destroy();
        }
    }
//  smoke_backtrace.restore_array();
//  rhs_smoke.restore_array();

//  PetscPrintf(p4est->mpicomm,"INSIDE OF THE FUNCTION: \n");

//  VecView(rhs_Tl.vec,PETSC_VIEWER_STDOUT_WORLD);

//  // Get derivatives of temperature fields if we are using Crank Nicholson:
//  vec_and_ptr_dim_t T_l_dd;
//  vec_and_ptr_dim_t T_s_dd;
//  if(method_ ==2){
//      T_s_dd.create(p4est,nodes);
//      ngbd->second_derivatives_central(T_s.vec,T_s_dd.vec[0],T_s_dd.vec[1]);
//      if(!do_advection) {
//          T_l_dd.create(p4est,nodes);
//          ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec[0],T_l_dd.vec[1]);
//        }
//    }

//  // Get Ts arrays:
//  rhs_Ts.get_array();
//  T_s.get_array();
//  if(method_ == 2){
//      T_s_dd.get_array();
//    }

//  // Get Tl arrays:
//  rhs_Tl.get_array();

//  if(method_ ==2 && !do_advection){
//      T_l_dd.get_array();
//    }

//  if(do_advection){
//      T_l_backtrace.get_array();
//      if(advection_sl_order==2) T_l_backtrace_nm1.get_array();
//    }
//  else{
//      T_l.get_array();
//    }

//  // Get Smoke arrays:
//  if(solve_smoke){
//      rhs_smoke.get_array();
//      smoke.get_array();

//      if(do_advection){
//          smoke_backtrace.get_array();
//          if(advection_sl_order==2) smoke_backtrace_nm1.get_array();
//        }
//    }


//  double alpha; double beta;
//  if(advection_sl_order ==2){
//      alpha = (2.*dt + dt_nm1)/(dt + dt_nm1);
//      beta = (-1.*dt)/(dt + dt_nm1);
//    }


//  // Loop through the nodes to build the RHS:
//  foreach_node(n,nodes){
//    if(method_ == 2){
//        rhs_Ts.ptr[n] = T_s.ptr[n]/dt + alpha_s*(T_s_dd.ptr[0][n] + T_s_dd.ptr[1][n]);
//      }
//    else{
//        rhs_Ts.ptr[n] = T_s.ptr[n]/dt;
//      }

//    if (do_advection) {
//        if(advection_sl_order ==2){
//            rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]*((alpha/dt) - (beta/dt_nm1)) + T_l_backtrace_nm1.ptr[n]*(beta/dt_nm1);
//          }
//        else{
//            rhs_Tl.ptr[n]= T_l_backtrace.ptr[n]/dt;
//          }
//      } // end of  if do advection
//    else {
//        if(method_ == 2){ // Crank nicholson
//            rhs_Tl.ptr[n] = T_l.ptr[n]/dt + alpha_l*(T_l_dd.ptr[0][n] + T_l_dd.ptr[1][n]);
//          }
//        else{
//            rhs_Tl.ptr[n] = T_l.ptr[n]/dt;
//          }
//      } // end of else: if not doing advection

//    if (solve_smoke){
//        if(do_advection){
//            if(advection_sl_order==2){
//                rhs_smoke.ptr[n] = smoke_backtrace.ptr[n]*((alpha/dt) - (beta/dt_nm1)) + smoke_backtrace_nm1.ptr[n]*(beta/dt_nm1);
//              }
//            else{
//                rhs_smoke.ptr[n] = smoke_backtrace.ptr[n];
//              }
//          } // end of if do advection
//        else{
//            rhs_smoke.ptr[n] = smoke.ptr[n];
//          } // end of else: if not doing advection
//      } // end of solve smoke
//  } // end of loop through the nodes

//  // Restore Ts arrays:
//  rhs_Ts.restore_array();
//  T_s.restore_array();
//  if(method_ == 2){
//      T_s_dd.restore_array();
//      T_s_dd.destroy();
//    }

//  // Restore Tl arrays:
//  rhs_Tl.restore_array();
//  if(method_ ==2 && !do_advection){
//      T_l_dd.restore_array();
//      T_l_dd.destroy();
//    }

//  if(do_advection){
//      T_l_backtrace.restore_array();
//      if(advection_sl_order==2) T_l_backtrace_nm1.restore_array();
//    }
//  else{
//      T_l.restore_array();
//    }

//  // Restore Smoke arrays:
//  if(solve_smoke){
//      rhs_smoke.restore_array();
//      smoke.restore_array();

//      if(do_advection){
//          smoke_backtrace.restore_array();
//          if(advection_sl_order==2) smoke_backtrace_nm1.restore_array();
//        }
//    }
}

void do_backtrace(vec_and_ptr_t T_l,vec_and_ptr_t T_l_backtrace,vec_and_ptr_dim_t v, vec_and_ptr_t smoke, vec_and_ptr_t smoke_backtrace, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t* ngbd, p4est_t *p4est_nm1, p4est_nodes_t *nodes_nm1, my_p4est_node_neighbors_t *ngbd_nm1,  vec_and_ptr_t T_l_backtrace_nm1, vec_and_ptr_dim_t v_nm1,vec_and_ptr_t smoke_backtrace_nm1, interpolation_method interp_method){
  // Get second derivatives of the velocity field:
  vec_and_ptr_dim_t v_dd[P4EST_DIM];
  vec_and_ptr_dim_t v_dd_nm1[P4EST_DIM];

  foreach_dimension(d){
    v_dd[d].create(p4est,nodes); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
    if(advection_sl_order ==2){
        v_dd_nm1[d].create(p4est_nm1,nodes_nm1);
      }
  }

  ngbd->second_derivatives_central(v.vec,v_dd[0].vec,v_dd[1].vec,P4EST_DIM);
  if(advection_sl_order ==2){
      ngbd_nm1->second_derivatives_central(v_nm1.vec,v_dd_nm1[0].vec,v_dd_nm1[1].vec,P4EST_DIM);
      PetscPrintf(p4est->mpicomm,"Gets the second derivatives of vnm1 \n");
    }

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];
  vector <double> xyz_d_nm1[P4EST_DIM];

  // Do the Semi-Lagrangian backtrace:
  if(advection_sl_order ==2){
      trajectory_from_np1_to_nm1(p4est,nodes,ngbd_nm1,ngbd,v_nm1.vec,v.vec,dt_nm1,dt,xyz_d_nm1,xyz_d);
    }
  else{
      trajectory_from_np1_to_n(p4est,nodes,ngbd,dt,v.vec,&v_dd->vec,xyz_d);
    }

  // Add the back-trace points to the interpolation object:
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd);
  my_p4est_interpolation_nodes_t SL_backtrace_interp_nm1(ngbd); // for handling nm1 backtraced points

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
  } // end of loop over nodes

  // Interpolate the Temperature data to back-traced points:
  // Note: We interpolate using an array of fields if we are solving for smoke to make interpolation more efficient
  // (rather than calling the interpolation over and over again)

  if(solve_smoke){
      Vec fields_in[2];
      Vec fields_out[2];

      fields_in[0] = T_l.vec;
      fields_in[1] = smoke.vec;

      fields_out[0] = T_l_backtrace.vec;
      fields_out[1] = smoke_backtrace.vec;

      SL_backtrace_interp.set_input(fields_in,interp_method,2);
      SL_backtrace_interp.interpolate(fields_out);

      if(advection_sl_order ==2){
          Vec fields_out_nm1[2];

          fields_out_nm1[0] = T_l_backtrace_nm1.vec;
          fields_out_nm1[1] = smoke_backtrace_nm1.vec;

          SL_backtrace_interp_nm1.set_input(fields_in,interp_method,2);
          SL_backtrace_interp_nm1.interpolate(fields_out_nm1);
        }
    }
  else{
      SL_backtrace_interp.set_input(T_l.vec,interp_method);
      SL_backtrace_interp.interpolate(T_l_backtrace.vec);

      if(advection_sl_order ==2){
          SL_backtrace_interp_nm1.set_input(T_l.vec,interp_method);
          SL_backtrace_interp_nm1.interpolate(T_l_backtrace_nm1.vec);
        }
    }

  // Destroy velocity derivatives now that not needed:
  v_dd->destroy();
  if(advection_sl_order ==2) v_dd_nm1->destroy();

}

void interpolate_values_onto_new_grid(vec_and_ptr_t T_l, vec_and_ptr_t T_l_new,
                                      vec_and_ptr_t T_s, vec_and_ptr_t T_s_new,
                                      vec_and_ptr_dim_t v_interface,vec_and_ptr_dim_t v_interface_new,
                                      vec_and_ptr_dim_t v_external,vec_and_ptr_dim_t v_external_new,
                                      vec_and_ptr_t smoke, vec_and_ptr_t smoke_new,
                                      p4est_nodes_t *nodes_new_grid, p4est_t *p4est_new, p4est_nodes_t *nodes_old_grid, p4est_t *p4est_old,
                                      my_p4est_node_neighbors_t *ngbd_old_grid,interpolation_method interp_method){
  // Need neighbors of old grid to create interpolation object
  // Need nodes of new grid to get the points that we must interpolate to

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_old_grid);

  // Create an array of the vectors for faster interpolation -- interpolate all fields at once:
  unsigned int num_fields =0;

  if(solve_navier_stokes || do_advection){
      num_fields+=2;
    }
  if(solve_stefan){
      num_fields+=4;
    }
  if(solve_smoke) num_fields+=1;

  Vec all_fields_old[num_fields];
  Vec all_fields_new[num_fields];

//  PetscErrorCode ierr;
//  for(unsigned int k = 0; k<num_fields; k++){
//      ierr = VecCreateGhostNodes(p4est_old, nodes_old_grid, &all_fields_old[k]); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est_new,nodes_new_grid,&all_fields_new[k]); CHKERRXX(ierr);
//    }
  // Set existing vectors as elements of the array of vectors:
  if(solve_stefan){
      all_fields_old[0] = T_l.vec;
      all_fields_old[1] = T_s.vec;
      all_fields_old[2] = v_interface.vec[0];
      all_fields_old[3] = v_interface.vec[1];


      all_fields_new[0] = T_l_new.vec;
      all_fields_new[1] = T_s_new.vec;
      all_fields_new[2] = v_interface_new.vec[0];
      all_fields_new[3] = v_interface_new.vec[1];


      if(do_advection || solve_navier_stokes){
          all_fields_old[4] = v_external.vec[0];
          all_fields_old[5] = v_external.vec[1];

          all_fields_new[4] = v_external_new.vec[0];
          all_fields_new[5] = v_external_new.vec[1];

          if(solve_smoke) {
              all_fields_old[6] = smoke.vec;
              all_fields_new[6] = smoke_new.vec;
            }
        }
      else{
          if(solve_smoke) {
              all_fields_old[4] = smoke.vec;
              all_fields_new[4] = smoke_new.vec;
            }
        }

    }
  else if(solve_navier_stokes){
      all_fields_old[0] = v_external.vec[0];
      all_fields_old[1] = v_external.vec[1];

      all_fields_new[0] = v_external_new.vec[0];
      all_fields_new[1] = v_external_new.vec[1];
    }

  interp_nodes.set_input(all_fields_old,interp_method,num_fields);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes_new_grid){
    node_xyz_fr_n(n,p4est_new,nodes_new_grid,xyz);
    interp_nodes.add_point(n,xyz);
  }

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

    jump.ptr[0][n] = (k_s*T_s_d.ptr[0][n] -k_l*T_l_d.ptr[0][n])/(L*rho_l);
    jump.ptr[1][n] = (k_s*T_s_d.ptr[1][n] -k_l*T_l_d.ptr[1][n])/(L*rho_l);
   }

  // Begin updating the ghost values of the layer nodes:
  foreach_dimension(d){
    VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
  }

  // Compute the jump in the local nodes:
  for(size_t i = 0; i<ngbd->get_local_size();i++){
      p4est_locidx_t n = ngbd->get_local_node(i);
      //ngbd->get_neighbors(n,qnnn);
      jump.ptr[0][n] = (k_s*T_s_d.ptr[0][n] -k_l*T_l_d.ptr[0][n])/(L*rho_l);
      jump.ptr[1][n] = (k_s*T_s_d.ptr[1][n] -k_l*T_l_d.ptr[1][n])/(L*rho_l);
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

  if(force_interfacial_velocity_to_zero){
        foreach_dimension(d){
            VecScaleGhost(v_interface.vec[d],0.0);
        }
  }
}

void compute_timestep(vec_and_ptr_dim_t v_interface, vec_and_ptr_t phi, double dxyz_close_to_interface, double dxyz_smallest[P4EST_DIM],p4est_nodes_t *nodes, p4est_t *p4est){

  // Check the values of v_interface locally:
  v_interface.get_array();
  phi.get_array();
  double max_v_norm = 0.0;
  foreach_local_node(n,nodes){
    if (fabs(phi.ptr[n]) < dxyz_close_to_interface){
        max_v_norm = max(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n])));
      }
  }
  v_interface.restore_array();
  phi.restore_array();

  // Get the maximum v norm across all the processors:
  MPI_Barrier(p4est->mpicomm);
  double global_max_vnorm = 0.0;
  int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
  SC_CHECK_MPI(mpi_ret);
  PetscPrintf(p4est->mpicomm,"\n \n \n Computed interfacial velocity and timestep: \n {");
  PetscPrintf(p4est->mpicomm,"\n Max v norm: %0.2e \n", global_max_vnorm);

//  // Save the previous timestep:
//  dt_nm1 = dt;
  // Compute new timestep:
  dt = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/min(global_max_vnorm,1.0);
  PetscPrintf(p4est->mpicomm,"Computed timestep: %0.3e \n",dt);

  dt = min(dt,dt_max_allowed);

  // Report computed timestep and minimum grid size:
  PetscPrintf(p4est->mpicomm,"Used timestep: %0.3e \n",dt);
  PetscPrintf(p4est->mpicomm,"dxyz close to interface : %0.3e \n } \n \n  ",dxyz_close_to_interface);

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
      curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]);
    }

  // Begin ghost update:
  VecGhostUpdateBegin(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);

  // Compute curvature on local nodes:
  for(size_t i = 0; i<ngbd->get_local_size(); i++){
      p4est_locidx_t n = ngbd->get_local_node(i);
      ngbd->get_neighbors(n,qnnn);
      curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]);
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
// FUNCTIONS FOR SAVING:
// --------------------------------------------------------------------------------------------------------------
void save_everything(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_t phi_2, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int,vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, vec_and_ptr_t smoke, char* filename){
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
 * smoke
 * */

  // First, need to scale the fields appropriately:

  // Scale velocities:
  foreach_dimension(d){
    VecScaleGhost(v_int.vec[d],1./scaling);
    VecScaleGhost(v_NS.vec[d],1./scaling);
  }

  // Scale pressure:
  VecScaleGhost(press.vec,scaling);

  // Get arrays:
  phi.get_array();
  if(example_ == ICE_AROUND_CYLINDER) phi_2.get_array();

  Tl.get_array(); Ts.get_array();

  v_int.get_array(); v_NS.get_array();

  press.get_array(); vorticity.get_array();

  if(solve_smoke) smoke.get_array();


  // Save data:
  std::vector<std::string> point_names;
  std::vector<double*> point_data;

  if(example_ == ICE_AROUND_CYLINDER && solve_smoke){
      point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure","smoke"};
      point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,smoke.ptr};
    }
  else if (example_ == ICE_AROUND_CYLINDER && !solve_smoke) {
      point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
      point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
    }
  else if (example_ !=ICE_AROUND_CYLINDER && solve_smoke){
      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure","smoke"};
      point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,smoke.ptr};

    }
  else{
      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
      point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
    }

  std::vector<std::string> cell_names = {};
  std::vector<double*> cell_data = {};

  my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


  // Restore arrays:

  phi.restore_array();
  if(example_ == ICE_AROUND_CYLINDER) phi_2.restore_array();

  Tl.restore_array(); Ts.restore_array();

  v_int.restore_array(); v_NS.restore_array();

  press.restore_array(); vorticity.restore_array();          // ------------------

  if(solve_smoke) smoke.restore_array();
  // Scale things back:
  foreach_dimension(d){
    VecScaleGhost(v_int.vec[d],scaling);
    VecScaleGhost(v_NS.vec[d],scaling);
  }

  // Scale pressure back:
  VecScaleGhost(press.vec,1./scaling);
}

void save_stefan_fields(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_t phi_2, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int, vec_and_ptr_t smoke, char* filename ){
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
    if(example_ == ICE_AROUND_CYLINDER) phi_2.get_array();

    Tl.get_array(); Ts.get_array();

    v_int.get_array();

    if(solve_smoke) smoke.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    if(example_ == ICE_AROUND_CYLINDER && solve_smoke){
        point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","smoke"};
        point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],smoke.ptr};
      }
    else if (example_ == ICE_AROUND_CYLINDER && !solve_smoke) {
        point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y"};
        point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1]};
      }
    else if (example_ !=ICE_AROUND_CYLINDER && solve_smoke){
        point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","smoke"};
        point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],smoke.ptr};
      }
    else{
        point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y"};
        point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1]};
      }

    std::vector<std::string> cell_names;
    std::vector<double*> cell_data;

    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


    // Restore arrays:

    phi.restore_array();
    if(example_ == ICE_AROUND_CYLINDER) phi_2.restore_array();

    Tl.restore_array(); Ts.restore_array();

    v_int.restore_array();

    if(solve_smoke) smoke.restore_array();
    // Scale things back:
    foreach_dimension(d){
      VecScaleGhost(v_int.vec[d],scaling);
    }
}
void save_navier_stokes_fields(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, vec_and_ptr_t smoke, char* filename){
  // Things we want to save:
  /*
   * LSF
   * v NS
   * pressure
   * vorticity
   * smoke
   * */

    // First, need to scale the fields appropriately:

    // Scale velocities:
    foreach_dimension(d){
      VecScaleGhost(v_NS.vec[d],1./scaling);
    }

    // Scale pressure:
    VecScaleGhost(press.vec,scaling);

    // Get arrays:
    phi.get_array();
    v_NS.get_array();

    press.get_array(); vorticity.get_array();

    if(solve_smoke) smoke.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

    if(solve_smoke){
        point_names = {"phi","u","v","vorticity","smoke"};
        point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,smoke.ptr};
      }
    else{
        point_names = {"phi","u","v","vorticity","pressure"};
        point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
      }

    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


    // Restore arrays:

    phi.restore_array();
    v_NS.restore_array();

    press.restore_array(); vorticity.restore_array();

    if(solve_smoke) smoke.restore_array();
    // Scale things back:
    foreach_dimension(d){
      VecScaleGhost(v_NS.vec[d],scaling);
    }

    // Scale pressure back:
    VecScaleGhost(press.vec,1./scaling);
} // end of save_navier_stokes_fields
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

  cmdParser cmd;

  pl.initialize_parser(cmd);
  cmd.parse(argc,argv);

  pl.get_all(cmd);
  select_solvers();
  solve_coupled = solve_navier_stokes && solve_stefan;


  PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d, method = %d \n",lmin,lmax,method_);
  PetscPrintf(mpi.comm(),"Stefan = %d, NS = %d, Coupled = %d \n",solve_stefan,solve_navier_stokes,solve_coupled);

  // stopwatch
  parStopWatch w;
  w.start("Running example: multialloy_with_fluids");

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


  // Make sure your flags are set to solve at least one of the problems:
  if(!solve_stefan && !solve_navier_stokes){
      throw std::invalid_argument("Woops, you haven't set options to solve either type of physical problem. \n"
                                  "You must at least set solve_stefan OR solve_navier_stokes to true. ");
    }
  // domain size information
  set_geometry();
  const int n_xyz[]      = { nx,  ny,  0};
  const double xyz_min[] = {xmin, ymin, 0};
  const double xyz_max[] = {xmax,  ymax,  0};
  const int periodic[]   = { px,  py,  0};

  // Get the simulation time info (it is example dependent):
  simulation_time_info();

  // -----------------------------------------------
  // Set properties for the Poisson node problem:
  // -----------------------------------------------
  int cube_refinement = 4;
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // Get diffusivity, conductivity, and interface bc info: (it is example dependent)
  if(solve_stefan){
      set_diffusivities();
      set_conductivities();
      interface_bc();
    }


  // -----------------------------------------------
  // Set properties for the Navier - Stokes problem (if applicable):
  // -----------------------------------------------
  if(solve_navier_stokes){
      set_NS_info();
      interface_bc_pressure();
      interface_bc_velocity_u();
      interface_bc_velocity_v();

    }
  double NS_norm; // for checking the maximum velocity norm of the navier-stokes solution
  PCType pc_face = PCSOR;
  KSPType face_solver_type = KSPBCGS;
  PCType pc_cell = PCSOR;
  KSPType cell_solver_type = KSPBCGS;


  // -----------------------------------------------
  // Scale the problem appropriately:
  // -----------------------------------------------
  double rho_physical = rho_l;
  rho_l/=(scaling*scaling*scaling);

  if(solve_stefan){
      k_s/=scaling;
      k_l/=scaling;
      L/=(scaling*scaling*scaling);

      alpha_l*=(scaling*scaling);
      alpha_s*=(scaling*scaling);
    }


  if(solve_navier_stokes){
      PetscPrintf(mpi.comm(),"Physical u0 = %0.3e \n"
                             "Physical v0 = %0.3e \n"
                             "Physical mu_l = %0.3e \n"
                             "Physical rho_l = %0.3e \n",u0,v0,mu_l,rho_physical);

    mu_l/=(scaling);         // Scale the viscosity depending on the domain
    u0*=scaling;             // Scale the initial velocities
    v0*=scaling;
    pressure_prescribed_value/=(scaling*scaling); // Scale the pressure BC prescribed value and flux
    pressure_prescribed_flux/=(scaling*scaling*scaling);

    PetscPrintf(mpi.comm(),"Reynolds number for this case is: %0.2f , %0.2f \n"
                           "Computational r0 = %0.4f \n"
                           "Computational mu = %0.3e \n"
                           "Computational u0 = %0.3e \n"
                           "Computational rho = %0.3e \n",Re_u, Re_v, r0,mu_l,u0,rho_l);

    PetscPrintf(mpi.comm(),"u initial is %0.3e, v initial is %0.3e \n",u0,v0);
        }

  PetscPrintf(mpi.comm(),"\n -------------------------------------------\n");
  PetscPrintf(mpi.comm(),"Scaling is : %0.2e \n",scaling);
  PetscPrintf(mpi.comm(),"\n Properties once scaled: \n");
  PetscPrintf(mpi.comm(),"kl = %0.2e \n",k_l);
  PetscPrintf(mpi.comm(),"ks = %0.2e \n",k_s);
  PetscPrintf(mpi.comm(),"alpha_l = %0.2e \n",alpha_l);
  PetscPrintf(mpi.comm(),"alpha_s = %0.2e \n",alpha_s);
  PetscPrintf(mpi.comm(),"rho_l = %0.2e \n",rho_l);
  PetscPrintf(mpi.comm(),"L = %0.2e \n",L);

  // -----------------------------------------------
  // Create the grid:
  // -----------------------------------------------
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

  // refine based on distance to a level-set
  //splitting_criteria_cf_t sp(lmin, lmax, &level_set,lip);

  splitting_criteria_cf_and_uniform_band_t sp(lmin,lmax,&level_set,uniform_band);
  p4est->user_pointer = &sp;


                                  // save the pointer to the forst splitting criteria
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL); // refine the grid according to the splitting criteria


  // partition the forest
  my_p4est_partition(p4est, P4EST_FALSE, NULL);                  // partition the forest, do not allow for coarsening --> Daniil does not allow (use P4EST_FALSE)

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL); // same

  // Expand ghost layer -- FOR NAVIER STOKES:
  my_p4est_ghost_expand(p4est,ghost);

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
  // LSF:
  vec_and_ptr_t phi;
  phi.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);

  // LSF for solid domain: -- This will be assigned within the loop as the negative of phi
  vec_and_ptr_t phi_solid;

  // LSF for the inner cylinder, if applicable (example 2):
  vec_and_ptr_t phi_cylinder;
  if(example_ == ICE_AROUND_CYLINDER){
      phi_cylinder.create(phi.vec);
      sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
    }

  // 2nd derivatives of LSF's
  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_cylinder_dd;

  // -----------------------------------------------
  // Initialize the interfacial velocity field (used for Stefan problem)
  // -----------------------------------------------
  vec_and_ptr_dim_t v_interface(p4est,nodes);
  vec_and_ptr_dim_t v_interface_new;
  if(solve_stefan){
      for (int dir = 0; dir<P4EST_DIM;dir++){
          sample_cf_on_nodes(p4est,nodes,zero_cf,v_interface.vec[dir]);
        }
    }


  // -----------------------------------------------
  // Initialize the fields relevant to the Poisson problem:
  // -----------------------------------------------
  // Vectors for T_liquid:
  vec_and_ptr_t T_l_n;
  vec_and_ptr_t rhs_Tl;

  if(solve_stefan){
      T_l_n.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,IC_temp,T_l_n.vec); // Sample this just so that we can save the initial temperature distribution
    }

  // Vector for advection of temperature:
  vec_and_ptr_t T_l_backtrace;
  vec_and_ptr_t T_l_backtrace_nm1;

  // Vectors for T_solid:
  vec_and_ptr_t T_s_n;
  vec_and_ptr_t rhs_Ts;

  if(solve_stefan){
      T_s_n.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,IC_temp,T_s_n.vec); // Sample this just so that we can save the initial temperature distribution
    }


  // Vectors to hold T values on old grid (for interpolation purposes)
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

  // vectors to hold smoke solution:
  vec_and_ptr_t smoke;
  vec_and_ptr_t smoke_new;
  vec_and_ptr_t smoke_np1;

  if (solve_smoke){
      smoke.create(p4est,nodes);
      sample_cf_on_nodes(p4est,nodes,ic_smoke,smoke.vec);
    }
  vec_and_ptr_t smoke_old;
  vec_and_ptr_dim_t smoke_dd; // for Semi Lagrangian backtrace

  vec_and_ptr_t smoke_backtrace;
  vec_and_ptr_t smoke_backtrace_nm1;
  vec_and_ptr_t rhs_smoke;

  // -----------------------------------------------
  // Initialize the Velocity field (if solving Navier-Stokes), and other Navier-Stokes relevant variables:
  // -----------------------------------------------
  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_n_new;

  vec_and_ptr_dim_t v_nm1;
  vec_and_ptr_dim_t v_np1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_cells_t press;
  vec_and_ptr_t press_nodes;

  const CF_DIM *v_init_cf[P4EST_DIM] = {&u_initial, &v_initial};
  const CF_DIM *v_init_NS_validate[P4EST_DIM] = {&u_ana_tn,&v_ana_tn};

  if(solve_navier_stokes){
      v_n.create(p4est,nodes);
      v_nm1.create(p4est,nodes);
      foreach_dimension(d){
        if(example_ ==NS_GIBOU_EXAMPLE){
            sample_cf_on_nodes(p4est,nodes,*v_init_NS_validate[d],v_n.vec[d]);
            sample_cf_on_nodes(p4est,nodes,*v_init_NS_validate[d],v_nm1.vec[d]);
          }
        else{
            sample_cf_on_nodes(p4est,nodes,*v_init_cf[d],v_n.vec[d]);
            sample_cf_on_nodes(p4est,nodes,*v_init_cf[d],v_nm1.vec[d]);
          }
      }

      if(example_ ==NS_GIBOU_EXAMPLE || COUPLED_PROBLEM_EXAMPLE){
          press_nodes.create(p4est,nodes);
          sample_cf_on_nodes(p4est,nodes,p_ana_tn,press_nodes.vec);
        }
    }

  vec_and_ptr_cells_t hodge_old;
  vec_and_ptr_cells_t hodge_new;
  vec_and_ptr_cells_t hodge_old_grid;



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
  sprintf(outdir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_%d",out_idx);

  // -----------------------------------------------
  // Initialize files to output various data of interest:
  // -----------------------------------------------
  // (1) For checking error for Frank Sphere analytical solution:
  FILE *fich;
  char name[1000];
  if (example_ == FRANK_SPHERE){
      if(elyce_laptop) sprintf(name,"/Users/elyce/workspace/projects/multialloy_with_fluids/Frank_Sphere_Errors/Frank_Sphere_error_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,method_);
      else sprintf(name,"/home/elyce/workspace/projects/multialloy_with_fluids/Frank_Sphere_Errors/Frank_Sphere_error_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,method_);


    ierr = PetscFOpen(mpi.comm(),name,"w",&fich); CHKERRXX(ierr);

    ierr = PetscFPrintf(mpi.comm(),fich,"time " "timestep " "iteration " "phi_error " "T_l_error " "T_s_error " "v_int_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich); CHKERRXX(ierr);
    }

  // (2) For checking error on LLNL NS benchmark case:
  FILE *fich_NS_LLNL;
  char name_NS_LLNL[1000];
  if (example_ == NS_GIBOU_EXAMPLE){
      if(elyce_laptop) sprintf(name_NS_LLNL,"/Users/elyce/workspace/projects/multialloy_with_fluids/NS_LLNL_errors/NS_LLNL_Errors_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,advection_sl_order);
      else sprintf(name_NS_LLNL,"/home/elyce/workspace/projects/multialloy_with_fluids/NS_LLNL_errors/NS_LLNL_Errors_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,advection_sl_order);


    ierr = PetscFOpen(mpi.comm(),name_NS_LLNL,"w",&fich_NS_LLNL); CHKERRXX(ierr);

    ierr = PetscFPrintf(mpi.comm(),fich_NS_LLNL,"time " "timestep " "iteration " "u_error " "v_error " "P_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_NS_LLNL); CHKERRXX(ierr);

    PetscPrintf(mpi.comm(),"Began writing file!\n");
    }

  // (3) For checking error on coupled example case:
  FILE *fich_coupled;
  char name_coupled[1000];
  if (example_ == COUPLED_PROBLEM_EXAMPLE){
      if(elyce_laptop) sprintf(name_coupled,"/Users/elyce/workspace/projects/multialloy_with_fluids/Coupled_errors/Coupled_Errors_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,advection_sl_order);
      else sprintf(name_coupled,"/home/elyce/workspace/projects/multialloy_with_fluids/Coupled_errors/Coupled_Errors_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,advection_sl_order);


    ierr = PetscFOpen(mpi.comm(),name_coupled,"w",&fich_coupled); CHKERRXX(ierr);

    ierr = PetscFPrintf(mpi.comm(),fich_coupled,"time " "timestep " "iteration " "u_error " "v_error " "P_error " "Tl_error ""number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_coupled); CHKERRXX(ierr);

    PetscPrintf(mpi.comm(),"Began writing file!\n");
    }



  // (3) For checking memory usage
  FILE *fich_mem;
  char name_mem[1000];
  sprintf(name_mem,"/home/elyce/workspace/projects/multialloy_with_fluids/memory_usage_stefan_%d_NS_%d_lmin_%d_lmax_%d.dat",solve_stefan,solve_navier_stokes,lmin,lmax);

  ierr = PetscFOpen(mpi.comm(),name_mem,"w",&fich_mem); CHKERRXX(ierr);

  ierr = PetscFPrintf(mpi.comm(),fich_mem,"time " "timestep " "iteration " "mem1 mem2 mem3 mem4 mem5 mem6 mem7 mem8 mem9 mem10 mem11 mem12 mem13 \n");CHKERRXX(ierr);
  ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);

  // -----------------------------------------------
  // Initialize the needed solvers for the Temperature problem
  // -----------------------------------------------
  my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

  my_p4est_poisson_nodes_mls_t *solver_smoke; // will solve for smoke over whole domain if being used

  // -----------------------------------------------
  // Initialize the needed solvers for the Navier-Stokes problem
  // -----------------------------------------------
  my_p4est_navier_stokes_t* ns;
  my_p4est_poisson_cells_t* cell_solver;
  my_p4est_poisson_faces_t* face_solver;


  // -----------------------------------------------
  // Begin stepping through time
  // -----------------------------------------------
  int tstep = 0;
  dt = delta_t;

  PetscPrintf(mpi.comm(),"Gets to here ");
  for (tn;tn<tfinal; tn+=dt, tstep++){
      if (!keep_going) break;
//      if(tstep>5) break; // TIMESTEP BREAK

      // Get current memory usage:
      PetscLogDouble mem1;
      PetscMemoryGetCurrentUsage(&mem1);

      // --------------------------------------------------------------------------------------------------------------
      // Print iteration information:
      // --------------------------------------------------------------------------------------------------------------

      PetscPrintf(mpi.comm(),"\n -------------------------------------------\n");
      ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3g , Timestep: %0.3e, Percent Done : %0.2f % \n ------------------------------------------- \n",tstep,tn,dt,(tn/tfinal)*100.0);
      if(solve_stefan){
          ierr = PetscPrintf(mpi.comm(),"\n Previous interfacial velocity (max norm) is %0.3e \n",v_interface_max_norm);
        }


      // --------------------------------------------------------------------------------------------------------------
      // Define some variables needed to specify how to extend across the interface:
      // --------------------------------------------------------------------------------------------------------------
      // Get smallest grid size:
      dxyz_min(p4est,dxyz_smallest);

      dxyz_close_to_interface = 1.0*max(dxyz_smallest[0],dxyz_smallest[1]);
      min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);
      extension_band_use_    = 8.*pow(min_volume_, 1./ double(P4EST_DIM)); //8
      extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
      extension_band_check_  = 6.*pow(min_volume_, 1./ double(P4EST_DIM)); // 6


      if(example_ == ICE_AROUND_CYLINDER){
          double delta_r = r0 - r_cyl;
          PetscPrintf(mpi.comm()," Your initial delta_r is %0.3e, and it must be at least %0.3e \n",delta_r,8.*dxyz_close_to_interface);
          P4EST_ASSERT(delta_r>= 5.*dxyz_close_to_interface);
          if(delta_r<8.*dxyz_close_to_interface){
              SC_ABORT("Your initial delta_r is too small \n");
            }
        }
      // If first iteration, perturb the LSF(s):
      my_p4est_level_set_t ls(ngbd);
      if(tstep<1){
          // Perturb the LSF on the first iteration

          ls.perturb_level_set_function(phi.vec,EPS);
          if(example_ ==ICE_AROUND_CYLINDER ) ls.perturb_level_set_function(phi_cylinder.vec,EPS);
        }
      // --------------------------------------------------------------------------------------------------------------
      // Extend Fields Across Interface (if solving Stefan):
      // -- Note: we do not extend NS velocity fields bc NS solver handles that internally
      // --------------------------------------------------------------------------------------------------------------
      // Define LSF for the solid domain (as just the negative of the liquid one):
      if(solve_stefan){
          phi_solid.create(p4est,nodes);
          VecScaleGhost(phi.vec,-1.0);
          VecCopyGhost(phi.vec,phi_solid.vec);
          VecScaleGhost(phi.vec,-1.0);

          if(check_temperature_values){
            // Check Temperature values:
            PetscPrintf(mpi.comm(),"\n Checking temperature values before field extension: \n [ ");
            PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
            check_T_values(phi,T_l_n,nodes,p4est, example_,phi_cylinder);
            PetscPrintf(mpi.comm(),"\nIn solid domain: ");
            check_T_values(phi_solid,T_s_n,nodes,p4est,example_,phi_cylinder);
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

          // Extend Temperature Fields across the interface: // WAS USING 1ST ORDER, NOW CHANGED TO SECOND
          ls.extend_Over_Interface_TVD_Full(phi.vec, T_l_n.vec, 50, 2, 1.e-9, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normals.vec, NULL, NULL, false, NULL, NULL);
          ls.extend_Over_Interface_TVD_Full(phi_solid.vec, T_s_n.vec, 50, 2, 1.e-9, extension_band_use_, extension_band_extend_, extension_band_check_, solid_normals.vec, NULL, NULL, false, NULL, NULL);

          // For the case where we have a second interface:
          if(example_ == ICE_AROUND_CYLINDER){
              phi_cylinder.create(p4est,nodes);
              sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);

              cyl_normals.create(p4est,nodes);
              compute_normals(*ngbd,phi_cylinder.vec,cyl_normals.vec);

              ls.extend_Over_Interface_TVD_Full(phi_cylinder.vec,T_s_n.vec,50,2,1.e-9,extension_band_use_,extension_band_extend_,extension_band_extend_,cyl_normals.vec,NULL,NULL,false,NULL,NULL);
              cyl_normals.destroy();
            }

          // Delete data for normals since it is no longer needed:
          liquid_normals.destroy();
          solid_normals.destroy();

          if (check_temperature_values){
            // Check Temperature values:
            PetscPrintf(mpi.comm(),"\n Checking temperature values after field extension: \n [ ");
            PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
            check_T_values(phi,T_l_n,nodes,p4est,example_,phi_cylinder);
            PetscPrintf(mpi.comm(),"\nIn solid domain: ");
            check_T_values(phi_solid,T_s_n,nodes,p4est,example_,phi_cylinder);
            PetscPrintf(mpi.comm()," ] \n");

            // Check Smoke values:
            PetscPrintf(mpi.comm(),"\n Checking smoke values after interpolating onto new grid: \n [ ");
            PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
            check_T_values(phi,smoke,nodes,p4est,example_,phi_cylinder);
            PetscPrintf(mpi.comm(),"\nIn solid domain: ");
            check_T_values(phi_solid,smoke,nodes,p4est,example_,phi_cylinder);
            PetscPrintf(mpi.comm()," ] \n");
            }


/*
          if (check_derivative_values){
              // Check Temperature derivative values:
              PetscPrintf(mpi.comm(),"\n Checking temperature derivative values after field extension: \n [ ");
              PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
              check_T_d_values(phi,T_l_d,nodes,p4est,0);
              PetscPrintf(mpi.comm(),"\nIn solid domain: ");
              check_T_d_values(phi_solid,T_s_d,nodes,p4est,0);
              PetscPrintf(mpi.comm()," ] \n");
            }
*/
        } // end of "if save stefan"

      PetscLogDouble mem2;
      PetscMemoryGetCurrentUsage(&mem2);
      // --------------------------------------------------------------------------------------------------------------
      // SAVING DATA: Save data every specified amout of timesteps: -- Do this after values are extended across interface to make visualization nicer
      // --------------------------------------------------------------------------------------------------------------
      // Check errors on NS  benchmark case: // TEMPORARY LOCATION
      if(example_ == NS_GIBOU_EXAMPLE){
          check_NS_LLNL_benchmark_error(phi,v_n,press_nodes,p4est,nodes,ghost,ngbd,dxyz_close_to_interface,name_NS_LLNL,fich_NS_LLNL,tstep);
        }
      if(example_ == COUPLED_PROBLEM_EXAMPLE){
          check_coupled_problem_error(phi,v_n,press_nodes,T_l_n,p4est,nodes,ngbd,dxyz_close_to_interface,name_coupled,fich_coupled,tstep);
        }
      int save_every_iter = 1;
      if((tstep>0 && tstep%save_every_iter ==0)){
          PetscPrintf(mpi.comm(),"Saving to vtk ... \n");
        char output[1000];
        if(save_coupled_fields){
            if(elyce_laptop) sprintf(output,"/Users/elyce/workspace/projects/multialloy_with_fluids/output/snapshot_full_%d",tstep);
            else sprintf(output,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_full_%d",tstep);
            save_everything(p4est,nodes,ghost,phi,phi_cylinder,T_l_n,T_s_n,v_interface,v_n,press_nodes,vorticity,smoke,output);

            if(example_ == COUPLED_PROBLEM_EXAMPLE){
                // Save NS analytical to compare:
                vec_and_ptr_dim_t vn_analytical;
                vec_and_ptr_t pn_analytical;
                vec_and_ptr_t Tn_analytical;

                vn_analytical.create(p4est,nodes);
                pn_analytical.create(p4est,nodes);
                Tn_analytical.create(p4est,nodes);

                sample_cf_on_nodes(p4est,nodes,u_ana_tn,vn_analytical.vec[0]);
                sample_cf_on_nodes(p4est,nodes,v_ana_tn,vn_analytical.vec[1]);
                sample_cf_on_nodes(p4est,nodes,p_ana_tn,pn_analytical.vec);
                sample_cf_on_nodes(p4est,nodes,T_ana_tn,Tn_analytical.vec);

                // Get errors:
                vec_and_ptr_dim_t vn_error;
                vec_and_ptr_t press_error;
                vec_and_ptr_t Tn_error;
                vn_error.create(p4est,nodes);
                press_error.create(p4est,nodes);
                Tn_error.create(p4est,nodes);

                vn_analytical.get_array(); vn_error.get_array(); v_n.get_array();
                pn_analytical.get_array(); press_error.get_array(); press_nodes.get_array();
                Tn_analytical.get_array();Tn_error.get_array();T_l_n.get_array();
                foreach_local_node(n,nodes){
                  press_error.ptr[n] = fabs(press_nodes.ptr[n] - pn_analytical.ptr[n]);
                  vn_error.ptr[0][n] = fabs(v_n.ptr[0][n] - vn_analytical.ptr[0][n]);
                  vn_error.ptr[1][n] = fabs(v_n.ptr[1][n] - vn_analytical.ptr[1][n]);
                  Tn_error.ptr[n] = fabs(T_l_n.ptr[n] - Tn_analytical.ptr[n]);

                }
                VecGhostUpdateBegin(press_error.vec,INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateBegin(vn_error.vec[0],INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateBegin(vn_error.vec[1],INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateBegin(Tn_error.vec,INSERT_VALUES,SCATTER_FORWARD);


                VecGhostUpdateEnd(press_error.vec,INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateEnd(vn_error.vec[0],INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateEnd(vn_error.vec[1],INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateEnd(Tn_error.vec,INSERT_VALUES,SCATTER_FORWARD);


                vn_analytical.restore_array(); vn_error.restore_array(); v_n.restore_array();
                pn_analytical.restore_array();press_error.restore_array(); press_nodes.restore_array();
                Tn_analytical.restore_array(); Tn_error.restore_array(); T_l_n.restore_array();

                sprintf(output,"/home/elyce/workspace/projects/multialloy_with_fluids//solidif_with_fluids_output/snapshot_coupled_analytical_%d",tstep);
                vn_analytical.get_array(); pn_analytical.get_array();
                vn_error.get_array(); press_error.get_array(); phi.get_array();
                Tn_error.get_array(); Tn_analytical.get_array();
                my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,9,0,output,
                                       VTK_POINT_DATA,"u_ana",vn_analytical.ptr[0],
                                       VTK_POINT_DATA,"v_ana",vn_analytical.ptr[1],
                                       VTK_POINT_DATA,"P_ana",pn_analytical.ptr,
                                       VTK_POINT_DATA,"Tl_ana",Tn_analytical.ptr,
                                       VTK_POINT_DATA,"u_err",vn_error.ptr[0],
                                       VTK_POINT_DATA,"v_err",vn_error.ptr[1],
                                       VTK_POINT_DATA,"P_err",press_error.ptr,
                                       VTK_POINT_DATA,"Tl_err",Tn_error.ptr,
                                       VTK_POINT_DATA,"phi",phi.ptr);
                vn_analytical.restore_array(); pn_analytical.restore_array();
                vn_error.restore_array(); press_error.restore_array(); phi.restore_array();
                Tn_analytical.restore_array();Tn_error.restore_array();

                vn_analytical.destroy();
                pn_analytical.destroy();
                Tn_analytical.destroy();
                vn_error.destroy();
                press_error.destroy();
                Tn_error.destroy();
              }
          }
        if(save_stefan){
            if(elyce_laptop) sprintf(output,"/Users/elyce/workspace/projects/multialloy_with_fluids/output/snapshot_stefan_%d",tstep);
            else sprintf(output,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_stefan_%d",tstep);
            PetscPrintf(mpi.comm(),"%s \n",output);

            save_stefan_fields(p4est,nodes,ghost,phi,phi_cylinder,T_l_n,T_s_n,v_interface,smoke,output);
          }
        if(save_navier_stokes){
            if(elyce_laptop) sprintf(output,"/Users/elyce/workspace/projects/multialloy_with_fluids/output/snapshot_NS_%d",tstep);
            else sprintf(output,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_NS_%d",tstep);
            save_navier_stokes_fields(p4est,nodes,ghost,phi,v_n,press_nodes,vorticity,smoke,output);

            if(example_ == NS_GIBOU_EXAMPLE){
                // Save NS analytical to compare:
                vec_and_ptr_dim_t vn_analytical;
                vec_and_ptr_t pn_analytical;
                vn_analytical.create(p4est,nodes);
                pn_analytical.create(p4est,nodes);

                sample_cf_on_nodes(p4est,nodes,u_ana_tn,vn_analytical.vec[0]);
                sample_cf_on_nodes(p4est,nodes,v_ana_tn,vn_analytical.vec[1]);
                sample_cf_on_nodes(p4est,nodes,p_ana_tn,pn_analytical.vec);

                // Get errors:
                vec_and_ptr_dim_t vn_error;
                vec_and_ptr_t press_error;
                vn_error.create(p4est,nodes);
                press_error.create(p4est,nodes);

                vn_analytical.get_array(); vn_error.get_array(); v_n.get_array();
                pn_analytical.get_array(); press_error.get_array(); press_nodes.get_array();
                foreach_local_node(n,nodes){
                  press_error.ptr[n] = fabs(press_nodes.ptr[n] - pn_analytical.ptr[n]);
                  vn_error.ptr[0][n] = fabs(v_n.ptr[0][n] - vn_analytical.ptr[0][n]);
                  vn_error.ptr[1][n] = fabs(v_n.ptr[1][n] - vn_analytical.ptr[1][n]);

                }
                VecGhostUpdateBegin(press_error.vec,INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateBegin(vn_error.vec[0],INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateBegin(vn_error.vec[1],INSERT_VALUES,SCATTER_FORWARD);

                VecGhostUpdateEnd(press_error.vec,INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateEnd(vn_error.vec[0],INSERT_VALUES,SCATTER_FORWARD);
                VecGhostUpdateEnd(vn_error.vec[1],INSERT_VALUES,SCATTER_FORWARD);


                vn_analytical.restore_array(); vn_error.restore_array(); v_n.restore_array();
                pn_analytical.restore_array();press_error.restore_array(); press_nodes.restore_array();

                sprintf(output,"/home/elyce/workspace/projects/multialloy_with_fluids//solidif_with_fluids_output/snapshot_NS_analytical_%d",tstep);
                vn_analytical.get_array(); pn_analytical.get_array();
                vn_error.get_array(); press_error.get_array(); phi.get_array();
                my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,7,0,output,
                                       VTK_POINT_DATA,"u_ana",vn_analytical.ptr[0],
                                       VTK_POINT_DATA,"v_ana",vn_analytical.ptr[1],
                                       VTK_POINT_DATA,"P_ana",pn_analytical.ptr,
                                       VTK_POINT_DATA,"u_err",vn_error.ptr[0],
                                       VTK_POINT_DATA,"v_err",vn_error.ptr[1],
                                       VTK_POINT_DATA,"P_err",press_error.ptr,
                                       VTK_POINT_DATA,"phi",phi.ptr);
                vn_analytical.restore_array(); pn_analytical.restore_array();
                vn_error.restore_array(); press_error.restore_array(); phi.restore_array();

                vn_analytical.destroy();
                pn_analytical.destroy();
                vn_error.destroy();
                press_error.destroy();
              }

          }
        }
//      char output[1000];

//      sprintf(output,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_stefan_%d",tstep);
//      save_stefan_fields(p4est,nodes,ghost,phi,phi_cylinder,T_l_n,T_s_n,v_interface,smoke,output);
//      // Enforce that the interfacial velocity is within a reasonable range specified by the user:
//      P4EST_ASSERT(v_interface_max_norm<v_int_max_allowed);

      PetscLogDouble mem3;
      PetscMemoryGetCurrentUsage(&mem3);

//      // Check if temperature is in reasonable range and break if not:
//      T_l_n.get_array();
//      double Tmax = 0.0;
//      foreach_node(n,nodes){
//        Tmax = max(Tmax,T_l_n.ptr[n]);
//      }
//      T_l_n.restore_array();
//      keep_going = (Tmax < Tmax_allowed);

      // --------------------------------------------------------------------------------------------------------------
      // Compute the jump in flux across the interface to use to advance the LSF (if solving Stefan:
      // --------------------------------------------------------------------------------------------------------------
      if(solve_stefan){

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
        }

      PetscLogDouble mem4;
      PetscMemoryGetCurrentUsage(&mem4);
      // --------------------------------------------------------------------------------------------------------------
      // Compute the timestep -- determined by velocity at the interface:
      // --------------------------------------------------------------------------------------------------------------
      // Save previous timestep:
      dt_nm1 = dt;
      if(solve_stefan){

          compute_timestep(v_interface, phi, dxyz_close_to_interface, dxyz_smallest,nodes,p4est); // this function modifies the variable dt
        }

      if(solve_navier_stokes){
          // STILL TO FIGURE OUT: GETTING NS TIMESTEP that makes sense
          // Take into consideration the Navier - Stokes timestep:
          // Take into account the NS timestep: -- probably better to do this with the max NS norm and CFL in the main file, not internally in NS
          if(solve_stefan){
              PetscPrintf(mpi.comm(),"\nComputed timesteps: \n"
                                     "Stefan: %0.3e \n"
                                     "Navier Stokes: %0.3e \n"
                                     "Official : %0.3e \n \n",dt,dt_NS,min(dt,dt_NS));
              dt = min(dt,dt_NS);
            }
          else{
              // If we are only solving Navier Stokes
              if(tstep>0) dt = dt_NS;
            }
        }

      // Adjust the timestep if we are near the end of our simulation, to get the proper end time:
      if(tn + dt > tfinal){
          dt = tfinal - tn;
        }

      PetscLogDouble mem5;
      PetscMemoryGetCurrentUsage(&mem5);
      // --------------------------------------------------------------------------------------------------------------
      // Advance the LSF/Update the grid :
      // --------------------------------------------------------------------------------------------------------------
      // In coupled case, we advect the LSF and update the grid according to both vorticity and phi
      // In Stefan case, we advect the LSF and update the grid according to the LSF
      // In NS case, we simply update the grid according to the LSF, with no advection of LSF

      // Make a copy of the grid objects for the next timestep:
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);

      // Expand the ghost layer for navier stokes:
      my_p4est_ghost_expand(p4est_np1,ghost_np1);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Create the semi-lagrangian object and do the advection:
      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

      // Build refinement criteria for Navier - Stokes problem:
      if(tstep == 0 && solve_navier_stokes){
          vorticity.create(p4est,nodes);
          sample_cf_on_nodes(p4est,nodes,zero_cf,vorticity.vec);
          NS_norm = max(u0,v0);
        }
      std::vector<compare_option_t> compare_opn;
      std::vector<compare_diagonal_option_t> diag_opn;
      std::vector<double> criteria;
      const int num_fields = 1;
      bool use_block = false;
      bool expand_ghost_layer = true;
      double threshold = 0.1;

      Vec fields_[num_fields];
      if(solve_navier_stokes){
          // Only use values of vorticity in the positive subdomain for refinement:
          vec_and_ptr_t vorticity_refine;
          vorticity_refine.create(p4est,nodes);

          vorticity.get_array();
          vorticity_refine.get_array();
          phi.get_array();

          foreach_local_node(n,nodes){
            if(phi.ptr[n] < dxyz_close_to_interface){
                vorticity_refine.ptr[n] = vorticity.ptr[n];
              }
            else{
                vorticity_refine.ptr[n] = 0.0;
              }
          }

          vorticity.restore_array();
          vorticity_refine.restore_array();
          phi.restore_array();

          fields_[0] = vorticity_refine.vec;
//          fields_[1] = phi.vec; // Will use this for a uniform band criteria

          // Coarsening instructions: (for vorticity)
          compare_opn.push_back(LESS_THAN);
          diag_opn.push_back(DIVIDE_BY);
          criteria.push_back(threshold*NS_norm/2.);

          // Refining instructions: (for vorticity)
          compare_opn.push_back(GREATER_THAN);
          diag_opn.push_back(DIVIDE_BY);
          criteria.push_back(threshold*NS_norm);

//          // Coarsening instructions: (for uniform band around interface)
//          compare_opn.push_back(GREATER_THAN);
//          diag_opn.push_back(ABSOLUTE);
//          criteria.push_back(uniform_band*dxyz_close_to_interface);

//          // Refining instructions: (for uniform band around interface)
//          compare_opn.push_back(LESS_THAN);
//          diag_opn.push_back(ABSOLUTE);
//          criteria.push_back(uniform_band*NS_norm);
        }


      // Advect the LSF and update the grid under the v_interface field:
      if(solve_coupled){
          example_ == ICE_AROUND_CYLINDER ?
            sl.update_p4est(v_interface.vec, dt, phi.vec, phi_dd.vec, phi_cylinder.vec,num_fields ,use_block ,fields_ ,NULL,criteria,compare_opn,diag_opn,expand_ghost_layer):
            sl.update_p4est(v_interface.vec, dt, phi.vec, phi_dd.vec, NULL,num_fields ,use_block ,fields_ ,NULL,criteria,compare_opn,diag_opn,expand_ghost_layer);
        }
      else if (solve_stefan){
          example_ == ICE_AROUND_CYLINDER ? // for example ice around cylinder, refine around both LSFs. Otherwise, refine around just the one
                sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec,phi_cylinder.vec):
                sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec);
        }
      else if (solve_navier_stokes){
          splitting_criteria_tag_t sp_NS(sp.min_lvl,sp.max_lvl,sp.lip);

          // Create a new vector which will hold the updated values of the fields -- since we will interpolate with each grid iteration
          Vec fields_new_[num_fields];
          for(unsigned int k = 0;k<num_fields; k++){
              ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
              ierr = VecCopyGhost(fields_[k],fields_new_[k]);
            }
          // Create a vector which will hold the updated values of the LSF:
          vec_and_ptr_t phi_new;
          phi_new.create(p4est,nodes);
          ierr = VecCopyGhost(phi.vec,phi_new.vec);

          bool is_grid_changing = true;
          int no_grid_changes = 0;
          while(is_grid_changing){
              is_grid_changing = sp_NS.refine_and_coarsen(p4est_np1,nodes_np1,phi_new.vec,num_fields,use_block,fields_new_,NULL,criteria,compare_opn,diag_opn);

              if(is_grid_changing){
                  no_grid_changes++;
                  PetscPrintf(mpi.comm(),"NS grid changed %d times \n",no_grid_changes);

                  my_p4est_partition(p4est_np1,P4EST_TRUE,NULL);
                  p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
                  p4est_ghost_expand(p4est_np1,ghost_np1);
                  p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);

//                  char outNS_file[1000];
//                  sprintf(outNS_file,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/NS_grid_intermediate_%d",no_grid_changes);
//                  my_p4est_vtk_write_all(p4est_np1,nodes_np1,ghost_np1,P4EST_TRUE,P4EST_TRUE,0,0,outNS_file);

                  // Destroy fields_new and create it on the new grid:
                  for(unsigned int k = 0;k<num_fields; k++){
                      ierr = VecDestroy(fields_new_[k]);
                      ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
                    }

                  // Destroy phi_new and create on new grid:
                  phi_new.destroy();
                  phi_new.create(p4est_np1,nodes_np1);

                  // Interpolate the fields and phi to the new grid:
                  my_p4est_interpolation_nodes_t interp_refine_and_coarsen(ngbd);
                  interp_refine_and_coarsen.set_input(fields_,quadratic_non_oscillatory_continuous_v2,num_fields);
                  double xyz_interp[P4EST_DIM];
                  foreach_node(n,nodes_np1){
                    node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz_interp);
                    interp_refine_and_coarsen.add_point(n,xyz_interp);
                  }
                  // Interpolate fields
                  interp_refine_and_coarsen.interpolate(fields_new_);

                  // Interpolate the phi onto the new grid:
                  interp_refine_and_coarsen.set_input(phi.vec,quadratic_non_oscillatory_continuous_v2);
                  interp_refine_and_coarsen.interpolate(phi_new.vec);

                } // End of if grid is changing


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

      // Destroy old derivative values:
      phi_dd.destroy();

      // Get the new neighbors:
      my_p4est_hierarchy_t *hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
      my_p4est_node_neighbors_t *ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

      // Initialize the neigbors:
      ngbd_np1->init_neighbors();

      // FOR FUTURE NOTICE :: functions that exist are: ngbd->update() and hierarchy->update() --> Look at how Daniil does it

      PetscLogDouble mem6;
      PetscMemoryGetCurrentUsage(&mem6);

      // Reinitialize the LSF on the new grid: -- NOT SURE IF WE NEED TO DO THIS EVERY TIME IF JUST NS AND NOTHING ELSE
      my_p4est_level_set_t ls_new(ngbd_np1);
      ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec, 50);
      ls_new.perturb_level_set_function(phi.vec,EPS);

      PetscLogDouble mem7;
      PetscMemoryGetCurrentUsage(&mem7);
      // --------------------------------------------------------------------------------------------------------------
      // Interpolate Values onto New Grid:
      // -------------------------------------------------------------------------------------------------------------
      // Create vectors to hold new values:
      if(solve_stefan){
          T_l_new.create(p4est_np1,nodes_np1);
          T_s_new.create(T_l_new.vec);
          v_interface_new.create(p4est_np1,nodes_np1);
        }

      if(solve_navier_stokes){
          v_n_new.create(p4est_np1,nodes_np1);
        }


      vec_and_ptr_t smoke_new;
      if (solve_smoke){
          smoke_new.create(T_l_new.vec);
        }

      // Interpolate things to the new grid:
      interpolate_values_onto_new_grid(T_l_n,T_l_new,
                                       T_s_n, T_s_new,
                                       v_interface, v_interface_new,
                                       v_n, v_n_new,
                                       smoke, smoke_new,
                                       nodes_np1, p4est_np1, nodes,p4est,
                                       ngbd, interp_bw_grids);

      // Copy new data over:
      // Transfer new values to the original objects:
      if(solve_stefan){
          T_l_n.destroy(); T_s_n.destroy();
          T_l_n.create(p4est_np1,nodes_np1); T_s_n.create(T_l_n.vec);

          v_interface.destroy();
          v_interface.create(p4est_np1,nodes_np1);

          VecCopyGhost(T_l_new.vec,T_l_n.vec);
          VecCopyGhost(T_s_new.vec,T_s_n.vec);

          foreach_dimension(d){
            VecCopyGhost(v_interface_new.vec[d],v_interface.vec[d]);
          }

          // Delete the "new value" objects until the next timestep:
          T_l_new.destroy(); T_s_new.destroy();
          v_interface_new.destroy();

          // Get the new solid LSF:
          phi_solid.destroy();
          phi_solid.create(p4est_np1,nodes_np1);
          VecScaleGhost(phi.vec,-1.0);
          VecCopyGhost(phi.vec,phi_solid.vec);
          VecScaleGhost(phi.vec,-1.0);


        } // end of if solve stefan

      if(solve_smoke) {
          smoke.destroy();
          smoke.create(T_l_n.vec);
          VecCopyGhost(smoke_new.vec,smoke.vec);
          smoke_new.destroy();
        }

      if(solve_navier_stokes){
          v_n.destroy(); v_n.create(p4est_np1,nodes_np1);
          foreach_dimension(d){
            VecCopyGhost(v_n_new.vec[d],v_n.vec[d]);
          }
          v_n_new.destroy();
        }

      PetscLogDouble mem8;
      PetscMemoryGetCurrentUsage(&mem8);
      // --------------------------------------------------------------------------------------------------------------
      // Compute the normal and curvature of the interface -- curvature is used in some of the interfacial boundary condition(s) on temperature
      // --------------------------------------------------------------------------------------------------------------
      vec_and_ptr_dim_t normal;
      vec_and_ptr_t curvature_tmp; // This one will hold computed curvature
      vec_and_ptr_t curvature;  // This one will hold curvature extended from interface to whole domain

      if(solve_stefan){
          normal.create(p4est_np1,nodes_np1);
          curvature.create(p4est_np1,nodes_np1);
          // Compute normals on the interface:
          compute_normals(*ngbd_np1,phi.vec,normal.vec);

          // Compute curvature on the interface:
          my_p4est_level_set_t ls_new_new(ngbd_np1);
          compute_curvature(phi,normal,curvature,ngbd_np1,ls_new_new);

          // Destroy normal:
          normal.destroy();

        }


      PetscLogDouble mem9;
      PetscMemoryGetCurrentUsage(&mem9);
      // --------------------------------------------------------------------------------------------------------------
      // Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
      // --------------------------------------------------------------------------------------------------------------
      // Get most updated derivatives of the LSF's (on current grid) -- Solver uses these:
      // ------------------------------------------------------------
      if(solve_stefan){
        PetscPrintf(mpi.comm(),"Beginning Poisson problem ... \n");
        phi_solid_dd.create(p4est_np1,nodes_np1);
        ngbd_np1->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

        phi_dd.create(p4est_np1,nodes_np1);
        ngbd_np1->second_derivatives_central(phi.vec,phi_dd.vec);

        if(example_ ==ICE_AROUND_CYLINDER){
            phi_cylinder.destroy();
            phi_cylinder.create(p4est_np1,nodes_np1);
            sample_cf_on_nodes(p4est_np1,nodes_np1,mini_level_set,phi_cylinder.vec);

            phi_cylinder_dd.create(p4est_np1,nodes_np1);
            ngbd_np1->second_derivatives_central(phi_cylinder.vec,phi_cylinder_dd.vec);
          }


        // Do quick optional check of values after interpolation: --> don't check till now bc we need phi_cylinder on new grid for ex 2
        if (check_temperature_values){
          // Check Temperature values:
          PetscPrintf(mpi.comm(),"\n Checking temperature values after interpolating onto new grid: \n [ ");
          PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
          check_T_values(phi,T_l_n,nodes_np1,p4est_np1,example_,phi_cylinder);
          PetscPrintf(mpi.comm(),"\nIn solid domain: ");
          check_T_values(phi_solid,T_s_n,nodes_np1,p4est_np1,example_,phi_cylinder);
          PetscPrintf(mpi.comm()," ] \n");

          // Check Smoke values:
          PetscPrintf(mpi.comm(),"\n Checking smoke values after interpolating onto new grid: \n [ ");
          PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
          check_T_values(phi,smoke,nodes_np1,p4est_np1,example_,phi_cylinder);
          PetscPrintf(mpi.comm(),"\nIn solid domain: ");
          check_T_values(phi_solid,smoke,nodes_np1,p4est_np1,example_,phi_cylinder);
          PetscPrintf(mpi.comm()," ] \n");
          }

        // ---------------------------------------
        // Compute advection terms (if applicable):
        // ---------------------------------------
        if (do_advection){
            PetscPrintf(mpi.comm(),"Gets into do advection \n\n");
            // Create backtrace vectors:
            T_l_backtrace.destroy();
            T_l_backtrace.create(p4est_np1,nodes_np1);

            if(advection_sl_order ==2){
                T_l_backtrace_nm1.destroy();
                T_l_backtrace_nm1.create(p4est_np1,nodes_np1);
              }

            if (solve_smoke){
                smoke_backtrace.destroy();
                smoke_backtrace.create(T_l_backtrace.vec);

                if(advection_sl_order ==2){
                    smoke_backtrace_nm1.destroy();
                    smoke_backtrace_nm1.create(p4est_np1,nodes_np1);
                  }
              }
            do_backtrace(T_l_n,T_l_backtrace,v_n,smoke,smoke_backtrace,p4est_np1,nodes_np1,ngbd_np1,p4est,nodes,ngbd, T_l_backtrace_nm1,v_nm1,smoke_backtrace_nm1,interp_bw_grids);

            // Do backtrace with v_n --> navier-stokes fluid velocity
        } // end of do_advection if statement

        // ------------------------------------------------------------
        // Setup RHS:
        // ------------------------------------------------------------
        // Create arrays to hold the RHS:
        rhs_Tl.create(p4est_np1,nodes_np1);
        rhs_Ts.create(p4est_np1,nodes_np1);
        if (solve_smoke) rhs_smoke.create(p4est_np1,nodes_np1);

        // Set up the RHS:

        setup_rhs(T_l_n,T_s_n,smoke,
                  rhs_Tl,rhs_Ts,rhs_smoke,
                  T_l_backtrace,smoke_backtrace, T_l_backtrace_nm1,smoke_backtrace_nm1,
                  p4est_np1,nodes_np1,ngbd_np1);

        // ------------------------------------------------------------
        // Setup the solvers:
        // ------------------------------------------------------------
        // Now, set up the solver(s):
        solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd_np1);
        solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd_np1);

        BC_interface_value bc_interface_val(ngbd_np1,normal,curvature);

        // Add the appropriate interfaces and interfacial boundary conditions:
        solver_Tl->add_boundary(MLS_INTERSECTION,phi.vec,phi_dd.vec[0],phi_dd.vec[1],interface_bc_type_temp,bc_interface_val,bc_interface_coeff);
        solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid.vec,phi_solid_dd.vec[0],phi_solid_dd.vec[1],interface_bc_type_temp,bc_interface_val,bc_interface_coeff);

        if(example_ == ICE_AROUND_CYLINDER){
          solver_Ts->add_boundary(MLS_INTERSECTION,phi_cylinder.vec,phi_cylinder_dd.vec[0],phi_cylinder_dd.vec[1],inner_interface_bc_type_temp,bc_interface_val_inner,bc_interface_coeff_inner);
          }


        // Set diagonal for Tl:
        if(do_advection){ // Cases with advection use semi lagrangian advection discretization in time
            if(advection_sl_order ==2){ // 2nd order semi lagrangian (BDF2 coefficients)
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

        solver_Tl->set_mu(alpha_l);
        solver_Tl->set_rhs(rhs_Tl.vec);

        solver_Ts->set_mu(alpha_s);
        solver_Ts->set_rhs(rhs_Ts.vec);

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
        solver_Tl ->set_wc(wall_bc_type_temp,wall_bc_value_temp);
        solver_Ts ->set_wc(wall_bc_type_temp,wall_bc_value_temp);

        // Preassemble the linear system
        solver_Tl->preassemble_linear_system();
        solver_Ts->preassemble_linear_system();

        // Create vector to hold the solution:
        T_l_np1.create(p4est_np1,nodes_np1);
        T_s_np1.create(T_l_np1.vec);

        // Solve the system:
        solver_Tl->solve(T_l_np1.vec);
        PetscPrintf(mpi.comm(),"Solved Tl \n");
        MPI_Barrier(mpi.comm());

        solver_Ts->solve(T_s_np1.vec);
        PetscPrintf(mpi.comm(),"Solved Ts \n");
        MPI_Barrier(mpi.comm());

        // Destroy the T_n values now and update them with the solution for the next timestep:
        T_l_n.destroy(); T_s_n.destroy();
        T_l_n.create(p4est_np1,nodes_np1); T_s_n.create(T_l_n.vec);


        VecCopyGhost(T_l_np1.vec,T_l_n.vec);
        VecCopyGhost(T_s_np1.vec,T_s_n.vec);

        MPI_Barrier(mpi.comm());
        if(solve_smoke){
            // FOR now: smoke has same diffusivity as the liquid phase, and we solve with no interfacial condition
            // Eventually: solve smoke with two RHSs, two diffusivities, and a jump condition on the interface -- enforce a zero jump in value and flux
            PetscPrintf(mpi.comm(),"Beginning solver setup for smoke \n");
            solver_smoke = new my_p4est_poisson_nodes_mls_t(ngbd_np1);

//            solver_smoke->add_boundary(MLS_ADD,phi.vec,phi_dd.vec[0],phi_dd.vec[1],NEUMANN,zero_cf,zero_cf);
            solver_smoke->set_diag(1./dt);

            solver_smoke->set_mu(alpha_l);//alpha_l

            solver_smoke->set_integration_order(1);
            solver_smoke->set_use_sc_scheme(0);
            solver_smoke->set_cube_refinement(cube_refinement);
            solver_smoke->set_store_finite_volumes(1);

            // Set wall BC and RHS:
            solver_smoke ->set_wc(bc_type_smoke,bc_smoke_value);
            solver_smoke->set_rhs(rhs_smoke.vec);
            solver_smoke->preassemble_linear_system();

            // Create vector to hold the solution:
            smoke_np1.create(p4est_np1,nodes_np1);

            // Solve the system:
            PetscPrintf(mpi.comm(),"Calling solve_smoke \n");
            solver_smoke->solve(smoke_np1.vec);
            PetscPrintf(mpi.comm(),"Completed solve smoke \n");

            // Destroy the n values now and update them with the solution for the next timestep:
            smoke.destroy();
            smoke.create(p4est_np1,nodes_np1);
            VecCopyGhost(smoke_np1.vec,smoke.vec);

            // Destroy np1 now that not needed:
            smoke_np1.destroy();
            rhs_smoke.destroy();

            delete solver_smoke;
          }

        // Destroy solvers once done:
        delete solver_Tl;
        delete solver_Ts;

        // Destroy np1 vectors now that theyre not needed:
        T_l_np1.destroy(); T_s_np1.destroy();

        phi_dd.destroy(); phi_solid_dd.destroy();
        phi_solid.destroy();
        curvature.destroy();

        // Destroy other fields that are no longer needed:
        if(example_ ==ICE_AROUND_CYLINDER){
            phi_cylinder.destroy();
            phi_cylinder_dd.destroy();
          }

        // Destroy rhs vectors now that no longer in use:
        rhs_Tl.destroy();
        rhs_Ts.destroy();
        if (check_temperature_values){
          // Check Temperature values:
          PetscPrintf(mpi.comm(),"\n Checking temperature values after acquiring solution: \n [ ");
          PetscPrintf(mpi.comm(),"\n In fluid domain: ");
          check_T_values(phi,T_l_n,nodes_np1,p4est_np1,example_,phi_cylinder);
          PetscPrintf(mpi.comm(),"\n In solid domain: ");
          check_T_values(phi_solid,T_s_n,nodes_np1,p4est_np1,example_,phi_cylinder);
          PetscPrintf(mpi.comm()," ] \n");
          }

        // ------------------------------------------------------------
        // Some example specific operations for the Poisson problem:
        // ------------------------------------------------------------
        // Check error on the Frank sphere, if relevant:
        if(example_ == FRANK_SPHERE){
            check_frank_sphere_error(T_l_n, T_s_n, phi, v_interface, p4est_np1, nodes_np1, dxyz_close_to_interface,name,fich,tstep);
          }
//        // Check if ice has melted, if relevant:
//        if (example_ == 0){
//            keep_going = check_ice_melted(phi,tn+dt,nodes_np1,p4est_np1);
//          }

        PetscPrintf(mpi.comm(),"Poisson step complete \n\n");
    } // end of "if solve stefan"

      PetscLogDouble mem10;
      PetscMemoryGetCurrentUsage(&mem10);
      // --------------------------------------------------------------------------------------------------------------
      // Navier-Stokes Problem: Setup and solve a NS problem in the liquid subdomain
      // --------------------------------------------------------------------------------------------------------------

      if (solve_navier_stokes){
          PetscPrintf(mpi.comm(),"Beginning Navier-Stokes problem ... \n");
          // Get the cell neighbors:
          my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_np1);

          // Create the faces:
          my_p4est_faces_t *faces_np1 = new my_p4est_faces_t(p4est_np1,ghost_np1,&brick,ngbd_c);

          // First, initialize the Navier-Stokes solver with the grid:
          ns = new my_p4est_navier_stokes_t(ngbd,ngbd_np1,faces_np1);

          // Set the LSF:
          ns->set_phi(phi.vec);

          // Set the parameters for the NS solver:
          ns->set_parameters(mu_l,rho_l,advection_sl_order,NULL,NULL,cfl);

          // Set the nth velocity:
          ns->set_velocities(v_nm1.vec,v_n.vec);

          // Set the timestep: // change to include both timesteps (dtnm1,dtn)
          if(advection_sl_order ==2){
              ns->set_dt(dt_nm1,dt);
            }
          else{
              ns->set_dt(dt);
            }

          // Call the appropriate functions to setup the interfacial boundary conditions :
          interface_bc_velocity_u(); interface_bc_velocity_v();

          // Now setup the bc interface objects -- must be initialized with the neighbors and computed interfacial velocity of the moving solid front
          BC_interface_value_velocity_u bc_interface_value_u(ngbd_np1,v_interface);
          BC_interface_value_velocity_v bc_interface_value_v(ngbd_np1,v_interface);

          // Special Pressure BC objects -- used in the NS LLNL validation case (example 3):
          vec_and_ptr_dim_t interface_normal(p4est_np1,nodes_np1);

          // Initialize the BC objects:
          BoundaryConditions2D bc_velocity[P4EST_DIM];
          BoundaryConditions2D bc_pressure;

          // Set the interfacial boundary conditions for velocity:
          bc_velocity[0].setInterfaceType(interface_bc_type_velocity_u);
          bc_velocity[1].setInterfaceType(interface_bc_type_velocity_v);

          bc_velocity[0].setInterfaceValue(bc_interface_value_u);
          bc_velocity[1].setInterfaceValue(bc_interface_value_v);

          // Set the wall boundary conditions for velocity:
          bc_velocity[0].setWallTypes(wall_bc_type_velocity_u); bc_velocity[1].setWallTypes(wall_bc_type_velocity_v);
          bc_velocity[0].setWallValues(wall_bc_value_velocity_u); bc_velocity[1].setWallValues(wall_bc_value_velocity_v);

          // Set the interfacial boundary conditions for pressure:
          interface_bc_pressure();
          bc_pressure.setInterfaceType(interface_bc_type_pressure);
          bc_pressure.setInterfaceValue(interface_bc_value_pressure);
          bc_pressure.setWallTypes(wall_bc_type_pressure);
          bc_pressure.setWallValues(wall_bc_value_pressure);

          // Set the boundary conditions:
          ns->set_bc(bc_velocity,&bc_pressure);

          // set_external_forces
          CF_DIM *external_forces[P4EST_DIM] = {&fx_ext_tn,&fy_ext_tn};
          if(example_ == NS_GIBOU_EXAMPLE){
              ns->set_external_forces(external_forces);
            }

          // Create the cell and face solvers:
          cell_solver = NULL;
          face_solver = NULL;

          // -----------------------------
          // Get hodge and begin iterating on hodge error
          // -----------------------------
          hodge_old.create(p4est_np1,ghost_np1);
          hodge_new.create(p4est_np1,ghost_np1);

          bool keep_iterating_hodge = true;
          double hodge_tolerance;
          if (tstep<1) hodge_tolerance = u0*hodge_percentage_of_max_u;
          else hodge_tolerance = NS_norm*hodge_percentage_of_max_u;
          PetscPrintf(mpi.comm(),"Hodge tolerance is %0.2e \n",hodge_tolerance);

          int hodge_max_it = 100;

          int hodge_iteration = 0;

          while(keep_iterating_hodge){
              double hodge_error = - 10.0;
              double hodge_global_error = -10.0;

              // Grab the old hodge variable before we go through the solution process:  Note: Have to copy it , because the hodge vector itself will be changed by the navier stokes solver
              hodge_new.set(ns->get_hodge());
              VecCopy(hodge_new.vec,hodge_old.vec);

              // ------------------------------------
              // Do NS Solution process:
              // ------------------------------------
              // Viscosity step:
              ns->solve_viscosity(face_solver,(face_solver!=NULL),face_solver_type,pc_face);

              // Projection step:
              ns->solve_projection(cell_solver,(cell_solver!=NULL),cell_solver_type,pc_cell);

              // -------------------------------------------------------------
              // Check the error on hodge:
              // -------------------------------------------------------------
              // Get the current hodge:
              hodge_new.set(ns->get_hodge());

              // Create interpolation object to interpolate phi to the quadrant location (since we are checking hodge error only in negative subdomain, we need to check value of LSF):
              my_p4est_interpolation_nodes_t *interp_phi = ns->get_interp_phi();

              // Get hodge arrays:
              hodge_old.get_array();
              hodge_new.get_array();

              // Loop over each quadrant in each tree, check the error in hodge
              foreach_tree(tr,p4est_np1){
                p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_np1->trees,tr);

                foreach_local_quad(q,tree){
                  // Get the global index of the quadrant:
                  p4est_locidx_t quad_idx = tree->quadrants_offset + q;

                  // Get xyz location of the quad center so we can interpolate phi there and check which domain we are in:
                  double xyz[P4EST_DIM];
                  quad_xyz_fr_q(quad_idx,tr,p4est_np1,ghost_np1,xyz);

                  double phi_val = (*interp_phi)(xyz[0],xyz[1]);
                  // Evaluate the hodge error:
                  if(phi_val < 0){
                      hodge_error = max(hodge_error,fabs(hodge_old.ptr[quad_idx] - hodge_new.ptr[quad_idx]));
                    }
                }
              }

              // Restore hodge arrays:
              hodge_old.restore_array();
              hodge_new.restore_array();

              // Get the global hodge error:
              int mpi_err = MPI_Allreduce(&hodge_error,&hodge_global_error,1,MPI_DOUBLE,MPI_MAX,mpi.comm()); SC_CHECK_MPI(mpi_err);
              PetscPrintf(mpi.comm(),"Hodge iteration : %d, hodge error: %0.3e \n",hodge_iteration,hodge_global_error);

              if((hodge_global_error < hodge_tolerance) || hodge_iteration>=hodge_max_it) keep_iterating_hodge = false;
              hodge_iteration++;
            }

          // Destroy hodge vectors:
          hodge_old.destroy();
          hodge_new.destroy();

          // Compute velocity at the nodes
          ns->compute_velocity_at_nodes();

          // Compute the pressure
          ns->compute_pressure();

          // Get pressure at cells:
          press.create(p4est_np1,ghost_np1);
          press.set(ns->get_pressure());

          // Get the pressure at the nodes (via interpolation):
          press_nodes.destroy();
          press_nodes.create(p4est_np1,nodes_np1);

          my_p4est_interpolation_cells_t interp_c(ngbd_c,ngbd_np1);
          foreach_node(n,nodes_np1){
            double xyz_press[P4EST_DIM];

            node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz_press);
            interp_c.add_point(n,xyz_press);
          }
          interp_c.set_input(press.vec, phi.vec, &bc_pressure);
          interp_c.interpolate(press_nodes.vec);

          // Destroy pressure at cells now:
          press.destroy();

          // Check the L2 norm of u to make sure nothing is blowing up
          NS_norm = ns->get_max_L2_norm_u();
          PetscPrintf(mpi.comm(),"\n max NS velocity norm is %0.3e \n",NS_norm);
          if(ns->get_max_L2_norm_u()>100.0){
              std::cerr<<"The simulation blew up \n"<<std::endl;
            }

          // Get the computed values of vorticity, and velocity:
          vorticity.destroy();
          vorticity.create(p4est_np1,nodes_np1);
          vorticity.set(ns->get_vorticity());

          // Before overwriting values of v_n, slide values in time for next timestep:
          v_nm1.destroy();
          v_nm1.create(p4est_np1,nodes_np1);

          foreach_dimension(d){
            VecCopyGhost(v_n.vec[d],v_nm1.vec[d]);
          }

          // Now set this step's "v_np1" to be "v_n" for the next timestep -- v_n for next step will be sampled at this grid for now, but will be interpolated onto new grid for next step in beginning of next step
          v_n.set(ns->get_velocity_np1()); // v_n is already constructed on p4est_np1, so no need to destroy old vector

          //Get a more appropriate dt for next timestep to consider:
          ns->compute_adapted_dt(u0);
//          ns->compute_dt(u0);
          dt_NS = ns->get_dt();

          // Delete solver now that it isn't being used
//          delete ns;

          PetscPrintf(mpi.comm(),"Navier-Stokes step complete \n \n");

//          // Check errors if this is the benchmark case:
//          if(example_ == 3){
//              check_NS_LLNL_benchmark_error(phi,v_n,press,p4est_np1,nodes_np1,dxyz_close_to_interface,name_NS_LLNL,fich_NS_LLNL,tstep);
//            }
        } // End of "if solve navier stokes"


      PetscLogDouble mem11;
      PetscMemoryGetCurrentUsage(&mem11);


      // --------------------------------------------------------------------------------------------------------------
      // Delete the old grid:
      // --------------------------------------------------------------------------------------------------------------
      // Delete the old grid and update with the new one:

      p4est_destroy(p4est); p4est = p4est_np1;
      p4est_ghost_destroy(ghost); ghost = ghost_np1;
      p4est_nodes_destroy(nodes); nodes = nodes_np1;

      delete hierarchy; hierarchy = hierarchy_np1;
      delete ngbd; ngbd = ngbd_np1;


      PetscLogDouble mem12;
      PetscMemoryGetCurrentUsage(&mem12);


      // Get current memory usage and print out all memory usage checkpoints:
      PetscLogDouble mem13;
      PetscMemoryGetCurrentUsage(&mem13);
      PetscPrintf(mpi.comm(),"Memory used %g \n\n",mem13);

      ierr = PetscFOpen(mpi.comm(),name_mem,"a",&fich_mem); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(),fich_mem,"%g %g %d %g %g %g %g %g %g %g %g %g %g %g %g %g \n",tn,dt,tstep,mem1,mem2,mem3,mem4,mem5,mem6,mem7,mem8,mem9,mem10,mem11,mem12,mem13);CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);
    } // <-- End of for loop through time

  phi.destroy();
  if(solve_stefan){
      T_l_n.destroy();
      T_s_n.destroy();
      smoke.destroy();
    }

  if(solve_navier_stokes){ v_n.destroy();v_nm1.destroy();vorticity.destroy();press.destroy();}
  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

