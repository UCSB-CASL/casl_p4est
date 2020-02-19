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
DEFINE_PARAMETER(pl,int,example_,3,"example number: \n"
                                   "0 - Frank Sphere (Stefan only) \n"
                                   "1 - NS Gibou example (Navier Stokes only) \n"
                                   "2 - work in progress \n"
                                   "3 - Coupled problem example for validation \n"
                                   "4 - Ice solidifying around a cooled cylinder \n"
                                   "5 - Flow past a cylinder (WIP) (Navier Stokes only)\n"
                                   "default: 4");

// ---------------------------------------
// Save options:
// ---------------------------------------
DEFINE_PARAMETER(pl,bool,save_stefan,false,"Save stefan ?");
DEFINE_PARAMETER(pl,bool,save_navier_stokes,false,"Save navier stokes?");
DEFINE_PARAMETER(pl,bool,save_coupled_fields,true,"Save the coupled problem?");
DEFINE_PARAMETER(pl,int,save_every_iter,50,"Saves vtk every n number of iterations (default is 1)");
DEFINE_PARAMETER(pl,bool,print_checkpoints,false,"Print checkpoints throughout script for debugging? ");
DEFINE_PARAMETER(pl,bool,mem_checkpoints,false,"checks various memory checkpoints for mem usage");
DEFINE_PARAMETER(pl,double,mem_safety_limit,60.e9,"Memory upper limit before closing the program -- in bytes");

// ---------------------------------------
// Solution options:
// ---------------------------------------
DEFINE_PARAMETER(pl,bool,solve_stefan,false,"Solve stefan ?");
DEFINE_PARAMETER(pl,bool,solve_navier_stokes,false,"Solve navier stokes?");
DEFINE_PARAMETER(pl,bool,solve_coupled,true,"Solve the coupled problem?");
DEFINE_PARAMETER(pl,bool,do_advection,1,"Boolean flag whether or not to do advection (default : 1)");
DEFINE_PARAMETER(pl,double,Re_overwrite,0.0,"overwrite the examples set Reynolds number");
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

DEFINE_PARAMETER(pl,int,advection_sl_order,2,"Integeer for advection solution order (can choose 1 or 2) (default:1) -- note: this also sets the NS solution order");
//DEFINE_PARAMETER(pl,bool,solve_smoke,0,"Boolean for whether to solve for smoke or not (a passive scalar), default: 0");
DEFINE_PARAMETER(pl,double,cfl,0.5,"CFL number (default:0.5)");
DEFINE_PARAMETER(pl,bool,force_interfacial_velocity_to_zero,false,"Force the interfacial velocity to zero? ");

bool stop_flag = false;
bool pressure_check_flag = false;
// ---------------------------------------
// Geometry options:
// ---------------------------------------
double xmin; double xmax;
double ymin; double ymax;

int nx, ny;       // number trees in each direction
int px, py;       // periodicity in each direction
double scaling;   // amount you want to scale the physical problem by -- scaling*physical_length_scale = computational_length_scale

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

void set_geometry(){
  switch(example_){
    case FRANK_SPHERE: // Frank sphere
      xmin = -5.0; xmax = 5.0; //5.0;
      ymin = -5.0; ymax = 5.0;
      nx = 1;
      ny = 1;
      px = 0; py = 1;

      scaling = 1.;


      s0 = 0.6286;
      r0 = s0; // for consistency, and for setting up NS problem (if wanted)
      T_inf = -0.2;
      Twall = T_inf;
      Tinterface = 0.0;
      break;

    case ICE_AROUND_CYLINDER:{ // Ice layer growing around a constant temperature cooled cylinder

      // Physical domain -- .05 meters high (5 cm) by .1 meters long (10 cm)
      xmin = 0.0; xmax = 3.5;//3.5;//2.0;//1.5;
      ymin = 0.0; ymax = 1.75;

      nx = 4;//4;
      ny = 2;

      px = 0;
      py = 1;

      scaling = 10.;

      double r_cyl_physical = 0.035/2;//0.016/2.;//0.01; // 1 cm

      double r_physical = r_cyl_physical + 0.035/5.;//0.0105;//.011; // 1.1 cm


      r0 = r_physical;//r_physical*scaling;
      r_cyl = r_cyl_physical;//r_cyl_physical*scaling;

      Twall = 276.; Tinterface = 273.0;//293.0,
      T_cyl = 260.;//273.0 - 20.0;
      back_wall_temp_flux = 0.0;

      //sigma is set with the conductivities
      break;}

    case NS_GIBOU_EXAMPLE: // Navier Stokes Validation case from Gibou 2015
      xmin = 0.0; xmax = PI;
      ymin = 0.0; ymax = PI;

      nx = 1; ny = 1;
      px = 0; py = 0;

      scaling = 1.0;
      r0 = 0.10;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      xmin = 0.0; xmax = PI;
      ymin = 0.0; ymax = PI;

      nx = 1; ny = 1;
      px = 0; py = 0;

      scaling = 1.0;
      r0 = 0.2;
      force_interfacial_velocity_to_zero = true;

      break;

    case FLOW_PAST_CYLINDER:
      xmin = 0.0; xmax = 5.0;
      ymin = 0.0; ymax = 5.0;

      nx = 2; ny = 2;
      px = 0; py = 1;

      scaling = 10.;
      r0 = 0.5;
      break;
    }
}

double v_interface_max_norm; // For keeping track of the interfacial velocity maximum norm

// ---------------------------------------
// Grid refinement:
// ---------------------------------------
DEFINE_PARAMETER(pl,int,lmin,3,"Minimum level of refinement");
DEFINE_PARAMETER(pl,int,lmax,6,"Maximum level of refinement");
DEFINE_PARAMETER(pl,double,lip,1.75,"Lipschitz coefficient");
DEFINE_PARAMETER(pl,int,method_,1,"Solver in time for solid domain, and for fluid if no advection. 1 - Backward Euler, 2 - Crank Nicholson");
DEFINE_PARAMETER(pl,int,num_splits,0,"Number of splits -- used for convergence tests");
// ---------------------------------------
// Time-stepping:
// ---------------------------------------
double tfinal;
double dt_max_allowed;
bool keep_going = true;

double tn;
double dt;
double dt_nm1;
DEFINE_PARAMETER(pl,double,t_ramp,0.5,"Time at which NS boundary conditions are ramped up to the freestream value \n");
DEFINE_PARAMETER(pl,bool,ramp_bcs,true,"Boolean option to ramp the BC's for the ice over cylinder case \n");
DEFINE_PARAMETER(pl,double,t_cyl_ramp,0.01,"Amount of time used to ramp the cylinder boundary condition to the cooled substrate condition \n");

void simulation_time_info(){
  switch(example_){
    case FRANK_SPHERE:
      tfinal = 1.30;
      dt_max_allowed = 0.1;
      tn = 1.0;
      break;
    case ICE_AROUND_CYLINDER: // ice solidifying around isothermally cooled cylinder
      tfinal = 40.*60.; // 40 minutes
      dt_max_allowed = 1.0;
      tn = 0.0;
      dt = 1.e-5;
      break;

    case NS_GIBOU_EXAMPLE:
      tfinal = PI/3.;
      dt_max_allowed = 1.e-2;
      tn = 0.0;
      dt = 1.e-3;
      break;

    case FLOW_PAST_CYLINDER:
      tfinal = 50.0;
      dt_max_allowed = 1.0;
      tn = 0.0;
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      tfinal = PI/3.;
      dt_max_allowed = 1.0e-1;
      tn = 0.0;
      dt = 1.e-3;
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
    case FRANK_SPHERE:
      alpha_s = 1.0;
      alpha_l = 1.0;
      break;

    case ICE_AROUND_CYLINDER:
      alpha_s = (1.1e-6); //ice - [m^2]/s
      alpha_l = (1.5e-7); //water- [m^2]/s
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
double rho_s;

void set_conductivities(){
  switch(example_){
    case FRANK_SPHERE:
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      rho_s = 1.0;
      break;

    case ICE_AROUND_CYLINDER:
      k_s = 2.22; // W/[m*K]
      k_l = 0.608; // W/[m*K]
      L = 334.e3;  // J/kg
      rho_l = 1000.0; // kg/m^3
      sigma = rho_l*L*(2.10e-10); //W/m
      rho_s = 920.; //[kg/m^3]
      break;

    case COUPLED_PROBLEM_EXAMPLE:
      k_s = 1.;
      k_l = 1.;
      L = 1.;
      rho_l = 1.;
      rho_s = 1.0;
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
int hodge_max_it = 30;
double uniform_band;
double dt_NS;

double hodge_global_error;

double NS_norm = 0.0; // To keep track of the NS norm
void set_NS_info(){
  pressure_prescribed_flux = 0.0; // For the Neumann condition on the two x walls and lower y wall
  pressure_prescribed_value = 0.0; // For the Dirichlet condition on the back y wall


//  uniform_band = 4.0;
  dt_NS = 1.e-3; // initial dt for NS
  switch(example_){
    case FRANK_SPHERE:throw std::invalid_argument("NS isnt setup for this example");
    case ICE_AROUND_CYLINDER:
      Re_u = 201.;//201.;//217.;
      Re_v = 0.;
      mu_l = 1.7106e-3;//1.793e-3;  // Viscosity of water , [Pa s]
      uniform_band = 4.;
      if(Re_overwrite>1.0){Re_u = Re_overwrite;}
      break;
    case NS_GIBOU_EXAMPLE:
      Re_u = 1.0;
      Re_v = 1.0;
      mu_l = 1.0;
      rho_l = 1.0;

      u0 = 1.0;
      v0 = 1.0;
      uniform_band = 2.;
      break;

    case FLOW_PAST_CYLINDER:
      Re_u = 300.0;
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
      uniform_band = 4.;
      break;
    }

  outflow_u = 0.0;
  outflow_v = 0.0;

  hodge_percentage_of_max_u = 1.e-3;

  // WAY OF SETTING VELOCITIES FOR NS SOLVER NEEDS TO BE FIXED -- THIS IS JUST A TEMPORARY WAY
  if(example_ == ICE_AROUND_CYLINDER){
      u0 = Re_u*mu_l/(rho_l*2.0*r_cyl);
      v0 = Re_v*mu_l/(rho_l*2.0*r_cyl);
    }

}

// ---------------------------------------
// Other parameters:
// ---------------------------------------
double v_int_max_allowed = 1.0;

bool move_interface_with_v_external = false;

bool check_temperature_values = false; // Whether or not you want to print out temperature value averages during various steps of the solution process -- for debugging

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

//------------------------------------------------------------------------
// For coupled problem validation:
// -----------------------------------------------------------------------
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
  double part1 = 2*SQR(n)*cos(pow(t,p)) - p*pow(t,p-1.)*sin(pow(t,p));
  double part2 = cos(n*x)*cos(n*y)*sin(x)*cos(y) + sin(n*x)*sin(n*y)*cos(x)*sin(y);
  return sin(n*x)*cos(n*y)*part1 + n*cos(t)*cos(pow(t,p))*part2;
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

double dT_dt(double x, double y, double t){
  double n=2.0; double p=2.0;
  return -1.0*p*pow(t,p-1.)*cos(n*y)*sin(pow(t,p))*sin(n*x);
}

double u_dT_dx(double x, double y, double t){
  double n=2.0; double p=2.0;
  return n*cos(pow(t,p))*cos(n*x)*cos(n*y)*cos(t)*cos(y)*sin(x);

}

double v_dT_dy(double x, double y, double t){
  double n=2.0; double p=2.0;
  return n*cos(pow(t,p))*sin(n*x)*sin(n*y)*cos(t)*cos(x)*sin(y);
}

class check_advection_term_n: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    double alpha_coeff_value;
    alpha_coeff_value = advection_sl_order==2? advection_alpha_coeff:1.0;
    return dT_dt(x,y,tn) + u_dT_dx(x,y,tn) + v_dT_dy(x,y,tn) - alpha_coeff_value*T_ana_tnp1(x,y)/dt;
  }
}check_advection_term_n;

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
      case ICE_AROUND_CYLINDER:
        return r0 - sqrt(SQR(x - (xmax/3.0)) + SQR(y - (ymax/2.0)));
      case NS_GIBOU_EXAMPLE:
        return 0.2 - sin(x)*sin(y);
      case FLOW_PAST_CYLINDER:
        return r0 - sqrt(SQR(x - (xmax/4.0)) + SQR(y - (ymax/2.0)));
      case COUPLED_PROBLEM_EXAMPLE:
        return 0.2 - sin(x)*sin(y);
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
      case ICE_AROUND_CYLINDER: return r_cyl - sqrt(SQR(x - (xmax/3.0)) + SQR(y - (ymax/2.0)));
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

class BC_interface_value_temp: public CF_DIM{
private:
  // Have interpolation objects for case with surface tension included in boundary condition: can interpolate the curvature in a timestep to the interface points while applying the boundary condition
  my_p4est_interpolation_nodes_t kappa_interp;

public:
  BC_interface_value_temp(my_p4est_node_neighbors_t *ngbd): kappa_interp(ngbd)
  {
//    // Set the curvature and normal inputs to be interpolated when the BC object is constructed:
//    kappa_interp.set_input(kappa.vec,linear);
  }
  void create(my_p4est_node_neighbors_t *ngbd,Vec kappa){
    kappa_interp.update_neighbors(ngbd);
    kappa_interp.set_input(kappa,linear);
  }
  void clear(){
    kappa_interp.clear();
  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE:{ // Frank sphere case, no surface tension
         double r = sqrt(SQR(x) + SQR(y));
         double sval = r/sqrt(tn+dt);
         return frank_sphere_solution_t(sval);
         //return Tinterface;
        }
      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder, with surface tension -- MAY ADD COMPLEXITY TO THIS LATER ON
//        printf("Interfacial temp BC is : %0.2f \n", Tinterface);
//        printf("Boundary condition is : %0.2f \n", Tinterface*(1. - (sigma/L)*kappa_interp(x,y)));

        return Tinterface*(1. - (sigma/L)*kappa_interp(x,y));
      case COUPLED_PROBLEM_EXAMPLE:
        return T_ana_tnp1(x,y);
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
    return ((fabs(x - xmin) <= EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS)); // front x wall, excluding the top and bottom corner points in y
  }
} xlower_wall;

struct XUPPER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return ((fabs(x - xmax) <= EPS) && (fabs(y - ymin)>EPS) && (fabs(y - ymax)>EPS)); // back x wall, excluding the top and bottom corner points in y
  }
} xupper_wall;

struct YLOWER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (fabs(y - ymin) <= EPS);
  }
} ylower_wall;

struct YUPPER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (fabs(y - ymax) <= EPS);
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

double temp_three_wall_dirichlet_val(DIM(double x, double y, double z)){
  if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
      return Twall;}
  else {
      return back_wall_temp_flux;
    }
}

double temp_three_wall_neumann_val(DIM(double x, double y, double z)){
  if (xupper_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
      return back_wall_temp_flux;}
  else {
      return Twall;
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

class WALL_BC_TYPE_TEMP: public WallBCDIM
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
          return temp_three_wall_dirichlet_val(x,y);
        }
      case COUPLED_PROBLEM_EXAMPLE:{
          if(xlower_wall(x,y) || xupper_wall(x,y) || ylower_wall(x,y) || yupper_wall(x,y)){
              return T_ana_tnp1(x,y);
            }
          break;
        }
      }
  }
} wall_bc_value_temp;

// --------------------------------------------------------------------------------------------------------------
// TEMPERATURE INITIAL CONDITION
// --------------------------------------------------------------------------------------------------------------
//class INITIAL_CONDITION_TEMP: public CF_DIM
//{
//public:
//  double operator() (DIM(double x, double y, double z)) const
//  {
//    double m;
//    double r;
//    double sval;
//    double Tsloped;
//    if (level_set(DIM(x,y,z)) > EPS){ // In the solid subdomain
//        switch(example_){
//          case FRANK_SPHERE:{
//            r = sqrt(SQR(x) + SQR(y));
//            sval = s(r,tn);
//            return frank_sphere_solution_t(sval); // Initial distribution is the analytical solution of Frank Sphere problem at t = 0
//          }
//          case ICE_AROUND_CYLINDER:{
//            return Tinterface;
//            }
//          case COUPLED_PROBLEM_EXAMPLE:{
//              return T_ana_tn(x,y);
//            }
//          }
//      }
//    else{// In the fluid subdomain:
//        switch(example_){
//          case FRANK_SPHERE: {// Analytical solution to frank sphere as initial condition
//            r = sqrt(SQR(x) + SQR(y));
//            sval = s(r,tn);
//            return frank_sphere_solution_t(sval);}
//          case ICE_AROUND_CYLINDER:{
//              return Twall;
////            m = (Twall - Tinterface)/(level_set(DIM(xmin,ymin,z)));
////            Tsloped = Tinterface + m*level_set(DIM(x,y,z));
////            if(Tsloped<Twall) return Tsloped;
////            else return Twall;
//            }
//          case COUPLED_PROBLEM_EXAMPLE:
//            {
//              return T_ana_tn(x,y);
//            }
//          }
//      }
//  }
//}IC_temp;

class INITIAL_CONDITION_TEMP_LIQ: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    double r;
    double sval;
    switch(example_){
      case FRANK_SPHERE:{
        r = sqrt(SQR(x) + SQR(y));
        sval = s(r,tn);
        return frank_sphere_solution_t(sval); // Initial distribution is the analytical solution of Frank Sphere problem at t = 0
      }
      case ICE_AROUND_CYLINDER:{
          return Twall;
//          else if(level_set(DIM(x,y,z)>0 && mini_level_set(DIM(x,y,z))<0)) return Tinterface;
//          else return Tinterface; // if(level_set(DIM(x,y,z))<0)

        }
      case COUPLED_PROBLEM_EXAMPLE:{
          return T_ana_tn(x,y);
        }
      }

      }

}IC_temp_liq;

class INITIAL_CONDITION_TEMP_SOL: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    double r;
    double sval;
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

}IC_temp_sol;
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
        //------------------------------------------------------------------
      case FRANK_SPHERE: throw std::invalid_argument("This option may not be used for the particular example being called");
        //------------------------------------------------------------------

      case ICE_AROUND_CYLINDER: {// water solidifying around a cylinder
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET; // Free stream
          }
        else if (xupper_wall(DIM(x,y,z))){
            return NEUMANN; // presribed outflow
          }
        break;
        }
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return DIRICHLET;
          }
        break;}
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:{
          if(ns_sides(x,y) || ns_top_bottom(x,y)){
              return DIRICHLET;
            }
        }
        //------------------------------------------------------------------

      } // end of switch case
  }
} wall_bc_type_velocity_u;

class WALL_BC_VALUE_VELOCITY_U: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      //------------------------------------------------------------------
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
        //------------------------------------------------------------------
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return u0; //Free stream velocity
          }

        else if(xupper_wall(DIM(x,y,z))){ // Homogenous Neumann condition on back wall
            return outflow_u;
          }
        break;
        }
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return u_ana_tnp1(x,y);
          }
        break;}
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:{
//          return 0.0;
          if(ns_sides(x,y) || ns_top_bottom(x,y)){
              return u_ana_tnp1(x,y);
            }
        }
        //------------------------------------------------------------------

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
      //------------------------------------------------------------------
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
      //------------------------------------------------------------------

      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET; // free stream
          }
        else if (xupper_wall(DIM(x,y,z))){
            return NEUMANN; // presribed outflow
          }
        break;
        }
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE:
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return DIRICHLET;
          }
        break;
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:{
          if(ns_sides(x,y) || ns_top_bottom(x,y)){
              return DIRICHLET;
            }
        }
        //------------------------------------------------------------------

      }
  }
} wall_bc_type_velocity_v;

class WALL_BC_VALUE_VELOCITY_V: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      //------------------------------------------------------------------
      case FRANK_SPHERE:
        throw std::invalid_argument("Navier Stokes solution is not compatible with this example, please choose another \n");
      //------------------------------------------------------------------
      case ICE_AROUND_CYLINDER:{
        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return v0; // Free stream
          }
        else if(xupper_wall(DIM(x,y,z))){ // prescribed outflow
            return outflow_v;
          }
        break;}
        //------------------------------------------------------------------


      case NS_GIBOU_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
          return v_ana_tnp1(x,y);
          }
        break;
        }
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return v_ana_tnp1(x,y);
          }
        }
        //------------------------------------------------------------------

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
    //------------------------------------------------------------------
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
    //------------------------------------------------------------------

    case ICE_AROUND_CYLINDER:
      interface_bc_type_velocity_u = DIRICHLET;
      break;
    //------------------------------------------------------------------

    case NS_GIBOU_EXAMPLE:
      interface_bc_type_velocity_u = DIRICHLET;
      break;
    //------------------------------------------------------------------

    case COUPLED_PROBLEM_EXAMPLE:{
        interface_bc_type_velocity_u = DIRICHLET;
        break;
      }

    }
}

// Interfacial condition for the u component:
class BC_interface_value_velocity_u: public CF_DIM{
private:
  my_p4est_interpolation_nodes_t v_interface_interp;

public:
//  BC_interface_value_velocity_u(my_p4est_node_neighbors_t *ngbd,vec_and_ptr_dim_t v_interface): v_interface_interp(ngbd){
//    // Set up the interpolation of the interfacial velocity x component:
//    v_interface_interp.set_input(v_interface.vec[0],linear);
//  }
  BC_interface_value_velocity_u(my_p4est_node_neighbors_t *ngbd):v_interface_interp(ngbd){};
  void create(my_p4est_node_neighbors_t *ngbd, Vec v_interface){
    v_interface_interp.update_neighbors(ngbd);
    v_interface_interp.set_input(v_interface,linear);
  }
  void clear(){
    v_interface_interp.clear();

  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
        //------------------------------------------------------------------

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
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE:
         return u_ana_tnp1(x,y);
         //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:
        return u_ana_tnp1(x,y);
        //------------------------------------------------------------------

      }
  }
};


BoundaryConditionType interface_bc_type_velocity_v;
void interface_bc_velocity_v(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      //------------------------------------------------------------------

    case ICE_AROUND_CYLINDER:
      interface_bc_type_velocity_v = DIRICHLET;
      break;

      //------------------------------------------------------------------

    case NS_GIBOU_EXAMPLE:
      interface_bc_type_velocity_v = DIRICHLET;
      break;
      //------------------------------------------------------------------

    case COUPLED_PROBLEM_EXAMPLE:{
        interface_bc_type_velocity_v = DIRICHLET;
        //------------------------------------------------------------------

      }
    }
}

// Interfacial condition for the v component:
class BC_interface_value_velocity_v: public CF_DIM{
private:
  my_p4est_interpolation_nodes_t v_interface_interp;

public:
//  BC_interface_value_velocity_u(my_p4est_node_neighbors_t *ngbd,vec_and_ptr_dim_t v_interface): v_interface_interp(ngbd){
//    // Set up the interpolation of the interfacial velocity x component:
//    v_interface_interp.set_input(v_interface.vec[0],linear);
//  }
  BC_interface_value_velocity_v(my_p4est_node_neighbors_t *ngbd):v_interface_interp(ngbd){};
  void create(my_p4est_node_neighbors_t *ngbd, Vec v_interface){
    v_interface_interp.update_neighbors(ngbd);
    v_interface_interp.set_input(v_interface,linear);
  }
  void clear(){
    v_interface_interp.clear();

  }
  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
        //------------------------------------------------------------------

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
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE:
         return v_ana_tnp1(x,y);
         //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:
        return v_ana_tnp1(x,y);
        //------------------------------------------------------------------

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
        //------------------------------------------------------------------

      case ICE_AROUND_CYLINDER:{
          if(fabs(x - xmax)<EPS){
              return DIRICHLET;
            }
          else{
              return NEUMANN;
            }
//        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
//            return NEUMANN;
//          }
//        else if (xupper_wall(DIM(x,y,z))){
//            return DIRICHLET;
//          }
        }
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE: {
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return NEUMANN;
          }
        break;}
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:{
          if(ns_sides(x,y) || ns_top_bottom(x,y)){
              return NEUMANN;
            }
        }
        //------------------------------------------------------------------

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
        //------------------------------------------------------------------

      case ICE_AROUND_CYLINDER:{ // coupled problem
          return 0.0;
/*        if (xlower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return pressure_prescribed_flux; // Neumann BC in pressure on all walls but back y wall
          }
        else if(xupper_wall(DIM(x,y,z))){ // Dirichlet condition on back wall (usually homogeneous, but could be nonhomogeneous)
            return pressure_prescribed_value;
          }
        break;*/}
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE: {// benchmark NS case
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return 0.0;//p_ana_tnp1(x,y);
          }
        break;}
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:{
        if(ns_sides(x,y) || ns_top_bottom(x,y)){
            return 0.0;
            //return p_ana_tnp1(x,y);
          }
        }
        //------------------------------------------------------------------

      }
  }
} wall_bc_value_pressure;

// vvv Used for NS LLNL validation case
/*
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
*/
// --------------------------------------------------------------------------------------------------------------
// PRESSURE INTERFACIAL CONDITION
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_pressure;
void interface_bc_pressure(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    //------------------------------------------------------------------

    case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
      //------------------------------------------------------------------

    case ICE_AROUND_CYLINDER:
      interface_bc_type_pressure = NEUMANN;
      break;
      //------------------------------------------------------------------

    case NS_GIBOU_EXAMPLE:
      interface_bc_type_pressure = NEUMANN;
      break;
      //------------------------------------------------------------------

    case COUPLED_PROBLEM_EXAMPLE:
      interface_bc_type_pressure = NEUMANN;
      break;
      //------------------------------------------------------------------


    }
}

class BC_interface_value_pressure: public CF_DIM{
public:

  double operator()(double x, double y) const
  {
    switch(example_){
      case FRANK_SPHERE: throw std::invalid_argument("Navier Stokes is not set up properly for this example \n");
        //------------------------------------------------------------------

      case ICE_AROUND_CYLINDER: // Ice solidifying around a cylinder
        return 0.0;
        //------------------------------------------------------------------

      case NS_GIBOU_EXAMPLE: // Benchmark NS
        return 0.0;
        //------------------------------------------------------------------

      case COUPLED_PROBLEM_EXAMPLE:
        return 0.0;
        //------------------------------------------------------------------

      }
  }
}interface_bc_value_pressure;
/*
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
    return 0.0;
  }
};
*/
// --------------------------------------------------------------------------------------------------------------
// Functions for checking the values of interest during the solution process
// --------------------------------------------------------------------------------------------------------------
void check_T_values(vec_and_ptr_t phi, vec_and_ptr_t T, p4est_nodes* nodes, p4est_t* p4est, int example, vec_and_ptr_t phi_cyl,bool check_for_reasonable_values) {
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
      throw std::invalid_argument("You must provide a phi_cylinder vector to run example 2 \n");
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
      if(T.ptr[n]>(Twall + 1.e-2) && check_for_reasonable_values && example_ == ICE_AROUND_CYLINDER){
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n,p4est,nodes,xyz);
          printf("\n Getting unreasonable T value of %0.2f = %0.4e at (%0.4f, %0.4f) \n",T.ptr[n],T.ptr[n],xyz[0],xyz[1]);
          stop_flag = true;
        }
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
  PetscPrintf(p4est->mpicomm,"Average value: %0.3e \n",global_avg_T);
  PetscPrintf(p4est->mpicomm,"Maximum value: %0.3e \n",global_max_T);
  PetscPrintf(p4est->mpicomm,"Minimum value: %0.3e \n",global_min_T);
  PetscPrintf(p4est->mpicomm,"Minimum value magnitude: %0.3e \n \n",global_min_mag_T);

  if(pressure_check_flag){
      PetscPrintf(p4est->mpicomm,"\n");
      PetscPrintf(p4est->mpicomm,"Physical Average value: %0.2e \n",global_avg_T/(SQR(scaling)));
      PetscPrintf(p4est->mpicomm,"Physical Maximum value: %0.2e \n",global_max_T/(SQR(scaling)));
      PetscPrintf(p4est->mpicomm,"Physical Minimum value: %0.2e \n",global_min_T/(SQR(scaling)));
      PetscPrintf(p4est->mpicomm,"Physical Minimum value magnitude: %0.2e \n \n",global_min_mag_T/(SQR(scaling)));
    }

//  if(global_max_T>1.01*(Twall) && check_for_reasonable_values && example_ == ICE_AROUND_CYLINDER){
//      PetscPrintf(p4est->mpicomm,"Aborting due to unreasonable T values \n");
//      stop_flag= true;
// //      MPI_Abort(p4est->mpicomm,1);
//    }

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
  double sval;
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
    sval = r/sqrt(tn+dt);

    double phi_exact = s0*sqrt(tn+dt) - r;
    double T_exact = frank_sphere_solution_t(sval);


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

void check_NS_validation_error(vec_and_ptr_t phi,vec_and_ptr_dim_t v_n, vec_and_ptr_t p, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_node_neighbors_t *ngbd, double dxyz_close_to_interface, char *name, FILE *fich, int tstep){
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

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,3,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n "
                             "Errors on NS Validation "
                             "\n -------------------------------------\n "
                             "Linf on u: %0.3e \n"
                             "Linf on v: %0.3e \n"
                             "Linf on P: %0.3e \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.3e \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:

  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],hodge_global_error,num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
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

  int mpi_err;

  mpi_err = MPI_Allreduce(local_Linf_errors,global_Linf_errors,4,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);SC_CHECK_MPI(mpi_err);

  // Print errors to application output:
  int num_nodes = nodes->indep_nodes.elem_count;
  PetscPrintf(p4est->mpicomm,"\n -------------------------------------\n "
                             "Errors on Coupled Problem Example "
                             "\n -------------------------------------\n "
                             "Linf on u: %0.4e \n"
                             "Linf on v: %0.4e \n"
                             "Linf on P: %0.4e \n"
                             "Linf on Tl: %0.4e \n"
                             "Number grid points used: %d \n"
                             "dxyz close to interface : %0.4f \n",
                              global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],
                              num_nodes,dxyz_close_to_interface);



  // Print errors to file:

  ierr = PetscFOpen(p4est->mpicomm,name,"a",&fich);CHKERRXX(ierr);
  ierr = PetscFPrintf(p4est->mpicomm,fich,"%g %g %d %g %g %g %g %d %g \n",tn,dt,tstep,global_Linf_errors[0],global_Linf_errors[1],global_Linf_errors[2],global_Linf_errors[3],num_nodes,dxyz_close_to_interface);CHKERRXX(ierr);
  ierr = PetscFClose(p4est->mpicomm,fich); CHKERRXX(ierr);



}

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
        double x = xyz[0] - (xmax)/3.0;
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
  ierr = PetscFPrintf(p4est->mpicomm,fich,"\n%0.4e %0.4e %d ",tn, v_interface_max_norm/scaling,global_theta_size);CHKERRXX(ierr);

  for (unsigned long i = 0; i<theta.size();i++){
      ierr = PetscSynchronizedFPrintf(p4est->mpicomm,fich,"%0.4e %0.4e ",theta[i],delta_r[i]/scaling);CHKERRXX(ierr);
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

void setup_rhs(vec_and_ptr_t phi,vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t rhs_Tl, vec_and_ptr_t rhs_Ts,vec_and_ptr_t T_l_backtrace, vec_and_ptr_t T_l_backtrace_nm1, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t *ngbd){

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
      if(advection_sl_order ==2) T_l_backtrace_nm1.get_array();
    }
  else{
      T_l.get_array();
    }

//  // Get smoke arrays:
//  if(solve_smoke){
//      rhs_smoke.get_array();
//      if(do_advection){
//          smoke_backtrace.get_array();
//          if(advection_sl_order ==2) smoke_backtrace_nm1.get_array();
//        }
//      else{
//          smoke.get_array();
//        }
//    }

  // Prep coefficients if we are doing 2nd order advection:
  if(do_advection && advection_sl_order==2){
      advection_alpha_coeff = (2.*dt + dt_nm1)/(dt + dt_nm1);
      advection_beta_coeff = (-1.*dt)/(dt + dt_nm1);
      PetscPrintf(p4est->mpicomm,"Alpha is %.3e, beta is %0.3e, dtnm1 = %0.3e, dt = %0.3e \n",advection_alpha_coeff,advection_beta_coeff, dt_nm1, dt);
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
//            if(fabs(T_l_backtrace_nm1.ptr[n]) < EPS){ // In case where backtrace_nm1 is close to 0, collapse discretization to first order method, since values near zero are nonphysical
//                rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]*alpha/dt;
//              }
//            else{
                rhs_Tl.ptr[n] = T_l_backtrace.ptr[n]*((advection_alpha_coeff/dt) - (advection_beta_coeff/dt_nm1)) + T_l_backtrace_nm1.ptr[n]*(advection_beta_coeff/dt_nm1);

//              }
//            double xyz[P4EST_DIM];
//            node_xyz_fr_n(n,p4est,nodes,xyz);
//            printf("rhs Tl is %0.4e, T_l_backtrace_nm1 is %0.4f , T_l_backtrace is %0.4f, phi is %0.4e, at (%0.4f, %0.4f) \n",rhs_Tl.ptr[n],T_l_backtrace_nm1.ptr[n], T_l_backtrace.ptr[n],phi.ptr[n],xyz[0],xyz[1]);
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

    // Now, if we are applying a forcing term, ie for an example:
    if(example_ == COUPLED_PROBLEM_EXAMPLE){
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n,p4est,nodes,xyz);
        rhs_Tl.ptr[n] += g_ana_tn(xyz[0],xyz[1]);
      }

  }// end of loop over nodes


  // Check the backtrace values and potential errors there:
  vec_and_ptr_t backtrace_check; vec_and_ptr_t backtrace_err;
  backtrace_check.create(p4est,nodes);
  backtrace_err.create(p4est,nodes);

  sample_cf_on_nodes(p4est,nodes,check_advection_term_n,backtrace_check.vec);

  backtrace_err.get_array();
  backtrace_check.get_array();
  foreach_local_node(n,nodes){
    if(phi.ptr[n]<0.){
      backtrace_err.ptr[n] = -1.0*backtrace_check.ptr[n] - rhs_Tl.ptr[n];
//      printf("Backtrace check is %0.3e, rhs_Tl is %0.3e, error is %0.3e \n",-1.0*backtrace_check.ptr[n], rhs_Tl.ptr[n], backtrace_err.ptr[n]);
      }

  }
  backtrace_err.restore_array();
  backtrace_check.restore_array();

  PetscPrintf(p4est->mpicomm,"\n ------------------------- \n");

  PetscPrintf(p4est->mpicomm,"CHECKING BACKTRACE ERRORS : -------------------------");
  check_T_values(phi,backtrace_err,nodes,p4est,example_,phi,false);
  PetscPrintf(p4est->mpicomm,"\n ------------------------- \n");
  PetscPrintf(p4est->mpicomm,"CHECKING RHS Values : -------------------------");
  check_T_values(phi,rhs_Tl,nodes,p4est,example_,phi,false);
  PetscPrintf(p4est->mpicomm,"\n ------------------------- \n");

  backtrace_err.destroy();
  backtrace_check.destroy();




  phi.restore_array();
  // Restore Ts arrays:
  T_s.restore_array();
  rhs_Ts.restore_array();

  // Restore Tl arrays:
  rhs_Tl.restore_array();
  if(do_advection){
      T_l_backtrace.restore_array();
      if(advection_sl_order==2) T_l_backtrace_nm1.restore_array();
    }
  else{
      T_l.restore_array();
    }
  // Restore smoke arrays:
//  if(solve_smoke){
//      rhs_smoke.restore_array();
//      if(do_advection){
//          if(advection_sl_order==2) smoke_backtrace_nm1.restore_array();
//          smoke_backtrace.restore_array();
//        }
//      else{
//          smoke.restore_array();
//        }
//    }

  if(method_ ==2){
      T_s_dd.restore_array();
      T_s_dd.destroy();
      if(!do_advection){
          T_l_dd.restore_array();
          T_l_dd.destroy();
        }
    }

  if(example_ == COUPLED_PROBLEM_EXAMPLE){
      PetscPrintf(p4est->mpicomm,"Updating rhs with forcing term values \n");
      VecGhostUpdateBegin(rhs_Tl.vec,INSERT_VALUES,SCATTER_FORWARD);
      VecGhostUpdateEnd(rhs_Tl.vec,INSERT_VALUES,SCATTER_FORWARD);
    }
}

void do_backtrace(vec_and_ptr_t T_l,vec_and_ptr_t T_l_nm1,vec_and_ptr_t T_l_backtrace,vec_and_ptr_dim_t v, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t* ngbd, p4est_t *p4est_nm1, p4est_nodes_t *nodes_nm1, my_p4est_node_neighbors_t *ngbd_nm1,  vec_and_ptr_t T_l_backtrace_nm1, vec_and_ptr_dim_t v_nm1,interpolation_method interp_method, vec_and_ptr_t phi){
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Beginning to do backtrace \n");

  // Get second derivatives of temp fields for interpolation purposes:
  vec_and_ptr_dim_t T_l_dd, T_l_dd_nm1;
  T_l_dd.create(p4est,nodes);
  ngbd->second_derivatives_central(T_l.vec,T_l_dd.vec);
  if(advection_sl_order==2) {
      T_l_dd_nm1.create(p4est_nm1,nodes_nm1);
      ngbd_nm1->second_derivatives_central(T_l_nm1.vec,T_l_dd_nm1.vec);
    }

  // Get second derivatives of the velocity field:
//  vec_and_ptr_dim_t v_dd[P4EST_DIM];
//  vec_and_ptr_dim_t v_dd_nm1[P4EST_DIM];
  Vec v_dd[P4EST_DIM][P4EST_DIM];
  Vec v_dd_nm1[P4EST_DIM][P4EST_DIM];

  PetscErrorCode ierr;
  foreach_dimension(d){
    foreach_dimension(dd){
      ierr = VecCreateGhostNodes(p4est, nodes, &v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order ==2){
          ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &v_dd_nm1[d][dd]); CHKERRXX(ierr);
        }
    }
  }

  // v_dd[k] is the second derivative of the velocity components n along cartesian direction k
  // v_dd_nm1[k] is the second derivative of the velocity components nm1 along cartesian direction k

  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Starts getting 2nd derivatives for backtrace \n");
  ngbd->second_derivatives_central(v.vec,v_dd[0],v_dd[1],P4EST_DIM);

  if(advection_sl_order ==2){
      ngbd_nm1->second_derivatives_central(v_nm1.vec, DIM(v_dd_nm1[0], v_dd_nm1[1], v_dd_nm1[2]), P4EST_DIM);
      if(print_checkpoints)PetscPrintf(p4est->mpicomm,"Gets the second derivatives of vnm1 \n");
    }

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];
  vector <double> xyz_d_nm1[P4EST_DIM];

  // Do the Semi-Lagrangian backtrace:
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Calling the backtrace trajectory \n");
  if(advection_sl_order ==2){
//      PetscPrintf(p4est->mpicomm,"dtnm1 : %0.3e, dtn: %0.3e \n",dt_nm1,dt);
      trajectory_from_np1_to_nm1(p4est,nodes,ngbd_nm1,ngbd,v_nm1.vec,v_dd_nm1,v.vec,v_dd,dt_nm1,dt,xyz_d_nm1,xyz_d);
    }
  else{
      trajectory_from_np1_to_n(p4est,nodes,ngbd,dt,v.vec,v_dd,xyz_d);
    }

  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Completes the backtrace trajectory \n");

  // Add the back-trace points to the interpolation object:
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd);
  my_p4est_interpolation_nodes_t SL_backtrace_interp_nm1(ngbd_nm1);

  // Add backtrace points to the interpolator(s):
  if(print_checkpoints)PetscPrintf(p4est->mpicomm,"Beginning interpolations for backtrace \n");
  foreach_local_node(n,nodes){
//    double xyz_node[P4EST_DIM];

    double xyz_temp[P4EST_DIM];
    double xyz_temp_nm1[P4EST_DIM];
    foreach_dimension(d){
      xyz_temp[d] = xyz_d[d][n];
      if(advection_sl_order ==2){
          xyz_temp_nm1[d] = xyz_d_nm1[d][n];
        }
    } // end of "for each dimension"

//    node_xyz_fr_n(n,p4est,nodes,xyz_node);
//    printf("Backtrace point for node %d at (%0.4e, %0.4e) on rank %d is (%0.4e, %0.4e)\n",n,xyz_node[0],xyz_node[1],p4est->mpirank,xyz_temp[0],xyz_temp[1]);

    SL_backtrace_interp.add_point(n,xyz_temp);
    if(advection_sl_order ==2 ) SL_backtrace_interp_nm1.add_point(n,xyz_temp_nm1);
  } // end of loop over nodes
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Adds inteperpolation points \n");

  // Interpolate the Temperature data to back-traced points:
  // Note: We interpolate using an array of fields if we are solving for smoke to make interpolation more efficient
  // (rather than calling the interpolation over and over again)
  if(print_checkpoints)PetscPrintf(p4est->mpicomm,"Beginning interpolations \n");

  SL_backtrace_interp.set_input(T_l.vec,T_l_dd.vec[0],T_l_dd.vec[1],interp_method);
  SL_backtrace_interp.interpolate(T_l_backtrace.vec);
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Successfully interpolates T_l backtrace \n");

  if(advection_sl_order ==2){
      SL_backtrace_interp_nm1.set_input(T_l_nm1.vec,T_l_dd_nm1.vec[0],T_l_dd_nm1.vec[1],  interp_method);
      SL_backtrace_interp_nm1.interpolate(T_l_backtrace_nm1.vec);
    }
  if(print_checkpoints) PetscPrintf(p4est->mpicomm,"Successfully interpolates T_l_nm1 backtrace \n");



// // Check values
//  T_l_backtrace.get_array(); v.get_array();
//  T_l_backtrace_nm1.get_array();
//  phi.get_array();
//  foreach_local_node(n,nodes){
//    double xyz[P4EST_DIM];
//    node_xyz_fr_n(n,p4est,nodes,xyz);
//    printf("Point: (%0.3f, %0.3f), T_l_d = %0.3f, T_l_d_nm1 = %0.3f, at pt (%0.3f, %0.3f), phi = %0.3e \n",xyz[0],xyz[1],T_l_backtrace.ptr[n],T_l_backtrace_nm1.ptr[n],xyz_d[0][n],xyz_d[1][n],phi.ptr[n]);
//    if ((T_l_backtrace.ptr[n] < EPS || T_l_backtrace_nm1.ptr[n] < EPS) && phi.ptr[n]<0. )  printf("^^^ HERE\n\n");
//  }
//  T_l_backtrace.restore_array(); v.restore_array();
//  T_l_backtrace_nm1.restore_array();
//  phi.restore_array();
//  if(advection_sl_order==2){
//    T_l_backtrace_nm1.get_array(); v_nm1.get_array();
//    foreach_local_node(n,nodes){
//      printf("T_l_d_nm1 = %0.3f at pt (%0.3f, %0.3f) \n",T_l_backtrace_nm1.ptr[n],xyz_d[0][n],xyz_d_nm1[1][n]);

//    }
//    T_l_backtrace_nm1.restore_array(); v_nm1.restore_array();
//  }




//  // end check values



  // Destroy velocity derivatives now that not needed:
  foreach_dimension(d){
    foreach_dimension(dd)
    {
      ierr = VecDestroy(v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      ierr = VecDestroy(v_dd_nm1[d][dd]); CHKERRXX(ierr);
    }
//    if(advection_sl_order ==2){
//        v_dd_nm1[d].destroy();
//      }
  }

  // Destroy temperature derivatives
  T_l_dd.destroy();
  if(advection_sl_order==2) {
      T_l_dd_nm1.destroy();
    }
}

void interpolate_values_onto_new_grid(vec_and_ptr_t T_l, vec_and_ptr_t T_l_new,
                                      vec_and_ptr_t T_s, vec_and_ptr_t T_s_new,
                                      vec_and_ptr_dim_t v_interface,vec_and_ptr_dim_t v_interface_new,
                                      vec_and_ptr_dim_t v_external,vec_and_ptr_dim_t v_external_new,
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
  Vec all_fields_old[num_fields];
  Vec all_fields_new[num_fields];

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

//          if(solve_smoke) {
//              all_fields_old[6] = smoke.vec;
//              all_fields_new[6] = smoke_new.vec;
//            }
        }
//      else{
//          if(solve_smoke) {
//              all_fields_old[4] = smoke.vec;
//              all_fields_new[4] = smoke_new.vec;
//            }
//        }

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
  interp_nodes.clear();

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

        if(fabs(phi.ptr[n])<dxyz_close_to_interface){
          foreach_dimension(d){
            jump.ptr[d][n] = (k_s*T_s_d.ptr[d][n] -k_l*T_l_d.ptr[d][n])/(L*rho_s);
          }
        }
       }

      // Begin updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }

      // Compute the jump in the local nodes:
      for(size_t i = 0; i<ngbd->get_local_size();i++){
          p4est_locidx_t n = ngbd->get_local_node(i);
          if(fabs(phi.ptr[n])<dxyz_close_to_interface){
            foreach_dimension(d){
              jump.ptr[d][n] = (k_s*T_s_d.ptr[d][n] -k_l*T_l_d.ptr[d][n])/(L*rho_s);
            }
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
      }}


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
  double global_max_vnorm = 0.0;
  int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
  SC_CHECK_MPI(mpi_ret);
  PetscPrintf(p4est->mpicomm,"\n"
                             "Computed interfacial velocity: \n"
                             " - Computational: %0.3e \n"
                             " - Physical: %0.3e [m/s] \n"
                             " - Physical: %0.3e [cm/s] \n",global_max_vnorm,global_max_vnorm/scaling,global_max_vnorm/scaling*100.);

//  // Save the previous timestep:
//  dt_nm1 = dt;
  // Compute new timestep:
  double dt_computed;
  dt_computed = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/global_max_vnorm;//min(global_max_vnorm,1.0);
  dt = min(dt_computed,dt_max_allowed);

  PetscPrintf(p4est->mpicomm,"\n"
                             "Computed Stefan driven timestep: \n"
                             " - dt computed: %0.3e \n"
                             " - dt maximum allowed: %0.3e \n"
                             " - dt used : %0.3e \n"
                             " - dxyz_close_to_interface: %0.3e \n",dt_computed,dt_max_allowed,dt,dxyz_close_to_interface);

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

// regularize_front() stolen from my_p4est_multialloy_t
//TODO: Inherit this from multialloy properly, implement it more nicely -- avoid copy paste of code
/*
void regularize_front(p4est_t* p4est_, p4est_nodes_t* nodes_, p4est_ghost_t* ghost_, my_p4est_node_neighbors_t* ngbd_, my_p4est_hierarchy_t* hierarchy_, my_p4est_brick_t brick_, double diag_, double dxyz_min_, vec_and_ptr_t front_phi_, bool front_smoothing_)
{
  PetscErrorCode ierr;  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
   //remove problem geometries
  PetscPrintf(p4est_->mpicomm, "Removing problem geometries...\n");

  vec_and_ptr_t front_phi_cur;

  front_phi_cur.set(front_phi_.vec);

  p4est_t       *p4est_cur = p4est_;
  p4est_nodes_t *nodes_cur = nodes_;
  p4est_ghost_t *ghost_cur = ghost_;
  my_p4est_node_neighbors_t *ngbd_cur = ngbd_;
  my_p4est_hierarchy_t *hierarchy_cur = hierarchy_;

  if (front_smoothing_ != 0)
  {
    p4est_cur = p4est_copy(p4est_, P4EST_FALSE);
    ghost_cur = my_p4est_ghost_new(p4est_cur, P4EST_CONNECT_FULL);
    nodes_cur = my_p4est_nodes_new(p4est_cur, ghost_cur);

    front_phi_cur.create(front_phi_.vec);
    VecCopyGhost(front_phi_.vec, front_phi_cur.vec);

    splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est_->user_pointer;
    bool is_grid_changing = true;
    while (is_grid_changing)
    {
      front_phi_cur.get_array();
      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl-front_smoothing_, sp_old->lip);
      is_grid_changing = sp.refine_and_coarsen(p4est_cur, nodes_cur, front_phi_cur.ptr);
      front_phi_cur.restore_array();

      if (is_grid_changing)
      {
        my_p4est_partition(p4est_cur, P4EST_TRUE, NULL);

        // reset nodes, ghost, and phi
        p4est_ghost_destroy(ghost_cur); ghost_cur = my_p4est_ghost_new(p4est_cur, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_cur); nodes_cur = my_p4est_nodes_new(p4est_cur, ghost_cur);

        front_phi_cur.destroy();
        front_phi_cur.create(p4est_cur, nodes_cur);

        my_p4est_interpolation_nodes_t interp(ngbd_);

        double xyz[P4EST_DIM];
        foreach_node(n, nodes_cur)
        {
          node_xyz_fr_n(n, p4est_cur, nodes_cur, xyz);
          interp.add_point(n, xyz);
        }

        interp.set_input(front_phi_.vec, linear); // we know that it is not really an interpolation, rather just a transfer, so therefore linear
        interp.interpolate(front_phi_cur.vec);
      }
    }

    hierarchy_cur = new my_p4est_hierarchy_t(p4est_cur, ghost_cur, &brick_);
    ngbd_cur = new my_p4est_node_neighbors_t(hierarchy_cur, nodes_cur);
    ngbd_cur->init_neighbors();
  }

  vec_and_ptr_t front_phi_tmp(front_phi_cur.vec);

  p4est_locidx_t nei_n[num_neighbors_cube];
  bool           nei_e[num_neighbors_cube];

  double band = diag_;

  front_phi_tmp.get_array();
  front_phi_cur.get_array();

  // first pass: smooth out extremely curved regions
  // TODO: make it iterative
  bool is_changed = false;
  foreach_local_node(n, nodes_cur)
  {
    if (fabs(front_phi_cur.ptr[n]) < band)
    {
      ngbd_cur->get_all_neighbors(n, nei_n, nei_e);

      unsigned short num_neg = 0;
      unsigned short num_pos = 0;

      for (unsigned short nn = 0; nn < num_neighbors_cube; ++nn)
      {
        front_phi_cur.ptr[nei_n[nn]] < 0 ? num_neg++ : num_pos++;
      }

      if ( (front_phi_cur.ptr[n] <  0 && num_neg < 3) ||
           (front_phi_cur.ptr[n] >= 0 && num_pos < 3) )
      {
        front_phi_cur.ptr[n] = front_phi_cur.ptr[n] <  0 ? EPS : -EPS;

        // check if node is a layer node (= a ghost node for another process)
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_cur->indep_nodes, n);
        if (ni->pad8 != 0) is_changed = true;
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  if (is_changed)
  {
    ierr = VecGhostUpdateBegin(front_phi_cur.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (front_phi_cur.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  VecCopyGhost(front_phi_cur.vec, front_phi_tmp.vec);

  // second pass: bridge narrow gaps
  // TODO: develop a more general approach that works in 3D as well
  double new_phi_val = .5*dxyz_min_*pow(2., front_smoothing_);
  is_changed = false;
  bool is_ghost_changed = false;
  foreach_local_node(n, nodes_cur)
  {
    if (front_phi_cur.ptr[n] < 0 && front_phi_cur.ptr[n] > -band)
    {
      ngbd_cur->get_all_neighbors(n, nei_n, nei_e);

      bool merge = (front_phi_cur.ptr[nei_n[nn_m00]] > 0 &&
                   front_phi_cur.ptr[nei_n[nn_p00]] > 0 &&
          front_phi_cur.ptr[nei_n[nn_0m0]] > 0 &&
          front_phi_cur.ptr[nei_n[nn_0p0]] > 0)
          || ((front_phi_cur.ptr[nei_n[nn_m00]] > 0 && front_phi_cur.ptr[nei_n[nn_p00]] > 0) &&
          (front_phi_cur.ptr[nei_n[nn_mm0]] < 0 || front_phi_cur.ptr[nei_n[nn_0m0]] < 0 || front_phi_cur.ptr[nei_n[nn_pm0]] < 0) &&
          (front_phi_cur.ptr[nei_n[nn_mp0]] < 0 || front_phi_cur.ptr[nei_n[nn_0p0]] < 0 || front_phi_cur.ptr[nei_n[nn_pp0]] < 0))
          || ((front_phi_cur.ptr[nei_n[nn_0m0]] > 0 && front_phi_cur.ptr[nei_n[nn_0p0]] > 0) &&
          (front_phi_cur.ptr[nei_n[nn_mm0]] < 0 || front_phi_cur.ptr[nei_n[nn_m00]] < 0 || front_phi_cur.ptr[nei_n[nn_mp0]] < 0) &&
          (front_phi_cur.ptr[nei_n[nn_pm0]] < 0 || front_phi_cur.ptr[nei_n[nn_p00]] < 0 || front_phi_cur.ptr[nei_n[nn_pp0]] < 0));

      if (merge)
      {
        front_phi_tmp.ptr[n] = new_phi_val;

        // check if node is a layer node (= a ghost node for another process)
        p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes_cur->indep_nodes, n);
        if (ni->pad8 != 0) is_ghost_changed = true;

        is_changed = true;
      }

    }
  }

  front_phi_tmp.restore_array();
  front_phi_cur.restore_array();

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_LOGICAL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  if (is_ghost_changed)
  {
    ierr = VecGhostUpdateBegin(front_phi_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (front_phi_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  // third pass: look for isolated pools of liquid and remove them
  if (is_changed) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
  {
    int num_islands = 0;
    vec_and_ptr_t island_number(front_phi_cur.vec);

    VecScaleGhost(front_phi_tmp.vec, -1.);
    compute_islands_numbers(*ngbd_cur, front_phi_tmp.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_tmp.vec, -1.);

    if (num_islands > 1)
    {
      island_number.get_array();
      front_phi_tmp.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes_cur)
      {
        if (island_number.ptr[n] >= 0)
        {
          ++island_area[ (int) island_number.ptr[n] ];
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

      // find the biggest liquid pool
      int main_island = 0;
      int island_area_max = island_area[0];

      for (int i = 1; i < num_islands; ++i)
      {
        if (island_area[i] > island_area_max)
        {
          main_island     = i;
          island_area_max = island_area[i];
        }
      }

      if (main_island < 0) throw;

      // solidify all but the biggest pool
      foreach_node(n, nodes_cur)
      {
        if (front_phi_tmp.ptr[n] < 0 && island_number.ptr[n] != main_island)
        {
          front_phi_tmp.ptr[n] = new_phi_val;
        }
      }

      island_number.restore_array();
      front_phi_tmp.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();
  }

  front_phi_cur.destroy();
  front_phi_cur.set(front_phi_tmp.vec);

  // iterpolate back onto fine grid
  if (front_smoothing_ != 0)
  {
    my_p4est_level_set_t ls(ngbd_cur);
    ls.reinitialize_1st_order_time_2nd_order_space(front_phi_cur.vec, 20);

    my_p4est_interpolation_nodes_t interp(ngbd_cur);

    double xyz[P4EST_DIM];
    foreach_node(n, nodes_)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      interp.add_point(n, xyz);
    }

    interp.set_input(front_phi_cur.vec, quadratic_non_oscillatory_continuous_v2); // we know that it is not really an interpolation, rather just a transfer, so therefore linear
    interp.interpolate(front_phi_.vec);

    front_phi_cur.destroy();
    delete ngbd_cur;
    delete hierarchy_cur;
    p4est_nodes_destroy(nodes_cur);
    p4est_ghost_destroy(ghost_cur);
    p4est_destroy(p4est_cur);
  } else {
    front_phi_.set(front_phi_cur.vec);
  }
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
}
*/
// --------------------------------------------------------------------------------------------------------------
// FUNCTIONS FOR SAVING TO VTK:
// --------------------------------------------------------------------------------------------------------------
void save_everything(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,vec_and_ptr_t phi, vec_and_ptr_t phi_2, vec_and_ptr_t Tl,vec_and_ptr_t Ts,vec_and_ptr_dim_t v_int,vec_and_ptr_dim_t v_NS, vec_and_ptr_t press, vec_and_ptr_t vorticity, vec_and_ptr_cells_t press_cells,char* filename){
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

  // First, need to scale the fields appropriately:

  // Scale velocities:
  foreach_dimension(d){
    VecScaleGhost(v_int.vec[d],1./scaling);
    VecScaleGhost(v_NS.vec[d],1./scaling);
  }

  // Scale pressure:
//  VecScaleGhost(press.vec,1./(SQR(scaling)));

  // Get arrays:
  phi.get_array();
  if(example_ == ICE_AROUND_CYLINDER) phi_2.get_array();

  Tl.get_array(); Ts.get_array();

  v_int.get_array(); v_NS.get_array();

  press.get_array(); vorticity.get_array();
  press_cells.get_array();

//  grad_p.get_array();

//  if(solve_smoke) smoke.get_array();


  // Save data:
  std::vector<std::string> point_names;
  std::vector<double*> point_data;

//  if(example_ == ICE_AROUND_CYLINDER && solve_smoke){
//      point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure","smoke"};
//      point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,smoke.ptr};
//    }
  if (example_ == ICE_AROUND_CYLINDER) {
      point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
      point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
//      vec_and_ptr_dim_t dTl, dTs, jump;
//      dTl.create(p4est,nodes); dTs.create(dTl.vec); jump.create(dTl.vec);



//      point_names = {"phi","phi_cyl","T_l","T_s","dTl_dx","dTl_dy","dTs_dx","dTs_dy","jump_x","jump_y","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
//      point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
    }
//  else if (example_ !=ICE_AROUND_CYLINDER && solve_smoke){
//      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure","smoke"};
//      point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr,smoke.ptr};

//    }
  else{
      point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","u","v","vorticity","pressure"};
      point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};
    }

  std::vector<std::string> cell_names = {"pressure_cells"};
  std::vector<double*> cell_data = {press_cells.ptr};

  my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);

  point_names.clear();point_data.clear();
  cell_names.clear(); cell_data.clear();
  // Restore arrays:

  press_cells.restore_array();
  phi.restore_array();
  if(example_ == ICE_AROUND_CYLINDER) phi_2.restore_array();

  Tl.restore_array(); Ts.restore_array();

  v_int.restore_array(); v_NS.restore_array();

  press.restore_array(); vorticity.restore_array();

//  grad_p.restore_array();
  // ------------------

//  if(solve_smoke) smoke.restore_array();
  // Scale things back:
  foreach_dimension(d){
    VecScaleGhost(v_int.vec[d],scaling);
    VecScaleGhost(v_NS.vec[d],scaling);
  }

  // Scale pressure back:
//  VecScaleGhost(press.vec,SQR(scaling));
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

//    if(solve_smoke) smoke.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

//    if(example_ == ICE_AROUND_CYLINDER && solve_smoke){
//        point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y","smoke"};
//        point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],smoke.ptr};
//      }
    if (example_ == ICE_AROUND_CYLINDER) {
        point_names = {"phi","phi_cyl","T_l","T_s","v_interface_x","v_interface_y"};
        point_data = {phi.ptr, phi_2.ptr,Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1]};
      }
//    else if (example_ !=ICE_AROUND_CYLINDER && solve_smoke){
//        point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y","smoke"};
//        point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1],smoke.ptr};
//      }
    else{
        point_names = {"phi","T_l","T_s","v_interface_x","v_interface_y"};
        point_data = {phi.ptr, Tl.ptr, Ts.ptr,v_int.ptr[0],v_int.ptr[1]};
      }

    std::vector<std::string> cell_names;
    std::vector<double*> cell_data;

    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);


    // Clear the vectors:
    cell_names.clear(); cell_data.clear();
    point_names.clear();point_data.clear();
    // Restore arrays:

    phi.restore_array();
    if(example_ == ICE_AROUND_CYLINDER) phi_2.restore_array();

    Tl.restore_array(); Ts.restore_array();

    v_int.restore_array();

//    if(solve_smoke) smoke.restore_array();
    // Scale things back:
    foreach_dimension(d){
      VecScaleGhost(v_int.vec[d],scaling);
    }
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
    foreach_dimension(d){
      VecScaleGhost(v_NS.vec[d],1./scaling);
    }

    // Scale pressure:
    VecScaleGhost(press.vec,scaling);

    // Get arrays:
    phi.get_array();
    v_NS.get_array();

    press.get_array(); vorticity.get_array();

//    if(solve_smoke) smoke.get_array();

    // Save data:
    std::vector<std::string> point_names;
    std::vector<double*> point_data;

//    if(solve_smoke){
//        point_names = {"phi","u","v","vorticity","smoke"};
//        point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,smoke.ptr};
//      }

    point_names = {"phi","u","v","vorticity","pressure"};
    point_data = {phi.ptr,v_NS.ptr[0],v_NS.ptr[1],vorticity.ptr,press.ptr};


    std::vector<std::string> cell_names = {};
    std::vector<double*> cell_data = {};

    my_p4est_vtk_write_all_vector_form(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,filename,point_data,point_names,cell_data,cell_names);

    point_names.clear(); point_data.clear();
    cell_names.clear(); cell_data.clear();


    // Restore arrays:

    phi.restore_array();
    v_NS.restore_array();

    press.restore_array(); vorticity.restore_array();


//    if(solve_smoke) smoke.restore_array();
    // Scale things back:
    foreach_dimension(d){
      VecScaleGhost(v_NS.vec[d],scaling);
    }

    // Scale pressure back:
    VecScaleGhost(press.vec,1./scaling);
} // end of save_navier_stokes_fields

// --------------------------------------------------------------------------------------------------------------
// FUNCTIONS FOr SAVING OR LOADING SIMULATION STATE:
// --------------------------------------------------------------------------------------------------------------

void fill_or_load_double_parameters(save_or_load flag, PetscReal *data){
  size_t idx=0;
  switch(flag){
    case SAVE:{
        data[idx++] = tn;
        data[idx++] = dt;
        if(advection_sl_order==2)data[idx++] = dt_nm1;
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
        data[idx++] = scaling;
        // Note: all physical parameters are saved in their *scaled* forms. To recover them, need to do the reverse scaling algebra.
        break;
      }
    case LOAD:{
        tn = data[idx++];
        dt = data[idx++];
        // Note: since these parameters depend on advection sl order, need to load integers first before doubles
        if(advection_sl_order==2) dt_nm1 = data[idx++];
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
        scaling= data[idx++];
      }

    }
  if(advection_sl_order==2) P4EST_ASSERT(idx == 14);
  else P4EST_ASSERT(idx == 13);
};

void fill_or_load_integer_parameters(save_or_load flag, PetscInt *data){
  size_t idx=0;
  switch(flag){
    case SAVE:{
        data[idx++] = lmin;
        data[idx++] = lmax;
        data[idx++] = advection_sl_order;
        data[idx++] = save_every_iter;
        break;
      }
    case LOAD:{
        lmin = data[idx++];
        lmax = data[idx++];
        advection_sl_order = data[idx++];
        save_every_iter = data[idx++];
      }

    }
  P4EST_ASSERT(idx == 4);
};
void save_or_load_parameters(const char* filename, splitting_criteria_t* splitting_crit, save_or_load flag, double &tn, const mpi_environment_t* mpi){
  PetscErrorCode ierr;

  // Double parameters we need to save:
  // - tn, dt, dt_nm1 (if 2nd order), k_l, k_s, alpha_l, alpha_s, rho_l, rho_s, mu_l, L, cfl, uniform_band, scaling
  PetscReal double_parameters[14];



  // Integer parameters we need to save:
  // - current lmin, current lmax, advection_sl_order, save_every_iter,
  PetscInt integer_parameters[4];

  int fd;
  char diskfilename[PATH_MAX];

  switch(flag){
    case SAVE:{
        if(mpi->rank() ==0){

            // Save the integer parameters to a file
            sprintf(diskfilename,"%s_integers",filename);
            fill_or_load_integer_parameters(flag,integer_parameters);
            ierr = PetscBinaryOpen(diskfilename,FILE_MODE_WRITE,&fd); CHKERRXX(ierr);
            ierr = PetscBinaryWrite(fd, integer_parameters, 4, PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);

            // Save the double parameters to a file:
            sprintf(diskfilename, "%s_doubles", filename);
            fill_or_load_double_parameters(flag, double_parameters);
            ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
            ierr = PetscBinaryWrite(fd, double_parameters, 14, PETSC_DOUBLE, PETSC_TRUE); CHKERRXX(ierr);
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
            ierr = PetscBinaryRead(fd, integer_parameters, 4, PETSC_INT); CHKERRXX(ierr);
            ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
          }
        int mpiret = MPI_Bcast(integer_parameters, 4, MPI_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
        fill_or_load_integer_parameters(flag, integer_parameters);

        // Now, load the double parameters:
        sprintf(diskfilename, "%s_doubles", filename);
        if(!file_exists(diskfilename))
          throw std::invalid_argument("The file storing the solver's double parameters could not be found");
        if(mpi->rank() == 0)
        {
          ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
          ierr = PetscBinaryRead(fd, double_parameters, 14, PETSC_DOUBLE); CHKERRXX(ierr);
          ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
        }
        mpiret = MPI_Bcast(double_parameters, 14, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
        fill_or_load_double_parameters(flag, double_parameters);

        break;
      }
    default:
      throw std::runtime_error("Unkown flag values were used when load/saving parameters \n");


    }



}

void save_state(const char* path_to_directory,double tn, unsigned int n_saved){


}

void load_state(const mpi_environment_t& mpi, const char* path_to_folder, double &tn){
  PetscErrorCode ierr;
  char filename[PATH_MAX];
  if(!is_folder(path_to_folder)) throw std::invalid_argument("Load state: path to directory is invalid \n");

  // First load the general solver parameters -- integers and doubles
  sprintf(filename, "%s/solver_parameters", path_to_folder);
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

  int num_16_to_17_increases = 0;
  int num_node_increases = 0;
  int num_17_to_18_decreases = 0;
  int num_ns_increases = 0;
  int num_ns_decreases = 0;

  int num_ngbd_increases = 0;
  int num_delete_grid_decreases = 0;

  for(int grid_res_iter=0;grid_res_iter<=num_splits;grid_res_iter++){

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
    int cube_refinement = 1;
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
    rho_s/=(scaling*scaling*scaling);

    if(solve_stefan){
        k_s/=scaling;
        k_l/=scaling;
        sigma/=scaling;

        alpha_l*=(scaling*scaling);
        alpha_s*=(scaling*scaling);
      }


    if(solve_navier_stokes){
        PetscPrintf(mpi.comm(),"Physical u0 = %0.3e \n"
                               "Physical v0 = %0.3e \n"
                               "Physical mu_l = %0.3e \n"
                               "Physical rho_l = %0.3e \n"
                               "Physical r0    = %0.3e [m] = %0.3e [cm] \n"
                               "Physical r_cyl = %0.3e [m] = %0.3e [cm] \n",u0,v0,mu_l,rho_physical,r0,r0*100.,r_cyl,r_cyl*100.);

      r0*=scaling;
      r_cyl*=scaling;

      //mu_l/=(scaling);         // No need to scale viscosity, it doesn't have a length scale in the units (that isn't embedded)
      u0*=scaling;             // Scale the initial velocities
      v0*=scaling;
      pressure_prescribed_value/=(scaling*scaling); // Scale the pressure BC prescribed value and flux
      pressure_prescribed_flux/=(scaling*scaling*scaling);

      PetscPrintf(mpi.comm(),"Reynolds number for this case is: %0.2f , %0.2f \n"
                             "Computational r0 = %0.4f \n"
                             "Computational mu = %0.3e \n"
                             "Computational u0 = %0.3e \n"
                             "Computational rho = %0.3e \n"
                             "Computational r0   = %0.3e \n"
                             "Computational r_cyl   = %0.3e \n",Re_u, Re_v, r0,mu_l,u0,rho_l,r0,r_cyl);

      PetscPrintf(mpi.comm(),"u initial is %0.3e, v initial is %0.3e \n",u0,v0);
          }

    // -----------------------------------------------
    // Create the grid:
    // -----------------------------------------------
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

    // create the forest
    p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

    // Create refinement splitting criteria related to level set and uniform band
    splitting_criteria_cf_and_uniform_band_t sp(lmin+grid_res_iter,lmax+grid_res_iter,&level_set,uniform_band);

    // Save the pointer to the forest splitting criteria
    p4est->user_pointer = &sp;

    // Refine grid according to the splitting criteria:
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    // partition the forest, do not allow for coarsening
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    // create ghost layer
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    // Expand ghost layer -- required for navier stokes | TO-DO: make this consistent with ghost layer boolean
    my_p4est_ghost_expand(p4est,ghost);

    // Create nodes
    nodes = my_p4est_nodes_new(p4est, ghost); //same

    // Create hierarchy
    my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);

    // Get node neighbors and initialize them
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
//    if(example_ == ICE_AROUND_CYLINDER){
//        phi_cylinder.create(phi.vec);
//        sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
//      }

    // 2nd derivatives of LSF's
    vec_and_ptr_dim_t phi_dd;
    vec_and_ptr_dim_t phi_solid_dd;
    vec_and_ptr_dim_t phi_cylinder_dd;

    // -----------------------------------------------
    // Initialize the interfacial velocity field (used for Stefan problem)
    // -----------------------------------------------
    vec_and_ptr_dim_t v_interface;/*(p4est,nodes)*/;
    vec_and_ptr_dim_t v_interface_new;
//    if(solve_stefan){
//        for (int dir = 0; dir<P4EST_DIM;dir++){
//            sample_cf_on_nodes(p4est,nodes,zero_cf,v_interface.vec[dir]);
//          }
//      }

    // Vectors to hold the normal and curvature information used to compute interfacial velocity due to jump in temp flux:
    vec_and_ptr_dim_t normal;
    vec_and_ptr_t curvature;
    vec_and_ptr_dim_t jump;



    // -----------------------------------------------
    // Initialize the fields relevant to the Poisson problem:
    // -----------------------------------------------
    // Vectors for T_liquid:
    vec_and_ptr_t T_l_n;
    vec_and_ptr_t rhs_Tl;

    vec_and_ptr_t T_l_nm1; // For storing the solution from the previous timestep

    if(solve_stefan){
        T_l_n.create(p4est,nodes);
        sample_cf_on_nodes(p4est,nodes,IC_temp_liq,T_l_n.vec); // Sample this just so that we can save the initial temperature distribution

        if(do_advection && advection_sl_order ==2){
            T_l_nm1.create(p4est,nodes);
            sample_cf_on_nodes(p4est,nodes,IC_temp_liq,T_l_nm1.vec);
          }
      }

    // Vector for advection of temperature:
    vec_and_ptr_t T_l_backtrace;
    vec_and_ptr_t T_l_backtrace_nm1;

    // Vectors for T_solid:
    vec_and_ptr_t T_s_n;
    vec_and_ptr_t rhs_Ts;

    if(solve_stefan){
        T_s_n.create(p4est,nodes);
        sample_cf_on_nodes(p4est,nodes,IC_temp_sol,T_s_n.vec); // Sample this just so that we can save the initial temperature distribution
      }

    // Initialize interfacial boundary condition
    BC_interface_value_temp bc_interface_val_temp(ngbd);



    // Vectors to hold T values on old grid (for interpolation purposes)
    vec_and_ptr_t T_l_new;
    vec_and_ptr_t T_s_new;

    // Vectors to hold first derivatives of T
    vec_and_ptr_dim_t T_l_d;
    vec_and_ptr_dim_t T_s_d;

    // Vectors to hold the normals of each domain:
    vec_and_ptr_dim_t liquid_normals;
    vec_and_ptr_dim_t solid_normals;

    // -----------------------------------------------
    // Initialize the Velocity field (if solving Navier-Stokes), and other Navier-Stokes relevant variables:
    // -----------------------------------------------
    vec_and_ptr_dim_t v_n;
    vec_and_ptr_dim_t v_n_new;

    vec_and_ptr_dim_t v_nm1;

    vec_and_ptr_t vorticity;
    vec_and_ptr_t vorticity_refine;
    vec_and_ptr_t u_component_refine;
    vec_and_ptr_cells_t press;
    vec_and_ptr_t press_nodes;
    vec_and_ptr_dim_t grad_p;

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

        if(example_ ==NS_GIBOU_EXAMPLE || example_ ==COUPLED_PROBLEM_EXAMPLE){
            press_nodes.create(p4est,nodes);
            sample_cf_on_nodes(p4est,nodes,p_ana_tn,press_nodes.vec);
          }
      }

    vec_and_ptr_cells_t hodge_old;
    vec_and_ptr_cells_t hodge_new;

    my_p4est_cell_neighbors_t *ngbd_c;
    my_p4est_faces_t *faces_np1;

    BC_interface_value_velocity_u bc_velocity_u_interfacial(ngbd);
    BC_interface_value_velocity_v bc_velocity_v_interfacial(ngbd);

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
    FILE *fich_stefan_errors;
    char name_stefan_errors[1000];
    if (example_ == FRANK_SPHERE){
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
      }



    // (2) For checking error on LLNL NS benchmark case:
    FILE *fich_NS_errors;
    char name_NS_errors[1000];
    if (example_ == NS_GIBOU_EXAMPLE){
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
      }

    // (3) For checking error on coupled example case:
    FILE *fich_coupled_errors;
    char name_coupled_errors[1000];
    if (example_ == COUPLED_PROBLEM_EXAMPLE){
      const char* out_dir_err_coupled = getenv("OUT_DIR_ERR_coupled");
      sprintf(name_coupled_errors,"%s/coupled_error_lmin_%d_lmax_%d_method_%d_advection_order_%d.dat",
              out_dir_err_coupled,lmin+grid_res_iter,lmax + grid_res_iter,method_,advection_sl_order);

      ierr = PetscFOpen(mpi.comm(),name_coupled_errors,"w",&fich_coupled_errors); CHKERRXX(ierr);
      ierr = PetscFPrintf(mpi.comm(),fich_coupled_errors,"time " "timestep " "iteration " "u_error " "v_error " "P_error " "Tl_error ""number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
      ierr = PetscFClose(mpi.comm(),fich_coupled_errors); CHKERRXX(ierr);
      }

    // (4) For checking memory usage
    FILE *fich_mem;
    char name_mem[1000];
    const char* out_dir_check_mem = getenv("OUT_DIR_MEM");
    sprintf(name_mem,"%s/memory_check_stefan_%d_NS_%d_lmin_%d_lmax_%d_method_%d_advection_order_%d.dat",
            out_dir_check_mem,solve_stefan,solve_navier_stokes,lmin+grid_res_iter,lmax+grid_res_iter,method_,advection_sl_order);
    ierr = PetscFOpen(mpi.comm(),name_mem,"w",&fich_mem); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_mem,"time " "timestep " "iteration "
                                            "mem values \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);

    // (5) For checking ice cylinder problem:
    FILE *fich_ice_radius_info;
    char name_ice_radius_info[1000];
    const char* out_dir_ice_cyl = getenv("OUT_DIR_ICE");
    sprintf(name_ice_radius_info,"%s/ice_cyl_info_lmin_%d_lmax_%d_method_%d_advection_order_%d.dat",
            out_dir_ice_cyl,lmin+grid_res_iter,lmax+grid_res_iter,method_,advection_sl_order);
    ierr = PetscFOpen(mpi.comm(),name_ice_radius_info,"w",&fich_ice_radius_info); CHKERRXX(ierr);
    ierr = PetscFPrintf(mpi.comm(),fich_ice_radius_info,"time "
                        "max_v_norm "
                        "number_elements "
                        "theta_N "
                        "delta_r_N ");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich_ice_radius_info); CHKERRXX(ierr);


    // -----------------------------------------------
    // Initialize the needed solvers for the Temperature problem
    // -----------------------------------------------
    my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
    my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

    // -----------------------------------------------
    // Initialize the needed solvers for the Navier-Stokes problem
    // -----------------------------------------------
    my_p4est_navier_stokes_t* ns;
    my_p4est_poisson_cells_t* cell_solver; // TO-DO: These may be unnecessary now
    my_p4est_poisson_faces_t* face_solver;

    BoundaryConditions2D bc_velocity[P4EST_DIM];
    BoundaryConditions2D bc_pressure;


    // -----------------------------------------------
    // Begin stepping through time
    // -----------------------------------------------
    int tstep = 0;
    double tstart = tn;
    for (tn;tn<tfinal; tn+=dt, tstep++){
//        if (!keep_going) break;
//        if(tstep>=10) keep_going = false; // TIMESTEP BREAK
        PetscLogDouble mem_safety_check;

        // Initialize variables for checking the current memory usage
        PetscLogDouble mem1=0.,mem2=0.,mem3=0.,mem4=0.,mem5=0.,mem6=0.,mem7=0.,mem8=0.,mem9=0.,mem10=0.,mem11=0.,mem12=0.,mem13=0.;
        PetscLogDouble mem_grid1=0.,mem_grid2=0.;

        PetscLogDouble memP1=0.,memP2=0.,memP3=0.,memP4=0.,memP5=0.,memP6=0.,memP7=0.;//mem9a,mem9b,mem9c,mem9d,mem9e,mem9f,mem9g;
        PetscLogDouble memNS1=0.,memNS2=0.,memNS3=0.,memNS4=0.,memNS5=0.,memNS6=0.;//mem10a,mem10b,mem10c,mem10d,mem10d1,mem10e;
        PetscLogDouble memNS_H1=0.,memNS_H2=0.,memNS_H3=0.,memNS_H4=0.,memNS_H5=0.;//mem10c1,mem10c2,mem10c3,mem10c4,mem10c11;

        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem1);
          }


        // --------------------------------------------------------------------------------------------------------------
        // Print iteration information:
        // --------------------------------------------------------------------------------------------------------------

        PetscPrintf(mpi.comm(),"\n -------------------------------------------\n");
        ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3g [s], %0.3g [min], Timestep: %0.3e [s], Percent Done : %0.2f % \n ------------------------------------------- \n",tstep,tn,tn/60.,dt,((tn-tstart)/tfinal)*100.0);
        if(solve_stefan){
            ierr = PetscPrintf(mpi.comm(),"\n Previous interfacial velocity (max norm) is %0.3g \n",v_interface_max_norm);
            if(v_interface_max_norm>v_int_max_allowed){
                PetscPrintf(mpi.comm(),"Interfacial velocity has exceeded its max allowable value \n");
                PetscPrintf(mpi.comm(),"Max allowed is : %g \n",v_int_max_allowed);
                MPI_Abort(mpi.comm(),1);
              }
          }
        // --------------------------------------------------------------------------------------------------------------
        // Define some variables needed to specify how to extend across the interface:
        // --------------------------------------------------------------------------------------------------------------
        // Get smallest grid size:
        dxyz_min(p4est,dxyz_smallest);

        dxyz_close_to_interface = 1.2*max(dxyz_smallest[0],dxyz_smallest[1]);
        min_volume_ = MULTD(dxyz_smallest[0], dxyz_smallest[1], dxyz_smallest[2]);
        extension_band_use_    = (8.)*pow(min_volume_, 1./ double(P4EST_DIM)); //8
        extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
        extension_band_check_  = (6.)*pow(min_volume_, 1./ double(P4EST_DIM)); // 6

        double dxyz_min_ = MIN(DIM(SQR(dxyz_smallest[0]),SQR(dxyz_smallest[1]), SQR(dxyz_smallest[2])));
        double diag_ = sqrt(SUMD(SQR(dxyz_smallest[0]),SQR(dxyz_smallest[1]), SQR(dxyz_smallest[2])));




        if(example_ == ICE_AROUND_CYLINDER){
            double delta_r = r0 - r_cyl;
            PetscPrintf(mpi.comm(),"The uniform band is %0.2f\n",uniform_band);

//            extension_band_extend_ = uniform_band;
//            extension_band_use_ = uniform_band/2.;
//            extension_band_check_ = uniform_band/3.;
            if(delta_r<3.*dxyz_close_to_interface){
                PetscPrintf(mpi.comm()," Your initial delta_r is %0.3e, and it must be at least %0.3e \n",delta_r,4.*dxyz_close_to_interface);
                SC_ABORT("Your initial delta_r is too small \n");
              }
          }
        // If first iteration, perturb the LSF(s):
        my_p4est_level_set_t ls(ngbd);
        if(tstep<1){
            // Perturb the LSF on the first iteration

            ls.perturb_level_set_function(phi.vec,EPS);
          }

        // --------------------------------------------------------------------------------------------------------------
        // Extend Fields Across Interface (if solving Stefan):
        // -- Note: we do not extend NS velocity fields bc NS solver handles that internally
        // --------------------------------------------------------------------------------------------------------------
        // Define LSF for the solid domain (as just the negative of the liquid one):
        if(solve_stefan){
            phi_solid.create(p4est,nodes);
            VecCopyGhost(phi.vec,phi_solid.vec);
            VecScaleGhost(phi_solid.vec,-1.0);


            // Compute normals for each domain:
            liquid_normals.create(p4est,nodes);
            compute_normals(*ngbd,phi.vec,liquid_normals.vec);

            solid_normals.create(p4est,nodes);
            compute_normals(*ngbd,phi_solid.vec,solid_normals.vec);

            // Extend Temperature Fields across the interface: // WAS USING 1ST ORDER, NOW CHANGED TO SECOND
            ls.extend_Over_Interface_TVD_Full(phi.vec, T_l_n.vec, 50, 2, 1.e-15, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normals.vec, NULL, NULL, false, NULL, NULL);
            ls.extend_Over_Interface_TVD_Full(phi_solid.vec, T_s_n.vec, 50, 2, 1.e-15, extension_band_use_, extension_band_extend_, extension_band_check_, solid_normals.vec, NULL, NULL, false, NULL, NULL);

//            foreach_dimension(d){
//              ls.extend_Over_Interface_TVD_Full(phi.vec, v_n.vec[d], 50, 2, 1.e-15, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normals.vec, NULL, NULL, false, NULL, NULL);
//            }




            // Delete data for normals since it is no longer needed:
            liquid_normals.destroy();
            solid_normals.destroy();

            if (check_temperature_values){
              // Check Temperature values:
              PetscPrintf(mpi.comm(),"\n Checking temperature values after field extension: \n [ ");
              PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
              check_T_values(phi,T_l_n,nodes,p4est,example_,phi_cylinder,true);
              PetscPrintf(mpi.comm(),"\nIn solid domain: ");
              check_T_values(phi_solid,T_s_n,nodes,p4est,example_,phi_cylinder,true);
              PetscPrintf(mpi.comm()," ] \n");

//              // Check Smoke values:
//              PetscPrintf(mpi.comm(),"\n Checking smoke values after interpolating onto new grid: \n [ ");
//              PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
//              check_T_values(phi,smoke,nodes,p4est,example_,phi_cylinder);
//              PetscPrintf(mpi.comm(),"\nIn solid domain: ");
//              check_T_values(phi_solid,smoke,nodes,p4est,example_,phi_cylinder);
//              PetscPrintf(mpi.comm()," ] \n");
              }

          } // end of "if save stefan"
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes field extension \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem2);}

        // --------------------------------------------------------------------------------------------------------------
        // SAVING DATA: Save data every specified amout of timesteps: -- Do this after values are extended across interface to make visualization nicer
        // --------------------------------------------------------------------------------------------------------------

        if((tstep>0 && (tstep%save_every_iter)==0)){
            PetscPrintf(mpi.comm(),"Saving to vtk ... \n");
          char output[1000];
          if(save_coupled_fields || save_stefan || save_navier_stokes){out_idx++;}
          if(save_coupled_fields){
              const char* out_dir_coupled = getenv("OUT_DIR_VTK_coupled");
              if(!out_dir_coupled){
                  throw std::invalid_argument("You need to set the output directory for coupled VTK: OUT_DIR_VTK_coupled");
                }
              phi_cylinder.create(p4est,nodes);
              sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);

              sprintf(output,"%s/output_lmin_%d_lmax_%d_advection_order_%d_stefan_%d_NS_%d_outidx_%d",out_dir_coupled,lmin+grid_res_iter,lmax+grid_res_iter,advection_sl_order,solve_stefan,solve_navier_stokes,out_idx);
              save_everything(p4est,nodes,ghost,phi,phi_cylinder,T_l_n,T_s_n,v_interface,v_n,press_nodes,vorticity,press,output);

              phi_cylinder.destroy();
/*
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

                  sprintf(output,"%s/snapshot_coupled_analytical_lmin_%d_lmax_%d_outiter_%d",out_dir_coupled,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
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
            */}
          if(save_stefan){
              const char* out_dir_stefan = getenv("OUT_DIR_VTK_stefan");
              sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_outiter_%d",out_dir_stefan,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);

              save_stefan_fields(p4est,nodes,ghost,phi,phi_cylinder,T_l_n,T_s_n,v_interface,output);
            }
          if(save_navier_stokes){
              const char* out_dir_ns = getenv("OUT_DIR_VTK_NS");
              sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_outiter_%d",out_dir_ns,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
              save_navier_stokes_fields(p4est,nodes,ghost,phi,v_n,press_nodes,vorticity,output);

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

                  sprintf(output,"%s/snapshot_NS_ana_and_errors_lmin_%d_lmax_%d_outiter_%d",out_dir_ns,lmin+grid_res_iter,lmax+grid_res_iter,out_idx);
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
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes saving to VTK \n");

          }

        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem3);
          }


        if(stop_flag) MPI_Abort(mpi.comm(),1);
        // --------------------------------------------------------------------------------------------------------------
        // Compute the jump in flux across the interface to use to advance the LSF (if solving Stefan:
        // --------------------------------------------------------------------------------------------------------------
        if(solve_stefan){

            // Get the first derivatives to compute the jump
            T_l_d.create(p4est,nodes); T_s_d.create(T_l_d.vec);
            ngbd->first_derivatives_central(T_l_n.vec,T_l_d.vec);
            ngbd->first_derivatives_central(T_s_n.vec,T_s_d.vec);

            // Create vector to hold the jump values:
            jump.create(p4est,nodes);
            v_interface.destroy();
            v_interface.create(p4est,nodes);

//            // Call the compute_velocity_function:
//            if(example_ == ICE_AROUND_CYLINDER && tstep<5){
//                force_interfacial_velocity_to_zero=true;
//              }
//            else if(example_ == ICE_AROUND_CYLINDER && tstep==5){force_interfacial_velocity_to_zero=false;}
            PetscPrintf(mpi.comm(),"Interfacial velocity forced to zero? %s \n",force_interfacial_velocity_to_zero?"Yes ": "No");

            compute_interfacial_velocity(T_l_d,T_s_d,jump,v_interface,phi,ngbd,dxyz_close_to_interface);

            // Destroy values once no longer needed:
            T_l_d.destroy();
            T_s_d.destroy();
            jump.destroy();
          }

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes computing the jump \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem4);
          }

        // --------------------------------------------------------------------------------------------------------------
        // Compute the timestep -- determined by velocity at the interface:
        // --------------------------------------------------------------------------------------------------------------
        // Save previous timestep:
        dt_nm1 = dt;
        if(solve_stefan){
            PetscPrintf(mpi.comm(),"\n"
                                   "[Stefan problem specific info:] \n"
                                   "----------------------------- \n");
            compute_timestep(v_interface, phi, dxyz_close_to_interface, dxyz_smallest,nodes,p4est); // this function modifies the variable dt
          }


        if(solve_navier_stokes){
            // STILL TO FIGURE OUT: GETTING NS TIMESTEP that makes sense
            // Take into consideration the Navier - Stokes timestep:
            // Take into account the NS timestep: -- probably better to do this with the max NS norm and CFL in the main file, not internally in NS
            if(solve_stefan){
                PetscPrintf(mpi.comm(),"\n"
                                       "NS contribution to timestep: \n"
                                       " - Navier Stokes: %0.3e \n"
                                       " - Official timestep used: %0.3e \n ",dt_NS,min(dt,dt_NS));
                if(advection_sl_order==2) PetscPrintf(mpi.comm()," - dt_nm1 : %0.3e \n ",dt_nm1);
                dt = min(dt,dt_NS);
              }
            else{
                if(tstep==0){dt_nm1 = dt_NS;}
                // If we are only solving Navier Stokes
                dt = dt_NS;
              }
          }

        if(tstep ==0){
            dt_nm1 = dt;
          }

        // Adjust the timestep if we are near the end of our simulation, to get the proper end time:
        if(tn + dt > tfinal){
            dt = tfinal - tn;
          }
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes computing timestep \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem5);
          }

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

        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem_grid1);
          }


        // Create the semi-lagrangian object and do the advection:
        my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

        // Build refinement criteria for Navier - Stokes problem:
        if(tstep == 0 && solve_navier_stokes){
            vorticity.create(p4est,nodes);
            sample_cf_on_nodes(p4est,nodes,zero_cf,vorticity.vec);
            NS_norm = max(u0,v0);

//            press_nodes.create(p4est,nodes);
//            sample_cf_on_nodes(p4est,nodes,zero_cf,press_nodes.vec);

//            grad_p.create(p4est,nodes);
//            ngbd->first_derivatives_central(press_nodes.vec,grad_p.vec);
//            press_nodes.destroy();
          }
        std::vector<compare_option_t> compare_opn;
        std::vector<compare_diagonal_option_t> diag_opn;
        std::vector<double> criteria;
        int num_fields = 2;//1;

        bool use_block = false;
        bool expand_ghost_layer = true;
        double threshold = 0.1;
//        vec_and_ptr_dim_t grad_p_refine;

        Vec fields_[num_fields];
        if(solve_navier_stokes && (num_fields!=0)){
            // Only use values of vorticity in the positive subdomain for refinement:
            vorticity_refine.create(p4est,nodes);

            u_component_refine.create(p4est,nodes);

//            grad_p_refine.create(p4est,nodes);



            vorticity.get_array();
            vorticity_refine.get_array();

            u_component_refine.get_array();

/*            grad_p_refine.get_array();
            grad_p.get_array()*/;

            phi.get_array();
            v_n.get_array();

            double xyz_b[P4EST_DIM];
            // NOTE: TO-DO : SEEMS LIKE YOU NEED TO UPDATE GHOST VALUES OTHERWISE YOU GET ISSUES WITH THE REFINEMENT
            foreach_local_node(n,nodes){
              if(phi.ptr[n] < 0.){
                  vorticity_refine.ptr[n] = vorticity.ptr[n];

                  u_component_refine.ptr[n] = (v_n.ptr[0][n]/u0 < 0.) ? 1.: 0.; // if this value is negative, we have backflow, and we will refine
//                  if(u_component_refine.ptr[n]/u0 < 0.) {
//                      node_xyz_fr_n(n,p4est,nodes,xyz_b);
//                      printf("BACKFLOW at (%0.5f, %0.5f), value: %0.4e, div: %0.4e \n",xyz_b[0],xyz_b[1],v_n.ptr[0][n],v_n.ptr[0][n]/u0);
//                    }
//                  foreach_dimension(d){
//                    grad_p_refine.ptr[d][n] = grad_p.ptr[d][n];
//                  }
                }
              else{
                  vorticity_refine.ptr[n] = 0.0;
                  u_component_refine.ptr[n] = 0.;
//                  foreach_dimension(d){
//                    grad_p_refine.ptr[d][n] = 0.0;
//                  }
                }
            }
            v_n.restore_array();
//            VecView(u_component_refine.vec,PETSC_VIEWER_STDOUT_WORLD);

            vorticity.restore_array();
            vorticity_refine.restore_array();
            phi.restore_array();

//            grad_p_refine.restore_array();
//            grad_p.restore_array();

            u_component_refine.restore_array();

            fields_[0] = vorticity_refine.vec;
            fields_[1] = u_component_refine.vec;

//            fields_[1] = grad_p_refine.vec[0];
//            fields_[2] = grad_p_refine.vec[1];

            // Coarsening instructions: (for vorticity)
            compare_opn.push_back(LESS_THAN);
            diag_opn.push_back(DIVIDE_BY);
            criteria.push_back(threshold*NS_norm/2.);

            // Refining instructions: (for vorticity)
            compare_opn.push_back(GREATER_THAN);
            diag_opn.push_back(DIVIDE_BY);
            criteria.push_back(threshold*NS_norm);

            // Coarsening instructions: (for u_component)
            compare_opn.push_back(LESS_THAN);
            diag_opn.push_back(ABSOLUTE);
            criteria.push_back(0.5); // Note: this coarsening instruction will never be triggered. Just put here as a placeholder.

            // Refining instructions: (for u_component)
            compare_opn.push_back(GREATER_THAN);
            diag_opn.push_back(ABSOLUTE);
            criteria.push_back(0.5);

//            // Coarsening instructions: (for dp/dx)
//            compare_opn.push_back(LESS_THAN);
//            diag_opn.push_back(DIVIDE_BY);
//            criteria.push_back(threshold*SQR(NS_norm)*rho_l/2.);

//            // Refining instructions: (for dp/dx)
//            compare_opn.push_back(GREATER_THAN);
//            diag_opn.push_back(DIVIDE_BY);
//            criteria.push_back(threshold*SQR(NS_norm)*rho_l);

//            // Coarsening instructions: (for dp/dy)
//            compare_opn.push_back(LESS_THAN);
//            diag_opn.push_back(DIVIDE_BY);
//            criteria.push_back(threshold*SQR(NS_norm)*rho_l/2.);

//            // Refining instructions: (for dp/dy)
//            compare_opn.push_back(GREATER_THAN);
//            diag_opn.push_back(DIVIDE_BY);
//            criteria.push_back(threshold*SQR(NS_norm)*rho_l);


          }

        bool nodes_increased = false;
        MPI_Barrier(mpi.comm());
        p4est_gloidx_t no_nodes1 = 0;
        for (int i = 0; i<p4est_np1->mpisize; i++){
          no_nodes1 += nodes_np1->global_owned_indeps[i];
        }
//        PetscPrintf(mpi.comm(),"BEFORE UPDATE: Current grid has %d nodes \n", no_nodes1);

        // Create second derivatives for phi in the case that we are using update_p4est:
        if(solve_stefan){
            phi_dd.create(p4est,nodes);
            ngbd->second_derivatives_central(phi.vec,phi_dd.vec);
          }

        // Advect the LSF and update the grid under the v_interface field:
        if(solve_coupled){
            phi_cylinder.create(p4est_np1,nodes_np1);
            sample_cf_on_nodes(p4est_np1,nodes_np1,mini_level_set,phi_cylinder.vec);
            if(example_ == ICE_AROUND_CYLINDER){
//                uniform_band = 4.;
//                VecView(v_interface.vec[0],PETSC_VIEWER_STDOUT_WORLD);
                sl.update_p4est(v_interface.vec, dt, phi.vec, phi_dd.vec, phi_cylinder.vec,num_fields ,use_block ,true,uniform_band,uniform_band*(1.5),fields_ ,NULL,criteria,compare_opn,diag_opn,expand_ghost_layer);
              }
            else{
                  sl.update_p4est(v_interface.vec, dt, phi.vec, phi_dd.vec, NULL,num_fields ,use_block ,true,uniform_band,uniform_band*(1.5),fields_ ,NULL,criteria,compare_opn,diag_opn,expand_ghost_layer);
              }

            phi_cylinder.destroy();
          }
        else if (solve_stefan && !solve_navier_stokes){
//              sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec);
              sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec,NULL,0,use_block,true,1.0,1.0,fields_,NULL,criteria,compare_opn,diag_opn,expand_ghost_layer);

          }
        else if (/*example_ == ICE_AROUND_CYLINDER*/solve_navier_stokes && !solve_stefan){
            bool use_standard_method = false;

//            if(use_standard_method){
//              phi_dd.create(p4est,nodes);
//              ngbd->second_derivatives_central(phi.vec,phi_dd.vec);

//              v_interface.create(p4est,nodes);
//              v_interface.get_array();
//              foreach_node(n,nodes){
//                foreach_dimension(d){
//                  v_interface.ptr[d][n] = 0.0;

//                }

//              }
//              v_interface.restore_array();
//              sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec);
//              v_interface.destroy();
//              phi_dd.destroy();
//            }

            if(!use_standard_method){
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
                      is_grid_changing = sp_NS.refine_and_coarsen(p4est_np1,nodes_np1,phi_new.vec,num_fields,use_block,true,1.0,1.0,fields_new_,NULL,criteria,compare_opn,diag_opn);

                      if(no_grid_changes>0 && !is_grid_changing){
                          last_grid_balance = true; // if the grid isn't changing anymore but it has changed, we need to do one more special interp of fields and balancing of the grid
                        }
                    }

                  if(is_grid_changing || last_grid_balance){
                      no_grid_changes++;
                      PetscPrintf(mpi.comm(),"NS grid changed %d times \n",no_grid_changes);
                      if(last_grid_balance){
                          p4est_balance(p4est_np1,P4EST_CONNECT_FULL,NULL);
                        }

                      my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);
                      p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
                      my_p4est_ghost_expand(p4est_np1,ghost_np1);
                      p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);

                      // Destroy fields_new and create it on the new grid:

                      if(num_fields!=0){
                          for(unsigned int k = 0;k<num_fields; k++){
                              ierr = VecDestroy(fields_new_[k]);
                              ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
                            }
                        }

                      // Destroy phi_new and create on new grid:
                      phi_new.destroy();
                      phi_new.create(p4est_np1,nodes_np1);


                      // Interpolate the fields and phi to the new grid:
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

                      // Interpolate the phi onto the new grid:
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
            }
          } // end of if only navier stokes


        // Destroy old derivative values in the case that we used it for update_p4est:
        if(solve_stefan)phi_dd.destroy();

        // Destroy refinement vorticity:
        if(solve_navier_stokes)vorticity_refine.destroy(); u_component_refine.destroy();/*grad_p_refine.destroy();*/


        // Clear up the memory from the std vectors holding refinement info:
        compare_opn.clear(); diag_opn.clear(); criteria.clear();
        compare_opn.shrink_to_fit(); diag_opn.shrink_to_fit(); criteria.shrink_to_fit();

        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem_grid2);
          }

        // Get the new neighbors:
        my_p4est_hierarchy_t *hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
        my_p4est_node_neighbors_t *ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

        // Initialize the neigbors:
        ngbd_np1->init_neighbors();

        // FOR FUTURE NOTICE :: functions that exist are: ngbd->update() and hierarchy->update() --> Look at how Daniil does it
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes grid refinement/LSF advection \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem6);
            if(mem6>mem_grid2){
                num_ngbd_increases++;
                PetscPrintf(mpi.comm(),"There have been %d ngbd increases \n",num_ngbd_increases);
                PetscPrintf(mpi.comm(),"ngbd increase value : %0.6e \n",mem6-mem_grid2);
              }
          }
        p4est_gloidx_t no_nodes2 = 0;
        for (int i = 0; i<p4est_np1->mpisize; i++){
          no_nodes2 += nodes_np1->global_owned_indeps[i];
        }
        if(no_nodes2>no_nodes1) num_node_increases++;

//        // Check solidification front to see if it needs to be regularlized: // TO-DO: THIS WILL CREATE A MEMORY LEAK -- THINK BEFORE UNCOMMENTING
//        phi_solid.create(p4est_np1,nodes_np1);
//        VecCopyGhost(phi.vec,phi_solid.vec);
//        VecScaleGhost(phi_solid.vec,-1.);
//        regularize_front(p4est_np1,nodes_np1,ghost_np1,ngbd_np1,hierarchy_np1,brick,diag_,dxyz_min_,phi_solid,true);
//        VecCopyGhost(phi_solid.vec,phi.vec);
//        VecScaleGhost(phi.vec,-1.0);



        // Reinitialize the LSF on the new grid: -- NOT SURE IF WE NEED TO DO THIS EVERY TIME IF JUST NS AND NOTHING ELSE
        my_p4est_level_set_t ls_new(ngbd_np1);
        ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec, 50);
        ls_new.perturb_level_set_function(phi.vec,EPS);
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Does reinitialization \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem7);
          }

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

        // Interpolate things to the new grid:
        interpolate_values_onto_new_grid(T_l_n,T_l_new,
                                         T_s_n, T_s_new,
                                         v_interface, v_interface_new,
                                         v_n, v_n_new,
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

        if(solve_navier_stokes){
            v_n.destroy(); v_n.create(p4est_np1,nodes_np1);
            foreach_dimension(d){
              VecCopyGhost(v_n_new.vec[d],v_n.vec[d]);
            }
            v_n_new.destroy();
          }
        if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes interpolation of values onto new grid \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem8);
          }

        // --------------------------------------------------------------------------------------------------------------
        // Compute the normal and curvature of the interface -- curvature is used in some of the interfacial boundary condition(s) on temperature
        // --------------------------------------------------------------------------------------------------------------

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

        if(print_checkpoints) PetscPrintf(mpi.comm(),"Computes normal and curvature \n");
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem9);
          }
        // --------------------------------------------------------------------------------------------------------------
        // Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
        // --------------------------------------------------------------------------------------------------------------
        // Get most updated derivatives of the LSF's (on current grid) -- Solver uses these:
        // ------------------------------------------------------------
        if(solve_stefan){
          PetscPrintf(mpi.comm(),"\n"
                                 "-----------------------------\n"
                                 "[Temperature problem specific info:] \n"
                                 "----------------------------- \n \n");
          if(print_checkpoints)PetscPrintf(mpi.comm(),"Beginning Poisson problem ... \n");
          phi_solid_dd.create(p4est_np1,nodes_np1);
          ngbd_np1->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

          phi_dd.create(p4est_np1,nodes_np1);
          ngbd_np1->second_derivatives_central(phi.vec,phi_dd.vec);

          if(example_ ==ICE_AROUND_CYLINDER){
              phi_cylinder.create(p4est_np1,nodes_np1);
              sample_cf_on_nodes(p4est_np1,nodes_np1,mini_level_set,phi_cylinder.vec);

              phi_cylinder_dd.create(p4est_np1,nodes_np1);
              ngbd_np1->second_derivatives_central(phi_cylinder.vec,phi_cylinder_dd.vec);
            }

          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP1);

            }

          // ---------------------------------------
          // Compute advection terms (if applicable):
          // ---------------------------------------
          if (do_advection){
              if(print_checkpoints) PetscPrintf(mpi.comm(),"Gets into do advection \n\n");
              // Create backtrace vectors:
              T_l_backtrace.create(p4est_np1,nodes_np1);

              if(advection_sl_order ==2){
                  T_l_backtrace_nm1.create(p4est_np1,nodes_np1);
                }
//              VecView(v_nm1.vec[0],PETSC_VIEWER_STDOUT_WORLD);
              do_backtrace(T_l_n,T_l_nm1,T_l_backtrace,v_n,p4est_np1,nodes_np1,ngbd_np1,p4est,nodes,ngbd, T_l_backtrace_nm1,v_nm1,interp_bw_grids,phi);

              if(true/*check_temperature_values*/){
                  PetscPrintf(mpi.comm(),"\n Checking temperature values for backtrace: \n [ ");
                  PetscPrintf(mpi.comm(),"\n for n: ");
                  check_T_values(phi,T_l_backtrace,nodes_np1,p4est_np1, example_,phi_cylinder,true);
                }
              if(false/*advection_sl_order ==2 && check_temperature_values*/){
                  PetscPrintf(mpi.comm(),"\n for nm1: ");
                  check_T_values(phi,T_l_backtrace_nm1,nodes_np1,p4est_np1,example_,phi_cylinder,true);
                  PetscPrintf(mpi.comm()," ] \n");

                }

              // Do backtrace with v_n --> navier-stokes fluid velocity
          } // end of do_advection if statement
          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP2);
            }
          // ------------------------------------------------------------
          // Setup RHS:
          // ------------------------------------------------------------
          // Create arrays to hold the RHS:
          rhs_Tl.create(p4est_np1,nodes_np1);
          rhs_Ts.create(p4est_np1,nodes_np1);
//          if (solve_smoke) rhs_smoke.create(p4est_np1,nodes_np1);

          // Set up the RHS:

          setup_rhs(phi,T_l_n,T_s_n,
                    rhs_Tl,rhs_Ts,
                    T_l_backtrace,T_l_backtrace_nm1,
                    p4est_np1,nodes_np1,ngbd_np1);
          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP3);
//              PetscPrintf(mpi.comm(),"memP3: index 14, Before starting Poisson process: %0.6e \n",memP3);
            }
          if(check_temperature_values){
              PetscPrintf(mpi.comm(),"\n Checking rhs values for T_l: \n [ ");
              check_T_values(phi,rhs_Tl,nodes_np1,p4est_np1, example_,phi_cylinder,false);
            }

          // ------------------------------------------------------------
          // Setup the solvers:
          // ------------------------------------------------------------
          // Now, set up the solver(s):
          solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd_np1);
          solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd_np1);

          bc_interface_val_temp.clear();
          bc_interface_val_temp.create(ngbd_np1,curvature.vec);

          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP4);
            }

          // Add the appropriate interfaces and interfacial boundary conditions:
          solver_Tl->add_boundary(MLS_INTERSECTION,phi.vec,phi_dd.vec[0],phi_dd.vec[1],interface_bc_type_temp,bc_interface_val_temp,bc_interface_coeff);
          solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid.vec,phi_solid_dd.vec[0],phi_solid_dd.vec[1],interface_bc_type_temp,bc_interface_val_temp,bc_interface_coeff);

          if(example_ == ICE_AROUND_CYLINDER){
            solver_Ts->add_boundary(MLS_INTERSECTION,phi_cylinder.vec,phi_cylinder_dd.vec[0],phi_cylinder_dd.vec[1],inner_interface_bc_type_temp,bc_interface_val_inner,bc_interface_coeff_inner);
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
          solver_Tl->set_wc(wall_bc_type_temp,wall_bc_value_temp);
          solver_Ts->set_wc(wall_bc_type_temp,wall_bc_value_temp);


          // Save the old T values if doing second order advection:
          if(do_advection && advection_sl_order ==2){
              T_l_nm1.destroy(); T_l_nm1.create(p4est_np1,nodes_np1);
              VecCopyGhost(T_l_n.vec,T_l_nm1.vec);
            }

          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP5);
              PetscPrintf(mpi.comm(),"memP5: idx 16: After setting up solvers: %0.6e \n",memP5);

            }

          // Preassemble the linear system
          solver_Tl->preassemble_linear_system();

          solver_Ts->preassemble_linear_system();

          // Solve the system:
          solver_Tl->solve(T_l_n.vec,false,true,KSPBCGS,PCHYPRE);
          solver_Ts->solve(T_s_n.vec,false,true,KSPBCGS,PCHYPRE);


          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP6);
              PetscPrintf(mpi.comm(),"16 TO 17 INCREASE: %0.10e \n \n",memP6 - memP5);
              PetscPrintf(mpi.comm(),"MEM INCREASE: %s\n",memP6 > memP5?"YES":"NO");

              if((memP6-memP5)>0.){ num_16_to_17_increases++;PetscPrintf(mpi.comm(),"Memory increased \n");}
              PetscPrintf(mpi.comm(),"There have been %d increases from 16 to 17 \n",num_16_to_17_increases);
              PetscPrintf(mpi.comm(),"There have been %d increases in grid size \n",num_node_increases);

            }

          delete solver_Tl;
          delete solver_Ts;

          // Check Temperature values:
          PetscPrintf(mpi.comm(),"--> Temperature values in the fluid domain: \n");
          check_T_values(phi,T_l_n,nodes_np1,p4est_np1,example_,phi_cylinder,true);
          PetscPrintf(mpi.comm(),"\n"
                                 "-->Temperature values in the solid domain: \n");
          check_T_values(phi_solid,T_s_n,nodes_np1,p4est_np1,example_,phi_cylinder,true);
          PetscPrintf(mpi.comm(),"\n \n");


          // Destroy auxiliary vectors:
          phi_dd.destroy();

          phi_solid.destroy();
          phi_solid_dd.destroy();

          curvature.destroy();


          // Destroy other fields that are no longer needed:
          if(example_ ==ICE_AROUND_CYLINDER){
              phi_cylinder.destroy();
              phi_cylinder_dd.destroy();

            }

          // Destroy rhs vectors now that no longer in use:
          rhs_Tl.destroy();
          rhs_Ts.destroy();

          // Destroy backtrace vectors now that no longer in use:
          if(do_advection) T_l_backtrace.destroy();
          if(do_advection && advection_sl_order==2) T_l_backtrace_nm1.destroy();

//          MPI_Barrier(mpi.comm());
//          PetscLogDouble m5;
//          PetscMemoryGetCurrentUsage(&m5);
//          PetscPrintf(mpi.comm(),"Deleted rhs vectors: %0.6e\n",m5);


          if(mem_checkpoints){
              MPI_Barrier(mpi.comm());
              PetscMemoryGetCurrentUsage(&memP7);
//              PetscPrintf(mpi.comm(),"Index 18 : %0.6e \n",memP7);
              PetscPrintf(mpi.comm(),"DECREASE FROM 17 TO 18: %0.6e \n",memP7 - memP6);
              if((memP7-memP6)<0.) {num_17_to_18_decreases++;PetscPrintf(mpi.comm(),"Memory decreased \n");}
              PetscPrintf(mpi.comm(),"There have been %d decreases from 17 to 18 \n",num_17_to_18_decreases);
            }
          // ------------------------------------------------------------
          // Some example specific operations for the Poisson problem:
          // ------------------------------------------------------------
          // Check error on the Frank sphere, if relevant:
          if(example_ == FRANK_SPHERE){
              check_frank_sphere_error(T_l_n, T_s_n, phi, v_interface, p4est_np1, nodes_np1, dxyz_close_to_interface,name_stefan_errors,fich_stefan_errors,tstep);
            }
          if(print_checkpoints) PetscPrintf(mpi.comm(),"Poisson step complete \n\n");
      } // end of "if solve stefan"


        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem10);
//            PetscPrintf(mpi.comm(),"index 19 - Exited Poisson problem: %0.6e \n",mem10);
          }

        // --------------------------------------------------------------------------------------------------------------
        // Navier-Stokes Problem: Setup and solve a NS problem in the liquid subdomain
        // --------------------------------------------------------------------------------------------------------------


        if (solve_navier_stokes){
            PetscPrintf(mpi.comm(),"\n"
                                   "-----------------------------\n"
                                   "[Navier-Stokes problem specific info:] \n"
                                   "----------------------------- \n \n");
            if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Navier-Stokes problem ... \n");
            // Get the cell neighbors:
            ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_np1);

            // Create the faces:
            faces_np1 = new my_p4est_faces_t(p4est_np1,ghost_np1,&brick,ngbd_c);

            // First, initialize the Navier-Stokes solver with the grid:
            if(tstep ==0){
              ns = new my_p4est_navier_stokes_t(ngbd,ngbd_np1,faces_np1);

              // Set the LSF:
              ns->set_phi(phi.vec);

              vec_and_ptr_dim_t v_n_NS(p4est_np1,nodes_np1);
              vec_and_ptr_dim_t v_nm1_NS(p4est,nodes);
              foreach_dimension(d){
                ierr = VecCopyGhost(v_n.vec[d],v_n_NS.vec[d]);
                ierr = VecCopyGhost(v_nm1.vec[d],v_nm1_NS.vec[d]);
              }
              // These get passed into the NS solver to handle, and NS solver will handle deleting them

              ns->set_parameters(mu_l,rho_l,2/*advection_sl_order*/,NULL,NULL,cfl);
              ns->set_velocities(v_nm1_NS.vec,v_n_NS.vec);
            }

            if(mem_checkpoints){
                MPI_Barrier(mpi.comm());
                PetscMemoryGetCurrentUsage(&memNS1);
              }

            // Set the timestep: // change to include both timesteps (dtnm1,dtn)
            if(print_checkpoints) PetscPrintf(mpi.comm()," Setting timestep for NS \n");
            if(advection_sl_order ==2){
                ns->set_dt(dt_nm1,dt);
              }
            else{
                ns->set_dt(dt);
              }

            // Update the NS grid:
            if(tstep>0){
                if(print_checkpoints) PetscPrintf(mpi.comm(),"Calling update grid from tn to tnp1... \n");
                ns->update_from_tn_to_tnp1_grid_external(phi.vec,p4est_np1,nodes_np1,ghost_np1,ngbd_np1,faces_np1,ngbd_c,hierarchy_np1);
              }
            // NOTE: we update NS grid first, THEN set new BCs and forces. This is because the update grid interpolation of the hodge variable
            // requires knowledge of the boundary conditions from that same timestep (the previous one, in our case)


            // Call the appropriate functions to setup the interfacial boundary conditions :
            interface_bc_velocity_u(); interface_bc_velocity_v();

            // Now setup the bc interface objects -- must be initialized with the neighbors and computed interfacial velocity of the moving solid front
//            BC_interface_value_velocity_u bc_interface_value_u(ngbd_np1,v_interface);
//            BC_interface_value_velocity_v bc_interface_value_v(ngbd_np1,v_interface);


            // Initialize the BC objects:
//            BoundaryConditions2D bc_velocity[P4EST_DIM];
//            BoundaryConditions2D bc_pressure;
            bc_velocity_u_interfacial.clear();
            bc_velocity_u_interfacial.create(ngbd_np1,v_interface.vec[0]);
            bc_velocity_v_interfacial.clear();
            bc_velocity_v_interfacial.create(ngbd_np1,v_interface.vec[1]);

            // Set the interfacial boundary conditions for velocity:
            bc_velocity[0].setInterfaceType(interface_bc_type_velocity_u);
            bc_velocity[1].setInterfaceType(interface_bc_type_velocity_v);

//            bc_velocity[0].setInterfaceValue(bc_interface_value_u);
//            bc_velocity[1].setInterfaceValue(bc_interface_value_v);
            bc_velocity[0].setInterfaceValue(bc_velocity_u_interfacial);
            bc_velocity[1].setInterfaceValue(bc_velocity_v_interfacial);

            // Set the wall boundary conditions for velocity:
            bc_velocity[0].setWallTypes(wall_bc_type_velocity_u); bc_velocity[1].setWallTypes(wall_bc_type_velocity_v);
            bc_velocity[0].setWallValues(wall_bc_value_velocity_u); bc_velocity[1].setWallValues(wall_bc_value_velocity_v);

            interface_bc_pressure();
            bc_pressure.setInterfaceType(interface_bc_type_pressure);
            bc_pressure.setInterfaceValue(interface_bc_value_pressure);
            bc_pressure.setWallTypes(wall_bc_type_pressure);
            bc_pressure.setWallValues(wall_bc_value_pressure);

            // Set the boundary conditions:
            if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting NS boundary conditions \n");
            ns->set_bc(bc_velocity,&bc_pressure);
            if(print_checkpoints) PetscPrintf(mpi.comm(),"Boundary conditions set. \n");


            // set_external_forces
            CF_DIM *external_forces[P4EST_DIM] = {&fx_ext_tn,&fy_ext_tn};
            if(example_ == NS_GIBOU_EXAMPLE || example_ == COUPLED_PROBLEM_EXAMPLE){
                if(print_checkpoints)PetscPrintf(mpi.comm(),"Sets external forces\n");
                ns->set_external_forces(external_forces);
              }

            if(mem_checkpoints){
                MPI_Barrier(mpi.comm());
                PetscMemoryGetCurrentUsage(&memNS2);
              }
            // -----------------------------
            // Get hodge and begin iterating on hodge error
            // -----------------------------
            hodge_old.create(p4est_np1,ghost_np1);
            hodge_new.create(p4est_np1,ghost_np1);

            bool keep_iterating_hodge = true;
            double hodge_tolerance;
            if (tstep<1) hodge_tolerance = u0*hodge_percentage_of_max_u;
            else hodge_tolerance = NS_norm*hodge_percentage_of_max_u;

            hodge_tolerance=1.e-4;
            PetscPrintf(mpi.comm(),"\n"
                                   "--> Hodge tolerance is %0.2e \n"
                                   "-----------\n",hodge_tolerance);

            int hodge_iteration = 0;

            if(mem_checkpoints){
                MPI_Barrier(mpi.comm());
                PetscMemoryGetCurrentUsage(&memNS3);
              }

            // Create interpolation object to interpolate phi to the quadrant location (since we are checking hodge error only in negative subdomain, we need to check value of LSF):
            my_p4est_interpolation_nodes_t *interp_phi = ns->get_interp_phi();


            face_solver = NULL;
            cell_solver = NULL;

            while(keep_iterating_hodge){


                double hodge_error = - 10.0;
                hodge_global_error = -10.0;

                // Grab the old hodge variable before we go through the solution process:  Note: Have to copy it , because the hodge vector itself will be changed by the navier stokes solver
                ns->copy_hodge(hodge_old.vec);

                // ------------------------------------
                // Do NS Solution process:
                // ------------------------------------
                // Viscosity step:
                if(mem_checkpoints){
                    MPI_Barrier(mpi.comm());
                    PetscMemoryGetCurrentUsage(&memNS_H1);
                  }

                ns->solve_viscosity(face_solver,(face_solver!=NULL),face_solver_type,pc_face);
//                delete face_solver;
//                ns->solve_viscosity();

                if(mem_checkpoints){
                    MPI_Barrier(mpi.comm());
                    PetscMemoryGetCurrentUsage(&memNS_H2);
                  }

                // Projection step:
                ns->solve_projection(cell_solver,(cell_solver!=NULL),cell_solver_type,pc_cell);

//                delete cell_solver;
//                ns->solve_projection();
                if(mem_checkpoints){
                    MPI_Barrier(mpi.comm());
                    PetscMemoryGetCurrentUsage(&memNS_H3);
                  }
                // -------------------------------------------------------------
                // Check the error on hodge:
                // -------------------------------------------------------------
                // Get the current hodge:
//                hodge_new.set(ns->get_hodge());
                ns->copy_hodge(hodge_new.vec);


                if(mem_checkpoints){
                    MPI_Barrier(mpi.comm());
                    PetscMemoryGetCurrentUsage(&memNS_H4);
                  }

                // Get hodge arrays:
                hodge_old.get_array();
                hodge_new.get_array();

                double xyz[P4EST_DIM];
                double phi_val;
                // Loop over each quadrant in each tree, check the error in hodge
                foreach_tree(tr,p4est_np1){
                  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_np1->trees,tr);

                  foreach_local_quad(q,tree){
                    // Get the global index of the quadrant:
                    p4est_locidx_t quad_idx = tree->quadrants_offset + q;

                    // Get xyz location of the quad center so we can interpolate phi there and check which domain we are in:
                    quad_xyz_fr_q(quad_idx,tr,p4est_np1,ghost_np1,xyz);

                    phi_val = (*interp_phi)(xyz[0],xyz[1]);
                    // Evaluate the hodge error:
                    if(phi_val < 0){
                        hodge_error = max(hodge_error,fabs(hodge_old.ptr[quad_idx] - hodge_new.ptr[quad_idx]));
                      }
                  }
                }

                // Restore hodge arrays:
                hodge_old.restore_array();
                hodge_new.restore_array();

                if(mem_checkpoints){
                    MPI_Barrier(mpi.comm());
                    PetscMemoryGetCurrentUsage(&memNS_H5);
                  }
                // Get the global hodge error:
                int mpi_err = MPI_Allreduce(&hodge_error,&hodge_global_error,1,MPI_DOUBLE,MPI_MAX,mpi.comm()); SC_CHECK_MPI(mpi_err);
                PetscPrintf(mpi.comm(),"Hodge iteration : %d, hodge error: %0.3e \n",hodge_iteration,hodge_global_error);

                hodge_iteration++;

                if((hodge_global_error < hodge_tolerance) || hodge_iteration>=hodge_max_it) keep_iterating_hodge = false;

              }
            PetscPrintf(mpi.comm(),"-----------\n");
            if(mem_checkpoints){
                MPI_Barrier(mpi.comm());
                PetscMemoryGetCurrentUsage(&memNS4);
              }

            // delete solvers:
            delete face_solver;
            delete cell_solver;
            // Compute velocity at the nodes
            ns->compute_velocity_at_nodes();

            // Set this timestep's "v_n" to be the "v_nm1" for the next timestep
            v_nm1.destroy(); v_nm1.create(p4est_np1,nodes_np1);
            foreach_dimension(d){
              VecCopyGhost(v_n.vec[d],v_nm1.vec[d]);
            }

            // Now set this step's "v_np1" to be "v_n" for the next timestep -- v_n for next step will be sampled at this grid for now, but will be interpolated onto new grid for next step in beginning of next step
            v_n.destroy(); v_n.create(p4est_np1,nodes_np1);
            ns->copy_velocity_np1(v_n.vec);

            // Compute the pressure
            ns->compute_pressure();

            // Destroy hodge vectors:
            hodge_old.destroy();
            hodge_new.destroy();

            // Get pressure at cells:
            press.destroy();
            press.create(p4est_np1,ghost_np1);
            ns->copy_pressure(press.vec);

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
            interp_c.clear();

            // Get pressure gradient:
            grad_p.destroy();
            grad_p.create(p4est_np1,nodes_np1);
            ngbd_np1->first_derivatives_central(press_nodes.vec,grad_p.vec);

            // Destroy pressure at cells now since no longer needed:
//            press.destroy();

            // Check pressure norms:
//            PetscPrintf(mpi.comm(),"Checking pressure norm info: \n ");
//            pressure_check_flag=true;
//            check_T_values(phi,press_nodes,nodes_np1,p4est_np1,example_,phi,false);
//            pressure_check_flag=false;

            // Get the computed values of vorticity
            vorticity.destroy();
            vorticity.create(p4est_np1,nodes_np1);
            ns->copy_vorticity(vorticity.vec);

            // Check the L2 norm of u to make sure nothing is blowing up
            NS_norm = ns->get_max_L2_norm_u();
            PetscPrintf(mpi.comm(),"Max NS velocity norm info: \n");
            PetscPrintf(mpi.comm()," - Computational value: %0.3e \n"
                                   " - Physical value: %0.3e [m/s] \n"
                                   " - Physical value: %0.3e [cm/s] \n \n",NS_norm,NS_norm/scaling,NS_norm/scaling*100.);
            if(ns->get_max_L2_norm_u()>100.0){
                std::cerr<<"The simulation blew up \n"<<std::endl;
                SC_ABORT("Navier Stokes velocity blew up \n");
              }

            // Clear the phi interpolate now that no longer needed:
            interp_phi->clear();

            // Get a more appropriate dt for next timestep to consider:
            ns->compute_adapted_dt(u0);
            dt_NS = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/(tstep==0? u0:NS_norm);
            PetscPrintf(mpi.comm(),"ELYCE COMPUTED DT : %0.4e \n",dt_NS);
            dt_NS = ns->get_dt();
            PetscPrintf(mpi.comm(),"NS COMPUTED DT : %0.4e \n",dt_NS);




            if(dt_NS>dt_max_allowed) dt_NS = dt_max_allowed;

//            // Clear BC interpolations:
//            bc_interface_value_u.clear();
//            bc_interface_value_v.clear();

            if(mem_checkpoints){
                MPI_Barrier(mpi.comm());
                PetscMemoryGetCurrentUsage(&memNS5);
              }

            if(mem_checkpoints){
                MPI_Barrier(mpi.comm());
                PetscMemoryGetCurrentUsage(&memNS6);
                if(memNS6>memNS1) {PetscPrintf(mpi.comm(),"NS MEM INCREASE : %0.6e \n",memNS6-memNS1); num_ns_increases++;
                    }
                if(memNS6<memNS1){
                    PetscPrintf(mpi.comm(),"NS MEM DECREASE : %0.6e \n",memNS1-memNS6); num_ns_decreases++;
                  }
//                if(nodes_increased){PetscPrintf(mpi.comm(),"NUM NODES CHANGED \n");}
                PetscPrintf(mpi.comm(),"NS1 : %0.6e \n",memNS1);
                PetscPrintf(mpi.comm(),"NS6 : %0.6e \n",memNS6);
                PetscPrintf(mpi.comm(),"There have been %d increases in NS mem usage \n \n ",num_ns_increases);
                PetscPrintf(mpi.comm(),"There have been %d decreases in NS mem usage \n \n ",num_ns_decreases);

                PetscPrintf(mpi.comm(),"There have been %d changes in grid size \n",num_node_increases);
              }




            if(example_ == NS_GIBOU_EXAMPLE){
                PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin + grid_res_iter,lmax + grid_res_iter);
                check_NS_validation_error(phi,v_n,press_nodes,p4est_np1,nodes_np1,ghost_np1,ngbd,dxyz_close_to_interface,name_NS_errors,fich_NS_errors,tstep);
              }

          } // End of "if solve navier stokes"

        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem11);
          }
        if(example_ == COUPLED_PROBLEM_EXAMPLE){
            PetscPrintf(mpi.comm(),"lmin = %d, lmax = %d \n",lmin + grid_res_iter,lmax + grid_res_iter);
            check_coupled_problem_error(phi,v_n,press_nodes,T_l_n,p4est_np1,nodes_np1,ngbd,dxyz_close_to_interface,name_coupled_errors,fich_coupled_errors,tstep);
          }
        if(example_ == ICE_AROUND_CYLINDER && tstep%save_every_iter ==0 ){
            check_ice_cylinder_v_and_radius(phi,p4est_np1,nodes_np1,dxyz_close_to_interface,name_ice_radius_info,fich_ice_radius_info);
          }
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
            PetscMemoryGetCurrentUsage(&mem12);
          }
        // --------------------------------------------------------------------------------------------------------------
        // Delete the old grid:
        // --------------------------------------------------------------------------------------------------------------
        // Delete the old grid and update with the new one:


        if(fabs(tn+dt - tfinal)<EPS && solve_navier_stokes){
            delete ns;
          }
        else{
            PetscLogDouble before_grid_delete, after_grid_delete;
            PetscMemoryGetCurrentUsage(&before_grid_delete);
            PetscPrintf(mpi.comm(),"Destroying old grid variables ... \n");
            p4est_destroy(p4est); ns->nullify_p4est_nm1();
            p4est_ghost_destroy(ghost);
            p4est_nodes_destroy(nodes);
            delete ngbd;
            delete hierarchy;

            p4est = p4est_np1;
            ghost = ghost_np1;
            nodes = nodes_np1;

            hierarchy = hierarchy_np1;
            ngbd = ngbd_np1;
          }


        // Get current memory usage and print out all memory usage checkpoints:
        if(mem_checkpoints){
            MPI_Barrier(mpi.comm());
//            int no_nodes = nodes->num_owned_indeps;
            p4est_gloidx_t no_nodes = 0;
            for (int i = 0; i<p4est->mpisize; i++){
              no_nodes += nodes->global_owned_indeps[i];
            }
            PetscPrintf(mpi.comm(),"Current grid has %d nodes \n", no_nodes);
            PetscMemoryGetCurrentUsage(&mem13);

            double mem_all[33] = {mem1, mem2, mem3, mem4, mem5,
                                  mem_grid1, mem_grid2, mem6, mem7, mem8,
                                  mem9,memP1,memP2,memP3,memP4,
                                  memP5,memP6,memP7,mem10,memNS1,
                                  memNS2,memNS3,memNS_H1,memNS_H2,memNS_H3,
                                  memNS_H4,memNS_H5,memNS4,memNS5,memNS6,
                                  mem11,mem12,mem13};

            ierr = PetscFOpen(mpi.comm(),name_mem,"a",&fich_mem); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(),fich_mem,"%g %g %d %d "
                                                    "%g %g %g %g %g %g %g %g %g %g "
                                                    "%g %g %g %g %g %g %g %g %g %g "
                                                    "%g %g %g %g %g %g %g %g %g %g "
                                                    "%g %g %g\n",tn,dt,tstep,no_nodes,
                                                    mem_all[0],mem_all[1],mem_all[2],mem_all[3],mem_all[4],mem_all[5],mem_all[6],mem_all[7],mem_all[8],mem_all[9],
                                                    mem_all[10],mem_all[11],mem_all[12],mem_all[13],mem_all[14],mem_all[15],mem_all[16],mem_all[17],mem_all[18],mem_all[19],
                                                    mem_all[20],mem_all[21],mem_all[22],mem_all[23],mem_all[24],mem_all[25],mem_all[26],mem_all[27],mem_all[28],mem_all[29],
                                                    mem_all[30],mem_all[31],mem_all[32]);CHKERRXX(ierr);
            ierr = PetscFClose(mpi.comm(),fich_mem); CHKERRXX(ierr);


            // Check the overall:
            PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"\nMemory information: \n");
          }
        MPI_Barrier(mpi.comm());

        PetscMemoryGetCurrentUsage(&mem_safety_check);

        PetscPrintf(mpi.comm(),"\n"
                               "Memory safety check:\n");
        PetscPrintf(mpi.comm()," - Current memory usage is : %0.5e GB \n"
                               " - Percent of safety limit: %0.2f % \n \n \n",mem_safety_check*mpi.size()*1.e-9,(mpi.size()*mem_safety_check)/(mem_safety_limit)*100.0);

//        PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"\nMemory information: \n");


        if(mem_safety_check>mem_safety_limit/mpi.size()){
            MPI_Barrier(mpi.comm());
            PetscPrintf(mpi.comm(),"We are encroaching upon the memory upper bound on this machine, calling MPI Abort...\n");
            MPI_Abort(mpi.comm(),1);
          }
      } // <-- End of for loop through time

    PetscPrintf(mpi.comm(),"Time loop exited \n");
    if(solve_stefan){
        T_l_n.destroy();
        T_s_n.destroy();
        if(advection_sl_order==2) T_l_nm1.destroy();
      }

    if(solve_navier_stokes){ v_n.destroy();v_nm1.destroy();vorticity.destroy();press_nodes.destroy();}
    else{ // TO-DO : Don't think this is being done 100% correctly... for example, in above code, should still need to destroy phi(?) maybe not ... look into this
        phi.destroy();

        // destroy the structures leftover (in non NS case)

        p4est_nodes_destroy(nodes);
        p4est_ghost_destroy(ghost);
        p4est_destroy      (p4est);
        my_p4est_brick_destroy(conn, &brick);
        delete hierarchy;
        delete ngbd;
        }

  }// end of loop through number of splits

  MPI_Barrier(mpi.comm());
  PetscLogDouble memfinal;
  PetscMemoryGetCurrentUsage(&memfinal);
  PetscPrintf(mpi.comm(),"Final memory usage is %.5e GB, which is %0.2f percent of max allowed \n",memfinal*1.e-9,(memfinal)/(mem_safety_limit)*100.0);
  w.stop(); w.read_duration();
  return 0;
}

