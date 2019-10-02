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


using namespace std;

// Examples to run:
int example_ = 2;  // 0 - Ice cube melting in water, 1 - Frank sphere, 2 - water solidifying around cooled cylinder

int method_ = 0; // 0 - Backward Euler, 1 - Crank Nicholson

bool elyce_laptop = true;
// ---------------------------------------
// Define geometry:
// ---------------------------------------
double xmin; double xmax;
double ymin; double ymax;

int nx, ny;
int px, py;
double box_size; // equivalent width (in x) in meters

// For frank sphere:
double s0;
double T_inf;

// For ice cube:
double r0;
double Twall;
double Tinterface;
double back_wall_flux;

// For solidifying ice problem:
double r_cyl;
double T_cyl;

void set_geometry(){
  switch(example_){
    case 0: // Ice cube melting
      xmin = 0.0; xmax = 1.0;
      ymin = 0.0; ymax = 1.0;

      nx = 1;
      ny = 1;

      px = 0;
      py = 1;

      box_size = 0.1;// Equivalent width [in meters]
      r0 = 0.10; // 0.25
      Twall = 298.0; Tinterface = 273.0;

      break;

    case 1: // Frank sphere
      xmin = -5.0; xmax = 5.0;
      ymin = -5.0; ymax = 5.0;
      box_size = 1.0;
      nx = 1;
      ny = 1;
      px = 0; py = 0;

      s0 = 1.65;
      T_inf = -0.5;
      Twall = -0.5;
      Tinterface = 0.0;
      break;

    case 2: // Ice layer growing around a constant temperature cooled cylinder
      xmin = 0.0; xmax = 1.0;
      ymin = 0.0; ymax = 1.0;

      nx = 1;
      ny = 1;

      px = 0;
      py = 1;

      box_size = 0.1;// Equivalent width [in meters]
      r0 = 0.15; // 0.25
      r_cyl = 0.10;
      Twall = 298.0; Tinterface = 273.0;
      T_cyl = 273.0 - 70.0;
      back_wall_flux = 0.0;
      // NOTE: TO DO: WILL NEED TO HAVE REFINEMENT CRITERIA AROUND BOTH INTERFACES(?) ... maybe not

      break;
    }
}
// ---------------------------------------
// Grid refinement:
// ---------------------------------------
int lmin = 4;
int lmax = 7;
// ---------------------------------------
// Time-stepping:
// ---------------------------------------
double tfinal;
double delta_t;
double dt_max_allowed;
bool keep_going = true;

double tn;
double dt;

void simulation_time_info(){
  switch(example_){
    case 0:
      tfinal = 3.6e3; // corresponds to 1 hour -- 3600 seconds
      delta_t = 1.e-1;
      dt_max_allowed = 1.e2;
      tn = 0.0;

      break;
    case 1:
      tfinal = 1.5;
      delta_t = 0.01;
      dt_max_allowed = 0.01;
      tn = 1.0;
      break;
    case 2:
      tfinal = 3.6e3;
      delta_t = 1.e-1;
      dt_max_allowed = 1.e1;
      tn = 0.0;

    }
}
// ---------------------------------------
// Physical properties:
// ---------------------------------------
double alpha_s;
double alpha_l;

void set_diffusivities(){
  switch(example_){
    case 0:
      alpha_s = (1.1820e-6); //ice - [m^2]/s
      alpha_l = (1.4547e-7); //water- [m^2]/s
      break;
    case 1:
      alpha_s = 1.0;
      alpha_l = 1.0;
      break;

    case 2:
      alpha_s = (1.1820e-6); //ice - [m^2]/s
      alpha_l = (1.4547e-7); //water- [m^2]/s
      break;
    }
}


double k_s;
double k_l;
double L; // Latent heat of fusion
double rho_l;

void set_conductivities(){
  switch(example_){
    case 0:
       k_s = 2.22; // W/[m*K]
       k_l = 0.608; // W/[m*K]
       L = 334.e3;  // J/kg
       rho_l = 1000.0; // kg/m^3
      break;
    case 1:
      k_s = 1.0;
      k_l = 1.0;
      L = 1.0;
      rho_l = 1.0;
      break;

    case 2:
      k_s = 2.22; // W/[m*K]
      k_l = 0.608; // W/[m*K]
      L = 334.e3;  // J/kg
      rho_l = 1000.0; // kg/m^3
      break;
    }
}
// ---------------------------------------
// Other parameters:
// ---------------------------------------
double cfl = 0.5;
double v_int_max_allowed = 250.0;

bool move_interface_with_v_external = false;

bool do_advection = true;

bool output_information = true;

bool check_temperature_values = true;

bool check_derivative_values = true;

bool check_interfacial_velocity = true;

bool save_temperature_derivative_fields = false;

bool solve_smoke = true;


// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
// Frank sphere functions
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
// LEVEL SET FUNCTION:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch (example_){
      case 0: return r0 - sqrt(SQR(x - (xmax/2.0)) + SQR(y - (ymax/2.0)));
      case 1: return 1.65 - sqrt(SQR(x) + SQR(y));
      case 2: return r0 - sqrt(SQR(x - (xmax/2.0)) + SQR(y - (ymax/2.0)));
      default: throw std::invalid_argument("You must choose an example type\n");
      }

  }
} level_set;

struct MINI_LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch(example_){
      case 0: throw std::invalid_argument("This option may not be used for the particular example being called");
      case 1: throw std::invalid_argument("This option may not be used for the particular example being called");
      case 2: return r_cyl - sqrt(SQR(x - (xmax/2.0)) + SQR(y - (ymax/2.0)));
      }
  }
} mini_level_set;

// --------------------------------------------------------------------------------------------------------------
// PRESCRIBED VELOCITY FIELD AT WHICH THE INTERFACE ADVANCES:
// --------------------------------------------------------------------------------------------------------------
struct u_advance : CF_DIM
{ double operator() (double x, double y) const{
  return 4.e-2;
  }

} u_adv;

struct v_advance: CF_DIM{
  double operator()(double x, double y) const
  {
    return 0.0;
  }
} v_adv;

// --------------------------------------------------------------------------------------------------------------
// SMOKE BC and IC's -- if passing smoke, a passive scalar
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
    if((x < xmax/10.0) && (y<(5./8.)*ymax ) && (y>(3./8.)*ymax)){
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
// INTERFACIAL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
BoundaryConditionType interface_bc_type_temp;
void interface_bc(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case 0:
      interface_bc_type_temp = DIRICHLET;
      break;
    case 1:
      interface_bc_type_temp = DIRICHLET;
      break;
    case 2:
      interface_bc_type_temp = DIRICHLET;

    }
}

BoundaryConditionType inner_interface_bc_type_temp;
void inner_interface_bc(){ //-- Call this function before setting interface bc in solver to get the interface bc type depending on the example
  switch(example_){
    case 0: throw std::invalid_argument("This option may not be used for the particular example being called");
    case 1: throw std::invalid_argument("This option may not be used for the particular example being called");
    case 2:
      inner_interface_bc_type_temp = DIRICHLET;
      break;

    }
}

class BC_interface_value: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    return Tinterface;
  }
}bc_interface_val;

class BC_interface_coeff: public CF_DIM{
public:
  double operator()(double x, double y) const
  { switch(example_){
      case 0: return 1.0;
      case 1: return 1.0; // maybe this should be 0?
      case 2: return 1.0;
      }
  }
}bc_interface_coeff;

class BC_interface_value_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  {
    return T_cyl;
  }
}bc_interface_val_inner;

class BC_interface_coeff_inner: public CF_DIM{
public:
  double operator()(double x, double y) const
  { return 1.0;
  }
}bc_interface_coeff_inner;
// -----
// Wall functions
// -----
struct XLOWER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (fabs(x - xmin) < EPS);
  }
} xlower_wall;

struct XUPPER_WALL : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (fabs(x - xmax) < EPS);
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


// --------------------------------------------------------------------------------------------------------------
// WALL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
class WALL_BC_TYPE_TEMP: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
    switch(example_){
      case 0: return DIRICHLET; // ice cube melting
      case 1: return DIRICHLET; // frank sphere
      case 2: // water solidifying around a cylinder
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            return DIRICHLET;
          }
        else if (yupper_wall(DIM(x,y,z))){
            return DIRICHLET;
          }
        //return DIRICHLET; // water solidifying around a cylinder
      }
  }
} wall_bc_type_temp;

class WALL_BC_VALUE_TEMP: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(example_){
      case 0:
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z))){
            if (level_set(DIM(x,y,z)) < EPS){
                return Twall;
              }
            else{
                return Tinterface;
                }
          }
        break;
      case 1:
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z)) || yupper_wall(DIM(x,y,z))){
            if (level_set(DIM(x,y,z)) < EPS){
                return Twall;
              }
            else{
                return Tinterface;
                }
          }
        break;
      case 2:
        if (xlower_wall(DIM(x,y,z)) || xupper_wall(DIM(x,y,z)) || ylower_wall(DIM(x,y,z))){
            if (level_set(DIM(x,y,z)) < EPS){
                return Twall;
              }
            else{
                return Tinterface;
                }
          }
        else if(yupper_wall(DIM(x,y,z))){ // Neumann condition on back wall
            return Twall;
          }
        break;
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
    if (level_set(DIM(x,y,z)) > EPS){

        switch(example_){
          case 0:
            //m = (Twall - Tinterface)/(level_set(DIM(xmin,ymin,z)));
            return Tinterface; //+ m*level_set(DIM(x,y,z));
          case 1:
            r = sqrt(SQR(x) + SQR(y));
            sval = s(r,tn);
            return frank_sphere_solution_t(sval); //CALCULATE WHAT IT SHOULD BE?
          case 2:
            return Tinterface;

          }
      }
    else{
        switch(example_){
          case 0:
            m = (Twall - Tinterface)/(level_set(DIM(xmin,ymin,z)));
            return Tinterface + m*level_set(DIM(x,y,z));
          case 1:
            r = sqrt(SQR(x) + SQR(y));
            sval = s(r,tn);
            return frank_sphere_solution_t(sval);
          case 2:
            m = (Twall - Tinterface)/(level_set(DIM(xmin,ymin,z)));
            return Tinterface + m*level_set(DIM(x,y,z));
          }
      }
  }
}IC_temp;


class INITIAL_CONDITION_TEMP_LIQUID: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    switch(example_){
      case 0:
        return Twall;

      case 1:
        double r = sqrt(SQR(x) + SQR(y));
        double sval = s(r,tn);
        return frank_sphere_solution_t(sval);
      }
  }
}IC_temp_liquid;

class INITIAL_CONDITION_TEMP_SOLID: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    double r;
    double sval;
    switch(example_){
      case 0:
        return Tinterface;

      case 1:
        r = sqrt(SQR(x) + SQR(y));
         sval = s(r,tn);
        return frank_sphere_solution_t(sval);
      case 2:
        return Tinterface;
      }
  }
}IC_temp_solid;

// --------------------------------------------------------------------------------------------------------------
// Function for checking the temperature values during the solution process
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
  if(example_ ==2 && phi_cyl.ptr == NULL){
      throw std::invalid_argument("You must provide a phi_cylinder vector to run example 2 \n");
    }

  // Loop over each node, check if node is in the subdomain we are considering. If so, compute average,max, and min values for the domain
  foreach_local_node(n,nodes){

    in_domain = false;
    // Check if the node is in the domain we are checking:
    if (example_ ==2 ){
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
  if(example_ ==2) phi_cyl.restore_array();

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

  double L1_Tl = 0.0;
  double L1_Ts = 0.0;
  double L1_phi = 0.0;
  double L1_v_int = 0.0;

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
        Linf_Ts = max(Linf_Tl,T_s_error);

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

void setup_rhs(vec_and_ptr_t T_l, vec_and_ptr_t T_s, vec_and_ptr_t smoke, vec_and_ptr_t rhs_Tl, vec_and_ptr_t rhs_Ts, vec_and_ptr_t rhs_smoke, vec_and_ptr_t T_l_backtrace, vec_and_ptr_t smoke_backtrace, p4est_t* p4est, p4est_nodes_t* nodes){

  // Get rhs arrays:
  rhs_Tl.get_array(); rhs_Ts.get_array();

  // Get current step arrays:
  T_s.get_array();


  if (do_advection) T_l_backtrace.get_array(); // Get backtrace arrays if doing advection
  else T_l.get_array();

  // Get relevant arrays for solving smoke:
   if(solve_smoke){
       rhs_smoke.get_array();
       smoke.get_array();

       if(do_advection) smoke_backtrace.get_array();
     }

  // Loop through the nodes to build the RHS:
  foreach_node(n,nodes){
    switch(method_){
      case 0: // Backward Euler
        if (do_advection) rhs_Tl.ptr[n]=T_l_backtrace.ptr[n]/dt;
        else rhs_Tl.ptr[n] = T_l.ptr[n]/dt;

        rhs_Ts.ptr[n] = T_s.ptr[n]/dt;

        break;
      case 1: // Crank Nicholson
        throw std::invalid_argument("Crank Nicholson is not yet fully implemented \n");
      }

    if (solve_smoke){
        switch(method_){
          case 0: // Backward Euler
            if (do_advection) rhs_smoke.ptr[n] = smoke_backtrace.ptr[n]/dt;
            else rhs_smoke.ptr[n] = smoke.ptr[n];
            break;

          case 1: // Crank Nicholson
            throw std::invalid_argument("Crank Nicholson is not yet fully implemented \n");
          }
      }
  }

  // Restore arrays:
  rhs_Tl.restore_array(); rhs_Ts.restore_array();

  if (do_advection) T_l_backtrace.restore_array();
  else T_l.restore_array();

  T_s.restore_array();

  if (solve_smoke){
      rhs_smoke.restore_array();
      if(do_advection) smoke_backtrace.restore_array();
      else smoke.restore_array();
    }
}

void do_backtrace(vec_and_ptr_t T_l,vec_and_ptr_t T_l_backtrace,vec_and_ptr_dim_t v, vec_and_ptr_t smoke, vec_and_ptr_t smoke_backtrace, p4est_t* p4est, p4est_nodes_t* nodes,my_p4est_node_neighbors_t* ngbd, interpolation_method interp_method){

  // Get second derivatives of the velocity field:
  vec_and_ptr_dim_t v_dd[P4EST_DIM];

  foreach_dimension(d){
    v_dd[d].create(p4est,nodes); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
    ngbd->second_derivatives_central(v.vec[d],v_dd[d].vec);
  }

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];

  // Do the Semi-Lagrangian backtrace:
  trajectory_from_np1_to_n(p4est,nodes,ngbd,dt,v.vec,&v_dd->vec,xyz_d);

  // Add the back-trace points to the interpolation object:
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd);
  foreach_local_node(n,nodes){
    double xyz_temp[P4EST_DIM];
    foreach_dimension(d){
      xyz_temp[d] = xyz_d[d][n];
    }
    SL_backtrace_interp.add_point(n,xyz_temp);
  }

  // Interpolate the Temperature data to back-traced points:
  SL_backtrace_interp.set_input(T_l.vec,interp_method);
  SL_backtrace_interp.interpolate(T_l_backtrace.vec);

  // Interpolate the smoke data to back-traced points (if applicable):
  if(solve_smoke){
      // Interpolate smoke data to back-traced points:
      SL_backtrace_interp.set_input(smoke.vec,interp_method);
      SL_backtrace_interp.interpolate(smoke_backtrace.vec);
    }

  // Destroy velocity derivatives now that not needed:
  v_dd->destroy();
}

void interpolate_values_onto_new_grid(vec_and_ptr_t T_l, vec_and_ptr_t T_l_new,
                                      vec_and_ptr_t T_s, vec_and_ptr_t T_s_new,
                                      vec_and_ptr_dim_t v_interface,vec_and_ptr_dim_t v_interface_new,
                                      vec_and_ptr_dim_t v_external,vec_and_ptr_dim_t v_external_new,
                                      vec_and_ptr_t smoke, vec_and_ptr_t smoke_new,
                                      p4est_nodes_t *nodes_new_grid, p4est_t *p4est_new,
                                      my_p4est_node_neighbors_t *ngbd_old_grid,interpolation_method interp_method){
  // Need neighbors of old grid to create interpolation object
  // Need nodes of new grid to get the points that we must interpolate to

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_old_grid);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n,nodes_new_grid){
    node_xyz_fr_n(n,p4est_new,nodes_new_grid,xyz);
    interp_nodes.add_point(n,xyz);
  }
  // Interpolate temperature fields:
  interp_nodes.set_input(T_l.vec,interp_method); interp_nodes.interpolate(T_l_new.vec);
  interp_nodes.set_input(T_s.vec,interp_method); interp_nodes.interpolate(T_s_new.vec);

  // Interpolate velocity fields:
  foreach_dimension(d){
    interp_nodes.set_input(v_interface.vec[d],interp_method); interp_nodes.interpolate(v_interface_new.vec[d]);
    interp_nodes.set_input(v_external.vec[d],interp_method); interp_nodes.interpolate(v_external_new.vec[d]);
  }

  // Interpolate smoke (if applicable):
  if(solve_smoke){
      interp_nodes.set_input(smoke.vec,interp_method); interp_nodes.interpolate(smoke_new.vec);
    }
}


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
  double global_max_v_norm = 0.0;
  int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_v_norm,1,MPI_DOUBLE,MPI_MAX,p4est->mpicomm);
  SC_CHECK_MPI(mpi_ret);
  PetscPrintf(p4est->mpicomm,"\n \n \n Computed interfacial velocity and timestep: \n {");
  PetscPrintf(p4est->mpicomm,"\n Max v norm: %0.2e \n", global_max_v_norm);

  // Compute new timestep:
  dt = cfl*min(dxyz_smallest[0],dxyz_smallest[1])/min(global_max_v_norm,1.0);
  dt = min(dt,dt_max_allowed);

  // Report computed timestep and minimum grid size:
  PetscPrintf(p4est->mpicomm,"Computed timestep: %0.3e \n",dt);
  PetscPrintf(p4est->mpicomm,"dxyz close to interface : %0.3e \n } \n \n  ",dxyz_close_to_interface);


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
  set_diffusivities();
  set_conductivities();

  interface_bc();

  // -----------------------------------------------
  // Scale the problem appropriately:
  // -----------------------------------------------
  double scaling = 1.0/box_size;
  rho_l/=(scaling*scaling*scaling);
  k_s/=scaling;
  k_l/=scaling;
  L/=(scaling*scaling*scaling);

  alpha_l*=(scaling*scaling);
  alpha_s*=(scaling*scaling);

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
  splitting_criteria_cf_t sp(lmin, lmax, &level_set,1.75);

  p4est->user_pointer = &sp;                                    // save the pointer to the forst splitting criteria
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL); // refine the grid according to the splitting criteria


  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);                  // partition the forest but allow for coarsening --> Daniil does not allow (use P4EST_FALSE)

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
  // Initialize the Level Set function:
  // -----------------------------------------------
  // LSF:
  vec_and_ptr_t phi;
  phi.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);

  // LSF for solid domain: -- This will be assigned within the loop as the negative of phi
  vec_and_ptr_t phi_solid;

  // LSF for the inner cylinder, if applicable (example 2):
  vec_and_ptr_t phi_cylinder;
  if(example_ == 2){
      phi_cylinder.create(phi.vec);
      sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);
    }

  // 2nd derivatives of LSF's
  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_cylinder_dd;

  // -----------------------------------------------
  // Initialize the Velocity field:
  // -----------------------------------------------
  vec_and_ptr_dim_t vel_n(p4est,nodes);
  vec_and_ptr_dim_t vel_new;

  const CF_DIM *vel_cf[P4EST_DIM] = {&u_adv, &v_adv};

  if (move_interface_with_v_external || do_advection){
    for (int dir = 0; dir<P4EST_DIM;dir++){
        sample_cf_on_nodes(p4est,nodes,*vel_cf[dir],vel_n.vec[dir]);
      }
    }

  vec_and_ptr_dim_t v_interface(p4est,nodes);
  vec_and_ptr_dim_t v_interface_new;
  for (int dir = 0; dir<P4EST_DIM;dir++){
      sample_cf_on_nodes(p4est,nodes,zero_cf,v_interface.vec[dir]);
    }

  // -----------------------------------------------
  // Initialize the Temperature field:
  // -----------------------------------------------
  // Vectors for T_liquid:
  vec_and_ptr_t T_l_n;
  T_l_n.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,IC_temp,T_l_n.vec); // Sample this just so that we can save the initial temperature distribution
  vec_and_ptr_t rhs_Tl;

  // Vector for advection of temperature:
  vec_and_ptr_t T_l_backtrace;

  // Vectors for T_solid:
  vec_and_ptr_t T_s_n;
  T_s_n.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,IC_temp_solid,T_s_n.vec); // Sample this just so that we can save the initial temperature distribution
  vec_and_ptr_t rhs_Ts;

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
  vec_and_ptr_t rhs_smoke;

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
  // Initialize a file to output the error data (for Frank Sphere case):
  // -----------------------------------------------
  FILE *fich;
  char name[1000];
  if (example_ == 1){
    sprintf(name,"/home/elyce/workspace/projects/multialloy_with_fluids/Frank_Sphere_error_lmin_%d_lmax_%d_method_%d.dat",lmin,lmax,method_);

    ierr = PetscFOpen(mpi.comm(),name,"w",&fich); CHKERRXX(ierr);

    ierr = PetscFPrintf(mpi.comm(),fich,"time " "timestep " "iteration " "phi_error " "T_l_error " "T_s_error " "v_int_error " "number_of_nodes" "min_grid_size \n");CHKERRXX(ierr);
    ierr = PetscFClose(mpi.comm(),fich); CHKERRXX(ierr);
    }

  // -----------------------------------------------
  // Initialize the needed solvers
  // -----------------------------------------------
  my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

  my_p4est_poisson_nodes_mls_t *solver_smoke; // will solve for smoke over whole domain if being used

  // Initialize interpolation objects:
  my_p4est_interpolation_nodes_t  *interp_nodes_l;
  my_p4est_interpolation_nodes_t  *interp_nodes_s;
  my_p4est_interpolation_nodes_t *interp_nodes_vint;
  // -----------------------------------------------
  // Initialize variables for keeping track of interface velocity
  //------------------------------------------------
  double global_max_v_norm = 0.0;
  double max_v_norm = 0.0;
  // -----------------------------------------------
  // Begin stepping through time
  // -----------------------------------------------
  int tstep = 0;
  int save = 1;
  dt = delta_t;

  for (tn;tn<tfinal; tn+=dt, tstep++){
      if (!keep_going) break;
      //if(tstep>1) break; // TIMESTEP BREAK

      // --------------------------------------------------------------------------------------------------------------
      // Print iteration information:
      // --------------------------------------------------------------------------------------------------------------

      PetscPrintf(mpi.comm(),"\n -------------------------------------------\n");
      ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3f \n ------------------------------------------- \n",tstep,tn);
      ierr = PetscPrintf(mpi.comm(),"\n Previous interfacial velocity (max norm) is %0.2f \n",global_max_v_norm);

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

      // Perturb the LSF on the first iteration
      my_p4est_level_set_t ls(ngbd);

      // If first iteration, perturb the LSF:
      if(tstep<1){
          ls.perturb_level_set_function(phi.vec,EPS);
          if(example_ ==2 ) ls.perturb_level_set_function(phi_cylinder.vec,EPS);
        }
      // --------------------------------------------------------------------------------------------------------------
      // Extend Fields Across Interface:
      // --------------------------------------------------------------------------------------------------------------
      // Define LSF for the solid domain (as just the negative of the liquid one):
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

      // Extend Temperature Fields across the interface:
      ls.extend_Over_Interface_TVD_Full(phi.vec, T_l_n.vec, 50, 1, 1.e-9, extension_band_use_, extension_band_extend_, extension_band_check_, liquid_normals.vec, NULL, NULL, false, NULL, NULL);

      ls.extend_Over_Interface_TVD_Full(phi_solid.vec, T_s_n.vec, 50, 1, 1.e-9, extension_band_use_, extension_band_extend_, extension_band_check_, solid_normals.vec, NULL, NULL, false, NULL, NULL);

      // Extend the fluid velocity and fluid pressure across the interface:
      // [ INSERT NS STUFF HERE]


      // For the case where we have a second interface:
      if(example_ == 2){
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
        }

      // --------------------------------------------------------------------------------------------------------------
      // Get derivatives of the temperature fields for saving to visualize and for computing the jump up ahead:
      // --------------------------------------------------------------------------------------------------------------
      T_l_d.create(p4est,nodes); T_s_d.create(T_l_d.vec);
      ngbd->first_derivatives_central(T_l_n.vec,T_l_d.vec);
      ngbd->first_derivatives_central(T_s_n.vec,T_s_d.vec);

      if (check_derivative_values){
          // Check Temperature derivative values:
          PetscPrintf(mpi.comm(),"\n Checking temperature derivative values after field extension: \n [ ");
          PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
          check_T_d_values(phi,T_l_d,nodes,p4est,0);
          PetscPrintf(mpi.comm(),"\nIn solid domain: ");
          check_T_d_values(phi_solid,T_s_d,nodes,p4est,0);
          PetscPrintf(mpi.comm()," ] \n");
        }

      // --------------------------------------------------------------------------------------------------------------
      // SAVING DATA: Save data every specified amout of timesteps: -- Do this after values are extended across interface to make visualization nicer
      // --------------------------------------------------------------------------------------------------------------

      if (tstep % save ==0){
          out_idx++;
          if(elyce_laptop) sprintf(outdir,"//Users/elyce/workspace/projects/multialloy_with_fluids/output/snapshot_%d",out_idx);
          else sprintf(outdir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_%d",out_idx);
          // -----------------------------------------------
          // Get local array to write initial LSF and velocity fields to vtk:
          // -----------------------------------------------

          phi.get_array();
          T_l_n.get_array();
          T_s_n.get_array();

          // Scale the velocity back to what the physical problem would be:
          foreach_dimension(d){
            VecScaleGhost(v_interface.vec[d],1./scaling);
          }
          v_interface.get_array();

          if(save_temperature_derivative_fields){
              T_l_d.get_array(); T_s_d.get_array();


              if(example_ ==2){
                  PetscPrintf(mpi.comm(),"Gets into where we should print phi_cyl \n");
                  phi_cylinder.get_array();
                  // Write out the data:
                  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                    10,0,outdir,
                                                    VTK_POINT_DATA,"phi",phi.ptr,
                                                    VTK_POINT_DATA,"phi_cyl",phi_cylinder.ptr,
                                                    VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                    VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                    VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                    VTK_POINT_DATA,"Ts",T_s_n.ptr,
                                                    VTK_POINT_DATA,"dTl_dx",T_l_d.ptr[0],
                                                    VTK_POINT_DATA,"dTl_dy",T_l_d.ptr[1],
                                                    VTK_POINT_DATA,"dTs_dx",T_s_d.ptr[0],
                                                    VTK_POINT_DATA,"dTs_dy",T_s_d.ptr[1]);
                  phi_cylinder.restore_array();
                }
              else{
                  // Write out the data:
                  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                    9,0,outdir,
                                                    VTK_POINT_DATA,"phi",phi.ptr,
                                                    VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                    VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                    VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                    VTK_POINT_DATA,"Ts",T_s_n.ptr,
                                                    VTK_POINT_DATA,"dTl_dx",T_l_d.ptr[0],
                                                    VTK_POINT_DATA,"dTl_dy",T_l_d.ptr[1],
                                                    VTK_POINT_DATA,"dTs_dx",T_s_d.ptr[0],
                                                    VTK_POINT_DATA,"dTs_dy",T_s_d.ptr[1]);
                }
              // Write out the data:
              my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                9,0,outdir,
                                                VTK_POINT_DATA,"phi",phi.ptr,
                                                VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                VTK_POINT_DATA,"Ts",T_s_n.ptr,
                                                VTK_POINT_DATA,"dTl_dx",T_l_d.ptr[0],
                                                VTK_POINT_DATA,"dTl_dy",T_l_d.ptr[1],
                                                VTK_POINT_DATA,"dTs_dx",T_s_d.ptr[0],
                                                VTK_POINT_DATA,"dTs_dy",T_s_d.ptr[1]);

              T_l_d.restore_array(); T_s_d.restore_array();
            }
          else{
              if (example_ ==2 && solve_smoke){
                  phi_cylinder.get_array();
                  smoke.get_array();
                  // Write out the data:
                  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                    7,0,outdir,
                                                    VTK_POINT_DATA,"phi",phi.ptr,
                                                    VTK_POINT_DATA,"phi_cyl",phi_cylinder.ptr,
                                                    VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                    VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                    VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                    VTK_POINT_DATA,"Ts",T_s_n.ptr,
                                                    VTK_POINT_DATA,"smoke",smoke.ptr);
                  phi_cylinder.restore_array();
                  smoke.restore_array();
                }
              else if( example_ ==2 && !solve_smoke){
                  phi_cylinder.get_array();
                  // Write out the data:
                  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                    6,0,outdir,
                                                    VTK_POINT_DATA,"phi",phi.ptr,
                                                    VTK_POINT_DATA,"phi_cyl",phi_cylinder.ptr,
                                                    VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                    VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                    VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                    VTK_POINT_DATA,"Ts",T_s_n.ptr);
                  phi_cylinder.restore_array();

                }
              else if(example_ !=2 && solve_smoke){
                  smoke.get_array();
                  // Write out the data:
                  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                    6,0,outdir,
                                                    VTK_POINT_DATA,"phi",phi.ptr,
                                                    VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                    VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                    VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                    VTK_POINT_DATA,"Ts",T_s_n.ptr,
                                                    VTK_POINT_DATA,"smoke",smoke.ptr);
                  smoke.restore_array();
                }
              else{
                  // Write out the data:
                  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                    5,0,outdir,
                                                    VTK_POINT_DATA,"phi",phi.ptr,
                                                    VTK_POINT_DATA,"vx",v_interface.ptr[0],
                                                    VTK_POINT_DATA,"vy",v_interface.ptr[1],
                                                    VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                                    VTK_POINT_DATA,"Ts",T_s_n.ptr);
                }
            }

          phi.restore_array();
          T_l_n.restore_array();
          T_s_n.restore_array();
          v_interface.restore_array();

          // Scale the velocity back to computational problem values:
          foreach_dimension(d){
            VecScaleGhost(v_interface.vec[d],scaling);
          }
        }

      // Enforce that the interfacial velocity is within a reasonable range specified by the user:
      P4EST_ASSERT(global_max_v_norm<v_int_max_allowed);

      // --------------------------------------------------------------------------------------------------------------
      // Compute the jump in flux across the interface to use to advance the LSF:
      // --------------------------------------------------------------------------------------------------------------
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

      // --------------------------------------------------------------------------------------------------------------
      // Compute the timestep -- determined by velocity at the interface:
      // --------------------------------------------------------------------------------------------------------------
      compute_timestep(v_interface, phi, dxyz_close_to_interface, dxyz_smallest,nodes,p4est);

      // --------------------------------------------------------------------------------------------------------------
      // Advance the LSF:
      // --------------------------------------------------------------------------------------------------------------
      // Make a copy of the grid objects for the next timestep:
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Create the semi-lagrangian object and do the advection:
      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd); // is this really the correct way to do this?

      // Advect the LSF and update the grid under the v_interface field:
      example_ == 2 ? // for example 2, refine around both LSFs. Otherwise, refine around just the one
            sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec,phi_cylinder.vec):
            sl.update_p4est(v_interface.vec,dt,phi.vec,phi_dd.vec);

      // --------------------------------------------------------------------------------------------------------------
      // Interpolate Values onto New Grid:
      // -------------------------------------------------------------------------------------------------------------
      // Create vectors to hold new values:
      T_l_new.create(p4est_np1,nodes_np1);
      T_s_new.create(T_l_new.vec);

      v_interface_new.create(p4est_np1,nodes_np1);
      vel_new.create(v_interface_new.vec);

      vec_and_ptr_t smoke_new;
      if (solve_smoke){
          smoke_new.create(T_l_new.vec);
        }

      // Interpolate things to the new grid:
      interpolate_values_onto_new_grid(T_l_n,T_l_new,
                                       T_s_n, T_s_new,
                                       v_interface, v_interface_new,
                                       vel_n, vel_new,
                                       smoke, smoke_new,
                                       nodes_np1, p4est_np1,
                                       ngbd, interp_bw_grids);


      // Copy new data over:
      // Transfer new values to the original objects:
      T_l_n.destroy(); T_s_n.destroy();

      T_l_n.create(p4est_np1,nodes_np1); T_s_n.create(T_l_n.vec);

      v_interface.destroy(); vel_n.destroy();
      v_interface.create(p4est_np1,nodes_np1); vel_n.create(v_interface.vec);

      if(solve_smoke) {smoke.destroy(); smoke.create(T_l_n.vec);}

      VecCopyGhost(T_l_new.vec,T_l_n.vec);
      VecCopyGhost(T_s_new.vec,T_s_n.vec);
      if (solve_smoke) VecCopyGhost(smoke_new.vec,smoke.vec);

      foreach_dimension(d){
        VecCopyGhost(v_interface_new.vec[d],v_interface.vec[d]);
        VecCopyGhost(vel_new.vec[d],vel_n.vec[d]);
      }

      // Delete the "new value" objects until the next timestep:
      T_l_new.destroy(); T_s_new.destroy();
      v_interface_new.destroy(); vel_new.destroy();

      // Check values after interpolation:
      if (check_temperature_values){
        // Check Temperature values:
        PetscPrintf(mpi.comm(),"\n Checking temperature values after interpolating onto new grid: \n [ ");
        PetscPrintf(mpi.comm(),"\nIn fluid domain: ");
        check_T_values(phi,T_l_n,nodes,p4est,example_,phi_cylinder);
        PetscPrintf(mpi.comm(),"\nIn solid domain: ");
        check_T_values(phi_solid,T_s_n,nodes,p4est,example_,phi_cylinder);
        PetscPrintf(mpi.comm()," ] \n");
        }
      // --------------------------------------------------------------------------------------------------------------
      // Delete the old grid and reinitialize phi:
      // --------------------------------------------------------------------------------------------------------------
      // Delete the old grid and update with the new one:
      p4est_destroy(p4est); p4est = p4est_np1;
      p4est_ghost_destroy(ghost); ghost = ghost_np1;
      p4est_nodes_destroy(nodes); nodes = nodes_np1;

      delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est,ghost,&brick);
      delete ngbd; ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
      ngbd->init_neighbors();

      // Create level set object and reinitialize it:
      my_p4est_level_set_t ls_new(ngbd);
      ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec, 100);

      // Get the new solid LSF:
      phi_solid.destroy();
      phi_solid.create(p4est,nodes);
      VecScaleGhost(phi.vec,-1.0);
      VecCopyGhost(phi.vec,phi_solid.vec);
      VecScaleGhost(phi.vec,-1.0);

      // --------------------------------------------------------------------------------------------------------------
      // Navier-Stokes Problem: Setup and solve a NS problem in the liquid subdomain
      // --------------------------------------------------------------------------------------------------------------




      // --------------------------------------------------------------------------------------------------------------
      // Compute the normal and curvature of the interface -- curvature is used in some of the interfacial boundary condition(s)
      // --------------------------------------------------------------------------------------------------------------
      if (example_ !=1){// compute this stuff, otherwise, don't bother
        }

      double deriv_norm;
      vec_and_ptr_dim_t normal;
      vec_and_ptr_t curvature_tmp; // This one will hold computed curvature
      vec_and_ptr_t curvature;  // This one will hold curvature extended from interface to whole domain

      normal.create(p4est,nodes);
      curvature_tmp.create(p4est,nodes);
      curvature.create(curvature_tmp.vec);

      // Get derivatives of phi:
      vec_and_ptr_dim_t phi_d;
      phi_d.create(p4est,nodes);
      ngbd->first_derivatives_central(phi.vec,phi_d.vec);

      // Get arrays needed:
      normal.get_array();
      curvature_tmp.get_array();
      phi_d.get_array();

      // First, compute the normal on the layer nodes:
      for(size_t i = 0; i<ngbd->get_layer_size(); i++){
          p4est_locidx_t n = ngbd->get_layer_node(i);

          deriv_norm = sqrt(SQR(phi_d.ptr[0][n]) + SQR(phi_d.ptr[1][n]));

          foreach_dimension(d){
            if (deriv_norm<EPS)deriv_norm = EPS;

            normal.ptr[d][n] = phi_d.ptr[d][n]/deriv_norm;;
          }
        }

      // Begin update:
      foreach_dimension(d){
        VecGhostUpdateBegin(normal.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }

      // Compute the normal on the local nodes:
      for(size_t i = 0; i<ngbd->get_local_size(); i++){
          p4est_locidx_t n = ngbd->get_local_node(i);
          deriv_norm = sqrt(SQR(phi_d.ptr[0][n]) + SQR(phi_d.ptr[1][n]));

          foreach_dimension(d){
            if (deriv_norm<EPS)deriv_norm = EPS;

            normal.ptr[d][n] = phi_d.ptr[d][n]/deriv_norm;;
          }

        }
      PetscPrintf(mpi.comm(),"Computed normal on local nodes \n");


      // End ghost update:
      foreach_dimension(d){
        VecGhostUpdateEnd(normal.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
      PetscPrintf(mpi.comm(),"Finished ghost update \n");

      // Define the qnnn object to help compute the derivatives of the normal:
      quad_neighbor_nodes_of_node_t qnnn;

      // Compute curvature on layer nodes:
      for(size_t i = 0; i<ngbd->get_layer_size(); i++){
          p4est_locidx_t n = ngbd->get_layer_node(i);
          ngbd->get_neighbors(n,qnnn);
          curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]);
        }
      PetscPrintf(mpi.comm(),"Computed curvature on layer nodes \n");


      // Begin ghost update:
      VecGhostUpdateBegin(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);
      PetscPrintf(mpi.comm(),"Begin curvature ghost update \n");


      // Compute curvature on local nodes:
      for(size_t i = 0; i<ngbd->get_local_size(); i++){
          p4est_locidx_t n = ngbd->get_local_node(i);
          ngbd->get_neighbors(n,qnnn);
          curvature_tmp.ptr[n] = qnnn.dx_central(normal.ptr[0]) + qnnn.dy_central(normal.ptr[1]);
        }

      PetscPrintf(mpi.comm(),"Computed curvature on local nodes \n");

      // End ghost update:
      VecGhostUpdateEnd(curvature_tmp.vec,INSERT_VALUES,SCATTER_FORWARD);

      PetscPrintf(mpi.comm(),"Finished curvature ghost update \n");

      // Restore arrays needed:
      curvature_tmp.restore_array();
      normal.restore_array();
      phi_d.restore_array();


      // Now go ahead and extend the curvature values to the whole domain
      //ls.extend_from_interface_to_whole_domain_TVD(phi.vec,curvature_tmp.vec,curvature.vec,20);
      ls.extend_from_interface_to_whole_domain_TVD(phi.vec,curvature_tmp.vec,curvature.vec);

      // --------------------------------------------------------------------------------------------------------------
      // Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
      // --------------------------------------------------------------------------------------------------------------
      // Get most updated derivatives of the LSF's (on current grid) -- Solver uses these:
      // ------------------------------------------------------------
      phi_solid_dd.destroy();
      phi_solid_dd.create(p4est,nodes);
      ngbd->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);

      phi_dd.destroy();
      phi_dd.create(p4est,nodes);
      ngbd->second_derivatives_central(phi_solid.vec,phi_solid_dd.vec);


      if(example_ ==2){
          phi_cylinder.destroy();
          phi_cylinder.create(p4est,nodes);
          sample_cf_on_nodes(p4est,nodes,mini_level_set,phi_cylinder.vec);

          phi_cylinder_dd.destroy();
          phi_cylinder_dd.create(p4est,nodes);
          ngbd->second_derivatives_central(phi_cylinder.vec,phi_cylinder_dd.vec);
        }

      // ---------------------------------------
      // Compute advection terms (if applicable):
      // ---------------------------------------
      if (do_advection){
          // Create backtrace vectors:
          T_l_backtrace.destroy();
          T_l_backtrace.create(p4est,nodes);

          if (solve_smoke){
              smoke_backtrace.destroy();
              smoke_backtrace.create(T_l_backtrace.vec);
            }

          // Do the Semi-Lagrangian backtrace:
          do_backtrace(T_l_n,T_l_backtrace,vel_n,smoke,smoke_backtrace,p4est,nodes,ngbd,interp_bw_grids);

      } // end of do_advection if statement

      // ------------------------------------------------------------
      // Setup RHS:
      // ------------------------------------------------------------
      // Create arrays to hold the RHS:
      rhs_Tl.create(p4est,nodes);
      rhs_Ts.create(p4est,nodes);
      if (solve_smoke) rhs_smoke.create(p4est,nodes);

      // Set up the RHS:
      setup_rhs(T_l_n,T_s_n,smoke,
                rhs_Tl,rhs_Ts,rhs_smoke,
                T_l_backtrace,smoke_backtrace,
                p4est,nodes);

      // ------------------------------------------------------------
      // Setup the solvers:
      // ------------------------------------------------------------
      // Now, set up the solver(s):
      solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd);
      solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd);
      solver_smoke = new my_p4est_poisson_nodes_mls_t(ngbd);

      solver_Tl->add_boundary(MLS_INTERSECTION,phi.vec,phi_dd.vec[0],phi_dd.vec[1],interface_bc_type_temp,bc_interface_val,bc_interface_coeff);
      solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid.vec,phi_solid_dd.vec[0],phi_solid_dd.vec[1],interface_bc_type_temp,bc_interface_val,bc_interface_coeff);
      if(example_ == 2){
        solver_Ts->add_boundary(MLS_INTERSECTION,phi_cylinder.vec,phi_cylinder_dd.vec[0],phi_cylinder_dd.vec[1],inner_interface_bc_type_temp,bc_interface_val_inner,bc_interface_coeff_inner);
        }

      // Set diagonal and diffusivity:
      switch(method_){
        case 0:
          solver_Tl->set_diag(1.0/dt);
          solver_Ts->set_diag(1.0/dt);
          if(solve_smoke) solver_smoke->set_diag(1./dt);
          break;

        case 1:
          solver_Tl->set_diag(2.0/dt);
          solver_Ts->set_diag(2.0/dt);
        }

      solver_Tl->set_mu(alpha_l);
      solver_Tl->set_rhs(rhs_Tl.vec);

      solver_Ts->set_mu(alpha_s);
      solver_Ts->set_rhs(rhs_Ts.vec);

      // Set some other solver properties:
      solver_Tl->set_integration_order(1);
      solver_Tl->set_use_sc_scheme(0);
      solver_Tl->set_cube_refinement(cube_refinement);
      solver_Tl->set_store_finite_volumes(1);

      solver_Ts->set_integration_order(1);
      solver_Ts->set_use_sc_scheme(0);
      solver_Ts->set_cube_refinement(cube_refinement);
      solver_Ts->set_store_finite_volumes(1);

      // Set the wall BC and RHS:
      solver_Tl ->set_wc(wall_bc_type_temp,wall_bc_value_temp);
      solver_Ts ->set_wc(wall_bc_type_temp,wall_bc_value_temp);

      // Preassemble the linear system
      solver_Tl->preassemble_linear_system();
      solver_Ts->preassemble_linear_system();

      // Create vector to hold the solution:
      T_l_np1.create(p4est,nodes);
      T_s_np1.create(T_l_np1.vec);

      // Solve the system:
      solver_Tl->solve(T_l_np1.vec);
      solver_Ts->solve(T_s_np1.vec);

      // Destroy the T_n values now and update them with the solution for the next timestep:
      T_l_n.destroy(); T_s_n.destroy();
      T_l_n.create(p4est,nodes); T_s_n.create(T_l_n.vec);

      VecCopyGhost(T_l_np1.vec,T_l_n.vec);
      VecCopyGhost(T_s_np1.vec,T_s_n.vec);

      if(solve_smoke){
          // FOR now: smoke has same diffusivity as the liquid phase, and we solve with no interfacial condition
          // Eventually: solve smoke with two RHSs, two diffusivities, and a jump condition on the interface -- enforce a zero jump in value and flux
          solver_smoke->set_mu(alpha_l);

          solver_smoke->set_integration_order(1);
          solver_smoke->set_use_sc_scheme(0);
          solver_smoke->set_cube_refinement(cube_refinement);
          solver_smoke->set_store_finite_volumes(1);

          // Set wall BC and RHS:
          solver_smoke ->set_wc(bc_type_smoke,bc_smoke_value);
          solver_smoke->set_rhs(rhs_smoke.vec);
          solver_smoke->preassemble_linear_system();

          // Create vector to hold the solution:
          smoke_np1.create(p4est,nodes);

          // Solve the system:
          solver_smoke->solve(smoke_np1.vec);

          // Destroy the n values now and update them with the solution for the next timestep:
          smoke.destroy();
          smoke.create(p4est,nodes);
          VecCopyGhost(smoke_np1.vec,smoke.vec);

          // Destroy np1 now that not needed:
          smoke_np1.destroy();
        }

      if (check_temperature_values){
        // Check Temperature values:
        PetscPrintf(mpi.comm(),"\n Checking temperature values after acquiring solution: \n [ ");
        PetscPrintf(mpi.comm(),"\n In fluid domain: ");
        check_T_values(phi,T_l_n,nodes,p4est,example_,phi_cylinder);
        PetscPrintf(mpi.comm(),"\n In solid domain: ");
        check_T_values(phi_solid,T_s_n,nodes,p4est,example_,phi_cylinder);
        PetscPrintf(mpi.comm()," ] \n");
        }

      // ------------------------------------------------------------
      // Some example specific operations:
      // ------------------------------------------------------------
      // Check error on the Frank sphere, if relevant:
      if(example_ == 1){
          check_frank_sphere_error(T_l_n, T_s_n, phi, v_interface, p4est, nodes, dxyz_close_to_interface,name,fich,tstep);
        }

      // Check if ice has melted, if relevant:
      if (example_ == 0){
          keep_going = check_ice_melted(phi,tn+dt,nodes,p4est);
        }

      // --------------------------------------------------------------------------------------------------------------
      // Destroy old information: (except phi, which gets updated by the update_p4est function)
      // --------------------------------------------------------------------------------------
      T_l_np1.destroy();
      T_s_np1.destroy();
      phi_solid.destroy();
      phi_dd.destroy();
      phi_solid_dd.destroy();
    } // <-- End of for loop through time

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

