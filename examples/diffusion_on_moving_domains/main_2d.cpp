/*
 * Title: diffusion_on_moving_domains
 * Description:
 * Author: dbochkov
 * Date Created: 04-16-2020
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
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

#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

// TEST SOLUTIONS
class sol_cf_t : public CF_DIM
{
public:
  int    *n;
  double *mag;
  cf_value_type_t what;
  sol_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (*n) {
      case 0: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(x)*cos(y)*exp(z);
          case DDX: return  (*mag)*cos(x)*cos(y)*exp(z);
          case DDY: return -(*mag)*sin(x)*sin(y)*exp(z);
          case DDZ: return  (*mag)*sin(x)*cos(y)*exp(z);
          case LAP: return -(*mag)*sin(x)*cos(y)*exp(z);
#else
          case VAL: return  (*mag)*sin(x)*cos(y);
          case DDX: return  (*mag)*cos(x)*cos(y);
          case DDY: return -(*mag)*sin(x)*sin(y);
          case LAP: return -(*mag)*2.*sin(x)*cos(y);
#endif
        }
      case 1: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*cos(x)*sin(y)*exp(z);
          case DDX: return -(*mag)*sin(x)*sin(y)*exp(z);
          case DDY: return  (*mag)*cos(x)*cos(y)*exp(z);
          case DDZ: return  (*mag)*cos(x)*sin(y)*exp(z);
          case LAP: return -(*mag)*cos(x)*sin(y)*exp(z);
#else
          case VAL: return  (*mag)*cos(x)*sin(y);
          case DDX: return -(*mag)*sin(x)*sin(y);
          case DDY: return  (*mag)*cos(x)*cos(y);
          case LAP: return -(*mag)*2.*cos(x)*sin(y);
#endif
        }
      case 2: switch (what) {
          case VAL: return (*mag)*exp(x);
          case DDX: return (*mag)*exp(x);
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
          case LAP: return (*mag)*exp(x);
        }
      case 3: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*0.5*log(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDX: return (*mag)*0.5*( 1.0+2.0*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDY: return (*mag)*0.5*( 0.5-1.4*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case DDZ: return (*mag)*0.5*(-0.3-1.8*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
          case LAP: return (*mag)*( 2.3/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))
                -0.5*( pow(1.+2.*(x-0.7*y-0.9*z),2.) + pow(0.5-1.4*(x-0.7*y-0.9*z),2.) + pow(-0.3-1.8*(x-0.7*y-0.9*z),2.) )/pow((x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.)), 2.) );
#else
          case VAL: return (*mag)*0.5*log( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case DDX: return (*mag)*(1.0*x+0.80*y+0.50)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case DDY: return (*mag)*(0.8*x+0.64*y-0.35)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
          case LAP: {
              double C = (x+0.8*y)*(x+0.8*y)+(x-0.7*y)+4.0;
              return (*mag)*( 1.64/C - ( pow(2.0*(x+0.8*y)+1.0, 2.0) + pow(1.6*(x+0.8*y)-0.7, 2.0) )/2.0/C/C );
            };
#endif
        }
      case 4: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*( log((x+y+3.)/(y+z+3.))*sin(x+0.5*y+0.7*z) );
          case DDX: return (*mag)*( 1.0*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)/(x+y+3.) );
          case DDY: return (*mag)*( 0.5*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(1.0/(x+y+3.)-1.0/(y+z+3.)) );
          case DDZ: return (*mag)*( 0.7*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(-1.0/(y+z+3.)) );
          case LAP: return (*mag)*( ( -1.74*log((x+y+3.)/(y+z+3.)) - 2./pow(x+y+3.,2.) + 2./pow(y+z+3.,2.) )*sin(x+0.5*y+0.7*z)
                + ( 3./(x+y+3.) - 2.4/(y+z+3.) )*cos(x+0.5*y+0.7*z) );
#else
          case VAL: return (*mag)*( log((0.7*x+3.0)/(y+3.0))*sin(x+0.5*y) );
          case DDX: return (*mag)*( ( 0.7/(0.7*x+3.) )*sin(x+0.5*y) + ( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y) );
          case DDY: return (*mag)*( ( - 1./(y+3.) )*sin(x+0.5*y) + 0.5*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y) );
          case LAP: return (*mag)*( ( 1./pow(y+3., 2.) - 0.49/pow(0.7*x+3., 2.) - 1.25*(log(0.7*x+3.)-log(y+3.)) )*sin(x+0.5*y)
                + ( 1.4/(0.7*x+3.) - 1./(y+3.) )*cos(x+0.5*y) );
#endif
        }
      case 5: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*exp(x+z-y*y)*(y+cos(x-z));
          case DDX: return (*mag)*exp(x+z-y*y)*(y+cos(x-z)-sin(x-z));
          case DDY: return (*mag)*exp(x+z-y*y)*(1.0 - 2.*y*(y+cos(x-z)));
          case DDZ: return (*mag)*exp(x+z-y*y)*(y+cos(x-z)+sin(x-z));
          case LAP: return (*mag)*exp(x+z-y*y)*(-4.*y-2.*cos(x-z)+4.*y*y*(y+cos(x-z)));
#else
          case VAL: return (*mag)*exp(x-y*y)*(y+cos(x));
          case DDX: return (*mag)*exp(x-y*y)*(y+cos(x)-sin(x));
          case DDY: return (*mag)*exp(x-y*y)*(1.-2.*y*(y+cos(x)));
          case LAP: return (*mag)*exp(x-y*y)*(y-2.*sin(x)) - 2.*exp(x-y*y)*(y*(3.-2.*y*y)+(1.-2.*y*y)*cos(x));
#endif
        }
      case 6: switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag)*( sin(x+0.3*y)*cos(x-0.7*y)*exp(z) + 3.*log(sqrt(x*x+y*y+z*z+0.5)) );
          case DDX: return (*mag)*( ( 1.0*cos(x+0.3*y)*cos(x-0.7*y) - 1.0*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*x/(x*x+y*y+z*z+0.5) );
          case DDY: return (*mag)*( ( 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*y/(x*x+y*y+z*z+0.5) );
          case DDZ: return (*mag)*( ( 1.0*cos(x-0.7*y)*sin(x+0.3*y)                                 )*exp(z) + 3.*z/(x*x+y*y+z*z+0.5) );
          case LAP: return (*mag)*( -1.58*( sin(x+0.3*y)*cos(x-0.7*y) + cos(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*(x*x+y*y+z*z+1.5)/pow(x*x+y*y+z*z+0.5, 2.) );
#else
          case VAL: return (*mag)*( sin(x+0.3*y)*cos(x-0.7*y) + 3.*log(sqrt(x*x+y*y+0.5)) );
          case DDX: return (*mag)*(  1.00*cos(x+0.3*y)*cos(x-0.7*y) - 1.00*sin(x+0.3*y)*sin(x-0.7*y) + 3.*x/(x*x+y*y+0.5) );
          case DDY: return (*mag)*(  0.30*cos(x+0.3*y)*cos(x-0.7*y) + 0.70*sin(x+0.3*y)*sin(x-0.7*y) + 3.*y/(x*x+y*y+0.5) );
          case LAP: return (*mag)*( -2.58*sin(x+0.3*y)*cos(x-0.7*y) - 1.58*cos(x+0.3*y)*sin(x-0.7*y) + 3./pow(x*x+y*y+0.5, 2.) );
#endif
        }
      case 7: {
        double p = mu_p_cf(DIM(x,y,z))/mu_m_cf(DIM(x,y,z));
        double r = sqrt(x*x+y*y);
        double r0 = 0.5+EPS;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
          case DDZ: return 0;
          case LAP: return 0;
#else
          case VAL: return (*mag)*( x*(p+1.) - x*(p-1.)*r0*r0/r/r )        /( p+1.+r0*r0*(p-1.) );
          case DDX: return (*mag)*( p+1.-r0*r0*(p-1.)*(y*y-x*x)/pow(r,4.) )/( p+1.+r0*r0*(p-1.) );
          case DDY: return (*mag)*( (p-1.)*r0*r0*2.*x*y/pow(r,4.) )        /( p+1.+r0*r0*(p-1.) );
          case LAP: return 0;
#endif
        }}
      case 9: {
#ifdef P4_TO_P8
        double r2 = x*x+y*y+z*z;
#else
        double r2 = x*x+y*y;
#endif
        if (r2 < EPS) r2 = EPS;
        switch (what) {
          case VAL: return (*mag)*(1+log(2.*sqrt(r2)));
          case DDX: return (*mag)*x/r2;
          case DDY: return (*mag)*y/r2;
#ifdef P4_TO_P8
          case DDZ: return (*mag)*z/r2;
          case LAP: return (*mag)/r2;
#else
          case LAP: return 0;
#endif
        }}
      case 10: {
        double phase_x =  0.13;
        double phase_y =  1.55;
        double phase_z =  0.7;
        double kx = 1;
        double ky = 1;
        double kz = 1;
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDX: return  (*mag)*PI*kx*cos(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDY: return  (*mag)*PI*ky*sin(PI*kx*x+phase_x)*cos(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
          case DDZ: return  (*mag)*PI*kz*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*cos(PI*kz*z+phase_z);
          case LAP: return -(*mag)*PI*PI*(kx*kx+ky*ky+kz*kz)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y)*sin(PI*kz*z+phase_z);
#else
          case VAL: return  (*mag)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
          case DDX: return  (*mag)*PI*kx*cos(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
          case DDY: return  (*mag)*PI*ky*sin(PI*kx*x+phase_x)*cos(PI*ky*y+phase_y);
          case LAP: return -(*mag)*PI*PI*(kx*kx+ky*ky)*sin(PI*kx*x+phase_x)*sin(PI*ky*y+phase_y);
#endif
        }}
      case 11: {
        double X    = (-x+y)/3.;
        double T5   = 16.*pow(X,5.)  - 20.*pow(X,3.) + 5.*X;
        double T5d  = 5.*(16.*pow(X,4.) - 12.*pow(X,2.) + 1.);
        double T5dd = 40.*X*(8.*X*X-3.);
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  T5*log(x+y+3.)*cos(z);
          case DDX: return  (T5/(x+y+3.) - T5d*log(x+y+3.)/3.)*cos(z);
          case DDY: return  (T5/(x+y+3.) + T5d*log(x+y+3.)/3.)*cos(z);
          case DDZ: return -T5*log(x+y+3.)*sin(z);
          case LAP: return  (2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.))*cos(z)
                - T5*log(x+y+3.)*cos(z);
#else
          case VAL: return T5*log(x+y+3.);
          case DDX: return T5/(x+y+3.) - T5d*log(x+y+3.)/3.;
          case DDY: return T5/(x+y+3.) + T5d*log(x+y+3.)/3.;
          case LAP: return 2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.);
#endif
        }}
      case 12: switch (what) {
#ifdef P4_TO_P8
          case VAL: return  (*mag)*sin(2.*x)*cos(2.*y)*exp(z);
          case DDX: return  (*mag)*2.*cos(2.*x)*cos(2.*y)*exp(z);
          case DDY: return -(*mag)*2.*sin(2.*x)*sin(2.*y)*exp(z);
          case DDZ: return  (*mag)*sin(2.*x)*cos(2.*y)*exp(z);
          case LAP: return -(*mag)*(2.*2.*2. - 1.)*sin(2.*x)*cos(2.*y)*exp(z);
#else
          case VAL: return  (*mag)*sin(2.*x)*cos(2.*y);
          case DDX: return  (*mag)*2.*cos(2.*x)*cos(2.*y);
          case DDY: return -(*mag)*2.*sin(2.*x)*sin(2.*y);
          case LAP: return -(*mag)*2.*2.*2.*sin(2.*x)*cos(2.*y);
#endif
        }
      case 13: switch (what){
          case VAL: return (*mag)*(exp((x - 0.5*(xmin.val+xmax.val))/(xmax.val -xmin.val)));
          case DDX: return (*mag)*(exp((x - 0.5*(xmin.val+xmax.val))/(xmax.val -xmin.val)))*(1/((xmax.val-xmin.val)));
          case DDY: return 0.0;
          case LAP: return (*mag)*(1/(SQR(xmax.val-xmin.val)))*(exp((x - 0.5*(xmin.val+xmax.val))/(xmax.val -xmin.val)));
        }
      case 14: switch (what){
#ifdef P4_TO_P8
          case VAL: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDX: return -(*mag)*(sin(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(1/(xmax.val-xmin.val))*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDY: return (*mag)*(cos(x/(xmax.val-xmin.val))*cos(y/(ymax.val-ymin.val)))*(1/(ymax.val-ymin.val))*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDZ: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(exp((z-zmin.val)/(zmax.val-zmin.val))*(1/(zmax.val-zmin.val)));
          case LAP: return (*mag)*((-1/(SQR(xmax.val-xmin.val)))+(-1/(SQR(ymax.val-ymin.val)))+1/SQR(zmax.val-zmin.val))*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val))*exp((z-zmin.val)/(zmax.val-zmin.val)));
#else
          case VAL: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
          case DDX: return -(*mag)*(sin(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(1/(xmax.val-xmin.val));
          case DDY: return (*mag)*(cos(x/(xmax.val-xmin.val))*cos(y/(ymax.val-ymin.val)))*(1/(ymax.val-ymin.val));
          case LAP: return -(*mag)*((1/(SQR(xmax.val-xmin.val)))+(1/(SQR(ymax.val-ymin.val))))*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
#endif
        }
#ifdef P4_TO_P8
      case 15: switch (what){
          case VAL: return (*mag)*exp((z-zmin.val)/(zmax.val-zmin.val));
          case DDX: return 0.0;
          case DDY: return 0.0;
          case DDZ: return (*mag)*(exp((z-zmin.val)/(zmax.val-zmin.val))*(1/(zmax.val-zmin.val)));
          case LAP: return (*mag)*(exp((z-zmin.val)/(zmax.val-zmin.val))*SQR(1/(zmax.val-zmin.val)));
        }
#endif
      case 16: switch (what){
          case VAL: return (*mag)*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
          case DDX: return -(*mag)*(sin(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)))*(1/(xmax.val-xmin.val));
          case DDY: return (*mag)*(cos(x/(xmax.val-xmin.val))*cos(y/(ymax.val-ymin.val)))*(1/(ymax.val-ymin.val));
#ifdef P4_TO_P8
          case DDZ: return 0.0;
#endif
          case LAP: return -(*mag)*((1/(SQR(xmax.val-xmin.val)))+(1/(SQR(ymax.val-ymin.val))))*(cos(x/(xmax.val-xmin.val))*sin(y/(ymax.val-ymin.val)));
        }
#ifdef P4_TO_P8
      case 17: switch (what){
          case VAL: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*sin(x/(xmax.val-xmin.val));
          case DDX: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*cos(x/(xmax.val-xmin.val))*(1/(xmax.val-xmin.val));
          case DDY: return ((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*sin(x/(xmax.val-xmin.val))*(1/(ymax.val-ymin.val));
          case DDZ: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*sin(x/(xmax.val-xmin.val))*(1/(zmax.val-zmin.val));
          case LAP: return ((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val))*sin(x/(xmax.val-xmin.val))*(-SQR(1/(xmax.val-xmin.val)));
        }
#endif
#ifdef P4_TO_P8
      case 18: switch (what){
          case VAL: return ((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val))*SQR((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))+pow((z-0.5*(zmin.val+zmax.val))/(zmax.val-zmin.val), 3.0);
          case DDX: return SQR((y-0.5*(ymin.val+ymax.val))/(ymax.val-ymin.val))*(1/(xmax.val-xmin.val));
          case DDY: return ((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val))*(2*(y-0.5*(ymin.val+ymax.val))/SQR(ymax.val-ymin.val));
          case DDZ: return 3.*SQR(z-0.5*(zmin.val+zmax.val))/(pow((zmax.val-zmin.val), 3.0));
          case LAP: return ((x-0.5*(xmin.val+xmax.val))/(xmax.val-xmin.val))*(2./SQR(ymax.val-ymin.val))+6.*(z-0.5*(zmin.val+zmax.val))/(pow((zmax.val-zmin.val), 3.0)) ;
        }
#endif
      case 19: {
        double X    = (-x+y)/3.;
        double T5   = 16.*pow(X,5.)  - 20.*pow(X,3.) + 5.*X;
        double T5d  = 5.*(16.*pow(X,4.) - 12.*pow(X,2.) + 1.);
        double T5dd = 40.*X*(8.*X*X-3.);
        switch (what) {
#ifdef P4_TO_P8
          case VAL: return  1.+T5*log(x+y+3.)*cos(z);
          case DDX: return  (T5/(x+y+3.) - T5d*log(x+y+3.)/3.)*cos(z);
          case DDY: return  (T5/(x+y+3.) + T5d*log(x+y+3.)/3.)*cos(z);
          case DDZ: return -T5*log(x+y+3.)*sin(z);
          case LAP: return  (2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.))*cos(z)
                - T5*log(x+y+3.)*cos(z);
#else
          case VAL: return 1.+T5*log(x+y+3.);
          case DDX: return T5/(x+y+3.) - T5d*log(x+y+3.)/3.;
          case DDY: return T5/(x+y+3.) + T5d*log(x+y+3.)/3.;
          case LAP: return 2.*T5dd*log(x+y+3.)/9. - 2.*T5/pow(x+y+3.,2.);
#endif
        }}
      case 20: switch (what) {
          case VAL: return (*mag);
          case DDX: return 0;
          case DDY: return 0;
#ifdef P4_TO_P8
          case DDZ: return 0;
#endif
          case LAP: return 0;
        }
      default:
        throw std::invalid_argument("Unknown test function\n");
    }
  }
};



double mag = 1;

#include <src/parameter_list.h>

param_list_t pl;

param_t<int> test_solution(pl, 0, "test_solution", "");

sol_cf_t sol_cf(VAL, test_solution.val, mag);
sol_cf_t sol_lap_cf(LAP, test_solution.val, mag);
sol_cf_t sol_t_cf(DDT, test_solution.val, mag);
sol_cf_t DIM(sol_x_cf(DDX, test_solution.val, mag),
             sol_y_cf(DDY, test_solution.val, mag),
             sol_z_cf(DDZ, test_solution.val, mag));

double DIM(xc, yc, zc);
double theta;

//inline double x_to_X(DIM(double x, double y, double z))
//{
//  double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
//  double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
//}

param_t<int> motion_type(pl, 0, "motion_type", "");

//double time_start = 0;
//double time_final = 1;

//double vx(double t)
//{
//  switch (motion_type.val) {
//    case 0: 1./(time_final - time_start);
//    case 1: 0;
//    case 2:
//    default: throw;
//  }
//}

param_t<int> test_geometry (pl, 0, "test_geometry", "");


class phi_cf_t : public CF_DIM
{
public:
  int    *idx;
  cf_value_type_t what;
  phi_cf_t(cf_value_type_t what, int &n, double &mag) : what(what), n(&n), mag(&mag) {}

  int num()
  {
    switch (test_geometry.val) {
      case 0: return 0; // no geometry
      case 1: return 1; // sphere
      case 2: return 1; // star-shaped
      case 3: return 2; // difference of two spheres
      case 4: return 2; // intersection of two spheres
      case 5: return 2; // union of two spheres
      case 6: return 2; // half a sphere
      case 7: return 2; // rectangle
      default: throw;
    }
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    switch (test_geometry.val) {
      case 0:
        switch (*idx) {
          default:
            switch (what) {
              case VAL: throw;
              case DDX: throw;
              case DDY: throw;
              default: throw;
            }
        }
      case 1:
        static
        switch (*idx) {
          default:
            switch (what) {
              case VAL: throw;
              case DDX: throw;
              case DDY: throw;
              default: throw;
            }
        }
    }
  }
};


void get_cfs(int& num, std::vector<mls_opn_t>& opn, std::vector<CF_DIM*>& phi_cfs, CF_DIM* rhs_cf, std::vector<CF_DIM*>& bc_values)
{
}


int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // stopwatch
  parStopWatch w;
  w.start("Running example: diffusion_on_moving_domains");

  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz[]      = { 1,  1,  1};
  const double xyz_min[] = {-1, -1, -1};
  const double xyz_max[] = { 1,  1,  1};
  const int periodic[]   = { 0,  0,  0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } circle;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5 - sqrt(SQR(x) + SQR(y));
    }
  } circle;
#endif

  splitting_criteria_cf_t sp(3, 8, &circle);
  p4est->user_pointer = &sp;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // save the grid into vtk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         0, 0, "diffusion_on_moving_domains");

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

