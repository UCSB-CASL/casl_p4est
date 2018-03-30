
double phase_x =  0.13;
double phase_y =  1.55;
double phase_z =  0.7;
#ifdef P4_TO_P8
class u_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return sin(x)*cos(y)*exp(z);
      case 1: return 2.*log(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))-3.;
      case 2: return 1.*log((x+y+3.)/(y+z+3.))*sin(x+0.5*y+0.7*z);
      case 3: return exp(x+z-y*y)*(y+cos(x-z));
      case 4: return sin(x+0.3*y)*cos(x-0.7*y)*exp(z) + 3.*log(sqrt(x*x+y*y+z*z+0.5));
      case 10: return sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} u_cf;
#else
class u_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return sin(x)*cos(y);
      case 1: return 2.*(log( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 )-1.5);
      case 2: return 4.*log((0.7*x+3.0)/(y+3.0))*sin(x+0.5*y);
      case 3: return exp(x-y*y)*(y+cos(x));
      case 4: return sin(x+0.3*y)*cos(x-0.7*y) + 3.*log(sqrt(x*x+y*y+0.5));
      case 10: return (sin(PI*x+phase_x)*sin(PI*y+phase_y));
    }
  }
} u_cf;
#endif

// EXACT DERIVATIVES
#ifdef P4_TO_P8
class ux_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return cos(x)*cos(y)*exp(z);
      case 1: return 2.*(1.+2.*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 1.*( log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)/(x+y+3.) );
      case 3: return exp(x+z-y*y)*(y+cos(x-z)-sin(x-z));
      case 4: return ( cos(x+0.3*y)*cos(x-0.7*y) - sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*x/(x*x+y*y+z*z+0.5);
    case 10: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} ux_cf;
class uy_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return -sin(x)*sin(y)*exp(z);
      case 1: return 2.*(0.5-1.4*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 1.*( 0.5*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(1.0/(x+y+3.)-1.0/(y+z+3.)) );
      case 3: return exp(x+z-y*y)*(1.0 - 2.*y*(y+cos(x-z)));
      case 4: return ( 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*y/(x*x+y*y+z*z+0.5);
    case 10: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} uy_cf;
class uz_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return sin(x)*cos(y)*exp(z);
      case 1: return 2.*(-0.3-1.8*(x-0.7*y-0.9*z))/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.));
      case 2: return 1.*( 0.7*log((x+y+3.)/(y+z+3.))*cos(x+0.5*y+0.7*z) + sin(x+0.5*y+0.7*z)*(-1.0/(y+z+3.)) );
      case 3: return exp(x+z-y*y)*(y+cos(x-z)+sin(x-z));
      case 4: return cos(x-0.7*y)*sin(x+0.3*y)*exp(z) + 3.*z/(x*x+y*y+z*z+0.5);
    case 10: return PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*cos(PI*z+phase_z);
    }
  }
} uz_cf;
class lap_u_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (n_test){
      case 0: return -1.0*sin(x)*cos(y)*exp(z);
      case 1: return 4.*( 2.3/(x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.))
            -0.5*( pow(1.+2.*(x-0.7*y-0.9*z),2.) + pow(0.5-1.4*(x-0.7*y-0.9*z),2.) + pow(-0.3-1.8*(x-0.7*y-0.9*z),2.) )/pow((x+0.5*y-0.3*z+3. + pow(x-0.7*y-0.9*z, 2.)), 2.) );
      case 2: return 1.*( -1.74*log((x+y+3.)/(y+z+3.)) - 2./pow(x+y+3.,2.) + 2./pow(y+z+3.,2.) )*sin(x+0.5*y+0.7*z)
            + 1.*( 3./(x+y+3.) - 2.4/(y+z+3.) )*cos(x+0.5*y+0.7*z);
      case 3: return exp(x+z-y*y)*(-4.*y-2.*cos(x-z)+4.*y*y*(y+cos(x-z)));
      case 4: return -1.58*( sin(x+0.3*y)*cos(x-0.7*y) + cos(x+0.3*y)*sin(x-0.7*y) )*exp(z) + 3.*(x*x+y*y+z*z+1.5)/pow(x*x+y*y+z*z+0.5, 2.);
    case 10: return -3.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y)*sin(PI*z+phase_z);
    }
  }
} lap_u_cf;
#else
class ux_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return cos(x)*cos(y);
      case 1: return 2.*2.*(x+0.8*y+0.5)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return 4.*( 0.7/(0.7*x+3.) )*sin(x+0.5*y)
            + 4.*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(y+cos(x)-sin(x));
      case 4: return cos(x+0.3*y)*cos(x-0.7*y) - sin(x+0.3*y)*sin(x-0.7*y)
            + 3.*x/(x*x+y*y+0.5);
      case 10: return PI*cos(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} ux_cf;
class uy_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -sin(x)*sin(y);
      case 1: return 2.*2.*(0.8*x+0.64*y-0.35)/( pow(x+0.8*y, 2.)+(x-0.7*y)+4.0 );
      case 2: return 4.*( - 1./(y+3.) )*sin(x+0.5*y)
            + 4.*0.5*( log(0.7*x+3.)-log(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(1.-2.*y*(y+cos(x)));
      case 4: return 0.3*cos(x+0.3*y)*cos(x-0.7*y) + 0.7*sin(x+0.3*y)*sin(x-0.7*y)
        + 3.*y/(x*x+y*y+0.5);
      case 10: return PI*sin(PI*x+phase_x)*cos(PI*y+phase_y);
    }
  }
} uy_cf;

class lap_u_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (n_test){
      case 0: return -2.0*sin(x)*cos(y);
      case 1: {
        double C = (x+0.8*y)*(x+0.8*y)+(x-0.7*y)+4.0;
        return 2.*2.*(1.64/C - ( pow(2.0*(x+0.8*y)+1.0, 2.0) + pow(1.6*(x+0.8*y)-0.7, 2.0) )/2.0/C/C);
      }
      case 2: return 4.*( 1./pow(y+3., 2.) - 0.49/pow(0.7*x+3., 2.) - 1.25*(log(0.7*x+3.)-log(y+3.)) )*sin(x+0.5*y)
            + 4.*( 1.4/(0.7*x+3.) - 1./(y+3.) )*cos(x+0.5*y);
      case 3: return exp(x-y*y)*(y-2.*sin(x)) - 2.*exp(x-y*y)*(y*(3.-2.*y*y)+(1.-2.*y*y)*cos(x));
      case 4: return -2.58*sin(x+0.3*y)*cos(x-0.7*y) - 1.58*cos(x+0.3*y)*sin(x-0.7*y)
        + 3./pow(x*x+y*y+0.5, 2.);
      case 10: return -2.0*PI*PI*sin(PI*x+phase_x)*sin(PI*y+phase_y);
    }
  }
} lap_u_cf;
#endif
