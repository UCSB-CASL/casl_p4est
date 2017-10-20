
//-----------------------------------------------------------------
// Diffusion coefficient in negative domain
//-----------------------------------------------------------------
#ifdef P4_TO_P8
class mu_m_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_m){
      case 0: return 1.;
      case 1: return 1.e5;
      case 2: return 1.+(0.2*cos(x)+0.3*sin(y))*sin(z);
      case 3: return 1.e5*(1.+(0.2*cos(x)+0.3*sin(y))*sin(z));
      case 4: return y*y*log(x+2.) + 4.;
      case 5: return exp(-z);
    }
  }
} mu_m_cf;
class mux_m_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_m)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return -0.2*sin(x)*sin(z);
      case 3: return -1.e5*0.2*sin(x)*sin(z);
      case 4: return y*y/(x+2.);
      case 5: return 0;
    }
  }
} mux_m_cf;
class muy_m_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_m)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.3*cos(y)*sin(z);
      case 3: return 1.e5*0.3*cos(y)*sin(z);
      case 4: return 2.*y*log(x+2.);
      case 5: return 0;
    }
  }
} muy_m_cf;
class muz_m_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_m)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return (0.2*sin(x)+0.3*cos(y))*cos(z);
      case 3: return 1.e5*(0.2*sin(x)+0.3*cos(y))*cos(z);
      case 4: return 0.;
      case 5: return -exp(-z);
    }
  }
} muz_m_cf;
#else
class mu_m_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_mu_m){
      case 0: return 1.;
      case 1: return 1.e5;
      case 2: return 1.+0.2*cos(x)+0.3*sin(y);
      case 3: return 1.e5*(1.+0.2*cos(x)+0.3*sin(y));
      case 4: return y*y*log(x+2.) + 4.;
      case 5: return exp(-y);
    }
  }
} mu_m_cf;
class mux_m_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_mu_m){
      case 0: return 0.;
      case 1: return 0.;
      case 2: return -.2*sin(x);
      case 3: return -1.e5*.2*sin(x);
      case 4: return y*y/(x+2.);
      case 5: return 0;
    }
  }
} mux_m_cf;
class muy_m_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_mu_m){
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.3*cos(y);
      case 3: return 1.e5*(0.3)*cos(y);
      case 4: return 2.*y*log(x+2.);
      case 5: return -exp(-y);
    }
  }
} muy_m_cf;
#endif


//-----------------------------------------------------------------
// Diffusion coefficient in positive domain
//-----------------------------------------------------------------
#ifdef P4_TO_P8
class mu_p_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_p){
      case 0: return 1.;
      case 1: return 1.e5;
      case 2: return 1.+(0.2*sin(x)+0.3*cos(y))*cos(z);
      case 3: return 1.e5*(1.+(0.2*sin(x)+0.3*cos(y))*cos(z));
      case 4: return y*y*log(x+2.) + 4.;
      case 5: return exp(-z);
    }
  }
} mu_p_cf;
class mux_p_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_p)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return 0.2*cos(x)*cos(z);
      case 3: return 1.e5*0.2*cos(x)*cos(z);
      case 4: return y*y/(x+2.);
      case 5: return 0;
    }
  }
} mux_p_cf;
class muy_p_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_p)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return -0.3*sin(y)*cos(z);
      case 3: return -0.3*1.e5*sin(y)*cos(z);
      case 4: return 2.*y*log(x+2.);
      case 5: return 0;
    }
  }
} muy_p_cf;
class muz_p_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_mu_p)
    {
      case 0: return 0.;
      case 1: return 0.;
      case 2: return -(0.2*sin(x)+0.3*cos(y))*sin(z);
      case 3: return -1.e5*(0.2*sin(x)+0.3*cos(y))*sin(z);
      case 4: return 0.;
      case 5: return -exp(-z);
    }
  }
} muz_p_cf;
#else
class mu_p_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_mu_p){
      case 0: return 1.;
      case 1: return 1.e5;
      case 2: return 1.+0.2*sin(x)+0.3*cos(y);
      case 3: return 1.e5*(1.+0.2*sin(x)+0.3*cos(y));
      case 4: return y*y*log(x+2.) + 4.;
      case 5: return exp(-y);
    }
  }
} mu_p_cf;
class mux_p_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_mu_p){
      case 0: return 0.;
      case 1: return 0.;
      case 2: return .2*cos(x);
      case 3: return 1.e5*.2*cos(x);
      case 4: return y*y/(x+2.);
      case 5: return 0;
    }
  }
} mux_p_cf;
class muy_p_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_mu_p){
      case 0: return 0.;
      case 1: return 0.;
      case 2: return -0.3*sin(y);
      case 3: return 1.e5*(-0.3)*sin(y);
      case 4: return 2.*y*log(x+2.);
      case 5: return -exp(-y);
    }
  }
} muy_p_cf;
#endif
