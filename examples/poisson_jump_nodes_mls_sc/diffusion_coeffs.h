
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
      case 1: return 1+(0.2*sin(x)+0.3*cos(y))*cos(z);
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
      case 1: return 0.2*cos(x)*cos(z);
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
      case 1: return -0.3*sin(y)*cos(z);
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
      case 1: return -(0.2*sin(x)+0.3*cos(y))*sin(z);
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
      case 1: return 1+0.2*sin(x)+0.3*cos(y);
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
      case 1: return .2*cos(x);
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
      case 1: return -0.3*sin(y);
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
      case 1: return 1+(0.2*sin(x)+0.3*cos(y))*cos(z);
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
      case 1: return 0.2*cos(x)*cos(z);
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
      case 1: return -0.3*sin(y)*cos(z);
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
      case 1: return -(0.2*sin(x)+0.3*cos(y))*sin(z);
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
      case 1: return 1+0.2*sin(x)+0.3*cos(y);
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
      case 1: return .2*cos(x);
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
      case 1: return -0.3*sin(y);
    }
  }
} muy_p_cf;
#endif
