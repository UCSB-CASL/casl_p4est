
//---------------------------------------------------------------
// Diagonal term in negative domain
//---------------------------------------------------------------
#ifdef P4_TO_P8
class diag_term_m_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_diag_term_m)
    {
      case 0: return 0.;
      case 1: return 1.;
      case 2: return cos(x+z)*exp(y);
    }
  }
} diag_term_m_cf;
#else
class diag_term_m_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_diag_term_m)
    {
      case 0: return 0;
      case 1: return 1.;
      case 2: return sin(x)*exp(y);
    }
  }
} diag_term_m_cf;
#endif

//---------------------------------------------------------------
// Diagonal term in positive domain
//---------------------------------------------------------------
#ifdef P4_TO_P8
class diag_term_p_cf_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch (num_test_diag_term_p)
    {
      case 0: return 0.;
      case 1: return 1.;
      case 2: return cos(x+z)*exp(y);
    }
  }
} diag_term_p_cf;
#else
class diag_term_p_cf_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_test_diag_term_p)
    {
      case 0: return 0;
      case 1: return 1.;
      case 2: return sin(x)*exp(y);
    }
  }
} diag_term_p_cf;
#endif
