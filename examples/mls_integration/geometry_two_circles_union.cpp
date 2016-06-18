/* geometry */

double xmin = -1.00;
double xmax =  1.00;
double ymin = -1.00;
double ymax =  1.00;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

double r0 = 0.5;
double d = 0.35;

double theta = 0.579;
#ifdef P4_TO_P8
double phy = 0.312;
#endif

double cosT = cos(theta);
double sinT = sin(theta);
#ifdef P4_TO_P8
double cosP = cos(phy);
double sinP = sin(phy);
#endif

#ifdef P4_TO_P8
double xc_0 = -d*sinT*cosP; double yc_0 =  d*cosT*cosP; double zc_0 =  d*sinP;
double xc_1 =  d*sinT*cosP; double yc_1 = -d*cosT*cosP; double zc_1 = -d*sinP;
#else
double xc_0 = -d*sinT; double yc_0 =  d*cosT;
double xc_1 =  d*sinT; double yc_1 = -d*cosT;
#endif

#ifdef P4_TO_P8
class LS_CIRCLE_0: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0) + SQR(z-zc_0)));
  }
} ls_circle_0;
#else
class LS_CIRCLE_0: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_0) + SQR(y-yc_0)));
  }
} ls_circle_0;
#endif


#ifdef P4_TO_P8
class LS_CIRCLE_1: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return -(r0 - sqrt(SQR(x-xc_1) + SQR(y-yc_1) + SQR(z-zc_1)));
  }
} ls_circle_1;
#else
class LS_CIRCLE_1: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r0 - sqrt(SQR(x-xc_1) + SQR(y-yc_1)));
  }
} ls_circle_1;
#endif

#ifdef P4_TO_P8
class LS_REF: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    double a = min(ls_circle_0(x,y,z), ls_circle_1(x,y,z));
    if (a < 0) a = 0;
    return a;
  }
} ls_ref;
#else
class LS_REF: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    double a = min(ls_circle_0(x,y), ls_circle_1(x,y));
    if (a < 0) a = 0;
    return a;
  }
} ls_ref;
#endif

#ifdef P4_TO_P8
class FUNC: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return 1.0;
  }
} func;
#else
class FUNC: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return 1.0;
  }
} func;
#endif

#ifdef P4_TO_P8
class FUNC_R2: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return x*x+y*y+z*z;
  }
} func_r2;
#else
class FUNC_R2: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return x*x+y*y;
  }
} func_r2;
#endif

class Exact {
public:
  double ID;
  double IB;
  double IDr2;
  double IBr2;
  vector<double> ISB, ISBr2;
  vector<double> IXr2;
  vector<int> IXc0, IXc1;
  double alpha;

  double n_subs = 2;
  double n_Xs = 1;

  bool provided = true;

  Exact()
  {
#ifdef P4_TO_P8
    alpha = acos(d/r0);
    /* the whole domain */
    ID = 2.0*4.0/3.0*PI*r0*r0*r0 - 2.0/3.0*PI*pow(r0-d,2.0)*(2.0*r0+d);
    IDr2 = 2.0*4.0/3.0*PI*r0*r0*r0*(1.5*0.4*r0*r0+d*d) -
           2.*(0.4*PI*pow(r0,5.)*(1-d/r0) - 0.1*PI*d*(pow(r0,4.)-pow(d,4.)) +
            1./3.*PI*pow(r0-d,2.)*(2.*r0+d)*(d*d-1.5*d*pow(r0+d,2.)/(2.*r0+d)));
    /* the whole boundary */
    IB = 2.0*(4.0*PI*r0*r0 - 2.*PI*r0*(r0-d));
    IBr2 = 2.0*(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d)-2.*PI*(1-d/r0)*r0*r0*(r0*r0+d*d-r0*d*(1+d/r0)));
    /* sub-boundaries */
    ISB.push_back(4.0*PI*r0*r0 - 2.*PI*r0*(r0-d));
    ISBr2.push_back(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d)-2.*PI*(1-d/r0)*r0*r0*(r0*r0+d*d-r0*d*(1+d/r0)));
    ISB.push_back(4.0*PI*r0*r0 - 2.*PI*r0*(r0-d));
    ISBr2.push_back(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d)-2.*PI*(1-d/r0)*r0*r0*(r0*r0+d*d-r0*d*(1+d/r0)));
    /* intersections */
    IXr2.push_back(2.*PI*pow(r0*r0-d*d,1.5));
    IXc0.push_back(0);
    IXc1.push_back(1);
#else
    // Auxiliary values
    alpha = acos(d/r0);
    double r_bar_A = 2.0*r0*sin(alpha)/alpha/3.0;
    double r_bar_B = r0*sin(alpha)/alpha;

    /* the whole domain */
    ID = 2.0*PI*r0*r0 - 2.0*(alpha*r0*r0-d*sqrt(r0*r0-d*d));
    IDr2 = 2.0*(0.5*PI*r0*r0*r0*r0 + PI*r0*r0*d*d) -
           (alpha*r0*r0*r0*r0 +
            2.0*alpha*r0*r0*d*(d-2.0*r_bar_A) -
            d*r0*r0*sqrt(r0*r0-d*d)/3.0);
    /* the whole boundary */
    IB = 2.0*2.0*(PI-alpha)*r0;
    IBr2 = 2.0*(2.0*PI*r0*(r0*r0+d*d) - 2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    /* sub-boundaries */
    ISB.push_back(2.0*(PI-alpha)*r0);
    ISBr2.push_back(2.0*PI*r0*(r0*r0+d*d) - 2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    ISB.push_back(2.0*(PI-alpha)*r0);
    ISBr2.push_back(2.0*PI*r0*(r0*r0+d*d) - 2.0*alpha*r0*(r0*r0+d*d-2.0*r_bar_B*d));
    /* intersections */
    IXr2.push_back(2.0*(r0*r0-d*d));
    IXc0.push_back(0);
    IXc1.push_back(1);
#endif
  }
} exact;

class Geometry
{
public:
#ifdef P4_TO_P8
  vector<CF_3 *> LSF;
#else
  vector<CF_2 *> LSF;
#endif
  vector<action_t> action;
  vector<int> color;
  Geometry()
  {
    LSF.push_back(&ls_circle_0); action.push_back(INTERSECTION); color.push_back(0);
    LSF.push_back(&ls_circle_1); action.push_back(ADDITION); color.push_back(1);
  }
} geometry;
