/* geometry */

double xmin = -1.0;
double xmax =  1.0;
double ymin = -1.0;
double ymax =  1.0;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

double r0 = 0.5;
double d = 0.2;

double theta = 0.579;
#ifdef P4_TO_P8
double phy = 0.123;
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
  vector<double> IXr2, IXc0, IXc1;

  bool provided = true;

  double n_subs = 0;
  double n_Xs = 0;

  double alpha;

  Exact()
  {
#ifdef P4_TO_P8
    /* the whole domain */
    ID = 4.0/3.0*PI*r0*r0*r0;
    IDr2 = 4.0/3.0*PI*r0*r0*r0*(1.5*0.4*r0*r0+d*d);
    /* the whole boundary */
    IB = 4.0*PI*r0*r0;
    IBr2 = 4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d);
    /* sub-boundaries */
    ISB.push_back(4.0*PI*r0*r0);
    ISBr2.push_back(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d));
    /* intersections */
#else
    /* the whole domain */
    ID = PI*r0*r0;
    IDr2 = 0.5*PI*r0*r0*r0*r0 + PI*r0*r0*d*d;
    /* the whole boundary */
    IB = 2*PI*r0;
    IBr2 = 2.0*PI*r0*(r0*r0+d*d);
    /* sub-boundaries */
    ISB.push_back(2*PI*r0);
    ISBr2.push_back(2.0*PI*r0*(r0*r0+d*d));
    /* intersections */
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
  }
} geometry;
