/* discretization */
int lmin = 4;
int lmax = 4;
#ifdef P4_TO_P8
int nb_splits = 4;
#else
int nb_splits = 6;
#endif

int nx = 1;
int ny = 1;
int nz = 1;

bool save_vtk = true;

/* geometry */

double xmin = -1.00;
double xmax =  1.00;
double ymin = -1.00;
double ymax =  1.00;
double zmin = -1;
double zmax =  1;

double r0 = 0.711;
double r1 = 0.639;
double d = 0.35;

double theta = 0.779;
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
double xc_0 = round((-d*sinT*cosP-0.02)*100.)/100.; double yc_0 = round(( d*cosT*cosP-0.07)*100.)/100.; double zc_0 = round(( d*sinP-0.03)*100.)/100.;
double xc_1 = round(( d*sinT*cosP-0.02)*100.)/100.; double yc_1 = round((-d*cosT*cosP-0.07)*100.)/100.; double zc_1 = round((-d*sinP-0.03)*100.)/100.;
#else
double xc_0 = round((-d*sinT+0.08)*100.)/100.; double yc_0 = round(( d*cosT-0.07)*100.)/100.;
double xc_1 = round(( d*sinT+0.08)*100.)/100.; double yc_1 = round((-d*cosT-0.07)*100.)/100.;
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
    return -(r1 - sqrt(SQR(x-xc_1) + SQR(y-yc_1) + SQR(z-zc_1)));
  }
} ls_circle_1;
#else
class LS_CIRCLE_1: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -(r1 - sqrt(SQR(x-xc_1) + SQR(y-yc_1)));
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

#ifdef P4_TO_P8
class FUNC_X: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return x;
  }
} func_x;
#else
class FUNC_X: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return x;
  }
} func_x;
#endif

#ifdef P4_TO_P8
class FUNC_Y: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return y;
  }
} func_y;
#else
class FUNC_Y: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return y;
  }
} func_y;
#endif

class Exact {
public:
  double ID;
  double IB;
  double IDr2;
  double IBr2;
  vector<double> ISB, ISBr2;
  vector<double> IX, IXx, IXy, IXr2;
  vector<int> IXc0, IXc1;
  double alpha;

  double n_subs = 2;
  double n_Xs = 1;

  bool provided = true;

  Exact()
  {
#ifdef P4_TO_P8
    // Auxiliary values
    double R0 = sqrt(xc_0*xc_0 + yc_0*yc_0 + zc_0*zc_0);
    double R1 = sqrt(xc_1*xc_1 + yc_1*yc_1 + zc_1*zc_1);

    double D = sqrt(pow(xc_1-xc_0,2.0)+pow(yc_1-yc_0,2.0)+pow(zc_1-zc_0,2.0));

    double d0 = (r0*r0+D*D-r1*r1)/(2.0*D);
    double d1 = D-d0;

    double d0x =  (xc_1-xc_0)/D*d0; double d0y =  (yc_1-yc_0)/D*d0; double d0z =  (zc_1-zc_0)/D*d0;
    double d1x = -(xc_1-xc_0)/D*d1; double d1y = -(yc_1-yc_0)/D*d1; double d1z = -(zc_1-zc_0)/D*d1;

    double r0d0 = xc_0*d0x + yc_0*d0y + zc_0*d0z;
    double r1d1 = xc_1*d1x + yc_1*d1y + zc_1*d1z;

    double alpha_0 = acos(d0/r0);
    double alpha_1 = acos(d1/r1);

    double mu_sph_0 = 4./3.*PI*r0*r0*r0; double mi_sph_0 = mu_sph_0*(0.6*r0*r0+R0*R0);
    double mu_sph_1 = 4./3.*PI*r1*r1*r1; double mi_sph_1 = mu_sph_1*(0.6*r1*r1+R1*R1);

    double mu_tri_0 = d0*PI*(r0*r0-d0*d0)/3.0; double mi_tri_0 = 0.3*mu_tri_0*(r0*r0+d0*d0) + mu_tri_0*(R0*R0+1.5*r0d0);
    double mu_tri_1 = d1*PI*(r1*r1-d1*d1)/3.0; double mi_tri_1 = 0.3*mu_tri_1*(r1*r1+d1*d1) + mu_tri_1*(R1*R1+1.5*r1d1);

    double mu_sec_0 = 2.*PI*r0*r0*(r0-d0)/3.; double mi_sec_0 = 0.4*PI*r0*r0*r0*r0*(r0-d0) + mu_sec_0*(R0*R0+0.75*r0d0*(r0/d0+1.));
    double mu_sec_1 = 2.*PI*r1*r1*(r1-d1)/3.; double mi_sec_1 = 0.4*PI*r1*r1*r1*r1*(r1-d1) + mu_sec_1*(R1*R1+0.75*r1d1*(r1/d1+1.));

    double mu_bnd_0 = 4.*PI*r0*r0; double mi_bnd_0 = mu_bnd_0*(r0*r0+R0*R0);
    double mu_bnd_1 = 4.*PI*r1*r1; double mi_bnd_1 = mu_bnd_1*(r1*r1+R1*R1);

    double mu_seg_0 = 2.*PI*r0*(r0-d0); double mi_seg_0 = 2.*PI*r0*r0*r0*(r0-d0) + mu_seg_0*(R0*R0+r0d0*(r0/d0+1.));
    double mu_seg_1 = 2.*PI*r1*(r1-d1); double mi_seg_1 = 2.*PI*r1*r1*r1*(r1-d1) + mu_seg_1*(R1*R1+r1d1*(r1/d1+1.));

#else
    // Auxiliary values
    double R0 = sqrt(xc_0*xc_0 + yc_0*yc_0);
    double R1 = sqrt(xc_1*xc_1 + yc_1*yc_1);

    double D = sqrt(pow(xc_1-xc_0,2.0)+pow(yc_1-yc_0,2.0));

    double d0 = (r0*r0+D*D-r1*r1)/(2.0*D);
    double d1 = D-d0;

    double d0x =  (xc_1-xc_0)/D*d0; double d0y =  (yc_1-yc_0)/D*d0;
    double d1x = -(xc_1-xc_0)/D*d1; double d1y = -(yc_1-yc_0)/D*d1;

    double r0d0 = xc_0*d0x + yc_0*d0y;
    double r1d1 = xc_1*d1x + yc_1*d1y;

    double alpha_0 = acos(d0/r0);
    double alpha_1 = acos(d1/r1);

    double mu_sph_0 = PI*r0*r0; double mi_sph_0 = PI*r0*r0*(0.5*r0*r0+R0*R0);
    double mu_sph_1 = PI*r1*r1; double mi_sph_1 = PI*r1*r1*(0.5*r1*r1+R1*R1);

    double mu_tri_0 = d0*sqrt(r0*r0-d0*d0); double mi_tri_0 = d0*sqrt(r0*r0-d0*d0)*(r0*r0/6.0+d0*d0/3.0+R0*R0 + 4.0*r0d0/3.0);
    double mu_tri_1 = d1*sqrt(r1*r1-d1*d1); double mi_tri_1 = d1*sqrt(r1*r1-d1*d1)*(r1*r1/6.0+d1*d1/3.0+R1*R1 + 4.0*r1d1/3.0);

    double mu_sec_0 = alpha_0*r0*r0; double mi_sec_0 = alpha_0*r0*r0*(0.5*r0*r0+R0*R0+2.0*r0d0*2.0*r0*sin(alpha_0)/alpha_0/d0/3.0);
    double mu_sec_1 = alpha_1*r1*r1; double mi_sec_1 = alpha_1*r1*r1*(0.5*r1*r1+R1*R1+2.0*r1d1*2.0*r1*sin(alpha_1)/alpha_1/d1/3.0);

    double mu_bnd_0 = 2.0*PI*r0; double mi_bnd_0 = 2.0*PI*r0*(r0*r0+R0*R0);
    double mu_bnd_1 = 2.0*PI*r1; double mi_bnd_1 = 2.0*PI*r1*(r1*r1+R1*R1);

    double mu_seg_0 = 2.0*alpha_0*r0; double mi_seg_0 = 2.0*alpha_0*r0*(r0*r0+R0*R0+2.0*r0d0*r0*sin(alpha_0)/alpha_0/d0);
    double mu_seg_1 = 2.0*alpha_1*r1; double mi_seg_1 = 2.0*alpha_1*r1*(r1*r1+R1*R1+2.0*r1d1*r1*sin(alpha_1)/alpha_1/d1);
#endif

    ID   = mu_sph_0+mu_sph_1 -mu_sec_0-mu_sec_1 +mu_tri_0+mu_tri_1;
    IDr2 = mi_sph_0+mi_sph_1 -mi_sec_0-mi_sec_1 +mi_tri_0+mi_tri_1;

    ISB.push_back(mu_bnd_0-mu_seg_0); ISBr2.push_back(mi_bnd_0-mi_seg_0);
    ISB.push_back(mu_bnd_1-mu_seg_1); ISBr2.push_back(mi_bnd_1-mi_seg_1);

    IXc0.push_back(0);
    IXc1.push_back(1);
#ifdef P4_TO_P8
    IX.push_back(PI*(r0*r0-d0*d0)); IXr2.push_back(PI*(r0*r0-d0*d0)*(r0*r0+R0*R0+2.*r0d0);
#else
    IX.push_back(2.0*(xc_0+d0x)); IXr2.push_back(2.0*(yc_0+d0y));
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
    std::cout << "Level-set 0: " << xc_0 << ", " << yc_0 << endl;
    std::cout << "Level-set 1: " << xc_1 << ", " << yc_1 << endl;
  }
} geometry;
