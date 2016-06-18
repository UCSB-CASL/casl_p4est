// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/cube3_refined_mls.h>
#include <src/simplex3_mls_vtk.h>
#else
#include <src/cube2_refined_mls.h>
#include <src/simplex2_mls_vtk.h>
#endif

#include <tools/plotting.h>

//#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

#undef MIN
#undef MAX

using namespace std;

/* discretization */
int lmin = 1;
int lmax = 4;
#ifdef P4_TO_P8
int nb_splits = 5;
#else
int nb_splits = 6;
#endif

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

bool save_vtk = false;

/* geometry */

double xmin = 0.;
double xmax = 1.;
double ymin = 0.;
double ymax = 1.;
#ifdef P4_TO_P8
double zmin = 0.;
double zmax = 1.;
#endif

double r0 = 0.532121;
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
double xc_0 = 0.; double yc_0 =  0.; double zc_0 =  0.;
#else
double xc_0 = 0.; double yc_0 = 0.;
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
  double M_in_dir;

  bool provided = true;

  double n_subs = 0;
  double n_Xs = 0;

  double alpha;

  Exact()
  {
#ifdef P4_TO_P8
    /* the whole domain */
    ID = 0.125*4.0/3.0*PI*r0*r0*r0;
    IDr2 = 4.0/3.0*PI*r0*r0*r0*(1.5*0.4*r0*r0+d*d);
    /* the whole boundary */
    IB = 0.125*4.0*PI*r0*r0;
    IBr2 = 4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d);
    /* sub-boundaries */
    ISB.push_back(0.125*4.0*PI*r0*r0);
    ISBr2.push_back(4.0*PI*r0*r0*(1.5*2.0/3.0*r0*r0+d*d));
    M_in_dir = 0.25*PI*r0*r0;
    /* intersections */
#else
    /* the whole domain */
    ID = 0.25*PI*r0*r0;
    IDr2 = 0.5*PI*r0*r0*r0*r0 + PI*r0*r0*d*d;
    /* the whole boundary */
    IB = 0.25*2*PI*r0;
    IBr2 = 2.0*PI*r0*(r0*r0+d*d);
    /* sub-boundaries */
    ISB.push_back(0.25*2*PI*r0);
    ISBr2.push_back(2.0*PI*r0*(r0*r0+d*d));
    M_in_dir = r0;
    /* intersections */
#endif
  }
} exact;


/* Vectors to store numerical results */
class Result
{
public:
  vector<double> ID, IB, IDr2, IBr2;
  vector< vector<double> > ISB, ISBr2, IXr2;
  vector<double> M_in_dir;
  Result()
  {
    for (int i = 0; i < exact.n_subs; i++)
    {
      ISB.push_back(vector<double>());
      ISBr2.push_back(vector<double>());
    }
    for (int i = 0; i < exact.n_Xs; i++)
    {
      IXr2.push_back(vector<double>());
    }
  }
};

Result res_mlt;

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

vector<double> level, h;

int main (int argc, char* argv[])
{
  for(int iter=0; iter<nb_splits; ++iter)
  {

    int mx = pow(2,lmin+iter);
    int my = pow(2,lmin+iter);
#ifdef P4_TO_P8
    int mz = pow(2,lmin+iter);
#endif

    int n_phi = geometry.LSF.size();

    int n_nodes = (mx+1)*(my+1);
#ifdef P4_TO_P8
    n_nodes *= mz+1;
#endif


    std::vector< std::vector<double> > phi_values   (n_phi, std::vector<double> (n_nodes,-1));

    std::vector< std::vector<double> > phi_xx_values(n_phi, std::vector<double> (n_nodes,0));
    std::vector< std::vector<double> > phi_yy_values(n_phi, std::vector<double> (n_nodes,0));
#ifdef P4_TO_P8
    std::vector< std::vector<double> > phi_zz_values(n_phi, std::vector<double> (n_nodes,0));
#endif

    double dx = (xmax-xmin)/(double)(mx);
    double dy = (ymax-ymin)/(double)(my);
#ifdef P4_TO_P8
    double dz = (zmax-zmin)/(double)(mz);
#endif
#ifdef P4_TO_P8
#endif

    double dx_min = dx;
    double dy_min = dy;
#ifdef P4_TO_P8
    double dz_min = dz;
#endif

    std::vector<double> X(mx+1, 0.); for (int i = 0; i < mx+1; i++) {X[i]=(xmin + dx*(double)(i));}
    std::vector<double> Y(my+1, 0.); for (int j = 0; j < my+1; j++) {Y[j]=(ymin + dy*(double)(j));}
#ifdef P4_TO_P8
    std::vector<double> Z(mz+1, 0.); for (int k = 0; k < mz+1; k++) {Z[k]=(zmin + dz*(double)(k));}
#endif

    for (int n = 0; n < n_phi; n++)
#ifdef P4_TO_P8
      for (int k = 0; k < mz+1; k++)
#endif
      for (int j = 0; j < my+1; j++)
        for (int i = 0; i < mx+1; i++)
        {
          int idx = i+j*(mx+1);
#ifdef P4_TO_P8
          idx += k*(mx+1)*(my+1);
#endif

#ifdef P4_TO_P8
          double phi_000 = (*geometry.LSF[n])(X[i], Y[j], Z[k]);
          double phi_m00 = (*geometry.LSF[n])(X[i]-dx_min, Y[j], Z[k]);
          double phi_p00 = (*geometry.LSF[n])(X[i]+dx_min, Y[j], Z[k]);
          double phi_0m0 = (*geometry.LSF[n])(X[i], Y[j]-dy_min, Z[k]);
          double phi_0p0 = (*geometry.LSF[n])(X[i], Y[j]+dy_min, Z[k]);
          double phi_00m = (*geometry.LSF[n])(X[i], Y[j], Z[k]-dz_min);
          double phi_00p = (*geometry.LSF[n])(X[i], Y[j], Z[k]+dz_min);
#else
          double phi_000 = (*geometry.LSF[n])(X[i], Y[j]);
          double phi_m00 = (*geometry.LSF[n])(X[i]-dx_min, Y[j]);
          double phi_p00 = (*geometry.LSF[n])(X[i]+dx_min, Y[j]);
          double phi_0m0 = (*geometry.LSF[n])(X[i], Y[j]-dy_min);
          double phi_0p0 = (*geometry.LSF[n])(X[i], Y[j]+dy_min);
#endif

          phi_values    [n][idx] = phi_000;
          phi_xx_values [n][idx] = 1.*(phi_p00+phi_m00-2.0*phi_000)/dx_min/dx_min;
          phi_yy_values [n][idx] = 1.*(phi_0p0+phi_0m0-2.0*phi_000)/dy_min/dy_min;
#ifdef P4_TO_P8
          phi_zz_values [n][idx] = 1.*(phi_00p+phi_00m-2.0*phi_000)/dz_min/dz_min;
#endif
        }

#ifdef P4_TO_P8
    cube3_refined_mls_t cube(xmin, xmax, ymin, ymax, zmin, zmax);
    cube.set_interpolation_grid(xmin, xmax, ymin, ymax, zmin, zmax, mx, my, mz);
    cube.set_phi(phi_values, phi_xx_values, phi_yy_values, phi_zz_values, geometry.action, geometry.color);
#else
    cube2_refined_mls_t cube(xmin, xmax, ymin, ymax);
    cube.set_interpolation_grid(xmin, xmax, ymin, ymax, mx, my);
    cube.set_phi(phi_values, phi_xx_values, phi_yy_values, geometry.action, geometry.color);
#endif

//    cube.set_phi(geometry.LSF, geometry.action, geometry.color);

#ifdef P4_TO_P8
//    cube.construct_domain(pow(2,lmin+iter), pow(2,lmin+iter), pow(2,lmin+iter), lmax-lmin);
//    cube.construct_domain(pow(2,lmin), pow(2,lmin), pow(2,lmin), iter);
    cube.construct_domain(mx, my, mz, lmax-lmin);
#else
//    cube.construct_domain(pow(2,lmin+iter), pow(2,lmin+iter), lmax-lmin);
//    cube.construct_domain(pow(2,lmin), pow(2,lmin), iter);
    cube.construct_domain(mx, my, lmax-lmin);
#endif


    std::cout << cube.n_nodes << " nodes & " << cube.n_leafs << " cubes" << std::endl;

    if (save_vtk)
    {
#ifdef P4_TO_P8
      vector<simplex3_mls_t *> simplices;
      int n_sps = NTETS;
#else
      vector<simplex2_mls_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < cube.cubes_mls.size(); k++)
        if (cube.cubes_mls[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&cube.cubes_mls[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#else
      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#endif
    }

    /* Calculate and store results */
    if (exact.provided || iter < nb_splits-1)
    {
      level.push_back(lmax+iter);
//      h.push_back((xmax-xmin)/pow(2.0,(double)(lmax+iter)));
      h.push_back(dx);

      res_mlt.ID.push_back(cube.measure_of_domain    ());
      res_mlt.IB.push_back(cube.measure_of_interface (-1));
      res_mlt.M_in_dir.push_back(cube.measure_in_dir(0));

//      res_mlt.IDr2.push_back(cube.integrate_over_domain    (func_r2));
//      res_mlt.IBr2.push_back(cube.integrate_over_interface (func_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        res_mlt.ISB[i].push_back(cube.measure_of_interface(geometry.color[i]));
//        res_mlt.ISBr2[i].push_back(cube.integrate_over_interface(func_r2, geometry.color[i]));
      }

//      for (int i = 0; i < exact.n_Xs; i++)
//      {
//        res_mlt.IXr2[i].push_back(cube.integrate_over_intersection(func_r2, exact.IXc0[i], exact.IXc1[i]));
//      }
    }
    else if (iter == nb_splits-1)
    {
      exact.ID    = (cube.measure_of_domain        ());
      exact.IB    = (cube.measure_of_interface     (-1));
//      exact.IDr2  = (cube.integrate_over_domain    (func_r2));
//      exact.IBr2  = (cube.integrate_over_interface (func_r2, -1));

      for (int i = 0; i < exact.n_subs; i++)
      {
        exact.ISB.push_back(cube.measure_of_interface(geometry.color[i]));
//        exact.ISBr2.push_back(cube.integrate_over_interface(func_r2, geometry.color[i]));
      }

//      for (int i = 0; i < exact.n_Xs; i++)
//      {
//        exact.IXr2.push_back(cube.integrate_over_intersection(func_r2, exact.IXc0[i], exact.IXc1[i]));
//      }
    }
  }

  Gnuplot plot_ID;
  print_Table("Domain", exact.ID, level, h, "MLT", res_mlt.ID, 1, &plot_ID);

  Gnuplot plot_IB;
  print_Table("Interface", exact.IB, level, h, "MLT", res_mlt.IB, 1, &plot_IB);

  Gnuplot plot_M;
  print_Table("In dir", exact.M_in_dir, level, h, "MLT", res_mlt.M_in_dir, 1, &plot_M);

//  Gnuplot plot_IDr2;
//  print_Table("2nd moment of domain", exact.IDr2, level, h, "MLT", res_mlt.IDr2, 2, &plot_IDr2);

//  Gnuplot plot_IBr2;
//  print_Table("2nd moment of interface", exact.IBr2, level, h, "MLT", res_mlt.IBr2, 2, &plot_IBr2);

  vector<Gnuplot *> plot_ISB;
  vector<Gnuplot *> plot_ISBr2;
  for (int i = 0; i < exact.n_subs; i++)
  {
    plot_ISB.push_back(new Gnuplot());
    print_Table("Interface #"+to_string(i), exact.ISB[i], level, h, "MLT", res_mlt.ISB[i], 2, plot_ISB[i]);

//    plot_ISBr2.push_back(new Gnuplot());
//    print_Table("2nd moment of interface #"+to_string(i), exact.ISBr2[i], level, h, "MLT", res_mlt.ISBr2[i], 2, plot_ISBr2[i]);
  }

//  vector<Gnuplot *> plot_IXr2;
//  for (int i = 0; i < exact.n_Xs; i++)
//  {
//    plot_IXr2.push_back(new Gnuplot());
//    print_Table("Intersection of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), exact.IXr2[i], level, h, "MLT", res_mlt.IXr2[i], 2, plot_IXr2[i]);
//  }

//  w.stop(); w.read_duration();

  std::cin.get();

  for (int i = 0; i < exact.n_subs; i++)
  {
    delete plot_ISB[i];
//    delete plot_ISBr2[i];
  }

//  for (int i = 0; i < exact.n_Xs; i++)
//  {
//    delete plot_IXr2[i];
//  }

  return 0;
}
