#ifdef CASL_OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include <lib/arrays/ArrayV.h>
#include <lib/amr/QuadTree.h>
#include <lib/amr/QuadTreeLevelSet.h>
#include <lib/tools/QuadraticInterpolationOnQuadTree.h>
#include <lib/tools/StopWatch.h>
#include <vector>

using namespace std;
using namespace CASL;

/*
 * for an explanation on NACA 4 digit airfoils, see
 * www.airfoiltools.com/airfoil/naca4digit
 */

//int sample_size = 5000;
int sample_size = 500000;

int lmax = 10;

double naca_angle = 1;

/* be carfeful if you use something like 0012, that's hexadecimal ! use 12 instead */
int naca_number = 4702;

/* default naca have an open trailing edge, which can be closed by tweaking a coefficient */
bool close_trailing_edge = true;

/* export the discrete sampling of the naca surface in a csv file */
bool save_sampling = false;

class NACA : public CF_2
{
private:
  vector<Point2> sample;
  unsigned int N;
  double naca_length;
  double naca_number;
  double naca_angle;
  double x_edge;
public:
  NACA(double number, double length, double angle)
  {
    N = sample_size;
    sample.resize(2*N);
    lip = 1.2;
    naca_length = length;
    naca_number = number;
    naca_angle = angle;
    x_edge = 8;

    double a0 =  0.2969;
    double a1 = -0.126;
    double a2 = -0.3516;
    double a3 =  0.2843;
    double a4 = close_trailing_edge ? -0.1036 : -0.1015;

    double T = mod(number,100); number-=T; T/=100;   /* thickness */
    double P = mod(number,1000); number-=P; P/=1000; /* position of maximum camber */
    double M = floor(number/1000) / 100;             /* camber */
    printf("Naca #%04g, params: chord=%g  position=%g  thickness=%g\n", naca_number, M, P, T);
    fflush(stdout);

    double dbeta = PI/(double)(N-1);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=0; i<N; ++i)
    {
      double xt = (1-cos(i*dbeta))/2;
      double yt = T/.2 * (a0*sqrt(xt) + xt*(a1 + xt*(a2 + xt*(a3 + a4*xt))));

      double yc, dycdx;
      if(xt<P)
      {
        yc = M/(P*P) * (2*P*xt - xt*xt);
        dycdx = 2*M/(P*P) * (P-xt);
      }
      else
      {
        yc = M/((1-P)*(1-P)) * (1 - 2*P + 2*P*xt - xt*xt);
        dycdx = 2*M/((1-P)*(1-P)) * (P-xt);
      }

      double theta = atan(dycdx);

      Point2 pu(xt - yt*sin(theta), yc + yt*cos(theta));
      Point2 pl(xt + yt*sin(theta), yc - yt*cos(theta));

      /* now apply scaling + rotation + translation */
      pu *= naca_length;
      pl *= naca_length;

      theta = -PI*naca_angle/180;
      double x_rot, y_rot;

      pu.x -= naca_length/2;
      x_rot = pu.x*cos(theta) - pu.y*sin(theta);
      y_rot = pu.x*sin(theta) + pu.y*cos(theta);
      pu.x = x_rot + naca_length/2 + x_edge;
      pu.y = y_rot;

      pl.x -= naca_length/2;
      x_rot = pl.x*cos(theta) - pl.y*sin(theta);
      y_rot = pl.x*sin(theta) + pl.y*cos(theta);
      pl.x = x_rot + naca_length/2 + x_edge;
      pl.y = y_rot;

      sample[2*i  ] = pl;
      sample[2*i+1] = pu;
    }
  }

  void save_sampling(const char *name)
  {
    FILE *fp;
    fp = fopen(name, "w");
    fprintf(fp, "x, y\n");
    for(unsigned int i=0; i<sample.size(); ++i)
    {
      fprintf(fp, "%e, %e\n", sample[i].x, sample[i].y);
    }
    fclose(fp);
  }

  double operator()(double x, double y) const
  {
    std::vector<double> dist_(omp_get_max_threads(),DBL_MAX);
    std::vector<unsigned int> ind_(omp_get_max_threads(), 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=0; i<sample.size(); ++i)
    {
      double d = sqrt(SQR(x-sample[i].x) + SQR(y-sample[i].y));
      if(d<dist_[omp_get_thread_num()])
      {
        dist_[omp_get_thread_num()] = d;
        ind_[omp_get_thread_num()] = i;
      }
    }

    double dist = DBL_MAX;
    unsigned int ind = 0;
    for(int i=0; i<omp_get_max_threads(); ++i)
    {
      if(dist_[i]<dist)
      {
        dist = dist_[i];
        ind = ind_[i];
      }
    }

    if(x<=x_edge || x>=x_edge+naca_length) dist = -dist;
    else
    {
      Point2 a;
      Point2 b;
      if(ind>sample.size()-3) { a = sample[ind]-sample[ind-2]; b = Point2(x,y)-sample[ind-2]; }
      else                    { a = sample[ind+2]-sample[ind]; b = Point2(x,y)-sample[ind]  ; }
      if(ind%2==0)
      {
        if(a.cross(b)<0) dist = -dist;
      }
      else
      {
        if(a.cross(b)>0) dist = -dist;
      }
    }

    if(close_trailing_edge) return dist;
    else                    return MIN(dist, -x+(naca_length+x_edge));
  }
};

int main()
{
#ifdef CASL_OPENMP
  (void) omp_set_dynamic(FALSE);
  if (omp_get_dynamic()) cout << "Warning: dynamic adjustment of threads has been set" << endl;
  (void) omp_set_num_threads(1);
  cout << "OpenMP - Max number of threads : " << omp_get_max_threads() << endl;
#endif

#ifdef CASL_THROWS
  cout << "CASL_THROWS on" << endl;
#endif

  StopWatch watch;
  watch.start();

  NACA naca(naca_number, 4, naca_angle);

  char name[1000];
  if(save_sampling)
  {
    sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d.csv", naca_number, lmax, sample_size);
    naca.save_sampling(name);
  }

  QuadTree tr;
  tr.set_Grid(0, 32, -8, 8);
  tr.construct_Quadtree_From_Level_Function(naca, 0, lmax);
  tr.initialize_Neighbors();

  ArrayV<double> phi(tr.number_Of_Nodes());
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(CaslInt n=0; n<tr.number_Of_Nodes(); ++n)
    phi[n] = naca(tr.x_fr_n(n), tr.y_fr_n(n));

  watch.stop_And_Read_Duration("building tree");

  watch.start();
  QuadTreeLevelSet ls(tr, phi);
  ls.reinitialize(TVD, 100);
  ls.perturb_Level_Function(EPSILON);

  watch.stop_And_Read_Duration("reinitializing");
  if(close_trailing_edge) sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_closed_tree.dat", naca_number, lmax, sample_size);
  else                    sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_tree.dat"       , naca_number, lmax, sample_size);
  tr.save(name);

  if(close_trailing_edge) sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_closed_phi.dat", naca_number, lmax, sample_size);
  else                    sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_phi.dat"       , naca_number, lmax, sample_size);
  phi.save(name);

  if(close_trailing_edge) sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_closed.vtk", naca_number, lmax, sample_size);
  else                    sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d.vtk"       , naca_number, lmax, sample_size);
  tr.print_VTK_Format(name);
  tr.print_VTK_Format(phi, "phi", name);

//  QuadraticInterpolationOnQuadTree interp(tr, phi);

//  QuadTree tr2;
//  tr2.set_Grid(0, 32, -8, 8);
//  tr2.construct_Quadtree_From_Level_Function(interp, 0, lmax);
//  tr2.initialize_Neighbors();

//  ArrayV<double> phi2(tr2.number_Of_Nodes());
//  for(CaslInt n=0; n<tr2.number_Of_Nodes(); ++n)
//    phi2[n] = interp(tr2.x_fr_n(n), tr2.y_fr_n(n));

//  std::cout << "done building tree" << std::endl;

//  QuadTreeLevelSet ls2(tr2, phi2);
//  ls2.reinitialize(TVD, 100);

//  std::cout << "done reinitializing" << std::endl;

//  sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_tree.dat", naca_number, lmax, sample_size);
//  tr2.save(name);
//  sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d_phi.dat", naca_number, lmax, sample_size);
//  phi2.save(name);

//  sprintf(name, "/home/guittet/code/data/casl_naca/naca_%04d_lmax%d_sampling%d.vtk", naca_number, lmax, sample_size);
//  tr2.print_VTK_Format(name);
//  tr2.print_VTK_Format(phi2, "phi", name);

  return 0;
}
