/*
 * Title: electroporation
 * Description:
 * Author:
 * Date Created: 09-22-2016
 */
/*
 * Some general notes on using voro++:
 * in 3D, we use the voro++ library to construct the voronoi mesh. The voro++ library should
 * be configured to account for the double precision numbers as well as decreasing the minimum
 * threshold for the definiton of tolerance. I get good results with tolerance=1e-13.
 * It is generally a good idea to turn on the VERBOSE mode in the voro++ library.
 * REMARK: there seems to be an issue while passing the xyz_min and xyz_max vectors to the Voronoi3D.cpp file.
 * Currently, I am enforcing the size everytime by hand as I don't have time to trace it! Use this at your own risk!
 * Specifically, in the construct_partition routine, the dimensions of the domain seem to be 1e-13 in all directions
 * if not brute forced to correct values. REMEMBER TO FIX THIS LATER!
*/
#define P4_TO_P8

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/voronoi2D.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_electroporation_solve.h>
#else
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_electroporation_solve.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#include <src/point3.h>
#include <src/voronoi3D.h>
#include <src/my_p8est_level_set.h>
#endif
//#include "nearpt3/nearpt3.cc"
#include <src/Parser.h>
#include <src/math.h>


#include "Halton/halton.cpp"

using namespace std;



int test = 9; //dynamic linear case=2, dynamic nonlinear case=4, static linear case=1, random spheroid=9

double cellDensity = 0.005;   // only if test = 8 || 9

double boxSide = 1e-3;      // only if test = 8
double alpha = 1;        // this is the scaling factor 1e-3, don't use! set to 1.


double omega = 1;  //example: w= 1e9 = 1 GHz angular frequency
double epsilon_0 = 8.85e-12; // farad/meter: permitivity in vacuum
/* 0 or 1 */
int implicit = 0;
/* order 1, 2 or 3. If choosing 3, implicit only */
int order = 1;

/* cell radius */
double r0 = test==5 ? 46e-6/alpha : (test==6 ? 53e-6/alpha : ((test==8 || test==9) ? 7e-6/alpha :50e-6/alpha));
double ellipse = test<5 ? 1 : (test==5 ? 1.225878312944962 : 1.250835468987754);
double a = test<5 ? r0 : (test==5 ? r0*ellipse : r0/ellipse);
double b = test<5 ? r0 : (test==5 ? r0*ellipse : r0/ellipse);
double c = test<5 ? r0 : (test==5 ? r0/ellipse : r0*ellipse);


double boxVolume = boxSide*boxSide*boxSide;
double ClusterRadius = 0.49*boxSide;
double SphereVolume = 4*PI*(ClusterRadius*ClusterRadius*ClusterRadius)/3;
double coeff = 1.;
double cellVolume = 4*PI*(coeff*r0)*(coeff*r0)*(coeff*r0)/3;
// 30 is the safety coefficient to avoid too-close cells corresponding to a minimum radius of ~3*r0



/* number of cells in x and y dimensions */
int x_cells = 4;
int y_cells = 4;
int z_cells = 4;
/* number of random cells */
int nb_cells = test==7 ? 64 : ((test==8 || test==9) ? int (cellDensity*SphereVolume/cellVolume) : x_cells*y_cells*z_cells);


double xmin = test<4 ? -2*x_cells*r0 :  (test == 7 ? -4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9) ? -boxSide/2 : -4*x_cells*r0));
double xmax = test<4 ?  2*x_cells*r0 :  (test == 7 ?  4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9) ?  boxSide/2 :  4*x_cells*r0));
double ymin = test<4 ? -2*y_cells*r0 :  (test == 7 ? -4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9) ? -boxSide/2 : -4*y_cells*r0));
double ymax = test<4 ?  2*y_cells*r0 :  (test == 7 ?  4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9) ?  boxSide/2 :  4*y_cells*r0));
double zminn = test<4 ? -2*z_cells*r0 :  (test == 7 ? -4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9) ? -boxSide/2 : -4*z_cells*r0));
double zmaxx = test<4 ?  2*z_cells*r0 :  (test == 7 ?  4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9) ?  boxSide/2 :  4*z_cells*r0));


int axial_nb = boxSide/r0/2;
int lmax_thr = (int)log2(axial_nb)+1;
int lmin = 2;
int lmax = MAX(lmax_thr, 7);
int nb_splits = 1;

double dt_scale = 200;
double tn;
double tf = 1.e-6;
double dt;

double E_unscaled = 40; /* applied electric field on the top electrode: kv/m */
double E = E_unscaled * 1e3 * ((zmaxx-zminn)*alpha)*alpha;  // this is the potential difference in SI units!

double sigma_c = 1;
double sigma_e = 15;

double Cm = test==1 ? 0 : 9.5e-3*alpha;
double SL = 1.9*alpha;
double S0 = 1.1e6*alpha;
double S1 = 1e4*alpha;
double X_0 = 0;
double X_1 = 0;

double Vep = 258e-3;
double Xep = 0.5;

double tau_ep   = 1e-6;
double tau_perm = 1e-6;//80*tau_ep;  with the modified X2 equation we need this one! Not here.
double tau_res  = 60;

double R1 = .25*MIN(xmax-xmin, ymax-ymin, zmaxx-zminn);
double R2 = 3*MAX(xmax-xmin, ymax-ymin, zmaxx-zminn);

bool save_vtk = true;
bool save_error = false;
int save_every_n = 1;
bool save_voro = false;
bool save_stats = true;
bool check_partition = false;
bool save_impedance = true;
bool save_hierarchy = false;





class LevelSet : public CF_3
{
public:
    vector<double> radii;
    vector<Point3> centers;
    vector<Point3> ex;
    vector<Point3> theta;
    double cellVolumes;
    double density = 0;

    LevelSet()
    {
        lip=1.2;
        if(test==7 || test==8 || test==9)
        {
            centers.resize(nb_cells);
            radii.resize(nb_cells);
            ex.resize(nb_cells);
            theta.resize(nb_cells);
            unsigned int seed = time(NULL);
            srand(seed);
            printf("The random seed is %u\n", seed);
            printf("number of cells is %u\n", nb_cells);
            fflush(stdout);
            std::vector<std::array<double,3> > v;
            std::array<double,3> p;
            double Radius=0;
            double azimuth = 0;
            double polar =0;

            double *r;
            int halton_counter = 0;
            r = halton(halton_counter,3);

            if(test==9){
                double azimuth = 0;
                double polar = 0;
                Radius = 0.49*(xmax-xmin)*r[0];
                azimuth = 2*PI*r[1];
                polar = PI*r[2];

                p[0] = Radius*sin(polar)*cos(azimuth);
                p[1] = Radius*sin(polar)*sin(azimuth);
                p[2] = Radius*cos(polar);
            } else {
                p[0] = 0.99*(xmax-xmin)*(r[0] - 0.5);
                p[1] = 0.99*(ymax-ymin)*(r[1] - 0.5);
                p[2] = 0.99*(zmaxx-zminn)*(r[2] - 0.5);
            }
            halton_counter++;
            v.push_back(p);
            int progress = 0;

            do
            {

                r = halton(halton_counter,3);

                if(test==9){
                    double azimuth = 0;
                    double polar = 0;
                    Radius = 0.49*(xmax-xmin)*r[0];
                    azimuth = 2*PI*r[1];
                    polar = PI*r[2];

                    p[0] = Radius*sin(polar)*cos(azimuth);
                    p[1] = Radius*sin(polar)*sin(azimuth);
                    p[2] = Radius*cos(polar);
                } else {
                    p[0] = 0.99*(xmax-xmin)*(r[0] - 0.5);
                    p[1] = 0.99*(ymax-ymin)*(r[1] - 0.5);
                    p[2] = 0.99*(zmaxx-zminn)*(r[2] - 0.5);
                }
                halton_counter++;
                bool far_enough = true;
                for(int ii=0;ii<v.size();++ii){
                    double mindist = sqrt(SQR(p[0]-v[ii][0])+ SQR(p[1]-v[ii][1])+SQR(p[2]-v[ii][2]));
                    if(mindist<1.5*r0){
                        far_enough = false;
                        break;

                    }
                }
                if(far_enough){
                    v.push_back(p);
                    //cellVolumes += 4*PI*(2*r0)*(2*r0)*(2*r0)/3;
                    if(v.size()%((int) nb_cells/10) == 0){
                        progress += 10;
                        printf("Cell Placement is in Progress. Currently at: %d %\n", progress);
                    }
                }
            }while(v.size()<nb_cells);

            cellVolumes = 0;
            for(int n=0; n<nb_cells; ++n)
            {
                centers[n].x = v[n][0];
                centers[n].y = v[n][1];
                centers[n].z = v[n][2];
                radii[n] = r0 + 1e-6*(6*((double)rand()/RAND_MAX) - 3);
                ex[n].x = 1 + .4*(double)rand()/RAND_MAX - .2;
                ex[n].y = 1 + .4*(double)rand()/RAND_MAX - .2;
                ex[n].z = 1 + .4*(double)rand()/RAND_MAX - .2;
                theta[n].x = PI*(double)rand()/RAND_MAX;
                theta[n].y = PI*(double)rand()/RAND_MAX;
                theta[n].z = PI*(double)rand()/RAND_MAX;
                cellVolumes += 4*PI*radii[n]*radii[n]*radii[n]*ex[n].x*ex[n].y*ex[n].z/3;
            }

            density = cellVolumes/SphereVolume;
            printf( "Done initializing random cells. The Cell volume density is = %g\n", density);
            fflush(stdout);

        }
    }

    double operator()(double x, double y, double z) const
    {
        double d = DBL_MAX;

        double xm, ym, zm; xm=xmin; ym=ymin; zm=zminn;
        double dx = xmax/(x_cells+1);
        double dy = ymax/(y_cells+1);
        double dz = zmaxx/(z_cells+1);
        double x_tmp, y_tmp, z_tmp;
        double x0, y0, z0;

        switch(test)
        {
        case 0: return sqrt(SQR(x) + SQR(y) + SQR(z)) - r0;
        case 1: return sqrt(SQR(x) + SQR(y) + SQR(z)) - R1;
        case 2: return sqrt(SQR(x) + SQR(y) + SQR(z)) - R1;
        case 3:
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    for(int k=0; k<z_cells; ++k)
                        d = MIN(d, sqrt(SQR(x-(xmin+i*4*r0+2*r0)) + SQR(y-(ymin+j*4*r0+2*r0)) + SQR(z-(zminn+k*4*r0+2*r0))) - r0);
            return d;
        case 4: return sqrt(SQR(x) + SQR(y) + SQR(z)) - R1;
        case 5:
        case 6:
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    for(int k=0; k<z_cells; ++k)
                        d = MIN(d, sqrt(SQR((x-(xm+(i+1)*dx))*r0/a) + SQR((y-(ym+(j+1)*dy))*r0/b) + SQR((z-(zm+(k+1)*dz))*r0/c)) - r0);
            return d;
        case 7:
            for(int n=0; n<nb_cells; ++n)
            {
                x0 = x - centers[n].x;
                y0 = y - centers[n].y;
                z0 = z - centers[n].z;

                x_tmp = x0;
                y_tmp = cos(theta[n].x)*y0 - sin(theta[n].x)*z0;
                z_tmp = sin(theta[n].x)*y0 + cos(theta[n].x)*z0;

                x0 = cos(theta[n].y)*x_tmp - sin(theta[n].y)*z_tmp;
                y0 = y_tmp;
                z0 = sin(theta[n].y)*x_tmp + cos(theta[n].y)*z_tmp;

                x_tmp = cos(theta[n].z)*x0 - sin(theta[n].z)*y0;
                y_tmp = sin(theta[n].z)*x0 + cos(theta[n].z)*y0;
                z_tmp = z0;

                d = MIN(d, sqrt(SQR(x_tmp/ex[n].x) + SQR(y_tmp/ex[n].y) + SQR(z_tmp/ex[n].z)) - radii[n]);
            }
            return d;
        case 8:
            for(int n=0; n<nb_cells; ++n)
            {
                x0 = x - centers[n].x;
                y0 = y - centers[n].y;
                z0 = z - centers[n].z;

                x_tmp = x0;
                y_tmp = cos(theta[n].x)*y0 - sin(theta[n].x)*z0;
                z_tmp = sin(theta[n].x)*y0 + cos(theta[n].x)*z0;

                x0 = cos(theta[n].y)*x_tmp - sin(theta[n].y)*z_tmp;
                y0 = y_tmp;
                z0 = sin(theta[n].y)*x_tmp + cos(theta[n].y)*z_tmp;

                x_tmp = cos(theta[n].z)*x0 - sin(theta[n].z)*y0;
                y_tmp = sin(theta[n].z)*x0 + cos(theta[n].z)*y0;
                z_tmp = z0;

                d = MIN(d, sqrt(SQR(x_tmp/ex[n].x) + SQR(y_tmp/ex[n].y) + SQR(z_tmp/ex[n].z)) - radii[n]);
            }
            return d;
        case 9:
            for(int n=0; n<nb_cells; ++n)
            {
                x0 = x - centers[n].x;
                y0 = y - centers[n].y;
                z0 = z - centers[n].z;

                x_tmp = x0;
                y_tmp = cos(theta[n].x)*y0 - sin(theta[n].x)*z0;
                z_tmp = sin(theta[n].x)*y0 + cos(theta[n].x)*z0;

                x0 = cos(theta[n].y)*x_tmp - sin(theta[n].y)*z_tmp;
                y0 = y_tmp;
                z0 = sin(theta[n].y)*x_tmp + cos(theta[n].y)*z_tmp;

                x_tmp = cos(theta[n].z)*x0 - sin(theta[n].z)*y0;
                y_tmp = sin(theta[n].z)*x0 + cos(theta[n].z)*y0;
                z_tmp = z0;

                d = MIN(d, sqrt(SQR(x_tmp/ex[n].x) + SQR(y_tmp/ex[n].y) + SQR(z_tmp/ex[n].z)) - radii[n]);
            }
            return d;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} level_set;

double u_exact(double x, double y, double z, double t, bool phi_is_pos)
{
    double SR1 = R1*alpha;
    double SR2 = R2*alpha;
    double SCm = Cm/alpha;
    double SSL= SL/alpha;


    double r = alpha*sqrt(x*x + y*y + z*z);
    double theta = atan2(sqrt(x*x+y*y),z);
    double g = E*R2;


    if(test==1)
    {
        double K = 1/(SR1*SR1*SR1*(sigma_e-sigma_c)+SR2*SR2*SR2*(2*sigma_e+sigma_c));

        double A = 3*sigma_c*sigma_e*SR2*SR2*K;
        double B = -sigma_c*sigma_e*(SR1*SR1 + 2*SR2*SR2*SR2/SR1)*K;
        double vv = A*g / (SSL - B);

        double alpha_e = SR2*SR2*(sigma_c+2*sigma_e)*K*g - SR1*SR1*sigma_c*K*vv;
        double beta_e = SR1*SR1*SR1*SR2*SR2*(sigma_e-sigma_c)*K*g + SR1*SR1*SR2*SR2*SR2*sigma_c*K*vv;
        double alpha_c = 3*sigma_e*SR2*SR2*K*g - sigma_e*(SR1*SR1+2*SR2*SR2*SR2/SR1)*K*vv;

        return phi_is_pos ? (alpha_e*r+beta_e/(r*r))*cos(theta) : alpha_c*r*cos(theta);
    }
    if(test==2)
    {
        double K = 1/(SR1*SR1*SR1*(sigma_e-sigma_c)+SR2*SR2*SR2*(2*sigma_e+sigma_c));

        double A = 3*sigma_c*sigma_e*SR2*SR2*K;
        double B = -sigma_c*sigma_e*(SR1*SR1 + 2*SR2*SR2*SR2/SR1)*K;
        double vv = A/(SSL-B)*g*(1-exp((B-SSL)/SCm*t));

        double alpha_e = SR2*SR2*(sigma_c+2*sigma_e)*K*g - SR1*SR1*sigma_c*K*vv;
        double beta_e = SR1*SR1*SR1*SR2*SR2*(sigma_e-sigma_c)*K*g + SR1*SR1*SR2*SR2*SR2*sigma_c*K*vv;
        double alpha_c = 3*sigma_e*SR2*SR2*K*g - sigma_e*(SR1*SR1+2*SR2*SR2*SR2/SR1)*K*vv;

        return phi_is_pos ? (alpha_e*r + beta_e/(r*r))*cos(theta) : alpha_c*r*cos(theta);
    }
    else
        return 0;
}

//PAM: square pulse to be asked from Clair
double pulse(double tn)
{

    return E*cos(omega*tn);
    //    int cycle = int(tn/half_period);
    //    if(cycle%2 == 0)
    //        return E;
    //    else
    //        return -E;
}

// rescale later!
double v_exact(double x, double y, double z, double t)
{
    double theta = atan2(sqrt(x*x+y*y),z);
    double g = E*R2;

    if(test==1)
    {
        double K = 1/(R1*R1*R1*(sigma_e-sigma_c)+R2*R2*R2*(2*sigma_e+sigma_c));

        double A = 3*sigma_c*sigma_e*R2*R2*K;
        double B = -sigma_c*sigma_e*(R1*R1 + 2*R2*R2*R2/R1)*K;

        return A/(SL-B)*g*cos(theta);
    }
    if(test==2)
    {
        double K = 1/(R1*R1*R1*(sigma_e-sigma_c)+R2*R2*R2*(2*sigma_e+sigma_c));

        double A = 3*sigma_c*sigma_e*R2*R2*K;
        double B = -sigma_c*sigma_e*(R1*R1 + 2*R2*R2*R2/R1)*K;

        return A/(SL-B)*g*(1-exp((B-SL)*t/Cm))*cos(theta);
    }
    else
        return 0;
}

struct BCWALLTYPE : WallBC3D
{
    BoundaryConditionType operator()(double x, double y, double z) const
    {
        switch(test)
        {
        case 1:
            return DIRICHLET;
        case 2:
            return DIRICHLET;
        case 3:
        case 4:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
            //return DIRICHLET;
        case 5:
        case 6:
        case 7:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 8:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 9:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} bc_wall_type_p;

struct BCWALLVALUE : CF_3
{
    double operator()(double x, double y, double z) const
    {
        switch(test)
        {
        case 1:
            return u_exact(x,y,z,0,true);
        case 2:
            return u_exact(x,y,z,t+dt,true);
        case 3:
        case 4:
            if(ABS(z-zminn)<EPS) return 0;
            if(ABS(z-zmaxx)<EPS) return E;
            return 0;
        case 5:
        case 6:
        case 7:
            if(ABS(z-zminn)<EPS) return 0;
            if(ABS(z-zmaxx)<EPS) return E;
            return 0;
        case 8:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
            return 0;
        case 9:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
            return 0;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} bc_wall_value_p;







double sigma(double x, double y, double z)
{
    return level_set(x,y,z)<0 ? sigma_c : sigma_e;
}

class SIGMA : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        return level_set(x,y,z)<0 ? sigma_c : sigma_e;
    }
}sigma_in;

class BETA_0 : public CF_1
{
public:
    double operator ()(double lambda) const
    {
        return exp(-Vep*Vep/(lambda*lambda));
    }
}beta_0_in;

class BETA_1 : public CF_1
{
public:
    double operator ()(double lambda) const
    {
        return exp(-Xep*Xep/(lambda*lambda));
    }
}beta_1_in;




class Initial_U : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} initial_u;



class Initial_Vn : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} initial_vn;

//Clair: scale!
class Initial_Vnm1 : public CF_3
{

public:
    double operator()(double x, double y, double z) const
    {
        if (test==2)
        {

            double K = 1/(R1*R1*R1*(sigma_e-sigma_c)+R2*R2*R2*(2*sigma_e+sigma_c));
            double A = 3*sigma_c*sigma_e*R2*R2*K;
            double B = -sigma_c*sigma_e*(R1*R1 + 2*R2*R2*R2/R1)*K;
            double g = E*R2;
            double theta = atan2(sqrt(x*x+y*y),z);

            if(order>1) return A/(SL-B)*g*(1-exp(-(SL-B)/Cm*(-dt)))*cos(theta);
            else        return 0;
        }
        else        return 0;
    }
} initial_vnm1;


class Initial_Vnm2 : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        if (test==4)
        {

            double K = 1/(R1*R1*R1*(sigma_e-sigma_c)+R2*R2*R2*(2*sigma_e+sigma_c));
            double A = 3*sigma_c*sigma_e*R2*R2*K;
            double B = -sigma_c*sigma_e*(R1*R1 + 2*R2*R2*R2/R1)*K;
            double g = E*R2;
            double theta = atan2(sqrt(x*x+y*y),z);

            if(order>2) return A/(SL-B)*g*(1-exp(-(SL-B)/Cm*(-2*dt)))*cos(theta);
            else        return 0;
        }
        else        return 0;
    }
} initial_vnm2;


class Initial_X0 : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return X_0;
    }
} initial_x0;

class Initial_X1 : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return X_1;
    }
} initial_x1;

class Initial_Sm : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return SL;
    }
} initial_sm;











class MU_M: public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return sigma_c;
    }
} mu_m;

class MU_P: public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return sigma_e;
    }
} mu_p;

struct U_M : CF_3
{
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} u_m;

struct U_P : CF_3
{
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} u_p;

class U_JUMP: public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} u_jump;



double grad_u_m(double x, double y, double z)
{
    return 0;
}

double grad_u_p(double x, double y, double z)
{
    return 0;
}


class MU_GRAD_U_JUMP: public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return mu_p(x,y,z)*grad_u_p(x,y,z) - mu_m(x,y,z)*grad_u_m(x,y,z);
    }
} mu_grad_u_jump;



void solve_Poisson_Jump( p4est_t *p4est, p4est_nodes_t *nodes,
                         my_p4est_node_neighbors_t *ngbd_n, my_p4est_cell_neighbors_t *ngbd_c,
                         Vec phi, Vec sol, double dt, Vec X0, Vec X1, Vec Sm, Vec vn, my_p4est_level_set_t ls, double tn, Vec vnm1, Vec vnm2, Vec grad_phi[3], double diag)
{
    PetscErrorCode ierr;

    Vec rhs_m, rhs_p;
    Vec mu_m_, mu_p_;
    Vec u_jump_;
    Vec mu_grad_u_jump_;
    ierr = VecDuplicate(phi, &rhs_m); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs_p); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_m_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_p_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &u_jump_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_grad_u_jump_); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mu_m, mu_m_);
    sample_cf_on_nodes(p4est, nodes, mu_p, mu_p_);
    sample_cf_on_nodes(p4est, nodes, u_jump, u_jump_);
    sample_cf_on_nodes(p4est, nodes, mu_grad_u_jump, mu_grad_u_jump_);





    double *rhs_m_p, *rhs_p_p;
    ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        rhs_m_p[n] = 0;
        rhs_p_p[n] = 0;
    }
    ierr = VecRestoreArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);


#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif
    bc_wall_value_p.t = tn+dt;
    bc.setWallTypes(bc_wall_type_p);
    bc.setWallValues(bc_wall_value_p);

    my_p4est_electroporation_solve_t solver(ngbd_n, ngbd_c);




    solver.set_parameters(implicit, order, dt, test, SL, tau_ep, tau_res, tau_perm, S0, S1, tn);
    solver.dt = dt;
    solver.Cm = Cm;
    solver.set_phi(phi);
    solver.set_grad_phi(grad_phi);
    solver.set_bc(bc);
    solver.set_mu(mu_m_, mu_p_);
    solver.set_rhs(rhs_m, rhs_p);
    solver.set_beta0(beta_0_in);
    solver.set_beta1(beta_1_in);

    Vec X_0_v, X_1_v, l, l0, l1;
    ierr = VecDuplicate(phi, &X_0_v); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &X_1_v); CHKERRXX(ierr);

    //    ierr = VecGhostGetLocalForm(X_0_v, &l); CHKERRXX(ierr);
    //    ierr = VecGhostGetLocalForm(X0, &l0); CHKERRXX(ierr);
    //    ierr = VecCopy(l0, l); CHKERRXX(ierr);
    //    ierr = VecGhostRestoreLocalForm(X_0_v, &l); CHKERRXX(ierr);
    //    ierr = VecGhostRestoreLocalForm(X0, &l0); CHKERRXX(ierr);

    //    ierr = VecGhostGetLocalForm(X_1_v, &l); CHKERRXX(ierr);
    //    ierr = VecGhostGetLocalForm(X1, &l1); CHKERRXX(ierr);
    //    ierr = VecCopy(l1, l); CHKERRXX(ierr);
    //    ierr = VecGhostRestoreLocalForm(X_1_v, &l); CHKERRXX(ierr);
    //    ierr = VecGhostRestoreLocalForm(X1, &l1); CHKERRXX(ierr);
    double convergence_Sm;

    int counter = 0;



    //my_p4est_interpolation_nodes_t interp_n(ngbd_n);
    //double xyz_np[3] = {0, 0, R1};
    //interp_n.add_point(0, xyz_np);
    double Sm_err = 0;
    //double convergence_Sm_old;
    // interp_n.set_input(Sm, linear);
    //interp_n.interpolate(&convergence_Sm_old);


    convergence_Sm = 0;
    solver.set_vn(vn);
    solver.set_vnm1(vnm1);
    solver.set_vnm2(vnm2);

    Vec vnp1;
    VecDuplicate(sol, &vnp1);

    double *grad_phi_p[P4EST_DIM];
    for(int j=0;j<P4EST_DIM;++j)
        VecGetArray(grad_phi[j], &grad_phi_p[j]);



    do
    {
        solver.set_Sm(Sm);
        solver.solve(sol);

        //        solver.set_X0(X0);
        //        solver.set_X1(X1);
        //        solver.compute_electroporation();
        //        solver.interpolate_electroporation_to_tree(X0, X1, Sm, vn);
        //        Vec vn_tmp;
        //        double *vn_p;
        //        VecDuplicate(vn,&vn_tmp);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, vn, vn_tmp);
        //        double *vn_tmp_p;
        //        VecGetArray(vn_tmp, &vn_tmp_p);
        //        VecGetArray(vn, &vn_p);
        //        for (size_t n = 0; n<nodes->indep_nodes.elem_count; n++)
        //        {
        //            vn_p[n] = vn_tmp_p[n];
        //        }
        //        VecRestoreArray(vn_tmp, &vn_tmp_p);
        //        VecRestoreArray(vn, &vn_p);


        // PAM BEGIN
        Vec u_plus, u_minus, u_plus_l, u_minus_l, sol_l;
        ierr = VecDuplicate(sol, &u_plus); CHKERRXX(ierr);
        ierr = VecDuplicate(sol, &u_minus); CHKERRXX(ierr);
        VecGhostGetLocalForm(sol, &sol_l);
        VecGhostGetLocalForm(u_plus, &u_plus_l);
        VecGhostGetLocalForm(u_minus, &u_minus_l);
        ierr = VecCopy(sol_l, u_plus_l); CHKERRXX(ierr);
        ierr = VecCopy(sol_l, u_minus_l); CHKERRXX(ierr);
        VecGhostRestoreLocalForm(sol, &sol_l);
        VecGhostRestoreLocalForm(u_plus, &u_plus_l);
        VecGhostRestoreLocalForm(u_minus, &u_minus_l);
        double *phi_p;
        VecGetArray(phi, &phi_p);
        ls.extend_Over_Interface_TVD(phi, u_minus,100);
        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
            phi_p[i] = -phi_p[i];
        ls.extend_Over_Interface_TVD(phi, u_plus,100);
        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
            phi_p[i] = -phi_p[i];
        Vec grad_up, grad_um;
        double *dup_p,*dum_p, *up_p, *um_p;
        ierr = VecGetArray(u_plus, &up_p); CHKERRXX(ierr);
        ierr = VecGetArray(u_minus, &um_p); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &grad_up); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &grad_um); CHKERRXX(ierr);
        ierr = VecGetArray(grad_up, &dup_p); CHKERRXX(ierr);
        ierr = VecGetArray(grad_um, &dum_p); CHKERRXX(ierr);
        for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
        {
            p4est_locidx_t n = ngbd_n->get_layer_node(i);
            const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
            dup_p[n] = qnnn.dx_central(up_p)*grad_phi_p[0][n] + qnnn.dy_central(up_p)*grad_phi_p[1][n] + qnnn.dz_central(up_p)*grad_phi_p[2][n];
            dum_p[n] = qnnn.dx_central(um_p)*grad_phi_p[0][n] + qnnn.dy_central(um_p)*grad_phi_p[1][n] + qnnn.dz_central(um_p)*grad_phi_p[2][n];
        }
        ierr = VecGhostUpdateBegin(grad_up, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(grad_um, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
        {
            p4est_locidx_t n = ngbd_n->get_local_node(i);
            const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
            dup_p[n] = qnnn.dx_central(up_p)*grad_phi_p[0][n] + qnnn.dy_central(up_p)*grad_phi_p[1][n] + qnnn.dz_central(up_p)*grad_phi_p[2][n];
            dum_p[n] = qnnn.dx_central(um_p)*grad_phi_p[0][n] + qnnn.dy_central(um_p)*grad_phi_p[1][n] + qnnn.dz_central(um_p)*grad_phi_p[2][n];
        }
        ierr = VecGhostUpdateEnd(grad_up, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(grad_um, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecRestoreArray(grad_up, &dup_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(grad_um, &dum_p); CHKERRXX(ierr);

        Vec du_plus_cte, du_minus_cte;
        ierr = VecDuplicate(sol,&du_plus_cte); CHKERRXX(ierr);
        ls.extend_from_interface_to_whole_domain_TVD(phi, grad_up, du_plus_cte);
        ierr = VecDuplicate(sol,&du_minus_cte); CHKERRXX(ierr);
        ls.extend_from_interface_to_whole_domain_TVD(phi, grad_um, du_minus_cte);
        double *du_plus_cte_p, *du_minus_cte_p;
        VecGetArray(du_plus_cte, &du_plus_cte_p);
        VecGetArray(du_minus_cte, &du_minus_cte_p);

        double *Sm_p, *vn_p, *vnp1_p;
        VecGetArray(vnp1, &vnp1_p);
        VecGetArray(vn, &vn_p);
        VecGetArray(Sm, &Sm_p);
        for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
        {
            vnp1_p[n] = (Cm*vn_p[n] + dt*(sigma_c*du_minus_cte_p[n] + sigma_e*du_plus_cte_p[n])/2.)/(Cm + dt*Sm_p[n]);
        }
        VecRestoreArray(Sm, &Sm_p);
        VecRestoreArray(vn, &vn_p);
        VecRestoreArray(vnp1, &vnp1_p);
        VecRestoreArray(du_plus_cte, &du_plus_cte_p);
        VecRestoreArray(du_minus_cte, &du_minus_cte_p);
        //PAM END



        // Daniil Begin
        //        // compute jump
        //        // make 2 other copies of the solution vector
        //        Vec u_plus_ext, u_minus_ext, u_plus_ext_l, u_minus_ext_l, sol_l;
        //        ierr = VecDuplicate(sol, &u_plus_ext); CHKERRXX(ierr);
        //        ierr = VecDuplicate(sol, &u_minus_ext); CHKERRXX(ierr);
        //        VecGhostGetLocalForm(sol, &sol_l);
        //        VecGhostGetLocalForm(u_plus_ext, &u_plus_ext_l);
        //        VecGhostGetLocalForm(u_minus_ext, &u_minus_ext_l);
        //        ierr = VecCopy(sol_l, u_plus_ext_l); CHKERRXX(ierr);
        //        ierr = VecCopy(sol_l, u_minus_ext_l); CHKERRXX(ierr);
        //        VecGhostRestoreLocalForm(sol, &sol_l);
        //        VecGhostRestoreLocalForm(u_plus_ext, &u_plus_ext_l);
        //        VecGhostRestoreLocalForm(u_minus_ext, &u_minus_ext_l);

        //        // project solutions onto the interface
        //        double *phi_p;
        //        VecGetArray(phi, &phi_p);

        //        ls.extend_Over_Interface_TVD(phi, u_plus_ext);
        //        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
        //            phi_p[i] = -phi_p[i];
        //        ls.extend_Over_Interface_TVD(phi, u_minus_ext);
        //        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
        //            phi_p[i] = -phi_p[i];




        //        Vec u_plus_cte, u_minus_cte;
        //        ierr = VecDuplicate(u_plus_ext,&u_plus_cte); CHKERRXX(ierr);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, u_plus_ext, u_plus_cte);
        //        ierr = VecDuplicate(u_minus_ext,&u_minus_cte); CHKERRXX(ierr);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, u_minus_ext, u_minus_cte);

        //        // subtract the projected vectors, on the interface it is the jump.
        //        double *vnp1_p, *u_minus_cte_p, *u_plus_cte_p;
        //        VecGetArray(u_minus_cte, &u_minus_cte_p);
        //        VecGetArray(u_plus_cte, &u_plus_cte_p);


        //        VecGetArray(vnp1, &vnp1_p);
        //        double *sol_p;
        //        VecGetArray(sol, &sol_p);
        //        for (size_t n = 0; n<nodes->indep_nodes.elem_count; n++)
        //        {
        //            vnp1_p[n] = u_minus_cte_p[n] - u_plus_cte_p[n];
        //        }
        //        VecRestoreArray(sol, &sol_p);
        //        VecRestoreArray(phi, &phi_p);
        //        VecRestoreArray(vnp1, &vnp1_p);
        //        VecRestoreArray(u_minus_cte, &u_minus_cte_p);
        //        VecRestoreArray(u_plus_cte, &u_plus_cte_p);
        //        Vec vn_tmp;
        //        VecDuplicate(phi,&vn_tmp);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, vnp1, vn_tmp);
        //        double *vn_tmp_p;
        //        VecGetArray(vn_tmp, &vn_tmp_p);
        //        VecGetArray(vnp1, &vnp1_p);
        //        for (size_t n = 0; n<nodes->indep_nodes.elem_count; n++)
        //        {
        //            vnp1_p[n] = vn_tmp_p[n];
        //        }
        //        VecRestoreArray(vn_tmp, &vn_tmp_p);
        //        VecRestoreArray(vnp1, &vnp1_p);
        //        ierr = VecDestroy(u_plus_ext); CHKERRXX(ierr);
        //        ierr = VecDestroy(u_minus_ext); CHKERRXX(ierr);
        //        ierr = VecDestroy(u_plus_cte); CHKERRXX(ierr);
        //        ierr = VecDestroy(u_minus_cte); CHKERRXX(ierr);
        //        VecDestroy(vn_tmp);
        // Daniil end













        if(check_partition)
            solver.check_voronoi_partition();

        char out_path[1000];
        char *out_dir = NULL;
        out_dir = getenv("OUT_DIR");
        if(out_dir==NULL)
        {
            ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
        }
        else
        {
            if(save_stats)
            {
                sprintf(out_path, "%s/stats.dat", out_dir);
                solver.write_stats(out_path);
            }
            if(save_voro)
            {
                snprintf(out_path,1000, "%s/voronoi", out_dir);
                solver.print_voronoi_VTK(out_path);
            }
        }


        //        Vec u_plus_ext, u_minus_ext, u_plus_ext_l, u_minus_ext_l, vn_l;
        //        ierr = VecDuplicate(vn, &u_plus_ext); CHKERRXX(ierr);
        //        ierr = VecDuplicate(vn, &u_minus_ext); CHKERRXX(ierr);
        //        VecGhostGetLocalForm(vn, &vn_l);
        //        VecGhostGetLocalForm(u_plus_ext, &u_plus_ext_l);
        //        VecGhostGetLocalForm(u_minus_ext, &u_minus_ext_l);
        //        ierr = VecCopy(vn_l, u_plus_ext_l); CHKERRXX(ierr);
        //        ierr = VecCopy(vn_l, u_minus_ext_l); CHKERRXX(ierr);
        //        VecGhostRestoreLocalForm(vn, &vn_l);
        //        VecGhostRestoreLocalForm(u_plus_ext, &u_plus_ext_l);
        //        VecGhostRestoreLocalForm(u_minus_ext, &u_minus_ext_l);

        //        //        project solutions onto the interface
        //        double *phi_p;
        //        VecGetArray(phi, &phi_p);

        //        ls.extend_Over_Interface_TVD(phi, u_plus_ext);
        //        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
        //            phi_p[i] = -phi_p[i];
        //        ls.extend_Over_Interface_TVD(phi, u_minus_ext);
        //        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
        //        {
        //            phi_p[i] = -phi_p[i];
        //        }
        //        VecRestoreArray(phi, &phi_p);
        //        double *vn_p, *u_minus_ext_p, *u_plus_ext_p;
        //        VecGetArray(vn, &vn_p);
        //        VecGetArray(u_minus_ext, &u_minus_ext_p);
        //        VecGetArray(u_plus_ext, &u_plus_ext_p);
        //        for (size_t n = 0; n<nodes->indep_nodes.elem_count; n++)
        //        {
        //            vn_p[n] = (u_plus_ext_p[n] + u_minus_ext_p[n])/2;
        //        }
        //        VecRestoreArray(vn, &vn_p);
        //        VecRestoreArray(u_minus_ext, &u_minus_ext_p);
        //        VecRestoreArray(u_plus_ext, &u_plus_ext_p);

        //        ls.extend_from_interface_to_whole_domain_TVD(phi, vn, vn);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, Sm, Sm);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, X0, X0);
        //        ls.extend_from_interface_to_whole_domain_TVD(phi, X1, X1);


        //        compute X and Sm
        if(test==1 || test==2)
        {
            Vec l;
            ierr = VecGhostGetLocalForm(Sm, &l); CHKERRXX(ierr);
            ierr = VecSet(l, SL); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(Sm, &l); CHKERRXX(ierr);

            ierr = VecGhostGetLocalForm(X0, &l); CHKERRXX(ierr);
            ierr = VecSet(l, 0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X0, &l); CHKERRXX(ierr);

            ierr = VecGhostGetLocalForm(X1, &l); CHKERRXX(ierr);
            ierr = VecSet(l, 0); CHKERRXX(ierr);
            ierr = VecGhostRestoreLocalForm(X1, &l); CHKERRXX(ierr);
            Sm_err = 1e-4;
        } else {

            double *vn_n_p, *Sm_n_p, *X0_np1, *X1_np1,*X_0_v_p, *X_1_v_p;
            ierr = VecGetArray(Sm, &Sm_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(vnp1, &vn_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(X0, &X0_np1); CHKERRXX(ierr);
            ierr = VecGetArray(X1, &X1_np1); CHKERRXX(ierr);
            ierr = VecGetArray(X_0_v, &X_0_v_p); CHKERRXX(ierr);
            ierr = VecGetArray(X_1_v, &X_1_v_p); CHKERRXX(ierr);
            double *phi_p;
            VecGetArray(phi, &phi_p);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
            {
                X_0_v_p[n] = X0_np1[n] + dt*((beta_0_in(vn_n_p[n]) - X0_np1[n])/tau_ep);
                X_1_v_p[n] = X1_np1[n] + dt*MAX((beta_1_in(X0_np1[n])-X1_np1[n])/tau_perm, (beta_1_in(X0_np1[n])-X1_np1[n])/tau_res); //dt/tau_perm*(X0_np1[n]-X1_np1[n]);
                Sm_n_p[n] = SL + S0*X_0_v_p[n] + S1*X_1_v_p[n];
            }
            VecRestoreArray(phi, &phi_p);
            ierr = VecRestoreArray(Sm, &Sm_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X0,&X0_np1); CHKERRXX(ierr);
            ierr = VecRestoreArray(X1,&X1_np1); CHKERRXX(ierr);
            ierr = VecRestoreArray(vnp1, &vn_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X_0_v, &X_0_v_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X_1_v, &X_1_v_p); CHKERRXX(ierr);
            counter++;
            // interp_n.set_input(Sm, linear);
            // interp_n.interpolate(&convergence_Sm);
            //Sm_err = ABS(convergence_Sm - convergence_Sm_old)/convergence_Sm_old;
            //PetscPrintf(p4est->mpicomm, "relative error in Sm is %g, old value %g, new value %g\n", Sm_err, convergence_Sm_old, convergence_Sm);
            //convergence_Sm_old = convergence_Sm;
        }


    }while(0 );//&& Sm_err>0.001);


    ierr = VecGhostGetLocalForm(X_0_v, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(X0, &l0); CHKERRXX(ierr);
    ierr = VecCopy(l, l0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X_0_v, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X0, &l0); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(X_1_v, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(X1, &l1); CHKERRXX(ierr);
    ierr = VecCopy(l, l1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X_1_v, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X1, &l1); CHKERRXX(ierr);

    if(order>2)
    {
        Vec vnm2_l, vnm1_l;
        VecGhostGetLocalForm(vnm1, &vnm1_l);
        VecGhostGetLocalForm(vnm2, &vnm2_l);
        ierr = VecCopy(vnm1_l, vnm2_l); CHKERRXX(ierr);
        VecGhostRestoreLocalForm(vnm1, &vnm1_l);
        VecGhostRestoreLocalForm(vnm2, &vnm2_l);
    }
    if(order>1)
    {
        Vec vn_l, vnm1_l;
        VecGhostGetLocalForm(vnm1, &vnm1_l);
        VecGhostGetLocalForm(vn, &vn_l);
        ierr = VecCopy(vn_l, vnm1_l); CHKERRXX(ierr);
        VecGhostRestoreLocalForm(vnm1, &vnm1_l);
        VecGhostRestoreLocalForm(vn, &vn_l);
    }


    ierr = VecGhostGetLocalForm(vnp1, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vn, &l1); CHKERRXX(ierr);
    ierr = VecCopy(l, l1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(vnp1, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(vn, &l1); CHKERRXX(ierr);


    ierr = VecDestroy(rhs_m); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_p); CHKERRXX(ierr);
    ierr = VecDestroy(mu_m_); CHKERRXX(ierr);
    ierr = VecDestroy(mu_p_); CHKERRXX(ierr);
    ierr = VecDestroy(u_jump_); CHKERRXX(ierr);
    ierr = VecDestroy(mu_grad_u_jump_); CHKERRXX(ierr);
    VecDestroy(vnp1);
    VecDestroy(X_0_v);
    VecDestroy(X_1_v);
}










void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, int compt, Vec X0, Vec X1, Vec Sm, Vec vn, Vec err)
{
    PetscErrorCode ierr;
    char *out_dir = NULL;
    out_dir = getenv("OUT_DIR");
    if(out_dir==NULL)
    {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save visuals\n"); CHKERRXX(ierr);
        return;
    }

    std::ostringstream oss;

    oss << out_dir << "/jump_"
        << p4est->mpisize << "_"
        << brick->nxyztrees[0] << "x"
        << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
           "x" << brick->nxyztrees[2] <<
       #endif
           "." << compt;

    double *phi_p, *sol_p, *X0_p, *X1_p, *Sm_p, *vn_p, *err_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArray(X0, &X0_p); CHKERRXX(ierr);
    ierr = VecGetArray(X1, &X1_p); CHKERRXX(ierr);
    ierr = VecGetArray(Sm, &Sm_p); CHKERRXX(ierr);
    ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);
    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    Vec mu;
    ierr = VecDuplicate(phi, &mu); CHKERRXX(ierr);
    double *mu_p_;
    ierr = VecGetArray(mu, &mu_p_); CHKERRXX(ierr);

    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        mu_p_[n] = phi_p[n]<0 ? mu_m(x,y,z) : mu_p(x,y,z);
#else
        mu_p_[n] = phi_p[n]<0 ? mu_m(x,y) : mu_p(x,y);
#endif
    }
    /* save the size of the leaves */
    Vec leaf_level;
    ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
    double *l_p;
    ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
            const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
            l_p[tree->quadrants_offset+q] = quad->level;
        }
    }

    for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
    {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
    }

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           8, 1, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "mu", mu_p_,
                           VTK_POINT_DATA, "sol", sol_p,
                           VTK_POINT_DATA, "X0", X0_p,
                           VTK_POINT_DATA, "X1", X1_p,
                           VTK_POINT_DATA, "vn", vn_p,
                           VTK_POINT_DATA, "err", err_p,
                           VTK_POINT_DATA, "Sm", Sm_p,
                           VTK_CELL_DATA , "leaf_level", l_p);

    ierr = VecRestoreArray(mu, &mu_p_); CHKERRXX(ierr);
    ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
    ierr = VecDestroy(mu); CHKERRXX(ierr);
    ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(Sm, &Sm_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(X0, &X0_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(X1, &X1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(vn, &vn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);

    PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



int main(int argc, char** argv) {
    PetscErrorCode ierr;
    // prepare parallel enviroment
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    // stopwatch
    parStopWatch w;
    w.start("Running example: electroporation");


    // p4est variables
    p4est_t*              p4est;
    p4est_nodes_t*        nodes;
    p4est_ghost_t*        ghost;
    p4est_connectivity_t* conn;
    my_p4est_brick_t      brick;

    // domain size information
    const int n_xyz []      = {2, 2, 2};
    PetscPrintf(mpi.comm(), "xyz_max for Box dimensions are set to be xmax = %g \t ymax = %g \t zmax = %g\n", xmax, ymax, zmaxx);
    const double xyz_min [] = {xmin, ymin, zminn}; //{-1, -1, -1};
    const double xyz_max [] = {xmax, ymax, zmaxx}; //{ 1,  1,  1};

    int periodic[] = {0, 0, 0};
    conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

    for(int repeat=0; repeat<nb_splits; ++repeat)
    {
        ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+repeat, lmax+repeat); CHKERRXX(ierr);


        // create the forest
        p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

        // refine based on distance to a level-set
        splitting_criteria_cf_t sp(lmin+repeat, lmax+repeat, &level_set, 1.2);

        p4est->user_pointer = &sp;
        for(int i=0; i<lmax; i++)
        {
            my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
            my_p4est_partition(p4est, P4EST_TRUE, NULL);
        }


        /* create the initial forest at time nm1 */
        p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
        my_p4est_partition(p4est, P4EST_TRUE, NULL);  //PAM

        // create ghost layer at time nm1: 2 layers to use Voronoi solver!
        ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
        my_p4est_ghost_expand(p4est, ghost);
        my_p4est_ghost_expand(p4est, ghost);
        // create node structure at time nm1
        nodes = my_p4est_nodes_new(p4est, ghost);

        if(p4est->mpirank==0)
        {
            p4est_gloidx_t nb_nodes = 0;
            for(int r=0; r<p4est->mpisize; ++r)
                nb_nodes += nodes->global_owned_indeps[r];
            ierr = PetscPrintf(p4est->mpicomm, "number of nodes : %d\n", nb_nodes); CHKERRXX(ierr);
        }

        my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);

        if(save_hierarchy){
            hierarchy.write_vtk("hierarchy");
            PetscPrintf(p4est->mpicomm, "Hierarchy structure saved to current directory. \n");
        }

        my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
        my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

        /* compute neighborhood information */
        ngbd_n.init_neighbors();

        /* initialize the variables */
        Vec phi, X0, X1, Sm, vn;
        ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, level_set, phi);


        /* perturb level set */
        my_p4est_level_set_t ls(&ngbd_n);
        ls.perturb_level_set_function(phi, EPS);

        Vec grad_phi[3];
        double *dphi_p[3],*phi_p;
        ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

        for(int i=0;i<3;++i)
        {
            ierr = VecCreateGhostNodes(p4est, nodes, &grad_phi[i]); CHKERRXX(ierr);
            ierr = VecGetArray(grad_phi[i], &dphi_p[i]); CHKERRXX(ierr);
        }

        for(size_t i=0; i<ngbd_n.get_layer_size(); ++i)
        {
            p4est_locidx_t n = ngbd_n.get_layer_node(i);

            const quad_neighbor_nodes_of_node_t qnnn = ngbd_n[n];
            double nx = qnnn.dx_central(phi_p);
            double ny = qnnn.dy_central(phi_p);
            double nz = qnnn.dz_central(phi_p);
            double norm = sqrt(nx*nx+ny*ny+nz*nz);
            if(norm>EPS) { nx /= norm; ny /= norm; nz/=norm;}
            else         { nx = 0; ny = 0; nz=0;}
            dphi_p[0][n] = nx;
            dphi_p[1][n] = ny;
            dphi_p[2][n] = nz;
        }

        for(int j=0;j<3;++j)
        {
            ierr = VecGhostUpdateBegin(grad_phi[j], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
        for(size_t i=0; i<ngbd_n.get_local_size(); ++i)
        {
            p4est_locidx_t n = ngbd_n.get_local_node(i);
            const quad_neighbor_nodes_of_node_t qnnn = ngbd_n[n];
            double nx = qnnn.dx_central(phi_p);
            double ny = qnnn.dy_central(phi_p);
            double nz = qnnn.dz_central(phi_p);
            double norm = sqrt(nx*nx+ny*ny+nz*nz);
            if(norm>EPS) { nx /= norm; ny /= norm; nz/=norm;}
            else         { nx = 0; ny = 0; nz=0;}

            dphi_p[0][n] = nx;
            dphi_p[1][n] = ny;
            dphi_p[2][n] = nz;
        }
        for(int j=0;j<3;++j)
        {
            ierr = VecGhostUpdateEnd(grad_phi[j], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
        for(int j=0;j<3;++j)
        {
            ierr = VecRestoreArray(grad_phi[j], &dphi_p[j]); CHKERRXX(ierr);
        }


        /* set initial time step *//* find dx and dy smallest */
        /* p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
        p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
        double xmin = p4est->connectivity->vertices[3*vm + 0];
        double ymin = p4est->connectivity->vertices[3*vm + 1];
        double xmax = p4est->connectivity->vertices[3*vp + 0];
        double ymax = p4est->connectivity->vertices[3*vp + 1];
        */
        double dx = (xmax-xmin) / pow(2., (double) sp.max_lvl);
        double dy = (ymax-ymin) / pow(2., (double) sp.max_lvl);
#ifdef P4_TO_P8
        //PetscPrintf(p4est->mpicomm, "22: zmax=%g, zmin=%g\n", zmax, zmin);
        //double zmin = p4est->connectivity->vertices[3*vm + 2];
        //double zmax = p4est->connectivity->vertices[3*vp + 2];
        double dz = (zmaxx-zminn) / pow(2.,(double) sp.max_lvl);
        //PetscPrintf(p4est->mpicomm, "3: xmin=%g, xmax=%g, ymin=%g, ymax=%g, zmin=%g, zmax=%g\n", xmin, xmax, ymin, ymax, zmin, zmax);
#endif
       double diag = sqrt(dx*dx + dy*dy + dz*dz);



        // loop over time
        int iteration = 0;


        FILE *fp;
        FILE *fp_err;
        char name[10000];
        char name_err[1000];
        if(save_error)
        {
            sprintf(name, "/home/pouria/Work/Electroporation_Output/2d/data.dat");
            ierr = PetscFOpen(mpi.comm(), name, "w", &fp); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(), fp, "%% time | avg S | avg poration | avg permeabilization | v_pole...\n"); CHKERRXX(ierr);
            ierr = PetscFPrintf(mpi.comm(), fp,  "%e", 0.); CHKERRXX(ierr);
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    ierr = PetscFPrintf(mpi.comm(), fp,  "\t %e\t %e\t %e\t %e", 0., 0., 0., 0.); CHKERRXX(ierr);

            ierr = PetscFClose(mpi.comm(), fp); CHKERRXX(ierr);

            if(test==4)
            {
                sprintf(name_err, "/home/pouria/Work/Electroporation_Output/2d/err_%d.dat", nb_splits);
                ierr = PetscFOpen(mpi.comm(), name_err, "w", &fp_err); CHKERRXX(ierr);
                ierr = PetscFClose(mpi.comm(), fp_err); CHKERRXX(ierr);
                ierr = PetscPrintf(mpi.comm(), "Saving error in %s\n", name_err); CHKERRXX(ierr);
            }
            ierr = PetscPrintf(mpi.comm(), "Saving data in %s\n", name); CHKERRXX(ierr);
        }

        double tn = 0;
        double err_n   = 0;
        double err_nm1 = 0;
        ierr = VecDuplicate(phi, &X0); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &X1); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &Sm); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, initial_x0, X0);
        sample_cf_on_nodes(p4est, nodes, initial_x1, X1);
        sample_cf_on_nodes(p4est, nodes, initial_sm, Sm);
        ierr = VecDuplicate(phi, &vn); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, initial_vn, vn);

        Vec vnm1, vnm2;
        ierr = VecDuplicate(phi, &vnm1); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &vnm2); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est, nodes, initial_vnm1, vnm1);
        sample_cf_on_nodes(p4est, nodes, initial_vnm2, vnm2);

        Vec sol;
        Vec err;
        ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

        Vec electrodes_phi, intensity;
        ierr = VecDuplicate(phi, &electrodes_phi); CHKERRXX(ierr);
        ierr = VecDuplicate(phi, &intensity); CHKERRXX(ierr);
        save_VTK(p4est, ghost, nodes, &brick, phi, sol, -1, X0, X1, Sm, vn, err);
        clock_t begin = clock();
        my_p4est_interpolation_nodes_t interp_n(&ngbd_n);

#ifdef P4_TO_P8
        dt = MIN(dx,dy,dz)/dt_scale;
#else
        dt = MIN(dx,dy)/dt_scale;
#endif
        dt=MIN(dt,0.1/omega);
        PetscPrintf(p4est->mpicomm, "Proceed with dt=%g, dx=%g, scaling %g \n", dt, dz,MIN(dx,dy,dz)/dt);
        while (tn<tf)
        {
            PetscPrintf(mpi.comm(), "####################################################\n");
            ierr = PetscPrintf(mpi.comm(), "Iteration %d, time %e\n", iteration, tn); CHKERRXX(ierr);
            ls.perturb_level_set_function(phi, EPS);
            solve_Poisson_Jump(p4est, nodes, &ngbd_n, &ngbd_c, phi, sol, dt, X0, X1, Sm, vn, ls,tn, vnm1, vnm2, grad_phi, diag);

            double u_Npole_exact = 0;
            double u_Npole = 0;
            double xyz_np[3] = {0, 1.*R1*cos(PI/4), 1.0*R1*sin(PI/4)};
            if(test==1 || test==2 || test==4)
            {
                interp_n.set_input(vn, linear);
                interp_n.add_point(0, xyz_np);
                interp_n.interpolate(&u_Npole);
                interp_n.clear();
                u_Npole_exact = v_exact(xyz_np[0], xyz_np[1], xyz_np[2], tn+dt);
            }

            /* compute the error on the tree*/
            double *err_p, *sol_p,*Ephi_p, *intensity_p;
            ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
            ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
            ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
            ierr = VecGetArray(electrodes_phi, &Ephi_p); CHKERRXX(ierr);
            ierr = VecGetArray(intensity, &intensity_p); CHKERRXX(ierr);

            err_nm1 = err_n;
            err_n = 0;

            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
                double x = node_x_fr_n(n, p4est, nodes);
                double y = node_y_fr_n(n, p4est, nodes);
                double z = node_z_fr_n(n, p4est, nodes);

                // a level-set just to represent the electrode surfaces for integration purposes
                Ephi_p[n] = z - zmaxx + EPS; // only on the top surface.
                // this is on both surfaces
                /*if(z>0)
                    Ephi_p[n] = z - zmaxx + EPS;
                else
                    Ephi_p[n] = -(z - zminn - EPS); */

                if(ABS(phi_p[n])>0.1*diag)
                {
                    err_p[n] = ABS(sol_p[n] - u_exact(x,y,z,tn,phi_p[n]>0));
                    err_n = MAX(err_n,err_p[n]);
                }
            }




            MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
            ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
            //PetscPrintf(p4est->mpicomm, "tests=1,2: Iter %d maximum error on solution: %g ORDER: %g\n", iteration, err_n, log(err_nm1/err_n)/log(2));

            double des_err = 0;
            if (test==1 || test==2 || test==4)
            {
                interp_n.set_input(err, linear);
                interp_n.add_point(0, xyz_np);
                interp_n.interpolate(&des_err);
                interp_n.clear();
            }

            // Computing the intensity vector
            for(size_t i=0; i<ngbd_n.get_layer_size(); ++i)
            {
                p4est_locidx_t n = ngbd_n.get_layer_node(i);
                quad_neighbor_nodes_of_node_t qnnn = ngbd_n[n];
                double normal_drv_potential = qnnn.dz_central(sol_p);
                intensity_p[n] = sigma_e*normal_drv_potential;
            }
            ierr = VecGhostUpdateBegin(intensity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd_n.get_local_size(); ++i)
            {
                p4est_locidx_t n = ngbd_n.get_local_node(i);
                quad_neighbor_nodes_of_node_t qnnn = ngbd_n[n];
                double normal_drv_potential = qnnn.dz_central(sol_p);
                intensity_p[n] = sigma_e*normal_drv_potential;
            }
            ierr = VecGhostUpdateEnd(intensity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(electrodes_phi, &Ephi_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(intensity, &intensity_p); CHKERRXX(ierr);


            // compute the important quantities: Pulse Intensity (A), impedance (Ohm), permitivity, conductivity
            double avg_Voltage = integrate_over_interface(p4est, nodes, electrodes_phi, sol)/((xmax-xmin)*(ymax-ymin));            // V*m^2
            double PulseIntensity = integrate_over_interface(p4est, nodes, electrodes_phi, intensity);   // current throughput (A): charge flux
            double impedance = avg_Voltage/PulseIntensity;                                               // Impedance (Ohm)

            double net_E = PulseIntensity/sigma_e;                                                       // E_net = epsilon_0*Q_top_plate = integral(E.dA) on top plate
            double epsilon_r = 1;                                                                        // relative permitivity, real part
            if(fabs(avg_Voltage)>EPS)
            {
                epsilon_r = fabs(net_E*(zmaxx-zminn)/((xmax-xmin)*(ymax-ymin)*avg_Voltage));     //PAM: the relative permittivity (dispersive term! eps = "eps_r"*eps_0 - j*sigma/omega)
                if (epsilon_r<1)
                    epsilon_r = 1;
            }
            double sigma_eff_Real, sigma_eff_Imaginary;
            double shape_factor = (xmax - xmin)*(ymax - ymin)/(zmaxx - zminn);

            sigma_eff_Real = 1/impedance/shape_factor;
            sigma_eff_Imaginary = -epsilon_r*epsilon_0*omega;

            if(save_impedance){
                char *out_dir = NULL;
                out_dir = getenv("OUT_DIR");
                if(out_dir==NULL)
                {
                    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
                }
                else
                {
                    char out_path_Z[1000];
                    sprintf(out_path_Z, "%s/impedance.dat", out_dir);
                    if(p4est->mpirank==0)
                    {
                        if(iteration ==0){
                            FILE *f = fopen(out_path_Z, "w");
                            fprintf(f, "time [s], \t impedance [Ohm], \t north pole TMP \t Pulse Intensity (A)  \t error \t exact TMP [V] \t relative permittivity \t Applied E(t) \t Re(sigma_eff) [S/m] \t Im(sigma_eff) [S/m] \t Omega [Hz] %g\n", omega);
                            fprintf(f, "%g \t %g \t %g \t  %g \t %g \t %g \t %g \t %g \t %g \t %g\n", tn+dt, impedance, u_Npole, PulseIntensity, des_err, u_Npole_exact, epsilon_r, pulse(tn), sigma_eff_Real, sigma_eff_Imaginary);
                            fclose(f);
                        }
                        else{
                            FILE *f = fopen(out_path_Z, "a");
                            fprintf(f, "%g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \n", tn+dt, impedance, u_Npole, PulseIntensity, des_err, u_Npole_exact, epsilon_r, pulse(tn), sigma_eff_Real, sigma_eff_Imaginary);
                            fclose(f);
                        }

                    }
                }
            }

            if(save_vtk && iteration%save_every_n == 0)
            {
                save_VTK(p4est, ghost, nodes, &brick, phi, sol, iteration, X0, X1, Sm, vn, err);
            }
            tn += dt;
            iteration++;
        }
        ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
        clock_t end = clock();
        double elapsed_secs_tmp = double(end - begin) / CLOCKS_PER_SEC;
        PetscPrintf(p4est->mpicomm, "\n###########################################\n\n");
        PetscPrintf(p4est->mpicomm, "\n########################################### TIMING DONE! elapsed time is:%g\n", elapsed_secs_tmp);
        PetscPrintf(p4est->mpicomm, "\n###########################################\n\n");

        ierr = VecDestroy(err); CHKERRXX(ierr);
        ierr = VecDestroy(vn); CHKERRXX(ierr);
        ierr = VecDestroy(vnm2); CHKERRXX(ierr);
        ierr = VecDestroy(vnm1); CHKERRXX(ierr);
        ierr = VecDestroy(Sm); CHKERRXX(ierr);
        ierr = VecDestroy(X0); CHKERRXX(ierr);
        ierr = VecDestroy(X1); CHKERRXX(ierr);


        // destroy the structures
        p4est_nodes_destroy(nodes);
        p4est_ghost_destroy(ghost);
        p4est_destroy      (p4est);
    }
    my_p8est_brick_destroy(conn, &brick);

    w.stop(); w.read_duration();

}
