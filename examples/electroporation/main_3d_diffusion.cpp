/*
 * Title: electroporation
 * Description: Solves electroporation and diffusion and advection and reaction equations for a tissue environment in parallel.
 * The models are from Leguebe and Poignard.
 *
 * Author: Pouria Mistani
 * Date Created: 09-22-2016
 */
/*
 * Notes on Stampede2:
 * On stampede, for some reason I don't know!, go to line 1247-1252 in my_p4est_electroporatio_solve.cpp and
 * change the xyz_min1 = {...} and xyz_max1={...} to xyz_min={..} and xyz_max={...}!!
 *
 * Some general notes on using voro++:
 * in 3D, we use the voro++ library to construct the voronoi mesh. The voro++ library should
 * be configured to account for the double precision numbers as well as decreasing the minimum
 * threshold for the definiton of tolerance. I get good results with tolerance=1e-13.
 * It is generally a good idea to turn on the VERBOSE mode in the voro++ library.
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
#include <src/voronoi2D.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_electroporation_solve.h>
#include <src/my_p4est_semi_lagrangian.h>
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
#include <src/my_p8est_semi_lagrangian.h>
#endif
//#include "nearpt3/nearpt3.cc"
#include <src/Parser.h>
#include <src/math.h>
#include <iostream>
#include <string>
#include <stack>
#include <algorithm>
#include "Halton/halton.cpp"

using namespace std;

template<typename T>
bool contains(const std::vector<T> &vec, const T& elem)
{
    return find(vec.begin(), vec.end(), elem)!=vec.end();
}


int test = 6;
/* -1: just a positive domain for test,
 * static linear case=1,
* dynamic linear case=2,
* dynamic nonlinear case=4,
* dynamic nonlinear case on a cubic lattice = 6, these are prolate ellipsoids
*  random cube box side enforced = 8,
*  random spheroid=9,
* 10=read from initial condition file.
* 11 = read from high packing distributions close random packings, max allowed density=65%
* test=1,2 use the exact solution on the boundary condition! Be careful!
* */

double cellDensity = 0.0001;   // only if test = 8 || 9
double density = 0;         // this is for measuring the density finally. don't change its declaration value.
double boxSide = 1e-3;      // only if test = 8



double omega = 0.5e6;  //angular frequency: if w=0.25 MHz every 1 micro-second pulse will pause, and repeat!
double epsilon_0 = 8.85e-12; // farad/meter: permitivity in vacuum
/* 0 or 1 */
int implicit = 0;
/* order 1, 2 or 3. If choosing 3, implicit only */
int order = 1;

/* cell radius */
double r0 = test==5 ? 46e-6 : (test==6 ? 53e-6 : ((test==8 || test==9) ? 7e-6 :50e-6));
double ellipse = test<5 ? 1 : (test==5 ? 1.225878312944962 : 1.250835468987754);
double a = test<5 ? r0 : (test==5 ? r0*ellipse : r0/ellipse);
double b = test<5 ? r0 : (test==5 ? r0*ellipse : r0/ellipse);
double c = test<5 ? r0 : (test==5 ? r0/ellipse : r0*ellipse);


double boxVolume = boxSide*boxSide*boxSide; // only for cases 8 or 9
double ClusterRadius = 0.49*boxSide;
double SphereVolume = 4*PI*(ClusterRadius*ClusterRadius*ClusterRadius)/3;
double coeff = 1.;
double cellVolume = 4*PI*(coeff*r0)*(coeff*r0)*(coeff*r0)/3;
// 30 is the safety coefficient to avoid too-close cells corresponding to a minimum radius of ~3*r0



/* number of cells in x and y dimensions */
int x_cells = 1;
int y_cells = 1;
int z_cells = 1;
/* number of random cells */
int nb_cells = test==7 ? 34 : (test==2 || test==4 || test==5)? 1 : ((test==8 || test==9) ? int (cellDensity*SphereVolume/cellVolume) : x_cells*y_cells*z_cells);


//Note: I changed xmin/ymin/zminn and max's for test<4 from 2*x_cells*r0 to 4*x_cells*r0
double xmin = test<4 ? -2*x_cells*r0 :  (test == 7 ? -4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9 || test==10 || test==11) ? -boxSide/2 : -2*x_cells*r0));
double xmax = test<4 ?  2*x_cells*r0 :  (test == 7 ?  4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9 || test==10 || test==11) ?  boxSide/2 :  2*x_cells*r0));
double ymin = test<4 ? -2*y_cells*r0 :  (test == 7 ? -4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9 || test==10 || test==11) ? -boxSide/2 : -2*y_cells*r0));
double ymax = test<4 ?  2*y_cells*r0 :  (test == 7 ?  4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9 || test==10 || test==11) ?  boxSide/2 :  2*y_cells*r0));
double zminn = test<4 ? -2*z_cells*r0 :  (test == 7 ? -4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9 || test==10 || test==11) ? -boxSide/2 : -2*z_cells*r0));
double zmaxx = test<4 ?  2*z_cells*r0 :  (test == 7 ?  4*pow(nb_cells, 1./3.)*r0  : ((test==8 || test==9 || test==10 || test==11) ?  boxSide/2 :  2*z_cells*r0));

const double xyz_min_[] = {xmin, ymin, zminn};
const double xyz_max_[] = {xmax, ymax, zmaxx};

int axial_nb = 2*zmaxx/r0/2;
int lmax_thr = (int)log2(axial_nb)+5;   // the mesh should be fine enough to have enough nodes on each cell for solver not to crash.
int lmin = 4;
int lmax =6;//MAX(lmax_thr, 5);
int nb_splits = 1;

double dt_scale = 40;
double tn;
double tf = 1e-3;//15.0/omega;
double dt;

double E_unscaled = 20;                       /* applied electric field on the top electrode: kv/m */
double E = E_unscaled * 1e3 * (zmaxx-zminn);  // this is the potential difference in SI units!

double sigma_c = 1;//0.455;
double sigma_e = 15;//1.5;

double Cm = test==1 ? 0 : 9.5e-3;
double SL = 1.9;
double S0 = 1.1e6;
double S1 = 1e4;
double X_0 = 0;
double X_1 = 0;
double Sm_threshold_value = 100*SL;   // threshold to be considered permeabilized: 100 times the rest value
double Vep = 258e-3;
double Xep = 0.5;

double tau_ep   = 1e-6;
double tau_perm = 80e-6;     //80*tau_ep;  with the modified X2 equation we need this one! Not here.
double tau_res  = 60;


/* diffusion constants */
double P0 = 0;     // initial permeability of membrane to nonpermeable molecule M.
double P1 = 1e-6;
double P2 = 1e-7;


double Faraday = 96485.3328;   //Faraday's constant C/mol
double R_gas = 8.314;          // J/C/mol
double T_env = 310;            // Kelvin
double d_c = 5e-6;             // diffusion in intracellular m*m/s
double d_e = 1e-5;             // diffusion in extracellular
double mu_e = 1e-3;            // motility = d_e*Faraday/R/T (Molecule motility in outer medium = 1e-6)
double mu_c = 0.0;             // Molecule motility in inner medium = 1e-7
const int number_ions = 2;     // Number of ions in the simulations

double M_0 = 1.0e-6;           // uniform initial condition for concentration
double M_boundary= 1.0e-6;     // concentration on the boundary of the box
/* end of diffusion modeling */

/* Electro-Elasticity */
double lambda = 10;





double R1 = .25*MIN(xmax-xmin, ymax-ymin, zmaxx-zminn);
double R2 = 3*MAX(xmax-xmin, ymax-ymin, zmaxx-zminn);

// save statistics
int save_every_n = 1;
bool save_impedance = true;
bool save_transport = true;
bool save_vtk = true;
bool save_stats = true;
bool save_dipoles = true;
bool save_avg_dipole_only = false;
bool save_shadowing = true;

bool save_error = false;
bool save_voro = false;
bool check_partition = false;
bool save_hierarchy = false;
// I hope you don't touch the last 4... unless you are troubled!

class LevelSet : public CF_3
{
private:
    PetscErrorCode      ierr;
    PetscMPIInt           rank;
    unsigned int seed = time(NULL);

public:
    vector<double> radii;
    vector<Point3> centers;
    vector<Point3> ex;
    vector<Point3> theta;
    double cellVolumes;

    double operator()(double x, double y, double z) const
    {
        double d = DBL_MAX;
        double  xm=xmin;
        double ym=ymin;
        double zm=zminn;
        double dx = (xmax-xmin)/(x_cells+1);
        double dy = (ymax-ymin)/(y_cells+1);
        double dz = (zmaxx-zminn)/(z_cells+1);
        double x_tmp, y_tmp, z_tmp;
        double x0, y0, z0;
        switch(test)
        {
        case -1: return 1;
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
        case 5: return sqrt(SQR(x) + SQR(y) + SQR(z)) - R1;
        case 6:
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    for(int k=0; k<z_cells; ++k)
                        d = MIN(d, sqrt(SQR((x-(xm+(i+1)*dx))*r0/a) + SQR((y-(ym+(j+1)*dy))*r0/b) + SQR((z-(zm+(k+1)*dz))*r0/c)) - r0);
            return d;
        case 7:
            for(int n=0; n<nb_cells; ++n)
            {
                x_tmp = x - centers[n].x;
                y_tmp = y - centers[n].y;
                z_tmp = z - centers[n].z;

                d = MIN(d, sqrt(SQR(x_tmp) + SQR(y_tmp) + SQR(z_tmp)) - radii[n]);
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
        case 10:
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
        case 11:
            // loading the random close packing spheres of Kenneth Desmond, Emory University
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

    void initialize()
    {
        MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
        lip=1.2;

        if(test==6)
        {
            double  xm=xmin;
            double ym=ymin;
            double zm=zminn;
            double dx = (xmax - xmin)/(x_cells+1);
            double dy = (ymax-ymin)/(y_cells+1);
            double dz = (zmaxx-zminn)/(z_cells+1);

            double cellVolumes = 0;
            int n = 0;
            centers.resize(nb_cells);
            radii.resize(nb_cells);
            ex.resize(nb_cells);
            theta.resize(nb_cells);
            for(int i=0; i<x_cells; ++i)
                for(int j=0; j<y_cells; ++j)
                    for(int k=0; k<z_cells; ++k)
                    {
                        centers[n].x = (xm+(i+1)*dx);//*r0/a;
                        centers[n].y = (ym+(j+1)*dy);//*r0/b;
                        centers[n].z = (zm+(k+1)*dz);//*r0/c;
                        radii[n] = 1;
                        ex[n].x = a;
                        ex[n].y = b;
                        ex[n].z = c;
                        theta[n].x = 0;
                        theta[n].y =0;
                        theta[n].z =0;
                        cellVolumes += 4*PI*ex[n].x*ex[n].y*ex[n].z/3;
                        n++;
                    }
            double density = cellVolumes/(xmax-xmin)/(ymax-ymin)/(zmaxx-zminn);
            if(rank==0)
                ierr = PetscPrintf(PETSC_COMM_SELF, "The volume fraction is = %g\n",density); CHKERRXX(ierr);
        }
        if(test==7)
        {
            centers.resize(nb_cells);
            radii.resize(nb_cells);
            ex.resize(nb_cells);
            theta.resize(nb_cells);
            srand(seed);
            if(rank==0)
            {
                ierr = PetscPrintf(PETSC_COMM_SELF, "The random seed is %u\n", seed); CHKERRXX(ierr);
                ierr = PetscPrintf(PETSC_COMM_SELF, "Number of SPHERICAL cells is %u\n", nb_cells); CHKERRXX(ierr);
            }
            std::vector<std::array<double,3> > v;
            std::array<double,3> p;
            double Radius=0.0;

            double *r;
            int halton_counter = 0;
            r = halton(halton_counter,3);
            double azimuth = 0.0;
            double polar = 0.0;
            Radius = 0.45*(xmax-xmin-3*r0)*r[0];
            azimuth = 2*PI*r[1];
            polar = PI*r[2];

            p[0] = Radius*sin(polar)*cos(azimuth);
            p[1] = Radius*sin(polar)*sin(azimuth);
            p[2] = Radius*cos(polar);

            halton_counter++;
            v.push_back(p);
            do
            {
                r = halton(halton_counter,3);
                double azimuth = 0;
                double polar = 0;
                Radius = 0.45*(xmax-xmin-3*r0)*r[0];
                azimuth = 2*PI*r[1];
                polar = PI*r[2];
                p[0] = Radius*sin(polar)*cos(azimuth);
                p[1] = Radius*sin(polar)*sin(azimuth);
                p[2] = Radius*cos(polar);
                halton_counter++;
                bool far_enough = true;
                for(unsigned int ii=0;ii<v.size();++ii){
                    double mindist = sqrt(SQR(p[0]-v[ii][0])+ SQR(p[1]-v[ii][1])+SQR(p[2]-v[ii][2]));
                    if(mindist<3*r0){
                        far_enough = false;
                        break;
                    }
                }
                if(far_enough){
                    v.push_back(p);
                    if(v.size()%((int) nb_cells/10) == 0){
                        double progress = 100*v.size()/nb_cells;
                        if(rank==0)
                            ierr = PetscPrintf(PETSC_COMM_SELF, "Cell Placement is in Progress. Currently at: %g %\n", progress); CHKERRXX(ierr);
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
                ex[n].x = 1;
                ex[n].y = 1;
                ex[n].z = 1;
                theta[n].x = 0;
                theta[n].y =0;
                theta[n].z =0;
                cellVolumes += 4*PI*radii[n]*radii[n]*radii[n]*ex[n].x*ex[n].y*ex[n].z/3;
            }
            density = cellVolumes/SphereVolume;
            if(rank==0)
                ierr = PetscPrintf(PETSC_COMM_SELF, "Done initializing random spherical cells. The volume fraction is = %g\n",density); CHKERRXX(ierr);
        }

        if(test==8 || test==9)
        {
            centers.resize(nb_cells);
            radii.resize(nb_cells);
            ex.resize(nb_cells);
            theta.resize(nb_cells);
            srand(seed);
            if(rank==0)
            {
                ierr = PetscPrintf(PETSC_COMM_SELF, "The random seed is %u\n", seed); CHKERRXX(ierr);
                ierr = PetscPrintf(PETSC_COMM_SELF, "Number of ELLIPSOID cells is %u\n", nb_cells); CHKERRXX(ierr);
            }
            std::vector<std::array<double,3> > v;
            std::array<double,3> p;
            double Radius=0.0;

            double *r;
            int halton_counter = 0;
            r = halton(halton_counter,3);

            if(test==9){
                double azimuth = 0.0;
                double polar = 0.0;
                Radius = 0.45*(xmax-xmin-3*r0)*r[0];
                azimuth = 2*PI*r[1];
                polar = PI*r[2];

                p[0] = Radius*sin(polar)*cos(azimuth);
                p[1] = Radius*sin(polar)*sin(azimuth);
                p[2] = Radius*cos(polar);
            } else {
                p[0] = (xmax-xmin - 3*r0)*(r[0] - 0.5);
                p[1] = (ymax-ymin - 3*r0)*(r[1] - 0.5);
                p[2] = (zmaxx-zminn - 3*r0)*(r[2] - 0.5);
            }
            halton_counter++;
            v.push_back(p);
            do
            {
                r = halton(halton_counter,3);
                if(test==9){
                    double azimuth = 0;
                    double polar = 0;
                    Radius = 0.45*(xmax-xmin-3*r0)*r[0];
                    azimuth = 2*PI*r[1];
                    polar = PI*r[2];

                    p[0] = Radius*sin(polar)*cos(azimuth);
                    p[1] = Radius*sin(polar)*sin(azimuth);
                    p[2] = Radius*cos(polar);
                } else {
                    p[0] = (xmax-xmin - 3*r0)*(r[0] - 0.5);
                    p[1] = (ymax-ymin - 3*r0)*(r[1] - 0.5);
                    p[2] = (zmaxx-zminn - 3*r0)*(r[2] - 0.5);
                }
                halton_counter++;
                bool far_enough = true;
                for(unsigned int ii=0;ii<v.size();++ii){
                    double mindist = sqrt(SQR(p[0]-v[ii][0])+ SQR(p[1]-v[ii][1])+SQR(p[2]-v[ii][2]));
                    if(mindist<3*r0){
                        far_enough = false;
                        break;

                    }
                }
                if(far_enough){
                    v.push_back(p);
                    if(v.size()%((int) nb_cells/10) == 0){
                        double progress = 100*v.size()/nb_cells;
                        if(rank==0)
                            ierr = PetscPrintf(PETSC_COMM_SELF, "Cell Placement is in Progress. Currently at: %g %\n", progress); CHKERRXX(ierr);
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

            if (test==8 || test==9)
                density = cellVolumes/SphereVolume;
            else
                density = cellVolumes/(xmax-xmin)/(ymax-ymin)/(zmaxx-zminn);

            if(rank==0)
                ierr = PetscPrintf(PETSC_COMM_SELF, "Done initializing random cells. The volume fraction is = %g\n",density); CHKERRXX(ierr);
        }

        if (test==10){
            double max_x = 0;
            double min_x=0;
            std::ifstream in;
            in.open("initial_conditions/init_n_10_N_27440.dat");
            std::string line;
            if(in.is_open())
            {
                int line_number = 0;
                int n = 0;

                while(std::getline(in, line)) //get 1 row as a string
                {

                    if(line_number==0 || line_number==2)
                    {
                        line_number++;
                        continue;
                    } else if (line_number==1)
                    {
                        std::istringstream iss(line); //put line into stringstream
                        std::string word;
                        iss >> word;
                        nb_cells = std::stoi(word);
                        centers.resize(nb_cells);
                        radii.resize(nb_cells);
                        ex.resize(nb_cells);
                        theta.resize(nb_cells);
                        line_number++;
                        continue;
                    } else {
                        std::istringstream iss(line); //put line into stringstream
                        std::string word;
                        iss >> word;
                        int ID = std::stoi(word);
                        iss >> word;
                        centers[n].x = std::stof(word);
                        min_x = MIN(min_x, centers[n].x);
                        max_x = MAX(max_x, centers[n].x);
                        iss >> word;
                        centers[n].y = std::stof(word);
                        iss >> word;
                        centers[n].z = std::stof(word);
                        iss >> word;
                        radii[n] = std::stof(word);
                        iss >> word;
                        ex[n].x = std::stof(word);
                        iss >> word;
                        ex[n].y = std::stof(word);
                        iss >> word;
                        ex[n].z = std::stof(word);
                        iss >> word;
                        theta[n].x = std::stof(word);
                        iss >> word;
                        theta[n].y = std::stof(word);
                        iss >> word;
                        theta[n].z = std::stof(word);
                        iss >> word;
                        cellVolumes += std::stof(word);
                        line_number++;
                        n++;
                    }
                }
            }
            ClusterRadius = (max_x - min_x)/2.0;
            SphereVolume = 4*PI*(ClusterRadius*ClusterRadius*ClusterRadius)/3;
            density = cellVolumes/SphereVolume;
            if(rank==0)
            {
                ierr = PetscPrintf(PETSC_COMM_SELF, "The spheroid is almost bounded between xmin = %g and x_max = %g\n", min_x, max_x); CHKERRXX(ierr);
                ierr = PetscPrintf(PETSC_COMM_SELF, "Done initializing %d number of random cells. The Spheroid volume fraction is = %g\n", nb_cells, density); CHKERRXX(ierr);
            }
        }

        if (test==11){
            double ratio = 0.75;
            double max_x = 0, max_y=0, max_z=0;
            double min_x=0, min_y=0, min_z=0;
            std::ifstream in;
            in.open("./Packing/monodisperse_1000/FinalConfig.dat");
            std::string line;
            if(in.is_open())
            {
                nb_cells = 1000;
                centers.resize(nb_cells);
                radii.resize(nb_cells);
                ex.resize(nb_cells);
                theta.resize(nb_cells);
                int n = 0;
                while(std::getline(in, line)) //get 1 row as a string
                {
                    theta[n].x = 0;
                    theta[n].y = 0;
                    theta[n].z = 0;
                    ex[n].x = 1;
                    ex[n].y = 1;
                    ex[n].z = 1;
                    std::istringstream iss(line); //put line into stringstream
                    std::string word;
                    iss >> word;
                    centers[n].x = (std::stof(word) - 0.5)*ratio*boxSide;
                    min_x = MIN(min_x, centers[n].x);
                    max_x = MAX(max_x, centers[n].x);
                    iss >> word;
                    centers[n].y = (std::stof(word) - 0.5)*ratio*boxSide;
                    min_y = MIN(min_y, centers[n].y);
                    max_y = MAX(max_y, centers[n].y);
                    iss >> word;
                    centers[n].z = (std::stof(word) - 0.5)*ratio*boxSide;
                    min_z = MIN(min_z, centers[n].z);
                    max_z = MAX(max_z, centers[n].z);
                    iss >> word;
                    int cellType = std::stoi(word);
                    if(cellType==1)
                        radii[n] = 0.05*boxSide*ratio;
                    else if (cellType==2)
                        radii[n] = 0.05*boxSide*ratio;
                    cellVolumes += 4*PI*radii[n]*radii[n]*radii[n]/3.0;
                    n++;
                }
            }
            ClusterRadius = (max_x - min_x)/2.0;
            double pack_Volume = (max_x - min_x)*(max_y - min_y)*(max_z - min_z);
            density = cellVolumes/pack_Volume;
            if(rank==0)
            {
                ierr = PetscPrintf(PETSC_COMM_SELF, "The cube is almost bounded between xmin = %g and x_max = %g\n", min_x, max_x); CHKERRXX(ierr);
                ierr = PetscPrintf(PETSC_COMM_SELF, "Done initializing %d number of random cells. The packing's volume fraction is = %g\n", nb_cells, density); CHKERRXX(ierr);
            }
        }
    }

    void save_cells(){
        if(test==6 || test==7 || test==8 || test==9 || test==10 || test==11)
        {
            MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
            if(rank==0)
            {
                char out_path[1000];
                char *out_dir = NULL;
                out_dir = getenv("OUT_DIR");
                if(out_dir==NULL)
                {
                    ierr = PetscPrintf(PETSC_COMM_SELF, "You need to set the environment variable OUT_DIR before running the code to save topologies...\n"); CHKERRXX(ierr);
                } else {
                    sprintf(out_path, "%s/Cells_Topology.dat", out_dir);
                    FILE *f = fopen(out_path, "w");
                    fprintf(f, "%% Number of cells is: %u, and the seed is set to: %u \n", nb_cells, seed);
                    fprintf(f, "%% ID  |\t X_c\t  |\t Y_c\t  |\t Z_c\t |\t radius\t |\t ex.x\t |\t ex.y\t |\t ex.z\t |\t theta.x |\t theta.y |\t theta.z |\t cell volume \n");
                    for(int n=0; n<nb_cells; ++n)
                    {
                        double tmp_volume = 4*PI*radii[n]*radii[n]*radii[n]*ex[n].x*ex[n].y*ex[n].z/3;
                        fprintf(f, "%d \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g\n", n, centers[n].x, centers[n].y, centers[n].z, radii[n], ex[n].x, ex[n].y, ex[n].z, theta[n].x, theta[n].y, theta[n].z,  tmp_volume);
                    }
                    fclose(f);
                }
            }
        }
    }
} level_set;


struct SingleCell : CF_3
{
public:
    int ID;
    double operator()(double x, double y, double z) const
    {

        double d = DBL_MAX;
        if(test==2 || test==4 || test==5)
        {
            return sqrt(SQR(x) + SQR(y) + SQR(z)) - R1;
        }else if(test==6 || test==7 || test==8 || test==9 || test==10 || test==11)
        {
            double x0 = x - level_set.centers[ID].x;
            double y0 = y - level_set.centers[ID].y;
            double z0 = z - level_set.centers[ID].z;

            double x_tmp = x0;
            double y_tmp = cos(level_set.theta[ID].x)*y0 - sin(level_set.theta[ID].x)*z0;
            double z_tmp = sin(level_set.theta[ID].x)*y0 + cos(level_set.theta[ID].x)*z0;

            x0 = cos(level_set.theta[ID].y)*x_tmp - sin(level_set.theta[ID].y)*z_tmp;
            y0 = y_tmp;
            z0 = sin(level_set.theta[ID].y)*x_tmp + cos(level_set.theta[ID].y)*z_tmp;

            x_tmp = cos(level_set.theta[ID].z)*x0 - sin(level_set.theta[ID].z)*y0;
            y_tmp = sin(level_set.theta[ID].z)*x0 + cos(level_set.theta[ID].z)*y0;
            z_tmp = z0;

            d = MIN(d, sqrt(SQR(x_tmp/level_set.ex[ID].x) + SQR(y_tmp/level_set.ex[ID].y) + SQR(z_tmp/level_set.ex[ID].z)) - level_set.radii[ID]);

            return d;
        }
        throw std::invalid_argument("Choose a valid test.");
    }
} single_cell_phi;

struct CellNumbers : CF_3
{
public:

    double operator()(double x, double y, double z) const
    {

        double d = DBL_MAX;
        if(test==2 || test==4 || test==5)
        {
            double d= sqrt(SQR(x) + SQR(y) + SQR(z)) - R1;
            if(d<=0)
                return 0;
            else
                return -1;
        }else{
            if(test==6 || test==7 || test==8 || test==9 || test==10 || test==11)
            {
                for(int ID=0; ID<nb_cells; ++ID)
                {
                    double x0 = x - level_set.centers[ID].x;
                    double y0 = y - level_set.centers[ID].y;
                    double z0 = z - level_set.centers[ID].z;

                    double x_tmp = x0;
                    double y_tmp = cos(level_set.theta[ID].x)*y0 - sin(level_set.theta[ID].x)*z0;
                    double z_tmp = sin(level_set.theta[ID].x)*y0 + cos(level_set.theta[ID].x)*z0;

                    x0 = cos(level_set.theta[ID].y)*x_tmp - sin(level_set.theta[ID].y)*z_tmp;
                    y0 = y_tmp;
                    z0 = sin(level_set.theta[ID].y)*x_tmp + cos(level_set.theta[ID].y)*z_tmp;

                    x_tmp = cos(level_set.theta[ID].z)*x0 - sin(level_set.theta[ID].z)*y0;
                    y_tmp = sin(level_set.theta[ID].z)*x0 + cos(level_set.theta[ID].z)*y0;
                    z_tmp = z0;

                    d = sqrt(SQR(x_tmp/level_set.ex[ID].x) + SQR(y_tmp/level_set.ex[ID].y) + SQR(z_tmp/level_set.ex[ID].z)) - level_set.radii[ID];
                    if(d<=0)
                        return ID;
                }
                return -1;
            }
        }
        throw std::invalid_argument("Choose a valid test.");
    }
} cell_numbering;

double u_exact(double x, double y, double z, double t, bool phi_is_pos)
{
    double SR1 = R1;
    double SR2 = R2;
    double SCm = Cm;
    double SSL= SL;


    double r = sqrt(x*x + y*y + z*z);
    double theta = atan2(sqrt(x*x+y*y),z);
    double g = E_unscaled*1e3*R2;


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
    return E*sin(2*PI*omega*tn);

    if(E*cos(2*PI*omega*tn)>=0)
        return E;
    else
        return 0;
}

// rescale later!
double v_exact(double x, double y, double z, double t)
{
    double theta = atan2(sqrt(x*x+y*y),z);
    double g = E_unscaled*1e3*R2;

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
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
            //return DIRICHLET;
        case 2:
            return DIRICHLET;
        case 3:
        case 4:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
            //return DIRICHLET;
        case 5:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 6:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 7:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 8:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 9:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 10:
            if(ABS(z-zminn)<EPS || ABS(z-zmaxx)<EPS) return DIRICHLET;
            else                                   return NEUMANN;
        case 11:
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
            if(ABS(z-zminn)<EPS) return 0;
            if(ABS(z-zmaxx)<EPS) return E;
            return 0;
            //return u_exact(x,y,z,0,true);
        case 2:
            return u_exact(x,y,z,t,true);
        case 3:
        case 4:
            if(ABS(z-zminn)<EPS) return 0;
            if(ABS(z-zmaxx)<EPS) return E;
            return 0;
        case 5:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
        case 6:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
        case 7:
            if(ABS(z-zminn)<EPS) return 0;
            if(ABS(z-zmaxx)<EPS) return pulse(t);
            return 0;
        case 8:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
            return 0;
        case 9:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
            return 0;
        case 10:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
            return 0;
        case 11:
            if(ABS(z-zmaxx)<EPS) return  pulse(t);
            if(ABS(z-zminn)<EPS) return 0;
            return 0;
        default: throw std::invalid_argument("Choose a valid test.");
        }
    }
} bc_wall_value_p;


struct MBCWALLTYPE : WallBC3D
{
    BoundaryConditionType operator()(double x, double y, double z) const
    {
        return NEUMANN;
        //        if(ABS(z-zmaxx)<EPS || ABS(z-zminn)<EPS) return DIRICHLET;
        //        else                 return NEUMANN;
    }
} M_bc_wall_type_p;

struct MBCWALLVALUE : CF_3
{
    double operator()(double x, double y, double z) const
    {
        return 0;

        //        if(ABS(z-zminn)<EPS) return 2*M_boundary;
        //        else if(ABS(z-zmaxx)<EPS) return 2*M_boundary;
        //        else if(ABS(x-xmin)<EPS || ABS(x-xmax)<EPS) return 0;
        //        else if(ABS(y-ymin)<EPS || ABS(y-ymax)<EPS) return 0;
    }
} M_bc_wall_value_p;



struct INTERFACE : CF_3
{
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} bc_interface_value_p;

double sigma(double x, double y, double z)
{
    return level_set(x,y,z)<=0 ? sigma_c : sigma_e;
}

class SIGMA : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        return level_set(x,y,z)<=0 ? sigma_c : sigma_e;
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


class Initial_Vn : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} initial_vn;

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




class Initial_Pm : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        return P0;
    }
} initial_pm;

class Initial_M : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
        if(level_set(x,y,z)>0)
            return M_0;//M_boundary+4*M_boundary*(zmaxx*zmaxx-z*z)/(zmaxx-zminn)/(zmaxx-zminn);//M_0
        else
            return 0;// M_boundary;//(M_0/R1)*(R1-ABS(level_set(x,y,z)));
    }
}initial_M;

class MOTILITY : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        return level_set(x,y,z)<0 ? mu_c : mu_e;
    }
}motility;

class MOTILITY_M : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        if(level_set(x,y,z)<0)
            return mu_c;
        else
            return 0;
    }
}motility_m;

class MOTILITY_P : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        if(level_set(x,y,z)>0)
            return mu_e;
        else
            return 0;
    }
}motility_p;


class DIFFUSION_M : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        if(level_set(x,y,z)<0)
            return d_c;
        else
            return 0;
    }
}diffusion_m;

class DIFFUSION_P : public CF_3
{
public:
    double operator()(double x, double y,double z) const
    {
        if(level_set(x,y,z)>0)
            return d_e;
        else
            return 0;
    }
}diffusion_p;

double is_interface(my_p4est_node_neighbors_t *ngbd_n, p4est_locidx_t n, double *phi_p)
{
    quad_neighbor_nodes_of_node_t qnnn;
    ngbd_n->get_neighbors(n, qnnn);
    double phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p;
    qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000 , phi_m00 , phi_p00 , phi_0m0 , phi_0p0, phi_00m, phi_00p);

    //    if(ABS(phi_000)<1e-8)
    //        return 0;
    if ((phi_000*phi_m00<0) || (phi_000*phi_p00<0)
            || (phi_000*phi_0m0<0) || (phi_000*phi_0p0<0)
            || (phi_000*phi_00m<0) || (phi_000*phi_00p<0))
        return 1;
    else
        return -1;
}


void compute_normal_and_curvature(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd, Vec phi, Vec normal[3], Vec &kappa, double dxyz_max)
{
    PetscErrorCode ierr;


    double *normal_p[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        if(normal[dir]!=NULL) { ierr = VecDestroy(normal[dir]); CHKERRXX(ierr); }
        ierr = VecCreateGhostNodes(p4est, nodes, &normal[dir]); CHKERRXX(ierr);
        ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    }
    if(kappa!=NULL) { ierr = VecDestroy(kappa); CHKERRXX(ierr); }

    ierr = VecDuplicate(phi, &kappa); CHKERRXX(ierr);

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    quad_neighbor_nodes_of_node_t qnnn;
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        ngbd->get_neighbors(n, qnnn);
        normal_p[0][n] = qnnn.dx_central(phi_p);
        normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        normal_p[2][n] = qnnn.dz_central(phi_p);
        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
#ifdef P4_TO_P8
        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
#endif
    }

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecGhostUpdateBegin(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_local_node(i);
        ngbd->get_neighbors(n, qnnn);
        normal_p[0][n] = qnnn.dx_central(phi_p);
        normal_p[1][n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        normal_p[2][n] = qnnn.dz_central(phi_p);
        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]) + SQR(normal_p[2][n]));
#else
        double norm = sqrt(SQR(normal_p[0][n]) + SQR(normal_p[1][n]));
#endif

        normal_p[0][n] = norm<EPS ? 0 : normal_p[0][n]/norm;
        normal_p[1][n] = norm<EPS ? 0 : normal_p[1][n]/norm;
#ifdef P4_TO_P8
        normal_p[2][n] = norm<EPS ? 0 : normal_p[2][n]/norm;
#endif
    }
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecGhostUpdateEnd(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    Vec kappa_tmp;
    ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);
    double *kappa_p;
    ierr = VecGetArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        ngbd->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1/dxyz_max), -1/dxyz_max);
#else
        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1/dxyz_max), -1/dxyz_max);
#endif
    }
    ierr = VecGhostUpdateBegin(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_local_node(i);
        ngbd->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]) + qnnn.dz_central(normal_p[2]), 1/dxyz_max), -1/dxyz_max);
#else
        kappa_p[n] = MAX(MIN(qnnn.dx_central(normal_p[0]) + qnnn.dy_central(normal_p[1]), 1/dxyz_max), -1/dxyz_max);
#endif
    }
    ierr = VecGhostUpdateEnd(kappa_tmp, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecRestoreArray(kappa_tmp, &kappa_p); CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    }

    my_p4est_level_set_t ls(ngbd);
    ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_tmp, kappa);
    ierr = VecDestroy(kappa_tmp); CHKERRXX(ierr);
}


void solve_electroelasticity( p4est_t *p4est, p4est_nodes_t *nodes,  p4est_ghost_t *ghost, my_p4est_level_set_t ls, my_p4est_node_neighbors_t *ngbd_n, Vec phi, double dt, double lambda, double dxyz_max)
{
    PetscErrorCode ierr;
    Vec kappa, normals[3], elastic_vel[P4EST_DIM], norm_phi, velo_n, curvature_force[P4EST_DIM], tmp;
    VecDuplicate(phi, &kappa);
    VecDuplicate(phi, &velo_n);
    VecDuplicate(phi, &tmp);
    for(unsigned int i=0; i<P4EST_DIM; ++i)
    {
        ierr = VecCreateGhostNodes(p4est, nodes, &curvature_force[i]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &elastic_vel[i]); CHKERRXX(ierr);
        VecDuplicate(phi, &normals[i]);
    }
    double diag = dxyz_max*sqrt(3);
    compute_normal_and_curvature(p4est, nodes, ngbd_n, phi, normals, kappa, dxyz_max);

    double *norm_phi_p, *phi_p, *curv_p[P4EST_DIM], *tmp_p, *kappa_p;
    ierr = VecCreateGhostNodes(p4est, nodes, &norm_phi); CHKERRXX(ierr);
    VecGetArray(norm_phi, &norm_phi_p);
    VecGetArray(phi, &phi_p);
    for(unsigned int i=0; i<P4EST_DIM; ++i)
        VecGetArray(curvature_force[i], &curv_p[i]);
    VecGetArray(kappa, &kappa_p);
    VecGetArray(norm_phi, &norm_phi_p);
    VecGetArray(tmp, &tmp_p);
    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_layer_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        double phi_x = qnnn.dx_central(phi_p);
        double phi_y = qnnn.dy_central(phi_p);
        double phi_z = qnnn.dz_central(phi_p);
        norm_phi_p[n] = sqrt(phi_x*phi_x+phi_y*phi_y+phi_z*phi_z);
        tmp_p[n] = kappa_p[n]*norm_phi_p[n];
    }
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(norm_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_local_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        double phi_x = qnnn.dx_central(phi_p);
        double phi_y = qnnn.dy_central(phi_p);
        double phi_z = qnnn.dz_central(phi_p);
        norm_phi_p[n] = sqrt(phi_x*phi_x+phi_y*phi_y+phi_z*phi_z);
        tmp_p[n] = kappa_p[n]*norm_phi_p[n];
    }
    ierr = VecGhostUpdateEnd(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(norm_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    VecRestoreArray(tmp, &tmp_p);
    VecRestoreArray(norm_phi, &norm_phi_p);

    VecGetArray(tmp, &tmp_p);
    VecGetArray(norm_phi, &norm_phi_p);
    double *normals_p[3];
    for(unsigned int i=0; i<P4EST_DIM; ++i)
        VecGetArray(normals[i], &normals_p[i]);


    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_layer_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        double tx = qnnn.dx_central(tmp_p);
        double ty = qnnn.dy_central(tmp_p);
        double tz = qnnn.dz_central(tmp_p);
        double ttt = (tx*normals_p[0][n]+ty*normals_p[1][n]+tz*normals_p[2][n]);
        double xcurve = tx-ttt*normals_p[0][n];
        double ycurve = ty-ttt*normals_p[1][n];
        double zcurve = tz-ttt*normals_p[2][n];
        curv_p[0][n] = -0.5*SQR(kappa_p[n])*normals_p[0][n]+(xcurve)/norm_phi_p[n];
        curv_p[1][n] = -0.5*SQR(kappa_p[n])*normals_p[1][n]+(ycurve)/norm_phi_p[n];
        curv_p[2][n] = -0.5*SQR(kappa_p[n])*normals_p[2][n]+(zcurve)/norm_phi_p[n];
    }
    ierr = VecGhostUpdateBegin(curvature_force[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(curvature_force[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(curvature_force[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_local_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);

        double tx = qnnn.dx_central(tmp_p);
        double ty = qnnn.dy_central(tmp_p);
        double tz = qnnn.dz_central(tmp_p);
        double ttt = (tx*normals_p[0][n]+ty*normals_p[1][n]+tz*normals_p[2][n]);
        double xcurve = tx-ttt*normals_p[0][n];
        double ycurve = ty-ttt*normals_p[1][n];
        double zcurve = tz-ttt*normals_p[2][n];
        curv_p[0][n] = -0.5*SQR(kappa_p[n])*normals_p[0][n]+(xcurve)/norm_phi_p[n];
        curv_p[1][n] = -0.5*SQR(kappa_p[n])*normals_p[1][n]+(ycurve)/norm_phi_p[n];
        curv_p[2][n] = -0.5*SQR(kappa_p[n])*normals_p[2][n]+(zcurve)/norm_phi_p[n];
    }
    ierr = VecGhostUpdateEnd(curvature_force[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(curvature_force[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(curvature_force[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    VecRestoreArray(norm_phi, &norm_phi_p);
    VecRestoreArray(tmp, &tmp_p);


    // compute velocities
    double *v_p[P4EST_DIM], *velo_n_p;
    VecGetArray(velo_n, &velo_n_p);
    VecGetArray(norm_phi, &norm_phi_p);
    for(unsigned int i=0; i<P4EST_DIM; ++i)
        VecGetArray(elastic_vel[i], &v_p[i]);
    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_layer_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        double npx = lambda*qnnn.dx_central(norm_phi_p);
        double npy = lambda*qnnn.dy_central(norm_phi_p);
        double npz = lambda*qnnn.dz_central(norm_phi_p);
        double normal_area_vel =  (npx*normals_p[0][n]+npy*normals_p[1][n]+npz*normals_p[2][n]);
        double E_prime = lambda*(norm_phi_p[n] - 1);
        double zeta = 0.5*(1+cos(PI*phi_p[n]/diag))/diag; // cut-off function
        // tensile resistance
        v_p[0][n] += 0.5*dt*((npx - normal_area_vel*normals_p[0][n])- E_prime*kappa_p[n]*normals_p[0][n])*norm_phi_p[n]*zeta;
        v_p[1][n] += 0.5*dt*((npy - normal_area_vel*normals_p[1][n])- E_prime*kappa_p[n]*normals_p[1][n])*norm_phi_p[n]*zeta;
        v_p[2][n] += 0.5*dt*((npz - normal_area_vel*normals_p[2][n])- E_prime*kappa_p[n]*normals_p[2][n])*norm_phi_p[n]*zeta;
        // curvature effects
        double cc = 0;
        cc += qnnn.dx_central(curv_p[0]);
        cc += qnnn.dy_central(curv_p[1]);
        cc += qnnn.dz_central(curv_p[2]);
        v_p[0][n] += 0.5*dt*cc*zeta*normals_p[0][n]*norm_phi_p[n];
        v_p[1][n] += 0.5*dt*cc*zeta*normals_p[1][n]*norm_phi_p[n];
        v_p[2][n] += 0.5*dt*cc*zeta*normals_p[2][n]*norm_phi_p[n];
        // Maxwell stress
        v_p[0][n] += 0.5*dt*0;
        v_p[1][n] += 0.5*dt*0;
        v_p[2][n] += 0.5*dt*0;

        // total velocity in normal direction
        velo_n_p[n] = v_p[0][n]*normals_p[0][n]+v_p[1][n]*normals_p[1][n]+v_p[2][n]*normals_p[2][n];
    }
    ierr = VecGhostUpdateBegin(velo_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(elastic_vel[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(elastic_vel[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(elastic_vel[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_local_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        double npx = lambda*qnnn.dx_central(norm_phi_p);
        double npy = lambda*qnnn.dy_central(norm_phi_p);
        double npz = lambda*qnnn.dz_central(norm_phi_p);
        double normal_area_vel =  (npx*normals_p[0][n]+npy*normals_p[1][n]+npz*normals_p[2][n]);
        double E_prime = lambda*(norm_phi_p[n] - 1);
        double zeta = 0.5*(1+cos(PI*phi_p[n]/diag))/diag; // cut-off function
        // tensile resistance
        v_p[0][n] += 0.5*dt*((npx - normal_area_vel*normals_p[0][n])- E_prime*kappa_p[n]*normals_p[0][n])*norm_phi_p[n]*zeta;
        v_p[1][n] += 0.5*dt*((npy - normal_area_vel*normals_p[1][n])- E_prime*kappa_p[n]*normals_p[1][n])*norm_phi_p[n]*zeta;
        v_p[2][n] += 0.5*dt*((npz - normal_area_vel*normals_p[2][n])- E_prime*kappa_p[n]*normals_p[2][n])*norm_phi_p[n]*zeta;
        // curvature effects
        double cc = 0;
        cc += qnnn.dx_central(curv_p[0]);
        cc += qnnn.dy_central(curv_p[1]);
        cc += qnnn.dz_central(curv_p[2]);
        v_p[0][n] += 0.5*dt*cc*zeta*normals_p[0][n]*norm_phi_p[n];
        v_p[1][n] += 0.5*dt*cc*zeta*normals_p[1][n]*norm_phi_p[n];
        v_p[2][n] += 0.5*dt*cc*zeta*normals_p[2][n]*norm_phi_p[n];
        // Maxwell stress
        v_p[0][n] += 0.5*dt*0;
        v_p[1][n] += 0.5*dt*0;
        v_p[2][n] += 0.5*dt*0;

        // total velocity in normal direction
        velo_n_p[n] = v_p[0][n]*normals_p[0][n]+v_p[1][n]*normals_p[1][n]+v_p[2][n]*normals_p[2][n];
    }
    ierr = VecGhostUpdateEnd(velo_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(elastic_vel[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(elastic_vel[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(elastic_vel[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    VecRestoreArray(velo_n, &velo_n_p);
    VecRestoreArray(norm_phi, &norm_phi_p);
    for(unsigned int i=0; i<P4EST_DIM; ++i)
    {
        VecRestoreArray(elastic_vel[i], &v_p[i]);
        VecRestoreArray(curvature_force[i], &curv_p[i]);
        VecRestoreArray(normals[i], &normals_p[i]);
    }
    VecRestoreArray(kappa, &kappa_p);
    VecRestoreArray(phi, &phi_p);

    ls.extend_from_interface_to_whole_domain_TVD(phi, velo_n, velo_n);
    ls.advect_in_normal_direction(velo_n, phi, dt);
    //my_p4est_semi_lagrangian_t sl(&p4est, &nodes, &ghost, ngbd_n);
    //sl.update_p4est(elastic_vel, dt, phi);
}





void solve_electric_potential( p4est_t *p4est, p4est_nodes_t *nodes,
                               my_p4est_node_neighbors_t *ngbd_n, my_p4est_cell_neighbors_t *ngbd_c,
                               Vec phi, Vec sol, double dt, Vec X0, Vec X1, Vec Sm, Vec vn, my_p4est_level_set_t ls, double tn, Vec vnm1, Vec vnm2, Vec grad_phi[3], Vec charge_rate, Vec grad_nm1, Vec grad_up, Vec grad_um)
{
    PetscErrorCode ierr;

    Vec rhs_m, rhs_p;
    Vec mu_m_, mu_p_;

    ierr = VecDuplicate(phi, &rhs_m); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs_p); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_m_); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mu_p_); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, mu_m, mu_m_);
    sample_cf_on_nodes(p4est, nodes, mu_p, mu_p_);


    double *rhs_m_p, *rhs_p_p, *c_rate_p;
    ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
    ierr = VecGetArray(charge_rate, &c_rate_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        rhs_m_p[n] = 0;//c_rate_p[n]/dt;
        rhs_p_p[n] = 0;//c_rate_p[n]/dt;
    }
    ierr = VecRestoreArray(charge_rate, &c_rate_p); CHKERRXX(ierr);
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

    int counter = 0;
    double Sm_err = 0;
    solver.set_vn(vn);
    solver.set_u_jump(vn);
    solver.set_vnm1(vnm1);
    solver.set_vnm2(vnm2);

    double *grad_phi_p[P4EST_DIM];
    for(int j=0;j<P4EST_DIM;++j)
        VecGetArray(grad_phi[j], &grad_phi_p[j]);

    do{
        solver.set_Sm(Sm);
        solver.solve(sol);

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
        ls.extend_Over_Interface_TVD(phi, u_minus);
        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
            phi_p[i] = -phi_p[i];
        ls.extend_Over_Interface_TVD(phi, u_plus);
        for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
            phi_p[i] = -phi_p[i];

        //   measure real jump values after solve with correct jumps
        /*            my_p4est_interpolation_nodes_t *interp_nm = new my_p4est_interpolation_nodes_t(ngbd_n);
                my_p4est_interpolation_nodes_t *interp_np = new my_p4est_interpolation_nodes_t(ngbd_n);
                interp_np->set_input(u_plus, linear);
                interp_nm->set_input(u_minus, linear); */
        double *u_minus_p, *u_plus_p, *u_jump_p;
        VecGetArray(vn, &u_jump_p);
        VecGetArray(u_plus, &u_plus_p);
        VecGetArray(u_minus, &u_minus_p);
        double diag = (zmaxx-zminn)/pow(2.0,(double) lmax)/2.0;
        for(unsigned int n=0; n<nodes->indep_nodes.elem_count;++n)
        {
            u_jump_p[n] = u_plus_p[n] - u_minus_p[n];
        }
        /*                for(unsigned int n=0; n<nodes->num_owned_indeps;++n)
                 {
                    if(ABS(phi_p[n])<EPS || is_interface(ngbd_n,n,phi_p)<0)
                    {
                        u_jump_p[n] = u_plus_p[n] - u_minus_p[n];
                        continue;
                    }
                    if(is_interface(ngbd_n,n,phi_p)>0)
                    {
                        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
                        double x = node_x_fr_n(n, p4est, nodes);
                        double y = node_y_fr_n(n, p4est, nodes);
                        double z = node_z_fr_n(n, p4est, nodes);
                        double xyz_np[3] = {x,y,z};
                        double nx = qnnn.dx_central(phi_p);
                        double ny = qnnn.dy_central(phi_p);
                        double nz = qnnn.dz_central(phi_p);
                        double norm = sqrt(nx*nx+ny*ny+nz*nz);
                        norm >EPS ? nx /= norm : nx = 0;
                        norm >EPS ? ny /= norm : ny = 0;
                        norm >EPS ? nz /= norm : nz = 0;
                        double m_in, m_out;
                        double dist = ABS(phi_p[n]);
                        if(phi_p[n]>0)
                        {
                            xyz_np[0] += nx*(diag/5 - dist);
                            xyz_np[1] += ny*(diag/5 - dist);
                            xyz_np[2] += nz*(diag/5 - dist);
                            //interp_nm.add_point(0, xyz_np);
                            //interp_nm.interpolate(&m_in);
                            m_in = (*interp_nm)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            m_out = (*interp_np)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            //                    interp_np.add_point(0, xyz_np);
                            //                    interp_np.interpolate(&m_out);
                            double tmp1 = (m_out - m_in);
                            //                    interp_np.clear();
                            //                    interp_nm.clear();

                            xyz_np[0] += -nx*2*diag/5;
                            xyz_np[1] += -ny*2*diag/5;
                            xyz_np[2] += -nz*2*diag/5;
                            m_in = (*interp_nm)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            m_out = (*interp_np)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            //                    interp_nm.add_point(0, xyz_np);
                            //                    interp_nm.interpolate(&m_in);
                            //                    interp_np.add_point(0, xyz_np);
                            //                    interp_np.interpolate(&m_out);

                            u_jump_p[n] = (tmp1 + (m_out - m_in))/2.0;
                            //                    interp_np.clear();
                            //                    interp_nm.clear();
                        }else{
                            xyz_np[0] += -nx*(diag/5 - dist);
                            xyz_np[1] += -ny*(diag/5 - dist);
                            xyz_np[2] += -nz*(diag/5 - dist);
                            m_in = (*interp_nm)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            m_out = (*interp_np)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            //                    interp_nm.add_point(0, xyz_np);
                            //                    interp_nm.interpolate(&m_in);
                            //                    interp_np.add_point(0, xyz_np);
                            //                    interp_np.interpolate(&m_out);
                            double tmp1 = (m_out - m_in);
                            //                    interp_np.clear();
                            //                    interp_nm.clear();

                            xyz_np[0] += nx*2*diag/5;
                            xyz_np[1] += ny*2*diag/5;
                            xyz_np[2] += nz*2*diag/5;
                            m_in = (*interp_nm)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            m_out = (*interp_np)(xyz_np[0],xyz_np[1],xyz_np[2]);
                            //                    interp_nm.add_point(0, xyz_np);
                            //                    interp_nm.interpolate(&m_in);
                            //                    interp_np.add_point(0, xyz_np);
                            //                    interp_np.interpolate(&m_out);

                            u_jump_p[n] = (tmp1 + (m_out - m_in))/2.0;
                            //                    interp_np.clear();
                            //                    interp_nm.clear();
                        }
                    }
                }
                delete dynamic_cast<my_p4est_interpolation_nodes_t*>(interp_nm);
                delete dynamic_cast<my_p4est_interpolation_nodes_t*>(interp_np); */
        VecRestoreArray(vn, &u_jump_p);
        VecRestoreArray(u_plus, &u_plus_p);
        VecRestoreArray(u_minus, &u_minus_p);
        ls.extend_from_interface_to_whole_domain(phi,vn,vn);
        // end of measure current jump values
        // potential directional gradients
        double *dup_p,*dum_p, *up_p, *um_p;
        ierr = VecGetArray(u_plus, &up_p); CHKERRXX(ierr);
        ierr = VecGetArray(u_minus, &um_p); CHKERRXX(ierr);
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
        ls.extend_from_interface_to_whole_domain_TVD(phi, grad_up, grad_up);
        ierr = VecDuplicate(sol,&du_minus_cte); CHKERRXX(ierr);
        ls.extend_from_interface_to_whole_domain_TVD(phi, grad_um, grad_um);

        double *du_plus_cte_p, *du_minus_cte_p;
        VecGetArray(grad_up, &du_plus_cte_p);
        VecGetArray(grad_um, &du_minus_cte_p);

        double *Sm_p, *grad_nm1_p;
        VecGetArray(Sm, &Sm_p);
        VecGetArray(grad_nm1,&grad_nm1_p);
        for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
        {
            du_plus_cte_p[n] = (sigma_c*du_minus_cte_p[n] + sigma_e*du_plus_cte_p[n])/sigma_e;
            du_minus_cte_p[n] = (sigma_c*du_minus_cte_p[n] + sigma_e*du_plus_cte_p[n])/sigma_c;
            grad_nm1_p[n] = sigma_e*du_plus_cte_p[n];
        }
        VecRestoreArray(Sm, &Sm_p);
        VecRestoreArray(grad_up, &du_plus_cte_p);
        VecRestoreArray(grad_um, &du_minus_cte_p);
        VecRestoreArray(grad_nm1,&grad_nm1_p);
        ls.extend_from_interface_to_whole_domain_TVD(phi, grad_nm1, grad_nm1);
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
            ierr = VecGetArray(vn, &vn_n_p); CHKERRXX(ierr);
            ierr = VecGetArray(X0, &X0_np1); CHKERRXX(ierr);
            ierr = VecGetArray(X1, &X1_np1); CHKERRXX(ierr);
            ierr = VecGetArray(X_0_v, &X_0_v_p); CHKERRXX(ierr);
            ierr = VecGetArray(X_1_v, &X_1_v_p); CHKERRXX(ierr);


            for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
            {
                X_0_v_p[n] = X0_np1[n] + dt*((beta_0_in(vn_n_p[n]) - X0_np1[n])/tau_ep);
                X_1_v_p[n] = X1_np1[n] + dt*MAX((beta_1_in(X0_np1[n])-X1_np1[n])/tau_perm, (beta_1_in(X0_np1[n])-X1_np1[n])/tau_res); //dt/tau_perm*(X0_np1[n]-X1_np1[n]);
                Sm_n_p[n] = SL + S0*X_0_v_p[n] + S1*X_1_v_p[n];
            }

            ierr = VecRestoreArray(Sm, &Sm_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X0,&X0_np1); CHKERRXX(ierr);
            ierr = VecRestoreArray(X1,&X1_np1); CHKERRXX(ierr);
            ierr = VecRestoreArray(vn, &vn_n_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X_0_v, &X_0_v_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(X_1_v, &X_1_v_p); CHKERRXX(ierr);
            counter++;
        }
    }while(0);

    for(int j=0;j<P4EST_DIM;++j)
        VecRestoreArray(grad_phi[j], &grad_phi_p[j]);
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
    ierr = VecDestroy(rhs_m); CHKERRXX(ierr);
    ierr = VecDestroy(rhs_p); CHKERRXX(ierr);
    ierr = VecDestroy(mu_m_); CHKERRXX(ierr);
    ierr = VecDestroy(mu_p_); CHKERRXX(ierr);
    VecDestroy(X_0_v);
    VecDestroy(X_1_v);
}





void advect_field_semi_lagrangian(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd_n, double dt_nm1, double dt_n,  Vec vnm1[P4EST_DIM], Vec **vxx_nm1, Vec v[P4EST_DIM], Vec **vxx, Vec M_n, Vec *M_xx_n, double *M_np1_p)
{
    PetscErrorCode ierr;
    my_p4est_interpolation_nodes_t interp_nm1(ngbd_n);
    my_p4est_interpolation_nodes_t interp(ngbd_n);

    std::vector<double> v_tmp_nm1[P4EST_DIM];
    std::vector<double> v_tmp[P4EST_DIM];


    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
        // phi_interp.add_point(n,xyz);
    }

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        v_tmp[dir].resize(nodes->indep_nodes.elem_count);
#ifdef P4_TO_P8
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2], quadratic);
#else
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], quadratic);
#endif
        interp.interpolate(v_tmp[dir].data());
    }
    interp.clear();
    // now find v_star
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        // Find initial xy points
        double xyz_star[] =
        {
            node_x_fr_n(n, p4est, nodes) - 0.5*dt_n*v_tmp[0][n],
            node_y_fr_n(n, p4est, nodes) - 0.5*dt_n*v_tmp[1][n]
    #ifdef P4_TO_P8
            , node_z_fr_n(n, p4est, nodes) - 0.5*dt_n*v_tmp[2][n]
    #endif
        };

        for(int dir=0; dir<P4EST_DIM; ++dir)
        {
            if        (is_periodic(p4est,dir) && xyz_star[dir]<xyz_min_[dir]) xyz_star[dir] += xyz_max_[dir]-xyz_min_[dir];
            else if (is_periodic(p4est,dir) && xyz_star[dir]>xyz_max_[dir]) xyz_star[dir] -= xyz_max_[dir]-xyz_min_[dir];
            else     xyz_star[dir] = MAX(xyz_min_[dir], MIN(xyz_max_[dir], xyz_star[dir]));
        }
        interp.add_point(n, xyz_star);
        interp_nm1.add_point(n, xyz_star);
    }

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        v_tmp_nm1[dir].resize(nodes->indep_nodes.elem_count);

#ifdef P4_TO_P8
        interp_nm1.set_input(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], vxx_nm1[dir][2], quadratic);
#else
        interp_nm1.set_input(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], quadratic);
#endif
        interp_nm1.interpolate(v_tmp_nm1[dir].data());


#ifdef P4_TO_P8
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2], quadratic);
#else
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], quadratic);
#endif
        interp.interpolate(v_tmp[dir].data());
    }
    interp_nm1.clear();
    interp.clear();
    // finally, find the backtracing value
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        double vx_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp[0][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[0][n];
        double vy_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp[1][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[1][n];
#ifdef P4_TO_P8
        double vz_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp[2][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[2][n];
#endif

        double xyz_d[] =
        {
            node_x_fr_n(n, p4est, nodes) - dt_n*vx_star,
            node_y_fr_n(n, p4est, nodes) - dt_n*vy_star
    #ifdef P4_TO_P8
            ,
            node_z_fr_n(n, p4est, nodes) - dt_n*vz_star
    #endif
        };

        for(int dir=0; dir<P4EST_DIM; ++dir)
        {
            if      (is_periodic(p4est,dir) && xyz_d[dir]<xyz_min_[dir]) xyz_d[dir] += xyz_max_[dir]-xyz_min_[dir];
            else if (is_periodic(p4est,dir) && xyz_d[dir]>xyz_max_[dir]) xyz_d[dir] -= xyz_max_[dir]-xyz_min_[dir];
            else                                               xyz_d[dir] = MAX(xyz_min_[dir], MIN(xyz_max_[dir], xyz_d[dir]));
        }

        interp.add_point(n, xyz_d);
    }

#ifdef P4_TO_P8
    interp.set_input(M_n, M_xx_n[0], M_xx_n[1], M_xx_n[2], quadratic_non_oscillatory);
#else
    interp.set_input(M_n, M_xx_n[0], M_xx_n[1], quadratic_non_oscillatory);
#endif
    interp.interpolate(M_np1_p);
}

void advect(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd_n, Vec v_nm1[3], Vec v[3], double dt_nm1, double dt_n, Vec &M)
{
    PetscErrorCode ierr;
    Vec *M_xx;
    // compute vx_xx, vx_yy
    Vec *vxx[P4EST_DIM];
    Vec *vnm1_xx[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        vxx[dir] = new Vec[P4EST_DIM];
        vnm1_xx[dir] = new Vec[P4EST_DIM];
        if(dir==0)
        {
            for(int dd=0; dd<P4EST_DIM; ++dd)
            {
                ierr = VecCreateGhostNodes(p4est, nodes, &vxx[dir][dd]); CHKERRXX(ierr);
                ierr = VecCreateGhostNodes(p4est, nodes, &vnm1_xx[dir][dd]); CHKERRXX(ierr);
            }
        }
        else
        {
            for(int dd=0; dd<P4EST_DIM; ++dd)
            {
                ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
                ierr = VecDuplicate(vnm1_xx[0][dd], &vnm1_xx[dir][dd]); CHKERRXX(ierr);
            }
        }
#ifdef P4_TO_P8
        ngbd_n->second_derivatives_central(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2]);
#else
        ngbd_n->second_derivatives_central(v[dir], vxx[dir][0], vxx[dir][1]);
#endif
#ifdef P4_TO_P8
        ngbd_n->second_derivatives_central(v_nm1[dir], vnm1_xx[dir][0], vnm1_xx[dir][1], vnm1_xx[dir][2]);
#else
        ngbd_n->second_derivatives_central(v_nm1[dir], vnm1_xx[dir][0], vnm1_xx[dir][1]);
#endif
    }
    // compute M_xx and M_yy
    M_xx = new Vec[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecDuplicate(vxx[0][dir], &M_xx[dir]); CHKERRXX(ierr);
    }
#ifdef P4_TO_P8
    ngbd_n->second_derivatives_central(M, M_xx[0], M_xx[1], M_xx[2]);
#else
    ngbd_n->second_derivatives_central(M, M_xx[0], M_xx[1]);
#endif

    double *M_np1_p;
    Vec M_np1;
    ierr = VecCreateGhostNodes(p4est, nodes, &M_np1); CHKERRXX(ierr);
    VecGetArray(M_np1, &M_np1_p);

    advect_field_semi_lagrangian(p4est, nodes, ngbd_n, dt_nm1, dt_n, v_nm1, vnm1_xx, v, vxx, M, M_xx, M_np1_p);

    VecRestoreArray(M_np1, &M_np1_p);
    ierr = VecDestroy(M); CHKERRXX(ierr);
    M = M_np1;
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        for(int dd=0; dd<P4EST_DIM; ++dd)
        {
            ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
            ierr = VecDestroy(vnm1_xx[dir][dd]); CHKERRXX(ierr);
        }
        delete[] vxx[dir];
        delete[] vnm1_xx[dir];
    }
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecDestroy(M_xx[dir]); CHKERRXX(ierr);
    }
    delete[] M_xx;
}
/*
double upwind_step(my_p4est_node_neighbors_t *ngbd_n, p4est_locidx_t n, double *f, double* fxx, double* fyy, double* fzz, double ux, double uy, double uz, double dt, double *phi_p, double *dxx, double *dyy, double *dzz)
{


    quad_neighbor_nodes_of_node_t qnnn;
    ngbd_n->get_neighbors(n, qnnn);

    if(is_interface(ngbd_n,n,phi_p)>0)
        return f[n];
    double fx = ux > 0 ? qnnn.dx_backward_quadratic(f, fxx) : qnnn.dx_forward_quadratic(f, fxx);
    double fy = uy > 0 ? qnnn.dy_backward_quadratic(f, fyy) : qnnn.dy_forward_quadratic(f, fyy);
    double fz = uz > 0 ? qnnn.dz_backward_quadratic(f, fzz) : qnnn.dz_forward_quadratic(f, fzz);
    return f[n] - dt*(ux*fx+uy*fy+uz*fz);




    // normal direction at this point
    double nx = qnnn.dx_central(phi_p);
    double ny = qnnn.dy_central(phi_p);
    double nz = qnnn.dz_central(phi_p);
    double norm = sqrt(nx*nx + ny*ny + nz*nz);
    fabs(norm) > EPS ? nx /= norm : nx = 0;
    fabs(norm) > EPS ? ny /= norm : ny = 0;
    fabs(norm) > EPS ? nz /= norm : nz = 0;

    // level-set values in the neighborhood
    double phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p;
    phi_000 = phi_p[n];
    phi_p00 = qnnn.f_p00_linear(phi_p);
    phi_m00 = qnnn.f_m00_linear(phi_p);
    phi_0m0 = qnnn.f_0m0_linear(phi_p);
    phi_0p0 = qnnn.f_0p0_linear(phi_p);
    phi_00m = qnnn.f_00m_linear(phi_p);
    phi_00p = qnnn.f_00p_linear(phi_p);

    if(fabs(phi_000) < EPS && phi_000>0)
        phi_000 = EPS;
    else if(fabs(phi_000) < EPS && phi_000<0)
        phi_000 = -EPS;


    // field values in the neighborhood
    double p_000 , p_m00 , p_p00 , p_0m0 , p_0p0, p_00m, p_00p;
    qnnn.ngbd_with_quadratic_interpolation(f, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0, p_00m, p_00p);

    double s_p00 = qnnn.d_p00; double s_m00 = qnnn.d_m00;
    double s_0p0 = qnnn.d_0p0; double s_0m0 = qnnn.d_0m0;
    double s_00p = qnnn.d_00p; double s_00m = qnnn.d_00m;
    //---------------------------------------------------------------------
    // Second Order derivatives
    //---------------------------------------------------------------------
    double pxx_000 = fxx[n];
    double pyy_000 = fyy[n];
    double pzz_000 = fzz[n];
    double pxx_m00 = qnnn.f_m00_linear(fxx);
    double pxx_p00 = qnnn.f_p00_linear(fxx);
    double pyy_0m0 = qnnn.f_0m0_linear(fyy);
    double pyy_0p0 = qnnn.f_0p0_linear(fyy);
    double pzz_00m = qnnn.f_00m_linear(fzz);
    double pzz_00p = qnnn.f_00p_linear(fzz);

    double phixx_000 = dxx[n];
    double phiyy_000 = dyy[n];
    double phizz_000 = dzz[n];
    double phixx_m00 = qnnn.f_m00_linear(dxx);
    double phixx_p00 = qnnn.f_p00_linear(dxx);
    double phiyy_0m0 = qnnn.f_0m0_linear(dyy);
    double phiyy_0p0 = qnnn.f_0p0_linear(dyy);
    double phizz_00m = qnnn.f_00m_linear(dzz);
    double phizz_00p = qnnn.f_00p_linear(dzz);


    // velocity normal to the interface
    double normal_velocity = ux*nx + uy*ny + uz*nz;
    //---------------------------------------------------------------------
    // Neumann boundary condition on the interface
    //---------------------------------------------------------------------
    if (normal_velocity>0 &&  ((phi_000*phi_m00<0) || (phi_000*phi_p00<0)
                               || (phi_000*phi_0m0<0) || (phi_000*phi_0p0<0)
                               || (phi_000*phi_00m<0) || (phi_000*phi_00p<0)))
    {
        // interface in the x direction
        if(phi_000*phi_m00<=0)
        {
            s_m00 =-interface_Location_With_Second_Order_Derivative(-s_m00,   0, phi_m00, phi_000, phixx_m00, phixx_000);
            p_m00=p_p00; pxx_000 = pxx_m00 = pxx_p00 = 0;
            //s_m00 = s_p00; p_m00=p_p00; pxx_000 = pxx_m00 = pxx_p00 = 0;
        }
        if(phi_000*phi_p00<=0)
        {
            s_p00 = interface_Location_With_Second_Order_Derivative(    0, s_p00, phi_000, phi_p00, phixx_000, phixx_p00);
            p_p00=p_m00; pxx_000 = pxx_m00 = pxx_p00 = 0;
            //s_p00 = s_m00; p_p00=p_m00; pxx_000 = pxx_m00 = pxx_p00 = 0;
        }

        // interface in the y direction
        if(phi_000*phi_0m0<=0)
        {
            s_0m0 =-interface_Location_With_Second_Order_Derivative(-s_0m0,   0, phi_0m0, phi_000, phiyy_0m0, phiyy_000);
            p_0m0=p_0p0; pyy_000 = pyy_0m0 = pyy_0p0 = 0;
            //s_0m0 = s_0p0; p_0m0=p_0p0; pyy_000 = pyy_0m0 = pyy_0p0 = 0;
        }
        if(phi_000*phi_0p0<=0)
        {
            s_0p0 = interface_Location_With_Second_Order_Derivative(    0, s_0p0, phi_000, p_0p0, phiyy_000, phiyy_0p0);
            p_0p0=p_0m0; pyy_000 = pyy_0m0 = pyy_0p0 = 0;
            //s_0p0 = s_0m0; p_0p0=p_0m0; pyy_000 = pyy_0m0 = pyy_0p0 = 0;
        }

        // interface in the z direction
        if(phi_000*phi_00m<=0)
        {
            s_00m =-interface_Location_With_Second_Order_Derivative(-s_00m,   0, phi_00m, phi_000, phizz_00m, phizz_000);
            p_00m=p_00p; pzz_000 = pzz_00m = pzz_00p = 0;
            //s_00m = s_00p; p_00m=p_00p; pzz_000 = pzz_00m = pzz_00p = 0;
        }
        if(phi_000*phi_00p<=0)
        {
            s_00p = interface_Location_With_Second_Order_Derivative(    0, s_00p, phi_000, phi_00p, phizz_000, phizz_00p);
            p_00p=p_00m; pzz_000 = pzz_00m = pzz_00p = 0;
            // s_00p = s_00m; p_00p=p_00m; pzz_000 = pzz_00m = pzz_00p = 0;
        }

        //        if(phi_000*phi_m00<0) { s_m00 =-interface_Location_With_Second_Order_Derivative(-s_m00,   0, phi_m00, phi_000, phixx_m00, phixx_000); p_m00=p_000; }
        //        if(phi_000*phi_p00<0) { s_p00 = interface_Location_With_Second_Order_Derivative(    0, s_p00, phi_000, phi_p00, phixx_000, phixx_p00); p_p00=p_000; }
        //        if(phi_000*phi_0m0<0) { s_0m0 =-interface_Location_With_Second_Order_Derivative(-s_0m0,   0, phi_0m0, phi_000, phiyy_0m0, phiyy_000); p_0m0=p_000;}
        //        if(phi_000*phi_0p0<0) { s_0p0 = interface_Location_With_Second_Order_Derivative(    0, s_0p0, phi_000, p_0p0, phiyy_000, phiyy_0p0); p_0p0=p_000;}
        //        if(phi_000*phi_00m<0) { s_00m =-interface_Location_With_Second_Order_Derivative(-s_00m,   0, phi_00m, phi_000, phizz_00m, phizz_000); p_00m=p_000;}
        //        if(phi_000*phi_00p<0) { s_00p = interface_Location_With_Second_Order_Derivative(    0, s_00p, phi_000, phi_00p, phizz_000, phizz_00p); p_00p=p_000;}
        //        s_m00 = MAX(s_m00,EPS);
        //        s_p00 = MAX(s_p00,EPS);
        //        s_0m0 = MAX(s_0m0,EPS);
        //        s_0p0 = MAX(s_0p0,EPS);
        //        s_00m = MAX(s_00m,EPS);
        //        s_00p = MAX(s_00p,EPS);

        //---------------------------------------------------------------------
        // First Order One-Sided Differencing
        //---------------------------------------------------------------------
        double px_p00 = (p_p00-p_000)/s_p00; double px_m00 = (p_000-p_m00)/s_m00;
        double py_0p0 = (p_0p0-p_000)/s_0p0; double py_0m0 = (p_000-p_0m0)/s_0m0;
        double pz_00p = (p_00p-p_000)/s_00p; double pz_00m = (p_000-p_00m)/s_00m;

        if(s_p00<EPS) px_p00 = 0; if(s_m00<EPS) px_m00 = 0;
        if(s_0p0<EPS) py_0p0 = 0; if(s_0m0<EPS) py_0m0 = 0;
        if(s_00p<EPS) pz_00p = 0; if(s_00m<EPS) pz_00m = 0;

        //---------------------------------------------------------------------
        // Second Order One-Sided Differencing
        //---------------------------------------------------------------------
        pxx_m00 = MINMOD(pxx_m00,pxx_000);   px_m00 += 0.5*s_m00*(pxx_m00);
        pxx_p00 = MINMOD(pxx_p00,pxx_000);   px_p00 -= 0.5*s_p00*(pxx_p00);
        pyy_0m0 = MINMOD(pyy_0m0,pyy_000);   py_0m0 += 0.5*s_0m0*(pyy_0m0);
        pyy_0p0 = MINMOD(pyy_0p0,pyy_000);   py_0p0 -= 0.5*s_0p0*(pyy_0p0);
        pzz_00m = MINMOD(pzz_00m,pzz_000);   pz_00m += 0.5*s_00m*(pzz_00m);
        pzz_00p = MINMOD(pzz_00p,pzz_000);   pz_00p -= 0.5*s_00p*(pzz_00p);


        if(px_p00>0) px_p00 = 0;
        if(px_m00<0) px_m00 = 0;
        if(py_0p0>0) py_0p0 = 0;
        if(py_0m0<0) py_0m0 = 0;
        if(pz_00p>0) pz_00p = 0;
        if(pz_00m<0) pz_00m = 0;



        //        double dt = MIN(s_m00,s_p00);
        //        dt = MIN(dt,s_0m0);
        //        dt = MIN(dt,s_0p0);
        //        dt = MIN(dt,s_00m);
        //        dt = MIN(dt,s_00p);
        //        dt /= 3.0;

        return p_000 - dt*normal_velocity*(sqrt(px_p00*px_p00 + px_m00*px_m00 +
                                                py_0p0*py_0p0 + py_0m0*py_0m0 +
                                                pz_00p*pz_00p + pz_00m*pz_00m));

    } else if (normal_velocity<0 &&  ((phi_000*phi_m00<0) || (phi_000*phi_p00<0)
                                      || (phi_000*phi_0m0<0) || (phi_000*phi_0p0<0)
                                      || (phi_000*phi_00m<0) || (phi_000*phi_00p<0)))
    {

        // boundary condition with no jump
        if(phi_000*phi_m00<0) { s_m00 =-interface_Location_With_Second_Order_Derivative(-s_m00,   0, phi_m00, phi_000, phixx_m00, phixx_000); p_m00=p_p00; }
        if(phi_000*phi_p00<0) { s_p00 = interface_Location_With_Second_Order_Derivative(    0, s_p00, phi_000, phi_p00, phixx_000, phixx_p00); p_p00=p_m00; }
        if(phi_000*phi_0m0<0) { s_0m0 =-interface_Location_With_Second_Order_Derivative(-s_0m0,   0, phi_0m0, phi_000, phiyy_0m0, phiyy_000); p_0m0=p_0p0;}
        if(phi_000*phi_0p0<0) { s_0p0 = interface_Location_With_Second_Order_Derivative(    0, s_0p0, phi_000, p_0p0, phiyy_000, phiyy_0p0); p_0p0=p_0m0;}
        if(phi_000*phi_00m<0) { s_00m =-interface_Location_With_Second_Order_Derivative(-s_00m,   0, phi_00m, phi_000, phizz_00m, phizz_000); p_00m=p_00p;}
        if(phi_000*phi_00p<0) { s_00p = interface_Location_With_Second_Order_Derivative(    0, s_00p, phi_000, phi_00p, phizz_000, phizz_00p); p_00p=p_00m;}
        s_m00 = MAX(s_m00,EPS);
        s_p00 = MAX(s_p00,EPS);
        s_0m0 = MAX(s_0m0,EPS);
        s_0p0 = MAX(s_0p0,EPS);
        s_00m = MAX(s_00m,EPS);
        s_00p = MAX(s_00p,EPS);

        //---------------------------------------------------------------------
        // First Order One-Sided Differencing
        //---------------------------------------------------------------------
        double px_p00 = (p_p00-p_000)/s_p00; double px_m00 = (p_000-p_m00)/s_m00;
        double py_0p0 = (p_0p0-p_000)/s_0p0; double py_0m0 = (p_000-p_0m0)/s_0m0;
        double pz_00p = (p_00p-p_000)/s_00p; double pz_00m = (p_000-p_00m)/s_00m;

        if(s_p00<EPS) px_p00 = 0; if(s_m00<EPS) px_m00 = 0;
        if(s_0p0<EPS) py_0p0 = 0; if(s_0m0<EPS) py_0m0 = 0;
        if(s_00p<EPS) pz_00p = 0; if(s_00m<EPS) pz_00m = 0;

        //---------------------------------------------------------------------
        // Second Order One-Sided Differencing
        //---------------------------------------------------------------------
        pxx_m00 = MINMOD(pxx_m00,pxx_000);   px_m00 += 0.5*s_m00*(pxx_m00);
        pxx_p00 = MINMOD(pxx_p00,pxx_000);   px_p00 -= 0.5*s_p00*(pxx_p00);
        pyy_0m0 = MINMOD(pyy_0m0,pyy_000);   py_0m0 += 0.5*s_0m0*(pyy_0m0);
        pyy_0p0 = MINMOD(pyy_0p0,pyy_000);   py_0p0 -= 0.5*s_0p0*(pyy_0p0);
        pzz_00m = MINMOD(pzz_00m,pzz_000);   pz_00m += 0.5*s_00m*(pzz_00m);
        pzz_00p = MINMOD(pzz_00p,pzz_000);   pz_00p -= 0.5*s_00p*(pzz_00p);

        if(px_p00<0) px_p00 = 0;
        if(px_m00>0) px_m00 = 0;
        if(py_0p0<0) py_0p0 = 0;
        if(py_0m0>0) py_0m0 = 0;
        if(pz_00p<0) pz_00p = 0;
        if(pz_00m>0) pz_00m = 0;


        return p_000 - dt*normal_velocity*(sqrt(px_p00*px_p00 + px_m00*px_m00 +
                                                py_0p0*py_0p0 + py_0m0*py_0m0 +
                                                pz_00p*pz_00p + pz_00m*pz_00m));
    } else
    {
        // far from the interface
        double fx = ux > 0 ? qnnn.dx_backward_quadratic(f, fxx) : qnnn.dx_forward_quadratic(f, fxx);
        double fy = uy > 0 ? qnnn.dy_backward_quadratic(f, fyy) : qnnn.dy_forward_quadratic(f, fyy);
        double fz = uz > 0 ? qnnn.dz_backward_quadratic(f, fzz) : qnnn.dz_forward_quadratic(f, fzz);
        return f[n] - dt*(ux*fx+uy*fy+uz*fz);
    }
}*/
/*    double fx = ux > 0 ? qnnn.dx_forward_quadratic(f, fxx)*phi_000/phi_p00 : qnnn.dx_backward_quadratic(f, fxx)*phi_000/phi_m00;
    double fy = uy > 0 ? qnnn.dy_forward_quadratic(f, fyy)*phi_000/phi_0p0 : qnnn.dy_backward_quadratic(f, fyy)*phi_000/phi_0m0;
    double fz = uz > 0 ? qnnn.dz_forward_quadratic(f, fzz)*phi_000/phi_00p : qnnn.dz_backward_quadratic(f, fzz)*phi_000/phi_00m;
    return f[n] - dt*(ux*fx+uy*fy+uz*fz);

    if(flux_d_p[n]<0 && phi_p[n]<0 && ABS(phi_p[n])<diag) // at entrance, gradient is preserved! no resistance exists. Elsewhere, normal thing!
    {
        quad_neighbor_nodes_of_node_t qnnn;
        ngbd_n->get_neighbors(n, qnnn);
        double phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p;
        phi_000 = phi_p[n];
        phi_p00 = ABS(qnnn.f_p00_linear(phi_p));
        phi_m00 = ABS(qnnn.f_m00_linear(phi_p));

        phi_0m0 = ABS(qnnn.f_0m0_linear(phi_p));
        phi_0p0 = ABS(qnnn.f_0p0_linear(phi_p));

        phi_00m = ABS(qnnn.f_00m_linear(phi_p));
        phi_00p = ABS(qnnn.f_00p_linear(phi_p));

        double nx = qnnn.dx_central(phi_p);
        double ny = qnnn.dy_central(phi_p);
        double nz = qnnn.dz_central(phi_p);
        double norm = sqrt(nx*nx + ny*ny + nz*nz);
        nx /= norm;
        ny /= norm;
        nz /= norm;

        //        double xyz_n[P4EST_DIM], xyz_pair[P4EST_DIM];
        //        node_xyz_fr_n(n, p4est, nodes, xyz_n);
        //        xyz_pair[0] = xyz_n[0] - 2*nx*phi_000;
        //        xyz_pair[1] = xyz_n[1] - 2*ny*phi_000;
        //        xyz_pair[2] = xyz_n[2] - 2*nz*phi_000;

        double fx, fy, fz;
        if(ux > 0){
            double fx_m00_mm = 0, fx_m00_pm = 0, fx_m00_mp = 0, fx_m00_pp = 0;
            double w_m00_mm = qnnn.d_m00_p0*qnnn.d_m00_0p;
            double w_m00_mp = qnnn.d_m00_p0*qnnn.d_m00_0m;
            double w_m00_pm = qnnn.d_m00_m0*qnnn.d_m00_0p;
            double w_m00_pp = qnnn.d_m00_m0*qnnn.d_m00_0m;
            if (w_m00_mm != 0) { fx_m00_mm = ngbd_n->get_neighbors(qnnn.node_m00_mm).dx_backward_linear(f); }
            if (w_m00_mp != 0) { fx_m00_mp = ngbd_n->get_neighbors(qnnn.node_m00_mp).dx_backward_linear(f); }
            if (w_m00_pm != 0) { fx_m00_pm = ngbd_n->get_neighbors(qnnn.node_m00_pm).dx_backward_linear(f); }
            if (w_m00_pp != 0) { fx_m00_pp = ngbd_n->get_neighbors(qnnn.node_m00_pp).dx_backward_linear(f); }
            double fx_m00 = (fx_m00_mm*w_m00_mm + fx_m00_mp*w_m00_mp + fx_m00_pm*w_m00_pm + fx_m00_pp*w_m00_pp )/(qnnn.d_m00_m0+qnnn.d_m00_p0)/(qnnn.d_m00_0m+qnnn.d_m00_0p);
            fx = qnnn.dx_backward_linear(f)*(phi_000 + phi_m00)/phi_000 - fx_m00*phi_m00/phi_000;
        } else {
            double fx_p00_mm = 0, fx_p00_pm = 0, fx_p00_mp = 0, fx_p00_pp = 0;
            double w_p00_mm = qnnn.d_p00_p0*qnnn.d_p00_0p;
            double w_p00_mp = qnnn.d_p00_p0*qnnn.d_p00_0m;
            double w_p00_pm = qnnn.d_p00_m0*qnnn.d_p00_0p;
            double w_p00_pp = qnnn.d_p00_m0*qnnn.d_p00_0m;
            if (w_p00_mm != 0) { fx_p00_mm = ngbd_n->get_neighbors(qnnn.node_p00_mm).dx_forward_linear(f); }
            if (w_p00_mp != 0) { fx_p00_mp = ngbd_n->get_neighbors(qnnn.node_p00_mp).dx_forward_linear(f); }
            if (w_p00_pm != 0) { fx_p00_pm = ngbd_n->get_neighbors(qnnn.node_p00_pm).dx_forward_linear(f); }
            if (w_p00_pp != 0) { fx_p00_pp = ngbd_n->get_neighbors(qnnn.node_p00_pp).dx_forward_linear(f); }
            double fx_p00 = (fx_p00_mm*w_p00_mm + fx_p00_mp*w_p00_mp + fx_p00_pm*w_p00_pm + fx_p00_pp*w_p00_pp )/(qnnn.d_p00_m0+qnnn.d_p00_p0)/(qnnn.d_p00_0m+qnnn.d_p00_0p);
            fx = qnnn.dx_forward_linear(f)*(phi_000 + phi_p00)/phi_000 - fx_p00*phi_p00/phi_000;
        }

        if(uy > 0){
            double fy_0m0_mm = 0, fy_0m0_pm = 0, fy_0m0_mp = 0, fy_0m0_pp = 0;
            double w_0m0_mm = qnnn.d_0m0_p0*qnnn.d_0m0_0p;
            double w_0m0_mp = qnnn.d_0m0_p0*qnnn.d_0m0_0m;
            double w_0m0_pm = qnnn.d_0m0_m0*qnnn.d_0m0_0p;
            double w_0m0_pp = qnnn.d_0m0_m0*qnnn.d_0m0_0m;
            if (w_0m0_mm != 0) { fy_0m0_mm = ngbd_n->get_neighbors(qnnn.node_0m0_mm).dy_backward_linear(f); }
            if (w_0m0_mp != 0) { fy_0m0_mp = ngbd_n->get_neighbors(qnnn.node_0m0_mp).dy_backward_linear(f); }
            if (w_0m0_pm != 0) { fy_0m0_pm = ngbd_n->get_neighbors(qnnn.node_0m0_pm).dy_backward_linear(f); }
            if (w_0m0_pp != 0) { fy_0m0_pp = ngbd_n->get_neighbors(qnnn.node_0m0_pp).dy_backward_linear(f); }
            double fy_0m0 = (fy_0m0_mm*w_0m0_mm + fy_0m0_mp*w_0m0_mp + fy_0m0_pm*w_0m0_pm + fy_0m0_pp*w_0m0_pp )/(qnnn.d_0m0_m0+qnnn.d_0m0_p0)/(qnnn.d_0m0_0m+qnnn.d_0m0_0p);
            fy = qnnn.dy_backward_linear(f)*(phi_000 + phi_0m0)/phi_000 - fy_0m0*phi_0m0/phi_000;
        } else {
            double fy_0p0_mm = 0, fy_0p0_pm = 0, fy_0p0_mp = 0, fy_0p0_pp = 0;
            double w_0p0_mm = qnnn.d_0p0_p0*qnnn.d_0p0_0p;
            double w_0p0_mp = qnnn.d_0p0_p0*qnnn.d_0p0_0m;
            double w_0p0_pm = qnnn.d_0p0_m0*qnnn.d_0p0_0p;
            double w_0p0_pp = qnnn.d_0p0_m0*qnnn.d_0p0_0m;
            if (w_0p0_mm != 0) { fy_0p0_mm = ngbd_n->get_neighbors(qnnn.node_0p0_mm).dy_forward_linear(f); }
            if (w_0p0_mp != 0) { fy_0p0_mp = ngbd_n->get_neighbors(qnnn.node_0p0_mp).dy_forward_linear(f); }
            if (w_0p0_pm != 0) { fy_0p0_pm = ngbd_n->get_neighbors(qnnn.node_0p0_pm).dy_forward_linear(f); }
            if (w_0p0_pp != 0) { fy_0p0_pp = ngbd_n->get_neighbors(qnnn.node_0p0_pp).dy_forward_linear(f); }
            double fy_0p0 = (fy_0p0_mm*w_0p0_mm + fy_0p0_mp*w_0p0_mp + fy_0p0_pm*w_0p0_pm + fy_0p0_pp*w_0p0_pp )/(qnnn.d_0p0_m0+qnnn.d_0p0_p0)/(qnnn.d_0p0_0m+qnnn.d_0p0_0p);
            fy = qnnn.dy_forward_linear(f)*(phi_000 + phi_0p0)/phi_000 - fy_0p0*phi_0p0/phi_000;
        }
Clair: scale!
        if(uz > 0){
            double fz_00m_mm = 0, fz_00m_pm = 0, fz_00m_mp = 0, fz_00m_pp = 0;
            double w_00m_mm = qnnn.d_00m_p0*qnnn.d_00m_0p;
            double w_00m_mp = qnnn.d_00m_p0*qnnn.d_00m_0m;
            double w_00m_pm = qnnn.d_00m_m0*qnnn.d_00m_0p;
            double w_00m_pp = qnnn.d_00m_m0*qnnn.d_00m_0m;
            if (w_00m_mm != 0) { fz_00m_mm = ngbd_n->get_neighbors(qnnn.node_00m_mm).dz_backward_linear(f); }
            if (w_00m_mp != 0) { fz_00m_mp = ngbd_n->get_neighbors(qnnn.node_00m_mp).dz_backward_linear(f); }
            if (w_00m_pm != 0) { fz_00m_pm = ngbd_n->get_neighbors(qnnn.node_00m_pm).dz_backward_linear(f); }
            if (w_00m_pp != 0) { fz_00m_pp = ngbd_n->get_neighbors(qnnn.node_00m_pp).dz_backward_linear(f); }
            double fz_00m = (fz_00m_mm*w_00m_mm + fz_00m_mp*w_00m_mp + fz_00m_pm*w_00m_pm + fz_00m_pp*w_00m_pp )/(qnnn.d_00m_m0+qnnn.d_00m_p0)/(qnnn.d_00m_0m+qnnn.d_00m_0p);
            fz = qnnn.dz_backward_linear(f)*(phi_000 + phi_00m)/phi_000 - fz_00m*phi_00m/phi_000;
        } else {
            double fz_00p_mm = 0, fz_00p_pm = 0, fz_00p_mp = 0, fz_00p_pp = 0;
            double w_00p_mm = qnnn.d_00p_p0*qnnn.d_00p_0p;
            double w_00p_mp = qnnn.d_00p_p0*qnnn.d_00p_0m;
            double w_00p_pm = qnnn.d_00p_m0*qnnn.d_00p_0p;
            double w_00p_pp = qnnn.d_00p_m0*qnnn.d_00p_0m;
            if (w_00p_mm != 0) { fz_00p_mm = ngbd_n->get_neighbors(qnnn.node_00p_mm).dz_forward_linear(f); }
            if (w_00p_mp != 0) { fz_00p_mp = ngbd_n->get_neighbors(qnnn.node_00p_mp).dz_forward_linear(f); }
            if (w_00p_pm != 0) { fz_00p_pm = ngbd_n->get_neighbors(qnnn.node_00p_pm).dz_forward_linear(f); }
            if (w_00p_pp != 0) { fz_00p_pp = ngbd_n->get_neighbors(qnnn.node_00p_pp).dz_forward_linear(f); }
            double fz_00p = (fz_00p_mm*w_00p_mm + fz_00p_mp*w_00p_mp + fz_00p_pm*w_00p_pm + fz_00p_pp*w_00p_pp )/(qnnn.d_00p_m0+qnnn.d_00p_p0)/(qnnn.d_00p_0m+qnnn.d_00p_0p);
            fz = qnnn.dz_forward_linear(f)*(phi_000 + phi_00p)/phi_000 - fz_00p*phi_00p/phi_000;
        }

        return f[n] - dt*(ux*fx+uy*fy+uz*fz);
    }
    else // in the bulk
    {
        quad_neighbor_nodes_of_node_t qnnn;
        ngbd_n->get_neighbors(n, qnnn);
        double fx = ux > 0 ? qnnn.dx_backward_quadratic(f, fxx) : qnnn.dx_forward_quadratic(f, fxx);
        double fy = uy > 0 ? qnnn.dy_backward_quadratic(f, fyy) : qnnn.dy_forward_quadratic(f, fyy);
        double fz = uz > 0 ? qnnn.dz_backward_quadratic(f, fzz) : qnnn.dz_forward_quadratic(f, fzz);
        return f[n] - dt*(ux*fx+uy*fy+uz*fz);
    }
}*/
/*
void advect_upwind(double dt, my_p4est_node_neighbors_t *ngbd_n, Vec field, Vec field_xx[3], Vec vel[3], double *phi_p, double *dxx, double *dyy, double *dzz)
{

    PetscErrorCode ierr;
    Vec field_np1;
    VecDuplicate(field_xx[0], &field_np1);

    double *field_xx_p, *field_yy_p, *field_zz_p, *field_np1_p, *vel_p[3], *field_p;
    for(unsigned int i=0;i<3;i++)
        VecGetArray(vel[i], &vel_p[i]);


    VecGetArray(field, &field_p);
    VecGetArray(field_np1, &field_np1_p);
    VecGetArray(field_xx[0], &field_xx_p);
    VecGetArray(field_xx[1], &field_yy_p);
    VecGetArray(field_xx[2], &field_zz_p);

    // 1) first half-step
    for (size_t i = 0; i<ngbd_n->get_layer_size(); i++) {
        p4est_locidx_t n = ngbd_n->get_layer_node(i);
        field_np1_p[n] = upwind_step(ngbd_n, n, field_p, field_xx_p, field_yy_p, field_zz_p, vel_p[0][n], vel_p[1][n], vel_p[2][n], dt, phi_p, dxx, dyy, dzz);
    }
    VecGhostUpdateBegin(field_np1, INSERT_VALUES, SCATTER_FORWARD);
    for (size_t i = 0; i<ngbd_n->get_local_size(); i++) {
        p4est_locidx_t n = ngbd_n->get_local_node(i);
        field_np1_p[n] = upwind_step(ngbd_n, n, field_p, field_xx_p, field_yy_p, field_zz_p, vel_p[0][n], vel_p[1][n], vel_p[2][n], dt, phi_p, dxx, dyy, dzz);
    }
    VecGhostUpdateEnd(field_np1, INSERT_VALUES, SCATTER_FORWARD);
    VecRestoreArray(field_np1, &field_np1_p);
    VecRestoreArray(field_xx[0], &field_xx_p);
    VecRestoreArray(field_xx[1], &field_yy_p);
    VecRestoreArray(field_xx[2], &field_zz_p);
    VecRestoreArray(field, &field_p);

    // 2) update gradients
    //ngbd_n->second_derivatives_central(field_np1, field_xx);
    ierr = VecGetArray(field_xx[0], &field_xx_p); CHKERRXX(ierr);
    ierr = VecGetArray(field_xx[1], &field_yy_p); CHKERRXX(ierr);
    ierr = VecGetArray(field_xx[2], &field_zz_p); CHKERRXX(ierr);
    ierr = VecGetArray(field_np1, &field_np1_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_layer_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        field_xx_p[n] = qnnn.dxx_central(field_np1_p);
        field_yy_p[n] = qnnn.dyy_central(field_np1_p);
        field_zz_p[n] = qnnn.dzz_central(field_np1_p);
    }
    ierr = VecGhostUpdateBegin(field_xx[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(field_xx[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(field_xx[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd_n->get_local_node(i);
        const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
        field_xx_p[n] = qnnn.dxx_central(field_np1_p);
        field_yy_p[n] = qnnn.dyy_central(field_np1_p);
        field_zz_p[n] = qnnn.dzz_central(field_np1_p);
    }
    ierr = VecGhostUpdateEnd(field_xx[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(field_xx[1], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(field_xx[2], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(field_xx[0], &field_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(field_xx[1], &field_yy_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(field_xx[2], &field_zz_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(field_np1, &field_np1_p); CHKERRXX(ierr);

    // 3) second half-step
    VecGetArray(field, &field_p);
    VecGetArray(field_np1, &field_np1_p);
    VecGetArray(field_xx[0], &field_xx_p);
    VecGetArray(field_xx[1], &field_yy_p);
    VecGetArray(field_xx[2], &field_zz_p);
    for (size_t i = 0; i<ngbd_n->get_layer_size(); i++) {
        p4est_locidx_t n = ngbd_n->get_layer_node(i);
        double field_np2 = upwind_step(ngbd_n, n, field_np1_p, field_xx_p, field_yy_p, field_zz_p, vel_p[0][n], vel_p[1][n], vel_p[2][n], dt, phi_p, dxx, dyy, dzz);
        field_p[n] = 0.5*(field_p[n] + field_np2);
    }
    VecGhostUpdateBegin(field, INSERT_VALUES, SCATTER_FORWARD);
    for (size_t i = 0; i<ngbd_n->get_local_size(); i++) {
        p4est_locidx_t n = ngbd_n->get_local_node(i);
        double field_np2 = upwind_step(ngbd_n, n, field_np1_p, field_xx_p, field_yy_p, field_zz_p, vel_p[0][n], vel_p[1][n], vel_p[2][n], dt, phi_p, dxx, dyy, dzz);
        field_p[n] = 0.5*(field_p[n] + field_np2);
    }
    VecGhostUpdateEnd(field, INSERT_VALUES, SCATTER_FORWARD);
    VecRestoreArray(field, &field_p);
    VecRestoreArray(field_np1, &field_np1_p);
    VecRestoreArray(field_xx[0], &field_xx_p);
    VecRestoreArray(field_xx[1], &field_yy_p);
    VecRestoreArray(field_xx[2], &field_zz_p);
    for(unsigned int i=0;i<3;i++)
    {
        VecRestoreArray(vel[i], &vel_p[i]);
    }
    VecDestroy(field_np1);
}

*/


void solve_transport( p4est_t *p4est,  p4est_ghost_t *ghost, p4est_nodes_t *nodes,
                      my_p4est_node_neighbors_t *ngbd_n, my_p4est_cell_neighbors_t *ngbd_c,
                      Vec phi, Vec sol, Vec X0, Vec X1, Vec Pm, Vec M_list[number_ions], my_p4est_level_set_t ls, double dt_nm1, double dt_n, Vec charge_rate, Vec vn, int number_ions, Vec grad_up, Vec ElectroPhoresis_nm1[P4EST_DIM], Vec ElectroPhoresis[P4EST_DIM])
{
    PetscErrorCode ierr;
    Vec rhs_m, rhs_p, grad_Mp, grad_Mm;
    ierr = VecCreateGhostNodes(p4est, nodes, &grad_Mp); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &grad_Mm); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_p); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs_m); CHKERRXX(ierr);

    Vec M_jump, grad_M_jump, mu_m_, mu_p_;
    VecDuplicate(phi, &M_jump);
    VecDuplicate(phi, &grad_M_jump);
    VecDuplicate(phi, &mu_m_);
    VecDuplicate(phi, &mu_p_);

    Vec Laplacian_u;
    ierr = VecCreateGhostNodes(p4est, nodes, &Laplacian_u); CHKERRXX(ierr);





    BoundaryConditions3D bc;
    bc.setWallTypes(M_bc_wall_type_p);
    bc.setWallValues(M_bc_wall_value_p);

    my_p4est_poisson_jump_nodes_voronoi_t solver(ngbd_n, ngbd_c);
    solver.set_bc(bc);
    solver.set_phi(phi);
    solver.set_diagonal(1);

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    Vec M_plus, M_minus, dM_plus_cte, dM_minus_cte;
    ierr = VecDuplicate(phi, &dM_minus_cte); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &dM_plus_cte); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &M_plus); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &M_minus); CHKERRXX(ierr);

    double *M_p[number_ions], *c_rate_p;
    ierr = VecGetArray(charge_rate, &c_rate_p); CHKERRXX(ierr);
    for(unsigned int ion=0; ion<number_ions-1; ++ion)
    {
        ierr = VecGetArray(M_list[ion], &M_p[ion]); CHKERRXX(ierr);
        for(size_t n=0; n<nodes->indep_nodes.elem_count;n++)
        {
            c_rate_p[n] = -M_p[ion][n]*Faraday;
        }
        ierr = VecRestoreArray(M_list[ion], &M_p[ion]); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(charge_rate, &c_rate_p); CHKERRXX(ierr);

    double *u_p, *u_ext_p;
    Vec sol_ext;
    VecDuplicate(sol,&sol_ext);
    VecGetArray(sol, &u_p);
    VecGetArray(sol_ext, &u_ext_p);
    for(size_t n=0; n<nodes->indep_nodes.elem_count;n++)
        u_ext_p[n] = u_p[n];
    VecRestoreArray(sol, &u_p);
    VecRestoreArray(sol_ext, &u_ext_p);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
        phi_p[i] = -phi_p[i];
    ls.extend_Over_Interface_TVD(phi, sol_ext);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
        phi_p[i] = -phi_p[i];

    Vec error;
    VecDuplicate(phi,&error);

    for(unsigned int ion=0; ion<number_ions-1; ++ion) //PAM: remove the -1 in front of the number_ions
    {
        double *Pm_p, *X0_p, *X1_p, *mu_m_p, *mu_p_p, *vn_p;
        ierr = VecGetArray(Pm, &Pm_p); CHKERRXX(ierr);
        ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);
        ierr = VecGetArray(X0, &X0_p); CHKERRXX(ierr);
        ierr = VecGetArray(X1, &X1_p); CHKERRXX(ierr);
        ierr = VecGetArray(mu_m_, &mu_m_p); CHKERRXX(ierr);
        ierr = VecGetArray(mu_p_, &mu_p_p); CHKERRXX(ierr);
        for(size_t n=0; n<nodes->indep_nodes.elem_count;n++)
        {
            mu_m_p[n] = dt_n*d_c;
            mu_p_p[n] = dt_n*d_e;

            double x0, x1;
            x0 = X0_p[n];
            x1 = X1_p[n];

            Pm_p[n] = P0 + P1*beta_0_in(vn_p[n]) + P2*x1;
        }
        ierr = VecRestoreArray(Pm, &Pm_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(vn, &vn_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(X0, &X0_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(X1, &X1_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(mu_m_, &mu_m_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(mu_p_, &mu_p_p); CHKERRXX(ierr);

        solver.set_mu(mu_m_, mu_p_);

        int counter = 0;
        double max_err;
        do{
            max_err = 0;
            // Compute Velocity, Laplacian!
            double *Laplacian_u_p, *ElectroPhoresis_p[3], *ElectroPhoresis_nm1_p[3], *du_p_p;
            ierr = VecGetArray(sol_ext, &u_p); CHKERRXX(ierr);
            ierr = VecGetArray(Laplacian_u, &Laplacian_u_p); CHKERRXX(ierr);
            for(int dir=0; dir<3; ++dir)
            {
                ierr = VecGetArray(ElectroPhoresis_nm1[dir], &ElectroPhoresis_nm1_p[dir]); CHKERRXX(ierr);
                ierr = VecGetArray(ElectroPhoresis[dir], &ElectroPhoresis_p[dir]); CHKERRXX(ierr);
            }
            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n){
                ElectroPhoresis_nm1_p[0][n] = ElectroPhoresis_p[0][n];
                ElectroPhoresis_nm1_p[1][n] = ElectroPhoresis_p[1][n];
                ElectroPhoresis_nm1_p[2][n] = ElectroPhoresis_p[2][n];
            }
            for(int dir=0; dir<3; ++dir)
                ierr = VecRestoreArray(ElectroPhoresis_nm1[dir], &ElectroPhoresis_nm1_p[dir]); CHKERRXX(ierr);

            for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
            {
                p4est_locidx_t n = ngbd_n->get_layer_node(i);
                const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
                if(phi_p[n]>0)
                {
                    ElectroPhoresis_p[0][n] = -mu_e*qnnn.dx_central(u_p);
                    ElectroPhoresis_p[1][n] = -mu_e*qnnn.dy_central(u_p);
                    ElectroPhoresis_p[2][n] = -mu_e*qnnn.dz_central(u_p);
                }
                if(is_interface(ngbd_n,n,phi_p)>0)
                    continue;
                Laplacian_u_p[n] = qnnn.dxx_central(u_p)+qnnn.dyy_central(u_p)+qnnn.dzz_central(u_p);
            }
            ierr = VecGhostUpdateBegin(Laplacian_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(int dir=0; dir<3; ++dir)
            {
                ierr = VecGhostUpdateBegin(ElectroPhoresis[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            }
            for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
            {
                p4est_locidx_t n = ngbd_n->get_local_node(i);
                const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
                if(phi_p[n]>0)
                {
                    ElectroPhoresis_p[0][n] = -mu_e*qnnn.dx_central(u_p);
                    ElectroPhoresis_p[1][n] = -mu_e*qnnn.dy_central(u_p);
                    ElectroPhoresis_p[2][n] = -mu_e*qnnn.dz_central(u_p);
                }
                if(is_interface(ngbd_n,n,phi_p)>0)
                    continue;
                Laplacian_u_p[n] = qnnn.dxx_central(u_p)+qnnn.dyy_central(u_p)+qnnn.dzz_central(u_p);
            }
            ierr = VecGhostUpdateEnd(Laplacian_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(int dir=0; dir<3; ++dir)
            {
                ierr = VecGhostUpdateEnd(ElectroPhoresis[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
                ierr = VecRestoreArray(ElectroPhoresis[dir], &ElectroPhoresis_p[dir]); CHKERRXX(ierr);
            }

            ierr = VecRestoreArray(Laplacian_u, &Laplacian_u_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(sol_ext, &u_p); CHKERRXX(ierr);

            // Advection by electric field using the semi-Lagrangian, we find departure point values M_departure and update them into M_list[ion].
            BoundaryConditions3D bc_interface;
            bc_interface.setInterfaceType(NEUMANN);
            bc_interface.setInterfaceValue(bc_interface_value_p);

            for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                phi_p[i] = -phi_p[i];
            for(int dir=0;dir<3; ++dir)
                ls.extend_Over_Interface(phi, ElectroPhoresis[dir], bc_interface, 2, 20);
            for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                phi_p[i] = -phi_p[i];
            for(int dir=0;dir<3; ++dir)
            {
                VecGetArray(ElectroPhoresis[dir], &ElectroPhoresis_p[dir]);
                for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                    if(phi_p[i]<0)
                        ElectroPhoresis_p[dir][i] = 0;
                VecRestoreArray(ElectroPhoresis[dir], &ElectroPhoresis_p[dir]);
            }
            advect(p4est, nodes, ngbd_n, ElectroPhoresis_nm1, ElectroPhoresis,  dt_nm1, dt_n,  M_list[ion]);

            // measure jump in concentration on the interface
            double *M_plus_p, *M_minus_p;
            VecGetArray(M_plus, &M_plus_p);
            VecGetArray(M_minus, &M_minus_p);
            VecGetArray(M_list[ion], &M_p[ion]);
            for (size_t n = 0; n<nodes->indep_nodes.elem_count; n++)
            {
                if(M_p[ion][n]<0)
                    M_p[ion][n]=EPS;

                M_plus_p[n] = M_p[ion][n];
                M_minus_p[n] = M_p[ion][n];
            }
            VecRestoreArray(M_plus, &M_plus_p);
            VecRestoreArray(M_minus, &M_minus_p);
            VecRestoreArray(M_list[ion], &M_p[ion]);

            ls.extend_Over_Interface_TVD(phi, M_minus);
            for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                phi_p[i] = -phi_p[i];
            ls.extend_Over_Interface_TVD(phi, M_plus);
            for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                phi_p[i] = -phi_p[i];

            // concentration gradients
            double *dMp_p, *dMm_p, *Mp_p, *Mm_p;
            ierr = VecGetArray(M_plus, &Mp_p); CHKERRXX(ierr);
            ierr = VecGetArray(M_minus, &Mm_p); CHKERRXX(ierr);
            ierr = VecGetArray(grad_Mp, &dMp_p); CHKERRXX(ierr);
            ierr = VecGetArray(grad_Mm, &dMm_p); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
            {
                p4est_locidx_t n = ngbd_n->get_layer_node(i);
                const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
                double nx = qnnn.dx_central(phi_p);
                double ny = qnnn.dy_central(phi_p);
                double nz = qnnn.dz_central(phi_p);
                double norm = sqrt(nx*nx+ny*ny+nz*nz);
                norm >EPS ? nx /= norm : nx = 0;
                norm >EPS ? ny /= norm : ny = 0;
                norm >EPS ? nz /= norm : nz = 0;

                dMm_p[n] = nx*qnnn.dx_central(Mm_p);
                dMp_p[n] = nx*qnnn.dx_central(Mp_p);
                dMm_p[n] += ny*qnnn.dy_central(Mm_p);
                dMp_p[n] += ny*qnnn.dy_central(Mp_p);
                dMm_p[n] += nz*qnnn.dz_central(Mm_p);
                dMp_p[n] += nz*qnnn.dz_central(Mp_p);
            }
            ierr = VecGhostUpdateBegin(grad_Mp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecGhostUpdateBegin(grad_Mm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
            {
                p4est_locidx_t n = ngbd_n->get_local_node(i);
                const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
                double nx = qnnn.dx_central(phi_p);
                double ny = qnnn.dy_central(phi_p);
                double nz = qnnn.dz_central(phi_p);
                double norm = sqrt(nx*nx+ny*ny+nz*nz);
                norm >EPS ? nx /= norm : nx = 0;
                norm >EPS ? ny /= norm : ny = 0;
                norm >EPS ? nz /= norm : nz = 0;

                dMm_p[n] = nx*qnnn.dx_central(Mm_p);
                dMp_p[n] = nx*qnnn.dx_central(Mp_p);
                dMm_p[n] += ny*qnnn.dy_central(Mm_p);
                dMp_p[n] += ny*qnnn.dy_central(Mp_p);
                dMm_p[n] += nz*qnnn.dz_central(Mm_p);
                dMp_p[n] += nz*qnnn.dz_central(Mp_p);
            }
            ierr = VecGhostUpdateEnd(grad_Mp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecGhostUpdateEnd(grad_Mm, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
            ierr = VecRestoreArray(grad_Mp, &dMp_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(grad_Mm, &dMm_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(M_plus, &Mp_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(M_minus, &Mm_p); CHKERRXX(ierr);
            //ls.extend_from_interface_to_whole_domain_TVD(phi, grad_Mp, dM_plus_cte);
            //ls.extend_from_interface_to_whole_domain_TVD(phi, grad_Mm, dM_minus_cte);
            ls.extend_Over_Interface_TVD(phi, grad_Mm);
            for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                phi_p[i] = -phi_p[i];
            ls.extend_Over_Interface_TVD(phi, grad_Mp);
            for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
                phi_p[i] = -phi_p[i];

            // measure real jump values after solve with correct jumps
            double *M_jump_p;
            VecGetArray(M_jump, &M_jump_p);
            VecGetArray(M_plus, &M_plus_p);
            VecGetArray(M_minus, &M_minus_p);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
            {
                M_jump_p[n] = M_plus_p[n] - M_minus_p[n];
            }
            /*         my_p4est_interpolation_nodes_t interp_np(ngbd_n);
           my_p4est_interpolation_nodes_t interp_nm(ngbd_n);
           interp_np.set_input(M_plus, linear);
           interp_nm.set_input(M_minus, linear);
                             double diag = (zmaxx-zminn)/pow(2.0,(double) lmax)/2.0;
                        for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
                        {
                            if(ABS(phi_p[n])<EPS || is_interface(ngbd_n,n,phi_p)<0)
                            {
                                M_jump_p[n] = M_plus_p[n] - M_minus_p[n];
                                continue;
                            }
                            if(is_interface(ngbd_n,n,phi_p)>0)
                            {
                                const quad_neighbor_nodes_of_node_t qnnn = ngbd_n->get_neighbors(n);
                                double x = node_x_fr_n(n, p4est, nodes);
                                double y = node_y_fr_n(n, p4est, nodes);
                                double z = node_z_fr_n(n, p4est, nodes);


                                double xyz_np[3] = {x,y,z};

                                double nx = qnnn.dx_central(phi_p);
                                double ny = qnnn.dy_central(phi_p);
                                double nz = qnnn.dz_central(phi_p);
                                double norm = sqrt(nx*nx+ny*ny+nz*nz);
                                norm >EPS ? nx /= norm : nx = 0;
                                norm >EPS ? ny /= norm : ny = 0;
                                norm >EPS ? nz /= norm : nz = 0;
                                double m_in, m_out;
                                double dist = ABS(phi_p[n]);
                                if(phi_p[n]>0)
                                {
                                    xyz_np[0] += nx*(diag/5 - dist);
                                    xyz_np[1] += ny*(diag/5 - dist);
                                    xyz_np[2] += nz*(diag/5 - dist);
                                    interp_nm.add_point(0, xyz_np);
                                    interp_nm.interpolate(&m_in);
                                    interp_np.add_point(0, xyz_np);
                                    interp_np.interpolate(&m_out);
                                    double tmp1 = (m_out - m_in);
                                    interp_np.clear();
                                    interp_nm.clear();

                                    xyz_np[0] += -nx*2*diag/5;
                                    xyz_np[1] += -ny*2*diag/5;
                                    xyz_np[2] += -nz*2*diag/5;
                                    interp_nm.add_point(0, xyz_np);
                                    interp_nm.interpolate(&m_in);
                                    interp_np.add_point(0, xyz_np);
                                    interp_np.interpolate(&m_out);

                                    M_jump_p[n] = (tmp1 + (m_out - m_in))/2.0;
                                    interp_np.clear();
                                    interp_nm.clear();
                                    continue;
                                }else{
                                    xyz_np[0] += -nx*(diag/5 - dist);
                                    xyz_np[1] += -ny*(diag/5 - dist);
                                    xyz_np[2] += -nz*(diag/5 - dist);
                                    interp_nm.add_point(0, xyz_np);
                                    interp_nm.interpolate(&m_in);
                                    interp_np.add_point(0, xyz_np);
                                    interp_np.interpolate(&m_out);
                                    double tmp1 = (m_out - m_in);
                                    interp_np.clear();
                                    interp_nm.clear();

                                    xyz_np[0] += nx*2*diag/5;
                                    xyz_np[1] += ny*2*diag/5;
                                    xyz_np[2] += nz*2*diag/5;
                                    interp_nm.add_point(0, xyz_np);
                                    interp_nm.interpolate(&m_in);
                                    interp_np.add_point(0, xyz_np);
                                    interp_np.interpolate(&m_out);

                                    M_jump_p[n] = (tmp1 + (m_out - m_in))/2.0;
                                    interp_np.clear();
                                    interp_nm.clear();
                                    continue;
                                }

                            }
                        } */
            VecRestoreArray(M_jump, &M_jump_p);
            VecRestoreArray(M_plus, &M_plus_p);
            VecRestoreArray(M_minus, &M_minus_p);
            ls.extend_from_interface_to_whole_domain(phi,M_jump,M_jump);
            // end of measure current jump values

            // compute gradient/RHS in solution across interface
            double max_jump = 0;
            double *grad_M_jump_p, *rhs_m_p, *rhs_p_p, *dM_m_p;
            ierr = VecGetArray(grad_up, &du_p_p); CHKERRXX(ierr);
            ierr = VecGetArray(grad_M_jump, &grad_M_jump_p); CHKERRXX(ierr);
            ierr = VecGetArray(M_list[ion],&M_p[ion]); CHKERRXX(ierr);
            ierr = VecGetArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
            ierr = VecGetArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
            ierr = VecGetArray(grad_Mm, &dM_m_p); CHKERRXX(ierr);
            ierr = VecGetArray(Pm, &Pm_p); CHKERRXX(ierr);
            ierr = VecGetArray(M_jump, &M_jump_p); CHKERRXX(ierr);
            ierr = VecGetArray(M_plus, &M_plus_p); CHKERRXX(ierr);
            ierr = VecGetArray(Laplacian_u, &Laplacian_u_p); CHKERRXX(ierr);
            for(size_t n=0; n<nodes->indep_nodes.elem_count;n++)
            {
                rhs_p_p[n] = M_p[ion][n];// +dt_n*mu_e*Laplacian_u_p[n]*M_p[ion][n];
                rhs_m_p[n] =M_p[ion][n];
                // no transport from inside to outside! E.n>0 => set advection to 0, continuous diffusion.
                grad_M_jump_p[n] = MIN(0.0, dt_n*mu_e*M_plus_p[n]*du_p_p[n]);

                if(is_interface(ngbd_n,n,phi_p)>0 &&  Pm_p[n]>EPS && ABS(dM_m_p[n])>EPS)
                {
                    if(ABS(M_jump_p[n])>ABS(d_c*dM_m_p[n]/Pm_p[n]))
                        M_jump_p[n] = d_c*dM_m_p[n]/Pm_p[n];
                }

                if(is_interface(ngbd_n, n,phi_p)>0)
                    max_jump = MAX(M_jump_p[n], ABS(max_jump));
            }
            ierr = VecRestoreArray(Laplacian_u, &Laplacian_u_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(M_plus, &M_plus_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(M_list[ion],&M_p[ion]); CHKERRXX(ierr);
            ierr = VecRestoreArray(grad_M_jump, &grad_M_jump_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(grad_up, &du_p_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(rhs_m, &rhs_m_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(rhs_p, &rhs_p_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(M_jump, &M_jump_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(grad_Mm, &dM_m_p); CHKERRXX(ierr);
            ierr = VecRestoreArray(Pm, &Pm_p); CHKERRXX(ierr);
            ls.extend_from_interface_to_whole_domain_TVD(phi, M_jump, M_jump);
            ls.extend_from_interface_to_whole_domain_TVD(phi, grad_M_jump, grad_M_jump);


            solver.set_u_jump(M_jump);
            solver.set_rhs(rhs_m,rhs_p);
            solver.set_mu_grad_u_jump(grad_M_jump);

            solver.solve(M_list[ion]);

            //   export jump values for evaluation.
            VecGetArray(M_list[1], &M_p[1]);
            VecGetArray(M_jump, &M_jump_p);
            for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
            {
                M_p[1][n] = M_jump_p[n];
            }
            VecRestoreArray(M_list[1], &M_p[1]);
            VecRestoreArray(M_jump, &M_jump_p);
            // measure error on interface, relative error in L2 norm.
            /*         double *error_p;
                        VecGetArray(error,&error_p);
                        ierr = VecGetArray(M_list[ion], &M_np_p[ion]); CHKERRXX(ierr);
                        ierr = VecGetArray(M_list_n[ion], &M_p[ion]); CHKERRXX(ierr);
                        for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
                        {
                            error_p[n] = M_p[ion][n]*M_p[ion][n];
                        }
                        VecRestoreArray(error,&error_p);
                        ierr = VecRestoreArray(M_list[ion], &M_np_p[ion]); CHKERRXX(ierr);
                        ierr = VecRestoreArray(M_list_n[ion], &M_p[ion]); CHKERRXX(ierr);
                        double M_k_integral = sqrt(integrate_over_interface(p4est,nodes,phi,error));

                        VecGetArray(error,&error_p);
                        ierr = VecGetArray(M_list[ion], &M_np_p[ion]); CHKERRXX(ierr);
                        ierr = VecGetArray(M_list_n[ion], &M_p[ion]); CHKERRXX(ierr);
                        for(unsigned int n=0; n<nodes->indep_nodes.elem_count;n++)
                        {
                            error_p[n] = (M_np_p[ion][n]-M_p[ion][n])*(M_np_p[ion][n]-M_p[ion][n]);
                            M_p[ion][n] = M_np_p[ion][n];
                        }
                        VecRestoreArray(error,&error_p);
                        ierr = VecRestoreArray(M_list[ion], &M_np_p[ion]); CHKERRXX(ierr);
                        ierr = VecRestoreArray(M_list_n[ion], &M_p[ion]); CHKERRXX(ierr);
                        max_err = sqrt(integrate_over_interface(p4est, nodes, phi, error))/M_k_integral; */
            counter++;
            /*MPI_Reduce(&max_err, &max_err, 1, MPI_DOUBLE,MPI_MAX, 0, p4est->mpicomm);
            MPI_Bcast(&max_err, 1, MPI_DOUBLE, 0, p4est->mpicomm);*/
            //          PetscPrintf(p4est->mpicomm, ">> >> >> Diffusion of ion # %d: iteration # %d just ended! Maximum relative error on membrane is %g\n", ion, counter, max_err);
        }while(counter<0);//max_err>1e-8);//max_err>1e-7);
    }
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    ierr = VecGetArray(charge_rate, &c_rate_p); CHKERRXX(ierr);
    for(unsigned int ion=0; ion<number_ions-1; ++ion)
    {
        ierr = VecGetArray(M_list[ion], &M_p[ion]); CHKERRXX(ierr);
        for(size_t n=0; n<nodes->indep_nodes.elem_count;n++)
        {
            c_rate_p[n] += (M_p[ion][n])*Faraday;
        }
        ierr = VecRestoreArray(M_list[ion], &M_p[ion]); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(charge_rate, &c_rate_p); CHKERRXX(ierr);

    VecDestroy(sol_ext);
    VecDestroy(grad_Mm);
    VecDestroy(grad_Mp);
    VecDestroy(error);
    VecDestroy(mu_m_);
    VecDestroy(mu_p_);
    VecDestroy(M_jump);
    VecDestroy(grad_M_jump);
    VecDestroy(rhs_m);
    VecDestroy(rhs_p);
    VecDestroy(Laplacian_u);
    VecDestroy(M_plus);
    VecDestroy(M_minus);
    VecDestroy(dM_plus_cte);
    VecDestroy(dM_minus_cte);

}


void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol, int compt, Vec X0, Vec X1, Vec Sm, Vec vn, Vec err, Vec M_list[number_ions], Vec Pm, Vec charge_rate, Vec cell_number)
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

    double *phi_p, *sol_p, *X0_p, *X1_p, *Sm_p, *vn_p, *err_p, *Pm_p, *c_rate_p, *cell_number_p;
    std::vector<const double *> M_p(number_ions);
    for(unsigned int i=0; i<2; ++i)
    {
        ierr = VecGetArrayRead(M_list[i], &M_p[i]); CHKERRXX(ierr);
    }

    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArray(X0, &X0_p); CHKERRXX(ierr);
    ierr = VecGetArray(X1, &X1_p); CHKERRXX(ierr);
    ierr = VecGetArray(Sm, &Sm_p); CHKERRXX(ierr);
    ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);
    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    ierr = VecGetArray(Pm, &Pm_p); CHKERRXX(ierr);
    ierr = VecGetArray(charge_rate, &c_rate_p); CHKERRXX(ierr);
    ierr = VecGetArray(cell_number, &cell_number_p); CHKERRXX(ierr);
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
                           13, 1, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "mu", mu_p_,
                           VTK_POINT_DATA, "sol", sol_p,
                           VTK_POINT_DATA, "X0", X0_p,
                           VTK_POINT_DATA, "X1", X1_p,
                           VTK_POINT_DATA, "vn", vn_p,
                           VTK_POINT_DATA, "err", err_p,
                           VTK_POINT_DATA, "Sm", Sm_p,
                           VTK_POINT_DATA, "M_1", M_p[0],
            VTK_POINT_DATA, "M_2", M_p[1],
            VTK_POINT_DATA, "Charge Diff", c_rate_p,
            VTK_POINT_DATA, "Cell Numbers", cell_number_p,
            VTK_POINT_DATA, "Pm", Pm_p,
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
    ierr = VecRestoreArray(Pm, &Pm_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(charge_rate, &c_rate_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(cell_number, &cell_number_p); CHKERRXX(ierr);
    for(unsigned int i=0; i<2; ++i)
    {
        ierr = VecRestoreArrayRead(M_list[i], &M_p[i]); CHKERRXX(ierr);
    }

    PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}




void fill_cell(my_p4est_node_neighbors_t *ngbd,  p4est_nodes_t *nodes, const double *phi_p, double *cell_number_p, int number, p4est_locidx_t n)
{
    stack<size_t> st;
    st.push(n);
    while(!st.empty())
    {
        size_t k = st.top();
        st.pop();
        cell_number_p[k] = number;
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[k];
        if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]<0 && cell_number_p[qnnn.node_m00_mm]<0) st.push(qnnn.node_m00_mm);
        if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]<0 && cell_number_p[qnnn.node_m00_pm]<0) st.push(qnnn.node_m00_pm);
        if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]<0 && cell_number_p[qnnn.node_p00_mm]<0) st.push(qnnn.node_p00_mm);
        if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]<0 && cell_number_p[qnnn.node_p00_pm]<0) st.push(qnnn.node_p00_pm);
        if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]<0 && cell_number_p[qnnn.node_0m0_mm]<0) st.push(qnnn.node_0m0_mm);
        if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]<0 && cell_number_p[qnnn.node_0m0_pm]<0) st.push(qnnn.node_0m0_pm);
        if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]<0 && cell_number_p[qnnn.node_0p0_mm]<0) st.push(qnnn.node_0p0_mm);
        if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]<0 && cell_number_p[qnnn.node_0p0_pm]<0) st.push(qnnn.node_0p0_pm);
        if(qnnn.node_00m_mm<nodes->num_owned_indeps && qnnn.d_00m_m0==0 && phi_p[qnnn.node_00m_mm]<0 && cell_number_p[qnnn.node_00m_mm]<0) st.push(qnnn.node_00m_mm);
        if(qnnn.node_00m_pm<nodes->num_owned_indeps && qnnn.d_00m_p0==0 && phi_p[qnnn.node_00m_pm]<0 && cell_number_p[qnnn.node_00m_pm]<0) st.push(qnnn.node_00m_pm);
        if(qnnn.node_00p_mm<nodes->num_owned_indeps && qnnn.d_00p_m0==0 && phi_p[qnnn.node_00p_mm]<0 && cell_number_p[qnnn.node_00p_mm]<0) st.push(qnnn.node_00p_mm);
        if(qnnn.node_00p_pm<nodes->num_owned_indeps && qnnn.d_00p_p0==0 && phi_p[qnnn.node_00p_pm]<0 && cell_number_p[qnnn.node_00p_pm]<0) st.push(qnnn.node_00p_pm);

        if(qnnn.node_m00_mp<nodes->num_owned_indeps && qnnn.d_m00_0m==0 && phi_p[qnnn.node_m00_mp]<0 && cell_number_p[qnnn.node_m00_mp]<0) st.push(qnnn.node_m00_mp);
        if(qnnn.node_m00_pp<nodes->num_owned_indeps && qnnn.d_m00_0p==0 && phi_p[qnnn.node_m00_pp]<0 && cell_number_p[qnnn.node_m00_pp]<0) st.push(qnnn.node_m00_pp);
        if(qnnn.node_p00_mp<nodes->num_owned_indeps && qnnn.d_p00_0m==0 && phi_p[qnnn.node_p00_mp]<0 && cell_number_p[qnnn.node_p00_mp]<0) st.push(qnnn.node_p00_mp);
        if(qnnn.node_p00_pp<nodes->num_owned_indeps && qnnn.d_p00_0p==0 && phi_p[qnnn.node_p00_pp]<0 && cell_number_p[qnnn.node_p00_pp]<0) st.push(qnnn.node_p00_pp);
        if(qnnn.node_0m0_mp<nodes->num_owned_indeps && qnnn.d_0m0_0m==0 && phi_p[qnnn.node_0m0_mp]<0 && cell_number_p[qnnn.node_0m0_mp]<0) st.push(qnnn.node_0m0_mp);
        if(qnnn.node_0m0_pp<nodes->num_owned_indeps && qnnn.d_0m0_0p==0 && phi_p[qnnn.node_0m0_pp]<0 && cell_number_p[qnnn.node_0m0_pp]<0) st.push(qnnn.node_0m0_pp);
        if(qnnn.node_0p0_mp<nodes->num_owned_indeps && qnnn.d_0p0_0m==0 && phi_p[qnnn.node_0p0_mp]<0 && cell_number_p[qnnn.node_0p0_mp]<0) st.push(qnnn.node_0p0_mp);
        if(qnnn.node_0p0_pp<nodes->num_owned_indeps && qnnn.d_0p0_0p==0 && phi_p[qnnn.node_0p0_pp]<0 && cell_number_p[qnnn.node_0p0_pp]<0) st.push(qnnn.node_0p0_pp);
        if(qnnn.node_00m_mp<nodes->num_owned_indeps && qnnn.d_00m_0m==0 && phi_p[qnnn.node_00m_mp]<0 && cell_number_p[qnnn.node_00m_mp]<0) st.push(qnnn.node_00m_mp);
        if(qnnn.node_00m_pp<nodes->num_owned_indeps && qnnn.d_00m_0p==0 && phi_p[qnnn.node_00m_pp]<0 && cell_number_p[qnnn.node_00m_pp]<0) st.push(qnnn.node_00m_pp);
        if(qnnn.node_00p_mp<nodes->num_owned_indeps && qnnn.d_00p_0m==0 && phi_p[qnnn.node_00p_mp]<0 && cell_number_p[qnnn.node_00p_mp]<0) st.push(qnnn.node_00p_mp);
        if(qnnn.node_00p_pp<nodes->num_owned_indeps && qnnn.d_00p_0p==0 && phi_p[qnnn.node_00p_pp]<0 && cell_number_p[qnnn.node_00p_pp]<0) st.push(qnnn.node_00p_pp);
    }
}


void find_connected_ghost_cells(my_p4est_node_neighbors_t *ngbd,  p4est_nodes_t *nodes, const double *phi_p, double *cell_number_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited)
{
    stack<size_t> st;
    st.push(n);
    while(!st.empty())
    {
        size_t k = st.top();
        st.pop();
        visited[k] = true;

        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[k];
        if(qnnn.node_m00_mm<nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]<0 && !visited[qnnn.node_m00_mm]) st.push(qnnn.node_m00_mm);
        if(qnnn.node_m00_pm<nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]<0 && !visited[qnnn.node_m00_pm]) st.push(qnnn.node_m00_pm);
        if(qnnn.node_p00_mm<nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]<0 && !visited[qnnn.node_p00_mm]) st.push(qnnn.node_p00_mm);
        if(qnnn.node_p00_pm<nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]<0 && !visited[qnnn.node_p00_pm]) st.push(qnnn.node_p00_pm);
        if(qnnn.node_0m0_mm<nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]<0 && !visited[qnnn.node_0m0_mm]) st.push(qnnn.node_0m0_mm);
        if(qnnn.node_0m0_pm<nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]<0 && !visited[qnnn.node_0m0_pm]) st.push(qnnn.node_0m0_pm);
        if(qnnn.node_0p0_mm<nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]<0 && !visited[qnnn.node_0p0_mm]) st.push(qnnn.node_0p0_mm);
        if(qnnn.node_0p0_pm<nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]<0 && !visited[qnnn.node_0p0_pm]) st.push(qnnn.node_0p0_pm);
        if(qnnn.node_00m_mm<nodes->num_owned_indeps && qnnn.d_00m_m0==0 && phi_p[qnnn.node_00m_mm]<0 && !visited[qnnn.node_00m_mm]) st.push(qnnn.node_00m_mm);
        if(qnnn.node_00m_pm<nodes->num_owned_indeps && qnnn.d_00m_p0==0 && phi_p[qnnn.node_00m_pm]<0 && !visited[qnnn.node_00m_pm]) st.push(qnnn.node_00m_pm);
        if(qnnn.node_00p_mm<nodes->num_owned_indeps && qnnn.d_00p_m0==0 && phi_p[qnnn.node_00p_mm]<0 && !visited[qnnn.node_00p_mm]) st.push(qnnn.node_00p_mm);
        if(qnnn.node_00p_pm<nodes->num_owned_indeps && qnnn.d_00p_p0==0 && phi_p[qnnn.node_00p_pm]<0 && !visited[qnnn.node_00p_pm]) st.push(qnnn.node_00p_pm);



        if(qnnn.node_m00_mp<nodes->num_owned_indeps && qnnn.d_m00_0m==0 && phi_p[qnnn.node_m00_mp]<0 && !visited[qnnn.node_m00_mp]) st.push(qnnn.node_m00_mp);
        if(qnnn.node_m00_pp<nodes->num_owned_indeps && qnnn.d_m00_0p==0 && phi_p[qnnn.node_m00_pp]<0 && !visited[qnnn.node_m00_pp]) st.push(qnnn.node_m00_pp);
        if(qnnn.node_p00_mp<nodes->num_owned_indeps && qnnn.d_p00_0m==0 && phi_p[qnnn.node_p00_mp]<0 && !visited[qnnn.node_p00_mp]) st.push(qnnn.node_p00_mp);
        if(qnnn.node_p00_pp<nodes->num_owned_indeps && qnnn.d_p00_0p==0 && phi_p[qnnn.node_p00_pp]<0 && !visited[qnnn.node_p00_pp]) st.push(qnnn.node_p00_pp);
        if(qnnn.node_0m0_mp<nodes->num_owned_indeps && qnnn.d_0m0_0m==0 && phi_p[qnnn.node_0m0_mp]<0 && !visited[qnnn.node_0m0_mp]) st.push(qnnn.node_0m0_mp);
        if(qnnn.node_0m0_pp<nodes->num_owned_indeps && qnnn.d_0m0_0p==0 && phi_p[qnnn.node_0m0_pp]<0 && !visited[qnnn.node_0m0_pp]) st.push(qnnn.node_0m0_pp);
        if(qnnn.node_0p0_mp<nodes->num_owned_indeps && qnnn.d_0p0_0m==0 && phi_p[qnnn.node_0p0_mp]<0 && !visited[qnnn.node_0p0_mp]) st.push(qnnn.node_0p0_mp);
        if(qnnn.node_0p0_pp<nodes->num_owned_indeps && qnnn.d_0p0_0p==0 && phi_p[qnnn.node_0p0_pp]<0 && !visited[qnnn.node_0p0_pp]) st.push(qnnn.node_0p0_pp);
        if(qnnn.node_00m_mp<nodes->num_owned_indeps && qnnn.d_00m_0m==0 && phi_p[qnnn.node_00m_mp]<0 && !visited[qnnn.node_00m_mp]) st.push(qnnn.node_00m_mp);
        if(qnnn.node_00m_pp<nodes->num_owned_indeps && qnnn.d_00m_0p==0 && phi_p[qnnn.node_00m_pp]<0 && !visited[qnnn.node_00m_pp]) st.push(qnnn.node_00m_pp);
        if(qnnn.node_00p_mp<nodes->num_owned_indeps && qnnn.d_00p_0m==0 && phi_p[qnnn.node_00p_mp]<0 && !visited[qnnn.node_00p_mp]) st.push(qnnn.node_00p_mp);
        if(qnnn.node_00p_pp<nodes->num_owned_indeps && qnnn.d_00p_0p==0 && phi_p[qnnn.node_00p_pp]<0 && !visited[qnnn.node_00p_pp]) st.push(qnnn.node_00p_pp);



        /* check connected ghost cell and add to list if new */
        if(qnnn.node_m00_mm>=nodes->num_owned_indeps && qnnn.d_m00_m0==0 && phi_p[qnnn.node_m00_mm]<0 && !contains(connected, cell_number_p[qnnn.node_m00_mm])) connected.push_back(cell_number_p[qnnn.node_m00_mm]);
        if(qnnn.node_m00_pm>=nodes->num_owned_indeps && qnnn.d_m00_p0==0 && phi_p[qnnn.node_m00_pm]<0 && !contains(connected, cell_number_p[qnnn.node_m00_pm])) connected.push_back(cell_number_p[qnnn.node_m00_pm]);
        if(qnnn.node_p00_mm>=nodes->num_owned_indeps && qnnn.d_p00_m0==0 && phi_p[qnnn.node_p00_mm]<0 && !contains(connected, cell_number_p[qnnn.node_p00_mm])) connected.push_back(cell_number_p[qnnn.node_p00_mm]);
        if(qnnn.node_p00_pm>=nodes->num_owned_indeps && qnnn.d_p00_p0==0 && phi_p[qnnn.node_p00_pm]<0 && !contains(connected, cell_number_p[qnnn.node_p00_pm])) connected.push_back(cell_number_p[qnnn.node_p00_pm]);
        if(qnnn.node_0m0_mm>=nodes->num_owned_indeps && qnnn.d_0m0_m0==0 && phi_p[qnnn.node_0m0_mm]<0 && !contains(connected, cell_number_p[qnnn.node_0m0_mm])) connected.push_back(cell_number_p[qnnn.node_0m0_mm]);
        if(qnnn.node_0m0_pm>=nodes->num_owned_indeps && qnnn.d_0m0_p0==0 && phi_p[qnnn.node_0m0_pm]<0 && !contains(connected, cell_number_p[qnnn.node_0m0_pm])) connected.push_back(cell_number_p[qnnn.node_0m0_pm]);
        if(qnnn.node_0p0_mm>=nodes->num_owned_indeps && qnnn.d_0p0_m0==0 && phi_p[qnnn.node_0p0_mm]<0 && !contains(connected, cell_number_p[qnnn.node_0p0_mm])) connected.push_back(cell_number_p[qnnn.node_0p0_mm]);
        if(qnnn.node_0p0_pm>=nodes->num_owned_indeps && qnnn.d_0p0_p0==0 && phi_p[qnnn.node_0p0_pm]<0 && !contains(connected, cell_number_p[qnnn.node_0p0_pm])) connected.push_back(cell_number_p[qnnn.node_0p0_pm]);
        if(qnnn.node_00m_mm>=nodes->num_owned_indeps && qnnn.d_00m_m0==0 && phi_p[qnnn.node_00m_mm]<0 && !contains(connected, cell_number_p[qnnn.node_00m_mm])) connected.push_back(cell_number_p[qnnn.node_00m_mm]);
        if(qnnn.node_00m_pm>=nodes->num_owned_indeps && qnnn.d_00m_p0==0 && phi_p[qnnn.node_00m_pm]<0 && !contains(connected, cell_number_p[qnnn.node_00m_pm])) connected.push_back(cell_number_p[qnnn.node_00m_pm]);
        if(qnnn.node_00p_mm>=nodes->num_owned_indeps && qnnn.d_00p_m0==0 && phi_p[qnnn.node_00p_mm]<0 && !contains(connected, cell_number_p[qnnn.node_00p_mm])) connected.push_back(cell_number_p[qnnn.node_00p_mm]);
        if(qnnn.node_00p_pm>=nodes->num_owned_indeps && qnnn.d_00p_p0==0 && phi_p[qnnn.node_00p_pm]<0 && !contains(connected, cell_number_p[qnnn.node_00p_pm])) connected.push_back(cell_number_p[qnnn.node_00p_pm]);


        if(qnnn.node_m00_mp>=nodes->num_owned_indeps && qnnn.d_m00_0m==0 && phi_p[qnnn.node_m00_mp]<0 && !contains(connected, cell_number_p[qnnn.node_m00_mp])) connected.push_back(cell_number_p[qnnn.node_m00_mp]);
        if(qnnn.node_m00_pp>=nodes->num_owned_indeps && qnnn.d_m00_0p==0 && phi_p[qnnn.node_m00_pp]<0 && !contains(connected, cell_number_p[qnnn.node_m00_pp])) connected.push_back(cell_number_p[qnnn.node_m00_pp]);
        if(qnnn.node_p00_mp>=nodes->num_owned_indeps && qnnn.d_p00_0m==0 && phi_p[qnnn.node_p00_mp]<0 && !contains(connected, cell_number_p[qnnn.node_p00_mp])) connected.push_back(cell_number_p[qnnn.node_p00_mp]);
        if(qnnn.node_p00_pp>=nodes->num_owned_indeps && qnnn.d_p00_0p==0 && phi_p[qnnn.node_p00_pp]<0 && !contains(connected, cell_number_p[qnnn.node_p00_pp])) connected.push_back(cell_number_p[qnnn.node_p00_pp]);
        if(qnnn.node_0m0_mp>=nodes->num_owned_indeps && qnnn.d_0m0_0m==0 && phi_p[qnnn.node_0m0_mp]<0 && !contains(connected, cell_number_p[qnnn.node_0m0_mp])) connected.push_back(cell_number_p[qnnn.node_0m0_mp]);
        if(qnnn.node_0m0_pp>=nodes->num_owned_indeps && qnnn.d_0m0_0p==0 && phi_p[qnnn.node_0m0_pp]<0 && !contains(connected, cell_number_p[qnnn.node_0m0_pp])) connected.push_back(cell_number_p[qnnn.node_0m0_pp]);
        if(qnnn.node_0p0_mp>=nodes->num_owned_indeps && qnnn.d_0p0_0m==0 && phi_p[qnnn.node_0p0_mp]<0 && !contains(connected, cell_number_p[qnnn.node_0p0_mp])) connected.push_back(cell_number_p[qnnn.node_0p0_mp]);
        if(qnnn.node_0p0_pp>=nodes->num_owned_indeps && qnnn.d_0p0_0p==0 && phi_p[qnnn.node_0p0_pp]<0 && !contains(connected, cell_number_p[qnnn.node_0p0_pp])) connected.push_back(cell_number_p[qnnn.node_0p0_pp]);
        if(qnnn.node_00m_mp>=nodes->num_owned_indeps && qnnn.d_00m_0m==0 && phi_p[qnnn.node_00m_mp]<0 && !contains(connected, cell_number_p[qnnn.node_00m_mp])) connected.push_back(cell_number_p[qnnn.node_00m_mp]);
        if(qnnn.node_00m_pp>=nodes->num_owned_indeps && qnnn.d_00m_0p==0 && phi_p[qnnn.node_00m_pp]<0 && !contains(connected, cell_number_p[qnnn.node_00m_pp])) connected.push_back(cell_number_p[qnnn.node_00m_pp]);
        if(qnnn.node_00p_mp>=nodes->num_owned_indeps && qnnn.d_00p_0m==0 && phi_p[qnnn.node_00p_mp]<0 && !contains(connected, cell_number_p[qnnn.node_00p_mp])) connected.push_back(cell_number_p[qnnn.node_00p_mp]);
        if(qnnn.node_00p_pp>=nodes->num_owned_indeps && qnnn.d_00p_0p==0 && phi_p[qnnn.node_00p_pp]<0 && !contains(connected, cell_number_p[qnnn.node_00p_pp])) connected.push_back(cell_number_p[qnnn.node_00p_pp]);
    }
}

void compute_cell_number(p4est_t *p4est, my_p4est_node_neighbors_t *ngbd,  p4est_nodes_t *nodes, Vec phi,  Vec &cell_number)
{
    PetscErrorCode ierr;
    int nb_cells_total = 0;
    int proc_padding = 1e6;

    /* first everyone compute the local numbers */
    std::vector<int> nb_cells_loc(p4est->mpisize);
    nb_cells_loc[p4est->mpirank] = p4est->mpirank*proc_padding;

    double *cell_number_p, *phi_p;
    ierr = VecGetArray(cell_number, &cell_number_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_layer_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_layer_node(i);
        if(phi_p[n]<0 && cell_number_p[n]<0)
        {
            fill_cell(ngbd, nodes, phi_p, cell_number_p, nb_cells_loc[p4est->mpirank], n);
            nb_cells_loc[p4est->mpirank]++;
        }
    }
    ierr = VecGhostUpdateBegin(cell_number, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t i=0; i<ngbd->get_local_size(); ++i)
    {
        p4est_locidx_t n = ngbd->get_local_node(i);
        if(phi_p[n]<0 && cell_number_p[n]<0)
        {
            fill_cell(ngbd, nodes, phi_p, cell_number_p, nb_cells_loc[p4est->mpirank], n);
            nb_cells_loc[p4est->mpirank]++;
        }
    }
    ierr = VecGhostUpdateEnd(cell_number, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(cell_number, &cell_number_p); CHKERRXX(ierr);
    ierr = VecGetArray(cell_number, &cell_number_p); CHKERRXX(ierr);
    /* get remote number of cells to prepare graph communication structure */
    int mpiret = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &nb_cells_loc[0], 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    /* compute offset for each process */
    std::vector<int> proc_offset(p4est->mpisize+1);
    proc_offset[0] = 0;
    for(int p=0; p<p4est->mpisize; ++p){
        proc_offset[p+1] = proc_offset[p] + (nb_cells_loc[p]%proc_padding);
    }

    /* build a local graph with
     *   - vertices = cell number
     *   - edges    = connected cells
     * in order to simplify the communications, the graph is stored as a full matrix. Given the sparsity, this can be optimized ...
     */

    int nb_cells_g = proc_offset[p4est->mpisize]; //PAMM
    std::vector<int> graph(nb_cells_g*nb_cells_g, 0);

    std::vector<double> connected;
    std::vector<bool> visited(nodes->num_owned_indeps, false);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
        if(cell_number_p[n]>=0 && !visited[n])
        {
            find_connected_ghost_cells(ngbd, nodes, phi_p, cell_number_p, n, connected, visited);
            for(unsigned int i=0; i<connected.size(); ++i)
            {
                int local_id = proc_offset[p4est->mpirank]+static_cast<int>(cell_number_p[n])%proc_padding;
                int remote_id = proc_offset[static_cast<int>(connected[i])/proc_padding] + (static_cast<int>(connected[i])%proc_padding);
                graph[nb_cells_g*local_id + remote_id] = 1;
            }
            connected.clear();
        }
    }

    std::vector<int> rcvcounts(p4est->mpisize);
    std::vector<int> displs(p4est->mpisize);
    for(int p=0; p<p4est->mpisize; ++p)
    {
        rcvcounts[p] = (nb_cells_loc[p]%proc_padding) * nb_cells_g;
        displs[p] = proc_offset[p]*nb_cells_g;
    }

    mpiret = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &graph[0], &rcvcounts[0], &displs[0], MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    std::vector<int> graph_numbering(nb_cells_g,-1);
    stack<int> st;
    for(int i=0; i<nb_cells_g; ++i)
    {
        if(graph_numbering[i]==-1)
        {
            st.push(i);
            while(!st.empty())
            {
                int k = st.top();
                st.pop();
                graph_numbering[k] = nb_cells_total;
                for(int j=0; j<nb_cells_g; ++j)
                {
                    int nj = k*nb_cells_g+j;
                    if(graph[nj] && graph_numbering[j]==-1)
                        st.push(j);
                }
            }
            nb_cells_total++;
        }
    }
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        if(cell_number_p[n]>=0)
        {
            int index = proc_offset[static_cast<int>(cell_number_p[n])/proc_padding] + (static_cast<int>(cell_number_p[n])%proc_padding);
            cell_number_p[n] = graph_numbering[index];
        }
    }

    ierr = VecRestoreArray(cell_number, &cell_number_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
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

    // scale domain to unit length
    double length_scale = 1;//(zmaxx - zminn);
    zmaxx *= length_scale;
    zminn *= length_scale;
    xmax *= length_scale;
    xmin *= length_scale;
    ymax *= length_scale;
    ymin *= length_scale;

    r0 *= length_scale;
    a *= length_scale;
    b *= length_scale;
    c *= length_scale;
    boxVolume *= (length_scale*length_scale*length_scale);
    ClusterRadius *= length_scale;
    SphereVolume *= (length_scale*length_scale*length_scale);
    cellVolume *= (length_scale*length_scale*length_scale);
    R1 *= length_scale;
    R2 *= length_scale;


    //scale electroporation and time
    E *= 1;//(length_scale); // in practice this only ensures BCs on the walls.
    sigma_c *= (length_scale*length_scale);// NOTE: when adding RHS for charge, should scale charges!
    sigma_e *= (length_scale*length_scale);
    S0 *= length_scale;
    S1 *= length_scale;
    SL *= length_scale;
    Cm *= length_scale;
    Vep *= 1;
    Sm_threshold_value *= length_scale;
    epsilon_0 *= 1;//(length_scale);
    omega *= 1;
    tf /= 1;
    tau_ep /= 1;
    tau_perm /= 1;
    tau_res /= 1;

    // now scale diffusion-advection
    int DIFF_TO_EP_RATIO = 1; // 10 is good with dt=1e-8
    M_0 *= 1;
    M_boundary *= 1;

    d_e *= (length_scale*length_scale); //this is b/c dt is scaled down with dx too! [physical time=t_n/length_scale]
    d_c *= (length_scale*length_scale);
    mu_e *= (length_scale*length_scale);
    mu_c *= (length_scale*length_scale);

    P0 *= length_scale;
    P1 *= length_scale;
    P2 *= length_scale;


    // domain size information
    const int n_xyz []      = {2, 2, 2};
    PetscPrintf(mpi.comm(), "xyz_max for Box dimensions are set to be xmax = %g \t ymax = %g \t zmax = %g\n", xmax, ymax, zmaxx);


    int periodic[] = {0, 0, 0};
    conn = my_p4est_brick_new(n_xyz, xyz_min_, xyz_max_, &brick, periodic);

    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin, lmax); CHKERRXX(ierr);
    // initialize level set
    level_set.initialize();
    ierr = PetscPrintf(mpi.comm(), "Level-set initialized!\n"); CHKERRXX(ierr);
    // saving cell data
    level_set.save_cells();
    ierr = PetscPrintf(mpi.comm(), "Saved cell information to file!\n"); CHKERRXX(ierr);
    // create the forest
    p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

    // refine based on distance to a level-set
    splitting_criteria_cf_t sp(lmin, lmax, &level_set, 1.2);
    p4est->user_pointer = &sp;
    for(int i=0; i<lmax; i++)
    {
        my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
        my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }


    /* create the initial forest at time nm1 */
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

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
    Vec phi, X0, X1, Sm, vn, Pm;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, level_set, phi);

    /* set initial time step *//* find dx and dy smallest */
    /* p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
        p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
        double xmin = p4est->connectivity->vertices[3*vm + 0];
        double ymin = p4est->connectivity->vertices[3*vm + 1];
        double xmax = p4est->connectivity->vertices[3*vp + 0];
        double ymax = p4est->connectivity->vertices[3*vp + 1];
        */
    double dx = (xmax-xmin) / pow(2., (double) sp.max_lvl)/n_xyz[0];
    double dy = (ymax-ymin) / pow(2., (double) sp.max_lvl)/n_xyz[1];
#ifdef P4_TO_P8
    //PetscPrintf(p4est->mpicomm, "22: zmax=%g, zmin=%g\n", zmax, zmin);
    //double zmin = p4est->connectivity->vertices[3*vm + 2];
    //double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmaxx-zminn) / pow(2.,(double) sp.max_lvl)/n_xyz[2];
    //PetscPrintf(p4est->mpicomm, "3: xmin=%g, xmax=%g, ymin=%g, ymax=%g, zmin=%g, zmax=%g\n", xmin, xmax, ymin, ymax, zmin, zmax);
#endif
    /* perturb level set */
    my_p4est_level_set_t ls(&ngbd_n);
    //  ls.reinitialize_2nd_order(phi);
    ls.perturb_level_set_function(phi, EPS);

    Vec cell_number;
    ierr = VecCreateGhostNodes(p4est, nodes, &cell_number); CHKERRXX(ierr);
    //    Vec loc;
    //    ierr = VecGhostGetLocalForm(cell_number, &loc); CHKERRXX(ierr);
    //    ierr = VecSet(loc, -1.0); CHKERRXX(ierr);
    //    ierr = VecGhostRestoreLocalForm(cell_number, &loc); CHKERRXX(ierr);
    //    compute_cell_number(p4est, &ngbd_n, nodes, phi, cell_number);
    sample_cf_on_nodes(p4est, nodes, cell_numbering, cell_number);

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

    /* for diffusion */
    Vec M_list[number_ions];  // for now only include 2 ions
    double *M_p[number_ions];
    for(unsigned int i=0;i<number_ions;++i)
    {
        ierr = VecCreateGhostNodes(p4est, nodes, &M_list[i]); CHKERRXX(ierr);
        // initialize concentrations for ions here. initial_M can be different function for each ion.
        sample_cf_on_nodes(p4est, nodes, initial_M, M_list[i]);
    }

    Vec ElectroPhoresis_nm1[3], ElectroPhoresis[3];
    for(int dir=0; dir<3; ++dir)
    {
        ierr = VecCreateGhostNodes(p4est, nodes, &ElectroPhoresis_nm1[dir]); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &ElectroPhoresis[dir]); CHKERRXX(ierr);
    }


    Vec charge_rate;
    ierr = VecDuplicate(phi, &charge_rate); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &Pm); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, initial_pm, Pm);

    Vec domain;
    ierr = VecDuplicate(phi, &domain); CHKERRXX(ierr);
    Vec lc;
    ierr = VecGhostGetLocalForm(domain, &lc); CHKERRXX(ierr);
    ierr = VecSet(lc, -1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(domain, &lc); CHKERRXX(ierr);
    double total_mass = integrate_over_negative_domain(p4est, nodes, domain, M_list[0]);
    PetscPrintf(mpi.comm(), "Initial TOTAL mass of first ion is %g\n", total_mass);

    Vec vnm1, vnm2;
    ierr = VecDuplicate(phi, &vnm1); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &vnm2); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, initial_vnm1, vnm1);
    sample_cf_on_nodes(p4est, nodes, initial_vnm2, vnm2);

    Vec sol;
    Vec err;
    ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

    Vec top_electrode_phi, bottom_electrode_phi, intensity, Sm_thresholded[4], ones;
    ierr = VecDuplicate(phi, &top_electrode_phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &bottom_electrode_phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &intensity); CHKERRXX(ierr);
    for(unsigned int i=0;i<4;++i)
        ierr = VecDuplicate(phi, &Sm_thresholded[i]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);

    clock_t begin = clock();
    my_p4est_interpolation_nodes_t interp_n(&ngbd_n);


#ifdef P4_TO_P8
    dt = MIN(dx,dy,dz)/length_scale/dt_scale;
#else
    dt = MIN(dx,dy)/length_scale/dt_scale;
#endif
    double dxyz_max = MIN(dx,dy,dz);
    //dt= 1e-7;
    // dt = MIN(dx,dy,dz)/mu_e/1000.0;  // by matching boundary conditions with a 0.1 V/m resolution
    //dt=MIN(dt, 0.2/omega, MIN(dx,dy,dz)/dt_scale);  // the last term is to due to diffusion of concentrations


    save_VTK(p4est, ghost, nodes, &brick, phi, sol, -1, X0, X1, Sm, vn, err, M_list, Pm, charge_rate, cell_number);
    PetscPrintf(p4est->mpicomm, "Proceed with dt=%g [s], dx=%g [m], scaling %g and final time is %g [s].\n", dt, dz/length_scale,MIN(dx,dy,dz)/dt/length_scale, tf);

    Vec grad_nm1;
    VecDuplicate(phi, &grad_nm1);
    Vec grad_um, grad_up;
    ierr = VecCreateGhostNodes(p4est, nodes, &grad_um); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &grad_up); CHKERRXX(ierr);
    Vec lu;
    ierr = VecGhostGetLocalForm(grad_up, &lu); CHKERRXX(ierr);
    ierr = VecSet(lu, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(grad_up, &lu); CHKERRXX(ierr);


    while (tn<tf)
    {

        PetscPrintf(mpi.comm(), "####################################################\n");
        ierr = PetscPrintf(mpi.comm(), "Iteration %d, time %e\n", iteration, tn); CHKERRXX(ierr);

        //        if(iteration%DIFF_TO_EP_RATIO==0)
        //        {
        //            ierr = PetscPrintf(mpi.comm(), ">> solving advection and diffusion with time-step %g [s]... \n", DIFF_TO_EP_RATIO*dt); CHKERRXX(ierr);
        //            solve_transport(p4est, ghost, nodes, &ngbd_n, &ngbd_c, phi, sol, X0, X1, Pm, M_list, ls, dt*DIFF_TO_EP_RATIO, dt*DIFF_TO_EP_RATIO, charge_rate, vn, number_ions, grad_up, ElectroPhoresis_nm1, ElectroPhoresis);
        //        }
        ierr = PetscPrintf(mpi.comm(), ">> solving electroporation with time-step %g [s]... \n", dt); CHKERRXX(ierr);
        solve_electric_potential(p4est, nodes,&ngbd_n, &ngbd_c, phi, sol, dt,  X0,  X1, Sm, vn, ls, tn, vnm1, vnm2, grad_phi, charge_rate,grad_nm1, grad_up, grad_um);
        ierr = PetscPrintf(mpi.comm(), ">> solving elasticity... \n"); CHKERRXX(ierr);
        solve_electroelasticity(p4est, nodes,  ghost, ls, &ngbd_n, phi, dt,  lambda, dxyz_max);
        ierr = PetscPrintf(mpi.comm(), ">> Done with elasticity! \n"); CHKERRXX(ierr);
        double maxval;
        VecMax(M_list[0],NULL,&maxval);
        PetscPrintf(mpi.comm(), "maximum concentration in first ion is %g\n", maxval);
        // test mass conservation after diffusion step!
        double total_mass = integrate_over_negative_domain(p4est, nodes, domain, M_list[0]);
        PetscPrintf(mpi.comm(), "TOTAL mass of first ion is %g\n", total_mass);

        ierr = PetscPrintf(mpi.comm(), "done with solving! Doing measurements... \n"); CHKERRXX(ierr);
        double u_Npole_exact = 0;
        double u_Npole = 0;
        double xyz_np[3] = {0, R1*cos(PI/4), R1*sin(PI/4)};
        if(test==1 || test==2 || test==4 || test==5)
        {
            interp_n.set_input(vn, linear);
            interp_n.add_point(0, xyz_np);
            interp_n.interpolate(&u_Npole);
            interp_n.clear();
            u_Npole_exact = v_exact(xyz_np[0], xyz_np[1], xyz_np[2], tn+dt);
        }

        /* compute the error on the tree*/
        double *err_p, *sol_p,*TEphi_p, *BEphi_p, *intensity_p, *Sm_p, *Sm_thresholded_p[4], *ones_p;
        ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
        ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
        ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
        ierr = VecGetArray(top_electrode_phi, &TEphi_p); CHKERRXX(ierr);
        ierr = VecGetArray(bottom_electrode_phi, &BEphi_p); CHKERRXX(ierr);
        ierr = VecGetArray(intensity, &intensity_p); CHKERRXX(ierr);
        ierr = VecGetArray(Sm, &Sm_p); CHKERRXX(ierr);
        for(unsigned int i=0;i<4;++i)
            ierr = VecGetArray(Sm_thresholded[i], &Sm_thresholded_p[i]); CHKERRXX(ierr);
        ierr = VecGetArray(ones, &ones_p); CHKERRXX(ierr);

        err_nm1 = err_n;
        err_n = 0;

        for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        {
            double x = node_x_fr_n(n, p4est, nodes);
            double y = node_y_fr_n(n, p4est, nodes);
            double z = node_z_fr_n(n, p4est, nodes);

            // a level-set just to represent the electrode surfaces for integration purposes
            TEphi_p[n] = z - (zmaxx - EPS); // only on the top surface.
            BEphi_p[n] = z - (zminn + EPS); // only on the top surface.
            // this is on both surfaces
            /*if(z>0)
                    Ephi_p[n] = z - zmaxx + EPS;
                else
                    Ephi_p[n] = -(z - zminn - EPS); */

            if(is_interface(&ngbd_n,n,phi_p)<0 && (test==1 || test==2 || test ==4))
            {
                err_p[n] = ABS(sol_p[n] - u_exact(x,y,z,tn,phi_p[n]>0));
                err_n = MAX(err_n,err_p[n]);
            }
            ones_p[n] = 1;
            if(is_interface(&ngbd_n,n,phi_p)>0)
            {
                if(ABS(Sm_p[n])>100*SL)
                    Sm_thresholded_p[0][n] = 1;                                        // this is to mark cells that are permeabilized above a threshold, to compute their area
                else
                    Sm_thresholded_p[0][n] = 0;

                if(ABS(Sm_p[n])>1000*SL)
                    Sm_thresholded_p[1][n] = 1;
                else
                    Sm_thresholded_p[1][n] = 0;

                if(ABS(Sm_p[n])>10000*SL)
                    Sm_thresholded_p[2][n] = 1;
                else
                    Sm_thresholded_p[2][n] = 0;

                if(ABS(Sm_p[n])>100000*SL)
                    Sm_thresholded_p[3][n] = 1;
                else
                    Sm_thresholded_p[3][n] = 0;
            }
        }

        ierr = VecRestoreArray(top_electrode_phi, &TEphi_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(bottom_electrode_phi, &BEphi_p); CHKERRXX(ierr);
        for(unsigned int i=0;i<4;++i)
            ierr = VecRestoreArray(Sm_thresholded[i], &Sm_thresholded_p[i]); CHKERRXX(ierr);
        ierr = VecRestoreArray(Sm, &Sm_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(ones, &ones_p); CHKERRXX(ierr);

        MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
        ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
        //PetscPrintf(p4est->mpicomm, "tests=1,2: Iter %d maximum error on solution: %g ORDER: %g\n", iteration, err_n, log(err_nm1/err_n)/log(2));

        double des_err = 0;
        if (test==1 || test==2 || test==4|| test==5)
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
            double normal_drv_potential = qnnn.dz_backward_linear(sol_p);//dz_central(sol_p);
            intensity_p[n] = sigma_e*normal_drv_potential;
        }
        ierr = VecGhostUpdateBegin(intensity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        for(size_t i=0; i<ngbd_n.get_local_size(); ++i)
        {
            p4est_locidx_t n = ngbd_n.get_local_node(i);
            quad_neighbor_nodes_of_node_t qnnn = ngbd_n[n];
            double normal_drv_potential = qnnn.dz_backward_linear(sol_p); //dz_central(sol_p);
            intensity_p[n] = sigma_e*normal_drv_potential;
        }
        ierr = VecGhostUpdateEnd(intensity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
        ierr = VecRestoreArray(intensity, &intensity_p); CHKERRXX(ierr);


        // compute the important quantities: Pulse Intensity (A), impedance (Ohm), permitivity, conductivity
        double avg_Voltage = integrate_over_interface(p4est, nodes, top_electrode_phi, sol) - integrate_over_interface(p4est, nodes, bottom_electrode_phi, sol);
        avg_Voltage /= ((xmax-xmin)*(ymax-ymin));                                                       // V*m^2
        double PulseIntensity = integrate_over_interface(p4est, nodes, top_electrode_phi, intensity);   // current throughput (A): charge flux
        double impedance = avg_Voltage/PulseIntensity;                                               // Impedance (Ohm)

        double net_E = PulseIntensity/sigma_e;                                                       // E_net = epsilon_0*Q_top_plate = integral(E.dA) on top plate
        double applied_E = net_E/(xmax-xmin)/(ymax-ymin);
        double epsilon_r = 1;                                                                        // relative permitivity, real part
        if(fabs(avg_Voltage)>EPS)
        {
            epsilon_r = fabs(net_E*(zmaxx-zminn)/((xmax-xmin)*(ymax-ymin)*avg_Voltage));     //PAM: the relative permittivity (dispersive term! eps = "eps_r"*eps_0 - j*sigma/omega)
            if (epsilon_r<1)
                epsilon_r = 1;
        }
        double epsilon_Im = (zmaxx-zminn)/epsilon_0/impedance/omega/(xmax-xmin)/(ymax-ymin);  // imaginary part of epsilon
        double sigma_eff_Real, sigma_eff_Imaginary;
        double shape_factor = (xmax - xmin)*(ymax - ymin)/(zmaxx - zminn);

        sigma_eff_Real = 1/impedance/shape_factor;
        sigma_eff_Imaginary = -epsilon_r*epsilon_0*omega;

        double total_area_permeabilized[4]={0.,0.,0.,0.};
        for(unsigned int i=0;i<4;++i)
            total_area_permeabilized[i] = integrate_over_interface(p4est, nodes, phi, Sm_thresholded[i]);

        MPI_Allreduce(MPI_IN_PLACE, &total_area_permeabilized[0], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);

        double total_area = integrate_over_interface(p4est, nodes, phi, ones);
        double avg_Sm = integrate_over_interface(p4est, nodes, phi, Sm)/total_area;
        double avg_X0 = integrate_over_interface(p4est, nodes, phi, X0)/total_area;
        double avg_X1 = integrate_over_interface(p4est, nodes, phi, X1)/total_area;
        double avg_vn = integrate_over_interface(p4est, nodes, phi, vn)/total_area;



        if(save_shadowing){
            std::vector<double> X1_cells(nb_cells);
            Vec single_phi;
            VecDuplicate(phi,&single_phi);
            for(int cell_ID=0; cell_ID<nb_cells; ++cell_ID)
            {
                single_cell_phi.ID = cell_ID;
                sample_cf_on_nodes(p4est, nodes, single_cell_phi, single_phi);
                X1_cells[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, X1);
            }
            if(p4est->mpirank==0)
            {
                char out_path[1000];
                char *out_dir = NULL;
                out_dir = getenv("OUT_DIR");
                if(out_dir==NULL)
                {
                    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save topologies...\n"); CHKERRXX(ierr);
                } else {
                    sprintf(out_path, "%s/ShadowingEffect_%d.dat", out_dir, iteration);
                    FILE *f = fopen(out_path, "w");
                    fprintf(f, "%% Number of cells is: %u. These are surface averages of X2 in the model over each cell. (\epsilon_m) \n", nb_cells);
                    fprintf(f, "%% ID  |\t X_c\t  |\t Y_c\t  |\t Z_c\t |\t <X2> \n");
                    for(int n=0; n<nb_cells; ++n)
                        fprintf(f, "%d \t %g \t %g \t %g \t %g  \n", n, level_set.centers[n].x, level_set.centers[n].y, level_set.centers[n].z, X1_cells[n]);
                    fclose(f);
                }
            }
            VecDestroy(single_phi);
        }


        //Begin compute dipoles!
        if(save_dipoles){
            std::vector<double> dipole_x(nb_cells);
            std::vector<double> dipole_y(nb_cells);
            std::vector<double> dipole_z(nb_cells);

            std::vector<double> Quad_xx(nb_cells);
            std::vector<double> Quad_yy(nb_cells);
            std::vector<double> Quad_zz(nb_cells);
            std::vector<double> Quad_xy(nb_cells);
            std::vector<double> Quad_xz(nb_cells);
            std::vector<double> Quad_yz(nb_cells);

            Vec single_phi, dipole[P4EST_DIM], Quadrupole[6];
            VecDuplicate(phi,&single_phi);
            double *dipole_p[P4EST_DIM], *Quad_p[6];
            for(int j=0;j<P4EST_DIM;++j)
            {
                ierr = VecGetArray(grad_phi[j], &dphi_p[j]); CHKERRXX(ierr);
                VecDuplicate(phi, &dipole[j]);
            }
            for(int j=0;j<6;++j)
            {
                VecDuplicate(phi,&Quadrupole[j]);
                VecGetArray(Quadrupole[j], &Quad_p[j]);
            }
            double *vn_p;
            VecGetArray(vn, &vn_p);
            for(int j=0;j<P4EST_DIM;++j)
                VecGetArray(dipole[j], &dipole_p[j]);
            for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
            {
                double x = node_x_fr_n(n, p4est, nodes);
                double y = node_y_fr_n(n, p4est, nodes);
                double z = node_z_fr_n(n, p4est, nodes);

                dipole_p[0][n] = epsilon_0*dphi_p[0][n]*vn_p[n];
                dipole_p[1][n] = epsilon_0*dphi_p[1][n]*vn_p[n];
                dipole_p[2][n] = epsilon_0*dphi_p[2][n]*vn_p[n];
                // measure Quadropole moments: Q_ij /epsilon_m= integral ( epsilon0*Vn*(3*r_i*r_j-r*r*delta_ij) dS)
                Quad_p[0][n] = 2*epsilon_0*vn_p[n]*( 2* x*dphi_p[0][n] - y*dphi_p[1][n] - z*dphi_p[2][n]);
                Quad_p[1][n] = 2*epsilon_0*vn_p[n]*(-x*dphi_p[0][n] + 2*y*dphi_p[1][n] - z*dphi_p[2][n]);
                Quad_p[2][n] = 2*epsilon_0*vn_p[n]*(-x*dphi_p[0][n] - y*dphi_p[1][n] + 2*z*dphi_p[2][n]);

                Quad_p[3][n] = 3*epsilon_0*vn_p[n]*(x*dphi_p[1][n] + y*dphi_p[0][n]);
                Quad_p[4][n] = 3*epsilon_0*vn_p[n]*(x*dphi_p[2][n] + z*dphi_p[0][n]);
                Quad_p[5][n] = 3*epsilon_0*vn_p[n]*(y*dphi_p[2][n] + z*dphi_p[1][n]);
            }
            for(int j=0;j<P4EST_DIM;++j)
                VecRestoreArray(dipole[j], &dipole_p[j]);
            VecRestoreArray(vn, &vn_p);
            for(int j=0;j<P4EST_DIM;++j)
                ierr = VecRestoreArray(grad_phi[j], &dphi_p[j]); CHKERRXX(ierr);
            for(int j=0;j<6;++j)
                VecRestoreArray(Quadrupole[j], &Quad_p[j]);

            double dipole_x1, dipole_y1, dipole_z1, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz;
            if(test==2 || test==4 || test==5 || save_avg_dipole_only)
            {
                dipole_x1= integrate_over_interface(p4est, nodes, phi, dipole[0]);
                dipole_y1 = integrate_over_interface(p4est, nodes, phi, dipole[1]);
                dipole_z1 = integrate_over_interface(p4est, nodes, phi, dipole[2]);

                Qxx = integrate_over_interface(p4est, nodes, phi, Quadrupole[0]);
                Qyy = integrate_over_interface(p4est, nodes, phi, Quadrupole[1]);
                Qzz = integrate_over_interface(p4est, nodes, phi, Quadrupole[2]);
                Qxy = integrate_over_interface(p4est, nodes, phi, Quadrupole[3]);
                Qxz = integrate_over_interface(p4est, nodes, phi, Quadrupole[4]);
                Qyz = integrate_over_interface(p4est, nodes, phi, Quadrupole[5]);
            } else  {
                for(int cell_ID=0; cell_ID<nb_cells; ++cell_ID)
                {
                    single_cell_phi.ID = cell_ID;
                    sample_cf_on_nodes(p4est, nodes, single_cell_phi, single_phi);
                    dipole_x[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, dipole[0]);
                    dipole_y[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, dipole[1]);
                    dipole_z[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, dipole[2]);

                    Quad_xx[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, Quadrupole[0]);
                    Quad_yy[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, Quadrupole[1]);
                    Quad_zz[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, Quadrupole[2]);
                    Quad_xy[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, Quadrupole[3]);
                    Quad_xz[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, Quadrupole[4]);
                    Quad_yz[cell_ID] = integrate_over_interface(p4est, nodes, single_phi, Quadrupole[5]);
                }
            }
            if(p4est->mpirank==0)
            {
                char out_path[1000];
                char *out_dir = NULL;
                out_dir = getenv("OUT_DIR");
                if(out_dir==NULL)
                {
                    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save topologies...\n"); CHKERRXX(ierr);
                } else {
                    sprintf(out_path, "%s/Electricity_%d.dat", out_dir, iteration);
                    FILE *f = fopen(out_path, "w");
                    fprintf(f, "%% Number of cells is: %u. Dipoles and Quadrupoles below are stored per relative permittivity of membrane (\epsilon_m) \n", nb_cells);
                    fprintf(f, "%% ID  |\t X_c\t  |\t Y_c\t  |\t Z_c\t |\t dipole_x \t dipole_y \t dipole_z \t Quad_xx \t Quad_yy \t Quad_zz \t Quad_xy \t Quad_xz \t Quad_yz \n");
                    if(test==2 || test==4 || test==5 || save_avg_dipole_only)
                        fprintf(f, "%d \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \n", 0, 0.0, 0.0,0.0, dipole_x1, dipole_y1, dipole_z1, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz);
                    else
                    {
                        for(int n=0; n<nb_cells; ++n)
                            fprintf(f, "%d \t %g \t %g \t %g \t %g \t %g \t %g %g \t %g \t %g \t %g \t %g \t %g  \n", n, level_set.centers[n].x, level_set.centers[n].y, level_set.centers[n].z, dipole_x[n], dipole_y[n], dipole_z[n], Quad_xx[n], Quad_yy[n], Quad_zz[n], Quad_xy[n], Quad_xz[n], Quad_yz[n]);
                    }
                    fclose(f);
                }
            }
            VecDestroy(single_phi);
            for(unsigned int i = 0; i<P4EST_DIM;++i)
                VecDestroy(dipole[i]);
            for(unsigned int i = 0; i<6;++i)
                VecDestroy(Quadrupole[i]);
        }
        // End of measuring cell dipoles

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
                        fprintf(f, "Simulation Parameters: Omega [Hz] %g \t cell volume fraction %g \t box side length [m] %g \n", omega, density, xmax-xmin);
                        fprintf(f, "time [s]    | impedance [Ohm] | TMP [V] | Intensity | Error |TMP_exact [V] |Re(epsilon)| Im(epsilon) | V(t)[V] | E(t) [V/m] | Re(sigma) [S/m] |  Im(sigma)  | (>1e2*SL) area [m*m] | (>1e3*SL) area [m*m] | (>1e4*SL) area [m*m] | (>1e5*SL) area [m*m] | Total area [m*m] | avg. Sm | avg. vn| avg. X0 | avg. X1 |\n");
                        fprintf(f, "%g \t %g \t %g \t  %g \t %g \t %g \t\t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g\t %g \t %g \t %g \t %g\n", tn+dt, impedance, u_Npole, PulseIntensity, des_err, u_Npole_exact, epsilon_r, epsilon_Im, pulse(tn), applied_E, sigma_eff_Real, sigma_eff_Imaginary, total_area_permeabilized[0], total_area_permeabilized[1], total_area_permeabilized[2], total_area_permeabilized[3], total_area, avg_Sm, avg_vn, avg_X0, avg_X1);
                        fclose(f);
                    }
                    else{
                        FILE *f = fopen(out_path_Z, "a");
                        fprintf(f, "%g \t %g \t %g \t %g \t %g \t %g \t\t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g \t %g\t %g \t %g \t %g \t %g\n", tn+dt, impedance, u_Npole, PulseIntensity, des_err, u_Npole_exact, epsilon_r, epsilon_Im, pulse(tn), applied_E, sigma_eff_Real, sigma_eff_Imaginary,  total_area_permeabilized[0], total_area_permeabilized[1], total_area_permeabilized[2], total_area_permeabilized[3], total_area, avg_Sm, avg_vn, avg_X0, avg_X1);
                        fclose(f);
                    }

                }
            }
        }

        if(save_transport){
            char *out_dir = NULL;
            out_dir = getenv("OUT_DIR");
            if(out_dir==NULL)
            {
                ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
            }
            else
            {
                char out_path_Z[1000];
                sprintf(out_path_Z, "%s/transport.dat", out_dir);
                if(p4est->mpirank==0)
                {
                    if(iteration ==0){
                        FILE *f = fopen(out_path_Z, "w");
                        fprintf(f, "Simulation Parameters: Omega [Hz] %g \t cell volume fraction %g \t box side length [m] %g \n", omega, density, xmax-xmin);
                        fprintf(f, "time [s]    | total mass [mol^3]  | V(t)[V] | E(t) [V/m] |  \n");
                        fprintf(f, "%g \t %g \t %g \t %g \n", tn+dt, total_mass, pulse(tn), applied_E);
                        fclose(f);
                    }
                    else{
                        FILE *f = fopen(out_path_Z, "a");
                        fprintf(f, "%g \t %g \t %g \t %g\n", tn+dt, total_mass, pulse(tn), applied_E);
                        fclose(f);
                    }
                }
            }
        }
        if(save_vtk && iteration%save_every_n == 0)
        {
            save_VTK(p4est, ghost, nodes, &brick, phi, sol, iteration, X0, X1, Sm, vn, err, M_list, Pm, charge_rate, cell_number);
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
    ierr = VecDestroy(Pm); CHKERRXX(ierr);
    ierr = VecDestroy(X0); CHKERRXX(ierr);
    ierr = VecDestroy(X1); CHKERRXX(ierr);
    ierr = VecDestroy(charge_rate); CHKERRXX(ierr);

    for(unsigned int i=0;i<4;++i)
        ierr = VecDestroy(Sm_thresholded[i]); CHKERRXX(ierr);

    for(unsigned int i=0;i<number_ions;++i)
        ierr = VecDestroy(M_list[i]); CHKERRXX(ierr);

    for(int dir=0; dir<3; ++dir)
    {
        VecDestroy(ElectroPhoresis_nm1[dir]);
        VecDestroy(ElectroPhoresis[dir]);
    }
    VecDestroy(grad_nm1);
    VecDestroy(grad_up);
    VecDestroy(grad_um);

    VecDestroy(domain);
    VecDestroy(lc);

    // destroy the structures
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);

    my_p8est_brick_destroy(conn, &brick);

    w.stop(); w.read_duration();
}
