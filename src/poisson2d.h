#ifndef POISSON2D_H
#define POISSON2D_H


#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#endif

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_utils.h>
#endif


#include<src/cube3.h>


#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

#ifdef P4_TO_P8
static struct:CF_3{
                  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
                  double operator()(double x, double y, double z) const {
                  return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
                  }
                  double  x0, y0, z0, r;
} circle ;

static class: public CF_3
{
                         public:
                         double operator()(double x, double y, double z) const {
                         return  cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
                         }
                         } u_ex;

static class: public CF_3
{
                         public:
                         double operator()(double x, double y, double z) const {
                         return  12*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
                         }
                         } f_ex;

static struct:WallBC3D{
                  BoundaryConditionType operator()(double x, double y, double z) const {
                  (void)x;
                  (void)y;
                  (void)z;
                  return NEUMANN;
                  }
                  } bc_wall_neumann_type;

static struct:WallBC3D{
                  BoundaryConditionType operator()(double x, double y, double z) const {
                  (void)x;
                  (void)y;
                  (void)z;
                  return DIRICHLET;
                  }
                  } bc_wall_dirichlet_type;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  (void) x;
                  (void) y;
                  (void) z;
                  return 0;
                  }
                  } bc_wall_neumann_value;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  return u_ex(x,y,z);
                  }
                  } bc_wall_dirichlet_value;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  return u_ex(x,y,z);
                  }
                  } bc_interface_dirichlet_value;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  double r  = sqrt(SQR(x-circle.x0) + SQR(y-circle.y0) + SQR(z-circle.z0));
                  double nx = (x-circle.x0) / r;
                  double ny = (y-circle.y0) / r;
                  double nz = (z-circle.z0) / r;
                  double norm = sqrt( nx*nx + ny*ny + nz*nz);
                  nx /= norm; ny /= norm; nz /= norm;
                  return ( 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z) * nx +
                           2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z) * ny +
                           2*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z) * nz );
                  }
                  } bc_interface_neumann_value;
#else
static struct:CF_2{
                  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
                  double operator()(double x, double y) const {
                  return r - sqrt(SQR(x-x0) + SQR(y-y0));
                  }
                  double  x0, y0, r;
} circle;

static class: public CF_2
{
                         public:
                         double operator()(double x, double y) const {
                         return  cos(2*M_PI*x)*cos(2*M_PI*y);
                         }
                         } u_ex;

static class: public CF_2
{
                         public:
                         double operator()(double x, double y) const {
                         return  8*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y);
                         }
                         } f_ex;

static struct:WallBC2D{
                  BoundaryConditionType operator()(double x, double y) const {
                  (void)x;
                  (void)y;
                  return NEUMANN;
                  }
                  } bc_wall_neumann_type;

static struct:WallBC2D{
                  BoundaryConditionType operator()(double x, double y) const {
                  (void)x;
                  (void)y;
                  return DIRICHLET;
                  }
                  } bc_wall_dirichlet_type;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  (void) x;
                  (void) y;
                  return 0;
                  }
                  } bc_wall_neumann_value;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  return u_ex(x,y);
                  }
                  } bc_wall_dirichlet_value;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  return u_ex(x,y);
                  }
                  } bc_interface_dirichlet_value;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  double r = sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
                  double nx = (x-circle.x0) / r;
                  double ny = (y-circle.y0) / r;
                  double norm = sqrt( nx*nx + ny*ny);
                  nx /= norm; ny /= norm;
                  return 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y) * nx + 2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y) * ny;
                  }
                  } bc_interface_neumann_value;
#endif


class poisson2d
{
private:

    //petsc fields

    Mat A;
    Mat ADense;
    KSP myKsp; PC pc;
    PetscScalar    neg_one      = -1.0,one = 1.0,value[3];
    PetscBool      nonzeroguess = PETSC_FALSE;
    Vec xx,b,u;
 PetscInt       n_petsc = 10,col[3],its;
PetscReal      norm,tol=1.e-14;  /* norm of solution error */

    // p4est fields

    mpi_context_t mpi_context, *mpi;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;
    cmdParser cmd;
    p4est_ghost_t* ghost;
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;

#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif

#ifdef P4_TO_P8
    CF_3 *bc_wall_value, *bc_interface_value;
    WallBC3D *wall_bc;
#else
    CF_2 *bc_wall_value, *bc_interface_value;
    WallBC2D *wall_bc;
#endif

    BoundaryConditionType bc_wall_type, bc_interface_type;
    int nb_splits, min_level, max_level;

    double *sol_p, *phi_p, *uex_p,*rhs_p, *bc_p;
    double *sol_p_cell; Vec phi, rhs, uex, sol;
    double *err;
    PoissonSolverNodeBase *solver;



    PetscInt rstart,rend,nlocal;

public:
    Session *mpi_session;
    poisson2d()
    {
        std::cout<<"Simple Constructor"<<std::endl;
    }
    poisson2d(int argc, char* argv[]);

    void poisson2d_initialyze_petsc(int argc, char* argv[]);
    void poisson2d_finalyze_petsc();




    std::string IO_path="/Users/gaddielouaknin/p4estLocal/";
    inline std::string convert2FullPath(std::string file_name)
    {
        std::stringstream oss;
        std::string mystr;
        oss <<this->IO_path <<file_name;
        mystr=oss.str();
        return mystr;
    }



    void createDensePetscMatrix(int argc, char *argv[]);
    void printDensePetscMatrix();
    int createDenseLinearAlgebraProbelm();
    int createDenseLinearAlgebraProbelmMPI();
    int destructDenseLinearAlgebraProblem();
    void printDenseLinearAlgebraProblem();
    void solveDenseLinearAlgebraProblem();
    void printDenseLinearAlgebraSolution();
    void createSparsePetscMatrix(int argc, char *argv[]);
    void printSparsePetscMatrix();
    void printForestNodes2TextFile();
    void printForestOctants2TextFile();
    void printForestQNodes2TextFile();
    void printGhostNodes();
    void printGhostCells();




};

#endif // POISSON2D_H
