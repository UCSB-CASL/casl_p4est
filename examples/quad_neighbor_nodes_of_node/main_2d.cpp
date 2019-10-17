/*
 * Title: quad_neighbor_nodes_of_node
 * Description: testing new features added to the my_p4est_quad_neighbor_nodes_of_nodes_t class
 * added features: block-structured vectors, possibility to store elementary operators for derivatives
 * Author: Raphael Egan
 * Date Created: 09-20-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif

#include <iostream>
#include <iomanip>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

const static std::string main_description = "\
 In this example, we test and illustrate the calculation of first and second derivatives of node-\n\
 sampled fields. We calculate the gradient and second derivatives (along all cartesian directions)\n\
 of nfields scalar fields, on nsplits grids that are finer and finer. The maximum pointwise errors\n\
 are evaluated for all nodes and the orders of convergence are estimated and successively shown (if\n\
 nsplits > 1). \n\
 The code's performance is assessed with built-in timers to compare various methods for evaluating \n\
 the derivatives. The available methods are:\n\
 - method 0: calculating the first and second derivatives of the nfields scalar fields sequentially,\n\
            one after another; \n\
 - method 1: calculating the first and second derivatives of the nfields scalar fields simultaneously\n\
            (calculating geometry-related information, only once); \n\
 - method 2: calculating the first and second derivatives of the nfields scalar fields simultaneously,\n\
            using block-structured parallel vectors to optimize parallel communications (calculating \n\
            geometry-related information, only once). \n\
 The three different methods should produce the EXACT same results regarding the orders of convergence.\n\
 (This example contains and illustrates performance facts pertaining to the optimization of procedures \n\
 related to data transfer between successive grids in grid-update steps, with quadratic interpolation.) \n\
 Example of application of interest: when interpolating (several) node-sampled data fields from one\n\
 grid to another with quadratic interpolation, the second derivatives of all fields are required for\n\
 all the fields\n\
 Developer: Raphael Egan (raphaelegan@ucsb.edu), October 2019.\n";


#ifdef P4_TO_P8
class test_function : public CF_3 {
#else
class test_function : public CF_2 {
#endif
public:
#ifdef P4_TO_P8
  virtual double dx   (double x, double y, double z) const=0;
  virtual double dy   (double x, double y, double z) const=0;
  virtual double dz   (double x, double y, double z) const=0;
  virtual double ddxx (double x, double y, double z) const=0;
  virtual double ddyy (double x, double y, double z) const=0;
  virtual double ddzz (double x, double y, double z) const=0;
  double dx   (double xyz[P4EST_DIM]) const { return dx(xyz[0], xyz[1], xyz[2]); }
  double dy   (double xyz[P4EST_DIM]) const { return dy(xyz[0], xyz[1], xyz[2]); }
  double dz   (double xyz[P4EST_DIM]) const { return dz(xyz[0], xyz[1], xyz[2]); }
  double ddxx (double xyz[P4EST_DIM]) const { return ddxx(xyz[0], xyz[1], xyz[2]); }
  double ddyy (double xyz[P4EST_DIM]) const { return ddyy(xyz[0], xyz[1], xyz[2]); }
  double ddzz (double xyz[P4EST_DIM]) const { return ddzz(xyz[0], xyz[1], xyz[2]); }
#else
  virtual double dx   (double x, double y) const=0;
  virtual double dy   (double x, double y) const=0;
  virtual double ddxx (double x, double y) const=0;
  virtual double ddyy (double x, double y) const=0;
  double dx   (double xyz[P4EST_DIM]) const { return dx(xyz[0], xyz[1]); }
  double dy   (double xyz[P4EST_DIM]) const { return dy(xyz[0], xyz[1]); }
  double ddxx (double xyz[P4EST_DIM]) const { return ddxx(xyz[0], xyz[1]); }
  double ddyy (double xyz[P4EST_DIM]) const { return ddyy(xyz[0], xyz[1]); }
#endif
  virtual ~test_function() {}
};

#ifdef P4_TO_P8
struct circle : CF_3 {
  circle(double x0_, double y0_, double z0_, double r_): x0(x0_), y0(y0_), z0(z0_), r(r_) {}
#else
struct circle : CF_2 {
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
#else
  double operator()(double x, double y) const {
#endif
    return r - sqrt(SQR(x-x0) + SQR(y-y0)
                #ifdef P4_TO_P8
                    + SQR(z-z0)
                #endif
                    );
  }

private:
  double x0, y0;
#ifdef P4_TO_P8
  double z0;
#endif
  double r;
};

static double random_generator(const double &min=0.0, const double &max=1.0)
{
  return (min+(max-min)*((double) rand())/((double) RAND_MAX));
}

void sample_test_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const test_function *cf_array[], Vec f)
{
  double *f_p;
  PetscInt bs;
  PetscErrorCode ierr;
  ierr = VecGetBlockSize(f, &bs); CHKERRXX(ierr);
  ierr = VecGetArray(f, &f_p); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);

    for (PetscInt j = 0; j<bs; j++) {
      const test_function &cf = *cf_array[j];
#ifdef P4_TO_P8
      f_p[i*bs + j] = cf(xyz[0], xyz[1], xyz[2]);
#else
      f_p[i*bs + j] = cf(xyz[0], xyz[1]);
#endif
    }
  }
  ierr = VecRestoreArray(f, &f_p); CHKERRXX(ierr);
  return;
}

void create_initial_grid_ghost_and_nodes(const mpi_environment_t &mpi, p4est_connectivity_t* conn, const unsigned int &lmin, const unsigned int &lmax,
                                         p4est_t* &forest, p4est_ghost_t* &ghosts, p4est_nodes_t* &nodes)
{
  if(forest!=NULL)
    p4est_destroy(forest);
  if(nodes!=NULL)
    p4est_nodes_destroy(nodes);
  if(ghosts!=NULL)
    p4est_ghost_destroy(ghosts);
  // creation and refinement of the p4est structure is not the purpose of this illustrative example
  // create the forest
  forest = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
  // sphere of random center in [r, 2-r]^P4EST_DIM or random radius in [r/2, r]
  const double r = 0.3;
#ifdef P4_TO_P8
  circle circ(random_generator(r, 2.0-r), random_generator(r, 2.0-r), random_generator(r, 2.0-r), random_generator(r/2, r));
#else
  circle circ(random_generator(r, 2.0-r), random_generator(r, 2.0-r), random_generator(r/2, r));
#endif
  splitting_criteria_cf_t cf_data(lmin, lmax, &circ, 1);

  // attach the splitting criterion
  forest->user_pointer = &cf_data;
  for (unsigned int k = 0; k < lmax; ++k) {
    // refine the forest once more
    my_p4est_refine(forest, P4EST_FALSE, refine_levelset_cf, NULL);
    // partition the forest
    my_p4est_partition(forest, P4EST_TRUE, NULL);
  }
  // create ghost layer
  ghosts = my_p4est_ghost_new(forest, P4EST_CONNECT_FULL);

  // create node structure: ALWAYS use 'my_p4est_nodes_new', never use 'p4est_nodes_new'
  // this is critical to ensure consistency with the rest of my_p4est_... library
  nodes = my_p4est_nodes_new(forest, ghosts);
  return;
}

void refine_my_grid(p4est_t* &forest, p4est_ghost_t* &ghosts, p4est_nodes_t* &nodes)
{
  if(forest==NULL)
    throw std::invalid_argument("refine_my_grid: needs a valid p4est structure to start with");
  // refine every cell once more
  my_p4est_refine(forest, P4EST_FALSE, refine_every_cell, NULL);
  // partition the forest
  my_p4est_partition(forest, P4EST_TRUE, NULL);

  // create new ghost layer
  if(ghosts!=NULL)
    p4est_ghost_destroy(ghosts);
  ghosts = my_p4est_ghost_new(forest, P4EST_CONNECT_FULL);

  // create new node structure: ALWAYS use 'my_p4est_nodes_new', never use 'p4est_nodes_new'
  // this is critical to ensure consistency with the rest of my_p4est_... library
  if(nodes!=NULL)
    p4est_nodes_destroy(nodes);
  nodes = my_p4est_nodes_new(forest, ghosts);
  return;
}

PetscErrorCode destroy_vectors_if_needed(const unsigned int &nfields,
                                         Vec &field_block, Vec &grad_field_block, Vec &second_derivatives_field_block,
                                         Vec field_[], Vec dx_[], Vec dy_[],
                                         #ifdef P4_TO_P8
                                         Vec dz_[],
                                         #endif
                                         Vec ddxx_[], Vec ddyy_[]
                                         #ifdef P4_TO_P8
                                         , Vec ddzz_[]
                                         #endif
                                         )
{
  PetscErrorCode ierr;
  for (unsigned int k = 0; k < nfields; ++k) {
    if(field_[k]!=NULL){
      ierr = VecDestroy(field_[k]);                     CHKERRQ(ierr); field_[k] = NULL; }
    if(dx_[k]!=NULL){
      ierr = VecDestroy(dx_[k]);                        CHKERRQ(ierr); dx_[k] = NULL; }
    if(dy_[k]!=NULL){
      ierr = VecDestroy(dy_[k]);                        CHKERRQ(ierr); dy_[k] = NULL; }
#ifdef P4_TO_P8
    if(dz_[k]!=NULL){
      ierr = VecDestroy(dz_[k]);                        CHKERRQ(ierr); dz_[k] = NULL; }
#endif
    if(ddxx_[k]!=NULL){
      ierr = VecDestroy(ddxx_[k]);                      CHKERRQ(ierr); ddxx_[k] = NULL; }
    if(ddyy_[k]!=NULL){
      ierr = VecDestroy(ddyy_[k]);                      CHKERRQ(ierr); ddyy_[k] = NULL; }
#ifdef P4_TO_P8
    if(ddzz_[k]!=NULL){
      ierr = VecDestroy(ddzz_[k]);                      CHKERRQ(ierr); ddzz_[k] = NULL; }
#endif
  }
  if(field_block!=NULL){
    ierr = VecDestroy(field_block);                     CHKERRQ(ierr); field_block = NULL; }
  if(grad_field_block!=NULL){
    ierr = VecDestroy(grad_field_block);                CHKERRQ(ierr); grad_field_block = NULL; }
  if(second_derivatives_field_block!=NULL){
    ierr = VecDestroy(second_derivatives_field_block);  CHKERRQ(ierr); second_derivatives_field_block = NULL; }
  return ierr;
}

PetscErrorCode create_vectors_and_sample_functions_on_nodes(const unsigned int &method, const p4est_t *p4est, p4est_nodes_t *nodes, const test_function *cf_field[], const unsigned int &nfields,
                                                            Vec &field_block, Vec &grad_field_block, Vec &second_derivatives_field_block,
                                                            Vec field_[], Vec dx_[], Vec dy_[],
                                                            #ifdef P4_TO_P8
                                                            Vec dz_[],
                                                            #endif
                                                            Vec ddxx_[], Vec ddyy_[]
                                                            #ifdef P4_TO_P8
                                                            , Vec ddzz_[]
                                                            #endif
                                                            )
{
  PetscErrorCode ierr;
  destroy_vectors_if_needed(nfields, field_block, grad_field_block, second_derivatives_field_block,
                            field_, dx_, dy_,
                          #ifdef P4_TO_P8
                            dz_,
                          #endif
                            ddxx_, ddyy_
                          #ifdef P4_TO_P8
                            , ddzz_
                          #endif
                            );
  if(method==2)
  {
    ierr = VecCreateGhostNodesBlock(p4est, nodes, nfields,           &field_block);                    CHKERRQ(ierr);
    ierr = VecCreateGhostNodesBlock(p4est, nodes, nfields*P4EST_DIM, &grad_field_block);               CHKERRQ(ierr);
    ierr = VecCreateGhostNodesBlock(p4est, nodes, nfields*P4EST_DIM, &second_derivatives_field_block); CHKERRQ(ierr);
    for (unsigned int k = 0; k < nfields; ++k) {
      field_[k] = NULL;
      dx_[k]    = NULL; ddxx_[k] = NULL;
      dy_[k]    = NULL; ddyy_[k] = NULL;
#ifdef P4_TO_P8
      dz_[k]    = NULL; ddzz_[k] = NULL;
#endif
    }
    sample_test_cf_on_nodes(p4est, nodes, cf_field, field_block);
  }
  else
  {
    field_block = NULL;
    grad_field_block = NULL;
    second_derivatives_field_block = NULL;
    for (unsigned int k = 0; k < nfields; ++k) {
      ierr = VecCreateGhostNodes(p4est, nodes, &field_[k]); CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &dx_[k]);    CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &ddxx_[k]);  CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &dy_[k]);    CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &ddyy_[k]);  CHKERRQ(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est, nodes, &dz_[k]);    CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &ddzz_[k]);  CHKERRQ(ierr);
#endif
    }
    for (unsigned int k = 0; k < nfields; ++k)
      sample_cf_on_nodes(p4est, nodes, *cf_field[k], field_[k]);
  }
  return ierr;
}

void evaluate_max_error_on_nodes(const unsigned int &method,const p4est_t *p4est, p4est_nodes_t *nodes, const test_function *cf_field[], const unsigned int &nfields,
                                 Vec &grad_field_block, Vec &second_derivatives_field_block,
                                 Vec dx_[], Vec dy_[],
                                 #ifdef P4_TO_P8
                                 Vec dz_[],
                                 #endif
                                 Vec ddxx_[], Vec ddyy_[],
                                 #ifdef P4_TO_P8
                                 Vec ddzz_[],
                                 #endif
                                 double err_gradient[][P4EST_DIM], double err_second_derivatives[][P4EST_DIM])
{
  PetscErrorCode ierr;
  const double *grad_field_block_p, *second_derivatives_field_block_p;
  const double *dx_p[nfields], *dy_p[nfields], *ddxx_p[nfields], *ddyy_p[nfields];
#ifdef P4_TO_P8
  const double *dz_p[nfields], *ddzz_p[nfields];
#endif


  if(method==2){
    ierr = VecGetArrayRead(grad_field_block, &grad_field_block_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(second_derivatives_field_block, &second_derivatives_field_block_p); CHKERRXX(ierr);
  }
  else
    for (unsigned char k = 0; k < nfields; ++k) {
      ierr = VecGetArrayRead(dx_[k],    &dx_p[k]);    CHKERRXX(ierr);
      ierr = VecGetArrayRead(dy_[k],    &dy_p[k]);    CHKERRXX(ierr);
      ierr = VecGetArrayRead(ddxx_[k],  &ddxx_p[k]);  CHKERRXX(ierr);
      ierr = VecGetArrayRead(ddyy_[k],  &ddyy_p[k]);  CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArrayRead(dz_[k],    &dz_p[k]);    CHKERRXX(ierr);
      ierr = VecGetArrayRead(ddzz_[k],  &ddzz_p[k]);  CHKERRXX(ierr);
#endif
    }
  for (unsigned int k = 0; k < nfields; ++k)
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      err_gradient[k][der] = 0.0;
      err_second_derivatives[k][der] = 0.0;
    }

  for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i)
  {
    double xyz [P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);
    if(method!=2)
    {
      for (unsigned int k = 0; k < nfields; ++k) {
        err_gradient[k][0] = MAX(err_gradient[k][0], fabs(dx_p[k][i]-cf_field[k]->dx(xyz)));
        err_gradient[k][1] = MAX(err_gradient[k][1], fabs(dy_p[k][i]-cf_field[k]->dy(xyz)));
#ifdef P4_TO_P8
        err_gradient[k][2] = MAX(err_gradient[k][2], fabs(dz_p[k][i]-cf_field[k]->dz(xyz)));
#endif
        err_second_derivatives[k][0] = MAX(err_second_derivatives[k][0], fabs(ddxx_p[k][i]-cf_field[k]->ddxx(xyz)));
        err_second_derivatives[k][1] = MAX(err_second_derivatives[k][1], fabs(ddyy_p[k][i]-cf_field[k]->ddyy(xyz)));
#ifdef P4_TO_P8
        err_second_derivatives[k][2] = MAX(err_second_derivatives[k][2], fabs(ddzz_p[k][i]-cf_field[k]->ddzz(xyz)));
#endif
      }
    }
    else
    {
      for (unsigned int k = 0; k < nfields; ++k) {
        err_gradient[k][0] = MAX(err_gradient[k][0], fabs(grad_field_block_p[i*nfields*P4EST_DIM+k*P4EST_DIM+0]-cf_field[k]->dx(xyz)));
        err_gradient[k][1] = MAX(err_gradient[k][1], fabs(grad_field_block_p[i*nfields*P4EST_DIM+k*P4EST_DIM+1]-cf_field[k]->dy(xyz)));
#ifdef P4_TO_P8
        err_gradient[k][2] = MAX(err_gradient[k][2], fabs(grad_field_block_p[i*nfields*P4EST_DIM+k*P4EST_DIM+2]-cf_field[k]->dz(xyz)));
#endif
        err_second_derivatives[k][0] = MAX(err_second_derivatives[k][0], fabs(second_derivatives_field_block_p[i*nfields*P4EST_DIM+k*P4EST_DIM+0]-cf_field[k]->ddxx(xyz)));
        err_second_derivatives[k][1] = MAX(err_second_derivatives[k][1], fabs(second_derivatives_field_block_p[i*nfields*P4EST_DIM+k*P4EST_DIM+1]-cf_field[k]->ddyy(xyz)));
#ifdef P4_TO_P8
        err_second_derivatives[k][2] = MAX(err_second_derivatives[k][2], fabs(second_derivatives_field_block_p[i*nfields*P4EST_DIM+k*P4EST_DIM+2]-cf_field[k]->ddzz(xyz)));
#endif
      }
    }
  }
  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, err_gradient,           nfields*P4EST_DIM,  MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, err_second_derivatives, nfields*P4EST_DIM,  MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  if(method==2){
    ierr = VecRestoreArrayRead(grad_field_block, &grad_field_block_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(second_derivatives_field_block, &second_derivatives_field_block_p); CHKERRXX(ierr);
  }
  else
    for (unsigned char k = 0; k < nfields; ++k) {
      ierr = VecRestoreArrayRead(dx_[k],    &dx_p[k]);    CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(dy_[k],    &dy_p[k]);    CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(ddxx_[k],  &ddxx_p[k]);  CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(ddyy_[k],  &ddyy_p[k]);  CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(dz_[k],    &dz_p[k]);    CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(ddzz_[k],  &ddzz_p[k]);  CHKERRXX(ierr);
#endif
    }
}

#ifdef P4_TO_P8
class uex : public test_function {
private:
  double a, b, c;
public:
  uex(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
#else
class uex : public test_function {
private:
  double a, b;
public:
  uex(double a_, double b_) : a(a_), b(b_) {}
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
    return 1.0/(cos(a*x*x + b*y*y + c*z*z)+1.5);
#else
  double operator()(double x, double y) const {
    return 1.0/(cos(a*x*x + b*y*y)+1.5);
#endif
  }

#ifdef P4_TO_P8
  double dx(double x, double y, double z) const {
    return 2.0*a*x*sin(a*x*x + b*y*y + c*z*z)/(SQR(cos(a*x*x + b*y*y + c*z*z)+1.5));
#else
  double dx(double x, double y) const {
    return 2.0*a*x*sin(a*x*x + b*y*y)/(SQR(cos(a*x*x + b*y*y)+1.5));
#endif
  }

#ifdef P4_TO_P8
  double dy(double x, double y, double z) const {
    return 2.0*b*y*sin(a*x*x + b*y*y + c*z*z)/(SQR(cos(a*x*x + b*y*y + c*z*z)+1.5));
#else
  double dy(double x, double y) const {
    return 2.0*b*y*sin(a*x*x + b*y*y)/(SQR(cos(a*x*x + b*y*y)+1.5));
#endif
  }

#ifdef P4_TO_P8
  double dz(double x, double y, double z) const {
    return 2.0*c*z*sin(a*x*x + b*y*y + c*z*z)/(SQR(cos(a*x*x + b*y*y + c*z*z)+1.5));
  }
#endif

#ifdef P4_TO_P8
  double ddxx(double x, double y, double z) const {
    return (4.0*a*a*x*x*cos(a*x*x + b*y*y + c*z*z)/SQR(1.5 + cos(a*x*x + b*y*y + c*z*z)) + 8.0*a*a*x*x*SQR(sin(a*x*x + b*y*y + c*z*z))/pow((1.5 + cos(a*x*x + b*y*y + c*z*z)), 3.0) + 2.0*a*sin(a*x*x + b*y*y + c*z*z)/SQR(1.5+cos(a*x*x + b*y*y + c*z*z)));
#else
  double ddxx(double x, double y) const {
    return (4.0*a*a*x*x*cos(a*x*x + b*y*y)/SQR(1.5 + cos(a*x*x + b*y*y)) + 8.0*a*a*x*x*SQR(sin(a*x*x + b*y*y))/pow((1.5 + cos(a*x*x + b*y*y)), 3.0) + 2.0*a*sin(a*x*x + b*y*y)/SQR(1.5+cos(a*x*x + b*y*y)));
#endif
  }

#ifdef P4_TO_P8
  double ddyy(double x, double y, double z) const {
    return (4.0*b*b*y*y*cos(a*x*x + b*y*y + c*z*z)/SQR(1.5 + cos(a*x*x + b*y*y + c*z*z)) + 8.0*b*b*y*y*SQR(sin(a*x*x + b*y*y + c*z*z))/pow((1.5 + cos(a*x*x + b*y*y + c*z*z)), 3.0) + 2.0*b*sin(a*x*x + b*y*y + c*z*z)/SQR(1.5+cos(a*x*x + b*y*y + c*z*z)));
#else
  double ddyy(double x, double y) const {
    return (4.0*b*b*y*y*cos(a*x*x + b*y*y)/SQR(1.5 + cos(a*x*x + b*y*y)) + 8.0*b*b*y*y*SQR(sin(a*x*x + b*y*y))/pow((1.5 + cos(a*x*x + b*y*y)), 3.0) + 2.0*b*sin(a*x*x + b*y*y)/SQR(1.5+cos(a*x*x + b*y*y)));
#endif
  }

#ifdef P4_TO_P8
  double ddzz(double x, double y, double z) const {
    return (4.0*c*c*z*z*cos(a*x*x + b*y*y + c*z*z)/SQR(1.5 + cos(a*x*x + b*y*y + c*z*z)) + 8.0*c*c*z*z*SQR(sin(a*x*x + b*y*y + c*z*z))/pow((1.5 + cos(a*x*x + b*y*y + c*z*z)), 3.0) + 2.0*c*sin(a*x*x + b*y*y + c*z*z)/SQR(1.5+cos(a*x*x + b*y*y + c*z*z)));
  }
#endif
};

#ifdef P4_TO_P8
class vex : public test_function {
private:
  double a, b, c;
public:
  vex(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
#else
class vex : public test_function {
private:
  double a, b;
public:
  vex(double a_, double b_) : a(a_), b(b_) {}
#endif

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
    return cos(a*x+y)*sin(b*y-x)*atan(c*z);
#else
  double operator()(double x, double y) const {
    return cos(a*x+y)*sin(b*y-x);
#endif
  }

#ifdef P4_TO_P8
  double dx(double x, double y, double z) const {
    return (-a*sin(a*x+y)*sin(b*y-x) - cos(a*x+y)*cos(b*y-x))*atan(c*z);
#else
  double dx(double x, double y) const {
    return (-a*sin(a*x+y)*sin(b*y-x) - cos(a*x+y)*cos(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double ddxx(double x, double y, double z) const {
    return (-a*a*cos(a*x+y)*sin(b*y-x) + 2.0*a*sin(a*x+y)*cos(b*y-x) - cos(a*x+y)*sin(b*y-x))*atan(c*z);
#else
  double ddxx(double x, double y) const {
    return (-a*a*cos(a*x+y)*sin(b*y-x) + 2.0*a*sin(a*x+y)*cos(b*y-x) - cos(a*x+y)*sin(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double dy(double x, double y, double z) const {
    return (-sin(a*x+y)*sin(b*y-x) + b*cos(a*x+y)*cos(b*y-x))*atan(c*z);
#else
  double dy(double x, double y) const {
    return (-sin(a*x+y)*sin(b*y-x) + b*cos(a*x+y)*cos(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double ddyy(double x, double y, double z) const {
    return (-cos(a*x+y)*sin(b*y-x) - 2.0*b*sin(a*x+y)*cos(b*y-x) - b*b*cos(a*x+y)*sin(b*y-x))*atan(c*z);
#else
  double ddyy(double x, double y) const {
    return (-cos(a*x+y)*sin(b*y-x) - 2.0*b*sin(a*x+y)*cos(b*y-x) - b*b*cos(a*x+y)*sin(b*y-x));
#endif
  }

#ifdef P4_TO_P8
  double dz(double x, double y, double z) const {
    return  c*cos(a*x+y)*sin(b*y-x)/(1+SQR(c*z));
  }

  double ddzz(double x, double y, double z) const {
    return  -2.0*c*c*c*z*cos(a*x+y)*sin(b*y-x)/(SQR(1+SQR(c*z)));
  }
#endif
};

#ifdef P4_TO_P8
class wex : public test_function {
private:
  double a, b, c;
public:
  wex(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
  double operator()(double x, double y, double z) const {
    return log(a*x*x+1)*atan(y)*sin(c*z+b*y);
  }

  double dx(double x, double y, double z) const {
    return atan(y)*sin(c*z+b*y)*2.0*a*x/(1.0+a*x*x);
  }

  double ddxx(double x, double y, double z) const {
    return atan(y)*sin(c*z+b*y)*(2.0*a*(1-a*x*x))/SQR(1.0+a*x*x);
  }

  double dy(double x, double y, double z) const {
    return log(a*x*x+1)*(sin(c*z+b*y)/(1 + y*y) + atan(y)*b*cos(c*z+b*y));
  }
  double ddyy(double x, double y, double z) const {
    return log(a*x*x+1)*(-2.0*y*sin(c*z+b*y)/(SQR(1 + y*y)) + 2.0*b*cos(c*z+b*y)/(1 + y*y) - atan(y)*b*b*sin(c*z+b*y));
  }

  double dz(double x, double y, double z) const {
    return log(a*x*x+1)*atan(y)*c*cos(c*z+b*y);
  }

  double ddzz(double x, double y, double z) const {
    return -log(a*x*x+1)*atan(y)*c*c*sin(c*z+b*y);
  }
};
#endif

int main (int argc, char* argv[]){
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  p4est_t            *p4est = NULL;
  p4est_ghost_t      *ghost = NULL;
  p4est_nodes_t      *nodes = NULL;
  PetscErrorCode      ierr;

  cmdParser cmd;
  cmd.add_option("seed",        "seed for random number generator (default is 279, don't ask why :-p)");
  cmd.add_option("ntrees",      "number of trees per dimensions (default is 2)");
  cmd.add_option("lmin",        "min level of the trees for the first grid to consider (default is 4)");
  cmd.add_option("lmax",        "max level of the trees for the first grid to consider (default is 6)");
  cmd.add_option("nsplits",     "number of grid splittings for accuracy check\n\
           (default is 3, accuracy is checked only if >1)");
  cmd.add_option("method",      "default is 0, available values are\n\
            0::calculating the derivatives, one field after another;\n\
            1::calculating the derivatives, all fields at once;\n\
            2::storing data by block, calculating all at once as well.");
  cmd.add_option("timing_off",  "disables timing if present");
  cmd.add_option("fields",      "number of fields to calculate first and second derivatives of\n\
           (default is number of dimensions, i.e., P4EST_DIM)");
  cmd.add_option("vtk_folder",  "exportation directory for vtk files if vtk exportation is activated\n\
           (default is the directory where the program is run from, i.e., './')");
  cmd.add_option("vtk",         "exports the (final) grid and hierarchy in vtk format, if present.");

  if(cmd.parse(argc, argv, main_description))
    return 0;

  // read user's input parameters
  const unsigned int method = cmd.get<unsigned int>("method", 0);
  unsigned int seed         = cmd.get<unsigned int>("seed", 279);
  bool timing_off           = cmd.contains("timing_off");
  unsigned int lmin         = cmd.get<unsigned int>("lmin", 4);
  unsigned int lmax         = cmd.get<unsigned int>("lmax", 6);
  unsigned int nsplits      = cmd.get<unsigned int>("nsplits", 3);
  int ntrees                = cmd.get<unsigned int>("ntrees", 2);
  unsigned int nfields      = cmd.get<unsigned int>("fields", P4EST_DIM);

  // check for valid inputs
  if(method > 2)
    throw std::invalid_argument("main: unknown desired method");
  if(nsplits == 0 || ntrees <= 0 || nfields == 0)
    throw std::invalid_argument("main: requires a strictly positive number for 'nsplits', 'ntrees' and 'fields'");
  if(lmax < lmin)
    throw std::invalid_argument("main: requires lmax >= lmin");
  /* [A note about throwing exceptions in a parallel framework]
   * Using exceptions (invalid_argument or others) is a fairly standard
   * procedure to handle non-standard code execution in object-oriented
   * programming. The idea is that the exception is "thrown" by the
   * current function back to the calling environment, i.e., up the next
   * level in the call stack, where it is either "caught" (via a try{}
   * catch{} block) or it will keep ascending up the call stack.
   * If the exception is never caught, std::terminate will finally be
   * invoked, causing the program to exit abnormally (usually printing
   * the exception's "what()" message though, which gives the user some
   * insight about what went wrong).
   *
   * While such a behavior for uncaught exception might be acceptable
   * in case of a serial application, things may get nastier in parallel.
   * Indeed if only one process throws an exception that is not caught,
   * it will cause that specific MPI process to stop, for sure, but it
   * will not notify the other processes in the MPI_COMM_WORLD. Therefore,
   * this may cause an MPI deadlock (other MPI processes desperately
   * waiting for the one interrupted MPI process to reach out to them,
   * indefinitely), without information about the problem being necessa-
   * rily shown to the user, making debugging hard if not impossible.
   *
   * In the above input checks, every single MPI process checks for the
   * exact same statements so that either all or none of them will throw,
   * which prevents such a deadlock situation.
   * Otherwise, if the desired COLLECTIVE behavior is to put an end to
   * ALL MPI processes when finding a critical error on a single MPI
   * process (but possibly not on others), it is advised to call some
   * collective termination like
   * MPI_Abort, PetscAbortErrorHandler or sc_abort, for instance.
   * [end of note] */

  // initialize the random number generator
  srand(seed);

  // create a timer
  parStopWatch timer;

  // create the test function(s), with random parameters
  const test_function *cf_field[nfields];
  for (unsigned int k = 0; k < nfields; ++k) {
#ifdef P4_TO_P8
    if(k%P4EST_DIM==0)
      cf_field[k] = new uex(random_generator(0.0, 1.0), random_generator(0.0, 1.0), random_generator(0.0, 1.0));
    else if (k%P4EST_DIM==1)
      cf_field[k] = new vex(random_generator(0.0, 2.0), random_generator(0.0, 0.7), random_generator(0.0, 0.5));
    else
      cf_field[k] = new wex(random_generator(0.0, 1.0), random_generator(-1.0, 1.0), random_generator(-1.0, 1.0));
#else
    if(k%P4EST_DIM==0)
      cf_field[k] = new uex(random_generator(0.0, 1.0), random_generator(0.0, 1.0));
    else
      cf_field[k] = new vex(random_generator(0.0, 2.0), random_generator(0.0, 0.7));
#endif
  }

  // Create the connectivity object, define the domain
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t my_brick, *brick = &my_brick;
  int n_xyz [] = {ntrees, ntrees, ntrees};
  double xyz_min [] = {0, 0, 0};
  double xyz_max [] = {2, 2, 2};
  int periodic []   = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);

  // Now create the forest
  create_initial_grid_ghost_and_nodes(mpi, connectivity, lmin, lmax, p4est, ghost, nodes);

  // Declare and initialize the relevant node-sampled parallel Petsc vectors to NULL (for
  // internal consistency in the function destroying them if needed, hereafter)
  // --------------------------------------------------------------------------------------------
  // For method 0 or 1: we use vectors of block size 1 for all the fields of interest, so we need
  // nfields Petsc vectors for the node-sampled function values, for every partial derivative
  // along a Cartesian direction, and also for every second partial derivative along a Cartesian
  // direction
  Vec field_[nfields];
  Vec dx_[nfields], dy_[nfields], ddxx_[nfields], ddyy_[nfields];
#ifdef P4_TO_P8
  Vec dz_[nfields], ddzz_[nfields];
#endif
  for (unsigned int k = 0; k < nfields; ++k) {
    field_[k] = NULL;
    dx_[k] = NULL; ddxx_[k] = NULL;
    dy_[k] = NULL; ddyy_[k] = NULL;
#ifdef P4_TO_P8
    dz_[k] = NULL; ddzz_[k] = NULL;
#endif
  }
  // For method 2: we use a parallel Petsc vector of block size nfields for storing the fields
  // of interest, and we use two parallel Petsc vectors of block size nfields*P4EST_DIM for the
  // first and second derivatives.
  Vec field_block, grad_field_block, second_derivatives_field_block;
  field_block = NULL; grad_field_block = NULL; second_derivatives_field_block = NULL;
  // The value of the kth field (0<= k < nfields) at node of local index i is
  //    field_block_p[nfields*i+k]
  // the value of the first or second derivative along cartesian direction dir, for the kth field
  // at node i are
  //    grad_field_block_p[nfields*P4EST_DIM*i+P4EST_DIM*k+dir]
  //    second_derivatives_field_block_p[nfields*P4EST_DIM*i+P4EST_DIM*k+dir]
  // where field_block_p, grad_field_block_p, second_derivatives_field_block_p are the local arrays
  // of the corresponding parallel vectors

  // The relevant tasks are here below:
  // We track the total time spent on node neighbor initialization and on the actual calculation of derivatives
  double time_spent_on_ngbd_initialization  = 0.0;
  double time_spent_on_derivative           = 0.0;
  // relevant error measures on former grid (--> '_m1') and on the current grid
  double err_gradient_step_m1[nfields][P4EST_DIM];
  double err_gradient_step[nfields][P4EST_DIM];
  double err_second_derivatives_step_m1[nfields][P4EST_DIM];
  double err_second_derivatives_step[nfields][P4EST_DIM];
  for (unsigned int ss = 0; ss < nsplits; ++ss) {
    if(ss > 0)
      refine_my_grid(p4est, ghost, nodes);

    // In order to calculate derivatives, we need to know about the neighbor nodes
    // of every node in the computational grid
    // ---------------------------------------------------------------------------
    // First we need a my_p4est_hierarchy (building brick to easily find points and
    // cells of interest in the domain).
    // --> check out my_p4est_hierarchy.h for more information
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    /* Once the hierarchy is known, the node neighbors can be found and/or constructed reliably, since the neighboring
     * quadrants can be found with the help of the hierarchy, as explained and illustrated above.
     */
    my_p4est_node_neighbors_t ngbd_n(&hierarchy, nodes);

#ifdef P4_TO_P8
    ierr = create_vectors_and_sample_functions_on_nodes(method, p4est, nodes, cf_field, nfields,
                                                        field_block, grad_field_block, second_derivatives_field_block,
                                                        field_, dx_, dy_, dz_, ddxx_, ddyy_, ddzz_); CHKERRXX(ierr);
#else
    ierr = create_vectors_and_sample_functions_on_nodes(method, p4est, nodes, cf_field, nfields,
                                                        field_block, grad_field_block, second_derivatives_field_block,
                                                        field_, dx_, dy_,      ddxx_, ddyy_); CHKERRXX(ierr);
#endif
    if(!timing_off)
      timer.start();

    ngbd_n.init_neighbors();
    if(!timing_off){
      timer.stop();
      time_spent_on_ngbd_initialization += timer.get_duration();
      timer.start();
    }
    switch (method) {
    case 0:
    {
      /* For method 0, we calculate the derivatives of the node-sampled scalar fields,
       * one after another. Every time the following functions are called, the purely
       * grid-related (i.e. not field-dependent) calculations are executed.
       * When nfields is large, these redundant grid-related calculations can become
       * a significant amount of the total execution time for this specific task
       * (possibly much more than 50%), especially on large 3D grids.
       */
      for (unsigned int k = 0; k < nfields; ++k) {
#ifdef P4_TO_P8
        ngbd_n.first_derivatives_central(field_[k], dx_[k], dy_[k], dz_[k]);
        ngbd_n.second_derivatives_central(field_[k], ddxx_[k], ddyy_[k], ddzz_[k]);
#else
        ngbd_n.first_derivatives_central(field_[k], dx_[k], dy_[k]);
        ngbd_n.second_derivatives_central(field_[k], ddxx_[k], ddyy_[k]);
#endif
      }
      break;
    }
    case 1:
    {
      /* For method 1, we calculate the derivatives of all the node-sampled scalar fields
       * at once. The purely grid-related (i.e. not field-dependent) calculations are
       * executed only once per node.
       * When nfields is large, one can save execution time by avoiding the redundant
       * grid-related calculations, especially on large 3D grids.
       */
#ifdef P4_TO_P8
      ngbd_n.first_derivatives_central(field_, dx_, dy_, dz_, nfields);
      ngbd_n.second_derivatives_central(field_, ddxx_, ddyy_, ddzz_, nfields);
#else
      ngbd_n.first_derivatives_central(field_, dx_, dy_, nfields);
      ngbd_n.second_derivatives_central(field_, ddxx_, ddyy_, nfields);
#endif
      break;
    }
    case 2:
    {
      /* For method 2, we calculate the derivatives of all the node-sampled scalar fields
       * at once, as in method 1. We also use block-structured parallel vectors instead of
       * a multitude of standard parallel vectors. This approach has the advantage of gathering
       * all relevant, similar data together and to encapsulate them. In particular, one needs
       * less communication calls to synchronize ghost data, and those data are packed in a
       * rather optimal way, leveraging better performance when using several computer nodes.
       */
      ngbd_n.first_derivatives_central(field_block, grad_field_block, nfields);
      ngbd_n.second_derivatives_central(field_block, second_derivatives_field_block, nfields);
      break;
    }
    default:
      throw std::invalid_argument("main: unknown desired method");
      break;
    }
    if(!timing_off){
      timer.stop();
      time_spent_on_derivative += timer.get_duration();
    }

    if(ss > 0)
      for (unsigned int k = 0; k < nfields; ++k)
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
        {
          err_gradient_step_m1[k][der]            = err_gradient_step[k][der];
          err_second_derivatives_step_m1[k][der]  = err_second_derivatives_step[k][der];
        }

#ifdef P4_TO_P8
    evaluate_max_error_on_nodes(method, p4est, nodes, cf_field, nfields,
                                grad_field_block, second_derivatives_field_block,
                                dx_, dy_, dz_, ddxx_, ddyy_, ddzz_,
                                err_gradient_step, err_second_derivatives_step);
    ierr = destroy_vectors_if_needed(nfields, field_block, grad_field_block, second_derivatives_field_block,
                                     field_, dx_, dy_, dz_, ddxx_, ddyy_, ddzz_); CHKERRXX(ierr);
#else
    evaluate_max_error_on_nodes(method, p4est, nodes, cf_field, nfields,
                                grad_field_block, second_derivatives_field_block,
                                dx_, dy_,      ddxx_, ddyy_,
                                err_gradient_step, err_second_derivatives_step);
    ierr = destroy_vectors_if_needed(nfields, field_block, grad_field_block, second_derivatives_field_block,
                                     field_, dx_, dy_,      ddxx_, ddyy_); CHKERRXX(ierr);
#endif

    if (mpi.rank() == 0 && ss>0)
    {
      if(ss ==1)
        std::cout << " ------------------ ORDERS OF CONVERGENCE AFTER " << ss << " split  ------------------- " << std::endl;
      else
        std::cout << " ------------------ ORDERS OF CONVERGENCE AFTER " << ss << " splits  ------------------ " << std::endl;
      std::cout << " --- orders of convergence when comparing results from " << lmin+ss-1 << "/" << lmax+ss-1 << " and " << lmin+ss << "/" << lmax+ss << " grids ---" << std::endl;

      std::cout << "\t - for the calculation of gradients: " << std::endl;
#ifdef P4_TO_P8
      std::cout << "\t\t\talong x\t\talong y\t\talong z" << std::endl;
#else
      std::cout << "\t\t\talong x\t\talong y" << std::endl;
#endif
      for (unsigned int comp = 0; comp < nfields; ++comp) {
        std::cout << "\tfield #" << comp << ": \t";
        std::cout << std::fixed;
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          std::cout << std::setprecision(4) << log(err_gradient_step_m1[comp][der]/err_gradient_step[comp][der])/log(2.0) << "\t\t";
        std::cout << std::endl;
      }
      std::cout << std::endl;
      std::cout << "\t - for the calculation of second derivatives: " << std::endl;
#ifdef P4_TO_P8
      std::cout << "\t\t\talong x\t\talong y\t\talong z" << std::endl;
#else
      std::cout << "\t\t\talong x\t\talong y" << std::endl;
#endif
      for (unsigned int comp = 0; comp < nfields; ++comp) {
        std::cout << "\tfield #" << comp << ": \t";
        std::cout << std::fixed;
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          std::cout << std::setprecision(4) << log(err_second_derivatives_step_m1[comp][der]/err_second_derivatives_step[comp][der])/log(2.0) << "\t\t";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    if(ss==nsplits-1 && cmd.contains("vtk"))
    {
      const string vtk_folder = cmd.get<string>("vtk_folder", "./");
      string grid_filename = vtk_folder+"p4est_grid";
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             0, 0, grid_filename.c_str());
      string hierarchy_filename = vtk_folder+"hierarchy";
      hierarchy.write_vtk(hierarchy_filename.c_str());
    }
  }

  if(!timing_off && p4est->mpirank == 0)
  {
    std::cout << "Time spent on initializing neighbors: " << time_spent_on_ngbd_initialization << " seconds." << std::endl;
    std::cout << "Time spent on calculating derivatives: " << time_spent_on_derivative << " seconds." << std::endl;
  }

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);

  my_p4est_brick_destroy(connectivity, brick);
  for (unsigned int k = 0; k < nfields; ++k)
    delete cf_field[k];

  return 0;
}

