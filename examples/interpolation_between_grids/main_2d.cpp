/*
 * Title: interpolation_between_grids
 * Description: testing new features added to the my_p4est_interpolation_nodes_t class
 * added features: block-structured vectors, possibility to interpolate several fields at once
 * Author: Raphael Egan
 * Date Created/revised: 10-16-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_macros.h>
#endif

#include <iostream>
#include <iomanip>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

const static std::string main_description = "\
 In this example, we test and illustrate the calculation of first and second derivatives of node-\n\
// sampled fields. We calculate the gradient and second derivatives (along all cartesian directions)\n\
// of nfields scalar fields, on nsplits grids that are finer and finer. The maximum pointwise errors\n\
// are evaluated for all inner nodes (i.e. excluding wall nodes) and the orders of convergence are\n\
// estimated and successively shown (if nsplits > 1). \n\
// The code's performance is assessed with built-in timers to compare various methods for evaluating \n\
// the derivatives. The available methods are:\n\
// - method 0: calculating the first and second derivatives of the nfields scalar fields sequentially,\n\
//            one after another; \n\
// - method 1: calculating the first and second derivatives of the nfields scalar fields simultaneously\n\
//            (calculating geometry-related information, only once); \n\
// - method 2: calculating the first and second derivatives of the nfields scalar fields simultaneously,\n\
//            using block-structured parallel vectors to optimize parallel communications (calculating \n\
//            geometry-related information, only once). \n\
// The three different methods should produce the EXACT same results regarding the orders of convergence.\n\
// (This example contains and illustrates performance facts pertaining to the optimization of procedures \n\
// related to data transfer between successive grids in grid-update procedures, with quadratic interpola-\n\
// tion.) \n\
// Example of application of interest: when interpolating (several) node-sampled data fields from one\n\
// grid to another with quadratic interpolation, the second derivatives of all fields are required for\n\
// all the fields\n\
 Developer: Raphael Egan (raphaelegan@ucsb.edu), October 2019.\n";

#ifdef P4_TO_P8
struct circle : CF_3 {
  circle(double x0_, double y0_, double z0_, double r_): x0(x0_), y0(y0_), z0(z0_), r(r_) {}
#else
struct circle : CF_2 {
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
#endif

#ifdef P4_TO_P8
  void update(double x0_, double y0_, double z0_, double r_)
#else
  void update(double x0_, double y0_, double r_)
#endif
  {
    x0 = x0_;
    y0 = y0_;
#ifdef P4_TO_P8
    z0 = z0_;
#endif
    r  = r_;
  }

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

#ifdef P4_TO_P8
class my_test_function : public CF_3 {
#else
class my_test_function : public CF_2 {
#endif
private:
  double aa, bb;
#ifdef P4_TO_P8
  double cc;
#endif

public:
#ifdef P4_TO_P8
  my_test_function(const double &aa_,const double &bb_, const double &cc_) : aa(aa_), bb(bb_), cc(cc_) {}
  double operator ()(double x, double y, double z) const
#else
  my_test_function(const double &aa_,const double &bb_) : aa(aa_), bb(bb_) {}
  double operator ()(double x, double y) const
#endif
  {
#ifdef P4_TO_P8
    return tanh(log(1 + fabs(aa) + SQR(x-bb*y)) + 1.0/(1.0+exp(-SQR(bb*y))) - cos(cc*(z-aa*x) + SQR(x-bb*y)))*(atan(aa*x-cc*z)/(1 + SQR(sin(aa*x-bb*y))));
#else
    return tanh(log(1 + fabs(aa) + SQR(x-bb*y)) + 1.0/(1.0+exp(-SQR(bb*y))))*(1.0/(1 + SQR(sin(aa*x-bb*y))));
#endif
  }
#ifdef P4_TO_P8
  double operator ()(double *xyz) const { return this->operator ()(xyz[0], xyz[1], xyz[2]); }
#else
  double operator ()(double *xyz) const { return this->operator ()(xyz[0], xyz[1]); }
#endif
};

void create_grid_ghost_and_nodes(const mpi_environment_t &mpi, p4est_connectivity_t* conn, const unsigned int &lmin, const unsigned int &lmax,
                                 circle &my_circle,
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
  splitting_criteria_cf_t cf_data(lmin, lmax, &my_circle, 1);

  // attach the splitting criterion
  forest->user_pointer = &cf_data;
  for (unsigned int k = 0; k < lmax; ++k) {
    // refine the forest once more
    my_p4est_refine(forest, P4EST_FALSE, refine_levelset_cf, NULL);
    // partition the forest
    my_p4est_partition(forest, P4EST_FALSE, NULL);
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
                                         Vec &field_block, Vec &second_derivatives_field_block,
                                         Vec field_[], Vec ddxx_[], Vec ddyy_[],
                                         #ifdef P4_TO_P8
                                         Vec ddzz_[],
                                         #endif
                                         Vec *results_ = NULL, Vec *results_block = NULL)
{
  PetscErrorCode ierr;
  for (unsigned int k = 0; k < nfields; ++k) {
    if(field_[k]!=NULL){
      ierr = VecDestroy(field_[k]);                     CHKERRQ(ierr); field_[k] = NULL; }
    if(ddxx_[k]!=NULL){
      ierr = VecDestroy(ddxx_[k]);                      CHKERRQ(ierr); ddxx_[k] = NULL; }
    if(ddyy_[k]!=NULL){
      ierr = VecDestroy(ddyy_[k]);                      CHKERRQ(ierr); ddyy_[k] = NULL; }
#ifdef P4_TO_P8
    if(ddzz_[k]!=NULL){
      ierr = VecDestroy(ddzz_[k]);                      CHKERRQ(ierr); ddzz_[k] = NULL; }
#endif
    if(results_!= NULL && results_[k]!= NULL){
      ierr = VecDestroy(results_[k]);                   CHKERRQ(ierr); results_[k] = NULL; }
  }
  if(field_block!=NULL){
    ierr = VecDestroy(field_block);                     CHKERRQ(ierr); field_block = NULL; }
  if(results_block!= NULL && *results_block!=NULL){
    ierr = VecDestroy(*results_block);                  CHKERRQ(ierr); *results_block = NULL; }
  if(second_derivatives_field_block!=NULL){
    ierr = VecDestroy(second_derivatives_field_block);  CHKERRQ(ierr); second_derivatives_field_block = NULL; }
  return ierr;
}

PetscErrorCode create_vectors_and_sample_functions_on_nodes(const unsigned int &method,
                                                            p4est_t *p4est, p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd,
                                                            const my_test_function *cf_field[], const unsigned int &nfields,
                                                            Vec &field_block, Vec &second_derivatives_field_block,
                                                            Vec field_[], Vec ddxx_[], Vec ddyy_[]
                                                            #ifdef P4_TO_P8
                                                            , Vec ddzz_[]
                                                            #endif
                                                            )
{
  PetscErrorCode ierr;
  destroy_vectors_if_needed(nfields, field_block, second_derivatives_field_block,
                            field_, ddxx_, ddyy_
                          #ifdef P4_TO_P8
                            , ddzz_
                          #endif
                            );

  if(method==2)
  {
    ierr = VecCreateGhostNodesBlock(p4est, nodes, nfields,           &field_block);                    CHKERRQ(ierr);
    ierr = VecCreateGhostNodesBlock(p4est, nodes, nfields*P4EST_DIM, &second_derivatives_field_block); CHKERRQ(ierr);
    double *field_block_p;
    ierr = VecGetArray(field_block, &field_block_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i) {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(i, p4est, nodes, xyz);
      for (unsigned int j = 0; j<nfields; j++)
#ifdef P4_TO_P8
        field_block_p[i*nfields + j] = (*cf_field[j])(xyz);
#else
        field_block_p[i*nfields + j] = (*cf_field[j])(xyz[0], xyz[1]);
#endif
    }
    ierr = VecRestoreArray(field_block, &field_block_p); CHKERRXX(ierr);
    ngbd->second_derivatives_central(field_block, second_derivatives_field_block, nfields);
  }
  else
  {
    for (unsigned int k = 0; k < nfields; ++k) {
      ierr = VecCreateGhostNodes(p4est, nodes, &field_[k]); CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &ddxx_[k]);  CHKERRQ(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &ddyy_[k]);  CHKERRQ(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est, nodes, &ddzz_[k]);  CHKERRQ(ierr);
#endif
    }
    for (unsigned int k = 0; k < nfields; ++k)
      sample_cf_on_nodes(p4est, nodes, *cf_field[k], field_[k]);
#ifdef P4_TO_P8
    ngbd->second_derivatives_central(field_, ddxx_, ddyy_, ddzz_, nfields);
#else
    ngbd->second_derivatives_central(field_, ddxx_, ddyy_, nfields);
#endif
  }
  return ierr;
}

void evaluate_max_error_on_destination_grid(const unsigned int &method,
                                            const p4est_t *p4est, p4est_nodes_t *nodes, const my_test_function *cf_field[], const unsigned int &nfields,
                                            Vec results_[], Vec results_block,
                                            double max_err[])
{
  PetscErrorCode ierr;
  const double *results_block_p;
  const double *results_p[nfields];

  if(method==2){
    ierr = VecGetArrayRead(results_block, &results_block_p); CHKERRXX(ierr);
  }
  else
    for (unsigned char k = 0; k < nfields; ++k) {
      ierr = VecGetArrayRead(results_[k], &results_p[k]); CHKERRXX(ierr);
    }
  for (unsigned int k = 0; k < nfields; ++k)
    max_err[k] = 0.0;

  for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i)
  {
    double xyz [P4EST_DIM];
    node_xyz_fr_n(i, p4est, nodes, xyz);
    if(method!=2)
      for (unsigned int k = 0; k < nfields; ++k)
        max_err[k] = MAX(max_err[k], fabs((*cf_field[k])(xyz) - results_p[k][i]));
    else
      for (unsigned int k = 0; k < nfields; ++k)
        max_err[k] = MAX(max_err[k], fabs((*cf_field[k])(xyz) - results_block_p[i*nfields + k]));
  }
  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, max_err, nfields, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  if(method==2){
    ierr = VecRestoreArrayRead(results_block, &results_block_p); CHKERRXX(ierr);
  }
  else
    for (unsigned char k = 0; k < nfields; ++k) {
      ierr = VecRestoreArrayRead(results_[k], &results_p[k]); CHKERRXX(ierr);
    }
}

int main (int argc, char* argv[]){

  mpi_environment_t mpi;
  mpi.init(argc, argv);

  p4est_t            *p4est_from = NULL, *p4est_to = NULL;
  p4est_ghost_t      *ghost_from = NULL, *ghost_to = NULL;
  p4est_nodes_t      *nodes_from = NULL, *nodes_to = NULL;
  PetscErrorCode      ierr;

  cmdParser cmd;
  cmd.add_option("seed",      "seed for random number generator (default is 279, totally arbitrarily chosen :-p)");
  cmd.add_option("ntrees",    "number of trees per Cartesian dimensions (default is 2)");
  cmd.add_option("lmin_from", "min level of the trees in the origin grid (i.e. before interpolation), defaut is 4");
  cmd.add_option("lmax_from", "max level of the trees in the origin grid (i.e. before interpolation), defaut is 6");
  cmd.add_option("lmin_to",   "min level of the trees in the destination grid (i.e. after interpolation), defaut is 5");
  cmd.add_option("lmax_to",   "max level of the trees in the destination grid (i.e. after interpolation), defaut is 8");
  cmd.add_option("nsplits",   "number of grid splittings for the origin grid (for accuracy check)\n\           (default is 3, accuracy is checked only if > 1)");
  cmd.add_option("method",      "default is 0, available values are\n\
            0::store the fields separately and interpolate one after another;\n\
            1::store the fields separately and interpolate all at once;\n\
            2::store the fields contiguously in block-structured vectors and interpolate all at once.");
  cmd.add_option("timing_off",  "disables timing if present");
  cmd.add_option("fields",      "number of node-sampled fields to interpolate between grids\n\           (default is number of dimensions, i.e., P4EST_DIM)");

  if(cmd.parse(argc, argv, main_description))
    return 0;

  // read user's input parameters
  const unsigned int method   = cmd.get<unsigned int>("method", 0);
  unsigned int seed           = cmd.get<unsigned int>("seed", 279);
  bool timing_off             = cmd.contains("timing_off");
  unsigned int lmin_from      = cmd.get<unsigned int>("lmin_from", 4);
  unsigned int lmax_from      = cmd.get<unsigned int>("lmax_from", 6);
  unsigned int lmin_to        = cmd.get<unsigned int>("lmin_to", 5);
  unsigned int lmax_to        = cmd.get<unsigned int>("lmax_to", 8);
  unsigned int nsplits        = cmd.get<unsigned int>("nsplits", 3);
  int ntrees                  = cmd.get<unsigned int>("ntrees", 2);
  const unsigned int nfields  = cmd.get<unsigned int>("fields", P4EST_DIM);

  // check for valid inputs
  if(method > 2)
    throw std::invalid_argument("main: unknown desired method");
  if(nsplits == 0 || ntrees <= 0 || nfields == 0)
    throw std::invalid_argument("main: requires a strictly positive number for 'nsplits', 'ntrees' and 'fields'");
  if(lmax_from < lmin_from)
    throw std::invalid_argument("main: requires lmax_from >= lmin_from");
  if(lmax_to < lmin_to)
    throw std::invalid_argument("main: requires lmax_to >= lmin_to");

  // initialize the random number generator
  srand(seed);

  // create a timer
  parStopWatch timer;

  // Create the connectivity object, define the domain
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t my_brick, *brick = &my_brick;
  int n_xyz [] = {ntrees, ntrees, ntrees};
  double xyz_min [] = {0, 0, 0};
  double xyz_max [] = {2, 2, 2};
  int periodic []   = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);

  // define a first circle for the original grid
  const double rmax_from = 0.3;
  const double radius_from = random_generator(0.5*rmax_from, rmax_from);
#ifdef P4_TO_P8
  circle circ(random_generator(rmax_from, 2.0-rmax_from), random_generator(rmax_from, 2.0-rmax_from), random_generator(rmax_from, 2.0-rmax_from), radius_from);
#else
  circle circ(random_generator(rmax_from, 2.0-rmax_from), random_generator(rmax_from, 2.0-rmax_from), radius_from);
#endif

  // Now create the (first) origin forest
  create_grid_ghost_and_nodes(mpi, connectivity, lmin_from, lmax_from, circ, p4est_from, ghost_from, nodes_from);

  // move the circle for the destination grid
  const double rmax_to = 0.3;
  const double radius_to = random_generator(0.5*rmax_to, rmax_to);
#ifdef P4_TO_P8
  circ.update(random_generator(rmax_to, 2.0-rmax_to), random_generator(rmax_to, 2.0-rmax_to), random_generator(rmax_to, 2.0-rmax_to), radius_to);
#else
  circ.update(random_generator(rmax_to, 2.0-rmax_to), random_generator(rmax_to, 2.0-rmax_to), radius_to);
#endif
  // Now create the destination forest
  create_grid_ghost_and_nodes(mpi, connectivity, lmin_to, lmax_to, circ, p4est_to, ghost_to, nodes_to);

  const my_test_function *cf_field[nfields];
  for (unsigned int k = 0; k < nfields; ++k)
#ifdef P4_TO_P8
    cf_field[k] = new my_test_function(random_generator(0.1, 0.9), random_generator(0.1, 0.9), random_generator(0.1, 0.9));
#else
    cf_field[k] = new my_test_function(random_generator(0.1, 0.9), random_generator(0.1, 0.9));
#endif

  Vec field_[nfields];
  Vec field_xx[nfields], field_yy[nfields];
#ifdef P4_TO_P8
  Vec field_zz[nfields];
#endif
  for (unsigned int k = 0; k < nfields; ++k) {
    field_[k] = NULL; field_xx[k] = NULL; field_yy[k] = NULL;
#ifdef P4_TO_P8
    field_zz[k] = NULL;
#endif
  }
  Vec field_block = NULL, field_block_xxyyzz = NULL;

  // The relevant tasks are here below:
  // We track the total time spent on interpolation
  double time_spent_on_interpolation = 0.0;
  // relevant error measures on former grid (--> '_m1') and on the current grid
  double max_err_ssm1[nfields], max_err_ss[nfields];

  for (unsigned int ss = 0; ss < nsplits; ++ss) {
    if(ss > 0)
      refine_my_grid(p4est_from, ghost_from, nodes_from);

    // In order to construct an interpolator tool, we need to be able to locate tha grid
    // cells that own any point of interest, hence we need a hierarchy
    my_p4est_hierarchy_t hierarchy_from(p4est_from, ghost_from, brick);
    // Along with the hierarchy, the interpolator object also needs to be able to access
    // node neighbors. Indeed, if the second derivatives of the fields to interpolate are not
    // provided by the user but if they are required anyways for the interpolation procedure
    // of interest, the interpolator object needs to be able to calculate them on-the-fly
    // hence, it needs to know about the node_neighbors
    // let's create and initialize it
    my_p4est_node_neighbors_t ngbd_from(&hierarchy_from, nodes_from);
    ngbd_from.init_neighbors();

    ierr = create_vectors_and_sample_functions_on_nodes(method, p4est_from, nodes_from, &ngbd_from, cf_field, nfields,
                                                        field_block, field_block_xxyyzz,
                                                        field_, field_xx, field_yy
                                                    #ifdef P4_TO_P8
                                                        , field_zz
                                                    #endif
                                                        ); CHKERRXX(ierr);

    // now let's create the node interpolator tool
    my_p4est_interpolation_nodes_t node_interpolator(&ngbd_from);
    // let's initialize results
    Vec results_[nfields], results_block;
    results_block = NULL;
    for (unsigned int k = 0; k < nfields; ++k)
      results_[k] = NULL;
    if(!timing_off)
      timer.start();
    switch (method) {
    case 0:
      for (unsigned int k = 0; k < nfields; ++k) {
#ifdef P4_TO_P8
        node_interpolator.set_input(field_[k], field_xx[k], field_yy[k], field_zz[k], quadratic);
#else
        node_interpolator.set_input(field_[k], field_xx[k], field_yy[k], quadratic);
#endif
        // add the points of the destination grid to the input buffer
        if(k == 0)
          for (p4est_locidx_t i=0; i<nodes_to->indep_nodes.elem_count; ++i)
          {
            double xyz_node[P4EST_DIM];
            node_xyz_fr_n(i, p4est_to, nodes_to, xyz_node);
            node_interpolator.add_point(i, xyz_node);
          }
        ierr = VecCreateGhostNodes(p4est_to, nodes_to, &results_[k]); CHKERRXX(ierr);
        node_interpolator.interpolate(results_[k]);
      }
      break;
    default:
      throw std::invalid_argument("main: unknown desired method");
      break;
    }
    if(!timing_off){
      timer.stop();
      time_spent_on_interpolation += timer.get_duration();
    }

    if(ss > 0)
      for (unsigned int k = 0; k < nfields; ++k)
        max_err_ssm1[k] = max_err_ss[k];

    evaluate_max_error_on_destination_grid(method, p4est_to, nodes_to, cf_field, nfields,
                                           results_, results_block, max_err_ss);
    ierr = destroy_vectors_if_needed(nfields, field_block, field_block_xxyyzz,
                                     field_, field_xx, field_yy,
                                 #ifdef P4_TO_P8
                                     field_zz,
                                 #endif
                                     results_, &results_block); CHKERRXX(ierr);


    if (mpi.rank() == 0 && ss>0)
    {
      if(ss ==1)
        std::cout << " ------------------ ORDERS OF CONVERGENCE AFTER " << ss << " split  ------------------- " << std::endl;
      else
        std::cout << " ------------------ ORDERS OF CONVERGENCE AFTER " << ss << " splits  ------------------ " << std::endl;
      std::cout << " --- orders of convergence when comparing results from " << lmin_from+ss-1 << "/" << lmax_from+ss-1 << " and " << lmin_from+ss << "/" << lmax_from+ss << " grids ---" << std::endl;
      std::cout << " for field\t";
      for (unsigned int k = 0; k < nfields; ++k)
        std::cout << "#" << k << "\t";
      std::cout << std::endl;
      std::cout << "\t\t";
      for (unsigned int k = 0; k < nfields; ++k)
        std::cout << std::setprecision(4) << log(max_err_ssm1[k]/max_err_ss[k])/log(2.0) << "\t";
      std::cout << std::endl;
    }
  }

  if(!timing_off && p4est_from->mpirank == 0)
    std::cout << "Time spent on calculating interpolations : " << time_spent_on_interpolation << " seconds." << std::endl;

  // destroy the p4ests, the ghosts, the nodes, the connectivity and the brick
  p4est_nodes_destroy (nodes_from); p4est_nodes_destroy(nodes_to);
  p4est_ghost_destroy(ghost_from);  p4est_ghost_destroy(ghost_to);
  p4est_destroy (p4est_from);       p4est_destroy (p4est_to);

  my_p4est_brick_destroy(connectivity, brick);

  return 0;
}

