// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolation.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolation.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

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
  double x0, y0, r;
#ifdef P4_TO_P8
  double z0;
#endif
};

#ifdef P4_TO_P8
static struct:CF_3{
#else
static struct:CF_2{
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
    return 1.0/(cos(x*x + y*y + z*z)+1.5);
#else
  double operator()(double x, double y) const {
    return 1.0/(cos(x*x + y*y)+1.5);
#endif
  }
} uex;

#ifdef P4_TO_P8
static struct:CF_3{
#else
static struct:CF_2{
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const {
    return cos(x)*sin(y)*atan(z);
#else
  double operator()(double x, double y) const {
    return cos(x)*sin(y);
#endif
  }
} vex;

#ifdef P4_TO_P8
static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return cos(x)*sin(y)*atan(z);

  }
} wex;
#endif
#ifdef P4_TO_P8
const CF_3 *vector_components_cf[P4EST_DIM] = {&uex, &vex, &wex};
#else
const CF_2 *vector_components_cf[P4EST_DIM] = {&uex, &vex};
#endif

int main (int argc, char* argv[]){

  try{
    mpi_environment_t mpi;
    mpi.init(argc, argv);

    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the original tree (default is 2)");
    cmd.add_option("lmax", "max level of the original tree (default is 9)");
    cmd.add_option("lmin_final", "min level of the final tree (default is lmin+1)");
    cmd.add_option("lmax_final", "max level of the final tree (default is lmax-2)");
    cmd.add_option("vtk_off", "disable vtk exportation if present");
    cmd.add_option("timing_off", "disable timing if present");
    cmd.add_option("interpolation_method", "0: linear, 1: quadratic, 2: quadratic_non_oscillatory (default is linear)");
    cmd.parse(argc, argv);

    bool vtk_off    = cmd.contains("vtk_off");
    bool timing_off = cmd.contains("timing_off");

    int interpolation_method_idx = cmd.get<int>("interpolation_method", 0);
    interpolation_method method = (interpolation_method_idx == 2) ? quadratic_non_oscillatory : ((interpolation_method_idx ==1) ? quadratic : linear);

#ifdef P4_TO_P8
    circle circ(1, 1, 1, .3);
#else
    circle circ(1, 1, .3);
#endif
    int lmin_original = cmd.get<int>("lmin", 2);
    int lmax_original = cmd.get<int>("lmax", 9);
    splitting_criteria_cf_t cf_data(lmin_original, lmax_original, &circ, 1);

    parStopWatch w1, w2;
    if(!timing_off)
      w1.start("total time");

    // Create the connectivity object
    if(!timing_off)
      w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
    int n_xyz [] = {2, 2, 2};
    double xyz_min [] = {0, 0, 0};
    double xyz_max [] = {2, 2, 2};
    int periodic []   = {0, 0, 0};

    connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, brick, periodic);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    // Now create the forest
    if(!timing_off)
      w2.start("p4est generation");
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    // Now refine the tree
    if(!timing_off)
      w2.start("refine");
    p4est->user_pointer = (void*)(&cf_data);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    // Finally re-partition
    if(!timing_off)
      w2.start("partition");
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    // Create the ghost structure
    if(!timing_off)
      w2.start("ghost");
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    // generate the node data structure
    if(!timing_off)
      w2.start("creating node structure");
    nodes = my_p4est_nodes_new(p4est, ghost);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    if(!timing_off)
      w2.start("computing phi and the vector field");
    Vec phi, vector_field;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &vector_field); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, circ, phi);
    sample_cf_on_nodes(p4est, nodes, vector_components_cf, vector_field);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    std::ostringstream grid_name; grid_name << P4EST_DIM << "d_grid";
    if(!vtk_off)
      my_p4est_vtk_write_all(p4est, nodes, NULL,
                             P4EST_TRUE, P4EST_TRUE,
                             0, 0, grid_name.str().c_str());


    // set up the qnnn neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    std::ostringstream hierarchy_name; hierarchy_name << P4EST_DIM << "d_hierrchy";
    if(!vtk_off)
      hierarchy.write_vtk(hierarchy_name.str().c_str());
    my_p4est_node_neighbors_t qnnn(&hierarchy, nodes);
    qnnn.init_neighbors();

    Vec phi_xxyyzz, vector_field_xxyyzz;
    if(method == quadratic || method == quadratic_non_oscillatory)
    {
      if(!timing_off)
      w2.start("computing second derivatives of phi and of the vector field");
      ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &phi_xxyyzz); CHKERRXX(ierr);
      ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM*P4EST_DIM, &vector_field_xxyyzz); CHKERRXX(ierr);
      qnnn.second_derivatives_central(phi, phi_xxyyzz, 1);
      qnnn.second_derivatives_central(vector_field, vector_field_xxyyzz, P4EST_DIM);
      if(!timing_off){
      w2.stop(); w2.print_duration(); }
    }
    else
    {
      phi_xxyyzz = NULL;
      vector_field_xxyyzz = NULL;
    }


    grid_name.str(""); grid_name << P4EST_DIM << "d_grid_qnnn_" << p4est->mpirank << "_" << p4est->mpisize;

    std::ostringstream oss; oss << P4EST_DIM << "d_phi_" << mpi.size();
    if(!vtk_off)
    {
      double *phi_p, *vector_field_p;
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(vector_field, &vector_field_p); CHKERRXX(ierr);
      my_p4est_vtk_write_all_general(p4est, nodes, ghost,
                                     P4EST_TRUE, P4EST_TRUE,
                                     1, 0, 1, 0, 0, 0, oss.str().c_str(),
                                     VTK_NODE_SCALAR, "phi", phi_p,
                                     VTK_NODE_VECTOR_BLOCK, "vector", vector_field_p);
      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(vector_field, &vector_field_p); CHKERRXX(ierr);
    }

    // move the circle to create another grid for creating the new grid
    cf_data.max_lvl = cmd.get<int>("lmax_final", lmax_original-2);
    cf_data.min_lvl = cmd.get<int>("lmin_final", lmin_original+1);
    circle original_circle = circ;
#ifdef P4_TO_P8
    circ.update(.75, 1.15, .57, .2);
#else
    circ.update(.75, 1.15, .2);
#endif

    // Create a new grid
    if(!timing_off)
      w2.start("creating/refining/partitioning new p4est");
    p4est_t *p4est_np1 = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est_np1->user_pointer = (void*)&cf_data;
    my_p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    /*
     * Here we create a new nodes structure. Note that in general if what you
     * want is the same procedure as before. This means if the previous grid
     * included ghost cells in the ghost node struture, usually the new one
     * should also include a NEW ghost structure.
     * Here, however, we do not care about this and simply pass NULL to for the
     * new node structure
     */
    if(!timing_off)
      w2.start("creating new node data structure");
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    // Create an interpolating function

    if(!timing_off)
      w2.start("interpolating");
    my_p4est_interpolation_nodes_t node_interpolator(&qnnn);
    node_interpolator.set_input(phi, phi_xxyyzz, method);

    for (p4est_locidx_t i=0; i<nodes_np1->num_owned_indeps; ++i)
    {
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est_np1, nodes_np1, xyz);
      // buffer the point
      node_interpolator.add_point(i, xyz);
    }

    // interpolate on to the new vector
    Vec phi_np1, vector_field_np1;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
    ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, P4EST_DIM, &vector_field_np1); CHKERRXX(ierr);

    node_interpolator.interpolate(phi_np1);
    node_interpolator.set_input(vector_field, vector_field_xxyyzz, method, P4EST_DIM);
    node_interpolator.interpolate(vector_field_np1);

    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vector_field_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vector_field_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(!timing_off){
      w2.stop(); w2.print_duration(); }

    oss.str(""); oss << P4EST_DIM << "d_phi_np1_" << mpi.size();

    double *phi_np1_p, *vector_field_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(vector_field_np1, &vector_field_np1_p); CHKERRXX(ierr);

    if(!vtk_off)
      my_p4est_vtk_write_all_general(p4est_np1, nodes_np1, NULL,
                                     P4EST_TRUE, P4EST_TRUE,
                                     1, 0, 1, 0, 0, 0, oss.str().c_str(),
                                     VTK_NODE_SCALAR, "phi_np1", phi_np1_p,
                                     VTK_NODE_VECTOR_BLOCK, "vector_np1", vector_field_np1_p);

    if(mpi.rank() == 0)
      std::cout << std::endl << "Interpolation errors in infinity norm on the final grid: " << std::endl << std::endl;
    double err_phi = 0.0;
    double err_vector[P4EST_DIM];
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
      err_vector[dim] = 0.0;
    for (p4est_locidx_t i=0; i<nodes_np1->num_owned_indeps; ++i)
    {
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est_np1, nodes_np1, xyz);
      err_phi = MAX(err_phi, fabs(phi_np1_p[i] - original_circle(xyz[0], xyz[1]
              #ifdef P4_TO_P8
                    , xyz[2]
    #endif
          )));
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        err_vector[dim] = MAX(err_vector[dim], fabs(vector_field_np1_p[P4EST_DIM*i+dim] - vector_components_cf[dim]->operator()(xyz[0], xyz[1]
    #ifdef P4_TO_P8
            , xyz[2]
    #endif
            )));
    }
    int mpiret  = MPI_Allreduce(MPI_IN_PLACE, &err_phi,   1,          MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    mpiret      = MPI_Allreduce(MPI_IN_PLACE, err_vector, P4EST_DIM,  MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
    if(mpi.rank() == 0)
    {
      std::cout << std::endl;
      std::cout << "The error in phi_np1, in infinity norm, is: " << err_phi << std::endl;
      std::cout << std::endl;
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        std::cout << "The error in vector component " << (int) dim << ", in infinity norm, is: " << err_vector[dim] << std::endl;
      std::cout << std::endl;
    }
    ierr = VecRestoreArray(vector_field_np1, &vector_field_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_np1,  &phi_np1_p); CHKERRXX(ierr);

    // finally, delete PETSc Vecs by calling 'VecDestroy' function
    ierr = VecDestroy(phi);                   CHKERRXX(ierr);
    ierr = VecDestroy(vector_field);          CHKERRXX(ierr);
    ierr = VecDestroy(phi_np1);               CHKERRXX(ierr);
    ierr = VecDestroy(vector_field_np1);      CHKERRXX(ierr);
    if(phi_xxyyzz != NULL) {
      ierr = VecDestroy(phi_xxyyzz);          CHKERRXX(ierr); }
    if(vector_field_xxyyzz != NULL) {
      ierr = VecDestroy(vector_field_xxyyzz); CHKERRXX(ierr); }

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    p4est_nodes_destroy (nodes_np1);
    p4est_destroy (p4est_np1);
    my_p4est_brick_destroy(connectivity, brick);

    if(!timing_off) {
      w1.stop();
      w1.print_duration();
    }

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

