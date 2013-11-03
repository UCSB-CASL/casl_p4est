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
#include <src/my_p8est_interpolating_function_cell_base.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolating_function_cell_base.h>
#endif

#include <src/CASL_math.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

#ifdef P4_TO_P8
struct circle:CF_3{
  circle(double x0_, double y0_, double z0_, double r_)
    : x0(x0_), y0(y0_), z0(z0_), r(r_)
  {}
  void update(double x0_, double y0_, double z0_, double r_)
  {
    x0 = x0_;
    y0 = y0_;
    z0 = z0_;
    r  = r_;
  }

  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
private:
  double x0, y0, z0, r;
};

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return x*x + y*y + z*z;
  }
} uex;

#else
struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update(double x0_, double y0_, double r_)
  {
    x0 = x0_;
    y0 = y0_;
    r  = r_;
  }

  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};

static struct:CF_2{
  double operator()(double x, double y) const {
    return x*x + y*y;
  }
} uex;

#endif

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("mode", "interpolation mode 0 = linear, 1 = quadratic, 2 = non-oscilatory quadratic");
    cmd.parse(argc, argv);


#ifdef P4_TO_P8
    circle circ(1, 1, 1, .3);
#else
    circle circ(1, 1, .3);
#endif
    splitting_criteria_cf_t cf_data(cmd.get("lmin", 0), cmd.get("lmax",5), &circ, 1);

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2, brick);
#else
    connectivity = my_p4est_brick_new(2, 2, brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine");
    p4est->user_pointer = (void*)(&cf_data);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating node structure");
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    w2.start("computing phi");
    Vec phi_node, phi_cell;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_node); CHKERRXX(ierr);
    ierr = VecCreateGhostCells(p4est, ghost, &phi_cell); CHKERRXX(ierr);
    Vec phi_c2n;
    ierr = VecDuplicate(phi_node, &phi_c2n); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, circ, phi_node);
    sample_cf_on_cells(p4est, ghost, circ, phi_cell);
    w2.stop(); w2.read_duration();

    // set up the qnnn neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_cell_neighbors_t cnnn(&hierarchy);
    InterpolatingFunctionCellBase interp(&cnnn);

    /*
     * Here we create a new nodes structure. Note that in general if what you
     * want is the same procedure as before. This means if the previous grid
     * included ghost cells in the ghost node struture, usually the new one
     * should also include a NEW ghost structure.
     * Here, however, we do not care about this and simply pass NULL to for the
     * new node structure
     */

    for (p4est_locidx_t i=0; i<nodes->num_owned_indeps; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
      double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif
      double xyz [P4EST_DIM] =
      {
        node_x_fr_i(node) + tree_xmin,
        node_y_fr_j(node) + tree_ymin
#ifdef P4_TO_P8
        ,
        node_z_fr_k(node) + tree_zmin
#endif
      };

      // buffer the point
      interp.add_point_to_buffer(i, xyz);
    }

    // interpolate
    switch (cmd.get("mode",1)){
    case 0:
      interp.set_input_parameters(phi_cell, linear);
      break;
    case 1:
      interp.set_input_parameters(phi_cell, IDW);
      break;
    case 2:
      interp.set_input_parameters(phi_cell, LSQR);
      break;
    default:
      throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }
    interp.interpolate(phi_c2n);

    ierr = VecGhostUpdateBegin(phi_c2n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(phi_c2n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    std::ostringstream oss; oss << P4EST_DIM << "d_simple_" << mpi->mpisize;

    double *phi_np, *phi_cp, *phi_c2np;
    ierr = VecGetArray(phi_node, &phi_np);  CHKERRXX(ierr);
    ierr = VecGetArray(phi_cell, &phi_cp);  CHKERRXX(ierr);
    ierr = VecGetArray(phi_c2n, &phi_c2np); CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 1, oss.str().c_str(),
                           VTK_POINT_DATA, "phi_node", phi_np,
                           VTK_POINT_DATA, "phi_c2n",  phi_c2np,
                           VTK_CELL_DATA,  "phi_cell", phi_cp);

    ierr = VecRestoreArray(phi_node, &phi_np);   CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_cell, &phi_cp);   CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_c2n, &phi_c2np); CHKERRXX(ierr);

    // finally, delete PETSc Vecs by calling 'VecDestroy' function
    ierr = VecDestroy(phi_node); CHKERRXX(ierr);
    ierr = VecDestroy(phi_cell); CHKERRXX(ierr);
    ierr = VecDestroy(phi_c2n); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

