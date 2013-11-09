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
    cmd.add_option("mode", "interpolation mode 0 = linear, 1 = IDW, 2 = LSQR");
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
    Vec phi;
    ierr = VecCreateGhostCells(p4est, ghost, &phi); CHKERRXX(ierr);
    sample_cf_on_cells(p4est, ghost, circ, phi);
    w2.stop(); w2.read_duration();

    std::ostringstream grid_name; grid_name << P4EST_DIM << "d_grid";
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 0, grid_name.str().c_str());

    // set up the qnnn neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_cell_neighbors_t cnnn(&hierarchy);
#ifndef P4_TO_P8
    cnnn.write_triangulation("triangulation");
#endif

    std::ostringstream oss; oss << P4EST_DIM << "d_phi_" << mpi->mpisize;
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 1, oss.str().c_str(),
                           VTK_CELL_DATA, "phi", phi_p);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    // move the circle to create another grid
    cf_data.max_lvl -= 2;
    cf_data.min_lvl += 1;

    circle circ_old(circ);
#ifdef P4_TO_P8
    circ.update(.75, 1.15, .57, .2);
#else
    circ.update(.75, 1.15, .2);
#endif

    // Create a new grid
    w2.start("creating/refining/partitioning new p4est");
    p4est_t *p4est_np1 = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    p4est_np1->user_pointer = (void*)&cf_data;
    p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);
    p4est_partition(p4est_np1, NULL);
    w2.stop(); w2.read_duration();

    /*
     * Here we create a new nodes structure. Note that in general if what you
     * want is the same procedure as before. This means if the previous grid
     * included ghost cells in the ghost node struture, usually the new one
     * should also include a NEW ghost structure.
     * Here, however, we do not care about this and simply pass NULL to for the
     * new node structure
     */
    w2.start("creating new node data structure");
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
    w2.stop(); w2.read_duration();

    // Create an interpolating function
    InterpolatingFunctionCellBase phi_func(&cnnn);

    for (p4est_locidx_t i=0; i<nodes_np1->num_owned_indeps; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, i+nodes_np1->offset_owned_indeps);
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
      phi_func.add_point_to_buffer(i, xyz);
    }

    // interpolate on to the new vector
    Vec phi_np1, phi_nodes;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_nodes); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_nodes, &phi_np1); CHKERRXX(ierr);

    // interpolate
    switch (cmd.get("mode",0)){
    case 0:
      phi_func.set_input_parameters(phi, linear);
      break;
    case 1:
      phi_func.set_input_parameters(phi, IDW);
      break;
    case 2:
      phi_func.set_input_parameters(phi, LSQR);
      break;
    default:
      throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }
    phi_func.interpolate(phi_np1);

    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_np1, nodes_np1, circ_old, phi_nodes);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    oss.str(""); oss << P4EST_DIM << "d_phi_np1_" << mpi->mpisize;

    double *phi_np1_p, *phi_nodes_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_nodes, &phi_nodes_p); CHKERRXX(ierr);
    
    my_p4est_vtk_write_all(p4est_np1, nodes_np1, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi_np1", phi_np1_p,
                           VTK_POINT_DATA, "phi_nodes", phi_nodes_p);

    ierr = VecRestoreArray(phi_np1,  &phi_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_nodes,  &phi_nodes_p); CHKERRXX(ierr);

    // finally, delete PETSc Vecs by calling 'VecDestroy' function
    ierr = VecDestroy(phi);          CHKERRXX(ierr);
    ierr = VecDestroy(phi_np1);  CHKERRXX(ierr);
    ierr = VecDestroy(phi_nodes);  CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    p4est_nodes_destroy (nodes_np1);
    p4est_destroy (p4est_np1);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

