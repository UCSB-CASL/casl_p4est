// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>
#include <p4est_mesh.h>

// casl_p4est
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <src/interpolating_function.h>

#include <src/neighbors.h>

#undef MIN
#undef MAX
#include <src/math.h>

#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_reinitialize.h>

using namespace std;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};

struct parabola:CF_2{
  parabola(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r*r - (SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};


struct splitting_criteria_custom_t {
    int max_lvl, min_lvl;
};


p4est_bool_t
refine_custom(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
{
    splitting_criteria_custom_t *data = (splitting_criteria_custom_t*)p4est->user_pointer;

    if (quad->level < data->min_lvl)
        return P4EST_TRUE;
    else if (quad->level >= data->max_lvl)
        return P4EST_FALSE;
    else
    {
        double dx, dy;
        dx_dy_dz_quadrant(p4est, which_tree, quad, &dx, &dy, NULL);

        double x = (double)quad->x/(double)P4EST_ROOT_LEN;
        double y = (double)quad->y/(double)P4EST_ROOT_LEN;

        c2p_coordinate_transform(p4est, which_tree, &x, &y, NULL);

        if( (x==1 && y==1) || (x+dx==1 && y+dx==1))
//        if(x+dx==1 && y+dy==1)
            return P4EST_TRUE;
        return P4EST_FALSE;
    }
}


int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    circle circ(1, 1, .3);
    parabola para(1, 1, .3);
    splitting_criteria_cf_t cf_circle = {&circ, 12, 0, 1};

    Session session;
    session.init(argc, argv, mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    /* Create the connectivity object */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
    connectivity = my_p4est_brick_new(2, 2, &brick);

    /* Now create the forest */
    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

    /* refine the forest using a refinement criteria */
    p4est->user_pointer = (void*)(&cf_circle);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    /* Finally re-partition */
    p4est_partition(p4est, NULL);

    /* Create the ghost structure */
    p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate the node data structure */
    nodes = my_p4est_nodes_new(p4est, ghost);

    /* initialize the neighbor nodes structure */
    my_p4est_hierarchy_t hierarchy(p4est,ghost);
    my_p4est_node_neighbors_t node_nb(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tree_xmin;
      double y = int2double_coordinate_transform(node->y) + tree_ymin;

      phi_p[p4est2petsc_local_numbering(nodes,i)] = para(x,y);
    }

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    my_p4est_level_set ls(p4est,nodes,&node_nb);
//    ls.reinitialize( phi, 100 );
    ls.reinitialize_2nd_order( phi, 100 );

    std::ostringstream oss; oss << "phi_" << mpi->mpisize;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);
    my_p4est_vtk_write_ghost_layer(p4est, ghost);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    /* make sure all mpi processes have time to write their info before continuing */
    int retval = MPI_Barrier (p4est->mpicomm);
    SC_CHECK_ABORT (!retval, P4EST_STRING "_vtk: Error synchronizing mpi");

    /* finally, delete PETSc Vecs by calling 'VecDestroy' function */
    ierr = VecDestroy(phi);     CHKERRXX(ierr);

    /* destroy the p4est and its connectivity structure */
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);
    p4est_connectivity_destroy (connectivity);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

