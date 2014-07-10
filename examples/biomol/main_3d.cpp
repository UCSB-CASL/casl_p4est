// System
#include <stdexcept>
#include <iostream>
#include <vector>

#include <src/my_p4est_to_p8est.h>
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base_jump.h>
#include <src/my_p8est_levelset.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>

#include "bio_molecule.h"

#undef MIN
#undef MAX

using namespace std;

int main(int argc, char *argv[]) {
  try{
    mpi_context_t mpi = {MPI_COMM_WORLD, 0, 0};
    Session mpi_session;
    mpi_session.init(argc, argv, mpi.mpicomm);
    MPI_Comm_size (mpi.mpicomm, &mpi.mpisize);
    MPI_Comm_rank (mpi.mpicomm, &mpi.mpirank);

    cmdParser cmd;
    cmd.add_option("lmin", "the min level of the tree");
    cmd.add_option("lmax", "the max level of the tree");
    cmd.add_option("lip", "Lipchitz constant for the levelset");
    cmd.add_option("pqr", "path to the pqr file");
    cmd.add_option("folder", "folder in which pqr files are located");
    cmd.parse(argc, argv);

    // decide on the type and value of the boundary conditions
    const int lmin   = cmd.get("lmin", 0);
    const int lmax   = cmd.get("lmax", 10);
    const string folder = cmd.get<string> ("folder", "../../examples/biomol");
    const string pqr = cmd.get<string>("pqr", "1d65");
    const double lip = cmd.get("lip", 1.5);

    parStopWatch w1, w2;
    w1.start("total time");

    /* create the macro mesh */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
    connectivity = my_p4est_brick_new(1, 1, 1, &brick);

    w2.start("reading pqr molecule");
    BioMolecule mol(brick, mpi);
    mol.read(folder + "/" + pqr + ".pqr");
    const double rp = 1.4;
//    mol.set_probe_radius(rp);
    w2.stop(); w2.read_duration();

    /* create the p4est */
    w2.start("initializing the grid");
    p4est_t *p4est = p4est_new(mpi.mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    /* refine the forest */
    splitting_criteria_cf_t split(lmin, lmax, &mol, lip);
    p4est->user_pointer = (void*)(&split);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    /* partition the p4est */
    p4est_partition(p4est, NULL);

    /* create the ghost layer */
    p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    /* reinitialize the level-set */
    w2.start("constructing SAS surface");
    Vec phi;
    PetscErrorCode ierr;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, mol, phi);
    w2.stop(); w2.read_duration();

    w2.start("reinitializing the level-set");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    neighbors.init_neighbors();
    my_p4est_level_set ls(&neighbors);

    ls.reinitialize_1st_order_time_2nd_order_space(phi);
    w2.stop(); w2.read_duration();

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    /* reconstruct the new grid */
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
//      phi_p[i] -= rp;
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, &brick, &neighbors);
    phi_interp.set_input_parameters(phi, quadratic_non_oscillatory);

    p4est_t *p4est_tmp = p4est_new(mpi.mpicomm, connectivity, 0, NULL, NULL);
    p4est_ghost_t *ghost_tmp = NULL;
    p4est_nodes_t *nodes_tmp = NULL;
    Vec phi_tmp = NULL;

    w2.start("regenerating the forest and level-set");
    PetscPrintf(mpi.mpicomm, "old number of global quadrants = %7d \n\n", p4est->global_num_quadrants);
    for (int l = 0; l<=lmax; l++) {
      break;
      /* partition and recompute the grid and levelset */
      w2.start("interpolating values onto the new tree");
      p4est_partition(p4est_tmp, NULL);
      ghost_tmp = p4est_ghost_new(p4est_tmp, P4EST_CONNECT_FULL);
      nodes_tmp = p4est_nodes_new(p4est_tmp, ghost_tmp);
      ierr = VecCreateGhostNodes(p4est_tmp, nodes_tmp, &phi_tmp); CHKERRXX(ierr);

      PetscPrintf(mpi.mpicomm, "current number of global quadrants = %7d \n", p4est_tmp->global_num_quadrants);

      /* buffer all the points in the current tree */
      for (size_t i = 0; i<nodes_tmp->indep_nodes.elem_count; i++) {
        const p4est_indep_t *ni = (const p4est_indep_t *)sc_array_index(&nodes_tmp->indep_nodes, i);

        p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*ni->p.piggy3.which_tree + 0];

        double tree_xmin = connectivity->vertices[3*v_mmm + 0];
        double tree_ymin = connectivity->vertices[3*v_mmm + 1];
        double tree_zmin = connectivity->vertices[3*v_mmm + 2];
        double xyz [P4EST_DIM] =
        {
          node_x_fr_i(ni) + tree_xmin,
          node_y_fr_j(ni) + tree_ymin,
          node_z_fr_k(ni) + tree_zmin
        };

        phi_interp.add_point_to_buffer(i, xyz);
      }
      phi_interp.interpolate(phi_tmp);
      w2.stop(); w2.read_duration();

      /* dont do the final refinement */
      if (l == lmax) break;

      /* refine the tree */
      w2.start("refinig the tree");
      double *phi_tmp_p;
      ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
      splitting_criteria_discrete_t sp(p4est_tmp, lmin, lmax, lip);
      sp.mark_cells_for_refinement(nodes_tmp, phi_tmp_p);
      ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

      p4est_tmp->user_pointer = &sp;
      p4est_refine(p4est_tmp, P4EST_FALSE, refine_marked_quadrants, NULL);

      /* free memory */
      ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
      p4est_ghost_destroy(ghost_tmp);
      p4est_nodes_destroy(nodes_tmp);
      w2.stop(); w2.read_duration();
    }
    w2.stop(); w2.read_duration();

    /* free memory and reset pointers */
//    p4est_destroy(p4est); p4est = p4est_tmp; p4est->user_pointer = &split; p4est_tmp = NULL;
//    p4est_ghost_destroy(ghost); ghost = ghost_tmp; ghost_tmp = NULL;
//    p4est_nodes_destroy(nodes); nodes = nodes_tmp; nodes_tmp = NULL;
//    ierr = VecDestroy(phi); phi = phi_tmp; phi_tmp = NULL;

    w2.start("saving vtk file");
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, pqr.c_str(),
                           VTK_POINT_DATA, "phi", phi_p);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w2.stop(); w2.read_duration();

    /* release memory */
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy (ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}
