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
#include <src/my_p8est_poisson_node_base.h>
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
    PetscErrorCode ierr;
    mpi_session.init(argc, argv, mpi.mpicomm);
    MPI_Comm_size (mpi.mpicomm, &mpi.mpisize);
    MPI_Comm_rank (mpi.mpicomm, &mpi.mpirank);

    cmdParser cmd;
    cmd.add_option("lmin", "the min level of the tree");
    cmd.add_option("lmax", "the max level of the tree");
    cmd.add_option("lip", "Lipchitz constant for the levelset");
    cmd.add_option("pqr", "path to the pqr file");
    cmd.add_option("input-dir", "folder in which pqr files are located");
    cmd.add_option("output-dir", "folder to save the results in");
    cmd.parse(argc, argv);

    // decide on the type and value of the boundary conditions
    const int lmin   = cmd.get("lmin", 0);
    const int lmax   = cmd.get("lmax", 10);
    const string folder = cmd.get<string> ("input-dir", "../../examples/biomol");
    const string output_folder = cmd.get<string>("output-dir");
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
    mol.set_probe_radius(1.4);
    w2.stop(); w2.read_duration();

    /* create the p4est */
    p4est_t *p4est = p4est_new(mpi.mpicomm, connectivity, 0, NULL, NULL);
    splitting_criteria_t split(lmin, lmax, lip);
    p4est->user_pointer = (void*)(&split);

    p4est_nodes_t *nodes;
    p4est_ghost_t *ghost;
    Vec phi;
    w2.start("constructing the grid and initializing level-set");
//    mol.construct_SES_by_advection(p4est, nodes, ghost, brick, phi);
    mol.construct_SES_by_reinitialization(p4est, nodes, ghost, brick, phi);
    w2.stop(); w2.read_duration();

    p4est_gloidx_t global_num_quadrants = p4est->global_num_quadrants;
    p4est_gloidx_t global_num_nodes = 0;
    for (int i = 0; i<p4est->mpisize; i++){
      global_num_nodes += nodes->global_owned_indeps[i];
    }

    PetscPrintf(p4est->mpicomm, "global number of nodes     = %7ld \n"
                                "global number of quadrants = %7ld \n", global_num_nodes, global_num_quadrants);

    w2.start("removing internal cavities");
    mol.remove_internal_cavities(p4est, nodes, ghost, brick, phi);
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    ostringstream oss; oss << output_folder + "/" + pqr + "_geometry";
    my_p4est_vtk_write_all(p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w2.stop(); w2.read_duration();

    global_num_quadrants = p4est->global_num_quadrants;
    global_num_nodes = 0;
    for (int i = 0; i<p4est->mpisize; i++)
      global_num_nodes += nodes->global_owned_indeps[i];

    PetscPrintf(p4est->mpicomm, "global number of nodes     = %7ld \n"
                                "global number of quadrants = %7ld \n", global_num_nodes, global_num_quadrants);

    w2.start("making the hierarchy and node numbering");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    w2.stop(); w2.read_duration();

    // Define boundary conditions
    struct: CF_3 {
      double operator()(double x, double y, double z) const {
        (void) x;
        (void) y;
        (void) z;

        return 1;
      }
    } bc_interface_value;

    struct: CF_3 {
      double operator()(double x, double y, double z) const {
        (void) x;
        (void) y;
        (void) z;

        return 0;
      }
    } bc_wall_value;

    struct: WallBC3D {
      BoundaryConditionType operator()(double x, double y, double z) const {
        (void) x;
        (void) y;
        (void) z;

        return DIRICHLET;
      }
    } bc_wall_type;

    BoundaryConditions3D bc;
    bc.setInterfaceType(DIRICHLET);
    bc.setInterfaceValue(bc_interface_value);
    bc.setWallTypes(bc_wall_type);
    bc.setWallValues(bc_wall_value);

    // Allocate memory for solution
    Vec psi;
    ierr = VecDuplicate(phi, &psi); CHKERRXX(ierr);

    PoissonSolverNodeBase poisson(&node_neighbors);
    poisson.set_bc(bc);
    poisson.set_phi(phi);

    w2.start("solving pb on the molecule");
    poisson.solve(psi);
    w2.stop(); w2.read_duration();

    w2.start("saving vtk file");
    double *psi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(psi, &psi_p);  CHKERRXX(ierr);

    oss.str(""); oss << output_folder + "/" + pqr + "_simple_poisson";
    my_p4est_vtk_write_all(p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "psi", psi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(psi, &psi_p);  CHKERRXX(ierr);
    w2.stop(); w2.read_duration();

    /* release memory */
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(psi); CHKERRXX(ierr);

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
