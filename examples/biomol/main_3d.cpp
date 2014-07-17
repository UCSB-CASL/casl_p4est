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
      PetscPrintf(p4est->mpicomm, "%ld\n", nodes->global_owned_indeps[i]);
      global_num_nodes += nodes->global_owned_indeps[i];
    }

    PetscPrintf(p4est->mpicomm, "global number of nodes     = %7ld \n"
                                "global number of quadrants = %7ld \n", global_num_nodes, global_num_quadrants);

    w2.start("removing internal cavities");
    mol.remove_internal_cavities(p4est, nodes, ghost, brick, phi);
    w2.stop(); w2.read_duration();

    global_num_quadrants = p4est->global_num_quadrants;
    global_num_nodes = 0;
    for (int i = 0; i<p4est->mpisize; i++)
      global_num_nodes += nodes->global_owned_indeps[i];

    PetscPrintf(p4est->mpicomm, "global number of nodes     = %7ld \n"
                                "global number of quadrants = %7ld \n", global_num_nodes, global_num_quadrants);

    BioMoleculeSolver solver(mol, p4est, nodes, ghost, brick);
    solver.set_electrolyte_parameters(10*sqrt(10), 2, 80);
    solver.set_phi(phi);

    w2.start("solving pb on the molecule");
    Vec psi_mol, psi_elec;
    solver.solve_nonlinear(psi_mol, psi_elec);
    w2.stop(); w2.read_duration();

    w2.start("saving vtk file");
    double *phi_p,*psi_mol_p, *psi_elec_p;
    ierr = VecGetArray(phi,      &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(psi_mol,  &psi_mol_p);  CHKERRXX(ierr);
    ierr = VecGetArray(psi_elec, &psi_elec_p); CHKERRXX(ierr);

    ostringstream oss; oss << output_folder + "/" + pqr;
    my_p4est_vtk_write_all(p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "psi_mol", psi_mol_p,
                           VTK_POINT_DATA, "phi_elec", psi_elec_p);

    ierr = VecRestoreArray(phi,      &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_mol,  &psi_mol_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(psi_elec, &psi_elec_p); CHKERRXX(ierr);
    w2.stop(); w2.read_duration();

    /* release memory */
    ierr = VecDestroy(psi_mol); CHKERRXX(ierr);
    ierr = VecDestroy(psi_elec); CHKERRXX(ierr);
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
