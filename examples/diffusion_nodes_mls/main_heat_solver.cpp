// main_heat_solver.cpp
#include <src/my_p4est_heat_solver_mls.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_utils.h>
#include <src/Parser.h>

#include "heat_problem_definitions.h"
#include "heat_test_cases.h"
#include "heat_convergence_analysis.h"

#include <src/parameter_list.h>
// Global parameter list
param_list_t pl;

// Grid parameters
param_t<int>    lmin (pl, 4, "lmin", "Min level of the tree");
param_t<int>    lmax (pl, 6, "lmax", "Max level of the tree");
param_t<double> xmin (pl, -1, "xmin", "Box xmin");
param_t<double> xmax (pl,  1, "xmax", "Box xmax");
param_t<double> ymin (pl, -1, "ymin", "Box ymin");
param_t<double> ymax (pl,  1, "ymax", "Box ymax");

// Time stepping parameters
param_t<double> dt_value    (pl, 0.01, "dt", "Time step size");
param_t<double> t_final     (pl, 1.0, "t_final", "Final time");
param_t<int>    time_method (pl, 0, "time_method", "0=BE, 1=CN, 2=BDF2");

// Test case
param_t<int>    test_case   (pl, 0, "test", "Test case number");
param_t<bool>   save_vtk    (pl, 1, "save_vtk", "Save VTK output");

int main(int argc, char* argv[])
{
    // Initialize MPI and PETSc
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);

    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Parse command line
    cmdParser cmd;
    pl.initialize_parser(cmd);
    cmd.parse(argc, argv);
    pl.set_from_cmd_all(cmd);

    if (rank == 0) {
        PetscPrintf(comm, "\n=== Heat Equation Solver ===\n");
        PetscPrintf(comm, "Test case: %d\n", test_case.val);
        PetscPrintf(comm, "Time step: %e\n", dt_value.val);
        PetscPrintf(comm, "Final time: %f\n", t_final.val);
    }

    // Set up test case
    HeatTestCase test;
    test.setup(test_case.val);

    // Create p4est grid
    double grid_min[] = {xmin.val, ymin.val};
    double grid_max[] = {xmax.val, ymax.val};
    int num_trees[] = {1, 1};
    int periodicity[] = {0, 0};

    my_p4est_brick_t brick;
    p4est_connectivity_t* connectivity =
        my_p4est_brick_new(num_trees, grid_min, grid_max, &brick, periodicity);

    p4est_t* p4est = p4est_new(comm, connectivity, 0, NULL, NULL);

    // Refine to desired level
    for (int i = 0; i < lmax.val; i++) {
        p4est_refine(p4est, 0, NULL, NULL);
        p4est_partition(p4est, 0, NULL);
    }

    // Create nodes
    p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t* nodes = p4est_nodes_new(p4est, ghost);

    // Set up heat solver
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy, nodes);
    ngbd_n.init_neighbors();

    my_p4est_heat_solver_mls_t heat_solver(&ngbd_n, p4est, nodes);

    // Configure solver
    heat_solver.set_time_step(dt_value.val);
    heat_solver.set_time_stepping_method(
        (my_p4est_heat_solver_mls_t::TimeSteppingMethod)time_method.val);

    // Set problem coefficients from test case
    test.set_coefficients(heat_solver, p4est, nodes);

    // Set initial condition
    Vec u0;
    VecCreateGhostNodes(p4est, nodes, &u0);
    test.set_initial_condition(p4est, nodes, u0);
    heat_solver.set_initial_condition(u0);

    // Time stepping loop
    double t = 0.0;
    int step = 0;

    while (t < t_final.val - 0.5*dt_value.val) {
        heat_solver.advance_one_time_step();
        t = heat_solver.get_current_time();
        step++;

        if (rank == 0 && step % 10 == 0) {
            PetscPrintf(comm, "Step %d: t = %f\n", step, t);

            // Check for steady state
            // if (heat_solver.(1e-10)) {
            //     PetscPrintf(comm, "Steady state reached at t = %f\n", t);
            //     break;
            // }
        }
    }

    // Compute error if exact solution exists
    if (test.has_exact_solution()) {
        Vec exact;
        VecCreateGhostNodes(p4est, nodes, &exact);
        test.get_exact_solution(t, p4est, nodes, exact);

        Vec solution = heat_solver.get_solution();
        double error = test.compute_error(solution, exact);

        if (rank == 0) {
            PetscPrintf(comm, "Final error at t = %f: %e\n", t, error);
        }

        VecDestroy(exact);
    }

    // Clean up
    VecDestroy(u0);
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy(p4est);
    p4est_connectivity_destroy(connectivity);

    PetscFinalize();
    return 0;
}