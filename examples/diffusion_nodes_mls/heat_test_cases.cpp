// heat_test_cases.cpp
#include "heat_test_cases.h"
#include <src/my_p4est_utils.h>

void HeatTestCase::setup(int test_case) {
    test_number = test_case;

    // Clean up old pointers
    if (exact_solution) { delete exact_solution; exact_solution = nullptr; }
    if (diffusion) { delete diffusion; diffusion = nullptr; }
    if (source) { delete source; source = nullptr; }

    switch(test_case) {
        case 0:  // Simple diffusion with constant coefficients
            diffusion = new DiffusionCoeffCF(1.0);
            source = new SourceTermCF(0.0);
            exact_solution = new TestSolutionCF(1, 1.0);  // Sinusoidal
            break;

        case 1:  // Heat equation with source
            diffusion = new DiffusionCoeffCF(0.1);
            source = new SourceTermCF(1.0);
            exact_solution = nullptr;  // No exact solution
            break;

        default:  // Default case
            diffusion = new DiffusionCoeffCF(1.0);
            source = new SourceTermCF(0.0);
            exact_solution = new TestSolutionCF(0, 0.0);  // Zero
            break;
    }
}

void HeatTestCase::set_coefficients(my_p4est_heat_solver_mls_t& solver,
                                    p4est_t* p4est, p4est_nodes_t* nodes) {
    // Create vectors for coefficients
    Vec mu_vec, source_vec, diag_vec;

    VecCreateGhostNodes(p4est, nodes, &mu_vec);
    VecCreateGhostNodes(p4est, nodes, &source_vec);
    VecCreateGhostNodes(p4est, nodes, &diag_vec);

    // Sample the coefficient functions
    if (diffusion) sample_cf_on_nodes(p4est, nodes, *diffusion, mu_vec);
    else VecSet(mu_vec, 1.0);

    if (source) sample_cf_on_nodes(p4est, nodes, *source, source_vec);
    else VecSet(source_vec, 0.0);

    VecSet(diag_vec, 0.0);  // No reaction term

    // Set in solver
    solver.set_mu(mu_vec, mu_vec);
    solver.set_rhs(source_vec, source_vec);
    solver.set_diag(diag_vec, diag_vec);

    // Clean up
    VecDestroy(mu_vec);
    VecDestroy(source_vec);
    VecDestroy(diag_vec);
}

void HeatTestCase::set_initial_condition(p4est_t* p4est, p4est_nodes_t* nodes, Vec u0) {
    if (exact_solution) {
        // Use exact solution at t=0 as initial condition
        sample_cf_on_nodes(p4est, nodes, *exact_solution, u0);
    } else {
        // Default: zero initial condition
        VecSet(u0, 0.0);
    }
}

void HeatTestCase::get_exact_solution(double t, p4est_t* p4est,
                                      p4est_nodes_t* nodes, Vec exact) {
    if (exact_solution) {
        // For now, just use the steady state solution
        // In a real implementation, you'd account for time evolution
        sample_cf_on_nodes(p4est, nodes, *exact_solution, exact);

        // Apply time decay for diffusion equation
        if (test_number == 0) {
            VecScale(exact, exp(-2.0 * M_PI * M_PI * t));
        }
    } else {
        VecSet(exact, 0.0);
    }
}

double HeatTestCase::compute_error(Vec numerical, Vec exact) {
    Vec error;
    VecDuplicate(numerical, &error);

    VecCopy(numerical, error);
    VecAXPY(error, -1.0, exact);  // error = numerical - exact

    double norm;
    VecNorm(error, NORM_2, &norm);

    VecDestroy(error);
    return norm;
}