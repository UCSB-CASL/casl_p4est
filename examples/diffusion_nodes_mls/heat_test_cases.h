// heat_test_cases.h
#ifndef HEAT_TEST_CASES_H
#define HEAT_TEST_CASES_H

#include <src/my_p4est_heat_solver_mls.h>
#include <src/my_p4est_macros.h>
#include "heat_problem_definitions.h"

class HeatTestCase {
private:
    int test_number;
    TestSolutionCF* exact_solution;
    DiffusionCoeffCF* diffusion;
    SourceTermCF* source;

public:
    HeatTestCase() : test_number(0), exact_solution(nullptr),
                     diffusion(nullptr), source(nullptr) {}

    ~HeatTestCase() {
        if (exact_solution) delete exact_solution;
        if (diffusion) delete diffusion;
        if (source) delete source;
    }

    void setup(int test_case);

    void set_coefficients(my_p4est_heat_solver_mls_t& solver,
                         p4est_t* p4est, p4est_nodes_t* nodes);

    void set_initial_condition(p4est_t* p4est, p4est_nodes_t* nodes, Vec u0);

    bool has_exact_solution() const { return exact_solution != nullptr; }

    void get_exact_solution(double t, p4est_t* p4est, p4est_nodes_t* nodes, Vec exact);

    double compute_error(Vec numerical, Vec exact);
};

#endif