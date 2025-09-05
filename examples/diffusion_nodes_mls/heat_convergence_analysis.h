// heat_convergence_analysis.h
#ifndef HEAT_CONVERGENCE_ANALYSIS_H
#define HEAT_CONVERGENCE_ANALYSIS_H

#include <vector>

class HeatConvergenceAnalysis {
public:
    struct ConvergenceData {
        double h;        // Grid spacing
        double dt;       // Time step
        double error;    // Error norm
        double rate;     // Convergence rate
    };

    std::vector<ConvergenceData> spatial_data;
    std::vector<ConvergenceData> temporal_data;

    void add_spatial_data(double h, double error);
    void add_temporal_data(double dt, double error);

    double compute_rate(double h1, double e1, double h2, double e2);
    void print_convergence_table();
};

#endif