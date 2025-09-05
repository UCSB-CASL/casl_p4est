// heat_convergence_analysis.cpp
#include "heat_convergence_analysis.h"
#include <cmath>
#include <iostream>
#include <iomanip>

void HeatConvergenceAnalysis::add_spatial_data(double h, double error) {
    ConvergenceData data;
    data.h = h;
    data.dt = 0;
    data.error = error;

    if (spatial_data.size() > 0) {
        auto& prev = spatial_data.back();
        data.rate = compute_rate(prev.h, prev.error, h, error);
    } else {
        data.rate = 0;
    }

    spatial_data.push_back(data);
}

void HeatConvergenceAnalysis::add_temporal_data(double dt, double error) {
    ConvergenceData data;
    data.h = 0;
    data.dt = dt;
    data.error = error;

    if (temporal_data.size() > 0) {
        auto& prev = temporal_data.back();
        data.rate = compute_rate(prev.dt, prev.error, dt, error);
    } else {
        data.rate = 0;
    }

    temporal_data.push_back(data);
}

double HeatConvergenceAnalysis::compute_rate(double h1, double e1, double h2, double e2) {
    if (h1 <= 0 || h2 <= 0 || e1 <= 0 || e2 <= 0) return 0;
    return log(e1/e2) / log(h1/h2);
}

void HeatConvergenceAnalysis::print_convergence_table() {
    if (spatial_data.size() > 0) {
        std::cout << "\n=== Spatial Convergence ===" << std::endl;
        std::cout << std::setw(12) << "h"
                  << std::setw(15) << "Error"
                  << std::setw(10) << "Rate" << std::endl;
        for (const auto& d : spatial_data) {
            std::cout << std::scientific << std::setprecision(4)
                      << std::setw(12) << d.h
                      << std::setw(15) << d.error
                      << std::fixed << std::setprecision(2)
                      << std::setw(10) << d.rate << std::endl;
        }
    }

    if (temporal_data.size() > 0) {
        std::cout << "\n=== Temporal Convergence ===" << std::endl;
        std::cout << std::setw(12) << "dt"
                  << std::setw(15) << "Error"
                  << std::setw(10) << "Rate" << std::endl;
        for (const auto& d : temporal_data) {
            std::cout << std::scientific << std::setprecision(4)
                      << std::setw(12) << d.dt
                      << std::setw(15) << d.error
                      << std::fixed << std::setprecision(2)
                      << std::setw(10) << d.rate << std::endl;
        }
    }
}