// heat_problem_definitions.h
#ifndef HEAT_PROBLEM_DEFINITIONS_H
#define HEAT_PROBLEM_DEFINITIONS_H

#include <src/my_p4est_macros.h>
#include <cmath>

// Simple test solution function
class TestSolutionCF : public CF_2 {
    int type;
    double amplitude;
public:
    TestSolutionCF(int t = 0, double amp = 1.0) : type(t), amplitude(amp) {}

    double operator()(double x, double y) const {
        switch(type) {
            case 0: return 0.0;  // Zero solution
            case 1: return amplitude * sin(M_PI * x) * cos(M_PI * y);  // Sinusoidal
            case 2: return amplitude * exp(-((x*x + y*y)));  // Gaussian
            default: return 0.0;
        }
    }
};

// Simple diffusion coefficient
class DiffusionCoeffCF : public CF_2 {
    double value;
public:
    DiffusionCoeffCF(double v = 1.0) : value(v) {}
    double operator()(double x, double y) const { return value; }
};

// Simple source term
class SourceTermCF : public CF_2 {
    double value;
public:
    SourceTermCF(double v = 0.0) : value(v) {}
    double operator()(double x, double y) const { return value; }
};

#endif