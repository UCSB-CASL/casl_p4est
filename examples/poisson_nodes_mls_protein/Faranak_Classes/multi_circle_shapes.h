//
// Created by Faranak on 2/6/25.
//

#ifndef MULTI_CIRCLE_SHAPE_H
#define MULTI_CIRCLE_SHAPE_H

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <utility>  // for std::pair

// System headers for p4est and shape functionality
#include "src/my_p4est_shapes.h"  // for CF_DIM and other base classes
#include "src/petsc_compatibility.h"
#include "src/Parser.h"
#include "src/parameter_list.h"

class MultiCircleShapePhi : public CF_2 {
public:
    double r0;      // radius of the circle
    double xc, yc;  // center coordinates
    double beta;    // deformation parameter (not used in simple circles, but kept for flexibility)
    double inside;  // interior (1) or exterior (-1)
    double theta, cos_theta, sin_theta;  // rotation parameters

    MultiCircleShapePhi(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                        double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        set_params(r0, xc, yc, beta, inside, theta);
    }

    void set_params(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        // Rotate and translate the point
        double X = (x - xc) * cos_theta - (y - yc) * sin_theta;
        double Y = (x - xc) * sin_theta + (y - yc) * cos_theta;

        double r = sqrt(X*X + Y*Y);

        // Avoid division by zero
        if (r < 1.0E-9) r = 1.0E-9;

        // Basic circular level set with optional deformation
        return inside * (r - r0 - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }
};

class MultiCircleShapePhiX : public CF_2 {
public:
    double r0;
    double xc, yc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    MultiCircleShapePhiX(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                         double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        set_params(r0, xc, yc, beta, inside, theta);
    }

    void set_params(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const
    {
        double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
        double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
        double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
                       -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0);
        double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
                       -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.);
        return phi_x*cos_theta+phi_y*sin_theta;
    }
};

class MultiCircleShapePhiY : public CF_2 {
public:
    double r0;
    double xc, yc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    MultiCircleShapePhiY(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                         double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        set_params(r0, xc, yc, beta, inside, theta);
    }

    void set_params(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const
    {
        double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
        double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
        double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
                       -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0);
        double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
                       -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.);
        return -phi_x*sin_theta+phi_y*cos_theta;
    }
};

struct MultiCircleShape {
    MultiCircleShapePhi phi;
    MultiCircleShapePhiX phi_x;
    MultiCircleShapePhiY phi_y;

    MultiCircleShape(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                     double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        phi.set_params(r0, xc, yc, beta, inside, theta);
        phi_x.set_params(r0, xc, yc, beta, inside, theta);
        phi_y.set_params(r0, xc, yc, beta, inside, theta);
    }

    void set_params(double r0 = 1.0, double xc = 0.0, double yc = 0.0,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0) {
        phi.set_params(r0, xc, yc, beta, inside, theta);
        phi_x.set_params(r0, xc, yc, beta, inside, theta);
        phi_y.set_params(r0, xc, yc, beta, inside, theta);
    }
};

class MultiCircleShapeGenerator {
private:
    // Domain bounds and parameters
    param_t<double>& xmin;
    param_t<double>& xmax;
    param_t<double>& ymin;
    param_t<double>& ymax;
    param_t<int>& nx;
    param_t<int>& ny;
    param_t<int>& lmin;
    param_t<int>& lmax;

    std::vector<MultiCircleShape> shapes;
    std::vector<CF_DIM*> phi_cf;
    std::vector<mls_opn_t> action;

    mutable size_t current_min_index;
    mutable double current_min_value;

    // store pre-computed level set
    mutable double precomputed_phi;
    mutable double precomputed_phi_x;
    mutable double precomputed_phi_y;

    // Modify isValidCenter to be more lenient
    bool isValidCenter(double x, double y, double radius,
                       const std::vector<MultiCircleShape>& existing_shapes) {
        double threshold = 0.5 * radius;  // Increased separation
        double inner_region = radius + threshold;

        if (x - inner_region < xmin.val || x + inner_region > xmax.val ||
            y - inner_region < ymin.val || y + inner_region > ymax.val) {
            std::cout << "Circle center (" << x << ", " << y
                      << ") outside domain bounds" << std::endl;
            return false;
        }

        for (const auto& shape : existing_shapes) {
            double dist = std::sqrt(std::pow(x - shape.phi.xc, 2) +
                                    std::pow(y - shape.phi.yc, 2));
            if (dist < 2.5 * radius) {
                std::cout << "Circle too close to existing circle. Distance: "
                          << dist << ", Minimum: " << 2.5 * radius << std::endl;
                return false;
            }
        }
        return true;
    }

    std::pair<double, double> generateRandomCenter(double radius) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist_x(xmin.val + radius, xmax.val - radius);
        std::uniform_real_distribution<double> dist_y(ymin.val + radius, ymax.val - radius);
        return {dist_x(rng), dist_y(rng)};
    }

    // Helper function to find the right shape index and value
    std::pair<size_t, double> find_extremum(double x, double y) const {
    if (shapes.empty()) {
        return {0, std::numeric_limits<double>::max()};
    }

    const double inside_param = shapes[0].phi.inside;
    size_t index = 0;
    double extremum = shapes[0].phi(x, y);

    for (size_t i = 1; i < shapes.size(); i++) {
        double current = shapes[i].phi(x, y);
        if ((inside_param > 0 && current < extremum) ||  // minimum for inside
            (inside_param < 0 && current > extremum)) {  // maximum for outside
            extremum = current;
            index = i;
        }
    }

    return {index, extremum};
    }


public:
    MultiCircleShapeGenerator(param_t<double>& xmin_, param_t<double>& xmax_,
                              param_t<double>& ymin_, param_t<double>& ymax_,
                              param_t<int>& nx_, param_t<int>& ny_,
                              param_t<int>& lmin_, param_t<int>& lmax_)
            : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_),
              nx(nx_), ny(ny_), lmin(lmin_), lmax(lmax_) {}

    void initializeCombinedLevelSet() {
        if (shapes.empty()) return;

        // Pre-compute combined level set for all shapes
        precomputed_phi = shapes[0].phi.r0;  // Initialize with first shape
        precomputed_phi_x = 0.0;
        precomputed_phi_y = 0.0;

        // Store first shape's parameters as reference
        const double inside_param = shapes[0].phi.inside;

        // Combine all shapes
        for (size_t i = 1; i < shapes.size(); i++) {
            if (inside_param > 0) {
                precomputed_phi = std::min(precomputed_phi, shapes[i].phi.r0);
            } else {
                precomputed_phi = std::max(precomputed_phi, shapes[i].phi.r0);
            }
        }
    }

    void generateCircles(int num_circles, double radius, int inside_value = 1, double beta = 0.0) {
        int max_attempts = 1000;
        shapes.clear();
        shapes.reserve(num_circles);

        std::cout << "Attempting to generate " << num_circles
                  << " circles with radius " << radius << std::endl;

        for (int i = 0; i < num_circles; ++i) {
            int attempts = 0;
            bool found_valid_center = false;
            double x, y;

            while (!found_valid_center && attempts < max_attempts) {
                auto [new_x, new_y] = generateRandomCenter(radius);
                if (isValidCenter(new_x, new_y, radius, shapes)) {
                    x = new_x;
                    y = new_y;
                    found_valid_center = true;
                }
                attempts++;
            }

            if (found_valid_center) {
                shapes.emplace_back(radius, x, y, beta, inside_value);
                std::cout << "Successfully placed circle " << i
                          << " at (" << x << ", " << y << ")" << std::endl;
            } else {
                std::cerr << "Warning: Could not place circle " << i + 1
                          << " after " << max_attempts << " attempts" << std::endl;
                break;
            }
        }

        std::cout << "Actually generated " << shapes.size() << " circles" << std::endl;
        initializeCombinedLevelSet();
    }

    double Phi(double x, double y, double z = 0) const {
        if (shapes.empty()) {
            return std::numeric_limits<double>::max();
        }
        return shapes[current_min_index].phi(x, y);
    }

    double Phi_x(double x, double y, double z = 0) const {
        if (shapes.empty()) {
            return 0.0;
        }
        return shapes[current_min_index].phi_x(x, y);
    }

    double Phi_y(double x, double y, double z = 0) const {
        if (shapes.empty()) {
            return 0.0;
        }
        return shapes[current_min_index].phi_y(x, y);
    }

    double evaluate(double x, double y, double z = 0) const {
        return find_extremum(x, y).second;
    }

    double evaluate_x(double x, double y, double z = 0) const {
        if (shapes.empty()) {
            return 0.0;
        }
        size_t index = find_extremum(x, y).first;
        return shapes[index].phi_x(x, y);
    }

    double evaluate_y(double x, double y, double z = 0) const {
        if (shapes.empty()) {
            return 0.0;
        }
        size_t index = find_extremum(x, y).first;
        return shapes[index].phi_y(x, y);
    }

    void setShapes(const std::vector<MultiCircleShape>& new_shapes) {
        shapes = new_shapes;
        initializeCombinedLevelSet();
    }

    // Compute resolution dynamically
    std::pair<int, int> computeResolution(int iter) const {
        int resolution_x = nx.val * pow(2, lmax.val + iter);
        int resolution_y = ny.val * pow(2, lmax.val + iter);
        return {resolution_x, resolution_y};
    }

    // Export level set values to file
    void exportToFile(const std::string& filename, int iter) {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // Use the dynamically computed resolution
        auto [resolution_x, resolution_y] = computeResolution(iter);

        double dx = (xmax.val - xmin.val) / (resolution_x - 1);
        double dy = (ymax.val - ymin.val) / (resolution_y - 1);

        for (int i = 0; i < resolution_y; ++i) {
            for (int j = 0; j < resolution_x; ++j) {
                double x = xmin.val + j * dx;
                double y = ymin.val + i * dy;

                outFile << x << "\t" << y << "\t"
                        << evaluate(x, y) << "\n";
            }
        }
        outFile.close();
    }
};

#endif // MULTI_CIRCLE_SHAPE_H

