//
// Created by Faranak on 2/6/25.
//

// random_shape_generator.h
#ifndef RANDOM_SHAPE_GENERATOR_H
#define RANDOM_SHAPE_GENERATOR_H

#include <vector>
#include <random>
#include <cmath>
#include "src/my_p4est_shapes.h"
#include "src/petsc_compatibility.h"
#include "src/Parser.h"
#include "src/parameter_list.h"

struct CircleShape {
    double x, y;  // center coordinates
    double radius;
    flower_shaped_domain_t shape;

    CircleShape(double x_, double y_, double radius_)
            : x(x_), y(y_), radius(radius_),
              shape(radius_, x_, y_, 0, 1) {} // Use existing flower_shaped_domain_t
};

class RandomShapeGenerator {
private:
    std::mt19937 rng;
    // Replace Grid2D with parameter references
    param_t<double>& xmin;
    param_t<double>& xmax;
    param_t<double>& ymin;
    param_t<double>& ymax;
    param_t<int>& nx;
    param_t<int>& ny;

    bool isValidCenter(double x, double y, double radius,
                       const std::vector<CircleShape>& existing_shapes) {
        double threshold = 0.2 * radius;
        double inner_region = radius + threshold;

        // Use .val to access parameter values
        if (x - inner_region < xmin.val || x + inner_region > xmax.val ||
            y - inner_region < ymin.val || y + inner_region > ymax.val) {
            return false;
        }

        // Rest remains the same
        for (const auto& shape : existing_shapes) {
            double dist = std::sqrt(std::pow(x - shape.x, 2) + std::pow(y - shape.y, 2));
            if (dist < 2 * radius) {
                return false;
            }
        }
        return true;
    }

    std::pair<double, double> generateRandomCenter(double radius) {
        // Use .val to access parameter values
        std::uniform_real_distribution<double> dist_x(xmin.val + radius, xmax.val - radius);
        std::uniform_real_distribution<double> dist_y(ymin.val + radius, ymax.val - radius);
        return {dist_x(rng), dist_y(rng)};
    }

public:
    // Constructor now takes parameter references
    RandomShapeGenerator(param_t<double>& xmin_, param_t<double>& xmax_,
                         param_t<double>& ymin_, param_t<double>& ymax_,
                         param_t<int>& nx_, param_t<int>& ny_)
            : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_),
              nx(nx_), ny(ny_), rng(std::random_device{}()) {}

    // computeMergedPhi needs to be modified to use parameter values
    std::vector<std::vector<double>> computeMergedPhi(
            const std::vector<CircleShape>& shapes) {
        std::vector<std::vector<double>> merged_phi(
                ny.val, std::vector<double>(nx.val, std::numeric_limits<double>::max()));

        double dx = (xmax.val - xmin.val) / (nx.val - 1);
        double dy = (ymax.val - ymin.val) / (ny.val - 1);

        for (int i = 0; i < ny.val; ++i) {
            double y = ymin.val + i * dy;
            for (int j = 0; j < nx.val; ++j) {
                double x = xmin.val + j * dx;

                double min_phi = std::numeric_limits<double>::max();
                for (const auto& shape : shapes) {
                    double phi_val = shape.shape.phi(x, y);
                    min_phi = std::min(min_phi, phi_val);
                }
                merged_phi[i][j] = min_phi;
            }
        }

        return merged_phi;
    }

    std::vector<CircleShape> generateCircles(int num_circles, double radius) {
        std::vector<CircleShape> shapes;

        int max_attempts = 1000; // Prevent infinite loops
        int attempts;

        for (int i = 0; i < num_circles; ++i) {
            attempts = 0;
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
                shapes.emplace_back(x, y, radius);
            } else {
                std::cerr << "Warning: Could not place circle " << i + 1 << std::endl;
                break;
            }
        }

        return shapes;
    }

    void exportToFile(const std::vector<std::vector<double>>& phi, const std::string& filename) {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // Calculate dx and dy
        double dx = (xmax.val - xmin.val) / (nx.val - 1);
        double dy = (ymax.val - ymin.val) / (ny.val - 1);

        // Write data in format: x y phi(x,y)
        for (int i = 0; i < ny.val; ++i) {
            double y = ymin.val + i * dy;
            for (int j = 0; j < nx.val; ++j) {
                double x = xmin.val + j * dx;
                outFile << x << " " << y << " " << phi[i][j] << "\n";
            }
            // Add blank line between rows for gnuplot
            outFile << "\n";
        }

        outFile.close();
    }
};

#endif // RANDOM_SHAPE_GENERATOR_H