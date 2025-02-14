//
// Created by Faranak on 2/6/25.
//

#include <iostream>
#include "multi_circle_shapes.h"

int main(int argc, char* argv[]) {
    // Parameter list setup
    param_list_t pl;
    param_t<int>    px   (pl, 0, "px", "Periodicity in the x-direction (0/1)");
    param_t<int>    py   (pl, 0, "py", "Periodicity in the y-direction (0/1)");
    param_t<int>    nx   (pl, 1, "nx", "Number of trees in the x-direction");
    param_t<int>    ny   (pl, 1, "ny", "Number of trees in the y-direction");
    param_t<double> xmin (pl, -1, "xmin", "Box xmin");
    param_t<double> ymin (pl, -1, "ymin", "Box ymin");
    param_t<double> xmax (pl,  1, "xmax", "Box xmax");
    param_t<double> ymax (pl,  1, "ymax", "Box ymax");

    // Refinement parameters
    param_t<int> lmin(pl, 6, "lmin", "Min level of the tree");
    param_t<int> lmax(pl, 6, "lmax", "Max level of the tree");
    param_t<double> band(pl, 4.0, "band", "Width of uniform band around interfaces");
    param_t<bool> balance_grid(pl, 1, "balance_grid", "Enforce 1:2 ratio");

    // Grid and shape parameters
    param_t<int> num_circles(pl, 6, "num_circles", "Number of circles");
    param_t<double> circle_radius(pl, 0.25, "circle_radius", "Circle radius");

    // Initialize shape generator
    MultiCircleShapeGenerator generator(xmin, xmax, ymin, ymax, nx, ny, lmin, lmax);
    generator.generateCircles(num_circles.val, circle_radius.val);

    // Export results with current iteration (0 in this case)
    std::string num_cs = std::to_string(num_circles.val);
    std::string filename = "merged_phi_" + num_cs + ".dat";
    generator.exportToFile(filename, 0);
    return 0;
}