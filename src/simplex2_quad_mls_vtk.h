#ifndef SIMPLEX2_QUAD_MLS_VTK_H
#define SIMPLEX2_QUAD_MLS_VTK_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "simplex2_quad_mls.h"

class simplex2_quad_mls_vtk
{
public:
  static void write_simplex_geometry(std::vector<simplex2_quad_mls_t *>& simplices, std::string dir, std::string suffix);
};

#endif // SIMPLEX2_QUAD_MLS_VTK_H
