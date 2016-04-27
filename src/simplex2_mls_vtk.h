#ifndef SIMPLEX2_MLS_VTK_H
#define SIMPLEX2_MLS_VTK_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "simplex2_mls.h"

class simplex2_mls_vtk
{
public:
  static void write_simplex_geometry(std::vector<simplex2_mls_t *>& simplices, std::string dir, std::string suffix);
};

#endif // SIMPLEX2_MLS_VTK_H
