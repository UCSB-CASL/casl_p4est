#ifndef SIMPLEX3_MLS_VTK_H
#define SIMPLEX3_MLS_VTK_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "simplex3_mls.h"

class simplex3_mls_vtk
{
public:
  static void write_simplex_geometry(std::vector<simplex3_mls_t *>& simplices, std::string dir, std::string suffix);
};

#endif // SIMPLEX3_MLS_VTK_H
