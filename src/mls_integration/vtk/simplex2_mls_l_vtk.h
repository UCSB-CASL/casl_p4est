#ifndef simplex2_mls_l_VTK_H
#define simplex2_mls_l_VTK_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "../simplex2_mls_l.h"

class simplex2_mls_l_vtk
{
public:
  static void write_simplex_geometry(std::vector<simplex2_mls_l_t *>& simplices, std::string dir, std::string suffix);
};

#endif // simplex2_mls_l_VTK_H
