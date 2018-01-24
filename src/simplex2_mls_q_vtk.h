#ifndef simplex2_mls_q_vtk_H
#define simplex2_mls_q_vtk_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "simplex2_mls_q.h"

class simplex2_mls_q_vtk
{
public:
  static void write_simplex_geometry(std::vector<simplex2_mls_q_t *>& simplices, std::string dir, std::string suffix);
};

#endif // simplex2_mls_q_vtk_H
