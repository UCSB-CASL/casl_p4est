#ifndef FIELDGENERATOR_H
#define FIELDGENERATOR_H

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>


//#include <mach/vm_statistics.h>
//#include <mach/mach_types.h>
//#include <mach/mach_init.h>
//#include <mach/mach_host.h>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_semi_lagrangian.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_levelset.h>
#include<src/my_p4est_semi_lagrangian.h>
#endif

#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>


using namespace std;

class FIeldGenerator
{



  };

#endif // STRESSTENSOR_H
