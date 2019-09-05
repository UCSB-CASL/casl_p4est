#ifndef MY_P4EST_MULTIALLOY_WITH_FLUIDS_H
#define MY_P4EST_MULTIALLOY_WITH_FLUIDS_H

// From my_p4est_multialloy.h:
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_multialloy.h>
#include <src/my_p4est_navier_stokes.h>

// From my_p4est_navier_stokes.h:
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_save_load.h>

#include <src/casl_math.h>


class my_p4est_multialloy_with_fluids
{
public:
  my_p4est_multialloy_with_fluids();
};

#endif // MY_P4EST_MULTIALLOY_WITH_FLUIDS_H
