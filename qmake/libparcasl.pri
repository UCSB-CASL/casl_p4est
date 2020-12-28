# --------------------------------------------------------------
# list of all files in src to be built
# --------------------------------------------------------------
# other common stuff
HEADERS += \
  $$PARCASL/src/casl_math.h \
  $$PARCASL/src/Parser.h \
  $$PARCASL/src/matrix.h \
  $$PARCASL/src/petsc_logging.h \
  $$PARCASL/src/cube2.h \
  $$PARCASL/src/cube3.h \
  $$PARCASL/src/point2.h \
  $$PARCASL/src/point3.h \
  $$PARCASL/src/simplex2.h \
  $$PARCASL/src/types.h \
  $$PARCASL/src/petsc_compatibility.h

SOURCES += \
  $$PARCASL/src/casl_math.cpp \
  $$PARCASL/src/Parser.cpp \
  $$PARCASL/src/matrix.cpp \
  $$PARCASL/src/petsc_logging.cpp \
  $$PARCASL/src/cube2.cpp \
  $$PARCASL/src/cube3.cpp \
  $$PARCASL/src/point2.cpp \
  $$PARCASL/src/point3.cpp \
  $$PARCASL/src/simplex2.cpp

# dimension-specific stuff
CONFIG(2d, 2d|3d): {

# we add the epitaxy stuff only if boost links are provided (it is the only class that uses it)
exists($$BOOST_INCLUDES/boost/random.hpp):exists($$BOOST_INCLUDES/boost/random/normal_distribution.hpp){
HEADERS += $$PARCASL/src/my_p4est_epitaxy.h
SOURCES += $$PARCASL/src/my_p4est_epitaxy.cpp
}

HEADERS += \
  $$PARCASL/src/my_p4est_cell_neighbors.h \
  $$PARCASL/src/my_p4est_faces.h \
  $$PARCASL/src/my_p4est_hierarchy.h \
  $$PARCASL/src/my_p4est_interpolation.h \
  $$PARCASL/src/my_p4est_interpolation_cells.h \
  $$PARCASL/src/my_p4est_interpolation_faces.h \
  $$PARCASL/src/my_p4est_interpolation_nodes.h \
  $$PARCASL/src/my_p4est_level_set.h \
  $$PARCASL/src/my_p4est_level_set_cells.h \
  $$PARCASL/src/my_p4est_level_set_faces.h \
  $$PARCASL/src/my_p4est_navier_stokes.h \
  $$PARCASL/src/my_p4est_node_neighbors.h \
  $$PARCASL/src/my_p4est_poisson_boltzmann_nodes.h \
  $$PARCASL/src/my_p4est_poisson_cells.h \
  $$PARCASL/src/my_p4est_poisson_faces.h \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_extended.h \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_voronoi.h \
  $$PARCASL/src/my_p4est_poisson_jump_voronoi_block.h \
  $$PARCASL/src/my_p4est_poisson_nodes.h \
  $$PARCASL/src/my_p4est_quad_neighbor_nodes_of_node.h \
  $$PARCASL/src/my_p4est_refine_coarsen.h \
  $$PARCASL/src/my_p4est_semi_lagrangian.h \
  $$PARCASL/src/my_p4est_shs_channel.h \
  $$PARCASL/src/my_p4est_solve_lsqr.h \
  $$PARCASL/src/my_p4est_trajectory_of_point.h \
  $$PARCASL/src/my_p4est_utils.h \
  $$PARCASL/src/my_p4est_vtk.h \
  $$PARCASL/src/my_p4est_log_wrappers.h \
  $$PARCASL/src/my_p4est_nodes.h \
  $$PARCASL/src/my_p4est_tools.h \
  $$PARCASL/src/voronoi2D.h

SOURCES += \
  $$PARCASL/src/my_p4est_cell_neighbors.cpp \
  $$PARCASL/src/my_p4est_faces.cpp \
  $$PARCASL/src/my_p4est_hierarchy.cpp \
  $$PARCASL/src/my_p4est_interpolation.cpp \
  $$PARCASL/src/my_p4est_interpolation_cells.cpp \
  $$PARCASL/src/my_p4est_interpolation_faces.cpp \
  $$PARCASL/src/my_p4est_interpolation_nodes.cpp \
  $$PARCASL/src/my_p4est_level_set.cpp \
  $$PARCASL/src/my_p4est_level_set_cells.cpp \
  $$PARCASL/src/my_p4est_level_set_faces.cpp \
  $$PARCASL/src/my_p4est_navier_stokes.cpp \
  $$PARCASL/src/my_p4est_node_neighbors.cpp \
  $$PARCASL/src/my_p4est_poisson_boltzmann_nodes.cpp \
  $$PARCASL/src/my_p4est_poisson_cells.cpp \
  $$PARCASL/src/my_p4est_poisson_faces.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_extended.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_voronoi.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_voronoi_block.cpp \
  $$PARCASL/src/my_p4est_poisson_nodes.cpp \
  $$PARCASL/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
  $$PARCASL/src/my_p4est_refine_coarsen.cpp \
  $$PARCASL/src/my_p4est_semi_lagrangian.cpp \
  $$PARCASL/src/my_p4est_solve_lsqr.cpp \
  $$PARCASL/src/my_p4est_trajectory_of_point.cpp \
  $$PARCASL/src/my_p4est_utils.cpp \
  $$PARCASL/src/my_p4est_vtk.cpp \
  $$PARCASL/src/my_p4est_log_wrappers.c \
  $$PARCASL/src/my_p4est_nodes.c \
  $$PARCASL/src/my_p4est_tools.c \
  $$PARCASL/src/voronoi2D.cpp

# headers-only stuff
HEADERS += \
  $$PARCASL/src/my_p4est_macros.h \
  $$PARCASL/src/my_p4est_shapes.h
}

CONFIG(3d, 2d|3d): {
HEADERS += \
  $$PARCASL/src/my_p8est_cell_neighbors.h \
  $$PARCASL/src/my_p8est_faces.h \
  $$PARCASL/src/my_p8est_hierarchy.h \
  $$PARCASL/src/my_p8est_interpolation.h \
  $$PARCASL/src/my_p8est_interpolation_cells.h \
  $$PARCASL/src/my_p8est_interpolation_faces.h \
  $$PARCASL/src/my_p8est_interpolation_nodes.h \
  $$PARCASL/src/my_p8est_level_set.h \
  $$PARCASL/src/my_p8est_level_set_cells.h \
  $$PARCASL/src/my_p8est_level_set_faces.h \
  $$PARCASL/src/my_p8est_navier_stokes.h \
  $$PARCASL/src/my_p8est_node_neighbors.h \
  $$PARCASL/src/my_p8est_poisson_boltzmann_nodes.h \
  $$PARCASL/src/my_p8est_poisson_cells.h \
  $$PARCASL/src/my_p8est_poisson_faces.h \
  $$PARCASL/src/my_p8est_poisson_jump_nodes_extended.h \
  $$PARCASL/src/my_p8est_poisson_jump_nodes_voronoi.h \
  $$PARCASL/src/my_p8est_poisson_jump_voronoi_block.h \
  $$PARCASL/src/my_p8est_poisson_nodes.h \
  $$PARCASL/src/my_p8est_quad_neighbor_nodes_of_node.h \
  $$PARCASL/src/my_p8est_refine_coarsen.h \
  $$PARCASL/src/my_p8est_semi_lagrangian.h \
  $$PARCASL/src/my_p8est_solve_lsqr.h \
  $$PARCASL/src/my_p8est_shs_channel.h \
  $$PARCASL/src/my_p8est_trajectory_of_point.h \
  $$PARCASL/src/my_p8est_utils.h \
  $$PARCASL/src/my_p8est_vtk.h \
  $$PARCASL/src/my_p8est_log_wrappers.h \
  $$PARCASL/src/my_p8est_nodes.h \
  $$PARCASL/src/my_p8est_tools.h \
  $$PARCASL/src/voronoi3D.h

SOURCES += \
  $$PARCASL/src/my_p8est_cell_neighbors.cpp \
  $$PARCASL/src/my_p8est_faces.cpp \
  $$PARCASL/src/my_p8est_hierarchy.cpp \
  $$PARCASL/src/my_p8est_interpolation.cpp \
  $$PARCASL/src/my_p8est_interpolation_cells.cpp \
  $$PARCASL/src/my_p8est_interpolation_faces.cpp \
  $$PARCASL/src/my_p8est_interpolation_nodes.cpp \
  $$PARCASL/src/my_p8est_level_set.cpp \
  $$PARCASL/src/my_p8est_level_set_cells.cpp \
  $$PARCASL/src/my_p8est_level_set_faces.cpp \
  $$PARCASL/src/my_p8est_navier_stokes.cpp \
  $$PARCASL/src/my_p8est_node_neighbors.cpp \
  $$PARCASL/src/my_p8est_poisson_boltzmann_nodes.cpp \
  $$PARCASL/src/my_p8est_poisson_cells.cpp \
  $$PARCASL/src/my_p8est_poisson_faces.cpp \
  $$PARCASL/src/my_p8est_poisson_jump_nodes_extended.cpp \
  $$PARCASL/src/my_p8est_poisson_jump_nodes_voronoi.cpp \
  $$PARCASL/src/my_p8est_poisson_jump_voronoi_block.cpp \
  $$PARCASL/src/my_p8est_poisson_nodes.cpp \
  $$PARCASL/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
  $$PARCASL/src/my_p8est_refine_coarsen.cpp \
  $$PARCASL/src/my_p8est_semi_lagrangian.cpp \
  $$PARCASL/src/my_p8est_solve_lsqr.cpp \
  $$PARCASL/src/my_p8est_trajectory_of_point.cpp \
  $$PARCASL/src/my_p8est_utils.cpp \
  $$PARCASL/src/my_p8est_vtk.cpp \
  $$PARCASL/src/my_p8est_log_wrappers.c \
  $$PARCASL/src/my_p8est_nodes.c \
  $$PARCASL/src/my_p8est_tools.c \
  $$PARCASL/src/voronoi3D.cpp

# headers-only stuff
HEADERS += \
  $$PARCASL/src/my_p8est_macros.h \
  $$PARCASL/src/my_p8est_shapes.h
}
