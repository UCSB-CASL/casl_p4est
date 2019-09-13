# --------------------------------------------------------------
# load config files based on environment
# --------------------------------------------------------------
CONFIG(linux, linux|macx|stampede): {
  include(linux.pri)
}

CONFIG(macx, linux|macx|stampede): {
  include(macx.pri)
}

CONFIG(stampede, linux|macx|stampede): {
  include(stampede.pri)
}

# --------------------------------------------------------------
# list of all files in src to be built
# --------------------------------------------------------------
CONFIG(2d, 2d|3d): {
SOURCES += \
  $$PARCASL/src/my_p4est_bialloy.cpp \
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
  $$PARCASL/src/my_p4est_ns_free_surface.cpp \
  $$PARCASL/src/my_p4est_poisson_cells.cpp \
  $$PARCASL/src/my_p4est_poisson_faces.cpp \
  $$PARCASL/src/my_p4est_poisson_boltzmann_nodes.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_voronoi.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_voronoi_block.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_extended.cpp \
  $$PARCASL/src/my_p4est_poisson_nodes.cpp \
  $$PARCASL/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
  $$PARCASL/src/my_p4est_refine_coarsen.cpp \
  $$PARCASL/src/my_p4est_save_load.cpp \
  $$PARCASL/src/my_p4est_semi_lagrangian.cpp \
  $$PARCASL/src/my_p4est_solve_lsqr.cpp \
  $$PARCASL/src/my_p4est_trajectory_of_point.cpp \
#  $$PARCASL/src/my_p4est_two_phase_flows.cpp \
  $$PARCASL/src/my_p4est_utils.cpp \
#  $$PARCASL/src/my_p4est_epitaxy.cpp \
  $$PARCASL/src/my_p4est_log_wrappers.c \
  $$PARCASL/src/my_p4est_nodes.c \
  $$PARCASL/src/my_p4est_tools.c \
  $$PARCASL/src/my_p4est_vtk.cpp \
  $$PARCASL/src/my_p4est_xgfm_cells.cpp \
  $$PARCASL/src/casl_math.cpp \
  $$PARCASL/src/Parser.cpp \
  $$PARCASL/src/cube2.cpp \
  $$PARCASL/src/matrix.cpp \
  $$PARCASL/src/petsc_logging.cpp \
  $$PARCASL/src/point2.cpp \
  $$PARCASL/src/simplex2.cpp \
  $$PARCASL/src/voronoi2D.cpp
}


CONFIG(3d, 2d|3d): {
SOURCES += \
  $$PARCASL/src/my_p8est_bialloy.cpp \
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
  $$PARCASL/src/my_p8est_ns_free_surface.cpp \
  $$PARCASL/src/my_p8est_poisson_cells.cpp \
  $$PARCASL/src/my_p8est_poisson_faces.cpp \
  $$PARCASL/src/my_p8est_poisson_boltzmann_nodes.cpp \
  $$PARCASL/src/my_p8est_poisson_jump_nodes_voronoi.cpp \
  $$PARCASL/src/my_p8est_poisson_jump_voronoi_block.cpp \
  $$PARCASL/src/my_p8est_poisson_jump_nodes_extended.cpp \
  $$PARCASL/src/my_p8est_poisson_nodes.cpp \
  $$PARCASL/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
  $$PARCASL/src/my_p8est_refine_coarsen.cpp \
  $$PARCASL/src/my_p8est_semi_lagrangian.cpp \
  $$PARCASL/src/my_p8est_save_load.cpp \
  $$PARCASL/src/my_p8est_solve_lsqr.cpp \
  $$PARCASL/src/my_p8est_trajectory_of_point.cpp \
  $$PARCASL/src/my_p8est_two_phase_flows.cpp \
  $$PARCASL/src/my_p8est_utils.cpp \
  $$PARCASL/src/my_p8est_log_wrappers.c \
  $$PARCASL/src/my_p8est_nodes.c \
  $$PARCASL/src/my_p8est_tools.c \
  $$PARCASL/src/my_p8est_vtk.cpp \
  $$PARCASL/src/my_p8est_xgfm_cells.cpp \
  $$PARCASL/src/casl_math.cpp \
  $$PARCASL/src/Parser.cpp \
  $$PARCASL/src/cube2.cpp \
  $$PARCASL/src/cube3.cpp \
  $$PARCASL/src/matrix.cpp \
  $$PARCASL/src/petsc_logging.cpp \
  $$PARCASL/src/point2.cpp \
  $$PARCASL/src/point3.cpp \
  $$PARCASL/src/simplex2.cpp \
  $$PARCASL/src/voronoi3D.cpp
}
