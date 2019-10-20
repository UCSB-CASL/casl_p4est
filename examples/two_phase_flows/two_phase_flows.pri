
CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d.cpp
}

# multiple level-set integration
HEADERS += \
  $$PARCASL/src/mls_integration/simplex_utils.h \
  $$PARCASL/src/mls_integration/simplex2_mls_l.h \
  $$PARCASL/src/mls_integration/simplex2_mls_q.h \
  $$PARCASL/src/mls_integration/simplex3_mls_l.h \
  $$PARCASL/src/mls_integration/simplex3_mls_q.h \
  $$PARCASL/src/mls_integration/cube2_mls.h \
  $$PARCASL/src/mls_integration/cube2_mls_l.h \
  $$PARCASL/src/mls_integration/cube2_mls_q.h \
  $$PARCASL/src/mls_integration/cube3_mls.h \
  $$PARCASL/src/mls_integration/cube3_mls_l.h \
  $$PARCASL/src/mls_integration/cube3_mls_q.h \
  $$PARCASL/src/mls_integration/vtk/simplex2_mls_l_vtk.h \
  $$PARCASL/src/mls_integration/vtk/simplex2_mls_q_vtk.h \
  $$PARCASL/src/mls_integration/vtk/simplex3_mls_l_vtk.h \
  $$PARCASL/src/mls_integration/vtk/simplex3_mls_q_vtk.h

SOURCES += \
  $$PARCASL/src/mls_integration/simplex2_mls_l.cpp \
  $$PARCASL/src/mls_integration/simplex2_mls_q.cpp \
  $$PARCASL/src/mls_integration/simplex3_mls_l.cpp \
  $$PARCASL/src/mls_integration/simplex3_mls_q.cpp \
  $$PARCASL/src/mls_integration/cube2_mls.cpp \
  $$PARCASL/src/mls_integration/cube2_mls_l.cpp \
  $$PARCASL/src/mls_integration/cube2_mls_q.cpp \
  $$PARCASL/src/mls_integration/cube3_mls.cpp \
  $$PARCASL/src/mls_integration/cube3_mls_l.cpp \
  $$PARCASL/src/mls_integration/cube3_mls_q.cpp \
  $$PARCASL/src/mls_integration/vtk/simplex2_mls_l_vtk.cpp \
  $$PARCASL/src/mls_integration/vtk/simplex2_mls_q_vtk.cpp \
  $$PARCASL/src/mls_integration/vtk/simplex3_mls_l_vtk.cpp \
  $$PARCASL/src/mls_integration/vtk/simplex3_mls_q_vtk.cpp

CONFIG(2d, 2d|3d): {
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
  $$PARCASL/src/my_p4est_poisson_cells.cpp \
  $$PARCASL/src/my_p4est_poisson_faces.cpp \
  $$PARCASL/src/my_p4est_poisson_boltzmann_nodes.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_voronoi.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_voronoi_block.cpp \
  $$PARCASL/src/my_p4est_poisson_jump_nodes_extended.cpp \
  $$PARCASL/src/my_p4est_poisson_nodes.cpp \
  $$PARCASL/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
  $$PARCASL/src/my_p4est_refine_coarsen.cpp \
  $$PARCASL/src/my_p4est_semi_lagrangian.cpp \
  $$PARCASL/src/my_p4est_solve_lsqr.cpp \
  $$PARCASL/src/my_p4est_trajectory_of_point.cpp \
  $$PARCASL/src/my_p4est_utils.cpp \
  $$PARCASL/src/my_p4est_log_wrappers.c \
  $$PARCASL/src/my_p4est_nodes.c \
  $$PARCASL/src/my_p4est_save_load.cpp \
  $$PARCASL/src/my_p4est_tools.c \
  $$PARCASL/src/my_p4est_two_phase_flows.cpp \
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
  $$PARCASL/src/my_p8est_solve_lsqr.cpp \
  $$PARCASL/src/my_p8est_trajectory_of_point.cpp \
  $$PARCASL/src/my_p8est_utils.cpp \
  $$PARCASL/src/my_p8est_log_wrappers.c \
  $$PARCASL/src/my_p8est_nodes.c \
  $$PARCASL/src/my_p8est_save_load.cpp \
  $$PARCASL/src/my_p8est_tools.c \
  $$PARCASL/src/my_p8est_two_phase_flows.cpp \
  $$PARCASL/src/my_p8est_vtk.cpp \
  $$PARCASL/src/my_p4est_xgfm_cells.h \
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

