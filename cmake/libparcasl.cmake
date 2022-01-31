# This file must be included in the example's CMakeLists.txt to pull the library source and header files.
# See https://stackoverflow.com/questions/17228677/how-to-include-an-additional-cmakelists-txt for more details.

# Contents extracted from qmake/libparcasl.pri.
# Multiple level-set integration.
list( APPEND HEADERS
		${PARCASL}src/mls_integration/simplex_utils.h
		${PARCASL}src/mls_integration/simplex2_mls_l.h
		${PARCASL}src/mls_integration/simplex2_mls_q.h
		${PARCASL}src/mls_integration/simplex3_mls_l.h
		${PARCASL}src/mls_integration/simplex3_mls_q.h
		${PARCASL}src/mls_integration/cube2_mls.h
		${PARCASL}src/mls_integration/cube2_mls_l.h
		${PARCASL}src/mls_integration/cube2_mls_q.h
		${PARCASL}src/mls_integration/cube3_mls.h
		${PARCASL}src/mls_integration/cube3_mls_l.h
		${PARCASL}src/mls_integration/cube3_mls_q.h
		${PARCASL}src/mls_integration/vtk/simplex2_mls_l_vtk.h
		${PARCASL}src/mls_integration/vtk/simplex2_mls_q_vtk.h
		${PARCASL}src/mls_integration/vtk/simplex3_mls_l_vtk.h
		${PARCASL}src/mls_integration/vtk/simplex3_mls_q_vtk.h )

list( APPEND SOURCES
		${PARCASL}src/mls_integration/simplex2_mls_l.cpp
		${PARCASL}src/mls_integration/simplex2_mls_q.cpp
		${PARCASL}src/mls_integration/simplex3_mls_l.cpp
		${PARCASL}src/mls_integration/simplex3_mls_q.cpp
		${PARCASL}src/mls_integration/cube2_mls.cpp
		${PARCASL}src/mls_integration/cube2_mls_l.cpp
		${PARCASL}src/mls_integration/cube2_mls_q.cpp
		${PARCASL}src/mls_integration/cube3_mls.cpp
		${PARCASL}src/mls_integration/cube3_mls_l.cpp
		${PARCASL}src/mls_integration/cube3_mls_q.cpp
		${PARCASL}src/mls_integration/vtk/simplex2_mls_l_vtk.cpp
		${PARCASL}src/mls_integration/vtk/simplex2_mls_q_vtk.cpp
		${PARCASL}src/mls_integration/vtk/simplex3_mls_l_vtk.cpp
		${PARCASL}src/mls_integration/vtk/simplex3_mls_q_vtk.cpp )

# Other common stuff.
list( APPEND HEADERS
		${PARCASL}src/parameter_list.h
		${PARCASL}src/casl_math.h
		${PARCASL}src/Parser.h
		${PARCASL}src/matrix.h
		${PARCASL}src/petsc_logging.h
		${PARCASL}src/cube2.h
		${PARCASL}src/cube3.h
		${PARCASL}src/point2.h
		${PARCASL}src/point3.h
		${PARCASL}src/simplex2.h
		${PARCASL}src/types.h
		${PARCASL}src/petsc_compatibility.h
		${PARCASL}src/casl_geometry.h )

list( APPEND SOURCES
		${PARCASL}src/parameter_list.cpp
		${PARCASL}src/casl_math.cpp
		${PARCASL}src/Parser.cpp
		${PARCASL}src/matrix.cpp
		${PARCASL}src/petsc_logging.cpp
		${PARCASL}src/cube2.cpp
		${PARCASL}src/cube3.cpp
		${PARCASL}src/point2.cpp
		${PARCASL}src/point3.cpp
		${PARCASL}src/simplex2.cpp )

# For dimension based compilation, we must create a Run Configuration.  Go to CLion -> Preferences
# -> Build, Execution, Deployment -> CMake.  Create a profile (debug or release) and add to "CMake options:" the
# following: -DDIMENSION=2d, for 2D, or -DDIMENSION=3d for 3D.
# To enable machine learning, provide the key-value pair -DENABLE_ML=1

# 2D case.
if( DIMENSION MATCHES 2d )
	# Epitaxy.
	list( APPEND HEADERS ${PARCASL}src/my_p4est_epitaxy.h )
	list( APPEND SOURCES ${PARCASL}src/my_p4est_epitaxy.cpp )

	# Machine learning.
	if( ENABLE_ML MATCHES 1 )
		list( APPEND HEADERS
				${PARCASL}src/my_p4est_semi_lagrangian_ml.h
				${PARCASL}src/my_p4est_curvature_ml.h )

		list( APPEND SOURCES
				${PARCASL}src/my_p4est_semi_lagrangian_ml.cpp
				${PARCASL}src/my_p4est_curvature_ml.cpp )
	endif()

	# Others.
	list( APPEND HEADERS
			${PARCASL}src/my_p4est_biofilm.h
			${PARCASL}src/my_p4est_biomolecules.h
			${PARCASL}src/my_p4est_cell_neighbors.h
			${PARCASL}src/my_p4est_faces.h
			${PARCASL}src/my_p4est_general_poisson_nodes_mls_solver.h
			${PARCASL}src/my_p4est_grid_aligned_extension.h
			${PARCASL}src/my_p4est_hierarchy.h
			${PARCASL}src/my_p4est_integration_mls.h
			${PARCASL}src/my_p4est_interface_manager.h
			${PARCASL}src/my_p4est_interpolation.h
			${PARCASL}src/my_p4est_interpolation_cells.h
			${PARCASL}src/my_p4est_interpolation_faces.h
			${PARCASL}src/my_p4est_interpolation_nodes.h
			${PARCASL}src/my_p4est_interpolation_nodes_local.h
			${PARCASL}src/my_p4est_level_set.h
			${PARCASL}src/my_p4est_level_set_cells.h
			${PARCASL}src/my_p4est_level_set_faces.h
			${PARCASL}src/my_p4est_multialloy.h
			${PARCASL}src/my_p4est_navier_stokes.h
			${PARCASL}src/my_p4est_node_neighbors.h
			${PARCASL}src/my_p4est_poisson_boltzmann_nodes.h
			${PARCASL}src/my_p4est_poisson_cells.h
			${PARCASL}src/my_p4est_poisson_faces.h
			${PARCASL}src/my_p4est_poisson_jump_cells.h
			${PARCASL}src/my_p4est_poisson_jump_cells_fv.h
			${PARCASL}src/my_p4est_poisson_jump_cells_xgfm.h
			${PARCASL}src/my_p4est_poisson_jump_faces.h
			${PARCASL}src/my_p4est_poisson_jump_faces_xgfm.h
			${PARCASL}src/my_p4est_poisson_jump_nodes_extended.h
			${PARCASL}src/my_p4est_poisson_jump_nodes_voronoi.h
			${PARCASL}src/my_p4est_poisson_jump_voronoi_block.h
			${PARCASL}src/my_p4est_poisson_nodes.h
			${PARCASL}src/my_p4est_poisson_nodes_mls.h
			${PARCASL}src/my_p4est_poisson_nodes_multialloy.h
			${PARCASL}src/my_p4est_quad_neighbor_nodes_of_node.h
			${PARCASL}src/my_p4est_refine_coarsen.h
			${PARCASL}src/my_p4est_save_load.h
			${PARCASL}src/my_p4est_scft.h
			${PARCASL}src/my_p4est_semi_lagrangian.h
			${PARCASL}src/my_p4est_shs_channel.h
			${PARCASL}src/my_p4est_solve_lsqr.h
			${PARCASL}src/my_p4est_surfactant.h
			${PARCASL}src/my_p4est_trajectory_of_point.h
			${PARCASL}src/my_p4est_utils.h
			${PARCASL}src/my_p4est_vtk.h
			${PARCASL}src/my_p4est_log_wrappers.h
			${PARCASL}src/my_p4est_nodes.h
			${PARCASL}src/my_p4est_tools.h
			${PARCASL}src/my_p4est_two_phase_flows.h
#			${PARCASL}src/my_p4est_xgfm_cells.h
			${PARCASL}src/voronoi2D.h
			${PARCASL}src/my_p4est_fast_sweeping.h
			${PARCASL}src/my_p4est_nodes_along_interface.h )

	list( APPEND SOURCES
			${PARCASL}src/my_p4est_biofilm.cpp
			${PARCASL}src/my_p4est_biomolecules.cpp
			${PARCASL}src/my_p4est_cell_neighbors.cpp
			${PARCASL}src/my_p4est_faces.cpp
			${PARCASL}src/my_p4est_general_poisson_nodes_mls_solver.cpp
			${PARCASL}src/my_p4est_grid_aligned_extension.cpp
			${PARCASL}src/my_p4est_hierarchy.cpp
			${PARCASL}src/my_p4est_integration_mls.cpp
			${PARCASL}src/my_p4est_interface_manager.cpp
			${PARCASL}src/my_p4est_interpolation.cpp
			${PARCASL}src/my_p4est_interpolation_cells.cpp
			${PARCASL}src/my_p4est_interpolation_faces.cpp
			${PARCASL}src/my_p4est_interpolation_nodes.cpp
			${PARCASL}src/my_p4est_interpolation_nodes_local.cpp
			${PARCASL}src/my_p4est_level_set.cpp
			${PARCASL}src/my_p4est_level_set_cells.cpp
			${PARCASL}src/my_p4est_level_set_faces.cpp
			${PARCASL}src/my_p4est_multialloy.cpp
			${PARCASL}src/my_p4est_navier_stokes.cpp
			${PARCASL}src/my_p4est_node_neighbors.cpp
			${PARCASL}src/my_p4est_poisson_boltzmann_nodes.cpp
			${PARCASL}src/my_p4est_poisson_cells.cpp
			${PARCASL}src/my_p4est_poisson_faces.cpp
			${PARCASL}src/my_p4est_poisson_jump_cells.cpp
			${PARCASL}src/my_p4est_poisson_jump_cells_fv.cpp
			${PARCASL}src/my_p4est_poisson_jump_cells_xgfm.cpp
			${PARCASL}src/my_p4est_poisson_jump_faces.cpp
			${PARCASL}src/my_p4est_poisson_jump_faces_xgfm.cpp
			${PARCASL}src/my_p4est_poisson_jump_nodes_extended.cpp
			${PARCASL}src/my_p4est_poisson_jump_nodes_voronoi.cpp
			${PARCASL}src/my_p4est_poisson_jump_voronoi_block.cpp
			${PARCASL}src/my_p4est_poisson_nodes.cpp
			${PARCASL}src/my_p4est_poisson_nodes_mls.cpp
			${PARCASL}src/my_p4est_poisson_nodes_multialloy.cpp
			${PARCASL}src/my_p4est_quad_neighbor_nodes_of_node.cpp
			${PARCASL}src/my_p4est_refine_coarsen.cpp
			${PARCASL}src/my_p4est_save_load.cpp
			${PARCASL}src/my_p4est_scft.cpp
			${PARCASL}src/my_p4est_semi_lagrangian.cpp
			${PARCASL}src/my_p4est_solve_lsqr.cpp
			${PARCASL}src/my_p4est_surfactant.cpp
			${PARCASL}src/my_p4est_trajectory_of_point.cpp
			${PARCASL}src/my_p4est_utils.cpp
			${PARCASL}src/my_p4est_vtk.cpp
			${PARCASL}src/my_p4est_log_wrappers.c
			${PARCASL}src/my_p4est_nodes.c
			${PARCASL}src/my_p4est_tools.c
			${PARCASL}src/my_p4est_two_phase_flows.cpp
#			${PARCASL}src/my_p4est_xgfm_cells.cpp
			${PARCASL}src/voronoi2D.cpp
			${PARCASL}src/my_p4est_fast_sweeping.cpp
			${PARCASL}src/my_p4est_nodes_along_interface.cpp )

	# headers-only stuff
	list( APPEND HEADERS
			${PARCASL}src/my_p4est_macros.h
			${PARCASL}src/my_p4est_shapes.h )
endif()

# 3D case.
if( DIMENSION MATCHES 3d )

	# Machine learning.
	if( ENABLE_ML MATCHES 1 )
		list( APPEND HEADERS
				${PARCASL}src/my_p8est_semi_lagrangian_ml.h
				${PARCASL}src/my_p8est_curvature_ml.h )

		list( APPEND SOURCES
				${PARCASL}src/my_p8est_semi_lagrangian_ml.cpp
				${PARCASL}src/my_p8est_curvature_ml.cpp )
	endif()

	# Others.
	list( APPEND HEADERS
			${PARCASL}src/my_p8est_biofilm.h
			${PARCASL}src/my_p8est_biomolecules.h
			${PARCASL}src/my_p8est_cell_neighbors.h
			${PARCASL}src/my_p8est_faces.h
			${PARCASL}src/my_p8est_general_poisson_nodes_mls_solver.h
			${PARCASL}src/my_p8est_hierarchy.h
			${PARCASL}src/my_p8est_integration_mls.h
			${PARCASL}src/my_p8est_interface_manager.h
			${PARCASL}src/my_p8est_interpolation.h
			${PARCASL}src/my_p8est_interpolation_cells.h
			${PARCASL}src/my_p8est_interpolation_faces.h
			${PARCASL}src/my_p8est_interpolation_nodes.h
			${PARCASL}src/my_p8est_interpolation_nodes_local.h
			${PARCASL}src/my_p8est_level_set.h
			${PARCASL}src/my_p8est_level_set_cells.h
			${PARCASL}src/my_p8est_level_set_faces.h
			#  ${PARCASL}src/my_p8est_multialloy.h
			${PARCASL}src/my_p8est_navier_stokes.h
			${PARCASL}src/my_p8est_node_neighbors.h
			${PARCASL}src/my_p8est_poisson_boltzmann_nodes.h
			${PARCASL}src/my_p8est_poisson_cells.h
			${PARCASL}src/my_p8est_poisson_faces.h
			${PARCASL}src/my_p8est_poisson_jump_cells.h
			${PARCASL}src/my_p8est_poisson_jump_cells_fv.h
			${PARCASL}src/my_p8est_poisson_jump_cells_xgfm.h
			${PARCASL}src/my_p8est_poisson_jump_faces.h
			${PARCASL}src/my_p8est_poisson_jump_faces_xgfm.h
			${PARCASL}src/my_p8est_poisson_jump_nodes_extended.h
			${PARCASL}src/my_p8est_poisson_jump_nodes_voronoi.h
			${PARCASL}src/my_p8est_poisson_jump_voronoi_block.h
			${PARCASL}src/my_p8est_poisson_nodes.h
			${PARCASL}src/my_p8est_poisson_nodes_mls.h
			${PARCASL}src/my_p8est_poisson_nodes_multialloy.h
			${PARCASL}src/my_p8est_quad_neighbor_nodes_of_node.h
			${PARCASL}src/my_p8est_refine_coarsen.h
			${PARCASL}src/my_p8est_save_load.h
			#  ${PARCASL}src/my_p8est_scft.h
			${PARCASL}src/my_p8est_semi_lagrangian.h
			${PARCASL}src/my_p8est_solve_lsqr.h
			${PARCASL}src/my_p8est_surfactant.h
			${PARCASL}src/my_p8est_shs_channel.h
			${PARCASL}src/my_p8est_trajectory_of_point.h
			${PARCASL}src/my_p8est_utils.h
			${PARCASL}src/my_p8est_vtk.h
			${PARCASL}src/my_p8est_log_wrappers.h
			${PARCASL}src/my_p8est_nodes.h
			${PARCASL}src/my_p8est_tools.h
			${PARCASL}src/my_p8est_two_phase_flows.h
#			${PARCASL}src/my_p8est_xgfm_cells.h
			${PARCASL}src/voronoi3D.h
			${PARCASL}src/my_p8est_fast_sweeping.h
			${PARCASL}src/my_p8est_nodes_along_interface.h )

	list( APPEND SOURCES
			${PARCASL}src/my_p8est_biofilm.cpp
			${PARCASL}src/my_p8est_biomolecules.cpp
			${PARCASL}src/my_p8est_cell_neighbors.cpp
			${PARCASL}src/my_p8est_faces.cpp
			${PARCASL}src/my_p8est_general_poisson_nodes_mls_solver.cpp
			${PARCASL}src/my_p8est_hierarchy.cpp
			${PARCASL}src/my_p8est_integration_mls.cpp
			${PARCASL}src/my_p8est_interface_manager.cpp
			${PARCASL}src/my_p8est_interpolation.cpp
			${PARCASL}src/my_p8est_interpolation_cells.cpp
			${PARCASL}src/my_p8est_interpolation_faces.cpp
			${PARCASL}src/my_p8est_interpolation_nodes.cpp
			${PARCASL}src/my_p8est_interpolation_nodes_local.cpp
			${PARCASL}src/my_p8est_level_set.cpp
			${PARCASL}src/my_p8est_level_set_cells.cpp
			${PARCASL}src/my_p8est_level_set_faces.cpp
			#  ${PARCASL}src/my_p8est_multialloy.cpp
			${PARCASL}src/my_p8est_navier_stokes.cpp
			${PARCASL}src/my_p8est_node_neighbors.cpp
			${PARCASL}src/my_p8est_poisson_boltzmann_nodes.cpp
			${PARCASL}src/my_p8est_poisson_cells.cpp
			${PARCASL}src/my_p8est_poisson_faces.cpp
			${PARCASL}src/my_p8est_poisson_jump_cells.cpp
			${PARCASL}src/my_p8est_poisson_jump_cells_fv.cpp
			${PARCASL}src/my_p8est_poisson_jump_cells_xgfm.cpp
			${PARCASL}src/my_p8est_poisson_jump_faces.cpp
			${PARCASL}src/my_p8est_poisson_jump_faces_xgfm.cpp
			${PARCASL}src/my_p8est_poisson_jump_nodes_extended.cpp
			${PARCASL}src/my_p8est_poisson_jump_nodes_voronoi.cpp
			${PARCASL}src/my_p8est_poisson_jump_voronoi_block.cpp
			${PARCASL}src/my_p8est_poisson_nodes.cpp
			${PARCASL}src/my_p8est_poisson_nodes_mls.cpp
			${PARCASL}src/my_p8est_poisson_nodes_multialloy.cpp
			${PARCASL}src/my_p8est_quad_neighbor_nodes_of_node.cpp
			${PARCASL}src/my_p8est_refine_coarsen.cpp
			${PARCASL}src/my_p8est_save_load.cpp
			#  ${PARCASL}src/my_p8est_scft.cpp
			${PARCASL}src/my_p8est_semi_lagrangian.cpp
			${PARCASL}src/my_p8est_solve_lsqr.cpp
			${PARCASL}src/my_p8est_surfactant.cpp
			${PARCASL}src/my_p8est_trajectory_of_point.cpp
			${PARCASL}src/my_p8est_utils.cpp
			${PARCASL}src/my_p8est_vtk.cpp
			${PARCASL}src/my_p8est_log_wrappers.c
			${PARCASL}src/my_p8est_nodes.c
			${PARCASL}src/my_p8est_tools.c
			${PARCASL}src/my_p8est_two_phase_flows.cpp
#			${PARCASL}src/my_p8est_xgfm_cells.cpp
			${PARCASL}src/voronoi3D.cpp
			${PARCASL}src/my_p8est_fast_sweeping.cpp
			${PARCASL}src/my_p8est_nodes_along_interface.cpp )

	# Headers-only stuff.
	list( APPEND HEADERS
			${PARCASL}src/my_p8est_macros.h
			${PARCASL}src/my_p8est_shapes.h )
endif()