# This CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/

cmake_minimum_required(VERSION 3.15)
project(casl_p4est)

set(CMAKE_CXX_STANDARD 11)

# Library directories.
link_directories(/usr/local/lib
        /usr/local/petsc/lib
        /usr/local/p4est-2.2/lib)

add_library(casl_p4est
        ### Multiple levelset integration ###
        src/mls_integration/simplex_utils.h
        src/mls_integration/simplex2_mls_l.h             src/mls_integration/simplex2_mls_l.cpp
        src/mls_integration/simplex2_mls_q.h             src/mls_integration/simplex2_mls_q.cpp
        src/mls_integration/simplex3_mls_l.h             src/mls_integration/simplex3_mls_l.cpp
        src/mls_integration/simplex3_mls_q.h             src/mls_integration/simplex3_mls_q.cpp
        src/mls_integration/cube2_mls.h                  src/mls_integration/cube2_mls.cpp
        src/mls_integration/cube2_mls_l.h                src/mls_integration/cube2_mls_l.cpp
        src/mls_integration/cube2_mls_q.h                src/mls_integration/cube2_mls_q.cpp
        src/mls_integration/cube3_mls.h                  src/mls_integration/cube3_mls.cpp
        src/mls_integration/cube3_mls_l.h                src/mls_integration/cube3_mls_l.cpp
        src/mls_integration/cube3_mls_q.h                src/mls_integration/cube3_mls_q.cpp
        src/mls_integration/vtk/simplex2_mls_l_vtk.h     src/mls_integration/vtk/simplex2_mls_l_vtk.cpp
        src/mls_integration/vtk/simplex2_mls_q_vtk.h     src/mls_integration/vtk/simplex2_mls_q_vtk.cpp
        src/mls_integration/vtk/simplex3_mls_l_vtk.h     src/mls_integration/vtk/simplex3_mls_l_vtk.cpp
        src/mls_integration/vtk/simplex3_mls_q_vtk.h     src/mls_integration/vtk/simplex3_mls_q_vtk.cpp

        ### Common functionality ###
        src/parameter_list.h         src/parameter_list.cpp
        src/casl_math.h              src/casl_math.cpp
        src/Parser.h                 src/Parser.cpp
        src/matrix.h                 src/matrix.cpp
        src/petsc_logging.h          src/petsc_logging.cpp
        src/cube2.h                  src/cube2.cpp
        src/cube3.h                  src/cube3.cpp
        src/point2.h                 src/point2.cpp
        src/point3.h                 src/point3.cpp
        src/simplex2.h               src/simplex2.cpp
        src/types.h
        src/petsc_compatibility.h

        ### The 2D Components ###
        src/my_p4est_macros.h
        src/my_p4est_shapes.h
        src/my_p4est_biofilm.h                           src/my_p4est_biofilm.cpp
		src/my_p4est_biomolecules.h						 src/my_p4est_biomolecules.cpp
        src/my_p4est_cell_neighbors.h                    src/my_p4est_cell_neighbors.cpp
        src/my_p4est_epitaxy.h                           src/my_p4est_epitaxy.cpp
        src/my_p4est_faces.h                             src/my_p4est_faces.cpp
		src/my_p4est_general_poisson_nodes_mls_solver.cpp
        src/my_p4est_hierarchy.h                         src/my_p4est_hierarchy.cpp
        src/my_p4est_integration_mls.h                   src/my_p4est_integration_mls.cpp
        src/my_p4est_interpolation.h                     src/my_p4est_interpolation.cpp
        src/my_p4est_interpolation_cells.h               src/my_p4est_interpolation_cells.cpp
        src/my_p4est_interpolation_faces.h               src/my_p4est_interpolation_faces.cpp
        src/my_p4est_interpolation_nodes.h               src/my_p4est_interpolation_nodes.cpp
        src/my_p4est_interpolation_nodes_local.h         src/my_p4est_interpolation_nodes_local.cpp
        src/my_p4est_level_set.h                         src/my_p4est_level_set.cpp
        src/my_p4est_level_set_cells.h                   src/my_p4est_level_set_cells.cpp
        src/my_p4est_level_set_faces.h                   src/my_p4est_level_set_faces.cpp
        src/my_p4est_multialloy.h                        src/my_p4est_multialloy.cpp
        src/my_p4est_navier_stokes.h                     src/my_p4est_navier_stokes.cpp
        src/my_p4est_node_neighbors.h                    src/my_p4est_node_neighbors.cpp
        src/my_p4est_poisson_boltzmann_nodes.h           src/my_p4est_poisson_boltzmann_nodes.cpp
        src/my_p4est_poisson_cells.h                     src/my_p4est_poisson_cells.cpp
        src/my_p4est_poisson_faces.h                     src/my_p4est_poisson_faces.cpp
        src/my_p4est_poisson_jump_nodes_extended.h       src/my_p4est_poisson_jump_nodes_extended.cpp
        src/my_p4est_poisson_jump_nodes_voronoi.h        src/my_p4est_poisson_jump_nodes_voronoi.cpp
        src/my_p4est_poisson_jump_voronoi_block.h        src/my_p4est_poisson_jump_voronoi_block.cpp
        src/my_p4est_poisson_nodes.h                     src/my_p4est_poisson_nodes.cpp
        src/my_p4est_poisson_nodes_mls.h                 src/my_p4est_poisson_nodes_mls.cpp
        src/my_p4est_poisson_nodes_multialloy.h          src/my_p4est_poisson_nodes_multialloy.cpp
        src/my_p4est_quad_neighbor_nodes_of_node.h       src/my_p4est_quad_neighbor_nodes_of_node.cpp
        src/my_p4est_refine_coarsen.h                    src/my_p4est_refine_coarsen.cpp
        src/my_p4est_scft.h                              src/my_p4est_scft.cpp
        src/my_p4est_semi_lagrangian.h                   src/my_p4est_semi_lagrangian.cpp
        src/my_p4est_solve_lsqr.h                        src/my_p4est_solve_lsqr.cpp
        src/my_p4est_trajectory_of_point.h               src/my_p4est_trajectory_of_point.cpp
        src/my_p4est_utils.h                             src/my_p4est_utils.cpp
        src/my_p4est_vtk.h                               src/my_p4est_vtk.cpp
        src/my_p4est_log_wrappers.h                      src/my_p4est_log_wrappers.c
        src/my_p4est_nodes.h                             src/my_p4est_nodes.c
        src/my_p4est_tools.h                             src/my_p4est_tools.c
        src/voronoi2D.h                                  src/voronoi2D.cpp
        )

find_package(MPI REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

target_link_libraries(casl_p4est PUBLIC
        mpi
        petsc
        p4est
        sc
        voro++
        ${Boost_FILESYSTEM_LIBRARY})

target_include_directories(casl_p4est PUBLIC
        .
        /usr/local/include
        /usr/local/petsc/include
        /usr/local/p4est-2.2/include
        /usr/local/include/voro++
        ${Boost_INCLUDE_DIR})