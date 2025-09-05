cmake_minimum_required(VERSION 3.10)
project(diffusion_nodes_mls)
set(CMAKE_CXX_STANDARD 11)

# ---- Build profile echo
message(STATUS "PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
message(STATUS "CMAKE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\"")

# ---- Locate repo root (two dirs above this example)
get_filename_component(CASL_P4EST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(CASL_P4EST_ROOT "${CASL_P4EST_ROOT}" DIRECTORY)
set(CASL_P4EST_ROOT "${CASL_P4EST_ROOT}/")

message(STATUS "CASL_P4EST_ROOT: ${CASL_P4EST_ROOT}")

################################################## Library Configuration ###################################################
if( CMAKE_BUILD_TYPE MATCHES Debug )
    message( "******* CASL CMAKE IN DEBUG MODE *******" )
    set( PETSC_DIR $ENV{HOME}/CASL/casl_code_base/external_libraries/petsc_debug )
    set( P4EST_DIR $ENV{HOME}/CASL/casl_code_base/external_libraries/p4est_debug )
    set( VORO_DIR  $ENV{HOME}/CASL/casl_code_base/external_libraries/voro_build )
elseif( CMAKE_BUILD_TYPE MATCHES Release )
    message( "******* CASL CMAKE IN RELEASE MODE *******" )
    set( PETSC_DIR $ENV{HOME}/CASL/casl_code_base/external_libraries/petsc_release )
    set( P4EST_DIR $ENV{HOME}/CASL/casl_code_base/external_libraries/p4est_release )
    set( VORO_DIR  $ENV{HOME}/CASL/casl_code_base/external_libraries/voro_build )
else()
    message( FATAL_ERROR "Invalid or missing CMAKE_BUILD_TYPE macro --it should be 'Debug' or 'Release'." )
endif()

set( MPI_DIR $ENV{HOME}/CASL/casl_code_base/external_libraries/mpich_casl_local_install )
set( BOOST_DIR $ENV{HOME}/CASL/casl_code_base/external_libraries/boost_build )

list( APPEND INC_DIRS
        ${PETSC_DIR}/include
        ${P4EST_DIR}/include
        ${VORO_DIR}/include/voro++
        ${MPI_DIR}/include
        ${BOOST_DIR}/include )

list( APPEND LIB_DIRS
        ${PETSC_DIR}/lib
        ${P4EST_DIR}/lib
        ${VORO_DIR}/lib
        ${MPI_DIR}/lib
        ${BOOST_DIR}/lib )

list( APPEND LIBS
        petsc
        p4est
        sc
        voro++
        mpi )

set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpicxx )

################################################## Build Configuration ###################################################

# ---- Global compile definitions
add_compile_definitions(CASL_THROWS)

# ---- Collect source files from src directory
file(GLOB_RECURSE CASL_SOURCES
        ${CASL_P4EST_ROOT}/src/*.cpp
        ${CASL_P4EST_ROOT}/src/*.c
)

file(GLOB_RECURSE CASL_HEADERS
        ${CASL_P4EST_ROOT}/src/*.h
        ${CASL_P4EST_ROOT}/src/*.hpp
)

# ---- Filter out test/example files
list(FILTER CASL_SOURCES EXCLUDE REGEX ".*/test_.*\\.cpp$")
list(FILTER CASL_SOURCES EXCLUDE REGEX ".*/main.*\\.cpp$")

# ---- Include and link directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CASL_P4EST_ROOT}
        ${INC_DIRS}
)

link_directories(${LIB_DIRS})

# ======================================================================
# Core object library
# ======================================================================
add_library(casl_core_obj OBJECT ${CASL_SOURCES} ${CASL_HEADERS})

target_include_directories(casl_core_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CASL_P4EST_ROOT}
        ${INC_DIRS}
)

# ======================================================================
# Heat solver executable
# ======================================================================
add_executable(diffusion_nodes_mls
        ${CMAKE_CURRENT_SOURCE_DIR}/main_heat_solver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/heat_test_cases.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/heat_convergence_analysis.cpp
        $<TARGET_OBJECTS:casl_core_obj>
)

# Link external libraries
target_link_libraries(diffusion_nodes_mls PRIVATE ${LIBS})

# Allow includes like <src/...>
target_include_directories(diffusion_nodes_mls PRIVATE ${CASL_P4EST_ROOT})

# Set output directory
set(WORKSPACE_DIR "$ENV{HOME}/CASL/workspace/built_examples")
set_target_properties(diffusion_nodes_mls PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${WORKSPACE_DIR}/${PROJECT_NAME}/bin
)

message(STATUS "====================================================")
message(STATUS "diffusion_nodes_mls will be built in:")
message(STATUS "  ${WORKSPACE_DIR}/${PROJECT_NAME}/bin")
message(STATUS "====================================================")