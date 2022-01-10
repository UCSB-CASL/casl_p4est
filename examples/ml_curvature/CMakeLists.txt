# This CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/
# To create different configurations, look into https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html
cmake_minimum_required( VERSION 3.17 )
project( ml_curvature )

set( CMAKE_CXX_STANDARD 14 )


######################## Compiler custom macros ########################

if( DIMENSION MATCHES 2d )
	list( APPEND SRC main_2d.cpp )
elseif( DIMENSION MATCHES 3d )
	list( APPEND SRC data_sets/paraboloid_test_3d.cpp )
else()
	message( FATAL_ERROR "Invalid or missing DIMENSION macro --it should be '2d' or '3d'." )
endif()

add_compile_definitions( PRIVATE CASL_THROWS )
add_compile_definitions( PRIVATE FDEEP_FLOAT_TYPE=float )		# Frugally deep floating-point precision.
add_compile_definitions( PRIVATE EIGEN_DONT_PARALLELIZE )		# Prevents parallelization in Eigen library.


######################## Execution mode ########################

# Checking mode: Created by visiting CLion | Preferences | Build, Execution, Deployment | CMake
# Based on https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000919039-Clion-how-to-build-cmake-to-support-debug-release
message( "" )
if( CMAKE_BUILD_TYPE MATCHES Debug )
	message( "******* CASL CMAKE IN DEBUG MODE *******" )
	set( PETSC_DIR /usr/local/petsc-3.12.0-debug )	# PETSc directory for debug mode.
	set( P4EST_DIR /usr/local/p4est-2.2-debug )		# p4est directory for debug mode.

	add_compile_definitions( PRIVATE DEBUG )		# Enable debug macro.
elseif( CMAKE_BUILD_TYPE MATCHES Release )
	message( "******* CASL CMAKE IN RELEASE MODE *******" )
	set( PETSC_DIR /usr/local/petsc )				# PETSc directory for release mode.
	set( P4EST_DIR /usr/local/p4est-2.2 )			# p4est directory for release mode.
endif()
message( "** Loading PETSc from: " ${PETSC_DIR} )
message( "** Loading p4est from: " ${P4EST_DIR} )
message( "" )


######################## Compiler custom flags and settings ########################

# CASL library root directory.
set( PARCASL ${CMAKE_CURRENT_SOURCE_DIR}/../../ )

# Compilers.
set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpicxx )

# Optimization flags.  Comment these out when using debugger.
#set( GCC_COVERAGE_COMPILE_FLAGS "-O2 -O3 -march=native" )
#set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )


######################## Loading libraries ########################
# These commands must be given prior to defining any target.

# OpenMP flags.
# See http://blog.llvm.org/2015/05/openmp-support_22.html
find_package( OpenMP )
if( OPENMP_FOUND )
	set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()

# JSON library.
find_package( nlohmann_json CONFIG REQUIRED )

# find_package( MPI REQUIRED )		# Not needed if using mpich rather than openmpi.
find_package( Boost COMPONENTS filesystem REQUIRED )

# Library linking directories.
link_directories(
		/usr/local/lib
		${PETSC_DIR}/lib
		${P4EST_DIR}/lib
		/usr/local/opt/libomp/lib
		/usr/local/OpenBlas/lib )

link_libraries(
		mpi
		petsc
		p4est
		sc
		voro++
		omp
		openblas
		nlohmann_json::nlohmann_json
		dlib
		${Boost_FILESYSTEM_LIBRARY} )

include_directories(
		.
		${PARCASL}
		/usr/local/include
		${PETSC_DIR}/include
		${P4EST_DIR}/include
		/usr/local/opt/libomp/include
		/usr/local/OpenBlas/include
		${Boost_INCLUDE_DIR} )


######################## Creating the executable ########################

include( ${PARCASL}/libparcasl.cmake )

add_executable( ml_curvature
		${SRC} ${HEADERS} ${SOURCES} )