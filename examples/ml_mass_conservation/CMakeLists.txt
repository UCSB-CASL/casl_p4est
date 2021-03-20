# This CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/
# To create different configurations, look into https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html
cmake_minimum_required( VERSION 3.17 )
project( ml_mass_conservation )

set( CMAKE_CXX_STANDARD 11 )

######################## Source code ########################

if( DIMENSION MATCHES 2d )
	list( APPEND SRC main_2d.cpp )
endif()

#if( DIMENSION MATCHES 3d )
#	list( APPEND SRC sphere_datasets_3d.cpp )
#endif()
#add_compile_definitions( PRIVATE P4_TO_P8 )
add_compile_definitions( PRIVATE CASL_THROWS )

######################## Settings ########################

# CASL library root directory.
set( PARCASL ${CMAKE_CURRENT_SOURCE_DIR}/../../ )

# Compilers.
set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpicxx )

######################## Libraries ########################
# These commands must be given prior to defining any target.

# find_package( MPI REQUIRED )		# Not needed if using mpich rather than openmpi.
find_package( Boost COMPONENTS filesystem REQUIRED )

# Library linking directories.
link_directories(
		/usr/local/lib
		/usr/local/petsc/lib
		/usr/local/p4est-2.2/lib )

link_libraries(
		mpi
		petsc
		p4est
		sc
		voro++
		${Boost_FILESYSTEM_LIBRARY} )

include_directories(
		.
		${PARCASL}
		/usr/local/include
		/usr/local/petsc/include
		/usr/local/p4est-2.2/include
		/usr/local/include/voro++
		${Boost_INCLUDE_DIR} )

######################## Creating the executable ########################

include( ${PARCASL}/libparcasl.cmake )

add_executable( ml_mass_conservation
		${SRC} ${HEADERS} ${SOURCES} )