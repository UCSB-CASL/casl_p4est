# This CMakeList.txt can be used to create new project or run existing one.
# You need to change:
#		1- the "COMPUTER_NAME" variable to match your computer name.
#		2- the "PROJECT_NAME" variable to be the example you consider. If it is a new research project,
# 		   choose a name that best represent it. That will be the name of the executable.
#

set( COMPUTER_NAME darkness )
set( PROJECT_NAME poisson_nodes_mls_nick )

# Note: this CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/
# To create different configurations, look into https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html

cmake_minimum_required( VERSION 3.17 )
project( ${PROJECT_NAME} )

set( CMAKE_CXX_STANDARD 14 )
set (DIMENSION 2d)

########## Compiler custom macros for current project

if( DIMENSION MATCHES 2d )
	list( APPEND SRC main_2d.cpp )
elseif( DIMENSION MATCHES 3d )
	list( APPEND SRC main_3d.cpp )
else()
	message( FATAL_ERROR "Invalid or missing DIMENSION macro --it should be '2d' or '3d'." )
endif()

add_compile_definitions( PRIVATE CASL_THROWS )
add_compile_definitions( PRIVATE FDEEP_FLOAT_TYPE=float )	# Frugally deep floating-point precision.
add_compile_definitions( PRIVATE EIGEN_DONT_PARALLELIZE )	# Prevents parallelization in Eigen library.

##########  Load paths to libraries and set compiler for current system 

set( PARCASL ${CMAKE_CURRENT_SOURCE_DIR}/../../ )		# CASL library root directory.
include( ${PARCASL}/cmake/${COMPUTER_NAME}.cmake )		# Set paths for this machine and populate LIBS,
								# INC_DIRS, and LIB_DIRS lists.

if( CMAKE_BUILD_TYPE MATCHES Debug )
	add_compile_definitions( PRIVATE DEBUG )		# Enable debug macro.
endif()

########## Compiler optimization flags 

# Comment these out when using debugger (not necessary).

#set( GCC_COVERAGE_COMPILE_FLAGS "-O2 -O3 -march=native" )
#set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

########## Loading libraries 

## In case we want to use a library not included in cmake/<machine>.cmake, e.g. the OpenMP library, we would
## add these commands here (must be given prior to defining any target).
## Here, we are commenting out the following because we do not want to use the OpenMP library
## Let's load OpenMP and its flags.  See http://blog.llvm.org/2015/05/openmp-support_22.html for more details.
# find_package( OpenMP )
# if( OPENMP_FOUND )
# 	set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
# 	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
# endif()

# Libraries and their linking/include directories (see cmake/<machine>.cmake).  Also, showing how to add OpenMP.
link_directories(
		${LIB_DIRS})
		#/usr/local/opt/libomp/lib )			# Uncomment if using the OpenMP library

link_libraries(
		${LIBS})
		#omp )						# Uncomment if using the OpenMP library

include_directories(
		.
		${PARCASL}
		${INC_DIRS})
		#/usr/local/opt/libomp/include )		# Uncomment if using the OpenMP library

########## Creating the executable

include( ${PARCASL}/cmake/libparcasl.cmake )

add_executable( ${PROJECT_NAME}
		${SRC} ${HEADERS} ${SOURCES}
		)
