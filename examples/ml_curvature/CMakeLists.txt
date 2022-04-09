# This CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/
# To create different configurations, look into https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html
cmake_minimum_required( VERSION 3.17 )
project( ml_curvature )

set( CMAKE_CXX_STANDARD 14 )

###################################### Compiler custom macros for current project ######################################

if( DIMENSION MATCHES 2d )
	list( APPEND SRC main_2d.cpp )
elseif( DIMENSION MATCHES 3d )
	list( APPEND SRC data_sets/sphere/sphere_dataset_3d_alt_rand.cpp )
else()
	message( FATAL_ERROR "Invalid or missing DIMENSION macro --it should be '2d' or '3d'." )
endif()

add_compile_definitions( PRIVATE CASL_THROWS )
add_compile_definitions( PRIVATE FDEEP_FLOAT_TYPE=float )		# Frugally deep floating-point precision.
add_compile_definitions( PRIVATE EIGEN_DONT_PARALLELIZE )		# Prevents parallelization in Eigen library.

############################# Load paths to libraries and set compiler for current system ##############################

set( PARCASL ${CMAKE_CURRENT_SOURCE_DIR}/../../ )				# CASL library root directory.

if( DEFINED ENV{CASL_CMAKE_PROFILE} )							# Set paths for this machine and populate LIBS, INC_DIRS,
	include( ${PARCASL}/cmake/$ENV{CASL_CMAKE_PROFILE} )		# and LIB_DIRS lists.  You must create the environment
else()															# variable CASL_CMAKE_PROFILE.
	message( FATAL_ERROR "Environment variable CASL_CMAKE_PROFILE is not set!" )
endif()

if( CMAKE_BUILD_TYPE MATCHES Debug )
	add_compile_definitions( PRIVATE DEBUG )					# Enable debug macro.
endif()

############################################# Compiler optimization flags ##############################################

if( CMAKE_BUILD_TYPE MATCHES Release )
	set( GCC_COVERAGE_COMPILE_FLAGS "-O2 -O3 -march=native" )
	set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
endif()

################################################## Loading libraries ###################################################
# These commands must be given prior to defining any target.

# Library just used for this project (i.e., not included in cmake/<machine>.cmake)
# Let's load OpenMP and its flags.  See http://blog.llvm.org/2015/05/openmp-support_22.html for more details.
find_package( OpenMP )
if( OPENMP_FOUND )
	set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	list( APPEND LIBS ${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES} )
	list( APPEND INC_DIRS ${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS} )
endif()

# Libraries and their linking/include directories (see cmake/<machine>.cmake).  Also adding OpenMP from above.
link_directories( ${LIB_DIRS} )

link_libraries( ${LIBS} )

include_directories(
		.
		${PARCASL}
		${INC_DIRS} )

############################################### Creating the executable ################################################

include( ${PARCASL}/cmake/libparcasl.cmake )

add_executable( ml_curvature
		${SRC} ${HEADERS} ${SOURCES} )