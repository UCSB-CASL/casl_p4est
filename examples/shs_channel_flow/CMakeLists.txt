# This CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/
# To create different configurations, look into https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html
cmake_minimum_required( VERSION 3.16 )
project( shs_channel_flow )

set( CMAKE_CXX_STANDARD 14 )

###################################### Compiler custom macros for current project ######################################

if( DIMENSION MATCHES 2d )
	list( APPEND SRC main_2d.cpp )
elseif( DIMENSION MATCHES 3d )
	list( APPEND SRC main_3d.cpp )
else()
	message( FATAL_ERROR "Invalid or missing DIMENSION macro --it should be '2d' or '3d'." )
endif()

add_compile_definitions( PRIVATE CASL_THROWS )

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
# Comment these out when using debugger.

#if( CMAKE_BUILD_TYPE MATCHES Release )
#	set( GCC_COVERAGE_COMPILE_FLAGS "-O2 -O3 -march=native" )
#	set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
#	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
#endif()

################################################## Loading libraries ###################################################
# These commands must be given prior to defining any target.
# Add here libraries just for this project.  See ml_curvature/CMakeLists.txt for an example adding OpenMP.

# Libraries and their linking/include directories (see cmake/<machine>.cmake).
link_directories( ${LIB_DIRS} )
link_libraries( ${LIBS} )
include_directories( . ${PARCASL} ${INC_DIRS} )

############################################### Creating the executable ################################################

include( ${PARCASL}/cmake/libparcasl.cmake )

add_executable( shs_channel_flow ${SRC} ${SOURCES} )	# Turns out we don't need the headers.
