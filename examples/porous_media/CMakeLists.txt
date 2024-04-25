# This CMakeList.txt can be used to create new project or run existing one.
# Note: this CMakeLists.txt file has been built manually from the common.pri and libparcasl.pri config files under qmake/
# To create different configurations, see https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html

# Choose to run a 2d or a 3d simulation:
set (DIMENSION 2d)

# Set minimum:
cmake_minimum_required( VERSION 3.17 )
set( CMAKE_CXX_STANDARD 14 )

# Automatically get the project's name by parsing the ${CMAKE_CURRENT_SOURCE_DIR}:
string(REGEX REPLACE .*/ "" THE_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} )

# Automatically set the project name:
project( ${THE_PROJECT_NAME} )

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

set( PARCASL ${CMAKE_CURRENT_SOURCE_DIR}/../../)		# CASL library root directory.
include( ${PARCASL}/cmake/casl_generic.cmake )		# Set paths for this machine and populate LIBS,
								# INC_DIRS, and LIB_DIRS lists.

if( CMAKE_BUILD_TYPE MATCHES Debug )
	add_compile_definitions( PRIVATE DEBUG )		# Enable debug macro.
endif()

# Libraries and their linking/include directories.
link_directories(
		${LIB_DIRS})

link_libraries(
		${LIBS})

include_directories(
		.
		${PARCASL}
		${INC_DIRS})

########## Creating the executable

include( ${PARCASL}/cmake/libparcasl.cmake )

add_executable( ${PROJECT_NAME}
		${SRC} ${HEADERS} ${SOURCES}
		)

message("*****************")
message("This CMake will generate the executable ${PROJECT_NAME} in ${DIMENSION}.")
message("*****************")
