# Paths to external libraries and compilers for build current system.
# Code creates three library-config lists:
# 1. INC_DIRS - Include directories,
# 2. LIB_DIRS - Link-library directories, and
# 3. LIBS     - Actual libraries.

################################################## Minimal libraries ###################################################

# Checking mode: Created by visiting CLion | Preferences | Build, Execution, Deployment | CMake
# Based on https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000919039-Clion-how-to-build-cmake-to-support-debug-release
message( "" )
if( CMAKE_BUILD_TYPE MATCHES Debug )
	message( "******* CASL CMAKE IN DEBUG MODE *******" )

	set( PETSC_DIR /usr/local/petsc-3.16.3 )	# PETSc.
	set( P4EST_DIR /usr/local/p4est-2.8 )		# p4est.
	set( VORO_DIR  /usr/local )						# Voro++.

elseif( CMAKE_BUILD_TYPE MATCHES Release )
	message( "******* CASL CMAKE IN RELEASE MODE *******" )

	set( PETSC_DIR /usr/local/petsc-3.16.3 )				# PETSc.
	set( P4EST_DIR /usr/local/p4est-2.8 )			# p4est.
	set( VORO_DIR  /usr/local )						# Voro++

else()
	message( FATAL_ERROR "Invalid or missing CMAKE_BUILD_TYPE macro --it should be 'Debug' or 'Release'." )
endif()

# MPI.
set( MPI_DIR /usr )

# Boost.  For header-only functions, you don't need to specify a particular component in
# `link_libraries`-- just add the path to boost headers in `include_directories`.  If you need
# some component use:
# find_package( Boost COMPONENTS filesystem REQUIRED )  <-- filesystem component.
# then `link_libraries( ${Boost_FILESYSTEM_LIBRARY} )` and  `include_directories( ${Boost_INCLUDE_DIR} )`.
set( BOOST_DIR /usr/local/boost-1.78.0 )

# Let's add libraries to the lists.
list( APPEND INC_DIRS					# Include directories.
		${PETSC_DIR}/include
		${P4EST_DIR}/include
		${VORO_DIR}/include/voro++
		${MPI_DIR}/include/mpich
		${BOOST_DIR}/include )

list( APPEND LIB_DIRS					# Library directories.
		${PETSC_DIR}/lib
		${P4EST_DIR}/lib
		${VORO_DIR}/lib
		${MPI_DIR}/lib/mpich
		${BOOST_DIR}/lib )

list( APPEND LIBS						# Libraries: Note we don't include Boost --we need header-only functions.
		petsc
		p4est
		sc
		voro++
		mpi )

message( "" )
message( "---------------- Minimal libraries ----------------" )
message( "** PETSc   : " ${PETSC_DIR} )
message( "** p4est   : " ${P4EST_DIR} )
message( "** Voro++  : " ${VORO_DIR} )
message( "** MPI     : " ${MPI_DIR} )
message( "** Boost   : " ${BOOST_DIR} )


######################################### Optional machine-learning libraries ##########################################

if( ENABLE_ML MATCHES 1 )		# Set this CMake variable as -DENABLE_ML=1.

	set( OpenBLAS_DIR / ) 	# OpenBLAS.
	set( DLIB_DIR / )					# dlib.
	set( JSON_DIR / )					# json: only headers.  You can also add `nlohmann_json::nlohmann_json`
												# to LIBS list if you use `find_package( nlohmann_json CONFIG REQUIRED )`.
	set( FDEEP_DIR / )					# frugally-deep: only headers.

	# Append to lists.
	list( APPEND INC_DIRS					# Include directories.
			${OpenBLAS_DIR}/include
			${DLIB_DIR}/include
			${JSON_DIR}/include				# nlohmann's json and frugally-deep are header-only libraries.
			${FDEEP_DIR}/include )

	list( APPEND LIB_DIRS					# Library directories.
			${OpenBLAS_DIR}/lib
			${DLIB_DIR}/lib )

	list( APPEND LIBS						# Libraries.
			openblas
			dlib )

	message( "" )
	message( "---------------- Machine-learning libraries ----------------" )
	message( "** OpenBLAS     : " ${OpenBLAS_DIR} )
	message( "** dlib         : " ${DLIB_DIR} )
	message( "** json         : " ${JSON_DIR} )
	message( "** frugally-deep: " ${FDEEP_DIR} )

endif()

message( "" )

###################################################### Compilers #######################################################

set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpicxx )
