# --------------------------------------------------------------
# Common settings for projects using on stampede supercomputer 
# --------------------------------------------------------------

CASL_P4EST = /home/elyce/workspace/libraries/casl_p4est

# PETSc
PETSC_DIR_DBG = /home/elyce/workspace/libraries/petsc/petsc_debug
PETSC_DIR_RLS =  /home/elyce/workspace/libraries/petsc/petsc_release

PETSC_INCLUDES_RELEASE = $$PETSC_DIR_RLS/include
PETSC_INCLUDES_DEBUG   = $$PETSC_DIR_DBG/include
PETSC_LIBS_RELEASE = -Wl,-rpath,$$PETSC_DIR_RLS/lib -L$$PETSC_DIR_RLS/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,$$PETSC_DIR_DBG/lib -L$$PETSC_DIR_DBG/lib -lpetsc

# p4est
P4EST_DIR_DBG =  /home/elyce/workspace/libraries/p4est/p4est_debug
P4EST_DIR_RLS =  /home/elyce/workspace/libraries/p4est/p4est_release

P4EST_INCLUDES_RELEASE = $$P4EST_DIR_RLS/include
P4EST_INCLUDES_DEBUG   = $$P4EST_DIR_DBG/include
P4EST_LIBS_RELEASE = -Wl,-rpath,$$P4EST_DIR_RLS/lib -L$$P4EST_DIR_RLS/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,$$P4EST_DIR_DBG/lib -L$$P4EST_DIR_DBG/lib -lp4est -lsc

# voro++
VORO_DIR_DBG =  /home/elyce/workspace/libraries/voro++
VORO_DIR_RLS = /home/elyce/workspace/libraries/voro++

VORO_INCLUDES_RELEASE = $$VORO_DIR_RLS/include/voro++
VORO_INCLUDES_DEBUG   = $$VORO_DIR_DBG/include/voro++
VORO_LIBS_RELEASE     = -L$$VORO_DIR_RLS/lib -lvoro++
VORO_LIBS_DEBUG       = -L$$VORO_DIR_DBG/lib -lvoro++

# matlab (for computing condition numbers, it's ok not to provide)
MATLAB_DIR = /home/elyce/workspace/libraries/MATLAB

MATLAB_INCLUDES = $$MATLAB_DIR/extern/include/
MATLAB_LIBS = -Wl,-rpath,$$MATLAB_DIR/bin/glnxa64/ -L$$MATLAB_DIR/bin/glnxa64/ -leng -lmx

# mpi
# use the mpich versions that were downloaded with petsc
CONFIG(debug, debug|release):{
  MPI_DIR = $$PETSC_DIR_DBG
}
CONFIG(release, debug|release):{
  MPI_DIR = $$PETSC_DIR_RLS
}

#MPI_DIR=/usr/lib/mpich

MPI_INCLUDES = $$MPI_DIR/include
MPI_LIBS = -Wl,-rpath,$$MPI_DIR/lib -L$$MPI_DIR/lib -lmpi -lmpicxx

# Boost
BOOST_INCLUDES = /home/elyce/workspace/source_libraries/boost_1_70_0

QMAKE_CC=mpicc.mpich
QMAKE_CXX=mpicxx.mpich
QMAKE_LINK=mpicxx.mpich
