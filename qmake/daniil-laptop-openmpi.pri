# PETSc
PETSC_DIR_DBG = /home/dbochkov/Software/PETSc/petsc-3.9.2/build-debug
PETSC_DIR_RLS = /home/dbochkov/Software/PETSc/petsc-3.9.2/build-release

PETSC_INCLUDES_RELEASE = $$PETSC_DIR_RLS/include $$PETSC_DIR_RLS/../include
PETSC_INCLUDES_DEBUG   = $$PETSC_DIR_DBG/include $$PETSC_DIR_DBG/../include
PETSC_LIBS_RELEASE = -Wl,-rpath,$$PETSC_DIR_RLS/lib -L$$PETSC_DIR_RLS/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,$$PETSC_DIR_DBG/lib -L$$PETSC_DIR_DBG/lib -lpetsc

# p4est
P4EST_DIR_DBG = /home/dbochkov/Software/p4est/p4est-2.0/build-release
P4EST_DIR_RLS = /home/dbochkov/Software/p4est/p4est-2.0/build-release

P4EST_INCLUDES_RELEASE = $$P4EST_DIR_RLS/include
P4EST_INCLUDES_DEBUG   = $$P4EST_DIR_DBG/include
P4EST_LIBS_RELEASE = -Wl,-rpath,$$P4EST_DIR_RLS/lib -L$$P4EST_DIR_RLS/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,$$P4EST_DIR_DBG/lib -L$$P4EST_DIR_DBG/lib -lp4est -lsc

# voro++
VORO_DIR_DBG = /home/dbochkov/Software/voro++/voro++-0.4.6/build-release
VORO_DIR_RLS = /home/dbochkov/Software/voro++/voro++-0.4.6/build-release

VORO_INCLUDES_RELEASE = $$VORO_DIR_RLS/include/voro++
VORO_INCLUDES_DEBUG   = $$VORO_DIR_DBG/include/voro++
VORO_LIBS_RELEASE     = -Wl,-rpath,$$VORO_DIR_RLS/lib -L$$VORO_DIR_RLS/lib -lvoro++
VORO_LIBS_DEBUG       = -Wl,-rpath,$$VORO_DIR_DBG/lib -L$$VORO_DIR_DBG/lib -lvoro++

# matlab (for computing condition numbers, it's ok not to provide)
MATLAB_DIR = /home/dbochkov/Software/MATLAB/R2018a

MATLAB_INCLUDES = $$MATLAB_DIR/extern/include/
MATLAB_LIBS = -Wl,-rpath,$$MATLAB_DIR/bin/glnxa64/ -L$$MATLAB_DIR/bin/glnxa64/ -leng -lmx

# mpi
MPI_DIR = /usr/lib/x86_64-linux-gnu/openmpi/

MPI_INCLUDES = /usr/lib/x86_64-linux-gnu/openmpi/include/
MPI_LIBS = -lmpi

# Boost
BOOST_INCLUDES = /home/dbochkov/Software/Boost/boost_1_70_0

QMAKE_CC=mpicc
QMAKE_CXX=mpicxx
QMAKE_LINK=mpicxx

