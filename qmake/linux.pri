# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE  = /home/egan/libraries/petsc-3.6.4/build-openmpi/include
PETSC_INCLUDES_RELEASE += /home/egan/libraries/petsc-3.6.4/include
PETSC_INCLUDES_DEBUG    = /home/egan/libraries/petsc-3.6.4/build-openmpi/include
PETSC_INCLUDES_DEBUG   += /home/egan/libraries/petsc-3.6.4/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/egan/libraries/petsc-3.6.4/build-openmpi/lib -L/home/egan/libraries/petsc-3.6.4/build-openmpi/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/egan/libraries/petsc-3.6.4/build-openmpi/lib -L/home/egan/libraries/petsc-3.6.4/build-openmpi/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/egan/libraries/p4est-1.1/local/include
P4EST_INCLUDES_DEBUG   = /home/egan/libraries/p4est-1.1/local/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/egan/libraries/p4est-1.1/local/lib -L/home/egan/libraries/p4est-1.1/local/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/egan/libraries/p4est-1.1/local/lib -L/home/egan/libraries/p4est-1.1/local/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /usr/local/include/voro++
VORO_INCLUDES_DEBUG   = /usr/local/include/voro++
VORO_LIBS_RELEASE     = /usr/local/lib/libvoro++.a
VORO_LIBS_DEBUG       = /usr/local/lib/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

