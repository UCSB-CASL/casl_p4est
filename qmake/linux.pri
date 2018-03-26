# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /home/regan/libraries/petsc/include
PETSC_INCLUDES_DEBUG   = /home/regan/libraries/petsc/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/regan/libraries/petsc/lib -L/home/regan/libraries/petsc/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/regan/libraries/petsc/lib -L/home/regan/libraries/petsc/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/regan/libraries/p4est/include
P4EST_INCLUDES_DEBUG   = /home/regan/libraries/p4est/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/regan/libraries/p4est/lib -L/home/regan/libraries/p4est/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/regan/libraries/p4est/lib -L/home/regan/libraries/p4est/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/regan/libraries/voro++/include/voro++
VORO_INCLUDES_DEBUG   = /home/regan/libraries/voro++/include/voro++
VORO_LIBS_RELEASE     = /home/regan/libraries/voro++/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/regan/libraries/voro++/lib/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

