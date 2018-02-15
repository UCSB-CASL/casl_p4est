# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /home/egan/libraries/petsc/include
PETSC_INCLUDES_DEBUG   = /home/egan/libraries/petsc/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/egan/libraries/petsc/lib -L/home/egan/libraries/petsc/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/egan/libraries/petsc/lib -L/home/egan/libraries/petsc/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/egan/libraries/p4est/include
P4EST_INCLUDES_DEBUG   = /home/egan/libraries/p4est/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/egan/libraries/p4est/lib -L/home/egan/libraries/p4est/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/egan/libraries/p4est/lib -L/home/egan/libraries/p4est/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/egan/libraries/voro++/include/voro++
VORO_INCLUDES_DEBUG   = /home/egan/libraries/voro++/include/voro++
VORO_LIBS_RELEASE     = /home/egan/libraries/voro++/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/egan/libraries/voro++/lib/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

