# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /usr/local/include
PETSC_INCLUDES_DEBUG   = /usr/local/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lpetsc

# voro++
VORO_INCLUDES_RELEASE = /usr/local/include/voro++
VORO_INCLUDES_DEBUG   = /usr/local/include/voro++
VORO_LIBS_RELEASE     = /usr/local/lib/libvoro++.a
VORO_LIBS_DEBUG       = /usr/local/lib/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

