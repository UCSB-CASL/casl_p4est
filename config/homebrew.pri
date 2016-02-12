# --------------------------------------------------------------
# Common settings for projects using homebrew builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /usr/local/include
PETSC_INCLUDES_DEBUG   = /usr/local/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /usr/local/include
P4EST_INCLUDES_DEBUG   = /usr/local/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /usr/local/include/voro++
VORO_INCLUDES_DEBUG   = /usr/local/include/voro++
VORO_LIBS_RELEASE     = /usr/local/lib/libvoro++.a
VORO_LIBS_DEBUG       = /usr/local/lib/libvoro++.a

# set the compiler path
QMAKE_CC=/usr/local/bin/mpicc
QMAKE_CXX=/usr/local/bin/mpicxx
QMAKE_LINK=$$QMAKE_CXX

# load commmon settings
include(common.pri)

