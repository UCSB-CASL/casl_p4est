# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

CONFIG(raphael) {
# PETSc
PETSC_INCLUDES_RELEASE = /home/regan/libraries/petsc/include
PETSC_INCLUDES_DEBUG   = /home/regan/libraries/petsc_debug/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/regan/libraries/petsc/lib -L/home/regan/libraries/petsc/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/regan/libraries/petsc_debug/lib -L/home/regan/libraries/petsc_debug/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/regan/libraries/p4est/include
P4EST_INCLUDES_DEBUG   = /home/regan/libraries/p4est_debug/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/regan/libraries/p4est/lib -L/home/regan/libraries/p4est/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/regan/libraries/p4est_debug/lib -L/home/regan/libraries/p4est_debug/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/regan/libraries/voro++/include/voro++
VORO_INCLUDES_DEBUG   = $VORO_INCLUDES_RELEASE
VORO_LIBS_RELEASE     = /home/regan/libraries/voro++/lib/libvoro++.a
VORO_LIBS_DEBUG       = $VORO_LIBS_RELEASE

# boost
VORO_INCLUDES_RELEASE = /home/regan/libraries/voro++/include/voro++
VORO_INCLUDES_DEBUG   = $VORO_INCLUDES_RELEASE
VORO_LIBS_RELEASE     = /home/regan/libraries/voro++/lib/libvoro++.a
VORO_LIBS_DEBUG       = $VORO_LIBS_RELEASE
QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx
}
else {
# PETSc
PETSC_INCLUDES_RELEASE = /home/rochi/libraries/petsc_release/include
PETSC_INCLUDES_DEBUG   = /home/rochi/libraries/petsc_debug/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/rochi/libraries/petsc_release/lib -L/home/rochi/libraries/petsc_release/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/rochi/libraries/petsc_debug/lib -L/home/rochi/libraries/petsc_debug/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/rochi/libraries/p4est_release/include
P4EST_INCLUDES_DEBUG   = /home/rochi/libraries/p4est_debug/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/rochi/libraries/p4est_release/lib -L/home/rochi/libraries/p4est_release/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/rochi/libraries/p4est_debug/lib -L/home/rochi/libraries/p4est_debug/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /usr/local/include/voro++
VORO_INCLUDES_DEBUG   = /usr/local/include/voro++
VORO_LIBS_RELEASE     = /usr/local/lib/libvoro++.a
VORO_LIBS_DEBUG       = /usr/local/lib/libvoro++.a

QMAKE_CC = /usr/local/bin/mpicc
QMAKE_CXX = /usr/local/bin/mpicxx
QMAKE_LINK = /usr/local/bin/mpicxx
}

include(common.pri)

