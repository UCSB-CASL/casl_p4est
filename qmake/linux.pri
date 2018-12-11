# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /home/temprano/Software/petsc-3.6.4/build-release/include
PETSC_INCLUDES_DEBUG   = /home/temprano/Software/petsc-3.6.4/build-debug/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/temprano/Software/petsc-3.6.4/build-release/include/lib -L/home/temprano/Software/petsc-3.6.4/build-release/include -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/temprano/Software/petsc-3.6.4/build-debug/include/lib -L/home/temprano/Software/petsc-3.6.4/build-debug/include -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/temprano/Software/p4est-2.0/build-release/include
P4EST_INCLUDES_DEBUG   = /home/temprano/Software/p4est-2.0/build-debug/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/temprano/Software/p4est-2.0/build-release/lib -L/home/temprano/Software/p4est-2.0/build-release/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/temprano/Software/p4est-2.0/build-debug/lib -L/home/temprano/Software/p4est-2.0/build-debug/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/temprano/Software/voro++-0.4.6/build/include/voro++
VORO_INCLUDES_DEBUG   = /home/temprano/Software/voro++-0.4.6/build/include/voro++
VORO_LIBS_RELEASE     = /home/temprano/Software/voro++-0.4.6/build/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/temprano/Software/voro++-0.4.6/build/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

