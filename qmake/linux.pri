# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /home/hlevy/Software/petsc-3.10.5/build-release/include
PETSC_INCLUDES_DEBUG   = /home/hlevy/Software/petsc-3.10.5/build-debug/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/hlevy/Software/petsc-3.10.5/build-release/lib -L/home/hlevy/Software/petsc-3.10.5/build-release/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/hlevy/Software/petsc-3.10.5/build-debug/lib -L/home/hlevy/Software/petsc-3.10.5/build-debug/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/hlevy/Software/p4est-2.0/build-release/include
P4EST_INCLUDES_DEBUG   = /home/hlevy/Software/p4est-2.0/build-debug/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/hlevy/Software/p4est-2.0/build-release/lib -L/home/hlevy/Software/p4est-2.0/build-release/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/hlevy/Software/p4est-2.0/build-debug/lib -L/home/hlevy/Software/p4est-2.0/build-debug/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/hlevy/Software/voro++-0.4.6/build/include/voro++
VORO_INCLUDES_DEBUG   = /home/hlevy/Software/voro++-0.4.6/build/include/voro++
VORO_LIBS_RELEASE     = /home/hlevy/Software/voro++-0.4.6/build/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/hlevy/Software/voro++-0.4.6/build/lib/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

