# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------
CONFIG(raphael, raphael|fernando|helene|elyce) {
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

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx
}

CONFIG(fernando, raphael|fernando|helene|elyce) {
# PETSc
PETSC_INCLUDES_RELEASE = /home/temprano/Software/petsc-3.10.3/build-release/include
PETSC_INCLUDES_DEBUG   = /home/temprano/Software/petsc-3.10.3/build-debug/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/temprano/Software/petsc-3.10.3/build-release/lib -L/home/temprano/Software/petsc-3.10.3/build-release/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/temprano/Software/petsc-3.10.3/build-debug/lib -L/home/temprano/Software/petsc-3.10.3/build-debug/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/temprano/Software/p4est-2.0/build-release/include
P4EST_INCLUDES_DEBUG   = /home/temprano/Software/p4est-2.0/build-debug/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/temprano/Software/p4est-2.0/build-release/lib -L/home/temprano/Software/p4est-2.0/build-release/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/temprano/Software/p4est-2.0/build-debug/lib -L/home/temprano/Software/p4est-2.0/build-debug/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/temprano/Software/voro++-0.4.6/build/include/voro++
VORO_INCLUDES_DEBUG   = /home/temprano/Software/voro++-0.4.6/build/include/voro++
VORO_LIBS_RELEASE     = /home/temprano/Software/voro++-0.4.6/build/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/temprano/Software/voro++-0.4.6/build/lib/libvoro++.a

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx
}

CONFIG(helene, raphael|fernando|helene|elyce) {
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
}

CONFIG(elyce, raphael|fernando|helene|elyce) {
# PETSc
PETSC_INCLUDES_RELEASE = /home/elyce/workspace/libraries/petsc/petsc_release/include
PETSC_INCLUDES_DEBUG   = /home/elyce/workspace/libraries/petsc/petsc_debug/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/elyce/workspace/libraries/petsc/petsc_release/lib -L/home/elyce/workspace/libraries/petsc/petsc_release/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/elyce/workspace/libraries/petsc/petsc_debug/lib -L/home/elyce/workspace/libraries/petsc/petsc_debug/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/elyce/workspace/libraries/p4est/p4est_release/include
P4EST_INCLUDES_DEBUG   = /home/elyce/workspace/libraries/p4est/p4est_debug/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/elyce/workspace/libraries/p4est/p4est_release/lib -L/home/elyce/workspace/libraries/p4est/p4est_release/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/elyce/workspace/libraries/p4est/p4est_debug/lib -L/home/elyce/workspace/libraries/p4est/p4est_debug/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/elyce/workspace/libraries/voro++/include/voro++
VORO_INCLUDES_DEBUG   = /home/elyce/workspace/libraries/voro++/include/voro++
VORO_LIBS_RELEASE     = /home/elyce/workspace/libraries/voro++/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/elyce/workspace/libraries/voro++/lib/libvoro++.a

# mpi
MPI_DIR = /usr/lib/mpich

MPI_INCLUDES = $$MPI_DIR/include
MPI_LIBS = -Wl,-rpath,$$MPI_DIR/lib -L$$MPI_DIR/lib -lmpi -lmpicxx

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx
}


include(common.pri)

