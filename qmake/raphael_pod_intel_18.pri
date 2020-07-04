# --------------------------------------------------------------
#    Paths to external libraries to be linked to casl_p4est
# --------------------------------------------------------------

DEFINES += POD_CLUSTER

# PETSc
PETSC_DIR_DEBUG         = /home/regan/libraries_with_intel_18/petsc_debug
PETSC_DIR_RELEASE       = /home/regan/libraries_with_intel_18/petsc

PETSC_INCLUDES_RELEASE  = $$PETSC_DIR_RELEASE/include
PETSC_INCLUDES_DEBUG    = $$PETSC_DIR_DEBUG/include
PETSC_LIBS_RELEASE      = -Wl,-rpath,$$PETSC_DIR_RELEASE/lib -L$$PETSC_DIR_RELEASE/lib -lpetsc
PETSC_LIBS_DEBUG        = -Wl,-rpath,$$PETSC_DIR_DEBUG/lib -L$$PETSC_DIR_DEBUG/lib -lpetsc

# p4est
P4EST_DIR_DEBUG         = /home/regan/libraries_with_intel_18/p4est_debug
P4EST_DIR_RELEASE       = /home/regan/libraries_with_intel_18/p4est

P4EST_INCLUDES_RELEASE  = $$P4EST_DIR_RELEASE/include
P4EST_INCLUDES_DEBUG    = $$P4EST_DIR_DEBUG/include
P4EST_LIBS_RELEASE      = -Wl,-rpath,$$P4EST_DIR_RELEASE/lib -L$$P4EST_DIR_RELEASE/lib -lp4est -lsc
P4EST_LIBS_DEBUG        = -Wl,-rpath,$$P4EST_DIR_DEBUG/lib -L$$P4EST_DIR_DEBUG/lib -lp4est -lsc

# voro++
VORO_DIR                = /home/regan/libraries_with_intel_18/voro++

VORO_INCLUDES_RELEASE   = $$VORO_DIR/include/voro++
VORO_INCLUDES_DEBUG     = $$VORO_DIR/include/voro++
VORO_LIBS_RELEASE       = -Wl,-rpath,$$VORO_DIR/lib -L$$VORO_DIR/lib -lvoro++
VORO_LIBS_DEBUG         = -Wl,-rpath,$$VORO_DIR/lib -L$$VORO_DIR/lib -lvoro++

QMAKE_CC                = /sw/intel/parallel_studio_xe_2018/compilers_and_libraries_2018/linux/mpi/intel64/bin/mpicc
QMAKE_CXX               = /sw/intel/parallel_studio_xe_2018/compilers_and_libraries_2018/linux/mpi/intel64/bin/mpicxx
QMAKE_LINK              = /sw/intel/parallel_studio_xe_2018/compilers_and_libraries_2018/linux/mpi/intel64/bin/mpicxx

