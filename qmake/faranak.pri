# --------------------------------------------------------------
#    Paths to external libraries to be linked to casl_p4est
# --------------------------------------------------------------

# PETSc
PETSC_DIR_DEBUG         = $$HOME/CASL/casl_code_base/external_libraries/petsc_debug
PETSC_DIR_RELEASE       = $$HOME/CASL/casl_code_base/external_libraries/petsc_release

PETSC_INCLUDES_RELEASE  = $$PETSC_DIR_RELEASE/include $$PETSC_DIR_RELEASE/include/petsc
PETSC_INCLUDES_DEBUG    = $$PETSC_DIR_DEBUG/include $$PETSC_DIR_DEBUG/include/petsc

PETSC_LIBS_RELEASE      = -Wl,-rpath,$$PETSC_DIR_RELEASE/lib -L$$PETSC_DIR_RELEASE/lib -lpetsc
PETSC_LIBS_DEBUG        = -Wl,-rpath,$$PETSC_DIR_DEBUG/lib -L$$PETSC_DIR_DEBUG/lib -lpetsc

# p4est
P4EST_DIR_DEBUG         = $$HOME/CASL/casl_code_base/external_libraries/p4est_debug
P4EST_DIR_RELEASE       = $$HOME/CASL/casl_code_base/external_libraries/p4est_release
P4EST_SRC               = $$HOME/CASL/casl_code_base/external_libraries/p4est-2.8/src

P4EST_INCLUDES_RELEASE  = $$P4EST_DIR_RELEASE/include $$P4EST_SRC
P4EST_INCLUDES_DEBUG    = $$P4EST_DIR_DEBUG/include $$P4EST_SRC

P4EST_LIBS_RELEASE      = -Wl,-rpath,$$P4EST_DIR_RELEASE/lib -L$$P4EST_DIR_RELEASE/lib -lp4est -lsc
P4EST_LIBS_DEBUG        = -Wl,-rpath,$$P4EST_DIR_DEBUG/lib -L$$P4EST_DIR_DEBUG/lib -lp4est -lsc

# voro++
VORO_DIR                = $$HOME/CASL/casl_code_base/external_libraries/voro++

VORO_INCLUDES_RELEASE   = $$VORO_DIR/include/voro++
VORO_INCLUDES_DEBUG     = $$VORO_DIR/include/voro++

VORO_LIBS_RELEASE       = -Wl,-rpath,$$VORO_DIR/lib -L$$VORO_DIR/lib -lvoro++
VORO_LIBS_DEBUG         = -Wl,-rpath,$$VORO_DIR/lib -L$$VORO_DIR/lib -lvoro++

# Compiler and linker
QMAKE_CC                = /usr/local/bin/mpicc
QMAKE_CXX               = /usr/local/bin/mpicxx
QMAKE_LINK              = /usr/local/bin/mpicxx
