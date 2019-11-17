# --------------------------------------------------------------
#    Paths to external libraries to be linked to casl_p4est
# --------------------------------------------------------------

# PETSc
PETSC_DIR_DEBUG         = /Users/rochi/Libraries_for_Code/petsc-3.11.4/build_debug
PETSC_DIR_RELEASE       = /Users/rochi/Libraries_for_Code/petsc-3.11.4/build_opt

PETSC_INCLUDES_RELEASE  = $$PETSC_DIR_RELEASE/include
PETSC_INCLUDES_DEBUG    = $$PETSC_DIR_DEBUG/include
PETSC_LIBS_RELEASE      = -Wl,-rpath,$$PETSC_DIR_RELEASE/lib -L$$PETSC_DIR_RELEASE/lib -lpetsc
PETSC_LIBS_DEBUG        = -Wl,-rpath,$$PETSC_DIR_DEBUG/lib -L$$PETSC_DIR_DEBUG/lib -lpetsc

# p4est
P4EST_DIR_DEBUG         = /Users/rochi/Libraries_for_Code/p4est-2.2/build_debug
P4EST_DIR_RELEASE       = /Users/rochi/Libraries_for_Code/p4est-2.2/build_opt

P4EST_INCLUDES_RELEASE  = $$P4EST_DIR_RELEASE/include
P4EST_INCLUDES_DEBUG    = $$P4EST_DIR_DEBUG/include
P4EST_LIBS_RELEASE      = -Wl,-rpath,$$P4EST_DIR_RELEASE/lib -L$$P4EST_DIR_RELEASE/lib -lp4est -lsc
P4EST_LIBS_DEBUG        = -Wl,-rpath,$$P4EST_DIR_DEBUG/lib -L$$P4EST_DIR_DEBUG/lib -lp4est -lsc

# voro++
VORO_DIR                = /Users/rochi/Libraries_for_Code/voro++-0.4.6/build

VORO_INCLUDES_RELEASE   = $$VORO_DIR/include/voro++
VORO_INCLUDES_DEBUG     = $$VORO_DIR/include/voro++
VORO_LIBS_RELEASE       = -Wl,-rpath,$$VORO_DIR/lib -L$$VORO_DIR/lib -lvoro++
VORO_LIBS_DEBUG         = -Wl,-rpath,$$VORO_DIR/lib -L$$VORO_DIR/lib -lvoro++

# matlab(for computing condition number, its optional)
#MATLAB_DIR              = /Applications/MATLAB_R2017b.app

#MATLAB_INCLUDES         = $$MATLAB_DIR/extern/include/
#MATLAB_LIBS             = $$MATLAB_DIR/extern/bin/

QMAKE_CC                = /usr/local/bin/mpicc
QMAKE_CXX               = /usr/local/bin/mpicxx
QMAKE_LINK              = /usr/local/bin/mpicxx

