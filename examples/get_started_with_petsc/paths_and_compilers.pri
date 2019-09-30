PARCASL = $$PWD/../..
PETSC_ROOT = foo
# Settings related to release/debug
# Change the following paths to the appropriate root directories on your machine!
CONFIG(debug, debug|release): {
    PETSC_ROOT = /home/regan/libraries/petsc_debug
}
CONFIG(release, debug|release): {
    PETSC_ROOT = /home/regan/libraries/petsc
}

# PETSc paths
PETSC_INCLUDES = $$PETSC_ROOT/include
PETSC_LIBS = -Wl,-rpath,$$PETSC_ROOT/lib -L$$PETSC_ROOT/lib -lpetsc

# compilers
QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx
