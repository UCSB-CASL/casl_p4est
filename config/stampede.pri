# --------------------------------------------------------------
# Configure options Stampede Supercomputer
# --------------------------------------------------------------

# p4est
P4EST_INCLUDES_DEBUG = $$(WORK)/soft/intel/p4est/debug/include
P4EST_INCLUDES_RELEASE = $$(WORK)/soft/intel/p4est/release/include
P4EST_LIBS_DEBUG = -L$$(WORK)/soft/intel/p4est/debug/lib -lp4est -lsc
P4EST_LIBS_RELEASE = -L$$(WORK)/soft/intel/p4est/release/lib -lp4est -lsc

# petsc -- WARNING, this is hardcoded, you might want to change to TACC macros?
TACC_PETSC_HOME = /opt/apps/intel13/mvapich2_1_9/petsc/3.4
TACC_PETSC_ARCH_RELEASE = sandybridge-cxx
TACC_PETSC_ARCH_DEBUG = sandybridge-cxxdebug
TACC_PETSC_LIB_RELEASE = $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_RELEASE/lib
TACC_PETSC_LIB_DEBUG = $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_DEBUG/lib

PETSC_INCLUDES_DEBUG = $$TACC_PETSC_HOME/include $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_DEBUG/include
PETSC_INCLUDES_RELEASE = $$TACC_PETSC_HOME/include $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_RELEASE/include
PETSC_LIBS_DEBUG = -Wl,-rpath,$$TACC_PETSC_LIB_DEBUG -L$$TACC_PETSC_LIB_DEBUG -lpetsc
PETSC_LIBS_RELEASE = -Wl,-rpath,$$TACC_PETSC_LIB_RELEASE -L$$TACC_PETSC_LIB_RELEASE -lpetsc

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

QMAKE_CFLAGS_RELEASE += -fast -vec-report0
QMAKE_CXXFLAGS_RELEASE += -fast -vec-report0
# for whatever reason intel compiler does not like it when -fast is passed to the linker
# and cannot find the petsc lib! "-fast" is for the most part equal to "-O3 -xHost -ipo"
QMAKE_LFLAGS_RELEASE += -vec-report0 -O3 -xHost -ipo

include(common.pri)
