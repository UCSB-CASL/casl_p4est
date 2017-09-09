# --------------------------------------------------------------
# Common settings for projects using on stampede supercomputer 
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = $$(TACC_PETSC_DIR)/include $$(TACC_PETSC_LIB)/../include
PETSC_INCLUDES_DEBUG   = $$(TACC_PETSC_DIR)/include $$(TACC_PETSC_LIB)/../include
PETSC_LIBS_RELEASE = -L$$(TACC_PETSC_LIB) -lpetsc
PETSC_LIBS_DEBUG = -L$$(TACC_PETSC_LIB) -lpetsc

# p4est
fix-memalign {
  P4EST_INCLUDES_RELEASE = $$(WORK)/soft/p4est-fix-memalign/release/include
  P4EST_INCLUDES_DEBUG   = $$(WORK)/soft/p4est-fix-memalign/debug/include
  P4EST_LIBS_RELEASE = -Wl,-rpath,$$(WORK)/soft/p4est-fix-memalign/release/lib -L$$(WORK)/soft/p4est-fix-memalign/release/lib -lp4est -lsc
  P4EST_LIBS_DEBUG = -Wl,-rpath,$$(WORK)/soft/p4est-fix-memalign/debug/lib -L$$(WORK)/soft/p4est-fix-memalign/debug/lib -lp4est -lsc
} else {
  P4EST_INCLUDES_RELEASE = $$(TACC_P4EST_DIR)/FAST/include
  P4EST_INCLUDES_DEBUG   = $$(TACC_P4EST_DIR)/DEBUG/include
  P4EST_LIBS_RELEASE = -Wl,-rpath,$$(TACC_P4EST_DIR)/FAST/lib -L$$(TACC_P4EST_DIR)/FAST/lib -lp4est -lsc
  P4EST_LIBS_DEBUG = -Wl,-rpath,$$(TACC_P4EST_DIR)/DEBUG/lib -L$$(TACC_P4EST_DIR)/DEBUG/lib -lp4est -lsc  
}

# voro++
VORO_INCLUDES_RELEASE = $$(TACC_VORO_DIR)/include
VORO_INCLUDES_DEBUG   = $$(TACC_VORO_DIR)/include
VORO_LIBS_RELEASE     = -L$$(TACC_VORO_DIR)/lib -lvoro++
VORO_LIBS_DEBUG       = -L$$(TACC_VORO_DIR)/lib -lvoro++

# Boost
BOOST_INCLUDE = $$(TACC_BOOST_INC)
BOOST_LIBS = $$(TACC_BOOST_LIB)

QMAKE_CC=mpicc
QMAKE_CXX=mpicxx
QMAKE_LINK=mpicxx

QMAKE_CXXFLAGS += -std=c++11 -Wall
QMAKE_CCFLAGS  += -std=c++11 -Wall
QMAKE_LFAGS    += -std=c++11 -Wall

release {
  OPT_FLAGS = -xHost -O3 -no-prec-div -ip -fp-model fast=2
  QMAKE_CCFLAGS += $$OPT_FLAGS
  QMAKE_CXXFLAGS += $$OPT_FLAGS
  QMAKE_LFLAGS += $$OPT_FLAGS
}
include(common.pri)

