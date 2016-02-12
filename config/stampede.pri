# --------------------------------------------------------------
# Common settings for projects using on stampede supercomputer 
# --------------------------------------------------------------

# p4est
P4EST_INCLUDES_RELEASE = $$(TACC_P4EST_DIR)/include
P4EST_INCLUDES_DEBUG   = $$(TACC_P4EST_DIR)/include
P4EST_LIBS_RELEASE = -L$$(TACC_P4EST_LIB) -lp4est -lsc
P4EST_LIBS_DEBUG   = -L$$(TACC_P4EST_LIB) -lp4est -lsc

# PETSc
PETSC_INCLUDES_RELEASE = $$(TACC_PETSC_DIR)/include $$(TACC_PETSC_LIB)/../include
PETSC_INCLUDES_DEBUG   = $$(TACC_PETSC_DIR)/include $$(TACC_PETSC_LIB)/../include
PETSC_LIBS_RELEASE = -L$$(TACC_PETSC_LIB) -lpetsc
PETSC_LIBS_DEBUG = -L$$(TACC_PETSC_LIB) -lpetsc

# voro++
VORO_INCLUDES_RELEASE = $$(TACC_VORO_DIR)/include
VORO_INCLUDES_DEBUG   = $$(TACC_VORO_DIR)/include
VORO_LIBS_RELEASE     = -L$$(TACC_VORO_DIR)/lib -lvoro++
VORO_LIBS_DEBUG       = -L$$(TACC_VORO_DIR)/lib -lvoro++

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

