# --------------------------------------------------------------
# Common settings for projects using on stampede supercomputer 
# --------------------------------------------------------------

# PETSc
PETSC_DIR_DBG = /home/rochi/libraries/petsc_debug
PETSC_DIR_RLS = /home/rochi/libraries/petsc_release

PETSC_INCLUDES_RELEASE = $$PETSC_DIR_RLS/include
PETSC_INCLUDES_DEBUG   = $$PETSC_DIR_DBG/include
PETSC_LIBS_RELEASE = -L$$PETSC_DIR_RLS/lib -lpetsc
PETSC_LIBS_DEBUG   = -L$$PETSC_DIR_DBG/lib -lpetsc

# p4est
P4EST_DIR_DBG = /home/rochi/libraries/p4est_debug
P4EST_DIR_RLS = /home/rochi/libraries/p4est_release

P4EST_INCLUDES_RELEASE = $$P4EST_DIR_RLS/include
P4EST_INCLUDES_DEBUG   = $$P4EST_DIR_DBG/include
P4EST_LIBS_RELEASE = -Wl,-rpath,$$P4EST_DIR_RLS/lib -L$$P4EST_DIR_RLS/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,$$P4EST_DIR_DBG/lib -L$$P4EST_DIR_DBG/lib -lp4est -lsc

# voro++
VORO_DIR_DBG = /usr/local/include/voro++
VORO_DIR_RLS = /usr/local/include/voro++

VORO_INCLUDES_RELEASE = $$VORO_DIR_RLS/include
VORO_INCLUDES_DEBUG   = $$VORO_DIR_DBG/include
VORO_LIBS_RELEASE     = -L$$VORO_DIR_RLS/lib -lvoro++
VORO_LIBS_DEBUG       = -L$$VORO_DIR_DBG/lib -lvoro++

# Boost
#BOOST_INCLUDES = /home/dbochkov/Software/Boost/boost_1_70_0

QMAKE_CC=mpicc
QMAKE_CXX=mpicxx
QMAKE_LINK=mpicxx

