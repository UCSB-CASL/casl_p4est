# --------------------------------------------------------------
# Common settings for projects using on stampede supercomputer 
# --------------------------------------------------------------

# PETSc
PETSC_DIR_DBG = /home/rochi/libraries/petsc_debug
PETSC_DIR_RLS = /home/rochi/libraries/petsc_release

PETSC_INCLUDES_RELEASE = $$PETSC_DIR_RLS/include
PETSC_INCLUDES_DEBUG   = $$PETSC_DIR_DBG/include
PETSC_LIBS_RELEASE = -Wl,-rpath,$$PETSC_DIR_RLS/lib -L$$PETSC_DIR_RLS/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,$$PETSC_DIR_DBG/lib -L$$PETSC_DIR_DBG/lib -lpetsc

# p4est
P4EST_DIR_DBG = /home/rochi/libraries/p4est_debug
P4EST_DIR_RLS = /home/rochi/libraries/p4est_release

P4EST_INCLUDES_RELEASE = $$P4EST_DIR_RLS/include
P4EST_INCLUDES_DEBUG   = $$P4EST_DIR_DBG/include
P4EST_LIBS_RELEASE = -Wl,-rpath,$$P4EST_DIR_RLS/lib -L$$P4EST_DIR_RLS/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,$$P4EST_DIR_DBG/lib -L$$P4EST_DIR_DBG/lib -lp4est -lsc

# voro++

VORO_INCLUDES_RELEASE = /usr/local/include/voro++
VORO_INCLUDES_DEBUG   = /usr/local/include/voro++
VORO_LIBS_RELEASE     = -Wl,-rpath,/usr/local/bin -L/usr/local/bin -lvoro++
VORO_LIBS_DEBUG       = -Wl,-rpath,/usr/local/bin -L/usr/local/bin -lvoro++

# Boost
BOOST_DIR       = /home/rochi/libraries/boost/release
BOOST_INCLUDES  = $$BOOST_DIR/include
BOOST_LIBS      = -Wl,-rpath,$$BOOST_DIR/lib -L$$BOOST_DIR/lib -lboost_filesystem

QMAKE_CC=/usr/local/bin/mpicc
QMAKE_CXX=/usr/local/bin/mpicxx
QMAKE_LINK=/usr/local/bin/mpicxx

