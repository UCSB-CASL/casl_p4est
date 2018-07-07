# --------------------------------------------------------------
# Common settings for projects using generic linux builds
# --------------------------------------------------------------

# PETSc
PETSC_INCLUDES_RELEASE = /home/pouria/libs/Petsc/include #/usr/local/include
PETSC_INCLUDES_DEBUG   = /home/pouria/libs/Petsc/include #/usr/local/include
PETSC_LIBS_RELEASE = -Wl,-rpath,/home/pouria/libs/Petsc/lib -L/home/pouria/libs/Petsc/lib -lpetsc
PETSC_LIBS_DEBUG   = -Wl,-rpath,/home/pouria/libs/Petsc/lib -L/home/pouria/libs/Petsc/lib -lpetsc

# p4est
P4EST_INCLUDES_RELEASE = /home/pouria/libs/p4est_install/include
P4EST_INCLUDES_DEBUG   = /home/pouria/libs/p4est_install/include
P4EST_LIBS_RELEASE = -Wl,-rpath,/home/pouria/libs/p4est_install/lib -L/home/pouria/libs/p4est_install/lib -lp4est -lsc
P4EST_LIBS_DEBUG   = -Wl,-rpath,/home/pouria/libs/p4est_install/lib -L/home/pouria/libs/p4est_install/lib -lp4est -lsc

# voro++
VORO_INCLUDES_RELEASE = /home/pouria/libs/voro++/include/voro++   #/usr/local/include/voro++
VORO_INCLUDES_DEBUG   = /home/pouria/libs/voro++/include/voro++   #/usr/local/include/voro++
VORO_LIBS_RELEASE     = /home/pouria/libs/voro++/lib/libvoro++.a   #/usr/local/lib/libvoro++.a
VORO_LIBS_DEBUG       = /home/pouria/libs/voro++/lib/libvoro++.a   #/usr/local/lib/libvoro++.a

# qhull
QHULL_INCLUDES = /home/pouria/libs/qhull_github/qhull/src
QHULL_LIBS = -Wl,-rpath,/home/pouria/libs/qhull_github/qhull/lib -L/home/pouria/libs/qhull_github/qhull/lib -lqhull_r

#boost
BOOST_INCLUDES_RELEASE= -I/home/pouria/libs/BOOST/include
BOOST_LIBS_RELEASE    = -L/home/pouria/libs/BOOST/lib

# ANN
ANN_INCLUDE = /home/pouria/Documents/ann_1.1.2/include/
ANN_LIBS = -Wl,-rpath,/home/pouria/Documents/ann_1.1.2/lib/ -L/home/pouria/Documents/ann_1.1.2/lib/ -lANN

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

include(common.pri)

