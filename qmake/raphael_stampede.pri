# --------------------------------------------------------------
#    Paths to external libraries to be linked to casl_p4est
# --------------------------------------------------------------

DEFINES += STAMPEDE

#HDF5 (This is required by their petsc/3.11 installation here, for some reason).
INCLUDEPATH   += $$(TACC_HDF5_INC)
LIBS          += -Wl,-rpath,$$(TACC_HDF5_LIB) -L$$(TACC_HDF5_LIB)

# PetSc

# include files are split into separate folders
INCLUDEPATH   += $$(TACC_PETSC_DIR)/include	# common to all installation (debug or not, complex values or nor, 64 bit integers or nor, etc.)
INCLUDEPATH   += $$(TACC_PETSC_INC)         # specific to the module you have loaded
LIBS          += -Wl,-rpath,$$(TACC_PETSC_LIB) -L$$(TACC_PETSC_LIB) -lpetsc

# p4est

INCLUDEPATH   += $$(TACC_P4EST_INC)
LIBS          += -Wl,-rpath,$$(TACC_P4EST_LIB) -L$$(TACC_P4EST_LIB) -lp4est -lsc

# voro++
VORO_DIR      = /work2/04965/tg842642/stampede2/libs/voro++

INCLUDEPATH   += $$(VORO_DIR)/include/voro++
LIBS          += -Wl,-rpath,$$(VORO_DIR)/lib -L$$(VORO_DIR)/lib -lvoro++

# boost (needed for epitaxy only, ok if not provided otherwise)
INCLUDEPATH   += $$(TACC_BOOST_INC)

# Intel Math Kernel Library --> blas/lapack(e) optimized for intel architecture
INCLUDEPATH   += $$(TACC_MKL_INC)
LIBS          += -L$$(TACC_MKL_LIB) -lmkl_intel_lp64 # seems to be enough, even for 64-bit integers (just load the appropriate 64-bit version of petsc in that case)

QMAKE_CC      = /opt/apps/intel18/impi/18.0.2/bin/mpicc
QMAKE_CXX     = /opt/apps/intel18/impi/18.0.2/bin/mpicxx
QMAKE_LINK    = /opt/apps/intel18/impi/18.0.2/bin/mpicxx

