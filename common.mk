# Configure PETSc
ifeq ($(TACC_CLUSTER), YES)
	PETSC_DIR = $(TACC_PETSC_DIR)
	include $(PETSC_DIR)conf/variables
	include $(PETSC_DIR)conf/rules
	ifeq ($(PETSC_INCLUDE),)
		PETSC_INCLUDE = -I$(PETSC_DIR)include -I$(PETSC_DIR)$(PETSC_ARCH)/include 
	endif
else
ifeq ($(BGQ_CLUSTER), YES)
	# assume petsc 3.6 is used
	include $(PETSC_DIR)/lib/petsc/conf/variables
	include $(PETSC_DIR)/lib/petsc/conf/rules
	PETSC_INCLUDE = -I$(PETSC_DIR)/include
else
	include $(PETSC_DIR)/conf/variables
	include $(PETSC_DIR)/conf/rules
	ifeq ($(PETSC_INCLUDE),)
		PETSC_INCLUDE = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include 
	endif
endif
endif


LINK_LIBS += $(PETSC_LIB)
INCLUDE_FLAGS += $(PETSC_INCLUDE)

# Configure p4est
INCLUDE_FLAGS += -I$(P4EST_DIR)/include
P4EST_LIBS = -L$(P4EST_DIR)/lib -lp4est -lsc
LINK_LIBS += $(P4EST_LIBS)

# Compilers and flags
#
ifeq ($(BGQ_CLUSTER), YES)
# This bracket is for Blue Gene/Q systems
# Compilers: choosing the reentrant version; drop the _r if not needed
MPICC = mpixlc_r
MPICXX = mpixlcxx_r
CXX_ARCH_FLAGS      = -qarch=qp -qtune=qp
CXX_FLAGS_debug     = -O0
CXX_WARN_FLAGS_ON   =
CXX_WARN_FLAGS_OFF  =
else
MPICC  = mpicc
MPICXX = mpicxx
CXX_ARCH_FLAGS      = -m64
CXX_FLAGS_debug     = -O0 -g
CXX_WARN_FLAGS_ON   = -W -Wall -Wextra 
CXX_WARN_FLAGS_OFF  = -w
endif
CXX_FLAGS_OPT_GNU   = -O3 -march=native
CXX_FLAGS_OPT_INTEL = -O3 -xHOST -ip
CXX_FLAGS_OPT_BGQ   = -O3 -qstrict

# configure 

CXX_FULL_PATH = `which $(MPICXX)`

CXX_FLAGS = $(CXX_ARCH_FLAGS)
ifeq ($(BUILD_TYPE), debug)
	CXX_FLAGS += $(CXX_FLAGS_debug)
else
	ifeq ($(CXX_COMPILER_TYPE), GNU)
		CXX_FLAGS += $(CXX_FLAGS_OPT_GNU)
	endif
	ifeq ($(CXX_COMPILER_TYPE), INTEL)
		CXX_FLAGS += $(CXX_FLAGS_OPT_INTEL)
	endif
	ifeq ($(CXX_COMPILER_TYPE), BGQ)
		CXX_FLAGS += $(CXX_FLAGS_OPT_BGQ)
	endif
endif

ifeq ($(WITH_DEBUG_SYMBOLS), YES)
	CXX_FLAGS += -g
endif

ifeq ($(WITH_OPENMP), YES)
	ifeq ($(CXX_COMPILER_TYPE), GNU)
		CXX_FLAGS += -fopenmp
	endif
	ifeq ($(CXX_COMPILER_TYPE), INTEL)
		CXX_FLAGS += -openmp
	endif
	ifeq ($(CXX_COMPILER_TYPE), BGQ)
		CXX_FLAGS += -qsmp=omp -qthreaded
	endif
endif

ifeq ($(PRINT_WARNINGS), YES)
	CXX_FLAGS += $(CXX_WARN_FLAGS_ON)
else
	CXX_FLAGS += $(CXX_WARN_FLAGS_OFF)
endif
CXX_FLAGS  += $(CXX_DEFINES) $(CXX_EXTRA_FLAGS)
LINK_FLAGS  = $(CXX_FLAGS)

# Some useful commands
RM    = rm -rf 
MKDIR = mkdir -p
ifeq ($(PRINT_MAKE_OUTPUT), YES)
	MAKE  = make
else 
	MAKE  = make --silent
endif
