# prepare sources that need to be compiled
#INCLUDE_FLAGS += -I$(CASL_DIR)

ifeq ($(BUILD_UTILITIES), YES)
	vpath %.cpp $(CASL_DIR)/lib/utilities
	SRCS += Macros.cpp
endif

ifeq ($(BUILD_ALGEBRA), YES)
	vpath %.cpp $(CASL_DIR)/lib/algebra
	SRCS += Matrix.cpp MatrixFull.cpp 
	ifeq ($(CASL_HAVE_PETSC), YES)
		SRCS += petscLinearSolver.cpp
	endif
endif

ifeq ($(BUILD_QUADTREE), YES)
	vpath Q%.cpp $(CASL_DIR)/lib/amr
	vpath Q%.cpp $(CASL_DIR)/lib/fastmarching
	SRCS += QuadTree.cpp QuadTree_FDM.cpp QuadTree_FVM.cpp Quad_Ngbd_Nodes_of_Node.cpp QuadTree_extension.cpp QuadTree_FastMarching.cpp
endif

ifeq ($(BUILD_OCTREE), YES)
	vpath O%.cpp $(CASL_DIR)/lib/amr
	vpath O%.cpp $(CASL_DIR)/lib/fastmarching
	SRCS += OcTree.cpp Oct_Ngbd_Nodes_of_Node.cpp Octree_FDM.cpp Octree_IP.cpp OcTree_extension.cpp OcTree_FastMarching.cpp  
endif

ifeq ($(BUILD_ARRAYS), YES)
	vpath %.cpp $(CASL_DIR)/lib/array
	SRCS += ArrayV_2D.cpp ArrayV_3D.cpp 
endif

ifeq ($(BUILD_IO), YES)
	vpath %.cpp $(CASL_DIR)/lib/io
	SRCS += vtuWriter.cpp xdmfWriter.cpp Parser.cpp
endif

ifeq ($(BUILD_EK), YES)
	vpath %.cpp $(CASL_DIR)/lib/ek
	SRCS += pbSolver_2d_FVM.cpp pnpSolver_2d_FVM.cpp
endif

ifeq ($(BUILD_GEOMETRY), YES)
	vpath %.cpp $(CASL_DIR)/lib/geometry
	SRCS += Point2.cpp Point2_new.cpp Point3.cpp
endif

ifeq ($(BUILD_RAYTRACING), YES)
	vpath %.cpp $(CASL_DIR)/lib/raytracing
	SRCS += Vector3.cpp
endif

ifeq ($(CASL_HAVE_PETSC), YES)
	CXX_DEFINES  += -DCASL_USE_PETSC
	CASL_HAVE_MPI = YES 

	PETSC_DIR = $(PETSC_HOME_DIR)

	# ifeq ($(BUILD_TYPE), debug)
	# 	PETSC_ARCH = $(PETSC_ARCH_DEBUG)
	# else
	# 	PETSC_ARCH = $(PETSC_ARCH_OPT)
	# endif

	include $(PETSC_HOME_DIR)/conf/variables
	include $(PETSC_HOME_DIR)/conf/rules

	LINK_LIBS += $(PETSC_LIB)
	INCLUDE_FLAGS += $(PETSC_INCLUDE)
#	INCLUDE_FLAGS += -I$(PETSC_INC_DIR)/include -I$(PETSC_INC_DIR)/$(PETSC_ARCH)/include
endif

ifeq ($(CASL_HAVE_P4EST), YES)
	CASL_HAVE_MPI = YES

	ifeq ($(BUILD_TYPE), debug)
		P4EST_ARCH  = debug
	else
		P4EST_ARCH  = release
	endif

	INCLUDE_FLAGS += -I$(P4EST_DIR)/include
	P4EST_LIBS = -L$(P4EST_DIR)/lib -lp4est -lsc
	LINK_LIBS += $(P4EST_LIBS)
endif


# Flags
CXX_ARCH_FLAGS      = -m64
CXX_FLAGS_debug     = -O0 -g
CXX_FLAGS_OPT_GNU   = -O2 -march=native
CXX_FLAGS_OPT_INTEL = -O2 -xHOST -ip
CXX_WARN_FLAGS_ON   = -W -Wall -Wextra 
CXX_WARN_FLAGS_OFF  = -w

CUDA_ARCH_FLAGS     = -arch sm_13
CUDA_FLAGS_debug    =  $(CXX_FLAGS_debug) -G
CUDA_FLAGS_OPT      = -O2
CUDA_WARN_FLAGS_ON  = $(CXX_WARN_FLAGS_ON)
CUDA_WARN_FLAGS_OFF = $(CXX_WARN_FLAGS_OFF)

# configure 
ifeq ($(CXX_COMPILER_TYPE), GNU)
	ifeq ($(CASL_HAVE_MPI), YES)
		CXX = mpicxx
	else
		CXX = g++
	endif
endif
ifeq ($(CXX_COMPILER_TYPE), INTEL)
	ifeq ($(CASL_HAVE_MPI), YES)
		CXX = mpicxx
	else
		CXX = icpc
	endif
endif

CXX_FULL_PATH = `which $(CXX)`

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
endif

ifeq ($(CASL_HAVE_OPENMP), YES)
	ifeq ($(CXX_COMPILER_TYPE), GNU)
		CXX_FLAGS += -fopenmp
	endif
	ifeq ($(CXX_COMPILER_TYPE), INTEL)
		CXX_FLAGS += -openmp
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
