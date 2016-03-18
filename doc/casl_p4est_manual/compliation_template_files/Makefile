####### Compiler, tools and options

build = 2d
debug = 0

ifeq ($(build),2d)
	OBJDIR = .obj_2d
	TARGET = p4est_exec_2d
else
	OBJDIR = .obj_3d
	TARGET = p4est_exec_3d
endif

ifeq ($(debug),1)
	P4EST_ARCH=debug
	PETSC_ARCH=debug
	CFLAGS += -g
	CXXFLAGS += -g
else
	P4EST_ARCH=release
	PETSC_ARCH=release
endif

# THESE NEED TO BE REPLACE WITH YOUR ENVIRONMENT DEPENDENT INFORMATION
P4EST_DIR = $(HOME)/softwares/build/p4est
PETSC_DIR = $(HOME)/softwares/build/petsc

# p4est
INCPATH += -I$(P4EST_DIR)/$(P4EST_ARCH)/include
LIBS += -Wl,-rpath,$(P4EST_DIR)/$(P4EST_ARCH)/lib -L$(P4EST_DIR)/$(P4EST_ARCH)/lib -lp4est -lsc

# petsc
INCPATH += -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LIBS += -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

# casl p4est
CASL_P4EST = $(HOME)/code/CASL_p4est
INCPATH += -I$(CASL_P4EST)

# you can give some #define flags to you program, here this is equivalent
# to giving the "#define COMET" preprocessing directive
#CFLAGS += -DCOMET
#CXXFLAGS += -DCOMET

CC = @echo compiling $< && mpicc
CXX = @echo compiling $< && mpic++
LINK = @echo linking $@ && mpic++

vpath %.c   $(CASL_P4EST)/src
vpath %.cpp $(CASL_P4EST)/src
vpath %.cpp $(CASL_P4EST)/examples/nodes

####### Files
ifeq ($(build),2d)
SOURCES = \
				$(CASL_P4EST)/examples/nodes/main_test_nodes_2d.cpp \
				$(CASL_P4EST)/src/my_p4est_utils.cpp \
				$(CASL_P4EST)/src/my_p4est_refine_coarsen.cpp \
				$(CASL_P4EST)/src/my_p4est_tools.c \
				$(CASL_P4EST)/src/my_p4est_nodes.c \
				$(CASL_P4EST)/src/cube2.cpp \
				$(CASL_P4EST)/src/point2.cpp \
				$(CASL_P4EST)/src/simplex2.cpp \
				$(CASL_P4EST)/src/my_p4est_log_wrappers.c \
				$(CASL_P4EST)/src/petsc_logging.cpp \
				$(CASL_P4EST)/src/Parser.cpp \
				$(CASL_P4EST)/src/math.cpp \
				$(CASL_P4EST)/src/my_p4est_vtk.cpp \
				$(CASL_P4EST)/src/my_p4est_hierarchy.cpp \
				$(CASL_P4EST)/src/my_p4est_node_neighbors.cpp \
				$(CASL_P4EST)/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
				$(CASL_P4EST)/src/my_p4est_interpolation.cpp \
				$(CASL_P4EST)/src/my_p4est_interpolation_nodes.cpp \
				$(CASL_P4EST)/src/my_p4est_level_set.cpp \
				$(CASL_P4EST)/src/my_p4est_poisson_nodes.cpp

else

SOURCES = \
				$(CASL_P4EST)/examples/nodes/main_test_nodes_3d.cpp \
				$(CASL_P4EST)/src/my_p8est_utils.cpp \
				$(CASL_P4EST)/src/my_p8est_refine_coarsen.cpp \
				$(CASL_P4EST)/src/my_p8est_tools.c \
				$(CASL_P4EST)/src/my_p8est_nodes.c \
				$(CASL_P4EST)/src/my_p8est_log_wrappers.c \
				$(CASL_P4EST)/src/petsc_logging.cpp \
				$(CASL_P4EST)/src/Parser.cpp \
				$(CASL_P4EST)/src/cube2.cpp \
				$(CASL_P4EST)/src/cube3.cpp \
				$(CASL_P4EST)/src/point2.cpp \
				$(CASL_P4EST)/src/point3.cpp \
				$(CASL_P4EST)/src/simplex2.cpp \
				$(CASL_P4EST)/src/math.cpp \
				$(CASL_P4EST)/src/my_p8est_vtk.cpp \
				$(CASL_P4EST)/src/my_p8est_hierarchy.cpp \
				$(CASL_P4EST)/src/my_p8est_node_neighbors.cpp \
				$(CASL_P4EST)/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
				$(CASL_P4EST)/src/my_p8est_interpolation.cpp \
				$(CASL_P4EST)/src/my_p8est_interpolation_nodes.cpp \
				$(CASL_P4EST)/src/my_p8est_level_set.cpp \
				$(CASL_P4EST)/src/my_p8est_poisson_nodes.cpp
endif

STRIP_SRC = $(notdir $(SOURCES))
OBJECTS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(STRIP_SRC)))	$(patsubst %.c, $(OBJDIR)/%.o, $(filter %.c, $(STRIP_SRC)))

####### Implicit rules

$(OBJDIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

$(OBJDIR)/%.o: %.c
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

#all: $(SOURCES) $(TARGET)
all: $(TARGET)

$(TARGET):  $(OBJECTS)
	$(LINK) -o $(TARGET) $(OBJECTS) $(LIBS) $(LFLAGS)

clean: 
	rm -rf $(OBJDIR)/*.o $(TARGET)
