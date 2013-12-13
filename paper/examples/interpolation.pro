TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt qui

P4EST_DIR  = /usr/local/p4est
CASL_P4EST = $$(HOME)/casl_p4est

# ----------------------------- Set configs parameters  ----------------------------- #
CONFIG += gcc log 2d

# --------------------------------- Define configs  --------------------------------- #
CONFIG(debug, debug|release): {
    INCLUDEPATH += $$P4EST_DIR/debug/include 
    LIBS += -L$$P4EST_DIR/debug/lib -lp4est -lsc
    DEFINES += DEBUG CASL_THROWS P4EST_DEBUG
}

CONFIG(release, debug|release): {
    INCLUDEPATH += $$P4EST_DIR/release/include
    LIBS += -L$$P4EST_DIR/release/lib -lp4est -lsc    
}

log{
    DEFINES += CASL_LOG_EVENTS
}

#DEFINES += GHOST_REMOTE_INTERPOLATION

2d{
TARGET = interpolation_2d
SOURCES +=\
    interpolation_2d.cpp\
    $$CASL_P4EST/src/my_p4est_utils.cpp\
    $$CASL_P4EST/src/my_p4est_refine_coarsen.cpp\
    $$CASL_P4EST/src/my_p4est_vtk.c \
    $$CASL_P4EST/src/my_p4est_tools.c\
    $$CASL_P4EST/src/my_p4est_nodes.c \
    $$CASL_P4EST/src/my_p4est_interpolating_function.cpp \
    $$CASL_P4EST/src/cube2.cpp \
    $$CASL_P4EST/src/point2.cpp \
    $$CASL_P4EST/src/simplex2.cpp \
    $$CASL_P4EST/src/my_p4est_hierarchy.cpp \
    $$CASL_P4EST/src/my_p4est_node_neighbors.cpp \
    $$CASL_P4EST/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
    $$CASL_P4EST/src/my_p4est_log_wrappers.cpp \
    $$CASL_P4EST/src/petsc_logging.cpp \
    $$CASL_P4EST/src/Parser.cpp
}

3d{
TARGET = interpolation_3d
SOURCES +=\
    interpolation_3d.cpp\
    $$CASL_P4EST/src/my_p8est_utils.cpp\
    $$CASL_P4EST/src/my_p8est_refine_coarsen.cpp\
    $$CASL_P4EST/src/my_p8est_vtk.c \
    $$CASL_P4EST/src/my_p8est_tools.c\
    $$CASL_P4EST/src/my_p8est_nodes.c \
    $$CASL_P4EST/src/my_p8est_interpolating_function.cpp \
    $$CASL_P4EST/src/cube3.cpp \
    $$CASL_P4EST/src/point3.cpp \
    $$CASL_P4EST/src/simplex2.cpp \
    $$CASL_P4EST/src/my_p8est_hierarchy.cpp \
    $$CASL_P4EST/src/my_p8est_node_neighbors.cpp \
    $$CASL_P4EST/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
    $$CASL_P4EST/src/my_p8est_log_wrappers.cpp \
    $$CASL_P4EST/src/petsc_logging.cpp \
    $$CASL_P4EST/src/Parser.cpp
}

# -------------------------------- PETSc Options  -------------------------------- #
# PETSc
INCLUDEPATH += $$(TACC_PETSC_LIB)/../../include $$(TACC_PETSC_LIB)/../include
LIBS += -Wl,-rpath,$$(TACC_PETSC_LIB) -L$$(TACC_PETSC_LIB) -lpetsc

# ------------------------------- Compiler Options ------------------------------- #
QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

CONFIG(intel, intel|gcc):{
    QMAKE_CFLAGS_RELEASE += -fast -vec-report
    QMAKE_CXXFLAGS_RELEASE += -fast -vec-report
}

CONFIG(gcc, intel|gcc):{
    QMAKE_CFLAGS_RELEASE += -O3 -march="native"
    QMAKE_CXXFLAGS_RELEASE += -O3 -march="native"
}

INCLUDEPATH += $$CASL_P4EST
DEPENDPATH += $$CASL_P4EST 
OBJECTS_DIR = .obj

# -------------------------------- Miscellaneous -------------------------------- #
DEFINES += "GIT_COMMIT_HASH_LONG=\\\"$$system(git rev-parse HEAD)\\\""
DEFINES += "GIT_COMMIT_HASH_SHORT=\\\"$$system(git rev-parse --short HEAD)\\\""

clean_data.target = clean_data
clean_data.commands = rm -rf *.vtk *.vtu *.pvtu *.visit *.csv *.bin *.dat
QMAKE_EXTRA_TARGETS += clean_data

commit.target = commit
commit.commands = git co run/stampede; \
     git ci -a -m \"Automatic commit\"; 

QMAKE_EXTRA_TARGETS += commit
