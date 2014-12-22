CONFIG -= console
CONFIG -= app_bundle
CONFIG -= qt qui

# ----------------------------- Set configs parameters  ----------------------------- #
CONFIG += log

CONFIG(stampede, stampede|office): {
    CONFIG += intel
    CASL_P4EST = $$(WORK)/casl_p4est
    P4EST_DIR  = $$(WORK)/soft/p4est-dev

    # p4est
    P4EST_INCLUDES_DEBUG = $$P4EST_DIR/debug/include
    P4EST_INCLUDES_RELEASE = $$P4EST_DIR/release/include
    P4EST_LIBS_DEBUG = -Wl,-rpath,$$P4EST_DIR/debug/lib -L$$P4EST_DIR/debug/lib -lp4est -lsc
    P4EST_LIBS_RELEASE = -Wl,-rpath,$$P4EST_DIR/release/lib -L$$P4EST_DIR/release/lib -lp4est -lsc

    # petsc -- WARNING, this is hardcoded, you might want to change to TACC macros?
    TACC_PETSC_HOME = /opt/apps/intel13/mvapich2_1_9/petsc/3.4
    TACC_PETSC_ARCH_RELEASE = sandybridge-cxx
    TACC_PETSC_ARCH_DEBUG = sandybridge-cxxdebug
    TACC_PETSC_LIB_RELEASE = $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_RELEASE/lib
    TACC_PETSC_LIB_DEBUG = $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_DEBUG/lib

    PETSC_INCLUDES_DEBUG = $$TACC_PETSC_HOME/include $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_DEBUG/include
    PETSC_INCLUDES_RELEASE = $$TACC_PETSC_HOME/include $$TACC_PETSC_HOME/$$TACC_PETSC_ARCH_RELEASE/include
    PETSC_LIBS_DEBUG = -Wl,-rpath,$$TACC_PETSC_LIB_DEBUG -L$$TACC_PETSC_LIB_DEBUG -lpetsc
    PETSC_LIBS_RELEASE = -Wl,-rpath,$$TACC_PETSC_LIB_RELEASE -L$$TACC_PETSC_LIB_RELEASE -lpetsc
}

CONFIG(office, stampede|office): {
    CONFIG += gcc
    CASL_P4EST = $(HOME)/casl_p4est
    P4EST_DIR = /usr/local/p4est

    # p4est
    P4EST_INCLUDES_DEBUG = $$P4EST_DIR/debug/include
    P4EST_INCLUDES_RELEASE = $$P4EST_DIR/release/include
    P4EST_LIBS_DEBUG = -Wl,-rpath,$$P4EST_DIR/debug/lib -L$$P4EST_DIR/debug/lib -lp4est -lsc
    P4EST_LIBS_RELEASE = -Wl,-rpath,$$P4EST_DIR/release/lib -L$$P4EST_DIR/release/lib -lp4est -lsc

    # petsc
    PETSC_INCLUDES_DEBUG = /usr/local/petsc/debug/include
    PETSC_INCLUDES_RELEASE = /usr/local/petsc/release/include
    PETSC_LIBS_DEBUG = -L/usr/local/petsc/debug/lib -Wl,-rpath,/usr/local/petsc/debug/lib -lpetsc
    PETSC_LIBS_RELEASE = -L/usr/local/petsc/release/lib -Wl,-rpath,/usr/local/petsc/release/lib -lpetsc

    INCLUDEPATH += /usr/local/mpich3/include
}

CONFIG(nonblocking_notify): {
    DEFINES += ENABLE_NONBLOCKING_NOTIFY
    CONFIG(stampede, office|stampede) : {
        DEFINES += ENABLE_MPI_EXTENSIONS
    }
}

# --------------------------------- Define configs  --------------------------------- #
CONFIG(debug, debug|release): {
    INCLUDEPATH += $$P4EST_INCLUDES_DEBUG $$PETSC_INCLUDES_DEBUG
    LIBS += $$P4EST_LIBS_DEBUG $$PETSC_LIBS_DEBUG
    DEFINES += DEBUG CASL_THROWS P4EST_DEBUG

}

CONFIG(release, debug|release): {
    INCLUDEPATH += $$P4EST_INCLUDES_RELEASE $$PETSC_INCLUDES_RELEASE
    LIBS += $$P4EST_LIBS_RELEASE $$PETSC_LIBS_RELEASE
}

CONFIG(log): {
    DEFINES += CASL_LOG_EVENTS
}

CONFIG(profile): {
    DEFINES += IPM_LOG_EVENTS
}

CONFIG(2d, 2d|3d): {
TARGET = interpolation_2d
SOURCES += \
    interpolation_balanced_2d.cpp\
    $$CASL_P4EST/src/my_p4est_utils.cpp \
    $$CASL_P4EST/src/my_p4est_refine_coarsen.cpp\
    $$CASL_P4EST/src/my_p4est_vtk.c \
    $$CASL_P4EST/src/my_p4est_tools.c\
    $$CASL_P4EST/src/my_p4est_nodes.c \
    $$CASL_P4EST/src/my_p4est_interpolating_function_nonblocking.cpp \
    $$CASL_P4EST/src/my_p4est_interpolating_function_host.cpp \
    $$CASL_P4EST/src/cube2.cpp \
    $$CASL_P4EST/src/point2.cpp \
    $$CASL_P4EST/src/simplex2.cpp \
    $$CASL_P4EST/src/my_p4est_hierarchy.cpp \
    $$CASL_P4EST/src/my_p4est_node_neighbors.cpp \
    $$CASL_P4EST/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
    $$CASL_P4EST/src/my_p4est_log_wrappers.c \
    $$CASL_P4EST/src/petsc_logging.cpp \
    $$CASL_P4EST/src/Parser.cpp \
    $$CASL_P4EST/src/CASL_math.cpp

}

CONFIG(3d, 2d|3d): {
TARGET = interpolation_3d.non
SOURCES += \
    interpolation_balanced_3d.cpp\
    $$CASL_P4EST/src/my_p8est_utils.cpp\
    $$CASL_P4EST/src/my_p8est_refine_coarsen.cpp\
    $$CASL_P4EST/src/my_p8est_vtk.c \
    $$CASL_P4EST/src/my_p8est_tools.c\
    $$CASL_P4EST/src/my_p8est_nodes.c \
    $$CASL_P4EST/src/my_p8est_interpolating_function_nonblocking.cpp \
    $$CASL_P4EST/src/my_p8est_interpolating_function_host.cpp \
    $$CASL_P4EST/src/cube3.cpp \
    $$CASL_P4EST/src/point3.cpp \
    $$CASL_P4EST/src/simplex2.cpp \
    $$CASL_P4EST/src/my_p8est_hierarchy.cpp \
    $$CASL_P4EST/src/my_p8est_node_neighbors.cpp \
    $$CASL_P4EST/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
    $$CASL_P4EST/src/my_p8est_log_wrappers.c \
    $$CASL_P4EST/src/petsc_logging.cpp \
    $$CASL_P4EST/src/Parser.cpp \
    $$CASL_P4EST/src/CASL_math.cpp
}

# ------------------------------- Compiler Options ------------------------------- #
CONFIG(gdb):{
    QMAKE_LFLAGS += -g
    QMAKE_CFLAGS += -g
    QMAKE_CXXFLAGS += -g
} 

QMAKE_CC = mpicc
QMAKE_CXX = mpicxx
QMAKE_LINK = mpicxx

CONFIG(intel, intel|gcc):{
    QMAKE_CFLAGS_RELEASE += -fast -vec-report0
    QMAKE_CXXFLAGS_RELEASE += -fast -vec-report0
        # for whatever reason intel compiler does not like it when -fast is passed to the linker
        # and cannot find the petsc lib! "-fast" is for the most part equal to "-O3 -xHost -ipo"
    QMAKE_LFLAGS_RELEASE += -vec-report0 -O3 -xHost -ipo 
}

CONFIG(gcc, intel|gcc):{
    QMAKE_CFLAGS_RELEASE += -O3 -march="native"
    QMAKE_CXXFLAGS_RELEASE += -O3 -march="native"
    QMAKE_LFLAGS_RELEASE += -O3 -march="native"
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

# -------------------------------- Print CONFIG -------------------------------- #
message("Config options set by qmake:" $$CONFIG)
