CONFIG -= console
CONFIG -= app_bundle
CONFIG -= qt qui

# ----------------------------- Set configs parameters  ----------------------------- #
CONFIG += 3d office

QMAKE_CC=/usr/local/bin/mpicc
QMAKE_CXX=/usr/local/bin/mpicxx
QMAKE_LINK=/usr/local/bin/mpicxx

CONFIG(stampede, stampede|office): {
    CONFIG += intel
    CASL_P4EST = $$(WORK)/casl_p4est

    # p4est
    P4EST_INCLUDES_DEBUG = $$(WORK)/soft/intel/p4est/debug/include
    P4EST_INCLUDES_RELEASE = $$(WORK)/soft/intel/p4est/release/include
    P4EST_LIBS_DEBUG = -L$$(WORK)/soft/intel/p4est/debug/lib -lp4est -lsc
    P4EST_LIBS_RELEASE = -L$$(WORK)/soft/intel/p4est/release/lib -lp4est -lsc

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

    DEFINES += ENABLE_MPI_EXTENSIONS
}

CONFIG(office, stampede|office): {
    CONFIG += gcc
    CASL_P4EST = $(HOME)/repos/parcasl

    # p4est
    P4EST_INCLUDES_DEBUG = /usr/local/include
    P4EST_INCLUDES_RELEASE = /usr/local/include
    P4EST_LIBS_DEBUG = -L/usr/local/lib -lp4est -lsc
    P4EST_LIBS_RELEASE = -L/usr/local/lib -lp4est -lsc

    # petsc
    PETSC_INCLUDES_DEBUG = /usr/local/include
    PETSC_INCLUDES_RELEASE = /usr/local/include
    PETSC_LIBS_DEBUG = -L/usr/local/lib -lpetsc
    PETSC_LIBS_RELEASE = -L/usr/local/lib -lpetsc

}

CONFIG(sc_notify): {
    DEFINES += ENABLE_NONBLOCKING_NOTIFY
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

log{
    DEFINES += CASL_LOG_EVENTS
}

CONFIG(profile): {
    DEFINES += IPM_LOG_EVENTS
}

CONFIG(2d, 2d|3d): {

CONFIG(profile): {
    TARGET = geometry_2d.prof
} else {
    TARGET = geometry_2d
}
SOURCES += \
#    oxygen_2d.cpp \
    charging_linear_2d.cpp \
    geometry_2d.cpp\
    $$CASL_P4EST/src/my_p4est_utils.cpp\
    $$CASL_P4EST/src/my_p4est_refine_coarsen.cpp\
    $$CASL_P4EST/src/my_p4est_semi_lagrangian.cpp\
    $$CASL_P4EST/src/my_p4est_level_set.cpp\
    $$CASL_P4EST/src/my_p4est_vtk.c \
    $$CASL_P4EST/src/my_p4est_tools.c\
    $$CASL_P4EST/src/my_p4est_nodes.c \
    $$CASL_P4EST/src/my_p4est_interpolation.cpp \
    $$CASL_P4EST/src/my_p4est_interpolation_nodes.cpp \
    $$CASL_P4EST/src/cube2.cpp \
    $$CASL_P4EST/src/point2.cpp \
    $$CASL_P4EST/src/simplex2.cpp \
    $$CASL_P4EST/src/my_p4est_hierarchy.cpp \
    $$CASL_P4EST/src/my_p4est_node_neighbors.cpp \
    $$CASL_P4EST/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
    $$CASL_P4EST/src/my_p4est_poisson_nodes.cpp \
    $$CASL_P4EST/src/my_p4est_log_wrappers.c \
    $$CASL_P4EST/src/petsc_logging.cpp \
    $$CASL_P4EST/src/Parser.cpp \
    $$CASL_P4EST/src/math.cpp
}

CONFIG(3d, 2d|3d): {
CONFIG(profile):{
        TARGET = geometry_3d.prof
} else {
        TARGET = geometry_3d
}
SOURCES += \
#    oxygen_3d.cpp \
    charging_linear_3d.cpp \
    geometry_3d.cpp\
    $$CASL_P4EST/src/my_p8est_utils.cpp\
    $$CASL_P4EST/src/my_p8est_refine_coarsen.cpp\
    $$CASL_P4EST/src/my_p8est_vtk.c \
    $$CASL_P4EST/src/my_p8est_semi_lagrangian.cpp\
    $$CASL_P4EST/src/my_p8est_level_set.cpp\
    $$CASL_P4EST/src/my_p8est_tools.c\
    $$CASL_P4EST/src/my_p8est_nodes.c \
    $$CASL_P4EST/src/my_p8est_interpolation.cpp \
    $$CASL_P4EST/src/my_p8est_interpolation_nodes.cpp \
    $$CASL_P4EST/src/cube3.cpp \
    $$CASL_P4EST/src/point3.cpp \
    $$CASL_P4EST/src/cube2.cpp \
    $$CASL_P4EST/src/point2.cpp \
    $$CASL_P4EST/src/simplex2.cpp \
    $$CASL_P4EST/src/my_p8est_hierarchy.cpp \
    $$CASL_P4EST/src/my_p8est_node_neighbors.cpp \
    $$CASL_P4EST/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
    $$CASL_P4EST/src/my_p8est_poisson_nodes.cpp \
    $$CASL_P4EST/src/my_p8est_log_wrappers.c \
    $$CASL_P4EST/src/petsc_logging.cpp \
    $$CASL_P4EST/src/Parser.cpp \
    $$CASL_P4EST/src/math.cpp
}

# ------------------------------- Compiler Options ------------------------------- #
CONFIG(profile):{
    QMAKE_LFLAGS += -g
    QMAKE_CFLAGS += -g
    QMAKE_CXXFLAGS += -g
} 

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
commit.commands = git co run/proposal; \
     git ci -a -m \"Automatic commit\"; 

QMAKE_EXTRA_TARGETS += commit
