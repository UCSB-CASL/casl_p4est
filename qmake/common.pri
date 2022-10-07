# --------------------------------------------------------------
# Common settings useful for all projects
# --------------------------------------------------------------
# Set the correct version of OSX if on a mac
macx: {
    OSX_VERSION_MAJOR = $$system("sw_vers -productVersion | awk -F '.' '{print $1}'")
    OSX_VERSION_MINOR = $$system("sw_vers -productVersion | awk -F '.' '{print $2}'")
    QMAKE_MACOSX_DEPLOYMENT_TARGET = $${OSX_VERSION_MAJOR}.$${OSX_VERSION_MINOR}
}

# Disable useless qt packages
CONFIG -= console app_bundle qt qui qt_framework

# Set the root directory for parcasl
PARCASL = $$PWD/..

# Settings related to logging and profiling
CONFIG(log): DEFINES += CASL_LOG_EVENTS
CONFIG(profile): {
    DEFINES += IPM_LOG_EVENTS
    QMAKE_LFLAGS += -g
    QMAKE_CFLAGS += -g
    QMAKE_CXXFLAGS += -g
    QMAKE_CXXFLAGS += c++11
}
#QMAKE_CXXFLAGS+=-shared
#QMAKE_CXXFLAGS+= -fPIC
#QMAKE_CFLAGS+= -fPIC

INCLUDEPATH += \
    $$BOOST_INCLUDES  \
    $$MATLAB_INCLUDES \
    $$MPI_INCLUDES    \
    $$LAPACKE_INCLUDE

LIBS += \
    $$MATLAB_LIBS \
    $$MPI_LIBS    \
    $$LAPACKE_LIBS

# Settings related to release/debug
CONFIG(debug, debug|release): {
    INCLUDEPATH += \
        $$P4EST_INCLUDES_DEBUG \
        $$PETSC_INCLUDES_DEBUG \
        $$VORO_INCLUDES_DEBUG

    LIBS += \
        $$P4EST_LIBS_DEBUG \
        $$PETSC_LIBS_DEBUG \
        $$VORO_LIBS_DEBUG

    DEFINES += DEBUG CASL_THROWS P4EST_DEBUG
}

CONFIG(release, debug|release): {
    INCLUDEPATH += \
        $$P4EST_INCLUDES_RELEASE \
        $$PETSC_INCLUDES_RELEASE \
        $$VORO_INCLUDES_RELEASE

    LIBS += \
        $$P4EST_LIBS_RELEASE \
        $$PETSC_LIBS_RELEASE \
        $$VORO_LIBS_RELEASE
}

exists($$MATLAB_INCLUDES/engine.h) {
  DEFINES += MATLAB_PROVIDED
}

INCLUDEPATH += $$PARCASL
DEPENDPATH  += $$PARCASL
OBJECTS_DIR = .obj

# enable C++11
#QMAKE_CXXFLAGS += -std=c++11
#QMAKE_CFLAGS   += -std=c++11 # [Raphael] this is irrelevant for the C compiler...
QMAKE_LFLAGS   += -std=c++11
QMAKE_CXXFLAGS+= -fPIC
QMAKE_CFLAGS+= -fPIC


contains(DEFINES, STAMPEDE) { # i.e. if DEFINES += STAMPEDE was added to the user-specific .pri file
QMAKE_CXXFLAGS  += "-xCORE-AVX2 -axCOMMON-AVX512,MIC-AVX512" # compiler issues were found when using the recommended $(TACC_VEC_FLAGS) with Intel 18 compilers
QMAKE_CFLAGS    += "-xCORE-AVX2 -axCOMMON-AVX512,MIC-AVX512" # compiler issues were found when using the recommended $(TACC_VEC_FLAGS) with Intel 18 compilers
QMAKE_LFLAGS    += "-xCORE-AVX2 -axCOMMON-AVX512,MIC-AVX512" # compiler issues were found when using the recommended $(TACC_VEC_FLAGS) with Intel 18 compilers
# --> enables execution of the code on KNL as well as SKX nodes on Stampede2
}

# Miscellaneous
DEFINES += "GIT_COMMIT_HASH_LONG=\\\"$$system(git rev-parse HEAD)\\\""
DEFINES += "GIT_COMMIT_HASH_SHORT=\\\"$$system(git rev-parse --short HEAD)\\\""

clean_data.target = clean_data
clean_data.commands = rm -rf *.vtk* *.vtu* *.pvtu* *.visit* *.csv* *.bin* *.dat*
QMAKE_EXTRA_TARGETS += clean_data

commit.target = commit
commit.commands = git co run/proposal; \
     git ci -a -m \"Automatic commit\";

QMAKE_EXTRA_TARGETS += commit
