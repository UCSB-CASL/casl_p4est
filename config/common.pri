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
CONFIG -= console app_bundle qt qui

# Set the root directory for parcasl
PARCASL = $$PWD/..

# Settings related to logging and profiling
CONFIG(log): DEFINES += CASL_LOG_EVENTS
CONFIG(profile): {
    DEFINES += IPM_LOG_EVENTS
    QMAKE_LFLAGS += -g
    QMAKE_CFLAGS += -g
    QMAKE_CXXFLAGS += -g
}

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

INCLUDEPATH += $$PARCASL
DEPENDPATH  += $$PARCASL
OBJECTS_DIR = .obj

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
