# Disable useless qt packages
CONFIG -= console app_bundle qt qui qt_framework

INCLUDEPATH += \
    $$PETSC_INCLUDES

LIBS += \
    $$PETSC_LIBS

# Settings related to release/debug
CONFIG(debug, debug|release): {
    DEFINES += DEBUG
}

OBJECTS_DIR = .obj

INCLUDEPATH += $$PARCASL
DEPENDPATH  += $$PARCASL

# enable C++11
QMAKE_CXXFLAGS += -std=c++11
QMAKE_CCFLAGS  += -std=c++11
QMAKE_LFAGS    += -std=c++11
