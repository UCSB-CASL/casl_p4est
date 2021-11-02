
CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d.cpp
}

HEADERS += \
    $$PARCASL/examples/scalar_jump_tests/scalar_tests.h

#include(../../qmake/libparcasl.pri)
include(../../qmake/libparcasl_poisson_jump_cells.pri)
