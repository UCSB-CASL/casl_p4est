
CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_test_viscosity_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_test_viscosity_3d.cpp
}

include(../../qmake/libparcasl.pri)

