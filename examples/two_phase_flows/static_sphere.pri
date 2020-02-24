
CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_static_sphere_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_static_sphere_3d.cpp
}

include(../../qmake/libparcasl.pri)

