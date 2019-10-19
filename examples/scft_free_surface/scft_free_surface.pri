CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d_free_surface.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d_free_surface.cpp
}

include($$PWD/../../qmake/libparcasl.pri)
