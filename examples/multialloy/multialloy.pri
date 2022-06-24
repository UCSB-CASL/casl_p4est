CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d_multicomponent_stefan.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d.cpp
}

include($$PWD/../../qmake/libparcasl.pri)
