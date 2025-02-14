CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d_dsa.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d_dsa.cpp
}

include($$PWD/../../qmake/libparcasl.pri)
