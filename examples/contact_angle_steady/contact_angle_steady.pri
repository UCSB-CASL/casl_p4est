CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d_steady.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d_steady.cpp
}

include($$PWD/../../qmake/libparcasl.pri)
