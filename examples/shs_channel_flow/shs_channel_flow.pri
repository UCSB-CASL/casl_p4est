
CONFIG(2d, 2d|3d): {
  HEADERS += $$PWD/my_p4est_shs_channel.h
  SOURCES += $$PWD/main_2d.cpp
}

CONFIG(3d, 2d|3d): {
  HEADERS += $$PWD/my_p8est_shs_channel.h
  SOURCES += $$PWD/main_3d.cpp
}

include($$PWD/../../qmake/libparcasl.pri)
