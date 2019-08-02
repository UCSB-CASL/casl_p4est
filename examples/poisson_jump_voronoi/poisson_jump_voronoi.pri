CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_jump_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_jump_3d.cpp
}

include(../../qmake/srclist.pri)

