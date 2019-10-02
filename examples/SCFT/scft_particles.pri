CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d_particles.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d_particles.cpp
}

include(../../qmake/srclist.pri)
