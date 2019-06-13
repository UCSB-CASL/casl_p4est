TARGET = poisson_nodes_mls

CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d.cpp
}

include(../../qmake/srclist.pri)
