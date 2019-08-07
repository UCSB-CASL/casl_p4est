PWD = $$CASL_P4EST/examples/navier_stokes

CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_flow_past_dendrites_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d.cpp
}

include(../../qmake/srclist.pri)
