
CONFIG(2d, 2d|3d): {
  SOURCES += $$PWD/main_2d.cpp
}

CONFIG(3d, 2d|3d): {
  SOURCES += $$PWD/main_3d.cpp
}

include($$PWD/../../qmake/libparcasl.pri)

HEADERS += \
  $$PARCASL/examples/two_phase_flows_validation/test_cases_for_two_phase_flows.h

