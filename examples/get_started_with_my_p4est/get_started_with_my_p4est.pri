CONFIG(2d, 2d|3d): {
  SOURCES +=  $$PARCASL/src/cube2.cpp \
              $$PARCASL/src/my_p4est_hierarchy.cpp \
              $$PARCASL/src/my_p4est_log_wrappers.c \
              $$PARCASL/src/my_p4est_node_neighbors.cpp \
              $$PARCASL/src/my_p4est_nodes.c \
              $$PARCASL/src/my_p4est_quad_neighbor_nodes_of_node.cpp \
              $$PARCASL/src/my_p4est_refine_coarsen.cpp \
              $$PARCASL/src/my_p4est_tools.c \
              $$PARCASL/src/my_p4est_utils.cpp \
              $$PARCASL/src/my_p4est_vtk.cpp \
              $$PARCASL/src/simplex2.cpp \
              $$PARCASL/src/point2.cpp \
              main_2d.cpp
  HEADERS +=  $$PARCASL/src/cube2.h \
              $$PARCASL/src/my_p4est_hierarchy.h \
              $$PARCASL/src/my_p4est_log_wrappers.h \
              $$PARCASL/src/my_p4est_node_neighbors.h \
              $$PARCASL/src/my_p4est_nodes.h \
              $$PARCASL/src/my_p4est_quad_neighbor_nodes_of_node.h \
              $$PARCASL/src/my_p4est_refine_coarsen.h \
              $$PARCASL/src/my_p4est_tools.h \
              $$PARCASL/src/my_p4est_utils.h \
              $$PARCASL/src/my_p4est_vtk.h \
              $$PARCASL/src/simplex2.h \
              $$PARCASL/src/point2.h
}

CONFIG(3d, 2d|3d): {
  SOURCES +=  $$PARCASL/src/cube3.cpp \
              $$PARCASL/src/my_p8est_hierarchy.cpp \
              $$PARCASL/src/my_p8est_log_wrappers.c \
              $$PARCASL/src/my_p8est_node_neighbors.cpp \
              $$PARCASL/src/my_p8est_nodes.c \
              $$PARCASL/src/my_p8est_quad_neighbor_nodes_of_node.cpp \
              $$PARCASL/src/my_p8est_refine_coarsen.cpp \
              $$PARCASL/src/my_p8est_tools.c \
              $$PARCASL/src/my_p8est_utils.cpp \
              $$PARCASL/src/my_p8est_vtk.cpp \
              $$PARCASL/src/point3.cpp \
              main_3d.cpp
  HEADERS +=  $$PARCASL/src/cube3.h \
              $$PARCASL/src/my_p8est_hierarchy.h \
              $$PARCASL/src/my_p8est_log_wrappers.h \
              $$PARCASL/src/my_p8est_node_neighbors.h \
              $$PARCASL/src/my_p8est_nodes.h \
              $$PARCASL/src/my_p8est_quad_neighbor_nodes_of_node.h \
              $$PARCASL/src/my_p8est_refine_coarsen.h \
              $$PARCASL/src/my_p8est_tools.h \
              $$PARCASL/src/my_p8est_utils.h \
              $$PARCASL/src/my_p8est_vtk.h \
              $$PARCASL/src/point3.h
}
