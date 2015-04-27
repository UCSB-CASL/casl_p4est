#ifndef TEST_BRICK_H
#define TEST_BRICK_H

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifndef P4_TO_P8
#include <p4est.h>
#else
#include <p8est.h>
#endif



class test_brick
{
public:
    static inline       p4est_locidx_t
    qidx (p4est_locidx_t m, p4est_locidx_t n,
          p4est_locidx_t i, p4est_locidx_t j, p4est_locidx_t k)
    {
    #ifndef P4_TO_P8
      return m * j + i;
    #else
      return m * n * k + m * j + i;
    #endif
    }

    static void
    #ifndef P4_TO_P8
    check_brick (p4est_connectivity_t * conn, int mi, int ni,
                 int periodic_a, int periodic_b)
    #else
    check_brick (p8est_connectivity_t * conn, int mi, int ni, int pi,
                 int periodic_a, int periodic_b, int periodic_c)
    #endif
    {
      p4est_topidx_t      m = (p4est_topidx_t) mi;
      p4est_topidx_t      n = (p4est_topidx_t) ni;
      int                 i;
      p4est_topidx_t      ti, tj, tk = 0;
      p4est_topidx_t     *tree_to_vertex = conn->tree_to_vertex;
      p4est_topidx_t     *tree_to_corner = conn->tree_to_corner;
      p4est_topidx_t     *tree_to_tree = conn->tree_to_tree;
      int8_t             *tree_to_face = conn->tree_to_face;
      p4est_topidx_t     *ctt_offset = conn->ctt_offset;
      p4est_topidx_t     *corner_to_tree = conn->corner_to_tree;
      int8_t             *corner_to_corner = conn->corner_to_corner;
      p4est_topidx_t      num_trees = conn->num_trees;
      p4est_topidx_t      num_vertices = conn->num_vertices;
      p4est_topidx_t      num_corners = conn->num_corners;
      double             *vertices = conn->vertices;
      double             *vertex[P4EST_CHILDREN];
      int8_t             *vert_counter, *corn_counter;
      p4est_topidx_t     *quad_counter;
      int8_t              total, face1, face2, face3, corn1;
      p4est_topidx_t      tx, ty, ttree1, ttree2, ttree3, tcorn1;
      p4est_topidx_t      tz = 0;
      p4est_topidx_t      diffx, diffy;
    #ifdef P4_TO_P8
      p4est_topidx_t      p = (p4est_topidx_t) pi;
      p4est_topidx_t     *tree_to_edge = conn->tree_to_edge;
      p4est_topidx_t     *ett_offset = conn->ett_offset;
      p4est_topidx_t     *edge_to_tree = conn->edge_to_tree;
      int8_t             *edge_to_edge = conn->edge_to_edge;
      p4est_topidx_t      num_edges = conn->num_edges;
      int8_t             *edge_counter;
      int8_t              edge1, edge2;
      p4est_topidx_t      tedge1, tedge2;
      p4est_topidx_t      diffz;
    #endif

      SC_CHECK_ABORT (num_trees > 0, "no trees");
    #ifndef P4_TO_P8
      SC_CHECK_ABORT (num_trees == m * n, "bad dimensions");
      SC_CHECK_ABORT (num_vertices == (m + 1) * (n + 1),
                      "wrong number of vertices");
    #else
      SC_CHECK_ABORT (num_trees == m * n * p, "bad dimensions");
      SC_CHECK_ABORT (num_vertices == (m + 1) * (n + 1) * (p + 1),
                      "wrong number of vertices");
    #endif

      quad_counter = P4EST_ALLOC (p4est_topidx_t, num_trees);
      memset (quad_counter, -1, num_trees * sizeof (p4est_topidx_t));
      vert_counter = P4EST_ALLOC_ZERO (int8_t, num_vertices);
      corn_counter = NULL;
      if (num_corners > 0) {
        corn_counter = P4EST_ALLOC_ZERO (int8_t, num_corners);
      }
    #ifdef P4_TO_P8
      edge_counter = NULL;
      if (num_edges > 0) {
        edge_counter = P4EST_ALLOC_ZERO (int8_t, num_edges);
      }
    #endif

      for (ti = 0; ti < num_trees; ti++) {
        for (i = 0; i < P4EST_CHILDREN; i++) {
          vertex[i] = vertices + 3 * tree_to_vertex[ti * P4EST_CHILDREN + i];
          vert_counter[tree_to_vertex[ti * P4EST_CHILDREN + i]]++;
          if (num_corners > 0 && tree_to_corner[ti * P4EST_CHILDREN + i] != -1) {
            corn_counter[tree_to_corner[ti * P4EST_CHILDREN + i]]++;
          }
        }
        tx = (p4est_topidx_t) vertex[0][0];
        ty = (p4est_topidx_t) vertex[0][1];
    #ifdef P4_TO_P8
        tz = (p4est_topidx_t) vertex[0][2];
    #endif
        SC_CHECK_ABORT (tx < m, "vertex coordinates out of range");
        SC_CHECK_ABORT (ty < n, "vertex coordinates out of range");
    #ifdef P4_TO_P8
        SC_CHECK_ABORT (tz < p, "vertex coordinates out of range");
    #endif
        quad_counter[qidx (m, n, tx, ty, tz)] = ti;
        for (i = 1; i < P4EST_CHILDREN; i++) {
          tx = (p4est_locidx_t) (vertex[i][0] - vertex[0][0]);
          ty = (p4est_locidx_t) (vertex[i][1] - vertex[0][1]);
    #ifdef P4_TO_P8
          tz = (p4est_locidx_t) (vertex[i][2] - vertex[0][2]);
    #endif
          if ((i & 1) == 1) {
            SC_CHECK_ABORT (tx == 1, "non-unit vertex difference");
          }
          else {
            SC_CHECK_ABORT (tx == 0, "non-unit vertex difference");
          }
          if (((i >> 1) & 1) == 1) {
            SC_CHECK_ABORT (ty == 1, "non-unit vertex difference");
          }
          else {
            SC_CHECK_ABORT (ty == 0, "non-unit vertex difference");
          }
    #ifdef P4_TO_P8
          if ((i >> 2) == 1) {
            SC_CHECK_ABORT (tz == 1, "non-unit vertex difference");
          }
          else {
            SC_CHECK_ABORT (tz == 0, "non-unit vertex difference");
          }
    #endif
        }
    #ifdef P4_TO_P8
        if (num_edges > 0) {
          for (i = 0; i < P8EST_EDGES; i++) {
            if (tree_to_edge[ti * P8EST_EDGES + i] != -1) {
              edge_counter[tree_to_edge[ti * P8EST_EDGES + i]]++;
            }
          }
        }
    #endif
      }

      for (ti = 0; ti < m; ti++) {
        for (tj = 0; tj < n; tj++) {
    #ifdef P4_TO_P8
          for (tk = 0; tk < p; tk++) {
    #endif
            SC_CHECK_ABORT (quad_counter[qidx (m, n, ti, tj, tk)] != -1,
                            "grid points has no tree");
    #ifdef P4_TO_P8
          }
    #endif
        }
      }

      for (ti = 0; ti < num_vertices; ti++) {
        tx = (p4est_topidx_t) vertices[ti * 3];
        ty = (p4est_topidx_t) vertices[ti * 3 + 1];
    #ifdef P4_TO_P8
        tz = (p4est_topidx_t) vertices[ti * 3 + 2];
    #endif
        total = P4EST_CHILDREN;
        if (tx == m || tx == 0) {
          total /= 2;
        }
        if (ty == n || ty == 0) {
          total /= 2;
        }
    #ifdef P4_TO_P8
        if (tz == p || tz == 0) {
          total /= 2;
        }
    #endif
        SC_CHECK_ABORT (vert_counter[ti] == total,
                        "vertex has too many or too few trees");
      }

      if (num_corners > 0) {
        for (ti = 0; ti < num_corners; ti++) {
          SC_CHECK_ABORT (corn_counter[ti] == P4EST_CHILDREN,
                          "corner has too many or too few trees");
          SC_CHECK_ABORT (ctt_offset[ti] == P4EST_CHILDREN * ti,
                          "corner offset incorrect");
        }
        SC_CHECK_ABORT (ctt_offset[ti] == P4EST_CHILDREN * ti,
                        "corner offset incorrect");
      }

    #ifdef P4_TO_P8
      if (num_edges > 0) {
        for (ti = 0; ti < num_edges; ti++) {
          SC_CHECK_ABORT (edge_counter[ti] == 4,
                          "edge has too many or too few trees");
          SC_CHECK_ABORT (ett_offset[ti] == 4 * ti, "edge offset incorrect");
        }
        SC_CHECK_ABORT (ett_offset[ti] == 4 * ti, "edge offset incorrect");
      }
    #endif

      for (ti = 0; ti < m; ti++) {
        for (tj = 0; tj < n; tj++) {
    #ifdef P4_TO_P8
          for (tk = 0; tk < p; tk++) {
    #endif
            ttree1 = quad_counter[qidx (m, n, ti, tj, tk)];
            for (face1 = 0; face1 < P4EST_FACES; face1++) {
              ttree2 = tree_to_tree[ttree1 * P4EST_FACES + face1];
              face2 = tree_to_face[ttree1 * P4EST_FACES + face1];
              if (!periodic_a &&
                  ((face1 == 0 && ti == 0) || (face1 == 1 && ti == m - 1))) {
                SC_CHECK_ABORT (ttree2 == ttree1 && face2 == face1,
                                "boundary tree without boundary face");
              }
              else if (!periodic_b &&
                       ((face1 == 2 && tj == 0) || (face1 == 3 && tj == n - 1))) {
                SC_CHECK_ABORT (ttree2 == ttree1 && face2 == face1,
                                "boundary tree without boundary face");
              }
    #ifdef P4_TO_P8
              else if (!periodic_c &&
                       ((face1 == 4 && tk == 0) || (face1 == 5 && tk == p - 1))) {
                SC_CHECK_ABORT (ttree2 == ttree1 && face2 == face1,
                                "boundary tree without boundary face");
              }
    #endif
              else {
                switch (face1) {
                case 0:
                  ttree3 = quad_counter[qidx (m, n, (ti + m - 1) % m, tj, tk)];
                  break;
                case 1:
                  ttree3 = quad_counter[qidx (m, n, (ti + 1) % m, tj, tk)];
                  break;
                case 2:
                  ttree3 = quad_counter[qidx (m, n, ti, (tj + n - 1) % n, tk)];
                  break;
                case 3:
                  ttree3 = quad_counter[qidx (m, n, ti, (tj + 1) % n, tk)];
                  break;
    #ifdef P4_TO_P8
                case 4:
                  ttree3 = quad_counter[qidx (m, n, ti, tj, (tk + p - 1) % p)];
                  break;
                case 5:
                  ttree3 = quad_counter[qidx (m, n, ti, tj, (tk + 1) % p)];
                  break;
    #endif
                default:
                  SC_ABORT_NOT_REACHED ();
                }
                face3 = face1 ^ 1;
                SC_CHECK_ABORT (ttree3 == ttree2 && face2 == face3,
                                "tree has incorrect neighbor");
                ttree3 = tree_to_tree[ttree2 * P4EST_FACES + face2];
                SC_CHECK_ABORT (ttree1 == ttree3, "tree mismatch");
                face3 = tree_to_face[ttree2 * P4EST_FACES + face2];
                SC_CHECK_ABORT (face1 == face3, "face mismatch");
              }
            }
    #ifdef P4_TO_P8
            if (num_edges > 0) {
              for (edge1 = 0; edge1 < P8EST_EDGES; edge1++) {
                if ((!periodic_b &&
                     (((edge1 == 0 || edge1 == 2) && (tj == 0)) ||
                      ((edge1 == 1 || edge1 == 3) && (tj == n - 1)))) ||
                    (!periodic_c &&
                     (((edge1 == 0 || edge1 == 1) && (tk == 0)) ||
                      ((edge1 == 2 || edge1 == 3) && (tk == p - 1))))) {
                  SC_CHECK_ABORT (tree_to_edge[ttree1 * P8EST_EDGES + edge1] ==
                                  -1, "boundary tree without boundary edge");
                }
                else if ((!periodic_a &&
                          (((edge1 == 4 || edge1 == 6) && (ti == 0)) ||
                           ((edge1 == 5 || edge1 == 7) && (ti == m - 1)))) ||
                         (!periodic_c &&
                          (((edge1 == 4 || edge1 == 5) && (tk == 0)) ||
                           ((edge1 == 6 || edge1 == 7) && (tk == p - 1))))) {
                  SC_CHECK_ABORT (tree_to_edge[ttree1 * P8EST_EDGES + edge1] ==
                                  -1, "boundary tree without boundary edge");
                }
                else if ((!periodic_a &&
                          (((edge1 == 8 || edge1 == 10) && (ti == 0)) ||
                           ((edge1 == 9 || edge1 == 11) && (ti == m - 1)))) ||
                         (!periodic_b &&
                          (((edge1 == 8 || edge1 == 9) && (tj == 0)) ||
                           ((edge1 == 10 || edge1 == 11) && (tj == n - 1))))) {
                  SC_CHECK_ABORT (tree_to_edge[ttree1 * P8EST_EDGES + edge1] ==
                                  -1, "boundary tree without boundary edge");
                }
                else {
                  tedge1 = tree_to_edge[ttree1 * P8EST_EDGES + edge1];
                  SC_CHECK_ABORT (edge_to_tree[4 * tedge1 + (3 - (edge1 % 4))] ==
                                  ttree1, "edge_to_tree mismatch");
                  SC_CHECK_ABORT (edge_to_edge[4 * tedge1 + (3 - (edge1 % 4))] ==
                                  edge1, "edge_to_edge mismatch");
                  ttree2 = tree_to_tree[ttree1 * 6 + p8est_edge_faces[edge1][0]];
                  edge2 = edge1 ^ 1;
                  tedge2 = tree_to_edge[ttree2 * P8EST_EDGES + edge2];
                  SC_CHECK_ABORT (tedge1 == tedge2,
                                  "face neighbor trees do not share edge");
                  SC_CHECK_ABORT (edge_to_tree[4 * tedge1 + (3 - (edge2 % 4))] ==
                                  ttree2,
                                  "edge does not recognize face neighbors");
                  SC_CHECK_ABORT (edge_to_edge[4 * tedge1 + (3 - (edge2 % 4))] ==
                                  edge2,
                                  "edge does not recognize face neighbors' edges");
                  ttree2 = tree_to_tree[ttree1 * 6 + p8est_edge_faces[edge1][1]];
                  edge2 = edge1 ^ 2;
                  tedge2 = tree_to_edge[ttree2 * P8EST_EDGES + edge2];
                  SC_CHECK_ABORT (tedge1 == tedge2,
                                  "face neighbor trees do not share edge");
                  SC_CHECK_ABORT (edge_to_tree[4 * tedge1 + (3 - (edge2 % 4))] ==
                                  ttree2,
                                  "edge does not recognize face neighbors");
                  SC_CHECK_ABORT (edge_to_edge[4 * tedge1 + (3 - (edge2 % 4))] ==
                                  edge2,
                                  "edge does not recognize face neighbors' edges");
                  ttree2 =
                    tree_to_tree[ttree2 * 6 + p8est_edge_faces[edge1 ^ 2][0]];
                  edge2 = edge1 ^ 3;
                  tedge2 = tree_to_edge[ttree2 * P8EST_EDGES + edge2];
                  SC_CHECK_ABORT (tedge1 == tedge2,
                                  "diagonal trees do not share edge");
                  SC_CHECK_ABORT (edge_to_tree[4 * tedge1 + (3 - (edge2 % 4))] ==
                                  ttree2,
                                  "edge does not recognize diagonal trees");
                  SC_CHECK_ABORT (edge_to_edge[4 * tedge1 + (3 - (edge2 % 4))] ==
                                  edge2,
                                  "edge does not recognize diagonal trees' edges");
                }
              }
            }
    #endif
            if (num_corners > 0) {
              for (corn1 = 0; corn1 < P4EST_CHILDREN; corn1++) {
                if ((!periodic_a &&
                     (((corn1 & 1) == 0 && ti == 0) ||
                      ((corn1 & 1) == 1 && ti == m - 1))) ||
                    (!periodic_b &&
                     ((((corn1 >> 1) & 1) == 0 && tj == 0) ||
                      (((corn1 >> 1) & 1) == 1 && tj == n - 1))) ||
    #ifdef P4_TO_P8
                    (!periodic_c &&
                     (((corn1 >> 2) == 0 && tk == 0) ||
                      ((corn1 >> 2) == 1 && tk == p - 1))) ||
    #endif
                    0) {
                  SC_CHECK_ABORT (tree_to_corner[ttree1 * P4EST_CHILDREN + corn1]
                                  == -1, "boundary tree without boundary corner");
                }
                else {
                  tcorn1 = tree_to_corner[ttree1 * P4EST_CHILDREN + corn1];
                  SC_CHECK_ABORT (corner_to_tree
                                  [tcorn1 * P4EST_CHILDREN +
                                   (P4EST_CHILDREN - 1 - corn1)] == ttree1,
                                  "corner_to_tree mismatch");
                  SC_CHECK_ABORT (corner_to_corner
                                  [tcorn1 * P4EST_CHILDREN + P4EST_CHILDREN - 1 -
                                   corn1] == corn1, "corner_to_corner mismatch");
                  for (i = 0; i < P4EST_CHILDREN; i++) {
                    ttree2 = corner_to_tree[tcorn1 * P4EST_CHILDREN + i];
                    tx =
                      (p4est_topidx_t) vertices[3 *
                                                tree_to_vertex[ttree2 *
                                                               P4EST_CHILDREN]];
                    ty = (p4est_topidx_t)
                      vertices[3 * tree_to_vertex[ttree2 * P4EST_CHILDREN] + 1];
    #ifdef P4_TO_P8
                    tz = (p4est_topidx_t)
                      vertices[3 * tree_to_vertex[ttree2 * P4EST_CHILDREN] + 2];
    #endif
                    diffx = (i & 1) - ((P4EST_CHILDREN - 1 - corn1) & 1);
                    diffy =
                      ((i >> 1) & 1) - (((P4EST_CHILDREN - 1 - corn1) >> 1) & 1);
    #ifdef P4_TO_P8
                    diffz = (i >> 2) - ((P4EST_CHILDREN - 1 - corn1) >> 2);
    #endif
                    SC_CHECK_ABORT ((ti + diffx + m) % m == tx,
                                    "unexpected trees around corner");
                    SC_CHECK_ABORT ((tj + diffy + n) % n == ty,
                                    "unexpected trees around corner");
    #ifdef P4_TO_P8
                    SC_CHECK_ABORT ((tk + diffz + p) % p == tz,
                                    "unexpected trees around corner");
    #endif
                  }
                }
              }
            }
    #ifdef P4_TO_P8
          }
    #endif
        }
      }

    #ifdef P4_TO_P8
      if (num_edges > 0) {
        P4EST_FREE (edge_counter);
      }
    #endif
      P4EST_FREE (vert_counter);
      if (num_corners > 0) {
        P4EST_FREE (corn_counter);
      }
      P4EST_FREE (quad_counter);

    }
    test_brick(int argc, char **argv);
};

#endif // TEST_BRICK_H
