#ifndef MY_P4EST_TRAJECTORY_OF_POINT_H
#define MY_P4EST_TRAJECTORY_OF_POINT_H


#ifdef P4_TO_P8
#include <p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <p4est.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_node_neighbors.h>
#endif

#include <vector>

/*!
 * \brief backtrace the nodes of p4est_n to find the departure points along the characteristics of the velocity
 *   field. This is a semi-Lagrangian algorithm with Backward Difference Formula along the characteristics.
 * \param[in] p4est   the forest on which the nodes to backtrace are defined
 * \param[in] nodes   the nodes to backtrace
 * \param[in] ngbd_n  the geometrical information
 * \param[in] v       the velocity field
 * \param[in] dt      the time step between n and np1
 * \param[out] xyz_d  the coordinates of the departure points at time n, to be filled
 */
void trajectory_from_np1_to_n( p4est_t *p4est, p4est_nodes_t *nodes,
                               my_p4est_node_neighbors_t *ngbd_n,
                               double dt,
                               Vec v[P4EST_DIM],
                               std::vector<double> xyz_d[P4EST_DIM] );


/*!
 * \brief backtrace the nodes of p4est_n to find the departure points along the characteristics of the velocity
 *   field. This is a semi-Lagrangian algorithm with Backward Difference Formula along the characteristics.
 * \param[in] p4est_n   the forest on which the nodes to backtrace are defined
 * \param[in] nodes_n   the nodes to backtrace
 * \param[in] ngbd_nm1  the geometrical information at time nm1
 * \param[in] ngbd_n    the geometrical information at time n
 * \param[in] vnm1      the velocity field at time nm1
 * \param[in] vn        the velocity field at time n
 * \param[in] dt_nm1    the time step between nm1 and n
 * \param[in] dt_n      the time step between n and np1
 * \param[out] xyz_nm1  the coordinates of the departure points at time nm1, to be filled
 * \param[out] xyz_n    the coordinates of the departure points at time n, to be filled
 */
void trajectory_from_np1_to_nm1( p4est_t *p4est_n, p4est_nodes_t *nodes_n,
                                 my_p4est_node_neighbors_t *ngbd_nm1,
                                 my_p4est_node_neighbors_t *ngbd_n,
                                 Vec vnm1[P4EST_DIM],
                                 Vec vn[P4EST_DIM],
                                 double dt_nm1, double dt_n,
                                 std::vector<double> xyz_nm1[P4EST_DIM],
                                 std::vector<double> xyz_n[P4EST_DIM] );


/*!
 * \brief backtrace the faces of p4est to find the departure points along the characteristics of the velocity
 *   field. This is a semi-Lagrangian algorithm with Backward Difference Formula along the characteristics.
 * \param[in] p4est_n   the forest on which the faces to backtrace are defined
 * \param[in] faces_n   the faces information
 * \param[in] ngbd_nm1  the geometrical information at time nm1
 * \param[in] ngbd_n    the geometrical information at time n
 * \param[in] vnm1      the velocity field at time nm1
 * \param[in] vn        the velocity field at time n
 * \param[in] dt_nm1    the time step between nm1 and n
 * \param[in] dt_n      the time step between n and np1
 * \param[out] xyz_nm1  the coordinates of the departure points at time nm1, to be filled
 * \param[out] xyz_n    the coordinates of the departure points at time n, to be filled
 * \param[in] dir       the direction in which the backtraced faces are, dir::x, dir::y or dir::z
 */
void trajectory_from_np1_to_nm1( p4est_t *p4est_n, my_p4est_faces_t *faces_n,
                                 my_p4est_node_neighbors_t *ngbd_nm1,
                                 my_p4est_node_neighbors_t *ngbd_n,
                                 Vec vnm1[P4EST_DIM],
                                 Vec vn[P4EST_DIM],
                                 double dt_nm1, double dt_n,
                                 std::vector<double> xyz_nm1[P4EST_DIM],
                                 std::vector<double> xyz_n[P4EST_DIM],
                                 int dir );

#endif /* MY_P4EST_TRAJECTORY_OF_POINT_H */
