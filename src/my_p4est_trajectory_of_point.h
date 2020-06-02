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
 * \brief trajectory_from_np1_to_n backtraces the nodes of p4est to find the one departure points along the characteristics
 * of the velocity field, at time n.
 * This is a semi-Lagrangian algorithm with Backward Difference Formula along the characteristics.
 * [WARNING:] The velocity fields are interpolated using quadratic interpolation from the nodes to the points of interest, this
 * procedure can be VERY slow if precalculated second derivatives are not provided and if the node neighbors are not initialized
 * (it is highly advised to provide precalculated second derivatives or, at least, initialize the node neighbors if the former is
 * not desired)
 * This function does NOT call trajectory_from_np1_bunch_of_points as it calculates characteristics back from nodes on the forest.
 * Hence, a bunch of interpolation can be shortcut (much faster to read values than doing dummy interpolations).
 * \param[in] p4est                   pointer to the forest on which the nodes to backtrace are defined
 * \param[in] nodes                   pointer to the nodes of that forest to trace back in time
 * \param[in] ngbd_n                  pointer to the node neighborhood information at time n
 * \param[in] dt                      the time step between n and np1
 * \param[in] v                       the P4EST_DIM components of the velocity field at time n (defined at the nodes of ngbd_n)
 * \param[in] second_derivatives_v    double array of precalculated second derivatives of the velocity field at time n
 *                                    second_derivatives_v[dd][dir] is the second derivative of the dir component of
 *                                    v along Cartesian direction dd
 *                                    If this is a NULL pointer itself or if any of its P4EST_DIM*P4EST_DIM component is NULL,
 *                                    precalculated second derivatives will NOT be used (might be much slower)!
 * \param[out] xyz_                   double array of vector<double> for the coordinates of the departure points at time n. This is
 *                                    to be filled (and possibly resized) by the function!
 */
void trajectory_from_np1_to_n(const p4est_t* p4est, const p4est_nodes_t* nodes, const my_p4est_node_neighbors_t* ngbd_n,
                              const double& dt,
                              Vec v[P4EST_DIM], Vec second_derivatives_v[P4EST_DIM][P4EST_DIM],
                              std::vector<double> xyz_d[P4EST_DIM]);
void trajectory_from_np1_to_n(const p4est_t* p4est, const p4est_nodes_t* nodes, const my_p4est_node_neighbors_t* ngbd_n,
                              const double& dt,
                              Vec v[P4EST_DIM],
                              std::vector<double> xyz_d[P4EST_DIM]);



/*!
 * \brief trajectory_from_np1_to_nm1 backtraces the nodes of p4est_n to find the two departure points along the characteristics
 * of the velocity field, at time n and nm1.
 * This is a semi-Lagrangian algorithm with Backward Difference Formula along the characteristics.
 * [WARNING:] The velocity fields are interpolated using quadratic interpolation from the nodes to the points of interest, this
 * procedure can be VERY slow if precalculated second derivatives are not provided and if the node neighbors are not initialized
 * (it is highly advised to provide precalculated second derivatives or, at least, initialize the node neighbors if the former is
 * not desired)
 * This function does NOT call trajectory_from_np1_bunch_of_points as it calculates characteristics back from nodes on the forest.
 * Hence, a bunch of interpolation can be shortcut (much faster to read values than doing dummy interpolations).
 * \param[in] p4est_n                 pointer to the forest on which the nodes to backtrace are defined
 * \param[in] nodes_n                 pointer to the nodes of that forest to trace back in time
 * \param[in] ngbd_nm1                pointer to the node neighborhood information at time nm1
 * \param[in] ngbd_n                  pointer to the node neighborhood information at time n
 * \param[in] vnm1                    the velocity field at time nm1
 * \param[in] second_derivatives_vnm1 double array of precalculated second derivatives of the velocity field at time nm1
 *                                    second_derivatives_vnm1[dd][dir] is the second derivative of the dir component of
 *                                    vnm1 along Cartesian direction dd
 *                                    If this is a NULL pointer itself or if any of its P4EST_DIM*P4EST_DIM component is NULL,
 *                                    precalculated second derivatives will NOT be used (might be much slower)!
 * \param[in] vn                      the P4EST_DIM components of the velocity field at time n (defined at the nodes of ngbd_n)
 * \param[in] second_derivatives_vn   double array of precalculated second derivatives of the velocity field at time n
 *                                    second_derivatives_vn[dd][dir] is the second derivative of the dir component of
 *                                    vn along Cartesian direction dd
 *                                    If this is a NULL pointer itself or if any of its P4EST_DIM*P4EST_DIM component is NULL,
 *                                    precalculated second derivatives will NOT be used (might be much slower)!
 * \param[in] dt_nm1                  the time step between nm1 and n
 * \param[in] dt_n                    the time step between n and np1
 * \param[out] xyz_nm1                double array of vector<double> for the coordinates of the departure points at time nm1. This is
 *                                    to be filled (and possibly resized) by the function!
 *                                    CANNOT BE NULL ON INPUT!
 * \param[out] xyz_n                  double array of vector<double> for the coordinates of the departure points at time n. This is
 *                                    to be filled (and possibly resized) by the function!
 */
void trajectory_from_np1_to_nm1(const p4est_t* p4est_n, const p4est_nodes_t* nodes_n,
                                const my_p4est_node_neighbors_t* ngbd_nm1,
                                const my_p4est_node_neighbors_t* ngbd_n,
                                Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                const double& dt_nm1, const double& dt_n,
                                std::vector<double> xyz_nm1[P4EST_DIM],
                                std::vector<double> xyz_n[P4EST_DIM]);
inline void trajectory_from_np1_to_nm1(const p4est_t* p4est_n, p4est_nodes_t* nodes_n,
                                       const my_p4est_node_neighbors_t* ngbd_nm1,
                                       const my_p4est_node_neighbors_t* ngbd_n,
                                       Vec vnm1[P4EST_DIM],
                                       Vec vn[P4EST_DIM],
                                       const double& dt_nm1, const double& dt_n,
                                       std::vector<double> xyz_nm1[P4EST_DIM],
                                       std::vector<double> xyz_n[P4EST_DIM])
{
  trajectory_from_np1_to_nm1(p4est_n, nodes_n, ngbd_nm1, ngbd_n, vnm1, NULL, vn, NULL, dt_nm1, dt_n, xyz_nm1, xyz_n);
}


/*!
 * \brief trajectory_from_np1_to_nm1 backtraces the faces aligned with the Cartesian direction dir to find the two
 * departure points along the characteristics of the velocity field, at time n and nm1.
 * This function calls trajectory_from_np1_bunch_of_points internally, see that function's details for more information
 * \param[in] faces_n   the faces information
 * \param[in] dir       Cartesian direction normal to the faces of interest
 */
void trajectory_from_np1_to_nm1(const my_p4est_faces_t* faces_n, const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                Vec vnm1[P4EST_DIM], Vec vn[P4EST_DIM],
                                const double& dt_nm1, const double& dt_n,
                                std::vector<double> xyz_nm1[P4EST_DIM],
                                std::vector<double> xyz_n[P4EST_DIM],
                                const unsigned char& dir);
/*!
 * \brief trajectory_from_np1_to_nm1 backtraces the faces of p4est aligned with the Cartesian direction dir to find the one
 * departure point along the characteristics of the velocity field, at time n.
 * This function calls trajectory_from_np1_bunch_of_points internally, see that function's details for more information
 * \param[in] faces_n   the faces information
 * \param[in] dir       Cartesian direction of interest
 */
inline void trajectory_from_np1_to_n(const my_p4est_faces_t* faces_n, const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                     Vec vnm1[P4EST_DIM], Vec vn[P4EST_DIM],
                                     const double& dt_nm1, const double& dt_n,
                                     std::vector<double> xyz_n[P4EST_DIM],
                                     const unsigned char& dir )
{
  trajectory_from_np1_to_nm1(faces_n, ngbd_nm1, ngbd_n, vnm1, vn, dt_nm1, dt_n, NULL, xyz_n, dir);
}

/*!
 * \brief trajectory_from_np1_all_faces backtraces ALL the faces of p4est (simultaneously) to find the (two) departure point
 * along the characteristics of the velocity field, at time n (and nm1). The calculation of the departure points at time nm1
 * is simply skipped if the last argument (xyz_nm1) is NULL.
 * This function calls trajectory_from_np1_bunch_of_points internally, see that function's details for more information
 * \param[in] faces_n                 the faces information
 */
void trajectory_from_np1_all_faces(const my_p4est_faces_t* faces_n, const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                   Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                   Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                   const double& dt_nm1, const double& dt_n,
                                   std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM],
                                   std::vector<double> xyz_nm1[P4EST_DIM][P4EST_DIM]);
inline void trajectory_from_np1_all_faces_transposed_second_derivatives(const my_p4est_faces_t* faces_n, const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                                                        Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                                                        Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                                                        const double& dt_nm1, const double& dt_n,
                                                                        std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM],
                                                                        std::vector<double> xyz_nm1[P4EST_DIM][P4EST_DIM])
{
  transpose_second_derivatives(second_derivatives_vnm1, second_derivatives_vnm1);
  trajectory_from_np1_all_faces(faces_n, ngbd_nm1, ngbd_n, vnm1, second_derivatives_vnm1, vn, second_derivatives_vn, dt_nm1, dt_n, xyz_n, xyz_nm1);
  transpose_second_derivatives(second_derivatives_vnm1, second_derivatives_vnm1);
}
inline void trajectory_from_np1_all_faces(const my_p4est_faces_t* faces_n, const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                          Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                          Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                          const double &dt_nm1, const double &dt_n,
                                          std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM])
{
  trajectory_from_np1_all_faces(faces_n, ngbd_nm1, ngbd_n, vnm1, second_derivatives_vnm1, vn, second_derivatives_vn, dt_nm1, dt_n, xyz_n, NULL);
}

inline void trajectory_from_np1_all_faces_transposed_second_derivatives(const my_p4est_faces_t* faces_n, const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                                                        Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                                                        Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                                                        const double &dt_nm1, const double &dt_n,
                                                                        std::vector<double> xyz_n[P4EST_DIM][P4EST_DIM])
{
  transpose_second_derivatives(second_derivatives_vnm1, second_derivatives_vnm1);
  trajectory_from_np1_all_faces(faces_n, ngbd_nm1, ngbd_n, vnm1, second_derivatives_vnm1, vn, second_derivatives_vn, dt_nm1, dt_n, xyz_n);
  transpose_second_derivatives(second_derivatives_vnm1, second_derivatives_vnm1);
}

/*!
 * \brief trajectory_from_np1_bunch_of_points backtraces a bunch of lists of points xyz_np1 (simultaneously) to find the (two)
 * departure point(s) along the characteristics of the velocity field, at time n (and nm1). The calculation of the departure
 * points at time nm1 is simply skipped if the last argument (xyz_nm1) is NULL.
 * This is a semi-Lagrangian algorithm with Backward Difference Formula along the characteristics.
 * [WARNING:] The velocity fields are interpolated using quadratic interpolation from the nodes to the points of interest, this
 * procedure can be VERY slow if precalculated second derivatives are not provided and if the node neighbors are not initialized
 * (it is highly advised to provide precalculated second derivatives or, at least, initialize the node neighbors if the former is
 * not desired)
 * \param[in] ngbd_nm1                pointer to the node neighborhood information at time nm1
 * \param[in] ngbd_n                  pointer to the node neighborhood information at time n
 * \param[in] vnm1                    the velocity field at time nm1
 * \param[in] second_derivatives_vnm1 double array of precalculated second derivatives of the velocity field at time nm1
 *                                    second_derivatives_vnm1[dd][dir] is the second derivative of the dir component of
 *                                    vnm1 along Cartesian direction dd
 *                                    If this is a NULL pointer itself or if any of its P4EST_DIM*P4EST_DIM component is NULL,
 *                                    precalculated second derivatives will NOT be used (might be much slower)!
 * \param[in] vn                      the P4EST_DIM components of the velocity field at time n (defined at the nodes of ngbd_n)
 * \param[in] second_derivatives_vn   double array of precalculated second derivatives of the velocity field at time n
 *                                    second_derivatives_vn[dd][dir] is the second derivative of the dir component of
 *                                    vn along Cartesian direction dd
 *                                    If this is a NULL pointer itself or if any of its P4EST_DIM*P4EST_DIM component is NULL,
 *                                    precalculated second derivatives will NOT be used (might be much slower)!
 * \param[in] dt_nm1                  the time step between nm1 and n
 * \param[in] dt_n                    the time step between n and np1
 * \param[in] xyz_np1                 double array of const vector<double> for the coordinates of the landing points at time np1. This
 *                                    points to the lists of coordinates for landing points to be backtraced (component by component)
 *                                    xyz_np1[k][i][j] is the ith component of xyz_np1 for the jth point in list k, 0 <= i < P4EST_DIM
 * \param[in] n_lists                 number of lists of input coordinates (i.e. first dimension size of xyz_np1)
 * \param[out] xyz_n                  double array of pointers to vector<double> for the coordinates of the departure points at time n.
 *                                    This is to be filled (and possibly resized) by the function!
 *                                    - xyz_n[k][i] must be of size xyz_np1[k][dir].size(), 0 <= dir < P4EST_DIM, for all i: 0 <= i < P4EST_DIM
 *                                    - xyz_n[k][i][j] is the ith component of xyz_n for the jth point in list k (i.e. corresponding to
 *                                      landing point xyz_np1[k][i][j])
 * \param[out] xyz_nm1                double array of pointers to vector<double> for the coordinates of the departure points at time nm1.
 *                                    This is to be filled (and possibly resized) by the function!
 *                                    - xyz_nm1[k][i] must be of size xyz_np1[k][dir].size(), 0 <= dir < P4EST_DIM, for all i: 0 <= i < P4EST_DIM
 *                                    - xyz_nm1[k][i][j] is the ith component of xyz_nm1 for the jth point in list k (i.e. corresponding to
 *                                      landing point xyz_np1[k][i][j])
 *                                    If this argument is NULL on input, the calculation of these is disregarded!
 */
void trajectory_from_np1_bunch_of_points(const my_p4est_node_neighbors_t* ngbd_nm1, const my_p4est_node_neighbors_t* ngbd_n,
                                         Vec vnm1[P4EST_DIM], Vec second_derivatives_vnm1[P4EST_DIM][P4EST_DIM],
                                         Vec vn[P4EST_DIM], Vec second_derivatives_vn[P4EST_DIM][P4EST_DIM],
                                         const double& dt_nm1, const double& dt_n,
                                         const std::vector<double> xyz_np1[][P4EST_DIM],
                                         const unsigned int& n_lists,
                                         std::vector<double>* xyz_n[][P4EST_DIM],
                                         std::vector<double>* xyz_nm1[][P4EST_DIM]);


#endif /* MY_P4EST_TRAJECTORY_OF_POINT_H */
