#ifndef MY_P4EST_LEVELSET_FACES_H
#define MY_P4EST_LEVELSET_FACES_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_level_set.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/casl_math.h>

class my_p4est_level_set_faces_t
{
  const p4est_t *p4est;
  const p4est_nodes_t *nodes;
  my_p4est_node_neighbors_t *ngbd_n;
  my_p4est_faces_t *faces;

public:
  my_p4est_level_set_faces_t(my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces )
    : p4est(ngbd_n->get_p4est()), nodes(ngbd_n->get_nodes()), ngbd_n(ngbd_n), faces(faces)
  {}

  /*!
   * \brief geometric_extrapolation_over_interface extrapolates a cell-sampled field cell_field from the negative to
   * positive domain associated with the node-sampled levelset function phi, at faces that are marked "not well-defined".
   * \param [inout] face_field_dir        : vector of face-sampled values for the dir^th component of the vector field to extrapolate
   *                                        (hence sampled at faces of normal orientation dir);
   * \param [in] interp_phi               : node-interpolator for interpolating the node-sampled levelset function phi at face centers;
   * \param [in] interp_grad_phi          : P4EST_DIM-valued node-interpolator for interpolating the node-sampled gradient of phi at face centers;
   * \param [in] bc_dir                   : boundary condition object associated with the dir^th component of the vector field of interest,
   *                                        used for determining and using the type of interface;
   * \param [in] dir                      : Cartesian directon of interest and orientation of the considered sampled faces
   * \param [in] face_is_well_defined_dir : mask marking faces as well-defined (1.0) or not (0.0),
   *                                        (e.g.:  in Navier-Stokes for insance, it is well-defined if the corresponding interpolated
   *                                                value of the levelset at the face center is negative or if the control volume cor-
   *                                                -responding to the face is at least partially in the negative domain and the local
   *                                                interface boundary condition is Neumann)
   * \param [in] dxyz_hodge_dir           : (optional) vector of face-sampled values of the partial derivative of the Hodge variable with respect to
   *                                        Cartesian direction dir. This is information is used to correct interface Dirichlet boundary condition if
   *                                        any. (hence sampled at faces of normal orientation dir);
   * \param [in] degree                   : desired degree of the interpolant to be built (0, 1 or 2);
   * \param [in] band_to_extend           : values of face_field_dir are calculated and set only where the interpolated value of phi at the face center
   *                                        is less than band_to_extend*(smallest diagonal) and where the local normal is well-defined.
   * -----------------------------
   * Internal procedure's details:
   * -----------------------------
   * For every face f of orientation dir that is _not_ marked well-defined, the interpolated levelset value phi_f of the node-sampled
   * values of the levelset is calculated. If 0.0 < phi_f < band_to_extend*diag_min and if the locally-interpolated normal is
   * well-defined, the method samples
   * 0) - the interface boundary condition type and value at the point projected on the interface from the center of face f of orientation dir
   *    - the value of dxyz_hodge_dir is also interpolated at that point if dxyz_hodge_dir was provided
   * 1) the node 2*diag away from the interface along the normal (if required to match the desired degree)
   * 2) the node 3*diag away from the interface along the normal (if required to match the desired degree)
   * Thereafter, the local value of face_field_dir is set by evaluating the interpolant of desired degree matching those sampled values and boundary
   * conditions. If the sampled interface boundary condition type is found to be DIRICHLET, and if dxyz_hodge_dir was provided, the interpolated value
   * of dxyz_hodge_dir is added to the user-defined interface boundary condition value and used as interface value to be matched when building the
   * extrapolation interpolant.
   */
  void geometric_extrapolation_over_interface(Vec face_field_dir, const my_p4est_interpolation_nodes_t &interp_phi, const my_p4est_interpolation_nodes_t &interp_grad_phi,
                                              const BoundaryConditionsDIM &bc_dir, const unsigned char &dir, Vec face_is_well_defined_dir,
                                              Vec dxyz_hodge_dir = NULL, const unsigned char& degree = 2, const unsigned int& band_to_extend = INT_MAX) const;

  /*!
   * \brief extend_over_interface extrapolates a cell-sampled field cell_field from the negative to
   * positive domain associated with the node-sampled levelset function phi, at faces that are marked "not well-defined".
   * \param [in] node_sampled_phi         : vector of node-sampled levelset values;
   * \param [inout] face_field_dir        : vector of face-sampled values for the dir^th component of the vector field to extrapolate
   *                                        (hence sampled at faces of normal orientation dir);
   * \param [in] bc_dir                   : boundary condition object associated with the dir^th component of the vector field of interest,
   *                                        used for determining and using the type of interface;
   * \param [in] dir                      : Cartesian directon of interest and orientation of the considered sampled faces
   * \param [in] face_is_well_defined_dir : mask marking faces as well-defined (1.0) or not (0.0),
   *                                        (e.g.:  in Navier-Stokes for insance, it is well-defined if the corresponding interpolated
   *                                                value of the levelset at the face center is negative or if the control volume cor-
   *                                                -responding to the face is at least partially in the negative domain and the local
   *                                                interface boundary condition is Neumann)
   * \param [in] dxyz_hodge_dir           : (optional) vector of face-sampled values of the partial derivative of the Hodge variable with respect to
   *                                        Cartesian direction dir. This is information is used to correct interface Dirichlet boundary condition if
   *                                        any. (hence sampled at faces of normal orientation dir);
   * \param [in] degree                   : desired degree of the interpolant to be built (0, 1 or 2);
   * \param [in] band_to_extend           : values of face_field_dir are calculated and set only where the interpolated value of phi at the face center
   *                                        is less than band_to_extend*(smallest diagonal) and where the local normal is well-defined.
   * (see more details in the header's comments of 'geometric_extrapolation_over_interface', which is called internally).
   */
  inline void extend_over_interface(Vec node_sampled_phi, Vec face_field_dir, const BoundaryConditionsDIM &bc_dir, const unsigned char& dir, Vec face_is_well_defined_dir,
                                    Vec dxyz_hodge_dir = NULL, const unsigned char& degree = 2, const unsigned int& band_to_extend = INT_MAX) const
  {
    PetscErrorCode ierr;

    /* first compute the gradient of phi and create node-interpolators for phi and its gradient */
    Vec node_sampled_grad_phi;
    ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &node_sampled_grad_phi); CHKERRXX(ierr);
    ngbd_n->first_derivatives_central(node_sampled_phi, node_sampled_grad_phi);

    my_p4est_interpolation_nodes_t interp_phi(ngbd_n);      interp_phi.set_input(node_sampled_phi, linear);
    my_p4est_interpolation_nodes_t interp_grad_phi(ngbd_n); interp_grad_phi.set_input(node_sampled_grad_phi, linear, P4EST_DIM);

    geometric_extrapolation_over_interface(face_field_dir, interp_phi, interp_grad_phi, bc_dir, dir, face_is_well_defined_dir, dxyz_hodge_dir, degree, band_to_extend);
    return;
  }
};

#endif /* MY_P4EST_LEVELSET_FACES_H */
