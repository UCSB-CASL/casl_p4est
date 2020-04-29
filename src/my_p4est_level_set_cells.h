#ifndef MY_P4EST_LEVELSET_CELLS_H
#define MY_P4EST_LEVELSET_CELLS_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/casl_math.h>

class my_p4est_level_set_cells_t
{
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_node_neighbors_t *ngbd_n;

public:
  my_p4est_level_set_cells_t(my_p4est_cell_neighbors_t *ngbd_c, my_p4est_node_neighbors_t *ngbd_n)
    : p4est(ngbd_n->p4est), ghost(ngbd_n->ghost), nodes(ngbd_n->nodes), ngbd_c(ngbd_c), ngbd_n(ngbd_n)
  {}

  /*!
   * \brief integrate_over_interface integrates integrates cell_field over the irregular interface phi.
   * The integral is approximated as
   * integral of cell_weight  ~= sum over the cell (cell_weight)*(length or area of the interface in cell)
   * where the length of area is approximated using standard techniques for node-sampled fields (decomposition
   * in simplices etc.).
   * \param node_sampled_phi  [in]  node-sampled levelset values
   * \param cell_weight       [in]  cell-sampled weight for the calculation of the integral
   * \return the value of the integral
   */
  double integrate_over_interface(Vec node_sampled_phi, Vec cell_field) const;

  /*!
   * \brief normal_vector_weighted_integral_over_interface integrates cell_field*(normal vector) over
   * the irregular interface phi --> typical application is the calculation of the force components due
   * to pressure forces onto a solid object. The integral is approximated as
   * integral of (cell_weight*normal vector)  ~= sum over the cell of (cell_weight)*integral over the cell of normal vector
   * where the integral over the cell of normal vector is approximated using standard techniques for
   * node-sampled fields (decomposition in simplices etc.).
   * \param node_sampled_phi      [in]  node-sampled levelset values
   * \param cell_weight           [in]  cell-sampled weight for the calculation of the integral
   * \param integral              [out] integral[dir] is the integral of cell_field*normal[dir] over the interface, (0 <= dir < P4EST_DIM)
   * \param node_sampled_grad_phi [in]  (optional) P4EST_DIM block-structured vector of node-sampled components of gradient of phi
   */
  void normal_vector_weighted_integral_over_interface(Vec node_sampled_phi, Vec cell_weight, double *integral, Vec node_sampled_grad_phi) const;
  inline  void normal_vector_weighted_integral_over_interface(Vec node_sampled_phi, Vec cell_weight, double *integral) const
  {
    Vec node_sampled_grad_phi;
    PetscErrorCode ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &node_sampled_grad_phi); CHKERRXX(ierr);
    ngbd_n->first_derivatives_central(node_sampled_phi, node_sampled_grad_phi);
    normal_vector_weighted_integral_over_interface(node_sampled_phi, cell_weight, integral, node_sampled_grad_phi);
    ierr = VecDestroy(node_sampled_grad_phi); CHKERRXX(ierr);
    return;
  }

  /*!
   * \brief integrate calculates the integral of a cell-sampled field cell_field over the negative domain associated
   * with the node-sampled levelset function node_sampled_phi. The integral is approximated as the Riemann sum of
   * cell-values multiplied by the measure of the cell volume/area in negative domain.
   * \param [in] node_sampled_phi : vector of node-sampled levelset value
   * \param [in] cell_field       : vector of cell-sampled field values to integrate
   * \return the desired integral
   */
  double integrate(Vec node_sampled_phi, Vec cell_field) const;

  /*!
   * \brief geometric_extrapolation_over_interface extrapolates a cell-sampled field cell_field from the negative to
   * positive domain associated with the node-sampled levelset function phi.
   * \param [inout] cell_field    : vector of cell-sampled field values to extrapolate, see note herebelow for more details;
   * \param [in] node_sampled_phi : vector of node-sampled levelset values;
   * \param [in] interp_grad_phi  : P4EST_DIM-valued node-interpolator for interpolating the node-sampled gradient of phi at cell centers;
   * \param [in] bc               : boundary condition object, used for determining and using the type of interface;
   * \param [in] degree           : desired degree of the interpolant to be built (0, 1 or 2);
   * \param [in] band_to_extend   : values of cell_field are calculated and set only where the phi_q (as defined here below)
   *                                is less than band_to_extend*(smallest diagonal) and where the local normal is well-defined.
   * -----------------------------
   * Internal procedure's details:
   * -----------------------------
   * For every cell C such that its value is _not_ well defined, i.e. arithmetic average phi_q of node-sampled value is positive
   * (+ all vertex values of node_sampled_levelset are positive as well if the local center is Neumann) plus if
   * 0.0 < phi_q < band_to_extend*diag_min and if the locally-interpolated normal is well-defined, then the method samples
   * 0) the interface boundary condition type and value at the point projected on the interface from the center of cell C
   * 1) the node 2*diag away from the interface along the normal (if required to match the desired degree)
   * 2) the node 3*diag away from the interface along the normal (if required to match the desired degree)
   * Thereafter, the local value of cell_field is set by evaluating the interpolant of desired degree matching those sampled values
   * and boundary conditions.
   */
  void geometric_extrapolation_over_interface(Vec cell_field, Vec node_sampled_phi, const my_p4est_interpolation_nodes_t& interp_grad_phi,
                                              const BoundaryConditionsDIM& bc, const unsigned char& degree = 2, const unsigned int& band_to_extend = INT_MAX) const;

  /*!
   * \brief extend_over_interface extrapolates a cell-sampled field cell_field from the negative to positive domain associated
   * with the node-sampled levelset function phi.
   * \param [in] node_sampled_phi : vector of node-sampled levelset values;
   * \param [inout] cell_field    : vector of cell-sampled field values to extrapolate, see geometric_extrapolation_over_interface for more details;
   * \param [in] bc               : POINTER TO a boundary condition object, used for determining and using the type of interface;
   * \param [in] degree           : desired degree of the interpolant to be built (0, 1 or 2);
   * \param [in] band_to_extend   : values of cell_field are calculated and set only where the phi_q is less than band_to_extend*(smallest diagonal).
   * (see more details in the header's comments of 'geometric_extrapolation_over_interface', which is called internally).
   */
  inline void extend_over_interface(Vec node_sampled_phi, Vec cell_field, BoundaryConditionsDIM *bc,
                                    const unsigned char& degree = 2, const unsigned int& band_to_extend = INT_MAX) const
  {
#ifdef CASL_THROWS
    if(bc == NULL)
      throw std::invalid_argument("my_p4est_level_set_cells_t::extend_over_interface: a valid pointer to boundary condition must be provided to this method.");
#endif
    PetscErrorCode ierr;

    /* first compute the gradient of phi and create a node-interpolator for that */
    Vec node_sampled_grad_phi;
    ierr = VecCreateGhostNodesBlock(p4est, nodes, P4EST_DIM, &node_sampled_grad_phi); CHKERRXX(ierr);
    ngbd_n->first_derivatives_central(node_sampled_phi, node_sampled_grad_phi);

    my_p4est_interpolation_nodes_t interp_grad_phi(ngbd_n);
    interp_grad_phi.set_input(node_sampled_grad_phi, linear, P4EST_DIM);

    geometric_extrapolation_over_interface(cell_field, node_sampled_phi, interp_grad_phi, *bc, degree, band_to_extend);
    return;
  }
};

#endif /* MY_P4EST_LEVELSET_CELLS_H */
