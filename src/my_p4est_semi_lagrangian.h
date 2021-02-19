#ifndef MY_P4EST_SEMI_LAGRANGIAN_H
#define MY_P4EST_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <iostream>

#ifdef P4_TO_P8
#include <p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/mls_integration/simplex3_mls_l.h>
#else
#include <p4est.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/mls_integration/simplex2_mls_l.h>
#endif

class my_p4est_semi_lagrangian_t
{
  friend class my_p4est_ns_free_surface_t;
  friend class my_p4est_surfactant_t;

protected:			// Added to allow child classes to access previously declared private members of this base class.
  /*
   * The pointer-to-pointer member variables (i.e. the 'p_***' variables here below) are required
   * to udpate the pointer to grid data that the user passed when constructing this object.
   * This object works on the objects pointed by p4est, nodes and ghost, it may modify them or even
   * destroy them create new ones. We want these changes of pointer to be 'transparent' to the user,
   * (so that they do not have to ask for a new pointer to the new grid data every time the grid changes)
   * --> this class updates the pointers as seen and provided by the user through the 'p_***' variables!
   * (A direct consequence is that this class does not take ownership of any such object, so it does not
   * destroy any grid data at the destructor)
   */
  p4est_t                   **p_p4est, *p4est;
  p4est_nodes_t             **p_nodes, *nodes;
  p4est_ghost_t             **p_ghost, *ghost;
  my_p4est_node_neighbors_t *ngbd_n;
  my_p4est_node_neighbors_t *ngbd_nm1;
  my_p4est_node_neighbors_t *ngbd_phi;

  void advect_from_n_to_np1(double dt, const CF_DIM **v,
                            Vec phi_n, Vec *phi_xx_n,
                            double *phi_np1);

  void advect_from_n_to_np1(double dt, Vec *v, Vec **vxx, Vec phi_n, Vec *phi_xx_n,
                            double *phi_np1);

  void advect_from_n_to_np1(double dt_nm1, double dt_n,
                            Vec *vnm1, Vec **vxx_nm1,
                            Vec *vn  , Vec **vxx_n,
                            Vec phi_n, Vec *phi_xx_n,
                            double *phi_np1);

  interpolation_method velo_interpolation;
  interpolation_method phi_interpolation;

  inline const double*  get_xyz_min()     const { return ngbd_n->get_brick()->xyz_min;                }
  inline const double*  get_xyz_max()     const { return ngbd_n->get_brick()->xyz_max;                }
  inline const int*     get_ntrees()      const { return ngbd_n->get_brick()->nxyztrees;              }
  inline const bool*    get_periodicity() const { return ngbd_n->get_hierarchy()->get_periodicity();  }

  inline double get_min_dx() const
  {
    splitting_criteria_t* data = (splitting_criteria_t*)p4est->user_pointer;
    const double *xyz_min = get_xyz_min();
    const double *xyz_max = get_xyz_max();
    const int* ntrees     = get_ntrees();
    return ((double)P4EST_QUADRANT_LEN(data->max_lvl)/(double)P4EST_ROOT_LEN)*MIN(DIM((xyz_max[0] - xyz_min[0])/ntrees[0], (xyz_max[1] - xyz_min[1])/ntrees[1], (xyz_max[2] - xyz_min[2])/ntrees[2]));
  }


public:
  my_p4est_semi_lagrangian_t(p4est_t **p4est_np1, p4est_nodes_t **nodes_np1,  p4est_ghost_t **ghost_np1,  my_p4est_node_neighbors_t *ngbd_n, my_p4est_node_neighbors_t *ngbd_nm1=NULL);

  inline void set_velo_interpolation(interpolation_method method) { velo_interpolation = method; }
  inline void set_phi_interpolation (interpolation_method method) { phi_interpolation  = method; }

  double compute_dt(DIM(const CF_DIM& vx, const CF_DIM& vy, const CF_DIM& vz));
  double compute_dt(DIM(Vec vx, Vec vy, Vec vz));

  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity field given as a continuous function. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt      the time step
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(const CF_DIM **v, double dt, Vec &phi, Vec *phi_xx=NULL);

  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity field. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt      the time step
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(Vec *v, double dt, Vec &phi, Vec *phi_xx=NULL, Vec phi_add_refine = NULL);

  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   *   This function gives you the option to refine not only around provided LSF(s), but also around additional scalar fields.
   *   Such fields may be provided either as an array of PETSc vectors with blocksize 1, or as a single PETSc block vector. The user must
   *   specify which type of input they are providing.
   *   Additionally, refine and coarsen criteria must be provided for each additional scalar field being provided, in a format that is described below.
   * \param v              [in]     the velocity field. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt             [in]     the time step
   * \param phi            [inout]  the level set function
   * \param phi_xx         [in]     the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   * \param phi_add_refine [in]     an additional level set function you may provide to refine around, but not advect
   * \param num_fields     [in]     number of scalar fields are being provided to refine/coarsen around, not including LSF(s)
   * \param use_block      [in]     a boolean specifying whether the provided scalar fields are in block vector format or an array of vectors. True --> block vector, false --> array of vectors
   * \param fields         [in]
   * \param fields_block   [in]
   * \param criteria       [in]
   * \param compare_opn    [in]
   * \param diag_opn       [in]
   * \param expand_ghost_layer [in]   a boolean specifying whether you want the new grid to have an expanded ghost layer
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(Vec *v, double dt, Vec &phi, Vec *phi_xx, Vec phi_add_refine, const unsigned int num_fields, bool use_block, bool enforce_uniform_band,double refine_band, double coarsen_band,Vec *fields, Vec fields_block, std::vector<double> criteria, std::vector<compare_option_t> compare_opn, std::vector<compare_diagonal_option_t> diag_opn, bool expand_ghost_layer);
  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with BDF along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param vnm1    the velocity field at time nm1 defined on p4est_n. This is a pointer to an array of dimension P4EST_DIM
   * \param vn      the velocity field at time n defined on p4est_n. This is a pointer to an array of dimension P4EST_DIM
   * \param dt_nm1  the time step from tnm1 to tn
   * \param dt_n    the time step from tn to tnp1
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(Vec *vnm1, Vec *vn, double dt_nm1, double dt_n, Vec &phi, Vec *phi_xx=NULL);


  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity fields. This is an array of size P4EST_DIM, each element is a vector with the list of velocities in the dimension
   * \param dt      the time step
   * \param phi     a vector of level set functions
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(std::vector<Vec> *v, double dt, std::vector<Vec> &phi);

  /*!
   * \brief update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity field. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt      the time step
   * \param phi     the level set function
   * \param phi_xx  the derivatives of the level set function. This is a pointer to an array of dimension P4EST_DIM
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(Vec *v, double dt, std::vector<Vec> &phi_parts, Vec &phi, Vec *phi_xx=NULL);

  /*!
   * \brief (multi level-set version) update a p4est from tn to tnp1, using a semi-Lagrangian scheme with Euler along the characteristic.
   *   The forest at time n is copied, and is then refined, coarsened and balance iteratively until convergence.
   * \param v       the velocity field. This is a pointer to an array of dimension P4EST_DIM.
   * \param dt      the time step
   * \param phi     the level set functions
   * \param action  the contructing operations
   * \param phi_idx the number of level-set function to be advected
   * \param phi_xx  the derivatives of the level set function to be advected. This is a pointer to an array of dimension P4EST_DIM
   * \note you need to update ngbd_n and hierarchy yourself !
   */
  void update_p4est(Vec *v, double dt, std::vector<Vec> &phi, std::vector<mls_opn_t> &action, int phi_idx, Vec *phi_xx=NULL);
  void update_p4est(Vec *vnm1, Vec *vn, double dt_nm1, double dt_n, std::vector<Vec> &phi, std::vector<mls_opn_t> &action, int phi_idx, Vec *phi_xx=NULL);

  void set_ngbd_phi(my_p4est_node_neighbors_t *ngbd_phi) { this->ngbd_phi = ngbd_phi; }

  /**
   * Update a p4est from tn to tnp1, using a semi-Lagrangian scheme with a single velocity step (no midpoint) with Euler
   * along the characteristics.
   * The forest at time tn is copied and then refined/coarsened and balance iteratively until convergence.
   * The method is based on:
   * [*] M. Mirzadeh, A. Guittet, C. Burstedde, and F. Gibou, Parallel Level-Set Method on Adaptive Tree-Based Grids.
   * @note You need to update the node neighborhood and hierarchy objects yourself!
   * @param [in] vel Array of velocity parallel vectors in each Cartesian direction.
   * @param [in] dt Time step.
   * @param [in,out] phi Level-set function values at time n, and then updated at time n + 1.
   * @param [in] band Desired minimum band around the interface, <= 1 to not refine based on band value.
   */
  void update_p4est_one_vel_step( Vec vel[], double dt, Vec& phi, double band=0 );

  /**
   * Advect level-set function using a semi-Lagrangian scheme with a single velocity step (no midpoint) with Euler along
   * the characteristics.
   * @param [in] dt Time step.
   * @param [in] vel Array of velocity parallel vectors in each Cartesian direction.
   * @param [in] vel_xx Array of second derivatives for each velocity component w.r.t. each Cartesian direction.
   * @param [in] phi Level-set function values at time n.
   * @param [in] phi_xx Array of second derivatives of phi w.r.t. each Cartesian direction.
   * @param [in,out] phiNewPtr Advected level-set function values.
   */
  void advect_from_n_to_np1_one_vel_step( double dt, Vec vel[], Vec *vel_xx[], Vec phi, Vec phi_xx[], double *phiNewPtr );

};

#endif // MY_P4EST_SEMI_LAGRANGIAN_H
