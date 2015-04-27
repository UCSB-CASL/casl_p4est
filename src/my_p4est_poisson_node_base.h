#ifndef POISSON_SOLVER_NODE_BASE_H
#define POISSON_SOLVER_NODE_BASE_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_utils.h>
#endif


#include<src/cube3.h>
#include<src/cube2.h>


class PoissonSolverNodeBase
{
public:

  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb_;
  InterpolatingFunctionNodeBase phi_interp;
#ifdef P4_TO_P8
  const CF_3* phi_cf;
#else
  const CF_2* phi_cf;
#endif

  double mu_, diag_add_;
  bool is_matrix_ready;
  bool matrix_has_nullspace;
  double dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc_;
#else
  BoundaryConditions2D *bc_;
#endif
  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  Vec rhs_, phi_, add_, phi_xx_, phi_yy_;
  Vec phi_is_all_positive;
  Vec phi_is_below_epsilon;
  PetscInt n_phi_is_all_positive;
  Vec is_crossed_neumann;
  PetscInt n_is_crossed_neuman;


#ifdef P4_TO_P8
  Vec phi_zz_;
#endif
  bool is_phi_dd_owned;
  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();

  // TODO: check how to optimize the memory allocation for a periodic matrix such that
  // the memory requirement is as the number of non zeros entries of the periodic matrix.
  /**
     * preallocate memory for the periodic Laplacian matrix
     * \param [in] na
     * \param [in] na
     * \param [in] na
     */
  void preallocate_periodic_matrix();


  /**
     * compute the spatial integral of the negative laplacian matrix taking into account
     * all the possible boundary conditions and scaling the elements up to the diagonal elements of the
     *  respective rows
     */
  void setup_negative_laplace_matrix();

  /**
     * compute the spatial integral of the rhs vector
     * and scale the elements with consistence with the laplacian matrix computed in the previous function
     */
  void setup_negative_laplace_rhsvec();  

  /**
     * setup a free periodic matrix
     * called from setupM2
     */
  void setupM();

  /**
     * setup a free periodic matrix
     * called from setupM2
     */
  void setupM2();

  /**
     * setup a free neuman matrix
     * called from setupM2
     */
  void setupM2_Neuman();


  /**
     * setup a free dirichlet matrix
     * called from setupM2
     */
  void setupM2_Dirichlet();


  /**
     * compute the spatial integral of the neuman matrix
     * taking into account the level set function and the
     * boundary conditions at the interface
     *
     */
  void setup_neuman_matrix();


  /**
     * compute the spatial integral of the neuman matrix
     * taking into account the level set function and the
     * boundary conditions at the interface
     *
     */
  void setup_dirichlet_matrix();


  /**
     * compute the spatial integral of the identity  matrix
     *
     */
  void setup_volume_matrix();

  /**
     * compute the spatial integral of the identity  matrix
     *
     */
  void setup_volume_matrix2();


  /**
     * compute the spatial interface integral of the identity  matrix
     * at the points where the zero level set is crossed
     */
  void setup_interface_matrix(Vec *v_interface);






  // disallow copy ctr and copy assignment
  PoissonSolverNodeBase(const PoissonSolverNodeBase& other);
  PoissonSolverNodeBase& operator=(const PoissonSolverNodeBase& other);

public:
  PoissonSolverNodeBase(const my_p4est_node_neighbors_t *node_neighbors);
  ~PoissonSolverNodeBase();

  // inlines setters
  /* FIXME: shouldn't those be references instead of copies ? I guess Vec is just a pointer ... but still ?
   * Mohammad: Vec is just a typedef to _p_Vec* so its merely a pointer under the hood.
   * If you are only passing the vector to access its data its fine to pass it as 'Vec v'
   * However, if 'v' is supposed to change itself, i.e. the the whole Vec object and not just its data
   * then it should either be passed via reference, Vec& v, or pointer, Vec* v, just like
   * any other object
   */
#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add; is_matrix_ready = false;}
  inline void set_diagonal(Vec add)
  {
      this->ierr=VecDuplicate(add,&this->add_); CHKERRXX(this->ierr);
      this->ierr=VecCopy(add,this->add_); CHKERRXX(this->ierr);

      VecGhostUpdateBegin(this->add_,INSERT_VALUES,SCATTER_FORWARD);
      VecGhostUpdateEnd(this->add_,INSERT_VALUES,SCATTER_FORWARD);

      this->is_matrix_ready = false;
  }
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {bc_       = &bc; is_matrix_ready = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {bc_       = &bc; is_matrix_ready = false;}
#endif
  inline void set_mu(double mu)                {mu_       = mu;  is_matrix_ready = false;}

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  std::string IO_path;//="/Users/gaddielouaknin/p4estLocal/";
  inline std::string convert2FullPath(std::string file_name)
  {
      std::stringstream oss;
      std::string mystr;
      oss <<this->IO_path <<file_name;
      mystr=oss.str();
      return mystr;
  }
  void printLaplaceMatrix();


  void print_quad_neighbor_nodes_of_node_t();
};

#endif // POISSON_SOLVER_NODE_BASE_H
