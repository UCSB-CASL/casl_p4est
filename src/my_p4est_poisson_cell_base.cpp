#ifdef P4_TO_P8
#include "my_p8est_poisson_cell_base.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#else
#include "my_p4est_poisson_cell_base.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_PoissonSolverCellBase_matrix_preallocation;
extern PetscLogEvent log_PoissonSolverCellBase_matrix_setup;
extern PetscLogEvent log_PoissonSolverCellBase_rhsvec_setup;
extern PetscLogEvent log_PoissonSolverCellBase_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0

PoissonSolverCellBase::PoissonSolverCellBase(const my_p4est_cell_neighbors_t *cell_neighbors,
                                             const my_p4est_node_neighbors_t *node_neighbors)
  : cell_neighbors_(cell_neighbors), node_neighbors_(node_neighbors),
    p4est_(cell_neighbors->p4est), nodes_(node_neighbors->nodes), ghost_(cell_neighbors->ghost), myb_(cell_neighbors->myb),
    phi_interp(p4est_, nodes_, ghost_, myb_, node_neighbors),
    phi_cf(NULL),
    mu_(1.), diag_add_(0.),
    is_matrix_ready(false), matrix_has_nullspace(false),
    bc_(NULL),
    A(NULL), A_null_space(NULL),
    rhs_(NULL), phi_(NULL), add_(NULL), phi_xx_(NULL), phi_yy_(NULL)
  #ifdef P4_TO_P8
  ,
    phi_zz_(NULL)
  #endif
{
  // set up the KSP solver
  ierr = KSPCreate(p4est_->mpicomm, &ksp); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  // compute grid parameters
  // NOTE: Assuming all trees are of the same size [0, 1]^d
  dx_min = 1.0 / pow(2.,(double) data->max_lvl);
  dy_min = dx_min;
#ifdef P4_TO_P8
  dz_min = dx_min;
#endif
#ifdef P4_TO_P8
  d_min = MIN(dx_min, dy_min, dz_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min + dz_min*dz_min);
#else
  d_min = MIN(dx_min, dy_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min);
#endif
}

PoissonSolverCellBase::~PoissonSolverCellBase()
{
  if (A             != NULL) ierr = MatDestroy(A);                      CHKERRXX(ierr);
  if (A_null_space  != NULL) ierr = MatNullSpaceDestroy (A_null_space); CHKERRXX(ierr);
  if (ksp           != NULL) ierr = KSPDestroy(ksp);                    CHKERRXX(ierr);
  if (is_phi_dd_owned){
    if (phi_xx_     != NULL) ierr = VecDestroy(phi_xx_);                CHKERRXX(ierr);
    if (phi_yy_     != NULL) ierr = VecDestroy(phi_yy_);                CHKERRXX(ierr);
#ifdef P4_TO_P8
    if (phi_zz_     != NULL) ierr = VecDestroy(phi_zz_);                CHKERRXX(ierr);
#endif
  }
}

void PoissonSolverCellBase::preallocate_matrix()
{
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_PoissonSolverCellBase_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = p4est_->global_num_quadrants;
  PetscInt num_owned_local  = p4est_->local_num_quadrants;

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  // set up the matrix
  ierr = MatCreate(p4est_->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATMPIAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);
  double *phi_p;
  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes_->local_nodes;
  const quad_info_t *cells[P4EST_FACES + 1];

  for (p4est_topidx_t tr_id = p4est_->first_local_tree; tr_id <= p4est_->last_local_tree; ++tr_id){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr_id);
    for (size_t q=0; q<tree->quadrants.elem_count; q++)
    {
      const p4est_quadrant_t *quad  = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      /* check if we are in the Omega^- domain */
      double phi_c = 0;
      for (short i=0; i<P4EST_CHILDREN; i++)
        phi_c += phi_p[q2n[quad_idx*P4EST_CHILDREN + i]];
      phi_c /= (double)P4EST_CHILDREN;

      if (phi_c > diag_min)
        continue;

      /*
     * Check for neighboring cells:
     * 1) If they exist and are local quads, increment d_nnz[n]
     * 2) If they exist but are not local quads, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */
      for (short i = 0; i<P4EST_FACES; i++)
        cells[i] = cell_neighbors_->begin(quad_idx, i);
      cells[P4EST_FACES] = cell_neighbors_->end(quad_idx, P4EST_FACES - 1); // use the last cell pointer as end iterator to make things easier to implement

      const quad_info_t *it;

      /* m00 direction */
      it = cells[dir::f_m00];
      if (it->level > quad->level) // all quadrants in m00 direction are smaller
        for (; it != cells[dir::f_m00 + 1]; ++it)
          it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      else if (it->level == quad->level) // one quadrant with the same size
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      else { // one larger quadrant -- must find all the neighbors in opposite direction
        // first add the cell itself
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;

        // now the neighbors in the opposite direction
        const quad_info_t *it_p00  = cell_neighbors_->begin(it->locidx, dir::f_p00);
        const quad_info_t *end     = cell_neighbors_->end(it->locidx, dir::f_p00);

        for (; it_p00 != end; ++it_p00)
          if (it_p00->locidx != quad_idx)
            it_p00->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      }

      /* p00 direction */
      it = cells[dir::f_p00];
      if (it->level > quad->level) {// all quadrants in p00 direction are smaller
        for (; it != cells[dir::f_p00 + 1]; ++it)
          it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else if (it->level == quad->level) {// one quadrant with the same size
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else { // one larger quadrant -- must find all the neighbors in opposite direction
        // first add the cell itself
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;

        // now the neighbors in the opposite direction
        const quad_info_t *it_m00  = cell_neighbors_->begin(it->locidx, dir::f_m00);
        const quad_info_t *end     = cell_neighbors_->end(it->locidx, dir::f_m00);

        for (; it_m00 != end; ++it_m00)
          if (it_m00->locidx != quad_idx)
            it_m00->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      }

      /* 0m0 direction */
      it = cells[dir::f_0m0];
      if (it->level > quad->level) {// all quadrants in 0m0 direction are smaller
        for (; it != cells[dir::f_0m0 + 1]; ++it)
          it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else if (it->level == quad->level) {// one quadrant with the same size
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else { // one larger quadrant -- must find all the neighbors in opposite direction
        // first add the cell itself
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;

        // now the neighbors in the opposite direction
        const quad_info_t *it_0p0  = cell_neighbors_->begin(it->locidx, dir::f_0p0);
        const quad_info_t *end     = cell_neighbors_->end(it->locidx, dir::f_0p0);

        for (; it_0p0 != end; ++it_0p0)
          if (it_0p0->locidx != quad_idx)
            it_0p0->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      }

      /* 0p0 direction */
      it = cells[dir::f_0p0];
      if (it->level > quad->level) {// all quadrants in 0p0 direction are smaller
        for (; it != cells[dir::f_0p0 + 1]; ++it)
          it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else if (it->level == quad->level) {// one quadrant with the same size
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else { // one larger quadrant -- must find all the neighbors in opposite direction
        // first add the cell itself
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;

        // now the neighbors in the opposite direction
        const quad_info_t *it_0m0  = cell_neighbors_->begin(it->locidx, dir::f_0m0);
        const quad_info_t *end     = cell_neighbors_->end(it->locidx, dir::f_0m0);

        for (; it_0m0 != end; ++it_0m0)
          if (it_0m0->locidx != quad_idx)
            it_0m0->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      }

#ifdef P4_TO_P8
      /* 00m direction */
      it = cells[dir::f_00m];
      if (it->level > quad->level) {// all quadrants in 00m direction are smaller
        for (; it != cells[dir::f_00m + 1]; ++it)
          it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else if (it->level == quad->level) {// one quadrant with the same size
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else { // one larger quadrant -- must find all the neighbors in opposite direction
        // first add the cell itself
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;

        // now the neighbors in the opposite direction
        const quad_info_t *it_00p  = cell_neighbors_->begin(it->locidx, dir::f_00p);
        const quad_info_t *end     = cell_neighbors_->end(it->locidx, dir::f_00p);

        for (; it_00p != end; ++it_00p)
          if (it_00p->locidx != quad_idx)
            it_00p->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      }

      /* 00p direction */
      it = cells[dir::f_00p];
      if (it->level > quad->level) {// all quadrants in 0p0 direction are smaller
        for (; it != cells[dir::f_00p + 1]; ++it)
          it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else if (it->level == quad->level) {// one quadrant with the same size
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      } else { // one larger quadrant -- must find all the neighbors in opposite direction
        // first add the cell itself
        it->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;

        // now the neighbors in the opposite direction
        const quad_info_t *it_00m  = cell_neighbors_->begin(it->locidx, dir::f_00m);
        const quad_info_t *end     = cell_neighbors_->end(it->locidx, dir::f_00m);

        for (; it_00m != end; ++it_00m)
          if (it_00m->locidx != quad_idx)
            it_00m->locidx < num_owned_local ? d_nnz[quad_idx]++ : o_nnz[quad_idx]++;
      }
#endif
    }
  }

  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverCellBase_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverCellBase::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_PoissonSolverCellBase_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(bc_ == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
    if (sol_size != p4est_->local_num_quadrants){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "solution.local_size = " << sol_size << " p4est->local_num_quadrants = " << p4est_->local_num_quadrants << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  // set local add if none was given
  bool local_add = false;
  if(add_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, p4est_->local_num_quadrants, &add_); CHKERRXX(ierr);
    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
  }

  // set a local phi if not was given
  bool local_phi = false;
  if(phi_ == NULL)
  {
    local_phi = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes_->num_owned_indeps, &phi_); CHKERRXX(ierr);
    ierr = VecSet(phi_, -1.); CHKERRXX(ierr);
  }

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_ready)
  {
    matrix_has_nullspace = true;
    setup_negative_laplace_matrix();

    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
    ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);  CHKERRXX(ierr);
  }

  // setup rhs
  setup_negative_laplace_rhsvec();

  // set the null-space if necessary
  if (matrix_has_nullspace)
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

  // Solve the system
  ierr = KSPSolve(ksp, rhs_, solution); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(add_); CHKERRXX(ierr);
    add_ = NULL;
  }
  if(local_phi)
  {
    ierr = VecDestroy(phi_); CHKERRXX(ierr);
    phi_ = NULL;
  }

  ierr = PetscLogEventEnd(log_PoissonSolverCellBase_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);
}

void PoissonSolverCellBase::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_PoissonSolverCellBase_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
  double *phi_p, *add_p, *phi_xx_p, *phi_yy_p;

  ierr = VecGetArray(phi_,    &phi_p);    CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArray(add_,    &add_p);    CHKERRXX(ierr);

#ifdef P4_TO_P8
  Cube3 cube;
#else
  Cube2 cube;
#endif

  // NOTE: d_eps should depend on the domain size which here we assume is 1.0
  double d_eps = 1.0 * EPS;
  const p4est_gloidx_t *gloidx = p4est_->global_first_quadrant;
  const p4est_locidx_t *q2n    = nodes_->local_nodes;
  const quad_info_t *cells[P4EST_FACES + 1];

  // Main loop over all local trees
  for(p4est_topidx_t tr_id = p4est_->first_local_tree; tr_id <= p4est_->last_local_tree; ++tr_id){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr_id);

    p4est_topidx_t v_mmm = p4est_->connectivity->tree_to_vertex[tr_id*P4EST_CHILDREN + 0];
    double tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
    double tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
    double tr_zmin = p4est_->connectivity->vertices[3*v_mmm + 2];
#endif

    // loop over all quadrants of this tree
    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t qu_locidx = q + tree->quadrants_offset;
      PetscInt qu_gloidx = qu_locidx + gloidx[p4est_->mpirank];

      p4est_locidx_t n_mmm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mmm];
      p4est_locidx_t n_pmm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_pmm];
      p4est_locidx_t n_mpm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mpm];
      p4est_locidx_t n_ppm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_ppm];
#ifdef P4_TO_P8
      p4est_locidx_t n_mmp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mmp];
      p4est_locidx_t n_pmp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_pmp];
      p4est_locidx_t n_mpp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mpp];
      p4est_locidx_t n_ppp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_ppp];
#endif

      // Get the cell coordinates and evaluate the level-set at the cell center
      double p_C = phi_cell(qu_locidx, phi_p);

      // NOTE: assuming quadrants are squares. This is only true if macro blocks are squares (which they are for now)
      double dx_C = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dy_C = dx_C;
#ifdef P4_TO_P8
      double dz_C = dx_C;
#endif
      double x_C = quad_x_fr_i(quad) + 0.5*dx_C + tr_xmin;
      double y_C = quad_y_fr_j(quad) + 0.5*dy_C + tr_ymin;
#ifdef P4_TO_P8
      double z_C = quad_y_fr_j(quad) + 0.5*dz_C + tr_zmin;
#endif

      // Interface condition
#ifdef P4_TO_P8
      bool is_cell_in_omega_plus = phi_p[n_mmm] > 0.0 && phi_p[n_pmm] > 0.0 && phi_p[n_mpm] > 0.0 && phi_p[n_ppm] > 0.0 &&
                                   phi_p[n_mmp] > 0.0 && phi_p[n_pmp] > 0.0 && phi_p[n_mpp] > 0.0 && phi_p[n_ppp] > 0.0 ;
#else
      bool is_cell_in_omega_plus = phi_p[n_mmm] > 0.0 && phi_p[n_pmm] > 0.0 && phi_p[n_mpm] > 0.0 && phi_p[n_ppm] > 0.0 ;
#endif

      // Way inside omega_plus and we dont care!
      if (is_cell_in_omega_plus || (p_C>0 && bc_->interfaceType()==DIRICHLET)){
        ierr = MatSetValue(A, qu_gloidx, qu_gloidx, 1.0, ADD_VALUES); CHKERRXX(ierr);
        continue;
      }

      cube.x0 = x_C - 0.5*dx_C;
      cube.x1 = x_C + 0.5*dx_C;
      cube.y0 = y_C - 0.5*dy_C;
      cube.y1 = y_C + 0.5*dy_C;

#ifdef P4_TO_P8
      OctValue  phi_buffer(phi_p[n_mmm], phi_p[n_mmp],
                           phi_p[n_mpm], phi_p[n_mpp],
                           phi_p[n_pmm], phi_p[n_pmp],
                           phi_p[n_ppm], phi_p[n_ppp]);

      cube.z0 = z_C - 0.5*dz_C;
      cube.z1 = z_C + 0.5*dz_C;
      double volume_cut_cell = cube.volume_In_Negative_Domain(phi_buffer);
#else
      QuadValue phi_buffer(phi_p[n_mmm],
                           phi_p[n_mpm],
                           phi_p[n_pmm],
                           phi_p[n_ppm]);
      double volume_cut_cell = cube.area_In_Negative_Domain(phi_buffer);
#endif


      // First add the diagonal term:
      ierr = MatSetValue(A, qu_gloidx, qu_gloidx, volume_cut_cell*add_p[qu_locidx], ADD_VALUES); CHKERRXX(ierr);

      /* get aceess to the neighboring cells of this cell */
      for (short i = 0; i<P4EST_FACES; i++)
        cells[i] = cell_neighbors_->begin(qu_locidx, i);
      cells[P4EST_FACES] = cell_neighbors_->end(qu_locidx, P4EST_FACES - 1); // last one used as the end iterator

      /*
       * For any direction of search (m00, p00, 0m0, 0p0, 00m, 00p), 3 cases can occur (examples for m00):
       * 1) There is only one cell to m00 and is larger: In this case, we need to get the
       * ngbs cells on the p00 side of the large cell
       * 2) There is only one cell to the left and is the same size. This is a regular cell
       * 3) There are multiple cells to the left: This is the easy case
       */

      /* 1 - m00 */
      double fxxa, fxxb;
      double theta;

#ifdef CASL_THROWS
      if (cells[dir::f_m00] == cells[dir::f_m00 + 1] && !is_quad_xmWall(p4est_, tr_id, quad))
        throw std::logic_error("[CASL_ERROR]: no ngbd cells were found for a non-boundary cell");
#endif

      // Case 0 - boundary cells
      if (cells[dir::f_m00] == cells[dir::f_m00 + 1]){ // no cell found -- must be boundary cell
#ifdef P4_TO_P8
        switch(bc_->wallType(x_C - 0.5*dx_C, y_C, z_C))
#else
        switch(bc_->wallType(x_C - 0.5*dx_C, y_C))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace = false;
          fxxa = phi_yy_p[n_mmm];
          fxxb = phi_yy_p[n_mpm];
          theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val00,phi_buffer.val01,fxxa,fxxb,dy_min);

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*2.0 * theta*dy_C/dx_C, ADD_VALUES); CHKERRXX(ierr);
          break;

        case NEUMANN:
          /* Nothing to be done here */
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }
      else if (cells[dir::f_m00] + 1 == cells[dir::f_m00 + 1]){ // one cell in m00 -- cases 1,2
        const quad_info_t* it_m00 = cells[dir::f_m00];

        double p_m00  = phi_cell(it_m00->locidx, phi_p);
        double dy_m00 = (double)P4EST_QUADRANT_LEN(it_m00->level)/(double)P4EST_ROOT_LEN;

        // Crossing the interface dirichlet
        if( bc_->interfaceType()==DIRICHLET && ABS(p_C)<=diag_min && p_C<=0. && p_m00*p_C < 0. )
        {
          matrix_has_nullspace = false;
          double theta_m00 = interface_Location(0., dx_C, p_C, p_m00)/dx_C;
          if (theta_m00<d_eps) theta_m00 = d_eps;
          if (theta_m00>1.0)   theta_m00 = 1.0;

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*1.0/theta_m00 * dy_C/dx_C, ADD_VALUES); CHKERRXX(ierr);
        }
        // Crossing the interface neumann
        else if( bc_->interfaceType()==NEUMANN && phi_buffer.val00*phi_buffer.val01 <= 0. )
        {
          fxxa = phi_yy_p[n_mmm];
          fxxb = phi_yy_p[n_mpm];
          dy_m00 *= fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val00,phi_buffer.val01,fxxa,fxxb,dy_min);

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx,       mu_*dy_m00/dx_C, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, qu_gloidx, it_m00->gloidx, -mu_*dy_m00/dx_C, ADD_VALUES); CHKERRXX(ierr);
        }
        else if(phi_buffer.val00<0 || phi_buffer.val01<0)
        {
          // get the ngbd cells on the right side of the left cell
          const quad_info_t* begin = cell_neighbors_->begin(it_m00->locidx, dir::f_p00);
          const quad_info_t* end   = cell_neighbors_->end(it_m00->locidx, dir::f_p00);
          const quad_info_t* it    = begin;

          std::vector<double> dy_ng (end - begin);
          for (int i = 0; i<end - begin; ++i, ++it)
            dy_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

          // First let's compute the average dx
          double dx = 0.0;
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            dx += dy_ng[i]/dy_m00 * 0.5 *  (double)(P4EST_QUADRANT_LEN(it_m00->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

          // Now add the contribution from all the cells on the right side of left cell
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            ierr = MatSetValue(A, qu_gloidx, it->gloidx, mu_*dy_C * dy_ng[i]/dy_m00/dx, ADD_VALUES); CHKERRXX(ierr);

          // Add the contribution from the left cell itself
          ierr = MatSetValue(A, qu_gloidx, it_m00->gloidx, -mu_*dy_C/dx, ADD_VALUES); CHKERRXX(ierr);
        }
      }
      else { // Case 3
        const quad_info_t *begin = cells[dir::f_m00];
        const quad_info_t *end   = cells[dir::f_m00 + 1];
        const quad_info_t *it    = begin;

        std::vector<double> dy_ng (end - begin);
        for (int i = 0; i<end - begin; ++i, ++it)
          dy_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

        // First let's compute the average dx
        double dx = 0.0;
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          dx += dy_ng[i]/dy_C * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

        // Add contribution from all the cells to the left
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          ierr = MatSetValue(A, qu_gloidx, it->gloidx, -mu_*dy_ng[i]/dx, ADD_VALUES); CHKERRXX(ierr);

        // Add contribution from the big cell (which is simply l itself)
        ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*dy_C/dx, ADD_VALUES); CHKERRXX(ierr);
      }

      /* 2 - p00 */
#ifdef CASL_THROWS
      if (cells[dir::f_p00] == cells[dir::f_p00 + 1] && !is_quad_xpWall(p4est_, tr_id, quad))
        throw std::logic_error("[CASL_ERROR]: no ngbd cells were found for a non-boundary cell");
#endif

      // Case 0 - boundary cells
      if (cells[dir::f_p00] == cells[dir::f_p00 + 1]){ // no cell found -- must be boundary cell
#ifdef P4_TO_P8
        switch(bc_->wallType(x_C + 0.5*dx_C, y_C, z_C))
#else
        switch(bc_->wallType(x_C + 0.5*dx_C, y_C))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace = false;
          fxxa = phi_yy_p[n_pmm];
          fxxb = phi_yy_p[n_ppm];
          theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val10,phi_buffer.val11,fxxa,fxxb,dy_min);

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*2.0 * theta*dy_C/dx_C, ADD_VALUES); CHKERRXX(ierr);
          break;

        case NEUMANN:
          /* Nothing to be done here */
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }
      else if (cells[dir::f_p00] + 1 == cells[dir::f_p00 + 1]){ // one cell in p00 -- cases 1,2
        const quad_info_t* it_p00 = cells[dir::f_p00];

        double p_p00  = phi_cell(it_p00->locidx, phi_p);
        double dy_p00 = (double)P4EST_QUADRANT_LEN(it_p00->level)/(double)P4EST_ROOT_LEN;

        // Crossing the interface dirichlet
        if( bc_->interfaceType()==DIRICHLET && ABS(p_C)<=diag_min && p_C<=0. && p_p00*p_C < 0. )
        {
          matrix_has_nullspace = false;
          double theta_p00 = interface_Location(0., dx_C, p_C, p_p00)/dx_C;
          if (theta_p00<d_eps) theta_p00 = d_eps;
          if (theta_p00>1.0)   theta_p00 = 1.0;

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*1.0/theta_p00 * dy_C/dx_C, ADD_VALUES); CHKERRXX(ierr);
        }
        // Crossing the interface neumann
        else if( bc_->interfaceType()==NEUMANN && phi_buffer.val10*phi_buffer.val11 <= 0. )
        {
          fxxa = phi_yy_p[n_pmm];
          fxxb = phi_yy_p[n_ppm];
          dy_p00 *= fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val10,phi_buffer.val11,fxxa,fxxb,dy_min);

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx,       mu_*dy_p00/dx_C, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, qu_gloidx, it_p00->gloidx, -mu_*dy_p00/dx_C, ADD_VALUES); CHKERRXX(ierr);
        }
        else if(phi_buffer.val10<0 || phi_buffer.val11<0)
        {
          // get the ngbd cells on the right side of the left cell
          const quad_info_t* begin = cell_neighbors_->begin(it_p00->locidx, dir::f_m00);
          const quad_info_t* end   = cell_neighbors_->end(it_p00->locidx, dir::f_m00);
          const quad_info_t* it    = begin;

          std::vector<double> dy_ng (end - begin);
          for (int i = 0; i<end - begin; ++i, ++it)
            dy_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

          // First let's compute the average dx
          double dx = 0.0;
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            dx += dy_ng[i]/dy_p00 * 0.5 *  (double)(P4EST_QUADRANT_LEN(it_p00->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

          // Now add the contribution from all the cells on the right side of left cell
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            ierr = MatSetValue(A, qu_gloidx, it->gloidx, mu_*dy_C * dy_ng[i]/dy_p00/dx, ADD_VALUES); CHKERRXX(ierr);

          // Add the contribution from the left cell itself
          ierr = MatSetValue(A, qu_gloidx, it_p00->gloidx, -mu_*dy_C/dx, ADD_VALUES); CHKERRXX(ierr);
        }
      }
      else { // Case 3
        const quad_info_t *begin = cells[dir::f_p00];
        const quad_info_t *end   = cells[dir::f_p00 + 1];
        const quad_info_t *it    = begin;

        std::vector<double> dy_ng (end - begin);
        for (int i = 0; i<end - begin; ++i, ++it)
          dy_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

        // First let's compute the average dx
        double dx = 0.0;
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          dx += dy_ng[i]/dy_C * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

        // Add contribution from all the cells to the left
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          ierr = MatSetValue(A, qu_gloidx, it->gloidx, -mu_*dy_ng[i]/dx, ADD_VALUES); CHKERRXX(ierr);

        // Add contribution from the big cell (which is simply l itself)
        ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*dy_C/dx, ADD_VALUES); CHKERRXX(ierr);
      }

      /* 3 - 0m0 */
#ifdef CASL_THROWS
      if (cells[dir::f_0m0] == cells[dir::f_0m0 + 1] && !is_quad_ymWall(p4est_, tr_id, quad))
        throw std::logic_error("[CASL_ERROR]: no ngbd cells were found for a non-boundary cell");
#endif

      if (cells[dir::f_0m0] == cells[dir::f_0m0 + 1]){ // no cell found -- must be boundary cell
#ifdef P4_TO_P8
        switch(bc_->wallType(x_C, y_C - 0.5*dy_C, z_C))
#else
        switch(bc_->wallType(x_C, y_C - 0.5*dy_C))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace = false;
          fxxa = phi_xx_p[n_mmm];
          fxxb = phi_xx_p[n_pmm];

          theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val00,phi_buffer.val10,fxxa,fxxb,dx_min);
          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*2.0 * theta*dx_C/dy_C, ADD_VALUES); CHKERRXX(ierr);

          break;
        case NEUMANN:
          /* Nothing to be done here */
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }
      else if (cells[dir::f_0m0] + 1 == cells[dir::f_0m0 +1]){ // one cell in 0m0 -- cases 1,2
        const quad_info_t* it_0m0 = cells[dir::f_0m0];

        double p_0m0  = phi_cell(it_0m0->locidx, phi_p);
        double dx_0m0 = (double)P4EST_QUADRANT_LEN(it_0m0->level)/(double)P4EST_ROOT_LEN;

        // Crossing the interface dirichlet
        if( bc_->interfaceType()==DIRICHLET && ABS(p_C)<=diag_min && p_C<=0. && p_0m0*p_C < 0. )
        {
          matrix_has_nullspace = false;
          double theta_0m0 = interface_Location(0., dy_C, p_C, p_0m0)/dy_C;
          if (theta_0m0<d_eps) theta_0m0 = d_eps;
          if (theta_0m0>1.0)   theta_0m0 = 1.0;

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*1.0/theta_0m0 * dx_C/dy_C, ADD_VALUES); CHKERRXX(ierr);
        }
        // Crossing the interface neumann
        else if( bc_->interfaceType()==NEUMANN && phi_buffer.val00*phi_buffer.val10 <= 0. )
        {
          fxxa = phi_xx_p[n_mmm];
          fxxb = phi_xx_p[n_pmm];
          dx_0m0 *= fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val00,phi_buffer.val10,fxxa,fxxb,dx_min);

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx,       mu_*dx_0m0/dy_C, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, qu_gloidx, it_0m0->gloidx, -mu_*dx_0m0/dy_C, ADD_VALUES); CHKERRXX(ierr);
        }
        else if(phi_buffer.val00<0 || phi_buffer.val10<0)
        {
          // get the ngbd cells on the 0p0 side of the 0m0 cell
          const quad_info_t* begin = cell_neighbors_->begin(it_0m0->locidx, dir::f_0p0);
          const quad_info_t* end   = cell_neighbors_->end(it_0m0->locidx, dir::f_0p0);
          const quad_info_t* it    = begin;

          std::vector<double> dx_ng (end - begin);
          for (int i = 0; i<end - begin; ++i, ++it)
            dx_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

          // First let's compute the average dy
          double dy = 0.0;
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            dy += dx_ng[i]/dx_0m0 * 0.5 * (double)(P4EST_QUADRANT_LEN(it_0m0->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

          // Now add the contribution from all the cells on the top side of bottom cell
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            ierr = MatSetValue(A, qu_gloidx, it->gloidx, mu_*dx_C * dx_ng[i]/dx_0m0/dy, ADD_VALUES); CHKERRXX(ierr);

          ierr = MatSetValue(A, qu_gloidx, it_0m0->gloidx, -mu_*dx_C/dy, ADD_VALUES); CHKERRXX(ierr);
        }
      } else { // Case 3
        const quad_info_t *begin = cells[dir::f_0m0];
        const quad_info_t *end   = cells[dir::f_0m0 + 1];
        const quad_info_t *it    = begin;

        std::vector<double> dx_ng (end - begin);
        for (int i = 0; i<end - begin; ++i, ++it)
          dx_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

        // First let's compute the average dx
        double dy = 0.0;
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          dy += dx_ng[i]/dx_C * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

        // Add contribution from all the cells to the bottom
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          ierr = MatSetValue(A, qu_gloidx, it->gloidx, -mu_*dx_ng[i]/dy, ADD_VALUES); CHKERRXX(ierr);

        // Add contribution from the big cell (which is simply l itself)
        ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*dx_C/dy, ADD_VALUES); CHKERRXX(ierr);
      }

      /* 4 - 0p0 */
#ifdef CASL_THROWS
      if (cells[dir::f_0p0] == cells[dir::f_0p0 + 1] && !is_quad_ypWall(p4est_, tr_id, quad))
        throw std::logic_error("[CASL_ERROR]: no ngbd cells were found for a non-boundary cell");
#endif

      if (cells[dir::f_0p0] == cells[dir::f_0p0 + 1]){ // no cell found -- must be boundary cell
#ifdef P4_TO_P8
        switch(bc_->wallType(x_C, y_C + 0.5*dy_C, z_C))
#else
        switch(bc_->wallType(x_C, y_C + 0.5*dy_C))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace = false;
          fxxa = phi_xx_p[n_mpm];
          fxxb = phi_xx_p[n_ppm];

          theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val01,phi_buffer.val11,fxxa,fxxb,dx_min);
          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*2.0 * theta*dx_C/dy_C, ADD_VALUES); CHKERRXX(ierr);

          break;
        case NEUMANN:
          /* Nothing to be done here */
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }
      else if (cells[dir::f_0p0] + 1 == cells[dir::f_0p0 +1]){ // one cell in 0p0 -- cases 1,2
        const quad_info_t* it_0p0 = cells[dir::f_0p0];

        double p_0p0  = phi_cell(it_0p0->locidx, phi_p);
        double dx_0p0 = (double)P4EST_QUADRANT_LEN(it_0p0->level)/(double)P4EST_ROOT_LEN;

        // Crossing the interface dirichlet
        if( bc_->interfaceType()==DIRICHLET && ABS(p_C)<=diag_min && p_C<=0. && p_0p0*p_C < 0. )
        {
          matrix_has_nullspace = false;
          double theta_0p0 = interface_Location(0., dy_C, p_C, p_0p0)/dy_C;
          if (theta_0p0<d_eps) theta_0p0 = d_eps;
          if (theta_0p0>1.0)   theta_0p0 = 1.0;

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*1.0/theta_0p0 * dx_C/dy_C, ADD_VALUES); CHKERRXX(ierr);
        }
        // Crossing the interface neumann
        else if( bc_->interfaceType()==NEUMANN && phi_buffer.val01*phi_buffer.val11 <= 0. )
        {
          fxxa = phi_xx_p[n_mpm];
          fxxb = phi_xx_p[n_ppm];
          dx_0p0 *= fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_buffer.val01,phi_buffer.val11,fxxa,fxxb,dx_min);

          ierr = MatSetValue(A, qu_gloidx, qu_gloidx,       mu_*dx_0p0/dy_C, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, qu_gloidx, it_0p0->gloidx, -mu_*dx_0p0/dy_C, ADD_VALUES); CHKERRXX(ierr);
        }
        else if(phi_buffer.val01<0 || phi_buffer.val11<0)
        {
          // get the ngbd cells on the 0m0 side of the 0p0 cell
          const quad_info_t* begin = cell_neighbors_->begin(it_0p0->locidx, dir::f_0m0);
          const quad_info_t* end   = cell_neighbors_->end(it_0p0->locidx, dir::f_0m0);
          const quad_info_t* it    = begin;

          std::vector<double> dx_ng (end - begin);
          for (int i = 0; i<end - begin; ++i, ++it)
            dx_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

          // First let's compute the average dy
          double dy = 0.0;
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            dy += dx_ng[i]/dx_0p0 * 0.5 * (double)(P4EST_QUADRANT_LEN(it_0p0->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

          // Now add the contribution from all the cells on the top side of bottom cell
          it = begin;
          for (int i = 0; i<end - begin; ++i, ++it)
            ierr = MatSetValue(A, qu_gloidx, it->gloidx, mu_*dx_C * dx_ng[i]/dx_0p0/dy, ADD_VALUES); CHKERRXX(ierr);

          ierr = MatSetValue(A, qu_gloidx, it_0p0->gloidx, -mu_*dx_C/dy, ADD_VALUES); CHKERRXX(ierr);
        }
      } else { // Case 3
        const quad_info_t *begin = cells[dir::f_0p0];
        const quad_info_t *end   = cells[dir::f_0p0 + 1];
        const quad_info_t *it    = begin;

        std::vector<double> dx_ng (end - begin);
        for (int i = 0; i<end - begin; ++i, ++it)
          dx_ng[i] = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;

        // First let's compute the average dx
        double dy = 0.0;
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          dy += dx_ng[i]/dx_C * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(it->level))/(double)P4EST_ROOT_LEN;

        // Add contribution from all the cells to the bottom
        it = begin;
        for (int i = 0; i<end - begin; ++i, ++it)
          ierr = MatSetValue(A, qu_gloidx, it->gloidx, -mu_*dx_ng[i]/dy, ADD_VALUES); CHKERRXX(ierr);

        // Add contribution from the big cell (which is simply l itself)
        ierr = MatSetValue(A, qu_gloidx, qu_gloidx, mu_*dx_C/dy, ADD_VALUES); CHKERRXX(ierr);
      }
    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);

  // check for null space
  if (matrix_has_nullspace)
  {
    if (A_null_space == NULL) // pun not intended!
      ierr = MatNullSpaceCreate(p4est_->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);

    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverCellBase_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverCellBase::setup_negative_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_PoissonSolverCellBase_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double   *phi_p, *phi_xx_p, *phi_yy_p, *add_p, *rhs_p;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_,    &rhs_p   ); CHKERRXX(ierr);

  // NOTE: this assumes that the domanin is of constant size [0, 1]^d
  double d_eps = 1.0 * EPS;

#ifdef P4_TO_P8
  Cube3 cube;
#else
  Cube2 cube;
#endif

  const p4est_locidx_t *q2n = nodes_->local_nodes;

  // Main loop over all local quadrant
  for(p4est_topidx_t tr_id = p4est_->first_local_tree; tr_id <= p4est_->last_local_tree; ++tr_id){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tr_id);

    p4est_topidx_t v_mmm = p4est_->connectivity->tree_to_vertex[tr_id*P4EST_CHILDREN + 0];
    double tr_xmin = p4est_->connectivity->vertices[3*v_mmm + 0];
    double tr_ymin = p4est_->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
    double tr_zmin = p4est_->connectivity->vertices[3*v_mmm + 2];
#endif

    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t qu_locidx = q + tree->quadrants_offset;

      p4est_locidx_t n_mmm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mmm];
      p4est_locidx_t n_pmm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_pmm];
      p4est_locidx_t n_mpm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mpm];
      p4est_locidx_t n_ppm = q2n[qu_locidx*P4EST_CHILDREN + dir::v_ppm];
#ifdef P4_TO_P8
      p4est_locidx_t n_mmp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mmp];
      p4est_locidx_t n_pmp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_pmp];
      p4est_locidx_t n_mpp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_mpp];
      p4est_locidx_t n_ppp = q2n[qu_locidx*P4EST_CHILDREN + dir::v_ppp];
#endif

      // Get the cell coordinates and evaluate the level-set at the cell center
      double p_C = phi_cell(qu_locidx, phi_p);

      // NOTE: assuming quadrants are squares. This is only true if macro blocks are squares (which they are for now)
      double dx_C = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dy_C = dx_C;
#ifdef P4_TO_P8
      double dz_C = dx_C;
#endif
      double x_C = quad_x_fr_i(quad) + 0.5*dx_C + tr_xmin;
      double y_C = quad_y_fr_j(quad) + 0.5*dy_C + tr_ymin;
#ifdef P4_TO_P8
      double z_C = quad_z_fr_k(quad) + 0.5*dz_C + tr_zmin;
#endif

#ifdef P4_TO_P8
      bool is_cell_in_omega_plus  = phi_p[n_mmm] >  0.0 && phi_p[n_pmm] >  0.0 && phi_p[n_mpm] >  0.0 && phi_p[n_ppm] >  0.0 &&
                                    phi_p[n_mmp] >  0.0 && phi_p[n_pmp] >  0.0 && phi_p[n_mpp] >  0.0 && phi_p[n_ppp] >  0.0 ;

      bool is_cell_in_omega_minus = phi_p[n_mmm] <= 0.0 && phi_p[n_pmm] <= 0.0 && phi_p[n_mpm] <= 0.0 && phi_p[n_ppm] <= 0.0 &&
                                    phi_p[n_mmp] <= 0.0 && phi_p[n_pmp] <= 0.0 && phi_p[n_mpp] <= 0.0 && phi_p[n_ppp] <= 0.0 ;
#else
      bool is_cell_in_omega_plus  = phi_p[n_mmm] >  0.0 && phi_p[n_pmm] >  0.0 && phi_p[n_mpm] >  0.0 && phi_p[n_ppm] >  0.0 ;
      bool is_cell_in_omega_minus = phi_p[n_mmm] <= 0.0 && phi_p[n_pmm] <= 0.0 && phi_p[n_mpm] <= 0.0 && phi_p[n_ppm] <= 0.0 ;
#endif

      // Way inside omega_plus and we dont care!
      if (is_cell_in_omega_plus || (p_C>0 && bc_->interfaceType()==DIRICHLET)){
        rhs_p[qu_locidx] = 0.0;
        continue;
      }
      bool is_cell_crossed_by_interface = !(is_cell_in_omega_minus || is_cell_in_omega_plus);

      cube.x0 = x_C - 0.5*dx_C;
      cube.x1 = x_C + 0.5*dx_C;
      cube.y0 = y_C - 0.5*dy_C;
      cube.y1 = y_C + 0.5*dy_C;

#ifdef P4_TO_P8
      OctValue  phi_buffer(phi_p[n_mmm], phi_p[n_mmp],
                           phi_p[n_mpm], phi_p[n_mpp],
                           phi_p[n_pmm], phi_p[n_pmp],
                           phi_p[n_ppm], phi_p[n_ppp]);

      cube.z0 = z_C - 0.5*dz_C;
      cube.z1 = z_C + 0.5*dz_C;
      double volume_cut_cell = cube.volume_In_Negative_Domain(phi_buffer);
#else
      QuadValue phi_buffer(phi_p[n_mmm],
                           phi_p[n_mpm],
                           phi_p[n_pmm],
                           phi_p[n_ppm]);
      double volume_cut_cell = cube.area_In_Negative_Domain(phi_buffer);
#endif
      rhs_p[qu_locidx] *= volume_cut_cell;

      // Neumann BC
      if( is_cell_crossed_by_interface && bc_->interfaceType() == NEUMANN )
      {
        // FIXME: this can be done much more efficiently if we pass the vec itself
#ifdef P4_TO_P8
        OctValue interface_values;
        interface_values.val000 = bc_->interfaceValue(x_C - 0.5*dx_C, y_C - 0.5*dy_C, z_C - 0.5*dz_C);
        interface_values.val001 = bc_->interfaceValue(x_C - 0.5*dx_C, y_C - 0.5*dy_C, z_C + 0.5*dz_C);
        interface_values.val010 = bc_->interfaceValue(x_C - 0.5*dx_C, y_C + 0.5*dy_C, z_C - 0.5*dz_C);
        interface_values.val011 = bc_->interfaceValue(x_C - 0.5*dx_C, y_C + 0.5*dy_C, z_C + 0.5*dz_C);
        interface_values.val100 = bc_->interfaceValue(x_C + 0.5*dx_C, y_C - 0.5*dy_C, z_C - 0.5*dz_C);
        interface_values.val101 = bc_->interfaceValue(x_C + 0.5*dx_C, y_C - 0.5*dy_C, z_C + 0.5*dz_C);
        interface_values.val110 = bc_->interfaceValue(x_C + 0.5*dx_C, y_C + 0.5*dy_C, z_C - 0.5*dz_C);
        interface_values.val111 = bc_->interfaceValue(x_C + 0.5*dx_C, y_C + 0.5*dy_C, z_C + 0.5*dz_C);
#else
        QuadValue interface_values;
        interface_values.val00 = bc_->interfaceValue(x_C - 0.5*dx_C, y_C - 0.5*dy_C);
        interface_values.val01 = bc_->interfaceValue(x_C - 0.5*dx_C, y_C + 0.5*dy_C);
        interface_values.val10 = bc_->interfaceValue(x_C + 0.5*dx_C, y_C - 0.5*dy_C);
        interface_values.val11 = bc_->interfaceValue(x_C + 0.5*dx_C, y_C + 0.5*dy_C);
#endif

        double val_interface = cube.integrate_Over_Interface(interface_values,phi_buffer);
        rhs_p[qu_locidx] += mu_*val_interface;
      }

      // Dirichlet BC
      if (ABS(p_C)<=diag_min && p_C<=0. && bc_->interfaceType() == DIRICHLET){
        if(!is_quad_xmWall(p4est_, tr_id, quad))
        {
          const quad_info_t * it_m00 = cell_neighbors_->begin(qu_locidx, dir::f_m00);
          double p_m00 = phi_cell(it_m00->locidx, phi_p);

          double theta_m00 = interface_Location(0., dx_C, p_C, p_m00)/dx_C;
          if (theta_m00 < d_eps) theta_m00 = d_eps;
          if (theta_m00 > 1.0  ) theta_m00 = 1.0;

          if (p_m00*p_C <= 0.) {
            double val_interface_m00 = bc_->interfaceValue(x_C - theta_m00 * dx_C, y_C);
            rhs_p[qu_locidx] += mu_*dy_C/dx_C * val_interface_m00/theta_m00;
          }
        }

        if(!is_quad_xpWall(p4est_, tr_id, quad))
        {
          const quad_info_t * it_p00 = cell_neighbors_->begin(qu_locidx, dir::f_p00);
          double p_p00 = phi_cell(it_p00->locidx, phi_p);

          double theta_p00 = interface_Location(0., dx_C, p_C, p_p00)/dx_C;
          if (theta_p00 < d_eps) theta_p00 = d_eps;
          if (theta_p00 > 1.0  ) theta_p00 = 1.0;

          if (p_p00*p_C <= 0.) {
            double val_interface_p00 = bc_->interfaceValue(x_C + theta_p00 * dx_C, y_C);
            rhs_p[qu_locidx] += mu_*dy_C/dx_C * val_interface_p00/theta_p00;
          }
        }

        if(!is_quad_ymWall(p4est_, tr_id, quad))
        {
          const quad_info_t * it_0m0 = cell_neighbors_->begin(qu_locidx, dir::f_0m0);
          double p_0m0 = phi_cell(it_0m0->locidx, phi_p);

          double theta_0m0 = interface_Location(0., dy_C, p_C, p_0m0)/dy_C;
          if (theta_0m0 < d_eps) theta_0m0 = d_eps;
          if (theta_0m0 > 1.0  ) theta_0m0 = 1.0;

          if (p_0m0*p_C <= 0.) {
            double val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0 * dy_C);
            rhs_p[qu_locidx] += mu_*dx_C/dy_C * val_interface_0m0/theta_0m0;
          }
        }

        if(!is_quad_ypWall(p4est_, tr_id, quad))
        {
          const quad_info_t * it_0p0 = cell_neighbors_->begin(qu_locidx, dir::f_0p0);
          double p_0p0 = phi_cell(it_0p0->locidx, phi_p);

          double theta_0p0 = interface_Location(0., dy_C, p_C, p_0p0)/dy_C;
          if (theta_0p0 < d_eps) theta_0p0 = d_eps;
          if (theta_0p0 > 1.0  ) theta_0p0 = 1.0;

          if (p_0p0*p_C <= 0.) {
            double val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0 * dy_C);
            rhs_p[qu_locidx] += mu_*dx_C/dy_C * val_interface_0p0/theta_0p0;
          }
        }
      }

      double fxxa, fxxb;

      /* accounting for the contributions from the wall */

      /* m00 */
      if (is_quad_xmWall(p4est_, tr_id, quad)){
        double val_wall = bc_->wallValue(x_C - 0.5*dx_C,y_C);
        fxxa = phi_yy_p[n_mmm];
        fxxb = phi_yy_p[n_mpm];
        double theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_p[n_mmm], phi_p[n_mpm], fxxa, fxxb, dy_min);
        switch(bc_->wallType(x_C - 0.5*dx_C,y_C))
        {
        case DIRICHLET:
          rhs_p[qu_locidx] += mu_*2.0*val_wall/dx_C * dy_C * theta;
          break;
        case NEUMANN:
          rhs_p[qu_locidx] += mu_*val_wall * dy_C * theta;
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }

      /* p00 */
      if (is_quad_xpWall(p4est_, tr_id, quad)){
        double val_wall = bc_->wallValue(x_C + 0.5*dx_C,y_C);
        fxxa = phi_yy_p[n_pmm];
        fxxb = phi_yy_p[n_ppm];
        double theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_p[n_pmm], phi_p[n_ppm], fxxa, fxxb, dy_min);
        switch(bc_->wallType(x_C + 0.5*dx_C,y_C))
        {
        case DIRICHLET:
          rhs_p[qu_locidx] += mu_*2.0*val_wall/dx_C * dy_C * theta;
          break;
        case NEUMANN:
          rhs_p[qu_locidx] += mu_*val_wall * dy_C * theta;
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }

      /* 0m0 */
      if (is_quad_ymWall(p4est_, tr_id, quad)) {
        double val_wall = bc_->wallValue(x_C, y_C - 0.5*dy_C);
        fxxa = phi_xx_p[n_mmm];
        fxxb = phi_xx_p[n_pmm];
        double theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_p[n_mmm], phi_p[n_pmm],fxxa,fxxb,dx_min);
        switch(bc_->wallType(x_C, y_C - 0.5*dy_C))
        {
        case DIRICHLET:
          rhs_p[qu_locidx] += mu_*2.0*val_wall/dy_C * dx_C * theta;
          break;
        case NEUMANN:
          rhs_p[qu_locidx] += mu_*val_wall * dx_C * theta;
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }

      /* 0p0 */
      if (is_quad_ypWall(p4est_, tr_id, quad)) {
        double val_wall = bc_->wallValue(x_C, y_C + 0.5*dy_C);
        fxxa = phi_xx_p[n_mpm];
        fxxb = phi_xx_p[n_ppm];
        double theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_p[n_mpm], phi_p[n_ppm],fxxa,fxxb,dx_min);
        switch(bc_->wallType(x_C, y_C + 0.5*dy_C))
        {
        case DIRICHLET:
          rhs_p[qu_locidx] += mu_*2.0*val_wall/dy_C * dx_C * theta;
          break;
        case NEUMANN:
          rhs_p[qu_locidx] += mu_*val_wall * dx_C * theta;
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: unknown boundary condition.");
        }
      }
    }
  }

  // restore the pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_,    &rhs_p   ); CHKERRXX(ierr);

  if (matrix_has_nullspace)
    ierr = MatNullSpaceRemove(A_null_space, rhs_, NULL); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverCellBase_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void PoissonSolverCellBase::set_phi(Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void PoissonSolverCellBase::set_phi(Vec phi, Vec phi_xx, Vec phi_yy)
#endif
{
  phi_ = phi;

#ifdef P4_TO_P8
  if (phi_xx != NULL && phi_yy != NULL && phi_zz != NULL)
#else
  if (phi_xx != NULL && phi_yy != NULL)
#endif
  {
    phi_xx_ = phi_xx;
    phi_yy_ = phi_yy;
#ifdef P4_TO_P8
    phi_zz_ = phi_zz;
#endif

    is_phi_dd_owned = false;
  } else {

    /*
     * We have two options here:
     * 1) Either compute phi_xx and phi_yy using the function that treats them
     * as two regular functions
     * or,
     * 2) Use the function that uses one block vector and then copy stuff into
     * these two vectors
     *
     * Case 1 requires less communications but case two inccuures additional copies
     * Which one is faster? I don't know!
     *
     * TODO: Going with case 1 for the moment -- to be tested
     */

    // Allocate memory for second derivaties
    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est_, nodes_, &phi_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    node_neighbors_->second_derivatives_central(phi_, phi_xx_, phi_yy_, phi_zz_);
#else
    node_neighbors_->second_derivatives_central(phi_, phi_xx_, phi_yy_);
#endif
    is_phi_dd_owned = true;
  }

  // set the interpolating function parameters
#ifdef P4_TO_P8
  phi_interp.set_input_parameters(phi_, quadratic_non_oscillatory, phi_xx_, phi_yy_, phi_zz_);
#else
  phi_interp.set_input_parameters(phi_, quadratic_non_oscillatory, phi_xx_, phi_yy_);
#endif
}
