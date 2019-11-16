#ifdef P4_TO_P8
#include "my_p8est_poisson_faces.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/cube2.h>
#include <src/cube3.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolation_faces.h>
#include "my_p4est_poisson_faces.h"
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <vector>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_faces_compute_voronoi_cell;
extern PetscLogEvent log_my_p4est_poisson_faces_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_faces_setup_linear_system;
extern PetscLogEvent log_my_p4est_poisson_faces_solve;
extern PetscLogEvent log_my_p4est_poisson_faces_KSPSolve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

using std::vector;

my_p4est_poisson_faces_t::my_p4est_poisson_faces_t(const my_p4est_faces_t *faces, const my_p4est_node_neighbors_t *ngbd_n)
  : faces(faces), p4est(faces->p4est), ngbd_c(faces->ngbd_c), ngbd_n(ngbd_n), interp_phi(ngbd_n),
    phi(NULL), apply_hodge_second_derivative_if_neumann(false), bc(NULL), dxyz_hodge(NULL)
{
  PetscErrorCode ierr;

  p4est_topidx_t vtx_0_max      = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + P4EST_CHILDREN - 1];
  p4est_topidx_t vtx_0_min      = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + 0];
  /* set up the KSP solvers and other parameters*/
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    A[dim]            = NULL;
    A_null_space[dim] = NULL;
    null_space[dim]   = NULL;
    ierr = KSPCreate(p4est->mpicomm, &ksp[dim]); CHKERRXX(ierr);
    ierr = KSPSetTolerances(ksp[dim], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
    pc_is_set_from_options[dim]   = false;
    ksp_is_set_from_options[dim]  = false;
    is_matrix_ready[dim]          = false;
    only_diag_is_modified[dim]    = false;
    matrix_has_nullspace[dim]     = false;
    current_diag[dim]             = 0.0;
    desired_diag[dim]             = 0.0;
    tree_dimensions[dim]          = p4est->connectivity->vertices[3*vtx_0_max + dim] - p4est->connectivity->vertices[3*vtx_0_min + dim] ;
  }

  xyz_min_max(p4est, xyz_min, xyz_max);
  dxyz_min(p4est, dxyz);

  compute_partition_on_the_fly = false;
  mu = 1;
}


my_p4est_poisson_faces_t::~my_p4est_poisson_faces_t()
{
  PetscErrorCode ierr;
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    if(A[dim]!=NULL)              { ierr = MatDestroy(A[dim]);                     CHKERRXX(ierr); }
    if(null_space[dim]!=NULL)     { ierr = VecDestroy(null_space[dim]);            CHKERRXX(ierr); }
    if(A_null_space[dim] != NULL) { ierr = MatNullSpaceDestroy(A_null_space[dim]); CHKERRXX(ierr); }
    if(ksp[dim]!=NULL)            { ierr = KSPDestroy(ksp[dim]);                   CHKERRXX(ierr); }
  }
}

void my_p4est_poisson_faces_t::setup_linear_solver(int dim, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  PetscErrorCode ierr;

  P4EST_ASSERT(ksp[dim] != NULL);
  /* set ksp type */
  KSPType ksp_type_as_such;
  ierr = KSPGetType(ksp[dim], &ksp_type_as_such); CHKERRXX(ierr);
  if(ksp_type != ksp_type_as_such)
  {
    ierr = KSPSetType(ksp[dim], ksp_type); CHKERRXX(ierr);
  }
  PetscBool ksp_initial_guess;
  ierr = KSPGetInitialGuessNonzero(ksp[dim], &ksp_initial_guess); CHKERRXX(ierr);
  if (ksp_initial_guess != ((PetscBool) use_nonzero_initial_guess))
  {
    ierr = KSPSetInitialGuessNonzero(ksp[dim], ((PetscBool) use_nonzero_initial_guess)); CHKERRXX(ierr);
  }
  if(!ksp_is_set_from_options[dim])
  {
    ierr = KSPSetFromOptions(ksp[dim]); CHKERRXX(ierr);
    ksp_is_set_from_options[dim] = true;
  }

  /* set pc type */
  PC pc;
  ierr = KSPGetPC(ksp[dim], &pc); CHKERRXX(ierr);
  P4EST_ASSERT(pc != NULL);
  PCType pc_type_as_such;
  ierr = PCGetType(pc, &pc_type_as_such); CHKERRXX(ierr);
  if(pc_type_as_such != pc_type)
  {
    ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

    /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
     * 1- Strong Threshold
     * 2- Coarsennig Type
     * 3- Truncation Factor
     *
     * Please refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
     * summary of HYPRE papers! Also for a complete list of all the options that can be set from PETSc, one can
     * consult the 'src/ksp/pc/impls/hypre.c' in the PETSc home directory.
     */
    if (!strcmp(pc_type, PCHYPRE)){
      /* 1- Strong threshold:
       * Between 0 to 1
       * "0 "gives better convergence rate (in 3D).
       * Suggested values (By Hypre manual): 0.25 for 2D, 0.5 for 3D
      */
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRXX(ierr);

      /* 2- Coarsening type
       * Available Options:
       * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
       */
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

      /* 3- Truncation factor
       * Greater than zero.
       * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
       */
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);
    }
  }
  if(!pc_is_set_from_options[dim])
  {
    ierr = PCSetFromOptions(pc); CHKERRXX(ierr);
    pc_is_set_from_options[dim] = true;
  }
}


void my_p4est_poisson_faces_t::set_phi(Vec phi_, const bool needs_solver_reset)
{
  this->phi = phi_;
  interp_phi.set_input(phi, linear);
  if(needs_solver_reset)
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      is_matrix_ready[dim]        = false;
      only_diag_is_modified[dim]  = false;
    }
}


void my_p4est_poisson_faces_t::set_rhs(Vec *rhs)
{
  this->rhs = rhs;
}


void my_p4est_poisson_faces_t::set_diagonal(double add)
{
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    desired_diag[dim]           = add;
    if((fabs(current_diag[dim] - desired_diag[dim]) >= EPS*MAX(fabs(current_diag[dim]), fabs(desired_diag[dim]))) && ((fabs(current_diag[dim] >= EPS) || (fabs(desired_diag[dim]) >= EPS))))
    {
      // actual modification of diag, do not change the flag values otherwise
      only_diag_is_modified[dim]  = is_matrix_ready[dim];
      is_matrix_ready[dim]        = false;
    }
  }
}


void my_p4est_poisson_faces_t::set_mu(double mu)
{
  P4EST_ASSERT(mu > 0.0);
  if(fabs(this->mu - mu) > EPS*MAX(this->mu, mu)) // actual modification of mu
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      only_diag_is_modified[dim]  = false;
      is_matrix_ready[dim]        = false;
    }
  this->mu = mu;
}

#ifdef P4_TO_P8
void my_p4est_poisson_faces_t::set_bc(const BoundaryConditions3D *bc, Vec *dxyz_hodge, Vec *face_is_well_defined, const bool needs_solver_reset)
#else
void my_p4est_poisson_faces_t::set_bc(const BoundaryConditions2D *bc, Vec *dxyz_hodge, Vec *face_is_well_defined, const bool needs_solver_reset)
#endif
{
  this->bc = bc;
  this->dxyz_hodge = dxyz_hodge;
  this->face_is_well_defined = face_is_well_defined;
  if (needs_solver_reset)
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      only_diag_is_modified[dim]  = false;
      is_matrix_ready[dim]        = false;
    }
}


void my_p4est_poisson_faces_t::set_compute_partition_on_the_fly(bool val)
{
  this->compute_partition_on_the_fly = val;
}


void my_p4est_poisson_faces_t::solve(Vec *solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  if(bc == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution[dir], &sol_size); CHKERRXX(ierr); break;
    if (sol_size != faces->num_local[dir]){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as the number of faces"
          << "solution.local_size = " << sol_size << " faces->num_local[" << dir << "] = " << faces->num_local[dir] << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_solve, A, rhs, solution, 0); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    /* assemble the linear system if required, and initialize the Krylov solver and its preconditioner based on that*/
    setup_linear_system(dir);

    setup_linear_solver(use_nonzero_initial_guess, matrix_has_nullspace[dir], ksp_type, pc_type);

    /* solve the system */
    ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_KSPSolve, ksp, rhs[dir], solution[dir], 0); CHKERRXX(ierr);

    ierr = KSPSolve(ksp[dir], rhs[dir], solution[dir]); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_KSPSolve, ksp, rhs[dir], solution[dir], 0); CHKERRXX(ierr);

    /* update ghosts */
    ierr = VecGhostUpdateBegin(solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    std::string filename = "/home/regan/workspace/projects/free_surface/2d/vtu/test/voronoi_faces" + std::to_string(dir) + ".vtk";
//    print_partition_VTK(filename.c_str(), dir);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_solve, A, rhs, solution, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_faces_t::compute_voronoi_cell(p4est_locidx_t f_idx, int dir)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Voronoi3D &voro_tmp = compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx];
#else
  Voronoi2D &voro_tmp = compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx];
#endif
  voro_tmp.clear();

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  const PetscScalar *face_is_well_defined_p;
  ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  int dir_m = 2*dir;
  int dir_p = 2*dir+1;

  faces->f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xtmp = p4est->connectivity->vertices[3*vp + 0];
  double ytmp = p4est->connectivity->vertices[3*vp + 1];
  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dx = (xtmp-xyz_min[0]) * dmin;
  double dy = (ytmp-xyz_min[1]) * dmin;
#ifdef P4_TO_P8
  double ztmp = p4est->connectivity->vertices[3*vp + 2];
  double dz = (ztmp-xyz_min[2]) * dmin;
#endif

  double x = faces->x_fr_f(f_idx, dir);
  double y = faces->y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
  double z = faces->z_fr_f(f_idx, dir);
#endif

  /* far in the positive domain */
  if(!face_is_well_defined_p[f_idx])
  {
    ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);
    return;
  }
  ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  p4est_locidx_t qm_idx=-1, qp_idx=-1;
  p4est_topidx_t tm_idx=-1, tp_idx=-1;
  p4est_quadrant_t qm, qp;
  vector<p4est_quadrant_t> ngbd;
  if(faces->q2f(quad_idx, dir_m)==f_idx)
  {
    qp_idx = quad_idx;
    tp_idx = tree_idx;
    qp = *quad; qp.p.piggy3.local_num = qp_idx;
    ngbd.clear();
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir_m);
    if(ngbd.size()>0)
    {
      qm = ngbd[0];
      qm_idx = ngbd[0].p.piggy3.local_num;
      tm_idx = ngbd[0].p.piggy3.which_tree;
    }
  }
  else
  {
    qm_idx = quad_idx;
    tm_idx = tree_idx;
    qm = *quad; qm.p.piggy3.local_num = qm_idx;
    ngbd.clear();
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir_p);
    if(ngbd.size()>0)
    {
      qp = ngbd[0];
      qp_idx = ngbd[0].p.piggy3.local_num;
      tp_idx = ngbd[0].p.piggy3.which_tree;
    }
  }

  /* check for walls */
#ifdef P4_TO_P8
  if(qm_idx==-1 && bc[dir].wallType(x,y,z)==DIRICHLET) { ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr); return; }
  if(qp_idx==-1 && bc[dir].wallType(x,y,z)==DIRICHLET) { ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr); return; }
#else
  if(qm_idx==-1 && bc[dir].wallType(x,y)==DIRICHLET) { ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr); return; }
  if(qp_idx==-1 && bc[dir].wallType(x,y)==DIRICHLET) { ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr); return; }
#endif

  /* find direct neighbors */
  vector<p4est_quadrant_t> ngbd_m_m0, ngbd_p_m0;
  vector<p4est_quadrant_t> ngbd_m_p0, ngbd_p_p0;
#ifdef P4_TO_P8
  vector<p4est_quadrant_t> ngbd_m_0m, ngbd_p_0m;
  vector<p4est_quadrant_t> ngbd_m_0p, ngbd_p_0p;
#endif
  if(qm_idx!=-1)
  {
    switch(dir)
    {
#ifdef P4_TO_P8
    case dir::x:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_m0, qm_idx, tm_idx, 0,-1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_p0, qm_idx, tm_idx, 0, 1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_0m, qm_idx, tm_idx, 0, 0,-1);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_0p, qm_idx, tm_idx, 0, 0, 1);
      break;
    case dir::y:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_m0, qm_idx, tm_idx,-1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_p0, qm_idx, tm_idx, 1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_0m, qm_idx, tm_idx, 0, 0,-1);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_0p, qm_idx, tm_idx, 0, 0, 1);
      break;
    case dir::z:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_m0, qm_idx, tm_idx,-1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_p0, qm_idx, tm_idx, 1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_0m, qm_idx, tm_idx, 0,-1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_0p, qm_idx, tm_idx, 0, 1, 0);
      break;
#else
    case dir::x:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_m0, qm_idx, tm_idx, 0,-1);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_p0, qm_idx, tm_idx, 0, 1);
      break;
    case dir::y:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_m0, qm_idx, tm_idx,-1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_m_p0, qm_idx, tm_idx, 1, 0);
      break;
#endif
    }
  }
  if(qp_idx!=-1)
  {
    switch(dir)
    {
#ifdef P4_TO_P8
    case dir::x:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_m0, qp_idx, tp_idx, 0,-1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_p0, qp_idx, tp_idx, 0, 1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_0m, qp_idx, tp_idx, 0, 0,-1);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_0p, qp_idx, tp_idx, 0, 0, 1);
      break;
    case dir::y:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_m0, qp_idx, tp_idx,-1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_p0, qp_idx, tp_idx, 1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_0m, qp_idx, tp_idx, 0, 0,-1);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_0p, qp_idx, tp_idx, 0, 0, 1);
      break;
    case dir::z:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_m0, qp_idx, tp_idx,-1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_p0, qp_idx, tp_idx, 1, 0, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_0m, qp_idx, tp_idx, 0,-1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_0p, qp_idx, tp_idx, 0, 1, 0);
      break;
#else
    case dir::x:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_m0, qp_idx, tp_idx, 0,-1);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_p0, qp_idx, tp_idx, 0, 1);
      break;
    case dir::y:
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_m0, qp_idx, tp_idx,-1, 0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd_p_p0, qp_idx, tp_idx, 1, 0);
      break;
#endif
    }
  }

  /* now gather the neighbor cells to get the potential voronoi neighbors */
#ifdef P4_TO_P8
  voro_tmp.set_center_point(f_idx,x,y,z);
#else
  voro_tmp.set_center_point(x,y);
#endif

  /* check for uniform case, if so build voronoi partition by hand */
  if(qm.level==qp.level &&
     (ngbd_p_m0.size()==1 && ngbd_m_m0.size()==1 && ngbd_m_m0[0].level==qm.level && ngbd_p_m0[0].level==qp.level &&
      faces->q2f(ngbd_m_m0[0].p.piggy3.local_num,2*dir)!=NO_VELOCITY && faces->q2f(ngbd_p_m0[0].p.piggy3.local_num,2*dir+1)!=NO_VELOCITY) &&
     (ngbd_p_p0.size()==1 && ngbd_m_p0.size()==1 && ngbd_m_p0[0].level==qm.level && ngbd_p_p0[0].level==qp.level &&
      faces->q2f(ngbd_m_p0[0].p.piggy3.local_num,2*dir)!=NO_VELOCITY && faces->q2f(ngbd_p_p0[0].p.piggy3.local_num,2*dir+1)!=NO_VELOCITY) &&
   #ifdef P4_TO_P8
     (ngbd_p_0m.size()==1 && ngbd_m_0m.size()==1 && ngbd_m_0m[0].level==qm.level && ngbd_p_0m[0].level==qp.level &&
      faces->q2f(ngbd_m_0m[0].p.piggy3.local_num,2*dir)!=NO_VELOCITY && faces->q2f(ngbd_p_0m[0].p.piggy3.local_num,2*dir+1)!=NO_VELOCITY) &&
     (ngbd_p_0p.size()==1 && ngbd_m_0p.size()==1 && ngbd_m_0p[0].level==qm.level && ngbd_p_0p[0].level==qp.level &&
      faces->q2f(ngbd_m_0p[0].p.piggy3.local_num,2*dir)!=NO_VELOCITY && faces->q2f(ngbd_p_0p[0].p.piggy3.local_num,2*dir+1)!=NO_VELOCITY) &&
   #endif
     faces->q2f(qm_idx,2*dir)!=NO_VELOCITY && faces->q2f(qp_idx,2*dir+1)!=NO_VELOCITY)
  {
#ifdef P4_TO_P8
    vector<ngbd3Dseed> points(6);
    points[0].n = faces->q2f(qp_idx,2*dir+1);
    points[0].p.x = faces->x_fr_f(points[0].n,dir);
    points[0].p.y = faces->y_fr_f(points[0].n,dir);
    points[0].p.z = faces->z_fr_f(points[0].n,dir);
    points[0].s = dir==dir::x ? dy*dz : (dir==dir::y ? dx*dz : dx*dy);

    points[1].n = faces->q2f(qm_idx,2*dir);
    points[1].p.x = faces->x_fr_f(points[1].n,dir);
    points[1].p.y = faces->y_fr_f(points[1].n,dir);
    points[1].p.z = faces->z_fr_f(points[1].n,dir);
    points[1].s = dir==dir::x ? dy*dz : (dir==dir::y ? dx*dz : dx*dy);

    points[2].n = faces->q2f(ngbd_p_m0[0].p.piggy3.local_num,2*dir);
    points[2].p.x = faces->x_fr_f(points[2].n,dir);
    points[2].p.y = faces->y_fr_f(points[2].n,dir);
    points[2].p.z = faces->z_fr_f(points[2].n,dir);
    points[2].s = dir==dir::x ? dx*dz : (dir==dir::y ? dy*dz : dy*dz);

    points[3].n = faces->q2f(ngbd_p_p0[0].p.piggy3.local_num,2*dir);
    points[3].p.x = faces->x_fr_f(points[3].n,dir);
    points[3].p.y = faces->y_fr_f(points[3].n,dir);
    points[3].p.z = faces->z_fr_f(points[3].n,dir);
    points[3].s = dir==dir::x ? dx*dz : (dir==dir::y ? dy*dz : dy*dz);

    points[4].n = faces->q2f(ngbd_p_0m[0].p.piggy3.local_num,2*dir);
    points[4].p.x = faces->x_fr_f(points[4].n,dir);
    points[4].p.y = faces->y_fr_f(points[4].n,dir);
    points[4].p.z = faces->z_fr_f(points[4].n,dir);
    points[4].s = dir==dir::x ? dx*dy : (dir==dir::y ? dx*dy : dx*dz);

    points[5].n = faces->q2f(ngbd_p_0p[0].p.piggy3.local_num,2*dir);
    points[5].p.x = faces->x_fr_f(points[5].n,dir);
    points[5].p.y = faces->y_fr_f(points[5].n,dir);
    points[5].p.z = faces->z_fr_f(points[5].n,dir);
    points[5].s = dir==dir::x ? dx*dy : (dir==dir::y ? dx*dy : dx*dz);

    voro_tmp.set_cell(points, dx*dy*dz);
#else
    vector<ngbd2Dseed> points(4);
    vector<Point2> partition(4);

    points[0].n = faces->q2f(qm_idx,2*dir);
    points[0].p.x = faces->x_fr_f(points[0].n,dir);
    points[0].p.y = faces->y_fr_f(points[0].n,dir);
    points[0].theta = 0;

    points[2].n = faces->q2f(qp_idx,2*dir+1);
    points[2].p.x = faces->x_fr_f(points[2].n,dir);
    points[2].p.y = faces->y_fr_f(points[2].n,dir);
    points[2].theta = PI;

    switch(dir)
    {
    case dir::x:
      partition[0].x = x-dx/2; partition[0].y = y-dy/2;
      partition[1].x = x+dx/2; partition[1].y = y-dy/2;
      partition[2].x = x+dx/2; partition[2].y = y+dy/2;
      partition[3].x = x-dx/2; partition[3].y = y+dy/2;

      points[1].n = faces->q2f(ngbd_p_m0[0].p.piggy3.local_num,2*dir);
      points[1].p.x = faces->x_fr_f(points[1].n,dir);
      points[1].p.y = faces->y_fr_f(points[1].n,dir);
      points[1].theta = PI/2;

      points[3].n = faces->q2f(ngbd_p_p0[0].p.piggy3.local_num,2*dir);
      points[3].p.x = faces->x_fr_f(points[3].n,dir);
      points[3].p.y = faces->y_fr_f(points[3].n,dir);
      points[3].theta = 3*PI/2;
      break;
    case dir::y:
      partition[0].x = x+dx/2; partition[0].y = y-dy/2;
      partition[1].x = x+dx/2; partition[1].y = y+dy/2;
      partition[2].x = x-dx/2; partition[2].y = y+dy/2;
      partition[3].x = x-dx/2; partition[3].y = y-dy/2;

      points[1].n = faces->q2f(ngbd_p_p0[0].p.piggy3.local_num,2*dir);
      points[1].p.x = faces->x_fr_f(points[1].n,dir);
      points[1].p.y = faces->y_fr_f(points[1].n,dir);
      points[1].theta = PI/2;

      points[3].n = faces->q2f(ngbd_p_m0[0].p.piggy3.local_num,2*dir);
      points[3].p.x = faces->x_fr_f(points[3].n,dir);
      points[3].p.y = faces->y_fr_f(points[3].n,dir);
      points[3].theta = 3*PI/2;
      break;
    }

    voro_tmp.set_neighbors_and_partition(points, partition, dx*dy);
#endif
  }

  /* otherwise, there is a T-junction and the grid is not uniform, need to compute the voronoi cell */
  else
  {
    const bool periodic[] = {is_periodic(p4est, dir::x), is_periodic(p4est, dir::y)
                         #ifdef P4_TO_P8
                             , is_periodic(p4est, dir::z)
                         #endif
                            };
    /* note that the walls are dealt with by voro++ in 3D */
#ifndef P4_TO_P8
    switch(dir)
    {
    case dir::x:
      if(qm_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_m00, x-dx, y, periodic, xyz_min, xyz_max);
      if(qp_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_p00, x+dx, y, periodic, xyz_min, xyz_max);
      if( (qm_idx==-1 || is_quad_ymWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ymWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_0m0, x, y-dy, periodic, xyz_min, xyz_max);
      if( (qm_idx==-1 || is_quad_ypWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ypWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_0p0, x, y+dy, periodic, xyz_min, xyz_max);
      break;
    case dir::y:
      if( (qm_idx==-1 || is_quad_xmWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xmWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_m00, x-dx, y, periodic, xyz_min, xyz_max);
      if( (qm_idx==-1 || is_quad_xpWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xpWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_p00, x+dx, y, periodic, xyz_min, xyz_max);
      if(qm_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_0m0, x, y-dy, periodic, xyz_min, xyz_max);
      if(qp_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_0p0, x, y+dy, periodic, xyz_min, xyz_max);
      break;
    }
#endif

    /* gather neighbor cells */
    ngbd.clear();
    if(qm_idx!=-1)
    {
      ngbd.push_back(qm);

      if(faces->q2f(qm_idx,2*dir)==NO_VELOCITY)
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 2*dir);

      switch(dir)
      {
#ifdef P4_TO_P8
      case dir::x:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1, 1);
        break;
      case dir::y:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1, 1);
        break;
      case dir::z:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 0);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 1, 0);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1,-1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 1,-1);
        break;
#else
      case dir::x:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1);
        break;
      case dir::y:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1);
        break;
#endif
      }

      for(unsigned int i=0; i<ngbd_m_m0.size(); ++i) ngbd.push_back(ngbd_m_m0[i]);
      for(unsigned int i=0; i<ngbd_m_p0.size(); ++i) ngbd.push_back(ngbd_m_p0[i]);
#ifdef P4_TO_P8
      for(unsigned int i=0; i<ngbd_m_0m.size(); ++i) ngbd.push_back(ngbd_m_0m[i]);
      for(unsigned int i=0; i<ngbd_m_0p.size(); ++i) ngbd.push_back(ngbd_m_0p[i]);
#endif
    }
    if(qp_idx!=-1)
    {
      ngbd.push_back(qp);

      if(faces->q2f(qp_idx,2*dir+1)==NO_VELOCITY)
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 2*dir+1);

      switch(dir)
      {
#ifdef P4_TO_P8
      case dir::x:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 1);
        break;
      case dir::y:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 1);
        break;
      case dir::z:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 0);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1, 0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 0);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 1);

        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 1);
        break;
#else
      case dir::x:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1);
        break;
      case dir::y:
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1);
        break;
#endif
      }

      for(unsigned int i=0; i<ngbd_p_m0.size(); ++i) ngbd.push_back(ngbd_p_m0[i]);
      for(unsigned int i=0; i<ngbd_p_p0.size(); ++i) ngbd.push_back(ngbd_p_p0[i]);
#ifdef P4_TO_P8
      for(unsigned int i=0; i<ngbd_p_0m.size(); ++i) ngbd.push_back(ngbd_p_0m[i]);
      for(unsigned int i=0; i<ngbd_p_0p.size(); ++i) ngbd.push_back(ngbd_p_0p[i]);
#endif
    }

    /* add the faces to the voronoi partition */
    for(unsigned int m=0; m<ngbd.size(); ++m)
    {
      p4est_locidx_t q_tmp = ngbd[m].p.piggy3.local_num;
      p4est_locidx_t f_tmp = faces->q2f(q_tmp, dir_m);
      if(f_tmp!=NO_VELOCITY && f_tmp!=f_idx)
      {
#ifdef P4_TO_P8
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir), faces->z_fr_f(f_tmp, dir), periodic, xyz_min, xyz_max);
#else
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir), periodic, xyz_min, xyz_max);
#endif
      }

      f_tmp = faces->q2f(q_tmp, dir_p);
      if(f_tmp!=NO_VELOCITY && f_tmp!=f_idx)
      {
#ifdef P4_TO_P8
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir), faces->z_fr_f(f_tmp, dir), periodic, xyz_min, xyz_max);
#else
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir), periodic, xyz_min, xyz_max);
#endif
      }
    }

#ifdef P4_TO_P8
//    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
//    double xyz_max[0]_ = p4est->connectivity->vertices[3*vp + 0];
//    double xyz_max[1]_ = p4est->connectivity->vertices[3*vp + 1];
//    double xyz_max[2]_ = p4est->connectivity->vertices[3*vp + 2];
    voro_tmp.construct_partition(xyz_min, xyz_max, periodic);
#else
    voro_tmp.construct_partition();
    voro_tmp.compute_volume();
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);
}



#ifndef P4_TO_P8
void my_p4est_poisson_faces_t::clip_voro_cell_by_interface(p4est_locidx_t f_idx, int dir)
{
  Voronoi2D &voro_tmp = compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx];
  vector<ngbd2Dseed> *points;
  vector<Point2> *partition;

  voro_tmp.get_neighbor_seeds(points);
  voro_tmp.get_partition(partition);

  /* first clip the voronoi partition at the boundary of the domain */
  if(dir==dir::x)
  {
    for(unsigned int m=0; m<points->size(); m++)
    {
      if((*points)[m].n == WALL_m00)
      {
        for(int i=-1; i<1; ++i)
        {
          int k   = mod(m+i            ,partition->size());
          int tmp = mod(k+(i==-1? -1:1),partition->size());
          Point2 dir = (*partition)[tmp]-(*partition)[k];
          double lambda = (xyz_min[0] - (*partition)[k].x)/dir.x;
          (*partition)[k].x = xyz_min[0];
          (*partition)[k].y = (*partition)[k].y + lambda*dir.y;
        }
      }

      if((*points)[m].n == WALL_p00)
      {
        for(int i=-1; i<1; ++i)
        {
          int k   = mod(m+i            ,partition->size());
          int tmp = mod(k+(i==-1? -1:1),partition->size());
          Point2 dir = (*partition)[tmp]-(*partition)[k];
          double lambda = (xyz_max[0] - (*partition)[k].x)/dir.x;
          (*partition)[k].x = xyz_max[0];
          (*partition)[k].y = (*partition)[k].y + lambda*dir.y;
        }
      }
    }
  }
  else /* dir::y */
  {
    for(unsigned int m=0; m<points->size(); m++)
    {
      if((*points)[m].n == WALL_0m0)
      {
        for(int i=-1; i<1; ++i)
        {
          int k   = mod(m+i            ,partition->size());
          int tmp = mod(k+(i==-1? -1:1),partition->size());
          Point2 dir = (*partition)[tmp]-(*partition)[k];
          double lambda = (xyz_min[1] - (*partition)[k].y)/dir.y;
          (*partition)[k].x = (*partition)[k].x + lambda*dir.x;
          (*partition)[k].y = xyz_min[1];
        }
      }

      if((*points)[m].n == WALL_0p0)
      {
        for(int i=-1; i<1; ++i)
        {
          int k   = mod(m+i            ,partition->size());
          int tmp = mod(k+(i==-1? -1:1),partition->size());
          Point2 dir = (*partition)[tmp]-(*partition)[k];
          double lambda = (xyz_max[1] - (*partition)[k].y)/dir.y;
          (*partition)[k].x = (*partition)[k].x + lambda*dir.x;
          (*partition)[k].y = xyz_max[1];
        }
      }
    }
  }

  /* clip the partition by the irregular interface */
  vector<double> phi_values(partition->size());
  bool is_pos = false;
  for(unsigned int m=0; m<partition->size(); m++)
  {
    phi_values[m] = interp_phi((*partition)[m].x, (*partition)[m].y);
    is_pos = is_pos || phi_values[m]>0;
  }

  /* clip the voronoi partition with the interface */
  if(is_pos)
  {
    double x = faces->x_fr_f(f_idx, dir);
    double y = faces->y_fr_f(f_idx, dir);

    double phi_c = interp_phi(x,y);
    voro_tmp.set_level_set_values(phi_values, phi_c);
    voro_tmp.clip_interface();
  }

  voro_tmp.compute_volume();
}
#endif


void my_p4est_poisson_faces_t::preallocate_matrix(int dir)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
  proc_offset[dir].resize(p4est->mpisize+1);
  proc_offset[dir][0] = 0;
  for(int r=1; r<=p4est->mpisize; ++r)
    proc_offset[dir][r] = proc_offset[dir][r-1] + faces->global_owned_indeps[dir][r-1];

  PetscInt num_owned_local  = (PetscInt) faces->num_local[dir];
  PetscInt num_owned_global = (PetscInt) proc_offset[dir][p4est->mpisize];

  if(A[dir]!=NULL){
    ierr = MatDestroy(A[dir]); CHKERRXX(ierr);}

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A[dir]); CHKERRXX(ierr);
  ierr = MatSetType(A[dir], MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A[dir], num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A[dir]); CHKERRXX(ierr);

  vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  if(compute_partition_on_the_fly) voro[dir].resize(1);
  else                           { voro[dir].clear(); voro[dir].resize(faces->num_local[dir]); }

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    double xyz[] = {
      faces->x_fr_f(f_idx, dir),
      faces->y_fr_f(f_idx, dir)
  #ifdef P4_TO_P8
      , faces->z_fr_f(f_idx, dir)
  #endif
    };
    double phi_c = interp_phi(xyz);

#ifdef P4_TO_P8
    if(bc[dir].interfaceType(xyz)==NOINTERFACE ||
       (bc[dir].interfaceType(xyz)==DIRICHLET && phi_c<0.5*MIN(dxyz[0],dxyz[1],dxyz[2])) ||
       (bc[dir].interfaceType(xyz)==NEUMANN   && phi_c<2*MAX(dxyz[0],dxyz[1],dxyz[2])) )
#else
    if(bc[dir].interfaceType(xyz)==NOINTERFACE ||
       (bc[dir].interfaceType(xyz)==DIRICHLET && phi_c<0.5*MIN(dxyz[0],dxyz[1])) ||
       (bc[dir].interfaceType(xyz)==NEUMANN   && phi_c<2*MAX(dxyz[0],dxyz[1])) )
#endif
    {
      compute_voronoi_cell(f_idx, dir);

#ifdef P4_TO_P8
      const vector<ngbd3Dseed> *points;
#else
      vector<ngbd2Dseed> *points;
#endif

      if(compute_partition_on_the_fly) voro[dir][0].get_neighbor_seeds(points);
      else                             voro[dir][f_idx].get_neighbor_seeds(points);

      for(unsigned int n=0; n<points->size(); ++n)
      {
        if((*points)[n].n>=0)
        {
          if((*points)[n].n<num_owned_local) d_nnz[f_idx]++;
          else                               o_nnz[f_idx]++;
        }
      }

#ifndef P4_TO_P8
      /* in 2D, clip the partition by the interface and by the walls of the domain */
      clip_voro_cell_by_interface(f_idx, dir);
#endif
    }
  }

  ierr = MatSeqAIJSetPreallocation(A[dir], 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A[dir], 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

//  print_partition_VTK("/home/guittet/code/Output/p4est_navier_stokes/voro_0.vtk");
//  if(dir==0)
//    throw std::invalid_argument("");
}


void my_p4est_poisson_faces_t::setup_linear_system(int dir)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_setup_linear_system, A, rhs[dir], 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(!only_diag_is_modified[dir] || !is_matrix_ready[dir]);
  // check that the current "diagonal" is as desired if the matrix is ready to go...
  P4EST_ASSERT(!is_matrix_ready[dir] || ((fabs(current_diag[dir] - desired_diag[dir]) < EPS*MAX(fabs(current_diag[dir]), fabs(desired_diag[dir]))) || ((fabs(current_diag[dir] < EPS) && (fabs(desired_diag[dir]) < EPS)))));
  if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
  {
    reset_current_diag(dir);
    /* preallocate the matrix and compute the voronoi partition */
    preallocate_matrix(dir);
  }

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  matrix_has_nullspace[dir] = true;

  int dir_m = 2*dir;
  int dir_p = 2*dir+1;

  const PetscScalar *face_is_well_defined_p;
  ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  if(null_space[dir] != NULL)
  {
    ierr = VecDestroy(null_space[dir]); CHKERRXX(ierr);
  }
  ierr = VecDuplicate(rhs[dir], &null_space[dir]); CHKERRXX(ierr);
  double *null_space_p;
  ierr = VecGetArray(null_space[dir], &null_space_p); CHKERRXX(ierr);

  std::vector<double> bc_coeffs;
  std::vector<p4est_locidx_t> bc_index; bc_index.resize(0);
  my_p4est_interpolation_faces_t interp_dxyz_hodge(ngbd_n, faces);
  interp_dxyz_hodge.set_input(dxyz_hodge[dir], dir, 1, face_is_well_defined[dir]);

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    null_space_p[f_idx] = 1;
    p4est_gloidx_t f_idx_g = f_idx + proc_offset[dir][p4est->mpirank];

    faces->f2q(f_idx, dir, quad_idx, tree_idx);

    p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

    double xyz[] = {
      faces->x_fr_f(f_idx, dir)
      , faces->y_fr_f(f_idx, dir)
  #ifdef P4_TO_P8
    , faces->z_fr_f(f_idx, dir)
  #endif
    };


    double phi_c = interp_phi(xyz);
    /* far in the positive domain */
    if(!face_is_well_defined_p[f_idx])
    {
      if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
      {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // only needs to be done if full reset
      }
      rhs_p[f_idx] = 0;
      null_space_p[f_idx] = 0;
      continue;
    }

    p4est_locidx_t qm_idx=-1, qp_idx=-1;
    vector<p4est_quadrant_t> ngbd;
    p4est_quadrant_t qm, qp;
    p4est_topidx_t tm_idx=-1, tp_idx=-1;
    if(faces->q2f(quad_idx, dir_m)==f_idx)
    {
      qp_idx = quad_idx;
      tp_idx = tree_idx;
      qp = *quad; qp.p.piggy3.local_num = qp_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir_m);
      /* note that the potential neighbor has to be the same size or bigger */
      if(ngbd.size()>0)
      {
        qm = ngbd[0];
        qm_idx = ngbd[0].p.piggy3.local_num;
        tm_idx = ngbd[0].p.piggy3.which_tree;
      }
    }
    else
    {
      qm_idx = quad_idx;
      tm_idx = tree_idx;
      qm = *quad; qm.p.piggy3.local_num = qm_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir_p);
      /* note that the potential neighbor has to be the same size or bigger */
      if(ngbd.size()>0)
      {
        qp = ngbd[0];
        qp_idx = ngbd[0].p.piggy3.local_num;
        tp_idx = ngbd[0].p.piggy3.which_tree;
      }
    }

    /* check for walls */
    if(qm_idx==-1 && bc[dir].wallType(xyz)==DIRICHLET)
    {
      matrix_has_nullspace[dir] = false;
      if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
      {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // only needs to be done if full reset
      }
      rhs_p[f_idx] = bc[dir].wallValue(xyz) + interp_dxyz_hodge(xyz);
      continue;
    }

    if(qp_idx==-1 && bc[dir].wallType(xyz)==DIRICHLET)
    {
      matrix_has_nullspace[dir] = false;
      if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
      {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // only needs to be done if full reset
      }
      rhs_p[f_idx] = bc[dir].wallValue(xyz) + interp_dxyz_hodge(xyz);
      continue;
    }

    bool wall[P4EST_FACES];
    for(int d=0; d<P4EST_DIM; ++d)
    {
      int f_m = 2*d;
      int f_p = 2*d+1;
      if(d==dir)
      {
        wall[f_m] = qm_idx==-1;
        wall[f_p] = qp_idx==-1;
      }
      else
      {
        wall[f_m] = (qm_idx==-1 || is_quad_Wall(p4est, tm_idx, &qm, f_m)) && (qp_idx==-1 || is_quad_Wall(p4est, tp_idx, &qp, f_m));
        wall[f_p] = (qm_idx==-1 || is_quad_Wall(p4est, tm_idx, &qm, f_p)) && (qp_idx==-1 || is_quad_Wall(p4est, tp_idx, &qp, f_p));
      }
    }

    if(compute_partition_on_the_fly)
    {
      compute_voronoi_cell(f_idx, dir);
#ifndef P4_TO_P8
      clip_voro_cell_by_interface(f_idx, dir);
#endif
    }

#ifdef P4_TO_P8
    Voronoi3D &voro_tmp = compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx];
    const vector<ngbd3Dseed> *points;
#else
    Voronoi2D &voro_tmp = compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx];
    vector<ngbd2Dseed> *points;
    vector<Point2> *partition;
    voro_tmp.get_partition(partition);
#endif
    voro_tmp.get_neighbor_seeds(points);

    /*
     * close to interface and dirichlet => finite differences
     */
#ifdef P4_TO_P8
    if(bc[dir].interfaceType(xyz)==DIRICHLET && phi_c>-2*MAX(dxyz[0],dxyz[1],dxyz[2]))
#else
    if(bc[dir].interfaceType(xyz)==DIRICHLET && phi_c>-2*MAX(dxyz[0],dxyz[1]))
#endif
    {
      if(fabs(phi_c) < EPS)
      {
        if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
        {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // only needs to be done if full reset
        }
        rhs_p[f_idx] = bc[dir].interfaceValue(xyz);
        interp_dxyz_hodge.add_point(bc_index.size(), xyz);
        bc_index.push_back(f_idx);
        bc_coeffs.push_back(1);
        matrix_has_nullspace[dir] = false;
        continue;
      }

      if(phi_c>0)
      {
        if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
        {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // only needs to be done if full reset
        }
        rhs_p[f_idx] = 0;
        null_space_p[f_idx] = 0;
        continue;
      }

      double phi[P4EST_FACES];
#ifdef P4_TO_P8
      phi[dir::f_m00] = interp_phi(xyz[0]-dxyz[0], xyz[1], xyz[2]);
      phi[dir::f_p00] = interp_phi(xyz[0]+dxyz[0], xyz[1], xyz[2]);
      phi[dir::f_0m0] = interp_phi(xyz[0], xyz[1]-dxyz[1], xyz[2]);
      phi[dir::f_0p0] = interp_phi(xyz[0], xyz[1]+dxyz[1], xyz[2]);
      phi[dir::f_00m] = interp_phi(xyz[0], xyz[1], xyz[2]-dxyz[2]);
      phi[dir::f_00p] = interp_phi(xyz[0], xyz[1], xyz[2]+dxyz[2]);
#else
      phi[dir::f_m00] = interp_phi(xyz[0]-dxyz[0], xyz[1]);
      phi[dir::f_p00] = interp_phi(xyz[0]+dxyz[0], xyz[1]);
      phi[dir::f_0m0] = interp_phi(xyz[0], xyz[1]-dxyz[1]);
      phi[dir::f_0p0] = interp_phi(xyz[0], xyz[1]+dxyz[1]);
#endif

      bool is_interface = false;
      for(int i=0; i<P4EST_FACES; ++i)
        is_interface = is_interface || phi[i]>0;
      for(unsigned int i=0; i<points->size(); ++i)
        is_interface = is_interface || (*points)[i].n==INTERFACE;

      if(is_interface)
      {
        bool is_interface[P4EST_FACES];
        bool is_crossed_by_interface = false;
        double val_interface[P4EST_FACES];
        for(int f=0; f<P4EST_FACES; ++f)
        {
          is_interface[f] = !wall[f] && phi[f]*phi_c<=0;
          is_crossed_by_interface = is_crossed_by_interface || is_interface[f];
          val_interface[f] = 0;
        }

        if(is_crossed_by_interface)
          matrix_has_nullspace[dir] = false;

        double d_[P4EST_FACES];
        d_[dir::f_m00] = wall[dir::f_m00] ? xyz[0]-xyz_min[0] : dxyz[0];
        d_[dir::f_p00] = wall[dir::f_p00] ? xyz_max[0]-xyz[0] : dxyz[0];
        d_[dir::f_0m0] = wall[dir::f_0m0] ? xyz[1]-xyz_min[1] : dxyz[1];
        d_[dir::f_0p0] = wall[dir::f_0p0] ? xyz_max[1]-xyz[1] : dxyz[1];
#ifdef P4_TO_P8
        d_[dir::f_00m] = wall[dir::f_00m] ? xyz[2]-xyz_min[2] : dxyz[2];
        d_[dir::f_00p] = wall[dir::f_00p] ? xyz_max[2]-xyz[2] : dxyz[2];
#endif

        double theta[P4EST_FACES];
        for(int f=0; f<P4EST_FACES; ++f)
        {
          if(is_interface[f]) {
            theta[f] = interface_Location(0, d_[f], phi_c, phi[f]);
            theta[f] = MAX(EPS, MIN(d_[f], theta[f]));
            d_[f] = theta[f];
            switch(f)
            {
#ifdef P4_TO_P8
            case dir::f_m00: val_interface[f] = bc[dir].interfaceValue(xyz[0] - theta[f], xyz[1], xyz[2]); break;
            case dir::f_p00: val_interface[f] = bc[dir].interfaceValue(xyz[0] + theta[f], xyz[1], xyz[2]); break;
            case dir::f_0m0: val_interface[f] = bc[dir].interfaceValue(xyz[0], xyz[1] - theta[f], xyz[2]); break;
            case dir::f_0p0: val_interface[f] = bc[dir].interfaceValue(xyz[0], xyz[1] + theta[f], xyz[2]); break;
            case dir::f_00m: val_interface[f] = bc[dir].interfaceValue(xyz[0], xyz[1], xyz[2] - theta[f]); break;
            case dir::f_00p: val_interface[f] = bc[dir].interfaceValue(xyz[0], xyz[1], xyz[2] + theta[f]); break;
#else
            case dir::f_m00: val_interface[f] = bc[dir].interfaceValue(xyz[0] - theta[f], xyz[1]); break;
            case dir::f_p00: val_interface[f] = bc[dir].interfaceValue(xyz[0] + theta[f], xyz[1]); break;
            case dir::f_0m0: val_interface[f] = bc[dir].interfaceValue(xyz[0], xyz[1] - theta[f]); break;
            case dir::f_0p0: val_interface[f] = bc[dir].interfaceValue(xyz[0], xyz[1] + theta[f]); break;
#endif
            }

          }
        }

        // if the face lies on a wall and the solver reaches this line, it must be a non-DIRICHLET wall boundary condition...
        // --> assumed to be NEUMANN...
        if(wall[dir_m]) d_[dir_m] = d_[dir_p];
        if(wall[dir_p]) d_[dir_p] = d_[dir_m];

        double desired_coeff[P4EST_FACES], current_coeff[P4EST_FACES];
        double desired_scaling = desired_diag[dir];
        double current_scaling = ((!only_diag_is_modified[dir] && !is_matrix_ready[dir])? 0.0: current_diag[dir]);
        for(int f=0; f<P4EST_FACES; ++f)
        {
          int ff = f%2==0 ? f+1 : f-1;
          desired_coeff[f] = current_coeff[f] = -2*mu/d_[f]/(d_[f]+d_[ff]);
          desired_scaling -= desired_coeff[f];
          current_scaling -= current_coeff[f];
        }

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        for(int f=0; f<P4EST_FACES; ++f)
        {
          desired_coeff[f] /= desired_scaling;
          if(only_diag_is_modified[dir] || is_matrix_ready[dir])
            current_coeff[f] /= current_scaling;
          else
            current_coeff[f] = 0.0;
        }

        rhs_p[f_idx] /= desired_scaling;

        if(desired_diag[dir] > 0) matrix_has_nullspace[dir] = false;

        //---------------------------------------------------------------------
        // insert the coefficients in the matrix
        //---------------------------------------------------------------------
        if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
        {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // needs to be done if full reset only, the (raw) diag term will always be 1.0 for this...
        }

        for(int f=0; f<P4EST_FACES; ++f)
        {
          /* this is the cartesian direction for which the linear system is assembled.
           * the treatment is different, for example x-velocity can be ON the x-walls
           */
          if(f/2==dir)
          {
            if(wall[f])
            {
              /*
               * if the code reaches this point, it must be a non-DIRICHLET wall boundary condition.
               * --> assumed to be NEUMANN! The treatment is using a ghost node across the wall, though:
               * the values at the wall and the closest inner faces neighbor (if defined) are still solved
               * for. The ghost value, however is defined as
               *
               * value_ghost = value_inner_face_neighbor (or value_inner_interface_point)
               *               + (d_[f] + d_[f_op])*(bc.wall_value(xyz) + second_order_derivative_of_hodge)
               * (the second_order_derivative_of_hodge term is calculated only if apply_hodge_second_derivative_if_neumann is true)
               *
               * NB: d_[f] and  d_[f_op] are made equal beforehand in this case (mirror symmetry, see above + comment)
               */
              int f_op = f%2==0 ? f+1 : f-1; /* the opposite direction. if f = f_m00, then f_op = f_p00 */
              double xyz_op[P4EST_DIM];
              if(!is_interface[f_op])
              {
                p4est_locidx_t face_opp_loc_idx = ((f%2 == 0)? faces->q2f(qp_idx, dir_p) : faces->q2f(qm_idx, dir_m));
                if(!is_matrix_ready[dir])
                {
                  p4est_gloidx_t f_tmp_g = face_global_number(face_opp_loc_idx, dir);
                  ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g, (desired_coeff[f] - current_coeff[f]), ADD_VALUES); CHKERRXX(ierr);
                }
                if(apply_hodge_second_derivative_if_neumann)
                {
                  faces->xyz_fr_f(face_opp_loc_idx, dir, xyz_op);
                  interp_dxyz_hodge.add_point(bc_index.size(),xyz_op);
                  bc_index.push_back(f_idx);
                  bc_coeffs.push_back(-desired_coeff[f]*(d_[f]+d_[f_op])*(-1.0/d_[f_op]));
                }
              }
              else
              {
                rhs_p[f_idx] -= desired_coeff[f] * val_interface[f_op];
                xyz_op[0] = (f_op==dir::f_m00 ? xyz[0]-theta[f_op] : (f_op==dir::f_p00 ? xyz[0]+theta[f_op] : xyz[0]));
                xyz_op[1] = (f_op==dir::f_0m0 ? xyz[1]-theta[f_op] : (f_op==dir::f_0p0 ? xyz[1]+theta[f_op] : xyz[1]));
#ifdef P4_TO_P8
                xyz_op[2] = (f_op==dir::f_00m ? xyz[2]-theta[f_op] : (f_op==dir::f_00p ? xyz[2]+theta[f_op] : xyz[2]));
#endif
                interp_dxyz_hodge.add_point(bc_index.size(),xyz_op);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(-desired_coeff[f]*(1.0 + (d_[f]+d_[f_op])*((apply_hodge_second_derivative_if_neumann) ? (-1.0/d_[f_op]): 0.0)));
              }
              rhs_p[f_idx] -= desired_coeff[f] * (d_[f]+d_[f_op]) * (bc[dir].wallValue(xyz) + ((apply_hodge_second_derivative_if_neumann)? (interp_dxyz_hodge(xyz)/d_[f_op]): 0.0));
              // modified by Raphael Egan, the former discretization of Neumann wall boundary conditions was dimensionally inconsistent
              // --> introduced a flag 'apply_hodge_second_derivative_if_neumann' to enforce an approximation of what is required
              // if apply_hodge_second_derivative_if_neumann == false, the correction is disregarded!
            }
            else if(!is_interface[f] && !is_matrix_ready[dir])
            {
              p4est_gloidx_t f_tmp_g = f%2==0 ? face_global_number(faces->q2f(qm_idx, dir_m), dir)
                                              : face_global_number(faces->q2f(qp_idx, dir_p), dir);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g, (desired_coeff[f] - current_coeff[f]), ADD_VALUES); CHKERRXX(ierr);
            }
            else
            {
              rhs_p[f_idx] -= desired_coeff[f]*val_interface[f];
              double xyz_[P4EST_DIM];
              xyz_[0] = (f==dir::f_m00 ? xyz[0]-theta[f] : (f==dir::f_p00 ? xyz[0]+theta[f] : xyz[0]));
              xyz_[1] = (f==dir::f_0m0 ? xyz[1]-theta[f] : (f==dir::f_0p0 ? xyz[1]+theta[f] : xyz[1]));
#ifdef P4_TO_P8
              xyz_[2] = (f==dir::f_00m ? xyz[2]-theta[f] : (f==dir::f_00p ? xyz[2]+theta[f] : xyz[2]));
#endif
              interp_dxyz_hodge.add_point(bc_index.size(),xyz_);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(-desired_coeff[f]);
            }
          }
          else /* if the direction f is not the direction in which the linear system is being assembled */
          {
            if(wall[f])
            {
              double w_xyz[P4EST_DIM];
              for (short dim = 0; dim < P4EST_DIM; ++dim)
              {
                if(dim != f/2)
                  w_xyz[dim] = xyz[dim];
                else
                  w_xyz[dim] = ((f%2==0)? xyz_min[dim] : xyz_max[dim]);
              }

              BoundaryConditionType bc_w_type = bc[dir].wallType(w_xyz);
              double bc_w_value = bc[dir].wallValue(w_xyz) + ((bc_w_type == DIRICHLET)? interp_dxyz_hodge(w_xyz) : ((apply_hodge_second_derivative_if_neumann)? ((interp_dxyz_hodge(w_xyz) - interp_dxyz_hodge(xyz))/d_[f]):0.0));

              switch(bc_w_type)
              {
              case DIRICHLET:
                 rhs_p[f_idx] -= desired_coeff[f]*bc_w_value;
                 break;
              case NEUMANN:
                if(!is_matrix_ready[dir])
                {
                  ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, (desired_coeff[f]-current_coeff[f]), ADD_VALUES); CHKERRXX(ierr); // should be correct like that (Raphael)
                }
                rhs_p[f_idx] -= desired_coeff[f] * d_[f] * bc_w_value;
                break;
              default:
                throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: invalid boundary condition.");
              }
            }

            else if(!is_interface[f] && !is_matrix_ready[dir])
            {
              ngbd.resize(0);
              ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, f);
#ifdef CASL_THROWS
              if(ngbd.size()!=1 || ngbd[0].level!=quad->level)
                throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: the grid is not uniform close to the interface.");
#endif
              p4est_gloidx_t f_tmp_g;
              if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
              else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g, (desired_coeff[f] - current_coeff[f]), ADD_VALUES); CHKERRXX(ierr);
            }
            else
            {
              rhs_p[f_idx] -= desired_coeff[f]*val_interface[f];
              double xyz_[P4EST_DIM];
              xyz_[0] = (f==dir::f_m00 ? xyz[0]-theta[f] : (f==dir::f_p00 ? xyz[0]+theta[f] : xyz[0]));
              xyz_[1] = (f==dir::f_0m0 ? xyz[1]-theta[f] : (f==dir::f_0p0 ? xyz[1]+theta[f] : xyz[1]));
#ifdef P4_TO_P8
              xyz_[2] = (f==dir::f_00m ? xyz[2]-theta[f] : (f==dir::f_00p ? xyz[2]+theta[f] : xyz[2]));
#endif
              interp_dxyz_hodge.add_point(bc_index.size(),xyz_);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(-desired_coeff[f]);
            }
          }
        } /* end of going through P4EST_FACES to assemble the system with finite differences */
        continue;
      }
    }

#ifdef P4_TO_P8
    /*
     * If close to the interface and Neumann bc, do finite volume by hand
     * since cutting the voronoi cells in 3D with voro++ to have a nice level set is a nightmare ...
     * In 2D, cutting the partition is easy ... so Neumann interface is handled in the bulk case
     */
    if(bc[dir].interfaceType(xyz)==NEUMANN && phi_c>-2*MAX(dxyz[0], dxyz[1], dxyz[2]))
    {
      Cube3 c3;
      OctValue op;
      OctValue iv;

      c3.x0 = (dir==dir::x && qm_idx==-1) ? xyz[0] : xyz[0]-dxyz[0]/2;
      c3.x1 = (dir==dir::x && qp_idx==-1) ? xyz[0] : xyz[0]+dxyz[0]/2;
      c3.y0 = (dir==dir::y && qm_idx==-1) ? xyz[1] : xyz[1]-dxyz[1]/2;
      c3.y1 = (dir==dir::y && qp_idx==-1) ? xyz[1] : xyz[1]+dxyz[1]/2;
      c3.z0 = (dir==dir::z && qm_idx==-1) ? xyz[2] : xyz[2]-dxyz[2]/2;
      c3.z1 = (dir==dir::z && qp_idx==-1) ? xyz[2] : xyz[2]+dxyz[2]/2;

      double hodge_correction = 0.0;
      if(apply_hodge_second_derivative_if_neumann)
      {
        double n_comp[P4EST_DIM];
        n_comp[0] = (interp_phi(c3.x1,  xyz[1], xyz[2]) - interp_phi(c3.x0,   xyz[1], xyz[2]))/(c3.x1 - c3.x0);
        n_comp[1] = (interp_phi(xyz[0], c3.y1,  xyz[2]) - interp_phi(xyz[0],  c3.y0,  xyz[2]))/(c3.y1 - c3.y0);
        n_comp[2] = (interp_phi(xyz[0], xyz[1], c3.z1)  - interp_phi(xyz[0],  xyz[1], c3.z0)) /(c3.z1 - c3.z0);
        double n_norm = 0.0;
        for (short dd = 0; dd < P4EST_DIM; ++dd)
          n_norm += SQR(n_comp[dd]);
        n_norm = sqrt(n_norm);
        P4EST_ASSERT(n_norm > EPS);
        n_comp[0] /= n_norm;
        hodge_correction += n_comp[0]*(interp_dxyz_hodge(c3.x1,  xyz[1], xyz[2])  - interp_dxyz_hodge(c3.x0,   xyz[1], xyz[2]))/(c3.x1 - c3.x0);
        n_comp[1] /= n_norm;
        hodge_correction += n_comp[1]*(interp_dxyz_hodge(xyz[0], c3.y1,  xyz[2])  - interp_dxyz_hodge(xyz[0],  c3.y0,  xyz[2]))/(c3.y1 - c3.y0);
        n_comp[2] /= n_norm;
        hodge_correction += n_comp[2]*(interp_dxyz_hodge(xyz[0], xyz[1], c3.z1)   - interp_dxyz_hodge(xyz[0],  xyz[1], c3.z0)) /(c3.z1 - c3.z0);
      }

      iv.val000 = bc[dir].interfaceValue(c3.x0, c3.y0, c3.z0) + hodge_correction;
      iv.val001 = bc[dir].interfaceValue(c3.x0, c3.y0, c3.z1) + hodge_correction;
      iv.val010 = bc[dir].interfaceValue(c3.x0, c3.y1, c3.z0) + hodge_correction;
      iv.val011 = bc[dir].interfaceValue(c3.x0, c3.y1, c3.z1) + hodge_correction;
      iv.val100 = bc[dir].interfaceValue(c3.x1, c3.y0, c3.z0) + hodge_correction;
      iv.val101 = bc[dir].interfaceValue(c3.x1, c3.y0, c3.z1) + hodge_correction;
      iv.val110 = bc[dir].interfaceValue(c3.x1, c3.y1, c3.z0) + hodge_correction;
      iv.val111 = bc[dir].interfaceValue(c3.x1, c3.y1, c3.z1) + hodge_correction;

      op.val000 = interp_phi(c3.x0, c3.y0, c3.z0);
      op.val001 = interp_phi(c3.x0, c3.y0, c3.z1);
      op.val010 = interp_phi(c3.x0, c3.y1, c3.z0);
      op.val011 = interp_phi(c3.x0, c3.y1, c3.z1);
      op.val100 = interp_phi(c3.x1, c3.y0, c3.z0);
      op.val101 = interp_phi(c3.x1, c3.y0, c3.z1);
      op.val110 = interp_phi(c3.x1, c3.y1, c3.z0);
      op.val111 = interp_phi(c3.x1, c3.y1, c3.z1);

      double volume = c3.volume_In_Negative_Domain(op);

      bool is_pos = (op.val000>0 || op.val001>0 || op.val010>0 || op.val011>0 ||
                     op.val100>0 || op.val101>0 || op.val110>0 || op.val111>0 );
      bool is_neg = (op.val000<=0 || op.val001<=0 || op.val010<=0 || op.val011<=0 ||
                     op.val100<=0 || op.val101<=0 || op.val110<=0 || op.val111<=0 );

      /* entirely in the positive domain */
      if(!is_neg) // should be !face_is_well_defined_p[f_idx], though...
      {
        if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
        {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
        }
        rhs_p[f_idx] = 0;
        null_space_p[f_idx] = 0;
        continue;
      }

      if(is_pos)
      {
        if(!is_matrix_ready[dir])
        {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, volume*(desired_diag[dir] - current_diag[dir]), ADD_VALUES); CHKERRXX(ierr);
        }
        if(desired_diag[dir] > 0) matrix_has_nullspace[dir] = false;
        rhs_p[f_idx] *= volume;
        rhs_p[f_idx] += mu * c3.integrate_Over_Interface(iv, op);

        Cube2 c2;
        QuadValue qp;

        /* m00 */
        c2.x0 = c3.y0; c2.y0 = c3.z0;
        c2.x1 = c3.y1; c2.y1 = c3.z1;
        qp.val00 = op.val000;
        qp.val01 = op.val001;
        qp.val10 = op.val010;
        qp.val11 = op.val011;

        double s_m00 = c2.area_In_Negative_Domain(qp);

        if(s_m00 > EPS*dxyz[1]*dxyz[2]) // --> non-zero interaction area between control volumes
        {
          if(wall[dir::f_m00])
          {
            double w_xyz[P4EST_DIM] = {xyz_min[0], xyz[1], xyz[2]};
            if(bc[dir].wallType(w_xyz) == DIRICHLET)
            {
              rhs_p[f_idx] += mu * s_m00 * bc[dir].wallValue(w_xyz) / (xyz[0] - w_xyz[0]); // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
              interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(mu * s_m00/ (xyz[0] - w_xyz[0]));
              if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
              {
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_m00/(xyz[0] - w_xyz[0]), ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
              }
            }
            else
            {
              rhs_p[f_idx] += mu * s_m00 * (bc[dir].wallValue(xyz_min[0], xyz[1], xyz[2]) + ((apply_hodge_second_derivative_if_neumann)? (-interp_dxyz_hodge(xyz)/(xyz[0] - w_xyz[0])): 0.0));
              if(apply_hodge_second_derivative_if_neumann)
              {
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * s_m00/ (xyz[0] - w_xyz[0]));
              }
            }
          }
          else
          {
            if(!only_diag_is_modified[dir] && !is_matrix_ready[dir]) // needs only to be done if fully reset
            {
              p4est_gloidx_t f_tmp_g;
              if(dir==dir::x)
                f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_m00), dir);
              else
              {
                ngbd.resize(0);
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0, 0);
                P4EST_ASSERT((ngbd.size()==1) && (ngbd[0].level == quad->level)); // must have a uniform tesselation with maximum refinement
                if(quad_idx==qm_idx)
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
                }
                else
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
                }
              }
              ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_m00/dxyz[0], ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*s_m00/dxyz[0], ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }

        /* p00 */
        qp.val00 = op.val100;
        qp.val01 = op.val101;
        qp.val10 = op.val110;
        qp.val11 = op.val111;

        double s_p00 = c2.area_In_Negative_Domain(qp);

        if(s_p00 > EPS*dxyz[1]*dxyz[2]) // --> non-zero interaction area between control volumes
        {
          if(wall[dir::f_p00])
          {
            double w_xyz[P4EST_DIM] = {xyz_max[0], xyz[1], xyz[2]};
            if(bc[dir].wallType(w_xyz) == DIRICHLET)
            {
              rhs_p[f_idx] += mu * s_p00 * bc[dir].wallValue(w_xyz) / (w_xyz[0] - xyz[0]); // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
              interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(mu * s_p00/ (w_xyz[0] - xyz[0]));
              if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
              {
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_p00/(w_xyz[0] - xyz[0]), ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
              }
            }
            else
            {
              rhs_p[f_idx] += mu * s_p00 * (bc[dir].wallValue(xyz_max[0], xyz[1], xyz[2]) + ((apply_hodge_second_derivative_if_neumann)? (-interp_dxyz_hodge(xyz)/(w_xyz[0] - xyz[0])): 0.0));
              if(apply_hodge_second_derivative_if_neumann)
              {
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * s_p00/ (w_xyz[0] - xyz[0]));
              }
            }
          }
          else
          {
            if(!only_diag_is_modified[dir] && !is_matrix_ready[dir]) // needs only to be done if fully reset
            {
              p4est_gloidx_t f_tmp_g;
              if(dir==dir::x)
                f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_p00), dir);
              else
              {
                ngbd.resize(0);
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
                P4EST_ASSERT((ngbd.size()==1) && (ngbd[0].level == quad->level)); // must have a uniform tesselation with maximum refinement
                if(quad_idx==qm_idx)
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
                }
                else
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
                }
              }
              ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_p00/dxyz[0], ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*s_p00/dxyz[0], ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }


        /* 0m0 */
        c2.x0 = c3.x0; c2.y0 = c3.z0;
        c2.x1 = c3.x1; c2.y1 = c3.z1;
        qp.val00 = op.val000;
        qp.val01 = op.val001;
        qp.val10 = op.val100;
        qp.val11 = op.val101;

        double s_0m0 = c2.area_In_Negative_Domain(qp);

        if(s_0m0 > EPS*dxyz[0]*dxyz[2]) // --> non-zero interaction area between control volumes
        {
          if(wall[dir::f_0m0])
          {
            double w_xyz[P4EST_DIM] = {xyz[0], xyz_min[1], xyz[2]};
            if(bc[dir].wallType(w_xyz) == DIRICHLET)
            {
              rhs_p[f_idx] += mu * s_0m0 * bc[dir].wallValue(w_xyz) / (xyz[1] - w_xyz[1]); // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
              interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(mu * s_0m0/ (xyz[1] - w_xyz[1]));
              if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
              {
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_0m0/(xyz[1] - w_xyz[1]), ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
              }
            }
            else
            {
              rhs_p[f_idx] += mu * s_0m0 * (bc[dir].wallValue(xyz[0], xyz_min[1], xyz[2]) + ((apply_hodge_second_derivative_if_neumann)? (-interp_dxyz_hodge(xyz)/(xyz[1] - w_xyz[1])): 0.0));
              if(apply_hodge_second_derivative_if_neumann)
              {
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * s_0m0/ (xyz[1] - w_xyz[1]));
              }
            }
          }
          else
          {
            if(!only_diag_is_modified[dir] && !is_matrix_ready[dir]) // needs only to be done if fully reset
            {
              p4est_gloidx_t f_tmp_g;
              if(dir==dir::y)
                f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_0m0), dir);
              else
              {
                ngbd.resize(0);
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
                P4EST_ASSERT((ngbd.size()==1) && (ngbd[0].level == quad->level)); // must have a uniform tesselation with maximum refinement
                if(quad_idx==qm_idx)
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
                }
                else
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
                }
              }
              ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_0m0/dxyz[1], ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*s_0m0/dxyz[1], ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }


        /* 0p0 */
        qp.val00 = op.val010;
        qp.val01 = op.val011;
        qp.val10 = op.val110;
        qp.val11 = op.val111;

        double s_0p0 = c2.area_In_Negative_Domain(qp);

        if(s_0p0 > EPS*dxyz[0]*dxyz[2]) // --> non-zero interaction area between control volumes
        {
          if(wall[dir::f_0m0])
          {
            double w_xyz[P4EST_DIM] = {xyz[0], xyz_max[1], xyz[2]};
            if(bc[dir].wallType(w_xyz) == DIRICHLET)
            {
              rhs_p[f_idx] += mu * s_0p0 * bc[dir].wallValue(w_xyz) / (w_xyz[1] - xyz[1]); // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
              interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(mu * s_0p0/ (w_xyz[1] - xyz[1]));
              if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
              {
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_0p0/(w_xyz[1] - xyz[1]), ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
              }
            }
            else
            {
              rhs_p[f_idx] += mu * s_0p0 * (bc[dir].wallValue(xyz[0], xyz_max[1], xyz[2])+ ((apply_hodge_second_derivative_if_neumann)? (-interp_dxyz_hodge(xyz)/(w_xyz[1] - xyz[1])): 0.0));
              if(apply_hodge_second_derivative_if_neumann)
              {
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * s_0p0/ (w_xyz[1] - xyz[1]));
              }
            }
          }
          else
          {
            if(!only_diag_is_modified[dir] && !is_matrix_ready[dir]) // needs only to be done if fully reset
            {
              p4est_gloidx_t f_tmp_g;
              if(dir==dir::y)
                f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_0p0), dir);
              else
              {
                ngbd.resize(0);
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
                P4EST_ASSERT((ngbd.size()==1) && (ngbd[0].level == quad->level)); // must have a uniform tesselation with maximum refinement
                if(quad_idx==qm_idx)
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
                }
                else
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
                }
              }
              ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_0p0/dxyz[1], ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*s_0p0/dxyz[1], ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }


        /* 00m */
        c2.x0 = c3.x0; c2.y0 = c3.y0;
        c2.x1 = c3.x1; c2.y1 = c3.y1;
        qp.val00 = op.val000;
        qp.val01 = op.val010;
        qp.val10 = op.val100;
        qp.val11 = op.val110;

        double s_00m = c2.area_In_Negative_Domain(qp);

        if(s_00m > EPS*dxyz[0]*dxyz[1]) // --> non-zero interaction area between control volumes
        {
          if(wall[dir::f_00m])
          {
            double w_xyz[P4EST_DIM] = {xyz[0], xyz[1], xyz_min[2]};
            if(bc[dir].wallType(w_xyz) == DIRICHLET)
            {
              rhs_p[f_idx] += mu * s_00m * bc[dir].wallValue(w_xyz) / (xyz[2] - w_xyz[2]); // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
              interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(mu * s_00m/ (xyz[2] - w_xyz[2]));
              if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
              {
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_00m/(xyz[2] - w_xyz[2]), ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
              }
            }
            else
            {
              rhs_p[f_idx] += mu * s_00m * (bc[dir].wallValue(xyz[0], xyz[1], xyz_min[2]) + ((apply_hodge_second_derivative_if_neumann)? (-interp_dxyz_hodge(xyz)/(xyz[2] - w_xyz[2])): 0.0));
              if(apply_hodge_second_derivative_if_neumann)
              {
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * s_00m/ (xyz[2] - w_xyz[2]));
              }
            }
          }
          else
          {
            if(!only_diag_is_modified[dir] && !is_matrix_ready[dir]) // needs only to be done if fully reset
            {
              p4est_gloidx_t f_tmp_g;
              if(dir==dir::z)
                f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_00m), dir);
              else
              {
                ngbd.resize(0);
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
                P4EST_ASSERT((ngbd.size()==1) && (ngbd[0].level == quad->level)); // must have a uniform tesselation with maximum refinement
                if(quad_idx==qm_idx)
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
                }
                else
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
                }
              }
              ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_00m/dxyz[2], ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*s_00m/dxyz[2], ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }


        /* 00p */
        qp.val00 = op.val001;
        qp.val01 = op.val011;
        qp.val10 = op.val101;
        qp.val11 = op.val111;

        double s_00p = c2.area_In_Negative_Domain(qp);

        if(s_00p > EPS*dxyz[0]*dxyz[1]) // --> non-zero interaction area between control volumes
        {
          if(wall[dir::f_00p])
          {
            double w_xyz[P4EST_DIM] = {xyz[0], xyz[1], xyz_max[2]};
            if(bc[dir].wallType(w_xyz) == DIRICHLET)
            {
              rhs_p[f_idx] += mu * s_00p * bc[dir].wallValue(w_xyz) / (w_xyz[2] - xyz[2]); // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
              interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(mu * s_00p/ (w_xyz[2] - xyz[2]));
              if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
              {
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_00p/(w_xyz[2] - xyz[2]), ADD_VALUES); CHKERRXX(ierr); // needs to be done only if fully reset
              }
            }
            else
            {
              rhs_p[f_idx] += mu * s_00p * (bc[dir].wallValue(xyz[0], xyz[1], xyz_max[2])+ ((apply_hodge_second_derivative_if_neumann)? (-interp_dxyz_hodge(xyz)/(w_xyz[2] - xyz[2])): 0.0));
              if(apply_hodge_second_derivative_if_neumann)
              {
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * s_00p/ (w_xyz[2] - xyz[2]));
              }
            }
          }
          else
          {
            if(!only_diag_is_modified[dir] && !is_matrix_ready[dir]) // needs only to be done if fully reset
            {
              p4est_gloidx_t f_tmp_g;
              if(dir==dir::z)
                f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_00p), dir);
              else
              {
                ngbd.resize(0);
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
                P4EST_ASSERT((ngbd.size()==1) && (ngbd[0].level == quad->level)); // must have a uniform tesselation with maximum refinement
                if(!((ngbd.size()==1) && (ngbd[0].level == quad->level)))
                {
                  std::cerr << "ngbd.size() = " << ngbd.size() << std::endl;
                  std::cerr << "ngbd[0].lveel = " << (int) ngbd[0].level << " while quad.level = " << (int) quad->level << std::endl;
                  std::cerr << "area between control volumes = " << s_00p << std::endl;
                  std::cerr << "location of the face of interest: x = " << xyz[0] << ", y = " << xyz[1] << ", z = " << xyz[2] << std::endl;
                  std::cerr << "the face has direction " << dir << std::endl;
                  for (size_t k = 0; k < ngbd.size(); ++k)
                    std::cerr << "neighbor " << k << " has local num" << ngbd[k].p.piggy3.local_num << " on proc " << p4est->mpirank << std::endl;
                }
                if(quad_idx==qm_idx)
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
                }
                else
                {
                  P4EST_ASSERT(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m) != NO_VELOCITY);
                  f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
                }
              }
              ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s_00p/dxyz[2], ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*s_00p/dxyz[2], ADD_VALUES); CHKERRXX(ierr);
            }
          }
        }

        continue;
      }
    }
#endif

    /*
     * Bulk case, away from the interface
     * Use finite volumes on the voronoi cells
     */

    /* integrally in positive domain */
#ifndef P4_TO_P8
    if(partition->size()==0)
    {
      if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
      {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
      }
      rhs_p[f_idx] = 0;
      null_space_p[f_idx] = 0;
      continue;
    }
#endif

    double volume = voro_tmp.get_volume();
    if(!is_matrix_ready[dir])
    {
      ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, volume*(desired_diag[dir] - current_diag[dir]), ADD_VALUES); CHKERRXX(ierr);
    }
    rhs_p[f_idx] *= volume;
    if(desired_diag[dir]>0) matrix_has_nullspace[dir] = false;

    /* bulk case, finite volume on voronoi cell */
#ifdef P4_TO_P8
    Point3 pc(xyz[0],xyz[1],xyz[2]);
#else
    Point2 pc(xyz[0],xyz[1]);
#endif

    double x_pert = xyz[0];
    double y_pert = xyz[1];
#ifdef P4_TO_P8
    double z_pert = xyz[2];
#endif

    switch(dir)
    {
    case dir::x:
      if(fabs(xyz[0]-xyz_min[0])<EPS*tree_dimensions[0]) x_pert = xyz_min[0]+2*EPS*(xyz_max[0]-xyz_min[0]);
      if(fabs(xyz[0]-xyz_max[0])<EPS*tree_dimensions[0]) x_pert = xyz_max[0]-2*EPS*(xyz_max[0]-xyz_min[0]);
      break;
    case dir::y:
      if(fabs(xyz[1]-xyz_min[1])<EPS*tree_dimensions[1]) y_pert = xyz_min[1]+2*EPS*(xyz_max[1]-xyz_min[1]);
      if(fabs(xyz[1]-xyz_max[1])<EPS*tree_dimensions[1]) y_pert = xyz_max[1]-2*EPS*(xyz_max[1]-xyz_min[1]);
      break;
#ifdef P4_TO_P8
    case dir::z:
      if(fabs(xyz[2]-xyz_min[2])<EPS*tree_dimensions[2]) z_pert = xyz_min[2]+2*EPS*(xyz_max[2]-xyz_min[2]);
      if(fabs(xyz[2]-xyz_max[2])<EPS*tree_dimensions[2]) z_pert = xyz_max[2]-2*EPS*(xyz_max[2]-xyz_min[2]);
      break;
#endif
    }

    for(unsigned int m=0; m<points->size(); ++m)
    {
      PetscInt m_idx_g;

#ifdef P4_TO_P8
      double s = (*points)[m].s;
#else
      int k = mod(m-1, points->size());
      double s = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      double d = ((*points)[m].p - pc).norm_L2();

      switch((*points)[m].n)
      {
      case WALL_m00:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(xyz_min[0],y_pert,z_pert))
#else
        switch(bc[dir].wallType(xyz_min[0],y_pert))
#endif
        {
        case DIRICHLET:
          if(dir==dir::x)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
          {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
          }
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_min[0],y_pert,z_pert) + interp_dxyz_hodge(xyz_min[0],y_pert,z_pert)) / d;
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_min[0],y_pert) + interp_dxyz_hodge(xyz_min[0],y_pert)) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_min[0],y_pert,z_pert) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_min[0],y_pert) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;



      case WALL_p00:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(xyz_max[0],y_pert,z_pert))
#else
        switch(bc[dir].wallType(xyz_max[0],y_pert))
#endif
        {
        case DIRICHLET:
          if(dir==dir::x)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
          {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
          }
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_max[0],y_pert,z_pert) + interp_dxyz_hodge(xyz_max[0],y_pert,z_pert)) / d;
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_max[0],y_pert) + interp_dxyz_hodge(xyz_max[0],y_pert)) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_max[0],y_pert,z_pert) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(xyz_max[0],y_pert) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      case WALL_0m0:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(x_pert,xyz_min[1],z_pert))
#else
        switch(bc[dir].wallType(x_pert,xyz_min[1]))
#endif
        {
        case DIRICHLET:
          if(dir==dir::y)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
          {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
          }
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_min[1],z_pert) + interp_dxyz_hodge(x_pert,xyz_min[1],z_pert)) / d;
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_min[1]) + interp_dxyz_hodge(x_pert,xyz_min[1])) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_min[1],z_pert) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_min[1]) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      case WALL_0p0:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(x_pert,xyz_max[1],z_pert))
#else
        switch(bc[dir].wallType(x_pert,xyz_max[1]))
#endif
        {
        case DIRICHLET:
          if(dir==dir::y)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
          {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
          }
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_max[1],z_pert) + interp_dxyz_hodge(x_pert,xyz_max[1],z_pert)) / d;
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_max[1]) + interp_dxyz_hodge(x_pert,xyz_max[1])) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_max[1],z_pert) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#else
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,xyz_max[1]) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

#ifdef P4_TO_P8
      case WALL_00m:
        switch(bc[dir].wallType(x_pert,y_pert,xyz_min[2]))
        {
        case DIRICHLET:
          if(dir==dir::z)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
          {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
          }
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,y_pert,xyz_min[2]) + interp_dxyz_hodge(x_pert,y_pert,xyz_min[2])) / d;
          break;
        case NEUMANN:
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,y_pert,xyz_min[2]) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      case WALL_00p:
        switch(bc[dir].wallType(x_pert,xyz[1],xyz_max[2]))
        {
        case DIRICHLET:
          if(dir==dir::z)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
          {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
          }
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,y_pert,xyz_max[2]) + interp_dxyz_hodge(x_pert,y_pert,xyz_max[2])) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_p[f_idx] += mu*s*(bc[dir].wallValue(x_pert,y_pert,xyz_max[2]) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;
#endif

      case INTERFACE:
        switch( bc[dir].interfaceType(xyz))
        {
        /* note that DIRICHLET done with finite differences */
        case NEUMANN:
#ifdef P4_TO_P8
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: Neumann boundary conditions should be treated separately in 3D ...");
#else
          rhs_p[f_idx] += mu*s*( bc[dir].interfaceValue(((*points)[m].p.x+xyz[0])/2.,((*points)[m].p.y+xyz[1])/2.) + ((apply_hodge_second_derivative_if_neumann)? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      default:
        if(!only_diag_is_modified[dir] && !is_matrix_ready[dir])
        {
          /* add coefficients in the matrix */
          m_idx_g = face_global_number((*points)[m].n, dir);

          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A[dir], f_idx_g, m_idx_g,-mu*s/d, ADD_VALUES); CHKERRXX(ierr);
        }
      }
    }
  }

  ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  int global_size_bc_index = bc_index.size();
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &global_size_bc_index, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  /* complete the right hand side with correct boundary condition: bc_v + grad(dxyz_hodge) */
  if(global_size_bc_index > 0 /*bc[dir].interfaceType()==DIRICHLET*/)
  {
    std::vector<double> bc_val(bc_index.size());
    interp_dxyz_hodge.interpolate(bc_val.data());
    interp_dxyz_hodge.clear();
    for(unsigned int n=0; n<bc_index.size(); ++n)
      rhs_p[bc_index[n]] += bc_coeffs[n]*bc_val[n];
    bc_val.clear();
    bc_index.clear();
    bc_coeffs.clear();
  }

  ierr = VecRestoreArray(null_space[dir], &null_space_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  if(!is_matrix_ready[dir])
  {
    /* Assemble the matrix */
    ierr = MatAssemblyBegin(A[dir], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A[dir], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  }


//  if(dir==1)
//  {
//    PetscViewer view;
//    char name[1000];
//    sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/mat_p4est_%d.m", p4est->mpisize);
//    ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//    ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//    ierr = MatView(A, view); CHKERRXX(ierr);

//    sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/rhs_p4est_%d.m", p4est->mpisize);
//    ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//    ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//    VecView(rhs[dir], view);

//    PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
//    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
//    PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
//    VecView(rhs[dir], PETSC_VIEWER_STDOUT_WORLD);
//  }

  /* take care of the nullspace if needed */
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace[dir], 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(matrix_has_nullspace[dir])
  {
    ierr = VecGhostUpdateBegin(null_space[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (null_space[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    double norm;
    ierr = VecNormalize(null_space[dir], &norm); CHKERRXX(ierr);

    ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_FALSE, 1, &null_space[dir], &A_null_space[dir]); CHKERRXX(ierr);
    ierr = MatSetNullSpace(A[dir], A_null_space[dir]); CHKERRXX(ierr);
    ierr = MatNullSpaceRemove(A_null_space[dir], rhs[dir], NULL); CHKERRXX(ierr);
    ierr = MatNullSpaceDestroy(A_null_space[dir]); CHKERRXX(ierr);
  }
  ierr = VecDestroy(null_space[dir]); CHKERRXX(ierr);
  null_space[dir] = NULL;

  if(!is_matrix_ready[dir])
  {
    ierr = KSPSetOperators(ksp[dir], A[dir], A[dir], SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  }
  is_matrix_ready[dir]  = true;
  current_diag[dir]     = desired_diag[dir];

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_setup_linear_system, A, rhs[dir], 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_faces_t::print_partition_VTK(const char *file, int dir)
{
  if(compute_partition_on_the_fly)
  {
    throw std::invalid_argument("[ERROR]: my_p4est_poisson_faces_t->print_partition_VTK: please don't use compute_partition_on_the_fly if you want to output the voronoi partition.");
  }
  else
  {
#ifdef P4_TO_P8
    bool periodic[] = {0, 0, 0};
    Voronoi3D::print_VTK_format(voro[dir], file, xyz_min, xyz_max, periodic);
#else
    Voronoi2D::print_VTK_format(voro[dir], file);
#endif
  }
}
