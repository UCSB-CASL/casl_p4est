#ifdef P4_TO_P8
#include "my_p8est_poisson_faces.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube2.h>
#include <src/cube3.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include "my_p4est_poisson_faces.h"
#endif

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>
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
    phi(NULL), rhs_u(NULL), rhs_v(NULL),
    #ifdef P4_TO_P8
    rhs_w(NULL),
    #endif
    A(PETSC_NULL), A_null_space(PETSC_NULL), ksp(PETSC_NULL)
{
  PetscErrorCode ierr;

  /* set up the KSP solver */
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  xmin = p4est->connectivity->vertices[3*vm + 0];
  xmax = p4est->connectivity->vertices[3*vp + 0];
  ymin = p4est->connectivity->vertices[3*vm + 1];
  ymax = p4est->connectivity->vertices[3*vp + 1];
  dx_min = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  dy_min = (ymax-ymin) / pow(2.,(double) data->max_lvl);
#ifdef P4_TO_P8
  zmin = p4est->connectivity->vertices[3*vm + 2];
  zmax = p4est->connectivity->vertices[3*vp + 2];
  dz_min = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif

  vp = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->trees->elem_count-1) + P4EST_CHILDREN-1];
  xmax = p4est->connectivity->vertices[3*vp + 0];
  ymax = p4est->connectivity->vertices[3*vp + 1];
#ifdef P4_TO_P8
  zmax = p4est->connectivity->vertices[3*vp + 2];
#endif

  compute_partition_on_the_fly = false;
  mu = 1;
  diag_add = 0;
}


my_p4est_poisson_faces_t::~my_p4est_poisson_faces_t()
{
  PetscErrorCode ierr;
  if(A!=NULL)                    { ierr = MatDestroy(A);                     CHKERRXX(ierr); }
  if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr); }
  if(ksp!=PETSC_NULL)            { ierr = KSPDestroy(ksp);                   CHKERRXX(ierr); }
}


void my_p4est_poisson_faces_t::reset_linear_solver(bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  PetscErrorCode ierr;
  if(ksp!=PETSC_NULL) { ierr = KSPDestroy(ksp); CHKERRXX(ierr);}
  if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr); }

  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  /* set ksp type */
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  /* set pc type */
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

  /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
   * 1- Strong Threshold
   * 2- Coarsennig Type
   * 3- Truncation Factor
   *
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
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

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

}


void my_p4est_poisson_faces_t::set_phi(Vec phi)
{
  this->phi = phi;
  interp_phi.set_input(phi, linear);
}


void my_p4est_poisson_faces_t::set_rhs(Vec *rhs)
{
  this->rhs = rhs;
}


void my_p4est_poisson_faces_t::set_diagonal(double add)
{
  this->diag_add = add;
}


void my_p4est_poisson_faces_t::set_mu(double mu)
{
  this->mu = mu;
}


#ifdef P4_TO_P8
void my_p4est_poisson_faces_t::set_bc(const BoundaryConditions3D *bc)
#else
void my_p4est_poisson_faces_t::set_bc(const BoundaryConditions2D *bc)
#endif
{
  this->bc = bc;
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
    /* preallocate the matrix and compute the voronoi partition */
    preallocate_matrix(dir);

    /* assemble the linear system */
    setup_linear_system(dir);

    reset_linear_solver(use_nonzero_initial_guess, ksp_type, pc_type);

    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);

    /* set the nullspace */
    if (matrix_has_nullspace[dir])
      ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

    /* solve the system */
    ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_KSPSolve, ksp, rhs[dir], solution[dir], 0); CHKERRXX(ierr);
    ierr = KSPSolve(ksp, rhs[dir], solution[dir]); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_KSPSolve, ksp, rhs[dir], solution[dir], 0); CHKERRXX(ierr);

    /* update ghosts */
    ierr = VecGhostUpdateBegin(solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_solve, A, rhs, solution, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_faces_t::compute_voronoi_cell(p4est_locidx_t f_idx, int dir)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_compute_voronoi_cell, A, 0, 0, 0); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Voronoi3D &voro_tmp = compute_partition_on_the_fly ? voro[0] : voro[f_idx];
#else
  Voronoi2D &voro_tmp = compute_partition_on_the_fly ? voro[0] : voro[f_idx];
#endif
  voro_tmp.clear();

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  int dir_m = 2*dir;
  int dir_p = 2*dir+1;

  faces->f2q(f_idx, dir, quad_idx, tree_idx);

  p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xtmp = p4est->connectivity->vertices[3*vp + 0];
  double ytmp = p4est->connectivity->vertices[3*vp + 1];
  double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dx = (xtmp-xmin) * dmin;
  double dy = (ytmp-ymin) * dmin;
#ifdef P4_TO_P8
  double ztmp = p4est->connectivity->vertices[3*vp + 2];
  double dz = (ztmp-zmin) * dmin;
#endif

  double x = faces->x_fr_f(f_idx, dir);
  double y = faces->y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
  double z = faces->z_fr_f(f_idx, dir);
#endif

#ifdef P4_TO_P8
  double phi_c = interp_phi(x,y,z);
#else
  double phi_c = interp_phi(x,y);
#endif
  /* far in the positive domain */
#ifdef P4_TO_P8
  if(phi_c > 2*MAX(dx_min,dy_min,dz_min))
#else
  if(phi_c > 2*MAX(dx_min,dy_min))
#endif
    return;

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
  if(qm_idx==-1 && bc[dir].wallType(x,y,z)==DIRICHLET) return;
  if(qp_idx==-1 && bc[dir].wallType(x,y,z)==DIRICHLET) return;
#else
  if(qm_idx==-1 && bc[dir].wallType(x,y)==DIRICHLET) return;
  if(qp_idx==-1 && bc[dir].wallType(x,y)==DIRICHLET) return;
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
  voro_tmp.set_Center_Point(f_idx,x,y,z);
#else
  voro_tmp.set_Center_Point(x,y);
#endif

  /* check for uniform case, if so build voronoi partition by hand */
  if(1 && qm.level==qp.level &&
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
    vector<Voronoi3DPoint> points(6);
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

    voro_tmp.set_Points(points, dx*dy*dz);
#else
    vector<Voronoi2DPoint> points(4);
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

    voro_tmp.set_Points_And_Partition(points, partition, dx*dy);
#endif
  }

  /* otherwise, there is a T-junction and the grid is not uniform, need to compute the voronoi cell */
  else
  {
    /* note that the walls are dealt with by voro++ in 3D */
  #ifndef P4_TO_P8
    switch(dir)
    {
    case dir::x:
      if(qm_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_m00, x-dx, y);
      if(qp_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_p00, x+dx, y);
      if( (qm_idx==-1 || is_quad_ymWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ymWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_0m0, x, y-dy);
      if( (qm_idx==-1 || is_quad_ypWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ypWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_0p0, x, y+dy);
      break;
    case dir::y:
      if( (qm_idx==-1 || is_quad_xmWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xmWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_m00, x-dx, y);
      if( (qm_idx==-1 || is_quad_xpWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xpWall(p4est, tp_idx, &qp)) ) voro_tmp.push(WALL_p00, x+dx, y);
      if(qm_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_0m0, x, y-dy);
      if(qp_idx==-1 && bc[dir].wallType(x,y)==NEUMANN) voro_tmp.push(WALL_0p0, x, y+dy);
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
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir), faces->z_fr_f(f_tmp, dir));
#else
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir));
#endif
      }

      f_tmp = faces->q2f(q_tmp, dir_p);
      if(f_tmp!=NO_VELOCITY && f_tmp!=f_idx)
      {
#ifdef P4_TO_P8
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir), faces->z_fr_f(f_tmp, dir));
#else
        voro_tmp.push(f_tmp, faces->x_fr_f(f_tmp, dir), faces->y_fr_f(f_tmp, dir));
#endif
      }
    }

#ifdef P4_TO_P8
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmax_ = p4est->connectivity->vertices[3*vp + 0];
    double ymax_ = p4est->connectivity->vertices[3*vp + 1];
    double zmax_ = p4est->connectivity->vertices[3*vp + 2];
    voro_tmp.construct_Partition(xmin, xmax, ymin, ymax, zmin, zmax, false, false, false);
#else
    voro_tmp.construct_Partition();
    voro_tmp.compute_volume();
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_compute_voronoi_cell, A, 0, 0, 0); CHKERRXX(ierr);
}



#ifndef P4_TO_P8
void my_p4est_poisson_faces_t::clip_voro_cell_by_interface(p4est_locidx_t f_idx, int dir)
{
  Voronoi2D &voro_tmp = compute_partition_on_the_fly ? voro[0] : voro[f_idx];
  vector<Voronoi2DPoint> *points;
  vector<Point2> *partition;

  voro_tmp.get_Points(points);
  voro_tmp.get_Partition(partition);

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
          double lambda = (xmin - (*partition)[k].x)/dir.x;
          (*partition)[k].x = xmin;
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
          double lambda = (xmax - (*partition)[k].x)/dir.x;
          (*partition)[k].x = xmax;
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
          double lambda = (ymin - (*partition)[k].y)/dir.y;
          (*partition)[k].x = (*partition)[k].x + lambda*dir.x;
          (*partition)[k].y = ymin;
        }
      }

      if((*points)[m].n == WALL_0p0)
      {
        for(int i=-1; i<1; ++i)
        {
          int k   = mod(m+i            ,partition->size());
          int tmp = mod(k+(i==-1? -1:1),partition->size());
          Point2 dir = (*partition)[tmp]-(*partition)[k];
          double lambda = (ymax - (*partition)[k].y)/dir.y;
          (*partition)[k].x = (*partition)[k].x + lambda*dir.x;
          (*partition)[k].y = ymax;
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

  // clip the voronoi partition with the interface
  if(is_pos)
  {
    double x = faces->x_fr_f(f_idx, dir);
    double y = faces->y_fr_f(f_idx, dir);

    double phi_c = interp_phi(x,y);
    voro_tmp.set_Level_Set_Values(phi_values, phi_c);
    voro_tmp.clip_Interface();
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

  if(A!=PETSC_NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  if(compute_partition_on_the_fly) voro.resize(1);
  else                           { voro.clear(); voro.resize(faces->num_local[dir]); }

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    double x = faces->x_fr_f(f_idx, dir);
    double y = faces->y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
    double z = faces->z_fr_f(f_idx, dir);
#endif

#ifdef P4_TO_P8
    double phi_c = interp_phi(x,y,z);
#else
    double phi_c = interp_phi(x,y);
#endif

#ifdef P4_TO_P8
    if(bc[dir].interfaceType()==NOINTERFACE ||
       (bc[dir].interfaceType()==DIRICHLET && phi_c<0.5*MIN(dx_min,dy_min,dz_min)) ||
       (bc[dir].interfaceType()==NEUMANN   && phi_c<2*MAX(dx_min,dy_min,dz_min)) )
#else
    if(bc[dir].interfaceType()==NOINTERFACE ||
       (bc[dir].interfaceType()==DIRICHLET && phi_c<0.5*MIN(dx_min,dy_min)) ||
       (bc[dir].interfaceType()==NEUMANN   && phi_c<2*MAX(dx_min,dy_min)) )
#endif
    {
      compute_voronoi_cell(f_idx, dir);

#ifdef P4_TO_P8
      const vector<Voronoi3DPoint> *points;
#else
      vector<Voronoi2DPoint> *points;
#endif

      if(compute_partition_on_the_fly) voro[0].get_Points(points);
      else                             voro[f_idx].get_Points(points);

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

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

//  print_partition_VTK("/home/guittet/code/Output/p4est_navier_stokes/voro_0.vtk");
//  if(dir==1) throw std::invalid_argument("");
}


void my_p4est_poisson_faces_t::setup_linear_system(int dir)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_setup_linear_system, A, rhs[dir], 0, 0); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  matrix_has_nullspace[dir] = true;

  int dir_m = 2*dir;
  int dir_p = 2*dir+1;

  double *rhs_p;
  ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    p4est_gloidx_t f_idx_g = f_idx + proc_offset[dir][p4est->mpirank];

    faces->f2q(f_idx, dir, quad_idx, tree_idx);

    p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

    double x = faces->x_fr_f(f_idx, dir);
    double y = faces->y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
    double z = faces->z_fr_f(f_idx, dir);
#endif

#ifdef P4_TO_P8
    double phi_c = interp_phi(x,y,z);
#else
    double phi_c = interp_phi(x,y);
#endif
    /* far in the positive domain */
  #ifdef P4_TO_P8
    if(phi_c > 2*MAX(dx_min,dy_min,dz_min))
  #else
    if(phi_c > 2*MAX(dx_min,dy_min))
  #endif
    {
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_p[f_idx] = 0;
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
#ifdef P4_TO_P8
    if(qm_idx==-1 && bc[dir].wallType(x,y,z)==DIRICHLET)
#else
    if(qm_idx==-1 && bc[dir].wallType(x,y)==DIRICHLET)
#endif
    {
      matrix_has_nullspace[dir] = false;
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
      rhs_p[f_idx] = bc[dir].wallValue(x,y,z);
#else
      rhs_p[f_idx] = bc[dir].wallValue(x,y);
#endif
      continue;
    }

#ifdef P4_TO_P8
    if(qp_idx==-1 && bc[dir].wallType(x,y,z)==DIRICHLET)
#else
    if(qp_idx==-1 && bc[dir].wallType(x,y)==DIRICHLET)
#endif
    {
      matrix_has_nullspace[dir] = false;
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
      rhs_p[f_idx] = bc[dir].wallValue(x,y,z);
#else
      rhs_p[f_idx] = bc[dir].wallValue(x,y);
#endif
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
    Voronoi3D &voro_tmp = compute_partition_on_the_fly ? voro[0] : voro[f_idx];
    const vector<Voronoi3DPoint> *points;
#else
    Voronoi2D &voro_tmp = compute_partition_on_the_fly ? voro[0] : voro[f_idx];
    vector<Voronoi2DPoint> *points;
    vector<Point2> *partition;
    voro_tmp.get_Partition(partition);
#endif
    voro_tmp.get_Points(points);

    /*
     * close to interface and dirichlet => finite differences
     */
#ifdef P4_TO_P8
    if(bc[dir].interfaceType()==DIRICHLET && phi_c>-2*MAX(dx_min,dy_min,dz_min))
#else
    if(bc[dir].interfaceType()==DIRICHLET && phi_c>-2*MAX(dx_min,dy_min))
#endif
    {
      if(fabs(phi_c) < EPS)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);

#ifdef P4_TO_P8
        rhs_p[f_idx] = bc[dir].interfaceValue(x,y,z);
#else
        rhs_p[f_idx] = bc[dir].interfaceValue(x,y);
#endif
        matrix_has_nullspace[dir] = false;
        continue;
      }

      if(phi_c>0)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
        rhs_p[f_idx] = 0;
        continue;
      }

      double phi[P4EST_FACES];
#ifdef P4_TO_P8
      phi[dir::f_m00] = interp_phi(x-dx_min, y, z);
      phi[dir::f_p00] = interp_phi(x+dx_min, y, z);
      phi[dir::f_0m0] = interp_phi(x, y-dy_min, z);
      phi[dir::f_0p0] = interp_phi(x, y+dy_min, z);
      phi[dir::f_00m] = interp_phi(x, y, z-dz_min);
      phi[dir::f_00p] = interp_phi(x, y, z+dz_min);
#else
      phi[dir::f_m00] = interp_phi(x-dx_min, y);
      phi[dir::f_p00] = interp_phi(x+dx_min, y);
      phi[dir::f_0m0] = interp_phi(x, y-dy_min);
      phi[dir::f_0p0] = interp_phi(x, y+dy_min);
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
        d_[dir::f_m00] = dx_min;
        d_[dir::f_p00] = dx_min;
        d_[dir::f_0m0] = dy_min;
        d_[dir::f_0p0] = dy_min;
#ifdef P4_TO_P8
        d_[dir::f_00m] = dz_min;
        d_[dir::f_00p] = dz_min;
#endif

        for(int f=0; f<P4EST_FACES; ++f)
        {
          if(is_interface[f]) {
            double theta = interface_Location(0, d_[f], phi_c, phi[f]);
            theta = MAX(EPS, MIN(d_[f], theta));
            d_[f] = theta;
            switch(f)
            {
#ifdef P4_TO_P8
            case dir::f_m00: val_interface[f] = bc[dir].interfaceValue(x - theta, y, z); break;
            case dir::f_p00: val_interface[f] = bc[dir].interfaceValue(x + theta, y, z); break;
            case dir::f_0m0: val_interface[f] = bc[dir].interfaceValue(x, y - theta, z); break;
            case dir::f_0p0: val_interface[f] = bc[dir].interfaceValue(x, y + theta, z); break;
            case dir::f_00m: val_interface[f] = bc[dir].interfaceValue(x, y, z - theta); break;
            case dir::f_00p: val_interface[f] = bc[dir].interfaceValue(x, y, z + theta); break;
#else
            case dir::f_m00: val_interface[f] = bc[dir].interfaceValue(x - theta, y); break;
            case dir::f_p00: val_interface[f] = bc[dir].interfaceValue(x + theta, y); break;
            case dir::f_0m0: val_interface[f] = bc[dir].interfaceValue(x, y - theta); break;
            case dir::f_0p0: val_interface[f] = bc[dir].interfaceValue(x, y + theta); break;
#endif
            }

          }
        }

        if(wall[dir_m]) d_[dir_m] = d_[dir_p];
        if(wall[dir_p]) d_[dir_p] = d_[dir_m];

        double coeff[P4EST_FACES];
        double diag = diag_add;
        for(int f=0; f<P4EST_FACES; ++f)
        {
          coeff[f] = -2*mu/d_[f]/(d_[f/2]+d_[f/2+1]);
          diag -= coeff[f];
        }

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        for(int f=0; f<P4EST_FACES; ++f)
          coeff[f] /= diag;

        rhs_p[f_idx] /= diag;

        if(diag_add > 0) matrix_has_nullspace[dir] = false;

        //---------------------------------------------------------------------
        // insert the coefficients in the matrix
        //---------------------------------------------------------------------
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);

        for(int f=0; f<P4EST_FACES; ++f)
        {
          /* this is the cartesian direction for which the linear system is assembled.
           * the treatment is different, for example x-velocity is ON the x-walls
           */
          if(f/2==dir)
          {
            if(wall[f])
            {
              int f_op = f%2==0 ? f+1 : f-1; /* the opposite direction. if f = f_m00, then f_op = f_p00 */
              if(!is_interface[f_op])
              {
                p4est_gloidx_t f_tmp_g = f%2==0 ? face_global_number(faces->q2f(qp_idx, dir_p), dir)
                                                : face_global_number(faces->q2f(qm_idx, dir_m), dir);
                ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff[f], ADD_VALUES); CHKERRXX(ierr);
              }
              else
                rhs_p[f_idx] -= coeff[f] * val_interface[f];
#ifdef P4_TO_P8
              rhs_p[f_idx] -= coeff[f] * (d_[f/2]+d_[f/2+1]) * bc[dir].wallValue(x,y,z);
#else
              rhs_p[f_idx] -= coeff[f] * (d_[f/2]+d_[f/2+1]) * bc[dir].wallValue(x,y);
#endif
            }
            else if(!is_interface[f])
            {
              p4est_gloidx_t f_tmp_g = f%2==0 ? face_global_number(faces->q2f(qm_idx, dir_m), dir)
                                              : face_global_number(faces->q2f(qp_idx, dir_p), dir);
              ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff[f], ADD_VALUES); CHKERRXX(ierr);
            }
            else
              rhs_p[f_idx] -= coeff[f]*val_interface[f];
          }
          else /* if the direction f is not the direction in which the linear system is being assembled */
          {
            if(wall[f])
            {
              BoundaryConditionType bc_w_type;
              double bc_w_value;
              switch(f)
              {
#ifdef P4_TO_P8
              case dir::f_m00:
                bc_w_type = bc[dir].wallType(xmin,y,z);
                bc_w_value = bc[dir].wallValue(xmin,y,z);
                break;
              case dir::f_p00:
                bc_w_type = bc[dir].wallType(xmax,y,z);
                bc_w_value = bc[dir].wallValue(xmax,y,z);
                break;
              case dir::f_0m0:
                bc_w_type = bc[dir].wallType(x,ymin,z);
                bc_w_value = bc[dir].wallValue(x,ymin,z);
                break;
              case dir::f_0p0:
                bc_w_type = bc[dir].wallType(x,ymax,z);
                bc_w_value = bc[dir].wallValue(x,ymax,z);
                break;
              case dir::f_00m:
                bc_w_type = bc[dir].wallType(x,y,zmin);
                bc_w_value = bc[dir].wallValue(x,y,zmin);
                break;
              case dir::f_00p:
                bc_w_type = bc[dir].wallType(x,y,zmax);
                bc_w_value = bc[dir].wallValue(x,y,zmax);
                break;
#else
              case dir::f_m00:
                bc_w_type = bc[dir].wallType(xmin,y);
                bc_w_value = bc[dir].wallValue(xmin,y);
                break;
              case dir::f_p00:
                bc_w_type = bc[dir].wallType(xmax,y);
                bc_w_value = bc[dir].wallValue(xmax,y);
                break;
              case dir::f_0m0:
                bc_w_type = bc[dir].wallType(x,ymin);
                bc_w_value = bc[dir].wallValue(x,ymin);
                break;
              case dir::f_0p0:
                bc_w_type = bc[dir].wallType(x,ymax);
                bc_w_value = bc[dir].wallValue(x,ymax);
                break;
#endif
              }

              switch(bc_w_type)
              {
              case DIRICHLET:
                 rhs_p[f_idx] -= coeff[f]*bc_w_value;
                 break;
              case NEUMANN:
                ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff[f], ADD_VALUES); CHKERRXX(ierr);
                rhs_p[f_idx] -= coeff[f] * d_[f] * bc_w_value;
                break;
              default:
                throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: invalid boundary condition.");
              }
            }

            else if(!is_interface[f])
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
              ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff[f], ADD_VALUES); CHKERRXX(ierr);
            }
            else rhs_p[f_idx] -= coeff[f]*val_interface[f];
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
    if(bc[dir].interfaceType()==NEUMANN && phi_c>-2*MAX(dx_min, dy_min, dz_min))
    {
      Cube3 c3;
      OctValue op;
      OctValue iv;

      c3.x0 = (dir==dir::x && qm_idx==-1) ? x : x-dx_min/2;
      c3.x1 = (dir==dir::x && qp_idx==-1) ? x : x+dx_min/2;
      c3.y0 = (dir==dir::y && qm_idx==-1) ? y : y-dy_min/2;
      c3.y1 = (dir==dir::y && qp_idx==-1) ? y : y+dy_min/2;
      c3.z0 = (dir==dir::z && qm_idx==-1) ? z : z-dz_min/2;
      c3.z1 = (dir==dir::z && qp_idx==-1) ? z : z+dz_min/2;

      iv.val000 = bc[dir].interfaceValue(c3.x0, c3.y0, c3.z0);
      iv.val001 = bc[dir].interfaceValue(c3.x0, c3.y0, c3.z1);
      iv.val010 = bc[dir].interfaceValue(c3.x0, c3.y1, c3.z0);
      iv.val011 = bc[dir].interfaceValue(c3.x0, c3.y1, c3.z1);
      iv.val100 = bc[dir].interfaceValue(c3.x1, c3.y0, c3.z0);
      iv.val101 = bc[dir].interfaceValue(c3.x1, c3.y0, c3.z1);
      iv.val110 = bc[dir].interfaceValue(c3.x1, c3.y1, c3.z0);
      iv.val111 = bc[dir].interfaceValue(c3.x1, c3.y1, c3.z1);

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
      bool is_neg = (op.val000<0 || op.val001<0 || op.val010<0 || op.val011<0 ||
                     op.val100<0 || op.val101<0 || op.val110<0 || op.val111<0 );

      /* entirely in the positive domain */
      if(!is_neg)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
        rhs_p[f_idx] = 0;
        continue;
      }

      if(is_pos)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, volume*diag_add, ADD_VALUES); CHKERRXX(ierr);
        if(diag_add > 0) matrix_has_nullspace[dir] = false;
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

        if(wall[dir::f_m00])
          rhs_p[f_idx] += mu * s_m00 * bc[dir].wallValue(xmin, y, z);
        else
        {
          p4est_gloidx_t f_tmp_g;
          if(dir==dir::x)
            f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_m00), dir);
          else
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0, 0);
            if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
            else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
          }
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_m00/dx_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_m00/dx_min, ADD_VALUES); CHKERRXX(ierr);
        }

        /* p00 */
        qp.val00 = op.val100;
        qp.val01 = op.val101;
        qp.val10 = op.val110;
        qp.val11 = op.val111;

        double s_p00 = c2.area_In_Negative_Domain(qp);

        if(wall[dir::f_p00])
          rhs_p[f_idx] += mu * s_p00 * bc[dir].wallValue(xmax, y, z);
        else
        {
          p4est_gloidx_t f_tmp_g;
          if(dir==dir::x)
            f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_p00), dir);
          else
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
            if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
            else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
          }
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_p00/dx_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_p00/dx_min, ADD_VALUES); CHKERRXX(ierr);
        }


        /* 0m0 */
        c2.x0 = c3.x0; c2.y0 = c3.z0;
        c2.x1 = c3.x1; c2.y1 = c3.z1;
        qp.val00 = op.val000;
        qp.val01 = op.val001;
        qp.val10 = op.val100;
        qp.val11 = op.val101;

        double s_0m0 = c2.area_In_Negative_Domain(qp);

        if(wall[dir::f_0m0])
          rhs_p[f_idx] += mu * s_0m0 * bc[dir].wallValue(x, ymin, z);
        else
        {
          p4est_gloidx_t f_tmp_g;
          if(dir==dir::y)
            f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_0m0), dir);
          else
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
            if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
            else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
          }
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_0m0/dy_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_0m0/dy_min, ADD_VALUES); CHKERRXX(ierr);
        }


        /* 0p0 */
        qp.val00 = op.val010;
        qp.val01 = op.val011;
        qp.val10 = op.val110;
        qp.val11 = op.val111;

        double s_0p0 = c2.area_In_Negative_Domain(qp);

        if(wall[dir::f_0p0])
          rhs_p[f_idx] += mu * s_0p0 * bc[dir].wallValue(x, ymax, z);
        else
        {
          p4est_gloidx_t f_tmp_g;
          if(dir==dir::y)
            f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_0p0), dir);
          else
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
            if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
            else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
          }
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_0p0/dy_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_0p0/dy_min, ADD_VALUES); CHKERRXX(ierr);
        }


        /* 00m */
        c2.x0 = c3.x0; c2.y0 = c3.y0;
        c2.x1 = c3.x1; c2.y1 = c3.y1;
        qp.val00 = op.val000;
        qp.val01 = op.val010;
        qp.val10 = op.val100;
        qp.val11 = op.val110;

        double s_00m = c2.area_In_Negative_Domain(qp);

        if(wall[dir::f_00m])
          rhs_p[f_idx] += mu * s_00m * bc[dir].wallValue(x, y, zmin);
        else
        {
          p4est_gloidx_t f_tmp_g;
          if(dir==dir::z)
            f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_00m), dir);
          else
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
            if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
            else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
          }
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_00m/dz_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_00m/dz_min, ADD_VALUES); CHKERRXX(ierr);
        }


        /* 00p */
        qp.val00 = op.val001;
        qp.val01 = op.val011;
        qp.val10 = op.val101;
        qp.val11 = op.val111;

        double s_00p = c2.area_In_Negative_Domain(qp);

        if(wall[dir::f_00p])
          rhs_p[f_idx] += mu * s_00p * bc[dir].wallValue(x, y, zmax);
        else
        {
          p4est_gloidx_t f_tmp_g;
          if(dir==dir::z)
            f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_00p), dir);
          else
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
            if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_p), dir);
            else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir_m), dir);
          }
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_00p/dz_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_00p/dz_min, ADD_VALUES); CHKERRXX(ierr);
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
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_p[f_idx] = 0;
      continue;
    }
#endif

    double volume = voro_tmp.get_volume();
    ierr = MatSetValue(A, f_idx_g, f_idx_g, volume*diag_add, ADD_VALUES); CHKERRXX(ierr);
    rhs_p[f_idx] *= volume;
    if(diag_add>0) matrix_has_nullspace[dir] = false;

    /* bulk case, finite volume on voronoi cell */
#ifdef P4_TO_P8
    Point3 pc(x,y,z);
#else
    Point2 pc(x,y);
#endif

    double x_pert = x;
    double y_pert = y;
#ifdef P4_TO_P8
    double z_pert = z;
#endif

    switch(dir)
    {
    case dir::x:
      if(fabs(x-xmin)<EPS) x_pert = xmin+2*EPS;
      if(fabs(x-xmax)<EPS) x_pert = xmax-2*EPS;
      break;
    case dir::y:
      if(fabs(y-ymin)<EPS) y_pert = ymin+2*EPS;
      if(fabs(y-ymax)<EPS) y_pert = ymax-2*EPS;
      break;
#ifdef P4_TO_P8
    case dir::z:
      if(fabs(z-zmin)<EPS) z_pert = zmin+2*EPS;
      if(fabs(z-zmax)<EPS) z_pert = zmax-2*EPS;
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
        switch(bc[dir].wallType(xmin,y_pert,z_pert))
#else
        switch(bc[dir].wallType(xmin,y_pert))
#endif
        {
        case DIRICHLET:
          if(dir==dir::x)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmin,y_pert,z) / d;
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmin,y_pert) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmin,y_pert,z_pert);
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmin,y_pert);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;



      case WALL_p00:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(xmax,y_pert,z_pert))
#else
        switch(bc[dir].wallType(xmax,y_pert))
#endif
        {
        case DIRICHLET:
          if(dir==dir::x)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmax,y_pert,z_pert) / d;
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmax,y_pert) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmax,y_pert,z_pert);
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(xmax,y_pert);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      case WALL_0m0:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(x_pert,ymin,z_pert))
#else
        switch(bc[dir].wallType(x_pert,ymin))
#endif
        {
        case DIRICHLET:
          if(dir==dir::y)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymin,z_pert) / d;
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymin) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymin,z_pert);
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymin);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      case WALL_0p0:
#ifdef P4_TO_P8
        switch(bc[dir].wallType(x_pert,ymax,z_pert))
#else
        switch(bc[dir].wallType(x_pert,ymax))
#endif
        {
        case DIRICHLET:
          if(dir==dir::y)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymax,z_pert) / d;
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymax) / d;
#endif
          break;
        case NEUMANN:
#ifdef P4_TO_P8
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymax,z_pert);
#else
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,ymax);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

#ifdef P4_TO_P8
      case WALL_00m:
        switch(bc[dir].wallType(x_pert,y_pert,zmin))
        {
        case DIRICHLET:
          if(dir==dir::z)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,y_pert,zmin) / d;
          break;
        case NEUMANN:
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,y_pert,zmin);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      case WALL_00p:
        switch(bc[dir].wallType(x_pert,y,zmax))
        {
        case DIRICHLET:
          if(dir==dir::z)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,y_pert,zmax) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_p[f_idx] += mu*s*bc[dir].wallValue(x_pert,y_pert,zmax);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;
#endif

      case INTERFACE:
        switch( bc[dir].interfaceType())
        {
        /* note that DIRICHLET done with finite differences */
        case NEUMANN:
#ifdef P4_TO_P8
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: Neumann boundary conditions should be treated separately in 3D ...");
#else
          rhs_p[f_idx] += mu*s*bc[dir].interfaceValue(((*points)[m].p.x+x)/2.,((*points)[m].p.y+y)/2.);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type.");
        }
        break;

      default:
        /* add coefficients in the matrix */
        m_idx_g = face_global_number((*points)[m].n, dir);

        ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, f_idx_g, m_idx_g,-mu*s/d, ADD_VALUES); CHKERRXX(ierr);
      }
    }
  }

  ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  /* take care of the nullspace if needed */
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace[dir], 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(matrix_has_nullspace[dir])
  {
    if(A_null_space == PETSC_NULL)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);
    }
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
    ierr = MatNullSpaceRemove(A_null_space, rhs[dir], NULL); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_setup_linear_system, A, rhs[dir], 0, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_faces_t::print_partition_VTK(const char *file)
{
  if(compute_partition_on_the_fly)
  {
    throw std::invalid_argument("[ERROR]: my_p4est_poisson_faces_t->print_partition_VTK: please don't use compute_partition_on_the_fly if you want to output the voronoi partition.");
  }
  else
  {
#ifdef P4_TO_P8
    Voronoi3D::print_VTK_Format(voro, file, xmin, xmax, ymin, ymax, zmin, zmax, false, false, false);
#else
    Voronoi2D::print_VTK_Format(voro, file);
#endif
  }
}
