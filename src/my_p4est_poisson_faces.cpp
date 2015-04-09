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
extern PetscLogEvent log_PoissonSolverFaces_matrix_preallocation;
extern PetscLogEvent log_PoissonSolverFaces_setup_linear_system;
extern PetscLogEvent log_PoissonSolverNodeBased_KSPSolve;
extern PetscLogEvent log_PoissonSolverNodeBased_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

using std::vector;

PoissonSolverFaces::PoissonSolverFaces(const my_p4est_faces_t *faces, const my_p4est_node_neighbors_t *ngbd_n)
  : faces(faces), p4est(faces->p4est), ngbd_c(faces->ngbd_c), ngbd_n(ngbd_n), interp_phi(*ngbd_n, linear),
    phi(NULL), rhs_u(NULL), rhs_v(NULL),
    #ifdef P4_TO_P8
    rhs_w(NULL),
    #endif
    A(PETSC_NULL), A_null_space(PETSC_NULL), ksp(PETSC_NULL)
{
  /* set up the KSP solver */
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  xmin = p4est->connectivity->vertices[3*vm + 0];
  ymin = p4est->connectivity->vertices[3*vm + 1];
  xmax = p4est->connectivity->vertices[3*vp + 0];
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

  mu = 1;
  diag_add = 0;
}


PoissonSolverFaces::~PoissonSolverFaces()
{
  PetscErrorCode ierr;
  if(A!=NULL)                    { ierr = MatDestroy(A);                     CHKERRXX(ierr); }
  if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr); }
  if(ksp!=PETSC_NULL)            { ierr = KSPDestroy(ksp);                   CHKERRXX(ierr); }
}


void PoissonSolverFaces::set_phi(Vec phi)
{
  this->phi = phi;
  interp_phi.set_input(phi);
}


#ifdef P4_TO_P8
void PoissonSolverFaces::set_rhs(Vec rhs_u, Vec rhs_v, Vec rhs_w)
#else
void PoissonSolverFaces::set_rhs(Vec rhs_u, Vec rhs_v)
#endif
{
  this->rhs_u = rhs_u;
  this->rhs_v = rhs_v;
#ifdef P4_TO_P8
  this->rhs_w = rhs_w;
#endif
}


void PoissonSolverFaces::set_diagonal(double add)
{
  this->diag_add = add;
}


void PoissonSolverFaces::set_mu(double mu)
{
  this->mu = mu;
}


#ifdef P4_TO_P8
void PoissonSolverFaces::set_bc(const BoundaryConditions3D& bc_u, const BoundaryConditions3D& bc_v, const BoundaryConditions3D& bc_w)
#else
void PoissonSolverFaces::set_bc(const BoundaryConditions2D& bc_u, const BoundaryConditions2D& bc_v)
#endif
{
  this->bc_u = &bc_u;
  this->bc_v = &bc_v;
#ifdef P4_TO_P8
  this->bc_w = &bc_w;
#endif
}


#ifdef P4_TO_P8
void PoissonSolverFaces::solve(Vec solution_u, Vec solution_v, Vec solution_w, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
#else
void PoissonSolverFaces::solve(Vec solution_u, Vec solution_v, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
#endif
{
#ifdef CASL_THROWS
  if(bc_u == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");
  if(bc_v == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");
#ifdef P4_TO_P8
  if(bc_w == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");
#endif
  PetscInt sol_size;
  ierr = VecGetLocalSize(solution_u, &sol_size); CHKERRXX(ierr);
  if (sol_size != faces->num_local[0]){
    std::ostringstream oss;
    oss << "[CASL_ERROR]: solution_u vector must be preallocated and locally have the same size as the number of x-faces"
        << "solution_u.local_size = " << sol_size << " faces->num_local[0] = " << faces->num_local[0] << std::endl;
    throw std::invalid_argument(oss.str());
  }
  ierr = VecGetLocalSize(solution_v, &sol_size); CHKERRXX(ierr);
  if (sol_size != faces->num_local[1]){
    std::ostringstream oss;
    oss << "[CASL_ERROR]: solution_1 vector must be preallocated and locally have the same size as the number of x-faces"
        << "solution_v.local_size = " << sol_size << " faces->num_local[1] = " << faces->num_local[1] << std::endl;
    throw std::invalid_argument(oss.str());
  }
#ifdef P4_TO_P8
  ierr = VecGetLocalSize(solution_w, &sol_size); CHKERRXX(ierr);
  if (sol_size != faces->num_local[2]){
    std::ostringstream oss;
    oss << "[CASL_ERROR]: solution_2 vector must be preallocated and locally have the same size as the number of x-faces"
        << "solution_w.local_size = " << sol_size << " faces->num_local[2] = " << faces->num_local[2] << std::endl;
    throw std::invalid_argument(oss.str());
  }
#endif
#endif

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

  solve_u(solution_u);
//  solve_v(solution_v);
#ifdef P4_TO_P8
//  solve_w(solution_w);
#endif
}




void PoissonSolverFaces::solve_u(Vec solution_u)
{
  /* preallocate the matrix and compute the voronoi partition */
  preallocate_matrix(dir::x);

  /* assemble the linear system */
  setup_linear_system_u();

  ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);

  /* set the nullspace */
  if (matrix_has_nullspace_u)
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

  /* solve the system */
  ierr = KSPSolve(ksp, rhs_u, solution_u); CHKERRXX(ierr);

  /* update ghosts */
  ierr = VecGhostUpdateBegin(solution_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (solution_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}



void PoissonSolverFaces::solve_v(Vec solution_v)
{
  /* preallocate the matrix and compute the voronoi partition */
  preallocate_matrix(dir::y);

  /* assemble the linear system */
  setup_linear_system_v();

  ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);

  /* set the nullspace */
  if (matrix_has_nullspace_v)
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

  /* solve the system */
  ierr = KSPSolve(ksp, rhs_v, solution_v); CHKERRXX(ierr);

  /* update ghosts */
  ierr = VecGhostUpdateBegin(solution_v, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (solution_v, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}


#ifdef P4_TO_P8
void PoissonSolverFaces::solve_w(Vec solution_w)
{
  /* preallocate the matrix and compute the voronoi partition */
  preallocate_matrix(dir::z);

  /* assemble the linear system */
  setup_linear_system_w();

  ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);

  /* set the nullspace */
  if (matrix_has_nullspace_w)
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

  /* solve the system */
  ierr = KSPSolve(ksp, rhs_w, solution_w); CHKERRXX(ierr);

  /* update ghosts */
  ierr = VecGhostUpdateBegin(solution_w, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (solution_w, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}
#endif


void PoissonSolverFaces::compute_voronoi_cell_u(p4est_locidx_t u_idx)
{
  voro[u_idx].clear();

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  faces->f2q(u_idx, dir::x, quad_idx, tree_idx);

  p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  double x = faces->x_fr_f(u_idx, dir::x);
  double y = faces->y_fr_f(u_idx, dir::x);
#ifdef P4_TO_P8
  double z = faces->z_fr_f(u_idx, dir::x);
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
  if(faces->q2f(quad_idx, dir::f_m00)==u_idx)
  {
    qp_idx = quad_idx;
    tp_idx = tree_idx;
    qp = *quad; qp.p.piggy3.local_num = qp_idx;
    ngbd.clear();
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0, 0);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0);
#endif
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
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
#endif
    if(ngbd.size()>0)
    {
      qp = ngbd[0];
      qp_idx = ngbd[0].p.piggy3.local_num;
      tp_idx = ngbd[0].p.piggy3.which_tree;
    }
  }

  /* check for walls */
#ifdef P4_TO_P8
  if(qm_idx==-1 && bc_u->wallType(x,y,z)==DIRICHLET) return;
  if(qp_idx==-1 && bc_u->wallType(x,y,z)==DIRICHLET) return;
#else
  if(qm_idx==-1 && bc_u->wallType(x,y)==DIRICHLET) return;
  if(qp_idx==-1 && bc_u->wallType(x,y)==DIRICHLET) return;
#endif

  /* now gather the neighbor cells to get the potential voronoi neighbors */
#ifdef P4_TO_P8
  voro[u_idx].set_Center_Point(u_idx,x,y,z);
#else
  voro[u_idx].set_Center_Point(x,y);
#endif

  /* note that the walls are dealt with by voro++ in 3D */
#ifndef P4_TO_P8
  if(qm_idx==-1 && bc_u->wallType(x,y)==NEUMANN) voro[u_idx].push(WALL_m00, x-dx, y);
  if(qp_idx==-1 && bc_u->wallType(x,y)==NEUMANN) voro[u_idx].push(WALL_p00, x+dx, y);
  if( (qm_idx==-1 || is_quad_ymWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ymWall(p4est, tp_idx, &qp)) ) voro[u_idx].push(WALL_0m0, x, y-dy);
  if( (qm_idx==-1 || is_quad_ypWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ypWall(p4est, tp_idx, &qp)) ) voro[u_idx].push(WALL_0p0, x, y+dy);
#endif

  /* gather neighbor cells */
  ngbd.clear();
  if(qm_idx!=-1)
  {
    ngbd.push_back(qm);
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1, 1);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1);
#endif
  }

  if(qp_idx!=-1)
  {
    ngbd.push_back(qp);
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 1);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1);
#endif
  }

  /* add the faces to the voronoi partition */
  for(unsigned int m=0; m<ngbd.size(); ++m)
  {
    p4est_locidx_t q_tmp = ngbd[m].p.piggy3.local_num;
    p4est_locidx_t f_tmp = faces->q2f(q_tmp, dir::f_m00);
    if(f_tmp!=NO_VELOCITY && f_tmp!=u_idx)
    {
#ifdef P4_TO_P8
      voro[u_idx].push(f_tmp, faces->x_fr_f(f_tmp, dir::x), faces->y_fr_f(f_tmp, dir::x), faces->z_fr_f(f_tmp, dir::x));
#else
      voro[u_idx].push(f_tmp, faces->x_fr_f(f_tmp, dir::x), faces->y_fr_f(f_tmp, dir::x));
#endif
    }

    f_tmp = faces->q2f(q_tmp, dir::f_p00);
    if(f_tmp!=NO_VELOCITY && f_tmp!=u_idx)
    {
#ifdef P4_TO_P8
      voro[u_idx].push(f_tmp, faces->x_fr_f(f_tmp, dir::x), faces->y_fr_f(f_tmp, dir::x), faces->z_fr_f(f_tmp, dir::x));
#else
      voro[u_idx].push(f_tmp, faces->x_fr_f(f_tmp, dir::x), faces->y_fr_f(f_tmp, dir::x));
#endif
    }
  }

#ifdef P4_TO_P8
  voro[u_idx].construct_Partition(xmin, xmax, ymin, ymax, zmin, zmax, false, false, false);
#else
  voro[u_idx].construct_Partition();
#endif

#ifndef P4_TO_P8
  vector<Voronoi2DPoint> *points;
  vector<Point2> *partition;

  voro[u_idx].get_Points(points);
  voro[u_idx].get_Partition(partition);

  /* first clip the voronoi partition at the boundary of the domain */
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
    voro[u_idx].set_Level_Set_Values(phi_values, phi_c);
    voro[u_idx].clip_Interface();
  }
#endif
}




void PoissonSolverFaces::compute_voronoi_cell_v(p4est_locidx_t v_idx)
{
  voro[v_idx].clear();

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  faces->f2q(v_idx, dir::y, quad_idx, tree_idx);

  p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  double x = faces->x_fr_f(v_idx, dir::y);
  double y = faces->y_fr_f(v_idx, dir::y);
#ifdef P4_TO_P8
  double z = faces->z_fr_f(v_idx, dir::y);
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
  if(faces->q2f(quad_idx, dir::f_0m0)==v_idx)
  {
    qp_idx = quad_idx;
    tp_idx = tree_idx;
    qp = *quad; qp.p.piggy3.local_num = qp_idx;
    ngbd.clear();
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1);
#endif
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
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
#endif
    if(ngbd.size()>0)
    {
      qp = ngbd[0];
      qp_idx = ngbd[0].p.piggy3.local_num;
      tp_idx = ngbd[0].p.piggy3.which_tree;
    }
  }

  /* check for walls */
#ifdef P4_TO_P8
  if(qm_idx==-1 && bc_v->wallType(x,y,z)==DIRICHLET) return;
  if(qp_idx==-1 && bc_v->wallType(x,y,z)==DIRICHLET) return;
#else
  if(qm_idx==-1 && bc_v->wallType(x,y)==DIRICHLET) return;
  if(qp_idx==-1 && bc_v->wallType(x,y)==DIRICHLET) return;
#endif

  /* now gather the neighbor cells to get the potential voronoi neighbors */
#ifdef P4_TO_P8
  voro[v_idx].set_Center_Point(v_idx,x,y,z);
#else
  voro[v_idx].set_Center_Point(x,y);
#endif

  /* note that the walls are dealt with by voro++ in 3D */
#ifndef P4_TO_P8
  if(qm_idx==-1 && bc_v->wallType(x,ymin)==NEUMANN) voro[v_idx].push(WALL_0m0, x, ymin-dy);
  if(qp_idx==-1 && bc_v->wallType(x,ymax)==NEUMANN) voro[v_idx].push(WALL_0p0, x, ymax+dy);
  if( (qm_idx==-1 || is_quad_xmWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xmWall(p4est, tp_idx, &qp)) ) voro[v_idx].push(WALL_m00, x-dx, y);
  if( (qm_idx==-1 || is_quad_xpWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xpWall(p4est, tp_idx, &qp)) ) voro[v_idx].push(WALL_p00, x+dx, y);
#endif

  /* gather neighbor cells */
  ngbd.clear();
  if(qm_idx!=-1)
  {
    ngbd.push_back(qm);
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1, 1);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 1, 0);
#endif
  }

  if(qp_idx!=-1)
  {
    ngbd.push_back(qp);
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 1);

    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1, 1);
#else
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0);
#endif
  }

  /* add the faces to the voronoi partition */
  for(unsigned int m=0; m<ngbd.size(); ++m)
  {
    p4est_locidx_t q_tmp = ngbd[m].p.piggy3.local_num;
    p4est_locidx_t v_tmp = faces->q2f(q_tmp, dir::f_0m0);
    if(v_tmp!=NO_VELOCITY && v_tmp!=v_idx)
    {
#ifdef P4_TO_P8
      voro[v_idx].push(v_tmp, faces->x_fr_f(v_tmp, dir::y), faces->y_fr_f(v_tmp, dir::y), faces->z_fr_f(v_tmp, dir::y));
#else
      voro[v_idx].push(v_tmp, faces->x_fr_f(v_tmp, dir::y), faces->y_fr_f(v_tmp, dir::y));
#endif
    }

    v_tmp = faces->q2f(q_tmp, dir::f_0p0);
    if(v_tmp!=NO_VELOCITY && v_tmp!=v_idx)
    {
#ifdef P4_TO_P8
      voro[v_idx].push(v_tmp, faces->x_fr_f(v_tmp, dir::y), faces->y_fr_f(v_tmp, dir::y), faces->z_fr_f(v_tmp, dir::y));
#else
      voro[v_idx].push(v_tmp, faces->x_fr_f(v_tmp, dir::y), faces->y_fr_f(v_tmp, dir::y));
#endif
    }
  }

#ifdef P4_TO_P8
  voro[v_idx].construct_Partition(xmin, xmax, ymin, ymax, zmin, zmax, false, false, false);
#else
  voro[v_idx].construct_Partition();
#endif

#ifndef P4_TO_P8
  vector<Voronoi2DPoint> *points;
  vector<Point2> *partition;

  voro[v_idx].get_Points(points);
  voro[v_idx].get_Partition(partition);

  /* first clip the voronoi partition at the boundary of the domain */
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
    voro[v_idx].set_Level_Set_Values(phi_values, phi_c);
    voro[v_idx].clip_Interface();
  }
#endif
}


#ifdef P4_TO_P8
void PoissonSolverFaces::compute_voronoi_cell_w(p4est_locidx_t w_idx)
{

}
#endif



void PoissonSolverFaces::preallocate_matrix(int dir)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_PoissonSolverFaces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
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

  voro.resize(faces->num_local[dir]);

  for(p4est_locidx_t idx=0; idx<faces->num_local[dir]; ++idx)
  {
    double x = faces->x_fr_f(idx, dir);
    double y = faces->y_fr_f(idx, dir);
#ifdef P4_TO_P8
    double z = faces->z_fr_f(idx, dir);
#endif

#ifdef P4_TO_P8
    double phi_c = interp_phi(x,y,z);
#else
    double phi_c = interp_phi(x,y);
#endif

#ifdef P4_TO_P8
    if(bc_u->interfaceType()==NOINTERFACE || phi_c<2*MAX(dx_min,dy_min,dz_min))
#else
    if(bc_u->interfaceType()==NOINTERFACE || phi_c<2*MAX(dx_min,dy_min))
#endif
    {
      switch(dir)
      {
      case dir::x:
        compute_voronoi_cell_u(idx);
        break;
      case dir::y:
        compute_voronoi_cell_v(idx);
        break;
#ifdef P4_TO_P8
      case dir::z:
        compute_voronoi_cell_w(idx);
        break;
#endif
      }


#ifdef P4_TO_P8
      const vector<Voronoi3DPoint> *points;
#else
      vector<Voronoi2DPoint> *points;
#endif
      voro[idx].get_Points(points);

      for(unsigned int n=0; n<points->size(); ++n)
      {
        if((*points)[n].n>=0)
        {
          if((*points)[n].n<num_owned_local) d_nnz[idx]++;
          else                               o_nnz[idx]++;
        }
      }
    }
  }


  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverFaces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}


void PoissonSolverFaces::setup_linear_system_u()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_PoissonSolverFaces_setup_linear_system, A, rhs_u, 0, 0); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  matrix_has_nullspace_u = true;

  double *rhs_u_p;
  ierr = VecGetArray(rhs_u, &rhs_u_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[0]; ++f_idx)
  {
    p4est_gloidx_t f_idx_g = f_idx + proc_offset[0][p4est->mpirank];

    faces->f2q(f_idx, dir::x, quad_idx, tree_idx);

    p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

    double x = faces->x_fr_f(f_idx, dir::x);
    double y = faces->y_fr_f(f_idx, dir::x);
#ifdef P4_TO_P8
    double z = faces->z_fr_f(f_idx, dir::x);
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
      rhs_u_p[f_idx] = 0;
      continue;
    }

    p4est_locidx_t qm_idx=-1, qp_idx=-1;
    vector<p4est_quadrant_t> ngbd;
    p4est_quadrant_t qm, qp;
    p4est_topidx_t tm_idx=-1, tp_idx=-1;
    if(faces->q2f(quad_idx, dir::f_m00)==f_idx)
    {
      qp_idx = quad_idx;
      tp_idx = tree_idx;
      qp = *quad; qp.p.piggy3.local_num = qp_idx;
      ngbd.clear();
#ifdef P4_TO_P8
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0, 0);
#else
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0);
#endif
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
#ifdef P4_TO_P8
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
#else
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
#endif
      if(ngbd.size()>0)
      {
        qp = ngbd[0];
        qp_idx = ngbd[0].p.piggy3.local_num;
        tp_idx = ngbd[0].p.piggy3.which_tree;
      }
    }

    /* check for walls */
#ifdef P4_TO_P8
    if(qm_idx==-1 && bc_u->wallType(xmin,y,z)==DIRICHLET)
#else
    if(qm_idx==-1 && bc_u->wallType(xmin,y)==DIRICHLET)
#endif
    {
      matrix_has_nullspace_u = false;
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
      rhs_u_p[f_idx] = bc_u->wallValue(xmin,y,z);
#else
      rhs_u_p[f_idx] = bc_u->wallValue(xmin,y);
#endif
      continue;
    }

#ifdef P4_TO_P8
    if(qp_idx==-1 && bc_u->wallType(xmax,y,z)==DIRICHLET)
#else
    if(qp_idx==-1 && bc_u->wallType(xmax,y)==DIRICHLET)
#endif
    {
      matrix_has_nullspace_u = false;
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
      rhs_u_p[f_idx] = bc_u->wallValue(xmax,y,z);
#else
      rhs_u_p[f_idx] = bc_u->wallValue(xmax,y);
#endif
      continue;
    }


    /*
     * close to interface and dirichlet => finite differences
     */
#ifdef P4_TO_P8
    if(bc_u->interfaceType()==DIRICHLET && phi_c>-2*MAX(dx_min,dy_min,dz_min))
#else
    if(bc_u->interfaceType()==DIRICHLET && phi_c>-2*MAX(dx_min,dy_min))
#endif
    {
      if(fabs(phi_c) < EPS)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
        rhs_u_p[f_idx] = bc_u->interfaceValue(x,y,z);
#else
        rhs_u_p[f_idx] = bc_u->interfaceValue(x,y);
#endif
        matrix_has_nullspace_u = false;
        continue;
      }

      if(phi_c>0)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
        rhs_u_p[f_idx] = 0;
        continue;
      }

#ifdef P4_TO_P8
      double phi_m00 = interp_phi(x-dx_min, y, z);
      double phi_p00 = interp_phi(x+dx_min, y, z);
      double phi_0m0 = interp_phi(x, y-dy_min, z);
      double phi_0p0 = interp_phi(x, y+dy_min, z);
      double phi_00m = interp_phi(x, y, z-dz_min);
      double phi_00p = interp_phi(x, y, z+dz_min);
#else
      double phi_m00 = interp_phi(x-dx_min, y);
      double phi_p00 = interp_phi(x+dx_min, y);
      double phi_0m0 = interp_phi(x, y-dy_min);
      double phi_0p0 = interp_phi(x, y+dy_min);
#endif

#ifdef P4_TO_P8
      if(phi_m00>0 || phi_p00>0. || phi_0m0>0 || phi_0p0>0 || phi_00m>0 || phi_00p>0)
#else
      if(phi_m00>0 || phi_p00>0. || phi_0m0>0 || phi_0p0>0)
#endif
      {
        bool wall_m00 = qm_idx==-1;
        bool wall_p00 = qp_idx==-1;
        bool wall_0m0 = (qm_idx==-1 || is_quad_ymWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ymWall(p4est, tp_idx, &qp));
        bool wall_0p0 = (qm_idx==-1 || is_quad_ypWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ypWall(p4est, tp_idx, &qp));
#ifdef P4_TO_P8
        bool wall_00m = (qm_idx==-1 || is_quad_zmWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_zmWall(p4est, tp_idx, &qp));
        bool wall_00p = (qm_idx==-1 || is_quad_zpWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_zpWall(p4est, tp_idx, &qp));
#endif

        bool is_interface_m00 = !wall_m00 && phi_m00*phi_c <= 0;
        bool is_interface_p00 = !wall_p00 && phi_p00*phi_c <= 0;
        bool is_interface_0m0 = !wall_0m0 && phi_0m0*phi_c <= 0;
        bool is_interface_0p0 = !wall_0p0 && phi_0p0*phi_c <= 0;
#ifdef P4_TO_P8
        bool is_interface_00m = !wall_00m && phi_00m*phi_c <= 0;
        bool is_interface_00p = !wall_00p && phi_00p*phi_c <= 0;
#endif

        if(  is_interface_m00 || is_interface_0m0 ||
             is_interface_p00 || is_interface_0p0
     #ifdef P4_TO_P8
             || is_interface_00m || is_interface_00p
     #endif
             )
          matrix_has_nullspace_u = false;

        double val_interface_m00 = 0;
        double val_interface_p00 = 0;
        double val_interface_0m0 = 0;
        double val_interface_0p0 = 0;
#ifdef P4_TO_P8
        double val_interface_00m = 0;
        double val_interface_00p = 0;
#endif

        double d_m00 = dx_min;
        double d_p00 = dx_min;
        double d_0m0 = dy_min;
        double d_0p0 = dy_min;
#ifdef P4_TO_P8
        double d_00m = dz_min;
        double d_00p = dz_min;
#endif

        if(is_interface_m00) {
          double theta = interface_Location(0, d_m00, phi_c, phi_m00);
          theta = MAX(EPS, MIN(d_m00, theta));
          d_m00 = theta;
#ifdef P4_TO_P8
          val_interface_m00 = bc_u->interfaceValue(x - theta, y, z);
#else
          val_interface_m00 = bc_u->interfaceValue(x - theta, y);
#endif
        }

        if(is_interface_p00) {
          double theta = interface_Location(0, d_p00, phi_c, phi_p00);
          theta = MAX(EPS, MIN(d_p00, theta));
          d_p00 = theta;
#ifdef P4_TO_P8
          val_interface_p00 = bc_u->interfaceValue(x + theta, y, z);
#else
          val_interface_p00 = bc_u->interfaceValue(x + theta, y);
#endif
        }

        if(is_interface_0m0) {
          double theta = interface_Location(0, d_0m0, phi_c, phi_0m0);
          theta = MAX(EPS, MIN(d_0m0, theta));
          d_0m0 = theta;
#ifdef P4_TO_P8
          val_interface_0m0 = bc_u->interfaceValue(x, y - theta, z);
#else
          val_interface_0m0 = bc_u->interfaceValue(x, y - theta);
#endif
        }

        if(is_interface_0p0) {
          double theta = interface_Location(0, d_0p0, phi_c, phi_0p0);
          theta = MAX(EPS, MIN(d_0p0, theta));
          d_0p0 = theta;
#ifdef P4_TO_P8
          val_interface_0p0 = bc_u->interfaceValue(x, y + theta, z);
#else
          val_interface_0p0 = bc_u->interfaceValue(x, y + theta);
#endif
        }

#ifdef P4_TO_P8
        if(is_interface_00m) {
          double theta = interface_Location(0, d_00m, phi_c, phi_00m);
          theta = MAX(EPS, MIN(d_00m, theta));
          d_00m = theta;
          val_interface_00m = bc_u->interfaceValue(x, y, z - theta);
        }

        if(is_interface_00p) {
          double theta = interface_Location(0, d_00p, phi_c, phi_00p);
          theta = MAX(EPS, MIN(d_00p, theta));
          d_00p = theta;
          val_interface_00p = bc_u->interfaceValue(x, y, z + theta);
        }
#endif

        if(wall_m00) d_m00 = d_p00;
        if(wall_p00) d_p00 = d_m00;

        double coeff_m00 = -2*mu/d_m00/(d_m00+d_p00);
        double coeff_p00 = -2*mu/d_p00/(d_m00+d_p00);
        double coeff_0m0 = -2*mu/d_0m0/(d_0m0+d_0p0);
        double coeff_0p0 = -2*mu/d_0p0/(d_0m0+d_0p0);
#ifdef P4_TO_P8
        double coeff_00m = -2*mu/d_00m/(d_00m+d_00p);
        double coeff_00p = -2*mu/d_00p/(d_00m+d_00p);
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
#ifdef P4_TO_P8
        double diag = diag_add - (coeff_m00+coeff_p00+coeff_0m0+coeff_0p0+coeff_00m+coeff_00p);
#else
        double diag = diag_add - (coeff_m00+coeff_p00+coeff_0m0+coeff_0p0);
#endif
        coeff_m00 /= diag;
        coeff_p00 /= diag;
        coeff_0m0 /= diag;
        coeff_0p0 /= diag;
#ifdef P4_TO_P8
        coeff_00m /= diag;
        coeff_00p /= diag;
#endif
        rhs_u_p[f_idx] /= diag;

        //---------------------------------------------------------------------
        // insert the coefficients in the matrix
        //---------------------------------------------------------------------

        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);

        if(wall_m00)
        {
          if(!is_interface_p00)
          {
            p4est_gloidx_t f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_p00), dir::x);
            ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_m00, ADD_VALUES); CHKERRXX(ierr);
          }
          else
            rhs_u_p[f_idx] -= coeff_m00 * val_interface_p00;
#ifdef P4_TO_P8
          rhs_u_p[f_idx] -= coeff_m00 * (d_m00+d_p00) * bc_u->wallValue(x,y,z);
#else
          rhs_u_p[f_idx] -= coeff_m00 * (d_m00+d_p00) * bc_u->wallValue(x,y);
#endif
        }
        else if(!is_interface_m00)
        {
          p4est_gloidx_t f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_m00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_m00, ADD_VALUES); CHKERRXX(ierr);
        }
        else
          rhs_u_p[f_idx] -= coeff_m00*val_interface_m00;


        if(wall_p00)
        {
          if(!is_interface_m00)
          {
            p4est_gloidx_t f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_m00), dir::x);
            ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_p00, ADD_VALUES); CHKERRXX(ierr);
          }
          else
            rhs_u_p[f_idx] -= coeff_p00 * val_interface_m00;
#ifdef P4_TO_P8
          rhs_u_p[f_idx] -= coeff_p00 * (d_m00+d_p00) * bc_u->wallValue(x,y,z);
#else
          rhs_u_p[f_idx] -= coeff_p00 * (d_m00+d_p00) * bc_u->wallValue(x,y);
#endif
        }
        else if(!is_interface_p00)
        {
          p4est_gloidx_t f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_p00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_p00, ADD_VALUES); CHKERRXX(ierr);
        }
        else
          rhs_u_p[f_idx] -= coeff_p00*val_interface_p00;


        if(wall_0m0)
        {
#ifdef P4_TO_P8
          if(bc_u->wallType(x,ymin,z) == DIRICHLET) rhs_u_p[f_idx] -= coeff_0m0*bc_u->wallValue(x,ymin,z);
          else if(bc_u->wallType(x,ymin,z) == NEUMANN)
#else
          if(bc_u->wallType(x,ymin) == DIRICHLET) rhs_u_p[f_idx] -= coeff_0m0*bc_u->wallValue(x,ymin);
          else if(bc_u->wallType(x,ymin) == NEUMANN)
#endif
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_0m0, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
            rhs_u_p[f_idx] -= coeff_0m0 * d_0m0 * bc_u->wallValue(x,ymin,z);
#else
            rhs_u_p[f_idx] -= coeff_0m0 * d_0m0 * bc_u->wallValue(x,ymin);
#endif
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: invalid boundary condition.");
#endif
        }
        else if(!is_interface_0m0)
        {
          ngbd.resize(0);
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
#else
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1);
#endif
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: the grid is not uniform close to the interface.");
#endif
          p4est_gloidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_0m0, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_u_p[f_idx] -= coeff_0m0*val_interface_0m0;


        if(wall_0p0)
        {
#ifdef P4_TO_P8
          if(bc_u->wallType(x,ymax,z) == DIRICHLET) rhs_u_p[f_idx] -= coeff_0p0*bc_u->wallValue(x,ymax,z);
          else if(bc_u->wallType(x,ymax,z) == NEUMANN)
#else
          if(bc_u->wallType(x,ymax) == DIRICHLET) rhs_u_p[f_idx] -= coeff_0p0*bc_u->wallValue(x,ymax);
          else if(bc_u->wallType(x,ymax) == NEUMANN)
#endif
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_0p0, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
            rhs_u_p[f_idx] -= coeff_0p0 * d_0p0 * bc_u->wallValue(x,ymax,z);
#else
            rhs_u_p[f_idx] -= coeff_0p0 * d_0p0 * bc_u->wallValue(x,ymax);
#endif
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: invalid boundary condition.");
#endif
        }
        else if(!is_interface_0p0)
        {
          ngbd.resize(0);
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
#else
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
#endif
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: the grid is not uniform close to the interface.");
#endif
          p4est_gloidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_0p0, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_u_p[f_idx] -= coeff_0p0*val_interface_0p0;

#ifdef P4_TO_P8
        if(wall_00m)
        {
          if(bc_u->wallType(x,y,zmin) == DIRICHLET) rhs_u_p[f_idx] -= coeff_00m*bc_u->wallValue(x,y,zmin);
          else if(bc_u->wallType(x,y,zmin) == NEUMANN)
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_00m, ADD_VALUES); CHKERRXX(ierr);
            rhs_u_p[f_idx] -= coeff_00m * d_00m * bc_u->wallValue(x,y,zmin);
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: invalid boundary condition.");
#endif
        }
        else if(!is_interface_00m)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: the grid is not uniform close to the interface.");
#endif
          p4est_gloidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_00m, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_u_p[f_idx] -= coeff_00m*val_interface_00m;


        if(wall_00p)
        {
          if(bc_u->wallType(x,y,zmax) == DIRICHLET) rhs_u_p[f_idx] -= coeff_00p*bc_u->wallValue(x,y,zmax);
          else if(bc_u->wallType(x,y,zmax) == NEUMANN)
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_00p, ADD_VALUES); CHKERRXX(ierr);
            rhs_u_p[f_idx] -= coeff_00p * d_00p * bc_u->wallValue(x,y,zmax);
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: invalid boundary condition.");
#endif
        }
        else if(!is_interface_00p)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: the grid is not uniform close to the interface.");
#endif
          p4est_gloidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_00p, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_u_p[f_idx] -= coeff_00p*val_interface_00p;
#endif

        if(diag_add > 0) matrix_has_nullspace_u = false;

        continue;
      }
    }

#ifdef P4_TO_P8
    /*
     * If close to the interface and Neumann bc, do finite volume by hand
     * since cutting the voronoi cells in 3D with voro++ to have a nice level set is a nightmare ...
     * In 2D, cutting the partition is easy ... so Neumann interface is handled in the bulk case
     */
    if(bc_u->interfaceType()==NEUMANN && phi_c>-2*MAX(dx_min, dy_min, dz_min))
    {
      Cube3 c3;
      OctValue op;
      OctValue bc;
      c3.x0 = qm_idx==-1 ? x : x-dx_min/2; c3.y0 = y-dy_min/2; c3.z0 = z-dz_min/2;
      c3.x1 = qp_idx==-1 ? x : x+dx_min/2; c3.y1 = y+dy_min/2; c3.z1 = z+dz_min/2;

      const p4est_locidx_t *q2n = ngbd_n->nodes->local_nodes;
      double *phi_p;
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

      if(qm_idx==-1) {
        op.val000 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; op.val001 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]];
        op.val010 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; op.val011 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]];
        bc.val000 = bc_u->interfaceValue(x, y-dy_min, z-dz_min);
        bc.val001 = bc_u->interfaceValue(x, y-dy_min, z+dz_min);
        bc.val010 = bc_u->interfaceValue(x, y+dy_min, z-dz_min);
        bc.val011 = bc_u->interfaceValue(x, y+dy_min, z+dz_min); }
      else {
        op.val000 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2);
        op.val001 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2);
        op.val010 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2);
        op.val011 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2);
        bc.val000 = bc_u->interfaceValue(x-dx_min, y-dy_min, z-dz_min);
        bc.val001 = bc_u->interfaceValue(x-dx_min, y-dy_min, z+dz_min);
        bc.val010 = bc_u->interfaceValue(x-dx_min, y+dy_min, z-dz_min);
        bc.val011 = bc_u->interfaceValue(x-dx_min, y+dy_min, z+dz_min); }

      if(qp_idx==-1) {
        op.val100 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; op.val101 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]];
        op.val110 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; op.val111 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]];
        bc.val100 = bc_u->interfaceValue(x, y-dy_min, z-dz_min);
        bc.val101 = bc_u->interfaceValue(x, y-dy_min, z+dz_min);
        bc.val110 = bc_u->interfaceValue(x, y+dy_min, z-dz_min);
        bc.val111 = bc_u->interfaceValue(x, y+dy_min, z+dz_min); }
      else {
        op.val100 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2);
        op.val101 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2);
        op.val110 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2);
        op.val111 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2);
        bc.val100 = bc_u->interfaceValue(x+dx_min, y-dy_min, z-dz_min);
        bc.val101 = bc_u->interfaceValue(x+dx_min, y-dy_min, z+dz_min);
        bc.val110 = bc_u->interfaceValue(x+dx_min, y+dy_min, z-dz_min);
        bc.val111 = bc_u->interfaceValue(x+dx_min, y+dy_min, z+dz_min); }

      double volume = c3.volume_In_Negative_Domain(op);

      bool is_pos = (op.val000>0 || op.val001>0 || op.val010>0 || op.val011>0 ||
                     op.val100>0 || op.val101>0 || op.val110>0 || op.val111>0 );
      bool is_neg = (op.val000<0 || op.val001<0 || op.val010<0 || op.val011<0 ||
                     op.val100<0 || op.val101<0 || op.val110<0 || op.val111<0 );

      /* entirely in the positive domain */
      if(!is_neg)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
        rhs_u_p[f_idx] = 0;
        continue;
      }

      if(is_pos)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, volume*diag_add, ADD_VALUES); CHKERRXX(ierr);
        if(diag_add > 0) matrix_has_nullspace_u = false;
        rhs_u_p[f_idx] *= volume;
        rhs_u_p[f_idx] += mu * c3.integrate_Over_Interface(bc, op);

        Cube2 c2;
        QuadValue qp;

        // m00
        c2.x0 = y-dy_min/2; c2.y0 = z-dz_min/2;
        c2.x1 = y+dy_min/2; c2.y1 = z+dz_min/2;
        if(qm_idx==-1) {
          qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]];
          qp.val10 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; qp.val11 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]]; }
        else {
          qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2);
          qp.val01 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2);
          qp.val10 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2);
          qp.val11 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_m00 = c2.area_In_Negative_Domain(qp);

        if(qm_idx==-1)
          rhs_u_p[f_idx] += mu * s_m00 * bc_u->wallValue(xmin, y, z);
        else
        {
          p4est_gloidx_t f_tmp_g = face_global_number(faces->q2f(qm_idx, dir::f_m00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_m00/dx_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_m00/dx_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // p00
        if(qp_idx==-1) {
          qp.val00 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; qp.val01 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]];
          qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]]; }
        else {
          qp.val00 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2);
          qp.val01 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2);
          qp.val10 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2);
          qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_p00 = c2.area_In_Negative_Domain(qp);

        if(qp_idx==-1)
          rhs_u_p[f_idx] += mu * s_p00 * bc_u->wallValue(xmax, y, z);
        else
        {
          p4est_gloidx_t f_tmp_g = face_global_number(faces->q2f(qp_idx, dir::f_p00), dir::x);
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_p00/dx_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_p00/dx_min, ADD_VALUES); CHKERRXX(ierr);
        }

        // 0m0
        c2.x0 = qm_idx==-1 ? x : x-dx_min/2; c2.y0 = z-dz_min/2;
        c2.x1 = qp_idx==-1 ? x : x+dx_min/2; c2.y1 = z+dz_min/2;
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2); }
        double s_0m0 = c2.area_In_Negative_Domain(qp);

        if(qm_idx==-1)
          rhs_u_p[f_idx] += mu * s_0m0 * bc_u->wallValue(x, ymin, z);
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
          p4est_topidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_0m0/dy_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_0m0/dy_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // 0p0
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_0p0 = c2.area_In_Negative_Domain(qp);

        if(qp_idx==-1)
          rhs_u_p[f_idx] += mu * s_0p0 * bc_u->wallValue(x, ymax, z);
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
          p4est_topidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_0p0/dy_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_0p0/dy_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // 00m
        c2.x0 = qm_idx==-1 ? x : x-dx_min/2; c2.y0 = y-dy_min/2;
        c2.x1 = qp_idx==-1 ? x : x+dx_min/2; c2.y1 = y+dy_min/2;
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2); }
        double s_00m = c2.area_In_Negative_Domain(qp);

        if(qm_idx==-1)
          rhs_u_p[f_idx] += mu * s_00m * bc_u->wallValue(x, y, zmin);
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
          p4est_topidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_00m/dz_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_00m/dz_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // 00p
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_00p = c2.area_In_Negative_Domain(qp);

        if(qp_idx==-1)
          rhs_u_p[f_idx] += mu * s_00p * bc_u->wallValue(x, y, zmax);
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
          p4est_topidx_t f_tmp_g;
          if(quad_idx==qm_idx) f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00), dir::x);
          else                 f_tmp_g = face_global_number(faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00), dir::x);

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_00p/dz_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_00p/dz_min, ADD_VALUES); CHKERRXX(ierr);
        }

        ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

        continue;
      }
    }
#endif

    /*
     * Bulk case, away from the interface
     * Use finite volumes on the voronoi cells
     */
#ifdef P4_TO_P8
    const vector<Voronoi3DPoint> *points;
#else
    vector<Voronoi2DPoint> *points;
    vector<Point2> *partition;
    voro[f_idx].get_Partition(partition);
#endif
    voro[f_idx].get_Points(points);

    /* integrally in positive domain */
#ifndef P4_TO_P8
    if(partition->size()==0)
    {
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_u_p[f_idx] = 0;
      continue;
    }
#endif

    double volume = voro[f_idx].volume();
    ierr = MatSetValue(A, f_idx_g, f_idx_g, volume*diag_add, ADD_VALUES); CHKERRXX(ierr);
    rhs_u_p[f_idx] *= volume;
    if(diag_add>0) matrix_has_nullspace_u = false;

    /* bulk case, finite volume on voronoi cell */
#ifdef P4_TO_P8
    Point3 pc(x,y,z);
#else
    Point2 pc(x,y);
#endif
    for(unsigned int m=0; m<points->size(); ++m)
    {
      PetscInt m_idx_g;
      double x_pert = x;
      if(fabs(x-xmax)<EPS) x_pert = xmax-2*EPS;
      if(fabs(x-xmin)<EPS) x_pert = xmin+2*EPS;

#ifdef P4_TO_P8
      double s = (*points)[m].s;
#else
      int k = mod(m-1, points->size());
      double s = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      double d = ((*points)[m].p - pc).norm_L2();;

      switch((*points)[m].n)
      {
      /* left wall (note that the dirichlet case has already been done at the beginning of the loop) */
      case WALL_m00:
#ifdef P4_TO_P8
        switch(bc_u->wallType(xmin,y,z))
#else
        switch(bc_u->wallType(xmin,y))
#endif
        {
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(xmin,y,z);
#else
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(xmin,y);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

        /* right wall (note that the dirichlet case has already been done at the beginning of the loop) */
      case WALL_p00:
#ifdef P4_TO_P8
        switch(bc_u->wallType(xmax,y,z))
#else
        switch(bc_u->wallType(xmax,y))
#endif
        {
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(xmax,y,z);
#else
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(xmax,y);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_0m0:
#ifdef P4_TO_P8
        switch(bc_u->wallType(x_pert,ymin,z))
#else
        switch(bc_u->wallType(x_pert,ymin))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymin,z) / d;
#else
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymin) / d;
#endif
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymin,z);
#else
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymin);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_0p0:
#ifdef P4_TO_P8
        switch(bc_u->wallType(x_pert,ymax,z))
#else
        switch(bc_u->wallType(x_pert,ymax))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymax,z) / d;
#else
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymax) / d;
#endif
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymax,z);
#else
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,ymax);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

#ifdef P4_TO_P8
      case WALL_00m:
        switch(bc_u->wallType(x_pert,y,zmin))
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,y,zmin) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,y,zmin);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_00p:
        switch(bc_u->wallType(x_pert,y,zmax))
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,y,zmax) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[f_idx] += mu*s*bc_u->wallValue(x_pert,y,zmax);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;
#endif

      case INTERFACE:
        switch( bc_u->interfaceType())
        {
        /* note that DIRICHLET done with finite differences */
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: Neumann boundary conditions should be treated separately in 3D ...");
#else
          rhs_u_p[f_idx] += mu*s*bc_u->interfaceValue(((*points)[m].p.x+x)/2.,((*points)[m].p.y+y)/2.);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      default:
        /* add coefficients in the matrix */
        m_idx_g = face_global_number((*points)[m].n, dir::x);

        ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, f_idx_g, m_idx_g,-mu*s/d, ADD_VALUES); CHKERRXX(ierr);
      }
    }
  }

  ierr = VecRestoreArray(rhs_u, &rhs_u_p); CHKERRXX(ierr);

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  /* take care of the nullspace if needed */
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace_u, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(matrix_has_nullspace_u)
  {
    if(A_null_space == PETSC_NULL)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);
    }
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
    ierr = MatNullSpaceRemove(A_null_space, rhs_u, NULL); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverFaces_setup_linear_system, A, rhs_u, 0, 0); CHKERRXX(ierr);
}



void PoissonSolverFaces::setup_linear_system_v()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_PoissonSolverFaces_setup_linear_system, A, rhs_v, 0, 0); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  matrix_has_nullspace_v = true;

  double *rhs_v_p;
  ierr = VecGetArray(rhs_u, &rhs_v_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[0]; ++f_idx)
  {
    p4est_gloidx_t f_idx_g = f_idx + proc_offset[0][p4est->mpirank];

    faces->f2q(f_idx, dir::y, quad_idx, tree_idx);

    p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

    double x = faces->x_fr_f(f_idx, dir::y);
    double y = faces->y_fr_f(f_idx, dir::y);
#ifdef P4_TO_P8
    double z = faces->z_fr_f(f_idx, dir::y);
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
      rhs_v_p[f_idx] = 0;
      continue;
    }

    p4est_locidx_t qm_idx=-1, qp_idx=-1;
    vector<p4est_quadrant_t> ngbd;
    p4est_quadrant_t qm, qp;
    p4est_topidx_t tm_idx=-1, tp_idx=-1;
    if(faces->q2f(quad_idx, dir::f_0m0)==f_idx)
    {
      qp_idx = quad_idx;
      tp_idx = tree_idx;
      qp = *quad; qp.p.piggy3.local_num = qp_idx;
      ngbd.clear();
#ifdef P4_TO_P8
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1, 0);
#else
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0,-1);
#endif
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
#ifdef P4_TO_P8
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1, 0);
#else
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
#endif
      if(ngbd.size()>0)
      {
        qp = ngbd[0];
        qp_idx = ngbd[0].p.piggy3.local_num;
        tp_idx = ngbd[0].p.piggy3.which_tree;
      }
    }

    /* check for walls */
#ifdef P4_TO_P8
    if(qm_idx==-1 && bc_v->wallType(x,ymin,z)==DIRICHLET)
#else
    if(qm_idx==-1 && bc_v->wallType(x,ymin)==DIRICHLET)
#endif
    {
      matrix_has_nullspace_v = false;
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
      rhs_v_p[f_idx] = bc_v->wallValue(x,ymin,z);
#else
      rhs_v_p[f_idx] = bc_v->wallValue(x,ymin);
#endif
      continue;
    }

#ifdef P4_TO_P8
    if(qp_idx==-1 && bc_v->wallType(x,ymax,z)==DIRICHLET)
#else
    if(qp_idx==-1 && bc_v->wallType(x,ymax)==DIRICHLET)
#endif
    {
      matrix_has_nullspace_v = false;
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
      rhs_v_p[f_idx] = bc_v->wallValue(x,ymax,z);
#else
      rhs_v_p[f_idx] = bc_v->wallValue(x,ymax);
#endif
      continue;
    }


    /*
     * close to interface and dirichlet => finite differences
     */
#ifdef P4_TO_P8
    if(bc_v->interfaceType()==DIRICHLET && phi_c>-2*MAX(dx_min,dy_min,dz_min))
#else
    if(bc_v->interfaceType()==DIRICHLET && phi_c>-2*MAX(dx_min,dy_min))
#endif
    {
      if(fabs(phi_c) < EPS)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
        rhs_v_p[f_idx] = bc_v->interfaceValue(x,y,z);
#else
        rhs_v_p[f_idx] = bc_v->interfaceValue(x,y);
#endif
        matrix_has_nullspace_v = false;
        continue;
      }

      if(phi_c>0)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
        rhs_v_p[f_idx] = 0;
        continue;
      }

#ifdef P4_TO_P8
      double phi_m00 = interp_phi(x-dx_min, y, z);
      double phi_p00 = interp_phi(x+dx_min, y, z);
      double phi_0m0 = interp_phi(x, y-dy_min, z);
      double phi_0p0 = interp_phi(x, y+dy_min, z);
      double phi_00m = interp_phi(x, y, z-dz_min);
      double phi_00p = interp_phi(x, y, z+dz_min);
#else
      double phi_m00 = interp_phi(x-dx_min, y);
      double phi_p00 = interp_phi(x+dx_min, y);
      double phi_0m0 = interp_phi(x, y-dy_min);
      double phi_0p0 = interp_phi(x, y+dy_min);
#endif

#ifdef P4_TO_P8
      if(phi_m00>0 || phi_p00>0. || phi_0m0>0 || phi_0p0>0 || phi_00m>0 || phi_00p>0)
#else
      if(phi_m00>0 || phi_p00>0. || phi_0m0>0 || phi_0p0>0)
#endif
      {
        bool wall_m00 = (qm_idx==-1 || is_quad_xmWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xmWall(p4est, tp_idx, &qp));
        bool wall_p00 = (qm_idx==-1 || is_quad_xpWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_xpWall(p4est, tp_idx, &qp));
        bool wall_0m0 = qm_idx==-1;
        bool wall_0p0 = qp_idx==-1;
#ifdef P4_TO_P8
        bool wall_00m = (qm_idx==-1 || is_quad_zmWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_zmWall(p4est, tp_idx, &qp));
        bool wall_00p = (qm_idx==-1 || is_quad_zpWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_zpWall(p4est, tp_idx, &qp));
#endif

        bool is_interface_m00 = !wall_m00 && phi_m00*phi_c <= 0;
        bool is_interface_p00 = !wall_p00 && phi_p00*phi_c <= 0;
        bool is_interface_0m0 = !wall_0m0 && phi_0m0*phi_c <= 0;
        bool is_interface_0p0 = !wall_0p0 && phi_0p0*phi_c <= 0;
#ifdef P4_TO_P8
        bool is_interface_00m = !wall_00m && phi_00m*phi_c <= 0;
        bool is_interface_00p = !wall_00p && phi_00p*phi_c <= 0;
#endif

        if(  is_interface_m00 || is_interface_0m0 ||
             is_interface_p00 || is_interface_0p0
     #ifdef P4_TO_P8
             || is_interface_00m || is_interface_00p
     #endif
             )
          matrix_has_nullspace_v = false;

        double val_interface_m00 = 0;
        double val_interface_p00 = 0;
        double val_interface_0m0 = 0;
        double val_interface_0p0 = 0;
#ifdef P4_TO_P8
        double val_interface_00m = 0;
        double val_interface_00p = 0;
#endif

        double d_m00 = dx_min;
        double d_p00 = dx_min;
        double d_0m0 = dy_min;
        double d_0p0 = dy_min;
#ifdef P4_TO_P8
        double d_00m = dz_min;
        double d_00p = dz_min;
#endif

        if(is_interface_m00) {
          double theta = interface_Location(0, d_m00, phi_c, phi_m00);
          theta = MAX(EPS, MIN(d_m00, theta));
          d_m00 = theta;
#ifdef P4_TO_P8
          val_interface_m00 = bc_v->interfaceValue(x - theta, y, z);
#else
          val_interface_m00 = bc_v->interfaceValue(x - theta, y);
#endif
        }

        if(is_interface_p00) {
          double theta = interface_Location(0, d_p00, phi_c, phi_p00);
          theta = MAX(EPS, MIN(d_p00, theta));
          d_p00 = theta;
#ifdef P4_TO_P8
          val_interface_p00 = bc_v->interfaceValue(x + theta, y, z);
#else
          val_interface_p00 = bc_v->interfaceValue(x + theta, y);
#endif
        }

        if(is_interface_0m0) {
          double theta = interface_Location(0, d_0m0, phi_c, phi_0m0);
          theta = MAX(EPS, MIN(d_0m0, theta));
          d_0m0 = theta;
#ifdef P4_TO_P8
          val_interface_0m0 = bc_v->interfaceValue(x, y - theta, z);
#else
          val_interface_0m0 = bc_v->interfaceValue(x, y - theta);
#endif
        }

        if(is_interface_0p0) {
          double theta = interface_Location(0, d_0p0, phi_c, phi_0p0);
          theta = MAX(EPS, MIN(d_0p0, theta));
          d_0p0 = theta;
#ifdef P4_TO_P8
          val_interface_0p0 = bc_v->interfaceValue(x, y + theta, z);
#else
          val_interface_0p0 = bc_v->interfaceValue(x, y + theta);
#endif
        }

#ifdef P4_TO_P8
        if(is_interface_00m) {
          double theta = interface_Location(0, d_00m, phi_c, phi_00m);
          theta = MAX(EPS, MIN(d_00m, theta));
          d_00m = theta;
          val_interface_00m = bc_v->interfaceValue(x, y, z - theta);
        }

        if(is_interface_00p) {
          double theta = interface_Location(0, d_00p, phi_c, phi_00p);
          theta = MAX(EPS, MIN(d_00p, theta));
          d_00p = theta;
          val_interface_00p = bc_v->interfaceValue(x, y, z + theta);
        }
#endif

        if(wall_m00) d_m00 = d_p00;
        if(wall_p00) d_p00 = d_m00;

        double coeff_m00 = -2*mu/d_m00/(d_m00+d_p00);
        double coeff_p00 = -2*mu/d_p00/(d_m00+d_p00);
        double coeff_0m0 = -2*mu/d_0m0/(d_0m0+d_0p0);
        double coeff_0p0 = -2*mu/d_0p0/(d_0m0+d_0p0);
#ifdef P4_TO_P8
        double coeff_00m = -2*mu/d_00m/(d_00m+d_00p);
        double coeff_00p = -2*mu/d_00p/(d_00m+d_00p);
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
#ifdef P4_TO_P8
        double diag = diag_add - (coeff_m00+coeff_p00+coeff_0m0+coeff_0p0+coeff_00m+coeff_00p);
#else
        double diag = diag_add - (coeff_m00+coeff_p00+coeff_0m0+coeff_0p0);
#endif
        coeff_m00 /= diag;
        coeff_p00 /= diag;
        coeff_0m0 /= diag;
        coeff_0p0 /= diag;
#ifdef P4_TO_P8
        coeff_00m /= diag;
        coeff_00p /= diag;
#endif
        rhs_v_p[f_idx] /= diag;

        if(diag_add > 0) matrix_has_nullspace_v = false;

        //---------------------------------------------------------------------
        // insert the coefficients in the matrix
        //---------------------------------------------------------------------

        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);

        if(wall_0m0)
        {
          if(!is_interface_0p0)
          {
            p4est_locidx_t f_tmp = faces->q2f(qp_idx, dir::f_0p0);
            p4est_gloidx_t f_tmp_g;
            if(f_tmp<faces->num_local[1]) f_tmp_g = f_tmp + proc_offset[1][p4est->mpirank];
            else { f_tmp -= faces->num_local[1]; f_tmp_g = faces->ghost_local_num[1][f_tmp] + proc_offset[1][faces->nonlocal_ranks[1][f_tmp]]; }
            ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_0m0, ADD_VALUES); CHKERRXX(ierr);
          }
          else
            rhs_v_p[f_idx] -= coeff_0m0 * val_interface_0p0;
#ifdef P4_TO_P8
          rhs_v_p[f_idx] -= coeff_0m0 * (d_0m0+d_0p0) * bc_v->wallValue(x,y,z);
#else
          rhs_v_p[f_idx] -= coeff_0m0 * (d_0m0+d_0p0) * bc_v->wallValue(x,y);
#endif
        }
        else if(!is_interface_0m0)
        {
          p4est_locidx_t f_tmp = faces->q2f(qm_idx, dir::f_0m0);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_0m0, ADD_VALUES); CHKERRXX(ierr);
        }
        else
          rhs_v_p[f_idx] -= coeff_0m0*val_interface_0m0;


        if(wall_0p0)
        {
          if(!is_interface_0m0)
          {
            p4est_locidx_t f_tmp = faces->q2f(qm_idx, dir::f_0m0);
            p4est_gloidx_t f_tmp_g;
            if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
            else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }
            ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_0p0, ADD_VALUES); CHKERRXX(ierr);
          }
          else
            rhs_v_p[f_idx] -= coeff_0p0 * val_interface_0m0;
#ifdef P4_TO_P8
          rhs_v_p[f_idx] -= coeff_0p0 * (d_0m0+d_0p0) * bc_v->wallValue(x,y,z);
#else
          rhs_v_p[f_idx] -= coeff_0p0 * (d_0m0+d_0p0) * bc_v->wallValue(x,y);
#endif
        }
        else if(!is_interface_0p0)
        {
          p4est_locidx_t f_tmp = faces->q2f(qp_idx, dir::f_0p0);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_0p0, ADD_VALUES); CHKERRXX(ierr);
        }
        else
          rhs_v_p[f_idx] -= coeff_0p0*val_interface_0p0;


        if(wall_m00)
        {
#ifdef P4_TO_P8
          if(bc_v->wallType(xmin,y,z) == DIRICHLET) rhs_v_p[f_idx] -= coeff_m00*bc_v->wallValue(xmin,y,z);
          else if(bc_v->wallType(xmin,y,z) == NEUMANN)
#else
          if(bc_v->wallType(xmin,y) == DIRICHLET) rhs_v_p[f_idx] -= coeff_m00*bc_v->wallValue(xmin,y);
          else if(bc_v->wallType(xmin,y) == NEUMANN)
#endif
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_m00, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
            rhs_v_p[f_idx] -= coeff_m00 * d_m00 * bc_v->wallValue(xmin,y,z);
#else
            rhs_v_p[f_idx] -= coeff_m00 * d_m00 * bc_v->wallValue(xmin,y);
#endif
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_v: invalid boundary condition.");
#endif
        }
        else if(!is_interface_m00)
        {
          ngbd.resize(0);
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0, 0);
#else
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0);
#endif
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_v: the grid is not uniform close to the interface.");
#endif
          p4est_locidx_t f_tmp;
          if(quad_idx==qm_idx) f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0p0);
          else                 f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0m0);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_m00, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_v_p[f_idx] -= coeff_m00*val_interface_m00;


        if(wall_p00)
        {
#ifdef P4_TO_P8
          if(bc_v->wallType(xmax,y,z) == DIRICHLET) rhs_v_p[f_idx] -= coeff_p00*bc_v->wallValue(xmax,y,z);
          else if(bc_v->wallType(xmax,y,z) == NEUMANN)
#else
          if(bc_v->wallType(xmax,y) == DIRICHLET) rhs_v_p[f_idx] -= coeff_p00*bc_v->wallValue(xmax,y);
          else if(bc_v->wallType(xmax,y) == NEUMANN)
#endif
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_p00, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
            rhs_v_p[f_idx] -= coeff_p00 * d_p00 * bc_v->wallValue(xmax,y,z);
#else
            rhs_v_p[f_idx] -= coeff_p00 * d_p00 * bc_v->wallValue(xmax,y);
#endif
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_v: invalid boundary condition.");
#endif
        }
        else if(!is_interface_p00)
        {
          ngbd.resize(0);
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0, 0);
#else
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
#endif
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_v: the grid is not uniform close to the interface.");
#endif
          p4est_locidx_t f_tmp;
          if(quad_idx==qm_idx) f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0p0);
          else                 f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0m0);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_p00, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_v_p[f_idx] -= coeff_p00*val_interface_p00;


#ifdef P4_TO_P8
        if(wall_00m)
        {
          if(bc_v->wallType(x,y,zmin) == DIRICHLET) rhs_v_p[f_idx] -= coeff_00m*bc_v->wallValue(x,y,zmin);
          else if(bc_v->wallType(x,y,zmin) == NEUMANN)
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_00m, ADD_VALUES); CHKERRXX(ierr);
            rhs_v_p[f_idx] -= coeff_00m * d_00m * bc_v->wallValue(x,y,zmin);
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_v: invalid boundary condition.");
#endif
        }
        else if(!is_interface_00m)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0,-1);
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: the grid is not uniform close to the interface.");
#endif
          p4est_locidx_t f_tmp;
          if(quad_idx==qm_idx) f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0p0);
          else                 f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0m0);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[1]) f_tmp_g = f_tmp + proc_offset[1][p4est->mpirank];
          else { f_tmp -= faces->num_local[1]; f_tmp_g = faces->ghost_local_num[1][f_tmp] + proc_offset[1][faces->nonlocal_ranks[1][f_tmp]]; }
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_00m, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_v_p[f_idx] -= coeff_00m*val_interface_00m;


        if(wall_00p)
        {
          if(bc_v->wallType(x,y,zmax) == DIRICHLET) rhs_v_p[f_idx] -= coeff_00p*bc_v->wallValue(x,y,zmax);
          else if(bc_v->wallType(x,y,zmax) == NEUMANN)
          {
            ierr = MatSetValue(A, f_idx_g, f_idx_g, coeff_00p, ADD_VALUES); CHKERRXX(ierr);
            rhs_v_p[f_idx] -= coeff_00p * d_00p * bc_v->wallValue(x,y,zmax);
          }
#ifdef CASL_THROWS
          else
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_v: invalid boundary condition.");
#endif
        }
        else if(!is_interface_00p)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 0, 1);
#ifdef CASL_THROWS
          if(ngbd.size()!=1 && ngbd[0].level==quad->level)
            throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces->setup_linear_system_u: the grid is not uniform close to the interface.");
#endif
          p4est_locidx_t f_tmp;
          if(quad_idx==qm_idx) f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0p0);
          else                 f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_0m0);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[1]) f_tmp_g = f_tmp + proc_offset[1][p4est->mpirank];
          else { f_tmp -= faces->num_local[1]; f_tmp_g = faces->ghost_local_num[1][f_tmp] + proc_offset[1][faces->nonlocal_ranks[1][f_tmp]]; }
          ierr = MatSetValue(A, f_idx_g, f_tmp_g, coeff_00p, ADD_VALUES); CHKERRXX(ierr);
        }
        else rhs_v_p[f_idx] -= coeff_00p*val_interface_00p;
#endif

        continue;
      }
    }

#ifdef P4_TO_P8
    /*
     * If close to the interface and Neumann bc, do finite volume by hand
     * since cutting the voronoi cells in 3D with voro++ to have a nice level set is a nightmare ...
     * In 2D, cutting the partition is easy ... so Neumann interface is handled in the bulk case
     */
    if(bc_v->interfaceType()==NEUMANN && phi_c>-2*MAX(dx_min, dy_min, dz_min))
    {
      Cube3 c3;
      OctValue op;
      OctValue bc;
      c3.x0 = qm_idx==-1 ? x : x-dx_min/2; c3.y0 = y-dy_min/2; c3.z0 = z-dz_min/2;
      c3.x1 = qp_idx==-1 ? x : x+dx_min/2; c3.y1 = y+dy_min/2; c3.z1 = z+dz_min/2;

      const p4est_locidx_t *q2n = ngbd_n->nodes->local_nodes;
      double *phi_p;
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

      if(qm_idx==-1) {
        op.val000 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; op.val001 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]];
        op.val010 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; op.val011 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]];
        bc.val000 = bc_v->interfaceValue(x, y-dy_min, z-dz_min);
        bc.val001 = bc_v->interfaceValue(x, y-dy_min, z+dz_min);
        bc.val010 = bc_v->interfaceValue(x, y+dy_min, z-dz_min);
        bc.val011 = bc_v->interfaceValue(x, y+dy_min, z+dz_min); }
      else {
        op.val000 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2);
        op.val001 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2);
        op.val010 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2);
        op.val011 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2);
        bc.val000 = bc_v->interfaceValue(x-dx_min, y-dy_min, z-dz_min);
        bc.val001 = bc_v->interfaceValue(x-dx_min, y-dy_min, z+dz_min);
        bc.val010 = bc_v->interfaceValue(x-dx_min, y+dy_min, z-dz_min);
        bc.val011 = bc_v->interfaceValue(x-dx_min, y+dy_min, z+dz_min); }

      if(qp_idx==-1) {
        op.val100 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; op.val101 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]];
        op.val110 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; op.val111 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]];
        bc.val100 = bc_v->interfaceValue(x, y-dy_min, z-dz_min);
        bc.val101 = bc_v->interfaceValue(x, y-dy_min, z+dz_min);
        bc.val110 = bc_v->interfaceValue(x, y+dy_min, z-dz_min);
        bc.val111 = bc_v->interfaceValue(x, y+dy_min, z+dz_min); }
      else {
        op.val100 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2);
        op.val101 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2);
        op.val110 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2);
        op.val111 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2);
        bc.val100 = bc_v->interfaceValue(x+dx_min, y-dy_min, z-dz_min);
        bc.val101 = bc_v->interfaceValue(x+dx_min, y-dy_min, z+dz_min);
        bc.val110 = bc_v->interfaceValue(x+dx_min, y+dy_min, z-dz_min);
        bc.val111 = bc_v->interfaceValue(x+dx_min, y+dy_min, z+dz_min); }

      double volume = c3.volume_In_Negative_Domain(op);

      bool is_pos = (op.val000>0 || op.val001>0 || op.val010>0 || op.val011>0 ||
                     op.val100>0 || op.val101>0 || op.val110>0 || op.val111>0 );
      bool is_neg = (op.val000<0 || op.val001<0 || op.val010<0 || op.val011<0 ||
                     op.val100<0 || op.val101<0 || op.val110<0 || op.val111<0 );

      /* entirely in the positive domain */
      if(!is_neg)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
        rhs_v_p[f_idx] = 0;
        continue;
      }

      if(is_pos)
      {
        ierr = MatSetValue(A, f_idx_g, f_idx_g, volume*diag_add, ADD_VALUES); CHKERRXX(ierr);
        if(diag_add > 0) matrix_has_nullspace_u = false;
        rhs_v_p[f_idx] *= volume;
        rhs_v_p[f_idx] += mu * c3.integrate_Over_Interface(bc, op);

        Cube2 c2;
        QuadValue qp;

        // m00
        c2.x0 = y-dy_min/2; c2.y0 = z-dz_min/2;
        c2.x1 = y+dy_min/2; c2.y1 = z+dz_min/2;
        if(qm_idx==-1) {
          qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]];
          qp.val10 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; qp.val11 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]]; }
        else {
          qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2);
          qp.val01 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2);
          qp.val10 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2);
          qp.val11 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_m00 = c2.area_In_Negative_Domain(qp);

        if(qm_idx==-1)
          rhs_v_p[f_idx] += mu * s_m00 * bc_v->wallValue(xmin, y, z);
        else
        {
          p4est_locidx_t f_tmp = faces->q2f(qm_idx, dir::f_m00);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_m00/dx_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_m00/dx_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // p00
        if(qp_idx==-1) {
          qp.val00 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; qp.val01 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]];
          qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]]; }
        else {
          qp.val00 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2);
          qp.val01 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2);
          qp.val10 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2);
          qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_p00 = c2.area_In_Negative_Domain(qp);

        if(qp_idx==-1)
          rhs_v_p[f_idx] += mu * s_p00 * bc_v->wallValue(xmax, y, z);
        else
        {
          p4est_locidx_t f_tmp = faces->q2f(qp_idx, dir::f_p00);
          p4est_gloidx_t f_tmp_g;
          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_p00/dx_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_p00/dx_min, ADD_VALUES); CHKERRXX(ierr);
        }

        // 0m0
        c2.x0 = qm_idx==-1 ? x : x-dx_min/2; c2.y0 = z-dz_min/2;
        c2.x1 = qp_idx==-1 ? x : x+dx_min/2; c2.y1 = z+dz_min/2;
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2); }
        double s_0m0 = c2.area_In_Negative_Domain(qp);

        if(qm_idx==-1)
          rhs_v_p[f_idx] += mu * s_0m0 * bc_v->wallValue(x, ymin, z);
        else
        {
          ngbd.resize(0);
          p4est_locidx_t f_tmp;
          p4est_topidx_t f_tmp_g;
          if(qp_idx==-1) {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1, 0);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00); }
          else {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1, 0);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00); }

          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_0m0/dy_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_0m0/dy_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // 0p0
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_0p0 = c2.area_In_Negative_Domain(qp);

        if(qp_idx==-1)
          rhs_v_p[f_idx] += mu * s_0p0 * bc_v->wallValue(x, ymax, z);
        else
        {
          ngbd.resize(0);
          p4est_locidx_t f_tmp;
          p4est_topidx_t f_tmp_g;
          if(qp_idx==-1) {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1, 0);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00); }
          else {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1, 0);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00); }

          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_0p0/dy_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_0p0/dy_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // 00m
        c2.x0 = qm_idx==-1 ? x : x-dx_min/2; c2.y0 = y-dy_min/2;
        c2.x1 = qp_idx==-1 ? x : x+dx_min/2; c2.y1 = y+dy_min/2;
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmm]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpm]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z-dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y+dy_min/2, z-dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmm]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppm]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y-dy_min/2, z-dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z-dz_min/2); }
        double s_00m = c2.area_In_Negative_Domain(qp);

        if(qm_idx==-1)
          rhs_v_p[f_idx] += mu * s_00m * bc_v->wallValue(x, y, zmin);
        else
        {
          ngbd.resize(0);
          p4est_locidx_t f_tmp;
          p4est_topidx_t f_tmp_g;
          if(qp_idx==-1) {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0,-1);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00); }
          else {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0,-1);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00); }

          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_00m/dz_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_00m/dz_min, ADD_VALUES); CHKERRXX(ierr);
        }


        // 00p
        if(qm_idx==-1) { qp.val00 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mmp]]; qp.val01 = phi_p[q2n[qp_idx*P4EST_CHILDREN + dir::v_mpp]]; }
        else           { qp.val00 = interp_phi(x-dx_min/2, y-dy_min/2, z+dz_min/2); qp.val01 = interp_phi(x-dx_min/2, y+dy_min/2, z+dz_min/2); }
        if(qp_idx==-1) { qp.val10 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_pmp]]; qp.val11 = phi_p[q2n[qm_idx*P4EST_CHILDREN + dir::v_ppp]]; }
        else           { qp.val10 = interp_phi(x+dx_min/2, y-dy_min/2, z+dz_min/2); qp.val11 = interp_phi(x+dx_min/2, y+dy_min/2, z+dz_min/2); }
        double s_00p = c2.area_In_Negative_Domain(qp);

        if(qp_idx==-1)
          rhs_v_p[f_idx] += mu * s_00p * bc_v->wallValue(x, y, zmax);
        else
        {
          ngbd.resize(0);
          p4est_locidx_t f_tmp;
          p4est_topidx_t f_tmp_g;
          if(qp_idx==-1) {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 0, 1);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_p00); }
          else {
            ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 0, 1);
            f_tmp = faces->q2f(ngbd[0].p.piggy3.local_num, dir::f_m00); }

          if(f_tmp<faces->num_local[0]) f_tmp_g = f_tmp + proc_offset[0][p4est->mpirank];
          else { f_tmp -= faces->num_local[0]; f_tmp_g = faces->ghost_local_num[0][f_tmp] + proc_offset[0][faces->nonlocal_ranks[0][f_tmp]]; }

          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s_00p/dz_min, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, f_idx_g, f_tmp_g,-mu*s_00p/dz_min, ADD_VALUES); CHKERRXX(ierr);
        }

        ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

        continue;
      }
    }
#endif


    /*
     * Bulk case, away from the interface
     * Use finite volumes on the voronoi cells
     */
#ifdef P4_TO_P8
    const vector<Voronoi3DPoint> *points;
#else
    vector<Voronoi2DPoint> *points;
    vector<Point2> *partition;
    voro[f_idx].get_Partition(partition);
#endif
    voro[f_idx].get_Points(points);

    /* integrally in positive domain */
#ifndef P4_TO_P8
    if(partition->size()==0)
    {
      ierr = MatSetValue(A, f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_v_p[f_idx] = 0;
      continue;
    }
#endif

    double volume = voro[f_idx].volume();
    ierr = MatSetValue(A, f_idx_g, f_idx_g, volume*diag_add, ADD_VALUES); CHKERRXX(ierr);
    rhs_v_p[f_idx] *= volume;
    if(diag_add>0) matrix_has_nullspace_u = false;

    /* bulk case, finite volume on voronoi cell */
#ifdef P4_TO_P8
    Point3 pc(x,y,z);
#else
    Point2 pc(x,y);
#endif
    for(unsigned int m=0; m<points->size(); ++m)
    {
      PetscInt m_idx_g;
      double x_pert = x;
      if(fabs(x-xmax)<EPS) x_pert = xmax-2*EPS;
      if(fabs(x-xmin)<EPS) x_pert = xmin+2*EPS;

#ifdef P4_TO_P8
      double s = (*points)[m].s;
#else
      int k = mod(m-1, points->size());
      double s = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      double d = ((*points)[m].p - pc).norm_L2();;

      switch((*points)[m].n)
      {
      /* left wall (note that the dirichlet case has already been done at the beginning of the loop) */
      case WALL_m00:
#ifdef P4_TO_P8
        switch(bc_v->wallType(xmin,y,z))
#else
        switch(bc_v->wallType(xmin,y))
#endif
        {
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(xmin,y,z);
#else
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(xmin,y);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

        /* right wall (note that the dirichlet case has already been done at the beginning of the loop) */
      case WALL_p00:
#ifdef P4_TO_P8
        switch(bc_v->wallType(xmax,y,z))
#else
        switch(bc_v->wallType(xmax,y))
#endif
        {
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(xmax,y,z);
#else
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(xmax,y);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_0m0:
#ifdef P4_TO_P8
        switch(bc_v->wallType(x_pert,ymin,z))
#else
        switch(bc_v->wallType(x_pert,ymin))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymin,z) / d;
#else
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymin) / d;
#endif
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymin,z);
#else
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymin);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_0p0:
#ifdef P4_TO_P8
        switch(bc_v->wallType(x_pert,ymax,z))
#else
        switch(bc_v->wallType(x_pert,ymax))
#endif
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
#ifdef P4_TO_P8
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymax,z) / d;
#else
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymax) / d;
#endif
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymax,z);
#else
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,ymax);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

#ifdef P4_TO_P8
      case WALL_00m:
        switch(bc_v->wallType(x_pert,y,zmin))
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,y,zmin) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,y,zmin);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_00p:
        switch(bc_v->wallType(x_pert,y,zmax))
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2;
          ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,y,zmax) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_v_p[f_idx] += mu*s*bc_v->wallValue(x_pert,y,zmax);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;
#endif

      case INTERFACE:
        switch( bc_v->interfaceType())
        {
        /* note that DIRICHLET done with finite differences */
        case NEUMANN:
          /* nothing to do for the matrix */
#ifdef P4_TO_P8
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: Neumann boundary conditions should be treated separately in 3D ...");
#else
          rhs_v_p[f_idx] += mu*s*bc_v->interfaceValue(((*points)[m].p.x+x)/2.,((*points)[m].p.y+y)/2.);
#endif
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      default:
        /* add coefficients in the matrix */
        m_idx_g = (*points)[m].n;
        if(m_idx_g<faces->num_local[0]) m_idx_g += proc_offset[0][p4est->mpirank];
        else
        {
          m_idx_g = m_idx_g - faces->num_local[0];
          m_idx_g = faces->ghost_local_num[0][m_idx_g] + proc_offset[0][faces->nonlocal_ranks[0][m_idx_g]];
        }

        ierr = MatSetValue(A, f_idx_g, f_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, f_idx_g, m_idx_g,-mu*s/d, ADD_VALUES); CHKERRXX(ierr);
      }
    }
  }

  ierr = VecRestoreArray(rhs_u, &rhs_v_p); CHKERRXX(ierr);

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

//  VecView(rhs_u, PETSC_VIEWER_STDOUT_WORLD);
//  MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  /* take care of the nullspace if needed */
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace_u, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(matrix_has_nullspace_u)
  {
    if(A_null_space == PETSC_NULL)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);
    }
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
    ierr = MatNullSpaceRemove(A_null_space, rhs_u, NULL); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverFaces_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);
}




void PoissonSolverFaces::setup_linear_system_w()
{
}




void PoissonSolverFaces::print_partition_u_VTK(const char *file)
{
//#ifdef P4_TO_P8
//  vector<Voronoi3D> voro(faces->num_local[0]);
//#else
//  vector<Voronoi2D> voro(faces->num_local[0]);
//#endif
//  for(p4est_locidx_t u=0; u<faces->num_local[0]; ++u)
//    compute_voronoi_cell_u(u, voro[u]);

#ifdef P4_TO_P8
  Voronoi3D::print_VTK_Format(voro, file, xmin, xmax, ymin, ymax, zmin, zmax, false, false, false);
#else
  Voronoi2D::print_VTK_Format(voro, file);
#endif
}
