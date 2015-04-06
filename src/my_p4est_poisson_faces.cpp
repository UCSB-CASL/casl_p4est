#ifdef P4_TO_P8
#include "my_p8est_poisson_node_base.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
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
    A(NULL), A_null_space(NULL), ksp(NULL)
{
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

  mu = 1;
  diag_add = 0;
}


PoissonSolverFaces::~PoissonSolverFaces()
{
  PetscErrorCode ierr;
  if(A!=NULL) { ierr = MatDestroy(A); CHKERRXX(ierr); }
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



void PoissonSolverFaces::solve(Vec solution_u, Vec solution_v)
{
  solve_u(solution_u);
}




void PoissonSolverFaces::solve_u(Vec solution_u)
{
  setup_linear_system_u();
}



void PoissonSolverFaces::compute_voronoi_cell_u(p4est_locidx_t u_idx, Voronoi2D& voro) const
{
  voro.clear();

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  faces->u2q(u_idx, quad_idx, tree_idx);

  p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
  double dy = dx;

  double x = faces->x_fr_u(u_idx);
  double y = faces->y_fr_u(u_idx);

  double phi_c = interp_phi(x,y);
  /* far in the positive domain */
#ifdef P4_TO_P8
  if(phi_c > 2*MAX(dx_min,dy_min,dz_min))
#else
  if(phi_c > 2*MAX(dx_min,dy_min))
#endif
    return;

  p4est_locidx_t qm_idx=-1, qp_idx=-1;
  p4est_topidx_t tm_idx, tp_idx;
  p4est_quadrant_t qm, qp;
  vector<p4est_quadrant_t> ngbd;
  if(faces->q2u(quad_idx, dir::f_m00)==u_idx)
  {
    qp_idx = quad_idx;
    tp_idx = tree_idx;
    qp = *quad; qp.p.piggy3.local_num = qp_idx;
    ngbd.clear();
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0);
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
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
    if(ngbd.size()>0)
    {
      qp = ngbd[0];
      qp_idx = ngbd[0].p.piggy3.local_num;
      tp_idx = ngbd[0].p.piggy3.which_tree;
    }
  }

  /* check for walls */
  if(qm_idx==-1 && bc_u->wallType(xmin,y)==DIRICHLET) return;
  if(qp_idx==-1 && bc_u->wallType(xmax,y)==DIRICHLET) return;

  /* now gather the neighbor cells to get the potential voronoi neighbors */
  voro.set_Center_Point(x,y);

  if(qm_idx==-1 && bc_u->wallType(xmin,y)==NEUMANN) voro.push(WALL_m00, xmin-dx, y);
  if(qp_idx==-1 && bc_u->wallType(xmax,y)==NEUMANN) voro.push(WALL_p00, xmax+dx, y);
  if( (qm_idx==-1 || is_quad_ymWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ymWall(p4est, tp_idx, &qp)) ) voro.push(WALL_0m0, x, y-dy);
  if( (qm_idx==-1 || is_quad_ypWall(p4est, tm_idx, &qm)) && (qp_idx==-1 || is_quad_ypWall(p4est, tp_idx, &qp)) ) voro.push(WALL_0p0, x, y+dy);

  /* gather neighbor cells */
  ngbd.clear();
  ngbd.push_back(qp);
  ngbd.push_back(qm);
  if(qm_idx!=-1)
  {
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qm_idx, tm_idx,-1, 1);
  }
  if(qp_idx!=-1)
  {
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 0, 1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1,-1);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, qp_idx, tp_idx, 1, 1);
  }

  /* add the faces to the voronoi partition */
  for(unsigned int m=0; m<ngbd.size(); ++m)
  {
    p4est_locidx_t q_tmp = ngbd[m].p.piggy3.local_num;
    p4est_locidx_t u_tmp = faces->q2u(q_tmp, dir::f_m00);
    if(u_tmp!=NO_VELOCITY && u_tmp!=u_idx) voro.push(u_tmp, faces->x_fr_u(u_tmp), faces->y_fr_u(u_tmp));
    u_tmp = faces->q2u(q_tmp, dir::f_p00);
    if(u_tmp!=NO_VELOCITY && u_tmp!=u_idx) voro.push(u_tmp, faces->x_fr_u(u_tmp), faces->y_fr_u(u_tmp));
  }

  voro.construct_Partition();

  vector<Voronoi2DPoint> *points;
  vector<Point2> *partition;
  voro.get_Points(points);
  voro.get_Partition(partition);

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
    voro.set_Level_Set_Values(phi_values, phi_c);
    voro.clip_Interface();
  }
}


void PoissonSolverFaces::preallocate_matrix_u()
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_PoissonSolverFaces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
  proc_offset[0].resize(p4est->mpisize+1);
  proc_offset[0][0] = 0;
  for(int r=1; r<=p4est->mpisize; ++r)
    proc_offset[0][r] = proc_offset[0][r-1] + faces->global_owned_indeps[0][r-1];

  PetscInt num_owned_local  = (PetscInt) faces->num_local[0];
  PetscInt num_owned_global = (PetscInt) proc_offset[0][p4est->mpisize];

  if(A!=NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  for(p4est_locidx_t u_idx=0; u_idx<faces->num_local[0]; ++u_idx)
  {
    Voronoi2D voro;
    compute_voronoi_cell_u(u_idx, voro);

    vector<Voronoi2DPoint> *points;
    voro.get_Points(points);

    for(unsigned int n=0; n<points->size(); ++n)
    {
      if((*points)[n].n>=0)
      {
        if((*points)[n].n<num_owned_local) d_nnz[u_idx]++;
        else                               o_nnz[u_idx]++;
      }
    }
  }

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverFaces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}


void PoissonSolverFaces::setup_linear_system_u()
{
  preallocate_matrix_u();

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_PoissonSolverFaces_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  matrix_has_nullspace_u = true;

  double *rhs_u_p;
  ierr = VecGetArray(rhs_u, &rhs_u_p); CHKERRXX(ierr);

  for(p4est_locidx_t u_idx=0; u_idx<faces->num_local[0]; ++u_idx)
  {
    p4est_gloidx_t u_idx_g = u_idx + proc_offset[0][p4est->mpirank];

    faces->u2q(u_idx, quad_idx, tree_idx);

    p4est_tree_t *tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
    double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dy = dx;

    double x = faces->x_fr_u(u_idx);
    double y = faces->y_fr_u(u_idx);

    double phi_c = interp_phi(x,y);
    /* far in the positive domain */
  #ifdef P4_TO_P8
    if(phi_c > 2*MAX(dx_min,dy_min,dz_min))
  #else
    if(phi_c > 2*MAX(dx_min,dy_min))
  #endif
    {
      ierr = MatSetValue(A, u_idx_g, u_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_u_p[u_idx] = 0;
      continue;
    }

    p4est_locidx_t qm_idx=-1, qp_idx=-1;
    vector<p4est_quadrant_t> ngbd;
    if(faces->q2u(quad_idx, dir::f_m00)==u_idx)
    {
      qp_idx = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx,-1, 0);
      if(ngbd.size()>0) qm_idx = ngbd[0].p.piggy3.local_num;
    }
    else
    {
      qm_idx = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
      if(ngbd.size()>0) qp_idx = ngbd[0].p.piggy3.local_num;
    }


    /* check for walls */
    if(qm_idx==-1 && bc_u->wallType(xmin,y)==DIRICHLET)
    {
      matrix_has_nullspace_u = false;
      ierr = MatSetValue(A, u_idx_g, u_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_u_p[u_idx] = bc_u->wallValue(xmin,y);
      continue;
    }

    if(qp_idx==-1 && bc_u->wallType(xmax,y)==DIRICHLET)
    {
      matrix_has_nullspace_u = false;
      ierr = MatSetValue(A, u_idx_g, u_idx_g, 1, ADD_VALUES); CHKERRXX(ierr);
      rhs_u_p[u_idx] = bc_u->wallValue(xmax,y);
      continue;
    }

    Voronoi2D voro;
    compute_voronoi_cell_u(u_idx, voro);

    /* close to interface and dirichlet => finite differences */
    if(bc_u->interfaceType()==DIRICHLET && voro.is_Interface())
    {

      continue;
    }

    vector<Voronoi2DPoint> *points;
    vector<Point2> *partition;
    voro.get_Points(points);
    voro.get_Partition(partition);

    /* integrally in positive domain */
    if(partition->size()==0)
    {
      ierr = MatSetValue(A, u_idx_g, u_idx_g, 0, ADD_VALUES); CHKERRXX(ierr);
      rhs_u_p[u_idx] = 0;
      continue;
    }

    /* bulk case, finite volume on voronoi cell */
    Point2 pc(x,y);
    for(unsigned int m=0; m<points->size(); ++m)
    {
      PetscInt m_idx_g;
      int k;
      double s, d;
      double x_pert = x;
      if(fabs(x-xmax)<EPS) x_pert = xmax-2*EPS;
      if(fabs(x-xmin)<EPS) x_pert = xmin+2*EPS;

      switch((*points)[m].n)
      {
      /* left wall (note that the dirichlet case has already been done at the beginning of the loop) */
      case WALL_m00:
        k = mod(m-1, points->size());
        s = ((*partition)[m] - (*partition)[k]).norm_L2();

        switch(bc_u->wallType(xmin,y))
        {
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[u_idx] += mu*s*bc_u->wallValue(xmin,y);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

        /* right wall (note that the dirichlet case has already been done at the beginning of the loop) */
      case WALL_p00:
        k = mod(m-1,points->size());
        s = ((*partition)[m] - (*partition)[k]).norm_L2();

        switch(bc_u->wallType(xmax,y))
        {
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[u_idx] += mu*s*bc_u->wallValue(xmax,y);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_0m0:
        k = mod(m-1,points->size());
        s = ((*partition)[m] - (*partition)[k]).norm_L2();
        d = ((*points)[m].p - pc).norm_L2();

        switch(bc_u->wallType(x_pert,ymin))
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2.;
          ierr = MatSetValue(A, u_idx_g, u_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_u_p[u_idx] += mu*s*bc_u->wallValue(x_pert,ymin) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[u_idx] += mu*s*bc_u->wallValue(x_pert,ymin);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case WALL_0p0:
        k = mod(m-1,points->size());
        s = ((*partition)[m] - (*partition)[k]).norm_L2();
        d = ((*points)[m].p - pc).norm_L2();

        switch(bc_u->wallType(x_pert,ymax))
        {
        case DIRICHLET:
          matrix_has_nullspace_u = false;
          d /= 2.;
          ierr = MatSetValue(A, u_idx_g, u_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          rhs_u_p[u_idx] += mu*s*bc_u->wallValue(x_pert,ymax) / d;
          break;
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[u_idx] += mu*s*bc_u->wallValue(x_pert,ymax);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      case INTERFACE:
        k = mod(m-1,points->size());
        s = ((*partition)[m]-(*partition)[k]).norm_L2();

        switch( bc_u->interfaceType())
        {
        /* note that DIRICHLET done with finite differences */
        case NEUMANN:
          /* nothing to do for the matrix */
          rhs_u_p[u_idx] += mu*s*bc_u->interfaceValue(((*points)[m].p.x+x)/2.,((*points)[m].p.y+y)/2.);
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: PoissonSolverFaces: unknown boundary condition type.");
        }
        break;

      default:
        k = mod(m-1,points->size());
        s = ((*partition)[k] - (*partition)[m]).norm_L2();
        d = ((*points)[m].p - pc).norm_L2();

        /* add coefficients in the matrix */
        m_idx_g = (*points)[m].n;
        if(m_idx_g<faces->num_local[0]) m_idx_g += proc_offset[0][p4est->mpirank];
        else                            m_idx_g = (m_idx_g-faces->num_local[0]) + proc_offset[0][faces->nonlocal_ranks[0][m_idx_g-faces->num_local[0]]];

        ierr = MatSetValue(A, u_idx_g, u_idx_g, mu*s/d, ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, u_idx_g, m_idx_g,-mu*s/d, ADD_VALUES); CHKERRXX(ierr);

      }
    }
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);


  /* take care of the nullspace if needed */
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace_u, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if (matrix_has_nullspace_u)
  {
    if (A_null_space == NULL)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);
    }
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
    ierr = MatNullSpaceRemove(A_null_space, rhs_u, NULL); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverFaces_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);
}




void PoissonSolverFaces::print_partition_u_VTK()
{
  vector<Voronoi2D> voro(faces->num_local[0]);
  for(p4est_locidx_t u=0; u<faces->num_local[0]; ++u)
    compute_voronoi_cell_u(u, voro[u]);

  Voronoi2D::print_VTK_Format(voro, "/home/guittet/code/Output/p4est_navier_stokes/voro_u.vtk");
}
