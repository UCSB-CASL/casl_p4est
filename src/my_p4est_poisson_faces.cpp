#ifdef P4_TO_P8
#include "my_p8est_poisson_faces.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/cube2.h>
#include <src/cube3.h>
#else
#include "my_p4est_poisson_faces.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolation_faces.h>
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
    phi(NULL), apply_hodge_second_derivative_if_neumann(false), bc(NULL), bc_hodge(NULL), dxyz_hodge(NULL)
{
  PetscErrorCode ierr;

  p4est_topidx_t vtx_0_max      = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + P4EST_CHILDREN - 1];
  p4est_topidx_t vtx_0_min      = p4est->connectivity->tree_to_vertex[0*P4EST_CHILDREN + 0];
  /* set up the KSP solvers and other parameters*/
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    A[dim]            = NULL;
    A_null_space[dim] = NULL;
    null_space[dim]   = NULL;
    ierr = KSPCreate(p4est->mpicomm, &ksp[dim]); CHKERRXX(ierr);
    ierr = KSPSetTolerances(ksp[dim], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
    pc_is_set_from_options[dim]   = false;
    ksp_is_set_from_options[dim]  = false;
    matrix_is_ready[dim]          = false;
    only_diag_is_modified[dim]    = false;
    matrix_has_nullspace[dim]     = false;
    current_diag[dim]             = 0.0;
    desired_diag[dim]             = 0.0;
    tree_dimensions[dim]          = p4est->connectivity->vertices[3*vtx_0_max + dim] - p4est->connectivity->vertices[3*vtx_0_min + dim] ;
    periodic[dim]                 = is_periodic(p4est, dim);
  }

  xyz_min_max(p4est, xyz_min, xyz_max);
  dxyz_min(p4est, dxyz);

  compute_partition_on_the_fly = false;
  mu = 1.0;
}

my_p4est_poisson_faces_t::~my_p4est_poisson_faces_t()
{
  PetscErrorCode ierr;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    if(A[dim] != NULL)              { ierr = MatDestroy(A[dim]);                     CHKERRXX(ierr); }
    if(null_space[dim] != NULL)     { ierr = VecDestroy(null_space[dim]);            CHKERRXX(ierr); }
    if(A_null_space[dim]  !=  NULL) { ierr = MatNullSpaceDestroy(A_null_space[dim]); CHKERRXX(ierr); }
    if(ksp[dim] != NULL)            { ierr = KSPDestroy(ksp[dim]);                   CHKERRXX(ierr); }
  }
}

void my_p4est_poisson_faces_t::setup_linear_solver(int dim, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  PetscErrorCode ierr;

  P4EST_ASSERT(ksp[dim] != NULL);
  /* set ksp type */
  KSPType ksp_type_as_such;
  ierr = KSPGetType(ksp[dim], &ksp_type_as_such); CHKERRXX(ierr);
  if(ksp_type != ksp_type_as_such){
    ierr = KSPSetType(ksp[dim], ksp_type); CHKERRXX(ierr); }

  PetscBool ksp_initial_guess;
  ierr = KSPGetInitialGuessNonzero(ksp[dim], &ksp_initial_guess); CHKERRXX(ierr);
  if (ksp_initial_guess != ((PetscBool) use_nonzero_initial_guess)){
    ierr = KSPSetInitialGuessNonzero(ksp[dim], ((PetscBool) use_nonzero_initial_guess)); CHKERRXX(ierr); }
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
       * Use zero for the best convergence. However, if you have memory problems, use greater than zero to save some memory.
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

void my_p4est_poisson_faces_t::set_phi(Vec phi_)
{
  this->phi = phi_;
  interp_phi.set_input(phi, linear);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    matrix_is_ready[dim]        = false;
    only_diag_is_modified[dim]  = false;
  }
}

void my_p4est_poisson_faces_t::set_rhs(Vec *rhs)
{
  this->rhs = rhs;
}

void my_p4est_poisson_faces_t::set_diagonal(double add)
{
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    desired_diag[dim]           = add;
    if(!current_diag_is_as_desired(dim))
    {
      // actual modification of diag, do not change the flag values otherwise, especially not of matrix_is_ready
      only_diag_is_modified[dim]  = matrix_is_ready[dim];
      matrix_is_ready[dim]        = false;
    }
  }
}


void my_p4est_poisson_faces_t::set_mu(double mu_)
{
  P4EST_ASSERT(mu > 0.0);
  if(fabs(this->mu - mu_) > EPS*MAX(this->mu, mu_)) // actual modification of mu
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      only_diag_is_modified[dim]  = false;
      matrix_is_ready[dim]        = false;
    }
  this->mu = mu_;
}

void my_p4est_poisson_faces_t::set_bc(const BoundaryConditionsDIM *bc_, Vec *dxyz_hodge_, Vec *face_is_well_defined_, const BoundaryConditionsDIM *bc_hodge_)
{
  this->bc = bc_;
  this->bc_hodge = bc_hodge_;
  this->dxyz_hodge = dxyz_hodge_;
  this->face_is_well_defined = face_is_well_defined_;
  // change of bc and/or face_is_well_defined --> a full reset is needed!
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    only_diag_is_modified[dim]  = false;
    matrix_is_ready[dim]        = false;
  }
}


void my_p4est_poisson_faces_t::set_compute_partition_on_the_fly(bool do_it_on_the_fly)
{
  this->compute_partition_on_the_fly = do_it_on_the_fly;
}


void my_p4est_poisson_faces_t::solve(Vec *solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  PetscErrorCode ierr;

#ifdef CASL_THROWS
  if(bc == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");
  for(unsigned char dir=0; dir < P4EST_DIM; ++dir)
  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution[dir], &sol_size); CHKERRXX(ierr);
    if (sol_size != faces->num_local[dir]){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as the number of faces"
          << "solution.local_size = " << sol_size << " faces->num_local[" << dir << "] = " << faces->num_local[dir] << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_solve, A, rhs, solution, 0); CHKERRXX(ierr);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    /* assemble the linear system if required, and initialize the Krylov solver and its preconditioner based on that*/
    setup_linear_system(dir);

    setup_linear_solver(dir, use_nonzero_initial_guess, ksp_type, pc_type);

    /* solve the system */
    ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_KSPSolve, ksp, rhs[dir], solution[dir], 0); CHKERRXX(ierr);

    ierr = KSPSolve(ksp[dir], rhs[dir], solution[dir]); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_KSPSolve, ksp, rhs[dir], solution[dir], 0); CHKERRXX(ierr);

    /* update ghosts */
    ierr = VecGhostUpdateBegin(solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_solve, A, rhs, solution, 0); CHKERRXX(ierr);
}

#ifndef P4_TO_P8
void my_p4est_poisson_faces_t::clip_voro_cell_by_interface(Voronoi2D &voro_cell, const p4est_locidx_t &f_idx, const unsigned char &dir)
{
  const vector<ngbd2Dseed> *points;
  vector<Point2> *partition;

  voro_cell.get_neighbor_seeds(points);
  voro_cell.get_partition(partition);

  /* first clip the voronoi partition at the boundary of the domain */
  const unsigned char other_dir = (dir + 1)%P4EST_DIM;
  for(size_t m = 0; m < points->size(); m++)
    if((*points)[m].n < 0 &&  (-1 - (*points)[m].n)/2 == dir) // there is a "wall" point added because of a parallel wall
    {
      const unsigned char which_wall = (-1 - (*points)[m].n)%2;
      for(char i = -1; i < 1; ++i)
      {
        size_t k                        = mod(m + i                 , partition->size());
        size_t tmp                      = mod(k + (i == -1 ? -1 : 1), partition->size());
        Point2 segment                  = (*partition)[tmp] - (*partition)[k];
        double lambda                   = ((which_wall == 0 ? xyz_min[dir] : xyz_max[dir]) - (*partition)[k].xyz(dir))/segment.xyz(dir);
        (*partition)[k].xyz(dir)        = (which_wall == 0 ? xyz_min[dir] : xyz_max[dir]);
        (*partition)[k].xyz(other_dir)  = (*partition)[k].xyz(other_dir) + lambda*segment.xyz(other_dir);
      }
    }

  /* clip the partition by the irregular interface */
  vector<double> phi_values(partition->size());
  bool is_pos = false;
  for(size_t m = 0; m < partition->size(); m++)
  {
    phi_values[m] = interp_phi((*partition)[m].x, (*partition)[m].y);
    is_pos = is_pos || phi_values[m] > 0.0;
  }

  /* clip the voronoi partition with the interface */
  if(is_pos)
  {
    double xyz_face[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz_face);
    double phi_c = interp_phi(xyz_face);
    voro_cell.set_level_set_values(phi_values, phi_c);
    voro_cell.clip_interface();
  }

  voro_cell.compute_volume();
}
#endif

void my_p4est_poisson_faces_t::preallocate_matrix(const unsigned char &dir)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_local  = (PetscInt) faces->num_local[dir];
  PetscInt num_owned_global = (PetscInt) faces->proc_offset[dir][p4est->mpisize];

  if(A[dir] != NULL){
    ierr = MatDestroy(A[dir]); CHKERRXX(ierr); }

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A[dir]); CHKERRXX(ierr);
  ierr = MatSetType(A[dir], MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A[dir], num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A[dir]); CHKERRXX(ierr);

  vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  if(compute_partition_on_the_fly) voro[dir].resize(1);
  else                           { voro[dir].clear(); voro[dir].resize(faces->num_local[dir]); }

  const PetscScalar *face_is_well_defined_p;
  ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx)
  {
    double xyz[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz);
    double phi_c = interp_phi(xyz);

    if(bc[dir].interfaceType(xyz) == NOINTERFACE ||
       (bc[dir].interfaceType(xyz) == DIRICHLET && phi_c < 0.5*MIN(DIM(dxyz[0],dxyz[1],dxyz[2]))) ||
       (bc[dir].interfaceType(xyz) == NEUMANN   && phi_c < 2.0*MAX(DIM(dxyz[0],dxyz[1],dxyz[2]))))
    {
      Voronoi_DIM &voro_cell = (compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx]);
      compute_voronoi_cell(voro_cell, faces, f_idx, dir, bc, face_is_well_defined_p);

      const vector<ngbdDIMseed > *points;
      voro_cell.get_neighbor_seeds(points);

      for(size_t n = 0; n < points->size(); ++n)
        if((*points)[n].n >= 0)
        {
          if((*points)[n].n < num_owned_local)  d_nnz[f_idx]++;
          else                                  o_nnz[f_idx]++;
        }

#ifndef P4_TO_P8
      /* in 2D, clip the partition by the interface and by the walls of the domain */
      try {
        clip_voro_cell_by_interface(voro_cell, f_idx, dir);
      } catch (std::exception e) {
        // [FIXME]: I found this issue but I have other urgent things to do for now... Raphael
        std::cout<<"Face index is : "<<f_idx<< " in direction "<<dir << " , x = "<<faces->x_fr_f(f_idx,dir) <<", y = "<< faces->y_fr_f(f_idx,dir) << " on process "<<p4est->mpirank<<std::endl;
        throw std::runtime_error("Error when clipping voronoi cell in 2D... consider using an aspect ratio closer to 1");
      }
#endif
    }
  }
  ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A[dir], 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A[dir], 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_faces_t::setup_linear_system(const unsigned char &dir)
{ //E: Sets up the linear system for a given direction --
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_poisson_faces_setup_linear_system, A, rhs[dir], 0, 0); CHKERRXX(ierr);

  // check that the current "diagonal" is as desired if the matrix is ready to go...
  P4EST_ASSERT(!matrix_is_ready[dir] || current_diag_is_as_desired(dir));
  if(!only_diag_is_modified[dir] && !matrix_is_ready[dir])
  {
    reset_current_diag(dir);
    /* preallocate the matrix and compute the voronoi partition */
    preallocate_matrix(dir);
  }

  matrix_has_nullspace[dir] = true;

  const PetscScalar *face_is_well_defined_p;
  double *rhs_p, *null_space_p;
  if(null_space[dir] != NULL) {
    ierr = VecDestroy(null_space[dir]); CHKERRXX(ierr); }
  ierr = VecDuplicate(rhs[dir], &null_space[dir]); CHKERRXX(ierr);
  ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(null_space[dir], &null_space_p); CHKERRXX(ierr);

  std::vector<double> bc_coeffs;
  std::vector<p4est_locidx_t> bc_index; bc_index.resize(0);
  my_p4est_interpolation_faces_t interp_dxyz_hodge(ngbd_n, faces);
  interp_dxyz_hodge.set_input(dxyz_hodge[dir], dir, 1, face_is_well_defined[dir]);

  // E: Loop over all faces in the given direction (ie. all x faces on local processor, or all y faces on local processor)
  for(p4est_locidx_t f_idx = 0; f_idx < faces->num_local[dir]; ++f_idx)
  {
    null_space_p[f_idx] = 1;
    p4est_gloidx_t f_idx_g = faces->global_index(f_idx, dir);

    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    faces->f2q(f_idx, dir, quad_idx, tree_idx); //E: Grabs the local index of quadrant that owns the face (in the given direction) (since faces are accounted for by their index and direction)

#ifdef CASL_THROWS
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
#endif

    // E: Get the coordinates of the center of the given face
    double xyz[P4EST_DIM]; faces->xyz_fr_f(f_idx, dir, xyz);

    //E: Get LSF value interpolated to the face center
    double phi_c = interp_phi(xyz);
    /* far in the positive domain, E: AKA in the part of the domain that we are not concerned in getting a solution*/
    if(!face_is_well_defined_p[f_idx])
    {
      if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
      rhs_p[f_idx] = 0;
      null_space_p[f_idx] = 0;
      continue;
    }

    p4est_quadrant_t qm, qp;
    faces->find_quads_touching_face(f_idx, dir, qm, qp);

    /* -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
    /* CHECK FOR WALLS -- Apply Dirichlet Wall Conditions if they are specified on the given wall*/
    /* -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

    // E: If the qm_idx or the qp_idx remain unchanged from their initialized values of (-1), that implies there is no neighbor in the given direction, thus there is a wall
    // E: If there is a wall, apply appropriate diagonal term to A matrix, and add BC value to the RHS
    if(qm.p.piggy3.local_num == -1 && bc[dir].wallType(xyz) == DIRICHLET)
    {
      matrix_has_nullspace[dir] = false;
      if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
      rhs_p[f_idx] = bc[dir].wallValue(xyz) + interp_dxyz_hodge(xyz);
      continue;
    }

    if(qp.p.piggy3.local_num == -1 && bc[dir].wallType(xyz) == DIRICHLET)
    {
      matrix_has_nullspace[dir] = false;
      if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); }// needs to be done only if fully reset
      rhs_p[f_idx] = bc[dir].wallValue(xyz) + interp_dxyz_hodge(xyz);
      continue;
    }

    bool wall[P4EST_FACES];
    //E: Loop over each direction and record for both plus and minus faces whether or not there is a wall present
    for(unsigned char d = 0; d < P4EST_DIM; ++d)
    {
      unsigned char f_m = 2*d;
      unsigned char f_p = 2*d+1;
      if(d == dir)
      {
        wall[f_m] = qm.p.piggy3.local_num == -1;
        wall[f_p] = qp.p.piggy3.local_num == -1;
      }
      else
      {
        wall[f_m] = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qm.p.piggy3.which_tree, &qm, f_m)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qp.p.piggy3.which_tree, &qp, f_m));
        wall[f_p] = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qm.p.piggy3.which_tree, &qm, f_p)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est, qp.p.piggy3.which_tree, &qp, f_p));
      }
    }

    /*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     * Compute the Voronoi Cell for the given face
     * Then store the Voronoi partition and points
     -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
    Voronoi_DIM &voro_tmp = compute_partition_on_the_fly ? voro[dir][0] : voro[dir][f_idx];

    if(compute_partition_on_the_fly)
    {
      compute_voronoi_cell(voro_tmp, faces, f_idx, dir, bc, face_is_well_defined_p);
#ifndef P4_TO_P8
      clip_voro_cell_by_interface(voro_tmp, f_idx, dir);
#endif
    }

    const vector<ngbdDIMseed> *points;
#ifndef P4_TO_P8
    vector<Point2> *partition;

    // E: Store Voronoi partition and points:
    voro_tmp.get_partition(partition);
#endif
    voro_tmp.get_neighbor_seeds(points);

    /* -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     * Apply Dirichelt Interfacial Conditions  -- close to interface and dirichlet => finite differences
     ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
    if(bc[dir].interfaceType(xyz) == DIRICHLET && phi_c > -2.0*MAX(DIM(dxyz[0], dxyz[1], dxyz[2])))
    {
      if(fabs(phi_c) < EPS*MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))) // E: If interface is directly at the face (with dimensionally-consistent check)
      {
        if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
        rhs_p[f_idx] = bc[dir].interfaceValue(xyz);        // E: Add Interface bc value to RHS
        interp_dxyz_hodge.add_point(bc_index.size(), xyz); // E: Add xyz location of face to list of points to interpolate grad(hodge) at
        bc_index.push_back(f_idx);                         // E: Add face index to list of bc index
        bc_coeffs.push_back(1);                            // E: Add 1 to lsit of bc coefficients
        matrix_has_nullspace[dir] = false;
        continue;
      }

      if(phi_c > 0.0)
      {
        if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
        rhs_p[f_idx] = 0;
        null_space_p[f_idx] = 0;
        continue;
      }

      // Now sample the values of the LSF in the area surrounding the face --> Used to detect the presence of the interface in the area around the current face
      // phi[ff] == value of the LSF at the face neighbor in (oriented) direction ff, assuming uniform local neighborhood
      double phi[P4EST_FACES];
      // E: Check for presence of interface via a sign change in LSF and by checking the Voronoi points
      bool is_interface = false;
      for (unsigned char ff = 0; ff < P4EST_FACES; ++ff) {
        double xyz_ngbd[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
        xyz_ngbd[ff/2] += (ff%2 == 1 ? +dxyz[ff/2] : -dxyz[ff/2]);
        phi[ff] = interp_phi(xyz_ngbd);
        is_interface = is_interface || phi[ff] > 0.0;
      }

#ifndef P4_TO_P8
      for (size_t i=0; i < points->size() && !is_interface; ++i)
        is_interface = is_interface || (*points)[i].n == INTERFACE;
#endif

      if(is_interface)
      {
        bool face_is_across[P4EST_FACES];
        double stencil_arm[P4EST_FACES], val_interface[P4EST_FACES];
        for (unsigned char ff = 0; ff < P4EST_FACES; ++ff)
        {
          face_is_across[ff] = !wall[ff] && phi[ff]*phi_c <= 0.0;
          matrix_has_nullspace[dir] = matrix_has_nullspace[dir] && !face_is_across[ff]; // interface point with Dirichlet boundary condition --> no nullspace
          stencil_arm[ff] = (wall[ff] ? (ff%2 ==1 ? xyz_max[ff/2] - xyz[ff/2] : xyz[ff/2] - xyz_min[ff/2]) : dxyz[ff/2]);
          val_interface[ff] = 0.0; // irrelevant, but undefined if no interface, otherwise (should not be a problem though)
          if(face_is_across[ff]) {
            stencil_arm[ff] = interface_Location(0, stencil_arm[ff], phi_c, phi[ff]);
            stencil_arm[ff] = MAX(EPS*tree_dimensions[ff/2], stencil_arm[ff]);
            double xyz_intfc[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
            xyz_intfc[ff/2]  += (ff%2 == 1 ? +stencil_arm[ff] : -stencil_arm[ff]);
            val_interface[ff] = bc[dir].interfaceValue(xyz_intfc);
          }
        }

        // if the face is a wall-face and the solver reaches this line, it MUST be a non-DIRICHLET wall boundary condition...
        // --> assumed to be NEUMANN...
        if(wall[2*dir])     stencil_arm[2*dir]     = stencil_arm[2*dir +1 ];
        if(wall[2*dir + 1]) stencil_arm[2*dir + 1] = stencil_arm[2*dir];

        double desired_coeff[P4EST_FACES], current_coeff[P4EST_FACES];
        double desired_scaling = desired_diag[dir];
        double current_scaling = (!only_diag_is_modified[dir] && !matrix_is_ready[dir] ? 0.0: current_diag[dir]);
        for(unsigned char ff = 0; ff < P4EST_FACES; ++ff)
        {
          unsigned char fff = ff%2 == 0 ? ff + 1 : ff - 1;
          desired_coeff[ff] = current_coeff[ff] = -2*mu/stencil_arm[ff]/(stencil_arm[ff]+stencil_arm[fff]);
          desired_scaling -= desired_coeff[ff];
          current_scaling -= current_coeff[ff];
        }

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        for(unsigned char ff = 0; ff < P4EST_FACES; ++ff)
        {
          desired_coeff[ff] /= desired_scaling;
          if(only_diag_is_modified[dir] || matrix_is_ready[dir])
            current_coeff[ff] /= current_scaling;
          else
            current_coeff[ff] = 0.0;
        }

        rhs_p[f_idx] /= desired_scaling;

        if(desired_diag[dir] > 0.0) matrix_has_nullspace[dir] = false;

        //---------------------------------------------------------------------
        // insert the coefficients in the matrix
        //---------------------------------------------------------------------
        if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs to be done if full reset only, the (raw) diag term will always be 1.0 for this...

        for(unsigned char ff = 0; ff < P4EST_FACES; ++ff)
        {
          /* this is the cartesian direction for which the linear system is assembled.
           * the treatment is different, for example x-velocity can be ON the x-walls
           */
          if(ff/2 == dir) // E: If the face is in the direction we are currently solving for
          {
            if(wall[ff]) // E: If the face f of the quad owning f_idx is on the wall
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
               * NB: d_[ff] and  d_[ff_op] are made equal beforehand in this case (mirror symmetry, see above + comment)
               */
              unsigned char ff_op = ff%2 == 0 ? ff + 1 : ff - 1; /* the opposite direction. if ff = f_m00, then ff_op = f_p00 */
              if(!face_is_across[ff_op]) // --> Neumann wall face but opposite face is not across the interface
              {
                p4est_locidx_t face_opp_loc_idx = (ff%2 == 0? faces->q2f(qp.p.piggy3.local_num, 2*dir + 1) : faces->q2f(qm.p.piggy3.local_num, 2*dir));
                if(!matrix_is_ready[dir])
                {
                  p4est_gloidx_t f_tmp_g = faces->global_index(face_opp_loc_idx, dir);
                  ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g, (desired_coeff[ff] - current_coeff[ff]), ADD_VALUES); CHKERRXX(ierr);
                }
                if(apply_hodge_second_derivative_if_neumann)
                {
                  double xyz_op[P4EST_DIM]; faces->xyz_fr_f(face_opp_loc_idx, dir, xyz_op);
                  interp_dxyz_hodge.add_point(bc_index.size(), xyz_op);
                  bc_index.push_back(f_idx);
                  bc_coeffs.push_back(-desired_coeff[ff]*(stencil_arm[ff] + stencil_arm[ff_op])*(-1.0/stencil_arm[ff_op]));
                }
              }
              else // --> Neumann wall face and opposite face is across the Dirichlet interface
              {
                rhs_p[f_idx] -= desired_coeff[ff] * val_interface[ff_op];
                double xyz_op[P4EST_DIM];
                for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
                  xyz_op[dd] = xyz[dd] + (ff_op/2 == dd ? (ff_op%2 == 1 ? +stencil_arm[ff_op] : -stencil_arm[ff_op]) : 0.0);
                interp_dxyz_hodge.add_point(bc_index.size(), xyz_op);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(-desired_coeff[ff]*(1.0 + (stencil_arm[ff] + stencil_arm[ff_op])*(apply_hodge_second_derivative_if_neumann ? -1.0/stencil_arm[ff_op]: 0.0)));
              }
              rhs_p[f_idx] -= desired_coeff[ff] * (stencil_arm[ff] + stencil_arm[ff_op]) * (bc[dir].wallValue(xyz) + (apply_hodge_second_derivative_if_neumann? interp_dxyz_hodge(xyz)/stencil_arm[ff_op]: 0.0));
              // modified by Raphael Egan, the former discretization of Neumann wall boundary conditions was dimensionally inconsistent
              // --> introduced a flag 'apply_hodge_second_derivative_if_neumann' to enforce an approximation of what is required
              // if apply_hodge_second_derivative_if_neumann == false, the correction is disregarded!
            }
            else if(!face_is_across[ff]) // regular stuff, nothing particular, no interface, no wall
            {
              if(!matrix_is_ready[dir])
              {
                p4est_gloidx_t f_tmp_g = ff%2 == 0 ? faces->global_index(faces->q2f(qm.p.piggy3.local_num, 2*dir), dir)
                                                   : faces->global_index(faces->q2f(qp.p.piggy3.local_num, 2*dir + 1), dir);
                ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g, (desired_coeff[ff] - current_coeff[ff]), ADD_VALUES); CHKERRXX(ierr);
              }
            }
            else // not a wall face but neighbor face is across the Dirichlet interface
            {
              rhs_p[f_idx] -= desired_coeff[ff]*val_interface[ff];
              double xyz_[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
              xyz_[ff/2] += (ff%2 == 1 ? stencil_arm[ff] : -stencil_arm[ff]);
              interp_dxyz_hodge.add_point(bc_index.size(),xyz_);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(-desired_coeff[ff]);
            }
          }
          else /* if the direction ff is not the direction in which the linear system is being assembled */
          {
            if(wall[ff]) // wall neighbor
            {
              double w_xyz[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
              w_xyz[ff/2] = (ff%2 == 0 ? xyz_min[ff/2] : xyz_max[ff/2]);

              BoundaryConditionType bc_w_type = bc[dir].wallType(w_xyz);
              double bc_w_value = bc[dir].wallValue(w_xyz) + (bc_w_type == DIRICHLET? interp_dxyz_hodge(w_xyz) : (apply_hodge_second_derivative_if_neumann ? (interp_dxyz_hodge(w_xyz) - interp_dxyz_hodge(xyz))/stencil_arm[ff] : 0.0));

              switch(bc_w_type)
              {
              case DIRICHLET:
                 rhs_p[f_idx] -= desired_coeff[ff]*bc_w_value;
                 break;
              case NEUMANN:
                if(!matrix_is_ready[dir]) {
                  ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, (desired_coeff[ff]-current_coeff[ff]), ADD_VALUES); CHKERRXX(ierr); } // should be correct like that (Raphael)
                rhs_p[f_idx] -= desired_coeff[ff] * stencil_arm[ff] * bc_w_value;
                break;
              default:
                throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: invalid boundary condition.");
              }
            }
            else if(!face_is_across[ff]) // regular neighbor
            {
              if(!matrix_is_ready[dir])
              {
                set_of_neighboring_quadrants ngbd; ngbd.clear();
                ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, ff);
#ifdef CASL_THROWS
                P4EST_ASSERT(ngbd.size() == 1 && ngbd.begin()->level == quad->level);
                // throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: the grid is not uniform close to the interface.");
#endif
                p4est_gloidx_t f_tmp_g;
                if(quad_idx == qm.p.piggy3.local_num) f_tmp_g = faces->global_index(faces->q2f(ngbd.begin()->p.piggy3.local_num, 2*dir + 1), dir);
                else                                  f_tmp_g = faces->global_index(faces->q2f(ngbd.begin()->p.piggy3.local_num, 2*dir), dir);
                ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g, (desired_coeff[ff] - current_coeff[ff]), ADD_VALUES); CHKERRXX(ierr);
              }
            }
            else // not a wall face but neighbor face is across the Dirichlet interface
            {
              rhs_p[f_idx] -= desired_coeff[ff]*val_interface[ff];
              double xyz_[P4EST_DIM] = {DIM(xyz[0], xyz[1],xyz[2])};
              xyz_[ff/2] += (ff%2 == 1 ? stencil_arm[ff] : -stencil_arm[ff]);
              interp_dxyz_hodge.add_point(bc_index.size(),xyz_);
              bc_index.push_back(f_idx);
              bc_coeffs.push_back(-desired_coeff[ff]);
            }
          }
        } /* end of going through P4EST_FACES to assemble the system with finite differences */
        continue;
      }
    } // End of if bcType.interface == DIRICHLET and LSF relatively close to 0

#ifdef P4_TO_P8
    /* --------------------------------------------------------------------------------------------------------------------------
     * If close to the interface and Neumann bc, do finite volume by hand
     * since cutting the voronoi cells in 3D with voro++ to have a nice level set is a nightmare ...
     * In 2D, cutting the partition is easy ... so Neumann interface is handled in the bulk case
     --------------------------------------------------------------------------------------------------------------------------*/
    if(bc[dir].interfaceType(xyz) == NEUMANN && phi_c > -2.0*MAX(dxyz[0], dxyz[1], dxyz[2]))
    {
      Cube3 c3;
      OctValue op;
      OctValue iv;

      for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
        c3.xyz_mmm[dd] = (dir == dd && qm.p.piggy3.local_num == -1 ? xyz[dd] : xyz[dd] - dxyz[dd]/2);
        c3.xyz_ppp[dd] = (dir == dd && qp.p.piggy3.local_num == -1 ? xyz[dd] : xyz[dd] + dxyz[dd]/2);
      }

      double hodge_correction = 0.0;
      if(apply_hodge_second_derivative_if_neumann)
      {
        double n_comp[P4EST_DIM];
        double n_norm = 0.0;
        n_comp[0] = (interp_phi(c3.xyz_ppp[0],  xyz[1], xyz[2]) - interp_phi(c3.xyz_mmm[0],   xyz[1], xyz[2]))/(c3.xyz_ppp[0] - c3.xyz_mmm[0]); n_norm += SQR(n_comp[0]);
        n_comp[1] = (interp_phi(xyz[0], c3.xyz_ppp[1],  xyz[2]) - interp_phi(xyz[0],  c3.xyz_mmm[1],  xyz[2]))/(c3.xyz_ppp[1] - c3.xyz_mmm[1]); n_norm += SQR(n_comp[1]);
        n_comp[2] = (interp_phi(xyz[0], xyz[1], c3.xyz_ppp[2])  - interp_phi(xyz[0],  xyz[1], c3.xyz_mmm[2] ))/(c3.xyz_ppp[2] - c3.xyz_mmm[2]); n_norm += SQR(n_comp[2]);
        n_norm = sqrt(n_norm);
        P4EST_ASSERT(n_norm > EPS);
        n_comp[0] /= n_norm;
        hodge_correction += n_comp[0]*(interp_dxyz_hodge(c3.xyz_ppp[0],  xyz[1], xyz[2])  - interp_dxyz_hodge(c3.xyz_mmm[0],   xyz[1], xyz[2]))/(c3.xyz_ppp[0] - c3.xyz_mmm[0]);
        n_comp[1] /= n_norm;
        hodge_correction += n_comp[1]*(interp_dxyz_hodge(xyz[0], c3.xyz_ppp[1],  xyz[2])  - interp_dxyz_hodge(xyz[0],  c3.xyz_mmm[1],  xyz[2]))/(c3.xyz_ppp[1] - c3.xyz_mmm[1]);
        n_comp[2] /= n_norm;
        hodge_correction += n_comp[2]*(interp_dxyz_hodge(xyz[0], xyz[1], c3.xyz_ppp[2])   - interp_dxyz_hodge(xyz[0],  xyz[1], c3.xyz_mmm[2])) /(c3.xyz_ppp[2] - c3.xyz_mmm[2]);
      }

      bool is_pos = false;
      bool is_neg = false;

      for (unsigned char xd = 0; xd < 2; ++xd)
        for (unsigned char yd = 0; yd < 2; ++yd)
          for (unsigned char zd = 0; zd < 2; ++zd)
          {
            double xyz_eval[3] = {(xd == 0 ? c3.xyz_mmm[0] : c3.xyz_ppp[0]), (yd == 0 ? c3.xyz_mmm[1] : c3.xyz_ppp[1]), (zd == 0 ? c3.xyz_mmm[2] : c3.xyz_ppp[2])};
            iv.val[4*xd + 2*yd + zd] = bc[dir].interfaceValue(xyz_eval) + hodge_correction;
            op.val[4*xd + 2*yd + zd] = interp_phi(xyz_eval);
            is_pos = is_pos || op.val[4*xd + 2*yd + zd] > 0.0;
            is_neg = is_neg || op.val[4*xd + 2*yd + zd] <= 0.0;
          }
      const double volume = c3.volume_In_Negative_Domain(op);

      /* entirely in the positive domain */
      if(!is_neg) // should be !face_is_well_defined_p[f_idx], though...
      {
        if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
        rhs_p[f_idx] = 0;
        null_space_p[f_idx] = 0;
        continue;
      }

      if(is_pos)
      {
        if(!matrix_is_ready[dir]) {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, volume*(desired_diag[dir] - current_diag[dir]), ADD_VALUES); CHKERRXX(ierr); }

        if(desired_diag[dir] > 0.0) matrix_has_nullspace[dir] = false;
        rhs_p[f_idx] *= volume;
        rhs_p[f_idx] += mu * c3.integrate_Over_Interface(iv, op);

        Cube2 cube_face;
        QuadValue ls_values_on_cube_face;

        const unsigned char increment[3] = {4, 2, 1};
        for (unsigned char ngbd_dir = 0; ngbd_dir < P4EST_FACES; ++ngbd_dir)
        {
          const unsigned char first_dir   = (ngbd_dir/2 == dir::x ? dir::y : dir::x);
          const unsigned char second_dir  = (ngbd_dir/2 == dir::z ? dir::y : dir::z);
          cube_face.xyz_mmm[0] = c3.xyz_mmm[first_dir];  cube_face.xyz_ppp[0] = c3.xyz_ppp[first_dir];
          cube_face.xyz_mmm[1] = c3.xyz_mmm[second_dir]; cube_face.xyz_ppp[1] = c3.xyz_ppp[second_dir];

          const unsigned char offset = (ngbd_dir%2 == 1 ? increment[ngbd_dir/2] : 0);
          for (unsigned char d1 = 0; d1 < 2; ++d1)
            for (unsigned char d2 = 0; d2 < 2; ++d2)
              ls_values_on_cube_face.val[2*d1 + d2] = op.val[offset + d1*increment[first_dir] + d2*increment[second_dir]];

          const double exchange_surface = cube_face.area_In_Negative_Domain(ls_values_on_cube_face);

          if(exchange_surface > EPS*dxyz[first_dir]*dxyz[second_dir]) // --> non-zero interaction area between control volumes
          {
            if(wall[ngbd_dir]) // neighbor is a wall
            {
              double w_xyz[P4EST_DIM] = {xyz[0], xyz[1], xyz[2]};
              w_xyz[ngbd_dir/2] = (ngbd_dir%2 == 1 ? xyz_max[ngbd_dir/2] : xyz_min[ngbd_dir/2]);
              const double wall_distance = (ngbd_dir%2 == 0 ? xyz[ngbd_dir/2] - w_xyz[ngbd_dir/2] : w_xyz[ngbd_dir/2] - xyz[ngbd_dir/2]);
              if(bc[dir].wallType(w_xyz) == DIRICHLET) // Dirichlet wall neighbor
              {
                rhs_p[f_idx] += mu * exchange_surface * bc[dir].wallValue(w_xyz)/wall_distance; // cannot be division by 0 as it would mean that the face is ON the wall --> treated earlier...
                interp_dxyz_hodge.add_point(bc_index.size(),w_xyz);
                bc_index.push_back(f_idx);
                bc_coeffs.push_back(mu * exchange_surface/wall_distance);
                if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
                  ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*exchange_surface/wall_distance, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
              }
              else // Neumann wall neighbor
              {
                rhs_p[f_idx] += mu * exchange_surface * (bc[dir].wallValue(w_xyz) + (apply_hodge_second_derivative_if_neumann ? -interp_dxyz_hodge(xyz)/wall_distance : 0.0));
                if(apply_hodge_second_derivative_if_neumann)
                {
                  interp_dxyz_hodge.add_point(bc_index.size(), w_xyz);
                  bc_index.push_back(f_idx);
                  bc_coeffs.push_back(mu * exchange_surface/ wall_distance);
                }
              }
            }
            else // neighbor is not a wall
            {
              if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) // needs only to be done if fully reset
              {
                p4est_gloidx_t f_tmp_g;
                // find the index of the neighbor
                if(dir == ngbd_dir/2)
                  f_tmp_g = faces->global_index(faces->q2f((ngbd_dir%2 == 1 ? qp.p.piggy3.local_num : qm.p.piggy3.local_num), ngbd_dir), dir);
                else
                {
                  set_of_neighboring_quadrants ngbd; ngbd.clear();
                  char search[P4EST_DIM] = {0, 0, 0}; search[ngbd_dir/2] = (ngbd_dir%2 == 1 ? 1 : -1);
                  ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, search[0], search[1], search[2]);
                  P4EST_ASSERT(ngbd.size() == 1 && ngbd.begin()->level == quad->level); // must have a uniform tesselation with maximum refinement
                  if(quad_idx == qm.p.piggy3.local_num)
                  {
                    P4EST_ASSERT(faces->q2f(ngbd.begin()->p.piggy3.local_num, 2*dir + 1) != NO_VELOCITY);
                    f_tmp_g = faces->global_index(faces->q2f(ngbd.begin()->p.piggy3.local_num, 2*dir + 1), dir);
                  }
                  else
                  {
                    P4EST_ASSERT(faces->q2f(ngbd.begin()->p.piggy3.local_num, 2*dir) != NO_VELOCITY);
                    f_tmp_g = faces->global_index(faces->q2f(ngbd.begin()->p.piggy3.local_num, 2*dir), dir);
                  }
                }
                ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*exchange_surface/dxyz[ngbd_dir/2], ADD_VALUES); CHKERRXX(ierr);
                ierr = MatSetValue(A[dir], f_idx_g, f_tmp_g,-mu*exchange_surface/dxyz[ngbd_dir/2], ADD_VALUES); CHKERRXX(ierr);
              }
            }
          }
        }

        continue;
      }
    }
#endif

    /* ----------------------------------------------------------------------------------------------------------------------------------------------------------------
     * BULK CASE: Finite Volume Discretization using Voronoi Cells
     ----------------------------------------------------------------------------------------------------------------------------------------------------------------*/

    /* integrally in positive domain */
#ifndef P4_TO_P8
    if(partition->size() == 0)
    {
      if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
        ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, 1, ADD_VALUES); CHKERRXX(ierr); } // needs only to be done if fully reset
      rhs_p[f_idx] = 0;
      null_space_p[f_idx] = 0;
      continue;
    }
#endif

    const double volume = voro_tmp.get_volume();
    if(!matrix_is_ready[dir]) {
      ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, volume*(desired_diag[dir] - current_diag[dir]), ADD_VALUES); CHKERRXX(ierr); }
    rhs_p[f_idx] *= volume;
    if(desired_diag[dir] > 0.0) matrix_has_nullspace[dir] = false;

    /* bulk case, finite volume on voronoi cell */
    PointDIM pc(DIM(xyz[0], xyz[1], xyz[2]));
    double xyz_pert[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
    if(fabs(xyz[dir] - xyz_min[dir]) < EPS*tree_dimensions[dir]) xyz_pert[dir] = xyz_min[dir] + 2.0*EPS*(xyz_max[dir] - xyz_min[dir]);
    if(fabs(xyz[dir] - xyz_max[dir]) < EPS*tree_dimensions[dir]) xyz_pert[dir] = xyz_max[dir] - 2.0*EPS*(xyz_max[dir] - xyz_min[dir]);

    for(size_t m = 0; m < points->size(); ++m) // E: Loop over the Voronoi points in the given face's associated Voronoi cell
    {
      PetscInt m_idx_g;

#ifdef P4_TO_P8
      const double surface = (*points)[m].s;
#else
      size_t k = mod(m - 1, points->size());
      const double surface = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      double distance_to_neighbor = ((*points)[m].p - pc).norm_L2(); // E: Distance between the center point of the Voronoi cell and the mth Voronoi point (distance to each neighbor value)
      P4EST_ASSERT(distance_to_neighbor <= 0.5*sqrt(SUMD(SQR(xyz_max[0] - xyz_min[0]), SQR(xyz_max[1] - xyz_min[1]), SQR(xyz_max[2] - xyz_min[2])))); // to check consistency, especially worth it in case sthg goes wrong when periodic

      switch((*points)[m].n) // E: *points[m].n returns the index of the mth neighbor face, except if it is negative (wall/interface cut)
      {
      case WALL_PARALLEL_TO_FACE:
      {
        matrix_has_nullspace[dir] = false;
        // no need to divide d by 2 in this special case, by construction in Voronoi2/3D
        if(!only_diag_is_modified[dir] && !matrix_is_ready[dir])
        {
          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*surface/distance_to_neighbor, ADD_VALUES); CHKERRXX(ierr); // needs only to be done if fully reset
        }
        try {
          if(bc_hodge != NULL && (bc_hodge->wallType(DIM((*points)[m].p.x, (*points)[m].p.y, (*points)[m].p.z)) == NEUMANN))
          {
#ifdef P4_TO_P8
            bool positive = (dir == dir::x ? (*points)[m].p.x > xyz[0] : (dir == dir::y ? (*points)[m].p.y > xyz[1] : (*points)[m].p.z > xyz[2]));
#else
            bool positive = (dir == dir::x ? (*points)[m].p.x > xyz[0] : (*points)[m].p.y > xyz[1]);
#endif
            rhs_p[f_idx] += mu*surface*(bc[dir].wallValue(DIM((*points)[m].p.x, (*points)[m].p.y, (*points)[m].p.z)) + (positive ? +1.0 : -1.0)*bc_hodge->wallValue(DIM((*points)[m].p.x, (*points)[m].p.y, (*points)[m].p.z)))/distance_to_neighbor;
          }
          else
          {
            // this is the least desirable scenario and main reason for the try-catch:
            // the user wants to use a stretched grid, but does not provide bc's that can easily be evaluated from everywhere so the solver's usage requires locality: if dxyz_hodge can't be interpolated where it's desired this is very likely to throw an exception
            rhs_p[f_idx] += mu*surface*(bc[dir].wallValue(DIM((*points)[m].p.x, (*points)[m].p.y, (*points)[m].p.z)) + interp_dxyz_hodge(DIM((*points)[m].p.x, (*points)[m].p.y, (*points)[m].p.z)))/distance_to_neighbor;
          }
        } catch (std::exception e) {
          throw std::runtime_error("my_p4est_poisson_faces_t: the boundary condition value needs to be readable from everywhere in the domain when using such stretched grids and non-periodic wall conditions, sorry...");
        }
        break;
      }
      case WALL_m00:
      case WALL_p00:
      case WALL_0m0:
      case WALL_0p0:
#ifdef P4_TO_P8
      case WALL_00m:
      case WALL_00p:
#endif
      {
        char wall_orientation = -1 - (*points)[m].n;
        P4EST_ASSERT(wall_orientation >= 0 && wall_orientation < P4EST_FACES);
        double wall_eval[P4EST_DIM] = {DIM(xyz_pert[0], xyz_pert[1], xyz_pert[2])}; wall_eval[wall_orientation/2] = (wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]);
        switch(bc[dir].wallType(wall_eval))
        {
        case DIRICHLET:
          if(dir == wall_orientation/2)
            throw std::runtime_error("[CASL_ERROR]: my_p4est_poisson_faces_t->setup_linear_system: dirichlet conditions on walls parallel to faces should have been done before. Are you using a rectangular grid ?");
          matrix_has_nullspace[dir] = false;
          distance_to_neighbor /= 2;
          if(!only_diag_is_modified[dir] && !matrix_is_ready[dir]) {
            ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*surface/distance_to_neighbor, ADD_VALUES); CHKERRXX(ierr); } // needs only to be done if fully reset
          rhs_p[f_idx] += mu*surface*(bc[dir].wallValue(wall_eval) + interp_dxyz_hodge(wall_eval))/distance_to_neighbor;
          break;
        case NEUMANN:
          rhs_p[f_idx] += mu*surface*(bc[dir].wallValue(wall_eval) + (apply_hodge_second_derivative_if_neumann ? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later --> good luck!
          break;
        default:
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type. Issue on Wall 00p");
        }
        break;
      }
      case INTERFACE:
        switch( bc[dir].interfaceType(xyz))
        {
        /* note that DIRICHLET done with finite differences */
        case NEUMANN:
#ifdef P4_TO_P8
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: Neumann boundary conditions should be treated separately in 3D ...");
#else
          rhs_p[f_idx] += mu*surface*(bc[dir].interfaceValue(((*points)[m].p.x + xyz[0])/2., ((*points)[m].p.y + xyz[1])/2.) + (apply_hodge_second_derivative_if_neumann ? 0.0 : 0.0)); // apply_hodge_second_derivative_if_neumann: would need to be fixed later on
#endif
          break;
        default:
          std::cout<<"\n Interface BC Type is: "<< bc[dir].interfaceType(xyz)<< " at ( "<< xyz[0]<< ", "<<xyz[1]<< ")"<<std::endl;
          throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_faces_t: unknown boundary condition type. Issue on interface");
        }
        break;

      default:
        if(!only_diag_is_modified[dir] && !matrix_is_ready[dir])
        {
          /* add coefficients in the matrix */
          m_idx_g = faces->global_index((*points)[m].n, dir);

          ierr = MatSetValue(A[dir], f_idx_g, f_idx_g, mu*surface/distance_to_neighbor, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A[dir], f_idx_g, m_idx_g,-mu*surface/distance_to_neighbor, ADD_VALUES); CHKERRXX(ierr);
        }
      }
    }
  } // End of loop through all faces in the direction dir (ie. end of loop through all x faces)

  ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

  int global_size_bc_index = bc_index.size();
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &global_size_bc_index, 1, MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret); //E: Sum the number of BC additions to the RHS to make across all processors,
                                                                                                                              //store on all processors as global_size_bc_index

  /* -----------------------------------------------------------------------------------------------------------------
   * Complete the right hand side with correct boundary condition: bc_v + grad(dxyz_hodge) (Aka, complete the applying of the Interfacial Dirichlet boundary condition)
   *-----------------------------------------------------------------------------------------------------------------*/
  if(global_size_bc_index > 0)
  {
    std::vector<double> bc_val(bc_index.size());
    interp_dxyz_hodge.interpolate(bc_val.data());
    interp_dxyz_hodge.clear();
    for(size_t n = 0; n < bc_index.size(); ++n)
      rhs_p[bc_index[n]] += bc_coeffs[n]*bc_val[n];
    bc_val.clear();
    bc_index.clear();
    bc_coeffs.clear();
  }
  /* -----------------------------------------------------------------------------------------------------------------
   * Finish Assembling the Linear System
   *-----------------------------------------------------------------------------------------------------------------*/
  ierr = VecRestoreArray(null_space[dir], &null_space_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  if(!matrix_is_ready[dir])
  {
    /* Assemble the matrix */
    ierr = MatAssemblyBegin(A[dir], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (A[dir], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  }

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

  ierr = KSPSetOperators(ksp[dir], A[dir], A[dir], SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  /* [Raphael Egan:] Starting from version 3.5, the last argument in KSPSetOperators became
   * irrelevant and is now simply disregarded in the above call. The matrices now keep track
   * of changes to their values and/or to their nonzero pattern by themselves. If no
   * modification was made to the matrix, the ksp environment can figure it out and knows
   * that the current preconditioner is still valid, thus it won't be recomputed.
   * If one desires to force reusing the current preconditioner EVEN IF a modification was
   * made to the matrix, one needs to call
   * ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRXX(ierr);
   * before the subsequent call to KSPSolve().
   * I have decided not to enforce that...
   */
  matrix_is_ready[dir]        = true;
  current_diag[dir]           = desired_diag[dir];
  P4EST_ASSERT(current_diag_is_as_desired(dir));

  ierr = PetscLogEventEnd(log_my_p4est_poisson_faces_setup_linear_system, A, rhs[dir], 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_faces_t::print_partition_VTK(const char *file, const unsigned char &dir)
{
  if(compute_partition_on_the_fly)
    throw std::invalid_argument("[ERROR]: my_p4est_poisson_faces_t->print_partition_VTK: please don't use compute_partition_on_the_fly if you want to output the voronoi partition.");
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
