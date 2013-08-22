#include "poisson_solver.h"
#include "petsc_compatibility.h"

PoissonSolver::PoissonSolver(p4est_t *p4est_, const CF_2& uex_, const CF_2 &f_)
  : p4est(p4est_), uex(&uex_), f(&f_)
{
  w.start("getting all neighboring information");
  cell_ngbds = new CellNeighbors(p4est);
  cell_ngbds->Init();
  w.stop(); w.read_duration();

  w.start("solver initialization");
  // Set up the matrix and the rhs
  ierr = MatCreate(p4est->mpicomm, &a); CHKERRXX(ierr);
  ierr = MatSetType(a, MATMPIAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(a, p4est->local_num_quadrants, p4est->local_num_quadrants,
                     PETSC_DECIDE, PETSC_DECIDE); CHKERRXX(ierr);
  ierr = MatSetFromOptions(a); CHKERRXX(ierr);
  ierr = MatSeqAIJSetPreallocation(a, 9, NULL); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(a, 9, NULL, 9, NULL); CHKERRXX(ierr);


  // Now vectors
  ierr = VecCreate(p4est->mpicomm, &x); CHKERRXX(ierr);
  ierr = VecSetType(x, VECMPI); CHKERRXX(ierr);
  ierr = VecSetSizes(x, p4est->local_num_quadrants, PETSC_DECIDE); CHKERRXX(ierr);
  ierr = VecSetFromOptions(x); CHKERRXX(ierr);
  ierr = VecSet(x, 0); CHKERRXX(ierr);

  ierr = VecDuplicate(x, &b); CHKERRXX(ierr);
  ierr = VecDuplicate(x, &xex); CHKERRXX(ierr);
  ierr = VecSet(b, 0); CHKERRXX(ierr);
  ierr = VecSet(xex, 0); CHKERRXX(ierr);

  // Now ksp
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  w.stop(); w.read_duration();

}

/*
 * This method sets up the negative laplace matrix for a very simple
 * case when there is no interface involved.
 *
 * Also, for now we do not consider coordinate transforms that involve distortion
 * of the lines. The reason for this is in that case one needs to consider either
 * full coordinate transfroms and solve the transformed equation or come up with
 * better ways of directly discretizing the equations in the physical domain. Thus
 * we only allow for coordinate systems that are at most scales of one another.
 */
void PoissonSolver::setUpNegativeLaplaceMatrix(){

  w.start("Setting up matrix");
  
  p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_gloidx_t qu_proc_offset = *(p4est->global_first_quadrant + p4est->mpirank);

  // Get a reference to all the neighbors
  const vector<CellNeighbors::quad_array> &all_ngbd_m0 = cell_ngbds->get_m0_neighbors();
  const vector<CellNeighbors::quad_array> &all_ngbd_p0 = cell_ngbds->get_p0_neighbors();
  const vector<CellNeighbors::quad_array> &all_ngbd_0m = cell_ngbds->get_0m_neighbors();
  const vector<CellNeighbors::quad_array> &all_ngbd_0p = cell_ngbds->get_0p_neighbors();

  // Loop over local trees
  for (p4est_topidx_t tr = p4est->first_local_tree; tr<= p4est->last_local_tree; ++tr){
    // Get a reference to the tree
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
    p4est_locidx_t qu_tree_offset = tree->quadrants_offset;

    // For a given tree, loop over all local cells
    for (size_t qu = 0; qu<tree->quadrants.elem_count; ++qu){
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu);

      p4est_gloidx_t qu_gloidx = qu_proc_offset + qu_tree_offset + qu;
      p4est_locidx_t qu_locidx = qu_tree_offset + qu;

      quad->p.piggy3.which_tree = tr;
      quad->p.piggy3.local_num  = qu_locidx;

      // xmWall
      p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);
      bool quad_is_boundary = (quad->x      == 0              && t2t[tr*4 + 0] == tr) ||
          (quad->x + qh == P4EST_ROOT_LEN && t2t[tr*4 + 1] == tr) ||
          (quad->y      == 0              && t2t[tr*4 + 2] == tr) ||
          (quad->y + qh == P4EST_ROOT_LEN && t2t[tr*4 + 3] == tr);

      if(quad_is_boundary){
        ierr = MatSetValue(a, qu_gloidx, qu_gloidx, 1.0, ADD_VALUES); CHKERRXX(ierr);
        continue;
      }

      // Get a referrence to the current neighbors
      const CellNeighbors::quad_array &ngbd_m0 = all_ngbd_m0[qu_locidx];
      const CellNeighbors::quad_array &ngbd_p0 = all_ngbd_p0[qu_locidx];
      const CellNeighbors::quad_array &ngbd_0m = all_ngbd_0m[qu_locidx];
      const CellNeighbors::quad_array &ngbd_0p = all_ngbd_0p[qu_locidx];

      double dx_C, dy_C;
      dx_dy_dz_quadrant(p4est, tr, quad, &dx_C, &dy_C, NULL);

      // Left cells
      if (ngbd_m0.size() == 1) { // one same size or bigger cell

        // First get all the neighbors right of the current quad
        CellNeighbors::quad_array ngbd_rev;
        ngbd_rev.push_back(quad);

        if (ngbd_m0[0]->level != quad->level) { // different size
          if (ngbd_m0[0]->y == quad->y){
#ifdef CASL_THROWS
            if (ngbd_0p.size() != 1 || ngbd_0p[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_0p[0]);
          } else {
#ifdef CASL_THROWS
            if (ngbd_0m.size() != 1 || ngbd_0m[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_0m[0]);
          }
        }

        double dx_L, dy_L;
        dx_dy_dz_quadrant(p4est, ngbd_m0[0]->p.piggy3.which_tree, ngbd_m0[0], &dx_L, &dy_L, NULL);

        double dx = 0.5 *(dx_L+dx_C);
        for (size_t i=0; i<ngbd_rev.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_rev[i]->p.piggy3.local_num), dy_C/dx * dy_C/dy_L, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_m0[0]->p.piggy3.local_num), -dy_C/dx, ADD_VALUES);
      } else { // two smaller cells
        // the two smaller cells have to of the same size
#ifdef CASL_THROWS
        if (ngbd_m0.size() != 2)
          throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        for (size_t i=0; i<ngbd_m0.size(); ++i){
          if (ngbd_m0[i]->level != quad->level + 1)
            throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        }
#endif
        double dx_L, dy_L;
        dx_dy_dz_quadrant(p4est, ngbd_m0[0]->p.piggy3.which_tree, ngbd_m0[0], &dx_L, &dy_L, NULL);

        double dx = 0.5*(dx_L + dx_C);
        for (size_t i=0; i<ngbd_m0.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_m0[i]->p.piggy3.local_num), -dy_L/dx, ADD_VALUES);
        }

        MatSetValue(a,qu_gloidx, qu_gloidx, dy_C/dx, ADD_VALUES);

      }

      // Right cells
      if (ngbd_p0.size() == 1) { // one same size or bigger cell

        // First get all the neighbors right of the current quad
        CellNeighbors::quad_array ngbd_rev;
        ngbd_rev.push_back(quad);

        if (ngbd_p0[0]->level != quad->level) { // different size
          if (ngbd_p0[0]->y == quad->y){
#ifdef CASL_THROWS
            if (ngbd_0p.size() != 1 || ngbd_0p[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_0p[0]);
          } else {
#ifdef CASL_THROWS
            if (ngbd_0m.size() != 1 || ngbd_0m[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_0m[0]);
          }
        }

        double dx_R, dy_R;
        dx_dy_dz_quadrant(p4est, ngbd_p0[0]->p.piggy3.which_tree, ngbd_p0[0], &dx_R, &dy_R, NULL);

        double dx = 0.5 *(dx_R+dx_C);
        for (size_t i=0; i<ngbd_rev.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_rev[i]->p.piggy3.local_num), dy_C/dx * dy_C/dy_R, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_p0[0]->p.piggy3.local_num), -dy_C/dx, ADD_VALUES);

      } else { // two smaller cells
        // the two smaller cells have to of the same size
#ifdef CASL_THROWS
        if (ngbd_p0.size() != 2)
          throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        for (size_t i=0; i<ngbd_p0.size(); ++i){
          if (ngbd_p0[i]->level != quad->level + 1)
            throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        }
#endif
        double dx_R, dy_R;
        dx_dy_dz_quadrant(p4est, ngbd_p0[0]->p.piggy3.which_tree, ngbd_p0[0], &dx_R, &dy_R, NULL);

        double dx = 0.5*(dx_R + dx_C);
        for (size_t i=0; i<ngbd_p0.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_p0[i]->p.piggy3.local_num), -dy_R/dx, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, qu_gloidx, dy_C/dx, ADD_VALUES);

      }

      // bottom cells
      if (ngbd_0m.size() == 1) { // one same size or bigger cell

        // First get all the neighbors right of the current quad
        CellNeighbors::quad_array ngbd_rev;
        ngbd_rev.push_back(quad);

        if (ngbd_0m[0]->level != quad->level) { // different size
          if (ngbd_0m[0]->x == quad->x){
#ifdef CASL_THROWS
            if (ngbd_p0.size() != 1 || ngbd_p0[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_p0[0]);
          } else {
#ifdef CASL_THROWS
            if (ngbd_m0.size() != 1 || ngbd_m0[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_m0[0]);
          }
        }

        double dx_B, dy_B;
        dx_dy_dz_quadrant(p4est, ngbd_0m[0]->p.piggy3.which_tree, ngbd_0m[0], &dx_B, &dy_B, NULL);

        double dy = 0.5 *(dy_B+dy_C);
        for (size_t i=0; i<ngbd_rev.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_rev[i]->p.piggy3.local_num), dx_C/dy * dx_C/dx_B, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_0m[0]->p.piggy3.local_num), -dx_C/dy, ADD_VALUES);

      } else { // two smaller cells
        // the two smaller cells have to of the same size
#ifdef CASL_THROWS
        if (ngbd_0m.size() != 2)
          throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        for (size_t i=0; i<ngbd_0m.size(); ++i){
          if (ngbd_0m[i]->level != quad->level + 1)
            throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        }
#endif
        double dx_B, dy_B;
        dx_dy_dz_quadrant(p4est, ngbd_0m[0]->p.piggy3.which_tree, ngbd_0m[0], &dx_B, &dy_B, NULL);

        double dy = 0.5*(dy_B + dy_C);
        for (size_t i=0; i<ngbd_0m.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_0m[i]->p.piggy3.local_num), -dx_B/dy, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, qu_gloidx, dx_C/dy, ADD_VALUES);

      }

      // top cells
      if (ngbd_0p.size() == 1) { // one same size or bigger cell

        // First get all the neighbors right of the current quad
        CellNeighbors::quad_array ngbd_rev;
        ngbd_rev.push_back(quad);

        if (ngbd_0p[0]->level != quad->level) { // different size
          if (ngbd_0p[0]->x == quad->x){
#ifdef CASL_THROWS
            if (ngbd_p0.size() != 1 || ngbd_p0[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_p0[0]);
          } else {
#ifdef CASL_THROWS
            if (ngbd_m0.size() != 1 || ngbd_m0[0]->level != quad->level)
              throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
#endif
            ngbd_rev.push_back(ngbd_m0[0]);
          }
        }

        double dx_T, dy_T;
        dx_dy_dz_quadrant(p4est, ngbd_0p[0]->p.piggy3.which_tree, ngbd_0p[0], &dx_T, &dy_T, NULL);

        double dy = 0.5 *(dy_T+dy_C);
        for (size_t i=0; i<ngbd_rev.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_rev[i]->p.piggy3.local_num), dx_C/dy * dx_C/dx_T, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_0p[0]->p.piggy3.local_num), -dx_C/dy, ADD_VALUES);

      } else { // two smaller cells
        // the two smaller cells have to of the same size
#ifdef CASL_THROWS
        if (ngbd_0p.size() != 2)
          throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        for (size_t i=0; i<ngbd_0p.size(); ++i){
          if (ngbd_0p[i]->level != quad->level + 1)
            throw std::runtime_error("[CASL_ERROR]: tree does not seem to be balanced");
        }
#endif
        double dx_T, dy_T;
        dx_dy_dz_quadrant(p4est, ngbd_0p[0]->p.piggy3.which_tree, ngbd_0p[0], &dx_T, &dy_T, NULL);

        double dy = 0.5*(dy_T + dy_C);
        for (size_t i=0; i<ngbd_0p.size(); ++i){
          MatSetValue(a, qu_gloidx, cell_ngbds->local2global(ngbd_0p[i]->p.piggy3.local_num), -dx_T/dy, ADD_VALUES);
        }

        MatSetValue(a, qu_gloidx, qu_gloidx, dx_C/dy, ADD_VALUES);

      }

    }
  }

  ierr = MatAssemblyBegin(a, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(a, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

  w.stop(); w.read_duration();

}

void PoissonSolver::setUpNegativeLaplaceRhsVec(){

  w.start("setting up rhsvec");

  p4est_topidx_t *t2t = p4est->connectivity->tree_to_tree;
  p4est_gloidx_t qu_proc_offset = *(p4est->global_first_quadrant + p4est->mpirank);

  // Loop over local trees
  for (p4est_topidx_t tr = p4est->first_local_tree; tr<= p4est->last_local_tree; ++tr){
    // Get a reference to the tree
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
    p4est_locidx_t qu_tree_offset = tree->quadrants_offset;

    // For a given tree, loop over all local cells
    for (size_t qu = 0; qu<tree->quadrants.elem_count; ++qu){
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu);
      p4est_gloidx_t qu_gloidx = qu_proc_offset + qu_tree_offset + qu;

      // xmWall
      p4est_qcoord_t qh = P4EST_QUADRANT_LEN(quad->level);
      bool quad_is_boundary = (quad->x      == 0              && t2t[tr*4 + 0] == tr) ||
          (quad->x + qh == P4EST_ROOT_LEN && t2t[tr*4 + 1] == tr) ||
          (quad->y      == 0              && t2t[tr*4 + 2] == tr) ||
          (quad->y + qh == P4EST_ROOT_LEN && t2t[tr*4 + 3] == tr);

      double x_c, y_c;
      xyz_quadrant(p4est, tr, quad, &x_c, &y_c, NULL);

      double val = (*uex)(x_c, y_c);
      ierr = VecSetValue(xex, qu_gloidx, val, INSERT_VALUES); CHKERRXX(ierr);

      if(quad_is_boundary){
        ierr = VecSetValue(b, qu_gloidx, val, INSERT_VALUES); CHKERRXX(ierr);
        continue;
      }

      double dx_C, dy_C;
      dx_dy_dz_quadrant(p4est, tr, quad, &dx_C, &dy_C, NULL);

      ierr = VecSetValue(b, qu_gloidx, dx_C*dy_C*(*f)(x_c, y_c), INSERT_VALUES); CHKERRXX(ierr);
    }
  }

  ierr = VecAssemblyBegin(b); CHKERRXX(ierr);
  ierr = VecAssemblyBegin(xex); CHKERRXX(ierr);

  ierr = VecAssemblyEnd(b); CHKERRXX(ierr);
  ierr = VecAssemblyEnd(xex); CHKERRXX(ierr);
  w.stop(); w.read_duration();

}

void PoissonSolver::save(const std::string &filename){

  PetscViewer viewer;

  ierr = PetscViewerBinaryOpen(p4est->mpicomm, (filename + "_mat").c_str(), FILE_MODE_WRITE, &viewer); CHKERRXX(ierr);
  ierr = MatView(a, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  ierr = PetscViewerBinaryOpen(p4est->mpicomm, (filename + "_vec").c_str(), FILE_MODE_WRITE, &viewer); CHKERRXX(ierr);
  ierr = VecView(b, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

}

void PoissonSolver::load(const std::string &filename){

  PetscViewer viewer;

  ierr = PetscViewerBinaryOpen(p4est->mpicomm, (filename + "_mat").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = MatLoad(a, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  ierr = PetscViewerBinaryOpen(p4est->mpicomm, (filename + "_vec").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = VecLoad(b, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
}

void PoissonSolver::solve(Vec& sol, Vec& sol_ex){

  w.start("solving the linear system");

  // set ksp options
  ierr = KSPSetOperators(ksp, a, a, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = KSPSetType(ksp, KSPBCGS); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  // set pc options
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, PCHYPRE); CHKERRXX(ierr);
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // solve the system
  ierr = KSPSolve(ksp, b, x); CHKERRXX(ierr);

  sol    = x;
  sol_ex = xex;

  w.stop(); w.read_duration();

}

PoissonSolver::~PoissonSolver(){
  // Destroy cell neighbors;
  delete cell_ngbds;

  // Destroy PETSc objects
  ierr = MatDestroy(a); CHKERRXX(ierr);
  ierr = VecDestroy(x); CHKERRXX(ierr);
  ierr = VecDestroy(b); CHKERRXX(ierr);
  ierr = VecDestroy(xex); CHKERRXX(ierr);
  ierr = KSPDestroy(ksp); CHKERRXX(ierr);
}



























