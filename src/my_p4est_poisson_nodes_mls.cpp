#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_mls.h"
#include <src/my_p8est_refine_coarsen.h>
#else
#include "my_p4est_poisson_nodes_mls.h"
#include <src/my_p4est_refine_coarsen.h>
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
extern PetscLogEvent log_my_p4est_poisson_nodes_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_matrix_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_rhsvec_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0

#ifdef P4_TO_P8
#define N_NBRS_MAX 27
#else
#define N_NBRS_MAX 9
#endif

my_p4est_poisson_nodes_mls_t::my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors(node_neighbors),
    p4est(node_neighbors->p4est), nodes(node_neighbors->nodes), ghost(node_neighbors->ghost), myb_(node_neighbors->myb),
    phi_interp(node_neighbors), interp_local(node_neighbors),
    is_matrix_computed(false), matrix_has_nullspace(false),
    A(NULL), rhs(NULL),
    node_vol(NULL),
    phi_dd_owned(false), phi_xx(NULL), phi_yy(NULL), phi_zz(NULL),
    keep_scalling(true), scalling(NULL), phi_eff(NULL), cube_refinement(0),
    kink_special_treatment(false)
  #ifdef P4_TO_P8
  #endif
{
  /*
   * TODO: We can compute the exact number of enteries in the matrix and just
   * allocate that many elements. My guess is its not going to change the memory
   * consumption that much anyway so we might as well allocate for the worst
   * case scenario which is 6 element per row. In places where the grid is
   * uniform we really need 5. In 3D this is 12 vs 7 so its more important ...
   *
   * Also, we only really should allocate 1 per row for points in omega^+ and
   * points for which we use Dirichlet. In the end we are allocating more than
   * we need which may or may not be a real issue in practice ...
   *
   * If we want to do this the correct way, we should first precompute all the
   * weights and probably put them in SparseCRS matrix (CASL) and then construct
   * PETSc matrix such that it uses the same memory space. Note that If copy the
   * stuff its (probably) going to both take longer to execute and consume more
   * memory eventually ...
   *
   * Another simpler approach is to forget about Dirichlet points and also
   * omega^+ domain, but consider the T-junctions and allocate the correct
   * number of elements at least for T-junctions. This does not require
   * precomputation and we only need to chech if a node is T-junction which is
   * much simpler ...
   *
   * We'll see if this becomes a real issue in memory consumption,. My GUESS is
   * it really does not matter in 2D but __might__ be important in 3D for really
   * big problems ...
   */

  // compute global numbering of nodes
  global_node_offset.resize(p4est->mpisize+1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_node_offset[r+1] = global_node_offset[r] + (PetscInt)nodes->global_owned_indeps[r];

  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  // compute grid parameters
  // NOTE: Assuming all trees are of the same size [0, 1]^d
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  dx_min = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  dy_min = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  dz_min = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif
#ifdef P4_TO_P8
  d_min = MIN(dx_min, dy_min, dz_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min + dz_min*dz_min);
#else
  d_min = MIN(dx_min, dy_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min);
#endif

  vol_min = dx_min*dy_min;
#ifdef P4_TO_P8
  vol_min *= dz_min;
#endif

  // construct petsc global indices
  petsc_gloidx.resize(nodes->indep_nodes.elem_count);

  // local nodes
  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++)
    petsc_gloidx[i] = global_node_offset[p4est->mpirank] + i;

  // ghost nodes
  p4est_locidx_t ghost_size = nodes->indep_nodes.elem_count - nodes->num_owned_indeps;
  for (p4est_locidx_t i = 0; i<ghost_size; i++){
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->num_owned_indeps);
    petsc_gloidx[i+nodes->num_owned_indeps] = global_node_offset[nodes->nonlocal_ranks[i]] + ni->p.piggy3.local_num;
  }

  /* We handle the all-neumann situation by fixing the solution at some point inside the domain.
   * In general this method is _NOT_ recommended as it pollutes the eigen-value spectrum.
   *
   * The other (and preferred) method to solve siongular matrices is to set the nullspace. If the
   * matrix is non-symmetric (which is the case for us) one has to compute the left nullspace, that
   * is the nuullspace of A^T instead of A. This is because it is the left nullspace that is orthogonal
   * complement of Range(A). Unfortunately teh general way of computing the nullspace is the SVD algorithm
   * which is more expensive than the linear system solution itself! Unless you can come up with a
   * smart way to "guess" the left nullspace, this is not recommened.
   *
   * To fix the solution, every processor sets up the matrix as if it was non-singular. Next, if all
   * processes agree that the matrix is singular, the process with the first interior node fixes the
   * value at that point to zero. This is done by modifing the row corresponding to 'fixed_value_idx_g'
   * index in the matrix.
   */
  fixed_value_idx_g = global_node_offset[p4est->mpisize];

  double eps = 1E-9*d_min;

#ifdef P4_TO_P8
  eps_dom = eps*eps*eps;
  eps_ifc = eps*eps;
#else
  eps_dom = eps*eps;
  eps_ifc = eps;
#endif

}

my_p4est_poisson_nodes_mls_t::~my_p4est_poisson_nodes_mls_t()
{
  if (A             != NULL) {ierr = MatDestroy(A);                      CHKERRXX(ierr);}
  if (ksp           != NULL) {ierr = KSPDestroy(ksp);                    CHKERRXX(ierr);}
  if (phi_dd_owned)
  {
    if (phi_xx != NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecDestroy(phi_xx->at(i)); CHKERRXX(ierr);}
      delete phi_xx;
    }

    if (phi_yy != NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecDestroy(phi_yy->at(i)); CHKERRXX(ierr);}
      delete phi_yy;
    }

#ifdef P4_TO_P8
    if (phi_zz != NULL)
    {
      for (int i = 0; i < n_phis; i++) {ierr = VecDestroy(phi_zz->at(i)); CHKERRXX(ierr);}
      delete phi_zz;
    }
#endif
  }

//  if (is_mue_dd_owned)
//  {
//    if (mue_xx_     != NULL) {ierr = VecDestroy(mue_xx_);                CHKERRXX(ierr);}
//    if (mue_yy_     != NULL) {ierr = VecDestroy(mue_yy_);                CHKERRXX(ierr);}
//#ifdef P4_TO_P8
//    if (mue_zz_     != NULL) {ierr = VecDestroy(mue_zz_);                CHKERRXX(ierr);}
//#endif
//  }

  if (scalling != NULL) {ierr = VecDestroy(scalling); CHKERRXX(ierr);}
  if (node_vol != NULL) {ierr = VecDestroy(node_vol); CHKERRXX(ierr);}
  if (phi_eff_owned)    {ierr = VecDestroy(phi_eff);  CHKERRXX(ierr);}
}

void my_p4est_poisson_nodes_mls_t::set_geometry(std::vector<Vec> &phi_,
                                                std::vector<Vec> &phi_xx_,
                                                std::vector<Vec> &phi_yy_,
                                                #ifdef P4_TO_P8
                                                std::vector<Vec> &phi_zz_,
                                                #endif
                                                std::vector<action_t> &action_, std::vector<int> &color_, Vec phi_eff_)
{
  phi = &phi_;
  phi_xx = &phi_xx_;
  phi_yy = &phi_yy_;
#ifdef P4_TO_P8
  phi_zz = &phi_zz_;
#endif
  action = &action_;
  color = &color_;

  n_phis = action->size();

  if (phi_eff_ == NULL) compute_phi_eff();
  else                  phi_eff = phi_eff_;
}

void my_p4est_poisson_nodes_mls_t::set_geometry(std::vector<Vec> &phi_,
                                                std::vector<action_t> &action_, std::vector<int> &color_, Vec phi_eff_)
{
  phi = &phi_;
  action = &action_;
  color = &color_;

  n_phis = action->size();

  if (phi_eff_ == NULL) compute_phi_eff();
  else                  phi_eff = phi_eff_;

  compute_phi_dd();
}

void my_p4est_poisson_nodes_mls_t::compute_phi_eff()
{
  if (phi_eff != NULL) {ierr = VecDestroy(phi_eff); CHKERRXX(ierr);}
  phi_eff_owned = true;

  ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff); CHKERRXX(ierr);

  std::vector<double *>   phi_p(n_phis, NULL);
  double                  *phi_eff_p;

  for (int i = 0; i < n_phis; i++) {ierr = VecGetArray(phi->at(i),  &phi_p[i]);  CHKERRXX(ierr);}
                                    ierr = VecGetArray(phi_eff,     &phi_eff_p); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    phi_eff_p[n] = -10.;

    for (int i = 0; i < n_phis; i++)
    {
      switch (action->at(i))
      {
      case INTERSECTION:  phi_eff_p[n] = (phi_eff_p[n] > phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
      case ADDITION:      phi_eff_p[n] = (phi_eff_p[n] < phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
      case COLORATION:    /* do nothing */ break;
      }
    }
  }

  for (int i = 0; i < n_phis; i++) {ierr = VecRestoreArray(phi->at(i), &phi_p[i]);  CHKERRXX(ierr);}
                                    ierr = VecRestoreArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(phi_eff, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (phi_eff, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::compute_phi_dd()
{
  // Allocate memory for second derivaties
  if (phi_xx != NULL && phi_dd_owned)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecDestroy(phi_xx->at(i)); CHKERRXX(ierr);}
    delete phi_xx;
  }
  phi_xx = new std::vector<Vec> ();

  if (phi_yy != NULL && phi_dd_owned)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecDestroy(phi_yy->at(i)); CHKERRXX(ierr);}
    delete phi_yy;
  }
  phi_yy = new std::vector<Vec> ();

#ifdef P4_TO_P8
  if (phi_zz != NULL && phi_dd_owned)
  {
    for (int i = 0; i < n_phis; i++) {ierr = VecDestroy(phi_zz->at(i)); CHKERRXX(ierr);}
    delete phi_zz;
  }
  phi_zz = new std::vector<Vec> ();
#endif

  for (unsigned int i = 0; i < n_phis; i++)
  {
    phi_xx->push_back(Vec()); ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx->at(i)); CHKERRXX(ierr);
    phi_yy->push_back(Vec()); ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy->at(i)); CHKERRXX(ierr);
#ifdef P4_TO_P8
    phi_zz->push_back(Vec()); ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz->at(i)); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    node_neighbors->second_derivatives_central(phi->at(i), phi_xx->at(i), phi_yy->at(i), phi_zz->at(i));
#else
    node_neighbors->second_derivatives_central(phi->at(i), phi_xx->at(i), phi_yy->at(i));
#endif
  }
  phi_dd_owned = true;
}

void my_p4est_poisson_nodes_mls_t::compute_volumes()
{
  /* TO FIX: doesn't work when the interface is too close to walls
   * more precisely, one needs to carefully choose quadrants to fetch nodes,
   * because near the walls some quadrants don't exist
   */

  // TO FIX: also the grid should be uniform at the interface

  //---------------------------------------------------------------------
  // get access to LSFs
  //---------------------------------------------------------------------
  std::vector<double *> phi_p (n_phis, NULL);
  std::vector<double *> phi_xx_p (n_phis, NULL);
  std::vector<double *> phi_yy_p (n_phis, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_zz_p (n_phis, NULL);
#endif

  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecGetArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  double *phi_eff_p;
  ierr = VecGetArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // get access to vec with volumes
  //---------------------------------------------------------------------
  double *node_vol_p;
  if (node_vol == NULL)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &node_vol); CHKERRXX(ierr);
  }
  ierr = VecGetArray(node_vol, &node_vol_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // allocate vectors for LSF values for a cube
  //---------------------------------------------------------------------
  std::vector< std::vector<double> > phi_cube(n_phis, std::vector<double> (N_NBRS_MAX, -1));

  std::vector< std::vector<double> > phi_xx_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
  std::vector< std::vector<double> > phi_yy_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#ifdef P4_TO_P8
  std::vector< std::vector<double> > phi_zz_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#endif

  //---------------------------------------------------------------------
  // create a cube
  //---------------------------------------------------------------------
#ifdef P4_TO_P8
  cube3_mls_t cube;
#else
  cube2_mls_t cube;
#endif

  //---------------------------------------------------------------------
  // some additional variables
  //---------------------------------------------------------------------
  bool neighbor_exists[N_NBRS_MAX];
  p4est_locidx_t neighbors[N_NBRS_MAX];

  double x_grid[3], y_grid[3];
#ifdef P4_TO_P8
  double z_grid[3];
#endif

  //---------------------------------------------------------------------
  // main loop over nodes
  //---------------------------------------------------------------------
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
  {
    if (phi_eff_p[n] >  1.0*diag_min) {node_vol_p[n] = 0.; continue;}
    if (phi_eff_p[n] < -1.0*diag_min) {node_vol_p[n] = 1.; continue;}

    double x_C  = node_x_fr_n(n, p4est, nodes);
    double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    bool xm_wall = is_node_xmWall(p4est, ni);
    bool xp_wall = is_node_xpWall(p4est, ni);
    bool ym_wall = is_node_ymWall(p4est, ni);
    bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
    bool zm_wall = is_node_zmWall(p4est, ni);
    bool zp_wall = is_node_zpWall(p4est, ni);
#endif

    double xm_grid = x_C, xp_grid = x_C, xm_cube = x_C, xp_cube = x_C; int nx_grid = 0;
    double ym_grid = y_C, yp_grid = y_C, ym_cube = y_C, yp_cube = y_C; int ny_grid = 0;
#ifdef P4_TO_P8
    double zm_grid = z_C, zp_grid = z_C, zm_cube = z_C, zp_cube = z_C; int nz_grid = 0;
#endif

    if (!xm_wall) {xm_cube -= 0.5*dx_min; xm_grid -= dx_min; nx_grid++;}
    if (!xp_wall) {xp_cube += 0.5*dx_min; xp_grid += dx_min; nx_grid++;}
    if (!ym_wall) {ym_cube -= 0.5*dy_min; ym_grid -= dy_min; ny_grid++;}
    if (!yp_wall) {yp_cube += 0.5*dy_min; yp_grid += dy_min; ny_grid++;}
#ifdef P4_TO_P8
    if (!zm_wall) {zm_cube -= 0.5*dz_min; zm_grid -= dz_min; nz_grid++;}
    if (!zp_wall) {zp_cube += 0.5*dz_min; zp_grid += dz_min; nz_grid++;}
#endif

    // count neighbors
    get_all_neighbors(n, neighbors, neighbor_exists);

    for (short i = 0; i < nx_grid+1; i++) x_grid[i] = xm_grid + (double)(i)*dx_min;
    for (short i = 0; i < ny_grid+1; i++) y_grid[i] = ym_grid + (double)(i)*dy_min;
#ifdef P4_TO_P8
    for (short i = 0; i < nz_grid+1; i++) z_grid[i] = zm_grid + (double)(i)*dz_min;
#endif

    interp_local.initialize(n);

    double phi_eff_n = phi_eff_p[n];

    // fetch values of LSF
    for (int i_phi = 0; i_phi < n_phis; i_phi++)
    {
#ifdef P4_TO_P8
        interp_local.set_input(phi_p[i_phi], phi_xx_p[i_phi], phi_yy_p[i_phi], phi_zz_p[i_phi], quadratic);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_xx_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_xx_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_yy_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_yy_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_zz_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_zz_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);
#else
        interp_local.set_input(phi_p[i_phi], phi_xx_p[i_phi], phi_yy_p[i_phi], quadratic);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);

        interp_local.set_input(phi_xx_p[i_phi], linear);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_xx_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);

        interp_local.set_input(phi_yy_p[i_phi], linear);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_yy_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);
#endif
//      int k = 0;
//      for (int i = 0; i < N_NBRS_MAX; i++)
//        if (neighbor_exists[i])
//        {
//          phi_cube[i_phi][k] = phi_p[i_phi][neighbors[i]];
//          phi_xx_cube[i_phi][k] = phi_xx_p[i_phi][neighbors[i]];
//          phi_yy_cube[i_phi][k] = phi_yy_p[i_phi][neighbors[i]];
//#ifdef P4_TO_P8
//          phi_zz_cube[i_phi][k] = phi_zz_p[i_phi][neighbors[i]];
//#endif
//          k++;
//        }
    }

    cube.x0 = xm_cube; cube.x1 = xp_cube;
    cube.y0 = ym_cube; cube.y1 = yp_cube;
#ifdef P4_TO_P8
    cube.z0 = zm_cube; cube.z1 = zp_cube;
#endif

#ifdef P4_TO_P8
    cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, phi_zz_cube, *action, *color);
    cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, zm_grid, zp_grid, nx_grid, ny_grid, nz_grid);
    double cell_vol = (xp_cube-xm_cube)*(yp_cube-ym_cube)*(zp_cube-zm_cube);
//    cube.construct_domain(1,1,1,0);
#else
    cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, *action, *color);
    cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, nx_grid, ny_grid);
    double cell_vol = (xp_cube-xm_cube)*(yp_cube-ym_cube);
//    cube.construct_domain(1,1,4);
#endif
    cube.construct_domain();

    node_vol_p[n] = cube.measure_of_domain()/cell_vol;
//    node_vol_p[n] = cube.measure_of_domain();

  }

  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecRestoreArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(phi_eff,   &phi_eff_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(node_vol, &node_vol_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(node_vol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (node_vol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}


void my_p4est_poisson_nodes_mls_t::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_node_offset[p4est->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes->num_owned_indeps);

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  /* preallocate space for matrix
   * This is done by computing the exact number of neighbors that will be used
   * to discretize Poisson equation which means it will adapt to T-junction
   * whenever necessary so it should have a fairly good estimate of non-zeros
   *
   * Note that this method overpredicts the actual number of non-zeros since
   * it assumes the PDE is discretized at boundary points and also points in
   * \Omega^+ that are within diag_min distance away from interface. For a
   * simple test (circle) this resulted in memory allocation for about 15%
   * extra points. Note that this does not mean there are actually 15% more
   * nonzeros, but simply that many more bytes are allocated and thus wasted.
   * This number will be smaller if there are small cells not only near the
   * interface but also inside the domain. (Note that use of worst-case estimate
   *, i.e d_nz = o_nz = 9 in 2D, in this case resulted in about 450% extra
   * memory consumption. So, getting 15% here with a simple change is a good
   * compromise here!)
   *
   * If this is still too much memory consumption, the ultimate choice is save
   * results in intermediate arrays and only allocate as much space as needed.
   * This is left for future optimizations if necessary.
   */
  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);
  double *phi_eff_p;
  ierr = VecGetArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  for (p4est_locidx_t n=0; n<num_owned_local; n++)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors->get_neighbors(n);

    /*
     * Check for neighboring nodes:
     * 1) If they exist and are local nodes, increment d_nnz[n]
     * 2) If they exist but are not local nodes, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */

    if (phi_eff_p[n] > 1.5*diag_min)
      continue;

//    if (node_loc[n] == NODE_OUT || node_loc[n] == NODE_MXO) continue;

#ifdef P4_TO_P8
    if (qnnn.d_m00_p0*qnnn.d_m00_0p != 0) // node_m00_mm will enter discretization
      qnnn.node_m00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_m0*qnnn.d_m00_0p != 0) // node_m00_pm will enter discretization
      qnnn.node_m00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_p0*qnnn.d_m00_0m != 0) // node_m00_mp will enter discretization
      qnnn.node_m00_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_m0*qnnn.d_m00_0m != 0) // node_m00_pp will enter discretization
      qnnn.node_m00_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_m00_p0 != 0) // node_m00_mm will enter discretization
      qnnn.node_m00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_m0 != 0) // node_m00_pm will enter discretization
      qnnn.node_m00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_p00_p0*qnnn.d_p00_0p != 0) // node_p00_mm will enter discretization
      qnnn.node_p00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_m0*qnnn.d_p00_0p != 0) // node_p00_pm will enter discretization
      qnnn.node_p00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_p0*qnnn.d_p00_0m != 0) // node_p00_mp will enter discretization
      qnnn.node_p00_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_m0*qnnn.d_p00_0m != 0) // node_p00_pp will enter discretization
      qnnn.node_p00_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_p00_p0 != 0) // node_p0_m will enter discretization
      qnnn.node_p00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_m0 != 0) // node_p0_p will enter discretization
      qnnn.node_p00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_0m0_p0*qnnn.d_0m0_0p != 0) // node_0m0_mm will enter discretization
      qnnn.node_0m0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_m0*qnnn.d_0m0_0p != 0) // node_0m0_pm will enter discretization
      qnnn.node_0m0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_p0*qnnn.d_0m0_0m != 0) // node_0m0_mp will enter discretization
      qnnn.node_0m0_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_m0*qnnn.d_0m0_0m != 0) // node_0m0_pp will enter discretization
      qnnn.node_0m0_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_0m0_p0 != 0) // node_0m_m will enter discretization
      qnnn.node_0m0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_m0 != 0) // node_0m_p will enter discretization
      qnnn.node_0m0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_0p0_p0*qnnn.d_0p0_0p != 0) // node_0p0_mm will enter discretization
      qnnn.node_0p0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_m0*qnnn.d_0p0_0p != 0) // node_0p0_pm will enter discretization
      qnnn.node_0p0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_p0*qnnn.d_0p0_0m != 0) // node_0p0_mp will enter discretization
      qnnn.node_0p0_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_m0*qnnn.d_0p0_0m != 0) // node_0p0_pp will enter discretization
      qnnn.node_0p0_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_0p0_p0 != 0) // node_0p_m will enter discretization
      qnnn.node_0p0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_m0 != 0) // node_0p_p will enter discretization
      qnnn.node_0p0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_00m_p0*qnnn.d_00m_0p != 0) // node_00m_mm will enter discretization
      qnnn.node_00m_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00m_m0*qnnn.d_00m_0p != 0) // node_00m_pm will enter discretization
      qnnn.node_00m_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00m_p0*qnnn.d_00m_0m != 0) // node_00m_mp will enter discretization
      qnnn.node_00m_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00m_m0*qnnn.d_00m_0m != 0) // node_00m_pp will enter discretization
      qnnn.node_00m_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;

    if (qnnn.d_00p_p0*qnnn.d_00p_0p != 0) // node_00p_mm will enter discretization
      qnnn.node_00p_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00p_m0*qnnn.d_00p_0p != 0) // node_00p_pm will enter discretization
      qnnn.node_00p_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00p_p0*qnnn.d_00p_0m != 0) // node_00p_mp will enter discretization
      qnnn.node_00p_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00p_m0*qnnn.d_00p_0m != 0) // node_00p_pp will enter discretization
      qnnn.node_00p_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

//    if (phi_eff_p[n] > -3.*diag_min)
//    {
//      d_nnz[n] += 9;
//      o_nnz[n] += 9;
//    }

  }

  ierr = VecRestoreArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
//  if(bc_ == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
    if (sol_size != nodes->num_owned_indeps){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "solution.local_size = " << sol_size << " nodes->num_owned_indeps = " << nodes->num_owned_indeps << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

//  // set local add if none was given
//  bool local_add = false;
//  if(add_ == NULL)
//  {
//    local_add = true;
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add_); CHKERRXX(ierr);
//    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
//  }

  // set a local phi if not was given
//  bool local_phi = false;
//  if(phi_ == NULL)
//  {
//    local_phi = true;
//    ierr = VecDuplicate(solution, &phi_); CHKERRXX(ierr);

//    Vec tmp;
//    ierr = VecGhostGetLocalForm(phi_, &tmp); CHKERRXX(ierr);
//    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(phi_, &tmp); CHKERRXX(ierr);
////    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
//    set_phi(phi_);
//  }

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  int method = 0;

  if (!is_matrix_computed)
  {
    matrix_has_nullspace = true;

    switch (method){
    case 0: setup_negative_laplace_matrix_sym(); break;
    }

    is_matrix_computed = true;

    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
    ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);  CHKERRXX(ierr);
  }

  // set pc type
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

//    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
//    if (matrix_has_nullspace){
//      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
//    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
  switch (method){
  case 0: setup_negative_laplace_rhsvec_sym(); break;
  }

  // Solve the system
  ierr = KSPSetTolerances(ksp, 1e-14, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs, solution, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs, solution, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  // get rid of local stuff
//  if(local_add)
//  {
//    ierr = VecDestroy(add_); CHKERRXX(ierr);
//    add_ = NULL;
//  }
//  if(local_phi)
//  {
//    ierr = VecDestroy(phi_); CHKERRXX(ierr);
//    phi_ = NULL;

//    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
//    phi_xx_ = NULL;

//    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
//    phi_yy_ = NULL;

//#ifdef P4_TO_P8
//    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
//    phi_zz_ = NULL;
//#endif
//  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::inv_mat3(double *in, double *out)
{
  double det = in[3*0+0]*(in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2]) -
               in[3*0+1]*(in[3*1+0]*in[3*2+2] - in[3*1+2]*in[3*2+0]) +
               in[3*0+2]*(in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1]);

  out[3*0+0] = (in[3*1+1]*in[3*2+2] - in[3*2+1]*in[3*1+2])/det;
  out[3*0+1] = (in[3*0+2]*in[3*2+1] - in[3*2+2]*in[3*0+1])/det;
  out[3*0+2] = (in[3*0+1]*in[3*1+2] - in[3*1+1]*in[3*0+2])/det;

  out[3*1+0] = (in[3*1+2]*in[3*2+0] - in[3*2+2]*in[3*1+0])/det;
  out[3*1+1] = (in[3*0+0]*in[3*2+2] - in[3*2+0]*in[3*0+2])/det;
  out[3*1+2] = (in[3*0+2]*in[3*1+0] - in[3*1+2]*in[3*0+0])/det;

  out[3*2+0] = (in[3*1+0]*in[3*2+1] - in[3*2+0]*in[3*1+1])/det;
  out[3*2+1] = (in[3*0+1]*in[3*2+0] - in[3*2+1]*in[3*0+0])/det;
  out[3*2+2] = (in[3*0+0]*in[3*1+1] - in[3*1+0]*in[3*0+1])/det;
}

void my_p4est_poisson_nodes_mls_t::inv_mat2(double *in, double *out)
{
  double det = in[0]*in[3]-in[1]*in[2];
  out[0] =  in[3]/det;
  out[1] = -in[1]/det;
  out[2] = -in[2]/det;
  out[3] =  in[0]/det;
}

void my_p4est_poisson_nodes_mls_t::setup_negative_laplace_matrix_sym()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // get access to LSFs
  //---------------------------------------------------------------------
  std::vector<double *> phi_p (n_phis, NULL);
  std::vector<double *> phi_xx_p (n_phis, NULL);
  std::vector<double *> phi_yy_p (n_phis, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_zz_p (n_phis, NULL);
#endif

  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecGetArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  double *phi_eff_p;
  ierr = VecGetArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // allocate vectors for LSF values for a cube
  //---------------------------------------------------------------------
  std::vector< std::vector<double> > phi_cube(n_phis, std::vector<double> (N_NBRS_MAX, -1));

  std::vector< std::vector<double> > phi_xx_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
  std::vector< std::vector<double> > phi_yy_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#ifdef P4_TO_P8
  std::vector< std::vector<double> > phi_zz_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#endif

  //---------------------------------------------------------------------
  // create a cube
  //---------------------------------------------------------------------
#ifdef P4_TO_P8
  cube3_mls_t cube;
#else
  cube2_mls_t cube;
#endif

//#ifdef P4_TO_P8
//    cube3_refined_mls_t cube;
//#else
//    cube2_refined_mls_t cube;
//#endif

  //---------------------------------------------------------------------
  // initialize quantities
  //---------------------------------------------------------------------
  mu.initialize();
  diag_add.initialize();
  for (int i = 0; i < n_phis; i++)
  {
    bc_coeffs[i].initialize();
    bc_values[i].initialize();
  }

  //---------------------------------------------------------------------
  // some additional variables
  //---------------------------------------------------------------------
  double *scalling_p;
  if (keep_scalling)
  {
    if (scalling == NULL)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &scalling); CHKERRXX(ierr);
    }
    ierr = VecGetArray(scalling, &scalling_p); CHKERRXX(ierr);
  }

  bool neighbor_exists[N_NBRS_MAX];
  p4est_locidx_t neighbors[N_NBRS_MAX];

  double integrand[N_NBRS_MAX];
  double dxyz_pr[P4EST_DIM];
  double xyz_pr[P4EST_DIM];
  double xyz_c[P4EST_DIM];
  double xyz_isxn[P4EST_DIM];
  double dist;
  double measure_of_iface;
  double measure_of_cut_cell;
  double mu_avg, bc_coeff_avg;

  double x_grid[3], y_grid[3];
  int nx_grid, ny_grid;
#ifdef P4_TO_P8
  double z_grid[3];
  int nz_grid;
#endif


  bool enforce_dirichlet_at_wall;

  node_loc.resize(nodes->num_owned_indeps, NODE_INS);

  //---------------------------------------------------------------------
  // main loop over nodes
  //---------------------------------------------------------------------
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++)
  {
    if      (phi_eff_p[n] >  2.0*diag_min)  node_loc[n] = NODE_OUT;
    else if (phi_eff_p[n] < -2.0*diag_min)  node_loc[n] = NODE_INS;
    else                                    node_loc[n] = NODE_NMN;

    double x_C  = node_x_fr_n(n, p4est, nodes); xyz_c[0] = x_C;
    double y_C  = node_y_fr_n(n, p4est, nodes); xyz_c[1] = y_C;
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes); xyz_c[2] = z_C;
#endif

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
#endif

    /*
     * NOTE: All nodes are in PETSc' local numbering
     */
    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
#endif

    p4est_locidx_t node_m00_mm=qnnn.node_m00_mm; p4est_locidx_t node_m00_pm=qnnn.node_m00_pm;
    p4est_locidx_t node_p00_mm=qnnn.node_p00_mm; p4est_locidx_t node_p00_pm=qnnn.node_p00_pm;
    p4est_locidx_t node_0m0_mm=qnnn.node_0m0_mm; p4est_locidx_t node_0m0_pm=qnnn.node_0m0_pm;
    p4est_locidx_t node_0p0_mm=qnnn.node_0p0_mm; p4est_locidx_t node_0p0_pm=qnnn.node_0p0_pm;
#ifdef P4_TO_P8
    p4est_locidx_t node_m00_mp=qnnn.node_m00_mp; p4est_locidx_t node_m00_pp=qnnn.node_m00_pp;
    p4est_locidx_t node_p00_mp=qnnn.node_p00_mp; p4est_locidx_t node_p00_pp=qnnn.node_p00_pp;
    p4est_locidx_t node_0m0_mp=qnnn.node_0m0_mp; p4est_locidx_t node_0m0_pp=qnnn.node_0m0_pp;
    p4est_locidx_t node_0p0_mp=qnnn.node_0p0_mp; p4est_locidx_t node_0p0_pp=qnnn.node_0p0_pp;

    p4est_locidx_t node_00m_mm=qnnn.node_00m_mm; p4est_locidx_t node_00m_mp=qnnn.node_00m_mp;
    p4est_locidx_t node_00m_pm=qnnn.node_00m_pm; p4est_locidx_t node_00m_pp=qnnn.node_00m_pp;
    p4est_locidx_t node_00p_mm=qnnn.node_00p_mm; p4est_locidx_t node_00p_mp=qnnn.node_00p_mp;
    p4est_locidx_t node_00p_pm=qnnn.node_00p_pm; p4est_locidx_t node_00p_pp=qnnn.node_00p_pp;
#endif

    PetscInt node_000_g = petsc_gloidx[qnnn.node_000];

    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // reconstruct interfaces
    //---------------------------------------------------------------------
    if (node_loc[n] == NODE_NMN)
    {
      // check if the node is a wall node
      bool xm_wall = is_node_xmWall(p4est, ni);
      bool xp_wall = is_node_xpWall(p4est, ni);
      bool ym_wall = is_node_ymWall(p4est, ni);
      bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
      bool zm_wall = is_node_zmWall(p4est, ni);
      bool zp_wall = is_node_zpWall(p4est, ni);
#endif

      // fetch dimensions of the cube and underlying interpolation grid
      /* at this moment a cube uses values of functions at nodes of computational grid for interpolation, reasons:
       * 1) to avoid double interpolation (first from nodes to vertices of a cube and then from vertices of the cube to points inside the cube)
       * 2) this would be very helpful if the procedure to remove small volume cells by Voronoi partition is to be implemented later
       */
      double xm_grid = x_C, xp_grid = x_C, xm_cube = x_C, xp_cube = x_C; nx_grid = 0;
      double ym_grid = y_C, yp_grid = y_C, ym_cube = y_C, yp_cube = y_C; ny_grid = 0;
#ifdef P4_TO_P8
      double zm_grid = z_C, zp_grid = z_C, zm_cube = z_C, zp_cube = z_C; nz_grid = 0;
#endif

      if (!xm_wall) {xm_cube -= 0.5*dx_min; xm_grid -= dx_min; nx_grid++;}
      if (!xp_wall) {xp_cube += 0.5*dx_min; xp_grid += dx_min; nx_grid++;}
      if (!ym_wall) {ym_cube -= 0.5*dy_min; ym_grid -= dy_min; ny_grid++;}
      if (!yp_wall) {yp_cube += 0.5*dy_min; yp_grid += dy_min; ny_grid++;}
#ifdef P4_TO_P8
      if (!zm_wall) {zm_cube -= 0.5*dz_min; zm_grid -= dz_min; nz_grid++;}
      if (!zp_wall) {zp_cube += 0.5*dz_min; zp_grid += dz_min; nz_grid++;}
#endif

      // count neighbors
//      get_all_neighbors(n, neighbors, neighbor_exists);

      for (short i = 0; i < nx_grid+1; i++) x_grid[i] = xm_grid + (double)(i)*dx_min;
      for (short i = 0; i < ny_grid+1; i++) y_grid[i] = ym_grid + (double)(i)*dy_min;
#ifdef P4_TO_P8
      for (short i = 0; i < nz_grid+1; i++) z_grid[i] = zm_grid + (double)(i)*dz_min;
#endif

      interp_local.initialize(n);

      // fetch values of LSF for the small uniform grid that is used by a cube for interpolation
      /* Looks terrible! In fact, most of the values are values at real nodes (so no interpolation),
       * only for missing nodes, when a neighboring quadrant has a smaller level, interpolation takes places.
       * Another option would be just to fetch values from pointers to vecs, but it might be potentially dangerous
       * (in case of a not_quite_uniform grid next to interface)
       */
      for (int i_phi = 0; i_phi < n_phis; i_phi++)
      {
#ifdef P4_TO_P8
        interp_local.set_input(phi_p[i_phi], phi_xx_p[i_phi], phi_yy_p[i_phi], phi_zz_p[i_phi], quadratic);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_xx_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_xx_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_yy_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_yy_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_zz_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_zz_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);
#else
        interp_local.set_input(phi_p[i_phi], phi_xx_p[i_phi], phi_yy_p[i_phi], quadratic);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);

        interp_local.set_input(phi_xx_p[i_phi], linear);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_xx_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);

        interp_local.set_input(phi_yy_p[i_phi], linear);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_yy_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);
#endif
//        sample_vec_at_neighbors(phi_p[i_phi], neighbors, neighbor_exists, phi_cube[i_phi]);
//        sample_vec_at_neighbors(phi_xx_p[i_phi], neighbors, neighbor_exists, phi_xx_cube[i_phi]);
//        sample_vec_at_neighbors(phi_yy_p[i_phi], neighbors, neighbor_exists, phi_yy_cube[i_phi]);
//#ifdef P4_TO_P8
//        sample_vec_at_neighbors(phi_zz_p[i_phi], neighbors, neighbor_exists, phi_zz_cube[i_phi]);
//#endif
      }

      cube.x0 = xm_cube; cube.x1 = xp_cube;
      cube.y0 = ym_cube; cube.y1 = yp_cube;
#ifdef P4_TO_P8
      cube.z0 = zm_cube; cube.z1 = zp_cube;
#endif

#ifdef P4_TO_P8
      cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, phi_zz_cube, *action, *color);
      cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, zm_grid, zp_grid, nx_grid, ny_grid, nz_grid);
//    cube.construct_domain(1,1,1,cube_refinement);
#else
      cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, *action, *color);
      cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, nx_grid, ny_grid);
//    cube.construct_domain(1,1,cube_refinement);
#endif

      cube.construct_domain();

      if (cube.measure_of_domain() < eps_dom) node_loc[n] = NODE_OUT;
      if (cube.measure_of_interface(-1) < eps_ifc && cube.measure_of_domain() > 0.5*vol_min) node_loc[n] = NODE_INS;

      enforce_dirichlet_at_wall = false;
      if      (xm_wall && cube.measure_in_dir(dir::f_m00) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (xp_wall && cube.measure_in_dir(dir::f_p00) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (ym_wall && cube.measure_in_dir(dir::f_0m0) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (yp_wall && cube.measure_in_dir(dir::f_0p0) > eps_ifc) enforce_dirichlet_at_wall = true;
#ifdef P4_TO_P8
      else if (zm_wall && cube.measure_in_dir(dir::f_00m) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (zp_wall && cube.measure_in_dir(dir::f_00p) > eps_ifc) enforce_dirichlet_at_wall = true;
#endif
    }

//    std::vector<double> phi_000(n_phi, 0), phi_p00(n_phi, 0), phi_m00(n_phi, 0), phi_0m0(n_phi, 0), phi_0p0(n_phi, 0);
//#ifdef P4_TO_P8
//    std::vector<double> phi_00m(n_phi, 0), phi_00p(n_phi, 0);
//#endif

//    for (int i = 0; i < n_phi; i++)
//    {
//#ifdef P4_TO_P8
//      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i], phi_00m[i], phi_00p[i]);
//#else
//      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i]);
//#endif
//    }

    if (is_node_Wall(p4est, ni) && (enforce_dirichlet_at_wall || node_loc[n] == NODE_INS))
    {
      ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
      if (is_inside(n)) matrix_has_nullspace = false;
      continue;
    } else {

      switch (node_loc[n])
      {
      // only Robin BC at the moment
      case NODE_OUT: ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr); break;
      case NODE_INS:
      {
        bool is_interface_m00 = false;
        bool is_interface_p00 = false;
        bool is_interface_0m0 = false;
        bool is_interface_0p0 = false;
#ifdef P4_TO_P8
        bool is_interface_00m = false;
        bool is_interface_00p = false;
#endif

#ifdef P4_TO_P8
        //------------------------------------
        // Dfxx =   fxx + a*fyy + b*fzz
        // Dfyy = c*fxx +   fyy + d*fzz
        // Dfzz = e*fxx + f*fyy +   fzz
        //------------------------------------
        double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
        double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

        double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
        double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

        double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
        double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

        //------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //
        // Laplace = wi*Dfxx +
        //           wj*Dfyy +
        //           wk*Dfzz
        //------------------------------------------------------------
        double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
        double wi = (1.-c-e+c*f+e*d-d*f)/det;
        double wj = (1.-a-f+a*e+f*b-b*e)/det;
        double wk = (1.-b-d+b*c+d*a-a*c)/det;

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

        if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
        else                               w_m00 += -2./d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
        else                               w_p00 += -2./d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
        else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
        else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

        if(is_node_zmWall(p4est, ni))      w_00p += -1./(d_00p*d_00p);
        else if(is_node_zpWall(p4est, ni)) w_00m += -1./(d_00m*d_00m);
        else                               w_00m += -2./d_00m/(d_00m+d_00p);

        if(is_node_zpWall(p4est, ni))      w_00m += -1./(d_00m*d_00m);
        else if(is_node_zmWall(p4est, ni)) w_00p += -1./(d_00p*d_00p);
        else                               w_00p += -2./d_00p/(d_00m+d_00p);

        // FIX FOR VARIABLE mu
        w_m00 *= wi * mu.val; w_p00 *= wi * mu.val;
        w_0m0 *= wj * mu.val; w_0p0 *= wj * mu.val;
        w_00m *= wk * mu.val; w_00p *= wk * mu.val;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        double w_000 = diag_add(n) - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
        w_00m /= w_000; w_00p /= w_000;

        //---------------------------------------------------------------------
        // add coefficients in the matrix
        //---------------------------------------------------------------------
        if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00)
        {
          double w_m00_mm = w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          double w_m00_mp = w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          double w_m00_pm = w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          double w_m00_pp = w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);

          if (w_m00_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mm], w_m00_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mp], w_m00_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pm], w_m00_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pp], w_m00_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_p00)
        {
          double w_p00_mm = w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          double w_p00_mp = w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          double w_p00_pm = w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          double w_p00_pp = w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);

          if (w_p00_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mm], w_p00_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mp], w_p00_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pm], w_p00_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pp], w_p00_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_0m0)
        {
          double w_0m0_mm = w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          double w_0m0_mp = w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          double w_0m0_pm = w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          double w_0m0_pp = w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);

          if (w_0m0_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mm], w_0m0_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mp], w_0m0_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pm], w_0m0_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pp], w_0m0_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_0p0)
        {
          double w_0p0_mm = w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          double w_0p0_mp = w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          double w_0p0_pm = w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          double w_0p0_pp = w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);

          if (w_0p0_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mm], w_0p0_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mp], w_0p0_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pm], w_0p0_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pp], w_0p0_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_00m)
        {
          double w_00m_mm = w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          double w_00m_mp = w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          double w_00m_pm = w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          double w_00m_pp = w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);

          if (w_00m_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mm], w_00m_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mp], w_00m_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pm], w_00m_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pp], w_00m_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_00p)
        {
          double w_00p_mm = w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          double w_00p_mp = w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          double w_00p_pm = w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          double w_00p_pp = w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);

          if (w_00p_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mm], w_00p_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mp], w_00p_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pm], w_00p_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pp], w_00p_pp, ADD_VALUES); CHKERRXX(ierr);}
        }
#else
        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

        if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
        else                               w_m00 += -2./d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
        else                               w_p00 += -2./d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
        else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
        else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

        //---------------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //---------------------------------------------------------------------
        double weight_on_Dyy = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);
        double weight_on_Dxx = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);

        // FIX THIS FOR VARIABLE mu
        w_m00 *= weight_on_Dxx*mu.constant;
        w_p00 *= weight_on_Dxx*mu.constant;
        w_0m0 *= weight_on_Dyy*mu.constant;
        w_0p0 *= weight_on_Dyy*mu.constant;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

        double w_000 = sample_qty(diag_add, n, xyz_c)-(w_m00+w_p00+w_0m0+w_0p0);
        w_m00 /= w_000;
        w_p00 /= w_000;
        w_0m0 /= w_000;
        w_0p0 /= w_000;

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00 && !is_node_xmWall(p4est, ni)) {
          PetscInt node_m00_pm_g = petsc_gloidx[node_m00_pm];
          PetscInt node_m00_mm_g = petsc_gloidx[node_m00_mm];

          if (d_m00_m0 != 0) ierr = MatSetValue(A, node_000_g, node_m00_pm_g, w_m00*d_m00_m0/(d_m00_m0+d_m00_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_m00_p0 != 0) ierr = MatSetValue(A, node_000_g, node_m00_mm_g, w_m00*d_m00_p0/(d_m00_m0+d_m00_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_p00 && !is_node_xpWall(p4est, ni)) {
          PetscInt node_p00_pm_g = petsc_gloidx[node_p00_pm];
          PetscInt node_p00_mm_g = petsc_gloidx[node_p00_mm];

          if (d_p00_m0 != 0) ierr = MatSetValue(A, node_000_g, node_p00_pm_g, w_p00*d_p00_m0/(d_p00_m0+d_p00_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_p00_p0 != 0) ierr = MatSetValue(A, node_000_g, node_p00_mm_g, w_p00*d_p00_p0/(d_p00_m0+d_p00_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_0m0 && !is_node_ymWall(p4est, ni)) {
          PetscInt node_0m0_pm_g = petsc_gloidx[node_0m0_pm];
          PetscInt node_0m0_mm_g = petsc_gloidx[node_0m0_mm];

          if (d_0m0_m0 != 0) ierr = MatSetValue(A, node_000_g, node_0m0_pm_g, w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_0m0_p0 != 0) ierr = MatSetValue(A, node_000_g, node_0m0_mm_g, w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_0p0 && !is_node_ypWall(p4est, ni)) {
          PetscInt node_0p0_pm_g = petsc_gloidx[node_0p0_pm];
          PetscInt node_0p0_mm_g = petsc_gloidx[node_0p0_mm];

          if (d_0p0_m0 != 0) ierr = MatSetValue(A, node_000_g, node_0p0_pm_g, w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_0p0_p0 != 0) ierr = MatSetValue(A, node_000_g, node_0p0_mm_g, w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0), ADD_VALUES); CHKERRXX(ierr);
        }
#endif

        if (keep_scalling) scalling_p[n] = w_000;
        if (sample_qty(diag_add, n, xyz_c)> 0) matrix_has_nullspace = false;
        continue;

      } break;

      case NODE_NMN:
      {
        measure_of_cut_cell = cube.measure_of_domain();

        double w_000 = 0.;

        //---------------------------------------------------------------------
        // contribution through interfaces
        //---------------------------------------------------------------------

        // count interfaces
        std::vector<int> present_ifaces;
        bool is_there_kink = false;
        for (int i_phi = 0; i_phi < n_phis; i_phi++)
        {
          measure_of_iface = cube.measure_of_interface(i_phi);
          if (bc_types->at(i_phi) == ROBIN && measure_of_iface > eps_ifc)
          {
            if (present_ifaces.size() > 0 and action->at(i_phi) != COLORATION) is_there_kink = true;
            present_ifaces.push_back(i_phi);
          }
        }

        int num_ifaces = present_ifaces.size();

        // check if there is really a kink
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            integrand[i + j*(nx_grid+1)] = 1.;

        int i0 = present_ifaces[0];
        int i1 = present_ifaces[1];
        double num_isxns = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

        if (num_isxns < 0.99) {is_there_kink = false;}


        if (is_there_kink && kink_special_treatment && num_ifaces == 2)
        {
          int i0 = present_ifaces[0];
          int i1 = present_ifaces[1];

//          // check if there is really a kink
//          for (short j = 0; j < ny_grid+1; j++)
//            for (short i = 0; i < nx_grid+1; i++)
//              integrand[i + j*(nx_grid+1)] = 1;
//          double num_isxns = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

//          if (num_isxns < 1.0) {is_there_kink = false; break;}

          double n0[2] = {0,0}, n1[2] = {0,0};
          double kappa[2] = {0,0}, g[2] = {0,0};
          double a_coeff[2] = {0,0}, b_coeff[2] = {0,0};

          // calculate normals to interfaces
          compute_normal(phi_p[i0], qnnn, n0);
          compute_normal(phi_p[i1], qnnn, n1);

          // get coordinates of the intersection
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              integrand[i + j*(nx_grid+1)] = x_grid[i];
          xyz_isxn[0] = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              integrand[i + j*(nx_grid+1)] = y_grid[j];
          xyz_isxn[1] = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

          // sample robin coeff at the intersection
          kappa[0] = sample_qty(bc_coeffs[i0], xyz_isxn);
          kappa[1] = sample_qty(bc_coeffs[i1], xyz_isxn);

          // sample g at the intersection
          g[0] = sample_qty(bc_values[i0], xyz_isxn);
          g[1] = sample_qty(bc_values[i1], xyz_isxn);

          // solve matrix and find coefficients
          double N_mat[4] = {n0[0], n0[1], n1[0], n1[1]};
          double N_inv_mat[4];

          inv_mat2(N_mat, N_inv_mat);

          for (int i = 0; i < 2; i++){
            a_coeff[i] = N_inv_mat[i*2 + 0]*kappa[0] + N_inv_mat[i*2 + 1]*kappa[1];
            b_coeff[i] = N_inv_mat[i*2 + 0]*g[0]     + N_inv_mat[i*2 + 1]*g[1];
          }

          // compute integrals
          for (short i_present_iface = 0; i_present_iface < present_ifaces.size(); ++i_present_iface)
          {
            int i_phi = present_ifaces[i_present_iface];

            for (short j = 0; j < ny_grid+1; j++)
              for (short i = 0; i < nx_grid+1; i++)
              {
                double xyz[P4EST_DIM] = {x_grid[i], y_grid[j]};
                integrand[i + j*(nx_grid+1)] = sample_qty(bc_coeffs[i_phi], xyz)
                    *(1.0 - a_coeff[0]*(x_grid[i]-x_C) - a_coeff[1]*(y_grid[j]-y_C));
              }

            w_000 += cube.integrate_over_interface(integrand, color->at(i_phi));

          }

        }

        if (!is_there_kink || !kink_special_treatment) {

          /* In case of COLORATION we need some correction:
           * A LSF that is used for COLORATION doesn't give any information about geometrical properties of interfaces.
           * To find such quantites as the distance to an interface or the projection point
           * one has to refer to a LSF that WAS colorated (not the colorating LSF).
           * That's why in case of COLORATION we loop through all LSFs,
           * which the colorating LSF could colorate.
           */
          for (int i_present_iface = 0; i_present_iface < present_ifaces.size(); i_present_iface++)
          {
            short i_phi = present_ifaces[i_present_iface];
            if (bc_types->at(i_phi) == ROBIN)
            {
              int num_iterations;

              if (action->at(i_phi) == COLORATION)  num_iterations = i_phi;
              else                                  num_iterations = 1;

              for (int j_phi = 0; j_phi < num_iterations; j_phi++)
              {
                if (action->at(i_phi) == COLORATION)  measure_of_iface = cube.measure_of_colored_interface(j_phi, i_phi);
                else                                  measure_of_iface = cube.measure_of_interface(i_phi);

                if (measure_of_iface > eps_ifc)
                {
                  if (action->at(i_phi) == COLORATION)  find_projection(phi_p[j_phi], qnnn, dxyz_pr, dist);
                  else                                  find_projection(phi_p[i_phi], qnnn, dxyz_pr, dist);

                  for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
                    xyz_pr[i_dim] = xyz_c[i_dim] + dxyz_pr[i_dim];

                  mu_avg       = sample_qty(mu,               xyz_pr);
                  bc_coeff_avg = sample_qty(bc_coeffs[i_phi], xyz_pr);

                  if (use_taylor_correction) { w_000 += mu_avg*bc_coeff_avg*measure_of_iface/(mu_avg-bc_coeff_avg*dist); }
                  else                       { w_000 += bc_coeff_avg*measure_of_iface; }

                  if (fabs(bc_coeff_avg) > 0) matrix_has_nullspace = false;
                }
              }
            }
          }
        }


        //---------------------------------------------------------------------
        // contribution through cell faces
        //---------------------------------------------------------------------
        double s_m00 = cube.measure_in_dir(dir::f_m00);
        double s_p00 = cube.measure_in_dir(dir::f_p00);
        double s_0m0 = cube.measure_in_dir(dir::f_0m0);
        double s_0p0 = cube.measure_in_dir(dir::f_0p0);
#ifdef P4_TO_P8
        double s_00m = cube.measure_in_dir(dir::f_00m);
        double s_00p = cube.measure_in_dir(dir::f_00p);
#endif

        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;
#ifdef P4_TO_P8
        double w_00m=0, w_00p=0;
#endif

        if(!is_node_xmWall(p4est, ni)) w_m00 += -mu.constant * s_m00/dx_min;
        if(!is_node_xpWall(p4est, ni)) w_p00 += -mu.constant * s_p00/dx_min;
        if(!is_node_ymWall(p4est, ni)) w_0m0 += -mu.constant * s_0m0/dy_min;
        if(!is_node_ypWall(p4est, ni)) w_0p0 += -mu.constant * s_0p0/dy_min;
#ifdef P4_TO_P8
        if(!is_node_zmWall(p4est, ni)) w_00m += -mu.constant * s_00m/dz_min;
        if(!is_node_zpWall(p4est, ni)) w_00p += -mu.constant * s_00p/dz_min;
#endif

#ifdef P4_TO_P8
        w_000 += sample_qty(diag_add, n, xyz_c)*measure_of_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
        w_000 += sample_qty(diag_add, n, xyz_c)*measure_of_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0);
#endif

        //---------------------------------------------------------------------
        // insert values into the matrix
        //---------------------------------------------------------------------
#ifdef P4_TO_P8
        PetscInt node_m00_g = petsc_gloidx[qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                                            : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp) ];
        PetscInt node_p00_g = petsc_gloidx[qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                                            : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp) ];
        PetscInt node_0m0_g = petsc_gloidx[qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                                            : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp) ];
        PetscInt node_0p0_g = petsc_gloidx[qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                                            : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp) ];
        PetscInt node_00m_g = petsc_gloidx[qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                                            : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp) ];
        PetscInt node_00p_g = petsc_gloidx[qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                                            : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp) ];
#else
        PetscInt node_m00_g = petsc_gloidx[qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm];
        PetscInt node_p00_g = petsc_gloidx[qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm];
        PetscInt node_0m0_g = petsc_gloidx[qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm];
        PetscInt node_0p0_g = petsc_gloidx[qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm];
#endif

        // diag scaling
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
#ifdef P4_TO_P8
        w_00m /= w_000; w_00p /= w_000;
#endif

        if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }

        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0,   ADD_VALUES); CHKERRXX(ierr);

        if(ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00, ADD_VALUES); CHKERRXX(ierr);}
        if(ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00, ADD_VALUES); CHKERRXX(ierr);}
        if(ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0, ADD_VALUES); CHKERRXX(ierr);}
        if(ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
        if(ABS(w_00m) > EPS) {ierr = MatSetValue(A, node_000_g, node_00m_g, w_00m, ADD_VALUES); CHKERRXX(ierr);}
        if(ABS(w_00p) > EPS) {ierr = MatSetValue(A, node_000_g, node_00p_g, w_00p, ADD_VALUES); CHKERRXX(ierr);}
#endif

        if (sample_qty(diag_add, n, xyz_c) > 0) matrix_has_nullspace = false;
        if (keep_scalling) scalling_p[n] = w_000;

      } break;
      }

    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  if (keep_scalling)
  {
    ierr = VecRestoreArray(scalling, &scalling_p); CHKERRXX(ierr);
  }

  //---------------------------------------------------------------------
  // finalize quantities
  //---------------------------------------------------------------------
  mu.finalize();
  diag_add.finalize();
  for (int i = 0; i < n_phis; i++)
  {
    bc_coeffs[i].finalize();
    bc_values[i].finalize();
  }

  //---------------------------------------------------------------------
  // close access to LSFs
  //---------------------------------------------------------------------
  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecRestoreArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }
  ierr = VecRestoreArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // check for null space
  //---------------------------------------------------------------------
  MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm);
  if (matrix_has_nullspace) {
    ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
    p4est_gloidx_t fixed_value_idx;
    MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
    if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
      fixed_value_idx_l = -1;
      fixed_value_idx_g = fixed_value_idx;
    } else {
      // reset the value
      ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::setup_negative_laplace_rhsvec_sym()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // get access to LSFs
  //---------------------------------------------------------------------
  std::vector<double *> phi_p (n_phis, NULL);
  std::vector<double *> phi_xx_p (n_phis, NULL);
  std::vector<double *> phi_yy_p (n_phis, NULL);
#ifdef P4_TO_P8
  std::vector<double *> phi_zz_p (n_phis, NULL);
#endif

  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecGetArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecGetArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  double *phi_eff_p;
  ierr = VecGetArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // allocate vectors for LSF values for a cube
  //---------------------------------------------------------------------
  std::vector< std::vector<double> > phi_cube(n_phis, std::vector<double> (N_NBRS_MAX, -1));

  std::vector< std::vector<double> > phi_xx_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
  std::vector< std::vector<double> > phi_yy_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#ifdef P4_TO_P8
  std::vector< std::vector<double> > phi_zz_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#endif

  //---------------------------------------------------------------------
  // create a cube
  //---------------------------------------------------------------------
#ifdef P4_TO_P8
  cube3_mls_t cube;
#else
  cube2_mls_t cube;
#endif

//#ifdef P4_TO_P8
//    cube3_refined_mls_t cube;
//#else
//    cube2_refined_mls_t cube;
//#endif

  //---------------------------------------------------------------------
  // initialize quantities
  //---------------------------------------------------------------------
  mu.initialize();
  diag_add.initialize();
  wall_value.initialize();
  for (int i = 0; i < n_phis; i++)
  {
    bc_coeffs[i].initialize();
    bc_values[i].initialize();
  }

  double *scalling_p;
  if (keep_scalling) {
    ierr = VecGetArray(scalling, &scalling_p); CHKERRXX(ierr);
  }

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  //---------------------------------------------------------------------
  // some additional variables
  //---------------------------------------------------------------------
  bool neighbor_exists[N_NBRS_MAX];
  p4est_locidx_t neighbors[N_NBRS_MAX];

  double integrand[N_NBRS_MAX];
  double dxyz_pr[P4EST_DIM];
  double xyz_pr[P4EST_DIM];
  double xyz_c[P4EST_DIM];
  double xyz_isxn[P4EST_DIM];
  double dist;
  double measure_of_iface;
  double measure_of_cut_cell;
  double mu_avg, bc_value_avg, bc_coeff_avg;

  double x_grid[3], y_grid[3];
  int nx_grid, ny_grid;
#ifdef P4_TO_P8
  double z_grid[3];
  int nz_grid;
#endif

  bool enforce_dirichlet_at_wall;

  //---------------------------------------------------------------------
  // main loop over nodes
  //---------------------------------------------------------------------
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    double x_C  = node_x_fr_n(n, p4est, nodes); xyz_c[0] = x_C;
    double y_C  = node_y_fr_n(n, p4est, nodes); xyz_c[1] = y_C;
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes); xyz_c[2] = z_C;
#endif

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
#endif

    /*
     * NOTE: All nodes are in PETSc' local numbering
     */
    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
#endif

    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // reconstruct interfaces
    //---------------------------------------------------------------------
    if (node_loc[n] == NODE_NMN)
    {
      // check if the node is a wall node
      bool xm_wall = is_node_xmWall(p4est, ni);
      bool xp_wall = is_node_xpWall(p4est, ni);
      bool ym_wall = is_node_ymWall(p4est, ni);
      bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
      bool zm_wall = is_node_zmWall(p4est, ni);
      bool zp_wall = is_node_zpWall(p4est, ni);
#endif

      // fetch dimensions of the cube and underlying interpolation grid
      double xm_grid = x_C, xp_grid = x_C, xm_cube = x_C, xp_cube = x_C; nx_grid = 0;
      double ym_grid = y_C, yp_grid = y_C, ym_cube = y_C, yp_cube = y_C; ny_grid = 0;
#ifdef P4_TO_P8
      double zm_grid = z_C, zp_grid = z_C, zm_cube = z_C, zp_cube = z_C; nz_grid = 0;
#endif

      if (!xm_wall) {xm_cube -= 0.5*dx_min; xm_grid -= dx_min; nx_grid++;}
      if (!xp_wall) {xp_cube += 0.5*dx_min; xp_grid += dx_min; nx_grid++;}
      if (!ym_wall) {ym_cube -= 0.5*dy_min; ym_grid -= dy_min; ny_grid++;}
      if (!yp_wall) {yp_cube += 0.5*dy_min; yp_grid += dy_min; ny_grid++;}
#ifdef P4_TO_P8
      if (!zm_wall) {zm_cube -= 0.5*dz_min; zm_grid -= dz_min; nz_grid++;}
      if (!zp_wall) {zp_cube += 0.5*dz_min; zp_grid += dz_min; nz_grid++;}
#endif

      // count neighbors
//      get_all_neighbors(n, neighbors, neighbor_exists);

      interp_local.initialize(n);

      for (short i = 0; i < nx_grid+1; i++) x_grid[i] = xm_grid + (double)(i)*dx_min;
      for (short i = 0; i < ny_grid+1; i++) y_grid[i] = ym_grid + (double)(i)*dy_min;
#ifdef P4_TO_P8
      for (short i = 0; i < nz_grid+1; i++) z_grid[i] = zm_grid + (double)(i)*dz_min;
#endif

      // fetch values of LSF
      for (int i_phi = 0; i_phi < n_phis; i_phi++)
      {
#ifdef P4_TO_P8
        interp_local.set_input(phi_p[i_phi], phi_xx_p[i_phi], phi_yy_p[i_phi], phi_zz_p[i_phi], quadratic);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_xx_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_xx_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_yy_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_yy_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);

        interp_local.set_input(phi_zz_p[i_phi], linear);
        for (short k = 0; k < nz_grid+1; k++)
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              phi_zz_cube[i_phi][i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j], z_grid[k]);
#else
        interp_local.set_input(phi_p[i_phi], phi_xx_p[i_phi], phi_yy_p[i_phi], quadratic);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);

        interp_local.set_input(phi_xx_p[i_phi], linear);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_xx_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);

        interp_local.set_input(phi_yy_p[i_phi], linear);
        for (short j = 0; j < ny_grid+1; j++)
          for (short i = 0; i < nx_grid+1; i++)
            phi_yy_cube[i_phi][i + j*(nx_grid+1)] = interp_local.interpolate(x_grid[i], y_grid[j]);
#endif
//        sample_vec_at_neighbors(phi_p[i_phi], neighbors, neighbor_exists, phi_cube[i_phi]);
//        sample_vec_at_neighbors(phi_xx_p[i_phi], neighbors, neighbor_exists, phi_xx_cube[i_phi]);
//        sample_vec_at_neighbors(phi_yy_p[i_phi], neighbors, neighbor_exists, phi_yy_cube[i_phi]);
//#ifdef P4_TO_P8
//        sample_vec_at_neighbors(phi_zz_p[i_phi], neighbors, neighbor_exists, phi_zz_cube[i_phi]);
//#endif
      }

      cube.x0 = xm_cube; cube.x1 = xp_cube;
      cube.y0 = ym_cube; cube.y1 = yp_cube;
#ifdef P4_TO_P8
      cube.z0 = zm_cube; cube.z1 = zp_cube;
#endif

#ifdef P4_TO_P8
      cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, phi_zz_cube, *action, *color);
      cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, zm_grid, zp_grid, nx_grid, ny_grid, nz_grid);
//    cube.construct_domain(1,1,1,cube_refinement);
#else
      cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, *action, *color);
      cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, nx_grid, ny_grid);
//    cube.construct_domain(1,1,cube_refinement);
#endif

      cube.construct_domain();
      enforce_dirichlet_at_wall = false;
      if      (xm_wall && cube.measure_in_dir(dir::f_m00) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (xp_wall && cube.measure_in_dir(dir::f_p00) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (ym_wall && cube.measure_in_dir(dir::f_0m0) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (yp_wall && cube.measure_in_dir(dir::f_0p0) > eps_ifc) enforce_dirichlet_at_wall = true;
#ifdef P4_TO_P8
      else if (zm_wall && cube.measure_in_dir(dir::f_00m) > eps_ifc) enforce_dirichlet_at_wall = true;
      else if (zp_wall && cube.measure_in_dir(dir::f_00p) > eps_ifc) enforce_dirichlet_at_wall = true;
#endif
    }

//    std::vector<double> phi_000(n_phi, 0), phi_p00(n_phi, 0), phi_m00(n_phi, 0), phi_0m0(n_phi, 0), phi_0p0(n_phi, 0);
//#ifdef P4_TO_P8
//    std::vector<double> phi_00m(n_phi, 0), phi_00p(n_phi, 0);
//#endif

//    for (int i = 0; i < n_phi; i++)
//    {
//#ifdef P4_TO_P8
//      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i], phi_00m[i], phi_00p[i]);
//#else
//      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i]);
//#endif
//    }

    if (is_node_Wall(p4est, ni) && (enforce_dirichlet_at_wall || node_loc[n] == NODE_INS))
    {
      rhs_p[n] = sample_qty(wall_value, n, xyz_c);
      continue;
    } else {

      switch (node_loc[n])
      {
        case NODE_OUT:
        {
          rhs_p[n] = 0.;
        } break;

        case NODE_INS:
        {
          double w_000;
          if (keep_scalling)
          {
            w_000 = scalling_p[n];
          } else {
            // TODO: past stuff from matrix
#ifdef P4_TO_P8
#else
#endif
          }
          rhs_p[n] /= w_000;
        } break;

        case NODE_NMN:
        {
          measure_of_cut_cell = cube.measure_of_domain();

          // LHS
          double w_000 = 0;

          if (keep_scalling) {
            w_000 = scalling_p[n];
          } else {
            // TODO: copy-past stuff from matrix
#ifdef P4_TO_P8
#else
#endif
          }

          // RHS
          rhs_p[n] *= measure_of_cut_cell;

          // count interfaces
          std::vector<int> present_ifaces;
          bool is_there_kink = false;
          for (int i_phi = 0; i_phi < n_phis; i_phi++)
          {
            measure_of_iface = cube.measure_of_interface(i_phi);
            if (bc_types->at(i_phi) == ROBIN && measure_of_iface > eps_ifc)
            {
              if (present_ifaces.size() > 0 and action->at(i_phi) != COLORATION) is_there_kink = true;
              present_ifaces.push_back(i_phi);
            }
          }

          int num_ifaces = present_ifaces.size();

          // Neumann term
          for (int i_present_iface = 0; i_present_iface < present_ifaces.size(); i_present_iface++)
          {
            short i_phi = present_ifaces[i_present_iface];
            if (bc_types->at(i_phi) == ROBIN)
#ifdef P4_TO_P8
              for (short k = 0; k < nz_grid+1; k++)
#endif
                for (short j = 0; j < ny_grid+1; j++)
                  for (short i = 0; i < nx_grid+1; i++)
                  {
#ifdef P4_TO_P8
                    double xyz[P4EST_DIM] = {x_grid[i], y_grid[j], z_grid[k]};
                    int p = i + j*(nx_grid+1) + k*(nx_grid+1)*(ny_grid+1);
#else
                    double xyz[P4EST_DIM] = {x_grid[i], y_grid[j]};
                    int p = i + j*(nx_grid+1);
#endif
                    integrand[p] = sample_qty(bc_values[i_phi], xyz);
                  }
            rhs_p[n] += cube.integrate_over_interface(integrand, color->at(i_phi));
          }

          // check if there is really a kink
          for (short j = 0; j < ny_grid+1; j++)
            for (short i = 0; i < nx_grid+1; i++)
              integrand[i + j*(nx_grid+1)] = 1.;

          int i0 = present_ifaces[0];
          int i1 = present_ifaces[1];
          double num_isxns = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

          if (num_isxns < 0.99) {is_there_kink = false;}

          // Robin term
          if (is_there_kink && kink_special_treatment && num_ifaces == 2)
          {
            int i0 = present_ifaces[0];
            int i1 = present_ifaces[1];


            double n0[2] = {0,0}, n1[2] = {0,0};
            double kappa[2] = {0,0}, g[2] = {0,0};
            double a_coeff[2] = {0,0}, b_coeff[2] = {0,0};

            // calculate normals to interfaces
            compute_normal(phi_p[i0], qnnn, n0);
            compute_normal(phi_p[i1], qnnn, n1);

            // get coordinates of the intersection
            for (short j = 0; j < ny_grid+1; j++)
              for (short i = 0; i < nx_grid+1; i++)
                integrand[i + j*(nx_grid+1)] = x_grid[i];
            xyz_isxn[0] = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

            for (short j = 0; j < ny_grid+1; j++)
              for (short i = 0; i < nx_grid+1; i++)
                integrand[i + j*(nx_grid+1)] = y_grid[j];
            xyz_isxn[1] = cube.integrate_over_intersection(integrand, color->at(i0), color->at(i1));

            // sample robin coeff at the intersection
            kappa[0] = sample_qty(bc_coeffs[i0], xyz_isxn);
            kappa[1] = sample_qty(bc_coeffs[i1], xyz_isxn);

            // sample g at the intersection
            g[0] = sample_qty(bc_values[i0], xyz_isxn);
            g[1] = sample_qty(bc_values[i1], xyz_isxn);

            // solve matrix and find coefficients
            double N_mat[4] = {n0[0], n0[1], n1[0], n1[1]};
            double N_inv_mat[4];

            inv_mat2(N_mat, N_inv_mat);

            for (int i = 0; i < 2; i++){
              a_coeff[i] = N_inv_mat[i*2 + 0]*kappa[0] + N_inv_mat[i*2 + 1]*kappa[1];
              b_coeff[i] = N_inv_mat[i*2 + 0]*g[0]     + N_inv_mat[i*2 + 1]*g[1];
            }

            // compute integrals
            for (short i_present_iface = 0; i_present_iface < present_ifaces.size(); ++i_present_iface)
            {
              int i_phi = present_ifaces[i_present_iface];

              for (short j = 0; j < ny_grid+1; j++)
                for (short i = 0; i < nx_grid+1; i++)
                {
                  double xyz[P4EST_DIM] = {x_grid[i], y_grid[j]};
                  integrand[i + j*(nx_grid+1)] = sample_qty(bc_coeffs[i_phi], xyz)
                      *(b_coeff[0]*(x_grid[i]-x_C) + b_coeff[1]*(y_grid[j]-y_C));
                }

              rhs_p[n] -= cube.integrate_over_interface(integrand, color->at(i_phi));
            }
          }

          if (use_taylor_correction && (!is_there_kink || !kink_special_treatment)) {

            /* In case of COLORATION we need some correction:
             * A LSF that is used for COLORATION doesn't give any information about geometrical properties of interfaces.
             * To find such quantites as the distance to an interface or the projection point
             * one has to refer to a LSF that WAS colorated (not the colorating LSF).
             * That's why in case of COLORATION we loop through all LSFs,
             * which the colorating LSF could colorate.
             */

            for (int i_present_iface = 0; i_present_iface < present_ifaces.size(); i_present_iface++)
            {
              short i_phi = present_ifaces[i_present_iface];
              if (bc_types->at(i_phi) == ROBIN)
              {
                int num_iterations;

                if (action->at(i_phi) == COLORATION)  num_iterations = i_phi;
                else                                  num_iterations = 1;

                for (int j_phi = 0; j_phi < num_iterations; j_phi++)
                {
                  if (action->at(i_phi) == COLORATION)  measure_of_iface = cube.measure_of_colored_interface(j_phi, i_phi);
                  else                                  measure_of_iface = cube.measure_of_interface(i_phi);

                  if (measure_of_iface > eps_ifc)
                  {
                    if (action->at(i_phi) == COLORATION)  find_projection(phi_p[j_phi], qnnn, dxyz_pr, dist);
                    else                                  find_projection(phi_p[i_phi], qnnn, dxyz_pr, dist);

                    for (short i_dim = 0; i_dim < P4EST_DIM; i_dim++)
                      xyz_pr[i_dim] = xyz_c[i_dim] + dxyz_pr[i_dim];

                    mu_avg       = sample_qty(mu,               xyz_pr);
                    bc_coeff_avg = sample_qty(bc_coeffs[i_phi], xyz_pr);
                    bc_value_avg = sample_qty(bc_values[i_phi], xyz_pr);

                    rhs_p[n] -= measure_of_iface*bc_coeff_avg*bc_value_avg*dist/(bc_coeff_avg*dist-mu_avg);
                  }
                }
              }
            }

          } // if for Robin term

          rhs_p[n] /= w_000;

        } break; // case NODE_NMN

      } // end of switch
    } // if a node is on wall
  } // loop over nodes

  if (matrix_has_nullspace && fixed_value_idx_l >= 0){
    rhs_p[fixed_value_idx_l] = 0;
  }

  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecRestoreArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  mu.finalize();
  diag_add.finalize();
  wall_value.finalize();
  for (int i = 0; i < n_phis; i++)
  {
    bc_coeffs[i].finalize();
    bc_values[i].finalize();
  }

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  if (keep_scalling) {
    ierr = VecRestoreArray(scalling, &scalling_p); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);
}

//double my_p4est_poisson_nodes_mls_t::calculate_trunc_error(CF_2 &exact)
//{
//  ierr = VecCreateGhostNodes(p4est, nodes, &trunc_error); CHKERRXX(ierr);
//  ierr = VecCreateGhostNodes(p4est, nodes, &exact_vec);   CHKERRXX(ierr);

//  sample_cf_on_nodes(p4est, nodes, exact, exact_vec);

//  ierr = MatMult(A, exact_vec, trunc_error); CHKERRXX(ierr);

//  double *scalling_p;     ierr = VecGetArray(scalling,    &scalling_p);     CHKERRXX(ierr);
//  double *rhs_p;          ierr = VecGetArray(rhs_,        &rhs_p);          CHKERRXX(ierr);
//  double *trunc_error_p;  ierr = VecGetArray(trunc_error, &trunc_error_p);  CHKERRXX(ierr);

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
//  {
//    switch (node_loc[n])
//    {
//    case NODE_OUT: trunc_error_p[n] = 0; break;
//    case NODE_INS: trunc_error_p[n] = fabs(  (trunc_error_p[n] - rhs_p[n])*scalling_p[n]  ); break;
//    case NODE_NMN: trunc_error_p[n] = fabs(  (trunc_error_p[n] - rhs_p[n])*scalling_p[n]/dx_min/dy_min  ); break;
//    }
//  }

//  ierr = VecRestoreArray(scalling,    &scalling_p);     CHKERRXX(ierr);
//  ierr = VecRestoreArray(rhs_,        &rhs_p);          CHKERRXX(ierr);
//  ierr = VecRestoreArray(trunc_error, &trunc_error_p);  CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(trunc_error, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (trunc_error, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  ierr = VecDestroy(exact_vec); CHKERRXX(ierr);

//  double max;
//  ierr = VecMax(trunc_error, NULL, &max); CHKERRXX(ierr);
//  return max;
//}

//void my_p4est_poisson_nodes_mls_t::calculate_gradient_error(Vec sol, Vec err_ux, Vec err_uy, CF_2 &ux, CF_2 &uy)
//{
//  double *sol_p;    ierr = VecGetArray(sol,     &sol_p);    CHKERRXX(ierr);
//  double *err_ux_p; ierr = VecGetArray(err_ux,  &err_ux_p); CHKERRXX(ierr);
//  double *err_uy_p; ierr = VecGetArray(err_uy,  &err_uy_p); CHKERRXX(ierr);

//  p4est_locidx_t  node[9];
//  double          x_cent[9];
//  double          y_cent[9];
//  bool            node_in[9];
//  bool            alt[9];

//#ifdef P4_TO_P8
//#else
//  p4est_locidx_t node_m00;
//  p4est_locidx_t node_p00;
//  p4est_locidx_t node_0m0;
//  p4est_locidx_t node_0p0;
//#endif

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
//  {
//    double x_C  = node_x_fr_n(n, p4est, nodes);
//    double y_C  = node_y_fr_n(n, p4est, nodes);
//#ifdef P4_TO_P8
//    double z_C  = node_z_fr_n(n, p4est, nodes);
//#endif

//    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

//    if (node_loc[n] == NODE_OUT)
//    {
//      err_ux_p[n] = 0;
//      err_uy_p[n] = 0;
//    }
//    else if (node_loc[n] == NODE_INS)
//    {
//      err_ux_p[n] = fabs(qnnn.dx_central(sol_p) - ux(x_C, y_C));
//      err_uy_p[n] = fabs(qnnn.dy_central(sol_p) - uy(x_C, y_C));
//    }
//    else if (node_loc[n] == NODE_NMN)
//    {
//      // find all neighbors
//#ifdef P4_TO_P8
//      node_m00 = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
//                                                 : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp);
//      node_p00 = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
//                                                 : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp);
//      node_0m0 = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
//                                                 : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp);
//      node_0p0 = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
//                                                 : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp);
//      node_00m = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
//                                                 : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp);
//      node_00p = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
//                                                 : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp);
//#else
//      node_m00 = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm;
//      node_p00 = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm;
//      node_0m0 = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
//      node_0p0 = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
//#endif
//      node[0] = n;
//      node[1] = node_m00; node[2] = node_p00; node[3] = node_0m0; node[4] = node_0p0;

//      for (int q = 1; q < 5; q++)
//      {
//        find_centroid(node_in[q], alt[q], x_cent[q], y_cent[q], node[q], NULL);
//      }
//      if (node_in[1] && node_in[2]) err_ux_p[n] = fabs(qnnn.dx_central(sol_p) - ux(x_C, y_C)); else err_ux_p[n] = 0;
//      if (node_in[3] && node_in[4]) err_uy_p[n] = fabs(qnnn.dy_central(sol_p) - uy(x_C, y_C)); else err_uy_p[n] = 0;
//    }
//  }

//  ierr = VecRestoreArray(sol,     &sol_p);    CHKERRXX(ierr);
//  ierr = VecRestoreArray(err_ux,  &err_ux_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(err_uy,  &err_uy_p); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(err_ux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateBegin(err_uy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (err_ux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (err_uy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//}

//void my_p4est_poisson_nodes_mls_t::calculate_equation_error(Vec sol, Vec err_eq)
//{
//  double *sol_p;    ierr = VecGetArray(sol,     &sol_p);    CHKERRXX(ierr);
//  double *err_eq_p; ierr = VecGetArray(err_eq,  &err_eq_p); CHKERRXX(ierr);

//  p4est_locidx_t  node[9];
//  double          x_cent[9];
//  double          y_cent[9];
//  bool            node_in[9];
//  bool            alt[9];

//#ifdef P4_TO_P8
//#else
//  p4est_locidx_t node_m00;
//  p4est_locidx_t node_p00;
//  p4est_locidx_t node_0m0;
//  p4est_locidx_t node_0p0;
//#endif

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
//  {
//    double x_C  = node_x_fr_n(n, p4est, nodes);
//    double y_C  = node_y_fr_n(n, p4est, nodes);
//#ifdef P4_TO_P8
//    double z_C  = node_z_fr_n(n, p4est, nodes);
//#endif

//    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

//    if (node_loc[n] == NODE_OUT)
//    {
//      err_eq_p[n] = 0;
//    }
//    else if (node_loc[n] == NODE_INS)
//    {
//      err_eq_p[n] = fabs(mu_*(qnnn.dxx_central(sol_p)+qnnn.dyy_central(sol_p)) + (*force_)(x_C, y_C));
//    }
//    else if (node_loc[n] == NODE_NMN)
//    {
//      // find all neighbors
//#ifdef P4_TO_P8
//      node_m00 = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
//                                                 : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp);
//      node_p00 = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
//                                                 : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp);
//      node_0m0 = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
//                                                 : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp);
//      node_0p0 = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
//                                                 : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp);
//      node_00m = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
//                                                 : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp);
//      node_00p = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
//                                                 : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp);
//#else
//      node_m00 = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm;
//      node_p00 = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm;
//      node_0m0 = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
//      node_0p0 = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;
//#endif
//      node[0] = n;
//      node[1] = node_m00; node[2] = node_p00; node[3] = node_0m0; node[4] = node_0p0;

//      for (int q = 1; q < 5; q++)
//      {
//        find_centroid(node_in[q], alt[q], x_cent[q], y_cent[q], node[q], NULL);
//      }
//      if (node_in[1] && node_in[2] && node_in[3] && node_in[4])
//        err_eq_p[n] = fabs(mu_*((sol_p[node_m00]+sol_p[node_p00]-2.*sol_p[n])/dx_min/dx_min + (sol_p[node_0m0]+sol_p[node_0p0]-2.*sol_p[n])/dy_min/dy_min ) + (*force_)(x_C, y_C));
////        err_eq_p[n] = fabs(mu_*(qnnn.dxx_central(sol_p)+qnnn.dyy_central(sol_p)) + (*force_)(x_C, y_C));
//    }
//  }

//  ierr = VecRestoreArray(sol,     &sol_p);    CHKERRXX(ierr);
//  ierr = VecRestoreArray(err_eq,  &err_eq_p); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(err_eq, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (err_eq, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//}

void my_p4est_poisson_nodes_mls_t::find_projection(double *phi_p, p4est_locidx_t *neighbors, bool *neighbor_exists, double dxyz_pr[], double &dist_pr)
{
  // find projection point
  double phi_x = 0., phi_y = 0.;
#ifdef P4_TO_P8
  double phi_z = 0;
#endif

  // x-derivative
  if (neighbor_exists[nn_m00] && neighbor_exists[nn_p00])
    phi_x = 0.5*(phi_p[neighbors[nn_p00]] - phi_p[neighbors[nn_m00]])/dx_min;
  else if (neighbor_exists[nn_p00])
    phi_x = (phi_p[neighbors[nn_p00]] - phi_p[neighbors[nn_000]])/dx_min;
  else if (neighbor_exists[nn_m00])
    phi_x = (phi_p[neighbors[nn_000]] - phi_p[neighbors[nn_m00]])/dx_min;
#ifdef CASL_THROWS
  else
    throw std::invalid_argument("[CASL_ERROR]: Not enough nodes to calculate x-derivative");
#endif

  // y-derivative
  if (neighbor_exists[nn_0m0] && neighbor_exists[nn_0p0])
    phi_y = 0.5*(phi_p[neighbors[nn_0p0]] - phi_p[neighbors[nn_0m0]])/dy_min;
  else if (neighbor_exists[nn_0p0])
    phi_y = (phi_p[neighbors[nn_0p0]] - phi_p[neighbors[nn_000]])/dy_min;
  else if (neighbor_exists[nn_0m0])
    phi_y = (phi_p[neighbors[nn_000]] - phi_p[neighbors[nn_0m0]])/dy_min;
#ifdef CASL_THROWS
  else
    throw std::invalid_argument("[CASL_ERROR]: Not enough nodes to calculate y-derivative");
#endif

#ifdef P4_TO_P8
  // z-derivative
  if (neighbor_exists[nn_00m] && neighbor_exists[nn_00p])
    phi_z = 0.5*(phi_p[neighbors[nn_00p]] - phi_p[neighbors[nn_00m]])/dz_min;
  else if (neighbor_exists[nn_00p])
    phi_z = (phi_p[neighbors[nn_00p]] - phi_p[neighbors[nn_000]])/dz_min;
  else if (neighbor_exists[nn_00m])
    phi_z = (phi_p[neighbors[nn_000]] - phi_p[neighbors[nn_00m]])/dz_min;
#ifdef CASL_THROWS
  else
    throw std::invalid_argument("[CASL_ERROR]: Not enough nodes to calculate y-derivative");
#endif
#endif

#ifdef P4_TO_P8
  double phi_d = sqrt(SQR(phi_x)+SQR(phi_y)+SQR(phi_z));
#else
  double phi_d = sqrt(SQR(phi_x)+SQR(phi_y));
#endif

  phi_x /= phi_d;
  phi_y /= phi_d;
#ifdef P4_TO_P8
  phi_z /= phi_d;
#endif

  dist_pr = phi_p[neighbors[nn_000]]/phi_d;

  dxyz_pr[0] = - dist_pr*phi_x;
  dxyz_pr[1] = - dist_pr*phi_y;
#ifdef P4_TO_P8
  dxyz_pr[2] = - dist_pr*phi_z;
#endif
}

void my_p4est_poisson_nodes_mls_t::find_projection(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr)
{
  // find projection point
  double phi_x = 0., phi_y = 0.;
#ifdef P4_TO_P8
  double phi_z = 0;
#endif

  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, qnnn.node_000);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est, ni);
  bool xp_wall = is_node_xpWall(p4est, ni);
  bool ym_wall = is_node_ymWall(p4est, ni);
  bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est, ni);
  bool zp_wall = is_node_zpWall(p4est, ni);
#endif

  if (!xm_wall && !xp_wall) phi_x = qnnn.dx_central(phi_p);
  else if (!xm_wall)        phi_x = qnnn.dx_backward_linear(phi_p);
  else if (!xp_wall)        phi_x = qnnn.dx_forward_linear(phi_p);

  if (!ym_wall && !yp_wall) phi_y = qnnn.dy_central(phi_p);
  else if (!ym_wall)        phi_y = qnnn.dy_backward_linear(phi_p);
  else if (!yp_wall)        phi_y = qnnn.dy_forward_linear(phi_p);

#ifdef P4_TO_P8
  if (!zm_wall && !zp_wall) phi_z = qnnn.dz_central(phi_p);
  else if (!zm_wall)        phi_z = qnnn.dz_backward_linear(phi_p);
  else if (!zp_wall)        phi_z = qnnn.dz_forward_linear(phi_p);
#endif

#ifdef P4_TO_P8
  double phi_d = sqrt(SQR(phi_x)+SQR(phi_y)+SQR(phi_z));
#else
  double phi_d = sqrt(SQR(phi_x)+SQR(phi_y));
#endif

  phi_x /= phi_d;
  phi_y /= phi_d;
#ifdef P4_TO_P8
  phi_z /= phi_d;
#endif

  dist_pr = phi_p[qnnn.node_000]/phi_d;

  dxyz_pr[0] = - dist_pr*phi_x;
  dxyz_pr[1] = - dist_pr*phi_y;
#ifdef P4_TO_P8
  dxyz_pr[2] = - dist_pr*phi_z;
#endif
}

void my_p4est_poisson_nodes_mls_t::compute_normal(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[])
{
  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, qnnn.node_000);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est, ni);
  bool xp_wall = is_node_xpWall(p4est, ni);
  bool ym_wall = is_node_ymWall(p4est, ni);
  bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est, ni);
  bool zp_wall = is_node_zpWall(p4est, ni);
#endif

  if (!xm_wall && !xp_wall) n[0] = qnnn.dx_central        (phi_p);
  else if (!xm_wall)        n[0] = qnnn.dx_backward_linear(phi_p);
  else if (!xp_wall)        n[0] = qnnn.dx_forward_linear (phi_p);

  if (!ym_wall && !yp_wall) n[1] = qnnn.dy_central        (phi_p);
  else if (!ym_wall)        n[1] = qnnn.dy_backward_linear(phi_p);
  else if (!yp_wall)        n[1] = qnnn.dy_forward_linear (phi_p);

#ifdef P4_TO_P8
  if (!zm_wall && !zp_wall) n[2] = qnnn.dz_central        (phi_p);
  else if (!zm_wall)        n[2] = qnnn.dz_backward_linear(phi_p);
  else if (!zp_wall)        n[2] = qnnn.dz_forward_linear (phi_p);
#endif

#ifdef P4_TO_P8
  double phi_d = sqrt(SQR(n[0])+SQR(n[1])+SQR(n[2]));
#else
  double phi_d = sqrt(SQR(n[0])+SQR(n[1]));
#endif

  n[0] /= phi_d;
  n[1] /= phi_d;
#ifdef P4_TO_P8
  n[2] /= phi_d;
#endif
}

//int my_p4est_poisson_nodes_mls_t::which_quad(double *dxyz[])
//{
//#ifdef P4_TO_P8
//  if      (dxyz[0] <= 0. && dxyz[1] <= 0. && dxyz[3] <= 0.) return dir::v_mmm;
//  else if (dxyz[0] >= 0. && dxyz[1] <= 0. && dxyz[3] <= 0.) return dir::v_pmm;
//  else if (dxyz[0] <= 0. && dxyz[1] >= 0. && dxyz[3] <= 0.) return dir::v_mpm;
//  else if (dxyz[0] >= 0. && dxyz[1] >= 0. && dxyz[3] <= 0.) return dir::v_ppm;
//  else if (dxyz[0] <= 0. && dxyz[1] <= 0. && dxyz[3] >= 0.) return dir::v_mmp;
//  else if (dxyz[0] >= 0. && dxyz[1] <= 0. && dxyz[3] >= 0.) return dir::v_pmp;
//  else if (dxyz[0] <= 0. && dxyz[1] >= 0. && dxyz[3] >= 0.) return dir::v_mpp;
//  else if (dxyz[0] >= 0. && dxyz[1] >= 0. && dxyz[3] >= 0.) return dir::v_ppp;
//#else
//  if      (dxyz[0] <= 0. && dxyz[1] <= 0.) return dir::v_mmm;
//  else if (dxyz[0] >= 0. && dxyz[1] <= 0.) return dir::v_pmm;
//  else if (dxyz[0] <= 0. && dxyz[1] >= 0.) return dir::v_mpm;
//  else if (dxyz[0] >= 0. && dxyz[1] >= 0.) return dir::v_ppm;
//#endif
//}

void my_p4est_poisson_nodes_mls_t::sample_vec_at_neighbors(double *in_p, int *neighbors, bool *neighbor_exists, double *output)
{
  int j = 0;
  for (int i = 0; i < N_NBRS_MAX; i++)
    if (neighbor_exists[i]) {output[j] = in_p[neighbors[i]]; j++;}
}

void my_p4est_poisson_nodes_mls_t::sample_vec_at_neighbors(double *in_p, int *neighbors, bool *neighbor_exists, std::vector<double> &output)
{
  int j = 0;
  for (int i = 0; i < N_NBRS_MAX; i++)
    if (neighbor_exists[i]) {output[j] = in_p[neighbors[i]]; j++;}
}

void my_p4est_poisson_nodes_mls_t::sample_qty_at_neighbors(quantity_t &qty, int *neighbors, bool *neighbor_exists, double *output)
{
  if (qty.constant) {
    for (int i = 0; i < N_NBRS_MAX; i++)
      output[i] = qty.constant;
  } else {
    int j = 0;
    for (int i = 0; i < N_NBRS_MAX; i++)
      if (neighbor_exists[i]) {output[j] = qty.vec_p[neighbors[i]]; j++;}
  }
}

double my_p4est_poisson_nodes_mls_t::sample_qty(quantity_t &qty, double *xyz)
{
  /* interp_local must be initialized for the node at hand before calling this function! */
  if (qty.is_constant) {
    return qty.constant;
  } else if (qty.is_vec) {
    interp_local.set_input(qty.vec_p, linear);
#ifdef P4_TO_P8
    return interp_local.interpolate(xyz[0], xyz[1], xyz[2]);
#else
    return interp_local.interpolate(xyz[0], xyz[1]);
#endif
  } else if (qty.is_cf) {
#ifdef P4_TO_P8
    return (*qty.cf)(xyz[0], xyz[1], xyz[2]);
#else
    return (*qty.cf)(xyz[0], xyz[1]);
#endif
  }
}

double my_p4est_poisson_nodes_mls_t::sample_qty(quantity_t &qty, p4est_locidx_t n)
{
  /* It does not work with cf quantities ATM */
  if (qty.is_constant) return qty.constant;
  else if (qty.is_vec) return qty.vec_p[n];
}

double my_p4est_poisson_nodes_mls_t::sample_qty(quantity_t &qty, p4est_locidx_t n, double *xyz)
{
  if (qty.is_constant) return qty.constant;
  else if (qty.is_vec) return qty.vec_p[n];
  else if (qty.is_cf) {
  #ifdef P4_TO_P8
      return (*qty.cf)(xyz[0], xyz[1], xyz[2]);
  #else
      return (*qty.cf)(xyz[0], xyz[1]);
  #endif
    }
}

void my_p4est_poisson_nodes_mls_t::get_all_neighbors(p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists)
{
  p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

  // check if the node is a wall node
  bool xm_wall = is_node_xmWall(p4est, ni);
  bool xp_wall = is_node_xpWall(p4est, ni);
  bool ym_wall = is_node_ymWall(p4est, ni);
  bool yp_wall = is_node_ypWall(p4est, ni);
#ifdef P4_TO_P8
  bool zm_wall = is_node_zmWall(p4est, ni);
  bool zp_wall = is_node_zpWall(p4est, ni);
#endif

  // count neighbors
  for (int i = 0; i < N_NBRS_MAX; i++) neighbor_exists[i] = true;

  if (xm_wall)
  {
    int i = 0;
    for (int j = 0; j < 3; j++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

  if (xp_wall)
  {
    int i = 2;
    for (int j = 0; j < 3; j++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

  if (ym_wall)
  {
    int j = 0;
    for (int i = 0; i < 3; i++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

  if (yp_wall)
  {
    int j = 2;
    for (int i = 0; i < 3; i++)
#ifdef P4_TO_P8
      for (int k = 0; k < 3; k++)
        neighbor_exists[i + j*3 + k*3*3] = false;
#else
      neighbor_exists[i + j*3] = false;
#endif
  }

#ifdef P4_TO_P8
  if (zm_wall)
  {
    int k = 0;
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        neighbor_exists[i + j*3 + k*3*3] = false;
  }

  if (zp_wall)
  {
    int k = 2;
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        neighbor_exists[i + j*3 + k*3*3] = false;
  }
#endif

  // find neighboring quadrants
  p4est_locidx_t quad_mmm_idx; p4est_topidx_t tree_mmm_idx;
  p4est_locidx_t quad_mpm_idx; p4est_topidx_t tree_mpm_idx;
  p4est_locidx_t quad_pmm_idx; p4est_topidx_t tree_pmm_idx;
  p4est_locidx_t quad_ppm_idx; p4est_topidx_t tree_ppm_idx;
#ifdef P4_TO_P8
  p4est_locidx_t quad_mmp_idx; p4est_topidx_t tree_mmp_idx;
  p4est_locidx_t quad_mpp_idx; p4est_topidx_t tree_mpp_idx;
  p4est_locidx_t quad_pmp_idx; p4est_topidx_t tree_pmp_idx;
  p4est_locidx_t quad_ppp_idx; p4est_topidx_t tree_ppp_idx;
#endif

#ifdef P4_TO_P8
  node_neighbors->find_neighbor_cell_of_node(n, -1, -1, -1, quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  node_neighbors->find_neighbor_cell_of_node(n, -1,  1, -1, quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  node_neighbors->find_neighbor_cell_of_node(n,  1, -1, -1, quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  node_neighbors->find_neighbor_cell_of_node(n,  1,  1, -1, quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
  node_neighbors->find_neighbor_cell_of_node(n, -1, -1,  1, quad_mmp_idx, tree_mmp_idx); //nei_quads[dir::v_mmp] = quad_mmp_idx;
  node_neighbors->find_neighbor_cell_of_node(n, -1,  1,  1, quad_mpp_idx, tree_mpp_idx); //nei_quads[dir::v_mpp] = quad_mpp_idx;
  node_neighbors->find_neighbor_cell_of_node(n,  1, -1,  1, quad_pmp_idx, tree_pmp_idx); //nei_quads[dir::v_pmp] = quad_pmp_idx;
  node_neighbors->find_neighbor_cell_of_node(n,  1,  1,  1, quad_ppp_idx, tree_ppp_idx); //nei_quads[dir::v_ppp] = quad_ppp_idx;
#else
  node_neighbors->find_neighbor_cell_of_node(n, -1, -1, quad_mmm_idx, tree_mmm_idx); //nei_quads[dir::v_mmm] = quad_mmm_idx;
  node_neighbors->find_neighbor_cell_of_node(n, -1, +1, quad_mpm_idx, tree_mpm_idx); //nei_quads[dir::v_mpm] = quad_mpm_idx;
  node_neighbors->find_neighbor_cell_of_node(n, +1, -1, quad_pmm_idx, tree_pmm_idx); //nei_quads[dir::v_pmm] = quad_pmm_idx;
  node_neighbors->find_neighbor_cell_of_node(n, +1, +1, quad_ppm_idx, tree_ppm_idx); //nei_quads[dir::v_ppm] = quad_ppm_idx;
#endif

  // find neighboring nodes
#ifdef P4_TO_P8
//  // zm plane
//  neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];

//  neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
//  neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

//  neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
//  neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];

//  neighbors[nn_mmm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
//  neighbors[nn_pmm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
//  neighbors[nn_mpm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
//  neighbors[nn_ppm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

//  // z0 plane
//  neighbors[nn_000] = n;

//  neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
//  neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

//  neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
//  neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

//  neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
//  neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
//  neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
//  neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

//  // zp plane
//  neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

//  neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
//  neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

//  neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
//  neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

//  neighbors[nn_mmp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
//  neighbors[nn_pmp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
//  neighbors[nn_mpp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
//  neighbors[nn_ppp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];

  neighbors[nn_000] = n;

  // m00
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpp];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmm];

  // p00
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppp];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppm];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

  // 0m0
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmp];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmm];

  // 0p0
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppp];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppm];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

  // 00m
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_pmm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mmm];

  // 00p
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_ppp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_pmp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

  // 0mm
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];
  // 0pm
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  // 0mp
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_mmp];
  // 0pp
  if      (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

  // m0m
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];
  // p0m
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];
  // m0p
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mmp];
  // p0p
  if      (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

  // mm0
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmp];
  else if (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
  // pm0
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmp];
  else if (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
  // mp0
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpp];
  else if (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
  // pp0
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppp];
  else if (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

  // mmm
  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mmm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  // pmm
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pmm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  // mpm
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mpm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  // ppm
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_ppm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

  // mmp
  if      (quad_mmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mmp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
  // pmp
  if      (quad_pmp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pmp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
  // mpp
  if      (quad_mpp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mpp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
  // ppp
  if      (quad_ppp_idx != NOT_A_VALID_QUADRANT) neighbors[nn_ppp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];
#else
  neighbors[nn_000] = n;

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mmm];

  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_ppm];
  else if (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
  else if (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_mmm];

  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];
  else if (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_ppm];

  if      (quad_mmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
  if      (quad_pmm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
  if      (quad_mpm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
  if      (quad_ppm_idx != NOT_A_VALID_QUADRANT) neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];
#endif
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_t::compute_error_sl(CF_3 &exact_cf, Vec sol, Vec err)
#else
void my_p4est_poisson_nodes_mls_t::compute_error_sl(CF_2 &exact_cf, Vec sol, Vec err)
#endif
{
  double *sol_p; ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  double *err_p; ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if (is_calc(n))
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
      err_p[n] = ABS(sol_p[n] - exact_cf(x,y,z));
#else
      err_p[n] = ABS(sol_p[n] - exact_cf(x,y));
#endif
    }
    else
      err_p[n] = 0;
  }

  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(err, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (err, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_t::compute_error_tr(CF_3 &exact_cf, Vec error)
#else
void my_p4est_poisson_nodes_mls_t::compute_error_tr(CF_2 &exact_cf, Vec error)
#endif
{
  Vec exact_vec; ierr = VecCreateGhostNodes(p4est, nodes, &exact_vec);   CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, exact_cf, exact_vec);

  ierr = MatMult(A, exact_vec, error); CHKERRXX(ierr);

  double *scalling_p; ierr = VecGetArray(scalling,  &scalling_p); CHKERRXX(ierr);
  double *rhs_p;      ierr = VecGetArray(rhs,       &rhs_p);      CHKERRXX(ierr);
  double *error_p;    ierr = VecGetArray(error,     &error_p);    CHKERRXX(ierr);

  double vol_min = dx_min*dy_min;
#ifdef P4_TO_P8
  vol_min *= dz_min;
#endif

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    switch (node_loc[n])
    {
    case NODE_OUT: error_p[n] = 0; break;
    case NODE_INS: error_p[n] = fabs(  (error_p[n] - rhs_p[n])*scalling_p[n]  ); break;
    case NODE_NMN: error_p[n] = fabs(  (error_p[n] - rhs_p[n])*scalling_p[n]/vol_min  ); break;
    }
  }

  ierr = VecRestoreArray(scalling,  &scalling_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs,       &rhs_p);          CHKERRXX(ierr);
  ierr = VecRestoreArray(error,     &error_p);  CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(error, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (error, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecDestroy(exact_vec); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_t::compute_error_gr(CF_3 &ux_cf, CF_3 &uy_cf, CF_3 &uz_cf, Vec sol, Vec err_ux, Vec err_uy, Vec err_uz)
#else
void my_p4est_poisson_nodes_mls_t::compute_error_gr(CF_2 &ux_cf, CF_2 &uy_cf, Vec sol, Vec err_ux, Vec err_uy)
#endif
{
  double *sol_p;    ierr = VecGetArray(sol,     &sol_p);    CHKERRXX(ierr);
  double *err_ux_p; ierr = VecGetArray(err_ux,  &err_ux_p); CHKERRXX(ierr);
  double *err_uy_p; ierr = VecGetArray(err_uy,  &err_uy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *err_uz_p; ierr = VecGetArray(err_uz,  &err_uz_p); CHKERRXX(ierr);
#endif
  double *node_vol_p; ierr = VecGetArray(node_vol,  &node_vol_p); CHKERRXX(ierr);

  p4est_locidx_t neighbors[N_NBRS_MAX];
  p4est_locidx_t neighbors_of_n[N_NBRS_MAX];
  bool neighbor_exists[N_NBRS_MAX];

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    double x_C  = node_x_fr_n(n, p4est, nodes);
    double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

#ifdef P4_TO_P8
    double ux_exact = ux_cf(x_C, y_C, z_C);
    double uy_exact = uy_cf(x_C, y_C, z_C);
    double uz_exact = uz_cf(x_C, y_C, z_C);
#else
    double ux_exact = ux_cf(x_C, y_C);
    double uy_exact = uy_cf(x_C, y_C);
#endif

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors->get_neighbors(n);

    if (node_loc[n] == NODE_OUT || node_vol_p[n] < .0e-1*vol_min)
    {
      err_ux_p[n] = 0;
      err_uy_p[n] = 0;
#ifdef P4_TO_P8
      err_uz_p[n] = 0;
#endif
    } else if (node_loc[n] == NODE_INS) {

      err_ux_p[n] = fabs(qnnn.dx_central(sol_p) - ux_exact);
      err_uy_p[n] = fabs(qnnn.dy_central(sol_p) - uy_exact);
#ifdef P4_TO_P8
      err_uz_p[n] = fabs(qnnn.dz_central(sol_p) - uz_exact);
#endif
    } else if (node_loc[n] == NODE_NMN) {
      get_all_neighbors(n, neighbors, neighbor_exists);

      // compute x-component
      if (node_vol_p[neighbors[nn_m00]] > eps_dom && node_vol_p[neighbors[nn_p00]] > eps_dom) { // using central differences

        err_ux_p[n] = fabs(qnnn.dx_central(sol_p) - ux_exact);

//      } else if (node_vol_p[neighbors[nn_m00]] > eps_dom) { // using backward differences

//        get_all_neighbors(neighbors[nn_m00], neighbors_of_n, neighbor_exists);
//        if (node_vol_p[neighbors_of_n[nn_m00]] > eps_dom)
//          err_ux_p[n] = fabs((1.5*sol_p[n] - 2.0*sol_p[neighbors[nn_m00]] + 0.5*sol_p[neighbors_of_n[nn_m00]])/dx_min - ux_exact);
//        else
//          err_ux_p[n] = 0.;

//      } else if (node_vol_p[neighbors[nn_p00]] > eps_dom) { // using forward differences

//        get_all_neighbors(neighbors[nn_p00], neighbors_of_n, neighbor_exists);
//        if (node_vol_p[neighbors_of_n[nn_p00]] > eps_dom)
//          err_ux_p[n] = fabs((- 1.5*sol_p[n] + 2.0*sol_p[neighbors[nn_p00]] - 0.5*sol_p[neighbors_of_n[nn_p00]])/dx_min - ux_exact);
//        else
//          err_ux_p[n] = 0.;

      } else err_ux_p[n] = 0.;

      // compute y-component
      if (node_vol_p[neighbors[nn_0m0]] > eps_dom && node_vol_p[neighbors[nn_0p0]] > eps_dom) { // using central differences

        err_uy_p[n] = fabs(qnnn.dy_central(sol_p) - uy_exact);

//      } else if (node_vol_p[neighbors[nn_0m0]] > eps_dom) { // using backward differences

//        get_all_neighbors(neighbors[nn_0m0], neighbors_of_n, neighbor_exists);
//        if (node_vol_p[neighbors_of_n[nn_0m0]] > eps_dom)
//          err_uy_p[n] = fabs((1.5*sol_p[n] - 2.0*sol_p[neighbors[nn_0m0]] + 0.5*sol_p[neighbors_of_n[nn_0m0]])/dy_min - uy_exact);
//        else
//          err_uy_p[n] = 0.;

//      } else if (node_vol_p[neighbors[nn_0p0]] > eps_dom) { // using forward differences

//        get_all_neighbors(neighbors[nn_0p0], neighbors_of_n, neighbor_exists);
//        if (node_vol_p[neighbors_of_n[nn_0p0]] > eps_dom)
//          err_uy_p[n] = fabs((- 1.5*sol_p[n] + 2.0*sol_p[neighbors[nn_0p0]] - 0.5*sol_p[neighbors_of_n[nn_0p0]])/dy_min - uy_exact);
//        else
//          err_uy_p[n] = 0.;

      } else err_uy_p[n] = 0;

#ifdef P4_TO_P8
      // compute z-component
      if (node_vol_p[neighbors[nn_00m]] > eps_dom && node_vol_p[neighbors[nn_00p]] > eps_dom) { // using central differences

        err_uz_p[n] = fabs(qnnn.dz_central(sol_p) - uz_exact);

//      } else if (node_vol_p[neighbors[nn_00m]] > eps_dom) { // using backward differences

//        get_all_neighbors(neighbors[nn_00m], neighbors_of_n, neighbor_exists);
//        if (node_vol_p[neighbors_of_n[nn_00m]] > eps_dom)
//          err_uz_p[n] = fabs((1.5*sol_p[n] - 2.0*sol_p[neighbors[nn_00m]] + 0.5*sol_p[neighbors_of_n[nn_00m]])/dz_min - uz_exact);
//        else
//          err_uz_p[n] = 0.;

//      } else if (node_vol_p[neighbors[nn_00p]] > eps_dom) { // using forward differences

//        get_all_neighbors(neighbors[nn_00p], neighbors_of_n, neighbor_exists);
//        if (node_vol_p[neighbors_of_n[nn_00p]] > eps_dom)
//          err_uz_p[n] = fabs((- 1.5*sol_p[n] + 2.0*sol_p[neighbors[nn_00p]] - 0.5*sol_p[neighbors_of_n[nn_00p]])/dz_min - uz_exact);
//        else
//          err_uz_p[n] = 0.;

      } else err_uz_p[n] = 0;
#endif
    }
  }

  ierr = VecRestoreArray(sol,     &sol_p);    CHKERRXX(ierr);
  ierr = VecRestoreArray(err_ux,  &err_ux_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_uy,  &err_uy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(err_uz,  &err_uz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(node_vol,  &node_vol_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(err_ux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(err_uy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateBegin(err_uz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
  ierr = VecGhostUpdateEnd  (err_ux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (err_uy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGhostUpdateEnd  (err_uz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
}

//#ifdef P4_TO_P8
//void my_p4est_poisson_nodes_mls_t::compute_error_gr(CF_3 &ux_cf, CF_3 &uy_cf, CF_3 &uz_cf, Vec sol, Vec err_ux, Vec err_uy, Vec err_uz)
//#else
//void my_p4est_poisson_nodes_mls_t::compute_error_xy(CF_2 &uxy_cf, Vec sol, Vec err_uxy)
//#endif
//{
//  double *sol_p;      ierr = VecGetArray(sol,       &sol_p);      CHKERRXX(ierr);
//  double *err_uxy_p;  ierr = VecGetArray(err_uxy,   &err_uxy_p);  CHKERRXX(ierr);
//  double *node_vol_p; ierr = VecGetArray(node_vol,  &node_vol_p); CHKERRXX(ierr);

//  p4est_locidx_t neighbors[N_NBRS_MAX];
//  bool neighbor_exists[N_NBRS_MAX];

//  int node_mm, node_pm, node_mp, node_pp;
//  double x_0, y_0, tmp;

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
//  {
//    double x_C  = node_x_fr_n(n, p4est, nodes);
//    double y_C  = node_y_fr_n(n, p4est, nodes);
//#ifdef P4_TO_P8
//    double z_C  = node_z_fr_n(n, p4est, nodes);
//#endif

//    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors->get_neighbors(n);

//    if (node_loc[n] == NODE_OUT || node_vol_p[n] < .0e-1*vol_min)
//    {
//      err_uxy_p[n] = 0;
//#ifdef P4_TO_P8
//#endif
//    } else if (node_loc[n] == NODE_NMN) {
//      get_all_neighbors(n, neighbors, neighbor_exists);

//      err_uxy_p[n] = 0.;
//      for (int i = 0; i < 4; i++)
//      {
//        switch (i) {
//          case 0: node_mm = neighbors[nn_mm0]; node_pm = neighbors[nn_0m0]; node_mp = neighbors[nn_m00]; node_pp = neighbors[nn_000]; x_0 = x_C - 0.5*dx_min; y_0 = y_C - 0.5*dy_min; break;
//          case 1: node_mm = neighbors[nn_0m0]; node_pm = neighbors[nn_pm0]; node_mp = neighbors[nn_000]; node_pp = neighbors[nn_p00]; x_0 = x_C + 0.5*dx_min; y_0 = y_C - 0.5*dy_min; break;
//          case 2: node_mm = neighbors[nn_m00]; node_pm = neighbors[nn_000]; node_mp = neighbors[nn_mp0]; node_pp = neighbors[nn_0p0]; x_0 = x_C - 0.5*dx_min; y_0 = y_C + 0.5*dy_min; break;
//          case 3: node_mm = neighbors[nn_000]; node_pm = neighbors[nn_p00]; node_mp = neighbors[nn_0p0]; node_pp = neighbors[nn_pp0]; x_0 = x_C + 0.5*dx_min; y_0 = y_C + 0.5*dy_min; break;
//        }
//        if (node_vol_p[node_mm] > eps_dom &&
//            node_vol_p[node_pm] > eps_dom &&
//            node_vol_p[node_mp] > eps_dom &&
//            node_vol_p[node_pp] > eps_dom) {
//          tmp = fabs((sol_p[node_mm] - sol_p[node_pm] - sol_p[node_mp] + sol_p[node_pp])/4.0/dx_min/dy_min - uxy_cf(x_0, y_0));
//          err_uxy_p[n] = MAX(err_uxy_p[n],tmp);
//        }
//      }
//    }
//  }

//  ierr = VecRestoreArray(sol,     &sol_p);    CHKERRXX(ierr);
//  ierr = VecRestoreArray(err_uxy,  &err_uxy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//#endif
//  ierr = VecRestoreArray(node_vol,  &node_vol_p); CHKERRXX(ierr);

//  ierr = VecGhostUpdateBegin(err_uxy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//#endif
//  ierr = VecGhostUpdateEnd  (err_uxy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//#endif
//}

//double my_p4est_poisson_nodes_mls_t::interpolate_near_node_linear(double *in_p, p4est_locidx_t *nei_quads, bool *nei_quad_exists, double x, double y, double z)
//{
//  int which_quad = -1;

//  // check if point right on the node
//  if (fabs(x) < eps && fabs(y) < eps)
//    return in_p[n];

//  // check if point

//  if (x < 0)
//  {
//    if (y < 0)
//      which_quad = dir::v_mmm;
//    else
//      which_quad = dir::v_mpm;
//  } else {
//    if (y < 0)
//      which_quad = dir::v_pmm;
//    else
//      which_quad = dir::v_ppm;
//  }
//}
