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
    phi_interp(node_neighbors),
    is_matrix_computed(false), matrix_has_nullspace(false),
    A(NULL),
    phi_dd_owned(false),
    keep_scalling(true), scalling(NULL), phi_eff(NULL), cube_refinement(1)
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

  // array to store types of nodes
  ierr = VecCreateGhostNodes(p4est, nodes, &node_vol); CHKERRXX(ierr);

  double eps = 1E-9*d_min;

#ifdef P4_TO_P8
  double eps_dom = eps*eps*eps;
  double eps_ifc = eps*eps;
#else
  double eps_dom = eps*eps;
  double eps_ifc = eps;
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
    phi_eff_p[n] = -1.;

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

  double *node_vol_p;
  ierr = VecGetArray(node_vol, &node_vol_p); CHKERRXX(ierr);

  std::vector< std::vector<double> > phi_cube(n_phis, std::vector<double> (N_NBRS_MAX, -1));

  std::vector< std::vector<double> > phi_xx_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
  std::vector< std::vector<double> > phi_yy_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#ifdef P4_TO_P8
  std::vector< std::vector<double> > phi_zz_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#endif

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    if (phi_eff_p[n] >  1.5*diag_min) {node_vol_p[n] = 0.; continue;}
    if (phi_eff_p[n] < -1.5*diag_min) {node_vol_p[n] = 1.; continue;}

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
    bool neighbor_exists[N_NBRS_MAX];

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
    node_neighbors->find_neighbor_cell_of_node(n, -1, -1, -1, quad_mmm_idx, tree_mmm_idx);
    node_neighbors->find_neighbor_cell_of_node(n, -1,  1, -1, quad_mpm_idx, tree_mpm_idx);
    node_neighbors->find_neighbor_cell_of_node(n,  1, -1, -1, quad_pmm_idx, tree_pmm_idx);
    node_neighbors->find_neighbor_cell_of_node(n,  1,  1, -1, quad_ppm_idx, tree_ppm_idx);
    node_neighbors->find_neighbor_cell_of_node(n, -1, -1,  1, quad_mmp_idx, tree_mmp_idx);
    node_neighbors->find_neighbor_cell_of_node(n, -1,  1,  1, quad_mpp_idx, tree_mpp_idx);
    node_neighbors->find_neighbor_cell_of_node(n,  1, -1,  1, quad_pmp_idx, tree_pmp_idx);
    node_neighbors->find_neighbor_cell_of_node(n,  1,  1,  1, quad_ppp_idx, tree_ppp_idx);
#else
    node_neighbors->find_neighbor_cell_of_node(n, -1, -1, quad_mmm_idx, tree_mmm_idx);
    node_neighbors->find_neighbor_cell_of_node(n, -1, +1, quad_mpm_idx, tree_mpm_idx);
    node_neighbors->find_neighbor_cell_of_node(n, +1, -1, quad_pmm_idx, tree_pmm_idx);
    node_neighbors->find_neighbor_cell_of_node(n, +1, +1, quad_ppm_idx, tree_ppm_idx);
#endif

    // find neighboring nodes

    int neighbors[N_NBRS_MAX];

#ifdef P4_TO_P8
    // zm plane
    neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];

    neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
    neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

    neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
    neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];

    neighbors[nn_mmm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
    neighbors[nn_pmm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
    neighbors[nn_mpm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
    neighbors[nn_ppm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

    // z0 plane
    neighbors[nn_000] = n;

    neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
    neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

    neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
    neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

    neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
    neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
    neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
    neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

    // zp plane
    neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

    neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
    neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

    neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
    neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

    neighbors[nn_mmp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
    neighbors[nn_pmp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
    neighbors[nn_mpp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
    neighbors[nn_ppp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];

#else
    neighbors[nn_000] = n;

    neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
    neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

    neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
    neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];

    neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
    neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
    neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
    neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];
#endif

//    double x_nei[N_NBRS_MAX];

//    for (int i = 0; i < N_NBRS_MAX; i++)
//      if (neighbor_exists[i])
//      {
//        x_nei[i] = node_y_fr_n(neighbors[i], p4est, nodes);
//      }

    // fetch values of LSF
    for (int i_phi = 0; i_phi < n_phis; i_phi++)
    {
      int k = 0;
      for (int i = 0; i < N_NBRS_MAX; i++)
        if (neighbor_exists[i])
        {
          phi_cube[i_phi][k] = phi_p[i_phi][neighbors[i]];
          phi_xx_cube[i_phi][k] = phi_xx_p[i_phi][neighbors[i]];
          phi_yy_cube[i_phi][k] = phi_yy_p[i_phi][neighbors[i]];
#ifdef P4_TO_P8
          phi_zz_cube[i_phi][k] = phi_zz_p[i_phi][neighbors[i]];
#endif
          k++;
        }
    }
#ifdef P4_TO_P8
    cube3_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube, zm_cube, zp_cube);
#else
    cube2_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube);
#endif

//#ifdef P4_TO_P8
//    cube3_refined_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube, zm_cube, zp_cube);
//#else
//    cube2_refined_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube);
//#endif

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

//    node_vol_p[n] = cube.measure_of_domain()/cell_vol;
    node_vol_p[n] = cube.measure_of_domain();

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
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(bc_ == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

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

  // set local add if none was given
  bool local_add = false;
  if(add_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add_); CHKERRXX(ierr);
    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
  }

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
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs_, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(add_); CHKERRXX(ierr);
    add_ = NULL;
  }
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

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);
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

void my_p4est_poisson_nodes_mls_t::setup_negative_laplace_matrix_non_sym()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  // get access to LSFs
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

  // allocate vectors for LSF values for a cube
  std::vector< std::vector<double> > phi_cube(n_phis, std::vector<double> (N_NBRS_MAX, -1));

  std::vector< std::vector<double> > phi_xx_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
  std::vector< std::vector<double> > phi_yy_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#ifdef P4_TO_P8
  std::vector< std::vector<double> > phi_zz_cube(n_phis, std::vector<double> (N_NBRS_MAX, 0));
#endif

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    if (phi_eff_p[n] >  1.5*diag_min) {node_vol_p[n] = 0.; continue;}
    if (phi_eff_p[n] < -1.5*diag_min) {node_vol_p[n] = 1.; continue;}

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



  }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    if      (phi_eff_p[n] >  1.5*diag_min)  node_loc[n] = NODE_OUT;
    else if (phi_eff_p[n] < -1.5*diag_min)  node_loc[n] = NODE_INS;
    else                                    node_loc[n] = NODE_NMN;

    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    double x_C  = node_x_fr_n(n, p4est, nodes);
    double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

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

    if (node_loc[n] == NODE_NMN)
    {
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
      bool neighbor_exists[N_NBRS_MAX];

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
      node_neighbors->find_neighbor_cell_of_node(n, -1, -1, -1, quad_mmm_idx, tree_mmm_idx);
      node_neighbors->find_neighbor_cell_of_node(n, -1,  1, -1, quad_mpm_idx, tree_mpm_idx);
      node_neighbors->find_neighbor_cell_of_node(n,  1, -1, -1, quad_pmm_idx, tree_pmm_idx);
      node_neighbors->find_neighbor_cell_of_node(n,  1,  1, -1, quad_ppm_idx, tree_ppm_idx);
      node_neighbors->find_neighbor_cell_of_node(n, -1, -1,  1, quad_mmp_idx, tree_mmp_idx);
      node_neighbors->find_neighbor_cell_of_node(n, -1,  1,  1, quad_mpp_idx, tree_mpp_idx);
      node_neighbors->find_neighbor_cell_of_node(n,  1, -1,  1, quad_pmp_idx, tree_pmp_idx);
      node_neighbors->find_neighbor_cell_of_node(n,  1,  1,  1, quad_ppp_idx, tree_ppp_idx);
#else
      node_neighbors->find_neighbor_cell_of_node(n, -1, -1, quad_mmm_idx, tree_mmm_idx);
      node_neighbors->find_neighbor_cell_of_node(n, -1, +1, quad_mpm_idx, tree_mpm_idx);
      node_neighbors->find_neighbor_cell_of_node(n, +1, -1, quad_pmm_idx, tree_pmm_idx);
      node_neighbors->find_neighbor_cell_of_node(n, +1, +1, quad_ppm_idx, tree_ppm_idx);
#endif

      // find neighboring nodes

      int neighbors[N_NBRS_MAX];

#ifdef P4_TO_P8
      // zm plane
      neighbors[nn_00m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm];

      neighbors[nn_m0m] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
      neighbors[nn_p0m] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

      neighbors[nn_0mm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
      neighbors[nn_0pm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];

      neighbors[nn_mmm] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
      neighbors[nn_pmm] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
      neighbors[nn_mpm] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
      neighbors[nn_ppm] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];

      // z0 plane
      neighbors[nn_000] = n;

      neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpm];
      neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm];

      neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmm];
      neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm];

      neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmm];
      neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmm];
      neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpm];
      neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppm];

      // zp plane
      neighbors[nn_00p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp];

      neighbors[nn_m0p] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mpp];
      neighbors[nn_p0p] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmp];

      neighbors[nn_0mp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_pmp];
      neighbors[nn_0pp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpp];

      neighbors[nn_mmp] = nodes->local_nodes[P4EST_CHILDREN*quad_mmp_idx + dir::v_mmp];
      neighbors[nn_pmp] = nodes->local_nodes[P4EST_CHILDREN*quad_pmp_idx + dir::v_pmp];
      neighbors[nn_mpp] = nodes->local_nodes[P4EST_CHILDREN*quad_mpp_idx + dir::v_mpp];
      neighbors[nn_ppp] = nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_ppp];

#else
      neighbors[nn_000] = n;

      neighbors[nn_m00] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm];
      neighbors[nn_p00] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm];

      neighbors[nn_0m0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm];
      neighbors[nn_0p0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm];

      neighbors[nn_mm0] = nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mmm];
      neighbors[nn_pm0] = nodes->local_nodes[P4EST_CHILDREN*quad_pmm_idx + dir::v_pmm];
      neighbors[nn_mp0] = nodes->local_nodes[P4EST_CHILDREN*quad_mpm_idx + dir::v_mpm];
      neighbors[nn_pp0] = nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_ppm];
#endif

      // fetch values of LSF
      for (int i_phi = 0; i_phi < n_phis; i_phi++)
      {
        int k = 0;
        for (int i = 0; i < N_NBRS_MAX; i++)
          if (neighbor_exists[i])
          {
            phi_cube[i_phi][k] = phi_p[i_phi][neighbors[i]];
            phi_xx_cube[i_phi][k] = phi_xx_p[i_phi][neighbors[i]];
            phi_yy_cube[i_phi][k] = phi_yy_p[i_phi][neighbors[i]];
#ifdef P4_TO_P8
            phi_zz_cube[i_phi][k] = phi_zz_p[i_phi][neighbors[i]];
#endif
            k++;
          }
      }

#ifdef P4_TO_P8
      cube3_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube, zm_cube, zp_cube);
#else
      cube2_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube);
#endif

//#ifdef P4_TO_P8
//    cube3_refined_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube, zm_cube, zp_cube);
//#else
//    cube2_refined_mls_t cube(xm_cube, xp_cube, ym_cube, yp_cube);
//#endif

#ifdef P4_TO_P8
      cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, phi_zz_cube, *action, *color);
      cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, zm_grid, zp_grid, nx_grid, ny_grid, nz_grid);
      double cell_vol = (xp_cube-xm_cube)*(yp_cube-ym_cube)*(zp_cube-zm_cube);
//    cube.construct_domain(1,1,1,cube_refinement);
#else
      cube.set_phi(phi_cube, phi_xx_cube, phi_yy_cube, *action, *color);
      cube.set_interpolation_grid(xm_grid, xp_grid, ym_grid, yp_grid, nx_grid, ny_grid);
      double cell_vol = (xp_cube-xm_cube)*(yp_cube-ym_cube);
//    cube.construct_domain(1,1,cube_refinement);
#endif
      cube.construct_domain();
    }

    std::vector<double> phi_000(n_phi, 0), phi_p00(n_phi, 0), phi_m00(n_phi, 0), phi_0m0(n_phi, 0), phi_0p0(n_phi, 0);
#ifdef P4_TO_P8
    std::vector<double> phi_00m(n_phi, 0), phi_00p(n_phi, 0);
#endif

    for (int i = 0; i < n_phi; i++)
    {
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i], phi_00m[i], phi_00p[i]);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i]);
#endif
    }

    if(is_node_Wall(p4est, ni) &&
   #ifdef P4_TO_P8
          (*bc_)[0].wallType(x_C,y_C,z_C) == DIRICHLET
   #else
          (*bc_)[0].wallType(x_C,y_C) == DIRICHLET
   #endif
       )
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

        w_m00 *= wi * mu_; w_p00 *= wi * mu_;
        w_0m0 *= wj * mu_; w_0p0 *= wj * mu_;
        w_00m *= wk * mu_; w_00p *= wk * mu_;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
        w_00m /= w_000; w_00p /= w_000;

        if (keep_scalling) scalling_p[n] = w_000;

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

        w_m00 *= weight_on_Dxx*mu_;
        w_p00 *= weight_on_Dxx*mu_;
        w_0m0 *= weight_on_Dyy*mu_;
        w_0p0 *= weight_on_Dyy*mu_;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

        double diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);
        w_m00 /= diag;
        w_p00 /= diag;
        w_0m0 /= diag;
        w_0p0 /= diag;

        if (keep_scalling) scalling_p[n] = diag;

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

        if(add_p[n] > 0) matrix_has_nullspace = false;
        continue;

      } break;

      case NODE_NMN:
      {
        double volume_cut_cell = cube.measure_of_domain();

          // LHS
          double w_000 = 0;

          double w[9] = {0,0,0,0,0,0,0,0,0};
          w_000 += add_p[n]*volume_cut_cell;

          for (int q = 5; q < 9; q++)
          {
            if (node_in[q])
            {
              double l = sqrt(SQR(x_cent[q]-x_cent[0])+SQR(y_cent[q]-y_cent[0]));
              w[q] = -mu_*cube_refined.measure_of_interface(-2000-q)/l;
              w_000 -= w[q];
            } else {
              w[q] = 0;
            }
          }

          int q;
//          q = 1; if (node_in[q]) {w[q] = -mu_*cube_refined.measure_in_dir(dir::f_m00)/dx_min; w_000 -= w[q];}
//          q = 2; if (node_in[q]) {w[q] = -mu_*cube_refined.measure_in_dir(dir::f_p00)/dx_min; w_000 -= w[q];}
//          q = 3; if (node_in[q]) {w[q] = -mu_*cube_refined.measure_in_dir(dir::f_0m0)/dy_min; w_000 -= w[q];}
//          q = 4; if (node_in[q]) {w[q] = -mu_*cube_refined.measure_in_dir(dir::f_0p0)/dy_min; w_000 -= w[q];}

          std::vector<double> f_x(n_nodes, 0), f_y(n_nodes,0);
          for (int i = 0; i < nx+1; i++)
            for (int j = 0; j < ny+1; j++)
            {
              f_x[i + j*(nx+1)] = (x_coord[i]-x_C)/dx_min;
              f_y[i + j*(nx+1)] = (y_coord[j]-y_C)/dy_min;
            }

          std::vector<double> f_x_aux(n_nodes_aux, 0), f_y_aux(n_nodes_aux,0);
          for (int i = 0; i < nx_aux+1; i++)
            for (int j = 0; j < ny_aux+1; j++)
            {
              f_x_aux[i + j*(nx_aux+1)] = (x_coord_aux[i]-x_C)/dx_min;
              f_y_aux[i + j*(nx_aux+1)] = (y_coord_aux[j]-y_C)/dy_min;
            }

          double s = 0;
          double eps_theta = 1.e-5;

          q = 1;
          s = cube_refined.measure_in_dir(dir::f_m00);
          if (s > eps_ifc)
          {
            double theta = cube_refined.integrate_in_dir(f_y.data(), dir::f_m00)/s;
            double Bn = 0, Bp = 0, at_c = 0;

            if (node_in[3] && node_in[5]) Bn = 1;
            if (node_in[4] && node_in[7]) Bp = 1;

            if (theta < -eps_theta) Bp = 0;
            else if (theta > eps_theta) Bn = 0;
            else {Bp = 0; Bn = 0;}

            if (Bp + Bn < EPS) at_c = 1;

            w_000 += mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dx_min;
            w[q]  -= mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dx_min;
            w[3]  += mu_*Bn*fabs(theta)*s/dx_min;
            w[5]  -= mu_*Bn*fabs(theta)*s/dx_min;
            w[4]  += mu_*Bp*fabs(theta)*s/dx_min;
            w[7]  -= mu_*Bp*fabs(theta)*s/dx_min;
          }

          q = 2;
          s = cube_refined.measure_in_dir(dir::f_p00);
          if (s > eps_ifc)
          {
            double theta = cube_refined.integrate_in_dir(f_y.data(), dir::f_p00)/s;
            double Bn = 0, Bp = 0, at_c = 0;

            if (node_in[3] && node_in[6]) Bn = 1;
            if (node_in[4] && node_in[8]) Bp = 1;

            if (theta < -eps_theta) Bp = 0;
            else if (theta > eps_theta) Bn = 0;
            else {Bp = 0; Bn = 0;}

            if (Bp + Bn < EPS) at_c = 1;

            w_000 += mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dx_min;
            w[q]  -= mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dx_min;
            w[3]  += mu_*Bn*fabs(theta)*s/dx_min;
            w[6]  -= mu_*Bn*fabs(theta)*s/dx_min;
            w[4]  += mu_*Bp*fabs(theta)*s/dx_min;
            w[8]  -= mu_*Bp*fabs(theta)*s/dx_min;
          }

          q = 3;
          s = cube_refined.measure_in_dir(dir::f_0m0);
          if (s > eps_ifc)
          {
            double theta = cube_refined.integrate_in_dir(f_x.data(), dir::f_0m0)/s;
            double Bn = 0, Bp = 0, at_c = 0;

            if (node_in[1] && node_in[5]) Bn = 1;
            if (node_in[2] && node_in[6]) Bp = 1;

            if (theta < -eps_theta) Bp = 0;
            else if (theta > eps_theta) Bn = 0;
            else {Bp = 0; Bn = 0;}

            if (Bp + Bn < EPS) at_c = 1;

            w_000 += mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dy_min;
            w[q]  -= mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dy_min;
            w[1]  += mu_*Bn*fabs(theta)*s/dy_min;
            w[5]  -= mu_*Bn*fabs(theta)*s/dy_min;
            w[2]  += mu_*Bp*fabs(theta)*s/dy_min;
            w[6]  -= mu_*Bp*fabs(theta)*s/dy_min;
          }

          q = 4;
          s = cube_refined.measure_in_dir(dir::f_0p0);
          if (s > eps_ifc)
          {
            double theta = cube_refined.integrate_in_dir(f_x.data(), dir::f_0p0)/s;
            double Bn = 0, Bp = 0, at_c = 0;

            if (node_in[1] && node_in[7]) Bn = 1;
            if (node_in[2] && node_in[8]) Bp = 1;

            if (theta < -eps_theta) Bp = 0;
            else if (theta > eps_theta) Bn = 0;
            else {Bp = 0; Bn = 0;}

            if (Bp + Bn < EPS) at_c = 1;

            w_000 += mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dy_min;
            w[q]  -= mu_*(at_c + (1.-at_c)*(1.-fabs(theta)))*s/dy_min;
            w[1]  += mu_*Bn*fabs(theta)*s/dy_min;
            w[7]  -= mu_*Bn*fabs(theta)*s/dy_min;
            w[2]  += mu_*Bp*fabs(theta)*s/dy_min;
            w[8]  -= mu_*Bp*fabs(theta)*s/dy_min;
          }

          // contribution through boundary
          std::vector<int> present_interfaces;
          for (int i = 0; i < n_phi; i++)
            if (cube_refined.measure_of_interface(i) > eps_ifc)
              present_interfaces.push_back(i);

          int n_fcs = present_interfaces.size();

          if (n_fcs > 0)
          {
            double A_000_b = 0, A_000_f = 0, A_000_c = 0;
            double A_0m0_b = 0, A_0m0_f = 0, A_0m0_c = 0;
            double A_0p0_b = 0, A_0p0_f = 0, A_0p0_c = 0;

            double B_000_b = 0, B_000_f = 0, B_000_c = 0;
            double B_m00_b = 0, B_m00_f = 0, B_m00_c = 0;
            double B_p00_b = 0, B_p00_f = 0, B_p00_c = 0;

            if      (node_in[1] && node_in[2])  {A_000_c = 1;}
            else if (node_in[1] && node_in[0])  {A_000_b = 1;}
            else if (node_in[0] && node_in[2])  {A_000_f = 1;}

            else if (node_in[5] && node_in[6])  {A_0m0_c = 1;}
            else if (node_in[7] && node_in[8])  {A_0p0_c = 1;}

            else if (node_in[5] && node_in[3])  {A_0m0_b = 1;}
            else if (node_in[3] && node_in[6])  {A_0m0_f = 1;}

            else if (node_in[7] && node_in[4])  {A_0p0_b = 1;}
            else if (node_in[4] && node_in[8])  {A_0p0_f = 1;}


            if      (node_in[3] && node_in[4])  {B_000_c = 1;}
            else if (node_in[3] && node_in[0])  {B_000_b = 1;}
            else if (node_in[0] && node_in[4])  {B_000_f = 1;}

            else if (node_in[5] && node_in[7])  {B_m00_c = 1;}
            else if (node_in[6] && node_in[8])  {B_p00_c = 1;}

            else if (node_in[5] && node_in[1])  {B_m00_b = 1;}
            else if (node_in[1] && node_in[7])  {B_m00_f = 1;}

            else if (node_in[6] && node_in[2])  {B_p00_b = 1;}
            else if (node_in[2] && node_in[8])  {B_p00_f = 1;}

            std::vector<double> a(n_neighbors,0), b(n_neighbors,0), c(n_neighbors,0);

            a[nn_000] = +(A_000_b-A_000_f);     b[nn_000] = +(B_000_b-B_000_f);     c[nn_000] = 1.;

            a[nn_m00] = -(0.5*A_000_c+A_000_b); b[nn_m00] = +(B_m00_b-B_m00_f);     c[nn_m00] = 0.;
            a[nn_p00] = +(0.5*A_000_c+A_000_f); b[nn_p00] = +(B_p00_b-B_p00_f);     c[nn_p00] = 0.;
            a[nn_0m0] = +(A_0m0_b-A_0m0_f);     b[nn_0m0] = -(0.5*B_000_c+B_000_b); c[nn_0m0] = 0.;
            a[nn_0p0] = +(A_0p0_b-A_0p0_f);     b[nn_0p0] = +(0.5*B_000_c+B_000_f); c[nn_0p0] = 0.;

            a[nn_mm0] = -(0.5*A_0m0_c+A_0m0_b); b[nn_mm0] = -(0.5*B_m00_c+B_m00_b); c[nn_mm0] = 0.;
            a[nn_pm0] = +(0.5*A_0m0_c+A_0m0_f); b[nn_pm0] = -(0.5*B_p00_c+B_p00_b); c[nn_pm0] = 0.;
            a[nn_mp0] = -(0.5*A_0p0_c+A_0p0_b); b[nn_mp0] = +(0.5*B_m00_c+B_m00_f); c[nn_mp0] = 0.;
            a[nn_pp0] = +(0.5*A_0p0_c+A_0p0_f); b[nn_pp0] = +(0.5*B_p00_c+B_p00_f); c[nn_pp0] = 0.;

            std::vector< std::vector<double> > u_coeff(n_neighbors, std::vector<double> (n_neighbors, 0));

            for (int i = 0; i < n_neighbors; i++)
              for (int j = 0; j < n_neighbors; j++)
                u_coeff[i][j] = a[i]*(x_n[j]-x_C)/dx_min + b[i]*(y_n[j]-y_C)/dy_min + c[i];

            std::vector<double> f2integrate(n_neighbors, 0);

            for (int q = 0; q < n_fcs; q++)
            {
              int i_phi = present_interfaces[q];
              if (bc_types[i_phi] == ROBIN)
              {
                if (fabs(robin_coef[i_phi](n,x_C,y_C)) > 0) matrix_has_nullspace = false;

                if ((*action_)[i_phi] == COLORATION)
                {
                  for (int j = 0; j < i_phi; j++)
                  {
                    w_000 += mu_*robin_coef_p[i_phi][n]*cube_refined.integrate_over_colored_interface(u_coeff.data(), j,i_phi);
                  }
                } else {
                  for (int j = 0; j < n_neighbors; j++)
                  {
                    for (int k = 0; k < n_neighbors; k++)
                      f2integrate[k] = robin_coef[i_phi](k, x_n[k], y_n[k])*u_coeff[j];

                    w[j] += mu_*cube_refined.integrate_over_interface(f2integrate, i_phi);
                  }
                }
              }
            }
          }

          for (int q = 1; q < 9; q++)
          {
            w[q] /= w_000;
            if (fabs(w[q]) > EPS && node_in[q])
            {
              PetscInt node_g = petsc_gloidx[node[q]];
              ierr = MatSetValue(A, node_000_g, node_g, w[q], ADD_VALUES); CHKERRXX(ierr);
            }
          }

          if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
            fixed_value_idx_l = n;
            fixed_value_idx_g = node_000_g;
          }
          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0,   ADD_VALUES); CHKERRXX(ierr);

          if(add_p[n] > 0) matrix_has_nullspace = false;

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

  ierr = VecRestoreArray(phi_eff, &phi_eff_p); CHKERRXX(ierr);

  for (int i = 0; i < n_phis; i++)
  {
    ierr = VecRestoreArray(phi->at(i), &phi_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_xx->at(i), &phi_xx_p[i]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_yy->at(i), &phi_yy_p[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_zz->at(i), &phi_zz_p[i]); CHKERRXX(ierr);
#endif
  }

  // check for null space
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

//void my_p4est_poisson_nodes_mls_t::setup_negative_laplace_rhsvec_non_sym()
//{
//  // register for logging purpose
//  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

////  double eps = 1E-6*d_min*d_min;
//  double eps = 1E-9*d_min;
//#ifdef P4_TO_P8
//  double eps_dom = eps*eps*eps;
//  double eps_ifc = eps*eps;
//#else
//  double eps_dom = eps*eps;
//  double eps_ifc = eps;
//#endif

//  double *add_p;  ierr = VecGetArray(add_, &add_p); CHKERRXX(ierr);
//  double *rhs_p;  ierr = VecGetArray(rhs_, &rhs_p); CHKERRXX(ierr);
////  Vec rhs_dup;
////  ierr = VecDuplicate(rhs_, &rhs_dup);  CHKERRXX(ierr);
////  ierr = VecCopy(rhs_, rhs_dup);  CHKERRXX(ierr);

//  int n_phi = phi_->size(); // number of level set functions

//  std::vector<double *> phi_p(n_phi, NULL);
//  for (int i_phi = 0; i_phi < n_phi; i_phi++) {ierr = VecGetArray((*phi_)[i_phi], &phi_p[i_phi]); CHKERRXX(ierr);}

//  std::vector<double *> robin_coef_p(n_phi, NULL);

//  for (int i = 0; i < n_phi; i++)
//  {
//    if (robin_coef_ && (*robin_coef_)[i]) {ierr = VecGetArray((*robin_coef_)[i], &robin_coef_p[i]); CHKERRXX(ierr);}
//    else                                {robin_coef_p[i] = NULL;}
//  }

//  std::vector<double> phi_cube(n_phi*P4EST_CHILDREN, -1);

//  double *scalling_p;
//  if (keep_scalling)
//  {
//    ierr = VecGetArray(scalling, &scalling_p); CHKERRXX(ierr);
//  }

//  int n_nodes = 1;

//  int nx = 2*cube_refinement;   std::vector<double> x_coord(nx+1, 0);   n_nodes *= (nx+1);
//  int ny = 2*cube_refinement;   std::vector<double> y_coord(ny+1, 0);   n_nodes *= (ny+1);
//#ifdef P4_TO_P8
//  int nz = 2*cube_refinement;   std::vector<double> z_coord(nz+1, 0);   n_nodes *= (nz+1);
//#endif

//  std::vector<double> phi_cube_refined(n_nodes*n_phi, -1);
//  std::vector<double> phix_cube_refined(n_nodes*n_phi, 0);
//  std::vector<double> phiy_cube_refined(n_nodes*n_phi, 0);
//  std::vector<double> phixx_cube_refined(n_nodes*n_phi, 0);
//  std::vector<double> phixy_cube_refined(n_nodes*n_phi, 0);
//  std::vector<double> phiyy_cube_refined(n_nodes*n_phi, 0);

//  int n_nodes_aux = 1;

//  int nx_aux = 4*cube_refinement;   std::vector<double> x_coord_aux(nx_aux+1, 0);   n_nodes_aux *= (nx_aux+1);
//  int ny_aux = 4*cube_refinement;   std::vector<double> y_coord_aux(ny_aux+1, 0);   n_nodes_aux *= (ny_aux+1);
//#ifdef P4_TO_P8
//  int nz_aux = 4*cube_refinement;   std::vector<double> z_coord_aux(nz_aux+1, 0);   n_nodes_aux *= (nz_aux+1);
//#endif

//  std::vector<double> phi_cube_refined_aux(n_nodes_aux*n_phi, -1);
//  // create a cube
//#ifdef P4_TO_P8
//  cube3_mls_t cube;
//  cube3_refined_mls_t cube_refined;
//#else
//  cube2_mls_t cube;
//  cube2_refined_quad_mls_t cube_refined;
//  cube2_refined_mls_t cube_refined_aux;
//#endif

//  bc_vec.clear();
////  std::vector<double *> bc_vec_p(n_phi, NULL);
//  for (int i_phi = 0; i_phi < n_phi; i_phi++)
//  {
//    bc_vec.push_back(Vec());
//    ierr = VecCreateGhostNodes(p4est, nodes, &bc_vec.back()); CHKERRXX(ierr);
//    sample_cf_on_nodes(p4est, nodes, (*bc_)[i_phi].getInterfaceValue(), bc_vec.back());
////    ierr = VecGetArray(bc_vec.back(), &(bc_vec_p[i_phi])); CHKERRXX(ierr);
//  }

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

//  p4est_locidx_t node_mm0;
//  p4est_locidx_t node_pm0;
//  p4est_locidx_t node_mp0;
//  p4est_locidx_t node_pp0;
//#endif

//  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
//  {
//    // tree information
//    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

//    //---------------------------------------------------------------------
//    // Information at neighboring nodes
//    //---------------------------------------------------------------------
//    double x_C  = node_x_fr_n(n, p4est, nodes);
//    double y_C  = node_y_fr_n(n, p4est, nodes);
//#ifdef P4_TO_P8
//    double z_C  = node_z_fr_n(n, p4est, nodes);
//#endif


//    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

//    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
//    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
//#ifdef P4_TO_P8
//    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
//#endif

//    /*
//     * NOTE: All nodes are in PETSc' local numbering
//     */
//    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
//    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
//    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
//    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
//#ifdef P4_TO_P8
//    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
//    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
//    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
//    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

//    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
//    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
//    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
//    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
//#endif

//    if (node_loc[n] == NODE_NMN) find_centroid(node_in[0], alt[0], x_cent[0], y_cent[0], n);


//    if (node_loc[n] == NODE_NMN)
//    {
////      p4est_locidx_t quad_mm0, quad_tree_mm0;
////      p4est_locidx_t quad_pm0, quad_tree_pm0;
////      p4est_locidx_t quad_mp0, quad_tree_mp0;
////      p4est_locidx_t quad_pp0, quad_tree_pp0;

////      node_neighbors_->find_neighbor_cell_of_node(n, -1, -1, quad_mm0, quad_tree_mm0);
////      node_neighbors_->find_neighbor_cell_of_node(n, +1, -1, quad_pm0, quad_tree_pm0);
////      node_neighbors_->find_neighbor_cell_of_node(n, -1, +1, quad_mp0, quad_tree_mp0);
////      node_neighbors_->find_neighbor_cell_of_node(n, +1, +1, quad_pp0, quad_tree_pp0);

////      double xyz_mm0 [] = {x_C - 0.5*dx_min, y_C - 0.5*dy_min};

////      p4est_quadrant_t quad_mm0;
////      std::vector<p4est_quadrant_t> remote_matches;
////      node_neighbors_->hierarchy->find_smallest_quadrant_containing_point(xyz_mm0, quad_mm0, remote_matches);
////      double F [] = {(*(*phi_cf_)[0])(x_C - dx_min, y_C - dy_min),
////                     (*(*phi_cf_)[0])(x_C, y_C - dy_min),
////                     (*(*phi_cf_)[0])(x_C - dx_min, y_C),
////                     (*(*phi_cf_)[0])(x_C, y_C)};
////      phi_interp.set_input((*phi_)[0], (*phi_xx_)[0], (*phi_yy_)[0], quadratic);
////      double phi_int = phi_interp(x_C - 0.5*dx_min, y_C - 0.5*dy_min);
////      double phi_exact = (*(*phi_cf_)[0])(x_C - 0.5*dx_min, y_C - 0.5*dy_min);
////      double phi_exact = phi_interp(x_C - 0.5*dx_min, y_C - 0.5*dy_min);

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

//      quad_neighbor_nodes_of_node_t qnnn_m00 = node_neighbors_->get_neighbors(node_m00);
//      quad_neighbor_nodes_of_node_t qnnn_p00 = node_neighbors_->get_neighbors(node_p00);

//#ifdef P4_TO_P8
//#else
//      node_mm0 = qnnn_m00.d_0m0_m0==0 ? qnnn_m00.node_0m0_mm : qnnn_m00.node_0m0_pm;
//      node_mp0 = qnnn_m00.d_0p0_m0==0 ? qnnn_m00.node_0p0_mm : qnnn_m00.node_0p0_pm;

//      node_pm0 = qnnn_p00.d_0m0_m0==0 ? qnnn_p00.node_0m0_mm : qnnn_p00.node_0m0_pm;
//      node_pp0 = qnnn_p00.d_0p0_m0==0 ? qnnn_p00.node_0p0_mm : qnnn_p00.node_0p0_pm;
//#endif

//      node[0] = n;
//      node[1] = node_m00; node[2] = node_p00; node[3] = node_0m0; node[4] = node_0p0;
//      node[5] = node_mm0; node[6] = node_pm0; node[7] = node_mp0; node[8] = node_pp0;

//      for (int q = 1; q < 9; q++)
//      {
//        find_centroid(node_in[q], alt[q], x_cent[q], y_cent[q], node[q], NULL);
//      }

//      cube.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-.5*dx_min;
//      cube.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+.5*dx_min;
//      cube.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-.5*dy_min;
//      cube.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+.5*dy_min;

//      cube_refined.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-.5*dx_min;
//      cube_refined.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+.5*dx_min;
//      cube_refined.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-.5*dy_min;
//      cube_refined.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+.5*dy_min;

//  #ifdef P4_TO_P8
//      cube.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-.5*dz_min;
//      cube.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+.5*dz_min;

//      cube_refined.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-.5*dz_min;
//      cube_refined.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+.5*dz_min;
//  #endif

//      nx = 2*cube_refinement;
//      ny = 2*cube_refinement;

//      // expand cube
//      if (!node_in[1]) {cube.x0 -= 0.5*dx_min; cube_refined.x0 -= 0.5*dx_min; nx += cube_refinement;}
//      if (!node_in[2]) {cube.x1 += 0.5*dx_min; cube_refined.x1 += 0.5*dx_min; nx += cube_refinement;}
//      if (!node_in[3]) {cube.y0 -= 0.5*dy_min; cube_refined.y0 -= 0.5*dy_min; ny += cube_refinement;}
//      if (!node_in[4]) {cube.y1 += 0.5*dy_min; cube_refined.y1 += 0.5*dy_min; ny += cube_refinement;}

////      nx = 2*cube_refinement;
////      ny = 2*cube_refinement;

////      // expand cube
////      if (!node_in[1]) {cube.x0 -= dx_min; cube_refined.x0 -= dx_min; nx += 2*cube_refinement;}
////      if (!node_in[2]) {cube.x1 += dx_min; cube_refined.x1 += dx_min; nx += 2*cube_refinement;}
////      if (!node_in[3]) {cube.y0 -= dy_min; cube_refined.y0 -= dy_min; ny += 2*cube_refinement;}
////      if (!node_in[4]) {cube.y1 += dy_min; cube_refined.y1 += dy_min; ny += 2*cube_refinement;}

//      x_coord.resize(nx+1);
//      y_coord.resize(ny+1);

//      double dx = (cube.x1 - cube.x0)/(double)(nx); for (int i = 0; i < nx+1; i++) {x_coord[i] = cube.x0 + dx*(double)(i);}
//      double dy = (cube.y1 - cube.y0)/(double)(ny); for (int j = 0; j < ny+1; j++) {y_coord[j] = cube.y0 + dy*(double)(j);}
//  #ifdef P4_TO_P8
//      double dz = (cube.z1 - cube.z0)/(double)(nz); for (int k = 0; k < nz+1; k++) {z_coord[k] = cube.z0 + dz*(double)(k);}
//  #endif

//      // count diagonal neighbors
//      int more_phi = 0;
//      if (node_in[5] && (!node_in[1] || !node_in[3])) more_phi++;
//      if (node_in[6] && (!node_in[2] || !node_in[3])) more_phi++;
//      if (node_in[7] && (!node_in[1] || !node_in[4])) more_phi++;
//      if (node_in[8] && (!node_in[2] || !node_in[4])) more_phi++;

//      n_nodes = (nx+1)*(ny+1);

//      // resize the array with LSF value to accomodate auxiliary LSF's
//      phi_cube_refined.resize((n_phi+more_phi)*n_nodes, -1);
//      phix_cube_refined.resize((n_phi+more_phi)*n_nodes, 0);
//      phiy_cube_refined.resize((n_phi+more_phi)*n_nodes, 0);
//      phixx_cube_refined.resize((n_phi+more_phi)*n_nodes, 0);
//      phixy_cube_refined.resize((n_phi+more_phi)*n_nodes, 0);
//      phiyy_cube_refined.resize((n_phi+more_phi)*n_nodes, 0);

//      // fetch values of normal LSF's
//      for (int i_phi = 0; i_phi < n_phi; i_phi++)
//      {
//  #ifdef P4_TO_P8
//        phi_interp.set_input((*phi_)[i_phi], (*phi_xx_)[i_phi], (*phi_yy_)[i_phi], (*phi_zz_)[i_phi], linear);
//  #else
//        phi_interp.set_input((*phi_)[i_phi], (*phi_xx_)[i_phi], (*phi_yy_)[i_phi], quadratic);
//  #endif
//        for (int i = 0; i < nx+1; i++)
//          for (int j = 0; j < ny+1; j++)
//  #ifdef P4_TO_P8
//            for (int k = 0; k < nz+1; k++)
//  //            phi_cube_refined[i_phi*n_nodes + k*(nx+1)*(ny+1) + j*(nx+1) + i] = phi_interp(x_coord[i], y_coord[j], z_coord[k]);
//              phi_cube_refined[i_phi*n_nodes + k*(nx+1)*(ny+1) + j*(nx+1) + i] = (*(*phi_cf_)[i_phi])(x_coord[i], y_coord[j], z_coord[k]);
//  #else
////            phi_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = phi_interp(x_coord[i], y_coord[j]);
//          {
//            phi_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phi_cf_)[i_phi])(x_coord[i], y_coord[j]);
//            phix_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phix_cf_)[i_phi])(x_coord[i], y_coord[j]);
//            phiy_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phiy_cf_)[i_phi])(x_coord[i], y_coord[j]);
//            phixx_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phixx_cf_)[i_phi])(x_coord[i], y_coord[j]);
//            phixy_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phixy_cf_)[i_phi])(x_coord[i], y_coord[j]);
//            phiyy_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phiyy_cf_)[i_phi])(x_coord[i], y_coord[j]);
//          }
//  #endif
//      }

//      // fetch values of auxiliary LSF's
//      std::vector<action_t> action_loc  = *action_;
//      std::vector<int>      color_loc   = *color_;

//      int i_phi = n_phi;

//      int q = 5;
//      if (node_in[q] && (!node_in[1] || !node_in[3]))
//      {
//        double x_0 = 0.5*(x_cent[q] + x_cent[0]);
//        double y_0 = 0.5*(y_cent[q] + y_cent[0]);
//        double l = sqrt(SQR(x_cent[q] - x_cent[0]) + SQR(y_cent[q] - y_cent[0]));
//        double a = (x_cent[q] - x_cent[0])/l + EPS;
//        double b = (y_cent[q] - y_cent[0])/l;
//        for (int i = 0; i < nx+1; i++)
//          for (int j = 0; j < ny+1; j++)
//            phi_cube_refined[i_phi*n_nodes + i + j*(nx+1)] = a*(x_coord[i] - x_0) + b*(y_coord[j] - y_0) + EPS;
//        action_loc.push_back(INTERSECTION);
//        color_loc.push_back(-2000-q);
//        i_phi++;
//      }

//      q = 6;
//      if (node_in[q] && (!node_in[2] || !node_in[3]))
//      {
//        double x_0 = 0.5*(x_cent[q] + x_cent[0]);
//        double y_0 = 0.5*(y_cent[q] + y_cent[0]);
//        double l = sqrt(SQR(x_cent[q] - x_cent[0]) + SQR(y_cent[q] - y_cent[0]));
//        double a = (x_cent[q] - x_cent[0])/l + EPS;
//        double b = (y_cent[q] - y_cent[0])/l;
//        for (int i = 0; i < nx+1; i++)
//          for (int j = 0; j < ny+1; j++)
//            phi_cube_refined[i_phi*n_nodes + i + j*(nx+1)] = a*(x_coord[i] - x_0) + b*(y_coord[j] - y_0) + EPS;
//        action_loc.push_back(INTERSECTION);
//        color_loc.push_back(-2000-q);
//        i_phi++;
//      }

//      q = 7;
//      if (node_in[q] && (!node_in[1] || !node_in[4]))
//      {
//        double x_0 = 0.5*(x_cent[q] + x_cent[0]);
//        double y_0 = 0.5*(y_cent[q] + y_cent[0]);
//        double l = sqrt(SQR(x_cent[q] - x_cent[0]) + SQR(y_cent[q] - y_cent[0]));
//        double a = (x_cent[q] - x_cent[0])/l + EPS;
//        double b = (y_cent[q] - y_cent[0])/l;
//        for (int i = 0; i < nx+1; i++)
//          for (int j = 0; j < ny+1; j++)
//            phi_cube_refined[i_phi*n_nodes + i + j*(nx+1)] = a*(x_coord[i] - x_0) + b*(y_coord[j] - y_0) + EPS;
//        action_loc.push_back(INTERSECTION);
//        color_loc.push_back(-2000-q);
//        i_phi++;
//      }

//      q = 8;
//      if (node_in[q] && (!node_in[2] || !node_in[4]))
//      {
//        double x_0 = 0.5*(x_cent[q] + x_cent[0]);
//        double y_0 = 0.5*(y_cent[q] + y_cent[0]);
//        double l = sqrt(SQR(x_cent[q] - x_cent[0]) + SQR(y_cent[q] - y_cent[0]));
//        double a = (x_cent[q] - x_cent[0])/l + EPS;
//        double b = (y_cent[q] - y_cent[0])/l;
//        for (int i = 0; i < nx+1; i++)
//          for (int j = 0; j < ny+1; j++)
//            phi_cube_refined[i_phi*n_nodes + i + j*(nx+1)] = a*(x_coord[i] - x_0) + b*(y_coord[j] - y_0) + EPS;
//        action_loc.push_back(INTERSECTION);
//        color_loc.push_back(-2000-q);
//        i_phi++;
//      }


//#ifdef P4_TO_P8
//    cube_refined.construct_domain(nx, ny, nz, phi_cube_refined.data(), *action_, *color_);
//#else
////    cube_refined.construct_domain(nx, ny, phi_cube_refined.data(), phixx_cube_refined.data(), phixy_cube_refined.data(), phiyy_cube_refined.data(), action_loc, color_loc);
//    cube_refined.construct_domain(nx, ny,
//                                  phi_cube_refined.data(),
//                                  phix_cube_refined.data(), phiy_cube_refined.data(),
//                                  phixx_cube_refined.data(), phixy_cube_refined.data(), phiyy_cube_refined.data(),
//                                  action_loc, color_loc);
//#endif

//    node_vol[n] = cube_refined.measure_of_domain();
//    }

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

//    if(is_node_Wall(p4est, ni)
//       &&
//#ifdef P4_TO_P8
//       (*bc_)[0].wallType(x_C,y_C,z_C) == DIRICHLET
//#else
//       (*bc_)[0].wallType(x_C,y_C) == DIRICHLET
//#endif
//       )
//    {
//#ifdef P4_TO_P8
//      rhs_p[n] = bc_strength*(*bc_)[0].wallValue(x_C,y_C,z_C);
//#else
//      rhs_p[n] = bc_strength*(*bc_)[0].wallValue(x_C,y_C);
//#endif
//      continue;
//    }
//    else
//    {
//      switch (node_loc[n])
//      {
//      case NODE_DIR: rhs_p[n] = 0; break; // no DIRICHLET BC at the moment
//      case NODE_OUT: rhs_p[n] = 0; break;
//      case NODE_INS:
//      {
//#ifdef P4_TO_P8
//        double diag;
//        if (!keep_scalling)
//        {
//          //------------------------------------
//          // Dfxx =   fxx + a*fyy + b*fzz
//          // Dfyy = c*fxx +   fyy + d*fzz
//          // Dfzz = e*fxx + f*fyy +   fzz
//          //------------------------------------
//          double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
//          double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

//          double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
//          double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

//          double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
//          double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

//          //------------------------------------------------------------
//          // compensating the error of linear interpolation at T-junction using
//          // the derivative in the transversal direction
//          //
//          // Laplace = wi*Dfxx +
//          //           wj*Dfyy +
//          //           wk*Dfzz
//          //------------------------------------------------------------
//          double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
//          double wi = (1.-c-e+c*f+e*d-d*f)/det;
//          double wj = (1.-a-f+a*e+f*b-b*e)/det;
//          double wk = (1.-b-d+b*c+d*a-a*c)/det;

//          //---------------------------------------------------------------------
//          // Shortley-Weller method, dimension by dimension
//          //---------------------------------------------------------------------
//          double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

//          if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
//          else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
//          else                               w_m00 += -2./d_m00/(d_m00+d_p00);

//          if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
//          else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
//          else                               w_p00 += -2./d_p00/(d_m00+d_p00);

//          if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
//          else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
//          else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

//          if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
//          else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
//          else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

//          if(is_node_zmWall(p4est, ni))      w_00p += -1./(d_00p*d_00p);
//          else if(is_node_zpWall(p4est, ni)) w_00m += -1./(d_00m*d_00m);
//          else                               w_00m += -2./d_00m/(d_00m+d_00p);

//          if(is_node_zpWall(p4est, ni))      w_00m += -1./(d_00m*d_00m);
//          else if(is_node_zmWall(p4est, ni)) w_00p += -1./(d_00p*d_00p);
//          else                               w_00p += -2./d_00p/(d_00m+d_00p);

//          w_m00 *= wi * mu_; w_p00 *= wi * mu_;
//          w_0m0 *= wj * mu_; w_0p0 *= wj * mu_;
//          w_00m *= wk * mu_; w_00p *= wk * mu_;

//          //---------------------------------------------------------------------
//          // diag scaling
//          //---------------------------------------------------------------------
//          diag = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
//        } else {
//          diag = scalling_p[n];
//        }
//        rhs_p[n] /= diag;
//#else
//        double diag;
//        if (!keep_scalling)
//        {
//          //---------------------------------------------------------------------
//          // Shortley-Weller method, dimension by dimension
//          //---------------------------------------------------------------------
//          double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

//          if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
//          else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
//          else                               w_m00 += -2./d_m00/(d_m00+d_p00);

//          if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
//          else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
//          else                               w_p00 += -2./d_p00/(d_m00+d_p00);

//          if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
//          else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
//          else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

//          if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
//          else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
//          else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

//          //---------------------------------------------------------------------
//          // compensating the error of linear interpolation at T-junction using
//          // the derivative in the transversal direction
//          //---------------------------------------------------------------------
//          double weight_on_Dyy = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);
//          double weight_on_Dxx = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);

//          w_m00 *= weight_on_Dxx*mu_;
//          w_p00 *= weight_on_Dxx*mu_;
//          w_0m0 *= weight_on_Dyy*mu_;
//          w_0p0 *= weight_on_Dyy*mu_;

//          diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);
//        } else {
//          diag = scalling_p[n];
//        }

//        rhs_p[n] /= diag;
//#endif
//      } break;

//      case NODE_NMN:
//      {
//        double volume_cut_cell = cube_refined.measure_of_domain();
//        // LHS
//        double w_000 = 0;

//        if (!keep_scalling) {
//        } else {
//          w_000 = scalling_p[n];
//        }

//        // RHS
//        double bc_value[P4EST_CHILDREN];

//        std::vector<double> bc_value_refined(n_neighbors, 0);

//        // integrate force
//        rhs_p[n] = rhs(n, x_C, y_C)*volume_cut_cell;

//        // integrate normal flux through interfaces
//        for (int i_phi = 0; i_phi < n_phi; i_phi++)
//        {
//          if (cube_refined.measure_of_interface(i_phi) > eps_ifc)
//          {
//            double integral_bc = 0;

//            if (interface_value[i_phi].cf == NULL) // if no cf is provided
//            {
//              for (int i = 0; i < n_neighbors; i++)
//                bc_value_refined[i] = interface_value[i_phi](neighbors[i], x_n[i], y_n[i]);

//              if ((*action_)[i_phi] == COLORATION)
//              {
//                for (int i = 0; i < i_phi; i++)
//                  integral_bc += cube_refined.integrate_over_colored_interface(bc_value_refined.data(), i, i_phi);
//              } else {
//                integral_bc = cube_refined.integrate_over_interface(bc_value_refined.data(), i_phi);
//              }
//            } else { // if cf is provided
//              if ((*action_)[i_phi] == COLORATION)
//              {
//                for (int i = 0; i < i_phi; i++)
//                  integral_bc += cube_refined.integrate_over_colored_interface(interface_value[i].cf, i, i_phi);
//              } else {
//                integral_bc = cube_refined.integrate_over_interface(interface_value[i].cf, i_phi);
//              }
//            }

//            rhs_p[n] += mu_*integral_bc;
//          }
//        }

//        rhs_p[n] /= w_000;
//      } break;
//      }

//    }
//  }

//  for (int i_phi = 0; i_phi < n_phi; i_phi++)
//  {
////    ierr = VecRestoreArray(bc_vec[i_phi], &(bc_vec_p[i_phi])); CHKERRXX(ierr);
//    ierr = VecDestroy(bc_vec[i_phi]); CHKERRXX(ierr);
//  }
//  bc_vec.clear();

//  if (matrix_has_nullspace && fixed_value_idx_l >= 0){
//    rhs_p[fixed_value_idx_l] = 0;
//  }

//  ierr = VecRestoreArray(add_, &add_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(rhs_, &rhs_p); CHKERRXX(ierr);
////  ierr = VecDestroy(rhs_dup); CHKERRXX(ierr);

//  if (keep_scalling) {ierr = VecRestoreArray(scalling, &scalling_p); CHKERRXX(ierr);}

//  for (int i = 0; i < n_phi; i++)
//  {
//    if (robin_coef_p[i]) {ierr = VecRestoreArray((*robin_coef_)[i], &robin_coef_p[i]);  CHKERRXX(ierr);}
//                          ierr = VecRestoreArray((*phi_)[i],        &phi_p[i]);         CHKERRXX(ierr);
//  }

//  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
//}

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
