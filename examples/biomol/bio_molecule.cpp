#include "bio_molecule.h"
#include <iostream>
#include <sstream>
#include <iomanip>

#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/my_p8est_poisson_node_base_jump.h>
#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

using namespace std;

#define MAX_BLOCK_SIZE 32
#define FAST_SURFACE_COMPUTATION

istream& operator >> (istream& is, Atom& atom) {
  string ignore [4];
  for (int i=0; i<4; i++) is >> ignore[i];
  is >> atom.x >> atom.y >> atom.z >> atom.q >> atom.r;
  return is;
}

ostream& operator << (ostream& os, Atom& atom) {
  os << "(x = " << atom.x << ", y = " << atom.y << ", z = " << atom.z << "; q = " << atom.q << ", r = " << atom.r << ")";
  return os;
}

BioMolecule::BioMolecule(my_p4est_brick_t& brick, const mpi_context_t &mpi)
  : mpi(mpi), xc_(0.), yc_(0.), zc_(0.), rp_(1.4)
{
  Dx_ = brick.nxyztrees[0];
  Dy_ = brick.nxyztrees[1];
  Dz_ = brick.nxyztrees[2];

  D_ = MIN(Dx_, MIN(Dy_, Dz_));
}

void BioMolecule::read(const string &pqr) {

  // only read on rank 0 and then broadcast the result to others
  if (mpi.mpirank == 0) {

    std::ifstream reader(pqr.c_str());
#ifdef CASL_THROWS
    if (!reader)
      throw std::invalid_argument("could not open the pqr file");
#endif

    // parse line by line
    string line;
    while(getline(reader, line)) {
      istringstream iss(line);

      string keyword; iss >> keyword;
      Atom atom;
      if (keyword == "ATOM") {
        iss >> atom;
        atoms.push_back(atom);
      }
    }
  }

  size_t msg_size = atoms.size()*sizeof(Atom);
  MPI_Bcast(&msg_size, 1, MPI_UNSIGNED_LONG, 0, mpi.mpicomm);
  if (mpi.mpirank != 0)
    atoms.resize(msg_size/sizeof(Atom));
  MPI_Bcast(&atoms[0], msg_size, MPI_BYTE, 0, mpi.mpicomm);

  // compute the center of mass
  xc_ = 0;
  yc_ = 0;
  zc_ = 0;
  for (size_t i = 0; i<atoms.size(); i++){
    xc_ += atoms[i].x;
    yc_ += atoms[i].y;
    zc_ += atoms[i].z;
  }
  xc_ /= atoms.size();
  yc_ /= atoms.size();
  zc_ /= atoms.size();

  // compute the size of the bounding box
  L_ = 0;
  for (size_t i = 0; i<atoms.size(); i++){
    L_ = MAX(L_, fabs(atoms[i].x - xc_));
    L_ = MAX(L_, fabs(atoms[i].y - yc_));
    L_ = MAX(L_, fabs(atoms[i].z - zc_));
  }
  L_ *= 2.5;

  // scale and recenter the molecule to middle
  translate(0.5*Dx_, 0.5*Dy_, 0.5*Dz_);
  scale(0.25);
  partition_atoms();
}

void BioMolecule::translate(double xc, double yc, double zc) {

  is_partitioned = false;

  // move the atoms to the new location
  for (size_t i = 0; i<atoms.size(); i++){
    atoms[i].x += (xc - xc_);
    atoms[i].y += (yc - yc_);
    atoms[i].z += (zc - zc_);
  }

  xc_ = xc;
  yc_ = yc;
  zc_ = zc;
}

void BioMolecule::scale(double s) {
  is_partitioned = false;

  // reset the position of atoms
  s_ = s*D_/L_;
  rmax_ = 0;
  for (size_t i = 0; i<atoms.size(); i++){
    Atom& a = atoms[i];

    a.x  = xc_ + s_*(a.x - xc_);
    a.y  = yc_ + s_*(a.y - yc_);
    a.z  = zc_ + s_*(a.z - zc_);
    a.r *= s_;

    rmax_ = MAX(rmax_, a.r);
  }

  rp_ *= s_;
  L_  *= s_;
}

void BioMolecule::partition_atoms(){
#ifdef FAST_SURFACE_COMPUTATION
  if (is_partitioned) return;

  int N = MIN((int)floor(L_/rmax_), MAX_BLOCK_SIZE); // to ensure locality

  double d = L_/N;
  double xmin = xc_ - 0.5*L_;
  double ymin = yc_ - 0.5*L_;
  double zmin = zc_ - 0.5*L_;

  cell2atom.resize(N*N*N);
  cell_buffer.resize((N+1)*(N+1)*(N+1));

  for (size_t i = 0; i<atoms.size(); i++) {
    const Atom& a = atoms[i];
    int ci = floor((a.x-xmin) / d);
    int cj = floor((a.y-ymin) / d);
    int ck = floor((a.z-zmin) / d);

    int cell_idx = N*N*ck + N*cj + ci;
    cell2atom[cell_idx].push_back(i);
  }

  // update buffer values
  for (int ck = 0; ck <= N; ck++){
    double z = ck*d + zmin;
    for (int cj = 0; cj <= N; cj++) {
      double y = cj*d + ymin;
      for (int ci = 0; ci <= N; ci++) {
        double x = ci*d + xmin;
        int idx = (N+1)*(N+1)*ck + (N+1)*cj + ci;

        const Atom& a = atoms[0];
        cell_buffer[idx] = rp_ + a.r - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z));
        for (size_t m = 1; m<atoms.size(); m++){
          const Atom& a = atoms[m];
          cell_buffer[idx] = MAX(cell_buffer[idx], rp_ + a.r - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));
        }
      }
    }
  }

  is_partitioned = true;
#endif
}

void BioMolecule::set_probe_radius(double rp) {
  rp_ = s_*rp;
}

void BioMolecule::subtract_probe_radius(Vec phi) {
  Vec phi_l;
  PetscErrorCode ierr = VecGhostGetLocalForm(phi, &phi_l); CHKERRXX(ierr);
  ierr = VecShift(phi_l, -rp_); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &phi_l); CHKERRXX(ierr);
}

double BioMolecule::operator ()(double x, double y, double z) const {
  double phi = -DBL_MAX;

#ifdef FAST_SURFACE_COMPUTATION
  // find the cell index for the point
  int N = MIN((int)floor(L_/rmax_), MAX_BLOCK_SIZE);
  double d = L_/N;
  double xmin = xc_ - 0.5*L_;
  double ymin = yc_ - 0.5*L_;
  double zmin = zc_ - 0.5*L_;

  int ci = floor((x-xmin) / d);
  int cj = floor((y-ymin) / d);
  int ck = floor((z-zmin) / d);

  if (ci < 0)        ci = 0;
  else if (ci > N-1) ci = N-1;
  if (cj < 0)        cj = 0;
  else if (cj > N-1) cj = N-1;
  if (ck < 0)        ck = 0;
  else if (ck > N-1) ck = N-1;

  bool is_phi_computed = false;
  for (int k = ck-1; k <= ck+1; k++){
    if (k<0 || k>=N) continue;

    for (int j = cj-1; j <= cj+1; j++){
      if (j<0 || j>=N) continue;

      for (int i = ci-1; i <= ci+1; i++){
        if (i<0 || i>=N) continue;

        int cell_idx = N*N*k + N*j + i;
        const vector<int>& mapping = cell2atom[cell_idx];

        for (size_t m = 0; m < mapping.size(); m++) {
          const Atom& a = atoms[mapping[m]];
          phi = MAX(phi, a.r + rp_ - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));
          is_phi_computed = true;
        }
      }
    }
  }

  if (!is_phi_computed) {
    double s [] = {(x-xmin)/d - ci, (ci+1) - (x-xmin)/d, (y-ymin)/d - cj, (cj+1) - (y-ymin)/d, (z-zmin)/d - ck, (ck+1) - (z-zmin)/d};
    bool is_outside_box = false;
    for (int i=0; i<2*P4EST_DIM; i++){
      if (s[i] < 0)      {s[i] = 0; is_outside_box = true;}
      else if (s[i] > 1) {s[i] = 1; is_outside_box = true;}
    }
    double w [] = {s[1]*s[3]*s[5],
                   s[0]*s[3]*s[5],
                   s[1]*s[2]*s[5],
                   s[0]*s[2]*s[5],
                   s[1]*s[3]*s[4],
                   s[0]*s[3]*s[4],
                   s[1]*s[2]*s[4],
                   s[0]*s[2]*s[4]};

    int idx = (N+1)*(N+1)*ck + (N+1)*cj + ci;
    phi = w[0]*cell_buffer[idx] +
          w[1]*cell_buffer[idx + 1] +
          w[2]*cell_buffer[idx + (N+1)] +
          w[3]*cell_buffer[idx + (N+1) + 1] +
          w[4]*cell_buffer[idx + (N+1)*(N+1)] +
          w[5]*cell_buffer[idx + (N+1)*(N+1) + 1] +
          w[6]*cell_buffer[idx + (N+1)*(N+1) + N+1] +
          w[7]*cell_buffer[idx + (N+1)*(N+1) + N+1 + 1];

    if (is_outside_box)
      phi -= sqrt(SQR(x - (ci+0.5)*d - xmin) + SQR(y - (cj+0.5)*d - ymin) + SQR(z - (ck+0.5)*d - zmin));
  }
#else
  for (size_t m = 0; m < atoms.size(); m++) {
    const Atom& a = atoms[m];
    phi = MAX(phi, a.r + rp_ - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));
  }
#endif
  return phi;
}

void BioMolecule::construct_SES_by_reinitialization(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &phi)
{
  splitting_criteria_t *sp = (splitting_criteria_t*) p4est->user_pointer;
  const p4est_connectivity_t *connectivity = p4est->connectivity;

  // split based on the SAS distance
  parStopWatch w;
  w.start("making intial tree");
  splitting_criteria_threshold_cf_t sp_thr(sp->min_lvl, sp->max_lvl, -1.0*rp_, 1.5*rp_, this, sp->lip);
  p4est->user_pointer = &sp_thr;

	// refine and partition
	// Note: Using recursive on large molecules causes the whole thing to be build in serial on 
	// one processor which is obviously not scalable 
	for (int l = 0; l<sp->max_lvl; l++){
	  my_p4est_refine(p4est, P4EST_FALSE, refine_threshold_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);
	}

  PetscPrintf(p4est->mpicomm, "num of global quadrants = %ld\n", p4est->global_num_quadrants);

  // create the ghost layer
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // generate unique node indices
  nodes = my_p4est_nodes_new(p4est, ghost);

  // calculate the SAS on
  PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *this, phi);
  w.stop(); w.read_duration();


  // subtract off the probe radius to get SES
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set ls(&neighbors);

  int it = floor(4.0*rp_/ (D_/ (1<<sp->max_lvl)));
  ls.reinitialize_2nd_order(phi, MAX(it, 20));
  subtract_probe_radius(phi);

  /* construct a newby refining only close to the SES. We do this in a
     * level by level approach and tag appropriate cells in each step for
     * refinement
     */
  p4est_t *p4est_tmp = p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, NULL);
  p4est_ghost_t *ghost_tmp = NULL;
  p4est_nodes_t *nodes_tmp = NULL;
  Vec phi_tmp = NULL;

  for (int l = 0; l <= sp->max_lvl; l++) {
    my_p4est_partition(p4est_tmp, P4EST_TRUE, NULL);
    ghost_tmp = p4est_ghost_new(p4est_tmp, P4EST_CONNECT_FULL);
    nodes_tmp = my_p4est_nodes_new(p4est_tmp, ghost_tmp);
    ierr = VecCreateGhostNodes(p4est_tmp, nodes_tmp, &phi_tmp); CHKERRXX(ierr);

    /* buffer all the points in the current tree */
    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, &brick, &neighbors);
    double *phi_tmp_p;
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes_tmp->indep_nodes.elem_count; i++) {
      const p4est_indep_t *ni = (const p4est_indep_t *)sc_array_index(&nodes_tmp->indep_nodes, i);

      p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*ni->p.piggy3.which_tree + 0];

      double tree_xmin = connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = connectivity->vertices[3*v_mmm + 1];
      double tree_zmin = connectivity->vertices[3*v_mmm + 2];
      double xyz [P4EST_DIM] =
      {
        node_x_fr_i(ni) + tree_xmin,
        node_y_fr_j(ni) + tree_ymin,
        node_z_fr_k(ni) + tree_zmin
      };

      phi_interp.add_point_to_buffer(i, xyz);
    }
    phi_interp.set_input_parameters(phi, linear);
    phi_interp.interpolate(phi_tmp);

    /* dont do the final refinement */
    if (l == sp->max_lvl) break;

    /* refine the tree */
    splitting_criteria_discrete_t sp_tag(p4est_tmp, sp->min_lvl, sp->max_lvl, sp->lip);
    sp_tag.mark_cells_for_refinement(nodes_tmp, phi_tmp_p);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    p4est_tmp->user_pointer = &sp_tag;
    my_p4est_refine(p4est_tmp, P4EST_FALSE, refine_marked_quadrants, NULL);

    /* free memory */
    ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
    p4est_ghost_destroy(ghost_tmp);
    p4est_nodes_destroy(nodes_tmp);
  }

  /* free memory and reset pointers */
  p4est_destroy(p4est); p4est = p4est_tmp; p4est->user_pointer = sp;
  p4est_ghost_destroy(ghost); ghost = ghost_tmp;
  p4est_nodes_destroy(nodes); nodes = nodes_tmp;
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_tmp;
}

void BioMolecule::construct_SES_by_advection(p8est_t *&p4est, p8est_nodes_t *&nodes, p8est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &phi)
{
  // generate the SAS first
  splitting_criteria_t *sp = (splitting_criteria_t*) p4est->user_pointer;
  const p4est_connectivity_t *connectivity = p4est->connectivity;

  // split based on the SAS distance
  splitting_criteria_cf_t sp_cf(sp->min_lvl, sp->max_lvl, this, sp->lip);
  p4est->user_pointer = &sp_cf;
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // partition the p4est
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

  // create the ghost layer
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // generate unique node indices
  nodes = my_p4est_nodes_new(p4est, ghost);

  // calculate the SAS on
  PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *this, phi);

  // reinitialize the levelset
  {
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    neighbors.init_neighbors();
    my_p4est_level_set ls(&neighbors);

    ls.reinitialize_2nd_order(phi);
  }

  struct:CF_3{
    double operator()(double, double, double) const {return 1.0;}
  } vn;
  double t = 0, dt = 0, tf = rp_;

  for (; t<tf; t += dt) {
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    my_p4est_level_set ls(&neighbors);

    dt = ls.advect_in_normal_direction(vn, phi, tf - t);

    /* reconstruct the grid */
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);

    // define interpolating function on the old stuff
    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, &brick, &neighbors);
    phi_interp.set_input_parameters(phi, linear);
    splitting_criteria_cf_t sp_interp (sp->min_lvl, sp->max_lvl, &phi_interp, sp->lip);
    p4est_np1->user_pointer = &sp_interp;

    // refine and coarsen new p4est
    my_p4est_coarsen(p4est_np1, P4EST_TRUE, coarsen_levelset_cf, NULL);
    my_p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);

    // partition
    my_p4est_partition(p4est_np1, P4EST_TRUE, NULL);

    // recompute ghost and nodes
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // transfer solution to the new grid
    Vec phi_np1;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    for (size_t n=0; n<nodes_np1->indep_nodes.elem_count; n++)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
      double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif

      double xyz [] =
      {
        node_x_fr_i(node) + tree_xmin,
        node_y_fr_j(node) + tree_ymin
  #ifdef P4_TO_P8
        ,
        node_z_fr_k(node) + tree_zmin
  #endif
      };

      phi_interp.add_point_to_buffer(n, xyz);
    }
    phi_interp.interpolate(phi_np1);

    // get rid of old stuff and replace them with new
    ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;
    p4est_destroy(p4est); p4est = p4est_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
  }
  p4est->user_pointer = sp;

  // reinitialize the levelset
  {
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    neighbors.init_neighbors();
    my_p4est_level_set ls(&neighbors);

    ls.reinitialize_2nd_order(phi);
  }
}

void BioMolecule::remove_internal_cavities(p8est_t *&p4est, p8est_nodes_t *&nodes, p8est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &phi)
{
  Vec sol;
  PetscErrorCode ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);
  ierr = VecSet(sol, 0); CHKERRXX(ierr);

  // solve an auxiliary poisson to determine the cavities
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  my_p4est_level_set ls(&neighbors);

  neighbors.init_neighbors();
  {
    struct:WallBC3D{
      BoundaryConditionType operator()(double, double, double) const {return DIRICHLET;}
    } bc_wall_type;

    struct:CF_3{
      double operator()(double, double, double) const { return 10.; }
    } bc_wall_value;

    struct:CF_3{
      double operator()(double, double, double) const { return 0.; }
    } bc_interface_value;

    BoundaryConditions3D bc;
    bc.setInterfaceType(DIRICHLET);
    bc.setInterfaceValue(bc_interface_value);
    bc.setWallTypes(bc_wall_type);
    bc.setWallValues(bc_wall_value);

    PoissonSolverNodeBase solver(&neighbors);
    solver.set_bc(bc);

    solver.set_phi(phi);
    solver.set_rhs(sol);
    solver.set_tolerances(1e-8, 10);
    solver.solve(sol);
  }

  // remove the cavities
  double *sol_p, *phi_p;
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
    if (phi_p[i] <= 0 && fabs(sol_p[i]) < EPS) { // internal cavity
      phi_p[i] = -phi_p[i];
    }
  }
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecDestroy(sol); CHKERRXX(ierr);

  // reinitialize the levelset
  ls.reinitialize_2nd_order(phi);

  // construct the grid in a level-by-level approach
  splitting_criteria_t *sp = (splitting_criteria_t*) p4est->user_pointer;
  const p4est_connectivity_t *connectivity = p4est->connectivity;

  p4est_t *p4est_tmp = p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, NULL);
  p4est_ghost_t *ghost_tmp = NULL;
  p4est_nodes_t *nodes_tmp = NULL;
  Vec phi_tmp = NULL;

  for (int l = 0; l <= sp->max_lvl; l++) {
    my_p4est_partition(p4est_tmp, P4EST_TRUE, NULL);
    ghost_tmp = p4est_ghost_new(p4est_tmp, P4EST_CONNECT_FULL);
    nodes_tmp = my_p4est_nodes_new(p4est_tmp, ghost_tmp);
    ierr = VecCreateGhostNodes(p4est_tmp, nodes_tmp, &phi_tmp); CHKERRXX(ierr);

    /* buffer all the points in the current tree */
    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, &brick, &neighbors);
    double *phi_tmp_p;
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes_tmp->indep_nodes.elem_count; i++) {
      const p4est_indep_t *ni = (const p4est_indep_t *)sc_array_index(&nodes_tmp->indep_nodes, i);

      p4est_topidx_t v_mmm = connectivity->tree_to_vertex[P4EST_CHILDREN*ni->p.piggy3.which_tree + 0];

      double tree_xmin = connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = connectivity->vertices[3*v_mmm + 1];
      double tree_zmin = connectivity->vertices[3*v_mmm + 2];
      double xyz [P4EST_DIM] =
      {
        node_x_fr_i(ni) + tree_xmin,
        node_y_fr_j(ni) + tree_ymin,
        node_z_fr_k(ni) + tree_zmin
      };

      phi_interp.add_point_to_buffer(i, xyz);
    }
    phi_interp.set_input_parameters(phi, linear);
    phi_interp.interpolate(phi_tmp);

    /* dont do the final refinement */
    if (l == sp->max_lvl) break;

    /* refine the tree */
    splitting_criteria_discrete_t sp_tag(p4est_tmp, sp->min_lvl, sp->max_lvl, sp->lip);
    sp_tag.mark_cells_for_refinement(nodes_tmp, phi_tmp_p);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    p4est_tmp->user_pointer = &sp_tag;
    my_p4est_refine(p4est_tmp, P4EST_FALSE, refine_marked_quadrants, NULL);

    /* free memory */
    ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
    p4est_ghost_destroy(ghost_tmp);
    p4est_nodes_destroy(nodes_tmp);
  }

  /* free memory and reset pointers */
  p4est_destroy(p4est); p4est = p4est_tmp; p4est->user_pointer = sp;
  p4est_ghost_destroy(ghost); ghost = ghost_tmp;
  p4est_nodes_destroy(nodes); nodes = nodes_tmp;
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_tmp;

}

BioMoleculeSolver::BioMoleculeSolver(const BioMolecule &mol, p8est_t *p4est, p8est_nodes_t *nodes, p8est_ghost_t *ghost, my_p4est_brick_t &brick)
  : mol(&mol),
    p4est(p4est), nodes(nodes), ghost(ghost), brick(brick),
    hierarchy(p4est, ghost, &brick),
    neighbors(&hierarchy, nodes)
{}

void BioMoleculeSolver::set_electrolyte_parameters(double edl, double epsilon_molecule, double epsilon_electrolyte)
{
  this->edl = edl*mol->s_;
  this->mue_m = epsilon_molecule;
  this->mue_p = epsilon_electrolyte;
}

void BioMoleculeSolver::set_phi(Vec phi)
{
  this->phi = phi;
}

void BioMoleculeSolver::solve_singular_part()
{
  // compute the Coulombic potential
  PetscErrorCode ierr;

  // TODO: use FMM to compute this thing fast
  struct psi_star_cf:CF_3{
    const BioMolecule *mol;
    psi_star_cf(const BioMolecule* mol)
      : mol(mol)
    {}
    double operator()(double x, double y, double z) const {
      double psi_star = 0;
      for (size_t i = 0; i<mol->atoms.size(); i++){
        const Atom& a = mol->atoms[i];
        psi_star += a.q/(4.0*M_PI) / sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z));
      }
      return -psi_star;
    }
  };
  psi_star_cf negative_coulomb(mol);

  Vec psi_star;
  ierr = VecDuplicate(phi, &psi_bar); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &psi_star); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, negative_coulomb, psi_star);

  // solve a poisson equation to obtain the singular solution
  InterpolatingFunctionNodeBase bc_interface_value(p4est, nodes, ghost, &brick, &neighbors);
  bc_interface_value.set_input_parameters(psi_star, linear);

  double *psi_bar_p, *psi_star_p, *phi_p;
  ierr = VecGetArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_star, &psi_star_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
    psi_bar_p[i] = 0;
    phi_p[i]     = -phi_p[i]; // we are solving on omega^+
  }

  BoundaryConditions3D bc;
  struct:CF_3{
    double operator()(double, double, double) const { return 0;}
  } bc_wall_value;

  struct:WallBC3D{
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;

  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);

  PoissonSolverNodeBase solver(&neighbors);
  solver.set_bc(bc);
  solver.set_phi(phi);
  solver.set_mu(mue_p);
  solver.set_rhs(psi_bar);
  solver.solve(psi_bar);

  // extend psi_bar
  my_p4est_level_set ls(&neighbors);
  ls.extend_Over_Interface(phi, psi_bar, DIRICHLET, psi_star);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
    psi_bar_p[i] -= psi_star_p[i];

  ierr = VecRestoreArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_star, &psi_star_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecDestroy(psi_star); CHKERRXX(ierr);
}

void BioMoleculeSolver::solve_linear(Vec &psi_molecule, Vec& psi_electrolyte)
{
  PetscErrorCode ierr;

  solve_singular_part();
  PoissonSolverNodeBaseJump solver(&neighbors);

  Vec psi, add, jump;
  ierr = VecDuplicate(phi, &add); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &jump); CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &psi); CHKERRXX(ierr);

  // set the rhs
  double *psi_p, *phi_p, *add_p, *jump_p, *psi_bar_p;
  ierr = VecGetArray(psi,     &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi,     &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(add,     &add_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  ierr = VecGetArray(jump,    &jump_p); CHKERRXX(ierr);

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++) {
    // use psi to set the rhs
    psi_p[2*i] = psi_p[2*i + 1] = 0;

    // set the add to diagonal
    if (phi_p[i] >= 0)
      add_p[i] = SQR(1.0/edl);
    else
      add_p[i] = 0;

    // compute the jump
    const quad_neighbor_nodes_of_node_t qnnn = neighbors.get_neighbors(i);
    double nx = qnnn.dx_central(phi_p);
    double ny = qnnn.dy_central(phi_p);
    double nz = qnnn.dz_central(phi_p);
    double abs = MAX(sqrt(SQR(nx) + SQR(ny) + SQR(nz)), EPS);
    nx /= abs; ny /= abs; nz /= abs;

    jump_p[i] = -(nx*qnnn.dx_central(psi_bar_p) +
                  ny*qnnn.dy_central(psi_bar_p) +
                  nz*qnnn.dz_central(psi_bar_p));
  }
  ierr = VecRestoreArray(add,     &add_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump,    &jump_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(jump, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecGhostUpdateEnd(jump, INSERT_VALUES, SCATTER_FORWARD);

  InterpolatingFunctionNodeBase jump_flux(p4est, nodes, ghost, &brick, &neighbors);
  jump_flux.set_input_parameters(jump, linear);

  struct:CF_3{
    double operator()(double, double, double) const { return 0; }
  } jump_sol;

  struct:CF_3{
    double operator()(double, double, double) const { return 0; }
  } bc_wall_value;

  struct:WallBC3D {
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;

  BoundaryConditions3D bc;
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  solver.set_mue(mue_p, mue_m);
  solver.set_phi(phi);
  solver.set_diagonal(add);
  solver.set_bc(bc);
  solver.set_jump(jump_sol, jump_flux);
  solver.set_rhs(psi);
  solver.solve(psi);

  // Destroy unecessary vectors
  ierr = VecDestroy(add); CHKERRXX(ierr);
  ierr = VecDestroy(jump); CHKERRXX(ierr);

  // separate solutions
  double *psi_mol_p, *psi_elec_p;
  ierr = VecDuplicate(phi, &psi_molecule); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &psi_electrolyte); CHKERRXX(ierr);
  ierr = VecGetArray(psi_molecule, &psi_mol_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_electrolyte, &psi_elec_p); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++) {
    psi_mol_p[i]  = psi_p[2*i];
    psi_elec_p[i] = psi_p[2*i + 1];
  }

  // extend solutions
  my_p4est_level_set ls(&neighbors);
  ls.extend_Over_Interface_TVD(phi, psi_molecule);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++) {
    phi_p[i]      = -phi_p[i];
  }
  ls.extend_Over_Interface_TVD(phi, psi_electrolyte);

  // restore pointers
  ierr = VecRestoreArray(phi,             &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi,             &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_bar,         &psi_bar_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_molecule,    &psi_mol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_electrolyte, &psi_elec_p); CHKERRXX(ierr);

  // destroy temporary solution
  ierr = VecDestroy(psi); CHKERRXX(ierr);
  ierr = VecDestroy(psi_bar); CHKERRXX(ierr);
}

void BioMoleculeSolver::solve_nonlinear(Vec &psi_molecule, Vec& psi_electrolyte, int itmax, double tol)
{
  PetscErrorCode ierr;

  // better off to initialize the neighbors
  neighbors.init_neighbors();
  solve_singular_part();

  Vec psi, add, jump;
  ierr = VecDuplicate(phi, &add); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &jump); CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &psi); CHKERRXX(ierr);

  double *psi_p, *phi_p, *add_p, *jump_p, *psi_bar_p;
  ierr = VecGetArray(psi,     &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi,     &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(add,     &add_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_bar, &psi_bar_p); CHKERRXX(ierr);
  ierr = VecGetArray(jump,    &jump_p); CHKERRXX(ierr);

  // separate solutions
  double *psi_mol_p, *psi_elec_p;
  ierr = VecDuplicate(phi, &psi_molecule); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &psi_electrolyte); CHKERRXX(ierr);
  ierr = VecGetArray(psi_molecule, &psi_mol_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_electrolyte, &psi_elec_p); CHKERRXX(ierr);

  int it = 0;
  double err = 1 + tol;
  double kappa_sqr = SQR(1.0/edl);

  for (size_t i = 0; i < nodes->indep_nodes.elem_count; i++) {
    psi_mol_p[i] = psi_elec_p[i] = 0;
  }

  //calculate the jump
  for (p4est_locidx_t i = 0; i < nodes->num_owned_indeps; i++) {
    psi_mol_p[i] = psi_elec_p[i] = 0;

    // compute the jump
    // TODO: do the layering thingy!
    const quad_neighbor_nodes_of_node_t& qnnn = neighbors[i];
    double nx = qnnn.dx_central(phi_p);
    double ny = qnnn.dy_central(phi_p);
    double nz = qnnn.dz_central(phi_p);
    double abs = MAX(sqrt(SQR(nx) + SQR(ny) + SQR(nz)), EPS);
    nx /= abs; ny /= abs; nz /= abs;

    jump_p[i] = -(nx*qnnn.dx_central(psi_bar_p) +
                  ny*qnnn.dy_central(psi_bar_p) +
                  nz*qnnn.dz_central(psi_bar_p));
  }
  ierr = VecGhostUpdateBegin(jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(jump,    &jump_p); CHKERRXX(ierr);

  InterpolatingFunctionNodeBase jump_flux(p4est, nodes, ghost, &brick, &neighbors);
  jump_flux.set_input_parameters(jump, linear);

  struct:CF_3{
    double operator()(double, double, double) const { return 0; }
  } jump_sol;

  struct:CF_3{
    double operator()(double, double, double) const { return 0; }
  } bc_wall_value;

  struct:WallBC3D {
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;

  BoundaryConditions3D bc;
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);

  my_p4est_level_set ls(&neighbors);
  while (it++ < itmax && err > tol) {
    PoissonSolverNodeBaseJump solver(&neighbors);

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++) {
      // use psi to set the rhs
      psi_p[2*i] = 0;
      psi_p[2*i + 1] = -kappa_sqr*sinh(psi_elec_p[i]) + kappa_sqr*psi_elec_p[i]*cosh(psi_elec_p[i]);

      // set the add to diagonal
      if (phi_p[i] <= 0)
        add_p[i] = 0;
      else
        add_p[i] = kappa_sqr*cosh(psi_elec_p[i]);
    }

    solver.set_mue(mue_p, mue_m);
    solver.set_phi(phi);
    solver.set_diagonal(add);
    solver.set_bc(bc);
    solver.set_jump(jump_sol, jump_flux);
    solver.set_rhs(psi);
    solver.solve(psi);

    err = 0;
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++) {
      if (phi_p[i] < 0)
        err = MAX(err, fabs(psi_mol_p[i]  - psi_p[2*i  ]));
      else
        err = MAX(err, fabs(psi_elec_p[i] - psi_p[2*i+1]));
    }

    // reset solutions
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++) {
      psi_mol_p[i]  = psi_p[2*i];
      psi_elec_p[i] = psi_p[2*i + 1];
    }

    // extend solutions
    ls.extend_Over_Interface_TVD(phi, psi_molecule, 10);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
      phi_p[i] = -phi_p[i];
    ls.extend_Over_Interface_TVD(phi, psi_electrolyte, 10);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
      phi_p[i] = -phi_p[i];

    PetscPrintf(p4est->mpicomm, "It = %2d \t err = %1.5e\n", it, err);
  }

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
    phi_p[i] = -phi_p[i];

  // restore pointers
  ierr = VecRestoreArray(add,             &add_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(jump,            &jump_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi,             &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi,             &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_bar,         &psi_bar_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_molecule,    &psi_mol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_electrolyte, &psi_elec_p); CHKERRXX(ierr);

  // destroy temporary solution
  ierr = VecDestroy(add); CHKERRXX(ierr);
  ierr = VecDestroy(jump); CHKERRXX(ierr);
  ierr = VecDestroy(psi); CHKERRXX(ierr);
  ierr = VecDestroy(psi_bar); CHKERRXX(ierr);
}

