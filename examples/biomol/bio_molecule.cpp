#include "bio_molecule.h"
#include <iostream>
#include <sstream>
#include <iomanip>

#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_vtk.h>
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

    reader.open(pqr.c_str());
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
    reader.close();
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
  L_ *= 2.1;

  // scale and recenter the molecule to middle
  translate(0.5, 0.5, 0.5);
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

  int N = MIN((int)floor(D_/rmax_), MAX_BLOCK_SIZE); // to ensure locality
  double dx = Dx_/N;
  double dy = Dy_/N;
  double dz = Dz_/N;

  cell2atom.resize(N*N*N);
  cell_buffer.resize((N+1)*(N+1)*(N+1));

  for (size_t i = 0; i<atoms.size(); i++) {
    const Atom& a = atoms[i];
    int ci = floor(a.x / dx);
    int cj = floor(a.y / dy);
    int ck = floor(a.z / dz);

    int cell_idx = N*N*ck + N*cj + ci;
    cell2atom[cell_idx].push_back(i);    
  }

  // update buffer values
  for (int ck = 0; ck <= N; ck++){
    double z = ck*dx;
    for (int cj = 0; cj <= N; cj++) {
      double y = cj*dy;
      for (int ci = 0; ci <= N; ci++) {
        double x = ci*dz;
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
  int N = MIN((int)floor(D_/rmax_), MAX_BLOCK_SIZE);
  double dx = Dx_/N;
  double dy = Dy_/N;
  double dz = Dz_/N;

  int ci = floor(x / dx);
  int cj = floor(y / dy);
  int ck = floor(z / dz);

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
    double s [] = {x/dx - ci, (ci+1) - x/dx, y/dy - cj, (cj+1) - y/dy, z/dz - ck, (ck+1) - z/dz};
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
  const splitting_criteria_t *sp = (const splitting_criteria_t*) p4est->user_pointer;
  const p4est_connectivity_t *connectivity = p4est->connectivity;

  // split based on the SAS distance
  splitting_criteria_threshold_cf_t sp_thr(sp->min_lvl, sp->max_lvl, -0.05*rp_, 1.5*rp_, this, sp->lip);
  p4est->user_pointer = &sp_thr;
  p4est_refine(p4est, P4EST_TRUE, refine_threshold_cf, NULL);

  // partition the p4est
  p4est_partition(p4est, NULL);

  // create the ghost layer
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // generate unique node indices
  nodes = my_p4est_nodes_new(p4est, ghost);

  // calculate the SAS on
  PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *this, phi);

  // subtract off the probe radius to get SES
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set ls(&neighbors);

  ls.reinitialize_2nd_order(phi, 10);
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
    p4est_partition(p4est_tmp, NULL);
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

    VecGhostUpdateBegin(phi_tmp, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(phi_tmp, INSERT_VALUES, SCATTER_FORWARD);

    ostringstream oss;
    oss << "mol." << l;
    my_p4est_vtk_write_all(p4est_tmp, nodes_tmp, ghost_tmp,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_tmp_p);

    /* dont do the final refinement */
    if (l == sp->max_lvl) break;

    /* refine the tree */
    splitting_criteria_discrete_t sp_tag(p4est_tmp, sp->min_lvl, sp->max_lvl, sp->lip);
    sp_tag.mark_cells_for_refinement(nodes_tmp, phi_tmp_p);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    p4est_tmp->user_pointer = &sp_tag;
    p4est_refine(p4est_tmp, P4EST_FALSE, refine_marked_quadrants, NULL);

    /* free memory */
    ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);
    p4est_ghost_destroy(ghost_tmp);
    p4est_nodes_destroy(nodes_tmp);
  }

  /* free memory and reset pointers */
  p4est_destroy(p4est); p4est = p4est_tmp; p4est->user_pointer = &sp;
  p4est_ghost_destroy(ghost); ghost = ghost_tmp;
  p4est_nodes_destroy(nodes); nodes = nodes_tmp;
  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_tmp;
}




