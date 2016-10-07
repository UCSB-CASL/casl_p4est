#include "bio_molecule.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iterator>

#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_poisson_jump_nodes_extended.h>
#include <src/petsc_compatibility.h>
#include <src/math.h>

using namespace std;

#define MAX_BLOCK_SIZE 32
#define FAST_SURFACE_COMPUTATION_2
//#define FAST_SURFACE_COMPUTATION_3

int atoms_checked;
int eval_count;


int AtomTree::morton_from_indices(int index1, int index2, int index3) const
    { // pack 3 10-bit indices into a 30-bit Morton code
      index1 &= 0x000003ff;
      index2 &= 0x000003ff;
      index3 &= 0x000003ff;
      index1 |= ( index1 << 16 );
      index2 |= ( index2 << 16 );
      index3 |= ( index3 << 16 );
      index1 &= 0x030000ff;
      index2 &= 0x030000ff;
      index3 &= 0x030000ff;
      index1 |= ( index1 << 8 );
      index2 |= ( index2 << 8 );
      index3 |= ( index3 << 8 );
      index1 &= 0x0300f00f;
      index2 &= 0x0300f00f;
      index3 &= 0x0300f00f;
      index1 |= ( index1 << 4 );
      index2 |= ( index2 << 4 );
      index3 |= ( index3 << 4 );
      index1 &= 0x030c30c3;
      index2 &= 0x030c30c3;
      index3 &= 0x030c30c3;
      index1 |= ( index1 << 2 );
      index2 |= ( index2 << 2 );
      index3 |= ( index3 << 2 );
      index1 &= 0x09249249;
      index2 &= 0x09249249;
      index3 &= 0x09249249;
      return( index1 | ( index2 << 1 ) | ( index3 << 2 ) );
    }


void  AtomTree::build_tree(const vector<Atom> &atoms, const my_p4est_brick_t &brick_)
{
    brick=brick_;

    int level = 0; //root
    int i = max_i>>1; //root cell
    int j = max_i>>1;
    int k = max_i>>1;
    build_subtree(atoms, level, i, j, k);

}

void  AtomTree::build_subtree(const vector<Atom> &atoms, int level, int i, int j, int k)
{

    //static int cell_count = 1;
    std::vector<Atom> new_atoms;

    double dx = (brick.xyz_max[0] - brick.xyz_min[0])/(1<<level);
    double dy = (brick.xyz_max[1] - brick.xyz_min[1])/(1<<level);
    double dz = (brick.xyz_max[2] - brick.xyz_min[2])/(1<<level);

    double x = i*(brick.xyz_max[0] - brick.xyz_min[0])/max_i;
    double y = j*(brick.xyz_max[1] - brick.xyz_min[1])/max_i;
    double z = k*(brick.xyz_max[2] - brick.xyz_min[2])/max_i;

    double d = sqrt(dx*dx+dy*dy+dz*dz);
    double lip = 1.5;

    for (int a=0; a<atoms.size(); a++)
    {
        if(sqrt( (atoms[a].x - x)*(atoms[a].x - x)+
                 (atoms[a].y - y)*(atoms[a].y - y)+
                 (atoms[a].z - z)*(atoms[a].z - z)) <= 2*lip*d) //(.5*(d+lip*d)+rp)) //2*lip*d
        {
            new_atoms.push_back(atoms[a]);
        }
    }


    if(new_atoms.size()>0)
    {



        int cell_index = atoms_by_cell.size();

        int morton_index = morton_from_indices(i,j,k);

        cell_table[morton_index] = cell_index;
        atoms_by_cell.push_back(new_atoms);



        //if(level==5)
         //       cout<<i<<"\t"<<j<<"\t"<<k<<"\t"<<level<<"\t"<<cell_index<<"\t"<<morton_index<<"\t"<<new_atoms.size()<<endl;

        if(level<max_level && new_atoms.size()>1)
        {
            int off = (max_i>>(level+2));
            //cell_count+=8;
            build_subtree(new_atoms, level+1, i+off, j+off, k+off);
            build_subtree(new_atoms, level+1, i+off, j+off, k-off);
            build_subtree(new_atoms, level+1, i+off, j-off, k+off);
            build_subtree(new_atoms, level+1, i+off, j-off, k-off);
            build_subtree(new_atoms, level+1, i-off, j+off, k+off);
            build_subtree(new_atoms, level+1, i-off, j+off, k-off);
            build_subtree(new_atoms, level+1, i-off, j-off, k+off);
            build_subtree(new_atoms, level+1, i-off, j-off, k-off);
        }

    }



}

double AtomTree::dist_from_surface(double x, double y, double z) const
{

    const vector<Atom> *atoms;
    int cell_index;

    if( x<brick.xyz_min[0] || x >brick.xyz_max[0] ||
        y<brick.xyz_min[1] || y >brick.xyz_max[1] ||
        z<brick.xyz_min[2] || z >brick.xyz_max[2] ){ //outside range

        atoms = &atoms_by_cell[0];

    }
    else //inside domain
    {
        cell_index = find_smallest_cell_containing_point(x,y,z);
        atoms = &atoms_by_cell[cell_index];

    }


    double phi = -DBL_MAX;
    for (int m = 0; m <(*atoms).size(); m++) {
      const Atom& a = ((*atoms)[m]);
      double test = a.r + rp - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z));
      phi = MAX(phi, test);
    }

    eval_count++;
    atoms_checked += (*atoms).size();

    return phi;

}

double AtomTree::num_atoms(double x, double y, double z) const
{

    const vector<Atom> *atoms;
    int cell_index;

    if( x<brick.xyz_min[0] || x >brick.xyz_max[0] ||
        y<brick.xyz_min[1] || y >brick.xyz_max[1] ||
        z<brick.xyz_min[2] || z >brick.xyz_max[2] ){ //outside range

        atoms = &atoms_by_cell[0];

    }
    else //inside domain
    {
        cell_index = find_smallest_cell_containing_point(x,y,z);
        atoms = &atoms_by_cell[cell_index];

    }


    return 1.*(*atoms).size();

}
void AtomTree::temp(){
    atoms_checked_ = atoms_checked; eval_count_ = eval_count;}


int AtomTree::find_smallest_cell_containing_point(double x, double y, double z) const
{
    int morton_index = morton_from_indices(max_i>>1, max_i>>1, max_i>>1);
    int level=0;
    int cell_index=0;

    while(cell_table.find(morton_index) != cell_table.end() && level<=max_level)
    {
        cell_index = cell_table.at(morton_index);// cell_table[morton_index];
        //compute next level morton_index
        level++;
        int i,j,k;
        i = cell_i_fr_x(x,level);
        j = cell_j_fr_y(y,level);
        k = cell_k_fr_z(z,level);


        morton_index = morton_from_indices(i,j,k);
        //cout<<i<<" "<<j<<" "<<k<<" "<<level<<" "<<morton_index<<endl;
    }
    //cout<<level-1<<endl;
    return cell_index;


}


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

BioMolecule::BioMolecule(my_p4est_brick_t& brick, const mpi_environment_t &mpi)
  : mpi(mpi), myb(brick), xc_(0.), yc_(0.), zc_(0.), rp_(1.4), atom_tree(brick,rp_)
{
  Dx_ = brick.xyz_max[0] - brick.xyz_min[0];
  Dy_ = brick.xyz_max[1] - brick.xyz_min[1];
  Dz_ = brick.xyz_max[2] - brick.xyz_min[2];

  D_ = MIN(Dx_, MIN(Dy_, Dz_));
}

void BioMolecule::read(const string &pqr) {

  // only read on rank 0 and then broadcast the result to others
  if (mpi.rank() == 0) {

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
  MPI_Bcast(&msg_size, 1, MPI_UNSIGNED_LONG, 0, mpi.comm());
  if (mpi.rank() != 0)
    atoms.resize(msg_size/sizeof(Atom));
  MPI_Bcast(&atoms[0], msg_size, MPI_BYTE, 0, mpi.comm());

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
  translate(myb.xyz_min[0] + 0.5*Dx_,
            myb.xyz_min[1] + 0.5*Dy_,
            myb.xyz_min[2] + 0.5*Dz_);
  set_scale(0.25);
  partition_atoms();

  atom_tree.build_tree(atoms, myb);

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

void BioMolecule::set_scale(double s) {
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


  atom_tree.rp = rp_;
  atom_tree.rmax = rmax_;
}

double BioMolecule::get_scale() const {
  return s_;
}
int BioMolecule::get_number_of_atoms() const {
  return atoms.size();
}

void BioMolecule::reduce_to_single_atom()
{
    std::vector<Atom> atoms_new;

    atoms_new.push_back(atoms[0]);
    atoms = atoms_new;

}

void BioMolecule::atoms_per_node(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &atom_count)
{

    PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &atom_count); CHKERRXX(ierr);
    double *atom_count_p;

    ierr = VecGetArray(atom_count, &atom_count_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++) {
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est, nodes, xyz);

      atom_count_p[i] = atom_tree.num_atoms(xyz[0], xyz[1], xyz[2]);
    }

    ierr = VecRestoreArray(atom_count, &atom_count_p); CHKERRXX(ierr);



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


#ifdef FAST_SURFACE_COMPUTATION_2

atoms_by_increasing_x = atoms;
std::sort(atoms_by_increasing_x.begin(), atoms_by_increasing_x.end(), AtomOrderX());
//PetscPrintf(MPI_COMM_WORLD,"Number of atoms %d min %f max %f\n",atoms_by_increasing_x.size(), atoms_by_increasing_x[0].x, atoms_by_increasing_x[atoms_by_increasing_x.size()-1].x );

is_partitioned = true;
#endif
}

void BioMolecule::set_probe_radius(double rp) {
  rp_ = s_*rp;
  atom_tree.rp = rp_;
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
#ifdef FAST_SURFACE_COMPUTATION_3


#else
//#ifdef FAST_SURFACE_COMPUTATION_2
if(fast_gen)
{
    /*
std::vector<Atom>::const_iterator m, a_beg, a_end;

Atom a1,a2;
a1.x = x -  3*(rmax_+rp_);
a_beg = std::lower_bound(atoms_by_increasing_x.begin(), atoms_by_increasing_x.end(), a1, AtomOrderX());
a2.x = x +  3*(rmax_+rp_);
a_end=std::upper_bound(atoms_by_increasing_x.begin(), atoms_by_increasing_x.end(), a2, AtomOrderX());

//printf("%f %f %d %d\n",a1.x, a2.x, atoms_by_increasing_x.size(), std::distance(a_beg, a_end));

phi = MAX(phi, atoms[0].r + rp_ - sqrt(SQR(x - atoms[0].x) + SQR(y - atoms[0].y) + SQR(z - atoms[0].z)));

for (m = a_beg; m < a_end; m++) {
  const Atom& a = *m;
  phi = MAX(phi, a.r + rp_ - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));


}

*/
    phi=atom_tree.dist_from_surface(x,y,z);
}
else
{
//#else
  for (size_t m = 0; m < atoms.size(); m++) {
    const Atom& a = atoms[m];
    phi = MAX(phi, a.r + rp_ - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));
  }
}
//#endif
#endif
#endif
  return phi;
}

void BioMolecule::construct_SES_by_reinitialization(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &phi)
{

    atoms_checked=0;
    eval_count=0;

  splitting_criteria_t *sp = (splitting_criteria_t*) p4est->user_pointer;
  const p4est_connectivity_t *connectivity = p4est->connectivity;

  // split based on the SAS distance
  parStopWatch w;
  //w.start("making intial tree");
//  class distance_mask_t: public CF_3{
//    const BioMolecule& mol;
//    double d;
//  public:
//    distance_mask_t(const BioMolecule* mol, double d): mol(*mol), d(d) {}
//    double operator ()(double x, double y, double z) const {
//      return d - fabs(mol(x,y,z));
//    }
//  };

//  distance_mask_t mask(this, 1.5*rp_);
//  splitting_criteria_thresh_t sp_thr(sp->min_lvl, sp->max_lvl, &mask, 0);
  splitting_criteria_cf_t sp_thr(sp->min_lvl, sp->max_lvl, this, 3);
  p4est->user_pointer = &sp_thr;

    // refine and partition
    // Note: Using recursive on large molecules causes the whole thing to be build in serial on
    // one processor which is obviously not scalable
    for (int l = 0; l<sp->max_lvl; l++){
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);
    cout<<l<<" "<<p4est->local_num_quadrants<<endl;
    }

  // create the ghost layer
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // generate unique node indices
  nodes = my_p4est_nodes_new(p4est, ghost);

  // calculate the SAS on
  PetscErrorCode ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *this, phi);
  //w.stop(); w.read_duration();


  // subtract off the probe radius to get SES
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
  neighbors.init_neighbors();
  my_p4est_level_set_t ls(&neighbors);

//  int it = floor(4.0*rp_/ (D_/ (1<<sp->max_lvl)));
  ls.reinitialize_2nd_order(phi, 20 /*MAX(it, 20)*/ );
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
    my_p4est_interpolation_nodes_t phi_interp(&neighbors);
    double *phi_tmp_p;
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes_tmp->indep_nodes.elem_count; i++) {
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est_tmp, nodes_tmp, xyz);

      phi_interp.add_point(i, xyz);
    }
    phi_interp.set_input(phi, linear);
    phi_interp.interpolate(phi_tmp);


    /* dont do the final refinement */
    if (l == sp->max_lvl) break;

    /* refine the tree */
    splitting_criteria_tag_t sp_tag(sp->min_lvl, sp->max_lvl, sp->lip);
    sp_tag.refine_and_coarsen(p4est_tmp, nodes_tmp, phi_tmp_p);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

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


void BioMolecule::construct_SES_by_reinitialization_fast(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &phi)
{
  splitting_criteria_t *sp = (splitting_criteria_t*) p4est->user_pointer;
  const p4est_connectivity_t *connectivity = p4est->connectivity;

  // split based on the SAS distance
  parStopWatch w;
  w.start("making intial tree");

  splitting_criteria_cf_t sp_thr(sp->min_lvl, sp->max_lvl, this, 3);
  p4est->user_pointer = &sp_thr;

    // refine and partition
    // Note: Using recursive on large molecules causes the whole thing to be build in serial on
    // one processor which is obviously not scalable
    for (int l = 0; l<sp->max_lvl; l++){
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_TRUE, NULL);
    }

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
  my_p4est_level_set_t ls(&neighbors);

//  int it = floor(4.0*rp_/ (D_/ (1<<sp->max_lvl)));
  ls.reinitialize_2nd_order(phi, 20 /*MAX(it, 20)*/ );
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
    my_p4est_interpolation_nodes_t phi_interp(&neighbors);
    double *phi_tmp_p;
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes_tmp->indep_nodes.elem_count; i++) {
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est_tmp, nodes_tmp, xyz);

      phi_interp.add_point(i, xyz);
    }
    phi_interp.set_input(phi, linear);
    phi_interp.interpolate(phi_tmp);

    /* dont do the final refinement */
    if (l == sp->max_lvl) break;

    /* refine the tree */
    splitting_criteria_tag_t sp_tag(sp->min_lvl, sp->max_lvl, sp->lip);
    sp_tag.refine_and_coarsen(p4est_tmp, nodes_tmp, phi_tmp_p);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

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
    my_p4est_level_set_t ls(&neighbors);

    ls.reinitialize_2nd_order(phi);
  }

  struct:CF_3{
    double operator()(double, double, double) const {return 1.0;}
  } vn;
  double t = 0, dt = 0, tf = rp_;

  for (; t<tf; t += dt) {
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    my_p4est_level_set_t ls(&neighbors);

    dt = ls.advect_in_normal_direction(vn, phi);

    /* reconstruct the grid */
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);

    // define interpolating function on the old stuff
    my_p4est_interpolation_nodes_t phi_interp(&neighbors);
    phi_interp.set_input(phi, linear);
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
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);

      phi_interp.add_point(n, xyz);
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
    my_p4est_level_set_t ls(&neighbors);

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
  my_p4est_level_set_t ls(&neighbors);

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

    my_p4est_poisson_nodes_t solver(&neighbors);
    solver.set_bc(bc);

    solver.set_phi(phi);
    solver.set_rhs(sol);
    solver.set_tolerances(1e-6);
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
    my_p4est_interpolation_nodes_t phi_interp(&neighbors);
    double *phi_tmp_p;
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes_tmp->indep_nodes.elem_count; i++) {
      double xyz [P4EST_DIM];
      node_xyz_fr_n(i, p4est_tmp, nodes_tmp, xyz);

      phi_interp.add_point(i, xyz);
    }
    phi_interp.set_input(phi, linear);
    phi_interp.interpolate(phi_tmp);

    /* dont do the final refinement */
    if (l == sp->max_lvl) break;

    /* refine the tree */
    splitting_criteria_tag_t sp_tag(sp->min_lvl, sp->max_lvl, sp->lip);
    sp_tag.refine_and_coarsen(p4est_tmp, nodes_tmp, phi_tmp_p);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

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
  my_p4est_interpolation_nodes_t bc_interface_value(&neighbors);
  bc_interface_value.set_input(psi_star, linear);

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

  my_p4est_poisson_nodes_t solver(&neighbors);
  solver.set_bc(bc);
  solver.set_phi(phi);
  solver.set_mu(mue_p);
  solver.set_rhs(psi_bar);
  solver.solve(psi_bar);

  // extend psi_bar
  my_p4est_level_set_t ls(&neighbors);
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
  my_p4est_poisson_jump_nodes_extended_t solver(&neighbors);

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

  my_p4est_interpolation_nodes_t jump_flux(&neighbors);
  jump_flux.set_input(jump, linear);

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
  my_p4est_level_set_t ls(&neighbors);
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

  my_p4est_interpolation_nodes_t jump_flux(&neighbors);
  jump_flux.set_input(jump, linear);

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

  my_p4est_level_set_t ls(&neighbors);
  while (it++ < itmax && err > tol) {
    my_p4est_poisson_jump_nodes_extended_t solver(&neighbors);

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
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);

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

