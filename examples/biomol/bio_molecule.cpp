#include "bio_molecule.h"
#include <iostream>
#include <sstream>
#include <iomanip>

#include <src/CASL_math.h>

using namespace std;

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
  : xc_(0.), yc_(0.), zc_(0.), s_(1.), mpi(mpi)
{
  D_ = MIN(brick.nxyztrees[0], MIN(brick.nxyztrees[1], brick.nxyztrees[2]));
  L_ = s_*D_;
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
  L_ *= 2.0;

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
  double alpha = s*D_/L_;
  rmax_ = 0;
  for (size_t i = 0; i<atoms.size(); i++){
    Atom& a = atoms[i];

    a.x  = xc_ + alpha*(a.x - xc_);
    a.y  = yc_ + alpha*(a.y - yc_);
    a.z  = zc_ + alpha*(a.z - zc_);
    a.r *= alpha;

    rmax_ = MAX(rmax_, a.r);
  }

  L_ = s*D_;
  s_ = s;
}

void BioMolecule::partition_atoms(){
  if (is_partitioned) return;

  int N = floor(L_/rmax_); // to ensure locality
  cell2atom.resize(N*N*N);

  double xmin = xc_ - 0.5*L_;
  double ymin = yc_ - 0.5*L_;
  double zmin = zc_ - 0.5*L_;

  for (size_t i = 0; i<atoms.size(); i++) {
    const Atom& a = atoms[i];
    int ci = floor((a.x - xmin) / L_);
    int cj = floor((a.y - ymin) / L_);
    int ck = floor((a.z - zmin) / L_);

    int cell_idx = N*N*ck + N*cj + ci;
    cell2atom[cell_idx].push_back(i);
  }

  is_partitioned = true;
}

double BioMolecule::operator ()(double x, double y, double z) const {
  double xmin = xc_ - 0.5*L_;
  double ymin = yc_ - 0.5*L_;
  double zmin = zc_ - 0.5*L_;

  // fins the cell index for the point
  int ci = floor((x - xmin) / L_);
  int cj = floor((y - ymin) / L_);
  int ck = floor((z - zmin) / L_);

  // clip to molecule's box boundary
  int N = floor(L_/rmax_);
  if      (ci <  0) ci = 0;
  else if (ci >= N) ci = N - 1;
  if      (cj <  0) cj = 0;
  else if (cj >= N) cj = N - 1;
  if      (ck <  0) ck = 0;
  else if (ck >= N) ck = N - 1;

  double phi = -100*L_;

//  for (size_t i = 0; i<atoms.size(); i++){
//    const Atom& a = atoms[i];
//    phi = MAX(phi, a.r - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));
//  }

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
          phi = MAX(phi, a.r - sqrt(SQR(x - a.x) + SQR(y - a.y) + SQR(z - a.z)));
        }
      }
    }
  }

  return phi;
}
