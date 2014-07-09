#ifndef BIO_MOLECULE_H
#define BIO_MOLECULE_H

#include <fstream>
#include <string>
#include <vector>

#include <src/my_p4est_to_p8est.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>

struct Atom {
  double x, y, z, q, r;
};

class BioMolecule: public CF_3
{
  std::ifstream reader;
  std::vector<Atom> atoms;

  double xc_, yc_, zc_, s_, rmax_;
  const mpi_context_t& mpi;

  double D_, L_;
  bool is_partitioned;

  typedef std::vector<std::vector<int> > atom_mapping_t;
  atom_mapping_t cell2atom;

public:
  BioMolecule(my_p4est_brick_t& brick, const mpi_context_t& mpi);
  void read(const std::string& pqr);
  void translate(double xc, double yc, double zc);
  void scale(double s);
  void partition_atoms();
  double operator()(double x, double y, double z) const;
};

#endif // BIO_MOLECULE_H
