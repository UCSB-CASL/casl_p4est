#ifndef BIO_MOLECULE_H
#define BIO_MOLECULE_H

#include <fstream>
#include <string>
#include <vector>

#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>

struct Atom {
  double x, y, z, q, r;
};

class BioMolecule: public CF_3
{
  std::ifstream reader;
  std::vector<Atom> atoms;

  const mpi_context_t& mpi;
  double xc_, yc_, zc_, s_, rmax_;
  double rp_;

  double D_, L_, Dx_, Dy_, Dz_;
  bool is_partitioned;

  typedef std::vector<std::vector<int> > atom_mapping_t;
  atom_mapping_t cell2atom;

  std::vector<double> cell_buffer;

public:
  BioMolecule(my_p4est_brick_t& brick, const mpi_context_t& mpi);
  void read(const std::string& pqr);
  void translate(double xc, double yc, double zc);
  void scale(double s);
  void set_probe_radius(double rp);
  void subtract_probe_radius(Vec phi);
  void partition_atoms();
  void construct_SES_by_reinitialization(p4est_t* &p4est, p4est_nodes_t *&nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void construct_SES_by_advection(const splitting_criteria_cf_t *sp, p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, Vec& phi);
  double operator()(double x, double y, double z) const;
};


#endif // BIO_MOLECULE_H
