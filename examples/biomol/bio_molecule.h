#ifndef BIO_MOLECULE_H
#define BIO_MOLECULE_H

#include <fstream>
#include <string>
#include <vector>

#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>

struct Atom {
  double x, y, z, q, r;
};

class BioMolecule: public CF_3
{
  friend class BioMoleculeSolver;
  std::vector<Atom> atoms;

  const mpi_environment_t& mpi;
  double xc_, yc_, zc_, s_, rmax_;
  double rp_;

  double D_, L_, Dx_, Dy_, Dz_;
  bool is_partitioned;

  typedef std::vector<std::vector<int> > atom_mapping_t;
  atom_mapping_t cell2atom;

  std::vector<double> cell_buffer;

public:
  BioMolecule(my_p4est_brick_t& brick, const mpi_environment_t& mpi);
  void read(const std::string& pqr);
  void translate(double xc, double yc, double zc);
  void set_scale(double s);
  double get_scale() const;
  void set_probe_radius(double rp);
  void subtract_probe_radius(Vec phi);
  void partition_atoms();
  void construct_SES_by_reinitialization(p4est_t* &p4est, p4est_nodes_t *&nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void construct_SES_by_advection(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void remove_internal_cavities(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  double operator()(double x, double y, double z) const;
};

class BioMoleculeSolver{
  const BioMolecule* mol;
  p4est_t* p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t& brick;

  my_p4est_hierarchy_t hierarchy;
  my_p4est_node_neighbors_t neighbors;

  typedef enum {
    linearPB,
    nonlinearPB
  } solver_type;

  double edl, mue_p, mue_m;

  Vec phi, psi_bar;

  void solve_singular_part();

public:
  BioMoleculeSolver(const BioMolecule& mol, p4est_t* p4est, p4est_nodes_t* nodes, p4est_ghost_t *ghost, my_p4est_brick_t& brick);
  void set_electrolyte_parameters(double edl, double mue_p, double mue_m);
  void set_phi(Vec phi);
  void solve_linear(Vec& psi_molecule, Vec &psi_electrolyte);
  void solve_nonlinear(Vec& psi_molecule, Vec &psi_electrolyte, int itmax = 10, double tol = 1e-6);
};

#endif // BIO_MOLECULE_H
