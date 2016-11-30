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
#include <AtomTree.h>


class BioMolecule: public CF_3
{
  friend class BioMoleculeSolver;
  std::vector<Atom> atoms;
  std::vector<Atom> atoms_by_increasing_x;





  const mpi_environment_t& mpi;
  const my_p4est_brick_t &myb;
  double xc_, yc_, zc_, s_, rmax_;
  double rp_;

  double D_, L_, Dx_, Dy_, Dz_;
  bool tree_built;

  std::vector<double> cell_buffer;

public:
  AtomTree atom_tree;
  bool fast_gen;
  BioMolecule(my_p4est_brick_t& brick, const mpi_environment_t& mpi);
  void read(const std::string& pqr);
  void translate(double xc, double yc, double zc);
  void set_scale(double s);
  double get_scale() const;
  int get_number_of_atoms() const;
  void set_probe_radius(double rp);
  void subtract_probe_radius(Vec phi);
  void construct_SES_by_reinitialization(p4est_t* &p4est, p4est_nodes_t *&nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void construct_SES_by_advection(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void remove_internal_cavities(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  double operator()(double x, double y, double z) const;
  void reduce_to_single_atom();
  void atoms_per_node(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &atom_count);
  void use_fast_surface_generation();
  void use_brute_force_surface_generation();
  void atoms_queried_per_node(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &atom_count);
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
