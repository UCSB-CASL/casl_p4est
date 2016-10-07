#ifndef BIO_MOLECULE_H
#define BIO_MOLECULE_H

#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>






struct Atom {
  double x, y, z, q, r;
};

struct AtomOrderX {
    inline bool operator() (const Atom& atom1, const Atom& atom2) {
            return (atom1.x < atom2.x);
        }

};


class AtomTree{

public:
    AtomTree(my_p4est_brick_t &brick_,double rp_): brick(brick_), rp(rp_){}
    void build_tree(const std::vector<Atom> &atoms, const my_p4est_brick_t &brick);
    double dist_from_surface(double x, double y, double z) const;
    double num_atoms(double x, double y, double z) const;



    std::unordered_map<int, int> cell_table;
    std::vector<std::vector<Atom>> atoms_by_cell;

    my_p4est_brick_t &brick;

    const static int max_i = 1024, max_level = 9;

    double min_dx;
    double rp;
    double rmax;

    int atoms_checked_;
    int eval_count_;

    void temp();


    int find_smallest_cell_containing_point(double x, double y, double z) const;



    int get_morton_index(int i, int j, int k) const;




    inline int cell_i_fr_x(double x, int level) const
    {
        double dx = (brick.xyz_max[0] - brick.xyz_min[0])/(1<<level);

        int i = (int) (x/dx)*(max_i>>level);
        i+= max_i>>(level+1);
        return i;
    }

    inline int cell_j_fr_y(double y, int level) const
    {
        double dy = (brick.xyz_max[1] - brick.xyz_min[1])/(1<<level);
        int j = (int)( y/dy)*(max_i>>level);
        j+= max_i>>(level+1);
        return j;
    }

    inline int cell_k_fr_z(double z, int level) const
    {
        double dz = (brick.xyz_max[2] - brick.xyz_min[2])/(1<<level);
        int k = (int) (z/dz)*(max_i>>level);
        k+= max_i>>(level+1);
        return k;
    }

    int morton_from_indices(int i, int j, int k) const;

    void build_subtree(const std::vector<Atom> &atoms, int level, int i, int j, int k);
    //double dist_from_surface(double x, double y, double z);


private:

};

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
  bool is_partitioned;


  typedef std::vector<std::vector<int> > atom_mapping_t;
  atom_mapping_t cell2atom;

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
  void partition_atoms();
  void construct_SES_by_reinitialization(p4est_t* &p4est, p4est_nodes_t *&nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void construct_SES_by_reinitialization_fast(p4est_t* &p4est, p4est_nodes_t *&nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void construct_SES_by_advection(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  void remove_internal_cavities(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t* &ghost, my_p4est_brick_t& brick, Vec& phi);
  double operator()(double x, double y, double z) const;
  void reduce_to_single_atom();
  void atoms_per_node(p4est_t* &p4est, p4est_nodes_t* &nodes, p4est_ghost_t *&ghost, my_p4est_brick_t &brick, Vec &atom_count);
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
