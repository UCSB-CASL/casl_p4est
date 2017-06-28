#ifndef ATOMTREE_H
#define ATOMTREE_H

#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include <unordered_map>
#include <iostream>


#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>

using namespace std;

//---------------------------------------------------------------------
//
//   Miles Detrixhe
//   2016 Fall, UCSB
//   Raphael Egan
//   2017 Spring, Summer UCSB
//   (comments by Raphael Egan, 2017 Spring & Summer UCSB)
//
//---------------------------------------------------------------------

class AtomTree;

/*!
 * \brief The Atom struct contains
 * the geometrical coordinates x, y, z of the atom center;
 * its electric charge q;
 * and the atom van der Waals radius r.
 */
struct Atom {
  double x, y, z, q, r;
};

/*!
 * \brief The ATCell class is the elementary building brick of the atom tree: any cell of the
 * atom tree is an atom cell (ATCell). It represents a cartesian region in the computational
 * domain and contains all the information required to calculate the distance from the SAS to
 * any point in that region.
 */
class ATCell{
private:
  // the list of local atoms that are within reachable distance to the ATCell;
  // (well-defined and non-empty if and only if the cell is a leaf of the atom tree)
  std::vector<Atom> local_atoms;
  // list of node indices (global indices, as stored in the node_table) of the cell nodes
  long long int node_indices[2][2][2];
public:
  // the logical coordinates i, j, k of its center (integers between 1 and 2^(abs_max_level+1) -1):
  const long int i,j,k;
  // the physical coordinates xc, yc, zc of its center:
  const double xc, yc, zc;
  // physical dimensions dx, dy, dz (and the corresponding diagonal length):
  const double dx, dy, dz, diag;
  // the level in the atom tree:
  const int level;
  // boolean flag (true if the cell is a leaf of the atom tree):
  bool is_leaf;


  /*!
   * \brief ATCell constructor
   * \param my_tree: the AtomTree to which the cell belongs, based on which the
   * physical coordinates will be calculated
   * \param i_ : logical x-coordinate of the center of the ATCell (integer between 1 and 2^abs_max_level - 1)
   * \param j_ : logical y-coordinate of the center of the ATCell (integer between 1 and 2^abs_max_level - 1)
   * \param k_ : logical z-coordinate of the center of the ATCell (integer between 1 and 2^abs_max_level - 1)
   * \param level_ : level of the ATCell in the AtomTree
   */
  ATCell(const AtomTree& my_tree, long int i_, long int j_, long int k_, int level_);

  /*!
   * \brief set_node_index: set the global node index (as stored in the node_table of the AtomTree)
   * of node [ioff][joff][koff] of the ATCell
   * \param ioff: 0/1 integer
   * \param joff: 0/1 integer
   * \param koff: 0/1 integer
   * \param node_index_in_node_table: self-explanatory
   */
  void set_node_index(const int ioff, const int joff, const int koff, const long long int node_index_in_node_table);
  void set_node_index(const bool ioff, const bool joff, const bool koff, const long long int node_index_in_node_table);

  /*!
   * \brief get_node_index: returns the global node index (as stored in the node_table of the AtomTree)
   * of node [ioff][joff][koff] of the ATCell
   * \param ioff: 0/1 integer
   * \param joff: 0/1 integer
   * \param koff: 0/1 integer
   * \return the global node index of the [ioff][joff][koff] node in ATCell
   */
  long long int get_node_index(const int ioff, const int joff, const int koff) const;
  long long int get_node_index(const bool ioff, const bool joff, const bool koff) const;

  /*!
   * \brief get_node_index_vtk_labeled: returns the global node index (as stored
   * in node_table) of the ATCell's node of given vtk label:
   * label:   0,   1,   2,   3,   4,   5,   6,   7
   * node:  mmm, pmm, ppm, mpm, mmp, pmp, ppp, mpp
   * \param vtk_label: local node's VTK integer label (integer0 to 7)
   * \return the global node index (as stored in node_table)
   */
  long long int get_node_index_vtk_labeled(const int vtk_label) const;

  /*!
   * \brief add_atom: adds an atom to the cell's list of local atoms
   * \param a: atom to be added
   */
  void add_atom(const Atom &a);

  /*!
   * \brief clear_local_atoms: self-explanatory
   */
  void clear_local_atoms();

  /*!
   * \brief get_local_atoms: self-explanatory
   * \return a copy of the cell's list of atoms
   */
  std::vector<Atom> get_local_atoms() const;

  /*!
   * \brief get_number_of_atoms: self-explanatory
   * \return the number of atoms within reachable distance from the ATCell
   */
  inline int get_number_of_atoms() const
  {
    return (int) local_atoms.size();
  }

  /*!
   * \brief dist_from_vdW_surface_to: calculates the signed distance from the van der Waals
   * surface to a point (x, y, z) in the atom cell.
   * \param x: x-coordinate of the point of interest
   * \param y: y-coordinate of the point of interest
   * \param z: z-coordinate of the point of interest
   * \return the signed distance to van der Waals surface
   */
  double dist_from_vdW_surface_to(const double& x, const double& y, const double& z) const;

  /*!
   * \brief contains_point: checks if a given point lies within the ATCell
   * \param x: x-coordinate
   * \param y: y-coordinate
   * \param z: z-coordinate
   * \return a boolean flag that is true iff the point)x, y, z) lies within the ATCell
   */
  inline bool contains_point(const double &x, const double &y, const double &z) const
  {
    return (fabs(x - xc) <= .5*dx && fabs(y - yc) <= .5*dy && fabs(z - zc) <= .5*dz);
  }

  /*!
   * \brief contains_atom: checks if the center of a given atom lies within the ATCell
   * \param a: the given atom
   * \return a boolean flag that is true iff the atom center lies within the ATCell
   */
  inline bool contains_atom(const Atom &a) const
  {
    return contains_point(a.x, a.y, a.z);
  }

  /*!
   * \brief is_empty: return a boolean flag that is true if the cell has no atoms in it
   * \return boolean flag (true if empty)
   */
  inline bool is_empty() const
  {
    return (local_atoms.size() == 0);
  }

  /*!
   * \brief shortest_distance_from_vdW_surf_of: calculates the shortest distance
   * for all points in ATCell to a given atom.
   * \param a: the considered atom;
   * \return the maximum, for all points in ATCell, of the signed distance
   * (positive inside the molecule, negative outside) from that point to the
   * vdW surface of atom a;
   */
  double shortest_distance_from_vdW_surf_of(const Atom &a) const;

};

/*!
 * \brief The ATNode struct represents a vertex of the atom tree (and of the atom cells).
 */
struct ATNode{
  // logical coordinates i, j, k (integers between 0 and 2^abs_max_level)
  const int i,j,k;

  /*!
   * \brief ATNode constructor
   * \param i_: logical x-coordinate (integer between 0 and 2^abs_max_level)
   * \param j_: logical y-coordinate (integer between 0 and 2^abs_max_level)
   * \param k_: logical z-coordinate (integer between 0 and 2^abs_max_level)
   */
  ATNode(int i_, int j_, int k_): i(i_), j(j_), k(k_){}
};

/*!
 * \brief The AtomTree class
 */
class AtomTree{
  friend class ATCell;
private:
  // the probe radius and the finest cell resolution are positive parameters
  class GuardedPositiveParameter
  {
  private:
    double value;
    const bool must_be_strictly_positive;
  public:
    GuardedPositiveParameter(const bool stric_pos = true, double const value_ = -1.): must_be_strictly_positive(stric_pos)
    {
      value = value_;
    }

    inline bool is_assigned() const
    {
      return (must_be_strictly_positive)? (value > 0): (value >=0);
    }

    inline double get_value() const
    {
      if (is_assigned())
        return value;
      else
        throw std::runtime_error("[CASL_ERROR] AtomTree::GuardedPositiveParameter::get_value(), the guarded positive value hasn't been assigned...");
    }

    inline void set_value(double const new_value)
    {
      if (is_assigned())
      {
        if (fabs(value - new_value) > EPS)
          throw std::runtime_error("[CASL_ERROR] AtomTree::GuardedPositiveParameter::set_value(double const ), the value is already assigned, it can't be changed (except if the tree is cleared)...");
      }
      else
        value = new_value;
    }

    inline void reset()
    {
      value = -1.;
    }
  };
  // the p4est brick that respresents the entire domain
  const my_p4est_brick_t &brick;
  // required grid reolution around the interface (minimal value = diag of smallest size of the COMPUTATIONAL grid)
  GuardedPositiveParameter interface_resolution;
  // probe radius
  GuardedPositiveParameter probe_radius;
  // maximum level of the tree (self-determined by atoms_per_cell or the finest_level)
  int max_level;
  // finest level (function of the required interface resolution, no greater than abs_max_level)
  int finest_level;
  // total list of atoms, as read from the .pqr files
  std::vector<Atom> total_atoms;
  // unordered_maps: keys are Morton indices, values are cell or node indices
  std::unordered_map<long long int, long long int> cell_table;
  std::unordered_map<long long int, long long int> node_table;
  // List of atom cells (i.e. cells of the atom tree) and nodes.
  // In the following vectors, the index of the point/of the cell whose center is at
  // logical coordinates (i, j, k) is given by node_table/cell_table.at(morton_from_indices(i,j,k));
  std::vector<ATCell> cells;
  std::vector<ATNode> nodes;
public:

  // Since the cells and nodes in the 3D computational space are mapped to 1D lists, one needs a biunivocal
  // mapping between integer triplets (i, j, k) representing the logical coordinates of the object that is
  // considered (ATCell or ATNode) and the integer index in that list. Z-ordering (Morton code) offers a way
  // to do so while preserving locality of the data. The Morton code idea is based on a bit-interleaving
  // construction built on the binary representation for i, j and k.
  // If one uses long integer representations (i.e. at least 32 bits signed integers) for i, j and k and
  // long long int (at least 64 bits) for the corresponding Morton code, 63 bits at most can be used for the
  // morton code and thus |_ 63/3 _| = 21 bits for i, j and k (we will restrict oursleves to 20 though, it's
  // easier to code the Morton conversion), which limits the maximum level of the atom tree.
  // Indeed, for a given abs_max_level, one needs logical coordinates running from 0 to 2**{abs_max_level+1}
  // (since we also need to index the cell centers). Therefore, (abs_max_level + 2) bits are needed for the
  // binary representation of logical coordinates i, j, k in that range. Since we limit ourselves to 20 bits
  // for those, the absolute maximum level must be 18.
  const static int abs_max_level = 18;
  const static long int max_i = 1 << (abs_max_level+1);
  const static int atoms_per_cell = 1;  // number of atoms per (non-empty) atom cell, in the limit of the maximum refinement level


  /*!
   * \brief AtomTree constructor
   * \param brick_: the p4est brick representing the computational domain with which the atom tree is associated
   */
  AtomTree(my_p4est_brick_t &brick_): brick(brick_), interface_resolution(true), probe_radius(false)
  {
    max_level = 0;
    finest_level = 0;
  }

  /*!
   * \brief build_tree: initiates the recursive procedure for building the atom tree.
   * \param atoms: the total list of atoms that are present in the computational domain.
   */
  void build_tree(const std::vector<Atom> &atoms);

  /*!
   * \brief is_built: indicates if the atom tree is built or not
   * \return a boolean flag that is true is the AtomTree is built
   */
  bool is_built() const;

  /*!
   * \brief get_max_level: self-explanatory
   * \return the maximum level of the atom tree
   */
  int inline get_max_level() const { return max_level;}

  /*!
   * \brief clear_tree: self-explanatory
   */
  void clear_tree();

  /*!
     * \brief dist_from_SAS: calculates the signed distance (negative outside the molecule)
     * from point (x, y, z) to the solvent-accessible surface (SAS)
     * \param x: x-coordinate
     * \param y: y-coordinate
     * \param z: z-coordinate
     * \return the distance to the SAS
     */
  double dist_from_SAS(const double& x, const double& y, const double& z) const;

  /*!
   * \brief num_atoms_queried gives the reduced number of atoms involved in the calculation
   * of the distance to the SAS from point (x, y, z).
   * \param x: x-coordinate;
   * \param y: y-coordinate;
   * \param z: z-coordinate;
   * \return the number of atoms in the atom list of the atom cell (ATCell)
   */
  int num_atoms_queried(const double& x, const double& y, const double& z) const;

  /*!
   * \brief number_of_nodes: self_explanatory
   * \return
   */
  inline long long int number_of_nodes() const{return nodes.size();}

  /*!
   * \brief number_of_cells: self_explanatory
   * \return
   */
  inline long long int number_of_cells() const{return cells.size();}

  /*!
   * \brief number_of_leaves: self_explanatory
   * \return
   */
  inline int number_of_leaves() const{
    long long int count=0;
    for (size_t n=0; n<cells.size(); n++)
      if(cells[n].is_leaf)
        ++count;
    return count;
  }

  /*!
   * \brief set_probe_radius: sets the probe radius (atomic/molecular radius of the solvent).
   * The probe radius can be set only once and for all, as it is a critical parameter in the
   * application. It can't be modified afterwards (except if the atom tree is cleared).
   * \param rp_: the value of the probe radius
   */
  void set_probe_radius(const double rp_);

  /*!
   * \brief reset_probe_radius: resets the probe radius (atomic/molecular radius of the solvent).
   * This method deletes the tree if needed and resets the probe radius if its redefinition poses
   * problems.
   * \param rp_: the value of the probe radius
   */
  void reset_probe_radius(const double rp_);

  /*!
   * \brief set_interface_resolution: sets the interface resolution, i.e. the minimal width
   * of the band around the SAS that must be captured exactly. In practice, set this
   * parameter to be slightly greater than the diagonal of the finest cell in the computational
   * grid.
   * This parameter can be set only once and for all, as it is a critical parameter in the
   * application. It can't be modified afterwards (except if the atom tree is cleared).
   * \param resolution_: the value of the interface resolution
   */
  void set_interface_resolution(const double resolution_);

  /*!
   * \brief reset_interface_resolution: resets the interface resolution (minimal width
   * of the band around the SAS that must be captured exactly).
   * This method deletes the tree if needed and resets the interface resolution if its
   * redefinition poses problems.
   * \param resolution_: the value of the interface resolution
   */
  void reset_interface_resolution(const double resolution_);

  /*!
   * \brief get_number_of_atoms_in_cell: self-explanatory
   * \param cell_index: self-explanatory (global index, i.e. value of cell_table.at(morton_from_indices(i, j, k)))
   * \return the number of atoms to be considered for the calculation of the distance from the SAS for any
   * point in the considered ATCell
   */
  int get_number_of_atoms_in_cell(const int& cell_index) const
  {
    return cells[cell_index].get_number_of_atoms();
  }

  inline double get_interface_resolution() const{ return interface_resolution.get_value();}

  /*!
   * \brief print_atom_count_per_cell: exports the AtomTree as a vtk file, and the number
   * of atoms per ATCell.
   * \param file_name: self-explanatory
   */
  void print_atom_count_per_cell( std::string file_name) const;

private:

  inline long int cell_i_fr_x(double x, int level) const
  {
    double dx = (brick.xyz_max[0] - brick.xyz_min[0])/(1<<level);
    long int i = (long int)((x-brick.xyz_min[0])/dx)*(max_i>>level);
    i+= max_i>>(level+1);
    i = (i>max_i) ? i-(max_i>>level) : i; //This accounts for points that land exactly on the maximum edge (reflexion symmetry)
    return i;
  }

  inline long int cell_j_fr_y(double y, int level) const
  {
    double dy = (brick.xyz_max[1] - brick.xyz_min[1])/(1<<level);
    long int j = (long int)((y-brick.xyz_min[1])/dy)*(max_i>>level);
    j+= max_i>>(level+1);
    j = (j>max_i) ? j-(max_i>>level) : j;
    return j;
  }

  inline long int cell_k_fr_z(double z, int level) const
  {
    double dz = (brick.xyz_max[2] - brick.xyz_min[2])/(1<<level);
    long int k = (long int)((z-brick.xyz_min[2])/dz)*(max_i>>level);
    k+= max_i>>(level+1);
    k = (k>max_i) ? k-(max_i>>level) : k;
    return k;
  }

  inline double x_fr_i(long int i) const
  {
    return brick.xyz_min[0] + i*(brick.xyz_max[0] - brick.xyz_min[0])/max_i;
  }

  inline double y_fr_j(long int j) const
  {
    return brick.xyz_min[1] + j*(brick.xyz_max[1] - brick.xyz_min[1])/max_i;
  }

  inline double z_fr_k(long int k) const
  {
    return brick.xyz_min[2] + k*(brick.xyz_max[2] - brick.xyz_min[2])/max_i;
  }

  /*!
   * \brief find_smallest_cell_containing_point: self-explanatory
   * \param x: x-coordinate of the point;
   * \param y: y-coordinate of the point;
   * \param z: z-coordinate of the point;
   * \return the cell index of the leaf atom cell ATCell that contains the point (x, y, z)
   */
  long long int find_smallest_cell_containing_point(const double& x, const double& y, const double& z) const;

  /*!
     * \brief morton_from_indices: calculates the Morton index value associated
     * with logical (integer) coordinates. The binary representation of the
     * Morton value is constructed by bit-interleaving of the binary representations
     * of the integer triplet (i, j, k).
     * i, j and k are considered 20-bits integers, i.e. \leq than 2**20-1. (in practice,
     * they are supposed to be smaller than or equal to 2**19, in this entire class)
     * \param i: logical x-coordinate
     * \param j: logical y-coordinate
     * \param k: logical z-coordinate
     * \return returns the Morton value calculated from i, j and k.
     */
  long long int morton_from_indices(long int index1, long int index2, long int index3) const;

  /*!
     * \brief build_subtree: elementary recursive function to build the atom tree
     * \param atoms: reduced list of atoms that are still considered at the parent cell level
     * (the full list of atoms in the domain if root cell level)
     * \param level: current level of the construction procedure, i.e. the actual level of the
     * cell that is the root for the corresponding subtree (i.e. the recursive call index)
     * \param i: logical x-index of that cell (root of the subtree) center;
     * \param j: logical y-index of that cell (root of the subtree) center;
     * \param k: logical z-index of that cell (root of the subtree) center;
     */
  void build_subtree(const std::vector<Atom> &atoms, int level, long int i, long int j, long int k);

  /*!
     * \brief add_cell: adds a new atom cell to the atom tree. i, j, k are the
     * logical coordinates of the cell center, the cell index is added to the cell
     * table (key == Morton index), the corresponding nodes are added to the node
     * table as well and the list of atoms in the constructed cell is created
     * \param atoms: extended list of atoms, including those in the created cell among others
     * \param level: level of the cell
     * \param i: logical x-coordinate of the cell-center
     * \param j: logical y-coordinate of the cell-center
     * \param k: logical z-coordinate of the cell-center
     * \return the function returns the cell index (also the value associated with
     * the Morton index of the cell center as a key in the cell table)
     */
  long long int add_cell(const std::vector<Atom> &atoms, int level, long int i, long int j, long int k);

  /*!
     * \brief set_cell_nodes: adds the cell nodes to the node table (the key entry is the
     * Morton index of the node) if needed, and sets the node indices of the cell.
     * \param cell: atom cell whose nodes are to be set
     */
  void set_cell_nodes(ATCell &cell);

  /*!
     * \brief set_atoms_belonging_to_cell: sets the list of atoms that are located within reachable
     * distance of the considered atom cell, i.e. for which there exists a point in the ATCell such that
     * its distance to the SAS is of the order of the required grid resolution distance.
     * If there is no such atoms in the considered cell (i.e. if any point in cell is far away from the
     * molecule), the list of atoms is set to any atoms in the molecule (the computed distance will be
     * negative anyways and it will be OK after reinitialization)
     * \param cell: the atom cell that is currently considered
     * \param atoms: extended list of atoms as considered at the parent level.
     */
  void set_atoms_belonging_to_cell(ATCell &cell, const std::vector<Atom> &atoms);

  /*!
   * \brief print_VTK_format: regular tree exportation function, vtk format for visualization purposes
   */
  void print_VTK_format( std::string file_name, double time=DBL_MIN) const;
  void print_VTK_format(std::vector<int> &F, std::string data_name, std::string file_name) const;
  void print_VTK_format(std::vector<double> &F, std::string data_name, std::string file_name) const;
};
#endif // ATOMTREE_H
