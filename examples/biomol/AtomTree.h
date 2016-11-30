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

//#include "bio_molecule.h"

using namespace std;

/*!
 * \file AtomTree.h
 * \brief AtomTree data structure based on a hashmap. Stores all the atoms of a biomolecule in a tree structure.
 */
//---------------------------------------------------------------------
//
//
//   Miles Detrixhe
//   2016 Fall, UCSB
//
//---------------------------------------------------------------------


struct Atom {
  double x, y, z, q, r;
};

struct AtomOrderX {
    inline bool operator() (const Atom& atom1, const Atom& atom2) {
            return (atom1.x < atom2.x);
        }

};


struct ATCell{
    int i,j,k;
    std::vector<Atom> atoms;
    bool refine;
    bool is_leaf;
    int level;
    int nodes[8];
    //int children[8];

};

struct ATNode{
    int i,j,k;

    ATNode(int i_, int j_, int k_){
        i=i_; j=j_; k=k_;}
};


class AtomTree{

    public:

    std::unordered_map<int, int> cell_table;
    std::unordered_map<int, int> node_table;
    std::vector<Atom> total_atoms;
    std::vector<ATCell> cells;
    std::vector<ATNode> nodes;

    my_p4est_brick_t &brick;

    const static int max_i = 512;
    const static int abs_max_level = 8;
    const static int atoms_per_cell = 1;

    int max_level;
    double threshold;
    double min_dx;
    double rp;
    double max_ar;
    double rmax;


    AtomTree(my_p4est_brick_t &brick_,double rp_): brick(brick_), rp(rp_){}
    void build_tree(const std::vector<Atom> &atoms, const my_p4est_brick_t &brick);
    void clear_tree();
    double dist_from_surface(double x, double y, double z) const;
    double num_atoms_queried(double x, double y, double z) const;
    int find_smallest_cell_containing_point(double x, double y, double z) const;
    inline int number_of_nodes() const{return nodes.size();}
    inline int number_of_cells() const{return cells.size();}
    inline int number_of_leaves() const{
        int count=0;
        for (int n=0; n<cells.size(); n++)
            if(cells[n].is_leaf)
                ++count;
        return count;
     }
    void set_probe_radius(double rp_);
    void set_max_atom_radius(double rmax_);




private:

    inline int cell_i_fr_x(double x, int level) const
    {
        double dx = (brick.xyz_max[0] - brick.xyz_min[0])/(1<<level);        
        int i = (int)(x/dx)*(max_i>>level);
        i+= max_i>>(level+1);
        i = (i>max_i) ? i-(max_i>>level) : i; //This accounts for points that land exactly on the maximum edge
        return i;
    }

    inline int cell_j_fr_y(double y, int level) const
    {
        double dy = (brick.xyz_max[1] - brick.xyz_min[1])/(1<<level);
        int j = (int)( y/dy)*(max_i>>level);
        j+= max_i>>(level+1);
        j = (j>max_i) ? j-(max_i>>level) : j;
        return j;
    }

    inline int cell_k_fr_z(double z, int level) const
    {
        double dz = (brick.xyz_max[2] - brick.xyz_min[2])/(1<<level);
        int k = (int) (z/dz)*(max_i>>level);
        k+= max_i>>(level+1);
        k = (k>max_i) ? k-(max_i>>level) : k;
        return k;
    }

    inline double x_fr_i(int i) const
    {
        return i*(brick.xyz_max[0] - brick.xyz_min[0])/max_i;
    }

    inline double y_fr_j(int j) const
    {
        return j*(brick.xyz_max[1] - brick.xyz_min[1])/max_i;
    }

    inline double z_fr_k(int k) const
    {
        return k*(brick.xyz_max[2] - brick.xyz_min[2])/max_i;
    }





    int morton_from_indices(int i, int j, int k) const;

    void build_subtree(const std::vector<Atom> &atoms, int level, int i, int j, int k);
    void build_subtree(int level, int i, int j, int k);
    int  add_cell(const std::vector<Atom> &atoms, int level, int i, int j, int k);
    void set_cell_nodes(ATCell &cell);
    void set_atoms_belonging_to_cell(ATCell &cell, const std::vector<Atom> &atoms);
    //double dist_from_surface(double x, double y, double z);

    void print_atom_count_per_cell( std::string file_name) const;

    void print_VTK_format( std::string file_name, double time=DBL_MIN) const;

    void print_VTK_format(std::vector<int> &F, std::string data_name, std::string file_name) const;
    void print_VTK_format(std::vector<double> &F, std::string data_name, std::string file_name) const;


};



#endif // ATOMTREE_H
