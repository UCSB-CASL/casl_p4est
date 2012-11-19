#ifndef PARQUADTREE_H
#define PARQUADTREE_H

// CASL
#include <lib/array/ArrayV.h>
#include <lib/amr/QuadTree.h>
#include <lib/io/xdmfWriter.h>

// p4est
#include <p4est.h>

class parQuadTree
{
  const p4est_t* p4est;
  ArrayV<QuadTree> tr_array;
  p4est_topidx_t num_of_trees;

public:
  explicit parQuadTree(const p4est_t* p4est_);

  void copyFromP4est();
  void copyToP4est();

  void print();

};

#endif // PARQUADTREE_H
