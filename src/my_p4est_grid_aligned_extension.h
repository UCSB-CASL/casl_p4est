#ifndef MY_P4EST_GRID_ALIGNED_EXTENSION_H
#define MY_P4EST_GRID_ALIGNED_EXTENSION_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/casl_math.h>

class my_p4est_grid_aligned_extension_t {
  PetscErrorCode  ierr;

  my_p4est_brick_t *myb_;
  p4est_t *p4est_;
  p4est_nodes_t *nodes_;
  p4est_ghost_t *ghost_;
  my_p4est_node_neighbors_t *ngbd_;
  my_p4est_interpolation_nodes_t interp_;

  Vec phi_;
  unsigned int order_;
  double band_extend_;
  double band_check_;
  unsigned int max_iters_;
  interpolation_method interp_method_;

  Vec well_defined_;
  unsigned int num_points_to_extend_;
  vector<p4est_locidx_t> points_to_extend_;
  vector<double> dist_;


public:
  my_p4est_grid_aligned_extension_t(my_p4est_node_neighbors_t *ngbd)
    : myb_(ngbd->myb), p4est_(ngbd->p4est), nodes_(ngbd->nodes), ghost_(ngbd->ghost), ngbd_(ngbd),
      interp_(ngbd), order_(2), band_extend_(10), band_check_(5), max_iters_(20),
      interp_method_(linear)
  {}

  ~my_p4est_grid_aligned_extension_t()
  {
    if (well_defined_ != NULL) {
      ierr = VecDestroy(well_defined_); CHKERRXX(ierr);
    }
  }

  void initialize(Vec phi, unsigned int order, unsigned int max_iters, double band_extend_, double band_check, Vec normal[], Vec mask);
  void extend(unsigned int num_fields, Vec fields[]);
};

#endif // MY_P4EST_GRID_ALIGNED_EXTENSION_H
