#ifndef INTERPOLATING_FUNCTION_H
#define INTERPOLATING_FUNCTION_H

#include <vector>
#include <map>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_node_neighbors.h>

enum interpolation_method{
  linear,
  quadratic,
  quadratic_non_oscillatory
};

class InterpolatingFunction: public CF_2
{
  interpolation_method method_;

  p4est_t *p4est_;
  p4est_nodes_t *nodes_;
  p4est_ghost_t *ghost_;
  my_p4est_brick_t *myb_;
  const my_p4est_node_neighbors_t *qnnn_;

  double xmin, xmax, ymin, ymax;

  PetscErrorCode ierr;
  Vec Fxx_, Fyy_;
  bool local_derivatives;

  std::vector<int> p4est2petsc;
  Vec input_vec_;

  struct point_buffer{
    std::vector<double> xy;
    std::vector<p4est_quadrant_t> quad;
    std::vector<p4est_locidx_t> node_locidx;

    double xmin, xmax;
    double ymin, ymax;

    size_t size() { return node_locidx.size(); }
    void clear()
    {
      xy.clear();
      quad.clear();
      node_locidx.clear();
    }
  };

  point_buffer local_point_buffer, ghost_point_buffer;

  typedef std::map<int, std::vector<double> > remote_transfer_map;
  remote_transfer_map remote_send_buffer, remote_recv_buffer;

  typedef std::map<int, std::vector<p4est_locidx_t> > nonlocal_node_map;
  nonlocal_node_map remote_node_index;

  std::vector<int> remote_receivers, remote_senders;
  bool is_buffer_prepared;

  std::vector<MPI_Request> remote_send_req, remote_recv_req;

  enum {
    remote_point_tag,
    remote_data_tag
  };

  // methods
  void send_point_buffers_begin();
  void recv_point_buffers_begin();
  void compute_second_derivatives();

  // rule of three -- disable copy ctr and assignment if not useful
  InterpolatingFunction(const InterpolatingFunction& other);
  InterpolatingFunction& operator=(const InterpolatingFunction& other);

public:
  InterpolatingFunction(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_brick_t *myb);
  InterpolatingFunction(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_brick_t *myb, const my_p4est_node_neighbors_t *qnnn);
  ~InterpolatingFunction();

  void add_point_to_buffer(p4est_locidx_t node_locidx, double x, double y);
  void set_input_parameters(Vec input_vec, interpolation_method method, Vec Fxx = NULL, Vec Fyy = NULL);

  // interpolation methods
  void interpolate(Vec output_vec);
  void interpolate(double *output_vec);
  double operator()(double x, double y) const;
};

#endif // INTERPOLATING_FUNCTION_H
