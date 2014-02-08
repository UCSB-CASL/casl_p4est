#ifndef INTERPOLATING_FUNCTION_H
#define INTERPOLATING_FUNCTION_H

#include <vector>
#include <map>
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_node_neighbors.h>
#endif

enum interpolation_method{
  linear,
  quadratic,
  quadratic_non_oscillatory
};

#ifdef P4_TO_P8
class InterpolatingFunctionNodeBase: public CF_3
#else
class InterpolatingFunctionNodeBase: public CF_2
#endif
{
  interpolation_method method_;

  p4est_t *p4est_;
  p4est_nodes_t *nodes_;
  p4est_ghost_t *ghost_;
  my_p4est_brick_t *myb_;
  const my_p4est_node_neighbors_t *neighbors_;

  double xyz_min[3], xyz_max[3];

  PetscErrorCode ierr;
  Vec Fxx_, Fyy_;
#ifdef P4_TO_P8
  Vec Fzz_;
#endif
  bool local_derivatives;

  Vec input_vec_;

  struct point_buffer{
    std::vector<double> xyz;
    std::vector<p4est_quadrant_t> quad;
    std::vector<p4est_locidx_t> node_locidx;

    size_t size() { return node_locidx.size(); }
    void clear()
    {
      xyz.clear();
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

  std::vector<MPI_Request> remote_send_req;

  enum {
    remote_point_tag,
    remote_data_tag,
    remote_notify_tag
  };

  // methods
  void process_remote_data(std::vector<double>& xyz_recv, std::vector<double>& f_send);
  void compute_second_derivatives();

  // rule of three -- disable copy ctr and assignment if not useful
  InterpolatingFunctionNodeBase(const InterpolatingFunctionNodeBase& other);
  InterpolatingFunctionNodeBase& operator=(const InterpolatingFunctionNodeBase& other);

public:
  InterpolatingFunctionNodeBase(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_brick_t *myb);
  InterpolatingFunctionNodeBase(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_brick_t *myb, const my_p4est_node_neighbors_t *neighbors);
  ~InterpolatingFunctionNodeBase();

  void add_point_to_buffer(p4est_locidx_t node_locidx, const double *xyz);
  void set_input_parameters(Vec input_vec, interpolation_method method, Vec Fxx = NULL, Vec Fyy = NULL
#ifdef P4_TO_P8
    , Vec Fzz = NULL
#endif
    );

  // interpolation methods
  void interpolate(Vec output_vec);
  void interpolate(double *output_vec);
  void save_comm_topology(const char* partition_name, const char *topology_name);
  double operator()(double x, double y
#ifdef P4_TO_P8
    , double z
#endif
    ) const;
};

#endif // INTERPOLATING_FUNCTION_H
