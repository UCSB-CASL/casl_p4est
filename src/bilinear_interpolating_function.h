#ifndef BILINEAR_INTERPOLATING_FUNCTION_H
#define BILINEAR_INTERPOLATING_FUNCTION_H

#include <vector>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>

class BilinearInterpolatingFunction: public CF_2
{
  p4est_t *p4est_;
  p4est_nodes_t *nodes_;
  p4est_ghost_t *ghost_;
  my_p4est_brick_t *myb_;
  std::vector<int> p4est2petsc;
  Vec input_vec_;

  struct{
    std::vector<double> xy;
    std::vector<p4est_quadrant_t> quad;
    std::vector<p4est_locidx_t> node_locidx;

    size_t size() { return node_locidx.size(); }
  } local_point_buffer;

  struct ghost_point_info{
    double xy[2];
    p4est_topidx_t tree_idx;
    p4est_locidx_t quad_locidx;
  };

  typedef std::map<int , std::vector<ghost_point_info> > ghost_transfer_map;
  ghost_transfer_map ghost_send_buffer, ghost_recv_buffer;

  typedef std::map<int, std::vector<double> > remote_transfer_map;
  remote_transfer_map remote_send_buffer, remote_recv_buffer;

  typedef std::map<int, std::vector<p4est_locidx_t> > nonlocal_node_map;
  nonlocal_node_map ghost_node_index, remote_node_index;

  std::vector<int> ghost_receivers, ghost_senders, remote_receivers, remote_senders;
  bool is_buffer_prepared;

  std::vector<MPI_Request> ghost_send_req, ghost_recv_req, remote_send_req, remote_recv_req;

  enum {
    ghost_point_tag,
    ghost_data_tag,
    remote_point_tag,
    remote_data_tag
  };

  // methods
  void send_point_buffers_begin();
  void recv_point_buffers_begin();
  void clear_buffer();

public:
  BilinearInterpolatingFunction(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_brick_t *myb);

  void add_point_to_buffer(p4est_locidx_t node_locidx, double x, double y);
  void set_input_vector(Vec& input_vec);
  void update_grid(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost);

  void interpolate(Vec& output_vec);
  double operator()(double x, double y) const;
};

#endif // BILINEAR_INTERPOLATING_FUNCTION_H
