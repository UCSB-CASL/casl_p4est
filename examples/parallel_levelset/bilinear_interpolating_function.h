#ifndef BILINEAR_INTERPOLATING_FUNCTION_H
#define BILINEAR_INTERPOLATING_FUNCTION_H

#include <vector>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>

namespace parallel{
class BilinearInterpolatingFunction: public CF_2
{
  p4est_t *p4est_;
  p4est_nodes_t *nodes_;
  p4est_ghost_t *ghost_;
  my_p4est_brick_t *myb_;

  struct local_point_buffer{
    std::vector<double> xy;
    std::vector<p4est_quadrant_t*> quad;
  };

  struct ghost_point_info{
    double xy[2];
    p4est_topidx_t tree_idx;
    p4est_locidx_t quad_locidx;
  };
  typedef std::map<int , std::vector<ghost_point_info> > ghost_transfer_map;
  ghost_transfer_map ghost_point_send_buffer, ghost_point_recv_buffer;

  typedef std::map<int, std::vector<double> > simple_transfer_map;
  simple_transfer_map remote_xy_send, remote_xy_recv, F_send, F_recv;

  std::vector<int> ghost_recievers, ghost_senders, remote_recievers, remote_senders;
  bool is_buffer_prepared;

  enum {
    size_tag,
    ghost_point_tag,
    ghost_data_tag,
    remote_point_tag,
    remote_data_tag
  };

  // methods
  double linear_interpolation(const double xy[]) const;
  void prepare_buffer();

public:
  BilinearInterpolatingFunction(p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, my_p4est_brick_t *myb);

  void add_point_to_buffer(double x, double y);
  void clear_buffer();

  void interpolate(Vec& F, const std::vector<p4est_locidx_t> node_locidx);
  double operator()(double x, double y) const;
};
} // namepace parallel

#endif // BILINEAR_INTERPOLATING_FUNCTION_H
