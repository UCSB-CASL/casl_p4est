#ifndef MY_P4EST_INTERPOLATING_FUNCTION_CELL_BASE_H
#define MY_P4EST_INTERPOLATING_FUNCTION_CELL_BASE_H

#include <vector>
#include <map>
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_cell_neighbors.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_cell_neighbors.h>
#endif

enum interpolation_method{
  linear,
  IDW,
  LSQR
};

#ifdef P4_TO_P8
class InterpolatingFunctionCellBase: public CF_3
#else
class InterpolatingFunctionCellBase: public CF_2
#endif
{
  p4est_t *p4est_;
  p4est_ghost_t *ghost_;
  my_p4est_brick_t *myb_;
  const my_p4est_cell_neighbors_t *cnnn_;
  interpolation_method method_;
  
  typedef my_p4est_cell_neighbors_t::quad_info_t quad_info_t;

  double xyz_min[3], xyz_max[3];

  PetscErrorCode ierr;
  Vec input_vec_;

  struct local_point_buffer_t{
    std::vector<double> xyz;
    std::vector<p4est_quadrant_t> quad;
    std::vector<p4est_locidx_t> output_idx;

    size_t size() { return output_idx.size(); }
    void clear()
    {
      xyz.clear();
      quad.clear();
      output_idx.clear();
    }
  };

  struct ghost_point_buffer_t{
    std::vector<double> xyz;
    std::vector<p4est_quadrant_t> quad;
    std::vector<p4est_locidx_t> output_idx;
    std::vector<int> rank;

    size_t size() { return output_idx.size(); }
    void clear()
    {
      xyz.clear();
      quad.clear();
      output_idx.clear();
    }
  };


  local_point_buffer_t local_point_buffer;
  ghost_point_buffer_t ghost_point_buffer;

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
  double cell_based_linear_interpolation(const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;
  double cell_based_IDW_interpolation   (const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;
  double cell_based_LSQR_interpolation  (const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;

  // rule of three -- disable copy ctr and assignment if not useful
  InterpolatingFunctionCellBase(const InterpolatingFunctionCellBase& other);
  InterpolatingFunctionCellBase& operator=(const InterpolatingFunctionCellBase& other);

public:
  InterpolatingFunctionCellBase(const my_p4est_cell_neighbors_t *cnnn);
  ~InterpolatingFunctionCellBase();

  void add_point_to_buffer(p4est_locidx_t node_locidx, const double *xyz);
  void set_input_parameters(Vec input_vec, interpolation_method method);

  // interpolation methods
  void interpolate(Vec output_vec);
  void interpolate(double *output_vec);
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif

};

#endif // MY_P4EST_INTERPOLATING_FUNCTION_CELL_BASE_H
