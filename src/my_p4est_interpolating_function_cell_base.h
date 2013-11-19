#ifndef MY_P4EST_INTERPOLATING_FUNCTION_CELL_BASE_H
#define MY_P4EST_INTERPOLATING_FUNCTION_CELL_BASE_H

#include <vector>
#include <map>
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/point3.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/point2.h>
#endif

enum interpolation_method{
  linear,
  IDW,
  LSQR,
  RBF_MQ
};

#ifdef P4_TO_P8
class InterpolatingFunctionCellBase: public CF_3
#else
class InterpolatingFunctionCellBase: public CF_2
#endif
{
public:
  typedef my_p4est_cell_neighbors_t::quad_info_t quad_info_t;

private:
  p4est_t *p4est_;
  p4est_ghost_t *ghost_;
  my_p4est_brick_t *myb_;
  const my_p4est_cell_neighbors_t *cnnn_;
  interpolation_method method_;
    
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
  double linear_interpolation(const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;
  double IDW_interpolation   (const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;
  double LSQR_interpolation  (const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;
  double RBF_MQ_interpolation(const p4est_quadrant_t& quad, p4est_locidx_t quad_idx, const double *Fi_p, const double *xyz) const;
#ifdef P4_TO_P8
  bool find_tetrahedron_containing_point_m00(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const;
  bool find_tetrahedron_containing_point_p00(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const;
  bool find_tetrahedron_containing_point_0m0(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const;
  bool find_tetrahedron_containing_point_0p0(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const;
  bool find_tetrahedron_containing_point_00m(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const;
  bool find_tetrahedron_containing_point_00p(p4est_locidx_t qu, p4est_topidx_t tr, Point3 &px, double *uvw, const quad_info_t **qu123) const;
#endif

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

#ifdef P4_TO_P8
namespace {
inline double compute_det(const Point3& p1, const Point3& p2, const Point3& p3)
{
  return ( (p1.x*p2.y*p3.z + p2.x*p3.y*p1.z + p3.x*p1.y*p2.z) -
           (p1.z*p2.y*p3.x + p2.z*p3.y*p1.x + p3.z*p1.y*p2.x) );
}

inline void quad_center(const p4est_t *p4est_, const InterpolatingFunctionCellBase::quad_info_t *it, Point3& p)
{
  static double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

  static const p4est_connectivity_t *conn = p4est_->connectivity;
  static p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
  static double tree_xmin = conn->vertices[3*v_mm + 0];
  static double tree_ymin = conn->vertices[3*v_mm + 1];
  static double tree_zmin = conn->vertices[3*v_mm + 2];

  p.x = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
  p.y = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
  p.z = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;
}

inline void compute_barycentric_coordinates(const Point3 p1, const Point3& p2, const Point3& p3, const Point3& px, double *uvw)
{
  double det = compute_det(p1, p2, p3);
  uvw[0] = compute_det(px, p2, p3)/det;
  uvw[1] = compute_det(p1, px, p3)/det;
  uvw[2] = compute_det(p1, p2, px)/det;
}
}
#endif

#endif // MY_P4EST_INTERPOLATING_FUNCTION_CELL_BASE_H
