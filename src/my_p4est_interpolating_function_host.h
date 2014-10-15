#ifndef INTERPOLATING_FUNCTION_HOST_H
#define INTERPOLATING_FUNCTION_HOST_H

#include <vector>
#include <queue>
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

#ifdef P4_TO_P8
class InterpolatingFunctionNodeBaseHost: public CF_3
#else
class InterpolatingFunctionNodeBaseHost: public CF_2
#endif
{
  const my_p4est_node_neighbors_t *neighbors_;
  const p4est_t *p4est_;
  const p4est_nodes_t *nodes_;
  const p4est_ghost_t *ghost_;
  const my_p4est_brick_t *myb_;
  Vec Fi;

  interpolation_method method_;

  double xyz_min[3], xyz_max[3];
  PetscErrorCode ierr;  

  struct input_buffer_t{
    std::vector<double> p_xyz;
    std::vector<p4est_locidx_t> node_idx;

    inline void push_back(p4est_locidx_t i, const double* xyz) {
      p_xyz.push_back(xyz[0]);
      p_xyz.push_back(xyz[1]);
#ifdef P4_TO_P8
      p_xyz.push_back(xyz[2]);
#endif

      node_idx.push_back(i);
    }

    inline size_t size() {
      return node_idx.size();
    }
  };

  struct local_buffer_t {
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
  };

  struct remote_buffer_t {
    double value;
    p4est_locidx_t input_buffer_idx;
  };
  
  std::map<int, input_buffer_t> input_buffer;
  std::vector<p4est_quadrant_t> local_buffer;
  std::map<int, std::vector<remote_buffer_t> > send_buffer;
  std::vector<MPI_Request> query_req;
  std::vector<MPI_Request> reply_req;
  std::vector<int> senders;

	// using a number other than 0 to ensure no other message uses the same tag by mistake
  enum {
    query_tag = 1234,
    reply_tag
  };

  // methods  
  void process_incoming_query(MPI_Status& status);
  void process_incoming_reply(MPI_Status& status, double *Fo_p);

  // rule of three -- disable copy ctr and assignment if not useful
  InterpolatingFunctionNodeBaseHost(const InterpolatingFunctionNodeBaseHost& other);
  InterpolatingFunctionNodeBaseHost& operator=(const InterpolatingFunctionNodeBaseHost& other);
public:
  InterpolatingFunctionNodeBaseHost(Vec F, const my_p4est_node_neighbors_t *neighbors, interpolation_method method = linear);
  ~InterpolatingFunctionNodeBaseHost();

  void add_point(p4est_locidx_t node_locidx, const double *xyz);

  // interpolation methods
  void interpolate(Vec Fo);
  void interpolate(double *Fo);
  double operator()(double x, double y
#ifdef P4_TO_P8
    , double z
#endif
    ) const;
};

#endif // INTERPOLATING_FUNCTION_HOST_H
