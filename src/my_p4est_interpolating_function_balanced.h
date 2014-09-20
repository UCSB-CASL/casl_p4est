#ifndef INTERPOLATING_FUNCTION_H
#define INTERPOLATING_FUNCTION_H

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
class InterpolatingFunctionNodeBaseBalanced: public CF_3
#else
class InterpolatingFunctionNodeBaseBalanced: public CF_2
#endif
{
  const my_p4est_node_neighbors_t *neighbors_;
  const p4est_t *p4est_;
  const p4est_nodes_t *nodes_;
  const p4est_ghost_t *ghost_;
  const my_p4est_brick_t *myb_;
  Vec Fi;

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

  struct cell_data_t {
    p4est_qcoord_t q_xyz[P4EST_DIM];
    p4est_topidx_t tree_idx;
    int8_t level;
    double f[P4EST_CHILDREN];
    p4est_locidx_t input_buffer_idx;
  };

  std::map<int, input_buffer_t> input_buffer;
  std::vector<cell_data_t> data_buffer;
  std::map<int, std::vector<cell_data_t> > send_buffer;
  std::vector<MPI_Request> point_send_req;
  std::vector<MPI_Request> data_send_req;
  std::vector<int> senders;

  enum {
    point_tag,
    data_tag
  };

  // methods  
  void process_data(const input_buffer_t* input, const cell_data_t& data, double *Fo_p);
  void process_message(MPI_Status& status, std::queue<std::pair<const input_buffer_t*, size_t> > &queue);

  // rule of three -- disable copy ctr and assignment if not useful
  InterpolatingFunctionNodeBaseBalanced(const InterpolatingFunctionNodeBaseBalanced& other);
  InterpolatingFunctionNodeBaseBalanced& operator=(const InterpolatingFunctionNodeBaseBalanced& other);
public:
  InterpolatingFunctionNodeBaseBalanced(Vec F, const my_p4est_node_neighbors_t *neighbors);
  ~InterpolatingFunctionNodeBaseBalanced();

  void add_point(p4est_locidx_t node_locidx, const double *xyz);

  // interpolation methods
  void interpolate(Vec Fo);
  void interpolate(double *Fo_p);
  double operator()(double x, double y
#ifdef P4_TO_P8
    , double z
#endif
    ) const;
};

#endif // INTERPOLATING_FUNCTION_H
