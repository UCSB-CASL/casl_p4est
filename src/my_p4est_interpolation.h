#ifndef MY_P4EST_INTERPOLATION
#define MY_P4EST_INTERPOLATION

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
class my_p4est_interpolation_t: public CF_3
#else
class my_p4est_interpolation_t: public CF_2
#endif
{
protected:

  const my_p4est_node_neighbors_t *ngbd_n;
  const p4est_t *p4est;
  p4est_ghost_t *ghost;
  const my_p4est_brick_t *myb;
  Vec Fi;

  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];
  PetscErrorCode ierr;
  int mpiret;

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
  void process_incoming_query(MPI_Status& status, InterpolatingFunctionLogEntry& entry);
  void process_incoming_reply(MPI_Status& status, double *Fo_p);

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_t(const my_p4est_interpolation_t& other);
  my_p4est_interpolation_t& operator=(const my_p4est_interpolation_t& other);
public:
  my_p4est_interpolation_t(const my_p4est_node_neighbors_t* neighbors);
  ~my_p4est_interpolation_t();

  /*!
   * \brief clear the points buffered for interpolation. Call this method to re-use an instantiation.
   */
  void clear();

  void set_input(Vec F);

  void add_point(p4est_locidx_t locidx, const double *xyz);

  // interpolation methods
  void interpolate(Vec Fo);
  void interpolate(double *Fo);

#ifdef P4_TO_P8
  virtual double operator()(double x, double y, double z) const = 0;
#else
  virtual double operator()(double x, double y) const = 0;
#endif

  virtual double interpolate(const p4est_quadrant_t &quad, const double *xyz) const = 0;

  void add_point_local(p4est_locidx_t locidx, const double *xyz);
  void interpolate_local(double *Fo_p);
};

#endif /* MY_P4EST_INTERPOLATION */
