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

using std::vector;

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
  vector<Vec> Fi;

  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];
  PetscErrorCode ierr;
  int mpiret;

  struct input_buffer_t{
    vector<double> p_xyz;
    vector<p4est_locidx_t> node_idx;

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
  vector<p4est_quadrant_t> local_buffer;
  std::map<int, vector<remote_buffer_t> > send_buffer;
  vector<MPI_Request> query_req;
  vector<MPI_Request> reply_req;
  vector<int> senders;

	// using a number other than 0 to ensure no other message uses the same tag by mistake
  enum {
    query_tag = 1234,
    reply_tag
  };

  inline unsigned int n_vecs() const { return Fi.size(); }

  // methods  
  void process_incoming_query(MPI_Status& status, InterpolatingFunctionLogEntry& entry);
  void process_incoming_reply(MPI_Status& status, double * const* Fo_p);

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


  void set_input(Vec *F, unsigned int n_vecs_);
  void set_input(Vec& F) {set_input(&F, 1);}

  void add_point(p4est_locidx_t locidx, const double *xyz);
  void add_point_local(p4est_locidx_t locidx, const double *xyz);

  // interpolation methods
  void interpolate(Vec Fo)
  {
    vector<Vec> Fos(1, Fo);
    interpolate(Fos);
  }
  void interpolate(vector<Vec>& Fo) {interpolate(Fo.data(), Fo.size());}
  void interpolate(Vec * Fos, unsigned int n_outputs);
  void interpolate(double *Fo)
  {
    interpolate(&Fo, 1);
  }
  void interpolate(double * const *Fo, unsigned int n_functions);

  void interpolate_local(double *Fo_p);

#ifdef P4_TO_P8
  virtual void operator()(double x, double y, double z, double* results) const = 0;
  inline double operator()(double x, double y, double z) const
  {
    P4EST_ASSERT(Fi.size() == 1);
    double to_return;
    this->operator()(x, y, z, &to_return);
    return to_return;
  }
#else
  virtual void operator()(double x, double y, double* results) const = 0;
  inline double operator()(double x, double y) const
  {
    P4EST_ASSERT(Fi.size() == 1);
    double to_return;
    this->operator()(x, y, &to_return);
    return to_return;
  }
#endif
  inline void operator()(const double xyz[], double* results) const
  {
#ifdef P4_TO_P8
    this->operator()(xyz[0], xyz[1], xyz[2], results);
#else
    this->operator()(xyz[0], xyz[1], results);
#endif
  }
  inline double operator()(const double xyz[]) const
  {
#ifdef P4_TO_P8
    return this->operator()(xyz[0], xyz[1], xyz[2]);
#else
    return this->operator()(xyz[0], xyz[1]);
#endif
  }

  virtual void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results) const = 0;
  inline double interpolate(const p4est_quadrant_t &quad, const double *xyz)
  {
    P4EST_ASSERT(Fi.size() == 1);
    double to_return;
    interpolate(quad, xyz, &to_return);
    return to_return;
  }
};

#endif /* MY_P4EST_INTERPOLATION */
