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
  unsigned int bs_f; // block_size of the fields to interpolate

  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];
  bool periodic[P4EST_DIM];
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
  void process_incoming_query(MPI_Status& status, InterpolatingFunctionLogEntry& entry, const unsigned int &comp);
  void process_incoming_reply(MPI_Status& status, double * const* Fo_p, const unsigned int &comp);

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_t(const my_p4est_interpolation_t& other);
  my_p4est_interpolation_t& operator=(const my_p4est_interpolation_t& other);

  void set_input(Vec *F, unsigned int n_vecs_, const unsigned int &block_size_f);
  inline void set_input(Vec& F, const unsigned int &block_size_f=1) {set_input(&F, 1, block_size_f);}
public:
  my_p4est_interpolation_t(const my_p4est_node_neighbors_t* neighbors);
  ~my_p4est_interpolation_t();

  const static unsigned int ALL_COMPONENTS=UINT_MAX;

  /*!
   * \brief clear the points buffered for interpolation. Call this method to re-use an instantiation.
   */
  void clear();

  void add_point(p4est_locidx_t locidx, const double *xyz);

  // interpolation methods and inline wrappers
  void interpolate(double * const *Fo, unsigned int n_functions, const unsigned int &comp=ALL_COMPONENTS);
  inline void interpolate(double *Fo, const unsigned int &comp=ALL_COMPONENTS) { interpolate(&Fo, 1, comp); }

  // the following 3 functions do not allow to interpolate one specific component only
  // Why? Because we would need to know whether the output vectors Fo have the same block
  // as the inputs or not and, if not, in which component of the ouput to store the results
  // This could become a serious hot mess... For instance: let's say you want to interpolate
  // one of the components of a P4EST_DIM-block-structured node-sampled velocity vector field
  // to a non-block-structured vector corresponding to a face-sampling vector, the extension
  // of the following function would be fairly confusing...
  // So we leave the following functions for interpolations from all components to all components
  // (the best way to proceed in a case like the example above, would be to call the above
  // function(s) immediately with the local array(s) of the destination vectors)
  inline void interpolate(Vec *Fos, unsigned int n_outputs)
  {
    P4EST_ASSERT(n_outputs > 0);
    double *Fo_p[n_outputs];
    for (unsigned int k = 0; k < n_outputs; ++k) {
      ierr = VecGetArray(Fos[k], &Fo_p[k]); CHKERRXX(ierr); }

    interpolate(Fo_p, n_outputs, ALL_COMPONENTS);
    for (unsigned int k = 0; k < n_outputs; ++k) {
      ierr = VecRestoreArray(Fos[k], &Fo_p[k]); CHKERRXX(ierr); }
  }
  inline void interpolate(Vec Fo){ interpolate(&Fo, 1); }
  inline void interpolate(vector<Vec>& Fo) { interpolate(Fo.data(), Fo.size()); }

#ifdef P4_TO_P8
  virtual void operator()(double x, double y, double z, double* results) const = 0;
  inline double operator()(double x, double y, double z) const
  {
    P4EST_ASSERT(Fi.size() == 1);
    P4EST_ASSERT(bs_f == 1);
    double to_return;
    this->operator()(x, y, z, &to_return);
    return to_return;
  }
#else
  virtual void operator()(double x, double y, double* results) const = 0;
  inline double operator()(double x, double y) const
  {
    P4EST_ASSERT(Fi.size() == 1);
    P4EST_ASSERT(bs_f == 1);
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

  // fully abstract function, needs to be defined in all child classes
  // the array results must be of size bs_f*n_vecs(),
  // on output, results[bs_f*k+comp] = interpolated value of the comp_th component of the kth field
  virtual void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &comp) const = 0;
  inline double interpolate(const p4est_quadrant_t &quad, const double *xyz)
  {
    P4EST_ASSERT(n_vecs() == 1);
    P4EST_ASSERT(bs_f == 1);
    double to_return;
    interpolate(quad, xyz, &to_return, 0);
    return to_return;
  }
  void add_point_local(p4est_locidx_t locidx, const double *xyz);
  void interpolate_local(double *Fo_p);
};

#endif /* MY_P4EST_INTERPOLATION */
