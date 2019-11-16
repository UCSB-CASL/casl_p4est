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

/*!
 * \brief The my_p4est_interpolation_t class provides a general framework for all types of
 * interpolation procedures. Regardless of the data-sampling used for the fields to inter-
 * polate (at nodes, at celles or faces), several features and basic routines are common to
 * all of them and shared. This is the purpose of this abstract class, which requires
 * specialization for a virtual method handling interpolation at the quadrant-level and for
 * a virtual operator().
 *
 * Generally speaking, this object (or, more specifically, any object of a class that inherits
 * from this abstract one) can be used in two different ways:
 * 1) as an enhanced CF_DIM structure, i.e. as an object augmented with an operator() (double *xyz)
 *    or, equivalently, an operator() (double x, double y[, double z]). When using such operators,
 *    this tool performs all the interpolation procedures and returns the interpolated value(s)
 *    at the point of interest.
 *    This usage is safe ONLY if the point coordinares xyz are known to be locally owned by the
 *    local process. Otherwise, it will trigger error(s) due to unknown field and/or data.
 * 2) as a general tool capable of evaluating interpolation(s) of field(s) at any point, anywhere
 *    by first buffering the points of interest and then handling appropriate communications with
 *    the owning processes whence "interpolate" is invoked. This usage is always safe to use but
 *    sometimes less convenient (since it requires buffering, etc.)
 *
 * See the description of
 *  void interpolate(double * const *Fo, const unsigned int &comp=ALL_COMPONENTS);
 * for mode details about the usage as in 2) here above.
 * Code revision and comments by Raphael Egan (raphaelegan@ucsb.edu), October 18, 2019.
 */
class my_p4est_interpolation_t: public CF_DIM
{
protected:


  const my_p4est_node_neighbors_t *ngbd_n;  /**< the node-neighborhood information for the "input" grid */
  const p4est_t *p4est;                     /**< the p4est structure for the "input" grid (p4est == ngbd_n->p4est) */
  p4est_ghost_t *ghost;                     /**< the ghost structure for the "input" grid (ghost == ngbd_n->ghost) */
  const my_p4est_brick_t *myb;              /**< the macromash information for the "input" grid (myb == ngbd_n->myb) */
  vector<Vec> Fi;                           /**< list of input field data
                                              (parallel, possibly block-structured, PetSc vectors) */
  unsigned int bs_f;                        /**< block size of (all) the input data fields in Fi */

  double xyz_min[P4EST_DIM];                /**< Cartesian coordinates of the "lower, left, front"
                                              corner of the "input" grid */
  double xyz_max[P4EST_DIM];                /**< Cartesian coordinates of the "upper, right, back"
                                              corner of the "input" grid */
  bool periodic[P4EST_DIM];                 /**< flags indicating if the "input" is periodic along
                                              the corresponding Cartesian direction */
  PetscErrorCode ierr;                      /**< PetscErrorCode, defined here for convenience */
  int mpiret;                               /**< mpi error code, defined here for convenience as well */


  /*!
   * \brief The input_buffer_t struct stores points that have been added to the set of interpolation points,
   * and that belong to the same owner process.
   */
  struct input_buffer_t{
    vector<double> p_xyz;                   /**< serialized point coordinates, the kth point added to this buffer
                                              has coordinates {p_xyz[P4EST_DIM*k], p_xyz[P4EST_DIM*k+1] (, p_xyz[P4EST_DIM*k+2])}*/
    vector<p4est_locidx_t> node_idx;        /**< serialized indices of the nodes in the buffer, the kth point added
                                              to this buffer has index node_idx[k]
                                              (index as desired for the insertion of the results in the output vector(s)) */

    /*!
     * \brief push_back adds a point to the buffer
     * \param [in] node_idx_on_output:    node index of the added node (as desired for the insertion
     *                                    of the results in the output vector(s))
     * \param [in] xyz:                   point Cartesian coordinates (array of P4EST_DIM doubles)
     */
    inline void push_back(const p4est_locidx_t &node_idx_on_output, const double* xyz) {
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        p_xyz.push_back(xyz[dir]);

      node_idx.push_back(node_idx_on_output);
    }

    /*!
     * \brief size
     * \return the number of points added to the buffer
     */
    inline size_t size() const {
      return node_idx.size();
    }
  };

  /*!
   * \brief The data_to_communicate union represents either type of data to be communicated
   * in the context of a "reply". The serialized replies needs to communicate the result(s)
   * of the interpolation, i.e., floating point values hence stored in "value", but also the
   * index of the corresponding point, as stored in the input_buffer_t of the process to
   * which the reply is sent, i.e. an integer number hence stored in index_in_input_buffer.
   *
   * [--- IMPORTANT NOTE ABOUT UNION (from https://en.cppreference.com/w/cpp/language/union) ---]
   * The union is only as big as necessary to hold its largest data member. The other data
   * members are allocated in the same bytes as part of that largest member. The details of
   * that allocation are implementation-defined, and it's undefined behavior to read from the
   * member of the union that wasn't most recently written.
   * [--------------------------   END OF IMPORTANT NOTE ABOUT UNION -------------------------- ]
   */
  union data_to_communicate
  {
    int     index_in_input_buffer;
    double  value;
    data_to_communicate(const int &index_in_input_buffer_=0): index_in_input_buffer(index_in_input_buffer_) {}
    data_to_communicate(const double &value_): value(value_) {}
  };
  
  std::map<int, input_buffer_t> input_buffer;               /**< map of input_buffer_t per process, i.e.,
                                                              input_buffer[r] = input_buffer_t corresponding to process of rank r */
  vector<p4est_quadrant_t> local_buffer;                    /**< list of local quadrants owning points that have been identified as local.
                                                              local_buffer[k] = quadrant owning the kth point in input_buffer[p4est->mpirank] */
  std::map<int, vector<data_to_communicate> > send_buffer;  /**< map of serialized data to be sent back to another process as a "reply" to a "query"
                                                              We serialize the data to be communicated in the following way:
                                                              index_of_point_0_in_the_input_buffer, value_0_at_point_0, value_1_at_point_0, ..., value_{nelements_per_node-1}_at_point_0,
                                                              index_of_point_1_in_the_input_buffer, value_0_at_point_1, value_1_at_point_1, ..., value_{nelements_per_node-1}_at_point_1,
                                                              ... */
  vector<MPI_Request> query_req;                            /**< list of MPI request associated with every non-blocking request sent */
  vector<MPI_Request> reply_req;                            /**< list of MPI request associated with every non-blocking reply sent */
  vector<int> senders;                                      /**< list of flags indicating if a remote process was queried
                                                              if senders[r] == 0, no query is to be sent to process r (hence no reply is expected from r either)
                                                              if senders[r] == 1, a  query is to be sent to process r (hence a reply is expected from r as well) */

  // arbitrary tags used to label the interpolation-related communications
  // between processes and to distinguish queries and replies
  enum {
    query_tag = 1234,
    reply_tag
  };

  /*!
   * \brief n_vecs
   * \return the number of input fields that have been provided
   */
  inline unsigned int n_vecs() const { return Fi.size(); }

  /*!
   * \brief process_incoming_query receives an incoming query, handles it, interpolates field(s) at the queried nodes
   * (only the ones that are locally owned) and sends a serialized reply back.
   * \param [in]    status: information structure about the MPI query message;
   * \param [in]    comp:   component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                        all components are considered if set to ALL_COMPONENTS
   * \param [inout] entry:  elementary logging entry (relevant only if CASL_LOG_EVENTS is defined)
   */
#ifdef CASL_LOG_EVENTS
  void process_incoming_query(const MPI_Status& status, const unsigned int &comp, InterpolatingFunctionLogEntry& entry);
#else
  void process_incoming_query(const MPI_Status& status, const unsigned int &comp);
#endif
  /*!
   * \brief process_incoming_reply receives a serialized reply and insert the result(s) into the appropriate output
   * vector(s)
   * \param [in]    status: information structure about the MPI reply message;
   * \param [inout] Fo_p:   constant array of pointers to local array(s) of Petsc Vectors in which the
   *                        results need to be inserted
   * \param [in]    comp:   component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                        all components are considered if set to ALL_COMPONENTS
   * \note if the block size of the input fields bs_f > 1 and if comp != ALL_COMPONENTS, we assume that the output
   *        vector(s) is (are) NOT block-structured (i.e., the ouputs have a block size of 1)
   */
  void process_incoming_reply(const MPI_Status& status, double * const* Fo_p, const unsigned int &comp);

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_t(const my_p4est_interpolation_t& other);
  my_p4est_interpolation_t& operator=(const my_p4est_interpolation_t& other);

  /*!
   * \brief set_input sets the input interpolation field(s). If communications are still pending (e.g., from a
   * former use of the same interpolator object) the application waits for those communications to be completed,
   * first. The MPI requests are then cleared as well as the send buffers. Finally, the list of input fields is
   * set as well as the block size of the input fields (must be the same for all fields).
   * \param [in] F:             array of input fields to interpolate. All fields must have the same structure
   *                            (i.e., all with the same sampling and same block-size)
   * \param [in] n_vecs_:       number of fields in the above array
   * \param [in] block_size_f:  block size of (all) the fields in the array
   */
  void set_input(Vec *F, unsigned int n_vecs_, const unsigned int &block_size_f);
  inline void set_input(Vec& F, const unsigned int &block_size_f=1) { set_input(&F, 1, block_size_f); }

  /*!
   * \brief add_point_general adds a point where the interpolated result(s) are required. This method first determines
   * which process owns the given point, then the point is added to the relevant buffer(s).
   * -  If the point is locally owned, it is added to input_buffer[p4est->mpirank] and the smallest local quadrant
   *    owning the point is added to local_buffer;
   * -  If the point is not locally owned, the point is added to the buffers of all candidate owning process(es),
   *    except if return_if_non_local is true.
   * \param [in] node_idx_on_output:  index in the output vector corresponding to the added node
   * \param [in] xyz:                 Cartesian coordinates of the added node in the computational domain
   * \param [in] return_if_not_local: flag forcing a return if set to true when the added point is non-local (default is false)
   */
  void add_point_general(const p4est_locidx_t &node_idx_on_output, const double *xyz, const bool &return_if_non_local);
public:
  /*!
   * \brief my_p4est_interpolation_t constructor. Sets ngbd_n, p4est, ghost, myb, initializes Fi to one
   * NULL input, bs_f is set to 0 and "senders" are all unset. xyz_min, xyz_max and periodic are all
   * determined from the information about the computational domain.
   * \param [in] neighbors: node neighborhood information
   */
  my_p4est_interpolation_t(const my_p4est_node_neighbors_t* neighbors);
  /*! \brief ~my_p4est_interpolation_t destructor: standard default destructor, except that pending
   * communications are awaited until completion before final destruction.
   */
  ~my_p4est_interpolation_t();

  /*!
   * \brief ALL_COMPONENTS is the default value used to indicate that ALL components of block-structured input
   * vectors need to be considered. The chosen value is set to UINT_MAX, because we reasonably assume that any
   * blocksize must be much smaller than that. (If not, you're probably wrong and you probably need to revise
   * your project)
   */
  const static unsigned int ALL_COMPONENTS=UINT_MAX;

  /*!
   * \brief clear waits for pending communications to complete, then clears all the buffers, the list of
   * MPI_requests, the list of senders is reset to 0 for all processes.
   */
  void clear();

  /*!
   * \brief add_point --> see add_point_general for more details.
   */
  void add_point(const p4est_locidx_t &node_idx_on_output, const double *xyz) { add_point_general(node_idx_on_output, xyz, false); };
  /*!
   * \brief add_point_local --> see add_point_general for more details.
   */
  inline void add_point_local(const p4est_locidx_t &node_idx_on_output, const double *xyz) { add_point_general(node_idx_on_output, xyz, true); };

  /*!
   * \brief interpolate executes the desired interpolation for all points that have been added to the input
   * buffers, based on the given input fields.
   * Upon calling this method, relevant communications are initiated:
   * 1) every process sends "queries" to remote process regarding the points that were provided but found
   *    to be non-local (non-blocking MPI_Isend);
   * 2) every process then proceeds to complete the interpolation over all required points, by "overlapping"
   *    the following tasks
   *    - calculating the interpolated values for all the nodes that are owned locally (hence non-remote);
   *    - receiving a "query" from another process and performing the interpolations at the queried nodes
   *      that have been communicated (if they are indeed locally owned). The results are assembled in a
   *      (serialized) "reply" which is sent back to the original process (i.e. the one that sent the
   *      "query" being handled). (non-blocking MPI_Isend);
   *    - receiving a serialized "reply" to a "query" that was sent in 1) and inserting these results back
   *      in the appropriate output
   * \param [inout] Fo:         constant array of pointers to double array(s) (may be local array(s) of Petsc vectors,
   *                            but not mandatorily) in which results need to be inserted
   * \param [in]    comp:       component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                            all components are considered if set to ALL_COMPONENTS
   * \note Fo MUST (and is assumed to) contain n_vecs() elements, for consistency with the given inputs
   * \note about the insertion of the results in elements of Fo:
   * if ((comp==ALL_COMPONENTS) && (bs_f > 1))
   *    --> all bs_f components of the kth input field for the node buffered with associated index "node_idx_on_output"
   *        (when using 'add_point()') are inserted in
   *        {Fo[k][bs_f*node_idx_on_output], Fo[k][bs_f*node_idx_on_output+1], ..., Fo[k][bs_f*node_idx_on_output+bs_f-1]}
   * if (bs_f == 1) or if (comp < bs_f)
   *    --> the only desired component of the kth input field for the node buffered with associated index "node_idx_on_output"
   *        (when using 'add_point()') is inserted in Fo[k][node_idx_on_output]
   */
  void interpolate(double * const *Fo, const unsigned int &comp=ALL_COMPONENTS);
  inline void interpolate(double *Fo, const unsigned int &comp=ALL_COMPONENTS) { P4EST_ASSERT(n_vecs()==1); interpolate(&Fo, comp); }

  /*!
   * \brief interpolate does the same task as the above method, except that it is specifically for Petsc parallel vector(s)
   * on output.
   * These functions do not allow to interpolate one specific component only: why? Because we would need to know whether
   * the output vectors in Fos have the same block structure as the inputs or not and, if not, in which component of the
   * ouput to store the results
   * --> This could become a serious hot mess of method: for instance, if you want to interpolate one of the components
   * of a P4EST_DIM-block-structured node-sampled vector field to a non-block-structured vector corresponding one of the
   * face-sampling vectors, the extension of the following function to handle such cases would be fairly confusing!
   * So we leave the following functions for interpolations from all components to all components only
   * (the best way to proceed in a case like the example given here above, would be to call the above function(s)
   * immediately with the local array(s) of the destination vectors and the appropriate indices given when passing
   * points with "add_point")
   * \param [inout] Fos:    array of Petsc Parallel vector(s) in which results need to be inserted
   * \note Fos is assumed to contain n_vecs() elements, for consistency with the given inputs
   * \note about the insertion of the results in elements of Fos:
   */
  inline void interpolate(Vec *Fos)
  {
    const unsigned int n_outputs = n_vecs();
    double *Fo_p[n_outputs];
    for (unsigned int k = 0; k < n_outputs; ++k) {
      ierr = VecGetArray(Fos[k], &Fo_p[k]); CHKERRXX(ierr); }

    interpolate(Fo_p, ALL_COMPONENTS);
    for (unsigned int k = 0; k < n_outputs; ++k) {
      ierr = VecRestoreArray(Fos[k], &Fo_p[k]); CHKERRXX(ierr); }
  }
  inline void interpolate(Vec Fo){ P4EST_ASSERT(n_vecs()==1); interpolate(&Fo); }
  inline void interpolate(vector<Vec>& Fo) { P4EST_ASSERT(n_vecs()==Fo.size()); interpolate(Fo.data()); }

  /*!
   * \brief operator () standard direct operator to interpolate value at any point that is locally owned.
   * This method is virtual and needs to be specified by any child class
   * \param [in] x: Cartesian coordinate along x
   * \param [in] y: Cartesian coordinate along y
   * \param [in] z: Cartesian coordinate along z (only in 3D)
   * \param [inout] results: array of results on output (size of array = bs_f*n_vecs() in general)
   * \note interpolates all components, so on output,
   * results[bs_f*k+comp]  = interpolated value of the comp_th component of the kth field
   */
#ifdef P4_TO_P8
  virtual void operator()(double x, double y, double z, double* results) const = 0;
  inline double operator()(double x, double y, double z) const
  {
    P4EST_ASSERT(n_vecs() == 1);
    P4EST_ASSERT(bs_f == 1);
    double to_return;
    this->operator()(x, y, z, &to_return);
    return to_return;
  }
#else
  virtual void operator()(double x, double y, double* results) const = 0;
  inline double operator()(double x, double y) const
  {
    P4EST_ASSERT(n_vecs() == 1);
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

  /*!
   * \brief interpolate most local version of the interpolation procedure: interpolates the values of the input field(s)
   * at the given location xyz, owned by the local quadrant quad.
   * This method is virtual and needs to be specified by any child class.
   * \param [in]    quad:     local p4est quadrant owning the point of interest
   * \param [in]    xyz:      Cartesian coordinates of the point of interest
   * \param [inout] results:  array of results on output (size of array must be ((comp==ALL_COMPONENTS && bs_f > 1) ? bs_f*n_vecs() : vecs()) in general)
   * \param [in]    comp:     component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                          all components are considered if set to ALL_COMPONENTS
   * \note on output,
   * if(comp == ALL_COMPONENTS) results[bs_f*k+comp]  = interpolated value of the comp_th component of the kth field
   * else                       results[k]            = interpolated value of the comp_th component of the kth field
   */
  virtual void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &comp) const = 0;
  inline double interpolate(const p4est_quadrant_t &quad, const double *xyz)
  {
    P4EST_ASSERT(n_vecs() == 1);
    P4EST_ASSERT(bs_f == 1);
    double to_return;
    interpolate(quad, xyz, &to_return, 0);
    return to_return;
  }
  /*!
   * \brief interpolate_local executes the interpolation only for the the nodes that were are locally owned
   * \param [inout] Fo_p: pointer to a double array (may be the local array of a Petsc vector, but not man-
   *                      datorily) in which results need to be inserted
   * \note This method is restricted to n_vecs()==1 and bs_f == 1.
   */
  void interpolate_local(double *Fo_p);
};

#endif /* MY_P4EST_INTERPOLATION */
