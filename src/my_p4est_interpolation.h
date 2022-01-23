#ifndef MY_P4EST_INTERPOLATION
#define MY_P4EST_INTERPOLATION

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_node_neighbors.h>
#endif

#include <map>

using std::vector;

/*!
 * \brief The my_p4est_interpolation_t class provides a general framework for synchronization
 * of nonlocal calculations/evaluations; in particular, this abstract class creates a general
 * contexte for all types of field interpolation procedures. Regardless of the data-sampling
 * used for the fields to interpolate (at nodes, at celles or faces), several features and basic
 * routines are common to all of the routines and inherited from this class. This abstract layer
 * requires specialization for a pure virtual method handling interpolation at the quadrant-level
 * and for a pure virtual operator().
 *
 * Generally speaking, this object (or, more specifically, any object of a class that inherits
 * from this abstract one) can be used in three different ways:
 * 1) as an enhanced CF_DIM structure, i.e. as an object augmented with an operator() (double *xyz)
 *    or, equivalently, an operator() (double x, double y[, double z]). When using such operators,
 *    this tool performs all the interpolation procedures and returns the interpolated value(s)
 *    at the point of interest.
 *    This usage is safe ONLY if the point coordinares xyz is locally owned by the local process
 *    or if all the otherwise required/calculated data is available in the ghost layer.
 *    (For cell- and face-sampled fields, this operator prints a warning message if called in a
 *    ghost cell, as the required neighborhood could be less than if the cell was local.
 *    For node-sampled fields, this operator is safe if the interpolation method is linear or
 *    if all second derivatives are precalculated and provided to the object).
 * 2) as a general tool capable of evaluating interpolation(s) of field(s) at any point, anywhere
 *    by first buffering the points of interest and then handling appropriate communications with
 *    the owning processes whence "interpolate" is invoked. This usage is always safe to use but
 *    sometimes less convenient (since it requires buffering, etc.)
 * 3) (only if called from a friend class) to sample interface boundary condition types and value
 *    at any point in the domain, local or remote (useful for geometric extrapolations, see the
 *    my_p4est_level_set* classes)
 *
 * See the description of
 * void interpolate(double * const *Fo_p, const u_int &comp = ALL_COMPONENTS, const bool &local_only = false);
 * here below for mode details about the usage as in 2) here above.
 * - Code revision and comments by Raphael Egan (raphaelegan@ucsb.edu), October 18, 2019.
 * - Changing communication-related routines to make them template in order to allow communications
 * of other data types. Subsequent addition of similar features for sampling interface boundary conditions
 * by Raphael Egan (raphaelegan@ucsb.edu), April 19, 2020.
 */
class my_p4est_interpolation_t : public CF_DIM
{
  friend class my_p4est_level_set_faces_t;
  friend class my_p4est_level_set_cells_t;
  friend class my_p4est_level_set_t;
private:
  /*!
   * \brief The input_buffer_t struct stores points that have been added to the set of interpolation points,
   * and that belong to the same owner process.
   */
  struct input_buffer_t{
    vector<double> p_xyz;             /**< serialized point coordinates, the kth point added to this buffer
                                        has coordinates {p_xyz[P4EST_DIM*k], p_xyz[P4EST_DIM*k + 1] (, p_xyz[P4EST_DIM*k + 2])}*/
    vector<p4est_locidx_t> node_idx;  /**< serialized indices of the nodes in the buffer, the kth point added
                                        to this buffer has index node_idx[k]
                                        (index as desired for the insertion of the results in the output vector(s),
                                        it does not necessarily need to be the actual local node/cell/face index) */

    /*!
     * \brief push_back adds a point to the buffer
     * \param [in] node_idx_on_output:    node index of the added node (as desired for the insertion
     *                                    of the results in the output data array(s))
     * \param [in] xyz:                   Cartesian coordinates of the point of interest (array of P4EST_DIM doubles)
     */
    inline void push_back(const p4est_locidx_t &node_idx_on_output, const double *xyz) {
      for (u_char dim = 0; dim < P4EST_DIM; ++dim)
        p_xyz.push_back(xyz[dim]);

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
   * in the context of a "reply". The serialized replies need to communicate the result(s)
   * of the interpolation/local evaluation, i.e., floating point values, boundary condition
   * samples, ... or any user-defined type T (--> stored in "value" in such a case), but also
   * the index of the corresponding point, as stored in the input_buffer_t of the process to
   * which the reply is sent, i.e. an integer number (stored in index_in_input_buffer).
   *
   * [--- IMPORTANT NOTE ABOUT UNION (from https://en.cppreference.com/w/cpp/language/union) ---]
   * The union is only as big as necessary to hold its largest data member. The other data
   * members are allocated in the same bytes as part of that largest member. The details of
   * that allocation are implementation-defined, and it's undefined behavior to read from the
   * member of the union that wasn't most recently written.
   * [--------------------------   END OF IMPORTANT NOTE ABOUT UNION -------------------------- ]
   */
  template<typename T>
  union data_to_communicate
  {
    int     index_in_input_buffer;
    T       value;
    data_to_communicate(const int &index_in_input_buffer_ = 0): index_in_input_buffer(index_in_input_buffer_) {}
    data_to_communicate(const T &value_) : value(value_) {}
  };

  std::map<int, input_buffer_t> input_buffer; /**< map of input_buffer_t per process, i.e.,
                                                input_buffer[r] = input_buffer_t corresponding to process of rank r */
  vector<p4est_quadrant_t> local_buffer;      /**< list of local quadrants owning points that have been identified as local.
                                                local_buffer[k] = quadrant owning the kth point in input_buffer[p4est->mpirank] */
  std::map<int, vector<data_to_communicate<double> > > send_buffer;
                                              /**< map of serialized floating-point value data to be sent back to another process as a "reply" to a "query"
                                                We serialize the data to be communicated in the following way:
                                                index_of_point_0_in_the_input_buffer, value_0_at_point_0, value_1_at_point_0, ..., value_{nelements_per_node-1}_at_point_0,
                                                index_of_point_1_in_the_input_buffer, value_0_at_point_1, value_1_at_point_1, ..., value_{nelements_per_node-1}_at_point_1,
                                                ... */

  vector<MPI_Request> query_req;              /**< list of MPI request associated with every non-blocking request sent */
  vector<MPI_Request> reply_req;              /**< list of MPI request associated with every non-blocking reply sent */
  vector<int> senders;                        /**< list of flags indicating if a remote process was queried
                                              if senders[r] == 0, no query is to be sent to process r (hence no reply is expected from r either)
                                              if senders[r] == 1, a  query is to be sent to process r (hence a reply is expected from r as well) */

  // arbitrary-defined tags used to label the interpolation-related communications
  // between processes and to distinguish queries and replies
  enum {
    query_tag = 1234,
    reply_tag
  };

  /*!
   * \brief receive_queried_coordinates_and_allocate_send_buffer_in_map : receive a list of point coordinates along from
   * another process that believe you might be the appropriate owner.
   * \param [out] xyz                   : on output, vector of queried point coordinates, serialized (x_point_0, y_point_0 [, z_point_0], x_point_1, y_point_1 [, z_point_1], ...)
   * \param [inout] map_of_send_buffers : map of buffered replies (per destination process) that the current process needs to fill before sending back
   * \param [in]  nelements_per_point   : number of queried data values per point that we will need to calculate and send back
   * \param [in]  status                : MPI_Status of the message we receive
   */
#ifdef CASL_LOG_EVENTS
  template<typename T> inline void receive_queried_coordinates_and_allocate_send_buffer_in_map(std::vector<double> &xyz, std::map<int, vector<data_to_communicate<T> > > &map_of_send_buffers, const size_t &nelements_per_point, const MPI_Status &status, InterpolatingFunctionLogEntry& entry)
#else
  template<typename T> inline void receive_queried_coordinates_and_allocate_send_buffer_in_map(std::vector<double> &xyz, std::map<int, vector<data_to_communicate<T> > > &map_of_send_buffers, const size_t &nelements_per_point, const MPI_Status &status)
#endif
  {
    // receive incoming queries about points and send back the interpolated result
    int vec_size;
    int mpiret = MPI_Get_count(&status, MPI_DOUBLE, &vec_size); SC_CHECK_MPI(mpiret);
    P4EST_ASSERT(vec_size%P4EST_DIM == 0);
    xyz.resize(vec_size);

#ifdef CASL_LOG_EVENTS
    // log information
    entry.num_recv_points += vec_size / P4EST_DIM;
    entry.num_recv_procs++;
#endif

    mpiret = MPI_Recv(&xyz[0], vec_size, MPI_DOUBLE, status.MPI_SOURCE, query_tag, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

    std::vector<data_to_communicate<T> > &buff = map_of_send_buffers[status.MPI_SOURCE];
    // For every point we take care of, we need to send back,
    // 1) the index (in the received serialized list of coordinates) of the queried point corresponding to the data == 1 'data_to_communicate' element (with 'index_in_input_buffer' well-defined)
    // 2) the data (serialized) --> nelements_per_point 'data_to_communicate' elements (all with 'value' well-defined)
    // that makes a (maximum) memory requirement of (vec_size/P4EST_DIM)*(1 + nelements_per_point)

    buff.reserve((vec_size/P4EST_DIM)*(nelements_per_point + 1));
//    // added conversion from int to size_t for consistency // Elyce 1/12/22 -- did not change bug though
//    buff.reserve((static_cast<size_t>(vec_size)/P4EST_DIM)*(nelements_per_point + 1));

    P4EST_ASSERT(buff.size() == 0);
    return;
  }

  /*!
   * \brief send_response_back_to_query self-explanatory template function
   * \param [in] buff   : serialized data_to_communicate that we need to send back as a response
   * \param [in] status : MPI_Status of the message we send
   */
  template<typename T> inline void send_response_back_to_query(const std::vector<data_to_communicate<T> > &buff, const MPI_Status &status)
  {
    MPI_Request req;
    int mpiret = MPI_Isend(buff.data(), buff.size()*sizeof (data_to_communicate<T>), MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    reply_req.push_back(req);
  }

  /*!
   * \brief receive_incoming_reply self-explanatory template function
   * \param [out] reply_buffer        : buffer storing the response from another queried process, the results are serialized as
   *                                    (node_idx_in_input_buffer, data_value_0, data_value_1, ..., data_value_{nelements_per_point - 1}, etc.)
   *                         stored in    'index_in_input_buffer',       'value',      'value', ...,                               'value', 'index_in_input_buffer'
   *                                    of the unions in reply_buffer
   * \param [in] nelements_per_point  : number of queried data values per point that we need to receive back
   * \param [in] status               : MPI_Status of the message we receive
   */
  template<typename T> inline void receive_incoming_reply(std::vector<data_to_communicate<T> > &reply_buffer, const size_t &nelements_per_point, const MPI_Status &status) const
  {
    // receive incoming reply we asked before
    int byte_count;
    int mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
    P4EST_ASSERT(byte_count%sizeof (data_to_communicate<T>) == 0);
    reply_buffer.resize(byte_count/sizeof (data_to_communicate<T>));
    P4EST_ASSERT(byte_count%((nelements_per_point + 1)*sizeof (data_to_communicate<T>)) == 0); (void) nelements_per_point; // to avoid compilation warning of unused variable in release

    mpiret = MPI_Recv(&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, reply_tag, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
    return;
  }

  /*!
   * \brief process_incoming_query* receives an incoming query, handles it, interpolates field(s) or evaluates the
   * interface boundary condition type and values locally at the queried points (only the ones that are locally owned)
   * and sends a serialized reply back.
   * \param [in]    status: information structure about the MPI query message;
   * \param [in]    comp:   component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                        all components are considered if set to ALL_COMPONENTS (irrelevant for interface boundary conditons)
   * \param [inout] entry:  elementary logging entry (relevant only if CASL_LOG_EVENTS is defined)
   */
#ifdef CASL_LOG_EVENTS
  void process_incoming_query(const MPI_Status &status, const u_int &comp, InterpolatingFunctionLogEntry& entry);
  void process_incoming_query_interface_bc(const BoundaryConditionsDIM& bc_to_sample, std::map<int, std::vector<data_to_communicate<bc_sample> > > &map_of_send_buffers_for_bc_samples, const MPI_Status &status, InterpolatingFunctionLogEntry& entry);
#else
  void process_incoming_query(const MPI_Status &status, const u_int &comp);
  void process_incoming_query_interface_bc(const BoundaryConditionsDIM& bc_to_sample, std::map<int, std::vector<data_to_communicate<bc_sample> > > &map_of_send_buffers_for_bc_samples, const MPI_Status &status);
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
  void process_incoming_reply(const MPI_Status& status, double * const *Fo_p, const u_int &comp) const;
  void process_incoming_reply_interface_bc(const MPI_Status& status, bc_sample* interface_bc) const;

  /*!
   * \brief complete_and_clear_communications self-explanatory
   */
  inline void complete_and_clear_communications()
  {
    if(query_req.size() > 0){
      int mpiret = MPI_Waitall(query_req.size(), &query_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret); }
    if(reply_req.size() > 0){
      int mpiret = MPI_Waitall(reply_req.size(), &reply_req[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret); }
    query_req.clear();
    reply_req.clear();
    return;
  }

  /*!
   * \brief determine_and_initiate_global_communications figures out how many queries the current process is
   * expected to received, how many queries the current process needs to send and how many of the points that
   * the current process had buffered need to be taken care of remotely
   * \param [out] num_remaining_replies : number of replies to expect (== number of queries the current process will send)
   * \param [out] num_remote_points     : number of the buffered points that will need to be handled remotely
   * \param [out] num_remaining_queries : number of queries that the current process will receive from other processes
   */
  void determine_and_initiate_global_communications(int &num_remaining_replies, int &num_remote_points, int&num_remaining_queries) ;

protected:

  const my_p4est_node_neighbors_t *ngbd_n;  /**< the node-neighborhood information for the "input" grid */
  const p4est_t *p4est;                     /**< the p4est structure for the "input" grid (p4est == ngbd_n->p4est) */
  const p4est_ghost_t *ghost;               /**< the ghost structure for the "input" grid (ghost == ngbd_n->ghost) */
  const my_p4est_brick_t *myb;              /**< the macromash information for the "input" grid (myb == ngbd_n->myb) */
  vector<Vec> Fi;                           /**< list of input field data
                                              (parallel, possibly block-structured, PetSc vectors) */
  u_int bs_f;                               /**< block size of (all) the input data fields in Fi */

  /*!
   * \brief get_xyz_min accesses the coordinates of the "lower, left, front" corner of the "input" grid
   * \return a pointer to the Cartesian coordinates
   */
  inline const double* get_xyz_min() const    { return myb->xyz_min; }
  /*!
   * \brief get_xyz_max accesses the coordinates of the "upper, right, back" corner of the "input" grid
   * \return a pointer to the Cartesian coordinates
   */
  inline const double* get_xyz_max() const    { return myb->xyz_max; }

  /*!
   * \brief get_periodicity accesses boolean flags indicating if the "input" grid is periodic along the
   * corresponding Cartesian direction
   * \return a pointer to the flags along Cartesian directions
   */
  inline const bool* get_periodicity() const  { return ngbd_n->get_hierarchy()->get_periodicity(); }

  /*!
   * \brief n_vecs
   * \return the number of input fields that have been provided
   */
  inline size_t n_vecs() const { return Fi.size(); }

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_t(const my_p4est_interpolation_t& other);
  my_p4est_interpolation_t& operator=(const my_p4est_interpolation_t& other);

  /*!
   * \brief set_input_fields sets the input interpolation field(s) given as PetSc parallel vector(s). If communications
   * are still pending (e.g., from a former use of the same interpolator object) the application waits for those
   * communications to be completed, first. The MPI requests are then cleared as well as the send buffers. Finally,
   * the list of input fields is set as well as the block size of the input fields (must be the same for all fields).
   * \param [in] inputs:        array of input fields to interpolate or boundary condition objects . In case of fields,
   *                            all PetSc vectors must have the same structure (i.e., all with the same sampling and
   *                            same block-size)
   * \param [in] n_vecs_:       number of objects in the above array
   * \param [in] block_size_f:  block size of (all) the PetSc vectors in the array (irrelevant if playing with Boundary
   *                            condition objects)
   */
  void set_input_fields(const Vec *F, const size_t &n_vecs_, const u_int &block_size_f);

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

  /*!
   * \brief clip_point_and_interpolate_all_on_the_fly : method designed to complete interpolation on-the-fly at a given point
   * --> to be used by local operator().
   * \param [in] xyz              : array of const P4EST_DIM doubles representing the coordinates of the point where
   *                                interpolation is queried.
   * \param [in] comp             : component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                                (all components are considered if set to ALL_COMPONENTS)
   * \param [out] results         : array of (all) results
   *                                (must be of size n_vecs()*bs_f if comp == ALL_COMPONENTS, n_vecs() otherwise)
   * \param [out] rank_found      : rank of the process found to be the owner of the quadrant in which the point was found
   * \param [out] remote_matches  : vector of remote_matches as constructed by hierarchy's find_smallest_quadrant_containing_point
   *                                (in case of no locally found quadrant)
   * \param [in] proceed_even_if_in_ghost_layer : flag indicating whether to proceed with calculation of the interpolation when the
   *                                point is found to belong to the ghost layer or not.
   *
   * [Note 1:] this method prioritizes local over ghost quadrants (if same size) when fetching the quadrant containing the point of
   * interest, in order to enable local calculations as much as possible.
   *
   * [Note 2:] This method throws an invalid_argument exception if xyz is found outside of the local partition (and out of its ghost layer, too
   * if proceed_even_if_in_ghost_layer was set to true).
   */
  inline void clip_point_and_interpolate_all_on_the_fly(const double* xyz, const u_int& comp, double* results, int &rank_found, std::vector<p4est_quadrant_t>& remote_matches,
                                                        const bool &proceed_even_if_in_ghost_layer) const
  {
    /* first clip the coordinates */
    double xyz_clip[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
    clip_in_domain(xyz_clip, get_xyz_min(), get_xyz_max(), get_periodicity());

    p4est_quadrant_t best_match;
    rank_found = ngbd_n->get_hierarchy()->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches, true, true);

    if (rank_found == p4est->mpirank || (proceed_even_if_in_ghost_layer && rank_found != -1))
    {
      interpolate(best_match, xyz, results, comp);
      return;
    }
    throw std::invalid_argument(std::string("my_p4est_interpolation_t::clip_point_and_interpolate(): the point does not belong to the local partition of the forest") + std::string(proceed_even_if_in_ghost_layer ? " nor to its ghosts." : "."));
  }

  /* the following does the same as 'interpolate' but it works on interface boundary condition samples
   * --> there is quite a bit of code duplication in there but the only way to avoid it (that I could
   * think of) would be to make this entire class template, including the method
   * 'interpolate(const p4est_quadrant&)' which is pure virtual function. However, that is not allowed
   * with template methods --> I spent way too much time trying to figure that out... I decided to template
   * all other common methods that are common to avoid code duplication there :-/
   * I make this method private because it's supposed to be called from friend my_p4est_level_set_*
   * classes only...
   * [Raphael] */
  void evaluate_interface_bc(const BoundaryConditionsDIM &bc_to_sample, bc_sample *interface_bc);

public:
  /*!
   * \brief my_p4est_interpolation_t constructor. Sets ngbd_n, p4est, ghost, myb, initializes Fi to one
   * NULL input, bs_f is set to 0 and "senders" are all unset.
   * \param [in] neighbors: pointer to a constant node neighborhood information
   */
  my_p4est_interpolation_t(const my_p4est_node_neighbors_t* neighbors);
  /*! \brief ~my_p4est_interpolation_t destructor: standard default destructor, except that pending
   * communications are awaited until completion before standard final destruction of all member (implicit).
   */
  ~my_p4est_interpolation_t();

  /*!
   * \brief ALL_COMPONENTS is the default value used to indicate that ALL components of block-structured input
   * vectors need to be considered. The chosen value is set to UINT_MAX, because we reasonably assume that any
   * blocksize must be much smaller than that. (If not, you're probably wrong and you probably need to revise
   * your project)
   */
  const static u_int ALL_COMPONENTS = UINT_MAX;

  /*!
   * \brief clear waits for pending communications to complete, then clears all the buffers, the list of
   * MPI_requests, the list of senders is reset to 0 for all processes.
   */
  void clear();

  /*!
   * \brief add_point --> see add_point_general for more details.
   */
  inline void add_point(const p4est_locidx_t &node_idx_on_output, const double *xyz){ add_point_general(node_idx_on_output, xyz, false); };
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
   * \param [inout] Fo_p: constant array of pointers to double/bc_sample array(s) (may be local array(s) of Petsc
   *                      vectors, but not mandatorily) in which results need to be inserted
   * \param [in]    comp: component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                      all components are considered if set to ALL_COMPONENTS (relevant only if interpolating
   *                      floating-point values)
   * \note Fo MUST (and is assumed to) contain n_vecs() elements, for consistency with the given inputs
   * \note about the insertion of the results in elements of Fo:
   * if (comp == ALL_COMPONENTS && bs_f > 1)
   *    --> all bs_f components of the kth input field for the node buffered with associated index "node_idx_on_output"
   *        (when using 'add_point()') are inserted in
   *        {Fo[k][bs_f*node_idx_on_output], Fo[k][bs_f*node_idx_on_output + 1], ..., Fo[k][bs_f*node_idx_on_output + bs_f - 1]}
   * if (bs_f == 1) or if (comp < bs_f)
   *    --> the only desired component of the kth input field for the node buffered with associated index "node_idx_on_output"
   *        (when using 'add_point()') is inserted in Fo[k][node_idx_on_output]
   */
  void interpolate(double * const *Fo_p, const u_int &comp = ALL_COMPONENTS, const bool &local_only = false);
  inline void interpolate(double *Fo, const u_int &comp = ALL_COMPONENTS, const bool &local_only = false) { P4EST_ASSERT(n_vecs() == 1); interpolate(&Fo, comp, local_only); }

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
    const u_int n_outputs = Fi.size();
    std::vector<double *> Fo_p(n_outputs);
    for (u_int k = 0; k < n_outputs; ++k) {
      PetscErrorCode ierr = VecGetArray(Fos[k], &Fo_p[k]); CHKERRXX(ierr); }

    interpolate(Fo_p.data(), ALL_COMPONENTS);
    for (u_int k = 0; k < n_outputs; ++k) {
      PetscErrorCode ierr = VecRestoreArray(Fos[k], &Fo_p[k]); CHKERRXX(ierr); }
  }
  inline void interpolate(Vec Fo){ P4EST_ASSERT(Fi.size() == 1); interpolate(&Fo); }
  inline void interpolate(vector<Vec> &Fo) { P4EST_ASSERT(Fi.size() == Fo.size()); interpolate(Fo.data()); }

  /*!
   * \brief operator () standard direct operator to interpolate value at any point that is locally owned.
   * This method is purely virtual and needs to be specified by any child class
   * \param [in] xyz: pointer to a constant array of P4EST_DIM Cartesian coordinates along x y and z
   * \param [in] comp: component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                   (all components are considered if set to ALL_COMPONENTS)
   * \param [in] results: pointer to an array of results to be set internally by the procedure so that
   *                  the values are accessible to the user afterwards
   *                  the size of the pointed array must be
   *                  - bs_f*n_vecs() if comp == ALL_COMPONENT;
   *                  - n_vecs() otherwise;
   * \note interpolates all components, so on output,
   * results[bs_f*k + comp] = interpolated value of the comp_th component of the kth field
   * --> bs_f > 1 not implemented yet for cell- or face-sampled fields
   */
  virtual void operator()(const double *xyz, double *results, const u_int& comp) const = 0;
  inline double operator()(const double *xyz, const u_int& comp = ALL_COMPONENTS) const
  {
    // we first check that there is indeed only 1 value to return in DEBUG (otherwise the user doesn't know what he is doing with this usage)
    P4EST_ASSERT(Fi.size() == 1);
    P4EST_ASSERT(comp < bs_f || (comp == ALL_COMPONENTS && bs_f == 1));
    double to_return;
    this->operator()(xyz, &to_return, comp);
    return to_return;
  }
  inline void operator() (DIM(double x, double y, double z), double *results, const u_int& comp = ALL_COMPONENTS) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    this->operator()(xyz, results, comp);
    return;
  }
  inline double operator()(DIM(double x, double y, double z), const u_int& comp) const
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return this->operator()(xyz, comp);
  }
  inline double operator()(DIM(double x, double y, double z)) const // needs to be defined explicitly because we inherit CF_DIM (can't use the standard default argument in the above...)
  {
    const double xyz[P4EST_DIM] = {DIM(x, y, z)};
    return this->operator()(xyz, ALL_COMPONENTS);
  }

  /*!
   * \brief interpolate: most local version of the interpolation procedure: interpolates the values of the input field(s)
   * at the given location xyz, owned by the local or ghost quadrant quad.
   * This method is virtual and needs to be specified by any child class.
   * \param [in]    quad:     local or ghost p4est quadrant owning the point of interest
   *                          [IMPORTANT REMARK 1:] the p.piggy3 value of the quadrant must be valid
   *                          [IMPORTANT REMARK 2:] the p.piggy3.local_num must be the CUMULATIVE local quadrant index over the
   *                                                local trees! (contrary to what is returned by find_smallest_quadrant() from
   *                                                my_p4est_hierarchy_t)!
   * \param [in]    xyz:      Cartesian coordinates of the point of interest
   * \param [inout] results:  array of results on output (size of array must be (comp == ALL_COMPONENTS && bs_f > 1 ? bs_f*n_vecs() : vecs()) in general)
   * \param [in]    comp:     component of the (possibly block-structured) Petsc parallel vector(s) to be interpolated
   *                          all components are considered if set to ALL_COMPONENTS
   * \note on output,
   * if(comp == ALL_COMPONENTS) results[bs_f*k + comp]  = interpolated value of the comp_th component of the kth field
   * else                       results[k]              = interpolated value of the comp_th component of the kth field
   */
  virtual void interpolate(const p4est_quadrant_t &quad, const double *xyz, double *results, const u_int &comp) const = 0;
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
   * \param [inout] Fo_p: pointer to a double/bc_sample array (may be the local array of a Petsc vector in
   *                      case of doubles, but not mandatorily) in which results need to be inserted
   * \note This method is restricted to n_vecs() == 1 and bs_f == 1.
   */
  inline void interpolate_local(double *Fo_p)
  {
    P4EST_ASSERT(n_vecs() == 1);
    P4EST_ASSERT(bs_f == 1);
    interpolate(&Fo_p, 0, true);
  }

  inline const std::vector<Vec>& get_input_fields() const { return Fi; }
  inline const u_int& get_blocksize_of_input_fields() const { return bs_f; }
};


#endif /* MY_P4EST_INTERPOLATION */
