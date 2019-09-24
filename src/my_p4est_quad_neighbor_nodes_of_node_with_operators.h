#ifndef MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H
#define MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#endif
#include <petscvec.h>
#include <type_traits>
#include <iostream>

//---------------------------------------------------------------------
//
// quad_neighbor_nodes_of_nodes_t : neighborhood of a node in Quadtree
//
//        node_m0_p                         node_p0_p
//             |                                 |
//      d_m0_p |                                 | d_p0_p
//             |                                 |
//             --- d_m0 ----- node_00 --- d_p0 ---
//             |                                 |
//      d_m0_m |                                 | d_p0_m
//             |                                 |
//        node_m0_m                         node_p0_m
//
//---------------------------------------------------------------------


#ifndef CASL_LOG_TINY_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
#warning "Use of 'CASL_LOG_TINY_EVENTS' macro is discouraged but supported. Logging tiny sections of the code may produce unreliable results due to overhead."
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_central_derivative;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_forward_derivative;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_backward_derivative;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_central_second_derivative;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_linear_interpolator;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

static const double zero_threshold = EPS;
static bool inline check_if_zero(const double& value) { return (fabs(value) < zero_threshold); } // needs a nondimensional argument, otherwise it's meaningless
static double inline dd_correction_weight_to_naive_first_derivative(const double &d_m, const double &d_p, const double &d_m_m, const double &d_m_p, const double &d_p_m, const double &d_p_p)
{
  return 0.5*(d_p_m*d_p_p*d_m/d_p - d_m_m*d_m_p*d_p/d_m)/(d_m+d_p);
}

// forward declaration
class my_p4est_node_neighbors_t;

template<typename T> inline T central_derivative(const T &fp, const T &f0, const T &fm, const double &dp, const double &dm)
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_central_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_LOG_FLOPS
  ierr = PetscLogFlops(4+(std::is_floating_point<T>::value?7:0)); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_central_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
  if(check_if_zero((dm-dp)/MAX(dm, dp))) // needs a nondimensional argument, that means that dm == dp basically, so 0.0 weight for f0, do not even calculate it, important if playing with noe_linear_combination types
    return (fp-fm)/(dm+dp);
  return ((fp-f0)*dm/dp + (f0-fm)*dp/dm)/(dp+dm);
}

template<typename T> inline T forward_derivative(const T &fp, const T &f0, const double &dp)
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_forward_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_LOG_FLOPS
  ierr = PetscLogFlops((std::is_floating_point<T>::value?2:0)); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_forward_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
  return (fp-f0)/dp;
}

template<typename T> inline T backward_derivative(const T &f0, const T &fm, const double &dm)
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_backward_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_LOG_FLOPS
  ierr = PetscLogFlops((std::is_floating_point<T>::value?2:0)); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_backward_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
  return (f0-fm)/dm;
}

template<typename T> inline T central_second_derivative(const T &fp, const T &f0, const T &fm, const double &dp, const double &dm)
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_central_second_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_LOG_FLOPS
  ierr = PetscLogFlops((std::is_floating_point<T>::value?8:0)); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_central_second_derivative, 0, 0, 0, 0); CHKERRXX(ierr);
  return ((fp-f0)/dp + (fm-f0)/dm)*2./(dp+dm);
}

typedef struct node_interpolation_weight{
  node_interpolation_weight(){};
  node_interpolation_weight(const p4est_locidx_t & idx, const double& ww_): weight(ww_), node_idx(idx) {}
  double weight;
  p4est_locidx_t node_idx;
  bool operator <(const node_interpolation_weight& other) const { return (node_idx < other.node_idx); }
  node_interpolation_weight operator- () const { node_interpolation_weight copy(*this); copy.weight *=-1; return copy; }
} node_interpolation_weight;

typedef struct node_linear_combination
{
  node_linear_combination() { elements.resize(0); }
  std::vector<node_interpolation_weight> elements;
  double calculate(const double *node_sampled_field, const unsigned int &block_size, const unsigned int &component) const
  {
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate, 0, 0, 0, 0); CHKERRXX(ierr);
    P4EST_ASSERT((component < block_size) && (block_size > 0));
    P4EST_ASSERT(elements.size()>0);
    double value = elements[0].weight*node_sampled_field[block_size*elements[0].node_idx+component];
    for (size_t k = 1; k < elements.size(); ++k)
      value += elements[k].weight*node_sampled_field[block_size*elements[k].node_idx+component];
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(2*interpolator.elements.size()-1); CHKERRXX(ierr);
#endif
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate, 0, 0, 0, 0); CHKERRXX(ierr);
    return value;
  }
  double calculate_dd (unsigned char der, const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size, const unsigned int &component) const;
  inline double calculate_dxx(const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size, const unsigned int &component) const
  {
    return calculate_dd(dir::x, node_sample_field, neighbors, block_size, component);
  }
  inline double calculate_dyy(const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size, const unsigned int &component) const
  {
    return calculate_dd(dir::y, node_sample_field, neighbors, block_size, component);
  }
#ifdef P4_TO_P8
  inline double calculate_dzz(const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size, const unsigned int &component) const
  {
    return calculate_dd(dir::z, node_sample_field, neighbors, block_size, component);
  }
#endif

  /* same thing : if multiplying by 0.0, this means, it becomes useless, be aware of it, if possible*/
  node_linear_combination& operator*=(const double& alpha) {
    for (size_t k = 0; k < elements.size(); ++k)
      elements[k].weight *= alpha;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(elements.size()); CHKERRXX(ierr);
#endif
    return *this;
  }
  /* same thing : if multiplying by 0.0, this means, it becomes useless, be aware of it, if possible*/
  node_linear_combination operator*(const double& alpha) const {
    node_linear_combination tmp(*this);
    for (size_t k = 0; k < tmp.elements.size(); ++k)
    {
      tmp.elements[k].node_idx  = this->elements[k].node_idx;
      tmp.elements[k].weight    = alpha*(this->elements[k].weight);
    }
    return tmp;
  }
  node_linear_combination& operator/=(const double& alpha) {
    return (*this*=(1/alpha));
  }
  /* same thing : if multiplying by 0.0, this means, it becomes useless, be aware of it, if possible*/
  node_linear_combination operator/(const double& alpha) const {
    return (*this)*(1.0/alpha);
  }

  /* same thing : avoid adding exactly 0.0 weights if possible */
  node_linear_combination& operator+=(const node_interpolation_weight& node_weight) {
#ifdef P4EST_DEBUG
    const size_t init_size = elements.size();
#endif
    // we add an element to the vector if it is not in there yet,
    // otherwise, we update the weight of the already existing vector
    std::vector<node_interpolation_weight>::iterator position = std::lower_bound(this->elements.begin(), this->elements.end(), node_weight);
    P4EST_ASSERT((position == this->elements.end()) || (position->node_idx >= node_weight.node_idx));
    if((position != this->elements.end()) && (position->node_idx == node_weight.node_idx))
      position->weight += node_weight.weight;
    else
      elements.insert(position, node_weight);
#ifdef P4EST_DEBUG
    P4EST_ASSERT((elements.size() >= init_size) || (elements.size() <= init_size+1));
    bool check_if_still_sorted_and_unique_indices = true;
    for (size_t k = 1; check_if_still_sorted_and_unique_indices && (k < elements.size()); ++k)
      check_if_still_sorted_and_unique_indices = check_if_still_sorted_and_unique_indices && (elements[k].node_idx > elements[k-1].node_idx);
    P4EST_ASSERT(check_if_still_sorted_and_unique_indices);
#endif
    return *this;
  }
  node_linear_combination& operator+=(const node_linear_combination& other) {
    // add one by one
    this->elements.reserve(this->elements.size() + other.elements.size());
    for (size_t k = 0; k < other.elements.size(); ++k)
      (*this) += other.elements[k];
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(other.elements.size()); CHKERRXX(ierr);
#endif
    return *this;
  }
  node_linear_combination& operator-=(const node_linear_combination& other) {
    // subtract one by one
    this->elements.reserve(this->elements.size() + other.elements.size());
    for (size_t k = 0; k < other.elements.size(); ++k)
      (*this) += (-other.elements[k]);
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(other.elements.size()); CHKERRXX(ierr);
#endif
    return *this;
  }
  node_linear_combination operator-(const node_linear_combination& other) const
  {
    node_linear_combination tmp(*this);
    tmp -= other;
    return tmp;
  }
  node_linear_combination operator+(const node_linear_combination& other) const
  {
    node_linear_combination tmp(*this);
    tmp += other;
    return tmp;
  }
} node_linear_combination;

inline node_linear_combination operator*(const double& alpha, const node_linear_combination &lc_tool) { return lc_tool*alpha; }

class quad_neighbor_nodes_of_node_t {
private:
public:
  p4est_nodes_t *nodes;

  p4est_locidx_t node_000;
  p4est_locidx_t node_m00_mm; p4est_locidx_t node_m00_pm;
  p4est_locidx_t node_p00_mm; p4est_locidx_t node_p00_pm;
  p4est_locidx_t node_0m0_mm; p4est_locidx_t node_0m0_pm;
  p4est_locidx_t node_0p0_mm; p4est_locidx_t node_0p0_pm;
#ifdef P4_TO_P8
  p4est_locidx_t node_m00_mp; p4est_locidx_t node_m00_pp;
  p4est_locidx_t node_p00_mp; p4est_locidx_t node_p00_pp;
  p4est_locidx_t node_0m0_mp; p4est_locidx_t node_0m0_pp;
  p4est_locidx_t node_0p0_mp; p4est_locidx_t node_0p0_pp;
  p4est_locidx_t node_00m_mm; p4est_locidx_t node_00m_pm;
  p4est_locidx_t node_00m_mp; p4est_locidx_t node_00m_pp;
  p4est_locidx_t node_00p_mm; p4est_locidx_t node_00p_pm;
  p4est_locidx_t node_00p_mp; p4est_locidx_t node_00p_pp;
#endif

  double d_m00; double d_m00_m0; double d_m00_p0;
  double d_p00; double d_p00_m0; double d_p00_p0;
  double d_0m0; double d_0m0_m0; double d_0m0_p0;
  double d_0p0; double d_0p0_m0; double d_0p0_p0;
#ifdef P4_TO_P8
  double d_00m, d_00p;
  double d_m00_0m; double d_m00_0p;
  double d_p00_0m; double d_p00_0p;
  double d_0m0_0m; double d_0m0_0p;
  double d_0p0_0m; double d_0p0_0p;
  double d_00m_m0; double d_00m_p0;
  double d_00m_0m; double d_00m_0p;
  double d_00p_m0; double d_00p_p0;
  double d_00p_0m; double d_00p_0p;
#endif
  double inverse_d_max;

  // pointers to the corresponding appropriate operators
  bool linear_interpolators_are_set, quadratic_interpolators_are_set, first_derivatives_are_set, second_derivatives_are_set;
  const node_linear_combination *linear_interpolator_000;
  const node_linear_combination *linear_interpolator_m00, *quadratic_interpolator_m00;
  const node_linear_combination *linear_interpolator_p00, *quadratic_interpolator_p00;
  const node_linear_combination *linear_interpolator_0m0, *quadratic_interpolator_0m0;
  const node_linear_combination *linear_interpolator_0p0, *quadratic_interpolator_0p0;
#ifdef P4_TO_P8
  const node_linear_combination *linear_interpolator_00m, *quadratic_interpolator_00m;
  const node_linear_combination *linear_interpolator_00p, *quadratic_interpolator_00p;
#endif
  const node_linear_combination *second_derivative_central[P4EST_DIM];
  const node_linear_combination *first_derivative_central[P4EST_DIM];

  quad_neighbor_nodes_of_node_t(): linear_interpolators_are_set(false), quadratic_interpolators_are_set(false),
    first_derivatives_are_set(false), second_derivatives_are_set(false),
    linear_interpolator_000(NULL),
    linear_interpolator_m00(NULL), quadratic_interpolator_m00(NULL),
    linear_interpolator_p00(NULL), quadratic_interpolator_p00(NULL),
    linear_interpolator_0m0(NULL), quadratic_interpolator_0m0(NULL),
    linear_interpolator_0p0(NULL), quadratic_interpolator_0p0(NULL)
#ifdef P4_TO_P8
  , linear_interpolator_00m(NULL), quadratic_interpolator_00m(NULL),
    linear_interpolator_00p(NULL), quadratic_interpolator_00p(NULL)
#endif
  {
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
    {
      second_derivative_central[der]  = NULL;
      first_derivative_central[der]   = NULL;
    }
  }

  ~quad_neighbor_nodes_of_node_t()
  {
    if(linear_interpolators_are_set)
    {
      P4EST_ASSERT(linear_interpolator_000!=NULL); delete linear_interpolator_000;
      P4EST_ASSERT(linear_interpolator_m00!=NULL); delete linear_interpolator_m00;
      P4EST_ASSERT(linear_interpolator_p00!=NULL); delete linear_interpolator_p00;
      P4EST_ASSERT(linear_interpolator_0m0!=NULL); delete linear_interpolator_0m0;
      P4EST_ASSERT(linear_interpolator_0p0!=NULL); delete linear_interpolator_0p0;
#ifdef P4_TO_P8
      P4EST_ASSERT(linear_interpolator_00m!=NULL); delete linear_interpolator_00m;
      P4EST_ASSERT(linear_interpolator_00p!=NULL); delete linear_interpolator_00p;
#endif
    }
    if(quadratic_interpolators_are_set)
    {
      P4EST_ASSERT(quadratic_interpolator_m00!=NULL); delete quadratic_interpolator_m00;
      P4EST_ASSERT(quadratic_interpolator_p00!=NULL); delete quadratic_interpolator_p00;
      P4EST_ASSERT(quadratic_interpolator_0m0!=NULL); delete quadratic_interpolator_0m0;
      P4EST_ASSERT(quadratic_interpolator_0p0!=NULL); delete quadratic_interpolator_0p0;
#ifdef P4_TO_P8
      P4EST_ASSERT(quadratic_interpolator_00m!=NULL); delete quadratic_interpolator_00m;
      P4EST_ASSERT(quadratic_interpolator_00p!=NULL); delete quadratic_interpolator_00p;
#endif
    }
    if(first_derivatives_are_set || second_derivatives_are_set)
      for (unsigned char der = 0; der < P4EST_DIM; ++der){
        if(second_derivatives_are_set){
          P4EST_ASSERT(second_derivative_central[der]!=NULL); delete second_derivative_central[der]; }
        if(first_derivatives_are_set){
          P4EST_ASSERT(first_derivative_central[der]!=NULL); delete first_derivative_central[der]; }
      }
  }

  template<typename T> inline T initialize_accumulator(const unsigned char &n_elem) const;
  template<typename T> inline void add_to_accumulator(T& accumulator, const p4est_locidx_t& node_ix, const double& weight, const double *f, const unsigned int &block_size, const unsigned int &component) const;
#if __cplusplus >= 201103L
  template<typename T> inline void shrink_accumulator(T& accumulator) const;
#endif
#ifdef P4_TO_P8
  template<typename T> inline T linear_interpolator(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm, const p4est_locidx_t &node_idx_mp, const p4est_locidx_t &node_idx_pp,
                                                    const double &d_m0, const double &d_p0, const double &d_0m, const double &d_0p,
                                                    const double *f=NULL, const unsigned int &block_size=1, const unsigned int &component=0) const;
#else
  template<typename T> inline T linear_interpolator(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm,
                                                    const double &d_m0, const double &d_p0,
                                                    const double *f=NULL, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  template<typename T=double> T f_000_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T=double> T f_m00_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T=double> T f_p00_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T=double> T f_0m0_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T=double> T f_0p0_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  template<typename T=double> T f_00m_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T=double> T f_00p_linear(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void linearly_interpolated_neighbors(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0, T &f_00m, T &f_00p, const unsigned int &block_size, const unsigned int &component) const;
#else
  template<typename T> void linearly_interpolated_neighbors(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0, const unsigned int &block_size, const unsigned int &component) const;
#endif
  void set_and_store_linear_interpolators();
  void free_linear_interpolators();

  // second-derivatives-related and third-order-interpolation-related procedures
  inline bool naive_dx_needs_yy_correction(double& yy_correction_weight_to_naive_Dx) const
  {
    yy_correction_weight_to_naive_Dx = dd_correction_weight_to_naive_first_derivative(d_m00, d_p00, d_m00_m0, d_m00_p0, d_p00_m0, d_p00_p0);
    return !check_if_zero(yy_correction_weight_to_naive_Dx*inverse_d_max);
  }
#ifdef P4_TO_P8
  inline bool naive_dx_needs_zz_correction(double& zz_correction_weight_to_naive_Dx) const
  {
    zz_correction_weight_to_naive_Dx = dd_correction_weight_to_naive_first_derivative(d_m00, d_p00, d_m00_0m, d_m00_0p, d_p00_0m, d_p00_0p);
    return !check_if_zero(zz_correction_weight_to_naive_Dx*inverse_d_max);
  }
#endif
  inline bool naive_dy_needs_xx_correction(double& xx_correction_weight_to_naive_Dy) const
  {
    xx_correction_weight_to_naive_Dy = dd_correction_weight_to_naive_first_derivative(d_0m0, d_0p0, d_0m0_m0, d_0m0_p0, d_0p0_m0, d_0p0_p0);
    return !check_if_zero(xx_correction_weight_to_naive_Dy*inverse_d_max);
  }
#ifdef P4_TO_P8
  inline bool naive_dy_needs_zz_correction(double& zz_correction_weight_to_naive_Dy) const
  {
    zz_correction_weight_to_naive_Dy = dd_correction_weight_to_naive_first_derivative(d_0m0, d_0p0, d_0m0_0m, d_0m0_0p, d_0p0_0m, d_0p0_0p);
    return !check_if_zero(zz_correction_weight_to_naive_Dy*inverse_d_max);
  }
  inline bool naive_dz_needs_xx_correction(double& xx_correction_weight_to_naive_Dz) const
  {
    xx_correction_weight_to_naive_Dz = dd_correction_weight_to_naive_first_derivative(d_00m, d_00p, d_00m_m0, d_00m_p0, d_00p_m0, d_00p_p0);
    return !check_if_zero(xx_correction_weight_to_naive_Dz*inverse_d_max);
  }
  inline bool naive_dz_needs_yy_correction(double& yy_correction_weight_to_naive_Dz) const
  {
    yy_correction_weight_to_naive_Dz = dd_correction_weight_to_naive_first_derivative(d_00m, d_00p, d_00m_0m, d_00m_0p, d_00p_0m, d_00p_0p);
    return !check_if_zero(yy_correction_weight_to_naive_Dz*inverse_d_max);
  }
#endif


#ifdef P4_TO_P8
  template<typename T> void correct_naive_second_derivatives(const T naive_Dxx, const T naive_Dyy, const T naive_Dzz, T &Dxx, T &Dyy, T &Dzz) const;
  template<typename T> void laplace_core(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0, T &f_00m, T &f_00p,
                                         T &fxx, T &fyy, T &fzz, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void laplace(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0, T &f_00m, T &f_00p,
                                    T &fxx, T &fyy, T &fzz, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> inline void laplace(const double *f, T &fxx, T &fyy, T &fzz, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
    T tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0, tmp_00m, tmp_00p;
    laplace<T>(f, tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0, tmp_00m, tmp_00p, fxx, fyy, fzz, block_size, component);
    return;
  }
#else
  template<typename T> void correct_naive_second_derivatives(const T naive_Dxx, const T naive_Dyy, T &Dxx, T &Dyy) const;
  template<typename T> void laplace_core(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0,
                                         T &fxx, T &fyy, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void laplace(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0,
                                    T &fxx, T &fyy, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> inline void laplace(const double *f, T &fxx, T &fyy, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
    T tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0;
    laplace<T>(f, tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0, fxx, fyy, block_size, component);
    return;
  }
#endif
  void set_and_store_second_derivative_operators();
  void free_second_derivative_operators();

  template<typename T> void ngbd_with_quadratic_interpolation(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0,
                                                            #ifdef P4_TO_P8
                                                              T &f_00m, T &f_00p,
                                                            #endif
                                                              const unsigned int &block_size=1, const unsigned int &component=0) const;

  template<typename T> void ngbd_with_quadratic_interpolation_core(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0,
                                                                 #ifdef P4_TO_P8
                                                                   T &f_00m, T &f_00p,
                                                                 #endif
                                                                   const unsigned int &block_size=1, const unsigned int &component=0) const;

  void x_ngbd_with_quadratic_interpolation(const double *f, double &f_m00, double &f_000, double &f_p00, const unsigned int &block_size=1, const unsigned int &component=0) const;
  void y_ngbd_with_quadratic_interpolation(const double *f, double &f_0m0, double &f_000, double &f_0p0, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  void z_ngbd_with_quadratic_interpolation(const double *f, double &f_00m, double &f_000, double &f_00p, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  void set_and_store_quadratic_interpolators();

  /* second derivatives */
  double dxx_central(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dyy_central(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  double dzz_central(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  inline double dd_central (const unsigned short& der, const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dxx_central(f, block_size, component) : ((der == dir::y)? dyy_central(f, block_size, component) : dzz_central(f, block_size, component)));
#else
    return ((der == dir::x)? dxx_central(f, block_size, component) : dyy_central(f, block_size, component));
#endif
  }
  // first-derivatives-related procedures
#ifdef P4_TO_P8
  template<typename T> void correct_naive_first_derivatives(const double*f, const T &naive_Dx, const T &naive_Dy, const T &naive_Dz, T &Dx, T &Dy, T &Dz, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void gradient_core(const double *f, T &fx, T &fy, T &fz, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void gradient(const double *f, T &fx, T &fy, T &fz, const unsigned int &block_size=1, const unsigned int &component=0) const;
#else
  template<typename T> void correct_naive_first_derivatives(const double*f, const T &naive_Dx, const T &naive_Dy, T &Dx, T &Dy, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void gradient_core(const double *f, T &fx, T &fy, const unsigned int &block_size=1, const unsigned int &component=0) const;
  template<typename T> void gradient(const double *f, T &fx, T &fy, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  void set_and_store_first_derivative_operators();


  /* first derivatives */
  double dx_central(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_central(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  double dz_central(const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  inline double d_central (const unsigned short& der, const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_central(f, block_size, component) : ((der == dir::y)? dy_central(f, block_size, component) : dz_central(f, block_size, component)));
#else
    return ((der == dir::x)? dx_central(f, block_size, component) : dy_central(f, block_size, component));
#endif
  }

  // biased-first-derivative-related procedures
  // based on linear-interpolation neighbors
  double dx_forward_linear (const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dx_backward_linear (const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_forward_linear (const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_backward_linear (const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  double dz_forward_linear (const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dz_backward_linear (const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  inline double d_forward_linear(const unsigned short& der, const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_linear(f, block_size, component) : ((der == dir::y)? dy_forward_linear(f, block_size, component) : dz_forward_linear(f, block_size, component)));
#else
    return ((der == dir::x)? dx_forward_linear(f, block_size, component) : dy_forward_linear(f, block_size, component));
#endif
  }
  inline double d_backward_linear(const unsigned short& der, const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_linear(f, block_size, component) : ((der == dir::y)? dy_backward_linear(f, block_size, component) : dz_backward_linear(f, block_size, component)));
#else
    return ((der == dir::x)? dx_backward_linear(f, block_size, component) : dy_backward_linear(f, block_size, component));
#endif
  }
  // based on quadratic-interpolation neighbors
  inline double d_backward_quadratic(const double &f_0_quad, const double &f_m_quad, const double &d_m, const double &f_dd_0, const double &f_dd_m) const
  {
    return (backward_derivative<double>(f_0_quad, f_m_quad, d_m) + 0.5*d_m*MINMOD(f_dd_0, f_dd_m));
  }
  inline double d_forward_quadratic(const double &f_p_quad, const double &f_0_quad, const double &d_p, const double &f_dd_0, const double &f_dd_p) const
  {
    return (forward_derivative<double>(f_p_quad, f_0_quad, d_p) - 0.5*d_p*MINMOD(f_dd_0, f_dd_p));
  }
  double dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dx_forward_quadratic (const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_forward_quadratic (const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  double dz_backward_quadratic(const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dz_forward_quadratic (const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  inline double d_backward_quadratic(const unsigned short& der, const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_quadratic(f, neighbors, block_size, component) : ((der == dir::y)? dy_backward_quadratic(f, neighbors, block_size, component) : dz_backward_quadratic(f, neighbors, block_size, component)));
#else
    return ((der == dir::x)? dx_backward_quadratic(f, neighbors, block_size, component) : dy_backward_quadratic(f, neighbors, block_size, component));
#endif
  }
  inline double d_forward_quadratic(const unsigned short& der, const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_quadratic(f, neighbors, block_size, component) : ((der == dir::y)? dy_forward_quadratic(f, neighbors, block_size, component) : dz_forward_quadratic(f, neighbors, block_size, component)));
#else
    return ((der == dir::x)? dx_forward_quadratic(f, neighbors, block_size, component) : dy_forward_quadratic(f, neighbors, block_size, component));
#endif
  }

  // VERY IMPORTANT NOTE: in the following, we assume fxx, fyy and fzz to have the same block structure as f!
  double dx_backward_quadratic(const double *f, const double *fxx, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dx_forward_quadratic (const double *f, const double *fxx, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_backward_quadratic(const double *f, const double *fyy, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dy_forward_quadratic (const double *f, const double *fyy, const unsigned int &block_size=1, const unsigned int &component=0) const;
#ifdef P4_TO_P8
  double dz_backward_quadratic(const double *f, const double *fzz, const unsigned int &block_size=1, const unsigned int &component=0) const;
  double dz_forward_quadratic (const double *f, const double *fzz, const unsigned int &block_size=1, const unsigned int &component=0) const;
#endif
  inline double d_backward_quadratic(const unsigned short& der, const double *f, const double *fderder, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_quadratic(f, fderder, block_size, component) : ((der == dir::y)? dy_backward_quadratic(f, fderder, block_size, component) : dz_backward_quadratic(f, fderder, block_size, component)));
#else
    return ((der == dir::x)? dx_backward_quadratic(f, fderder, block_size, component) : dy_backward_quadratic(f, fderder, block_size, component));
#endif
  }
  inline double d_forward_quadratic(const unsigned short& der, const double *f, const double *fderder, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_quadratic(f, fderder, block_size, component) : ((der == dir::y)? dy_forward_quadratic(f, fderder, block_size, component) : dz_forward_quadratic(f, fderder, block_size, component)));
#else
    return ((der == dir::x)? dx_forward_quadratic(f, fderder, block_size, component) : dy_forward_quadratic(f, fderder, block_size, component));
#endif
  }

  void print_debug(FILE* pFile) const
  {
    p4est_indep_t *n_m00_mm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_m00_mm);
    p4est_indep_t *n_m00_pm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_m00_pm);
    p4est_indep_t *n_p00_mm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_p00_mm);
    p4est_indep_t *n_p00_pm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_p00_pm);
    p4est_indep_t *n_0m0_mm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0m0_mm);
    p4est_indep_t *n_0m0_pm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0m0_pm);
    p4est_indep_t *n_0p0_mm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0p0_mm);
    p4est_indep_t *n_0p0_pm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0p0_pm);
#ifdef P4_TO_P8
    p4est_indep_t *n_m00_mp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_m00_mp);
    p4est_indep_t *n_m00_pp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_m00_pp);
    p4est_indep_t *n_p00_mp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_p00_mp);
    p4est_indep_t *n_p00_pp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_p00_pp);
    p4est_indep_t *n_0m0_mp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0m0_mp);
    p4est_indep_t *n_0m0_pp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0m0_pp);
    p4est_indep_t *n_0p0_mp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0p0_mp);
    p4est_indep_t *n_0p0_pp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_0p0_pp);

    p4est_indep_t *n_00m_mm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00m_mp);
    p4est_indep_t *n_00m_pm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00m_pp);
    p4est_indep_t *n_00p_mm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00p_mp);
    p4est_indep_t *n_00p_pm = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00p_pp);
    p4est_indep_t *n_00m_mp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00m_mp);
    p4est_indep_t *n_00m_pp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00m_pp);
    p4est_indep_t *n_00p_mp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00p_mp);
    p4est_indep_t *n_00p_pp = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,node_00p_pp);
#endif
    fprintf(pFile,"------------- Printing QNNN for node %d ------------\n",node_000);
    fprintf(pFile,"node_m00_mm : %d - ( %f , %f )  -  %f\n",node_m00_mm, n_m00_mm->x / (double) P4EST_ROOT_LEN,n_m00_mm->y / (double) P4EST_ROOT_LEN,d_m00_m0);
    fprintf(pFile,"node_m00_pm : %d - ( %f , %f )  -  %f\n",node_m00_pm, n_m00_pm->x / (double) P4EST_ROOT_LEN,n_m00_pm->y / (double) P4EST_ROOT_LEN,d_m00_p0);
#ifdef P4_TO_P8
    fprintf(pFile,"node_m00_mp : %d - ( %f , %f )  -  %f\n",node_m00_mp, n_m00_mp->x / (double) P4EST_ROOT_LEN,n_m00_mp->y / (double) P4EST_ROOT_LEN,d_m00_0m);
    fprintf(pFile,"node_m00_pp : %d - ( %f , %f )  -  %f\n",node_m00_pp, n_m00_pp->x / (double) P4EST_ROOT_LEN,n_m00_pp->y / (double) P4EST_ROOT_LEN,d_m00_0p);
#endif
    fprintf(pFile,"node_p00_mm : %d - ( %f , %f )  -  %f\n",node_p00_mm, n_p00_mm->x / (double) P4EST_ROOT_LEN,n_p00_mm->y / (double) P4EST_ROOT_LEN,d_p00_m0);
    fprintf(pFile,"node_p00_pm : %d - ( %f , %f )  -  %f\n",node_p00_pm, n_p00_pm->x / (double) P4EST_ROOT_LEN,n_p00_pm->y / (double) P4EST_ROOT_LEN,d_p00_p0);
#ifdef P4_TO_P8
    fprintf(pFile,"node_p00_mp : %d - ( %f , %f )  -  %f\n",node_p00_mp, n_p00_mp->x / (double) P4EST_ROOT_LEN,n_p00_mp->y / (double) P4EST_ROOT_LEN,d_p00_0m);
    fprintf(pFile,"node_p00_pp : %d - ( %f , %f )  -  %f\n",node_p00_pp, n_p00_pp->x / (double) P4EST_ROOT_LEN,n_p00_pp->y / (double) P4EST_ROOT_LEN,d_p00_0p);
#endif
    fprintf(pFile,"node_0m0_mm : %d - ( %f , %f )  -  %f\n",node_0m0_mm, n_0m0_mm->x / (double) P4EST_ROOT_LEN,n_0m0_mm->y / (double) P4EST_ROOT_LEN,d_0m0_m0);
    fprintf(pFile,"node_0m0_pm : %d - ( %f , %f )  -  %f\n",node_0m0_pm, n_0m0_pm->x / (double) P4EST_ROOT_LEN,n_0m0_pm->y / (double) P4EST_ROOT_LEN,d_0m0_p0);
#ifdef P4_TO_P8
    fprintf(pFile,"node_0m0_mp : %d - ( %f , %f )  -  %f\n",node_0m0_mp, n_0m0_mp->x / (double) P4EST_ROOT_LEN,n_0m0_mp->y / (double) P4EST_ROOT_LEN,d_0m0_0m);
    fprintf(pFile,"node_0m0_pp : %d - ( %f , %f )  -  %f\n",node_0m0_pp, n_0m0_pp->x / (double) P4EST_ROOT_LEN,n_0m0_pp->y / (double) P4EST_ROOT_LEN,d_0m0_0p);
#endif
    fprintf(pFile,"node_0p0_mm : %d - ( %f , %f )  -  %f\n",node_0p0_mm, n_0p0_mm->x / (double) P4EST_ROOT_LEN,n_0p0_mm->y / (double) P4EST_ROOT_LEN,d_0p0_m0);
    fprintf(pFile,"node_0p0_pm : %d - ( %f , %f )  -  %f\n",node_0p0_pm, n_0p0_pm->x / (double) P4EST_ROOT_LEN,n_0p0_pm->y / (double) P4EST_ROOT_LEN,d_0p0_p0);
#ifdef P4_TO_P8
    fprintf(pFile,"node_0p0_mp : %d - ( %f , %f )  -  %f\n",node_0p0_mp, n_0p0_mp->x / (double) P4EST_ROOT_LEN,n_0p0_mp->y / (double) P4EST_ROOT_LEN,d_0p0_0m);
    fprintf(pFile,"node_0p0_pp : %d - ( %f , %f )  -  %f\n",node_0p0_pp, n_0p0_pp->x / (double) P4EST_ROOT_LEN,n_0p0_pp->y / (double) P4EST_ROOT_LEN,d_0p0_0p);
#endif
#ifdef P4_TO_P8
    fprintf(pFile,"node_00m_mm : %d - ( %f , %f )  -  %f\n",node_00m_mm, n_00m_mm->x / (double) P4EST_ROOT_LEN,n_00m_mm->y / (double) P4EST_ROOT_LEN,d_00m_m0);
    fprintf(pFile,"node_00m_pm : %d - ( %f , %f )  -  %f\n",node_00m_pm, n_00m_pm->x / (double) P4EST_ROOT_LEN,n_00m_pm->y / (double) P4EST_ROOT_LEN,d_00m_p0);
    fprintf(pFile,"node_00m_mp : %d - ( %f , %f )  -  %f\n",node_00m_mp, n_00m_mp->x / (double) P4EST_ROOT_LEN,n_00m_mp->y / (double) P4EST_ROOT_LEN,d_00m_0m);
    fprintf(pFile,"node_00m_pp : %d - ( %f , %f )  -  %f\n",node_00m_pp, n_00m_pp->x / (double) P4EST_ROOT_LEN,n_00m_pp->y / (double) P4EST_ROOT_LEN,d_00m_0p);
    fprintf(pFile,"node_00p_mm : %d - ( %f , %f )  -  %f\n",node_00p_mm, n_00p_mm->x / (double) P4EST_ROOT_LEN,n_00p_mm->y / (double) P4EST_ROOT_LEN,d_00p_m0);
    fprintf(pFile,"node_00p_pm : %d - ( %f , %f )  -  %f\n",node_00p_pm, n_00p_pm->x / (double) P4EST_ROOT_LEN,n_00p_pm->y / (double) P4EST_ROOT_LEN,d_00p_p0);
    fprintf(pFile,"node_00p_mp : %d - ( %f , %f )  -  %f\n",node_00p_mp, n_00p_mp->x / (double) P4EST_ROOT_LEN,n_00p_mp->y / (double) P4EST_ROOT_LEN,d_00p_0m);
    fprintf(pFile,"node_00p_pp : %d - ( %f , %f )  -  %f\n",node_00p_pp, n_00p_pp->x / (double) P4EST_ROOT_LEN,n_00p_pp->y / (double) P4EST_ROOT_LEN,d_00p_0p);
#endif
#ifdef P4_TO_P8
    fprintf(pFile,"d_m00 : %f\nd_p00 : %f\nd_0m0 : %f\nd_0p0 : %f\nd_00m : %f\nd_00p : %f\n",d_m00,d_p00,d_0m0,d_0p0,d_00m,d_00p);
#else
    fprintf(pFile,"d_m00 : %f\nd_p00 : %f\nd_0m0 : %f\nd_0p0 : %f\n",d_m00,d_p00,d_0m0,d_0p0);
#endif
  }
};

template<> inline double quad_neighbor_nodes_of_node_t::initialize_accumulator<double>(const unsigned char &) const { return 0.0;}
template<> inline node_linear_combination quad_neighbor_nodes_of_node_t::initialize_accumulator<node_linear_combination>(const unsigned char &n_elem) const
{
  node_linear_combination empty_one;
  empty_one.elements.reserve(n_elem);
  return empty_one;
}
template<> inline void quad_neighbor_nodes_of_node_t::add_to_accumulator<double>(double& accumulator, const p4est_locidx_t& node_ix, const double& weight, const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  accumulator += weight*f[bs*node_ix+comp];
  return;
}
template<> inline void quad_neighbor_nodes_of_node_t::add_to_accumulator<node_linear_combination>(node_linear_combination& accumulator, const p4est_locidx_t& node_ix, const double& weight, const double *, const unsigned int &, const unsigned int &) const
{
  accumulator += node_interpolation_weight(node_ix, weight);
  return;
}
#if __cplusplus >= 201103L
template<> inline void quad_neighbor_nodes_of_node_t::shrink_accumulator(double&) const { return; }
template<> inline void quad_neighbor_nodes_of_node_t::shrink_accumulator(node_linear_combination& lc_tool) const { lc_tool.elements.shrink_to_fit(); return; }
#endif


#endif /* !MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H */
