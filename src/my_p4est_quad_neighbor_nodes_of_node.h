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
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_laplace;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_gradient;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

typedef struct node_interpolation_weight{
  node_interpolation_weight(const p4est_locidx_t & idx, const double& ww_): weight(ww_), node_idx(idx) {}
  double weight;
  p4est_locidx_t node_idx;
  inline bool operator <(const node_interpolation_weight& other) const { return (node_idx < other.node_idx); }
  inline node_interpolation_weight operator-() const { node_interpolation_weight copy(*this); copy.weight *=-1.0; return copy; }
  // inline node_interpolation_weight operator*(const double &alpha) const { node_interpolation_weight copy(*this); copy.weight *=alpha; return copy; }
} node_interpolation_weight;

class my_p4est_node_neighbors_t;
typedef struct node_linear_combination
{
  node_linear_combination(size_t nelem=0) { elements.reserve(nelem); }
  std::vector<node_interpolation_weight> elements;
  inline void calculate_all_components(const double *node_sampled_field[], double* serialized_results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    unsigned int bs_node_idx = bs*elements[0].node_idx;
    unsigned int bsk;
    for (unsigned int k = 0; k < n_arrays; ++k){
      bsk = bs*k;
      for (unsigned int comp = 0; comp < bs; ++comp)
        serialized_results[bsk + comp] = elements[0].weight*node_sampled_field[k][bs_node_idx + comp];
    }
    for (size_t nn = 1; nn < elements.size(); ++nn){
      bs_node_idx = bs*elements[nn].node_idx;
      for (unsigned int k = 0; k < n_arrays; ++k){
        bsk = bs*k;
        for (unsigned int comp = 0; comp < bs; ++comp)
          serialized_results[bsk + comp] += elements[nn].weight*node_sampled_field[k][bs_node_idx + comp];
      }
    }
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(narrays*bs*(4.0*elements.size()-1)+elements.size()*n_arrays+elements.size()); CHKERRXX(ierr_flops);
#endif
    return;
  }

  inline void calculate_component(const double *node_sampled_field[], double* serialized_results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    unsigned int bs_node_idx = bs*elements[0].node_idx;
    for (unsigned int k = 0; k < n_arrays; ++k)
      serialized_results[k] = elements[0].weight*node_sampled_field[k][bs_node_idx + comp];
    for (size_t nn = 1; nn < elements.size(); ++nn){
      bs_node_idx = bs*elements[nn].node_idx;
      for (unsigned int k = 0; k < n_arrays; ++k)
        serialized_results[k] += elements[nn].weight*node_sampled_field[k][bs_node_idx + comp];
    }
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(narrays*(3.0*elements.size()-1)+elements.size()); CHKERRXX(ierr_flops);
#endif
    return;
  }

  inline void calculate(const double *node_sampled_field[], double* serialized_results, const unsigned int &n_arrays) const
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      serialized_results[k] = elements[0].weight*node_sampled_field[k][elements[0].node_idx];
    for (size_t nn = 1; nn < elements.size(); ++nn)
      for (unsigned int k = 0; k < n_arrays; ++k)
        serialized_results[k] += elements[nn].weight*node_sampled_field[k][elements[nn].node_idx];
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(n_arrays*(2.0*elements.size()-1)); CHKERRXX(ierr_flops);
#endif
    return;
  }

  double calculate_dd (unsigned char der, const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors) const;
  inline double calculate_dxx(const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors) const
  {
    return calculate_dd(dir::x, node_sample_field, neighbors);
  }
  inline double calculate_dyy(const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors) const
  {
    return calculate_dd(dir::y, node_sample_field, neighbors);
  }
#ifdef P4_TO_P8
  inline double calculate_dzz(const double *node_sample_field, const my_p4est_node_neighbors_t& neighbors) const
  {
    return calculate_dd(dir::z, node_sample_field, neighbors);
  }
#endif

  /*
  inline void add_terms(const node_interpolation_weight& node_weight, const double &factor) {
#ifdef P4EST_DEBUG
    const size_t init_size = elements.size();
#endif
    // we add an element to the vector if it is not in there yet,
    // otherwise, we update the weight of the already existing vector
    std::vector<node_interpolation_weight>::iterator position = std::lower_bound(this->elements.begin(), this->elements.end(), node_weight);
    P4EST_ASSERT((position == this->elements.end()) || (position->node_idx >= node_weight.node_idx));
    if((position != this->elements.end()) && (position->node_idx == node_weight.node_idx))
      position->weight += factor*node_weight.weight;
    else
      elements.insert(position, node_weight*factor);
#ifdef P4EST_DEBUG
    P4EST_ASSERT((elements.size() >= init_size) || (elements.size() <= init_size+1));
    bool check_if_still_sorted_and_unique_indices = true;
    for (size_t k = 1; check_if_still_sorted_and_unique_indices && (k < elements.size()); ++k)
      check_if_still_sorted_and_unique_indices = check_if_still_sorted_and_unique_indices && (elements[k].node_idx > elements[k-1].node_idx);
    P4EST_ASSERT(check_if_still_sorted_and_unique_indices);
#endif
    return;
  }

  inline void add_terms(const node_linear_combination& lc_tool, const double &factor) {
    elements.reserve(elements.size()+lc_tool.elements.size());
    for (size_t k = 0; k < lc_tool.elements.size(); ++k)
      add_terms(lc_tool.elements[k], factor);
    return;
  }
  */
} node_linear_combination;

const double zero_threshold_qnnn = EPS;

class quad_neighbor_nodes_of_node_t {
private:
  static inline bool check_if_zero(const double& value) { return (fabs(value) < zero_threshold_qnnn); } // needs a nondimensional argument, otherwise it's meaningless
  // very elementary operations in the most synthetic forms
  inline double dd_correction_weight_to_naive_first_derivative(const double &d_m, const double &d_p, const double &d_m_m, const double &d_m_p, const double &d_p_m, const double &d_p_p) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(10); CHKERRXX(ierr_flops);
#endif
    return 0.5*(d_p_m*d_p_p*d_m/d_p - d_m_m*d_m_p*d_p/d_m)/(d_m+d_p);
  }
  inline double central_derivative(const double &fp, const double &f0, const double &fm, const double &dp, const double &dm) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(9); CHKERRXX(ierr_flops);
#endif
    return ((fp-f0)*dm/dp + (f0-fm)*dp/dm)/(dp+dm);
  }
  inline double forward_derivative(const double &fp, const double &f0, const double &dp) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(2); CHKERRXX(ierr_flops);
#endif
    return (fp-f0)/dp;
  }
  inline double backward_derivative(const double &f0, const double &fm, const double &dm) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(2); CHKERRXX(ierr_flops);
#endif
    return (f0-fm)/dm;
  }
  inline double central_second_derivative(const double &fp, const double &f0, const double &fm, const double &dp, const double &dm) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr_flops = PetscLogFlops(8); CHKERRXX(ierr_flops);
#endif
    return ((fp-f0)/dp + (fm-f0)/dm)*2./(dp+dm);
  }

  inline void get_linear_interpolation_weights(const double &d_m0, const double &d_p0 ONLY3D(COMMA const double &d_0m COMMA const double &d_0p),
                                               double &w_mm, double &w_pm ONLY3D(COMMA double &w_mp COMMA double &w_pp), double &normalization) const
  {
    /* (f[node_idx_mm]*d_p0*d_0p +
         *  f[node_idx_mp]*d_p0*d_0m +
         *  f[node_idx_pm]*d_m0*d_0p +
         *  f[node_idx_pp]*d_m0*d_0m )/(d_m0 + d_p0)/(d_0m + d_0p);
         */
    w_mm = d_p0 ONLY3D(*d_0p);
    w_pm = d_m0 ONLY3D(*d_0p);
#ifdef P4_TO_P8
    w_mp = d_p0*d_0m;
    w_pp = d_m0*d_0m;
#endif
    normalization = (d_m0 + d_p0)ONLY3D(*(d_0m + d_0p));
#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    PetscErrorCode ierr_flops = PetscLogFlops(7); CHKERRXX(ierr_flops);
#else
    PetscErrorCode ierr_flops = PetscLogFlops(1); CHKERRXX(ierr_flops);
#endif
#endif
    return;
  }

  inline void interpolate_linearly_all_components(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm ONLY3D(COMMA const p4est_locidx_t &node_idx_mp COMMA const p4est_locidx_t &node_idx_pp),
                                                  const double &d_m0, const double &d_p0 ONLY3D(COMMA const double &d_0m COMMA const double &d_0p),
                                                  const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    double normalization, w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp);
    get_linear_interpolation_weights(d_m0, d_p0 ONLY3D(COMMA d_0m COMMA d_0p), w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp), normalization);
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        results[kbs + comp] = (ONLY3D(f[k][bs*node_idx_pp + comp]*w_pp +) f[k][bs*node_idx_pm + comp]*w_pm ONLY3D(+ f[k][bs*node_idx_mp + comp]*w_mp) + f[k][bs*node_idx_mm + comp]*w_mm)/normalization;
    }
#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    PetscErrorCode ierr_flops = PetscLogFlops(n_arrays*(1 + 17*bs)); CHKERRXX(ierr_flops);
#else
    PetscErrorCode ierr_flops = PetscLogFlops(n_arrays*(1 + 9*bs)); CHKERRXX(ierr_flops);
#endif
#endif
    return;
  }

  inline void interpolate_linearly_component(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm ONLY3D(COMMA const p4est_locidx_t &node_idx_mp COMMA const p4est_locidx_t &node_idx_pp),
                                             const double &d_m0, const double &d_p0 ONLY3D(COMMA const double &d_0m COMMA const double &d_0p),
                                             const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    double normalization, w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp);
    get_linear_interpolation_weights(d_m0, d_p0 ONLY3D(COMMA d_0m COMMA d_0p), w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp), normalization);
    for (unsigned int k = 0; k < n_arrays; ++k)
      results[k] = (ONLY3D(f[k][bs*node_idx_pp + comp]*w_pp +) f[k][bs*node_idx_pm + comp]*w_pm ONLY3D(+ f[k][bs*node_idx_mp + comp]*w_mp) + f[k][bs*node_idx_mm + comp]*w_mm)/normalization;
#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    PetscErrorCode ierr_flops = PetscLogFlops(16*n_arrays); CHKERRXX(ierr_flops);
#else
    PetscErrorCode ierr_flops = PetscLogFlops(8*n_arrays); CHKERRXX(ierr_flops);
#endif
#endif
    return;
  }

  inline void interpolate_linearly(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm ONLY3D(COMMA const p4est_locidx_t &node_idx_mp COMMA const p4est_locidx_t &node_idx_pp),
                                   const double &d_m0, const double &d_p0 ONLY3D(COMMA const double &d_0m COMMA const double &d_0p),
                                   const double *f[], double *results, const unsigned int &n_arrays) const
  {
    double normalization, w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp);
    get_linear_interpolation_weights(d_m0, d_p0 ONLY3D(COMMA d_0m COMMA d_0p), w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp), normalization);
    for (unsigned int k = 0; k < n_arrays; ++k)
      results[k] = (ONLY3D(f[k][node_idx_pp]*w_pp +) f[k][node_idx_pm]*w_pm ONLY3D(+ f[k][node_idx_mp]*w_mp) + f[k][node_idx_mm]*w_mm)/normalization;
#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    PetscErrorCode ierr_flops = PetscLogFlops(8*n_arrays); CHKERRXX(ierr_flops);
#else
    PetscErrorCode ierr_flops = PetscLogFlops(4*n_arrays); CHKERRXX(ierr_flops);
#endif
#endif
    return;
  }

  inline void get_linear_interpolator(node_linear_combination &lc_tool, const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm ONLY3D(COMMA const p4est_locidx_t &node_idx_mp COMMA const p4est_locidx_t &node_idx_pp),
                                      const double &d_m0, const double &d_p0 ONLY3D(COMMA const double &d_0m COMMA const double &d_0p)) const
  {
    P4EST_ASSERT(lc_tool.elements.size()==0);
    double normalization, w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp);
    get_linear_interpolation_weights(d_m0, d_p0 ONLY3D(COMMA d_0m COMMA d_0p), w_mm, w_pm ONLY3D(COMMA w_mp COMMA w_pp), normalization);
    w_pm /= normalization; if(!check_if_zero(w_pm)) { lc_tool.elements.push_back(node_interpolation_weight(node_idx_pm, w_pm)); }
    w_mm /= normalization; if(!check_if_zero(w_mm)) { lc_tool.elements.push_back(node_interpolation_weight(node_idx_mm, w_mm)); }
#ifdef P4_TO_P8
    w_pp /= normalization; if(!check_if_zero(w_pp)) { lc_tool.elements.push_back(node_interpolation_weight(node_idx_pp, w_pp)); }
    w_mp /= normalization; if(!check_if_zero(w_mp)) { lc_tool.elements.push_back(node_interpolation_weight(node_idx_mp, w_mp)); }
#endif
    lc_tool.elements.shrink_to_fit();
    return;
  }

  inline void f_m00_linear_all_components(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    interpolate_linearly_all_components(node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p), f, results, n_arrays, bs);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_m00].calculate_all_components(f, results, n_arrays, bs) : interpolate_linearly_all_components(node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p), f, results, n_arrays, bs);
    */
    return;
  }
  inline void f_m00_linear_component(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    interpolate_linearly_component(node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p), f, results, n_arrays, bs, comp);

    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_m00].calculate_component(f, results, n_arrays, bs, comp) : interpolate_linearly_component(node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p), f, results, n_arrays, bs, comp);
    */
    return;
  }
  inline void f_m00_linear(const double *f[], double *results, const unsigned int &n_arrays) const
  {
    interpolate_linearly(node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p), f, results, n_arrays);

    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_m00].calculate(f, results, n_arrays) : interpolate_linearly(node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p), f, results, n_arrays);
    */
    return;
  }
  inline void f_p00_linear_all_components(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    interpolate_linearly_all_components(node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0 ONLY3D(COMMA d_p00_0m COMMA d_p00_0p), f, results, n_arrays, bs);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_p00].calculate_all_components(f, results, n_arrays, bs) : interpolate_linearly_all_components(node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p, f, results, n_arrays, bs);
    */
    return;
  }
  inline void f_p00_linear_component(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    interpolate_linearly_component(node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0 ONLY3D(COMMA d_p00_0m COMMA d_p00_0p), f, results, n_arrays, bs, comp);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_p00].calculate_component(f, results, n_arrays, bs, comp) : interpolate_linearly_component(node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p, f, results, n_arrays, bs, comp);
    */
    return;
  }
  inline void f_p00_linear(const double *f[], double *results, const unsigned int &n_arrays) const
  {
    interpolate_linearly(node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0 ONLY3D(COMMA d_p00_0m COMMA d_p00_0p), f, results, n_arrays);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_p00].calculate(f, results, n_arrays) : interpolate_linearly(node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p, f, results, n_arrays);
    */
    return;
  }
  inline void f_0m0_linear_all_components(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    interpolate_linearly_all_components(node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p), f, results, n_arrays, bs);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_0m0].calculate_all_components(f, results, n_arrays, bs) : interpolate_linearly_all_components(node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p), f, results, n_arrays, bs);
    */
    return;
  }
  inline void f_0m0_linear_component(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    interpolate_linearly_component(node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p), f, results, n_arrays, bs, comp);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_0m0].calculate_component(f, results, n_arrays, bs, comp) : interpolate_linearly_component(node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p), f, results, n_arrays, bs, comp);
    */
    return;
  }
  inline void f_0m0_linear(const double *f[], double *results, const unsigned int &n_arrays) const
  {
    interpolate_linearly(node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p), f, results, n_arrays);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_0m0].calculate(f, results, n_arrays) : interpolate_linearly(node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p), f, results, n_arrays);
    */
    return;
  }
  inline void f_0p0_linear_all_components(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    interpolate_linearly_all_components(node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p), f, results, n_arrays, bs);

    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_0p0].calculate_all_components(f, results, n_arrays, bs) : interpolate_linearly_all_components(node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p), f, results, n_arrays, bs);
    */
    return;
  }
  inline void f_0p0_linear_component(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    interpolate_linearly_component(node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p), f, results, n_arrays, bs, comp);
    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_0p0].calculate_component(f, results, n_arrays, bs, comp) : interpolate_linearly_component(node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p), f, results, n_arrays, bs, comp);
    */
    return;
  }
  inline void f_0p0_linear(const double *f[], double *results, const unsigned int &n_arrays) const
  {
    interpolate_linearly(node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p), f, results, n_arrays);

    /*
    linear_interpolators_are_set? linear_interpolator[dir::f_0p0].calculate(f, results, n_arrays) : interpolate_linearly(node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p), f, results, n_arrays);
    */
    return;
  }
#ifdef P4_TO_P8
  inline void f_00m_linear_all_components(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    interpolate_linearly_all_components(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, results, n_arrays, bs);
    /*    linear_interpolators_are_set? linear_interpolator[dir::f_00m].calculate_all_components(f, results, n_arrays, bs) : interpolate_linearly_all_components(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, results, n_arrays, bs);*/
    return;
  }
  inline void f_00m_linear_component(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    interpolate_linearly_component(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, results, n_arrays, bs, comp);
    /*    linear_interpolators_are_set? linear_interpolator[dir::f_00m].calculate_component(f, results, n_arrays, bs, comp) : interpolate_linearly_component(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, results, n_arrays, bs, comp);*/
    return;
  }
  inline void f_00m_linear(const double *f[], double *results, const unsigned int &n_arrays) const
  {
    interpolate_linearly(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, results, n_arrays);
    /*    linear_interpolators_are_set? linear_interpolator[dir::f_00m].calculate(f, results, n_arrays) : interpolate_linearly(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, results, n_arrays);*/
    return;
  }
  inline void f_00p_linear_all_components(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    interpolate_linearly_all_components(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, results, n_arrays, bs);
    /*    linear_interpolators_are_set? linear_interpolator[dir::f_00p].calculate_all_components(f, results, n_arrays, bs) : interpolate_linearly_all_components(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, results, n_arrays, bs);*/
    return;
  }
  inline void f_00p_linear_component(const double *f[], double *results, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    interpolate_linearly_component(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, results, n_arrays, bs, comp);
    /*    linear_interpolators_are_set? linear_interpolator[dir::f_00p].calculate_component(f, results, n_arrays, bs, comp) : interpolate_linearly_component(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, results, n_arrays, bs, comp);*/
    return;
  }
  inline void f_00p_linear(const double *f[], double *results, const unsigned int &n_arrays) const
  {
    interpolate_linearly(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, results, n_arrays);
    /*    linear_interpolators_are_set? linear_interpolator[dir::f_00p].calculate(f, results, n_arrays) : interpolate_linearly(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, results, n_arrays);*/
    return;
  }
#endif

  void linearly_interpolated_neighbors_all_components(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p),
                                                      const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    const unsigned bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k){
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs + comp] = f[k][bs_node_000 + comp];
    }
    f_p00_linear_all_components(f, f_p00, n_arrays, bs); f_m00_linear_all_components(f, f_m00, n_arrays, bs);
    f_0p0_linear_all_components(f, f_0p0, n_arrays, bs); f_0m0_linear_all_components(f, f_0m0, n_arrays, bs);
#ifdef P4_TO_P8
    f_00p_linear_all_components(f, f_00p, n_arrays, bs); f_00m_linear_all_components(f, f_00m, n_arrays, bs);
#endif
    return;
  }
  void linearly_interpolated_neighbors_component(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p),
                                                 const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    const unsigned bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000 + comp];
    f_p00_linear_component(f, f_p00, n_arrays, bs, comp); f_m00_linear_component(f, f_m00, n_arrays, bs, comp);
    f_0p0_linear_component(f, f_0p0, n_arrays, bs, comp); f_0m0_linear_component(f, f_0m0, n_arrays, bs, comp);
#ifdef P4_TO_P8
    f_00p_linear_component(f, f_00p, n_arrays, bs, comp); f_00m_linear_component(f, f_00m, n_arrays, bs, comp);
#endif
    return;
  }
  void linearly_interpolated_neighbors(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p),
                                       const unsigned int &n_arrays) const
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_p00_linear(f, f_p00, n_arrays); f_m00_linear(f, f_m00, n_arrays);
    f_0p0_linear(f, f_0p0, n_arrays); f_0m0_linear(f, f_0m0, n_arrays);
#ifdef P4_TO_P8
    f_00p_linear(f, f_00p, n_arrays); f_00m_linear(f, f_00m, n_arrays);
#endif
    return;
  }

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

  void correct_naive_second_derivatives(DIM(double *Dxx, double *Dyy, double *Dzz),  const unsigned int &n_values) const;
  //  void correct_naive_second_derivatives(DIM(node_linear_combination &Dxx, node_linear_combination &Dyy, node_linear_combination &Dzz)) const;

  inline void laplace_all_components(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p),
                                     DIM(double *fxx, double *fyy, double *fzz),
                                     const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    linearly_interpolated_neighbors_all_components(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), n_arrays, bs);
    // naive calculations
    for (unsigned int k = 0; k < n_arrays*bs; ++k) {
      fxx[k] = central_second_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
      fyy[k] = central_second_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
#ifdef P4_TO_P8
      fzz[k] = central_second_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
#endif
    }
    correct_naive_second_derivatives(DIM(fxx, fyy, fzz), n_arrays*bs);
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  inline void laplace_component(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p),
                                DIM(double *fxx, double *fyy, double *fzz),
                                const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    linearly_interpolated_neighbors_component(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), n_arrays, bs, comp);
    // naive calculations
    for (unsigned int k = 0; k < n_arrays; ++k) {
      fxx[k] = central_second_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
      fyy[k] = central_second_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
#ifdef P4_TO_P8
      fzz[k] = central_second_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
#endif
    }
    correct_naive_second_derivatives(DIM(fxx, fyy, fzz),  n_arrays);
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  inline void laplace(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p),
                      DIM(double *fxx, double *fyy, double *fzz), const unsigned int &n_arrays) const
  {
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    linearly_interpolated_neighbors(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), n_arrays);
    // naive calculations
    for (unsigned int k = 0; k < n_arrays; ++k) {
      fxx[k] = central_second_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
      fyy[k] = central_second_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
#ifdef P4_TO_P8
      fzz[k] = central_second_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
#endif
    }
    correct_naive_second_derivatives(DIM(fxx, fyy, fzz),  n_arrays);
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  inline void laplace_all_components(const double *f[], DIM(double *fxx, double *fyy, double *fzz), const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[0].calculate_all_components(f, fxx, n_arrays, bs);
      second_derivative_operator[1].calculate_all_components(f, fyy, n_arrays, bs);
#ifdef P4_TO_P8
      second_derivative_operator[2].calculate_all_components(f, fzz, n_arrays, bs);
#endif
      return;
    }
    */
    double tmp_000[n_arrays*bs], tmp_m00[n_arrays*bs], tmp_p00[n_arrays*bs], tmp_0m0[n_arrays*bs], tmp_0p0[n_arrays*bs] ONLY3D(COMMA tmp_00m[n_arrays*bs] COMMA tmp_00p[n_arrays*bs]);
    laplace_all_components(f, tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0 ONLY3D(COMMA tmp_00m COMMA tmp_00p), DIM(fxx, fyy, fzz), n_arrays, bs);
    return;
  }

  inline void laplace_component(const double *f[], DIM(double *fxx, double *fyy, double *fzz),  const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[0].calculate_component(f, fxx, n_arrays, bs, comp);
      second_derivative_operator[1].calculate_component(f, fyy, n_arrays, bs, comp);
#ifdef P4_TO_P8
      second_derivative_operator[2].calculate_component(f, fzz, n_arrays, bs, comp);
#endif
      return;
    }
    */
    double tmp_000[n_arrays], tmp_m00[n_arrays], tmp_p00[n_arrays], tmp_0m0[n_arrays], tmp_0p0[n_arrays] ONLY3D(COMMA tmp_00m[n_arrays] COMMA tmp_00p[n_arrays]);
    laplace_component(f, tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0 ONLY3D(COMMA tmp_00m COMMA tmp_00p), DIM(fxx, fyy, fzz), n_arrays, bs, comp);
    return;
  }

  inline void laplace(const double *f[], DIM(double *fxx, double *fyy, double *fzz),  const unsigned int &n_arrays) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[0].calculate(f, fxx, n_arrays);
      second_derivative_operator[1].calculate(f, fyy, n_arrays);
#ifdef P4_TO_P8
      second_derivative_operator[2].calculate(f, fzz, n_arrays);
#endif
      return;
    }
    */
    double tmp_000[n_arrays], tmp_m00[n_arrays], tmp_p00[n_arrays], tmp_0m0[n_arrays], tmp_0p0[n_arrays] ONLY3D(COMMA tmp_00m[n_arrays] COMMA tmp_00p[n_arrays]);
    laplace(f, tmp_000, tmp_m00, tmp_p00, tmp_0m0, tmp_0p0 ONLY3D(COMMA tmp_00m COMMA tmp_00p), DIM(fxx, fyy, fzz), n_arrays);
    return;
  }


  inline void ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p), const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    /*
    if(quadratic_interpolators_are_set)
    {
      const unsigned int bs_node_000 = bs*node_000;
      for (unsigned int k = 0; k < n_arrays; ++k){
        const unsigned int kbs = k*bs;
        for (unsigned int comp = 0; comp < bs; ++comp)
          f_000[kbs + comp] = f[k][bs_node_000 + comp];
      }
      quadratic_interpolator[dir::f_m00].calculate_all_components(f, f_m00, n_arrays, bs);
      quadratic_interpolator[dir::f_p00].calculate_all_components(f, f_p00, n_arrays, bs);
      quadratic_interpolator[dir::f_0m0].calculate_all_components(f, f_0m0, n_arrays, bs);
      quadratic_interpolator[dir::f_0p0].calculate_all_components(f, f_0p0, n_arrays, bs);
#ifdef P4_TO_P8
      quadratic_interpolator[dir::f_00m].calculate_all_components(f, f_00m, n_arrays, bs);
      quadratic_interpolator[dir::f_00p].calculate_all_components(f, f_00p, n_arrays, bs);
#endif
      return;
    }
    */

    double DIM(fxx[n_arrays*bs], fyy[n_arrays*bs], fzz[n_arrays*bs]);
    laplace_all_components(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), DIM(fxx, fyy, fzz),  n_arrays, bs);
    // third order interpolation
    for (unsigned int k = 0; k < n_arrays*bs; ++k) {
      f_m00[k] -= (0.5*d_m00_m0*d_m00_p0*fyy[k] ONLY3D( + 0.5*d_m00_0m*d_m00_0p*fzz[k]));
      f_p00[k] -= (0.5*d_p00_m0*d_p00_p0*fyy[k] ONLY3D( + 0.5*d_p00_0m*d_p00_0p*fzz[k]));
      f_0m0[k] -= (0.5*d_0m0_m0*d_0m0_p0*fxx[k] ONLY3D( + 0.5*d_0m0_0m*d_0m0_0p*fzz[k]));
      f_0p0[k] -= (0.5*d_0p0_m0*d_0p0_p0*fxx[k] ONLY3D( + 0.5*d_0p0_0m*d_0p0_0p*fzz[k]));
#ifdef P4_TO_P8
      f_00m[k] -= (0.5*d_00m_m0*d_00m_p0*fxx[k]         + 0.5*d_00m_0m*d_00m_0p*fyy[k]);
      f_00p[k] -= (0.5*d_00p_m0*d_00p_p0*fxx[k]         + 0.5*d_00p_0m*d_00p_0p*fyy[k]);
#endif
    }

#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    ierr_flops = PetscLogFlops(48*n_arrays*bs); CHKERRXX(ierr_flops);
#else
    ierr_flops = PetscLogFlops(16*n_arrays*bs); CHKERRXX(ierr_flops);
#endif
#endif
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  inline void ngbd_with_quadratic_interpolation_component(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p), const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    /*
    if(quadratic_interpolators_are_set)
    {
      const unsigned int bs_node_000 = bs*node_000;
      for (unsigned int k = 0; k < n_arrays; ++k)
        f_000[k] = f[k][bs_node_000 + comp];
      quadratic_interpolator[dir::f_m00].calculate_all_components(f, f_m00, n_arrays, bs);
      quadratic_interpolator[dir::f_p00].calculate_all_components(f, f_p00, n_arrays, bs);
      quadratic_interpolator[dir::f_0m0].calculate_all_components(f, f_0m0, n_arrays, bs);
      quadratic_interpolator[dir::f_0p0].calculate_all_components(f, f_0p0, n_arrays, bs);
#ifdef P4_TO_P8
      quadratic_interpolator[dir::f_00m].calculate_all_components(f, f_00m, n_arrays, bs);
      quadratic_interpolator[dir::f_00p].calculate_all_components(f, f_00p, n_arrays, bs);
#endif
      return;
    }
    */

    double DIM(fxx[n_arrays], fyy[n_arrays], fzz[n_arrays]);
    laplace_component(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), DIM(fxx, fyy, fzz),  n_arrays, bs, comp);
    // third order interpolation
    for (unsigned int k = 0; k < n_arrays; ++k) {
      f_m00[k] -= (0.5*d_m00_m0*d_m00_p0*fyy[k] ONLY3D( + 0.5*d_m00_0m*d_m00_0p*fzz[k]));
      f_p00[k] -= (0.5*d_p00_m0*d_p00_p0*fyy[k] ONLY3D( + 0.5*d_p00_0m*d_p00_0p*fzz[k]));
      f_0m0[k] -= (0.5*d_0m0_m0*d_0m0_p0*fxx[k] ONLY3D( + 0.5*d_0m0_0m*d_0m0_0p*fzz[k]));
      f_0p0[k] -= (0.5*d_0p0_m0*d_0p0_p0*fxx[k] ONLY3D( + 0.5*d_0p0_0m*d_0p0_0p*fzz[k]));
#ifdef P4_TO_P8
      f_00m[k] -= (0.5*d_00m_m0*d_00m_p0*fxx[k]         + 0.5*d_00m_0m*d_00m_0p*fyy[k]);
      f_00p[k] -= (0.5*d_00p_m0*d_00p_p0*fxx[k]         + 0.5*d_00p_0m*d_00p_0p*fyy[k]);
#endif
    }

#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    ierr_flops = PetscLogFlops(48*n_arrays); CHKERRXX(ierr_flops);
#else
    ierr_flops = PetscLogFlops(16*n_arrays); CHKERRXX(ierr_flops);
#endif
#endif
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  void x_ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned int &n_arrays, const unsigned int &bs) const;
  void x_ngbd_with_quadratic_interpolation_component(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  void x_ngbd_with_quadratic_interpolation(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned int &n_arrays) const;
  void y_ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned int &n_arrays, const unsigned int &bs) const;
  void y_ngbd_with_quadratic_interpolation_component(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  void y_ngbd_with_quadratic_interpolation(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned int &n_arrays) const;
#ifdef P4_TO_P8
  void z_ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned int &n_arrays, const unsigned int &bs) const;
  void z_ngbd_with_quadratic_interpolation_component(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  void z_ngbd_with_quadratic_interpolation(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned int &n_arrays) const;
#endif
  inline void dxx_central_all_components(const double *f[], double *fxx, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[0].calculate_all_components(f, fxx, n_arrays, bs);
      return;
    }
    */
    double f_m00[n_arrays*bs], f_000[n_arrays*bs], f_p00[n_arrays*bs];
    x_ngbd_with_quadratic_interpolation_all_components(f, f_m00, f_000, f_p00, n_arrays, bs);
    for (unsigned int k = 0; k < n_arrays*bs; ++k)
      fxx[k] = central_second_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
    return;
  }
  inline void dxx_central_component(const double *f[], double *fxx, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[0].calculate_component(f, fxx, n_arrays, bs, comp);
      return;
    }
    */
    double f_m00[n_arrays], f_000[n_arrays], f_p00[n_arrays];
    x_ngbd_with_quadratic_interpolation_component(f, f_m00, f_000, f_p00, n_arrays, bs, comp);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fxx[k] = central_second_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
    return;
  }
  inline void dxx_central(const double *f[], double *fxx, const unsigned int &n_arrays) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[0].calculate(f, fxx, n_arrays);
      return;
    }
    */
    double f_m00[n_arrays], f_000[n_arrays], f_p00[n_arrays];
    x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fxx[k] = central_second_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
    return;
  }
  inline void dyy_central_all_components(const double *f[], double *fyy, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[1].calculate_all_components(f, fyy, n_arrays, bs);
      return;
    }
    */
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    double f_0m0[n_arrays*bs], f_000[n_arrays*bs], f_0p0[n_arrays*bs];
    y_ngbd_with_quadratic_interpolation_all_components(f, f_0m0, f_000, f_0p0, n_arrays, bs);
    for (unsigned int k = 0; k < n_arrays*bs; ++k)
      fyy[k] = central_second_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
    return;
  }
  inline void dyy_central_component(const double *f[], double *fyy, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[1].calculate_component(f, fyy, n_arrays, bs, comp);
      return;
    }
    */
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    double f_0m0[n_arrays], f_000[n_arrays], f_0p0[n_arrays];
    y_ngbd_with_quadratic_interpolation_component(f, f_0m0, f_000, f_0p0, n_arrays, bs, comp);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fyy[k] = central_second_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
    return;
  }
  inline void dyy_central(const double *f[], double *fyy, const unsigned int &n_arrays) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[1].calculate(f, fyy, n_arrays);
      return;
    }
    */
    double f_0m0[n_arrays], f_000[n_arrays], f_0p0[n_arrays];
    y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fyy[k] = central_second_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
    return;
  }
#ifdef P4_TO_P8
  inline void dzz_central_all_components(const double *f[], double *fzz, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[2].calculate_all_components(f, fzz, n_arrays, bs);
      return;
    }
    */
    double f_00m[n_arrays*bs],f_000[n_arrays*bs],f_00p[n_arrays*bs];
    z_ngbd_with_quadratic_interpolation_all_components(f, f_00m, f_000, f_00p, n_arrays, bs);
    for (unsigned int k = 0; k < n_arrays*bs; ++k)
      fzz[k] = central_second_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
    return;
  }
  inline void dzz_central_component(const double *f[], double *fzz, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[2].calculate_component(f, fzz, n_arrays, bs, comp);
      return;
    }
    */
    double f_00m[n_arrays],f_000[n_arrays],f_00p[n_arrays];
    z_ngbd_with_quadratic_interpolation_component(f, f_00m, f_000, f_00p, n_arrays, bs, comp);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fzz[k] = central_second_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
    return;
  }
  inline void dzz_central(const double *f[], double *fzz, const unsigned int &n_arrays) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[2].calculate(f, fzz, n_arrays);
      return;
    }
    */
    double f_00m[n_arrays],f_000[n_arrays],f_00p[n_arrays];
    z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fzz[k] = central_second_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
    return;
  }
#endif
  inline void dd_central_all_components(const unsigned short &der, const double *f[], double *fdd, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[der].calculate_all_components(f, fdd, n_arrays, bs);
      return;
    }
    */
    switch (der) {
    case dir::x:
      dxx_central_all_components(f, fdd, n_arrays, bs);
      break;
    case dir::y:
      dyy_central_all_components(f, fdd, n_arrays, bs);
      break;
#ifdef P4_TO_P8
    case dir::z:
      dzz_central_all_components(f, fdd, n_arrays, bs);
      break;
#endif
    default:
#ifdef CASL_THROWS
      throw std::invalid_argument("quad_neighbor_nodes_of_node_t::dd_central(der, ...) : unknown differentiation direction.");
#endif
      break;
    }
    return;
  }
  inline void dd_central_component(const unsigned short &der, const double *f[], double *fdd, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[der].calculate_component(f, fdd, n_arrays, bs, comp);
      return;
    }
    */
    switch (der) {
    case dir::x:
      dxx_central_component(f, fdd, n_arrays, bs, comp);
      break;
    case dir::y:
      dyy_central_component(f, fdd, n_arrays, bs, comp);
      break;
#ifdef P4_TO_P8
    case dir::z:
      dzz_central_component(f, fdd, n_arrays, bs, comp);
      break;
#endif
    default:
#ifdef CASL_THROWS
      throw std::invalid_argument("quad_neighbor_nodes_of_node_t::dd_central(der, ...) : unknown differentiation direction.");
#endif
      break;
    }
    return;
  }
  inline void dd_central(const unsigned short &der, const double *f[], double *fdd, const unsigned int &n_arrays) const
  {
    /*
    if(second_derivative_operators_are_set)
    {
      second_derivative_operator[der].calculate(f, fdd, n_arrays);
      return;
    }
    */
    switch (der) {
    case dir::x:
      dxx_central(f, fdd, n_arrays);
      break;
    case dir::y:
      dyy_central(f, fdd, n_arrays);
      break;
#ifdef P4_TO_P8
    case dir::z:
      dzz_central(f, fdd, n_arrays);
      break;
#endif
    default:
#ifdef CASL_THROWS
      throw std::invalid_argument("quad_neighbor_nodes_of_node_t::dd_central(der, ...) : unknown differentiation direction.");
#endif
      break;
    }
    return;
  }

  // first-derivatives-related procedures
  inline void gradient_all_components(const double *f[], DIM(double *fx, double *fy, double *fz), const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    /*
    if(gradient_operator_is_set)
    {
      gradient_operator[0].calculate_all_components(f, fx, n_arrays, bs);
      gradient_operator[1].calculate_all_components(f, fy, n_arrays, bs);
#ifdef P4_TO_P8
      gradient_operator[2].calculate_all_components(f, fz, n_arrays, bs);
#endif
      return;
    }
    */
    double f_000[n_arrays*bs], f_m00[n_arrays*bs], f_p00[n_arrays*bs], f_0m0[n_arrays*bs], f_0p0[n_arrays*bs] ONLY3D(COMMA f_00m[n_arrays*bs] COMMA f_00p[n_arrays*bs]);
    linearly_interpolated_neighbors_all_components(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), n_arrays, bs);

    double DIM(naive_Dx[n_arrays*bs], naive_Dy[n_arrays*bs], naive_Dz[n_arrays*bs]);
    for (unsigned int k = 0; k < n_arrays*bs; ++k) {
      naive_Dx[k] = central_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
      naive_Dy[k] = central_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
#ifdef P4_TO_P8
      naive_Dz[k] = central_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
#endif
    }
    correct_naive_first_derivatives(f, DIM(naive_Dx, naive_Dy, naive_Dz), DIM(fx, fy, fz), n_arrays, bs, bs);
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  inline void gradient_component(const double *f[], DIM(double *fx, double *fy, double *fz), const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    P4EST_ASSERT(bs > 1);
    P4EST_ASSERT(comp < bs);
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    /*
    if(gradient_operator_is_set)
    {
      gradient_operator[0].calculate_component(f, fx, n_arrays, bs, comp);
      gradient_operator[1].calculate_component(f, fy, n_arrays, bs, comp);
#ifdef P4_TO_P8
      gradient_operator[2].calculate_component(f, fz, n_arrays, bs, comp);
#endif
      return;
    }
    */
    double f_000[n_arrays], f_m00[n_arrays], f_p00[n_arrays], f_0m0[n_arrays], f_0p0[n_arrays] ONLY3D(COMMA f_00m[n_arrays] COMMA f_00p[n_arrays]);
    linearly_interpolated_neighbors_component(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), n_arrays, bs, comp);

    double DIM(naive_Dx[n_arrays], naive_Dy[n_arrays], naive_Dz[n_arrays]);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      naive_Dx[k] = central_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
      naive_Dy[k] = central_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
#ifdef P4_TO_P8
      naive_Dz[k] = central_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
#endif
    }
    correct_naive_first_derivatives(f, DIM(naive_Dx, naive_Dy, naive_Dz), DIM(fx, fy, fz), n_arrays, bs, comp);
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }
  inline void gradient(const double *f[], DIM(double *fx, double *fy, double *fz), const unsigned int &n_arrays) const
  {
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    /*
    if(gradient_operator_is_set)
    {
      gradient_operator[0].calculate(f, fx, n_arrays);
      gradient_operator[1].calculate(f, fy, n_arrays);
#ifdef P4_TO_P8
      gradient_operator[2].calculate(f, fz, n_arrays);
#endif
      return;
    }
    */
    double f_000[n_arrays], f_m00[n_arrays], f_p00[n_arrays], f_0m0[n_arrays], f_0p0[n_arrays] ONLY3D(COMMA f_00m[n_arrays] COMMA f_00p[n_arrays]);
    linearly_interpolated_neighbors(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), n_arrays);

    double DIM(naive_Dx[n_arrays], naive_Dy[n_arrays], naive_Dz[n_arrays]);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      naive_Dx[k] = central_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00);
      naive_Dy[k] = central_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0);
#ifdef P4_TO_P8
      naive_Dz[k] = central_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m);
#endif
    }
    correct_naive_first_derivatives(f, DIM(naive_Dx, naive_Dy, naive_Dz), DIM(fx, fy, fz), n_arrays, 1, 1);
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  void dx_central_internal(const double *f[], double *fx, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  inline void dx_central_all_components (const double *f[], double *fx, const unsigned int &n_arrays, const unsigned int &bs) const                           { dx_central_internal(f, fx, n_arrays, bs, bs);    }
  inline void dx_central_component      (const double *f[], double *fx, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const { dx_central_internal(f, fx, n_arrays, bs, comp);  }
  inline void dx_central                (const double *f[], double *fx, const unsigned int &n_arrays) const                                                   { dx_central_internal(f, fx, n_arrays, 1,  1);     }
  void dy_central_internal(const double *f[], double *fy, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  inline void dy_central_all_components (const double *f[], double *fy, const unsigned int &n_arrays, const unsigned int &bs) const                           { dy_central_internal(f, fy, n_arrays, bs, bs);    }
  inline void dy_central_component      (const double *f[], double *fy, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const { dy_central_internal(f, fy, n_arrays, bs, comp);  }
  inline void dy_central                (const double *f[], double *fy, const unsigned int &n_arrays) const                                                   { dy_central_internal(f, fy, n_arrays, 1,  1);     }
#ifdef P4_TO_P8
  void dz_central_internal(const double *f[], double *fz, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  inline void dz_central_all_components (const double *f[], double *fz, const unsigned int &n_arrays, const unsigned int &bs) const                           { dz_central_internal(f, fz, n_arrays, bs, bs);    }
  inline void dz_central_component      (const double *f[], double *fz, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const { dz_central_internal(f, fz, n_arrays, bs, comp);  }
  inline void dz_central                (const double *f[], double *fz, const unsigned int &n_arrays) const                                                   { dz_central_internal(f, fz, n_arrays, 1,  1);     }
#endif

  void correct_naive_first_derivatives(const double *f[], DIM(const double *naive_Dx, const double *naive_Dy, const double *naive_Dz), DIM(double *Dx, double *Dy, double *Dz), const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const;
  //  void correct_naive_first_derivatives(DIM(node_linear_combination &Dx, node_linear_combination &Dy, node_linear_combination &Dz), DIM(const node_linear_combination &Dxx, const node_linear_combination &Dyy, const node_linear_combination &Dzz)) const;

  // biased-first-derivative-related procedures
  // based on quadratic-interpolation neighbors
  inline double d_backward_quadratic(const double &f_0_quad, const double &f_m_quad, const double &d_m, const double &f_dd_0, const double &f_dd_m) const
  {
    return (backward_derivative(f_0_quad, f_m_quad, d_m) + 0.5*d_m*MINMOD(f_dd_0, f_dd_m));
  }
  inline double d_forward_quadratic(const double &f_p_quad, const double &f_0_quad, const double &d_p, const double &f_dd_0, const double &f_dd_p) const
  {
    return (forward_derivative(f_p_quad, f_0_quad, d_p) - 0.5*d_p*MINMOD(f_dd_0, f_dd_p));
  }

  /*
  bool linear_interpolators_are_set, quadratic_interpolators_are_set, second_derivative_operators_are_set, gradient_operator_is_set;
  node_linear_combination linear_interpolator[P4EST_FACES], quadratic_interpolator[P4EST_DIM];
  node_linear_combination gradient_operator[P4EST_DIM];
  node_linear_combination second_derivative_operator[P4EST_DIM];

  inline void create_linear_interpolators(node_linear_combination lin_interp[P4EST_FACES]) const
  {
    for (unsigned char dir = 0; dir < P4EST_FACES; ++dir)
      lin_interp[dir].elements.reserve(1<<(P4EST_DIM-1));
#ifdef P4_TO_P8
    get_linear_interpolator(lin_interp[dir::f_m00], node_m00_mm, node_m00_pm, node_m00_mp, node_m00_pp, d_m00_m0, d_m00_p0, d_m00_0m, d_m00_0p);
    get_linear_interpolator(lin_interp[dir::f_p00], node_p00_mm, node_p00_pm, node_p00_mp, node_p00_pp, d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p);
    get_linear_interpolator(lin_interp[dir::f_0m0], node_0m0_mm, node_0m0_pm, node_0m0_mp, node_0m0_pp, d_0m0_m0, d_0m0_p0, d_0m0_0m, d_0m0_0p);
    get_linear_interpolator(lin_interp[dir::f_0p0], node_0p0_mm, node_0p0_pm, node_0p0_mp, node_0p0_pp, d_0p0_m0, d_0p0_p0, d_0p0_0m, d_0p0_0p);
    get_linear_interpolator(lin_interp[dir::f_00m], node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p);
    get_linear_interpolator(lin_interp[dir::f_00p], node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p);
#else
    get_linear_interpolator(lin_interp[dir::f_m00], node_m00_mm, node_m00_pm,                           d_m00_m0, d_m00_p0);
    get_linear_interpolator(lin_interp[dir::f_p00], node_p00_mm, node_p00_pm,                           d_p00_m0, d_p00_p0);
    get_linear_interpolator(lin_interp[dir::f_0m0], node_0m0_mm, node_0m0_pm,                           d_0m0_m0, d_0m0_p0);
    get_linear_interpolator(lin_interp[dir::f_0p0], node_0p0_mm, node_0p0_pm,                           d_0p0_m0, d_0p0_p0);
#endif
    return;
  }

  inline void create_laplace_operators(const node_linear_combination lin_op[P4EST_FACES], node_linear_combination DD[P4EST_DIM]) const
  {
    const node_interpolation_weight ww_000(node_000, 1.0);
    // naive second derivatives operators
    DD[0].elements.reserve(lin_op[dir::f_m00].elements.size() + lin_op[dir::f_p00].elements.size() + 1);
    DD[1].elements.reserve(lin_op[dir::f_0m0].elements.size() + lin_op[dir::f_0p0].elements.size() + 1);
#ifdef P4_TO_P8
    DD[2].elements.reserve(lin_op[dir::f_00m].elements.size() + lin_op[dir::f_00p].elements.size() + 1);
#endif
    DD[0].add_terms(lin_op[dir::f_p00], 2.0/(d_p00*(d_p00+d_m00))); DD[0].add_terms(lin_op[dir::f_m00], 2.0/(d_m00*(d_p00+d_m00))); DD[0].add_terms(ww_000, -2.0/(d_m00*d_p00));
    DD[1].add_terms(lin_op[dir::f_0p0], 2.0/(d_0p0*(d_0p0+d_0m0))); DD[1].add_terms(lin_op[dir::f_0m0], 2.0/(d_0m0*(d_0p0+d_0m0))); DD[1].add_terms(ww_000, -2.0/(d_0m0*d_0p0));
#ifdef P4_TO_P8
    DD[2].add_terms(lin_op[dir::f_00p], 2.0/(d_00p*(d_00p+d_00m))); DD[2].add_terms(lin_op[dir::f_00m], 2.0/(d_00m*(d_00p+d_00m))); DD[2].add_terms(ww_000, -2.0/(d_00m*d_00p));
#endif

#ifdef P4_TO_P8
    correct_naive_second_derivatives(DD[0], DD[1], DD[2]);
#else
    correct_naive_second_derivatives(DD[0], DD[1]);
#endif
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      DD[der].elements.shrink_to_fit();
    return;
  }

  inline void create_quadratic_interpolators(const node_linear_combination lin_op[P4EST_FACES], const node_linear_combination DD[P4EST_DIM],
                                             node_linear_combination quad_op[P4EST_FACES]) const
  {
    // third order interpolation
    for (unsigned char fdir = 0; fdir < P4EST_FACES; ++fdir)
      quad_op[fdir].add_terms(lin_op[fdir], 1.0);
    if(!check_if_zero(d_m00_m0*inverse_d_max) && !check_if_zero(d_m00_p0*inverse_d_max))
      quad_op[dir::f_m00].add_terms(DD[1], -0.5*d_m00_m0*d_m00_p0);
#ifdef P4_TO_P8
    if(!check_if_zero(d_m00_0m*inverse_d_max) && !check_if_zero(d_m00_0p*inverse_d_max))
      quad_op[dir::f_m00].add_terms(DD[2], -0.5*d_m00_0m*d_m00_0p);
#endif
    if(!check_if_zero(d_p00_m0*inverse_d_max) && !check_if_zero(d_p00_p0*inverse_d_max))
      quad_op[dir::f_p00].add_terms(DD[1], -0.5*d_p00_m0*d_p00_p0);
#ifdef P4_TO_P8
    if(!check_if_zero(d_p00_0m*inverse_d_max) && !check_if_zero(d_p00_0p*inverse_d_max))
      quad_op[dir::f_p00].add_terms(DD[2], -0.5*d_p00_0m*d_p00_0p);
#endif
    if(!check_if_zero(d_0m0_m0*inverse_d_max) && !check_if_zero(d_0m0_p0*inverse_d_max))
      quad_op[dir::f_0m0].add_terms(DD[0], -0.5*d_0m0_m0*d_0m0_p0);
#ifdef P4_TO_P8
    if(!check_if_zero(d_0m0_0m*inverse_d_max) && !check_if_zero(d_0m0_0p*inverse_d_max))
      quad_op[dir::f_0m0].add_terms(DD[2], -0.5*d_0m0_0m*d_0m0_0p);
#endif
    if(!check_if_zero(d_0p0_m0*inverse_d_max) && !check_if_zero(d_0p0_p0*inverse_d_max))
      quad_op[dir::f_0p0].add_terms(DD[0], -0.5*d_0p0_m0*d_0p0_p0);
#ifdef P4_TO_P8
    if(!check_if_zero(d_0p0_0m*inverse_d_max) && !check_if_zero(d_0p0_0p*inverse_d_max))
      quad_op[dir::f_0p0].add_terms(DD[2], -0.5*d_0p0_0m*d_0p0_0p);
    if(!check_if_zero(d_00m_m0*inverse_d_max) && !check_if_zero(d_00m_p0*inverse_d_max))
      quad_op[dir::f_00m].add_terms(DD[0], -0.5*d_00m_m0*d_00m_p0);
    if(!check_if_zero(d_00m_0m*inverse_d_max) && !check_if_zero(d_00m_0p*inverse_d_max))
      quad_op[dir::f_00m].add_terms(DD[1], -0.5*d_00m_0m*d_00m_0p);
    if(!check_if_zero(d_00p_m0*inverse_d_max) && !check_if_zero(d_00p_p0*inverse_d_max))
      quad_op[dir::f_00p].add_terms(DD[0], -0.5*d_00p_m0*d_00p_p0);
    if(!check_if_zero(d_00p_0m*inverse_d_max) && !check_if_zero(d_00p_0p*inverse_d_max))
      quad_op[dir::f_00p].add_terms(DD[1], -0.5*d_00p_0m*d_00p_0p);
#endif
    return;
  }


  inline void create_gradient_interpolators(const node_linear_combination lin_op[P4EST_FACES], const node_linear_combination DD[P4EST_DIM],
                                             node_linear_combination grad_op[P4EST_DIM]) const
  {
    const node_interpolation_weight ww_000(node_000, 1.0);
    grad_op[0].add_terms(lin_op[dir::f_p00], d_m00/(d_p00*(d_p00+d_m00)));
    grad_op[0].add_terms(lin_op[dir::f_m00], -d_p00/(d_m00*(d_p00+d_m00)));
    if(!check_if_zero(d_p00/d_m00-d_m00/d_p00))
      grad_op[0].add_terms(ww_000, (d_p00/d_m00-d_m00/d_p00)/(d_p00+d_m00));

    grad_op[1].add_terms(lin_op[dir::f_0p0], d_0m0/(d_0p0*(d_0p0+d_0m0)));
    grad_op[1].add_terms(lin_op[dir::f_0m0], -d_0p0/(d_0m0*(d_0p0+d_0m0)));
    if(!check_if_zero(d_0p0/d_0m0-d_0m0/d_0p0))
      grad_op[1].add_terms(ww_000, (d_0p0/d_0m0-d_0m0/d_0p0)/(d_0p0+d_0m0));

#ifdef P4_TO_P8
    grad_op[2].add_terms(lin_op[dir::f_00p], d_00m/(d_00p*(d_00p+d_00m)));
    grad_op[2].add_terms(lin_op[dir::f_00m], -d_00p/(d_00m*(d_00p+d_00m)));
    if(!check_if_zero(d_00p/d_00m-d_00m/d_00p))
      grad_op[2].add_terms(ww_000, (d_00p/d_00m-d_00m/d_00p)/(d_00p+d_00m));
#endif
#ifdef P4_TO_P8
    correct_naive_first_derivatives(grad_op[0], grad_op[1], grad_op[2], DD[0], DD[1], DD[2]);
#else
    correct_naive_first_derivatives(grad_op[0], grad_op[1],             DD[0], DD[1]);
#endif
    return;
  }
  */

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
  /*
  quad_neighbor_nodes_of_node_t() : linear_interpolators_are_set(false), quadratic_interpolators_are_set(false),
    second_derivative_operators_are_set(false), gradient_operator_is_set(false) {}

  inline void set_and_store_linear_interpolators()
  {
    if(linear_interpolators_are_set)
      return;
    create_linear_interpolators(linear_interpolator);
    linear_interpolators_are_set = true;
  }

  inline void set_and_store_second_derivative_operators()
  {
    if(second_derivative_operators_are_set)
      return;
    if(linear_interpolators_are_set)
    {
      create_laplace_operators(linear_interpolator, second_derivative_operator);
      second_derivative_operators_are_set = true;
      return;
    }
    node_linear_combination *lin_op = new node_linear_combination[P4EST_FACES];
    create_linear_interpolators(lin_op);

    create_laplace_operators(lin_op, second_derivative_operator);
    second_derivative_operators_are_set = true;
    delete[] lin_op;
    return;
  }

  inline void set_and_store_quadratic_interpolators()
  {
    if(quadratic_interpolators_are_set)
      return;
    if(linear_interpolators_are_set && second_derivative_operators_are_set)
    {
      create_quadratic_interpolators(linear_interpolator, second_derivative_operator, quadratic_interpolator);
      quadratic_interpolators_are_set = true;
      return;
    }
    else if(linear_interpolators_are_set && !second_derivative_operators_are_set)
    {
      node_linear_combination *DD = new node_linear_combination[P4EST_DIM];
      create_laplace_operators(linear_interpolator, DD);
      create_quadratic_interpolators(linear_interpolator, DD, quadratic_interpolator);
      delete[] DD;
      quadratic_interpolators_are_set = true;
      return;
    }
    else if(!linear_interpolators_are_set && second_derivative_operators_are_set)
    {
      node_linear_combination *lin_op = new node_linear_combination[P4EST_FACES];
      create_linear_interpolators(lin_op);
      create_quadratic_interpolators(lin_op, second_derivative_operator, quadratic_interpolator);
      delete[] lin_op;
      quadratic_interpolators_are_set = true;
      return;
    }
    node_linear_combination *lin_op = new node_linear_combination[P4EST_FACES];
    node_linear_combination *DD = new node_linear_combination[P4EST_DIM];
    create_linear_interpolators(lin_op);
    create_laplace_operators(lin_op, DD);
    create_quadratic_interpolators(lin_op, DD, quadratic_interpolator);
    delete[] lin_op;
    delete[] DD;
    quadratic_interpolators_are_set = true;
    return;
  }

  inline void set_and_store_gradient_operator()
  {
    if(gradient_operator_is_set)
      return;
    if(linear_interpolators_are_set && second_derivative_operators_are_set)
    {
      create_gradient_interpolators(linear_interpolator, second_derivative_operator, gradient_operator);
      gradient_operator_is_set = true;
      return;
    }
    else if(linear_interpolators_are_set && !second_derivative_operators_are_set)
    {
      node_linear_combination *DD = new node_linear_combination[P4EST_DIM];
      create_laplace_operators(linear_interpolator, DD);
      create_gradient_interpolators(linear_interpolator, DD, gradient_operator);
      delete[] DD;
      gradient_operator_is_set = true;
      return;
    }
    else if(!linear_interpolators_are_set && second_derivative_operators_are_set)
    {
      node_linear_combination *lin_op = new node_linear_combination[P4EST_FACES];
      create_linear_interpolators(lin_op);
      create_gradient_interpolators(lin_op, second_derivative_operator, gradient_operator);
      delete[] lin_op;
      gradient_operator_is_set = true;
      return;
    }
    node_linear_combination *lin_op = new node_linear_combination[P4EST_FACES];
    node_linear_combination *DD = new node_linear_combination[P4EST_DIM];
    create_linear_interpolators(lin_op);
    create_laplace_operators(lin_op, DD);
    create_gradient_interpolators(lin_op, DD, gradient_operator);
    delete[] lin_op;
    delete[] DD;
    gradient_operator_is_set = true;
    return;
  }
  */

  inline double f_m00_linear(const double *f) const
  {
    double result;
    f_m00_linear(&f, &result, 1);
    return result;
  }
  inline double f_p00_linear(const double *f) const
  {
    double result;
    f_p00_linear(&f, &result, 1);
    return result;
  }
  inline double f_0m0_linear(const double *f) const
  {
    double result;
    f_0m0_linear(&f, &result, 1);
    return result;
  }
  inline double f_0p0_linear(const double *f) const
  {
    double result;
    f_0p0_linear(&f, &result, 1);
    return result;
  }
#ifdef P4_TO_P8
  inline double f_00m_linear(const double *f) const
  {
    double result;
    f_00m_linear(&f, &result, 1);
    return result;
  }
  inline double f_00p_linear(const double *f) const
  {
    double result;
    f_00p_linear(&f, &result, 1);
    return result;
  }
#endif

  // second-derivatives-related and third-order-interpolation-related procedures
  inline void laplace(const double *f, double fxxyyzz[P4EST_DIM]) const
  {
    laplace(&f, DIM(&fxxyyzz[0], &fxxyyzz[1], &fxxyyzz[2]), 1);
    return;
  }

  inline void laplace(const double *f, DIM(double &fxx, double &fyy, double &fzz)) const
  {
    laplace(&f, DIM(&fxx, &fyy, &fzz), 1);
    return;
  }

  inline void laplace(const double *f[], double *serialized_fxxyyzz, const unsigned int &n_arrays) const
  {
    double DIM(fxx[n_arrays], fyy[n_arrays], fzz[n_arrays]);
    laplace(f, DIM(fxx, fyy, fzz), n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int l_offset = P4EST_DIM*k;
      serialized_fxxyyzz[l_offset + 0] = fxx[k];
      serialized_fxxyyzz[l_offset + 1] = fyy[k];
#ifdef P4_TO_P8
      serialized_fxxyyzz[l_offset + 2] = fzz[k];
#endif
    }
    return;
  }

  inline void laplace_insert_in_vectors(const double *f[], DIM(double *fxx[], double *fyy[], double *fzz[]), const unsigned int &n_arrays) const
  {
    double DIM(fxx_serial[n_arrays], fyy_serial[n_arrays], fzz_serial[n_arrays]);
    laplace(f, DIM(fxx_serial, fyy_serial, fzz_serial),  n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      fxx[k][node_000] = fxx_serial[k];
      fyy[k][node_000] = fyy_serial[k];
#ifdef P4_TO_P8
      fzz[k][node_000] = fzz_serial[k];
#endif
    }
    return;
  }

  inline void laplace_insert_in_block_vectors(const double *f[], double *fxxyyzz[], const unsigned int &n_arrays) const
  {
    double DIM(fxx_serial[n_arrays], fyy_serial[n_arrays], fzz_serial[n_arrays]);
    laplace(f, DIM(fxx_serial, fyy_serial, fzz_serial),  n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int l_offset = P4EST_DIM*node_000;
      fxxyyzz[k][l_offset]      = fxx_serial[k];
      fxxyyzz[k][l_offset + 1]  = fyy_serial[k];
#ifdef P4_TO_P8
      fxxyyzz[k][l_offset + 2]  = fzz_serial[k];
#endif
    }
    return;
  }

  inline void laplace_all_components(const double *f[], double *serialized_fxxyyzz, const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1); // the elementary functions for bs==1 use less flops (straightfward indices) --> functions have been duplicated for efficiency
    const unsigned int n_arrays_bs = n_arrays*bs;
    double DIM(fxx[n_arrays_bs], fyy[n_arrays_bs], fzz[n_arrays_bs]);
    laplace_all_components(f, DIM(fxx, fyy, fzz), n_arrays, bs);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp) {
        const unsigned int l_offset = P4EST_DIM*(kbs + comp);
        const unsigned int r_idx = kbs + comp;
        serialized_fxxyyzz[l_offset + 0] = fxx[r_idx];
        serialized_fxxyyzz[l_offset + 1] = fyy[r_idx];
#ifdef P4_TO_P8
        serialized_fxxyyzz[l_offset + 2] = fzz[r_idx];
#endif
      }
    }
    return;
  }

  inline void laplace_all_components_insert_in_vectors(const double *f[], DIM(double *fxx[], double *fyy[], double *fzz[]), const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    const unsigned int n_arrays_bs =n_arrays*bs;
    double DIM(fxx_serial[n_arrays_bs], fyy_serial[n_arrays_bs], fzz_serial[n_arrays_bs]);
    laplace_all_components(f, DIM(fxx_serial, fyy_serial, fzz_serial),  n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp) {
        const unsigned int l_index = l_offset + comp;
        const unsigned int r_idx = kbs + comp;
        fxx[k][l_index] = fxx_serial[r_idx];
        fyy[k][l_index] = fyy_serial[r_idx];
#ifdef P4_TO_P8
        fzz[k][l_index] = fzz_serial[r_idx];
#endif
      }
    }
    return;
  }

  inline void laplace_all_components_insert_in_block_vectors(const double *f[], double *fxxyyzz[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    const unsigned int n_arrays_bs =n_arrays*bs;
    double DIM(fxx_serial[n_arrays_bs], fyy_serial[n_arrays_bs], fzz_serial[n_arrays_bs]);
    laplace_all_components(f, DIM(fxx_serial, fyy_serial, fzz_serial),  n_arrays, bs);
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp) {
        const unsigned int l_offset = P4EST_DIM*(bs_node_000 + comp);
        const unsigned int r_idx = kbs + comp;
        fxxyyzz[k][l_offset]      = fxx_serial[r_idx];
        fxxyyzz[k][l_offset + 1]  = fyy_serial[r_idx];
#ifdef P4_TO_P8
        fxxyyzz[k][l_offset + 2]  = fzz_serial[r_idx];
#endif
      }
    }
    return;
  }

  inline void laplace_component(const double *f[], double *serialized_fxxyyzz, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
  {
    double DIM(fxx[n_arrays], fyy[n_arrays], fzz[n_arrays]);
    laplace_component(f, DIM(fxx, fyy, fzz), n_arrays, bs, comp);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int l_offset = P4EST_DIM*k;
      serialized_fxxyyzz[l_offset + 0] = fxx[k];
      serialized_fxxyyzz[l_offset + 1] = fyy[k];
#ifdef P4_TO_P8
      serialized_fxxyyzz[l_offset + 2] = fzz[k];
#endif
    }
    return;
  }

  inline void ngbd_with_quadratic_interpolation(const double *f[], double *f_000, double *f_m00, double *f_p00, double *f_0m0, double *f_0p0 ONLY3D(COMMA double *f_00m COMMA double *f_00p), const unsigned int &n_arrays) const
  {
#ifdef CASL_LOG_TINY_EVENTS
    PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    /*
    if(quadratic_interpolators_are_set)
    {
      for (unsigned int k = 0; k < n_arrays; ++k)
        f_000[k] = f[k][node_000];
      quadratic_interpolator[dir::f_m00].calculate(f, f_m00, n_arrays);
      quadratic_interpolator[dir::f_p00].calculate(f, f_p00, n_arrays);
      quadratic_interpolator[dir::f_0m0].calculate(f, f_0m0, n_arrays);
      quadratic_interpolator[dir::f_0p0].calculate(f, f_0p0, n_arrays);
#ifdef P4_TO_P8
      quadratic_interpolator[dir::f_00m].calculate(f, f_00m, n_arrays);
      quadratic_interpolator[dir::f_00p].calculate(f, f_00p, n_arrays);
#endif
      return;
    }
    */

    double DIM(fxx[n_arrays], fyy[n_arrays], fzz[n_arrays]);
    laplace(f, f_000, f_m00, f_p00, f_0m0, f_0p0 ONLY3D(COMMA f_00m COMMA f_00p), DIM(fxx, fyy, fzz),  n_arrays);
    // third order interpolation
    for (unsigned int k = 0; k < n_arrays; ++k) {
      f_m00[k] -= (0.5*d_m00_m0*d_m00_p0*fyy[k] ONLY3D( + 0.5*d_m00_0m*d_m00_0p*fzz[k]));
      f_p00[k] -= (0.5*d_p00_m0*d_p00_p0*fyy[k] ONLY3D( + 0.5*d_p00_0m*d_p00_0p*fzz[k]));
      f_0m0[k] -= (0.5*d_0m0_m0*d_0m0_p0*fxx[k] ONLY3D( + 0.5*d_0m0_0m*d_0m0_0p*fzz[k]));
      f_0p0[k] -= (0.5*d_0p0_m0*d_0p0_p0*fxx[k] ONLY3D( + 0.5*d_0p0_0m*d_0p0_0p*fzz[k]));
#ifdef P4_TO_P8
      f_00m[k] -= (0.5*d_00m_m0*d_00m_p0*fxx[k]         + 0.5*d_00m_0m*d_00m_0p*fyy[k]);
      f_00p[k] -= (0.5*d_00p_m0*d_00p_p0*fxx[k]         + 0.5*d_00p_0m*d_00p_0p*fyy[k]);
#endif
    }

#ifdef CASL_LOG_FLOPS
#ifdef P4_TO_P8
    ierr_flops = PetscLogFlops(48*n_arrays); CHKERRXX(ierr_flops);
#else
    ierr_flops = PetscLogFlops(16*n_arrays); CHKERRXX(ierr_flops);
#endif
#endif
#ifdef CASL_LOG_TINY_EVENTS
    ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
    return;
  }

  inline void ngbd_with_quadratic_interpolation(const double *f, double &f_000, double &f_m00, double &f_p00, double &f_0m0, double &f_0p0 ONLY3D(COMMA double &f_00m COMMA double &f_00p)) const
  {
    ngbd_with_quadratic_interpolation(&f, &f_000, &f_m00, &f_p00, &f_0m0, &f_0p0 ONLY3D(COMMA &f_00m COMMA &f_00p), 1);
    return;
  }

  inline void x_ngbd_with_quadratic_interpolation(const double *f, double &f_m00, double &f_000, double &f_p00) const
  {
    x_ngbd_with_quadratic_interpolation(&f, &f_m00, &f_000, &f_p00, 1);
    return;
  }
  inline void y_ngbd_with_quadratic_interpolation(const double *f, double &f_0m0, double &f_000, double &f_0p0) const
  {
    y_ngbd_with_quadratic_interpolation(&f, &f_0m0, &f_000, &f_0p0, 1);
    return;
  }
#ifdef P4_TO_P8
  inline void z_ngbd_with_quadratic_interpolation(const double *f, double &f_00m, double &f_000, double &f_00p) const
  {
    z_ngbd_with_quadratic_interpolation(&f, &f_00m, &f_000, &f_00p, 1);
    return;
  }
#endif

  /* second derivatives */
  inline double dxx_central(const double *f) const
  {
    double fxx;
    dxx_central(&f, &fxx, 1);
    return fxx;
  }
  inline void dxx_central_insert_in_vectors(const double *f[], double *fxx[], const unsigned int &n_arrays) const
  {
    double fxx_serial[n_arrays];
    dxx_central(f, fxx_serial, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fxx[k][node_000] = fxx_serial[k];
    return;
  }
  inline void dxx_central_all_components_insert_in_vectors(const double *f[], double *fxx[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double fxx_serial[n_arrays*bs];
    dxx_central_all_components(f, fxx_serial, n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        fxx[k][l_offset + comp] = fxx_serial[kbs + comp];
    }
    return;
  }
  double dyy_central(const double *f) const
  {
    double fyy;
    dyy_central(&f, &fyy, 1);
    return fyy;
  }
  inline void dyy_central_insert_in_vectors(const double *f[], double *fyy[], const unsigned int &n_arrays) const
  {
    double fyy_serial[n_arrays];
    dyy_central(f, fyy_serial, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fyy[k][node_000] = fyy_serial[k];
    return;
  }
  inline void dyy_central_all_components_insert_in_vectors(const double *f[], double *fyy[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double fyy_serial[n_arrays*bs];
    dyy_central_all_components(f, fyy_serial, n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        fyy[k][l_offset + comp] = fyy_serial[kbs + comp];
    }
    return;
  }
#ifdef P4_TO_P8
  double dzz_central(const double *f) const
  {
    double fzz;
    dzz_central(&f, &fzz, 1);
    return fzz;
  }
  inline void dzz_central_insert_in_vectors(const double *f[], double *fzz[], const unsigned int &n_arrays) const
  {
    double fzz_serial[n_arrays];
    dzz_central(f, fzz_serial, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fzz[k][node_000] = fzz_serial[k];
    return;
  }
  inline void dzz_central_all_components_insert_in_vectors(const double *f[], double *fzz[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double fzz_serial[n_arrays*bs];
    dzz_central_all_components(f, fzz_serial, n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        fzz[k][l_offset + comp] = fzz_serial[kbs + comp];
    }
    return;
  }
#endif
  inline double dd_central(const unsigned short &der, const double *f) const
  {
    double fdd;
    dd_central(der, &f, &fdd, 1);
    return fdd;
  }

  // first-derivatives-related procedures
  inline void gradient(const double *f, double fxyx[P4EST_DIM]) const
  {
    gradient(&f, DIM(&fxyx[0], &fxyx[1], &fxyx[2]),  1);
    return;
  }
  inline void gradient(const double **f, double **fxyz, const unsigned int &n_vecs) const
  {
    double DIM(fx_serial[n_vecs], fy_serial[n_vecs], fz_serial[n_vecs]);
    gradient(f, DIM(fx_serial, fy_serial, fz_serial), n_vecs);
    for (unsigned int k = 0; k < n_vecs; ++k) {
      fxyz[k][0] = fx_serial[k];
      fxyz[k][1] = fy_serial[k];
#ifdef P4_TO_P8
      fxyz[k][2] = fz_serial[k];
#endif
    }
    return;
  }
  inline void gradient(const double *f, DIM(double &fx, double &fy, double &fz)) const
  {
    gradient(&f, DIM(&fx, &fy, &fz), 1);
    return;
  }
  inline void gradient_all_components(const double *f[], double *fxyz[], const unsigned int &n_vecs, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    const unsigned bsnvecs = bs*n_vecs;
    double DIM(fx_serial[bsnvecs], fy_serial[bsnvecs], fz_serial[bsnvecs]);
    gradient_all_components(f, DIM(fx_serial, fy_serial, fz_serial),  n_vecs, bs);
    for (unsigned int k = 0; k < n_vecs; ++k) {
      const unsigned int bsk = bs*k;
      for (unsigned int comp = 0; comp < bs; ++comp) {
        const unsigned int comp_dim = comp*P4EST_DIM;
        fxyz[k][comp_dim + 0] = fx_serial[bsk + comp];
        fxyz[k][comp_dim + 1] = fy_serial[bsk + comp];
#ifdef P4_TO_P8
        fxyz[k][comp_dim + 2] = fz_serial[bsk + comp];
#endif
      }
    }
    return;
  }
  inline void gradient_all_components(const double *f, double *fxyz, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double DIM(fx_serial[bs], fy_serial[bs], fz_serial[bs]);
    gradient_all_components(&f, DIM(fx_serial, fy_serial, fz_serial),  1, bs);
    for (unsigned int comp = 0; comp < bs; ++comp) {
      const unsigned int comp_dim = comp*P4EST_DIM;
      fxyz[comp_dim + 0] = fx_serial[comp];
      fxyz[comp_dim + 1] = fy_serial[comp];
#ifdef P4_TO_P8
      fxyz[comp_dim + 2] = fz_serial[comp];
#endif
    }
    return;
  }
  inline void gradient_insert_in_vectors(const double *f[], DIM(double *fx[], double *fy[], double *fz[]), const unsigned int &n_arrays) const
  {
    double DIM(fx_serial[n_arrays], fy_serial[n_arrays], fz_serial[n_arrays]);
    gradient(f, DIM(fx_serial, fy_serial, fz_serial),  n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      fx[k][node_000] = fx_serial[k];
      fy[k][node_000] = fy_serial[k];
#ifdef P4_TO_P8
      fz[k][node_000] = fz_serial[k];
#endif
    }
    return;
  }
  inline void gradient_insert_in_block_vectors(const double *f[], double *fxyz[], const unsigned int &n_arrays) const
  {
    double DIM(fx_serial[n_arrays], fy_serial[n_arrays], fz_serial[n_arrays]);
    gradient(f, DIM(fx_serial, fy_serial, fz_serial), n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int l_offset = P4EST_DIM*node_000;
      fxyz[k][l_offset]     = fx_serial[k];
      fxyz[k][l_offset + 1] = fy_serial[k];
#ifdef P4_TO_P8
      fxyz[k][l_offset + 2] = fz_serial[k];
#endif
    }
    return;
  }

  inline void gradient_all_components_insert_in_vectors(const double *f[], DIM(double *fx[], double *fy[], double *fz[]), const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    const unsigned int n_arrays_bs = n_arrays*bs;
    double DIM(fx_serial[n_arrays_bs], fy_serial[n_arrays_bs], fz_serial[n_arrays_bs]);
    gradient_all_components(f, DIM(fx_serial, fy_serial, fz_serial),  n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k) {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp) {
        const unsigned int l_index = l_offset + comp;
        const unsigned int r_index = kbs + comp;
        fx[k][l_index] = fx_serial[r_index];
        fy[k][l_index] = fy_serial[r_index];
#ifdef P4_TO_P8
        fz[k][l_index] = fz_serial[r_index];
#endif
      }
    }
    return;
  }
  inline void gradient_all_components_insert_in_block_vectors(const double *f[], double *fxyz[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    const unsigned int n_arrays_bs = n_arrays*bs;
    double DIM(fx_serial[n_arrays_bs], fy_serial[n_arrays_bs], fz_serial[n_arrays_bs]);
    gradient_all_components(f, DIM(fx_serial, fy_serial, fz_serial),  n_arrays, bs);
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k) {
      unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp) {
        const unsigned int l_offset = P4EST_DIM*(bs_node_000 + comp);
        const unsigned int r_idx = kbs + comp;
        fxyz[k][l_offset]     = fx_serial[r_idx];
        fxyz[k][l_offset + 1] = fy_serial[r_idx];
#ifdef P4_TO_P8
        fxyz[k][l_offset + 2] = fz_serial[r_idx];
#endif
      }
    }
    return;
  }

  /* first derivatives */
  inline double dx_central(const double *f) const
  {
    double fx;
    dx_central(&f, &fx, 1);
    return fx;
  }
  inline void dx_central_insert_in_vectors(const double *f[], double *fx[], const unsigned int &n_arrays) const
  {
    double fx_serial[n_arrays];
    dx_central(f, fx_serial, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fx[k][node_000] = fx_serial[k];
    return;
  }
  inline double dx_central_component(const double *f, const unsigned int &bs, const unsigned int &comp) const
  {
    double fx;
    dx_central_component(&f, &fx, 1, bs, comp);
    return fx;
  }
  inline void dx_central_all_components_insert_in_vectors(const double *f[], double *fx[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double fx_serial[n_arrays*bs];
    dx_central_all_components(f, fx_serial, n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        fx[k][l_offset + comp] = fx_serial[kbs + comp];
    }
    return;
  }
  inline double dy_central(const double *f) const
  {
    double fy;
    dy_central(&f, &fy, 1);
    return fy;
  }
  inline void dy_central_insert_in_vectors(const double *f[], double *fy[], const unsigned int &n_arrays) const
  {
    double fy_serial[n_arrays];
    dy_central(f, fy_serial, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fy[k][node_000] = fy_serial[k];
    return;
  }
  inline double dy_central_component(const double *f, const unsigned int &bs, const unsigned int &comp) const
  {
    double fy;
    dy_central_component(&f, &fy, 1, bs, comp);
    return fy;
  }
  inline void dy_central_all_components_insert_in_vectors(const double *f[], double *fy[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double fy_serial[n_arrays*bs];
    dy_central_all_components(f, fy_serial, n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        fy[k][l_offset + comp] = fy_serial[kbs + comp];
    }
    return;
  }
#ifdef P4_TO_P8
  inline double dz_central(const double *f) const
  {
    double fz;
    dz_central(&f, &fz, 1);
    return fz;
  }
  inline void dz_central_insert_in_vectors(const double *f[], double *fz[], const unsigned int &n_arrays) const
  {
    double fz_serial[n_arrays];
    dz_central(f, fz_serial, n_arrays);
    for (unsigned int k = 0; k < n_arrays; ++k)
      fz[k][node_000] = fz_serial[k];
    return;
  }
  inline double dz_central_component(const double *f, const unsigned int &bs, const unsigned int &comp) const
  {
    double fz;
    dz_central_component(&f, &fz, 1, bs, comp);
    return fz;
  }
  inline void dz_central_all_components_insert_in_vectors(const double *f[], double *fz[], const unsigned int &n_arrays, const unsigned int &bs) const
  {
    P4EST_ASSERT(bs > 1);
    double fz_serial[n_arrays*bs];
    dz_central_all_components(f, fz_serial, n_arrays, bs);
    const unsigned int l_offset = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        fz[k][l_offset + comp] = fz_serial[kbs + comp];
    }
    return;
  }
#endif
  inline double d_central (const unsigned short &der, const double *f) const
  {
    return (der == dir::x ? dx_central(f) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_central(f) ONLY3D(: dz_central(f)CLOSE_PARENTHESIS));
  }

  /*!
   * \brief get_curvature evaluates the local mean curvature as divergence of (nabla phi/(magnitude of nabla of phi)).
   * The user MUST provide nabla of phi (either by component or block-structued) to enable local calculation!
   * \param [in] phi_p              : pointer to the data of a node-sampled vector storing the values of phi
   * \param [in] grad_phi_block_p   : pointer to the data of a P4EST_DIM block-structured node-sampled vector storing the components of the gradient of phi
   * \param [in] grad_phi_p         : pointer to an array of size P4EST_DIM for the components of the gradient of phi.
   * \param [in] phi_xxyyzz_block_p : pointer to the data of a P4EST_DIM block-structured node-sampled vector storing the second derivatives of phi with
   *                                  respect to the P4EST_DIM Cartesian directions x, y, z
   * \param [in] phi_xxyyzz_p       : pointer to an array of size P4EST_DIM for the second derivatives of phi with
   *                                  respect to the P4EST_DIM Cartesian directions x, y, z
   * NOTE 1: if phi_xxyyzz_block_p or phi_xxyyzz_p is provided, the second derivatives of phi are read from that it and phi_p is disregarded.
   * If they are not provided, they are calculated from phi_p.
   * NOTE 2: if the magnitude of the local gradient of phi falls below EPS, the value is not calculated (ill-defined case) and the returned
   * value is set to 0.0, by default.
   * \return the value of the local mean curvature.
   */
  inline double get_curvature(const double *phi_p,
                              const double *grad_phi_block_p, const double *grad_phi_p[P4EST_DIM],
                              const double *phi_xxyyzz_block_p, const double *phi_xxyyzz_p[P4EST_DIM]) const
  {
    P4EST_ASSERT((grad_phi_block_p != NULL && grad_phi_p == NULL) || (grad_phi_block_p == NULL && grad_phi_p != NULL)); // must provide the gradient, one way or the other
    P4EST_ASSERT(phi_p != NULL || phi_xxyyzz_block_p != NULL || phi_xxyyzz_p != NULL);
    // fetch first derivatives
    double mag_of_grad = 0.0;
    const double dx = (grad_phi_block_p != NULL ? grad_phi_block_p[P4EST_DIM*node_000 + 0] : grad_phi_p[0][node_000]); mag_of_grad += SQR(dx);
    const double dy = (grad_phi_block_p != NULL ? grad_phi_block_p[P4EST_DIM*node_000 + 1] : grad_phi_p[1][node_000]); mag_of_grad += SQR(dy);
#ifdef P4_TO_P8
    const double dz = (grad_phi_block_p != NULL ? grad_phi_block_p[P4EST_DIM*node_000 + 2] : grad_phi_p[2][node_000]); mag_of_grad += SQR(dz);
#endif
    mag_of_grad = sqrt(mag_of_grad);

    if(mag_of_grad > EPS)
    {
      // compute second derivatives
      double dxxyyzz[P4EST_DIM];
      if(phi_xxyyzz_block_p != NULL || phi_xxyyzz_p != NULL){
        for (u_char der = 0; der < P4EST_DIM; ++der)
          dxxyyzz[der] = (phi_xxyyzz_block_p != NULL ? phi_xxyyzz_block_p[P4EST_DIM*node_000 + der] : phi_xxyyzz_p[der][node_000]);
      }
      else
        laplace(phi_p, dxxyyzz);

      const double dxy = (grad_phi_block_p != NULL ? dy_central_component(grad_phi_block_p, P4EST_DIM, dir::x) : dy_central(grad_phi_p[0])); // d/dy{d/dx}
#ifdef P4_TO_P8
      const double dxz = (grad_phi_block_p != NULL ? dz_central_component(grad_phi_block_p, P4EST_DIM, dir::x) : dz_central(grad_phi_p[0])); // d/dz{d/dx}
      const double dyz = (grad_phi_block_p != NULL ? dz_central_component(grad_phi_block_p, P4EST_DIM, dir::y) : dz_central(grad_phi_p[1])); // d/dz{d/dy}
#endif
      return ((dxxyyzz[1] ONLY3D(+ dxxyyzz[2]))*SQR(dx) + (dxxyyzz[0] ONLY3D(+ dxxyyzz[2]))*SQR(dy) ONLY3D(+ (dxxyyzz[0] + dxxyyzz[1])*SQR(dz)) -
          2.0*(dx*dy*dxy ONLY3D(+ dx*dz*dxz + dy*dz*dyz))) / (mag_of_grad*mag_of_grad*mag_of_grad);
    }
    else
      return 0.0; // nothing better to suggest for now, sorry
  }

  inline double get_curvature(const double *grad_phi_p, const double *phi_p, const double *phi_xxyyzz_p = NULL) const
  {
    return get_curvature(phi_p, grad_phi_p, NULL, phi_xxyyzz_p, NULL);
  }

  inline double get_curvature(const double *grad_phi_p[P4EST_DIM], const double *phi_p, const double *phi_xxyyzz_p[P4EST_DIM] = NULL) const
  {
    return get_curvature(phi_p, NULL, grad_phi_p, NULL, phi_xxyyzz_p);
  }

  /*!
   * \brief get_curvature evaluates the local mean curvature as divergence of (normals). It is the user's responsibility
   * to provide the ***normal vector*** (i.e. the normalized gradient, not the standard gradient)
   * The user MUST provide the normal vectors (in a component by component) fashion to enable local calculation
   * \param normals pointer to an array of size P4EST_DIM of the normals. CANNOT be NULL.
   * \return mean curvature at a single point
   */
  inline double get_curvature(const double* normals_p[P4EST_DIM]) const
  {
    P4EST_ASSERT(normals_p != NULL);
    return SUMD(dx_central(normals_p[0]), dy_central(normals_p[1]), dz_central(normals_p[2]));
  }

  // biased-first-derivative-related procedures
  // based on linear-interpolation neighbors
  inline double dx_forward_linear (const double *f) const
  {
    return forward_derivative(f_p00_linear(f), f[node_000], d_p00);
  }
  inline double dx_backward_linear (const double *f) const
  {
    return backward_derivative(f[node_000], f_m00_linear(f), d_m00);
  }
  inline double dy_forward_linear (const double *f) const
  {
    return forward_derivative(f_0p0_linear(f), f[node_000], d_0p0);
  }
  inline double dy_backward_linear (const double *f) const
  {
    return backward_derivative(f[node_000], f_0m0_linear(f), d_0m0);
  }
#ifdef P4_TO_P8
  inline double dz_forward_linear (const double *f) const
  {
    return forward_derivative(f_00p_linear(f), f[node_000], d_00p);
  }
  inline double dz_backward_linear (const double *f) const
  {
    return backward_derivative(f[node_000], f_00m_linear(f), d_00m);
  }
#endif
  inline double d_forward_linear(const unsigned short &der, const double *f) const
  {
    return (der == dir::x ? dx_forward_linear(f) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_forward_linear(f) ONLY3D(: dz_forward_linear(f) CLOSE_PARENTHESIS));
  }
  inline double d_backward_linear(const unsigned short &der, const double *f) const
  {
    return (der == dir::x ? dx_backward_linear(f) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_backward_linear(f) ONLY3D(: dz_backward_linear(f) CLOSE_PARENTHESIS));
  }
  // based on quadratic-interpolation neighbors
  double dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const;
  double dx_forward_quadratic (const double *f, const my_p4est_node_neighbors_t &neighbors) const;
  double dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const;
  double dy_forward_quadratic (const double *f, const my_p4est_node_neighbors_t &neighbors) const;
#ifdef P4_TO_P8
  double dz_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const;
  double dz_forward_quadratic (const double *f, const my_p4est_node_neighbors_t &neighbors) const;
#endif
  inline double d_backward_quadratic(const unsigned short &der, const double *f, const my_p4est_node_neighbors_t &neighbors) const
  {
    return (der == dir::x ? dx_backward_quadratic(f, neighbors) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_backward_quadratic(f, neighbors) ONLY3D(: dz_backward_quadratic(f, neighbors) CLOSE_PARENTHESIS));
  }
  inline double d_forward_quadratic(const unsigned short &der, const double *f, const my_p4est_node_neighbors_t &neighbors) const
  {
    return (der == dir::x ? dx_forward_quadratic(f, neighbors) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_forward_quadratic(f, neighbors) ONLY3D(: dz_forward_quadratic(f, neighbors) CLOSE_PARENTHESIS));
  }

  // VERY IMPORTANT NOTE: in the following, we assume fxx, fyy and fzz to have the same block structure as f!
  double dx_backward_quadratic(const double *f, const double *fxx) const
  {
    double f_000, f_m00, f_p00;
    x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);
    const double f_xx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
    const double f_xx_m00 = f_m00_linear(fxx);
    return d_backward_quadratic(f_000, f_m00, d_m00, f_xx_000, f_xx_m00);
  }
  double dx_forward_quadratic (const double *f, const double *fxx) const
  {
    double f_000, f_m00, f_p00;
    x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);
    const double f_xx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
    const double f_xx_p00 = f_p00_linear(fxx);
    return d_forward_quadratic(f_p00, f_000, d_p00, f_xx_000, f_xx_p00);
  }
  double dy_backward_quadratic(const double *f, const double *fyy) const
  {
    double f_000, f_0m0, f_0p0;
    y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);
    const double f_yy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
    const double f_yy_0m0 = f_0m0_linear(fyy);
    return d_backward_quadratic(f_000, f_0m0, d_0m0, f_yy_000, f_yy_0m0);
  }
  double dy_forward_quadratic (const double *f, const double *fyy) const
  {
    double f_000, f_0m0, f_0p0;
    y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);
    const double f_yy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
    const double f_yy_0p0 = f_0p0_linear(fyy);
    return d_forward_quadratic(f_0p0, f_000, d_0p0, f_yy_000, f_yy_0p0);
  }
#ifdef P4_TO_P8
  double dz_backward_quadratic(const double *f, const double *fzz) const
  {
    double f_000, f_00m, f_00p;
    z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);
    const double f_zz_000 = central_second_derivative(f_00p, f_000, f_00m, d_00p, d_00m);
    const double f_zz_00m = f_00m_linear(fzz);
    return d_backward_quadratic(f_000, f_00m, d_00m, f_zz_000, f_zz_00m);
  }
  double dz_forward_quadratic (const double *f, const double *fzz) const
  {
    double f_000, f_00m, f_00p;
    z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);
    const double f_zz_000 = central_second_derivative(f_00p, f_000, f_00m, d_00p, d_00m);
    const double f_zz_00p = f_00p_linear(fzz);
    return d_forward_quadratic(f_00p, f_000, d_00p, f_zz_000, f_zz_00p);
  }
#endif
  inline double d_backward_quadratic(const unsigned short &der, const double *f, const double *fderder) const
  {
    return (der == dir::x ? dx_backward_quadratic(f, fderder) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_backward_quadratic(f, fderder) ONLY3D(: dz_backward_quadratic(f, fderder) CLOSE_PARENTHESIS));
  }
  inline double d_forward_quadratic(const unsigned short &der, const double *f, const double *fderder) const
  {
    return (der == dir::x ? dx_forward_quadratic(f, fderder) : ONLY3D(OPEN_PARENTHESIS der == dir::y ?) dy_forward_quadratic(f, fderder) ONLY3D(: dz_forward_quadratic(f, fderder) CLOSE_PARENTHESIS));
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


  inline p4est_locidx_t neighbor_m00() const
  {
    const double max_hy = MAX(d_0m0, d_0p0);
#ifdef P4_TO_P8
    const double max_hz = MAX(d_00m, d_00p);
#endif
    if      (check_if_zero(d_m00_p0/max_hy) ONLY3D(&& check_if_zero(d_m00_0m/max_hz)))  return node_m00_pm;
    else if (check_if_zero(d_m00_m0/max_hy) ONLY3D(&& check_if_zero(d_m00_0m/max_hz)))  return node_m00_mm;
#ifdef P4_TO_P8
    else if (check_if_zero(d_m00_m0/max_hy)        && check_if_zero(d_m00_0p/max_hz))   return node_m00_mp;
    else if (check_if_zero(d_m00_p0/max_hy)        && check_if_zero(d_m00_0p/max_hz))   return node_m00_pp;
#endif
    else return -1;
    //  else            throw std::invalid_argument("No neighbor in m00 direction \n");
  }
  inline p4est_locidx_t neighbor_p00() const
  {
    const double max_hy = MAX(d_0m0, d_0p0);
#ifdef P4_TO_P8
    const double max_hz = MAX(d_00m, d_00p);
#endif
    if      (check_if_zero(d_p00_m0/max_hy) ONLY3D(&& check_if_zero(d_p00_0m/max_hz)))  return node_p00_mm;
    else if (check_if_zero(d_p00_p0/max_hy) ONLY3D(&& check_if_zero(d_p00_0m/max_hz)))  return node_p00_pm;
#ifdef P4_TO_P8
    else if (check_if_zero(d_p00_m0/max_hy)        && check_if_zero(d_p00_0p/max_hz))   return node_p00_mp;
    else if (check_if_zero(d_p00_p0/max_hy)        && check_if_zero(d_p00_0p/max_hz))   return node_p00_pp;
#endif
    else return -1;
    //    else            throw std::invalid_argument("No neighbor in p00 direction \n");
  }
  inline p4est_locidx_t neighbor_0m0() const
  {
    const double max_hx = MAX(d_m00, d_p00);
#ifdef P4_TO_P8
    const double max_hz = MAX(d_00m, d_00p);
#endif
    if      (check_if_zero(d_0m0_m0/max_hx) ONLY3D(&& check_if_zero(d_0m0_0m/max_hz)))  return node_0m0_mm;
    else if (check_if_zero(d_0m0_p0/max_hx) ONLY3D(&& check_if_zero(d_0m0_0m/max_hz)))  return node_0m0_pm;
#ifdef P4_TO_P8
    else if (check_if_zero(d_0m0_m0/max_hx)        && check_if_zero(d_0m0_0p/max_hz))   return node_0m0_mp;
    else if (check_if_zero(d_0m0_p0/max_hx)        && check_if_zero(d_0m0_0p/max_hz))   return node_0m0_pp;
#endif
    else return -1;
    //    else            throw std::invalid_argument("No neighbor in 0m0 direction \n");
  }
  inline p4est_locidx_t neighbor_0p0() const
  {
    const double max_hx = MAX(d_m00, d_p00);
#ifdef P4_TO_P8
    const double max_hz = MAX(d_00m, d_00p);
#endif
    if      (check_if_zero(d_0p0_m0/max_hx) ONLY3D(&& check_if_zero(d_0p0_0m/max_hz)))  return node_0p0_mm;
    else if (check_if_zero(d_0p0_p0/max_hx) ONLY3D(&& check_if_zero(d_0p0_0m/max_hz)))  return node_0p0_pm;
#ifdef P4_TO_P8
    else if (check_if_zero(d_0p0_m0/max_hx)        && check_if_zero(d_0p0_0p/max_hz))   return node_0p0_mp;
    else if (check_if_zero(d_0p0_p0/max_hx)        && check_if_zero(d_0p0_0p/max_hz))   return node_0p0_pp;
#endif
    else return -1;
    //    else            throw std::invalid_argument("No neighbor in 0p0 direction \n");
  }
#ifdef P4_TO_P8
  inline p4est_locidx_t neighbor_00m() const
  {
    const double max_hx = MAX(d_m00, d_p00);
    const double max_hy = MAX(d_0m0, d_0p0);
    if      (check_if_zero(d_00m_m0/max_hx) && check_if_zero(d_00m_0m/max_hy)) return node_00m_mm;
    else if (check_if_zero(d_00m_p0/max_hx) && check_if_zero(d_00m_0m/max_hy)) return node_00m_pm;
    else if (check_if_zero(d_00m_m0/max_hx) && check_if_zero(d_00m_0p/max_hy)) return node_00m_mp;
    else if (check_if_zero(d_00m_p0/max_hx) && check_if_zero(d_00m_0p/max_hy)) return node_00m_pp;
    else return -1;
    //  else throw std::invalid_argument("No neighbor in m00 direction \n");
  }
  inline p4est_locidx_t neighbor_00p() const
  {
    const double max_hx = MAX(d_m00, d_p00);
    const double max_hy = MAX(d_0m0, d_0p0);
    if      (check_if_zero(d_00p_m0/max_hx) && check_if_zero(d_00p_0m/max_hy)) return node_00p_mm;
    else if (check_if_zero(d_00p_p0/max_hx) && check_if_zero(d_00p_0m/max_hy)) return node_00p_pm;
    else if (check_if_zero(d_00p_m0/max_hx) && check_if_zero(d_00p_0p/max_hy)) return node_00p_mp;
    else if (check_if_zero(d_00p_p0/max_hx) && check_if_zero(d_00p_0p/max_hy)) return node_00p_pp;
    else return -1;
    //  else throw std::invalid_argument("No neighbor in m00 direction \n");
  }
#endif

  inline p4est_locidx_t neighbor(int dir) const
  {
    switch (dir)
    {
    case dir::f_m00: return neighbor_m00();
    case dir::f_p00: return neighbor_p00();
    case dir::f_0m0: return neighbor_0m0();
    case dir::f_0p0: return neighbor_0p0();
#ifdef P4_TO_P8
    case dir::f_00m: return neighbor_00m();
    case dir::f_00p: return neighbor_00p();
#endif
    default: throw std::invalid_argument("Invalid direction\n");
    }
  }
  inline double distance(int dir) const
  {
    switch (dir)
    {
    case dir::f_m00: return d_m00;
    case dir::f_p00: return d_p00;
    case dir::f_0m0: return d_0m0;
    case dir::f_0p0: return d_0p0;
#ifdef P4_TO_P8
    case dir::f_00m: return d_00m;
    case dir::f_00p: return d_00p;
#endif
    default: throw std::invalid_argument("Invalid direction\n");
    }
  }

  inline bool is_stencil_in_negative_domain(double *phi_p) const
  {
    return phi_p[this->node_000] < -EPS &&
    #ifdef P4_TO_P8
        ( phi_p[this->node_m00_mm] < -EPS || fabs(this->d_m00_p0) < EPS || fabs(this->d_m00_0p) < EPS) &&
        ( phi_p[this->node_m00_mp] < -EPS || fabs(this->d_m00_p0) < EPS || fabs(this->d_m00_0m) < EPS) &&
        ( phi_p[this->node_m00_pm] < -EPS || fabs(this->d_m00_m0) < EPS || fabs(this->d_m00_0p) < EPS) &&
        ( phi_p[this->node_m00_pp] < -EPS || fabs(this->d_m00_m0) < EPS || fabs(this->d_m00_0m) < EPS) &&
        ( phi_p[this->node_p00_mm] < -EPS || fabs(this->d_p00_p0) < EPS || fabs(this->d_p00_0p) < EPS) &&
        ( phi_p[this->node_p00_mp] < -EPS || fabs(this->d_p00_p0) < EPS || fabs(this->d_p00_0m) < EPS) &&
        ( phi_p[this->node_p00_pm] < -EPS || fabs(this->d_p00_m0) < EPS || fabs(this->d_p00_0p) < EPS) &&
        ( phi_p[this->node_p00_pp] < -EPS || fabs(this->d_p00_m0) < EPS || fabs(this->d_p00_0m) < EPS) &&
        ( phi_p[this->node_0m0_mm] < -EPS || fabs(this->d_0m0_p0) < EPS || fabs(this->d_0m0_0p) < EPS) &&
        ( phi_p[this->node_0m0_mp] < -EPS || fabs(this->d_0m0_p0) < EPS || fabs(this->d_0m0_0m) < EPS) &&
        ( phi_p[this->node_0m0_pm] < -EPS || fabs(this->d_0m0_m0) < EPS || fabs(this->d_0m0_0p) < EPS) &&
        ( phi_p[this->node_0m0_pp] < -EPS || fabs(this->d_0m0_m0) < EPS || fabs(this->d_0m0_0m) < EPS) &&
        ( phi_p[this->node_0p0_mm] < -EPS || fabs(this->d_0p0_p0) < EPS || fabs(this->d_0p0_0p) < EPS) &&
        ( phi_p[this->node_0p0_mp] < -EPS || fabs(this->d_0p0_p0) < EPS || fabs(this->d_0p0_0m) < EPS) &&
        ( phi_p[this->node_0p0_pm] < -EPS || fabs(this->d_0p0_m0) < EPS || fabs(this->d_0p0_0p) < EPS) &&
        ( phi_p[this->node_0p0_pp] < -EPS || fabs(this->d_0p0_m0) < EPS || fabs(this->d_0p0_0m) < EPS) &&
        ( phi_p[this->node_00m_mm] < -EPS || fabs(this->d_00m_p0) < EPS || fabs(this->d_00m_0p) < EPS) &&
        ( phi_p[this->node_00m_mp] < -EPS || fabs(this->d_00m_p0) < EPS || fabs(this->d_00m_0m) < EPS) &&
        ( phi_p[this->node_00m_pm] < -EPS || fabs(this->d_00m_m0) < EPS || fabs(this->d_00m_0p) < EPS) &&
        ( phi_p[this->node_00m_pp] < -EPS || fabs(this->d_00m_m0) < EPS || fabs(this->d_00m_0m) < EPS) &&
        ( phi_p[this->node_00p_mm] < -EPS || fabs(this->d_00p_p0) < EPS || fabs(this->d_00p_0p) < EPS) &&
        ( phi_p[this->node_00p_mp] < -EPS || fabs(this->d_00p_p0) < EPS || fabs(this->d_00p_0m) < EPS) &&
        ( phi_p[this->node_00p_pm] < -EPS || fabs(this->d_00p_m0) < EPS || fabs(this->d_00p_0p) < EPS) &&
        ( phi_p[this->node_00p_pp] < -EPS || fabs(this->d_00p_m0) < EPS || fabs(this->d_00p_0m) < EPS);
#else
        ( phi_p[this->node_m00_mm] < -EPS || fabs(this->d_m00_p0) < EPS) &&
        ( phi_p[this->node_m00_pm] < -EPS || fabs(this->d_m00_m0) < EPS) &&
        ( phi_p[this->node_p00_mm] < -EPS || fabs(this->d_p00_p0) < EPS) &&
        ( phi_p[this->node_p00_pm] < -EPS || fabs(this->d_p00_m0) < EPS) &&
        ( phi_p[this->node_0m0_mm] < -EPS || fabs(this->d_0m0_p0) < EPS) &&
        ( phi_p[this->node_0m0_pm] < -EPS || fabs(this->d_0m0_m0) < EPS) &&
        ( phi_p[this->node_0p0_mm] < -EPS || fabs(this->d_0p0_p0) < EPS) &&
        ( phi_p[this->node_0p0_pm] < -EPS || fabs(this->d_0p0_m0) < EPS);
#endif
  }

  inline double interpolate_in_dir(int dir, double dist, double *f_ptr) const
  {
    p4est_locidx_t node_nei = neighbor(dir);
    double         h   = distance(dir);
    if (node_nei == -1) throw std::domain_error("interpolate_in_dir does not support non-uniform grids yet\n");
    return f_ptr[node_000]*(1-dist/h) + f_ptr[node_nei]*dist/h;
  }

  inline double interpolate_in_dir(int dir, double dist, double *f_ptr, double *f_dd_ptr) const
  {
    p4est_locidx_t node_nei = neighbor(dir);
    double         h   = distance(dir);
    if (node_nei == -1) throw std::domain_error("interpolate_in_dir doesn not support non-uniform grids yet\n");
    return f_ptr[node_000]*(1-dist/h) + f_ptr[node_nei]*dist/h + 0.5*dist*(dist-h)*MINMOD(f_dd_ptr[node_000], f_dd_ptr[node_nei]);
  }

  inline double interpolate_in_dir(int dir, double dist, double *f_ptr, double *f_dd_ptr[]) const
  {
    p4est_locidx_t node_nei = neighbor(dir);
    double         h   = distance(dir);
    int            dim = dir / 2;
    if (node_nei == -1) throw std::domain_error("interpolate_in_dir does not support non-uniform grids yet\n");
    return f_ptr[node_000]*(1-dist/h) + f_ptr[node_nei]*dist/h + 0.5*dist*(dist-h)*MINMOD(f_dd_ptr[dim][node_000], f_dd_ptr[dim][node_nei]);
  }
};

#endif /* !MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H */
