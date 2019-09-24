#include "my_p4est_quad_neighbor_nodes_of_node.h"
#include <src/my_p4est_node_neighbors.h>
#include <petsclog.h>

#ifndef CASL_LOG_TINY_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
#warning "Use of 'CASL_LOG_TINY_EVENTS' macro is discouraged but supported. Logging tiny sections of the code may produce unreliable results due to overhead."
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate_dd;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_laplace_core;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation_core;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_gradient_core;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_assemble_first_derivative_operators;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_gradient;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double node_linear_combination::calculate_dd( const unsigned char der, const double *node_sample_field, const my_p4est_node_neighbors_t &neighbors, const unsigned int &bs, const unsigned int &comp) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate_dd, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(der < P4EST_DIM);
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(elements.size()>0);
  double value = elements[0].weight*(neighbors.get_neighbors(elements[0].node_idx).dd_central(der, node_sample_field, bs, comp));
  for (size_t k = 1; k < elements.size(); ++k)
    value += elements[k].weight*(neighbors.get_neighbors(elements[k].node_idx).dd_central(der, node_sample_field, bs, comp));
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate_dd, 0, 0, 0, 0); CHKERRXX(ierr);
  return value;
}

#ifdef P4_TO_P8
template<typename T> inline T quad_neighbor_nodes_of_node_t::linear_interpolator(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm, const p4est_locidx_t &node_idx_mp, const p4est_locidx_t &node_idx_pp,
                                                                                 const double &d_m0, const double &d_p0, const double &d_0m, const double &d_0p,
                                                                                 const double *f, const unsigned int &bs, const unsigned int &comp) const
#else
template<typename T> inline T quad_neighbor_nodes_of_node_t::linear_interpolator(const p4est_locidx_t &node_idx_mm, const p4est_locidx_t &node_idx_pm,
                                                                                 const double &d_m0, const double &d_p0,
                                                                                 const double *f, const unsigned int &bs, const unsigned int &comp) const
#endif
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_linear_interpolator, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef P4_TO_P8
  unsigned char n_elem = (1<<(P4EST_DIM - 1 - ((check_if_zero(d_p0*inverse_d_max) || check_if_zero(d_m0*inverse_d_max))?1:0) - ((check_if_zero(d_0p*inverse_d_max) || check_if_zero(d_0m*inverse_d_max))?1:0)));
#else
  unsigned char n_elem = (1<<(P4EST_DIM - 1 - ((check_if_zero(d_p0*inverse_d_max) || check_if_zero(d_m0*inverse_d_max))?1:0)));
#endif
  T accumulator = initialize_accumulator<T>(n_elem);
  unsigned char idx=0; double sum_of_weights = 0.0; double ww;
#ifdef P4_TO_P8
  double weight1;
  double weight2;
  /* (f[node_idx_mm]*d_p0*d_0p +
     *  f[node_idx_mp]*d_p0*d_0m +
     *  f[node_idx_pm]*d_m0*d_0p +
     *  f[node_idx_pp]*d_m0*d_0m )/(d_m0 + d_p0)/(d_0m + d_0p);
     */
  weight1     = d_m0/(d_m0 + d_p0);
  weight2     = d_0m/(d_0m + d_0p);
  if(!check_if_zero(weight1) && !check_if_zero(weight2))
  {
    ww = (++idx==n_elem)? (1.0-sum_of_weights) : weight1*weight2;
    sum_of_weights += ww;
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(8); CHKERRXX(ierr); // worst case scenario
#endif
    add_to_accumulator<T>(accumulator, node_idx_pp, ww, f, bs, comp);
  }
  weight1     = d_m0/(d_m0 + d_p0);
  weight2     = d_0p/(d_0m + d_0p);
  if(!check_if_zero(weight1) && !check_if_zero(weight2))
  {
    ww = (++idx==n_elem)? (1.0-sum_of_weights) : weight1*weight2;
    sum_of_weights += ww;
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(8); CHKERRXX(ierr); // worst case scenario
#endif
    add_to_accumulator<T>(accumulator, node_idx_pm, ww, f, bs, comp);
  }
  weight1     = d_p0/(d_m0 + d_p0);
  weight2     = d_0m/(d_0m + d_0p);
  if(!check_if_zero(weight1) && !check_if_zero(weight2))
  {
    ww = (++idx==n_elem)? (1.0-sum_of_weights) : weight1*weight2;
    sum_of_weights += ww;
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(8); CHKERRXX(ierr); // worst case scenario
#endif
    add_to_accumulator<T>(accumulator, node_idx_mp, ww, f, bs, comp);
  }
  weight1     = d_p0/(d_m0 + d_p0);
  weight2     = d_0p/(d_0m + d_0p);
  if(!check_if_zero(weight1) && !check_if_zero(weight2))
  {
    ww = (++idx==n_elem)? (1.0-sum_of_weights) : weight1*weight2;
    sum_of_weights +=ww;
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(8); CHKERRXX(ierr); // worst case scenario
#endif
    add_to_accumulator<T>(accumulator, node_idx_mm, ww, f, bs, comp);
  }
#else
  /* (f[node_idx_mm]*d_p0 +
     *  f[node_idx_pm]*d_m0)/(d_m0+d_p0);
     */
  ww = d_p0/(d_m0+d_p0);
  if(!check_if_zero(ww))
  {
    if (++idx==n_elem)
      ww = (1.0-sum_of_weights);
    sum_of_weights +=ww;
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(6); CHKERRXX(ierr); // worst case scenario
#endif
    add_to_accumulator<T>(accumulator, node_idx_mm, ww, f, bs, comp);

  }
  ww = d_m0/(d_m0+d_p0);
  if(!check_if_zero(d_m0*inverse_d_max))
  {
    if (++idx==n_elem)
      ww = (1.0-sum_of_weights);
    sum_of_weights +=ww;
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(6); CHKERRXX(ierr); // worst case scenario
#endif
    add_to_accumulator<T>(accumulator, node_idx_pm, ww, f, bs, comp);
  }
#endif
  P4EST_ASSERT(idx==n_elem && check_if_zero(sum_of_weights - 1.0));
#if __cplusplus >=201103L
  shrink_accumulator<T>(accumulator);
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_linear_interpolator, 0, 0, 0, 0); CHKERRXX(ierr);
  return accumulator;
}

template<> double quad_neighbor_nodes_of_node_t::f_000_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  // this one is a real dummy, so shortcut the intellectual masturbation here
  P4EST_ASSERT(f!=NULL);
  return f[bs*node_000+comp];
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_000_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_000);
  // this one is a real dummy, so shortcut the intellectual masturbation and build it right away
  node_linear_combination lc_tool;
  lc_tool += node_interpolation_weight(node_000, 1.0);
  return lc_tool;
}
template<> double quad_neighbor_nodes_of_node_t::f_m00_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(linear_interpolators_are_set)
    return linear_interpolator_m00->calculate(f, bs, comp);
#ifdef P4_TO_P8
  return linear_interpolator<double>(node_m00_mm, node_m00_pm, node_m00_mp, node_m00_pp, d_m00_m0, d_m00_p0, d_m00_0m, d_m00_0p, f, bs, comp);
#else
  return linear_interpolator<double>(node_m00_mm, node_m00_pm,                           d_m00_m0, d_m00_p0,                     f, bs, comp);
#endif
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_m00_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_m00);
#ifdef P4_TO_P8
  return linear_interpolator<node_linear_combination>(node_m00_mm, node_m00_pm, node_m00_mp, node_m00_pp, d_m00_m0, d_m00_p0, d_m00_0m, d_m00_0p);
#else
  return linear_interpolator<node_linear_combination>(node_m00_mm, node_m00_pm,                           d_m00_m0, d_m00_p0);
#endif
}
template<> double quad_neighbor_nodes_of_node_t::f_p00_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(linear_interpolators_are_set)
    return linear_interpolator_p00->calculate(f, bs, comp);
#ifdef P4_TO_P8
  return linear_interpolator<double>(node_p00_mm, node_p00_pm, node_p00_mp, node_p00_pp, d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p, f, bs, comp);
#else
  return linear_interpolator<double>(node_p00_mm, node_p00_pm,                           d_p00_m0, d_p00_p0,                     f, bs, comp);
#endif
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_p00_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_p00);
#ifdef P4_TO_P8
  return linear_interpolator<node_linear_combination>(node_p00_mm, node_p00_pm, node_p00_mp, node_p00_pp, d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p);
#else
  return linear_interpolator<node_linear_combination>(node_p00_mm, node_p00_pm,                           d_p00_m0, d_p00_p0);
#endif
}
template<> double quad_neighbor_nodes_of_node_t::f_0m0_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(linear_interpolators_are_set)
    return linear_interpolator_0m0->calculate(f, bs, comp);
#ifdef P4_TO_P8
  return linear_interpolator<double>(node_0m0_mm, node_0m0_pm, node_0m0_mp, node_0m0_pp, d_0m0_m0, d_0m0_p0, d_0m0_0m, d_0m0_0p, f, bs, comp);
#else
  return linear_interpolator<double>(node_0m0_mm, node_0m0_pm,                           d_0m0_m0, d_0m0_p0,                     f, bs, comp);
#endif
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_0m0_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_0m0);
#ifdef P4_TO_P8
  return linear_interpolator<node_linear_combination>(node_0m0_mm, node_0m0_pm, node_0m0_mp, node_0m0_pp, d_0m0_m0, d_0m0_p0, d_0m0_0m, d_0m0_0p);
#else
  return linear_interpolator<node_linear_combination>(node_0m0_mm, node_0m0_pm,                           d_0m0_m0, d_0m0_p0);
#endif
}
template<> double quad_neighbor_nodes_of_node_t::f_0p0_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(linear_interpolators_are_set)
    return linear_interpolator_0p0->calculate(f, bs, comp);
#ifdef P4_TO_P8
  return linear_interpolator<double>(node_0p0_mm, node_0p0_pm, node_0p0_mp, node_0p0_pp, d_0p0_m0, d_0p0_p0, d_0p0_0m, d_0p0_0p, f, bs, comp);
#else
  return linear_interpolator<double>(node_0p0_mm, node_0p0_pm,                           d_0p0_m0, d_0p0_p0,                     f, bs, comp);
#endif
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_0p0_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_0p0);
#ifdef P4_TO_P8
  return linear_interpolator<node_linear_combination>(node_0p0_mm, node_0p0_pm, node_0p0_mp, node_0p0_pp, d_0p0_m0, d_0p0_p0, d_0p0_0m, d_0p0_0p);
#else
  return linear_interpolator<node_linear_combination>(node_0p0_mm, node_0p0_pm,                           d_0p0_m0, d_0p0_p0);
#endif
}
#ifdef P4_TO_P8
template<> double quad_neighbor_nodes_of_node_t::f_00m_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(linear_interpolators_are_set)
    return linear_interpolator_00m->calculate(f, bs, comp);
  return linear_interpolator<double>(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p, f, bs, comp);
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_00m_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_00m);
  return linear_interpolator<node_linear_combination>(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p);
}
template<> double quad_neighbor_nodes_of_node_t::f_00p_linear(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(linear_interpolators_are_set)
    return linear_interpolator_00p->calculate(f, bs, comp);
  return linear_interpolator<double>(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p, f, bs, comp);
}
template<> node_linear_combination quad_neighbor_nodes_of_node_t::f_00p_linear(const double *, const unsigned int &, const unsigned int &) const
{
  if(linear_interpolators_are_set)
    return (*linear_interpolator_00p);
  return linear_interpolator<node_linear_combination>(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p);
}
#endif

#ifdef P4_TO_P8
template<typename T> void quad_neighbor_nodes_of_node_t::linearly_interpolated_neighbors(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0, T &f_00m, T &f_00p,
                                                                                         const unsigned int &bs, const unsigned int &comp) const
#else
template<typename T> void quad_neighbor_nodes_of_node_t::linearly_interpolated_neighbors(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0,
                                                                                         const unsigned int &bs, const unsigned int &comp) const
#endif
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  f_000 = f_000_linear<T>(f, bs, comp);
  f_p00 = f_p00_linear<T>(f, bs, comp); f_m00 = f_m00_linear<T>(f, bs, comp);
  f_0p0 = f_0p0_linear<T>(f, bs, comp); f_0m0 = f_0m0_linear<T>(f, bs, comp);
#ifdef P4_TO_P8
  f_00p = f_00p_linear<T>(f, bs, comp); f_00m = f_00m_linear<T>(f, bs, comp);
#endif
  return;
}

#ifdef P4_TO_P8
template<typename T> void quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(const T naive_Dxx, const T naive_Dyy, const T naive_Dzz, T &Dxx, T &Dyy, T &Dzz) const
#else
template<typename T> void quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(const T naive_Dxx, const T naive_Dyy, T &Dxx, T &Dyy) const
#endif
{
  const double m12 = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ; // 9 flops
#ifdef P4_TO_P8
  const double m13 = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ; // 9 flops
#endif
  const double m21 = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ; // 9 flops
#ifdef P4_TO_P8
  const double m23 = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ; // 9 flops
  const double m31 = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ; // 9 flops
  const double m32 = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ; // 9 flops
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr = PetscLogFlops(54); CHKERRXX(ierr);
#endif
  // naive_Dfxx = 1.0*fxx + m12*fyy + m13*fzz
  // naive_Dfyy = m21*fxx + 1.0*fyy + m23*fzz
  // naive_Dfzz = m31*fxx + m32*fyy + 1.0*fzz
  // either (m12 and m13) are 0.0 or (m21 and m23) are 0.0 or (m31 and m32) are 0.0
  P4EST_ASSERT((check_if_zero(m12) && check_if_zero(m13)) || (check_if_zero(m21) && check_if_zero(m23)) || (check_if_zero(m31) && check_if_zero(m32)));
  if (check_if_zero(m12) && check_if_zero(m13))
  {
    Dxx = naive_Dxx;
    // now, either m23 or m32 or both must 0.0
    P4EST_ASSERT(check_if_zero(m23) || check_if_zero(m32));
    if(check_if_zero(m23))
    {
      Dyy = naive_Dyy;
      if(!check_if_zero(m21))
        Dyy -= m21*Dxx;
      Dzz = naive_Dzz;
      if(!check_if_zero(m31))
        Dzz -= m31*Dxx;
      if(!check_if_zero(m32))
        Dzz -= m32*Dyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
    else if (check_if_zero(m32))
    {
      Dzz = naive_Dzz;
      if(!check_if_zero(m31))
        Dzz -= m31*Dxx;
      // m23 is non-zero, otherwise the if statement above would have been entered
      Dyy = naive_Dyy - m23*Dzz;
      if(!check_if_zero(m21))
        Dyy -= m21*Dxx;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
    else
    {
      // this is supposed to NEVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
      std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement a." << std::endl;
      const double idet = 1.0/(1.0-m23*m32);
      Dyy = ((check_if_zero(m21)? naive_Dyy : (naive_Dyy-m21*Dxx)) - m23*(check_if_zero(m31)? naive_Dzz : (naive_Dzz-m31*Dxx)))*idet;
      Dzz = ((check_if_zero(m31)? naive_Dzz : (naive_Dzz-m31*Dxx)) - m32*(check_if_zero(m21)? naive_Dyy : (naive_Dyy-m21*Dxx)))*idet;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(3+(std::is_floating_point<T>::value?14:0)); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
  }
  else if (check_if_zero(m21) && check_if_zero(m23))
  {
    Dyy = naive_Dyy;
    // now, either m13 or m31 or both must 0.0
    P4EST_ASSERT(check_if_zero(m13) || check_if_zero(m31));
    if(check_if_zero(m13))
    {
      Dxx = naive_Dxx;
      if(!check_if_zero(m12))
        Dxx -= m12*Dyy;
      Dzz = naive_Dzz;
      if(!check_if_zero(m31))
        Dzz -= m31*Dxx;
      if(!check_if_zero(m32))
        Dzz -= m32*Dyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
    else if (check_if_zero(m31))
    {
      Dzz = naive_Dzz;
      if(!check_if_zero(m32))
        Dzz -= m32*Dyy;
      // m13 is non-zero, otherwise the if statement above would have been entered
      Dxx = naive_Dxx - m13*Dzz;
      if(!check_if_zero(m12))
        Dxx -= m12*Dyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
    else
    {
      // this is supposed to NEVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
      std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement b." << std::endl;
      const double idet = 1.0/(1.0-m13*m31);
      Dxx = ((check_if_zero(m12)? naive_Dxx : (naive_Dxx-m12*Dyy)) - m13*(check_if_zero(m32)? naive_Dzz : (naive_Dzz-m32*Dyy)))*idet;
      Dzz = ((check_if_zero(m32)? naive_Dzz : (naive_Dzz-m32*Dyy)) - m31*(check_if_zero(m12)? naive_Dxx : (naive_Dxx-m12*Dyy)))*idet;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(3+(std::is_floating_point<T>::value?14:0)); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
  }
  else if (check_if_zero(m31) && check_if_zero(m32))
  {
    Dzz = naive_Dzz;
    // now, either m12 or m21 or both must 0.0
    P4EST_ASSERT(check_if_zero(m12) || check_if_zero(m21));
    if(check_if_zero(m12))
    {
      Dxx = naive_Dxx;
      if(!check_if_zero(m13))
        Dxx -= m13*Dzz;
      Dyy = naive_Dyy;
      if(!check_if_zero(m21))
        Dyy -= m21*Dxx;
      if(!check_if_zero(m23))
        Dyy -= m23*Dzz;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
    else if (check_if_zero(m21))
    {
      Dyy = naive_Dyy;
      if(!check_if_zero(m23))
        Dyy -= m23*Dzz;
      // m12 is non-zero, otherwise the if statement above would have been entered
      Dxx = naive_Dxx - m12*Dyy;
      if(!check_if_zero(m13))
        Dxx -= m13*Dzz;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
    else
    {
      // this is supposed to NEVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
      std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement c." << std::endl;
      const double idet = 1.0/(1.0-m12*m21);
      Dxx = ((check_if_zero(m13)? naive_Dxx : (naive_Dxx-m13*Dzz)) - m12*(check_if_zero(m23)? naive_Dyy : (naive_Dyy-m23*Dzz)))*idet;
      Dzz = ((check_if_zero(m23)? naive_Dyy : (naive_Dyy-m23*Dzz)) - m21*(check_if_zero(m13)? naive_Dxx : (naive_Dxx-m13*Dzz)))*idet;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?6:0); CHKERRXX(ierr); // (upper bound), flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
  }
  else
  {
    // this is supposed to NEVER-EVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
    std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement D, this really sucks!" << std::endl;
    const double idet = 1.0/(1.0 + m12*m23*m31 + m13*m21*m32 - m12*m21 - m13*m31 - m23*m32); // 13 flops
    Dxx = (naive_Dxx*(idet*(1.0-m23*m32)) + naive_Dyy*(idet*(m32*m13-m12)) + naive_Dzz*(idet*(m12*m23-m13))); // 14 flops, counted only if double
    Dyy = (naive_Dxx*(idet*(m31*m23-m21)) + naive_Dyy*(idet*(1.0-m31*m13)) + naive_Dzz*(idet*(m21*m13-m23))); // 14 flops, counted only if double
    Dzz = (naive_Dxx*(idet*(m21*m32-m31)) + naive_Dyy*(idet*(m31*m12-m32)) + naive_Dzz*(idet*(1.0-m21*m12))); // 14 flops, counted only if double
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(13+(std::is_floating_point<T>::value?42:0)); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  }
#else
#ifdef CASL_LOG_FLOPS
  ierr = PetscLogFlops(18); CHKERRXX(ierr);
#endif
  // naive_Dfxx = 1.0*fxx + m12*fyy
  // naive_Dfyy = m21*fxx + 1.0*fyy
  // either m12 or m21 MUST be 0
  P4EST_ASSERT(check_if_zero(m12) || check_if_zero(m21));
  if(check_if_zero(m12))
  {
    Dxx = naive_Dxx;
    if(check_if_zero(m21))
      Dyy = naive_Dyy;
    else {
      Dyy = naive_Dyy - m21*naive_Dxx;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(std::is_floating_point<T>::value?2:0); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
    }
  }
  else if (check_if_zero(m21))
  {
    Dyy = naive_Dyy;
    Dxx = naive_Dxx - m12*naive_Dyy; // m12 is necessarily not 0.0, otherwise the above statement would have been activated
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(std::is_floating_point<T>::value?2:0); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  }
  else
  {
    // this is supposed to NEVER-EVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
    std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement A (2D), this really sucks!" << std::endl;
    const double idet = 1.0/(1.0 - m12*m21); // 3 flops
    Dxx = naive_Dxx*idet - naive_Dyy*(m12*idet);// 4 flops, counted only if double
    Dyy = naive_Dyy*idet - naive_Dxx*(m21*idet);// 4 flops, counted only if double
#ifdef CASL_LOG_FLOPS
    ierr = PetscLogFlops(3+(std::is_floating_point<T>::value?8:0)); CHKERRXX(ierr); // flops are counting in other subsequent operator functions in node_linear_combination if not double
#endif
  }
#endif
}

#ifdef P4_TO_P8
template<> void quad_neighbor_nodes_of_node_t::laplace(const double *f, double &f_000, double &f_m00, double &f_p00, double &f_0m0, double &f_0p0, double &f_00m, double &f_00p,
                                                       double &fxx, double &fyy, double &fzz, const unsigned int &bs, const unsigned int &comp) const
#else
template<> void quad_neighbor_nodes_of_node_t::laplace(const double *f, double &f_000, double &f_m00, double &f_p00, double &f_0m0, double &f_0p0,
                                                       double &fxx, double &fyy, const unsigned int &bs, const unsigned int &comp) const
#endif
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(second_derivatives_are_set) {
    fxx = second_derivative_central[0]->calculate(f, bs, comp);
    fyy = second_derivative_central[1]->calculate(f, bs, comp);
#ifdef P4_TO_P8
    fzz = second_derivative_central[2]->calculate(f, bs, comp);
#endif
    return;
  }
#ifdef P4_TO_P8
  laplace_core<double>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p, fxx, fyy, fzz, bs, comp);
#else
  laplace_core<double>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, fxx, fyy, bs, comp);
#endif
  return;
}
#ifdef P4_TO_P8
template<> void quad_neighbor_nodes_of_node_t::laplace(const double *, node_linear_combination &f_000, node_linear_combination &f_m00, node_linear_combination &f_p00, node_linear_combination &f_0m0, node_linear_combination &f_0p0, node_linear_combination &f_00m, node_linear_combination &f_00p,
                                                       node_linear_combination &fxx, node_linear_combination &fyy, node_linear_combination &fzz, const unsigned int &, const unsigned int &) const
#else
template<> void quad_neighbor_nodes_of_node_t::laplace(const double *, node_linear_combination &f_000, node_linear_combination &f_m00, node_linear_combination &f_p00, node_linear_combination &f_0m0, node_linear_combination &f_0p0,
                                                       node_linear_combination &fxx, node_linear_combination &fyy, const unsigned int &, const unsigned int &) const
#endif
{
  if(second_derivatives_are_set) {
    fxx = *second_derivative_central[0];
    fyy = *second_derivative_central[1];
#ifdef P4_TO_P8
    fzz = *second_derivative_central[2];
#endif
    return;
  }
#ifdef P4_TO_P8
  laplace_core<node_linear_combination>(NULL, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p, fxx, fyy, fzz);
#else
  laplace_core<node_linear_combination>(NULL, f_000, f_m00, f_p00, f_0m0, f_0p0, fxx, fyy);
#endif
  return;
}

#ifdef P4_TO_P8
template<typename T> void quad_neighbor_nodes_of_node_t::laplace_core(const double *f, T&f_000, T&f_m00, T&f_p00, T&f_0m0, T&f_0p0, T&f_00m, T&f_00p,  T&fxx, T&fyy, T&fzz,  const unsigned int &bs, const unsigned int &comp) const
#else
template<typename T> void quad_neighbor_nodes_of_node_t::laplace_core(const double *f, T&f_000, T&f_m00, T&f_p00, T&f_0m0, T&f_0p0,                    T&fxx, T&fyy,         const unsigned int &bs, const unsigned int &comp) const
#endif
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace_core, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef P4_TO_P8
  linearly_interpolated_neighbors<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p,  bs, comp);
#else
  linearly_interpolated_neighbors<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0,                bs, comp);
#endif

  const T naive_Dxx = central_second_derivative<T>(f_p00, f_000, f_m00, d_p00, d_m00);
  const T naive_Dyy = central_second_derivative<T>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
#ifdef P4_TO_P8
  const T naive_Dzz = central_second_derivative<T>(f_00p, f_000, f_00m, d_00p, d_00m);
  correct_naive_second_derivatives<T>(naive_Dxx, naive_Dyy, naive_Dzz, fxx, fyy, fzz);
#else
  correct_naive_second_derivatives<T>(naive_Dxx, naive_Dyy, fxx, fyy);
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace_core, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

template<> void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation(const double *f, double &f_000, double &f_m00, double &f_p00, double &f_0m0, double &f_0p0,
                                                                                 #ifdef P4_TO_P8
                                                                                 double &f_00m, double &f_00p,
                                                                                 #endif
                                                                                 const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if (quadratic_interpolators_are_set)
  {
    f_000 = f_000_linear<double>(f, bs, comp);
    f_m00 = quadratic_interpolator_m00->calculate(f, bs, comp);
    f_p00 = quadratic_interpolator_p00->calculate(f, bs, comp);
    f_0m0 = quadratic_interpolator_0m0->calculate(f, bs, comp);
    f_0p0 = quadratic_interpolator_0p0->calculate(f, bs, comp);
#ifdef P4_TO_P8
    f_00m = quadratic_interpolator_00m->calculate(f, bs, comp);
    f_00p = quadratic_interpolator_00p->calculate(f, bs, comp);
#endif
    return;
  }
#ifdef P4_TO_P8
  ngbd_with_quadratic_interpolation_core<double>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p, bs, comp);
#else
  ngbd_with_quadratic_interpolation_core<double>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, bs, comp);
#endif
  return;
}

template<> void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation(const double *, node_linear_combination &f_000, node_linear_combination &f_m00, node_linear_combination &f_p00, node_linear_combination &f_0m0, node_linear_combination &f_0p0,
                                                                                 #ifdef P4_TO_P8
                                                                                 node_linear_combination &f_00m, node_linear_combination &f_00p,
                                                                                 #endif
                                                                                 const unsigned int &, const unsigned int &) const
{
  if (quadratic_interpolators_are_set)
  {
    f_000 = f_000_linear<node_linear_combination>(NULL);
    f_m00 = (*quadratic_interpolator_m00);
    f_p00 = (*quadratic_interpolator_p00);
    f_0m0 = (*quadratic_interpolator_0m0);
    f_0p0 = (*quadratic_interpolator_0p0);
#ifdef P4_TO_P8
    f_00m = (*quadratic_interpolator_00m);
    f_00p = (*quadratic_interpolator_00p);
#endif
    return;
  }
#ifdef P4_TO_P8
  ngbd_with_quadratic_interpolation_core<node_linear_combination>(NULL, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p);
#else
  ngbd_with_quadratic_interpolation_core<node_linear_combination>(NULL, f_000, f_m00, f_p00, f_0m0, f_0p0);
#endif
  return;
}

template<typename T> void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation_core(const double *f, T &f_000, T &f_m00, T &f_p00, T &f_0m0, T &f_0p0,
                                                                                                #ifdef P4_TO_P8
                                                                                                T &f_00m, T &f_00p,
                                                                                                #endif
                                                                                                const unsigned int &bs, const unsigned int &comp) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation_core, 0, 0, 0, 0); CHKERRXX(ierr);

  bool need_second_derivatives = false;
  const bool f_m00_need_yy_correction = (!check_if_zero(d_m00_m0*inverse_d_max) && !check_if_zero(d_m00_p0*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_m00_need_yy_correction;
#ifdef P4_TO_P8
  const bool f_m00_need_zz_correction = (!check_if_zero(d_m00_0m*inverse_d_max) && !check_if_zero(d_m00_0p*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_m00_need_zz_correction;
#endif
  const bool f_p00_need_yy_correction = (!check_if_zero(d_p00_m0*inverse_d_max) && !check_if_zero(d_p00_p0*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_p00_need_yy_correction;
#ifdef P4_TO_P8
  const bool f_p00_need_zz_correction = (!check_if_zero(d_p00_0m*inverse_d_max) && !check_if_zero(d_p00_0p*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_p00_need_zz_correction;
#endif
  const bool f_0m0_need_xx_correction = (!check_if_zero(d_0m0_m0*inverse_d_max) && !check_if_zero(d_0m0_p0*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_0m0_need_xx_correction;
#ifdef P4_TO_P8
  const bool f_0m0_need_zz_correction = (!check_if_zero(d_0m0_0m*inverse_d_max) && !check_if_zero(d_0m0_0p*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_0m0_need_zz_correction;
#endif
  const bool f_0p0_need_xx_correction = (!check_if_zero(d_0p0_m0*inverse_d_max) && !check_if_zero(d_0p0_p0*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_0p0_need_xx_correction;
#ifdef P4_TO_P8
  const bool f_0p0_need_zz_correction = (!check_if_zero(d_0p0_0m*inverse_d_max) && !check_if_zero(d_0p0_0p*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_0p0_need_zz_correction;
  const bool f_00m_need_xx_correction = (!check_if_zero(d_00m_m0*inverse_d_max) && !check_if_zero(d_00m_p0*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_00m_need_xx_correction;
  const bool f_00m_need_yy_correction = (!check_if_zero(d_00m_0m*inverse_d_max) && !check_if_zero(d_00m_0p*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_00m_need_yy_correction;
  const bool f_00p_need_xx_correction = (!check_if_zero(d_00p_m0*inverse_d_max) && !check_if_zero(d_00p_p0*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_00p_need_xx_correction;
  const bool f_00p_need_yy_correction = (!check_if_zero(d_00p_0m*inverse_d_max) && !check_if_zero(d_00p_0p*inverse_d_max)); need_second_derivatives = need_second_derivatives || f_00p_need_yy_correction;
#endif

  if(need_second_derivatives)
  {
    T fxx, fyy;
#ifdef P4_TO_P8
    T fzz;
    laplace<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p, fxx, fyy, fzz, bs, comp);
#else
    laplace<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, fxx, fyy,  bs, comp);
#endif

    // third order interpolation
    if(f_m00_need_yy_correction)
    {
      f_m00 -= 0.5*d_m00_m0*d_m00_p0*fyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#ifdef P4_TO_P8
    if(f_m00_need_zz_correction)
    {
      f_m00 -= 0.5*d_m00_0m*d_m00_0p*fzz;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#endif
    if(f_p00_need_yy_correction)
    {
      f_p00 -= 0.5*d_p00_m0*d_p00_p0*fyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#ifdef P4_TO_P8
    if(f_p00_need_zz_correction)
    {
      f_p00 -= 0.5*d_p00_0m*d_p00_0p*fzz;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#endif
    if(f_0m0_need_xx_correction)
    {
      f_0m0 -= 0.5*d_0m0_m0*d_0m0_p0*fxx;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#ifdef P4_TO_P8
    if(f_0m0_need_zz_correction)
    {
      f_0m0 -= 0.5*d_0m0_0m*d_0m0_0p*fzz;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#endif
    if(f_0p0_need_xx_correction)
    {
      f_0p0 -= 0.5*d_0p0_m0*d_0p0_p0*fxx;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#ifdef P4_TO_P8
    if(f_0p0_need_zz_correction)
    {
      f_0p0 -= 0.5*d_0p0_0m*d_0p0_0p*fzz;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
    if(f_00m_need_xx_correction)
    {
      f_00m -= 0.5*d_00m_m0*d_00m_p0*fxx;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
    if(f_00m_need_yy_correction)
    {
      f_00m -= 0.5*d_00m_0m*d_00m_0p*fyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
    if(f_00p_need_xx_correction)
    {
      f_00p -= 0.5*d_00p_m0*d_00p_p0*fxx;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
    if(f_00p_need_yy_correction)
    {
      f_00p -= 0.5*d_00p_0m*d_00p_0p*fyy;
#ifdef CASL_LOG_FLOPS
      ierr = PetscLogFlops(4); CHKERRXX(ierr);
#endif
    }
#endif
  }
  else
    linearly_interpolated_neighbors<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0,
                                   #ifdef P4_TO_P8
                                       f_00m, f_00p,
                                   #endif
                                       bs, comp);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quadratic_interpolation_core, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}


void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation(const double *f, double &f_m00, double &f_000, double &f_p00, const unsigned int &bs, const unsigned int &comp) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if((check_if_zero(d_m00_m0*inverse_d_max) || check_if_zero(d_m00_p0*inverse_d_max)) && (check_if_zero(d_p00_m0*inverse_d_max) || check_if_zero(d_p00_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_m00_0m*inverse_d_max) || check_if_zero(d_m00_0p*inverse_d_max)) && (check_if_zero(d_p00_0m*inverse_d_max) || check_if_zero(d_p00_0p*inverse_d_max))
   #endif
     )
  {
    f_000 = f_000_linear<double>(f, bs, comp);
    f_m00 = f_m00_linear<double>(f, bs, comp);
    f_p00 = f_p00_linear<double>(f, bs, comp);
  }
  else
  {
    double temp_0, temp_1;
#ifdef P4_TO_P8
    double temp_2, temp_3;
#endif
    ngbd_with_quadratic_interpolation<double>(f, f_000, f_m00, f_p00, temp_0, temp_1,
                                          #ifdef P4_TO_P8
                                              temp_2, temp_3,
                                          #endif
                                              bs, comp);
  }
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}



void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation(const double* f,
                                                                        double& f_0m0, double& f_000, double& f_0p0,
                                                                        const unsigned int &bs, const unsigned int &comp) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if((check_if_zero(d_0m0_m0*inverse_d_max) || check_if_zero(d_0m0_p0*inverse_d_max)) && (check_if_zero(d_0p0_m0*inverse_d_max) || check_if_zero(d_0p0_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_0m0_0m*inverse_d_max) || check_if_zero(d_0m0_0p*inverse_d_max)) && (check_if_zero(d_0p0_0m*inverse_d_max) || check_if_zero(d_0p0_0p*inverse_d_max))
   #endif
     )
  {
    f_000 = f_000_linear<double>(f, bs, comp);
    f_0m0 = f_0m0_linear<double>(f, bs, comp);
    f_0p0 = f_0p0_linear<double>(f, bs, comp);
  }
  else
  {
    double temp_0, temp_1;
#ifdef P4_TO_P8
    double temp_2, temp_3;
#endif
    ngbd_with_quadratic_interpolation<double>(f, f_000, temp_0, temp_1, f_0m0, f_0p0,
                                          #ifdef P4_TO_P8
                                              temp_2, temp_3,
                                          #endif
                                              bs, comp);
  }
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

#ifdef P4_TO_P8
void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation(const double* f,
                                                                        double& f_00m, double& f_000, double& f_00p,
                                                                        const unsigned int &bs, const unsigned int &comp) const
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if((check_if_zero(d_00m_m0*inverse_d_max) || check_if_zero(d_00m_p0*inverse_d_max)) && (check_if_zero(d_00p_m0*inverse_d_max) || check_if_zero(d_00p_p0*inverse_d_max))
     && (check_if_zero(d_00m_0m*inverse_d_max) || check_if_zero(d_00m_0p*inverse_d_max)) && (check_if_zero(d_00p_0m*inverse_d_max) || check_if_zero(d_00p_0p*inverse_d_max))
     )
  {
    f_000 = f_000_linear<double>(f, bs, comp);
    f_00m = f_00m_linear<double>(f, bs, comp);
    f_00p = f_00p_linear<double>(f, bs, comp);
  }
  else
  {
    double temp_0, temp_1, temp_2, temp_3;
    ngbd_with_quadratic_interpolation<double>(f, f_000, temp_0, temp_1, temp_2, temp_3, f_00m, f_00p, bs, comp);
  }
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}
#endif

double quad_neighbor_nodes_of_node_t::dxx_central(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if (second_derivatives_are_set)
    return second_derivative_central[0]->calculate(f, bs, comp);
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  return central_second_derivative<double>(f_p00, f_000, f_m00, d_p00, d_m00);
}

double quad_neighbor_nodes_of_node_t::dyy_central(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if (second_derivatives_are_set)
    return second_derivative_central[1]->calculate(f, bs, comp);
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  return central_second_derivative<double>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

#ifdef P4_TO_P8
double quad_neighbor_nodes_of_node_t::dzz_central(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if (second_derivatives_are_set)
    return second_derivative_central[2]->calculate(f, bs, comp);
  double f_00m,f_000,f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p, bs, comp);
  return central_second_derivative<double>(f_00p, f_000, f_00m, d_0p0, d_0m0);
}
#endif

#ifdef P4_TO_P8
template<typename T> void quad_neighbor_nodes_of_node_t::correct_naive_first_derivatives(const double*f, const T &naive_Dx, const T &naive_Dy, const T &naive_Dz,  T &Dx, T &Dy, T &Dz, const unsigned int &bs, const unsigned int &comp) const
#else
template<typename T> void quad_neighbor_nodes_of_node_t::correct_naive_first_derivatives(const double*f, const T &naive_Dx, const T &naive_Dy,                     T &Dx, T &Dy, const unsigned int &bs, const unsigned int &comp) const
#endif
{
  Dx = naive_Dx;
  Dy = naive_Dy;
  double yy_correction_weight_to_naive_Dx, xx_correction_weight_to_naive_Dy;
#ifdef P4_TO_P8
  double zz_correction_weight_to_naive_Dx, zz_correction_weight_to_naive_Dy;
  double xx_correction_weight_to_naive_Dz, yy_correction_weight_to_naive_Dz;
  Dz = naive_Dz;
#endif
  // correct them if needed
  bool second_derivatives_needed = false;
  const bool Dx_needs_yy_correction = naive_dx_needs_yy_correction(yy_correction_weight_to_naive_Dx); second_derivatives_needed = second_derivatives_needed || Dx_needs_yy_correction;
#ifdef P4_TO_P8
  const bool Dx_needs_zz_correction = naive_dx_needs_zz_correction(zz_correction_weight_to_naive_Dx); second_derivatives_needed = second_derivatives_needed || Dx_needs_zz_correction;
#endif
  const bool Dy_needs_xx_correction = naive_dy_needs_xx_correction(xx_correction_weight_to_naive_Dy); second_derivatives_needed = second_derivatives_needed || Dy_needs_xx_correction;
#ifdef P4_TO_P8
  const bool Dy_needs_zz_correction = naive_dy_needs_zz_correction(zz_correction_weight_to_naive_Dy); second_derivatives_needed = second_derivatives_needed || Dy_needs_zz_correction;
  const bool Dz_needs_xx_correction = naive_dz_needs_xx_correction(xx_correction_weight_to_naive_Dz); second_derivatives_needed = second_derivatives_needed || Dz_needs_xx_correction;
  const bool Dz_needs_yy_correction = naive_dz_needs_yy_correction(yy_correction_weight_to_naive_Dz); second_derivatives_needed = second_derivatives_needed || Dz_needs_yy_correction;
#endif

  if(second_derivatives_needed)
  {
    T fxx,fyy;
#ifdef P4_TO_P8
    T fzz;
    laplace(f,fxx,fyy,fzz,  bs, comp);
#else
    laplace(f,fxx,fyy,      bs, comp);
#endif
    if(Dx_needs_yy_correction)
      Dx -= fyy*yy_correction_weight_to_naive_Dx;
#ifdef P4_TO_P8
    if(Dx_needs_zz_correction)
      Dx -= fzz*zz_correction_weight_to_naive_Dx;
#endif
    if(Dy_needs_xx_correction)
      Dy -= fxx*xx_correction_weight_to_naive_Dy;
#ifdef P4_TO_P8
    if(Dy_needs_zz_correction)
      Dy -= fzz*zz_correction_weight_to_naive_Dy;
    if(Dz_needs_xx_correction)
      Dz -= fxx*xx_correction_weight_to_naive_Dz;
    if(Dz_needs_yy_correction)
      Dz -= fyy*yy_correction_weight_to_naive_Dz;
#endif
  }
  return;
}

#ifdef P4_TO_P8
template<> void quad_neighbor_nodes_of_node_t::gradient(const double *f, double &fx, double &fy, double &fz,  const unsigned int &bs, const unsigned int &comp) const
#else
template<> void quad_neighbor_nodes_of_node_t::gradient(const double *f, double &fx, double &fy,              const unsigned int &bs, const unsigned int &comp) const
#endif
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(first_derivatives_are_set) {
    fx = first_derivative_central[0]->calculate(f, bs, comp);
    fy = first_derivative_central[1]->calculate(f, bs, comp);
#ifdef P4_TO_P8
    fz = first_derivative_central[2]->calculate(f, bs, comp);
#endif
    return;
  }
#ifdef P4_TO_P8
  gradient_core<double>(f, fx, fy, fz,  bs, comp);
#else
  gradient_core<double>(f, fx, fy,      bs, comp);
#endif
  return;
}

#ifdef P4_TO_P8
template<> void quad_neighbor_nodes_of_node_t::gradient(const double *, node_linear_combination &fx, node_linear_combination &fy, node_linear_combination &fz, const unsigned int &, const unsigned int &) const
#else
template<> void quad_neighbor_nodes_of_node_t::gradient(const double *, node_linear_combination &fx, node_linear_combination &fy,                              const unsigned int &, const unsigned int &) const
#endif
{
  if(first_derivatives_are_set) {
    fx = (*first_derivative_central[0]);
    fy = (*first_derivative_central[1]);
#ifdef P4_TO_P8
    fz = (*first_derivative_central[2]);
#endif
    return;
  }
#ifdef P4_TO_P8
  gradient_core<node_linear_combination>(NULL, fx, fy, fz);
#else
  gradient_core<node_linear_combination>(NULL, fx, fy);
#endif
  return;
}

#ifdef P4_TO_P8
template<typename T> void quad_neighbor_nodes_of_node_t::gradient_core(const double *f, T&fx, T&fy, T&fz, const unsigned int &bs, const unsigned int &comp) const
#else
template<typename T> void quad_neighbor_nodes_of_node_t::gradient_core(const double *f, T&fx, T&fy,       const unsigned int &bs, const unsigned int &comp) const
#endif
{
  PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient_core, 0, 0, 0, 0); CHKERRXX(ierr);
  T f_000, f_m00, f_p00, f_0m0, f_0p0;
#ifdef P4_TO_P8
  T f_00m, f_00p;
  linearly_interpolated_neighbors<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p,  bs, comp);
#else
  linearly_interpolated_neighbors<T>(f, f_000, f_m00, f_p00, f_0m0, f_0p0,                bs, comp);
#endif

  const T naive_Dx = central_derivative<T>(f_p00, f_000, f_m00, d_p00, d_m00);
  const T naive_Dy = central_derivative<T>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
#ifdef P4_TO_P8
  const T naive_Dz = central_derivative<T>(f_00p, f_000, f_00m, d_00p, d_00m);
  correct_naive_first_derivatives<T>(f, naive_Dx, naive_Dy, naive_Dz, fx, fy, fz, bs, comp);
#else
  correct_naive_first_derivatives<T>(f, naive_Dx, naive_Dy,           fx, fy,     bs, comp);
#endif
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient_core, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

double quad_neighbor_nodes_of_node_t::dx_central(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(first_derivatives_are_set)
    return first_derivative_central[0]->calculate(f, bs, comp);

  const double f_000 = f_000_linear<double>(f, bs, comp);
  const double f_m00 = f_m00_linear<double>(f, bs, comp);
  const double f_p00 = f_p00_linear<double>(f, bs, comp);
  double df_dx = central_derivative<double>(f_p00, f_000, f_m00, d_p00, d_m00); // naive approach so far

  double yy_correction_weight_to_df_dx;
#ifdef P4_TO_P8
  double zz_correction_weight_to_df_dx;
#endif
  // correct it if needed
  bool second_derivatives_needed = false;
  const bool Dx_needs_yy_correction = naive_dx_needs_yy_correction(yy_correction_weight_to_df_dx); second_derivatives_needed = second_derivatives_needed || Dx_needs_yy_correction;
#ifdef P4_TO_P8
  const bool Dx_needs_zz_correction = naive_dx_needs_zz_correction(zz_correction_weight_to_df_dx); second_derivatives_needed = second_derivatives_needed || Dx_needs_zz_correction;
#endif

  if(second_derivatives_needed)
  {
    double fxx,fyy;
#ifdef P4_TO_P8
    double fzz;
    laplace<double>(f,fxx,fyy,fzz, bs, comp);
#else
    laplace<double>(f,fxx,fyy, bs, comp);
#endif
    if(Dx_needs_yy_correction)
      df_dx -= fyy*yy_correction_weight_to_df_dx;
#ifdef P4_TO_P8
    if(Dx_needs_zz_correction)
      df_dx -= fzz*zz_correction_weight_to_df_dx;
#endif
  }
  return df_dx;
}

double quad_neighbor_nodes_of_node_t::dy_central(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(first_derivatives_are_set)
    return first_derivative_central[1]->calculate(f, bs, comp);

  const double f_000 = f_000_linear<double>(f, bs, comp);
  const double f_0m0 = f_0m0_linear<double>(f, bs, comp);
  const double f_0p0 = f_0p0_linear<double>(f, bs, comp);
  double df_dy = central_derivative<double>(f_0p0, f_000, f_0m0, d_0p0, d_0m0); // naive approach so far

  double xx_correction_weight_to_df_dy;
#ifdef P4_TO_P8
  double zz_correction_weight_to_df_dy;
#endif
  // correct it if needed
  bool second_derivatives_needed = false;
  const bool Dy_needs_xx_correction = naive_dy_needs_xx_correction(xx_correction_weight_to_df_dy); second_derivatives_needed = second_derivatives_needed || Dy_needs_xx_correction;
#ifdef P4_TO_P8
  const bool Dy_needs_zz_correction = naive_dy_needs_zz_correction(zz_correction_weight_to_df_dy); second_derivatives_needed = second_derivatives_needed || Dy_needs_zz_correction;
#endif

  if(second_derivatives_needed)
  {
    double fxx,fyy;
#ifdef P4_TO_P8
    double fzz;
    laplace<double>(f,fxx,fyy,fzz, bs, comp);
#else
    laplace<double>(f,fxx,fyy, bs, comp);
#endif
    if(Dy_needs_xx_correction)
      df_dy -= fxx*xx_correction_weight_to_df_dy;
#ifdef P4_TO_P8
    if(Dy_needs_zz_correction)
      df_dy -= fzz*zz_correction_weight_to_df_dy;
#endif
  }
  return df_dy;
}

#ifdef P4_TO_P8
double quad_neighbor_nodes_of_node_t::dz_central(const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  if(first_derivatives_are_set)
    return first_derivative_central[2]->calculate(f, bs, comp);

  const double f_000 = f_000_linear<double>(f, bs, comp);
  const double f_00m = f_00m_linear<double>(f, bs, comp);
  const double f_00p = f_00p_linear<double>(f, bs, comp);
  double df_dz = central_derivative<double>(f_00p, f_000, f_00m, d_00p, d_00m); // naive approach so far

  double xx_correction_weight_to_df_dz;
#ifdef P4_TO_P8
  double yy_correction_weight_to_df_dz;
#endif
  // correct it if needed
  bool second_derivatives_needed = false;
  const bool Dz_needs_xx_correction = naive_dz_needs_xx_correction(xx_correction_weight_to_df_dz); second_derivatives_needed = second_derivatives_needed || Dz_needs_xx_correction;
#ifdef P4_TO_P8
  const bool Dz_needs_yy_correction = naive_dz_needs_yy_correction(yy_correction_weight_to_df_dz); second_derivatives_needed = second_derivatives_needed || Dz_needs_yy_correction;
#endif

  if(second_derivatives_needed)
  {
    double fxx,fyy, fzz;
    laplace<double>(f,fxx,fyy,fzz, bs, comp);
    if(Dz_needs_xx_correction)
      df_dz -= fxx*xx_correction_weight_to_df_dz;
    if(Dz_needs_yy_correction)
      df_dz -= fyy*yy_correction_weight_to_df_dz;
  }
  return df_dz;
}
#endif

double quad_neighbor_nodes_of_node_t::dx_forward_linear (const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  return forward_derivative<double>(f_p00_linear<double>(f, bs, comp), f_000_linear<double>(f, bs, comp), d_p00);
}
double quad_neighbor_nodes_of_node_t::dx_backward_linear (const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  return backward_derivative<double>(f_000_linear<double>(f, bs, comp), f_m00_linear<double>(f, bs, comp), d_m00);
}
double quad_neighbor_nodes_of_node_t::dy_forward_linear (const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  return forward_derivative<double>(f_0p0_linear<double>(f, bs, comp), f_000_linear<double>(f, bs, comp), d_0p0);
}
double quad_neighbor_nodes_of_node_t::dy_backward_linear (const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  return backward_derivative<double>(f_000_linear<double>(f, bs, comp), f_0m0_linear<double>(f, bs, comp), d_0m0);
}
#ifdef P4_TO_P8
double quad_neighbor_nodes_of_node_t::dz_forward_linear (const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  return forward_derivative<double>(f_00p_linear<double>(f, bs, comp), f_000_linear<double>(f, bs, comp), d_00p);
}
double quad_neighbor_nodes_of_node_t::dz_backward_linear (const double *f, const unsigned int &bs, const unsigned int &comp) const
{
  return backward_derivative<double>(f_000_linear<double>(f, bs, comp), f_00m_linear<double>(f, bs, comp), d_00m);
}
#endif

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  const double f_xx_000 = central_second_derivative<double>(f_p00, f_000, f_m00, d_p00, d_m00);
  double f_xx_m00;
  if(linear_interpolators_are_set)
    f_xx_m00 = linear_interpolator_m00->calculate_dd(dir::x, f, neighbors, bs, comp);
  else
  {
    node_linear_combination lin_m00 = f_m00_linear<node_linear_combination>(NULL);
    f_xx_m00 = lin_m00.calculate_dd(dir::x, f, neighbors, bs, comp);
  }
  return d_backward_quadratic(f_000, f_m00, d_m00, f_xx_000, f_xx_m00);
}
double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  const double f_xx_000 = central_second_derivative<double>(f_p00, f_000, f_m00, d_p00, d_m00);
  double f_xx_p00;
  if(linear_interpolators_are_set)
    f_xx_p00 = linear_interpolator_p00->calculate_dd(dir::x, f, neighbors, bs, comp);
  else
  {
    node_linear_combination lin_p00 = f_p00_linear<node_linear_combination>(NULL);
    f_xx_p00 = lin_p00.calculate_dd(dir::x, f, neighbors, bs, comp);
  }
  return d_forward_quadratic(f_p00, f_000, d_p00, f_xx_000, f_xx_p00);
}
double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  const double f_yy_000 = central_second_derivative<double>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double f_yy_0m0;
  if(linear_interpolators_are_set)
    f_yy_0m0 = linear_interpolator_0m0->calculate_dd(dir::y, f, neighbors, bs, comp);
  else
  {
    node_linear_combination lin_0m0 = f_0m0_linear<node_linear_combination>(NULL);
    f_yy_0m0 = lin_0m0.calculate_dd(dir::y, f, neighbors, bs, comp);
  }
  return d_backward_quadratic(f_000, f_0m0, d_0m0, f_yy_000, f_yy_0m0);
}
double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  const double f_yy_000 = central_second_derivative<double>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double f_yy_0p0;
  if(linear_interpolators_are_set)
    f_yy_0p0 = linear_interpolator_0p0->calculate_dd(dir::y, f, neighbors, bs, comp);
  else
  {
    node_linear_combination lin_0p0 = f_0p0_linear<node_linear_combination>(NULL);
    f_yy_0p0 = lin_0p0.calculate_dd(dir::y, f, neighbors, bs, comp);
  }
  return d_forward_quadratic(f_0p0, f_000, d_0p0, f_yy_000, f_yy_0p0);
}
#ifdef P4_TO_P8
double quad_neighbor_nodes_of_node_t::dz_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p, bs, comp);
  const double f_zz_000 = central_second_derivative<double>(f_00p, f_000, f_00m, d_00p, d_00m);
  double f_zz_00m;
  if(linear_interpolators_are_set)
    f_zz_00m = linear_interpolator_00m->calculate_dd(dir::z, f, neighbors, bs, comp);
  else
  {
    node_linear_combination lin_00m = f_00m_linear<node_linear_combination>(NULL);
    f_zz_00m = lin_00m.calculate_dd(dir::z, f, neighbors, bs, comp);
  }
  return d_backward_quadratic(f_000, f_00m, d_00m, f_zz_000, f_zz_00m);
}
double quad_neighbor_nodes_of_node_t::dz_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p, bs, comp);
  const double f_zz_000 = central_second_derivative<double>(f_00p, f_000, f_00m, d_00p, d_00m);
  double f_zz_00p;
  if(linear_interpolators_are_set)
    f_zz_00p = linear_interpolator_00p->calculate_dd(dir::z, f, neighbors, bs, comp);
  else
  {
    node_linear_combination lin_00p = f_00p_linear<node_linear_combination>(NULL);
    f_zz_00p = lin_00p.calculate_dd(dir::z, f, neighbors, bs, comp);
  }
  return d_forward_quadratic(f_00p, f_000, d_00p, f_zz_000, f_zz_00p);
}
#endif

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const double *fxx, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  const double f_xx_000 = central_second_derivative<double>(f_p00, f_000, f_m00, d_p00, d_m00);
  const double f_xx_m00 = f_m00_linear<double>(fxx, bs, comp);
  return d_backward_quadratic(f_000, f_m00, d_m00, f_xx_000, f_xx_m00);
}
double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const double *fxx, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  const double f_xx_000 = central_second_derivative<double>(f_p00, f_000, f_m00, d_p00, d_m00);
  const double f_xx_p00 = f_p00_linear<double>(fxx, bs, comp);
  return d_forward_quadratic(f_p00, f_000, d_p00, f_xx_000, f_xx_p00);
}
double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const double *fyy, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  const double f_yy_000 = central_second_derivative<double>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  const double f_yy_0m0 = f_0m0_linear<double>(fyy, bs, comp);
  return d_backward_quadratic(f_000, f_0m0, d_0m0, f_yy_000, f_yy_0m0);
}
double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const double *fyy, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  const double f_yy_000 = central_second_derivative<double>(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  const double f_yy_0p0 = f_0p0_linear<double>(fyy, bs, comp);
  return d_forward_quadratic(f_0p0, f_000, d_0p0, f_yy_000, f_yy_0p0);
}
#ifdef P4_TO_P8
double quad_neighbor_nodes_of_node_t::dz_backward_quadratic(const double *f, const double *fzz, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p, bs, comp);
  const double f_zz_000 = central_second_derivative<double>(f_00p, f_000, f_00m, d_00p, d_00m);
  const double f_zz_00m = f_00m_linear<double>(fzz, bs, comp);
  return d_backward_quadratic(f_000, f_00m, d_00m, f_zz_000, f_zz_00m);
}
double quad_neighbor_nodes_of_node_t::dz_forward_quadratic(const double *f, const double *fzz, const unsigned int&bs, const unsigned int &comp) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p, bs, comp);
  const double f_zz_000 = central_second_derivative<double>(f_00p, f_000, f_00m, d_00p, d_00m);
  const double f_zz_00p = f_00p_linear<double>(fzz, bs, comp);
  return d_forward_quadratic(f_00p, f_000, d_00p, f_zz_000, f_zz_00p);
}
#endif

void quad_neighbor_nodes_of_node_t::set_and_store_linear_interpolators()
{
  if(linear_interpolators_are_set)
    return;
  linear_interpolator_000 = new node_linear_combination(f_000_linear<node_linear_combination>(NULL));
  linear_interpolator_m00 = new node_linear_combination(f_m00_linear<node_linear_combination>(NULL));
  linear_interpolator_p00 = new node_linear_combination(f_p00_linear<node_linear_combination>(NULL));
  linear_interpolator_0m0 = new node_linear_combination(f_0m0_linear<node_linear_combination>(NULL));
  linear_interpolator_0p0 = new node_linear_combination(f_0p0_linear<node_linear_combination>(NULL));
#ifdef P4_TO_P8
  linear_interpolator_00m = new node_linear_combination(f_00m_linear<node_linear_combination>(NULL));
  linear_interpolator_00p = new node_linear_combination(f_0p0_linear<node_linear_combination>(NULL));
#endif
  linear_interpolators_are_set = true;
}

void quad_neighbor_nodes_of_node_t::free_linear_interpolators()
{
  if(!linear_interpolators_are_set)
    return;
  delete linear_interpolator_000; linear_interpolator_000 = NULL;
  delete linear_interpolator_m00; linear_interpolator_m00 = NULL;
  delete linear_interpolator_p00; linear_interpolator_p00 = NULL;
  delete linear_interpolator_0m0; linear_interpolator_0m0 = NULL;
  delete linear_interpolator_0p0; linear_interpolator_0p0 = NULL;
#ifdef P4_TO_P8
  delete linear_interpolator_00m; linear_interpolator_00m = NULL;
  delete linear_interpolator_00p; linear_interpolator_00p = NULL;
#endif
  linear_interpolators_are_set = false;
}

void quad_neighbor_nodes_of_node_t::set_and_store_second_derivative_operators()
{
  if(second_derivatives_are_set)
    return;
  // if not done yet, build them!
  node_linear_combination *tmp[P4EST_DIM];
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
    tmp[der] = new node_linear_combination;
#ifdef P4_TO_P8
  laplace<node_linear_combination>(NULL, *tmp[0], *tmp[1], *tmp[2]);
#else
  laplace<node_linear_combination>(NULL, *tmp[0], *tmp[1]);
#endif
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
    second_derivative_central[der] = tmp[der];
  second_derivatives_are_set = true;
  return;
}

void quad_neighbor_nodes_of_node_t::free_second_derivative_operators()
{
  if(!second_derivatives_are_set)
    return;
  for (unsigned char der = 0; der < P4EST_DIM; ++der) {
    delete second_derivative_central[der]; second_derivative_central[der] = NULL;
  }
  second_derivatives_are_set = false;
  return;
}

void quad_neighbor_nodes_of_node_t::set_and_store_quadratic_interpolators()
{
  if(quadratic_interpolators_are_set)
    return;
  node_linear_combination tmp_000;
  node_linear_combination *ptr_m00 = new node_linear_combination;
  node_linear_combination *ptr_p00 = new node_linear_combination;
  node_linear_combination *ptr_0m0 = new node_linear_combination;
  node_linear_combination *ptr_0p0 = new node_linear_combination;
#ifdef P4_TO_P8
  node_linear_combination *ptr_00m = new node_linear_combination;
  node_linear_combination *ptr_00p = new node_linear_combination;
  ngbd_with_quadratic_interpolation<node_linear_combination>(NULL, tmp_000, (*ptr_m00), (*ptr_p00), (*ptr_0m0), (*ptr_0p0), (*ptr_00m), (*ptr_00p));
#else
  ngbd_with_quadratic_interpolation<node_linear_combination>(NULL, tmp_000, (*ptr_m00), (*ptr_p00), (*ptr_0m0), (*ptr_0p0));
#endif
  quadratic_interpolator_m00 = ptr_m00;
  quadratic_interpolator_p00 = ptr_p00;
  quadratic_interpolator_0m0 = ptr_0m0;
  quadratic_interpolator_0p0 = ptr_0p0;
#ifdef P4_TO_P8
  quadratic_interpolator_00m = ptr_00m;
  quadratic_interpolator_00p = ptr_00p;
#endif
  quadratic_interpolators_are_set = true;
  return;
}

void quad_neighbor_nodes_of_node_t::set_and_store_first_derivative_operators()
{
  if(first_derivatives_are_set)
    return;
  // if not done yet, build them!
  node_linear_combination *tmp[P4EST_DIM];
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
    tmp[der] = new node_linear_combination;
#ifdef P4_TO_P8
  gradient<node_linear_combination>(NULL, *tmp[0], *tmp[1], *tmp[2]);
#else
  gradient<node_linear_combination>(NULL, *tmp[0], *tmp[1]);
#endif
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
    first_derivative_central[der] = tmp[der];
  first_derivatives_are_set = true;
  return;
}
