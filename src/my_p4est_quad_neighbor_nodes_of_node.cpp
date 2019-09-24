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
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double node_linear_combination::calculate_dd( const unsigned char der, const double *node_sample_field, const my_p4est_node_neighbors_t &neighbors) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate_dd, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  P4EST_ASSERT(der < P4EST_DIM);
  P4EST_ASSERT(elements.size()>0);
  double value = elements[0].weight*(neighbors.get_neighbors(elements[0].node_idx).dd_central(der, node_sample_field));
  for (size_t k = 1; k < elements.size(); ++k)
    value += elements[k].weight*(neighbors.get_neighbors(elements[k].node_idx).dd_central(der, node_sample_field));
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_node_linear_combination_calculate_dd, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return value;
}

#ifdef P4_TO_P8
void quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(const double *naive_Dxx, const double *naive_Dyy, const double *naive_Dzz,  double *Dxx, double *Dyy, double *Dzz, const unsigned int &n_values) const
#else
void quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(const double *naive_Dxx, const double *naive_Dyy,                           double *Dxx, double *Dyy,              const unsigned int &n_values) const
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
  PetscErrorCode ierr_flops = PetscLogFlops(54); CHKERRXX(ierr_flops);
#endif
  // naive_Dfxx = 1.0*fxx + m12*fyy + m13*fzz
  // naive_Dfyy = m21*fxx + 1.0*fyy + m23*fzz
  // naive_Dfzz = m31*fxx + m32*fyy + 1.0*fzz
  // either (m12 and m13) are 0.0 or (m21 and m23) are 0.0 or (m31 and m32) are 0.0
  P4EST_ASSERT((check_if_zero(m12) && check_if_zero(m13)) || (check_if_zero(m21) && check_if_zero(m23)) || (check_if_zero(m31) && check_if_zero(m32)));
  if (check_if_zero(m12) && check_if_zero(m13))
  {
    // now, either m23 or m32 or both must 0.0
    P4EST_ASSERT(check_if_zero(m23) || check_if_zero(m32));
    if(check_if_zero(m23))
    {
      const bool m21_non_zero = !check_if_zero(m21);
      const bool m31_non_zero = !check_if_zero(m31);
      const bool m32_non_zero = !check_if_zero(m32);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dxx[k] = naive_Dxx[k];
        Dyy[k] = naive_Dyy[k];
        if(m21_non_zero)
          Dyy[k] -= m21*Dxx[k];
        Dzz[k] = naive_Dzz[k];
        if(m31_non_zero)
          Dzz[k] -= m31*Dxx[k];
        if(m32_non_zero)
          Dzz[k] -= m32*Dyy[k];
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
    else if (check_if_zero(m32))
    {
      const bool m21_non_zero = !check_if_zero(m21);
      const bool m31_non_zero = !check_if_zero(m31);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dxx[k] = naive_Dxx[k];
        Dzz[k] = naive_Dzz[k];
        if(m31_non_zero)
          Dzz[k] -= m31*Dxx[k];
        // m23 is non-zero, otherwise the if statement above would have been entered
        Dyy[k] = naive_Dyy[k] - m23*Dzz[k];
        if(m21_non_zero)
          Dyy[k] -= m21*Dxx[k];
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
    else
    {
      // this is supposed to NEVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
      std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement a." << std::endl;
      const double idet = 1.0/(1.0-m23*m32);
      const double m21_is_zero = check_if_zero(m21);
      const double m31_is_zero = check_if_zero(m31);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dxx[k] = naive_Dxx[k];
        Dyy[k] = ((m21_is_zero? naive_Dyy[k] : (naive_Dyy[k]-m21*Dxx[k])) - m23*(m31_is_zero? naive_Dzz[k] : (naive_Dzz[k]-m31*Dxx[k])))*idet;
        Dzz[k] = ((m31_is_zero? naive_Dzz[k] : (naive_Dzz[k]-m31*Dxx[k])) - m32*(m21_is_zero? naive_Dyy[k] : (naive_Dyy[k]-m21*Dxx[k])))*idet;
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(17*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
  }
  else if (check_if_zero(m21) && check_if_zero(m23))
  {
    // now, either m13 or m31 or both must 0.0
    P4EST_ASSERT(check_if_zero(m13) || check_if_zero(m31));
    if(check_if_zero(m13))
    {
      const bool m12_non_zero= !check_if_zero(m12);
      const bool m31_non_zero= !check_if_zero(m31);
      const bool m32_non_zero= !check_if_zero(m32);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dyy[k] = naive_Dyy[k];
        Dxx[k] = naive_Dxx[k];
        if(m12_non_zero)
          Dxx[k] -= m12*Dyy[k];
        Dzz[k] = naive_Dzz[k];
        if(m31_non_zero)
          Dzz[k] -= m31*Dxx[k];
        if(m32_non_zero)
          Dzz[k] -= m32*Dyy[k];
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
    else if (check_if_zero(m31))
    {
      const bool m12_non_zero= !check_if_zero(m12);
      const bool m32_non_zero= !check_if_zero(m32);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dyy[k] = naive_Dyy[k];
        Dzz[k] = naive_Dzz[k];
        if(m32_non_zero)
          Dzz[k] -= m32*Dyy[k];
        // m13 is non-zero, otherwise the if statement above would have been entered
        Dxx[k] = naive_Dxx[k] - m13*Dzz[k];
        if(m12_non_zero)
          Dxx[k] -= m12*Dyy[k];
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
    else
    {
      // this is supposed to NEVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
      std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement b." << std::endl;
      const double idet = 1.0/(1.0-m13*m31);
      const bool m12_is_zero = check_if_zero(m12);
      const bool m32_is_zero = check_if_zero(m32);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dyy[k] = naive_Dyy[k];
        Dxx[k] = ((m12_is_zero? naive_Dxx[k] : (naive_Dxx[k]-m12*Dyy[k])) - m13*(m32_is_zero? naive_Dzz[k] : (naive_Dzz[k]-m32*Dyy[k])))*idet;
        Dzz [k]= ((m32_is_zero? naive_Dzz[k] : (naive_Dzz[k]-m32*Dyy[k])) - m31*(m12_is_zero? naive_Dxx[k] : (naive_Dxx[k]-m12*Dyy[k])))*idet;
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(17*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
  }
  else if (check_if_zero(m31) && check_if_zero(m32))
  {
    // now, either m12 or m21 or both must 0.0
    P4EST_ASSERT(check_if_zero(m12) || check_if_zero(m21));
    if(check_if_zero(m12))
    {
      const bool m13_non_zero= !check_if_zero(m13);
      const bool m21_non_zero= !check_if_zero(m21);
      const bool m23_non_zero= !check_if_zero(m23);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dzz[k] = naive_Dzz[k];
        Dxx[k] = naive_Dxx[k];
        if(m13_non_zero)
          Dxx[k] -= m13*Dzz[k];
        Dyy[k] = naive_Dyy[k];
        if(m21_non_zero)
          Dyy[k] -= m21*Dxx[k];
        if(m23_non_zero)
          Dyy[k] -= m23*Dzz[k];
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
    else if (check_if_zero(m21))
    {
      const bool m13_non_zero= !check_if_zero(m13);
      const bool m23_non_zero= !check_if_zero(m23);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dzz[k] = naive_Dzz[k];
        Dyy[k] = naive_Dyy[k];
        if(m23_non_zero)
          Dyy[k] -= m23*Dzz[k];
        // m12 is non-zero, otherwise the if statement above would have been entered
        Dxx[k] = naive_Dxx[k] - m12*Dyy[k];
        if(m13_non_zero)
          Dxx[k] -= m13*Dzz[k];
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_values); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
    else
    {
      // this is supposed to NEVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
      std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement c." << std::endl;
      const double idet = 1.0/(1.0-m12*m21);
      const bool m13_is_zero = check_if_zero(m13);
      const bool m23_is_zero = check_if_zero(m23);
      for (unsigned int k = 0; k < n_values; ++k) {
        Dzz[k] = naive_Dzz[k];
        Dxx[k] = ((m13_is_zero? naive_Dxx[k] : (naive_Dxx[k]-m13*Dzz[k])) - m12*(m23_is_zero? naive_Dyy[k] : (naive_Dyy[k]-m23*Dzz[k])))*idet;
        Dzz[k] = ((m23_is_zero? naive_Dyy[k] : (naive_Dyy[k]-m23*Dzz[k])) - m21*(m13_is_zero? naive_Dxx[k] : (naive_Dxx[k]-m13*Dzz[k])))*idet;
      }
#ifdef CASL_LOG_FLOPS
      ierr_flops = PetscLogFlops(6*n_vales); CHKERRXX(ierr_flops); // (upper bound)
#endif
    }
  }
  else
  {
    // this is supposed to NEVER-EVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
    std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement D, this really sucks!" << std::endl;
    const double idet = 1.0/(1.0 + m12*m23*m31 + m13*m21*m32 - m12*m21 - m13*m31 - m23*m32); // 13 flops
    for (unsigned int k = 0; k < n_values; ++k) {
      Dxx[k] = (naive_Dxx[k]*(idet*(1.0-m23*m32)) + naive_Dyy[k]*(idet*(m32*m13-m12)) + naive_Dzz[k]*(idet*(m12*m23-m13))); // 14 flops, counted only if double
      Dyy[k] = (naive_Dxx[k]*(idet*(m31*m23-m21)) + naive_Dyy[k]*(idet*(1.0-m31*m13)) + naive_Dzz[k]*(idet*(m21*m13-m23))); // 14 flops, counted only if double
      Dzz[k] = (naive_Dxx[k]*(idet*(m21*m32-m31)) + naive_Dyy[k]*(idet*(m31*m12-m32)) + naive_Dzz[k]*(idet*(1.0-m21*m12))); // 14 flops, counted only if double
    }
#ifdef CASL_LOG_FLOPS
    ierr_flops = PetscLogFlops(13+42*n_values); CHKERRXX(ierr_flops);
#endif
  }
#else
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr_flops = PetscLogFlops(18); CHKERRXX(ierr_flops);
#endif
  // naive_Dfxx = 1.0*fxx + m12*fyy
  // naive_Dfyy = m21*fxx + 1.0*fyy
  // either m12 or m21 MUST be 0
  P4EST_ASSERT(check_if_zero(m12) || check_if_zero(m21));
  if(check_if_zero(m12))
  {
    const bool m21_is_zero = check_if_zero(m21);
    for (unsigned int k = 0; k < n_values; ++k) {
      Dxx[k] = naive_Dxx[k];
      if(m21_is_zero)
        Dyy[k] = naive_Dyy[k];
      else {
        Dyy[k] = naive_Dyy[k] - m21*naive_Dxx[k];
      }
    }
#ifdef CASL_LOG_FLOPS
    if(!m21_is_zero){
      ierr_flops = PetscLogFlops(2*n_values); CHKERRXX(ierr_flops); }
#endif
  }
  else if (check_if_zero(m21))
  {
    for (unsigned int k = 0; k < n_values; ++k) {
      Dyy[k] = naive_Dyy[k];
      Dxx[k] = naive_Dxx[k] - m12*naive_Dyy[k]; // m12 is necessarily not 0.0, otherwise the above statement would have been activated
    }
#ifdef CASL_LOG_FLOPS
    ierr_flops = PetscLogFlops(2*n_values); CHKERRXX(ierr_flops);
#endif
  }
  else
  {
    // this is supposed to NEVER-EVER happen, but it is still implemented to ensure "proper behavior" of the code in release mode
    std::cerr << "quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(): basic alignment hypothesis have been invalidated, statement A (2D), this really sucks!" << std::endl;
    const double idet = 1.0/(1.0 - m12*m21); // 3 flops
    for (unsigned int k = 0; k < n_values; ++k) {
      Dxx[k] = naive_Dxx[k]*idet - naive_Dyy[k]*(m12*idet);// 4 flops
      Dyy[k] = naive_Dyy[k]*idet - naive_Dxx[k]*(m21*idet);// 4 flops
    }
#ifdef CASL_LOG_FLOPS
    ierr_flops = PetscLogFlops(3+8*n_values); CHKERRXX(ierr_flops);
#endif
  }
#endif
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned &n_arrays, const unsigned int &bs) const
{
  P4EST_ASSERT(bs>1);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  if((check_if_zero(d_m00_m0*inverse_d_max) || check_if_zero(d_m00_p0*inverse_d_max)) && (check_if_zero(d_p00_m0*inverse_d_max) || check_if_zero(d_p00_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_m00_0m*inverse_d_max) || check_if_zero(d_m00_0p*inverse_d_max)) && (check_if_zero(d_p00_0m*inverse_d_max) || check_if_zero(d_p00_0p*inverse_d_max))
   #endif
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs+comp] = f[k][bs_node_000+comp];
    }
    f_m00_linear(f, f_m00, n_arrays, bs);
    f_p00_linear(f, f_p00, n_arrays, bs);
  }
  else
  {
    double temp_0[n_arrays*bs], temp_1[n_arrays*bs];
#ifdef P4_TO_P8
    double temp_2[n_arrays*bs], temp_3[n_arrays*bs];
#endif
    ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, temp_0, temp_1,
                                  #ifdef P4_TO_P8
                                      temp_2, temp_3,
                                  #endif
                                      n_arrays, bs);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation_bs_one(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned &n_arrays) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  if((check_if_zero(d_m00_m0*inverse_d_max) || check_if_zero(d_m00_p0*inverse_d_max)) && (check_if_zero(d_p00_m0*inverse_d_max) || check_if_zero(d_p00_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_m00_0m*inverse_d_max) || check_if_zero(d_m00_0p*inverse_d_max)) && (check_if_zero(d_p00_0m*inverse_d_max) || check_if_zero(d_p00_0p*inverse_d_max))
   #endif
     )
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_m00_linear_bs_one(f, f_m00, n_arrays);
    f_p00_linear_bs_one(f, f_p00, n_arrays);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays];
#ifdef P4_TO_P8
    double temp_2[n_arrays], temp_3[n_arrays];
#endif
    ngbd_with_quadratic_interpolation_bs_one(f, f_000, f_m00, f_p00, temp_0, temp_1,
                                         #ifdef P4_TO_P8
                                             temp_2, temp_3,
                                         #endif
                                             n_arrays);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned &n_arrays, const unsigned int &bs) const
{
  P4EST_ASSERT(bs>1);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  if((check_if_zero(d_0m0_m0*inverse_d_max) || check_if_zero(d_0m0_p0*inverse_d_max)) && (check_if_zero(d_0p0_m0*inverse_d_max) || check_if_zero(d_0p0_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_0m0_0m*inverse_d_max) || check_if_zero(d_0m0_0p*inverse_d_max)) && (check_if_zero(d_0p0_0m*inverse_d_max) || check_if_zero(d_0p0_0p*inverse_d_max))
   #endif
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs+comp] = f[k][bs_node_000+comp];
    }
    f_0m0_linear(f, f_0m0, n_arrays, bs);
    f_0p0_linear(f, f_0p0, n_arrays, bs);
  }
  else
  {
    double temp_0[n_arrays*bs], temp_1[n_arrays*bs];
#ifdef P4_TO_P8
    double temp_2[n_arrays*bs], temp_3[n_arrays*bs];
#endif
    ngbd_with_quadratic_interpolation(f, f_000, temp_0, temp_1, f_0m0, f_0p0,
                                  #ifdef P4_TO_P8
                                      temp_2, temp_3,
                                  #endif
                                      n_arrays, bs);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation_bs_one(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned &n_arrays) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  if((check_if_zero(d_0m0_m0*inverse_d_max) || check_if_zero(d_0m0_p0*inverse_d_max)) && (check_if_zero(d_0p0_m0*inverse_d_max) || check_if_zero(d_0p0_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_0m0_0m*inverse_d_max) || check_if_zero(d_0m0_0p*inverse_d_max)) && (check_if_zero(d_0p0_0m*inverse_d_max) || check_if_zero(d_0p0_0p*inverse_d_max))
   #endif
     )
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_0m0_linear_bs_one(f, f_0m0, n_arrays);
    f_0p0_linear_bs_one(f, f_0p0, n_arrays);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays];
#ifdef P4_TO_P8
    double temp_2[n_arrays], temp_3[n_arrays];
#endif
    ngbd_with_quadratic_interpolation_bs_one(f, f_000, temp_0, temp_1, f_0m0, f_0p0,
                                         #ifdef P4_TO_P8
                                             temp_2, temp_3,
                                         #endif
                                             n_arrays);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

#ifdef P4_TO_P8
void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned &n_arrays, const unsigned int &bs) const
{
  P4EST_ASSERT(bs>1);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  if((check_if_zero(d_00m_m0*inverse_d_max) || check_if_zero(d_00m_p0*inverse_d_max)) && (check_if_zero(d_00p_m0*inverse_d_max) || check_if_zero(d_00p_p0*inverse_d_max))
     && (check_if_zero(d_00m_0m*inverse_d_max) || check_if_zero(d_00m_0p*inverse_d_max)) && (check_if_zero(d_00p_0m*inverse_d_max) || check_if_zero(d_00p_0p*inverse_d_max))
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
      unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs+comp] = f[k][bs_node_000+comp];
    }
    f_00m_linear(f, f_00m, n_arrays, bs);
    f_00p_linear(f, f_00p, n_arrays, bs);
  }
  else
  {
    double temp_0[n_arrays*bs], temp_1[n_arrays*bs], temp_2[n_arrays*bs], temp_3[n_arrays*bs];
    ngbd_with_quadratic_interpolation(f, f_000, temp_0, temp_1, temp_2, temp_3, f_00m, f_00p, n_arrays, bs);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}
void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation_bs_one(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned &n_arrays) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  if((check_if_zero(d_00m_m0*inverse_d_max) || check_if_zero(d_00m_p0*inverse_d_max)) && (check_if_zero(d_00p_m0*inverse_d_max) || check_if_zero(d_00p_p0*inverse_d_max))
     && (check_if_zero(d_00m_0m*inverse_d_max) || check_if_zero(d_00m_0p*inverse_d_max)) && (check_if_zero(d_00p_0m*inverse_d_max) || check_if_zero(d_00p_0p*inverse_d_max))
     )
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_00m_linear_bs_one(f, f_00m, n_arrays);
    f_00p_linear_bs_one(f, f_00p, n_arrays);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays], temp_2[n_arrays], temp_3[n_arrays];
    ngbd_with_quadratic_interpolation_bs_one(f, f_000, temp_0, temp_1, temp_2, temp_3, f_00m, f_00p, n_arrays);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}
#endif

#ifdef P4_TO_P8
void quad_neighbor_nodes_of_node_t::correct_naive_first_derivatives(const double *f[], const double *naive_Dx, const double *naive_Dy, const double *naive_Dz,  double *Dx, double *Dy, double *Dz, const unsigned int &n_arrays, const unsigned int &bs) const
#else
void quad_neighbor_nodes_of_node_t::correct_naive_first_derivatives(const double *f[], const double *naive_Dx, const double *naive_Dy,                          double *Dx, double *Dy,             const unsigned int &n_arrays, const unsigned int &bs) const
#endif
{
  const unsigned int n_arrays_bs = n_arrays*bs;
  for (unsigned int k = 0; k < n_arrays_bs; ++k) {
    Dx[k] = naive_Dx[k];
    Dy[k] = naive_Dy[k];
#ifdef P4_TO_P8
    Dz[k] = naive_Dz[k];
#endif
  }
  double yy_correction_weight_to_naive_Dx, xx_correction_weight_to_naive_Dy;
#ifdef P4_TO_P8
  double zz_correction_weight_to_naive_Dx, zz_correction_weight_to_naive_Dy;
  double xx_correction_weight_to_naive_Dz, yy_correction_weight_to_naive_Dz;
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
    double fxx[n_arrays_bs],fyy[n_arrays_bs];
#ifdef P4_TO_P8
    double fzz[n_arrays_bs];
    if(bs>1)
      laplace(f, fxx, fyy, fzz, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, fzz, n_arrays);
#else
    if(bs>1)
      laplace(f, fxx, fyy,      n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, n_arrays);
#endif
    for (unsigned int k = 0; k < n_arrays_bs; ++k) {
      if(Dx_needs_yy_correction)
        Dx[k] -= fyy[k]*yy_correction_weight_to_naive_Dx;
#ifdef P4_TO_P8
      if(Dx_needs_zz_correction)
        Dx[k] -= fzz[k]*zz_correction_weight_to_naive_Dx;
#endif
      if(Dy_needs_xx_correction)
        Dy[k] -= fxx[k]*xx_correction_weight_to_naive_Dy;
#ifdef P4_TO_P8
      if(Dy_needs_zz_correction)
        Dy[k] -= fzz[k]*zz_correction_weight_to_naive_Dy;
      if(Dz_needs_xx_correction)
        Dz[k] -= fxx[k]*xx_correction_weight_to_naive_Dz;
      if(Dz_needs_yy_correction)
        Dz[k] -= fyy[k]*yy_correction_weight_to_naive_Dz;
#endif
    }
  }
  return;
}

void quad_neighbor_nodes_of_node_t::dx_central(const double *f[], double *fx, const unsigned &n_arrays, const unsigned int &bs) const
{
  const unsigned int n_arrays_bs = n_arrays*bs;
  double f_000[n_arrays_bs], f_p00[n_arrays_bs], f_m00[n_arrays_bs];
  if(bs>1){
    for (unsigned int k = 0; k < n_arrays; ++k)
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[k*bs+comp] = f[k][bs*node_000+comp];
    f_m00_linear(f, f_m00, n_arrays, bs);
    f_p00_linear(f, f_p00, n_arrays, bs);
  } else {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_m00_linear_bs_one(f, f_m00, n_arrays);
    f_p00_linear_bs_one(f, f_p00, n_arrays);
  }
  for (unsigned int k = 0; k < n_arrays_bs; ++k)
    fx[k] = central_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00); // naive approach so far

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
    double fxx[n_arrays_bs],fyy[n_arrays_bs];
#ifdef P4_TO_P8
    double fzz[n_arrays_bs];
    if(bs>1)
      laplace(f, fxx, fyy, fzz, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, fzz, n_arrays);
#else
    if(bs>1)
      laplace(f, fxx, fyy, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, n_arrays);
#endif
    for (unsigned int k = 0; k < n_arrays_bs; ++k) {
      if(Dx_needs_yy_correction)
        fx[k] -= fyy[k]*yy_correction_weight_to_df_dx;
#ifdef P4_TO_P8
      if(Dx_needs_zz_correction)
        fx[k] -= fzz[k]*zz_correction_weight_to_df_dx;
#endif
    }
  }
  return;
}

void quad_neighbor_nodes_of_node_t::dy_central(const double *f[], double *fy, const unsigned &n_arrays, const unsigned int &bs) const
{
  const unsigned int n_arrays_bs = n_arrays*bs;
  double f_000[n_arrays_bs], f_0p0[n_arrays_bs], f_0m0[n_arrays_bs];
  if(bs>1){
    for (unsigned int k = 0; k < n_arrays; ++k)
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[k*bs+comp] = f[k][bs*node_000+comp];
    f_0m0_linear(f, f_0m0, n_arrays, bs);
    f_0p0_linear(f, f_0p0, n_arrays, bs);
  } else {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_0m0_linear_bs_one(f, f_0m0, n_arrays);
    f_0p0_linear_bs_one(f, f_0p0, n_arrays);
  }
  for (unsigned int k = 0; k < n_arrays_bs; ++k)
    fy[k] = central_derivative(f_0p0[k], f_000[k], f_0m0[k], d_p00, d_m00); // naive approach so far

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
    double fxx[n_arrays_bs],fyy[n_arrays_bs];
#ifdef P4_TO_P8
    double fzz[n_arrays_bs];
    if(bs>1)
      laplace(f, fxx, fyy, fzz, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, fzz, n_arrays);
#else
    if(bs>1)
      laplace(f, fxx, fyy, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, n_arrays);
#endif
    for (unsigned int k = 0; k < n_arrays_bs; ++k) {
      if(Dy_needs_xx_correction)
        fy[k] -= fxx[k]*xx_correction_weight_to_df_dy;
#ifdef P4_TO_P8
      if(Dy_needs_zz_correction)
        fy[k] -= fzz[k]*zz_correction_weight_to_df_dy;
#endif
    }
  }
  return;
}

#ifdef P4_TO_P8
void quad_neighbor_nodes_of_node_t::dz_central(const double *f[], double *fz, const unsigned &n_arrays, const unsigned int &bs) const
{
  const unsigned int n_arrays_bs = n_arrays*bs;
  double f_000[n_arrays_bs], f_00p[n_arrays_bs], f_00m[n_arrays_bs];
  if(bs>1){
    for (unsigned int k = 0; k < n_arrays; ++k)
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[k*bs+comp] = f[k][bs*node_000+comp];
    f_00m_linear(f, f_00m, n_arrays, bs);
    f_00p_linear(f, f_00p, n_arrays, bs);
  } else {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_00m_linear_bs_one(f, f_00m, n_arrays);
    f_00p_linear_bs_one(f, f_00p, n_arrays);
  }
  for (unsigned int k = 0; k < n_arrays*bs; ++k)
    fz[k] = central_derivative(f_00p[k], f_000[k], f_00m[k], d_p00, d_m00); // naive approach so far

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
    double fxx[n_arrays*bs],fyy[n_arrays*bs];
#ifdef P4_TO_P8
    double fzz[n_arrays*bs];
    if(bs>1)
      laplace(f, fxx, fyy, fzz, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, fzz, n_arrays);
#else
    if(bs>1)
      laplace(f, fxx, fyy, n_arrays, bs);
    else
      laplace_bs_one(f, fxx, fyy, n_arrays);
#endif
    for (unsigned int k = 0; k < n_arrays*bs; ++k) {
      if(Dz_needs_xx_correction)
        fz[k] -= fxx[k]*xx_correction_weight_to_df_dz;
      if(Dz_needs_yy_correction)
        fz[k] -= fyy[k]*yy_correction_weight_to_df_dz;
    }
  }
  return;
}
#endif

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);
  const double f_xx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
#ifdef P4_TO_P8
  node_linear_combination lin_m00 = linear_interpolator(node_m00_mm, node_m00_pm, node_m00_mp, node_m00_pp, d_m00_m0, d_m00_p0, d_m00_0m, d_m00_0p);
#else
  node_linear_combination lin_m00 = linear_interpolator(node_m00_mm, node_m00_pm,                           d_m00_m0, d_m00_p0);
#endif
  double f_xx_m00 = lin_m00.calculate_dd(dir::x, f, neighbors);
  return d_backward_quadratic(f_000, f_m00, d_m00, f_xx_000, f_xx_m00);
}
double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);
  const double f_xx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
#ifdef P4_TO_P8
  node_linear_combination lin_p00 = linear_interpolator(node_p00_mm, node_p00_pm, node_p00_mp, node_p00_pp, d_p00_m0, d_p00_p0, d_p00_0m, d_p00_0p);
#else
  node_linear_combination lin_p00 = linear_interpolator(node_p00_mm, node_p00_pm,                           d_p00_m0, d_p00_p0);
#endif
  double f_xx_p00 = lin_p00.calculate_dd(dir::x, f, neighbors);
  return d_forward_quadratic(f_p00, f_000, d_p00, f_xx_000, f_xx_p00);
}
double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);
  const double f_yy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
#ifdef P4_TO_P8
  node_linear_combination lin_0m0 = linear_interpolator(node_0m0_mm, node_0m0_pm, node_0m0_mp, node_0m0_pp, d_0m0_m0, d_0m0_p0, d_0m0_0m, d_0m0_0p);
#else
  node_linear_combination lin_0m0 = linear_interpolator(node_0m0_mm, node_0m0_pm,                           d_0m0_m0, d_0m0_p0);
#endif
  double f_yy_0m0 = lin_0m0.calculate_dd(dir::y, f, neighbors);
  return d_backward_quadratic(f_000, f_0m0, d_0m0, f_yy_000, f_yy_0m0);
}
double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);
  const double f_yy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
#ifdef P4_TO_P8
  node_linear_combination lin_0p0 = linear_interpolator(node_0p0_mm, node_0p0_pm, node_0p0_mp, node_0p0_pp, d_0p0_m0, d_0p0_p0, d_0p0_0m, d_0p0_0p);
#else
  node_linear_combination lin_0p0 = linear_interpolator(node_0p0_mm, node_0p0_pm,                           d_0p0_m0, d_0p0_p0);
#endif
  double f_yy_0p0 = lin_0p0.calculate_dd(dir::y, f, neighbors);
  return d_forward_quadratic(f_0p0, f_000, d_0p0, f_yy_000, f_yy_0p0);
}
#ifdef P4_TO_P8
double quad_neighbor_nodes_of_node_t::dz_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);
  const double f_zz_000 = central_second_derivative(f_00p, f_000, f_00m, d_00p, d_00m);
#ifdef P4_TO_P8
  node_linear_combination lin_00m = linear_interpolator(node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p);
#else
  node_linear_combination lin_00m = linear_interpolator(node_00m_mm, node_00m_pm,                           d_00m_m0, d_00m_p0);
#endif
  double f_zz_00m = lin_00m.calculate_dd(dir::z, f, neighbors);
  return d_backward_quadratic(f_000, f_00m, d_00m, f_zz_000, f_zz_00m);
}
double quad_neighbor_nodes_of_node_t::dz_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);
  const double f_zz_000 = central_second_derivative(f_00p, f_000, f_00m, d_00p, d_00m);
#ifdef P4_TO_P8
  node_linear_combination lin_00p = linear_interpolator(node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p);
#else
  node_linear_combination lin_00p = linear_interpolator(node_00p_mm, node_00p_pm,                           d_00p_m0, d_00p_p0);
#endif
  double f_zz_00p = lin_00p.calculate_dd(dir::z, f, neighbors);
  return d_forward_quadratic(f_00p, f_000, d_00p, f_zz_000, f_zz_00p);
}
#endif


