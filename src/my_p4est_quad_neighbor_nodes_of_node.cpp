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
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central_m00;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central_p00;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central_0m0;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central_0p0;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double quad_neighbor_nodes_of_node_t::f_m00_linear( const double *f ) const
{
  PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
  if(d_m00_p0==0) return f[node_m00_pm];
  if(d_m00_m0==0) return f[node_m00_mm];
  else          return(f[node_m00_mm]*d_m00_p0 + f[node_m00_pm]*d_m00_m0)/ (d_m00_m0+d_m00_p0);
}

void quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(DIM(double *Dxx, double *Dyy, double *Dzz), const unsigned int &n_values) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_p00_p0==0) return f[node_p00_pm];
    if(d_p00_m0==0) return f[node_p00_mm];
    else          return(f[node_p00_mm]*d_p00_p0 + f[node_p00_pm]*d_p00_m0)/ (d_p00_m0+d_p00_p0);
}

/*
void quad_neighbor_nodes_of_node_t::correct_naive_second_derivatives(DIM(node_linear_combination &Dxx, node_linear_combination &Dyy, node_linear_combination &Dzz)) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0m0_m0==0) return f[node_0m0_mm];
    if(d_0m0_p0==0) return f[node_0m0_pm];
    else          return(f[node_0m0_pm]*d_0m0_m0 + f[node_0m0_mm]*d_0m0_p0)/ (d_0m0_m0+d_0m0_p0);
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned int &n_arrays, const unsigned int &bs) const
{
  P4EST_ASSERT(bs > 1);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k){
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs+comp] = f[k][bs_node_000+comp];
    }
    quadratic_interpolator[dir::f_m00].calculate_all_components(f, f_m00, n_arrays, bs);
    quadratic_interpolator[dir::f_p00].calculate_all_components(f, f_p00, n_arrays, bs);
    return;
  }
  */
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
    f_m00_linear_all_components(f, f_m00, n_arrays, bs);
    f_p00_linear_all_components(f, f_p00, n_arrays, bs);
  }
  else
  {
    double temp_0[n_arrays*bs], temp_1[n_arrays*bs] ONLY3D(COMMA temp_2[n_arrays*bs] COMMA temp_3[n_arrays*bs]);
    ngbd_with_quadratic_interpolation_all_components(f, f_000, f_m00, f_p00, temp_0, temp_1 ONLY3D(COMMA temp_2 COMMA temp_3), n_arrays, bs);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}


void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation_component(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT(bs > 1);
  P4EST_ASSERT(comp < bs);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000+comp];
    quadratic_interpolator[dir::f_m00].calculate_component(f, f_m00, n_arrays, bs, comp);
    quadratic_interpolator[dir::f_p00].calculate_component(f, f_p00, n_arrays, bs, comp);
    return;
  }
  */
  if((check_if_zero(d_m00_m0*inverse_d_max) || check_if_zero(d_m00_p0*inverse_d_max)) && (check_if_zero(d_p00_m0*inverse_d_max) || check_if_zero(d_p00_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_m00_0m*inverse_d_max) || check_if_zero(d_m00_0p*inverse_d_max)) && (check_if_zero(d_p00_0m*inverse_d_max) || check_if_zero(d_p00_0p*inverse_d_max))
   #endif
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000+comp];
    f_m00_linear_component(f, f_m00, n_arrays, bs, comp);
    f_p00_linear_component(f, f_p00, n_arrays, bs, comp);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays] ONLY3D(COMMA temp_2[n_arrays] COMMA temp_3[n_arrays]);
    ngbd_with_quadratic_interpolation_component(f, f_000, f_m00, f_p00, temp_0, temp_1 ONLY3D(COMMA temp_2 COMMA temp_3), n_arrays, bs, comp);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation(const double *f[], double *f_m00, double *f_000, double *f_p00, const unsigned int &n_arrays) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    quadratic_interpolator[dir::f_m00].calculate(f, f_m00, n_arrays);
    quadratic_interpolator[dir::f_p00].calculate(f, f_p00, n_arrays);
    return;
  }
  */
  if((check_if_zero(d_m00_m0*inverse_d_max) || check_if_zero(d_m00_p0*inverse_d_max)) && (check_if_zero(d_p00_m0*inverse_d_max) || check_if_zero(d_p00_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_m00_0m*inverse_d_max) || check_if_zero(d_m00_0p*inverse_d_max)) && (check_if_zero(d_p00_0m*inverse_d_max) || check_if_zero(d_p00_0p*inverse_d_max))
   #endif
     )
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_m00_linear(f, f_m00, n_arrays);
    f_p00_linear(f, f_p00, n_arrays);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays] ONLY3D(COMMA temp_2[n_arrays] COMMA temp_3[n_arrays]);
    ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, temp_0, temp_1 ONLY3D(COMMA temp_2 COMMA temp_3), n_arrays);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned int &n_arrays, const unsigned int &bs) const
{
  P4EST_ASSERT(bs>1);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k){
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs+comp] = f[k][bs_node_000+comp];
    }
    quadratic_interpolator[dir::f_0m0].calculate_all_components(f, f_0m0, n_arrays, bs);
    quadratic_interpolator[dir::f_0p0].calculate_all_components(f, f_0p0, n_arrays, bs);
    return;
  }
  */
  if((check_if_zero(d_0m0_m0*inverse_d_max) || check_if_zero(d_0m0_p0*inverse_d_max)) && (check_if_zero(d_0p0_m0*inverse_d_max) || check_if_zero(d_0p0_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_0m0_0m*inverse_d_max) || check_if_zero(d_0m0_0p*inverse_d_max)) && (check_if_zero(d_0p0_0m*inverse_d_max) || check_if_zero(d_0p0_0p*inverse_d_max))
   #endif
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
        double f_0m0 = f_0m0_linear(f);
        double f_0p0 = f_0p0_linear(f);
        fyy = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0 )*2./(d_0p0+d_0m0);
    }
    f_0m0_linear_all_components(f, f_0m0, n_arrays, bs);
    f_0p0_linear_all_components(f, f_0p0, n_arrays, bs);
  }
  else
  {
    double temp_0[n_arrays*bs], temp_1[n_arrays*bs] ONLY3D(COMMA temp_2[n_arrays*bs] COMMA temp_3[n_arrays*bs]);
    ngbd_with_quadratic_interpolation_all_components(f, f_000, temp_0, temp_1, f_0m0, f_0p0 ONLY3D(COMMA temp_2 COMMA temp_3), n_arrays, bs);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation_component(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT(bs > 1);
  P4EST_ASSERT(comp < bs);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000+comp];
    quadratic_interpolator[dir::f_0m0].calculate_component(f, f_0m0, n_arrays, bs, comp);
    quadratic_interpolator[dir::f_0p0].calculate_component(f, f_0p0, n_arrays, bs, comp);
    return;
  }
  */
  if((check_if_zero(d_0m0_m0*inverse_d_max) || check_if_zero(d_0m0_p0*inverse_d_max)) && (check_if_zero(d_0p0_m0*inverse_d_max) || check_if_zero(d_0p0_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_0m0_0m*inverse_d_max) || check_if_zero(d_0m0_0p*inverse_d_max)) && (check_if_zero(d_0p0_0m*inverse_d_max) || check_if_zero(d_0p0_0p*inverse_d_max))
   #endif
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000+comp];
    f_0m0_linear_component(f, f_0m0, n_arrays, bs, comp);
    f_0p0_linear_component(f, f_0p0, n_arrays, bs, comp);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays] ONLY3D(COMMA temp_2[n_arrays] COMMA temp_3[n_arrays]);
    ngbd_with_quadratic_interpolation_component(f, f_000, temp_0, temp_1, f_0m0, f_0p0 ONLY3D(COMMA temp_2 COMMA temp_3), n_arrays, bs, comp);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation(const double *f[], double *f_0m0, double *f_000, double *f_0p0, const unsigned int &n_arrays) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    quadratic_interpolator[dir::f_0m0].calculate(f, f_0m0, n_arrays);
    quadratic_interpolator[dir::f_0p0].calculate(f, f_0p0, n_arrays);
    return;
  }
  */
  if((check_if_zero(d_0m0_m0*inverse_d_max) || check_if_zero(d_0m0_p0*inverse_d_max)) && (check_if_zero(d_0p0_m0*inverse_d_max) || check_if_zero(d_0p0_p0*inverse_d_max))
   #ifdef P4_TO_P8
     && (check_if_zero(d_0m0_0m*inverse_d_max) || check_if_zero(d_0m0_0p*inverse_d_max)) && (check_if_zero(d_0p0_0m*inverse_d_max) || check_if_zero(d_0p0_0p*inverse_d_max))
   #endif
     )
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_0m0_linear(f, f_0m0, n_arrays);
    f_0p0_linear(f, f_0p0, n_arrays);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays] ONLY3D(COMMA temp_2[n_arrays] COMMA temp_3[n_arrays]);
    ngbd_with_quadratic_interpolation(f, f_000, temp_0, temp_1, f_0m0, f_0p0 ONLY3D(COMMA temp_2 COMMA temp_3), n_arrays);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}

#ifdef P4_TO_P8
void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation_all_components(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned int &n_arrays, const unsigned int &bs) const
{
  P4EST_ASSERT(bs>1);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k){
      const unsigned int kbs = k*bs;
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[kbs+comp] = f[k][bs_node_000+comp];
    }
    quadratic_interpolator[dir::f_00m].calculate_all_components(f, f_00m, n_arrays, bs);
    quadratic_interpolator[dir::f_00p].calculate_all_components(f, f_00p, n_arrays, bs);
    return;
  }
  */
  if((check_if_zero(d_00m_m0*inverse_d_max) || check_if_zero(d_00m_p0*inverse_d_max)) && (check_if_zero(d_00p_m0*inverse_d_max) || check_if_zero(d_00p_p0*inverse_d_max))
     && (check_if_zero(d_00m_0m*inverse_d_max) || check_if_zero(d_00m_0p*inverse_d_max)) && (check_if_zero(d_00p_0m*inverse_d_max) || check_if_zero(d_00p_0p*inverse_d_max))
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
    {
        double f_m00 = f_m00_linear(f);
        double f_p00 = f_p00_linear(f);
        fxx = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2./(d_p00+d_m00);
    }
    f_00m_linear_all_components(f, f_00m, n_arrays, bs);
    f_00p_linear_all_components(f, f_00p, n_arrays, bs);
  }
  else
  {
    double temp_0[n_arrays*bs], temp_1[n_arrays*bs], temp_2[n_arrays*bs], temp_3[n_arrays*bs];
    ngbd_with_quadratic_interpolation_all_components(f, f_000, temp_0, temp_1, temp_2, temp_3, f_00m, f_00p, n_arrays, bs);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}
void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation_component(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT(bs>1);
  P4EST_ASSERT(comp < bs);
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    const unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000+comp];
    quadratic_interpolator[dir::f_00m].calculate_component(f, f_00m, n_arrays, bs, comp);
    quadratic_interpolator[dir::f_00p].calculate_component(f, f_00p, n_arrays, bs, comp);
    return;
  }
  */
  if((check_if_zero(d_00m_m0*inverse_d_max) || check_if_zero(d_00m_p0*inverse_d_max)) && (check_if_zero(d_00p_m0*inverse_d_max) || check_if_zero(d_00p_p0*inverse_d_max))
     && (check_if_zero(d_00m_0m*inverse_d_max) || check_if_zero(d_00m_0p*inverse_d_max)) && (check_if_zero(d_00p_0m*inverse_d_max) || check_if_zero(d_00p_0p*inverse_d_max))
     )
  {
    unsigned int bs_node_000 = bs*node_000;
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs_node_000+comp];
    f_00m_linear_component(f, f_00m, n_arrays, bs, comp);
    f_00p_linear_component(f, f_00p, n_arrays, bs, comp);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays], temp_2[n_arrays], temp_3[n_arrays];
    ngbd_with_quadratic_interpolation_component(f, f_000, temp_0, temp_1, temp_2, temp_3, f_00m, f_00p, n_arrays, bs, comp);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}
void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation(const double *f[], double *f_00m, double *f_000, double *f_00p, const unsigned int &n_arrays) const
{
#ifdef CASL_LOG_TINY_EVENTS
  PetscErrorCode ierr_log_event = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  /*
  if(quadratic_interpolators_are_set)
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    quadratic_interpolator[dir::f_00m].calculate(f, f_00m, n_arrays);
    quadratic_interpolator[dir::f_00p].calculate(f, f_00p, n_arrays);
    return;
  }
  */
  if((check_if_zero(d_00m_m0*inverse_d_max) || check_if_zero(d_00m_p0*inverse_d_max)) && (check_if_zero(d_00p_m0*inverse_d_max) || check_if_zero(d_00p_p0*inverse_d_max))
     && (check_if_zero(d_00m_0m*inverse_d_max) || check_if_zero(d_00m_0p*inverse_d_max)) && (check_if_zero(d_00p_0m*inverse_d_max) || check_if_zero(d_00p_0p*inverse_d_max))
     )
  {
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_00m_linear(f, f_00m, n_arrays);
    f_00p_linear(f, f_00p, n_arrays);
  }
  else
  {
    double temp_0[n_arrays], temp_1[n_arrays], temp_2[n_arrays], temp_3[n_arrays];
    ngbd_with_quadratic_interpolation(f, f_000, temp_0, temp_1, temp_2, temp_3, f_00m, f_00p, n_arrays);
  }
#ifdef CASL_LOG_TINY_EVENTS
  ierr_log_event = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quadratic_interpolation, 0, 0, 0, 0); CHKERRXX(ierr_log_event);
#endif
  return;
}
#endif

void quad_neighbor_nodes_of_node_t::correct_naive_first_derivatives(const double *f[], DIM(const double *naive_Dx, const double *naive_Dy, const double *naive_Dz),  DIM(double *Dx, double *Dy, double *Dz), const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT(comp <= bs);
  // comp == bs means "all components", comp < bs means, only one, comp > bs is not accepted
  const unsigned int nelements = n_arrays*(((bs>1) && (comp==bs))? bs : 1);
  for (unsigned int k = 0; k < nelements; ++k) {
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
    double DIM(fxx[nelements],fyy[nelements], fzz[nelements]);
    if (bs == 1)
      laplace(f, DIM(fxx, fyy, fzz), n_arrays);
    else if (bs > 1 && comp < bs)
      laplace_component(f, DIM(fxx, fyy, fzz), n_arrays, bs, comp);
    else
      laplace_all_components(f, DIM(fxx, fyy, fzz), n_arrays, bs);

    for (unsigned int k = 0; k < nelements; ++k) {
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

/*
void quad_neighbor_nodes_of_node_t::correct_naive_first_derivatives(DIM(node_linear_combination &Dx, node_linear_combination &Dy, node_linear_combination &Dz), DIM(const node_linear_combination &Dxx, const node_linear_combination &Dyy, const node_linear_combination &Dzz)) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    double f_0m0,f_000,f_0p0; y_ngbd_with_quadratic_interpolation(f,f_0m0,f_000,f_0p0);
    double val = ((f_0p0-f_000)/d_0p0*d_0m0+
                  (f_000-f_0m0)/d_0m0*d_0p0)/(d_0m0+d_0p0);

    ierr = PetscLogFlops(7); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return val;
}

void quad_neighbor_nodes_of_node_t::dx_central_internal(const double *f[], double *fx, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  /*
  if(gradient_operator_is_set)
  {
    if(bs > 1){
      if(comp == bs)
        gradient_operator[dir::x].calculate_all_components(f, fx, n_arrays, bs);
      else
        gradient_operator[dir::x].calculate_component(f, fx, n_arrays, bs, comp);
    }
    else
      gradient_operator[dir::x].calculate(f, fx, n_arrays);
    return;
  }
  */
  P4EST_ASSERT(comp <= bs);
  // comp == bs means "all components", comp < bs means, only one, comp > bs is not accepted
  const unsigned int nelements = n_arrays*(((bs>1) && (comp==bs))? bs : 1);
  double f_000[nelements], f_p00[nelements], f_m00[nelements];
  if (bs==1){
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_m00_linear(f, f_m00, n_arrays);
    f_p00_linear(f, f_p00, n_arrays);
  } else if ((bs>1) && (comp < bs)){
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs*node_000+comp];
    f_m00_linear_component(f, f_m00, n_arrays, bs, comp);
    f_p00_linear_component(f, f_p00, n_arrays, bs, comp);
  } else {
    for (unsigned int k = 0; k < n_arrays; ++k)
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[k*bs+comp] = f[k][bs*node_000+comp];
    f_m00_linear_all_components(f, f_m00, n_arrays, bs);
    f_p00_linear_all_components(f, f_p00, n_arrays, bs);
  }
  for (unsigned int k = 0; k < nelements; ++k)
    fx[k] = central_derivative(f_p00[k], f_000[k], f_m00[k], d_p00, d_m00); // naive approach so far

  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0);

  if(second_derivatives_needed)
  {
    double DIM(fxx[nelements],fyy[nelements], fzz[nelements]);
    if (bs == 1)
      laplace(f, DIM(fxx, fyy, fzz), n_arrays);
    else if (bs > 1 && comp < bs)
      laplace_component(f, DIM(fxx, fyy, fzz), n_arrays, bs, comp);
    else
      laplace_all_components(f, DIM(fxx, fyy, fzz), n_arrays, bs);

    for (unsigned int k = 0; k < nelements; ++k) {
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

void quad_neighbor_nodes_of_node_t::dy_central_internal(const double *f[], double *fy, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  /*
  if(gradient_operator_is_set)
  {
    if(bs > 1){
      if(comp == bs)
        gradient_operator[dir::y].calculate_all_components(f, fx, n_arrays, bs);
      else
        gradient_operator[dir::y].calculate_component(f, fx, n_arrays, bs, comp);
    }
    else
      gradient_operator[dir::y].calculate(f, fx, n_arrays);
    return;
  }
  */
  P4EST_ASSERT(comp <= bs);
  // comp == bs means "all components", comp < bs means, only one, comp > bs is not accepted
  const unsigned int nelements = n_arrays*(((bs>1) && (comp==bs))? bs : 1);
  double f_000[nelements], f_0p0[nelements], f_0m0[nelements];
  if (bs==1){
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_0m0_linear(f, f_0m0, n_arrays);
    f_0p0_linear(f, f_0p0, n_arrays);
  } else if ((bs>1) && (comp < bs)){
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs*node_000+comp];
    f_0m0_linear_component(f, f_0m0, n_arrays, bs, comp);
    f_0p0_linear_component(f, f_0p0, n_arrays, bs, comp);
  } else {
    for (unsigned int k = 0; k < n_arrays; ++k)
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[k*bs+comp] = f[k][bs*node_000+comp];
    f_0m0_linear_all_components(f, f_0m0, n_arrays, bs);
    f_0p0_linear_all_components(f, f_0p0, n_arrays, bs);
  }
  for (unsigned int k = 0; k < nelements; ++k)
    fy[k] = central_derivative(f_0p0[k], f_000[k], f_0m0[k], d_0p0, d_0m0); // naive approach so far

    return (f_p00_linear(f)-f[node_000])/d_p00;
}

  if(second_derivatives_needed)
  {
    double DIM(fxx[nelements],fyy[nelements], fzz[nelements]);
    if (bs == 1)
      laplace(f, DIM(fxx, fyy, fzz), n_arrays);
    else if (bs > 1 && comp < bs)
      laplace_component(f, DIM(fxx, fyy, fzz), n_arrays, bs, comp);
    else
      laplace_all_components(f, DIM(fxx, fyy, fzz), n_arrays, bs);
    for (unsigned int k = 0; k < nelements; ++k) {
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
void quad_neighbor_nodes_of_node_t::dz_central_internal(const double *f[], double *fz, const unsigned int &n_arrays, const unsigned int &bs, const unsigned int &comp) const
{
  /*
  if(gradient_operator_is_set)
  {
    if(bs > 1){
      if(comp == bs)
        gradient_operator[dir::z].calculate_all_components(f, fx, n_arrays, bs);
      else
        gradient_operator[dir::z].calculate_component(f, fx, n_arrays, bs, comp);
    }
    else
      gradient_operator[dir::z].calculate(f, fx, n_arrays);
    return;
  }
  */
  P4EST_ASSERT(comp <= bs);
  // comp == bs means "all components", comp < bs means, only one, comp > bs is not accepted
  const unsigned int nelements = n_arrays*(((bs>1) && (comp==bs))? bs : 1);
  double f_000[nelements], f_00p[nelements], f_00m[nelements];
  if (bs==1){
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][node_000];
    f_00m_linear(f, f_00m, n_arrays);
    f_00p_linear(f, f_00p, n_arrays);
  } else if ((bs>1) && (comp < bs)){
    for (unsigned int k = 0; k < n_arrays; ++k)
      f_000[k] = f[k][bs*node_000+comp];
    f_00m_linear_component(f, f_00m, n_arrays, bs, comp);
    f_00p_linear_component(f, f_00p, n_arrays, bs, comp);
  } else {
    for (unsigned int k = 0; k < n_arrays; ++k)
      for (unsigned int comp = 0; comp < bs; ++comp)
        f_000[k*bs+comp] = f[k][bs*node_000+comp];
    f_00m_linear_all_components(f, f_00m, n_arrays, bs);
    f_00p_linear_all_components(f, f_00p, n_arrays, bs);
  }
  for (unsigned int k = 0; k < nelements; ++k)
    fz[k] = central_derivative(f_00p[k], f_000[k], f_00m[k], d_00p, d_00m); // naive approach so far

  double xx_correction_weight_to_df_dz, yy_correction_weight_to_df_dz;
  // correct it if needed
  bool second_derivatives_needed = false;
  const bool Dz_needs_xx_correction = naive_dz_needs_xx_correction(xx_correction_weight_to_df_dz); second_derivatives_needed = second_derivatives_needed || Dz_needs_xx_correction;
  const bool Dz_needs_yy_correction = naive_dz_needs_yy_correction(yy_correction_weight_to_df_dz); second_derivatives_needed = second_derivatives_needed || Dz_needs_yy_correction;

  if(second_derivatives_needed)
  {
    double fxx[nelements],fyy[nelements], fzz[nelements];
    if (bs==1)
      laplace(f, fxx, fyy, fzz, n_arrays);
    else if ((bs > 1) && (comp < bs))
      laplace_component(f, fxx, fyy, fzz, n_arrays, bs, comp);
    else
      laplace_all_components(f, fxx, fyy, fzz, n_arrays, bs);
    for (unsigned int k = 0; k < nelements; ++k) {
      if(Dz_needs_xx_correction)
        fz[k] -= fxx[k]*xx_correction_weight_to_df_dz;
      if(Dz_needs_yy_correction)
        fz[k] -= fyy[k]*yy_correction_weight_to_df_dz;
    }
  }
  return;
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);
  const double f_xx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  double f_xx_m00;
  /*
  if(linear_interpolators_are_set)
    f_xx_m00 = linear_interpolator[dir::f_m00].calculate_dd(dir::x, f, neighbors);
  else
  {*/
  node_linear_combination lin_m00(1<<(P4EST_DIM-1));
  get_linear_interpolator(lin_m00, node_m00_mm, node_m00_pm ONLY3D(COMMA node_m00_mp COMMA node_m00_pp), d_m00_m0, d_m00_p0 ONLY3D(COMMA d_m00_0m COMMA d_m00_0p));

  f_xx_m00 = lin_m00.calculate_dd(dir::x, f, neighbors);
  /*}*/
  return d_backward_quadratic(f_000, f_m00, d_m00, f_xx_000, f_xx_m00);
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);
  const double f_xx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  double f_xx_p00;
  /*
  if(linear_interpolators_are_set)
    f_xx_p00 = linear_interpolator[dir::f_p00].calculate_dd(dir::x, f, neighbors);
  else
  {*/
  node_linear_combination lin_p00(1<<(P4EST_DIM-1));
  get_linear_interpolator(lin_p00, node_p00_mm, node_p00_pm ONLY3D(COMMA node_p00_mp COMMA node_p00_pp), d_p00_m0, d_p00_p0 ONLY3D(COMMA d_p00_0m COMMA d_p00_0p));
  f_xx_p00 = lin_p00.calculate_dd(dir::x, f, neighbors);
  /*}*/
  return d_forward_quadratic(f_p00, f_000, d_p00, f_xx_000, f_xx_p00);
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);
  const double f_yy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double f_yy_0m0;
  /*
  if(linear_interpolators_are_set)
    linear_interpolator[dir::f_0m0].calculate_dd(dir::y, f, neighbors);
  else
  {*/
  node_linear_combination lin_0m0(1<<(P4EST_DIM-1));
  get_linear_interpolator(lin_0m0, node_0m0_mm, node_0m0_pm ONLY3D(COMMA node_0m0_mp COMMA node_0m0_pp), d_0m0_m0, d_0m0_p0 ONLY3D(COMMA d_0m0_0m COMMA d_0m0_0p));
  f_yy_0m0 = lin_0m0.calculate_dd(dir::y, f, neighbors);
  /*}*/
  return d_backward_quadratic(f_000, f_0m0, d_0m0, f_yy_000, f_yy_0m0);
}

double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);
  const double f_yy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double f_yy_0p0;
  /*
  if(linear_interpolators_are_set)
    linear_interpolator[dir::f_0p0].calculate_dd(dir::y, f, neighbors);
  else
  {*/
  node_linear_combination lin_0p0(1<<(P4EST_DIM-1));
  get_linear_interpolator(lin_0p0, node_0p0_mm, node_0p0_pm ONLY3D(COMMA node_0p0_mp COMMA node_0p0_pp), d_0p0_m0, d_0p0_p0 ONLY3D(COMMA d_0p0_0m COMMA d_0p0_0p));
  f_yy_0p0 = lin_0p0.calculate_dd(dir::y, f, neighbors);
  /*}*/
  return d_forward_quadratic(f_0p0, f_000, d_0p0, f_yy_000, f_yy_0p0);
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const double *fxx) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);
  const double f_zz_000 = central_second_derivative(f_00p, f_000, f_00m, d_00p, d_00m);
  double f_zz_00m;
  /*
  if(linear_interpolators_are_set)
    linear_interpolator[dir::f_00m].calculate_dd(dir::z, f, neighbors);
  else
  {*/
  node_linear_combination lin_00m(1<<(P4EST_DIM-1));
  get_linear_interpolator(lin_00m, node_00m_mm, node_00m_pm, node_00m_mp, node_00m_pp, d_00m_m0, d_00m_p0, d_00m_0m, d_00m_0p);
  f_zz_00m = lin_00m.calculate_dd(dir::z, f, neighbors);
  /*}*/
  return d_backward_quadratic(f_000, f_00m, d_00m, f_zz_000, f_zz_00m);
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const double *fxx) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);
  const double f_zz_000 = central_second_derivative(f_00p, f_000, f_00m, d_00p, d_00m);
  double f_zz_00p;
  /*
  if(linear_interpolators_are_set)
    linear_interpolator[dir::f_00p].calculate_dd(dir::z, f, neighbors);
  else
  {*/
  node_linear_combination lin_00p(1<<(P4EST_DIM-1));
  get_linear_interpolator(lin_00p, node_00p_mm, node_00p_pm, node_00p_mp, node_00p_pp, d_00p_m0, d_00p_p0, d_00p_0m, d_00p_0p);
  f_zz_00p = lin_00p.calculate_dd(dir::z, f, neighbors);
  /*}*/
  return d_forward_quadratic(f_00p, f_000, d_00p, f_zz_000, f_zz_00p);
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const double *fyy) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);

  double fyy_000 = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2.0/(d_0m0+d_0p0);
  double fyy_0m0 = f_0m0_linear(fyy);

  return (f_000-f_0m0)/d_0m0 + 0.5*d_0m0*MINMOD(fyy_000,fyy_0m0);
}

double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const double *fyy) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);

  double fyy_000 = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2.0/(d_0m0+d_0p0);
  double fyy_0p0 = f_0p0_linear(fyy);

  return (f_0p0-f_000)/d_0p0 - 0.5*d_0p0*MINMOD(fyy_000,fyy_0p0);
}

double quad_neighbor_nodes_of_node_t::dxx_central( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    double f_m00,f_000,f_p00; x_ngbd_with_quadratic_interpolation(f,f_m00,f_000,f_p00);
    double val = ((f_p00-f_000)/d_p00+(f_m00-f_000)/d_m00)*2./(d_m00+d_p00);

    ierr = PetscLogFlops(8); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return val;
}

double quad_neighbor_nodes_of_node_t::dyy_central( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    double f_0m0,f_000,f_0p0; y_ngbd_with_quadratic_interpolation(f,f_0m0,f_000,f_0p0);
    double val = ((f_0p0-f_000)/d_0p0+(f_0m0-f_000)/d_0m0)*2./(d_0m0+d_0p0);

    ierr = PetscLogFlops(8); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return val;
}

void quad_neighbor_nodes_of_node_t::laplace(const double *f, double &fxx, double &fyy) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr);

  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0);

  fxx = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2.0/(d_m00+d_p00);
  fyy = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2.0/(d_0m0+d_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr);
}

double quad_neighbor_nodes_of_node_t::dxx_central_on_m00(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central_m00, 0, 0, 0, 0); CHKERRXX(ierr);

  // FIXME: These kind of operations would be expensive if neighbors is not initialized!
  double fxx_m00_mm = 0, fxx_m00_pm = 0;
  if (d_m00_p0 != 0) { fxx_m00_mm = neighbors.get_neighbors(node_m00_mm).dxx_central(f); }
  if (d_m00_m0 != 0) { fxx_m00_pm = neighbors.get_neighbors(node_m00_pm).dxx_central(f); }

  ierr = PetscLogFlops(5); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central_m00, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fxx_m00_mm*d_m00_p0 + fxx_m00_pm*d_m00_m0)/(d_m00_m0+d_m00_p0);
}

double quad_neighbor_nodes_of_node_t::dxx_central_on_p00(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central_p00, 0, 0, 0, 0); CHKERRXX(ierr);

  double fxx_p00_mm = 0, fxx_p00_pm = 0;
  if (d_p00_p0 != 0) { fxx_p00_mm = neighbors.get_neighbors(node_p00_mm).dxx_central(f); }
  if (d_p00_m0 != 0) { fxx_p00_pm = neighbors.get_neighbors(node_p00_pm).dxx_central(f); }

  ierr = PetscLogFlops(5); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central_p00, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fxx_p00_mm*d_p00_p0 + fxx_p00_pm*d_p00_m0)/(d_p00_m0+d_p00_p0);
}

double quad_neighbor_nodes_of_node_t::dyy_central_on_0m0(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central_0m0, 0, 0, 0, 0); CHKERRXX(ierr);

  double fyy_0m0_mm = 0, fyy_0m0_pm = 0;
  if (d_0m0_p0 != 0) { fyy_0m0_mm = neighbors.get_neighbors(node_0m0_mm).dyy_central(f); }
  if (d_0m0_m0 != 0) { fyy_0m0_pm = neighbors.get_neighbors(node_0m0_pm).dyy_central(f); }

  ierr = PetscLogFlops(5); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central_0m0, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fyy_0m0_mm*d_0m0_p0 + fyy_0m0_pm*d_0m0_m0)/(d_0m0_m0+d_0m0_p0);
}

double quad_neighbor_nodes_of_node_t::dyy_central_on_0p0(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central_0p0, 0, 0, 0, 0); CHKERRXX(ierr);

  double fyy_0p0_mm = 0, fyy_0p0_pm = 0;
  if (d_0p0_p0 != 0) { fyy_0p0_mm = neighbors.get_neighbors(node_0p0_mm).dyy_central(f); }
  if (d_0p0_m0 != 0) { fyy_0p0_pm = neighbors.get_neighbors(node_0p0_pm).dyy_central(f); }

  ierr = PetscLogFlops(5); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central_0p0, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fyy_0p0_mm*d_0p0_p0 + fyy_0p0_pm*d_0p0_m0)/(d_0p0_m0+d_0p0_p0);
}
