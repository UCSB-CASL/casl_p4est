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
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_forward_linear;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_backward_linear;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_forward_linear;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_backward_linear;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_forward_quadratic;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_backward_quadratic;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_forward_quadratic;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_backward_quadratic;
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

double node_interpolator::interpolate_dxx(const double *node_sample_field, const my_p4est_node_neighbors_t &neighbors) const
{
  P4EST_ASSERT(elements.size()>0);
  double value = elements[0].weight*(neighbors.get_neighbors(elements[0].node_idx).dxx_central(node_sample_field));
  for (size_t k = 1; k < elements.size(); ++k)
    value += elements[k].weight*(neighbors.get_neighbors(elements[k].node_idx).dxx_central(node_sample_field));
  return value;
}
double node_interpolator::interpolate_dyy(const double *node_sample_field, const my_p4est_node_neighbors_t &neighbors) const
{
  P4EST_ASSERT(elements.size()>0);
  double value = elements[0].weight*(neighbors.get_neighbors(elements[0].node_idx).dyy_central(node_sample_field));
  for (size_t k = 1; k < elements.size(); ++k)
    value += elements[k].weight*(neighbors.get_neighbors(elements[k].node_idx).dyy_central(node_sample_field));
  return value;
}

void quad_neighbor_nodes_of_node_t::f_m00_linear( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields>0);
  node_interpolator interpolator;
  linear_interpolator_m00(interpolator);
  for (unsigned int k = 0; k < n_fields; ++k)
  {
    results[k] = interpolator.interpolate(f[k]);
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2*interpolator.elements.size()-1); CHKERRXX(ierr);
#endif
  }
}

void quad_neighbor_nodes_of_node_t::f_p00_linear( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields>0);
  node_interpolator interpolator;
  linear_interpolator_p00(interpolator);
  for (unsigned int k = 0; k < n_fields; ++k)
  {
    results[k] = interpolator.interpolate(f[k]);
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2*interpolator.elements.size()-1); CHKERRXX(ierr);
#endif
  }
}

void quad_neighbor_nodes_of_node_t::f_0m0_linear( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields>0);
  node_interpolator interpolator;
  linear_interpolator_0m0(interpolator);
  for (unsigned int k = 0; k < n_fields; ++k)
  {
    results[k] = interpolator.interpolate(f[k]);
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2*interpolator.elements.size()-1); CHKERRXX(ierr);
#endif
  }
}

void quad_neighbor_nodes_of_node_t::f_0p0_linear( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields>0);
  node_interpolator interpolator;
  linear_interpolator_0p0(interpolator);
  for (unsigned int k = 0; k < n_fields; ++k)
  {
    results[k] = interpolator.interpolate(f[k]);
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2*interpolator.elements.size()-1); CHKERRXX(ierr);
#endif
  }
}

void quad_neighbor_nodes_of_node_t::linear_interpolator_m00(node_interpolator& interpolator) const
{
  if(d_m00_p0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_m00_pm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else if(d_m00_m0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_m00_mm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else
  {
    interpolator.elements.resize(2);
    interpolator.elements[0].node_idx  = node_m00_mm;
    interpolator.elements[0].weight    = d_m00_p0/(d_m00_m0+d_m00_p0);
    interpolator.elements[1].node_idx  = node_m00_pm;
    interpolator.elements[1].weight    = 1.0-interpolator.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
    return;
  }
}

void quad_neighbor_nodes_of_node_t::linear_interpolator_p00(node_interpolator& interpolator) const
{
  if(d_p00_p0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_p00_pm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else if(d_p00_m0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_p00_mm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else
  {
    interpolator.elements.resize(2);
    interpolator.elements[0].node_idx  = node_p00_mm;
    interpolator.elements[0].weight    = d_p00_p0/(d_p00_m0+d_p00_p0);
    interpolator.elements[1].node_idx  = node_p00_pm;
    interpolator.elements[1].weight    = 1.0-interpolator.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
    return;
  }
}

void quad_neighbor_nodes_of_node_t::linear_interpolator_0m0(node_interpolator& interpolator) const
{
  if(d_0m0_m0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_0m0_mm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else if(d_0m0_p0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_0m0_pm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else
  {
    interpolator.elements.resize(2);
    interpolator.elements[0].node_idx  = node_0m0_pm;
    interpolator.elements[0].weight    = d_0m0_m0/(d_0m0_m0+d_0m0_p0);
    interpolator.elements[1].node_idx  = node_0m0_mm;
    interpolator.elements[1].weight    = 1.0-interpolator.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
    return;
  }
}

void quad_neighbor_nodes_of_node_t::linear_interpolator_0p0(node_interpolator& interpolator) const
{
  if(d_0p0_m0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_0p0_mm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else if(d_0p0_p0==0)
  {
    interpolator.elements.resize(1);
    interpolator.elements[0].node_idx  = node_0p0_pm;
    interpolator.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return;
  }
  else
  {
    interpolator.elements.resize(2);
    interpolator.elements[0].node_idx  = node_0p0_pm;
    interpolator.elements[0].weight    = d_0p0_m0/(d_0p0_m0+d_0p0_p0);
    interpolator.elements[1].node_idx  = node_0p0_mm;
    interpolator.elements[1].weight    = 1.0-interpolator.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
    return;
  }
}

void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation(const double *f,
                                                                      double &f_000, double &f_m00, double &f_p00, double &f_0m0, double &f_0p0,
                                                                      const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                                      const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  f_000 = f[node_000];

  double fyy=0; if(d_p00_m0*d_p00_p0!=0 || d_m00_m0*d_m00_p0!=0) fyy = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double fxx=0; if(d_0m0_m0*d_0m0_p0!=0 || d_0p0_m0*d_0p0_p0!=0) fxx = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);

  f_m00 = interp_m00.interpolate(f) - 0.5*d_m00_m0*d_m00_p0*fyy;
  f_p00 = interp_p00.interpolate(f) - 0.5*d_p00_m0*d_p00_p0*fyy;
  f_0m0 = interp_0m0.interpolate(f) - 0.5*d_0m0_m0*d_0m0_p0*fxx;
  f_0p0 = interp_0p0.interpolate(f) - 0.5*d_0p0_m0*d_0p0_p0*fxx;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(12); CHKERRXX(ierr);
#endif
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation( const double *f,
                                                                         double& f_m00, double& f_000, double& f_p00,
                                                                         const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                                                         const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{
  f_000 = f[node_000];

  double fyy=0;
  if(d_p00_m0*d_p00_p0!=0 || d_m00_m0*d_m00_p0!=0)
  {
    double f_0m0 = interp_0m0.interpolate(f);
    double f_0p0 = interp_0p0.interpolate(f);
    fyy = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  }
  f_m00 = interp_m00.interpolate(f) - 0.5*d_m00_m0*d_m00_p0*fyy;
  f_p00 = interp_p00.interpolate(f) - 0.5*d_p00_m0*d_p00_p0*fyy;
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr = PetscLogFlops(6); CHKERRXX(ierr);
#endif
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation( const double *f, double& f_0m0, double& f_000, double& f_0p0,
                                                                         const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                                                         const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{
  f_000 = f[node_000];

  double fxx=0;
  if(d_0m0_m0*d_0m0_p0!=0 || d_0p0_m0*d_0p0_p0!=0)
  {
    double f_m00 = interp_m00.interpolate(f);
    double f_p00 = interp_p00.interpolate(f);
    fxx = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  }
  f_0m0 = interp_0m0.interpolate(f) - 0.5*d_m00_m0*d_m00_p0*fxx;
  f_0p0 = interp_0p0.interpolate(f) - 0.5*d_p00_m0*d_p00_p0*fxx;
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr = PetscLogFlops(6); CHKERRXX(ierr);
#endif
}

double quad_neighbor_nodes_of_node_t::dx_central ( const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00, const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, interp_m00, interp_p00, interp_0m0, interp_0p0);
  return  central_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
}

double quad_neighbor_nodes_of_node_t::dx_forward_linear ( const double *f, const node_interpolator& interp_p00) const
{
  return  forward_derivative(interp_p00.interpolate(f), f[node_000], d_p00);
}

double quad_neighbor_nodes_of_node_t::dx_backward_linear ( const double *f, const node_interpolator& interp_m00) const
{
  return  backward_derivative(f[node_000], interp_m00.interpolate(f), d_m00);
}

double quad_neighbor_nodes_of_node_t::dy_forward_linear ( const double *f, const node_interpolator& interp_0p0) const
{
  return  forward_derivative(interp_0p0.interpolate(f), f[node_000], d_0p0);
}

double quad_neighbor_nodes_of_node_t::dy_backward_linear ( const double *f, const node_interpolator& interp_0m0) const
{
  return  backward_derivative(f[node_000], interp_0m0.interpolate(f), d_0m0);
}

double quad_neighbor_nodes_of_node_t::dy_central ( const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00, const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);
  return  central_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

double quad_neighbor_nodes_of_node_t::dxx_central( const double *f,
                                                   const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                                   const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, interp_m00, interp_p00, interp_0m0, interp_0p0);
  return central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
}

double quad_neighbor_nodes_of_node_t::dyy_central( const double *f,
                                                   const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                                   const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);
  return central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

void quad_neighbor_nodes_of_node_t::laplace( const double *f,
                                             double &fxx, double &fyy,
                                             const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                             const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const
{

  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);
  fxx = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  fyy = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

void quad_neighbor_nodes_of_node_t::grad(const double *f, double &df_dx, double &df_dy,
                                         const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                         const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);
  df_dx = central_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  df_dy = central_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                            const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                            const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fxx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  double fxx_m00 = interp_m00.interpolate_dxx(f, neighbors);
  return (backward_derivative(f_000, f_m00, d_m00) + 0.5*d_m00*MINMOD(fxx_000,fxx_m00));
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const double *fxx,
                                                            const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                            const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fxx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  double fxx_m00 = interp_m00.interpolate(fxx);
  return (backward_derivative(f_000, f_m00, d_m00) + 0.5*d_m00*MINMOD(fxx_000,fxx_m00));
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                           const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                           const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fxx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  double fxx_p00 = interp_p00.interpolate_dxx(f, neighbors);
  return (forward_derivative(f_p00, f_000, d_p00) - 0.5*d_m00*MINMOD(fxx_000,fxx_p00));
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const double* fxx,
                                                           const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                           const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fxx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  double fxx_p00 = interp_p00.interpolate(fxx);
  return (forward_derivative(f_p00, f_000, d_p00) - 0.5*d_m00*MINMOD(fxx_000,fxx_p00));
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                            const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                            const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fyy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double fyy_0m0 = interp_0m0.interpolate_dyy(f, neighbors);
  return (backward_derivative(f_000, f_0m0, d_0m0) + 0.5*d_0m0*MINMOD(fyy_000,fyy_0m0));
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const double *fyy,
                                                            const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                            const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fyy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double fyy_0m0 = interp_0m0.interpolate(fyy);
  return (backward_derivative(f_000, f_0m0, d_0m0) + 0.5*d_0m0*MINMOD(fyy_000,fyy_0m0));
}

double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                           const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                           const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{

  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fyy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double fyy_0p0 = interp_0p0.interpolate_dyy(f, neighbors);
  return (forward_derivative(f_0p0, f_000, d_0p0) - 0.5*d_0p0*MINMOD(fyy_000,fyy_0p0));
}

double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const double *fyy,
                                                           const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                                                           const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const
{

  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, interp_m00, interp_p00, interp_0m0, interp_0p0);

  double fyy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double fyy_0p0 = interp_0p0.interpolate(fyy);
  return (forward_derivative(f_0p0, f_000, d_0p0) - 0.5*d_0p0*MINMOD(fyy_000,fyy_0p0));
}

void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation(const double *f[], double f_000[],
                                                                      double f_m00[], double f_p00[],
                                                                      double f_0m0[], double f_0p0[],
                                                                      const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields>0);
  PetscErrorCode ierr;
  P4EST_ASSERT(n_fields > 0);
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    ngbd_with_quadratic_interpolation(f[k], f_000[k], f_m00[k], f_p00[k], f_0m0[k], f_0p0[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation(const double *f[],
                                                                        double f_m00[], double f_000[], double f_p00[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    x_ngbd_with_quadratic_interpolation(f[k], f_m00[k], f_000[k], f_p00[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation(const double *f[],
                                                                        double f_0m0[], double f_000[], double f_0p0[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    y_ngbd_with_quadratic_interpolation(f[k], f_0m0[k], f_000[k], f_0p0[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_central ( const double *f[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_central(f[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_central ( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_central(f[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::gradient(const double *f[], double fx[], double fy[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields>0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);
  for (unsigned int k = 0; k < n_fields; ++k)
    grad(f[k], fx[k], fy[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::gradient(const double *f[], double grad_f[][P4EST_DIM], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields>0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);
  for (unsigned int k = 0; k < n_fields; ++k)
    grad(f[k], grad_f[k][0], grad_f[k][1], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_forward_linear ( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_forward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_p00;
  linear_interpolator_p00(interpolator_p00);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_forward_linear(f[k], interpolator_p00);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_forward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_backward_linear( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_backward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00;
  linear_interpolator_m00(interpolator_m00);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_backward_linear(f[k], interpolator_m00);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_backward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_forward_linear ( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_forward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_0p0;
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_forward_linear(f[k], interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_forward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_backward_linear( const double *f[], double results[], const unsigned int& n_fields ) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_backward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_0m0;
  linear_interpolator_0m0(interpolator_0m0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_backward_linear(f[k], interpolator_0m0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_backward_linear, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f[], const my_p4est_node_neighbors_t &neighbors, double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_backward_quadratic(f[k], neighbors, interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f[], const my_p4est_node_neighbors_t &neighbors, double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_forward_quadratic(f[k], neighbors, interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f[], const my_p4est_node_neighbors_t &neighbors, double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_backward_quadratic(f[k], neighbors, interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f[], const my_p4est_node_neighbors_t &neighbors, double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_forward_quadratic(f[k], neighbors, interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f[], const double *fxx[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_backward_quadratic(f[k], fxx[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f[], const double *fxx[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dx_forward_quadratic(f[k], fxx[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f[], const double *fyy[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_backward_quadratic(f[k], fyy[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_backward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f[], const double *fyy[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);

  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dy_forward_quadratic(f[k], fyy[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_forward_quadratic, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dxx_central( const double *f[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);
  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dxx_central(f[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::dyy_central( const double *f[], double results[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);
  for (unsigned int k = 0; k < n_fields; ++k)
    results[k] = dyy_central(f[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::laplace(const double *f[], double fxx[], double fyy[], const unsigned int& n_fields) const
{
  P4EST_ASSERT(n_fields > 0);
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr);
  node_interpolator interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0;
  linear_interpolator_m00(interpolator_m00);
  linear_interpolator_p00(interpolator_p00);
  linear_interpolator_0m0(interpolator_0m0);
  linear_interpolator_0p0(interpolator_0p0);
  for (unsigned int k = 0; k < n_fields; ++k)
    laplace(f[k], fxx[k], fyy[k], interpolator_m00, interpolator_p00, interpolator_0m0, interpolator_0p0);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr);
}


// the following are no longer needed --> replaced within the node_interpolator structure

//double quad_neighbor_nodes_of_node_t::dxx_central_on_m00(const double *f, const my_p4est_node_neighbors_t &neighbors) const
//{
//  PetscErrorCode ierr;
//  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central_m00, 0, 0, 0, 0); CHKERRXX(ierr);

//  // FIXME: These kind of operations would be expensive if neighbors is not initialized!
//  double fxx_m00_mm = 0, fxx_m00_pm = 0;
//  if (d_m00_p0 != 0) { fxx_m00_mm = neighbors.get_neighbors(node_m00_mm).dxx_central(f); }
//  if (d_m00_m0 != 0) { fxx_m00_pm = neighbors.get_neighbors(node_m00_pm).dxx_central(f); }

//  ierr = PetscLogFlops(5); CHKERRXX(ierr);
//  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central_m00, 0, 0, 0, 0); CHKERRXX(ierr);

//  return (fxx_m00_mm*d_m00_p0 + fxx_m00_pm*d_m00_m0)/(d_m00_m0+d_m00_p0);
//}

//double quad_neighbor_nodes_of_node_t::dxx_central_on_p00(const double *f, const my_p4est_node_neighbors_t &neighbors) const
//{
//  PetscErrorCode ierr;
//  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central_p00, 0, 0, 0, 0); CHKERRXX(ierr);

//  double fxx_p00_mm = 0, fxx_p00_pm = 0;
//  if (d_p00_p0 != 0) { fxx_p00_mm = neighbors.get_neighbors(node_p00_mm).dxx_central(f); }
//  if (d_p00_m0 != 0) { fxx_p00_pm = neighbors.get_neighbors(node_p00_pm).dxx_central(f); }

//  ierr = PetscLogFlops(5); CHKERRXX(ierr);
//  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central_p00, 0, 0, 0, 0); CHKERRXX(ierr);

//  return (fxx_p00_mm*d_p00_p0 + fxx_p00_pm*d_p00_m0)/(d_p00_m0+d_p00_p0);
//}

//double quad_neighbor_nodes_of_node_t::dyy_central_on_0m0(const double *f, const my_p4est_node_neighbors_t &neighbors) const
//{
//  PetscErrorCode ierr;
//  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central_0m0, 0, 0, 0, 0); CHKERRXX(ierr);

//  double fyy_0m0_mm = 0, fyy_0m0_pm = 0;
//  if (d_0m0_p0 != 0) { fyy_0m0_mm = neighbors.get_neighbors(node_0m0_mm).dyy_central(f); }
//  if (d_0m0_m0 != 0) { fyy_0m0_pm = neighbors.get_neighbors(node_0m0_pm).dyy_central(f); }

//  ierr = PetscLogFlops(5); CHKERRXX(ierr);
//  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central_0m0, 0, 0, 0, 0); CHKERRXX(ierr);

//  return (fyy_0m0_mm*d_0m0_p0 + fyy_0m0_pm*d_0m0_m0)/(d_0m0_m0+d_0m0_p0);
//}

//double quad_neighbor_nodes_of_node_t::dyy_central_on_0p0(const double *f, const my_p4est_node_neighbors_t &neighbors) const
//{
//  PetscErrorCode ierr;
//  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central_0p0, 0, 0, 0, 0); CHKERRXX(ierr);

//  double fyy_0p0_mm = 0, fyy_0p0_pm = 0;
//  if (d_0p0_p0 != 0) { fyy_0p0_mm = neighbors.get_neighbors(node_0p0_mm).dyy_central(f); }
//  if (d_0p0_m0 != 0) { fyy_0p0_pm = neighbors.get_neighbors(node_0p0_pm).dyy_central(f); }

//  ierr = PetscLogFlops(5); CHKERRXX(ierr);
//  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central_0p0, 0, 0, 0, 0); CHKERRXX(ierr);

//  return (fyy_0p0_mm*d_0p0_p0 + fyy_0p0_pm*d_0p0_m0)/(d_0p0_m0+d_0p0_p0);
//}
