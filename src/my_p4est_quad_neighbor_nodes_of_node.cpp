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

inline double node_interpolator::interpolate_dd( const unsigned char der, const double *node_sample_field, const my_p4est_node_neighbors_t &neighbors, const unsigned int &bs, const unsigned int &comp) const
{
  P4EST_ASSERT(der < P4EST_DIM);
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(elements.size()>0);
  double value = elements[0].weight*(neighbors.get_neighbors(elements[0].node_idx).dd_central(der, node_sample_field, bs, comp));
  for (size_t k = 1; k < elements.size(); ++k)
    value += elements[k].weight*(neighbors.get_neighbors(elements[k].node_idx).dd_central(der, node_sample_field, bs, comp));
  return value;
}

void quad_neighbor_nodes_of_node_t::set_linear_interpolator_m00( )
{
  if(interpolator_m00.is_set)
    return;
  if(d_m00_p0==0)
  {
    interpolator_m00.elements.resize(1);
    interpolator_m00.elements[0].node_idx  = node_m00_pm;
    interpolator_m00.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else if(d_m00_m0==0)
  {
    interpolator_m00.elements.resize(1);
    interpolator_m00.elements[0].node_idx  = node_m00_mm;
    interpolator_m00.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else
  {
    interpolator_m00.elements.resize(2);
    interpolator_m00.elements[0].node_idx  = node_m00_mm;
    interpolator_m00.elements[0].weight    = d_m00_p0/(d_m00_m0+d_m00_p0);
    interpolator_m00.elements[1].node_idx  = node_m00_pm;
    interpolator_m00.elements[1].weight    = 1.0-interpolator_m00.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
  }
  interpolator_m00.is_set = true;
  return;
}

void quad_neighbor_nodes_of_node_t::set_linear_interpolator_p00( )
{
  if(interpolator_p00.is_set)
    return;
  if(d_p00_p0==0)
  {
    interpolator_p00.elements.resize(1);
    interpolator_p00.elements[0].node_idx  = node_p00_pm;
    interpolator_p00.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else if(d_p00_m0==0)
  {
    interpolator_p00.elements.resize(1);
    interpolator_p00.elements[0].node_idx  = node_p00_mm;
    interpolator_p00.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else
  {
    interpolator_p00.elements.resize(2);
    interpolator_p00.elements[0].node_idx  = node_p00_mm;
    interpolator_p00.elements[0].weight    = d_p00_p0/(d_p00_m0+d_p00_p0);
    interpolator_p00.elements[1].node_idx  = node_p00_pm;
    interpolator_p00.elements[1].weight    = 1.0-interpolator_p00.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
  }
  interpolator_p00.is_set = true;
  return;
}

void quad_neighbor_nodes_of_node_t::set_linear_interpolator_0m0( )
{
  if(interpolator_0m0.is_set)
    return;
  if(d_0m0_m0==0)
  {
    interpolator_0m0.elements.resize(1);
    interpolator_0m0.elements[0].node_idx  = node_0m0_mm;
    interpolator_0m0.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else if(d_0m0_p0==0)
  {
    interpolator_0m0.elements.resize(1);
    interpolator_0m0.elements[0].node_idx  = node_0m0_pm;
    interpolator_0m0.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else
  {
    interpolator_0m0.elements.resize(2);
    interpolator_0m0.elements[0].node_idx  = node_0m0_pm;
    interpolator_0m0.elements[0].weight    = d_0m0_m0/(d_0m0_m0+d_0m0_p0);
    interpolator_0m0.elements[1].node_idx  = node_0m0_mm;
    interpolator_0m0.elements[1].weight    = 1.0-interpolator_0m0.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
  }
  interpolator_0m0.is_set = true;
  return;
}

void quad_neighbor_nodes_of_node_t::set_linear_interpolator_0p0( )
{
  if(interpolator_0p0.is_set)
    return;
  if(d_0p0_m0==0)
  {
    interpolator_0p0.elements.resize(1);
    interpolator_0p0.elements[0].node_idx  = node_0p0_mm;
    interpolator_0p0.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else if(d_0p0_p0==0)
  {
    interpolator_0p0.elements.resize(1);
    interpolator_0p0.elements[0].node_idx  = node_0p0_pm;
    interpolator_0p0.elements[0].weight    = 1.0;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
  }
  else
  {
    interpolator_0p0.elements.resize(2);
    interpolator_0p0.elements[0].node_idx  = node_0p0_pm;
    interpolator_0p0.elements[0].weight    = d_0p0_m0/(d_0p0_m0+d_0p0_p0);
    interpolator_0p0.elements[1].node_idx  = node_0p0_mm;
    interpolator_0p0.elements[1].weight    = 1.0-interpolator_0p0.elements[0].weight;
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(5); CHKERRXX(ierr);
#endif
  }
  interpolator_0p0.is_set = true;
  return;
}

void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation( const double *f,
                                                                       double& f_000, double& f_m00, double& f_p00, double& f_0m0, double& f_0p0,
                                                                       const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_m00.is_set && interpolator_p00.is_set && interpolator_0m0.is_set && interpolator_0p0.is_set);
  f_000 = f[bs*node_000+comp];
  f_m00 = interpolator_m00.interpolate(f, bs, comp);
  f_p00 = interpolator_p00.interpolate(f, bs, comp);
  f_0m0 = interpolator_0m0.interpolate(f, bs, comp);
  f_0p0 = interpolator_0p0.interpolate(f, bs, comp);

  double fyy=0; if(d_p00_m0*d_p00_p0!=0 || d_m00_m0*d_m00_p0!=0) fyy = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  double fxx=0; if(d_0m0_m0*d_0m0_p0!=0 || d_0p0_m0*d_0p0_p0!=0) fxx = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);

  f_m00 -= 0.5*d_m00_m0*d_m00_p0*fyy;
  f_p00 -= 0.5*d_p00_m0*d_p00_p0*fyy;
  f_0m0 -= 0.5*d_0m0_m0*d_0m0_p0*fxx;
  f_0p0 -= 0.5*d_0p0_m0*d_0p0_p0*fxx;
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr = PetscLogFlops(12); CHKERRXX(ierr);
#endif
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation( const double *f,
                                                                         double& f_m00, double& f_000, double& f_p00,
                                                                         const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_m00.is_set && interpolator_p00.is_set && interpolator_0m0.is_set && interpolator_0p0.is_set);
  f_000 = f[bs*node_000+comp];

  double fyy=0;
  if(d_p00_m0*d_p00_p0!=0 || d_m00_m0*d_m00_p0!=0)
  {
    double f_0m0 = interpolator_0m0.interpolate(f, bs, comp);
    double f_0p0 = interpolator_0p0.interpolate(f, bs, comp);
    fyy = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  }
  f_m00 = interpolator_m00.interpolate(f, bs, comp) - 0.5*d_m00_m0*d_m00_p0*fyy;
  f_p00 = interpolator_p00.interpolate(f, bs, comp) - 0.5*d_p00_m0*d_p00_p0*fyy;
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr = PetscLogFlops(6); CHKERRXX(ierr);
#endif
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation( const double *f,
                                                                         double& f_0m0, double& f_000, double& f_0p0,
                                                                         const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_m00.is_set && interpolator_p00.is_set && interpolator_0m0.is_set && interpolator_0p0.is_set);
  f_000 = f[bs*node_000+comp];

  double fxx=0;
  if(d_0m0_m0*d_0m0_p0!=0 || d_0p0_m0*d_0p0_p0!=0)
  {
    double f_m00 = interpolator_m00.interpolate(f, bs, comp);
    double f_p00 = interpolator_p00.interpolate(f, bs, comp);
    fxx = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  }
  f_0m0 = interpolator_0m0.interpolate(f, bs, comp) - 0.5*d_m00_m0*d_m00_p0*fxx;
  f_0p0 = interpolator_0p0.interpolate(f, bs, comp) - 0.5*d_p00_m0*d_p00_p0*fxx;
#ifdef CASL_LOG_FLOPS
  PetscErrorCode ierr = PetscLogFlops(6); CHKERRXX(ierr);
#endif
}

double quad_neighbor_nodes_of_node_t::dx_central ( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  return  central_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
}

double quad_neighbor_nodes_of_node_t::dy_central ( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  return  central_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

void quad_neighbor_nodes_of_node_t::gradient( const double *f, double &fx, double &fy, const unsigned int& bs, const unsigned int& comp) const
{
  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0, bs, comp);
  fx = central_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  fy = central_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

double quad_neighbor_nodes_of_node_t::dx_forward_linear ( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_p00.is_set);
  return  forward_derivative(interpolator_p00.interpolate(f, bs, comp), f[bs*node_000+comp], d_p00);
}

double quad_neighbor_nodes_of_node_t::dx_backward_linear ( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_m00.is_set);
  return  backward_derivative(f[bs*node_000+comp], interpolator_m00.interpolate(f, bs, comp), d_m00);
}

double quad_neighbor_nodes_of_node_t::dy_forward_linear ( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_0p0.is_set);
  return  forward_derivative(interpolator_0p0.interpolate(f, bs, comp), f[bs*node_000+comp], d_0p0);
}

double quad_neighbor_nodes_of_node_t::dy_backward_linear ( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_0m0.is_set);
  return  backward_derivative(f[bs*node_000+comp], interpolator_0m0.interpolate(f, bs, comp), d_0m0);
}

inline double quad_neighbor_nodes_of_node_t::dx_backward_quadratic( const double *f, const double& fxx_m00,
                                                                    const unsigned int& bs, const unsigned int& comp) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  double fxx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  return (backward_derivative(f_000, f_m00, d_m00) + 0.5*d_m00*MINMOD(fxx_000,fxx_m00));
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic( const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                             const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_m00.is_set);
  double fxx_m00 = interpolator_m00.interpolate_dxx(f, neighbors, bs, comp);
  return dx_backward_quadratic(f, fxx_m00, bs, comp);
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic( const double *f, const double *fxx, const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_m00.is_set);
  double fxx_m00 = interpolator_m00.interpolate(fxx, bs, comp);
  return dx_backward_quadratic(f, fxx_m00, bs, comp);
}

inline double quad_neighbor_nodes_of_node_t::dx_forward_quadratic( const double *f, const double& fxx_p00,
                                                                   const unsigned int& bs, const unsigned int& comp) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  double fxx_000 = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  return (forward_derivative(f_p00, f_000, d_p00) - 0.5*d_p00*MINMOD(fxx_000,fxx_p00));
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic( const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                            const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_p00.is_set);
  double fxx_p00 = interpolator_p00.interpolate_dxx(f, neighbors, bs, comp);
  return dx_forward_quadratic(f, fxx_p00, bs, comp);
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic ( const double *f, const double *fxx, const unsigned int& bs, const unsigned int& comp ) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_p00.is_set);
  double fxx_p00 = interpolator_p00.interpolate(fxx, bs, comp);
  return dx_forward_quadratic(f, fxx_p00, bs, comp);
}

inline double quad_neighbor_nodes_of_node_t::dy_backward_quadratic( const double *f, const double& fyy_0m0,
                                                                    const unsigned int& bs, const unsigned int& comp) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  double fyy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  return (backward_derivative(f_000, f_0m0, d_0m0) + 0.5*d_0m0*MINMOD(fyy_000,fyy_0m0));
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic( const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                             const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_0m0.is_set);
  double fyy_0m0 = interpolator_0m0.interpolate_dyy(f, neighbors, bs, comp);
  return dy_backward_quadratic(f, fyy_0m0, bs, comp);
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic( const double *f, const double *fyy, const unsigned int& bs, const unsigned int& comp ) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_0m0.is_set);
  double fyy_0m0 = interpolator_0m0.interpolate(fyy, bs, comp);
  return dy_backward_quadratic(f, fyy_0m0, bs, comp);
}

inline double quad_neighbor_nodes_of_node_t::dy_forward_quadratic( const double *f, const double& fyy_0p0,
                                                                   const unsigned int& bs, const unsigned int& comp) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  double fyy_000 = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
  return (forward_derivative(f_0p0, f_000, d_0p0) - 0.5*d_0p0*MINMOD(fyy_000,fyy_0p0));
}

double quad_neighbor_nodes_of_node_t::dy_forward_quadratic( const double *f, const my_p4est_node_neighbors_t &neighbors,
                                                            const unsigned int& bs, const unsigned int& comp) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_0p0.is_set);
  double fyy_0p0 = interpolator_0p0.interpolate_dyy(f, neighbors, bs, comp);
  return dy_forward_quadratic(f, fyy_0p0, bs, comp);
}
double quad_neighbor_nodes_of_node_t::dy_forward_quadratic ( const double *f, const double *fyy, const unsigned int& bs, const unsigned int& comp ) const
{
  P4EST_ASSERT((comp < bs) && (bs > 0));
  P4EST_ASSERT(interpolator_0p0.is_set);
  double fyy_0p0 = interpolator_0p0.interpolate(fyy, bs, comp);
  return dy_forward_quadratic(f, fyy_0p0, bs, comp);
}

double quad_neighbor_nodes_of_node_t::dxx_central( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  double f_m00,f_000,f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00, bs, comp);
  return central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
}

double quad_neighbor_nodes_of_node_t::dyy_central( const double *f, const unsigned int& bs, const unsigned int& comp) const
{
  double f_0m0,f_000,f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0, bs, comp);
  return central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
}

void quad_neighbor_nodes_of_node_t::laplace( const double *f, double &fxx, double &fyy, const unsigned int& bs, const unsigned int& comp) const
{

  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0, bs, comp);
  fxx = central_second_derivative(f_p00, f_000, f_m00, d_p00, d_m00);
  fyy = central_second_derivative(f_0p0, f_000, f_0m0, d_0p0, d_0m0);
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
