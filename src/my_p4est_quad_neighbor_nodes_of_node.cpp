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

double quad_neighbor_nodes_of_node_t::f_p00_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_p00_p0==0) return f[node_p00_pm];
    if(d_p00_m0==0) return f[node_p00_mm];
    else          return(f[node_p00_mm]*d_p00_p0 + f[node_p00_pm]*d_p00_m0)/ (d_p00_m0+d_p00_p0);
}

double quad_neighbor_nodes_of_node_t::f_0m0_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0m0_m0==0) return f[node_0m0_mm];
    if(d_0m0_p0==0) return f[node_0m0_pm];
    else          return(f[node_0m0_pm]*d_0m0_m0 + f[node_0m0_mm]*d_0m0_p0)/ (d_0m0_m0+d_0m0_p0);
}

double quad_neighbor_nodes_of_node_t::f_0p0_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0p0_m0==0) return f[node_0p0_mm];
    if(d_0p0_p0==0) return f[node_0p0_pm];
    else          return(f[node_0p0_pm]*d_0p0_m0 + f[node_0p0_mm]*d_0p0_p0)/ (d_0p0_m0+d_0p0_p0);
}

void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation(const double *f,
                                                                       double& f_000,
                                                                       double& f_m00, double& f_p00,
                                                                       double& f_0m0, double& f_0p0
                                                                      ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    f_000 = f[node_000];

    f_m00 = f_m00_linear(f);
    f_p00 = f_p00_linear(f);
    f_0m0 = f_0m0_linear(f);
    f_0p0 = f_0p0_linear(f);

    double fyy=0; if(d_p00_m0*d_p00_p0!=0 || d_m00_m0*d_m00_p0!=0) fyy = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2/(d_0p0+d_0m0);
    double fxx=0; if(d_0m0_m0*d_0m0_p0!=0 || d_0p0_m0*d_0p0_p0!=0) fxx = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2/(d_p00+d_m00);

    f_m00 -= 0.5*d_m00_m0*d_m00_p0*fyy;
    f_p00 -= 0.5*d_p00_m0*d_p00_p0*fyy;
    f_0m0 -= 0.5*d_0m0_m0*d_0m0_p0*fxx;
    f_0p0 -= 0.5*d_0p0_m0*d_0p0_p0*fxx;

    ierr = PetscLogFlops(22); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation(const double *f, double& f_m00,
                                                                         double& f_000,
                                                                         double& f_p00) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    f_000 = f[node_000];

    double fyy=0;
    if(d_p00_m0*d_p00_p0!=0 || d_m00_m0*d_m00_p0!=0)
    {
        double f_0m0 = f_0m0_linear(f);
        double f_0p0 = f_0p0_linear(f);
        fyy = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0 )*2./(d_0p0+d_0m0);
    }
    f_m00 = f_m00_linear(f) - 0.5*d_m00_m0*d_m00_p0*fyy;
    f_p00 = f_p00_linear(f) - 0.5*d_p00_m0*d_p00_p0*fyy;

    ierr = PetscLogFlops(11); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation(const double *f, double& f_0m0,
                                                                         double& f_000,
                                                                         double& f_0p0) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    f_000 = f[node_000];

    double fxx=0;
    if(d_0m0_m0*d_0m0_p0!=0 || d_0p0_m0*d_0p0_p0!=0)
    {
        double f_m00 = f_m00_linear(f);
        double f_p00 = f_p00_linear(f);
        fxx = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2./(d_p00+d_m00);
    }
    f_0m0 = f_0m0_linear(f) - 0.5*d_0m0_m0*d_0m0_p0*fxx;
    f_0p0 = f_0p0_linear(f) - 0.5*d_0p0_m0*d_0p0_p0*fxx;

    ierr = PetscLogFlops(11); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

double quad_neighbor_nodes_of_node_t::dx_central ( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);

    double f_m00,f_000,f_p00; x_ngbd_with_quadratic_interpolation(f,f_m00,f_000,f_p00);
    double val = ((f_p00-f_000)/d_p00*d_m00+
                  (f_000-f_m00)/d_m00*d_p00)/(d_m00+d_p00);

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(7); CHKERRXX(ierr);

    return val;
}

double quad_neighbor_nodes_of_node_t::dy_central ( const double *f ) const
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

void quad_neighbor_nodes_of_node_t::gradient(const double *f, double &fx, double &fy) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);

  double f_000, f_m00, f_p00, f_0m0, f_0p0;
  ngbd_with_quadratic_interpolation(f, f_000, f_m00, f_p00, f_0m0, f_0p0);

  fx = ((f_p00-f_000)/d_p00*d_m00 + (f_000-f_m00)/d_m00*d_p00)/(d_m00+d_p00);
  fy = ((f_0p0-f_000)/d_0p0*d_0m0 + (f_000-f_0m0)/d_0m0*d_0p0)/(d_0m0+d_0p0);

  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
}

double quad_neighbor_nodes_of_node_t::dx_forward_linear ( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f_p00_linear(f)-f[node_000])/d_p00;
}

double quad_neighbor_nodes_of_node_t::dx_backward_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f[node_000]-f_m00_linear(f))/d_m00;
}

double quad_neighbor_nodes_of_node_t::dy_forward_linear ( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f_0p0_linear(f)-f[node_000])/d_0p0;
}

double quad_neighbor_nodes_of_node_t::dy_backward_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f[node_000]-f_0m0_linear(f))/d_0m0;
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);

  double fxx_000 = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2.0/(d_m00+d_p00);
  double fxx_m00 = dxx_central_on_m00(f, neighbors);

  return (f_000-f_m00)/d_m00 + 0.5*d_m00*MINMOD(fxx_000,fxx_m00);
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);

  double fxx_000 = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2.0/(d_m00+d_p00);
  double fxx_p00 = dxx_central_on_p00(f, neighbors);

  return (f_p00-f_000)/d_p00 - 0.5*d_p00*MINMOD(fxx_000,fxx_p00);
}

double quad_neighbor_nodes_of_node_t::dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);

  double fyy_000 = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2.0/(d_0m0+d_0p0);
  double fyy_0m0 = dyy_central_on_0m0(f, neighbors);

  return (f_000-f_0m0)/d_0m0 + 0.5*d_0m0*MINMOD(fyy_000,fyy_0m0);
}

double quad_neighbor_nodes_of_node_t::dy_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_0m0, f_0p0;
  y_ngbd_with_quadratic_interpolation(f, f_0m0, f_000, f_0p0);

  double fyy_000 = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2.0/(d_0m0+d_0p0);
  double fyy_0p0 = dyy_central_on_0p0(f, neighbors);

  return (f_0p0-f_000)/d_0p0 - 0.5*d_0p0*MINMOD(fyy_000,fyy_0p0);
}

double quad_neighbor_nodes_of_node_t::dx_backward_quadratic(const double *f, const double *fxx) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);

  double fxx_000 = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2.0/(d_m00+d_p00);
  double fxx_m00 = f_m00_linear(fxx);

  return (f_000-f_m00)/d_m00 + 0.5*d_m00*MINMOD(fxx_000,fxx_m00);
}

double quad_neighbor_nodes_of_node_t::dx_forward_quadratic(const double *f, const double *fxx) const
{
  double f_000, f_m00, f_p00;
  x_ngbd_with_quadratic_interpolation(f, f_m00, f_000, f_p00);

  double fxx_000 = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2.0/(d_m00+d_p00);
  double fxx_p00 = f_p00_linear(fxx);

  return (f_p00-f_000)/d_p00 - 0.5*d_p00*MINMOD(fxx_000,fxx_p00);
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
