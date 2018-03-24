#include "my_p8est_quad_neighbor_nodes_of_node.h"
#include <src/my_p8est_node_neighbors.h>
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
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quad_interp;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dy_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dz_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dx_gradient;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dzz_central;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central_m00;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dxx_central_p00;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central_0m0;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dyy_central_0p0;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dzz_central_00m;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_dzz_central_00p;
extern PetscLogEvent log_quad_neighbor_nodes_of_node_t_laplace;
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double quad_neighbor_nodes_of_node_t::f_m00_linear(const double* f) const{
    PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
    return (f[node_m00_mm]*d_m00_p0*d_m00_0p +
            f[node_m00_mp]*d_m00_p0*d_m00_0m +
            f[node_m00_pm]*d_m00_m0*d_m00_0p +
            f[node_m00_pp]*d_m00_m0*d_m00_0m )/(d_m00_m0 + d_m00_p0)/(d_m00_0m + d_m00_0p);
}
double quad_neighbor_nodes_of_node_t::f_p00_linear(const double* f) const{
    PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
    return (f[node_p00_mm]*d_p00_p0*d_p00_0p +
            f[node_p00_mp]*d_p00_p0*d_p00_0m +
            f[node_p00_pm]*d_p00_m0*d_p00_0p +
            f[node_p00_pp]*d_p00_m0*d_p00_0m )/(d_p00_m0 + d_p00_p0)/(d_p00_0m + d_p00_0p);
}
double quad_neighbor_nodes_of_node_t::f_0m0_linear(const double* f) const{
    PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
    return (f[node_0m0_mm]*d_0m0_p0*d_0m0_0p +
            f[node_0m0_mp]*d_0m0_p0*d_0m0_0m +
            f[node_0m0_pm]*d_0m0_m0*d_0m0_0p +
            f[node_0m0_pp]*d_0m0_m0*d_0m0_0m )/(d_0m0_m0 + d_0m0_p0)/(d_0m0_0m + d_0m0_0p);
}
double quad_neighbor_nodes_of_node_t::f_0p0_linear(const double* f) const{
    PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
    return (f[node_0p0_mm]*d_0p0_p0*d_0p0_0p +
            f[node_0p0_mp]*d_0p0_p0*d_0p0_0m +
            f[node_0p0_pm]*d_0p0_m0*d_0p0_0p +
            f[node_0p0_pp]*d_0p0_m0*d_0p0_0m )/(d_0p0_m0 + d_0p0_p0)/(d_0p0_0m + d_0p0_0p);
}
double quad_neighbor_nodes_of_node_t::f_00m_linear(const double* f) const{
    PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
    return (f[node_00m_mm]*d_00m_p0*d_00m_0p +
            f[node_00m_mp]*d_00m_p0*d_00m_0m +
            f[node_00m_pm]*d_00m_m0*d_00m_0p +
            f[node_00m_pp]*d_00m_m0*d_00m_0m )/(d_00m_m0 + d_00m_p0)/(d_00m_0m + d_00m_0p);
}
double quad_neighbor_nodes_of_node_t::f_00p_linear(const double* f) const{
    PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
    return (f[node_00p_mm]*d_00p_p0*d_00p_0p +
            f[node_00p_mp]*d_00p_p0*d_00p_0m +
            f[node_00p_pm]*d_00p_m0*d_00p_0p +
            f[node_00p_pp]*d_00p_m0*d_00p_0m )/(d_00p_m0 + d_00p_p0)/(d_00p_0m + d_00p_0p);
}

double quad_neighbor_nodes_of_node_t::dx_central( const double* f ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(35); CHKERRXX(ierr);

    // Dfx =  fx  + afyy + bfzz
    double Dfx = ( (f_p00_linear(f)-f[node_000])/d_p00*d_m00
                  +(f[node_000]-f_m00_linear(f))/d_m00*d_p00 ) / (d_p00+d_m00);

    double a =-d_m00_m0*d_m00_p0/d_m00*d_p00/2.
              +d_p00_m0*d_p00_p0/d_p00*d_m00/2.; a /= (d_m00+d_p00);
    double b =-d_m00_0m*d_m00_0p/d_m00*d_p00/2.
              +d_p00_0m*d_p00_0p/d_p00*d_m00/2.; b /= (d_m00+d_p00);

    if( a!=0 || b!=0 )
    {
        double fxx,fyy,fzz; laplace(f,fxx,fyy,fzz);
        Dfx -= a*fyy + b*fzz;
    }
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    return Dfx;
}

double quad_neighbor_nodes_of_node_t::dy_central( const double* f ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(35); CHKERRXX(ierr);

    // Dfy = cfxx +  fy  + dfzz
    double Dfy = ( (f_0p0_linear(f)-f[node_000])/d_0p0*d_0m0
                  +(f[node_000]-f_0m0_linear(f))/d_0m0*d_0p0 ) / (d_0p0+d_0m0);

    double c =-d_0m0_m0*d_0m0_p0/d_0m0*d_0p0/2.
              +d_0p0_m0*d_0p0_p0/d_0p0*d_0m0/2.; c /= (d_0m0+d_0p0);
    double d =-d_0m0_0m*d_0m0_0p/d_0m0*d_0p0/2.
              +d_0p0_0m*d_0p0_0p/d_0p0*d_0m0/2.; d /= (d_0m0+d_0p0);

    if( c!=0 || d!=0 )
    {
        double fxx,fyy,fzz; laplace(f,fxx,fyy,fzz);
        Dfy -= c*fxx + d*fzz;
    }
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);
    return Dfy;
}

double quad_neighbor_nodes_of_node_t::dz_central( const double* f ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dz_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(35); CHKERRXX(ierr);

    // Dfz = efxx + ffyy +  fz
    double Dfz = ( (f_00p_linear(f)-f[node_000])/d_00p*d_00m
                  +(f[node_000]-f_00m_linear(f))/d_00m*d_00p ) / (d_00p+d_00m);

    double e =-d_00m_m0*d_00m_p0/d_00m*d_00p/2.
              +d_00p_m0*d_00p_p0/d_00p*d_00m/2.; e /= (d_00m+d_00p);
    double F =-d_00m_0m*d_00m_0p/d_00m*d_00p/2.
              +d_00p_0m*d_00p_0p/d_00p*d_00m/2.; F /= (d_00m+d_00p);

    if(e!=0 || F!=0)
    {
        double fxx,fyy,fzz; laplace(f,fxx,fyy,fzz);
        Dfz -= e*fxx + F*fyy;
    }
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dz_central, 0, 0, 0, 0); CHKERRXX(ierr);
    return Dfz;
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

double quad_neighbor_nodes_of_node_t::dz_forward_linear ( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f_00p_linear(f)-f[node_000])/d_00p;
}

double quad_neighbor_nodes_of_node_t::dz_backward_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f[node_000]-f_00m_linear(f))/d_00m;
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

double quad_neighbor_nodes_of_node_t::dz_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);

  double fzz_000 = ((f_00p-f_000)/d_00p + (f_00m-f_000)/d_00m)*2.0/(d_00m+d_00p);
  double fzz_00m = dzz_central_on_00m(f, neighbors);

  return (f_000-f_00m)/d_00m + 0.5*d_00m*MINMOD(fzz_000,fzz_00m);
}

double quad_neighbor_nodes_of_node_t::dz_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);

  double fzz_000 = ((f_00p-f_000)/d_00p + (f_00m-f_000)/d_00m)*2.0/(d_00m+d_00p);
  double fzz_00p = dzz_central_on_00p(f, neighbors);

  return (f_00p-f_000)/d_00p - 0.5*d_00p*MINMOD(fzz_000,fzz_00p);
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

double quad_neighbor_nodes_of_node_t::dz_backward_quadratic(const double *f, const double *fzz) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);

  double fzz_000 = ((f_00p-f_000)/d_00p + (f_00m-f_000)/d_00m)*2.0/(d_00m+d_00p);
  double fzz_00m = f_00m_linear(fzz);

  return (f_000-f_00m)/d_00m + 0.5*d_00m*MINMOD(fzz_000,fzz_00m);
}

double quad_neighbor_nodes_of_node_t::dz_forward_quadratic(const double *f, const double *fzz) const
{
  double f_000, f_00m, f_00p;
  z_ngbd_with_quadratic_interpolation(f, f_00m, f_000, f_00p);

  double fzz_000 = ((f_00p-f_000)/d_00p + (f_00m-f_000)/d_00m)*2.0/(d_00m+d_00p);
  double fzz_00p = f_00p_linear(fzz);

  return (f_00p-f_000)/d_00p - 0.5*d_00p*MINMOD(fzz_000,fzz_00p);
}

double quad_neighbor_nodes_of_node_t::dxx_central( const double* f ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(7); CHKERRXX(ierr);

    double fm,fc,fp;
    x_ngbd_with_quadratic_interpolation(f,fm,fc,fp);

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return ((fp-fc)/d_p00+(fm-fc)/d_m00)*2./(d_p00+d_m00);
}

double quad_neighbor_nodes_of_node_t::dyy_central( const double* f ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(7); CHKERRXX(ierr);

    double fm,fc,fp;
    y_ngbd_with_quadratic_interpolation(f,fm,fc,fp);

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return ((fp-fc)/d_0p0+(fm-fc)/d_0m0)*2./(d_0p0+d_0m0);
}

double quad_neighbor_nodes_of_node_t::dzz_central( const double* f ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dzz_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(7); CHKERRXX(ierr);

    double fm,fc,fp;
    z_ngbd_with_quadratic_interpolation(f,fm,fc,fp);

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dzz_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return ((fp-fc)/d_00p+(fm-fc)/d_00m)*2./(d_00p+d_00m);
}

double quad_neighbor_nodes_of_node_t::dxx_central_on_m00(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central_m00, 0, 0, 0, 0); CHKERRXX(ierr);

  double fxx_m00_mm = 0, fxx_m00_pm = 0, fxx_m00_mp = 0, fxx_m00_pp = 0;
  double w_m00_mm = d_m00_p0*d_m00_0p;
  double w_m00_mp = d_m00_p0*d_m00_0m;
  double w_m00_pm = d_m00_m0*d_m00_0p;
  double w_m00_pp = d_m00_m0*d_m00_0m;

  if (w_m00_mm != 0) { fxx_m00_mm = neighbors[node_m00_mm].dxx_central(f); }
  if (w_m00_mp != 0) { fxx_m00_mp = neighbors[node_m00_mp].dxx_central(f); }
  if (w_m00_pm != 0) { fxx_m00_pm = neighbors[node_m00_pm].dxx_central(f); }
  if (w_m00_pp != 0) { fxx_m00_pp = neighbors[node_m00_pp].dxx_central(f); }

  ierr = PetscLogFlops(15); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central_m00, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fxx_m00_mm*w_m00_mm + fxx_m00_mp*w_m00_mp +
          fxx_m00_pm*w_m00_pm + fxx_m00_pp*w_m00_pp )/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
}

double quad_neighbor_nodes_of_node_t::dxx_central_on_p00(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central_p00, 0, 0, 0, 0); CHKERRXX(ierr);

  double fxx_p00_mm = 0, fxx_p00_pm = 0, fxx_p00_mp = 0, fxx_p00_pp = 0;
  double w_p00_mm = d_p00_p0*d_p00_0p;
  double w_p00_mp = d_p00_p0*d_p00_0m;
  double w_p00_pm = d_p00_m0*d_p00_0p;
  double w_p00_pp = d_p00_m0*d_p00_0m;

  if (w_p00_mm != 0) { fxx_p00_mm = neighbors[node_p00_mm].dxx_central(f); }
  if (w_p00_mp != 0) { fxx_p00_mp = neighbors[node_p00_mp].dxx_central(f); }
  if (w_p00_pm != 0) { fxx_p00_pm = neighbors[node_p00_pm].dxx_central(f); }
  if (w_p00_pp != 0) { fxx_p00_pp = neighbors[node_p00_pp].dxx_central(f); }

  ierr = PetscLogFlops(15); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central_p00, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fxx_p00_mm*w_p00_mm + fxx_p00_mp*w_p00_mp +
          fxx_p00_pm*w_p00_pm + fxx_p00_pp*w_p00_pp )/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
}

double quad_neighbor_nodes_of_node_t::dyy_central_on_0m0(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central_0m0, 0, 0, 0, 0); CHKERRXX(ierr);

  double fyy_0m0_mm = 0, fyy_0m0_pm = 0, fyy_0m0_mp = 0, fyy_0m0_pp = 0;
  double w_0m0_mm = d_0m0_p0*d_0m0_0p;
  double w_0m0_mp = d_0m0_p0*d_0m0_0m;
  double w_0m0_pm = d_0m0_m0*d_0m0_0p;
  double w_0m0_pp = d_0m0_m0*d_0m0_0m;

  if (w_0m0_mm != 0) { fyy_0m0_mm = neighbors[node_0m0_mm].dyy_central(f); }
  if (w_0m0_mp != 0) { fyy_0m0_mp = neighbors[node_0m0_mp].dyy_central(f); }
  if (w_0m0_pm != 0) { fyy_0m0_pm = neighbors[node_0m0_pm].dyy_central(f); }
  if (w_0m0_pp != 0) { fyy_0m0_pp = neighbors[node_0m0_pp].dyy_central(f); }

  ierr = PetscLogFlops(15); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central_0m0, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fyy_0m0_mm*w_0m0_mm + fyy_0m0_mp*w_0m0_mp +
          fyy_0m0_pm*w_0m0_pm + fyy_0m0_pp*w_0m0_pp )/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
}

double quad_neighbor_nodes_of_node_t::dyy_central_on_0p0(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central_0p0, 0, 0, 0, 0); CHKERRXX(ierr);

  double fyy_0p0_mm = 0, fyy_0p0_pm = 0, fyy_0p0_mp = 0, fyy_0p0_pp = 0;
  double w_0p0_mm = d_0p0_p0*d_0p0_0p;
  double w_0p0_mp = d_0p0_p0*d_0p0_0m;
  double w_0p0_pm = d_0p0_m0*d_0p0_0p;
  double w_0p0_pp = d_0p0_m0*d_0p0_0m;

  if (w_0p0_mm != 0) { fyy_0p0_mm = neighbors[node_0p0_mm].dyy_central(f); }
  if (w_0p0_mp != 0) { fyy_0p0_mp = neighbors[node_0p0_mp].dyy_central(f); }
  if (w_0p0_pm != 0) { fyy_0p0_pm = neighbors[node_0p0_pm].dyy_central(f); }
  if (w_0p0_pp != 0) { fyy_0p0_pp = neighbors[node_0p0_pp].dyy_central(f); }

  ierr = PetscLogFlops(15); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central_0p0, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fyy_0p0_mm*w_0p0_mm + fyy_0p0_mp*w_0p0_mp +
          fyy_0p0_pm*w_0p0_pm + fyy_0p0_pp*w_0p0_pp )/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
}

double quad_neighbor_nodes_of_node_t::dzz_central_on_00m(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dzz_central_00m, 0, 0, 0, 0); CHKERRXX(ierr);

  double fzz_00m_mm = 0, fzz_00m_pm = 0, fzz_00m_mp = 0, fzz_00m_pp = 0;
  double w_00m_mm = d_00m_p0*d_00m_0p;
  double w_00m_mp = d_00m_p0*d_00m_0m;
  double w_00m_pm = d_00m_m0*d_00m_0p;
  double w_00m_pp = d_00m_m0*d_00m_0m;

  if (w_00m_mm != 0) { fzz_00m_mm = neighbors[node_00m_mm].dzz_central(f); }
  if (w_00m_mp != 0) { fzz_00m_mp = neighbors[node_00m_mp].dzz_central(f); }
  if (w_00m_pm != 0) { fzz_00m_pm = neighbors[node_00m_pm].dzz_central(f); }
  if (w_00m_pp != 0) { fzz_00m_pp = neighbors[node_00m_pp].dzz_central(f); }

  ierr = PetscLogFlops(15); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dzz_central_00m, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fzz_00m_mm*w_00m_mm + fzz_00m_mp*w_00m_mp +
          fzz_00m_pm*w_00m_pm + fzz_00m_pp*w_00m_pp )/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
}

double quad_neighbor_nodes_of_node_t::dzz_central_on_00p(const double *f, const my_p4est_node_neighbors_t &neighbors) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dzz_central_00p, 0, 0, 0, 0); CHKERRXX(ierr);

  double fzz_00p_mm = 0, fzz_00p_pm = 0, fzz_00p_mp = 0, fzz_00p_pp = 0;
  double w_00p_mm = d_00p_p0*d_00p_0p;
  double w_00p_mp = d_00p_p0*d_00p_0m;
  double w_00p_pm = d_00p_m0*d_00p_0p;
  double w_00p_pp = d_00p_m0*d_00p_0m;

  if (w_00p_mm != 0) { fzz_00p_mm = neighbors[node_00p_mm].dzz_central(f); }
  if (w_00p_mp != 0) { fzz_00p_mp = neighbors[node_00p_mp].dzz_central(f); }
  if (w_00p_pm != 0) { fzz_00p_pm = neighbors[node_00p_pm].dzz_central(f); }
  if (w_00p_pp != 0) { fzz_00p_pp = neighbors[node_00p_pp].dzz_central(f); }

  ierr = PetscLogFlops(15); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dzz_central_00p, 0, 0, 0, 0); CHKERRXX(ierr);

  return (fzz_00p_mm*w_00p_mm + fzz_00p_mp*w_00p_mp +
          fzz_00p_pm*w_00p_pm + fzz_00p_pp*w_00p_pp )/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
}

void quad_neighbor_nodes_of_node_t::gradient( const double* f, double& fx, double& fy, double& fz ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(78); CHKERRXX(ierr);

    double fxx,fyy,fzz; laplace(f,fxx,fyy,fzz);

    // Dfx =  fx  + afyy + bfzz
    // Dfy = cfxx +  fy  + dfzz
    // Dfz = efxx + ffyy +  fz
    double Dfx = ( (f_p00_linear(f)-f[node_000])/d_p00*d_m00
                  +(f[node_000]-f_m00_linear(f))/d_m00*d_p00)/(d_p00+d_m00);
    double Dfy = ( (f_0p0_linear(f)-f[node_000])/d_0p0*d_0m0
                  +(f[node_000]-f_0m0_linear(f))/d_0m0*d_0p0)/(d_0p0+d_0m0);
    double Dfz = ( (f_00p_linear(f)-f[node_000])/d_00p*d_00m
                  +(f[node_000]-f_00m_linear(f))/d_00m*d_00p)/(d_00p+d_00m);

    double a =-d_m00_m0*d_m00_p0/d_m00*d_p00/2.
              +d_p00_m0*d_p00_p0/d_p00*d_m00/2.; a /= (d_m00+d_p00);
    double b =-d_m00_0m*d_m00_0p/d_m00*d_p00/2.
              +d_p00_0m*d_p00_0p/d_p00*d_m00/2.; b /= (d_m00+d_p00);
    double c =-d_0m0_m0*d_0m0_p0/d_0m0*d_0p0/2.
              +d_0p0_m0*d_0p0_p0/d_0p0*d_0m0/2.; c /= (d_0m0+d_0p0);
    double d =-d_0m0_0m*d_0m0_0p/d_0m0*d_0p0/2.
              +d_0p0_0m*d_0p0_0p/d_0p0*d_0m0/2.; d /= (d_0m0+d_0p0);
    double e =-d_00m_m0*d_00m_p0/d_00m*d_00p/2.
              +d_00p_m0*d_00p_p0/d_00p*d_00m/2.; e /= (d_00m+d_00p);
    double F =-d_00m_0m*d_00m_0p/d_00m*d_00p/2.
              +d_00p_0m*d_00p_0p/d_00p*d_00m/2.; F /= (d_00m+d_00p);

    fx = Dfx - a*fyy - b*fzz;
    fy = Dfy - c*fxx - d*fzz;
    fz = Dfz - e*fxx - F*fyy;

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_gradient, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::laplace( const double* f, double& fxx, double& fyy, double& fzz ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(111); CHKERRXX(ierr);

    double f_000 = f[node_000];
    double f_p00 = f_p00_linear(f); double f_m00 = f_m00_linear(f);
    double f_0p0 = f_0p0_linear(f); double f_0m0 = f_0m0_linear(f);
    double f_00p = f_00p_linear(f); double f_00m = f_00m_linear(f);

    double Dfxx = ((f_p00-f_000)/d_p00 + (f_m00-f_000)/d_m00)*2./(d_p00+d_m00);
    double Dfyy = ((f_0p0-f_000)/d_0p0 + (f_0m0-f_000)/d_0m0)*2./(d_0p0+d_0m0);
    double Dfzz = ((f_00p-f_000)/d_00p + (f_00m-f_000)/d_00m)*2./(d_00p+d_00m);

    // Dfxx =   fxx + a*fyy + b*fzz
    // Dfyy = c*fxx +   fyy + d*fzz
    // Dfzz = e*fxx + f*fyy +   fzz
    double a =(d_m00_m0*d_m00_p0/d_m00+d_p00_m0*d_p00_p0/d_p00)/(d_p00+d_m00) ;
    double b =(d_m00_0m*d_m00_0p/d_m00+d_p00_0m*d_p00_0p/d_p00)/(d_p00+d_m00) ;
    double c =(d_0m0_m0*d_0m0_p0/d_0m0+d_0p0_m0*d_0p0_p0/d_0p0)/(d_0p0+d_0m0) ;
    double d =(d_0m0_0m*d_0m0_0p/d_0m0+d_0p0_0m*d_0p0_0p/d_0p0)/(d_0p0+d_0m0) ;
    double e =(d_00m_m0*d_00m_p0/d_00m+d_00p_m0*d_00p_p0/d_00p)/(d_00p+d_00m) ;
    double F =(d_00m_0m*d_00m_0p/d_00m+d_00p_0m*d_00p_0p/d_00p)/(d_00p+d_00m) ;

    // fxx,fyy,fzz are linear sums of Dfxx, Dfyy, Dfzz
    double det = a*c+b*e+d*F-a*d*e-b*c*F-1;

    fxx = (Dfxx*(d*F-1) + Dfyy*(a-b*F) + Dfzz*(b-a*d))/det;
    fyy = (Dfxx*(c-d*e) + Dfyy*(b*e-1) + Dfzz*(d-b*c))/det;
    fzz = (Dfxx*(e-c*F) + Dfyy*(F-a*e) + Dfzz*(a*c-1))/det;

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_laplace, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation( const double* f, double& f_m00,
                                                              double& f_000,
                                                              double& f_p00 ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    if( (d_m00_m0==0||d_m00_p0==0) && (d_m00_0m==0||d_m00_0p==0) &&
        (d_p00_m0==0||d_p00_p0==0) && (d_p00_0m==0||d_p00_0p==0))
    {
        f_000 = f[node_000];
        f_m00 = f_m00_linear(f);
        f_p00 = f_p00_linear(f);
    }
    else {double temp; ngbd_with_quadratic_interpolation(f,f_000,f_m00,f_p00,temp,temp,temp,temp);}

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation( const double* f, double& f_0m0,
                                                              double& f_000,
                                                              double& f_0p0 ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    if( (d_0m0_m0==0||d_0m0_p0==0) && (d_0m0_0m==0||d_0m0_0p==0) &&
        (d_0p0_m0==0||d_0p0_p0==0) && (d_0p0_0m==0||d_0p0_0p==0))
    {
        f_000 = f[node_000];
        f_0m0 = f_0m0_linear(f);
        f_0p0 = f_0p0_linear(f);
    }
    else {double temp; ngbd_with_quadratic_interpolation(f,f_000,temp,temp,f_0m0,f_0p0,temp,temp);}

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::z_ngbd_with_quadratic_interpolation( const double* f, double& f_00m,
                                                              double& f_000,
                                                              double& f_00p ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    if( (d_00m_m0==0||d_00m_p0==0) && (d_00m_0m==0||d_00m_0p==0) &&
        (d_00p_m0==0||d_00p_p0==0) && (d_00p_0m==0||d_00p_0p==0))
    {
        f_000 = f[node_000];
        f_00m = f_00m_linear(f);
        f_00p = f_00p_linear(f);
    }
    else {double temp; ngbd_with_quadratic_interpolation(f,f_000,temp,temp,temp,temp,f_00m,f_00p);}

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_z_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}


void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation( const double* f, double& f_000,
                                                            double& f_m00, double& f_p00,
                                                            double& f_0m0, double& f_0p0,
                                                            double& f_00m, double& f_00p ) const
{
    PetscErrorCode ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(174); CHKERRXX(ierr);

    double f_000_ = f[node_000];
    double f_m00_ = f_m00_linear(f); double f_p00_ = f_p00_linear(f);
    double f_0m0_ = f_0m0_linear(f); double f_0p0_ = f_0p0_linear(f);
    double f_00m_ = f_00m_linear(f); double f_00p_ = f_00p_linear(f);

    //
    double Dfxx =((f_p00_-f_000_)/d_p00 + (f_m00_-f_000_)/d_m00)*2./(d_m00+d_p00);
    double Dfyy =((f_0p0_-f_000_)/d_0p0 + (f_0m0_-f_000_)/d_0m0)*2./(d_0m0+d_0p0);
    double Dfzz =((f_00p_-f_000_)/d_00p + (f_00m_-f_000_)/d_00m)*2./(d_00m+d_00p);

    // Dfxx =   fxx + a*fyy + b*fzz
    // Dfyy = c*fxx +   fyy + d*fzz
    // Dfzz = e*fxx + f*fyy +   fzz
    double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
    double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;
    double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
    double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;
    double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
    double F = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

    // fxx,fyy,fzz are linear sums of Dfxx, Dfyy, Dfzz
    double det = a*c+b*e+d*F-a*d*e-b*c*F-1;

    double fxx = (Dfxx*(d*F-1) + Dfyy*(a-b*F) + Dfzz*(b-a*d))/det;
    double fyy = (Dfxx*(c-d*e) + Dfyy*(b*e-1) + Dfzz*(d-b*c))/det;
    double fzz = (Dfxx*(e-c*F) + Dfyy*(F-a*e) + Dfzz*(a*c-1))/det;

    // third order interpolation
    f_000 = f_000_;
    f_m00 = f_m00_ - 0.5*d_m00_m0*d_m00_p0*fyy - 0.5*d_m00_0m*d_m00_0p*fzz;
    f_p00 = f_p00_ - 0.5*d_p00_m0*d_p00_p0*fyy - 0.5*d_p00_0m*d_p00_0p*fzz;
    f_0m0 = f_0m0_ - 0.5*d_0m0_m0*d_0m0_p0*fxx - 0.5*d_0m0_0m*d_0m0_0p*fzz;
    f_0p0 = f_0p0_ - 0.5*d_0p0_m0*d_0p0_p0*fxx - 0.5*d_0p0_0m*d_0p0_0p*fzz;
    f_00m = f_00m_ - 0.5*d_00m_m0*d_00m_p0*fxx - 0.5*d_00m_0m*d_00m_0p*fyy;
    f_00p = f_00p_ - 0.5*d_00p_m0*d_00p_p0*fxx - 0.5*d_00p_0m*d_00p_0p*fyy;

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

p4est_locidx_t quad_neighbor_nodes_of_node_t::neighbor_m00() const{
  if      (d_m00_m0 == 0 && d_m00_0m == 0) return node_m00_mm;
  else if (d_m00_p0 == 0 && d_m00_0m == 0) return node_m00_pm;
  else if (d_m00_m0 == 0 && d_m00_0p == 0) return node_m00_mp;
  else if (d_m00_p0 == 0 && d_m00_0p == 0) return node_m00_pp;
  else return -1;
//  else throw std::invalid_argument("No neighbor in m00 direction \n");
}
p4est_locidx_t quad_neighbor_nodes_of_node_t::neighbor_p00() const{
  if      (d_m00_m0 == 0 && d_m00_0m == 0) return node_m00_mm;
  else if (d_m00_p0 == 0 && d_m00_0m == 0) return node_m00_pm;
  else if (d_m00_m0 == 0 && d_m00_0p == 0) return node_m00_mp;
  else if (d_m00_p0 == 0 && d_m00_0p == 0) return node_m00_pp;
  else return -1;
//  else throw std::invalid_argument("No neighbor in m00 direction \n");
}
p4est_locidx_t quad_neighbor_nodes_of_node_t::neighbor_0m0() const{
  if      (d_m00_m0 == 0 && d_m00_0m == 0) return node_m00_mm;
  else if (d_m00_p0 == 0 && d_m00_0m == 0) return node_m00_pm;
  else if (d_m00_m0 == 0 && d_m00_0p == 0) return node_m00_mp;
  else if (d_m00_p0 == 0 && d_m00_0p == 0) return node_m00_pp;
  else return -1;
//  else throw std::invalid_argument("No neighbor in m00 direction \n");
}
p4est_locidx_t quad_neighbor_nodes_of_node_t::neighbor_0p0() const{
  if      (d_m00_m0 == 0 && d_m00_0m == 0) return node_m00_mm;
  else if (d_m00_p0 == 0 && d_m00_0m == 0) return node_m00_pm;
  else if (d_m00_m0 == 0 && d_m00_0p == 0) return node_m00_mp;
  else if (d_m00_p0 == 0 && d_m00_0p == 0) return node_m00_pp;
  else return -1;
//  else throw std::invalid_argument("No neighbor in m00 direction \n");
}
p4est_locidx_t quad_neighbor_nodes_of_node_t::neighbor_00m() const{
  if      (d_m00_m0 == 0 && d_m00_0m == 0) return node_m00_mm;
  else if (d_m00_p0 == 0 && d_m00_0m == 0) return node_m00_pm;
  else if (d_m00_m0 == 0 && d_m00_0p == 0) return node_m00_mp;
  else if (d_m00_p0 == 0 && d_m00_0p == 0) return node_m00_pp;
  else return -1;
//  else throw std::invalid_argument("No neighbor in m00 direction \n");
}
p4est_locidx_t quad_neighbor_nodes_of_node_t::neighbor_00p() const{
  if      (d_m00_m0 == 0 && d_m00_0m == 0) return node_m00_mm;
  else if (d_m00_p0 == 0 && d_m00_0m == 0) return node_m00_pm;
  else if (d_m00_m0 == 0 && d_m00_0p == 0) return node_m00_mp;
  else if (d_m00_p0 == 0 && d_m00_0p == 0) return node_m00_pp;
  else return -1;
//  else throw std::invalid_argument("No neighbor in m00 direction \n");
}

