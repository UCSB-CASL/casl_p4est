#include "my_p4est_quad_neighbor_nodes_of_node.h"
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
#endif
#ifndef CASL_LOG_FLOPS
#undef  PetscLogFlops
#define PetscLogFlops(n) 0
#endif

double quad_neighbor_nodes_of_node_t::f_m0_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_m0_p==0) return f[node_m0_p];
    if(d_m0_m==0) return f[node_m0_m];
    else          return(f[node_m0_m]*d_m0_p + f[node_m0_p]*d_m0_m)/ (d_m0_m+d_m0_p);
}

double quad_neighbor_nodes_of_node_t::f_p0_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_p0_p==0) return f[node_p0_p];
    if(d_p0_m==0) return f[node_p0_m];
    else          return(f[node_p0_m]*d_p0_p + f[node_p0_p]*d_p0_m)/ (d_p0_m+d_p0_p);
}

double quad_neighbor_nodes_of_node_t::f_0m_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0m_m==0) return f[node_0m_m];
    if(d_0m_p==0) return f[node_0m_p];
    else          return(f[node_0m_p]*d_0m_m + f[node_0m_m]*d_0m_p)/ (d_0m_m+d_0m_p);
}

double quad_neighbor_nodes_of_node_t::f_0p_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0p_m==0) return f[node_0p_m];
    if(d_0p_p==0) return f[node_0p_p];
    else          return(f[node_0p_p]*d_0p_m + f[node_0p_m]*d_0p_p)/ (d_0p_m+d_0p_p);
}

void quad_neighbor_nodes_of_node_t::ngbd_with_quadratic_interpolation( const double *f, double& f_00,
                                                                       double& f_m0,
                                                                       double& f_p0,
                                                                       double& f_0m,
                                                                       double& f_0p) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    f_00 = f[node_00];

    f_m0 = f_m0_linear(f);
    f_p0 = f_p0_linear(f);
    f_0m = f_0m_linear(f);
    f_0p = f_0p_linear(f);

    double fyy=0; if(d_p0_m*d_p0_p!=0 || d_m0_m*d_m0_p!=0) fyy = ((f_0p-f_00)/d_0p + (f_0m-f_00)/d_0m)*2/(d_0p+d_0m);
    double fxx=0; if(d_0m_m*d_0m_p!=0 || d_0p_m*d_0p_p!=0) fxx = ((f_p0-f_00)/d_p0 + (f_m0-f_00)/d_m0)*2/(d_p0+d_m0);

    f_m0 -= 0.5*d_m0_m*d_m0_p*fyy;
    f_p0 -= 0.5*d_p0_m*d_p0_p*fyy;
    f_0m -= 0.5*d_0m_m*d_0m_p*fxx;
    f_0p -= 0.5*d_0p_m*d_0p_p*fxx;

    ierr = PetscLogFlops(22); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::x_ngbd_with_quadratic_interpolation( const double *f, double& f_m0,
                                                                         double& f_00,
                                                                         double& f_p0) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    f_00 = f[node_00];

    double fyy=0;
    if(d_p0_m*d_p0_p!=0 || d_m0_m*d_m0_p!=0)
    {
        double f_0m = f_0m_linear(f);
        double f_0p = f_0p_linear(f);
        fyy = ((f_0p-f_00)/d_0p + (f_0m-f_00)/d_0m )*2./(d_0p+d_0m);
    }
    f_m0 = f_m0_linear(f) - 0.5*d_m0_m*d_m0_p*fyy;
    f_p0 = f_p0_linear(f) - 0.5*d_p0_m*d_p0_p*fyy;

    ierr = PetscLogFlops(11); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_x_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

void quad_neighbor_nodes_of_node_t::y_ngbd_with_quadratic_interpolation( const double *f, double& f_0m,
                                                                         double& f_00,
                                                                         double& f_0p) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);

    f_00 = f[node_00];

    double fxx=0;
    if(d_0m_m*d_0m_p!=0 || d_0p_m*d_0p_p!=0)
    {
        double f_m0 = f_m0_linear(f);
        double f_p0 = f_p0_linear(f);
        fxx = ((f_p0-f_00)/d_p0 + (f_m0-f_00)/d_m0)*2./(d_p0+d_m0);
    }
    f_0m = f_0m_linear(f) - 0.5*d_0m_m*d_0m_p*fxx;
    f_0p = f_0p_linear(f) - 0.5*d_0p_m*d_0p_p*fxx;

    ierr = PetscLogFlops(11); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_y_ngbd_with_quad_interp, 0, 0, 0, 0); CHKERRXX(ierr);
}

double quad_neighbor_nodes_of_node_t::dx_central ( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);

    double f_m0,f_00,f_p0; x_ngbd_with_quadratic_interpolation(f,f_m0,f_00,f_p0);
    double val = ((f_p0-f_00)/d_p0*d_m0+
                  (f_00-f_m0)/d_m0*d_p0)/(d_m0+d_p0);

    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogFlops(7); CHKERRXX(ierr);

    return val;
}

double quad_neighbor_nodes_of_node_t::dy_central ( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    double f_0m,f_00,f_0p; y_ngbd_with_quadratic_interpolation(f,f_0m,f_00,f_0p);
    double val = ((f_0p-f_00)/d_0p*d_0m+
                  (f_00-f_0m)/d_0m*d_0p)/(d_0m+d_0p);

    ierr = PetscLogFlops(7); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return val;
}

double quad_neighbor_nodes_of_node_t::dx_forward_linear ( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f_p0_linear(f)-f[node_00])/d_p0;
}

double quad_neighbor_nodes_of_node_t::dx_backward_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f[node_00]-f_m0_linear(f))/d_m0;
}

double quad_neighbor_nodes_of_node_t::dy_forward_linear ( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f_0p_linear(f)-f[node_00])/d_0p;
}

double quad_neighbor_nodes_of_node_t::dy_backward_linear( const double *f ) const
{
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);

    return (f[node_00]-f_0m_linear(f))/d_0m;
}

double quad_neighbor_nodes_of_node_t::dxx_central( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);
    double f_m0,f_00,f_p0; x_ngbd_with_quadratic_interpolation(f,f_m0,f_00,f_p0);
    double val = ((f_p0-f_00)/d_p0+(f_m0-f_00)/d_m0)*2./(d_m0+d_p0);

    ierr = PetscLogFlops(8); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dxx_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return val;
}

double quad_neighbor_nodes_of_node_t::dyy_central( const double *f ) const
{
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    double f_0m,f_00,f_0p; y_ngbd_with_quadratic_interpolation(f,f_0m,f_00,f_0p);
    double val = ((f_0p-f_00)/d_0p+(f_0m-f_00)/d_0m)*2./(d_0m+d_0p);

    ierr = PetscLogFlops(8); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_quad_neighbor_nodes_of_node_t_dyy_central, 0, 0, 0, 0); CHKERRXX(ierr);

    return val;
}
