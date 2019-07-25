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

// forward declaration
class my_p4est_node_neighbors_t;

typedef struct {
  double weight;
  p4est_locidx_t node_idx;
} node_interpolation_weight;

typedef struct
{
  std::vector<node_interpolation_weight> elements;
  double interpolate(const double* node_sampled_field) const
  {
    P4EST_ASSERT(elements.size()>0);
    double value = elements[0].weight*node_sampled_field[elements[0].node_idx];
    for (size_t k = 1; k < elements.size(); ++k)
      value += elements[k].weight*node_sampled_field[elements[k].node_idx];
    return value;
  }
  double interpolate_dxx(const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors) const;
  double interpolate_dyy(const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors) const;
#ifdef P4_TO_P8
  double interpolate_dzz(const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors) const;
#endif
} node_interpolator;

struct quad_neighbor_nodes_of_node_t {
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

  inline double central_derivative(const double& fp, const double& f0, const double& fm, const double& dp, const double& dm) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(7); CHKERRXX(ierr);
#endif
    return ((fp-f0)*dm/dp + (f0-fm)*dp/dm)/(dp+dm);
  }

  inline double forward_derivative(const double& fp, const double& f0, const double& dp) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return (fp-f0)/dp;
  }

  inline double backward_derivative(const double& f0, const double& fm, const double& dm) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2); CHKERRXX(ierr);
#endif
    return (f0-fm)/dm;
  }

  inline double central_second_derivative(const double& fp, const double& f0, const double& fm, const double& dp, const double& dm) const
  {
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(8); CHKERRXX(ierr);
#endif
    return ((fp-f0)/dp + (fm-f0)/dm)*2./(dp+dm);
  }

  inline double f_m00_linear( const double *f ) const
  {
    double result;
    f_m00_linear(&f, &result, 1);
    return result;
  }
  inline double f_p00_linear( const double *f ) const
  {
    double result;
    f_p00_linear(&f, &result, 1);
    return result;
  }
  inline double f_0m0_linear( const double *f ) const
  {
    double result;
    f_0m0_linear(&f, &result, 1);
    return result;
  }
  inline double f_0p0_linear( const double *f ) const
  {
    double result;
    f_0p0_linear(&f, &result, 1);
    return result;
  }
#ifdef P4_TO_P8
  inline double f_00m_linear( const double *f ) const
  {
    double result;
    f_00m_linear(&f, &result, 1);
    return result;
  }
  inline double f_00p_linear( const double *f ) const
  {
    double result;
    f_00p_linear(&f, &result, 1);
    return result;
  }
#endif

  void f_m00_linear( const double *f[], double results[], const unsigned int& n_fields ) const;
  void f_p00_linear( const double *f[], double results[], const unsigned int& n_fields ) const;
  void f_0m0_linear( const double *f[], double results[], const unsigned int& n_fields ) const;
  void f_0p0_linear( const double *f[], double results[], const unsigned int& n_fields ) const;
#ifdef P4_TO_P8
  void f_00m_linear( const double *f[], const unsigned int& n_fields ) const;
  void f_00p_linear( const double *f[], const unsigned int& n_fields ) const;
#endif

  void  linear_interpolator_m00( node_interpolator& interpolator) const;
  void  linear_interpolator_p00( node_interpolator& interpolator) const;
  void  linear_interpolator_0m0( node_interpolator& interpolator) const;
  void  linear_interpolator_0p0( node_interpolator& interpolator) const;
#ifdef P4_TO_P8
  void  linear_interpolator_00m( node_interpolator& interpolator) const;
  void  linear_interpolator_00p( node_interpolator& interpolator) const;
#endif

  inline void ngbd_with_quadratic_interpolation( const double *f,
                                          double& f_000,
                                          double& f_m00, double& f_p00,
                                          double& f_0m0, double& f_0p0
                                        #ifdef P4_TO_P8
                                          ,double& f_00m, double& f_00p
                                        #endif
                                          ) const
  {
    ngbd_with_quadratic_interpolation(&f, &f_000, &f_m00, &f_p00, &f_0m0, &f_0p0,
                                  #ifdef P4_TO_P8
                                      &f_00m, &f_00p,
                                  #endif
                                      1);
  }
  void ngbd_with_quadratic_interpolation( const double *f[], double f_000[], double f_m00[], double f_p00[], double f_0m0[], double f_0p0[],
                                        #ifdef P4_TO_P8
                                          double f_00m[], double f_00p[],
                                        #endif
                                          const unsigned int& n_fields) const;
  void ngbd_with_quadratic_interpolation( const double *f, double& f_000, double& f_m00, double& f_p00, double& f_0m0, double& f_0p0,
                                        #ifdef P4_TO_P8
                                          double& f_00m, double& f_00p,
                                        #endif
                                          const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                          const node_interpolator& interp_0m0, const node_interpolator& interp_0p0
                                        #ifdef P4_TO_P8
                                          , const node_interpolator& interp_00m, const node_interpolator& interp_00p
                                        #endif
                                          ) const;

  inline void x_ngbd_with_quadratic_interpolation( const double *f, double& f_m00, double& f_000, double& f_p00) const
  {
    x_ngbd_with_quadratic_interpolation(&f, &f_m00, &f_000, &f_p00, 1);
  }
  void x_ngbd_with_quadratic_interpolation( const double *f[], double f_m00[], double f_000[], double f_p00[], const unsigned int& n_fields) const;
  void x_ngbd_with_quadratic_interpolation( const double *f, double& f_m00, double& f_000, double& f_p00,
                                            const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                            const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;

  inline void y_ngbd_with_quadratic_interpolation( const double *f, double& f_0m0, double& f_000, double& f_0p0) const
  {
    y_ngbd_with_quadratic_interpolation(&f, &f_0m0, &f_000, &f_0p0, 1);
  }
  void y_ngbd_with_quadratic_interpolation( const double *f[], double f_0m0[], double f_000[], double f_0p0[], const unsigned int& n_fields) const;
  void y_ngbd_with_quadratic_interpolation( const double *f, double& f_0m0, double& f_000, double& f_0p0,
                                            const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                            const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;

#ifdef P4_TO_P8
  void z_ngbd_with_quadratic_interpolation( const double *f, double& f_00m, double& f_000, double& f_00p) const;
  void z_ngbd_with_quadratic_interpolation( const double *f[], double f_00m[], double f_000[], double f_00p[], const unsigned int& n_fields) const;
  void z_ngbd_with_quadratic_interpolation( const double *f, double& f_00m, double& f_000, double& f_00p,
                                            const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                                            const node_interpolator& interp_0m0, const node_interpolator& interp_0p0,
                                            const node_interpolator& interp_00m, const node_interpolator& interp_00p) const;
#endif

  inline double dx_central ( const double *f ) const
  {
    double result;
    dx_central(&f, &result, 1);
    return result;
  }
  inline double dy_central ( const double *f ) const
  {
    double result;
    dy_central(&f, &result, 1);
    return result;
  }
#ifdef P4_TO_P8
  inline double dz_central ( const double *f ) const
  {
    double result;
    dz_central(&f, &result, 1);
    return result;
  }
#endif
  void dx_central (const double *f[], double results[], const unsigned int& n_fields) const;
  double dx_central (const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                     const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;
  void dy_central (const double *f[], double results[], const unsigned int& n_fields) const;
  double dy_central (const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                     const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;
#ifdef P4_TO_P8
  void dz_central (const double *f[], double results[], const unsigned int& n_fields) const;
  double dz_central (const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                     const node_interpolator& interp_0m0, const node_interpolator& interp_0p0,
                     const node_interpolator& interp_00m, const node_interpolator& interp_00p) const;
  void grad(const double *f, double& df_dx, double& df_dy, double& df_dz,
            const node_interpolator& interp_m00, const node_interpolator& interp_p00,
            const node_interpolator& interp_0m0, const node_interpolator& interp_0p0,
            const node_interpolator& interp_00m, const node_interpolator& interp_00p) const;
#else
  void grad(const double *f, double& df_dx, double& df_dy,
            const node_interpolator& interp_m00, const node_interpolator& interp_p00,
            const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;
#endif

  inline double d_central (const unsigned short& der, const double *f) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_central(f) : ((der == dir::y)? dy_central(f) : dz_central(f)));
#else
    return ((der == dir::x)? dx_central(f) : dy_central(f));
#endif
  }

#ifdef P4_TO_P8
  void gradient( const double* f, double& fx , double& fy , double& fz  ) const
  {
    gradient(&f, &fx, &fy, &fz, 1);
  }
  void gradient( const double* f[], double fx[], double fy[], double fz[], const unsigned int& n_fields) const;
#else
  void gradient( const double* f, double& fx , double& fy  ) const
  {
    gradient(&f, &fx, &fy, 1);
  }
  void gradient( const double* f[], double fx[], double fy[], const unsigned int& n_fields) const;
#endif
  void gradient( const double* f[], double grad_f[][P4EST_DIM], const unsigned int& n_fields) const;

  inline double dx_forward_linear ( const double *f ) const
  {
    double result;
    dx_forward_linear(&f, &result, 1);
    return result;
  }
  void dx_forward_linear (const double *f[], double results[], const unsigned int& n_fields) const;
  double dx_forward_linear (const double *f, const node_interpolator& interp_p00) const;

  inline double dx_backward_linear( const double *f ) const
  {
    double result;
    dx_backward_linear(&f, &result, 1);
    return result;
  }
  void dx_backward_linear (const double *f[], double results[], const unsigned int& n_fields) const;
  double dx_backward_linear (const double *f, const node_interpolator& interp_m00) const;

  inline double dy_forward_linear ( const double *f ) const
  {
    double result;
    dy_forward_linear(&f, &result, 1);
    return result;
  }
  void dy_forward_linear (const double *f[], double results[], const unsigned int& n_fields) const;
  double dy_forward_linear (const double *f, const node_interpolator& interp_0p0) const;

  inline double dy_backward_linear( const double *f ) const
  {
    double result;
    dy_backward_linear(&f, &result, 1);
    return result;
  }
  void dy_backward_linear (const double *f[], double results[], const unsigned int& n_fields) const;
  double dy_backward_linear (const double *f, const node_interpolator& interp_0m0) const;

#ifdef P4_TO_P8
  inline double dz_forward_linear ( const double *f ) const
  {
    double result;
    dz_forward_linear (&f, &result, 1);
    return result;
  }
  void dz_forward_linear (const double *f[], double results[], const unsigned int& n_fields) const;
  double dz_forward_linear (const double *f, const node_interpolator& interp_00p) const;

  inline double dz_backward_linear( const double *f ) const
  {
    double result;
    dz_backward_linear (&f, &result, 1);
    return result;
  }
  void dz_backward_linear (const double *f[], double results[], const unsigned int& n_fields) const;
  double dz_backward_linear (const double *f, const node_interpolator& interp_00m) const;

#endif
  inline double d_forward_linear(const unsigned short& der, const double *f) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_linear(f) : ((der == dir::y)? dy_forward_linear(f) : dz_forward_linear(f)));
#else
    return ((der == dir::x)? dx_forward_linear(f) : dy_forward_linear(f));
#endif
  }
  inline double d_backward_linear(const unsigned short& der, const double *f) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_linear(f) : ((der == dir::y)? dy_backward_linear(f) : dz_backward_linear(f)));
#else
    return ((der == dir::x)? dx_backward_linear(f) : dy_backward_linear(f));
#endif
  }


  double dx_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors ) const
  {
    double result;
    dx_backward_quadratic(&f, neighbors, &result, 1);
    return result;
  }
  void dx_backward_quadratic( const double *f[], const my_p4est_node_neighbors_t& neighbors, double results[], const unsigned int& n_fields) const;
  double dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                               const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                               const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
  double dx_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors ) const
  {
    double result;
    dx_forward_quadratic(&f, neighbors, &result, 1);
    return result;
  }
  void dx_forward_quadratic( const double *f[], const my_p4est_node_neighbors_t& neighbors, double results[], const unsigned int& n_fields) const;
  double dx_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                              const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                              const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
  double dy_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors ) const
  {
    double result;
    dy_forward_quadratic(&f, neighbors, &result, 1);
    return result;
  }
  void dy_forward_quadratic( const double *f[], const my_p4est_node_neighbors_t& neighbors, double results[], const unsigned int& n_fields) const;
  double dy_forward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                              const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                              const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
  double dy_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors ) const
  {
    double result;
    dy_backward_quadratic(&f, neighbors, &result, 1);
    return result;
  }
  void dy_backward_quadratic( const double *f[], const my_p4est_node_neighbors_t& neighbors, double results[], const unsigned int& n_fields) const;
  double dy_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors,
                              const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                              const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
#ifdef P4_TO_P8
  double dz_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
  double dz_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
#endif
  inline double d_forward_quadratic(const unsigned short& der, const double *f, const my_p4est_node_neighbors_t& neighbors ) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_quadratic(f, neighbors) : ((der == dir::y)? dy_forward_quadratic(f, neighbors) : dz_forward_quadratic(f, neighbors)));
#else
    return ((der == dir::x)? dx_forward_quadratic(f, neighbors) : dy_forward_quadratic(f, neighbors));
#endif
  }
  inline double d_backward_quadratic(const unsigned short& der, const double *f, const my_p4est_node_neighbors_t& neighbors ) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_quadratic(f, neighbors) : ((der == dir::y)? dy_backward_quadratic(f, neighbors) : dz_backward_quadratic(f, neighbors)));
#else
    return ((der == dir::x)? dx_backward_quadratic(f, neighbors) : dy_backward_quadratic(f, neighbors));
#endif
  }

  double dx_forward_quadratic ( const double *f, const double *fxx ) const
  {
    double result;
    dx_forward_quadratic (&f, &fxx, &result, 1);
    return result;
  }
  void dx_forward_quadratic ( const double *f[], const double *fxx[], double results[], const unsigned int& n_fields ) const;
  double dx_forward_quadratic(const double *f, const double* fxx,
                              const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                              const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;

  double dx_backward_quadratic( const double *f, const double *fxx ) const
  {
    double result;
    dx_backward_quadratic (&f, &fxx, &result, 1);
    return result;
  }
  void dx_backward_quadratic ( const double *f[], const double *fxx[], double results[], const unsigned int& n_fields ) const;
  double dx_backward_quadratic(const double *f, const double* fxx,
                               const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                               const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
  double dy_forward_quadratic ( const double *f, const double *fyy ) const
  {
    double result;
    dy_forward_quadratic (&f, &fyy, &result, 1);
    return result;
  }
  void dy_forward_quadratic ( const double *f[], const double *fyy[], double results[], const unsigned int& n_fields ) const;
  double dy_forward_quadratic(const double *f, const double* fyy,
                              const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                              const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
  double dy_backward_quadratic( const double *f, const double *fyy ) const
  {
    double result;
    dy_backward_quadratic (&f, &fyy, &result, 1);
    return result;
  }
  void dy_backward_quadratic ( const double *f[], const double *fyy[], double results[], const unsigned int& n_fields ) const;
  double dy_backward_quadratic(const double *f, const double* fyy,
                               const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                               const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
#ifdef P4_TO_P8
  double dz_forward_quadratic ( const double *f, const double *fzz ) const
  {
    double result;
    dz_forward_quadratic (&f, &fyy, &result, 1);
    return result;
  }
  void dz_forward_quadratic ( const double *f[], const double *fzz[], double results[], const unsigned int& n_fields ) const;
  double dz_forward_quadratic(const double *f, const double* fzz,
                              const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                              const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
  double dz_backward_quadratic( const double *f, const double *fzz ) const
  {
    double result;
    dz_backward_quadratic (&f, &fyy, &result, 1);
    return result;
  }
  void dz_backward_quadratic ( const double *f[], const double *fzz[], double results[], const unsigned int& n_fields ) const;
  double dz_backward_quadratic(const double *f, const double* fzz,
                               const node_interpolator &interp_m00, const node_interpolator &interp_p00,
                               const node_interpolator &interp_0m0, const node_interpolator &interp_0p0) const;
#endif
  inline double d_forward_quadratic(const unsigned short& der, const double *f, const double *fderder) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_quadratic(f, fderder) : ((der == dir::y)? dy_forward_quadratic(f, fderder) : dz_forward_quadratic(f, fderder)));
#else
    return ((der == dir::x)? dx_forward_quadratic(f, fderder) : dy_forward_quadratic(f, fderder));
#endif
  }
  inline double d_backward_quadratic(const unsigned short& der, const double *f, const double *fderder) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_quadratic(f, fderder) : ((der == dir::y)? dy_backward_quadratic(f, fderder) : dz_backward_quadratic(f, fderder)));
#else
    return ((der == dir::x)? dx_backward_quadratic(f, fderder) : dy_backward_quadratic(f, fderder));
#endif
  }

  /* second-order derivatives */
  inline double dxx_central( const double *f ) const
  {
    double result;
    dxx_central(&f, &result, 1);
    return result;
  }
  void dxx_central( const double *f[], double results[], const unsigned int& n_fields ) const;
  double dxx_central( const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00, const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;

  double dyy_central( const double *f ) const
  {
    double result;
    dyy_central(&f, &result, 1);
    return result;
  }
  void dyy_central( const double *f[], double results[], const unsigned int& n_fields ) const;
  double dyy_central( const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00, const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;
#ifdef P4_TO_P8
  double dzz_central( const double *f ) const
  {
    double result;
    dzz_central(&f, &result, 1);
    return result;
  }
  void dzz_central( const double *f[], double results[], const unsigned int& n_fields ) const;
  double dzz_central( const double *f, const node_interpolator& interp_m00, const node_interpolator& interp_p00, const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;
#endif
  inline double dd_central (const unsigned short& der, const double *f) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dxx_central(f) : ((der == dir::y)? dyy_central(f) : dzz_central(f)));
#else
    return ((der == dir::x)? dxx_central(f) : dyy_central(f));
#endif
  }

#ifdef P4_TO_P8
  inline void laplace ( const double* f, double& fxx, double& fyy, double& fzz ) const
  {
    laplace(&f, &fxx, &fyy, &fzz, 1);
  }
  void laplace ( const double* f[], double fxx[], double fyy[], double fzz[], const unsigned int& n_fields) const;
  void laplace( const double *f, double &fxx, double &fyy,
                const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                const node_interpolator& interp_0m0, const node_interpolator& interp_0p0,
                const node_interpolator& interp_00m, const node_interpolator& interp_00p) const;
#else
  inline void laplace ( const double* f, double& fxx, double& fyy ) const
  {
    laplace(&f, &fxx, &fyy, 1);
  }
  void laplace ( const double* f[], double fxx[], double fyy[], const unsigned int& n_fields) const;
  void laplace( const double *f, double &fxx, double &fyy,
                const node_interpolator& interp_m00, const node_interpolator& interp_p00,
                const node_interpolator& interp_0m0, const node_interpolator& interp_0p0) const;
#endif

  // the following are no longer needed, given the capabilities of node_interpolator

//  double dxx_central_on_m00(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
//  double dxx_central_on_p00(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
//  double dyy_central_on_0m0(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
//  double dyy_central_on_0p0(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
//#ifdef P4_TO_P8
//  double dzz_central_on_00m(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
//  double dzz_central_on_00p(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
//#endif

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

#endif /* !MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H */
