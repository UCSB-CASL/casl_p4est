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

typedef struct node_interpolator
{
  node_interpolator() : is_set(false) {}
  bool is_set;
  std::vector<node_interpolation_weight> elements;
  double interpolate(const double* node_sampled_field, const unsigned int& block_size, const unsigned int& component) const
  {
    P4EST_ASSERT((component < block_size) && (block_size > 0));
    P4EST_ASSERT(elements.size()>0);
    double value = elements[0].weight*node_sampled_field[block_size*elements[0].node_idx+component];
    for (size_t k = 1; k < elements.size(); ++k)
      value += elements[k].weight*node_sampled_field[block_size*elements[k].node_idx+component];
#ifdef CASL_LOG_FLOPS
    PetscErrorCode ierr = PetscLogFlops(2*interpolator.elements.size()-1); CHKERRXX(ierr);
#endif
    return value;
  }
  double interpolate_dd (unsigned char der, const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size, const unsigned int& component) const;
  inline double interpolate_dxx(const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size, const unsigned int& component) const
  {
    return interpolate_dd(dir::x, node_sample_field, neighbors, block_size, component);
  }
  inline double interpolate_dyy(const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size, const unsigned int& component) const
  {
    return interpolate_dd(dir::y, node_sample_field, neighbors, block_size, component);
  }
#ifdef P4_TO_P8
  inline double interpolate_dzz(const double* node_sample_field, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size, const unsigned int& component) const
  {
    return interpolate_dd(dir::z, node_sample_field, neighbors, block_size, component);
  }
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

  node_interpolator interpolator_m00;
  node_interpolator interpolator_p00;
  node_interpolator interpolator_0m0;
  node_interpolator interpolator_0p0;
#ifdef P4_TO_P8
  node_interpolator interpolator_00m;
  node_interpolator interpolator_00p;
#endif

  void  set_linear_interpolator_m00();
  void  set_linear_interpolator_p00();
  void  set_linear_interpolator_0m0();
  void  set_linear_interpolator_0p0();
#ifdef P4_TO_P8
  void  set_linear_interpolator_00m();
  void  set_linear_interpolator_00p();
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

  inline double f_m00_linear( const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
    P4EST_ASSERT((component < block_size) && (block_size > 0));
    P4EST_ASSERT(interpolator_m00.is_set);
    return interpolator_m00.interpolate(f, block_size, component);
  }

  inline double f_p00_linear( const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
    P4EST_ASSERT((component < block_size) && (block_size > 0));
    P4EST_ASSERT(interpolator_p00.is_set);
    return interpolator_p00.interpolate(f, block_size, component);
  }

  inline double f_0m0_linear( const double *f, const unsigned int &block_size=1, const unsigned int &component=0) const
  {
    P4EST_ASSERT((component < block_size) && (block_size > 0));
    P4EST_ASSERT(interpolator_0m0.is_set);
    return interpolator_0m0.interpolate(f, block_size, component);
  }

  inline double f_0p0_linear( const double *f, const unsigned int& block_size=1, const unsigned int &component=0) const
  {
    P4EST_ASSERT((component < block_size) && (block_size > 0));
    P4EST_ASSERT(interpolator_0p0.is_set);
    return interpolator_0p0.interpolate(f, block_size, component);
  }
#ifdef P4_TO_P8
  inline double f_00m_linear( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  inline double f_00p_linear( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif



  void ngbd_with_quadratic_interpolation( const double *f,
                                          double& f_000,
                                          double& f_m00, double& f_p00,
                                          double& f_0m0, double& f_0p0,
                                        #ifdef P4_TO_P8
                                          double& f_00m, double& f_00p,
                                        #endif
                                          const unsigned int& block_size=1, const unsigned int& component=0) const;

  void x_ngbd_with_quadratic_interpolation( const double *f, double& f_m00, double& f_000, double& f_p00, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  void y_ngbd_with_quadratic_interpolation( const double *f, double& f_0m0, double& f_000, double& f_0p0, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#ifdef P4_TO_P8
  void z_ngbd_with_quadratic_interpolation( const double *f, double& f_00m, double& f_000, double& f_00p, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif

  double dx_central ( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dy_central ( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#ifdef P4_TO_P8
  double dz_central ( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif

  inline double d_central (const unsigned short& der, const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_central(f, block_size, component) : ((der == dir::y)? dy_central(f, block_size, component) : dz_central(f, block_size, component)));
#else
    return ((der == dir::x)? dx_central(f, block_size, component) : dy_central(f, block_size, component));
#endif
  }

#ifdef P4_TO_P8
  void gradient( const double* f, double& fx , double& fy , double& fz, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#else
  void gradient( const double* f, double& fx , double& fy, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif

  double dx_forward_linear (const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dx_backward_linear (const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dy_forward_linear (const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dy_backward_linear (const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#ifdef P4_TO_P8
  double dz_forward_linear (const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dz_backward_linear (const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif

  inline double d_forward_linear(const unsigned short& der, const double *f, const unsigned int& block_size=1, const unsigned int& component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_linear(f, block_size, component) : ((der == dir::y)? dy_forward_linear(f, block_size, component) : dz_forward_linear(f, block_size, component)));
#else
    return ((der == dir::x)? dx_forward_linear(f, block_size, component) : dy_forward_linear(f, block_size, component));
#endif
  }
  inline double d_backward_linear(const unsigned short& der, const double *f, const unsigned int& block_size=1, const unsigned int& component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_linear(f, block_size, component) : ((der == dir::y)? dy_backward_linear(f, block_size, component) : dz_backward_linear(f, block_size, component)));
#else
    return ((der == dir::x)? dx_backward_linear(f, block_size, component) : dy_backward_linear(f, block_size, component));
#endif
  }

  double dx_backward_quadratic( const double *f, const double& fxx_m00, const unsigned int& bs, const unsigned int& comp) const;
  double dx_forward_quadratic( const double *f, const double& fxx_p00, const unsigned int& bs, const unsigned int& comp) const;
  double dy_backward_quadratic( const double *f, const double& fyy_0m0, const unsigned int& bs, const unsigned int& comp) const;
  double dy_forward_quadratic( const double *f, const double& fyy_0p0, const unsigned int& bs, const unsigned int& comp) const;
#ifdef P4_TO_P8
  double dz_backward_quadratic( const double *f, const double& fzz_00m, const unsigned int& bs, const unsigned int& comp) const;
  double dz_forward_quadratic( const double *f, const double& fzz_00p, const unsigned int& bs, const unsigned int& comp) const;
#endif
  double dx_backward_quadratic(const double *f, const my_p4est_node_neighbors_t &neighbors, const unsigned int& block_size=1, const unsigned int& component=0) const;
  double dx_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0) const;
  double dy_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0) const;
  double dy_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0) const;
#ifdef P4_TO_P8
  double dz_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dz_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif
  inline double d_backward_quadratic(const unsigned short& der, const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_quadratic(f, neighbors, block_size, component) : ((der == dir::y)? dy_backward_quadratic(f, neighbors, block_size, component) : dz_backward_quadratic(f, neighbors, block_size, component)));
#else
    return ((der == dir::x)? dx_backward_quadratic(f, neighbors, block_size, component) : dy_backward_quadratic(f, neighbors, block_size, component));
#endif
  }
  inline double d_forward_quadratic(const unsigned short& der, const double *f, const my_p4est_node_neighbors_t& neighbors, const unsigned int& block_size=1, const unsigned int& component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_quadratic(f, neighbors, block_size, component) : ((der == dir::y)? dy_forward_quadratic(f, neighbors, block_size, component) : dz_forward_quadratic(f, neighbors, block_size, component)));
#else
    return ((der == dir::x)? dx_forward_quadratic(f, neighbors, block_size, component) : dy_forward_quadratic(f, neighbors, block_size, component));
#endif
  }

  double dx_backward_quadratic( const double *f, const double *fxx, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dx_forward_quadratic ( const double *f, const double *fxx, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dy_backward_quadratic( const double *f, const double *fyy, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dy_forward_quadratic ( const double *f, const double *fyy, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#ifdef P4_TO_P8
  double dz_backward_quadratic( const double *f, const double *fzz, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dz_forward_quadratic ( const double *f, const double *fzz, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif
  inline double d_backward_quadratic(const unsigned short& der, const double *f, const double *fderder, const unsigned int& block_size=1, const unsigned int& component=0 ) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_backward_quadratic(f, fderder, block_size, component) : ((der == dir::y)? dy_backward_quadratic(f, fderder, block_size, component) : dz_backward_quadratic(f, fderder, block_size, component)));
#else
    return ((der == dir::x)? dx_backward_quadratic(f, fderder, block_size, component) : dy_backward_quadratic(f, fderder, block_size, component));
#endif
  }
  inline double d_forward_quadratic(const unsigned short& der, const double *f, const double *fderder, const unsigned int& block_size=1, const unsigned int& component=0) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dx_forward_quadratic(f, fderder, block_size, component) : ((der == dir::y)? dy_forward_quadratic(f, fderder, block_size, component) : dz_forward_quadratic(f, fderder, block_size, component)));
#else
    return ((der == dir::x)? dx_forward_quadratic(f, fderder, block_size, component) : dy_forward_quadratic(f, fderder, block_size, component));
#endif
  }

  /* second-order derivatives */
  double dxx_central( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
  double dyy_central( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#ifdef P4_TO_P8
  double dzz_central( const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif
  inline double dd_central (const unsigned short& der, const double *f, const unsigned int& block_size=1, const unsigned int& component=0 ) const
  {
#ifdef P4_TO_P8
    return ((der == dir::x)? dxx_central(f, block_size, component) : ((der == dir::y)? dyy_central(f, block_size, component) : dzz_central(f, block_size, component)));
#else
    return ((der == dir::x)? dxx_central(f, block_size, component) : dyy_central(f, block_size, component));
#endif
  }

#ifdef P4_TO_P8
  inline void laplace ( const double* f, double& fxx, double& fyy, double& fzz, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#else
  inline void laplace ( const double* f, double& fxx, double& fyy, const unsigned int& block_size=1, const unsigned int& component=0 ) const;
#endif

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
