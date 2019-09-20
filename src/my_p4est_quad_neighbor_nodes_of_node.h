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

  double f_m00_linear( const double *f ) const;
  double f_p00_linear( const double *f ) const;
  double f_0m0_linear( const double *f ) const;
  double f_0p0_linear( const double *f ) const;
#ifdef P4_TO_P8
  double f_00m_linear( const double *f ) const;
  double f_00p_linear( const double *f ) const;
#endif

  void ngbd_with_quadratic_interpolation( const double *f,
                                          double& f_000,
                                          double& f_m00, double& f_p00,
                                          double& f_0m0, double& f_0p0
                                        #ifdef P4_TO_P8
                                          ,double& f_00m, double& f_00p
                                        #endif
                                          ) const;

  inline void ngbd_with_quadratic_interpolation( const double *f, double& f_000,double f_nei[]) const
  {
#ifdef P4_TO_P8
    ngbd_with_quadratic_interpolation(f, f_000, f_nei[dir::f_m00], f_nei[dir::f_p00], f_nei[dir::f_0m0], f_nei[dir::f_0p0], f_nei[dir::f_00m], f_nei[dir::f_00p]);
#else
    ngbd_with_quadratic_interpolation(f, f_000, f_nei[dir::f_m00], f_nei[dir::f_p00], f_nei[dir::f_0m0], f_nei[dir::f_0p0]);
#endif
  }

  void x_ngbd_with_quadratic_interpolation( const double *f,
                                            double& f_m00, double& f_000, double& f_p00) const;

  void y_ngbd_with_quadratic_interpolation( const double *f,
                                            double& f_0m0, double& f_000, double& f_0p0) const;
#ifdef P4_TO_P8
  void z_ngbd_with_quadratic_interpolation( const double *f,
                                            double& f_00m, double& f_000, double& f_00p) const;
#endif

  double dx_central ( const double *f ) const;
  double dy_central ( const double *f ) const;
#ifdef P4_TO_P8
  double dz_central ( const double *f ) const;
#endif

#ifdef P4_TO_P8
  void gradient( const double* f, double& fx , double& fy , double& fz  ) const;
#else
  void gradient( const double* f, double& fx , double& fy  ) const;
#endif

  double dx_forward_linear ( const double *f ) const;  
  double dx_backward_linear( const double *f ) const;
  double dy_forward_linear ( const double *f ) const;
  double dy_backward_linear( const double *f ) const;
#ifdef P4_TO_P8
  double dz_forward_linear ( const double *f ) const;
  double dz_backward_linear( const double *f ) const;
#endif

  double dx_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
  double dx_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
  double dy_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
  double dy_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
#ifdef P4_TO_P8
  double dz_forward_quadratic ( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
  double dz_backward_quadratic( const double *f, const my_p4est_node_neighbors_t& neighbors ) const;
#endif

  double dx_forward_quadratic ( const double *f, const double *fxx ) const;
  double dx_backward_quadratic( const double *f, const double *fxx ) const;
  double dy_forward_quadratic ( const double *f, const double *fyy ) const;
  double dy_backward_quadratic( const double *f, const double *fyy ) const;
#ifdef P4_TO_P8
  double dz_forward_quadratic ( const double *f, const double *fzz ) const;
  double dz_backward_quadratic( const double *f, const double *fzz ) const;
#endif

  /* second-order derivatives */
  double dxx_central( const double *f ) const;
  double dyy_central( const double *f ) const;
#ifdef P4_TO_P8
  double dzz_central( const double *f ) const;
#endif

#ifdef P4_TO_P8
  void laplace ( const double* f, double& fxx, double& fyy, double& fzz ) const;
#else
  void laplace ( const double* f, double& fxx, double& fyy ) const;
#endif

  double dxx_central_on_m00(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dxx_central_on_p00(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dyy_central_on_0m0(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dyy_central_on_0p0(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dy_central_on_m00(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dy_central_on_p00(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dx_central_on_0m0(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dx_central_on_0p0(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
#ifdef P4_TO_P8
  double dzz_central_on_00m(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
  double dzz_central_on_00p(const double *f, const my_p4est_node_neighbors_t& neighbors) const;
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

  p4est_locidx_t neighbor_m00() const;
  p4est_locidx_t neighbor_p00() const;
  p4est_locidx_t neighbor_0m0() const;
  p4est_locidx_t neighbor_0p0() const;
#ifdef P4_TO_P8
  p4est_locidx_t neighbor_00m() const;
  p4est_locidx_t neighbor_00p() const;
#endif

  inline p4est_locidx_t neighbor(int dir) const
  {
    switch (dir)
    {
      case dir::f_m00: return neighbor_m00();
      case dir::f_p00: return neighbor_p00();
      case dir::f_0m0: return neighbor_0m0();
      case dir::f_0p0: return neighbor_0p0();
#ifdef P4_TO_P8
      case dir::f_00m: return neighbor_00m();
      case dir::f_00p: return neighbor_00p();
#endif
      default: throw std::invalid_argument("Invalid direction\n");
    }
  }

  inline double distance(int dir) const
  {
    switch (dir)
    {
      case dir::f_m00: return d_m00;
      case dir::f_p00: return d_p00;
      case dir::f_0m0: return d_0m0;
      case dir::f_0p0: return d_0p0;
#ifdef P4_TO_P8
      case dir::f_00m: return d_00m;
      case dir::f_00p: return d_00p;
#endif
      default: throw std::invalid_argument("Invalid direction\n");
    }
  }

  inline bool is_stencil_in_negative_domain(double *phi_p) const
  {
    return phi_p[this->node_000]<-EPS &&
    #ifdef P4_TO_P8
        ( phi_p[this->node_m00_mm]<-EPS || fabs(this->d_m00_p0)<EPS || fabs(this->d_m00_0p)<EPS) &&
        ( phi_p[this->node_m00_mp]<-EPS || fabs(this->d_m00_p0)<EPS || fabs(this->d_m00_0m)<EPS) &&
        ( phi_p[this->node_m00_pm]<-EPS || fabs(this->d_m00_m0)<EPS || fabs(this->d_m00_0p)<EPS) &&
        ( phi_p[this->node_m00_pp]<-EPS || fabs(this->d_m00_m0)<EPS || fabs(this->d_m00_0m)<EPS) &&
        ( phi_p[this->node_p00_mm]<-EPS || fabs(this->d_p00_p0)<EPS || fabs(this->d_p00_0p)<EPS) &&
        ( phi_p[this->node_p00_mp]<-EPS || fabs(this->d_p00_p0)<EPS || fabs(this->d_p00_0m)<EPS) &&
        ( phi_p[this->node_p00_pm]<-EPS || fabs(this->d_p00_m0)<EPS || fabs(this->d_p00_0p)<EPS) &&
        ( phi_p[this->node_p00_pp]<-EPS || fabs(this->d_p00_m0)<EPS || fabs(this->d_p00_0m)<EPS) &&
        ( phi_p[this->node_0m0_mm]<-EPS || fabs(this->d_0m0_p0)<EPS || fabs(this->d_0m0_0p)<EPS) &&
        ( phi_p[this->node_0m0_mp]<-EPS || fabs(this->d_0m0_p0)<EPS || fabs(this->d_0m0_0m)<EPS) &&
        ( phi_p[this->node_0m0_pm]<-EPS || fabs(this->d_0m0_m0)<EPS || fabs(this->d_0m0_0p)<EPS) &&
        ( phi_p[this->node_0m0_pp]<-EPS || fabs(this->d_0m0_m0)<EPS || fabs(this->d_0m0_0m)<EPS) &&
        ( phi_p[this->node_0p0_mm]<-EPS || fabs(this->d_0p0_p0)<EPS || fabs(this->d_0p0_0p)<EPS) &&
        ( phi_p[this->node_0p0_mp]<-EPS || fabs(this->d_0p0_p0)<EPS || fabs(this->d_0p0_0m)<EPS) &&
        ( phi_p[this->node_0p0_pm]<-EPS || fabs(this->d_0p0_m0)<EPS || fabs(this->d_0p0_0p)<EPS) &&
        ( phi_p[this->node_0p0_pp]<-EPS || fabs(this->d_0p0_m0)<EPS || fabs(this->d_0p0_0m)<EPS) &&
        ( phi_p[this->node_00m_mm]<-EPS || fabs(this->d_00m_p0)<EPS || fabs(this->d_00m_0p)<EPS) &&
        ( phi_p[this->node_00m_mp]<-EPS || fabs(this->d_00m_p0)<EPS || fabs(this->d_00m_0m)<EPS) &&
        ( phi_p[this->node_00m_pm]<-EPS || fabs(this->d_00m_m0)<EPS || fabs(this->d_00m_0p)<EPS) &&
        ( phi_p[this->node_00m_pp]<-EPS || fabs(this->d_00m_m0)<EPS || fabs(this->d_00m_0m)<EPS) &&
        ( phi_p[this->node_00p_mm]<-EPS || fabs(this->d_00p_p0)<EPS || fabs(this->d_00p_0p)<EPS) &&
        ( phi_p[this->node_00p_mp]<-EPS || fabs(this->d_00p_p0)<EPS || fabs(this->d_00p_0m)<EPS) &&
        ( phi_p[this->node_00p_pm]<-EPS || fabs(this->d_00p_m0)<EPS || fabs(this->d_00p_0p)<EPS) &&
        ( phi_p[this->node_00p_pp]<-EPS || fabs(this->d_00p_m0)<EPS || fabs(this->d_00p_0m)<EPS);
    #else
        ( phi_p[this->node_m00_mm]<-EPS || fabs(this->d_m00_p0)<EPS) &&
        ( phi_p[this->node_m00_pm]<-EPS || fabs(this->d_m00_m0)<EPS) &&
        ( phi_p[this->node_p00_mm]<-EPS || fabs(this->d_p00_p0)<EPS) &&
        ( phi_p[this->node_p00_pm]<-EPS || fabs(this->d_p00_m0)<EPS) &&
        ( phi_p[this->node_0m0_mm]<-EPS || fabs(this->d_0m0_p0)<EPS) &&
        ( phi_p[this->node_0m0_pm]<-EPS || fabs(this->d_0m0_m0)<EPS) &&
        ( phi_p[this->node_0p0_mm]<-EPS || fabs(this->d_0p0_p0)<EPS) &&
        ( phi_p[this->node_0p0_pm]<-EPS || fabs(this->d_0p0_m0)<EPS);
#endif
  }

  inline double interpolate_in_dir(int dir, double dist, double *f_ptr) const
  {
    p4est_locidx_t node_nei = neighbor(dir);
    double         h   = distance(dir);
    if (node_nei == -1) throw std::domain_error("interpolate_in_dir does not support non-uniform grids yet\n");
    return f_ptr[node_000]*(1-dist/h) + f_ptr[node_nei]*dist/h;
  }

  inline double interpolate_in_dir(int dir, double dist, double *f_ptr, double *f_dd_ptr) const
  {
    p4est_locidx_t node_nei = neighbor(dir);
    double         h   = distance(dir);
    if (node_nei == -1) throw std::domain_error("interpolate_in_dir doesn not support non-uniform grids yet\n");
    return f_ptr[node_000]*(1-dist/h) + f_ptr[node_nei]*dist/h + 0.5*dist*(dist-h)*MINMOD(f_dd_ptr[node_000], f_dd_ptr[node_nei]);
  }

  inline double interpolate_in_dir(int dir, double dist, double *f_ptr, double *f_dd_ptr[]) const
  {
    p4est_locidx_t node_nei = neighbor(dir);
    double         h   = distance(dir);
    int            dim = dir / 2;
    if (node_nei == -1) throw std::domain_error("interpolate_in_dir does not support non-uniform grids yet\n");
    return f_ptr[node_000]*(1-dist/h) + f_ptr[node_nei]*dist/h + 0.5*dist*(dist-h)*MINMOD(f_dd_ptr[dim][node_000], f_dd_ptr[dim][node_nei]);
  }

  // inlined duplicates

#ifdef P4_TO_P8
  inline double inl_f_m00_linear(const double* f) const{
//      PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
      return (f[node_m00_mm]*d_m00_p0*d_m00_0p +
              f[node_m00_mp]*d_m00_p0*d_m00_0m +
              f[node_m00_pm]*d_m00_m0*d_m00_0p +
              f[node_m00_pp]*d_m00_m0*d_m00_0m )/(d_m00_m0 + d_m00_p0)/(d_m00_0m + d_m00_0p);
  }
  inline double inl_f_p00_linear(const double* f) const{
//      PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
      return (f[node_p00_mm]*d_p00_p0*d_p00_0p +
              f[node_p00_mp]*d_p00_p0*d_p00_0m +
              f[node_p00_pm]*d_p00_m0*d_p00_0p +
              f[node_p00_pp]*d_p00_m0*d_p00_0m )/(d_p00_m0 + d_p00_p0)/(d_p00_0m + d_p00_0p);
  }
  inline double inl_f_0m0_linear(const double* f) const{
//      PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
      return (f[node_0m0_mm]*d_0m0_p0*d_0m0_0p +
              f[node_0m0_mp]*d_0m0_p0*d_0m0_0m +
              f[node_0m0_pm]*d_0m0_m0*d_0m0_0p +
              f[node_0m0_pp]*d_0m0_m0*d_0m0_0m )/(d_0m0_m0 + d_0m0_p0)/(d_0m0_0m + d_0m0_0p);
  }
  inline double inl_f_0p0_linear(const double* f) const{
//      PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
      return (f[node_0p0_mm]*d_0p0_p0*d_0p0_0p +
              f[node_0p0_mp]*d_0p0_p0*d_0p0_0m +
              f[node_0p0_pm]*d_0p0_m0*d_0p0_0p +
              f[node_0p0_pp]*d_0p0_m0*d_0p0_0m )/(d_0p0_m0 + d_0p0_p0)/(d_0p0_0m + d_0p0_0p);
  }
  inline double inl_f_00m_linear(const double* f) const{
//      PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
      return (f[node_00m_mm]*d_00m_p0*d_00m_0p +
              f[node_00m_mp]*d_00m_p0*d_00m_0m +
              f[node_00m_pm]*d_00m_m0*d_00m_0p +
              f[node_00m_pp]*d_00m_m0*d_00m_0m )/(d_00m_m0 + d_00m_p0)/(d_00m_0m + d_00m_0p);
  }
  inline double inl_f_00p_linear(const double* f) const{
//      PetscErrorCode ierr = PetscLogFlops(17); CHKERRXX(ierr);
      return (f[node_00p_mm]*d_00p_p0*d_00p_0p +
              f[node_00p_mp]*d_00p_p0*d_00p_0m +
              f[node_00p_pm]*d_00p_m0*d_00p_0p +
              f[node_00p_pp]*d_00p_m0*d_00p_0m )/(d_00p_m0 + d_00p_p0)/(d_00p_0m + d_00p_0p);
  }
#else
  inline double inl_f_m00_linear( const double *f ) const
  {
//    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_m00_p0==0) return f[node_m00_pm];
    if(d_m00_m0==0) return f[node_m00_mm];
    else          return(f[node_m00_mm]*d_m00_p0 + f[node_m00_pm]*d_m00_m0)/ (d_m00_m0+d_m00_p0);
  }
  inline double inl_f_p00_linear( const double *f ) const
  {
//    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_p00_p0==0) return f[node_p00_pm];
    if(d_p00_m0==0) return f[node_p00_mm];
    else          return(f[node_p00_mm]*d_p00_p0 + f[node_p00_pm]*d_p00_m0)/ (d_p00_m0+d_p00_p0);
  }
  inline double inl_f_0m0_linear( const double *f ) const
  {
//    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0m0_m0==0) return f[node_0m0_mm];
    if(d_0m0_p0==0) return f[node_0m0_pm];
    else          return(f[node_0m0_pm]*d_0m0_m0 + f[node_0m0_mm]*d_0m0_p0)/ (d_0m0_m0+d_0m0_p0);
  }
  inline double inl_f_0p0_linear( const double *f ) const
  {
//    PetscErrorCode ierr = PetscLogFlops(2.5); CHKERRXX(ierr); // 50% propability
    if(d_0p0_m0==0) return f[node_0p0_mm];
    if(d_0p0_p0==0) return f[node_0p0_pm];
    else          return(f[node_0p0_pm]*d_0p0_m0 + f[node_0p0_mm]*d_0p0_p0)/ (d_0p0_m0+d_0p0_p0);
  }
#endif
};

#endif /* !MY_P4EST_QUAD_NEIGHBOR_NODES_OF_NODE_H */
