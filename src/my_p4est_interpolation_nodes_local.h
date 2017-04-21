#ifndef MY_P4EST_INTERPOLATION_NODES_LOCAL
#define MY_P4EST_INTERPOLATION_NODES_LOCAL

#ifdef P4_TO_P8
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_node_neighbors.h>
#endif

class my_p4est_interpolation_nodes_local_t
{
//private:
public:
  // p4est info
  p4est_nodes_t *nodes;
  p4est_t       *p4est;
  p4est_ghost_t *ghost;

  const my_p4est_node_neighbors_t *node_neighbors;

  // dimensions of computational box
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  // interpolation input (either as Vec's or pointers)
  bool is_input_in_vec;

  const double *Fi_p;
  Vec Fi;

  const double *Fxx_p, *Fyy_p;
  Vec Fxx, Fyy;

#ifdef P4_TO_P8
  const double *Fzz_p;
  Vec Fzz;
#endif

  // local values of input
  double f  [P4EST_CHILDREN];
  double fxx[P4EST_CHILDREN];
  double fyy[P4EST_CHILDREN];
#ifdef P4_TO_P8
  double fzz[P4EST_CHILDREN];
#endif

#ifdef P4_TO_P8
  grid_interpolation3_t interp;
#else
  grid_interpolation2_t interp;
#endif

  // neighboring quadrants
  p4est_locidx_t quad_idx[P4EST_CHILDREN];
  p4est_topidx_t tree_idx[P4EST_CHILDREN];

  double xyz_quad_max[P4EST_DIM*P4EST_CHILDREN];
  double xyz_quad_min[P4EST_DIM*P4EST_CHILDREN];

  short level_of_quad[P4EST_CHILDREN];

  // order of interpolation
  interpolation_method method;

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_nodes_local_t(const my_p4est_interpolation_nodes_local_t& other);
  my_p4est_interpolation_nodes_local_t& operator=(const my_p4est_interpolation_nodes_local_t& other);

  double eps;

  my_p4est_interpolation_nodes_local_t(const my_p4est_node_neighbors_t* ngbd_n)
    : nodes(ngbd_n->nodes), node_neighbors(ngbd_n), p4est(ngbd_n->p4est), ghost(ngbd_n->ghost),
      Fxx(NULL), Fxx_p(NULL), Fyy(NULL), Fyy_p(NULL),
      #ifdef P4_TO_P8
      Fzz(NULL), Fzz_p(NULL),
      #endif
      method(linear), eps(1.0E-15)
  {
    // compute domain sizes
    double *v2c = p4est->connectivity->vertices;
    p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
    p4est_topidx_t first_tree = 0, last_tree = p4est->trees->elem_count-1;
    p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

    for (short i=0; i<P4EST_DIM; i++)
    {
      xyz_min[i] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + i];
      xyz_max[i] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + i];
    }
  }

  // initialize interpolator
  void initialize(p4est_locidx_t n);

  // set input as Vec's
  void set_input(Vec Fi_in, interpolation_method method_in)
  {
    Fi = Fi_in; method = method_in; is_input_in_vec = true;
  }

#ifdef P4_TO_P8
  void set_input(Vec Fi_in, Vec Fxx_in, Vec Fyy_in, Vec Fzz_in, interpolation_method method_in)
  {
    Fi = Fi_in; Fxx = Fxx_in; Fyy = Fyy_in; Fzz = Fzz_in;
    method = method_in; is_input_in_vec = true;
  }
#else
  void set_input(Vec Fi_in, Vec Fxx_in, Vec Fyy_in, interpolation_method method_in)
  {
    Fi = Fi_in; Fxx = Fxx_in; Fyy = Fyy_in;
    method = method_in; is_input_in_vec = true;
  }
#endif

  // set input as pointers
  void set_input(double* Fi_p_in, interpolation_method method_in)
  {
    Fi_p = Fi_p_in;
    method = method_in; is_input_in_vec = false;
  }

#ifdef P4_TO_P8
  void set_input(double* Fi_p_in, double* Fxx_p_in, double* Fyy_p_in, double* Fzz_p_in, interpolation_method method_in)
  {
    Fi_p = Fi_p_in; Fxx_p = Fxx_p_in; Fyy_p = Fyy_p_in; Fzz_p = Fzz_p_in;
    method = method_in; is_input_in_vec = false;
  }
#else
  void set_input(double* Fi_p_in, double* Fxx_p_in, double* Fyy_p_in, interpolation_method method_in)
  {
    Fi_p = Fi_p_in; Fxx_p = Fxx_p_in; Fyy_p = Fyy_p_in;
    method = method_in; is_input_in_vec = false;
  }
#endif

  // interpolation method
#ifdef P4_TO_P8
  double interpolate(double x, double y, double z);
#else
  double interpolate(double x, double y);
#endif

  void set_eps(double eps_in) {eps = eps_in; interp.set_eps(eps_in);}
};

#endif /* MY_P4EST_INTERPOLATION_NODES_LOCAL */
