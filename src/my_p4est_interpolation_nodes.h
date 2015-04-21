#ifndef MY_P4EST_INTERPOLATION_NODES
#define MY_P4EST_INTERPOLATION_NODES

#ifdef P4_TO_P8
#include <src/my_p4est_nodes.h>
#include <src/my_p8est_interpolation.h>
#else
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_interpolation.h>
#endif

class my_p4est_interpolation_nodes_t : public my_p4est_interpolation_t
{
private:
  p4est_nodes_t *nodes;
  Vec Fxx, Fyy;
#ifdef P4_TO_P8
  Vec Fzz;
#endif

  interpolation_method method;

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_nodes_t(const my_p4est_interpolation_nodes_t& other);
  my_p4est_interpolation_nodes_t& operator=(const my_p4est_interpolation_nodes_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n);

  void set_input(Vec F, interpolation_method method);

#ifdef P4_TO_P8
  void set_input(Vec F, Vec Fxx, Vec Fyy, Vec Fzz, interpolation_method method);
#else
  void set_input(Vec F, Vec Fxx, Vec Fyy, interpolation_method method);
#endif

  // interpolation methods
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif

  double interpolate(const p4est_quadrant_t &quad, const double *xyz) const;
};

#endif /* MY_P4EST_INTERPOLATION_NODES */
