#ifndef MY_P4EST_INTEGRATION_REFINED_QUAD_MLS_H
#define MY_P4EST_INTEGRATION_REFINED_QUAD_MLS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
//#include <p8est_ghost.h>
#include "cube3_mls.h"
#else
#include <p4est.h>
#include <p4est_nodes.h>
//#include <p4est_ghost.h>
#include "cube2_refined_quad_mls.h"
#endif

#include <src/petsc_logging.h>
#include <src/petsc_compatibility.h>
#include <petsc.h>
//#include <stdexcept>
//#include <sstream>
#include <vector>
#include <src/casl_types.h>

class my_p4est_integration_refined_quad_mls_t
{
public:
  enum int_type_t {DOM, FC1, FC2, FC3};

  p4est_t       *p4est;
  p4est_nodes_t *nodes;

  std::vector<Vec>      *phi, *phixx, *phixy, *phiyy;
  std::vector<int>      *color;
  std::vector<action_t> *action;

  std::vector<CF_2 *>   *phi_cf;
  std::vector<CF_2 *>   *phix_cf, *phiy_cf;
  std::vector<CF_2 *>   *phixx_cf, *phixy_cf, *phiyy_cf;

  bool initialized;

  int nx, ny, n_cubes, n_nodes;

#ifdef P4_TO_P8
  std::vector<cube3_mls_t> cubes;
#else
  std::vector<cube2_refined_quad_mls_t> cubes;
#endif


  my_p4est_integration_refined_quad_mls_t()
    : p4est(NULL), nodes(NULL), phi(NULL), color(NULL), action(NULL), initialized(false), nx(1), ny(1), n_cubes(1), n_nodes(4)
  {}

  void initialize();

  void set_p4est  (p4est_t *p4est_, p4est_nodes_t *nodes_)                                {p4est = p4est_; nodes = nodes_;}
  void set_phi    (std::vector<Vec> &phi_, std::vector<Vec> &phixx_, std::vector<Vec> &phixy_, std::vector<Vec> &phiyy_,
                   std::vector<action_t> &action_, std::vector<int> &color_)
  {
    phi = &phi_; phixx = &phixx_; phixy = &phixy_; phiyy = &phiyy_; action = &action_; color = &color_;
  }

  void set_phi    (std::vector<CF_2 *> &phi_, std::vector<CF_2 *> &phixx_, std::vector<CF_2 *> &phixy_, std::vector<CF_2 *> &phiyy_,
                   std::vector<action_t> &action_, std::vector<int> &color_)
  {
    phi_cf = &phi_; phixx_cf = &phixx_; phixy_cf = &phixy_; phiyy_cf = &phiyy_; action = &action_; color = &color_;
  }

  void set_phi    (std::vector<CF_2 *> &phi_,
                   std::vector<CF_2 *> &phixx_, std::vector<CF_2 *> &phixy_, std::vector<CF_2 *> &phiyy_,
                   std::vector<CF_2 *> &phix_, std::vector<CF_2 *> &phiy_,
                   std::vector<action_t> &action_, std::vector<int> &color_)
  {
    phi_cf = &phi_;
    phixx_cf = &phixx_; phixy_cf = &phixy_; phiyy_cf = &phiyy_;
    phix_cf = &phix_; phiy_cf = &phiy_;
    action = &action_; color = &color_;
  }

  void set_refinement(int nx_, int ny_) {nx = nx_; ny = ny_; n_cubes = nx*ny; n_nodes = (nx+1)*(ny+1);}

  double perform(int_type_t int_type, CF_2 *f = NULL, int n0 = -1, int n1 = -1, int n2 = -1);

  double integrate_over_domain        (CF_2 *f)                         {return perform(DOM,f,-1,-1,-1);}
  double integrate_over_interface     (CF_2 *f, int n0)                 {return perform(FC1,f,n0,-1,-1);}
  double integrate_over_intersection  (CF_2 *f, int n0, int n1)         {return perform(FC2,f,n0,n1,-1);}
  #ifdef P4_TO_P8
  double integrate_over_intersection  (Vec f, int n0, int n1, int n2) {return perform(FC3,f,n0,n1,n2);}
  #endif

  double measure_of_domain        ()                {return perform(DOM,NULL,-1,-1,-1);}
  double measure_of_interface     (int n0)          {return perform(FC1,NULL,n0,-1,-1);}
#ifdef P4_TO_P8
  double measure_of_intersection  (int n0, int n1)  {return perform(FC2,NULL,n0,n1,-1);}
#endif
};

#endif // MY_P4EST_INTEGRATION_REFINED_QUAD_MLS_H
