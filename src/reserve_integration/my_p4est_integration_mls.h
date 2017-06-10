#ifndef MY_P4EST_INTEGRATION_MLS_H
#define MY_P4EST_INTEGRATION_MLS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include "cube3_mls.h"
#include "cube3_mls_quadratic.h"
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_utils.h>
#include "cube2_mls.h"
#include "cube2_mls_quadratic.h"
#endif

#include <src/petsc_logging.h>
#include <src/petsc_compatibility.h>
#include <petsc.h>
#include <vector>

class my_p4est_integration_mls_t
{
  std::vector< quadrant_interp_t > phi_interp_;
public:
  enum int_type_t {DOM, FC1, FC2, FC3};

  p4est_t       *p4est;
  p4est_nodes_t *nodes;

  std::vector<Vec>      *phi;
  std::vector<Vec>      *phi_xx, *phi_yy;
#ifdef P4_TO_P8
  std::vector<Vec>      *phi_zz;
#endif

  std::vector<int>      *color;
  std::vector<action_t> *action;

  bool initialized;
  bool linear_integration;

#ifdef P4_TO_P8
  std::vector<cube3_mls_t> cubes_linear;
  std::vector<cube3_mls_quadratic_t> cubes_quadratic;
#else
  std::vector<cube2_mls_t> cubes_linear;
  std::vector<cube2_mls_quadratic_t> cubes_quadratic;
#endif


  my_p4est_integration_mls_t(p4est_t *p4est_, p4est_nodes_t *nodes_)
    : p4est(NULL), nodes(NULL), phi(NULL), color(NULL), action(NULL), initialized(false), linear_integration(true)
  {p4est = p4est_; nodes = nodes_;}

  void initialize();

  void set_p4est  (p4est_t *p4est_, p4est_nodes_t *nodes_) {p4est = p4est_; nodes = nodes_;}

#ifdef P4_TO_P8
  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = NULL; phi_yy = NULL; phi_zz = NULL;
    action  = &acn_;
    color   = &clr_;
    linear_integration = true;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<Vec> &phi_xx_,
                   std::vector<Vec> &phi_yy_,
                   std::vector<Vec> &phi_zz_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_; phi_zz = &phi_zz_;
    action  = &acn_;
    color   = &clr_;
    linear_integration = false;
  }
#else
  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = NULL; phi_yy = NULL;
    action  = &acn_;
    color   = &clr_;
    linear_integration = true;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<Vec> &phi_xx_,
                   std::vector<Vec> &phi_yy_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_;
    action  = &acn_;
    color   = &clr_;
    linear_integration = false;
  }
#endif

  double perform(int_type_t int_type, int n0 = -1, int n1 = -1, int n2 = -1);

#ifdef P4_TO_P8
  CF_3 *f_cf;
#else
  CF_2 *f_cf;
#endif
  Vec  f;
  Vec *fdd;

  enum { integrand_type_cf, integrand_type_vec, integrand_type_vec_dd } integrand_type;

  inline void set_input(CF_3 *f_)             { f_cf = &f_;         integrand_type = integrand_type_cf;     }
  inline void set_input(Vec   f_)             { f = f_;             integrand_type = integrand_type_vec;    }
  inline void set_input(Vec   f_, Vec *fdd_)  { f = f_; fdd = fdd_; integrand_type = integrand_type_vec_dd; }

  // linear integration
  double integrate_over_domain        (                      ) {return perform(DOM,-1,-1,-1);}
  double integrate_over_interface     (int n0,               ) {return perform(FC1,n0,-1,-1);}
  double integrate_over_intersection  (int n0, int n1,       ) {return perform(FC2,n0,n1,-1);}
#ifdef P4_TO_P8
  double integrate_over_intersection  (int n0, int n1, int n2) {return perform(FC3,n0,n1,n2);}
#endif

  double integrate_everywhere         (Vec f);

  double measure_of_domain        ()                {return perform(DOM,-1,-1,-1);}
  double measure_of_interface     (int n0)          {return perform(FC1,n0,-1,-1);}
#ifdef P4_TO_P8
  double measure_of_intersection  (int n0, int n1)  {return perform(FC2,n0,n1,-1);}
#endif
};

#endif // MY_P4EST_INTEGRATION_MLS_H
