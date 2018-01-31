#ifndef MY_P4EST_INTEGRATION_MLS_H
#define MY_P4EST_INTEGRATION_MLS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <src/my_p8est_utils.h>
#include <src/mls_integration/cube3_mls.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/mls_integration/cube2_mls.h>
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

#ifdef P4_TO_P8
  std::vector<CF_3 *> *phi_cf;
#else
  std::vector<CF_2 *> *phi_cf;
#endif

  std::vector<int>      *color;
  std::vector<action_t> *action;

  bool initialized;
  bool linear_integration;

#ifdef P4_TO_P8
  std::vector<cube3_mls_t> cubes;
#else
  std::vector<cube2_mls_t> cubes;
#endif

//#ifdef P4_TO_P8
//  std::vector<cube3_mls_l_t> cubes_linear;
//  std::vector<cube3_mls_q_t> cubes_quadratic;
//#else
//  std::vector<cube2_mls_l_t> cubes_linear;
//  std::vector<cube2_mls_q_t> cubes_quadratic;
//#endif


  my_p4est_integration_mls_t(p4est_t *p4est_, p4est_nodes_t *nodes_)
    : p4est(NULL), nodes(NULL), phi(NULL), phi_cf(NULL), color(NULL), action(NULL), initialized(false), linear_integration(true)
  {p4est = p4est_; nodes = nodes_;}

  void initialize();

  void set_p4est  (p4est_t *p4est_, p4est_nodes_t *nodes_) {p4est = p4est_; nodes = nodes_;}

#ifdef P4_TO_P8
  void set_phi    (std::vector<CF_3 *> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_, bool linear)
  {
    phi_cf  = &phi_;
    phi     = NULL; phi_xx = NULL; phi_yy = NULL;
    action  = &acn_;
    color   = &clr_;
    linear_integration = linear;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi_cf  = NULL;
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
    phi_cf  = NULL;
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_; phi_zz = &phi_zz_;
    action  = &acn_;
    color   = &clr_;
    linear_integration = false;
  }
#else
  void set_phi    (std::vector<CF_2 *> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_, bool linear)
  {
    phi_cf  = &phi_;
    phi     = NULL; phi_xx = NULL; phi_yy = NULL;
    action  = &acn_;
    color   = &clr_;
    linear_integration = linear;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi_cf  = NULL;
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
    phi_cf  = NULL;
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_;
    action  = &acn_;
    color   = &clr_;
    linear_integration = false;
  }
#endif

  double perform(int_type_t int_type, int n0 = -1, int n1 = -1, int n2 = -1, Vec f = NULL, Vec *fdd = NULL);

  // linear integration
  double integrate_over_domain        (                         Vec f, Vec *fdd = NULL) {return perform(DOM,-1,-1,-1,f,fdd);}
  double integrate_over_interface     (int n0,                  Vec f, Vec *fdd = NULL) {return perform(FC1,n0,-1,-1,f,fdd);}
  double integrate_over_intersection  (int n0, int n1,          Vec f, Vec *fdd = NULL) {return perform(FC2,n0,n1,-1,f,fdd);}
#ifdef P4_TO_P8
  double integrate_over_intersection  (int n0, int n1, int n2,  Vec f, Vec *fdd = NULL) {return perform(FC3,n0,n1,n2,f,fdd);}
#endif

  double integrate_everywhere         (Vec f);

  double measure_of_domain        ()                {return perform(DOM,-1,-1,-1,NULL,NULL);}
  double measure_of_interface     (int n0)          {return perform(FC1,n0,-1,-1,NULL,NULL);}
#ifdef P4_TO_P8
  double measure_of_intersection  (int n0, int n1)  {return perform(FC2,n0,n1,-1,NULL,NULL);}
#endif
};

#endif // MY_P4EST_INTEGRATION_MLS_H
