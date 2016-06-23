#ifndef MY_P4EST_INTEGRATION_MLS_H
#define MY_P4EST_INTEGRATION_MLS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
//#include <p8est_ghost.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
//#include <p4est_ghost.h>
#endif

#include "cube3_mls.h"
#include "cube3_refined_mls.h"
#include "cube2_mls.h"
#include "cube2_refined_mls.h"

#include <src/petsc_logging.h>
#include <src/petsc_compatibility.h>
#include <petsc.h>
//#include <stdexcept>
//#include <sstream>
#include <vector>
//#include <src/casl_types.h>

class my_p4est_integration_mls_t
{
public:
  enum int_type_t {DOM, FC1, FC2, FC3};

  p4est_t       *p4est;
  p4est_nodes_t *nodes;

#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif

  std::vector<Vec>      *phi;
  std::vector<Vec>      *phi_xx, *phi_yy;
#ifdef P4_TO_P8
  std::vector<Vec>      *phi_zz;
#endif

  std::vector<int>      *color;
  std::vector<action_t> *action;

  bool use_cube_refined;
  int level;

  bool initialized;

#ifdef P4_TO_P8
  std::vector<cube3_mls_t> cubes;
  std::vector<cube3_refined_mls_t> cubes_refined;
#else
  std::vector<cube2_mls_t> cubes;
  std::vector<cube2_refined_mls_t> cubes_refined;
#endif


  my_p4est_integration_mls_t()
    : p4est(NULL), nodes(NULL), phi(NULL), color(NULL), action(NULL), initialized(false), use_cube_refined(false)
  {}

  void initialize();

  void set_p4est  (p4est_t *p4est_, p4est_nodes_t *nodes_)                                {p4est = p4est_; nodes = nodes_;}

#ifdef P4_TO_P8
  void set_phi    (std::vector<CF_3 *> &phi_cf_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi_cf  = &phi_cf_; phi_xx = NULL; phi_yy = NULL; phi_zz = NULL;
    action  = &acn_;
    color   = &clr_;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = NULL; phi_yy = NULL; phi_zz = NULL; phi_cf  = NULL;
    action  = &acn_;
    color   = &clr_;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<Vec> &phi_xx_,
                   std::vector<Vec> &phi_yy_,
                   std::vector<Vec> &phi_zz_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_; phi_zz = &phi_zz_; phi_cf  = NULL;
    action  = &acn_;
    color   = &clr_;
  }
#else
  void set_phi    (std::vector<CF_2 *> &phi_cf_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi_cf  = &phi_cf_; phi_xx = NULL; phi_yy = NULL;
    action  = &acn_;
    color   = &clr_;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = NULL; phi_yy = NULL; phi_cf = NULL;
    action  = &acn_;
    color   = &clr_;
  }

  void set_phi    (std::vector<Vec> &phi_,
                   std::vector<Vec> &phi_xx_,
                   std::vector<Vec> &phi_yy_,
                   std::vector<action_t> &acn_, std::vector<int> &clr_)
  {
    phi     = &phi_; phi_xx = &phi_xx_; phi_yy = &phi_yy_; phi_cf = NULL;
    action  = &acn_;
    color   = &clr_;
  }
#endif

  void set_use_cube_refined(int level_) {level = level_; use_cube_refined = true;}
  void unset_use_cube_refined() {use_cube_refined = false;}

  double perform(int_type_t int_type, Vec f = NULL, int n0 = -1, int n1 = -1, int n2 = -1);

  double integrate_over_domain        (Vec f)                         {return perform(DOM,f,-1,-1,-1);}
  double integrate_over_interface     (Vec f, int n0)                 {return perform(FC1,f,n0,-1,-1);}
  double integrate_over_intersection  (Vec f, int n0, int n1)         {return perform(FC2,f,n0,n1,-1);}
  #ifdef P4_TO_P8
  double integrate_over_intersection  (Vec f, int n0, int n1, int n2) {return perform(FC3,f,n0,n1,n2);}
  #endif

  double measure_of_domain        ()                {return perform(DOM,NULL,-1,-1,-1);}
  double measure_of_interface     (int n0)          {return perform(FC1,NULL,n0,-1,-1);}
#ifdef P4_TO_P8
  double measure_of_intersection  (int n0, int n1)  {return perform(FC2,NULL,n0,n1,-1);}
#endif
};

#endif // MY_P4EST_INTEGRATION_MLS_H
