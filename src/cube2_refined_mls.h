#ifndef CUBE2_REFINED_MLS_H
#define CUBE2_REFINED_MLS_H

#include "cube2_mls.h"
#include "vector"
#include "grid_interpolation2.h"
#include "my_p4est_utils.h"

class cube2_refined_mls_t
{
  struct node_t {
    double  x, y;

    node_t(double x, double y) : x(x), y(y) {}
  };

  struct edge_t {
    int   v0, v1;
    bool  is_split;
    int   e0, e1, v01;

    edge_t(int v0, int v1) : v0(v0), v1(v1), is_split(false), e0(-1), e1(-1), v01(-1) {}
  };

  struct cube_t {
    bool  is_split;
    bool  wall[4];
    int e_m0, e_p0, e_0m, e_0p;

    cube_t(int e_m0_ = -1, int e_p0_ = -1, int e_0m_ = -1, int e_0p_ = -1)
    {
      e_m0 = e_m0_;
      e_p0 = e_p0_;
      e_0m = e_0m_;
      e_0p = e_0p_;
      is_split = false;
      wall[0] = false;
      wall[1] = false;
      wall[2] = false;
      wall[3] = false;
    }
  };

public:
  static const int N_CHILDREN=4;
  double  x0, x1, y0, y1;
  loc_t   loc;
  int     n_cubes, n_nodes, n_leafs, nx, ny;

  double dx_min, dy_min;

  // LSF by a function
  std::vector<CF_2 *> *phi_cf_in;

  // LSF by array of values
  std::vector< std::vector<double> > *phi_in;
  std::vector< std::vector<double> > *phi_xx_in;
  std::vector< std::vector<double> > *phi_yy_in;
  grid_interpolation2_t interp;

  // quadtree structure
  std::vector<node_t> nodes;
  std::vector<edge_t> edges;
  std::vector<cube_t> cubes;

  std::vector<int> leaf_to_node;
  std::vector<int> get_cube;

  // LSF related stuff
  std::vector<action_t> *action;
  std::vector<int>      *color;

  std::vector< std::vector<double> > phi;
  std::vector< std::vector<double> > phi_xx;
  std::vector< std::vector<double> > phi_yy;

  // array
  std::vector<cube2_mls_t> cubes_mls;

  cube2_refined_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), phi_cf_in(NULL) {set_interpolation_grid(x0, x1, y0, y1, 1, 1);}

  // set LSF by function
//  void set_phi_cf (std::vector<CF_2 *> &phi_cf_) {phi_cf_in = &phi_cf_;}

  // set interpolation parameters
  void set_interpolation_grid(double xm, double xp, double ym, double yp, int nx, int ny)
  {
    interp.initialize(xm, xp, ym, yp, nx, ny);
  }
//  void set_phi    (std::vector< std::vector<double> > &phi_)    {phi_in = &phi_;}
//  void set_phi_dd (std::vector< std::vector<double> > &phi_xx_,
//                   std::vector< std::vector<double> > &phi_yy_) {phi_xx_in = &phi_xx_;
//                                                                 phi_yy_in = &phi_yy_;}

//  // action and color
//  void set_action (std::vector<action_t>  &action_) {action = &action_;}
//  void set_color  (std::vector<int>       &color_)  {color = &color_;}

  void set_phi(std::vector<CF_2 *> &phi_cf_,
               std::vector<action_t> &action_,
               std::vector<int> &color_)
  {
    phi_cf_in = &phi_cf_;
    phi_xx_in = NULL;
    phi_yy_in = NULL;
    action    = &action_;
    color     = &color_;
  }

  void set_phi(std::vector< std::vector<double> > &phi_,
               std::vector<action_t> &action_,
               std::vector<int> &color_)
  {
    phi_in    = &phi_;
    phi_xx_in = NULL;
    phi_yy_in = NULL;
    action    = &action_;
    color     = &color_;
  }

  void set_phi(std::vector< std::vector<double> > &phi_,
               std::vector< std::vector<double> > &phi_xx_,
               std::vector< std::vector<double> > &phi_yy_,
               std::vector<action_t> &action_,
               std::vector<int> &color_)
  {
    phi_in    = &phi_;
    phi_xx_in = &phi_xx_;
    phi_yy_in = &phi_yy_;
    action    = &action_;
    color     = &color_;
  }

  void construct_domain(int nx_, int ny_, int level);

  void split_edge(int n);
  void split_cube(int n);
  bool need_split(int n);

  void sample_all(std::vector<double> &f, std::vector<double> &f_values);
  void sample_all(CF_2 &f, std::vector<double> &f_values);

  // integration tools
  double perform(std::vector<double> &f, int type, int num0, int num1);
  double perform(CF_2 &f, int type, int num0, int num1);
  double perform(double f, int type, int num0, int num1);

  double integrate_over_domain            (std::vector<double> &f)                  {return perform(f,0,-1,-1);}
  double integrate_over_interface         (std::vector<double> &f, int n0)          {return perform(f,1,n0,-1);}
  double integrate_over_intersection      (std::vector<double> &f, int n0, int n1)  {return perform(f,2,n0,n1);}
  double integrate_over_colored_interface (std::vector<double> &f, int n0, int n1)  {return perform(f,3,n0,n1);}
  double integrate_in_non_cart_dir        (std::vector<double> &f, int n0)          {return perform(f,4,n0,-1);}

  double integrate_over_domain            (CF_2 &f)                  {return perform(f,0,-1,-1);}
  double integrate_over_interface         (CF_2 &f, int n0)          {return perform(f,1,n0,-1);}
  double integrate_over_intersection      (CF_2 &f, int n0, int n1)  {return perform(f,2,n0,n1);}
  double integrate_over_colored_interface (CF_2 &f, int n0, int n1)  {return perform(f,3,n0,n1);}
  double integrate_in_non_cart_dir        (CF_2 &f, int n0)          {return perform(f,4,n0,-1);}

  double measure_of_domain            ()                {return perform(1.,0,-1,-1);}
  double measure_of_interface         (int n0)          {return perform(1.,1,n0,-1);}
  double measure_of_intersection      (int n0, int n1)  {return perform(1.,2,n0,n1);}
  double measure_of_colored_interface (int n0, int n1)  {return perform(1.,3,n0,n1);}
  double measure_in_non_cart_dir      (int n0)          {return perform(1.,4,n0,-1);}

  double integrate_in_dir(std::vector<double> &f, int dir);
  double integrate_in_dir(CF_2 &f, int dir);
  double integrate_in_dir(double f, int dir);

  double measure_in_dir (int n0) {return integrate_in_dir(1.,n0);}
};

#endif // CUBE2_REFINED_MLS_H
