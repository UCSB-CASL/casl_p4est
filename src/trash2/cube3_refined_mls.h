#ifndef CUBE2_REFINED_MLS_H
#define CUBE2_REFINED_MLS_H

#include "cube3_mls.h"
#include "vector"
#include "grid_interpolation3.h"
#include "my_p4est_utils.h"

class cube3_refined_mls_t
{
public:
  static const int N_CHILDREN = 8;
  static const int N_FACES = 6;

  double  x0, x1, y0, y1, z0, z1;
  loc_t   loc;
  int     n_cubes, n_nodes, n_leafs, nx, ny, nz;

  double dx_min, dy_min, dz_min;

  // LSF by a function
  std::vector<CF_3 *> *phi_cf_in;

  // LSF by array of values
  std::vector< std::vector<double> > *phi_in;
  std::vector< std::vector<double> > *phi_xx_in;
  std::vector< std::vector<double> > *phi_yy_in;
  std::vector< std::vector<double> > *phi_zz_in;
  grid_interpolation3_t interp;

  std::vector<int> leaf_to_node;
  std::vector<int> get_cube;

  // LSF related stuff
  std::vector<action_t> *action;
  std::vector<int>      *color;

  std::vector< std::vector<double> > phi;
  std::vector< std::vector<double> > phi_xx;
  std::vector< std::vector<double> > phi_yy;
  std::vector< std::vector<double> > phi_zz;

  std::vector<double> x_coord;
  std::vector<double> y_coord;
  std::vector<double> z_coord;

  // array
  std::vector<cube3_mls_t> cubes_mls;

  cube3_refined_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1., double z0 = 0., double z1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1), phi_cf_in(NULL)
  {set_interpolation_grid(x0, x1, y0, y1, z0, z1, 1, 1, 1);}

  // set LSF by function
//  void set_phi_cf (std::vector<CF_3 *> &phi_cf_) {phi_cf_in = &phi_cf_;}

  // set interpolation parameters
  void set_interpolation_grid(double xm, double xp, double ym, double yp, double zm, double zp, int nx, int ny, int nz)
  {
    interp.initialize(xm, xp, ym, yp, zm, zp, nx, ny, nz);
  }
//  void set_phi    (std::vector< std::vector<double> > &phi_)    {phi_in = &phi_;}
//  void set_phi_dd (std::vector< std::vector<double> > &phi_xx_,
//                   std::vector< std::vector<double> > &phi_yy_,
//                   std::vector< std::vector<double> > &phi_zz_) {phi_xx_in = &phi_xx_;
//                                                                 phi_yy_in = &phi_yy_;
//                                                                 phi_zz_in = &phi_zz_;}

//  // action and color
//  void set_action (std::vector<action_t>  &action_) {action = &action_;}
//  void set_color  (std::vector<int>       &color_)  {color = &color_;}


  void set_phi(std::vector<CF_3 *> &phi_cf_,
               std::vector<action_t> &action_,
               std::vector<int> &color_)
  {
    phi_cf_in = &phi_cf_;
    phi_xx_in = NULL;
    phi_yy_in = NULL;
    phi_zz_in = NULL;
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
    phi_zz_in = NULL;
    action    = &action_;
    color     = &color_;
  }

  void set_phi(std::vector< std::vector<double> > &phi_,
               std::vector< std::vector<double> > &phi_xx_,
               std::vector< std::vector<double> > &phi_yy_,
               std::vector< std::vector<double> > &phi_zz_,
               std::vector<action_t> &action_,
               std::vector<int> &color_)
  {
    phi_in    = &phi_;
    phi_xx_in = &phi_xx_;
    phi_yy_in = &phi_yy_;
    phi_zz_in = &phi_zz_;
    action    = &action_;
    color     = &color_;
  }

  void construct_domain(int nx_, int ny_, int nz_, int level);

  void split_edge(int n);
  void split_face(int n);
  void split_cube(int n);
  bool need_split(int n);

  void sample_all(std::vector<double> &f, std::vector<double> &f_values);
  void sample_all(CF_3 &f, std::vector<double> &f_values);

  // integration tools
  double perform(std::vector<double> &f, int type, int num0, int num1, int num2);
  double perform(CF_3 &f, int type, int num0, int num1, int num2);
  double perform(double f, int type, int num0, int num1, int num2);

  double integrate_over_domain            (std::vector<double> &f)                  {return perform(f,0,-1,-1,-1);}
  double integrate_over_interface         (std::vector<double> &f, int n0)          {return perform(f,1,n0,-1,-1);}
  double integrate_over_intersection      (std::vector<double> &f, int n0, int n1)  {return perform(f,2,n0,n1,-1);}
  double integrate_over_intersection      (std::vector<double> &f, int n0, int n1, int n2)  {return perform(f,3,n0,n1,n2);}
  double integrate_over_colored_interface (std::vector<double> &f, int n0, int n1)  {return perform(f,4,n0,n1,-1);}
//  double integrate_in_non_cart_dir        (std::vector<double> &f, int n0)          {return perform(f,5,n0,-1,-1);}

  double integrate_over_domain            (CF_3 &f)                  {return perform(f,0,-1,-1,-1);}
  double integrate_over_interface         (CF_3 &f, int n0)          {return perform(f,1,n0,-1,-1);}
  double integrate_over_intersection      (CF_3 &f, int n0, int n1)  {return perform(f,2,n0,n1,-1);}
  double integrate_over_intersection      (CF_3 &f, int n0, int n1, int n2)  {return perform(f,3,n0,n1,n2);}
  double integrate_over_colored_interface (CF_3 &f, int n0, int n1)  {return perform(f,4,n0,n1,-1);}
//  double integrate_in_non_cart_dir        (CF_3 &f, int n0)          {return perform(f,5,n0,-1,-1);}

  double measure_of_domain            ()                {return perform(1.,0,-1,-1,-1);}
  double measure_of_interface         (int n0)          {return perform(1.,1,n0,-1,-1);}
  double measure_of_intersection      (int n0, int n1)  {return perform(1.,2,n0,n1,-1);}
  double measure_of_intersection      (int n0, int n1, int n2)  {return perform(1.,2,n0,n1,n2);}
  double measure_of_colored_interface (int n0, int n1)  {return perform(1.,3,n0,n1,-1);}
//  double measure_in_non_cart_dir      (int n0)          {return perform(1.,4,n0,-1,-1);}

  double integrate_in_dir(std::vector<double> &f, int dir);
  double integrate_in_dir(CF_3 &f, int dir);
  double integrate_in_dir(double f, int dir);

  double measure_in_dir (int n0) {return integrate_in_dir(1.,n0);}
};

#endif // CUBE2_REFINED_MLS_H
