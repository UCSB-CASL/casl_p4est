#ifndef CUBE2_REFINED_MLS_H
#define CUBE2_REFINED_MLS_H

#include "cube3_mls.h"
#include "vector"
#include "grid_interpolation3.h"
#include "my_p4est_utils.h"

//#define N_CHILDREN 8


class cube3_refined_mls_t
{
  struct node_t {
    double  x, y, z;

    node_t(double x, double y, double z) : x(x), y(y), z(z) {}
  };

  struct edge_t {
    int   v0, v1;
    bool  is_split;
    int   e0, e1, v01;

    edge_t(int v0, int v1) : v0(v0), v1(v1), is_split(false), e0(-1), e1(-1), v01(-1) {}
  };

  struct face_t {
    int e_m0, e_p0, e_0m, e_0p;
    int f_mm, f_pm, f_mp, f_pp;
    int v_00;
    int ce_m0, ce_p0, ce_0m, ce_0p;
    bool is_split;
    int level;

    face_t(int e_m0_ = -1, int e_p0_ = -1, int e_0m_ = -1, int e_0p_ = -1)
      : e_m0(e_m0_), e_p0(e_p0_), e_0m(e_0m_), e_0p(e_0p_), is_split(false),
        f_mm(-1), f_pm(-1), f_mp(-1), f_pp(-1),
        v_00(-1), ce_m0(-1), ce_p0(-1), ce_0m(-1), ce_0p(-1),
        level(0) {}
  };

  struct cube_t {
    bool  is_split;
    bool  wall[6];
    int f_m00, f_p00, f_0m0, f_0p0, f_00m, f_00p;
    int level;

    cube_t(int f_m00_ = -1, int f_p00_ = -1, int f_0m0_ = -1, int f_0p0_ = -1, int f_00m_ = -1, int f_00p_ = -1)
    {
      f_m00 = f_m00_;
      f_p00 = f_p00_;
      f_0m0 = f_0m0_;
      f_0p0 = f_0p0_;
      f_00m = f_00m_;
      f_00p = f_00p_;
      level = 0;
      is_split = false;
      for (int i = 0; i < N_FACES; i++) wall[i] = false;
    }
  };


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
  std::vector< std::vector<double> > *phi_x_in;
  std::vector< std::vector<double> > *phi_y_in;
  std::vector< std::vector<double> > *phi_z_in;
  std::vector< std::vector<double> > *phi_xx_in;
  std::vector< std::vector<double> > *phi_yy_in;
  std::vector< std::vector<double> > *phi_zz_in;
  grid_interpolation3_t interp;

  // quadtree structure
  std::vector<node_t> nodes;
  std::vector<edge_t> edges;
  std::vector<face_t> faces;
  std::vector<cube_t> cubes;

  std::vector<int> leaf_to_node;
  std::vector<int> get_cube;

  // LSF related stuff
  std::vector<action_t> *action;
  std::vector<int>      *color;

  std::vector< std::vector<double> > phi;
  std::vector< std::vector<double> > phi_x;
  std::vector< std::vector<double> > phi_y;
  std::vector< std::vector<double> > phi_z;
  std::vector< std::vector<double> > phi_xx;
  std::vector< std::vector<double> > phi_yy;
  std::vector< std::vector<double> > phi_zz;

  // array
  std::vector<cube3_mls_t> cubes_mls;

  cube3_refined_mls_t(double x0 = 0., double x1 = 1., double y0 = 0., double y1 = 1., double z0 = 0., double z1 = 1.)
    : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1), phi_cf_in(NULL)
  {set_interpolation_grid(x0, x1, y0, y1, z0, z1, 1, 1, 1);}

  // set LSF by function
  void set_phi_cf (std::vector<CF_3 *> &phi_cf_) {phi_cf_in = &phi_cf_;}

  // set interpolation parameters
  void set_interpolation_grid(double xm, double xp, double ym, double yp, double zm, double zp, int nx, int ny, int nz)
  {
    interp.initialize(xm, xp, ym, yp, zm, zp, nx, ny, nz);
  }
  void set_phi    (std::vector< std::vector<double> > &phi_)    {phi_in = &phi_;}
  void set_phi_d  (std::vector< std::vector<double> > &phi_x_,
                   std::vector< std::vector<double> > &phi_y_,
                   std::vector< std::vector<double> > &phi_z_)  {phi_x_in = &phi_x_;
                                                                 phi_y_in = &phi_y_;
                                                                 phi_z_in = &phi_z_;}
  void set_phi_dd (std::vector< std::vector<double> > &phi_xx_,
                   std::vector< std::vector<double> > &phi_yy_,
                   std::vector< std::vector<double> > &phi_zz_) {phi_xx_in = &phi_xx_;
                                                                 phi_yy_in = &phi_yy_;
                                                                 phi_zz_in = &phi_zz_;}

  // action and color
  void set_action (std::vector<action_t>  &action_) {action = &action_;}
  void set_color  (std::vector<int>       &color_)  {color = &color_;}

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
