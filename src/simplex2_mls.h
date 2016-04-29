#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef SIMPLEX2_MLS_H
#define SIMPLEX2_MLS_H

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>

class simplex2_mls_t
{
public:
  struct vtx2_t // vertex
  {
    /* Structure and properties */
    double  x, y;   // coordinates
    int     c0, c1; // colors
    double  value;  // stored value
    loc_t   loc;    // location

    int     n_vtx0, n_vtx1; // neighbors
    double  ratio;          // placement between nv0 and nv1

#ifdef CASL_THROWS
    int p_edg; // parent edge
#endif

    vtx2_t(double x = 0.0, double y = 0.0)
      : x(x), y(y), c0(-1), c1(-1), value(0.0), loc(INS)
#ifdef CASL_THROWS
      , n_vtx0(-1), n_vtx1(-1), ratio(1.0),
        p_edg(-1)
#endif
    {}

    void set(loc_t loc_, int c0_, int c1_) {loc = loc_; c0 = c0_; c1 = c1_;}
  };

  struct edg2_t // edge
  {
    /* Structure and properties */
    int   vtx0, vtx1; // vertices
    int   c0;         // colors
    bool  is_split;   // has the edge been split
    loc_t loc;        // location
    int   dir;        // to keep track of edges of a cube
                      // (-1 - inside; 0 - m0; 1 - p0; 2 - 0m; 3 - 0p;)
    int   p_lsf;      // # of lsf that created an edge

    /* Child objects */
    int c_vtx01;        // splitting vertex
    int c_edg0, c_edg1; // edges

#ifdef CASL_THROWS
    int type;                 // type of splitting
    int p_edg, p_tri, p_tet;  // parental objects
#endif

    edg2_t(int v0 = -1, int v1 = -1)
      : vtx0(v0), vtx1(v1), c0(-1), is_split(false), loc(INS), dir(-1), p_lsf(-1)
#ifdef CASL_THROWS
      , c_vtx01(-1), c_edg0(-1), c_edg1(-1),
        type(-1), p_edg(-1), p_tri(-1), p_tet(-1)
#endif
    {}

    void set(loc_t loc_, int c0_) {loc = loc_; c0 = c0_;}
  };

  struct tri2_t  // triangle
  {
    /* Structure and properties */
    int   vtx0, vtx1, vtx2; // vertices
    int   edg0, edg1, edg2; // edges
    loc_t loc;              // location
    bool  is_split;         // has the triangle been split

    /* Child objects */
    int c_vtx01, c_vtx02, c_vtx12;  // vertices
    int c_edg0,  c_edg1;            // edges
    int c_tri0,  c_tri1,  c_tri2;   // triangles

#ifdef CASL_THROWS
    int type;
    int p_tri, p_tet;
#endif

    tri2_t(int v0 = -1, int v1 = -1, int v2 = -1,
           int e0 = -1, int e1 = -1, int e2 = -1)
      : vtx0(v0), vtx1(v1), vtx2(v2),
        edg0(e0), edg1(e1), edg2(e2),
        loc(INS), is_split(false)
#ifdef CASL_THROWS
      , c_vtx01(-1), c_vtx02(-1), c_vtx12(-1),
        c_edg0(-1), c_edg1(-1),
        c_tri0(-1), c_tri1(-1), c_tri2(-1),
        type(-1), p_tri(-1), p_tet(-1)
#endif
    {}
    void set(loc_t loc_) {loc = loc_;}
  };

  simplex2_mls_t();
  simplex2_mls_t(double x0, double y0,
                 double x1, double y1,
                 double x2, double y2);

//  double x[3], y[3];

  std::vector<vtx2_t> vtxs;
  std::vector<edg2_t> edgs;
  std::vector<tri2_t> tris;

  std::vector<double> *phi;
  std::vector<double> *phi_x;
  std::vector<double> *phi_y;

  void do_action(std::vector<double> *phi_, std::vector<double> *phi_x_, std::vector<double> *phi_y_, int cn, action_t action);
  void do_action_vtx(int n_vtx, int cn, action_t action);
  void do_action_edg(int n_edg, int cn, action_t action);
  void do_action_tri(int n_tri, int cn, action_t action);

  bool need_swap(int v0, int v1);

  void interpolate_all(double &p0, double &p1, double &p2);
  void interpolate_from_neighbors(int v);

  double integrate_over_domain            (double f0, double f1, double f2);
  double integrate_over_interface         (double f0, double f1, double f2, int num0);
  double integrate_over_colored_interface (double f0, double f1, double f2, int num0, int num1);
  double integrate_over_intersection      (double f0, double f1, double f2, int num0, int num1);
  double integrate_in_dir                 (double f0, double f1, double f2, int dir);
  double integrate_in_non_cart_dir        (double f0, double f1, double f2, int dir);

  double length (int vtx0, int vtx1);
  double area   (int vtx0, int vtx1, int vtx2);

  bool use_linear;
  double find_intersection_linear   (int v0, int v1);
  double find_intersection_quadratic(int v0, int v1);

  template<typename X>
  void swap(X &x, X &y)
  {
    X tmp;
    tmp = x; x = y; y = tmp;
  }

  void perturb(double &f, double epsilon){
    if(fabs(f) < epsilon){
      if(f >= 0) f =  epsilon;
      else      f = -epsilon;
    }
  }

#ifdef CASL_THROWS
  bool tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2);
  bool tri_is_ok(int t);
#endif

};

#endif // SIMPLEX2_MLS_H
