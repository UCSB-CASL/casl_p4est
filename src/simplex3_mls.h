#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef SIMPLEX3_MLS_H
#define SIMPLEX3_MLS_H

//#include <src/point3.h>
#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>
//#include <string>
//#include <fstream>
//#include <tools/plotting.h>
//#include <src/my_p4est_utils.h>
//#include <src/mls_types.h>

class simplex3_mls_t
{
public:
  double eps;

  struct vtx3_t // vertex
  {
    /* Structure and properties */
    double  x, y, z;    // coordinates
    int     c0, c1, c2; // colors
    double  value;      // stored value
    loc_t   loc;        // location

    int     n_vtx0, n_vtx1; // neighbors
    double  ratio;    // placement between nv0 and nv1

#ifdef CASL_THROWS
    int p_edg; // parent edge
#endif

    vtx3_t(double x = 0.0, double y = 0.0, double z = 0.0)
      : x(x), y(y), z(z), c0(-1), c1(-1), c2(-1), value(0.0), loc(INS)
#ifdef CASL_THROWS
      , n_vtx0(-1), n_vtx1(-1), ratio(1.0),
        p_edg(-1)
#endif
    {}

    void set(loc_t loc_, int c0_, int c1_, int c2_) {loc = loc_; c0 = c0_; c1 = c1_; c2 = c2_;}
  };

  struct edg3_t // edge
  {
    /* Structure and properties */
    int   vtx0, vtx1; // vertices
    int   c0, c1;     // colors
    bool  is_split;   // has the edge been split
    loc_t loc;        // location
    double value;    // stored value at midpoint

    /* Child objects */
    int c_vtx01;        // splitting vertex
    int c_edg0, c_edg1; // edges

#ifdef CASL_THROWS
    int type;                 // type of splitting
    int p_edg, p_tri, p_tet;  // parental objects
#endif

    edg3_t(int v0 = -1, int v1 = -1)
      : vtx0(v0), vtx1(v1), c0(-1), c1(-1), is_split(false), loc(INS)
#ifdef CASL_THROWS
      , c_vtx01(-1), c_edg0(-1), c_edg1(-1),
        type(-1), p_edg(-1), p_tri(-1), p_tet(-1)
#endif
    {}

    void set(loc_t loc_, int c0_, int c1_) {loc = loc_; c0 = c0_; c1 = c1_;}
  };

  struct tri3_t  // triangle
  {
    /* Structure and properties */
    int   vtx0, vtx1, vtx2; // vertices
    int   edg0, edg1, edg2; // edges
    int   c;                // color
    loc_t loc;              // location
    bool  is_split;         // has the triangle been split
    int   dir;              // to keep track of faces of a cube
    int   p_lsf;            // parent level-set function

    /* Child objects */
    int c_vtx01, c_vtx02, c_vtx12;  // vertices
    int c_edg0,  c_edg1;            // edges
    int c_tri0,  c_tri1,  c_tri2;   // triangles

#ifdef CASL_THROWS
    int type;
    int p_tri, p_tet;
#endif

    tri3_t(int v0 = -1, int v1 = -1, int v2 = -1,
           int e0 = -1, int e1 = -1, int e2 = -1)
      : vtx0(v0), vtx1(v1), vtx2(v2),
        edg0(e0), edg1(e1), edg2(e2),
        c(-1), loc(INS), is_split(false), dir(-1), p_lsf(-1)
#ifdef CASL_THROWS
      , c_vtx01(-1), c_vtx02(-1), c_vtx12(-1),
        c_edg0(-1), c_edg1(-1),
        c_tri0(-1), c_tri1(-1), c_tri2(-1),
        type(-1), p_tri(-1), p_tet(-1)
#endif
    {}
    void set(loc_t loc_, int c_) {loc = loc_; c = c_;}
  };

  struct tet3_t // tetrahedron
  {
    /* Structure and properties */
    int   vtx0, vtx1, vtx2, vtx3; // vertices
    int   tri0, tri1, tri2, tri3; // triangles
    loc_t loc;                    // location
    bool  is_split;               // has the tetrahedron been split

    /* Child objects */
    int c_vtx01, c_vtx02, c_vtx03, c_vtx12, c_vtx13, c_vtx23; // up to 6 splitting vertices
    int c_edg;                                                // there might be an additional edge
    int c_tri0, c_tri1, c_tri2, c_tri3, c_tri4, c_tri5;       // up to 6 child triangles
    int c_tet0, c_tet1, c_tet2, c_tet3, c_tet4, c_tet5;       // up to 6 child tetrahedra

#ifdef CASL_THROWS
    int type;
    int p_tet;
#endif

    tet3_t(int v0 = -1, int v1 = -1, int v2 = -1, int v3 = -1,
           int t0 = -1, int t1 = -1, int t2 = -1, int t3 = -1)
      : vtx0(v0), vtx1(v1), vtx2(v2), vtx3(v3),
        tri0(t0), tri1(t1), tri2(t2), tri3(t3),
        loc(INS), is_split(false)
#ifdef CASL_THROWS
      , c_vtx01(-1), c_vtx02(-1), c_vtx03(-1), c_vtx12(-1), c_vtx13(-1), c_vtx23(-1),
        c_tri0(-1), c_tri1(-1), c_tri2(-1), c_tri3(-1), c_tri4(-1), c_tri5(-1),
        c_tet0(-1), c_tet1(-1), c_tet2(-1), c_tet3(-1), c_tet4(-1), c_tet5(-1),
        type(-1), p_tet(-1)
#endif
    {}

    void set(loc_t loc_) {loc = loc_;}
  };

  simplex3_mls_t();
  simplex3_mls_t(double x0, double y0, double z0,
                 double x1, double y1, double z1,
                 double x2, double y2, double z2,
                 double x3, double y3, double z3);

  std::vector<vtx3_t> vtxs;
  std::vector<edg3_t> edgs;
  std::vector<tri3_t> tris;
  std::vector<tet3_t> tets;


  void do_action(int cn, action_t action);
  void do_action_vtx(int n_vtx, int cn, action_t action);
  void do_action_edg(int n_edg, int cn, action_t action);
  void do_action_tri(int n_tri, int cn, action_t action);
  void do_action_tet(int n_tet, int cn, action_t action);

  bool need_swap(int v0, int v1);

  void interpolate_all(double &p0, double &p1, double &p2, double &p3);
  void interpolate_from_neighbors(int v);
  void interpolate_from_parent(int v);
  void interpolate_from_parent(vtx3_t &v);

  double integrate_over_domain            (double f0, double f1, double f2, double f3);
  double integrate_over_interface         (double f0, double f1, double f2, double f3, int num0);
  double integrate_over_colored_interface (double f0, double f1, double f2, double f3, int num0, int num1);
  double integrate_over_intersection      (double f0, double f1, double f2, double f3, int num0, int num1);
  double integrate_over_intersection      (double f0, double f1, double f2, double f3, int num0, int num1, int num2);
  double integrate_in_dir                 (double f0, double f1, double f2, double f3, int dir);

  double length (int vtx0, int vtx1);
  double area   (int vtx0, int vtx1, int vtx2);
  double volume (int vtx0, int vtx1, int vtx2, int vtx3);
  double volume(vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3);

  bool use_linear;
  double find_intersection_linear   (int v0, int v1);
  double find_intersection_quadratic(int e);

  void get_edge_coords(int e, double xyz[]);

  template<typename X>
  void swap(X &x, X &y)
  {
    X tmp;
    tmp = x; x = y; y = tmp;
  }

  void perturb(double &f, double epsilon){
    if(fabs(f) < epsilon){
      if(f > 0) f =  epsilon;
      else      f = -epsilon;
    }
  }

  void set_use_linear(bool val) { use_linear = val; }

#ifdef CASL_THROWS
  bool tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2);
  bool tri_is_ok(int t);
  bool tet_is_ok(int s);
#endif

};

#endif // SIMPLEX3_MLS_H
