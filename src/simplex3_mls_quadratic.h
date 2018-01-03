#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef SIMPLEX3_MLS_QUADRATIC_H
#define SIMPLEX3_MLS_QUADRATIC_H

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_utils.h>
#endif

class simplex3_mls_quadratic_t
{
public:
  double eps;

  bool not_pure;

  const static int nodes_per_tri = 6;
  const static int nodes_per_tet = 10;

  const double curvature_limit_ = 0.2;

  std::vector<CF_3 *> *phi_;

  double max_dist_error_;

  double diag;

  struct vtx3_t // vertex
  {
    /* Structure and properties */
    double  x, y, z;    // coordinates
    int     c0, c1, c2; // colors
    double  value;      // stored value
    loc_t   loc;        // location

    int     n_vtx0, n_vtx1; // neighbors
    double  ratio;    // placement between nv0 and nv1
    bool    is_recycled;

#ifdef CASL_THROWS
    int p_edg; // parent edge
#endif

    vtx3_t(double x, double y, double z)
      : x(x), y(y), z(z), c0(-1), c1(-1), c2(-1), value(0.0), loc(INS), is_recycled(false)
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
    int   vtx0, vtx1, vtx2; // vertices
    int   c0, c1;     // colors
    bool  is_split;   // has the edge been split
    loc_t loc;        // location
    double value;    // stored value at midpoint

    /* Child objects */
    double a;           // location of the intersection point in reference element
    int c_vtx_x;        // intersection point vertex
    int c_edg0, c_edg1; // edges

#ifdef CASL_THROWS
    int type;                 // type of splitting
    int p_edg, p_tri, p_tet;  // parental objects
#endif

    edg3_t(int v0, int v1, int v2)
      : vtx0(v0), vtx1(v1), vtx2(v2), c0(-1), c1(-1), is_split(false), loc(INS)
#ifdef CASL_THROWS
      , c_vtx_x(-11), c_edg0(-12), c_edg1(-13),
        type(-1), p_edg(-1), p_tri(-1), p_tet(-1)
#endif
    {
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0)
        throw std::domain_error("[CASL_ERROR]: Invalid edge.");
    }

    void set(loc_t loc_, int c0_, int c1_) {
      loc = loc_; c0 = c0_; c1 = c1_;
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0)
        throw std::domain_error("[CASL_ERROR]: Invalid edge.");
    }
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

    /* some stuff for better reconstruction */
    double g_vtx01[3], g_vtx12[3], g_vtx02[3]; // midpoints in normal direction
    double ab01[2], ab12[2], ab02[2];
    bool is_curved;


    /* Child objects */
    int c_vtx01, c_vtx02, c_vtx12;  // vertices
    int c_edg0,  c_edg1,  c_edg2;   // edges
    int c_tri0,  c_tri1,  c_tri2,  c_tri3;   // triangles

#ifdef CASL_THROWS
    int type;
    int p_tri, p_tet;
#endif

    tri3_t(int v0, int v1, int v2,
           int e0, int e1, int e2)
      : vtx0(v0), vtx1(v1), vtx2(v2),
        edg0(e0), edg1(e1), edg2(e2),
        c(-1), loc(INS), is_split(false), dir(-1), p_lsf(-1),
        is_curved(false)
#ifdef CASL_THROWS
      , c_vtx01(-21), c_vtx02(-22), c_vtx12(-23),
        c_edg0(-24), c_edg1(-25), c_edg2(-26),
        c_tri0(-27), c_tri1(-28), c_tri2(-29), c_tri3(-20),
        type(-1), p_tri(-1), p_tet(-1)
#endif
    {
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 ||
          edg0 < 0 || edg1 < 0 || edg2 < 0)
        throw std::domain_error("[CASL_ERROR]: Invalid triangle.");
    }
    void set(loc_t loc_, int c_) {
      loc = loc_; c = c_;
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 ||
          edg0 < 0 || edg1 < 0 || edg2 < 0)
        throw std::domain_error("[CASL_ERROR]: Invalid triangle.");
    }
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

    tet3_t(int v0, int v1, int v2, int v3,
           int t0, int t1, int t2, int t3)
      : vtx0(v0), vtx1(v1), vtx2(v2), vtx3(v3),
        tri0(t0), tri1(t1), tri2(t2), tri3(t3),
        loc(INS), is_split(false)
#ifdef CASL_THROWS
      , c_vtx01(-30), c_vtx02(-31), c_vtx03(-32), c_vtx12(-33), c_vtx13(-34), c_vtx23(-35),
        c_tri0(-36), c_tri1(-37), c_tri2(-38), c_tri3(-39), c_tri4(-40), c_tri5(-41),
        c_tet0(-42), c_tet1(-43), c_tet2(-44), c_tet3(-45), c_tet4(-46), c_tet5(-47),
        type(-1), p_tet(-1)
#endif
    {
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 || vtx3 < 0 ||
          tri0 < 0 || tri1 < 0 || tri2 < 0 || tri3 < 0)
        throw std::domain_error("[CASL_ERROR]: Invalid tetrahedron.");
    }

    void set(loc_t loc_) {
      loc = loc_;
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 || vtx3 < 0 ||
          tri0 < 0 || tri1 < 0 || tri2 < 0 || tri3 < 0)
        throw std::domain_error("[CASL_ERROR]: Invalid tetrahedron.");
    }
  };

  simplex3_mls_quadratic_t();
  simplex3_mls_quadratic_t(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double x2, double y2, double z2,
                           double x3, double y3, double z3,
                           double x4, double y4, double z4,
                           double x5, double y5, double z5,
                           double x6, double y6, double z6,
                           double x7, double y7, double z7,
                           double x8, double y8, double z8,
                           double x9, double y9, double z9);

  std::vector<vtx3_t> vtxs;
  std::vector<edg3_t> edgs;
  std::vector<tri3_t> tris;
  std::vector<tet3_t> tets;

  void construct_domain(std::vector<CF_3 *> &phi, std::vector<action_t> &acn, std::vector<int> &clr);
  void do_action_vtx(int n_vtx, int cn, action_t action);
  void do_action_edg(int n_edg, int cn, action_t action);
  void do_action_tri(int n_tri, int cn, action_t action);
  void do_action_tet(int n_tet, int cn, action_t action);

  double find_intersection_quadratic(int e);
  void find_middle_node(double &x_out, double &y_out, double x0, double y0, double x1, double y1, int n_tri);
  void find_middle_node_tet(double abc_out[], int n_tet);
  bool need_swap(int v0, int v1);
  void deform_middle_node(double &x_out, double &y_out,
                          double x, double y,
                          double x0, double y0,
                          double x1, double y1,
                          double x2, double y2,
                          double x3, double y3,
                          double x01, double y01);

  void refine_all();
  void refine_edg(int n_edg);
  void refine_tri(int n_tri);
  void refine_tet(int n_tet);

//  void interpolate_all(double &p0, double &p1, double &p2, double &p3);
//  void interpolate_from_neighbors(int v);
//  void interpolate_from_parent(int v);
//  void interpolate_from_parent(vtx3_t &v);

  double integrate_over_domain            (CF_3 &f);
  double integrate_over_interface         (CF_3 &f, int num0);
  double integrate_over_colored_interface (CF_3 &f, int num0, int num1);
  double integrate_over_intersection      (CF_3 &f, int num0, int num1);
  double integrate_over_intersection      (CF_3 &f, int num0, int num1, int num2);
  double integrate_in_dir                 (CF_3 &f, int dir);

  double jacobian_edg(int n_edg, double a);
  double jacobian_tri(int n_tri, double *ab);
  double jacobian_tet(int n_tet, double *abc);

  void mapping_edg(double* xyz, int n_edg, double a);
  void mapping_tri(double* xyz, int n_tri, double* ab);
  void mapping_tet(double *xyz, int n_tet, double* abc);

//  double interpolate_from_parent(std::vector<double> &f, double* xyz);
//  void inv_mat3(double *in, double *out);

//  double length (int vtx0, int vtx1);
  double area   (int vtx0, int vtx1, int vtx2);
  double volume (int vtx0, int vtx1, int vtx2, int vtx3);
  double volume(vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3);

//  bool use_linear;
//  double find_intersection_linear   (int v0, int v1);
//  double find_intersection_quadratic(int e);

//  void get_edge_coords(int e, double xyz[]);

  void construct_proper_mapping(int tri_idx, int phi_idx);
  double find_root(double phi, double phi_n, double phi_nn);
  double interpolate_from_parent(std::vector<double> &f, double* xyz);
  void inv_mat3(double *in, double *out);
  double interpolate_from_parent_with_derivatives(double* xyz, double normal[3], double &F, double &Fn, double &Fnn);
  double interpolate_from_parent_with_derivatives(double* xyz, double &F, double &Fn, double &Fnn, double *normal);
  void deform_edge_in_normal_dir(int n_edg);
  void invert_mapping_tri(int tri_idx, double xyz[3], double ab[2]);

  template<typename X>
  void swap(X &x, X &y)
  {
    X tmp;
    tmp = x; x = y; y = tmp;
  }

  template<typename X>
  void swap(X *x, X *y)
  {
    X *tmp;
    tmp = x; x = y; y = tmp;
  }

  void perturb(double &f, double epsilon){
    if(fabs(f) < epsilon){
      if(f > 0) f =  epsilon;
      else      f = -epsilon;
    }
  }

  bool invalid_reconstruction;

//  void set_use_linear(bool val) { use_linear = val; }
//  void set_use_linear(bool val) { use_linear = true; }

#ifdef CASL_THROWS
  bool tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2);
  bool tri_is_ok(int t);
  bool tet_is_ok(int s);
#endif

};

#endif // SIMPLEX3_MLS_QUADRATIC_H
