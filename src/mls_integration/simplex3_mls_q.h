#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef simplex3_mls_q_H
#define simplex3_mls_q_H

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cfloat>
#include <climits>

#define SIMPLEX3_MLS_Q_DEBUG

class simplex3_mls_q_t
{
  friend class simplex3_mls_q_vtk;
  friend class cube3_mls_q_t;

public:

  //--------------------------------------------------
  // Class Constructors
  //--------------------------------------------------
  simplex3_mls_q_t(double x0, double y0, double z0,
                   double x1, double y1, double z1,
                   double x2, double y2, double z2,
                   double x3, double y3, double z3,
                   double x4, double y4, double z4,
                   double x5, double y5, double z5,
                   double x6, double y6, double z6,
                   double x7, double y7, double z7,
                   double x8, double y8, double z8,
                   double x9, double y9, double z9);

  //--------------------------------------------------
  // Domain Reconstruction
  //--------------------------------------------------
  void construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr);

  //--------------------------------------------------
  // Quadrature Points
  //--------------------------------------------------
  void quadrature_over_domain       (                              std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_interface    (int num,                      std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_intersection (int num0, int num1,           std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_intersection (int num0, int num1, int num2, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_in_dir            (int dir,                      std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);

private:

  // some geometric info
  const static int nodes_per_tri_  = 6;
  const static int nodes_per_tet_  = 10;

  // resolution limit (eps_ = eps_rel*lmin)
  const double eps_rel_ = 1.e-12;
  double eps_;
  double lmin_;

  double phi_max_;
  double phi_eps_;

  // for close to interface vertices
  const double phi_perturbance_ = 1.e10*DBL_MIN;
  const double phi_tolerance_   = 2.e10*DBL_MIN;

  // inverse mapping for interpolation purposes
  double map_parent_to_ref_[9];

  // number of interfaces
  int num_phi_;

  // curvature
  double kappa_;

  // flag to discard a reconstruction
  bool invalid_reconstruction_;

  // some parameters of reconstructions
  const static int max_refinement_ = 8;
  const double snap_limit_ = 0.2;
  const double kappa_scale_ = 1;
  const double kappa_eps_ = 1.e-12;
  const bool check_for_curvature_          = 1;
  const bool check_for_edge_intersections_ = 1;
  const bool try_to_fix_outside_vertices_  = 1;
  const bool check_for_overlapping_        = 1;
  const bool check_for_valid_data_         = 1;
  const bool refine_in_normal_dir_         = 1;
  const bool adjust_auxiliary_midpoint_    = 0;

  //--------------------------------------------------
  // Elementary Geometric Elements
  //--------------------------------------------------
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

#ifdef SIMPLEX3_MLS_Q_DEBUG
    int p_edg; // parent edge
#endif

    vtx3_t(double x, double y, double z)
      : x(x), y(y), z(z), c0(-1), c1(-1), c2(-1), value(0.0), loc(INS), is_recycled(false)
#ifdef SIMPLEX3_MLS_Q_DEBUG
      , n_vtx0(-1), n_vtx1(-1), ratio(1.0),
        p_edg(-1)
#endif
    {
#ifdef SIMPLEX3_MLS_Q_DEBUG
          if (x != x ||
              y != y ||
              z != z )
                throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Something went wrong during integration.");
#endif
    }

    inline void set(loc_t loc_, int c0_, int c1_, int c2_) {loc = loc_; c0 = c0_; c1 = c1_; c2 = c2_;}
  };

  struct edg3_t // edge
  {
    /* Structure and properties */
    int   vtx0, vtx1, vtx2; // vertices
    int   c0, c1;     // colors
    bool  is_split;   // has the edge been split
    loc_t loc;        // location
    double value;    // stored value at midpoint
    bool to_refine;

    /* Child objects */
    double a;           // location of the intersection point in reference element
    int c_vtx_x;        // intersection point vertex
    int c_edg0, c_edg1; // edges

#ifdef SIMPLEX3_MLS_Q_DEBUG
    int type;                 // type of splitting
    int p_edg, p_tri, p_tet;  // parental objects
#endif

    edg3_t(int v0, int v1, int v2)
      : vtx0(v0), vtx1(v1), vtx2(v2), c0(-1), c1(-1), is_split(false), loc(INS), to_refine(false), a(0.5)
#ifdef SIMPLEX3_MLS_Q_DEBUG
      , c_vtx_x(-11), c_edg0(-12), c_edg1(-13),
        type(-1), p_edg(-1), p_tri(-1), p_tet(-1)
#endif
    {
#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0)
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Invalid edge.");
#endif
    }

    inline void set(loc_t loc_, int c0_, int c1_) {
      loc = loc_; c0 = c0_; c1 = c1_;
#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0)
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Invalid edge.");
#endif
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
    bool to_refine;

    double a, b;

    /* some stuff for better reconstruction */
    double g_vtx01[3], g_vtx12[3], g_vtx02[3]; // midpoints in normal direction
    double ab01[2], ab12[2], ab02[2];
    bool is_curved;


    /* Child objects */
    int c_vtx01, c_vtx02, c_vtx12;  // vertices
    int c_edg0,  c_edg1,  c_edg2;   // edges
    int c_tri0,  c_tri1,  c_tri2,  c_tri3;   // triangles

#ifdef SIMPLEX3_MLS_Q_DEBUG
    int type;
    int p_tri, p_tet;
#endif

    tri3_t(int v0, int v1, int v2,
           int e0, int e1, int e2)
      : vtx0(v0), vtx1(v1), vtx2(v2),
        edg0(e0), edg1(e1), edg2(e2),
        c(-1), loc(INS), is_split(false), dir(-1), p_lsf(-1),
        is_curved(false), to_refine(false), a(0), b(0)
#ifdef SIMPLEX3_MLS_Q_DEBUG
      , c_vtx01(-21), c_vtx02(-22), c_vtx12(-23),
        c_edg0(-24), c_edg1(-25), c_edg2(-26),
        c_tri0(-27), c_tri1(-28), c_tri2(-29), c_tri3(-20),
        type(-1), p_tri(-1), p_tet(-1)
#endif
    {
#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 ||
          edg0 < 0 || edg1 < 0 || edg2 < 0)
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Invalid triangle.");
#endif
    }
    inline void set(loc_t loc_, int c_) {
      loc = loc_; c = c_;
#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 ||
          edg0 < 0 || edg1 < 0 || edg2 < 0)
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Invalid triangle.");
#endif
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

#ifdef SIMPLEX3_MLS_Q_DEBUG
    int type;
    int p_tet;
#endif

    tet3_t(int v0, int v1, int v2, int v3,
           int t0, int t1, int t2, int t3)
      : vtx0(v0), vtx1(v1), vtx2(v2), vtx3(v3),
        tri0(t0), tri1(t1), tri2(t2), tri3(t3),
        loc(INS), is_split(false)
#ifdef SIMPLEX3_MLS_Q_DEBUG
      , c_vtx01(-30), c_vtx02(-31), c_vtx03(-32), c_vtx12(-33), c_vtx13(-34), c_vtx23(-35),
        c_tri0(-36), c_tri1(-37), c_tri2(-38), c_tri3(-39), c_tri4(-40), c_tri5(-41),
        c_tet0(-42), c_tet1(-43), c_tet2(-44), c_tet3(-45), c_tet4(-46), c_tet5(-47),
        type(-1), p_tet(-1)
#endif
    {
#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 || vtx3 < 0 ||
          tri0 < 0 || tri1 < 0 || tri2 < 0 || tri3 < 0)
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Invalid tetrahedron.");
#endif
    }

    inline void set(loc_t loc_) {
      loc = loc_;
#ifdef SIMPLEX3_MLS_Q_DEBUG
      if (vtx0 < 0 || vtx1 < 0 || vtx2 < 0 || vtx3 < 0 ||
          tri0 < 0 || tri1 < 0 || tri2 < 0 || tri3 < 0)
        throw std::domain_error("[CASL_ERROR]: (simplex3_mls_q_t) Invalid tetrahedron.");
#endif
    }
  };

  //--------------------------------------------------
  // Arrays Containing Geometric Structure
  //--------------------------------------------------
  std::vector<vtx3_t> vtxs_;
  std::vector<edg3_t> edgs_;
  std::vector<tri3_t> tris_;
  std::vector<tet3_t> tets_;

  std::vector<vtx3_t> vtxs_tmp_;
  std::vector<edg3_t> edgs_tmp_;
  std::vector<tri3_t> tris_tmp_;
  std::vector<tet3_t> tets_tmp_;

  //--------------------------------------------------
  // Splitting
  //--------------------------------------------------
  void do_action_vtx(int n_vtx, int cn, action_t action);
  void do_action_edg(int n_edg, int cn, action_t action);
  void do_action_tri(int n_tri, int cn, action_t action);
  void do_action_tet(int n_tet, int cn, action_t action);

  double find_intersection_quadratic(int e);

  bool find_middle_node(double *xyz_out, double *xyz0, double *xyz1, int n_tri, double *t = NULL);
  bool find_middle_node_tet(double abc_out[], int n_tet, double *t = NULL);

  void adjust_middle_node(double *xyz_out,
                          double *xyz_in,
                          double *xyz0,
                          double *xyz1,
                          double *xyz2,
                          double *xyz3,
                          double *xyz01);

  //--------------------------------------------------
  // Simple Refinement
  //--------------------------------------------------
  void refine_edg(int n_edg);
  void refine_tri(int n_tri);
  void refine_tet(int n_tet);

  //--------------------------------------------------
  // Geometry Aware Refinement
  //--------------------------------------------------
  void smart_refine_edg(int n_edg);
  void smart_refine_tri(int n_tri);
  void smart_refine_tri(int n_tri, double a, double b);
  void smart_refine_tet(int n_tet);

  //--------------------------------------------------
  // Jacobians
  //--------------------------------------------------
  double jacobian_edg(int n_edg, double a);
  double jacobian_tri(int n_tri, double *ab);
  double jacobian_tet(int n_tet, double *abc);

  //--------------------------------------------------
  // Mappings
  //--------------------------------------------------
  void mapping_edg(double *xyz, int n_edg, double a);
  void mapping_tri(double *xyz, int n_tri, double *ab);
  void mapping_tet(double *xyz, int n_tet, double *abc);

  //--------------------------------------------------
  // Computation Tools
  //--------------------------------------------------
  double length (int vtx0, int vtx1);
  double length (int e);
  double area   (int vtx0, int vtx1, int vtx2);
  double volume (int vtx0, int vtx1, int vtx2, int vtx3);
  double volume (vtx3_t &vtx0, vtx3_t &vtx1, vtx3_t &vtx2, vtx3_t &vtx3);

  void compute_curvature();

  void inv_mat3(double *in, double *out);

  inline double signum(double x)
  {
    return (x > 0.) ? 1. : ((x < 0.) ? -1. : 0.);
  }

  inline bool same_sign(double &x, double &y)
  {
    return x < 0 && y < 0 || x > 0 && y > 0;
  }

  inline bool not_finite(double x)
  {
    return x != x || x <= -DBL_MAX || x >= DBL_MAX;
  }

  //--------------------------------------------------
  // Interpolation
  //--------------------------------------------------
  double interpolate_from_parent(double* xyz);
  double interpolate_from_parent(std::vector<double> &f, double* xyz);

  //--------------------------------------------------
  // Sorting
  //--------------------------------------------------
  void sort_edg(int n_edg);
  void sort_tri(int n_tri);
  void sort_tet(int n_tet);

  inline bool need_swap(int v0, int v1)
  {
    if (vtxs_[v0].value > vtxs_[v1].value) return true;
    if (vtxs_[v0].value < vtxs_[v1].value) return false;
    return (v0 > v1 ? true : false);
  }

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

  inline void perturb(double &f, double epsilon)
  {
    if(fabs(f) < epsilon)
    {
      if(f >= 0) f =  epsilon;
      else       f = -epsilon;
    }
  }

  //--------------------------------------------------
  // Debugging
  //--------------------------------------------------
#ifdef SIMPLEX3_MLS_Q_DEBUG
  bool tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2);
  bool tri_is_ok(int t);
  bool tet_is_ok(int s);
#endif

};

#endif // simplex3_mls_q_H
