#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef simplex2_mls_q_H
#define simplex2_mls_q_H

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cfloat>
#include <climits>

#define SIMPLEX2_MLS_Q_DEBUG

class simplex2_mls_q_t
{
  friend class simplex2_mls_q_vtk;
  friend class cube2_mls_q_t;

public:

  //--------------------------------------------------
  // Class Constructors
  //--------------------------------------------------
  simplex2_mls_q_t(double x0, double y0,
                   double x1, double y1,
                   double x2, double y2,
                   double x3, double y3,
                   double x4, double y4,
                   double x5, double y5);

  //--------------------------------------------------
  // Domain Reconstruction
  //--------------------------------------------------
  void construct_domain(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr);


  //--------------------------------------------------
  // Quadrature Points
  //--------------------------------------------------
  void quadrature_over_domain       (                    std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_interface    (int num,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_intersection (int num0, int num1, std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_in_dir            (int dir,            std::vector<double> &weights, std::vector<double> &X, std::vector<double> &Y);

private:

  // some geometric info
  const static int nodes_per_tri_ = 6;

  // resolution limit (eps_ = eps_rel*lmin)
  double eps_rel_;
  double eps_;
  double lmin_;

  double phi_max_;
  double phi_eps_;

  // inverse mapping for interpolation purposes
  double map_parent_to_ref_[4];

  // number of interfaces
  int num_phi_;

  // flag to discard intermediate reconstruction
  bool invalid_reconstruction_;

  // parameters
  int max_refinement_;
  double snap_limit_;
  bool check_for_curvature_         ;
  bool check_for_edge_intersections_;
  bool check_for_overlapping_       ;
  bool refine_in_normal_dir_        ;
  bool adjust_auxiliary_midpoint_   ;

  //--------------------------------------------------
  // Elementary Geometric Elements
  //--------------------------------------------------
  //--------------------------------------------------
  // Vertex
  //--------------------------------------------------
  struct vtx2_t
  {
    /* Structure and properties */
    double  x, y;   // coordinates
    int     c0, c1; // colors
    double  value;  // stored value
    loc_t   loc;    // location

    int     n_vtx0, n_vtx1; // neighbors
    double  ratio;          // placement between nv0 and nv1
    bool    is_recycled;    // for quadratic elements nodes might become unused

#ifdef SIMPLEX2_MLS_Q_DEBUG
    int p_edg; // parent edge
#endif

    vtx2_t(double x = 0.0, double y = 0.0)
      : x(x), y(y), c0(-1), c1(-1), value(0.0), loc(INS), is_recycled(false)
#ifdef SIMPLEX2_MLS_Q_DEBUG
      , n_vtx0(-1), n_vtx1(-1), ratio(1.0),
        p_edg(-1)
#endif
    {}

    void set(loc_t loc_, int c0_, int c1_) {loc = loc_; c0 = c0_; c1 = c1_;}
  };

  //--------------------------------------------------
  // Edge
  //--------------------------------------------------
  struct edg2_t
  {
    /* Structure and properties */
    int   vtx0, vtx1, vtx2; // vertices
    int   c0;               // colors
    bool  is_split;         // has the edge been split
    loc_t loc;              // location
    int   dir;              // to keep track of edges of a cube
                            // (-1 - inside; 0 - m0; 1 - p0; 2 - 0m; 3 - 0p;)
    int   p_lsf;            // # of lsf that created an edge

    bool to_refine;


    /* Child objects */
    double a;           // location of the intersection point in reference element
    int c_vtx_x;        // intersection point vertex
    int c_edg0, c_edg1; // edges

#ifdef SIMPLEX2_MLS_Q_DEBUG
    int type;                 // type of splitting
    int p_edg, p_tri, p_tet;  // parental objects
#endif

    edg2_t(int v0, int v1, int v2)
      : vtx0(v0), vtx1(v1), vtx2(v2), c0(-1), is_split(false), loc(INS), dir(-1), p_lsf(-1), to_refine(false)
#ifdef SIMPLEX2_MLS_Q_DEBUG
      , c_vtx_x(-1), c_edg0(-1), c_edg1(-1),
        type(-1), p_edg(-1), p_tri(-1), p_tet(-1)
#endif
    {}

    void set(loc_t loc_, int c0_) {loc = loc_; c0 = c0_;}
  };

  //--------------------------------------------------
  // Triangle
  //--------------------------------------------------
  struct tri2_t
  {
    /* Structure and properties */
    int   vtx0, vtx1, vtx2; // vertices
    int   edg0, edg1, edg2; // edges
    loc_t loc;              // location
    bool  is_split;         // has the triangle been split
    bool to_refine;

    /* Child objects */
    int c_vtx01, c_vtx02, c_vtx12;  // vertices
    int c_edg0,  c_edg1,  c_edg2;   // edges
    int c_tri0,  c_tri1,  c_tri2;   // triangles

#ifdef SIMPLEX2_MLS_Q_DEBUG
    int type;
    int p_tri, p_tet;
#endif

    tri2_t(int v0 = -1, int v1 = -1, int v2 = -1,
           int e0 = -1, int e1 = -1, int e2 = -1)
      : vtx0(v0), vtx1(v1), vtx2(v2),
        edg0(e0), edg1(e1), edg2(e2),
        loc(INS), is_split(false), to_refine(false)
#ifdef SIMPLEX2_MLS_Q_DEBUG
      , c_vtx01(-1), c_vtx02(-1), c_vtx12(-1),
        c_edg0(-1), c_edg1(-1),
        c_tri0(-1), c_tri1(-1), c_tri2(-1),
        type(-1), p_tri(-1), p_tet(-1)
#endif
    {}
    void set(loc_t loc_) {loc = loc_;}
  };

  //--------------------------------------------------
  // Arrays Containing Geometric Structure
  //--------------------------------------------------
  std::vector<vtx2_t> vtxs_;
  std::vector<edg2_t> edgs_;
  std::vector<tri2_t> tris_;

  std::vector<vtx2_t> vtxs_tmp_;
  std::vector<edg2_t> edgs_tmp_;
  std::vector<tri2_t> tris_tmp_;

  //--------------------------------------------------
  // Splitting
  //--------------------------------------------------
  void do_action_vtx(int n_vtx, int cn, action_t action);
  void do_action_edg(int n_edg, int cn, action_t action);
  void do_action_tri(int n_tri, int cn, action_t action);

  double find_intersection_quadratic(int e);

  bool find_middle_node(double *xyz_out, double *xyz0, double *xyz1, int n_tri, double *t = NULL);

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

  //--------------------------------------------------
  // Geometry Aware Refinement
  //--------------------------------------------------
  void smart_refine_edg(int n_edg);
  void smart_refine_tri(int n_tri);

  //--------------------------------------------------
  // Jacobians
  //--------------------------------------------------
  double jacobian_edg(int n_edg, double a);
  double jacobian_tri(int n_edg, double a, double b);

  //--------------------------------------------------
  // Mappings
  //--------------------------------------------------
  void mapping_edg(double *xyz, int n_edg, double  a);
  void mapping_tri(double *xyz, int n_tri, double *ab);

  //--------------------------------------------------
  // Computation tools
  //--------------------------------------------------
  double length (int v0, int v1);
  double length (int e);
  double area   (int v0, int vtx1, int vtx2);

  inline double signum(double x)
  {
    return (x > 0.) ? 1. : ((x < 0.) ? -1. : 0.);
  }

  inline bool same_sign(double &x, double &y)
  {
    return x < 0 && y < 0 || x > 0 && y > 0;
  }

  inline bool not_finite(double &x)
  {
    return x != x || x <= -DBL_MAX || x >= DBL_MAX;
  }

  //--------------------------------------------------
  // Interpolation
  //--------------------------------------------------
  double interpolate_from_parent(std::vector<double> &f, double x, double y);
  double interpolate_from_parent(double x, double y);

  //--------------------------------------------------
  // Sorting
  //--------------------------------------------------
  inline bool need_swap(int v0, int v1)
  {
    if (vtxs_[v0].value > vtxs_[v1].value) return true;
    if (vtxs_[v0].value < vtxs_[v1].value) return false;
    return (v0 > v1 ? true : false);
  }

  template<typename X>
  inline void swap(X &x, X &y)
  {
    X tmp;
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
#ifdef SIMPLEX2_MLS_Q_DEBUG
  bool tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2);
  bool tri_is_ok(int t);
#endif
};

#endif // simplex2_mls_q_H
