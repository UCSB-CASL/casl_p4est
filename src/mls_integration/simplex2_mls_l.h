#ifndef MLS_INTEGRATION_TYPES
#define MLS_INTEGRATION_TYPES
enum loc_t {INS, OUT, FCE, LNE, PNT};
enum action_t {INTERSECTION, ADDITION, COLORATION};
#endif

#ifndef simplex2_mls_l_H
#define simplex2_mls_l_H

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cfloat>

#define SIMPLEX2_MLS_L_T_DEBUG

class simplex2_mls_l_t
{
private:
  friend class simplex2_mls_l_vtk;
  friend class cube2_mls_l_t;
  // some geometric info
  const static int nodes_per_tri_ = 3;

  // simplex area (for interpolation)
  double A_;

  // resolution limit (eps_ = eps_rel*lmin)
  double eps_rel_;
  double eps_;
  double lmin_;

  double phi_max_;
  double phi_eps_;

  // for close to interface vertices
  const double phi_perturbance_ = 10.*DBL_MIN;
  const double phi_tolerance_   = 12.*DBL_MIN;

  // number of interfaces
  int num_phi_;

  bool use_linear_;

  //--------------------------------------------------
  // Elementary Geometric Elements
  //--------------------------------------------------
  struct vtx2_t // vertex
  {
    /* Structure and properties */
    double  x, y;   // coordinates
    int     c0, c1; // colors
    double  value;  // stored value
    loc_t   loc;    // location

    int     n_vtx0, n_vtx1; // neighbors
    double  ratio;          // placement between nv0 and nv1

#ifdef SIMPLEX2_MLS_L_T_DEBUG
    int p_edg; // parent edge
#endif

    vtx2_t(double x = 0.0, double y = 0.0)
      : x(x), y(y), c0(-1), c1(-1), value(0.0), loc(INS)
#ifdef SIMPLEX2_MLS_L_T_DEBUG
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
    double value;     // stored value at midpoint

    /* Child objects */
    int c_vtx01;        // splitting vertex
    int c_edg0, c_edg1; // edges

#ifdef SIMPLEX2_MLS_L_T_DEBUG
    int type;                 // type of splitting
    int p_edg, p_tri, p_tet;  // parental objects
#endif

    edg2_t(int v0 = -1, int v1 = -1)
      : vtx0(v0), vtx1(v1), c0(-1), is_split(false), loc(INS), dir(-1), p_lsf(-1)
#ifdef SIMPLEX2_MLS_L_T_DEBUG
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

#ifdef SIMPLEX2_MLS_L_T_DEBUG
    int type;
    int p_tri, p_tet;
#endif

    tri2_t(int v0 = -1, int v1 = -1, int v2 = -1,
           int e0 = -1, int e1 = -1, int e2 = -1)
      : vtx0(v0), vtx1(v1), vtx2(v2),
        edg0(e0), edg1(e1), edg2(e2),
        loc(INS), is_split(false)
#ifdef SIMPLEX2_MLS_L_T_DEBUG
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

  //--------------------------------------------------
  // Splitting
  //--------------------------------------------------
  void do_action_vtx(int n_vtx, int cn, action_t action);
  void do_action_edg(int n_edg, int cn, action_t action);
  void do_action_tri(int n_tri, int cn, action_t action);

  //--------------------------------------------------
  // Intersections
  //--------------------------------------------------
  double find_intersection_linear   (int v0, int v1);
  double find_intersection_quadratic(int e);

  //--------------------------------------------------
  // Interpolation
  //--------------------------------------------------
  void interpolate_from_neighbors(int v);
  void interpolate_from_parent(int v);
  void interpolate_from_parent(vtx2_t &vtx);

  //--------------------------------------------------
  // Computation tools
  //--------------------------------------------------
  double length (int v0, int v1);
  double area   (int v0, int v1, int v2);
  double area   (vtx2_t &vtx0, vtx2_t &vtx1, vtx2_t &vtx2);

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
#ifdef SIMPLEX2_MLS_L_T_DEBUG
  bool tri_is_ok(int v0, int v1, int v2, int e0, int e1, int e2);
  bool tri_is_ok(int t);
#endif

public:

  //--------------------------------------------------
  // Class Constructors
  //--------------------------------------------------
  simplex2_mls_l_t(double x0, double y0,
                   double x1, double y1,
                   double x2, double y2,
                   double eps_rel = 1.e-10);

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

  //--------------------------------------------------
  // Various
  //--------------------------------------------------
  inline void set_use_linear(bool use_linear) { use_linear_ = use_linear; }
};

#endif // simplex2_mls_l_H
