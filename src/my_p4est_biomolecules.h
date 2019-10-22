#ifndef MY_P4EST_BIOMOLECULES_H
#define MY_P4EST_BIOMOLECULES_H

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_poisson_jump_nodes_voronoi.h>
#include <src/my_p8est_general_poisson_nodes_mls_solver.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <p8est_extended.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/my_p4est_general_poisson_nodes_mls_solver.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <p4est_extended.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <numeric>

static const double domain_side_length = 1.0;

// for validating my solver
static const double xyz_c_val[P4EST_DIM] = {0.5*domain_side_length, 0.5*domain_side_length
                                            #ifdef P4_TO_P8
                                            , 0.5*domain_side_length
                                            #endif
                                           };
static const double mag_val     = 1.0;
static const double radius_val  = domain_side_length*0.4;
static const double alpha_val   = -0.2/domain_side_length;
static const double beta_val    = -M_PI/domain_side_length;
static const double gamma_val   = 0.4/domain_side_length;


static struct:
  #ifdef P4_TO_P8
  CF_3
  #else
  CF_2
  #endif
{
    #ifdef P4_TO_P8
    double operator()(double x, double y, double z) const
{
    return (mag_val*SQR(radius_val)/(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2])));
}
double x_derivative(double x, double y, double z) const
{
  return (-2.0*mag_val*SQR(radius_val)*(x-xyz_c_val[0])/(SQR(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2]))));
}
double xx_derivative(double x, double y, double z) const
{
  return (+2.0*mag_val*SQR(radius_val)*(3.0*SQR(x-xyz_c_val[0]) - SQR(y - xyz_c_val[1]) - SQR(z - xyz_c_val[2]) - SQR(radius_val))/(pow(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2]), 3.0)));
}
double y_derivative(double x, double y, double z) const
{
  return (-2.0*mag_val*SQR(radius_val)*(y-xyz_c_val[1])/(SQR(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2]))));
}
double yy_derivative(double x, double y, double z) const
{
  return (+2.0*mag_val*SQR(radius_val)*(3.0*SQR(y-xyz_c_val[1]) - SQR(x - xyz_c_val[0]) - SQR(z - xyz_c_val[2]) - SQR(radius_val))/(pow(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2]), 3.0)));
}
double z_derivative(double x, double y, double z) const
{
  return (-2.0*mag_val*SQR(radius_val)*(z-xyz_c_val[2])/(SQR(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2]))));
}
double zz_derivative(double x, double y, double z) const
{
  return (+2.0*mag_val*SQR(radius_val)*(3.0*SQR(z-xyz_c_val[2]) - SQR(x - xyz_c_val[0]) - SQR(y - xyz_c_val[1]) - SQR(radius_val))/(pow(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])+SQR(z-xyz_c_val[2]), 3.0)));
}
#else
    double operator()(double x, double y) const
{
    return (mag_val*SQR(radius_val)/(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1])));
}
double x_derivative(double x, double y) const
{
  return (-2.0*mag_val*SQR(radius_val)*(x-xyz_c_val[0])/(SQR(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1]))));
}
double xx_derivative(double x, double y) const
{
  return (+2.0*mag_val*SQR(radius_val)*(3.0*SQR(x-xyz_c_val[0]) - SQR(y - xyz_c_val[1]) - SQR(radius_val))/(pow(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1]), 3.0)));
}
double y_derivative(double x, double y) const
{
  return (-2.0*mag_val*SQR(radius_val)*(y-xyz_c_val[1])/(SQR(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1]))));
}
double yy_derivative(double x, double y) const
{
  return (+2.0*mag_val*SQR(radius_val)*(3.0*SQR(y-xyz_c_val[1]) - SQR(x - xyz_c_val[0]) - SQR(radius_val))/(pow(SQR(radius_val)+SQR(x-xyz_c_val[0])+SQR(y-xyz_c_val[1]), 3.0)));
}
#endif
} first_term;

static struct:
  #ifdef P4_TO_P8
  CF_3
  #else
  CF_2
  #endif
{
    #ifdef P4_TO_P8
    double operator()(double x, double y, double z) const
{
    return (atan(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2])));
}
double x_derivative(double x, double y, double z) const
{
  return (alpha_val/(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2]))));
}
double xx_derivative(double x, double y, double z) const
{
  return (-2.0*SQR(alpha_val)*(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2]))/(SQR(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2])))));
}
double y_derivative(double x, double y, double z) const
{
  return (beta_val/(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2]))));
}
double yy_derivative(double x, double y, double z) const
{
  return (-2.0*SQR(beta_val)*(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2]))/(SQR(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2])))));
}
double z_derivative(double x, double y, double z) const
{
  return (gamma_val/(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2]))));
}
double zz_derivative(double x, double y, double z) const
{
  return (-2.0*SQR(gamma_val)*(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2]))/(SQR(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])+gamma_val*(z-xyz_c_val[2])))));
}

#else
    double operator()(double x, double y) const
{
    return (atan(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])));
}
double x_derivative(double x, double y) const
{
  return (alpha_val/(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1]))));
}
double xx_derivative(double x, double y) const
{
  return (-2.0*SQR(alpha_val)*(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1]))/(SQR(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])))));
}
double y_derivative(double x, double y) const
{
  return (beta_val/(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1]))));
}
double yy_derivative(double x, double y) const
{
  return (-2.0*SQR(beta_val)*(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1]))/(SQR(1.0 + SQR(alpha_val*(x-xyz_c_val[0])+beta_val*(y-xyz_c_val[1])))));
}
#endif
} second_factor;

static struct:
  #ifdef P4_TO_P8
  CF_3
  #else
  CF_2
  #endif
{
    #ifdef P4_TO_P8
    double operator()(double x, double y, double z) const
{
    return (first_term(x, y, z)*(1.0 + second_factor(x, y, z)));
    }
    double x_derivative(double x, double y, double z) const
{
    return (first_term.x_derivative(x, y, z)*(1.0 + second_factor(x, y, z)) + first_term(x, y, z)*second_factor.x_derivative(x, y, z));
    }
    double xx_derivative(double x, double y, double z) const
{
    return (first_term.xx_derivative(x, y, z)*(1.0 + second_factor(x, y, z)) + 2.0*first_term.x_derivative(x, y, z)*second_factor.x_derivative(x, y, z) +  first_term(x, y, z)*second_factor.xx_derivative(x, y, z));
    }
    double y_derivative(double x, double y, double z) const
{
    return (first_term.y_derivative(x, y, z)*(1.0 + second_factor(x, y, z)) + first_term(x, y, z)*second_factor.y_derivative(x, y, z));
    }
    double yy_derivative(double x, double y, double z) const
{
    return (first_term.yy_derivative(x, y, z)*(1.0 + second_factor(x, y, z)) + 2.0*first_term.y_derivative(x, y, z)*second_factor.y_derivative(x, y, z) +  first_term(x, y, z)*second_factor.yy_derivative(x, y, z));
    }
    double z_derivative(double x, double y, double z) const
{
    return (first_term.z_derivative(x, y, z)*(1.0 + second_factor(x, y, z)) + first_term(x, y, z)*second_factor.z_derivative(x, y, z));
    }
    double zz_derivative(double x, double y, double z) const
{
    return (first_term.zz_derivative(x, y, z)*(1.0 + second_factor(x, y, z)) + 2.0*first_term.z_derivative(x, y, z)*second_factor.z_derivative(x, y, z) +  first_term(x, y, z)*second_factor.zz_derivative(x, y, z));
    }
    double   laplacian(double x, double y, double z)
{
    return (xx_derivative(x, y, z) + yy_derivative(x, y, z) + zz_derivative(x, y, z));
    }
    #else
    double operator()(double x, double y) const
{
    return (first_term(x, y)*(1.0 + second_factor(x, y)));
    }
    double x_derivative(double x, double y) const
{
    return (first_term.x_derivative(x, y)*(1.0 + second_factor(x, y)) + first_term(x, y)*second_factor.x_derivative(x, y));
    }
    double xx_derivative(double x, double y) const
{
    return (first_term.xx_derivative(x, y)*(1.0 + second_factor(x, y)) + 2.0*first_term.x_derivative(x, y)*second_factor.x_derivative(x, y) +  first_term(x, y)*second_factor.xx_derivative(x, y));
    }
    double y_derivative(double x, double y) const
{
    return (first_term.y_derivative(x, y)*(1.0 + second_factor(x, y)) + first_term(x, y)*second_factor.y_derivative(x, y));
    }
    double yy_derivative(double x, double y) const
{
    return (first_term.yy_derivative(x, y)*(1.0 + second_factor(x, y)) + 2.0*first_term.y_derivative(x, y)*second_factor.y_derivative(x, y) +  first_term(x, y)*second_factor.yy_derivative(x, y));
    }
    double   laplacian(double x, double y)
{
    return (xx_derivative(x, y) + yy_derivative(x, y));
    }
    #endif
    } validation_function;



using namespace std;
#if (__cplusplus < 201103L) // for the dumbass outdated compilers
namespace std {
inline string to_string ( size_t x ) {
  return to_string(static_cast<long long>(x));
}
inline string to_string ( int x ) {
  return to_string(static_cast<long long>(x));
}
inline string to_string ( double x ) {
  return to_string(static_cast<long double>(x));
}
template<class ForwardIterator, class T>
void iota(ForwardIterator first, ForwardIterator last, T value)
{
  while(first != last)
    *first++ = value++;
}
}
#endif

//---------------------------------------------------------------------
//
//   Raphael Egan
//   2017 Spring, Summer, Fall, CASL, UCSB
//
//---------------------------------------------------------------------


/*!
 * \brief The Atom struct contains
 * the geometrical coordinates x, y, z of the atom center;
 * its electric charge q;
 * and the atom van der Waals radius r.
 */
struct Atom {
#ifdef P4_TO_P8
  double x, y, z, q, r;
#else
  double x, y, q, r;
#endif
  static const string ATOM;
  inline double dist_to_vdW_surface(const double& xp, const double& yp
                                  #ifdef P4_TO_P8
                                    , const double& zp
                                  #endif
                                    ) const
  {
    return r - sqrt(SQR(xp-x) + SQR(yp-y)
                #ifdef P4_TO_P8
                    + SQR(zp-z)
                #endif
                    );
  }
  inline double dist_to_vdW_surface(const double* xyz) const
  {
    return dist_to_vdW_surface(xyz[0], xyz[1]
    #ifdef P4_TO_P8
        , xyz[2]
    #endif
        );
  }
  inline double dist_to_vdW_surface(const vector<double> xyz) const
  {
    P4EST_ASSERT(xyz.size() == P4EST_DIM);
    return dist_to_vdW_surface(&xyz.at(0));
  }
  double max_phi_vdW_in_quad(double* xyz_c, double* dxdydz, double* xyzM = NULL) const
  {
    if((fabs(x - xyz_c[0]) <= 0.5*dxdydz[0])
       && (fabs(y - xyz_c[1]) <= 0.5*dxdydz[1])
   #ifdef P4_TO_P8
       && (fabs(z - xyz_c[2]) <= 0.5*dxdydz[2])
   #endif
       ) // the cell cointains the atom center
    {
      if(xyzM != NULL)
      {
        xyzM[0] = x;
        xyzM[1] = y;
#ifdef P4_TO_P8
        xyzM[2] = z;
#endif
      }
      return r;
    }
    int ioff = (x > xyz_c[0]+0.5*dxdydz[0])? 1: (x < xyz_c[0]-0.5*dxdydz[0])? -1: 0;
    int joff = (y > xyz_c[1]+0.5*dxdydz[1])? 1: (y < xyz_c[1]-0.5*dxdydz[1])? -1: 0;
#ifdef P4_TO_P8
    int koff = (z > xyz_c[2]+0.5*dxdydz[2])? 1: (z < xyz_c[2]-0.5*dxdydz[2])? -1: 0;
#endif
    if (xyzM != NULL)
    {
      xyzM[0] = (ioff!=0)?(xyz_c[0]+ioff*0.5*dxdydz[0]):x;
      xyzM[1] = (joff!=0)?(xyz_c[1]+joff*0.5*dxdydz[1]):y;
#ifdef P4_TO_P8
      xyzM[2] = (koff!=0)?(xyz_c[2]+koff*0.5*dxdydz[2]):z;
#endif
    }
    return r - sqrt(SQR(ioff*(x - (xyz_c[0]+ioff*0.5*dxdydz[0])))
        + SQR(joff*(y - (xyz_c[1]+joff*0.5*dxdydz[1])))
    #ifdef P4_TO_P8
        + SQR(koff*(z - (xyz_c[2]+koff*0.5*dxdydz[2])))
    #endif
        );
  }
};

// following comparisons used for surface construction only, q is irrelevant
inline bool operator ==(const Atom& lhs, const Atom& rhs)
{
  return
      ((fabs(rhs.x - lhs.x) > EPS*MAX(MAX(EPS, fabs(lhs.x)), fabs(rhs.x)))?
         false : (
           (fabs(rhs.y - lhs.y) > EPS*MAX(MAX(EPS, fabs(lhs.y)), fabs(rhs.y)))?
             false : (
             #ifdef P4_TO_P8
               fabs(rhs.z - lhs.z) > EPS*MAX(MAX(EPS, fabs(lhs.z)), fabs(rhs.z))?
                 false:(
                 #endif
                   (fabs(rhs.r - lhs.r) > EPS*MAX(MAX(EPS, fabs(lhs.r)), fabs(rhs.r))?
                      false:true
                    #ifdef P4_TO_P8
                      )
                 #endif
                   )
                 )
             )
         );
}
inline bool operator !=(const Atom& lhs, const Atom& rhs) {return !(lhs==rhs);}
inline bool operator <(const Atom& lhs, const Atom& rhs)
{
  return
      ((fabs(rhs.x - lhs.x) > EPS*MAX(MAX(EPS, fabs(lhs.x)), fabs(rhs.x)))?
         (lhs.x < rhs.x) : (
           (fabs(rhs.y - lhs.y) > EPS*MAX(MAX(EPS, fabs(lhs.y)), fabs(rhs.y)))?
             (lhs.y < rhs.y) :(
             #ifdef P4_TO_P8
               (fabs(rhs.z - lhs.z) > EPS*MAX(MAX(EPS, fabs(lhs.z)), fabs(rhs.z)))?
                 (lhs.z < rhs.z) :(
                 #endif
                   (fabs(rhs.r - lhs.r) > EPS*MAX(MAX(EPS, fabs(lhs.r)), fabs(rhs.r)))?
                     (lhs.r < rhs.r) : true
                   #ifdef P4_TO_P8
                     )
               #endif
                 )
             )
         );
}

inline istream& operator >> (istream& is, Atom& atom) {
#ifdef P4_TO_P8
  string ignore [4];
#else
  string ignore [5];
#endif
  for (int i=0; i<4; i++) is >> ignore[i];
  is >> atom.x >> atom.y ;
#ifdef P4_TO_P8
  is >> atom.z ;
#else
  is >> ignore[4] ;
#endif
  is >> atom.q >> atom.r;

  return is;
}


inline ostream& operator << (ostream& os, Atom& atom) {
  os << "(x = " << atom.x << ", y = " << atom.y;
#ifdef P4_TO_P8
  os << ", z = " << atom.z;
#endif
  os << "; q = " << atom.q << ", r = " << atom.r << ")";
  return os;
}

inline bool operator >>(string& line, Atom& atom)
{
  string word = line.substr(0, 6); // first word in structured line
  if(Atom::ATOM.compare(word))
    return false;
  else
  {
    atom.x = stod(line.substr(30, 8));
    atom.y = stod(line.substr(38, 8));
#ifdef P4_TO_P8
    atom.z = stod(line.substr(46, 8));
#endif
    atom.q = stod(line.substr(54, 8));
    atom.r = stod(line.substr(62, 8));
    std::cout << atom << std::endl;
    return true;
  }
}

struct sorted_atom
{
  int global_atom_idx;
  int mol_idx;
  double distance_from_xyz;
  double distance_from_xyz_i;
  double distance_from_graal;
};

/*!
 * \brief The surface_generation_method enum allows a distinction between
 * the two methods used for the calculation of the SAS levelset function.
 * - brute_force method: one loops through all the atoms in the
 * list for every single grid point;
 * - list_reduction: reduced lists of atoms to consider are built
 * recursively as the cells are created. The value of the level set
 * function is exact for grid point such that
 *     \varphi_{\SAS] \geq -order_of_accuracy*diag_of_finest_cell.
 * For points that are further than that, the list of atoms is such
 * that it includes atoms that are close enough to possibly satisfy
 * the Lipschitz refinement criterion so that the computational grid
 * is correct even though the value of the level set function might be
 * inexact.
 */
enum sas_generation_method{
  brute_force,
  list_reduction,
  list_reduction_with_exact_phi
};

enum cavity_removal_method{
  poisson = 314159,
  region_growing
};

class reduced_list
{
private:
  // count the number of such objects (to avoid memory leaks)
  static int  nb_reduced_lists;
public:
  // the actual reduced list
  vector<int> atom_global_idx;
  reduced_list(const int& n = 0)
  {
    atom_global_idx.resize(n);
    iota(atom_global_idx.begin(), atom_global_idx.end(), 0);
    nb_reduced_lists++;
  }
  reduced_list(const int& n, const int& value)
  {
    atom_global_idx.resize(n, value);
    nb_reduced_lists++;
  }
  int inline size() const {return ((int) atom_global_idx.size());}
  static int inline get_nb_reduced_lists() {return nb_reduced_lists;}
  ~reduced_list()
  {
    nb_reduced_lists--;
  }
};

class biomol_grid_parameters
{
private:
  bool                  is_splitting_criterion_set;
  bool                  is_probe_radius_set;
  bool                  is_layer_thickness_set;
  // relevant data
  splitting_criteria_t  sp; // min_lvl, max_lvl, lip
  double                rp; // probe_radius
  int                   min_level_to_capture_probe_radius;
  int                   OOA; // order of accuracy (thickness of the accuracy layer)
  double                accuracy_layer_thickness;
  const double          diag_of_root_cells;
  double                smallest_diag;
  void set_thickness_of_accuracy_layer() {accuracy_layer_thickness = ((double) OOA)*smallest_diag;}
  void set_smallest_diag() {smallest_diag = diag_of_root_cells/(1<<(sp.max_lvl));}
  void change_lmax(const int& new_lmax)
  {
    if(new_lmax != sp.max_lvl)
    {
      sp.max_lvl = new_lmax;
      set_smallest_diag();
      set_thickness_of_accuracy_layer();
    }
  }
public:
  biomol_grid_parameters(const double& root_cell_diag_, const int& l_min = -1, const int& l_max = -1, const double& lip_ = -1., const double& rp_ = -1.0, const int& OOA_ = -1):
    sp(l_min, l_max, lip_), diag_of_root_cells(root_cell_diag_)
  {
    is_splitting_criterion_set = (l_min >= 0 && l_min <= l_max && l_max <= P4EST_QMAXLEVEL && lip_ >= 1.0);
    rp = rp_;
    is_probe_radius_set = rp_ > 0.;
    if(is_probe_radius_set)
      min_level_to_capture_probe_radius = (int) ceil(log2(diag_of_root_cells/rp));
    set_smallest_diag();
    OOA = OOA_;
    set_thickness_of_accuracy_layer();
    is_layer_thickness_set = (((OOA_ == 1) || (OOA_ == 2)) && (sp.max_lvl >= 0));
  }
  inline int min_level() const {return sp.min_lvl;}
  inline int max_level() const{return sp.max_lvl;}
  inline double lip_cst() const{return sp.lip;}
  inline int threshold_level() const{return min_level_to_capture_probe_radius;}
  inline double probe_radius() const{return rp;}
  inline int order_of_accuracy() const {return OOA;}
  inline double layer_thickness() const{return accuracy_layer_thickness;}
  inline double root_diag() const{return diag_of_root_cells;}

  inline bool are_set(){return (is_splitting_criterion_set && is_probe_radius_set && is_layer_thickness_set);}
  bool set_splitting_criterion(const int& l_min, const int& l_max, const double& lip_)
  {
    bool need_to_reset_the_forest = is_splitting_criterion_set && ((l_min != sp.min_lvl) || (l_max != sp.max_lvl) || (fabs(sp.lip - lip_) >= EPS*fabs(sp.lip)));
    sp.min_lvl = l_min;
    change_lmax(l_max);
    sp.lip = lip_;
    is_splitting_criterion_set = (l_min >= 0 && l_min <= l_max && l_max <= P4EST_QMAXLEVEL && lip_ >= 1.0);
    return need_to_reset_the_forest;
  }
  bool set_probe_radius(const double& rp_)
  {
    bool need_to_reset_the_forest = is_probe_radius_set && (fabs(probe_radius() - rp_) > EPS*rp);
    rp = rp_;
    is_probe_radius_set = rp_ > 0.;
    if(is_probe_radius_set)
      min_level_to_capture_probe_radius = (int) ceil(log2(diag_of_root_cells/rp));
    return need_to_reset_the_forest;
  }
  bool set_OOA(const int& OOA_)
  {
    bool need_to_reset_the_forest = is_layer_thickness_set && (OOA != OOA_);
    OOA = OOA_;
    set_thickness_of_accuracy_layer();
    is_layer_thickness_set = (((OOA_ == 1) || (OOA_ == 2)) && (sp.max_lvl >= 0));
    return need_to_reset_the_forest;
  }
};

typedef shared_ptr<reduced_list>  reduced_list_ptr;

class my_p4est_biomolecules_t:public
    #ifdef P4_TO_P8
    CF_3
    #else
    CF_2
    #endif
{
  friend class my_p4est_biomolecules_solver_t;
private:
  class par_error_manager
  {
  private:
    const int       my_rank;
    const int       mpi_size;
    const MPI_Comm  comm;
    FILE*           error_file;
  public:
    par_error_manager(const int& rank_, const int& mpi_size_, MPI_Comm const& comm_, FILE*& err_file_):
      my_rank(rank_), mpi_size(mpi_size_), comm(comm_), error_file(err_file_)
    {
      if(err_file_ == NULL)
      {
        if(my_rank == 0)
          fprintf(stderr, "The error file was not defined, it's set to stderr by default...\n");
        error_file = stderr;
      }
    }
    void check_my_local_error(int& local_error, string& general_message) const
    {
      int             mpiret;
      vector<int>     general_errors; general_errors.resize(mpi_size);
      bool is_there_an_error = false;
      mpiret = MPI_Allgather(&local_error, 1, MPI_INT, &general_errors.at(0), 1, MPI_INT, comm); SC_CHECK_MPI(mpiret);
      for (int k = 0; k < mpi_size; ++k) {
        is_there_an_error |= general_errors.at(k);
        if (is_there_an_error) {
          break;
        }
      }
      if (is_there_an_error)
      {
        if (my_rank == 0)
        {
          PetscFPrintf(comm, error_file, general_message.c_str());
          for (int k = 0; k < mpi_size; ++k) {
            if (general_errors.at(k)) {
              string msg = "------ a local error came from proc " + to_string(k) + "\n";
              PetscFPrintf(comm, error_file, msg.c_str());
            }
          }
        }
        MPI_Finalize();
        exit(is_there_an_error);
      }
    }
    void print_message_and_abort(string& message, int error_code) const
    {
      fprintf(error_file, "%s", message.c_str());
      MPI_Abort(comm, error_code);
    }
  };
  class molecule:public
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
  private:
#ifdef CASL_THROWS
    const par_error_manager       mol_err_manager;
#endif
    static p4est_connectivity_t*  domain_connectivity; // class pointer to the p4est domain connectivity
    static mpi_environment_t*     mpi;                 // class pointer to the mpi environment
    // dimensional variables: need an update when angstrom_to_domain is modified
    double        angstrom_to_domain;                 // scaling factor: distance in the domain = angstrom_to_domain*actual distance (in angstrom)
    vector<Atom>  atoms;                              // list of atoms in the molecule
    int           n_charged_atoms;
    vector<int>   index_of_charged_atom;

    double        molecule_centroid[P4EST_DIM];
    double        side_length_of_bounding_cube;       // self-explanatory
    double        largest_radius;
    bool          scale_is_set;                       // true if the molecule has been scaled (activates the "check if molecule is in box" error management)
    /*!
     * \brief read: reads the pqr file (in parallel by chunks, and then allgather) and
     * computes the molecule centroid;
     * \param pqr: path to the pqr file
     * \param overlap: max number of characters per (relevant) line in the pqr file (or any integer greater than that!)
     */
    void                      read(const string& pqr, const int overlap);
    /*
    //
    // \brief read_serial: reads the pqr file on root proc computes the molecule centroid,
    // and broadcasts the results
    // \param pqr: path to the pqr file
    // ...DEPRECATED...
    void                      read_serial(const string& pqr);
    */
    /*!
     * \brief check_if_file: checks if the file path is a valid path toward a file.
     * Return true if it is a file!
     * \param file_path: self-explanatory
     */
    bool              check_if_file(const string& file_path) const;
    /*!
     * \brief calculate_center_of_domain: self_explicit
     * \param domain_center[out]: pointer to domain center (an array of double[P4EST_DIM]).
     */
    void                      calculate_center_of_domain(double* domain_center) const;
  public:
    // the two following functions return 0 in case of success, 1 in case of failure
    static int                update_connectivity(p4est_connectivity_t* conn_);
    static int                update_mpi(mpi_environment_t* mpi_);
    /*!
     * \brief molecule constructor.
     * \param pqr_: path to the pqr file to read;
     * \param xyz_c (optional): pointer to new_centroid (an array of double[P4EST_DIM]).
     * new_centroid = s*old_centroid if the pointer is NULL (or disregarded);
     * \param angles (optional): pointer to an array of angles (in radians) defining the rotation
     * matrix R.
     * --> In 2D, the right-handed rotation angle (one value).
     * --> in 3D, the angles psi, theta_n, phi_n (three values), representing a right-handed angle
     * of rotation psi around the axis n pointed by polar and azimuthal angles theta_n and phi_n.
     * R = identity if the pointer is NULL (or disregarded);
     * \param angstrom_to_domain_ (optional): scaling factor from angstrom to domain dimensions
     * (default is 1, i.e. no scaling --> validity check is skipped in debug mode in that case);
     * \param overlap (optional): max number of characters per (relevant) line in the pqr file
     * (default value is 70, as observed from my own pqr files, including the '\n' characters)
     */
    molecule(const string& pqr_, const double* xyz_c = NULL, double* angles = NULL, const double angstrom_to_domain_ = 1.0, const int overlap = 70);
    /*!
     * \brief calculate_scaling_factor: calculates the angstrom_to_domain factor that would set the
     * ratio of the side length of the centroid-centered cube bounding the molecule to the minimal
     * domain size equal to the desired value.
     * The method does NOT modify the molecule, it simply calculates the angstrom_to_domain
     * \param cube_side_length_to_min_domain_size: desired ratio (double)
     * \return the value of angstrom_to_domain scaling factor.
     */
    double                    calculate_scaling_factor(const double cube_side_length_to_min_domain_size) const;
    /*!
     * \brief scale_rotate_and_translate: this method loops through all atoms in the molecule and scales
     * and relocates them as
     *    xyz_new = new_centroid + s*R*(xyz_old - old_centroid)
     *    new_atom_radius = s*old_atom_radius
     * \param angstrom_to_domain_ (optional): pointer to the new scaling factor, based on which the
     * multiplicator s is computed. s = 1 if the pointer if NULL (or disregarded);
     * \param xyz_c (optional): pointer to new_centroid (an array of double[P4EST_DIM]).
     * new_centroid = s*old_centroid if the pointer if NULL (or disregarded);
     * \param angles (optional): pointer to an array of angles (in radians) defining the rotation
     * matrix R.
     * --> In 2D, the right-handed rotation angle (one value).
     * --> in 3D, the angles psi, theta_n, phi_n (three values), representing a right-handed angle
     * of rotation psi around the axis n pointed by polar and azimuthal angles theta_n and phi_n.
     * R = identity matrix if the pointer if NULL (or disregarded);
     * The side_length_of_bounding_cube is recalculated (on-the-fly).
     * An exception is thrown in debug mode if the molecule is already scaled and if its bounding
     * box is not entirely in the domain.
     */
    void                      scale_rotate_and_translate(const double* angstrom_to_domain_= NULL, const double* xyz_c = NULL, double* angles = NULL);
    /*!
     * \brief translate: translates the entire molecule to the new desired centroid point
     * \param xyz_c (optional): pointer to the new desired centroid location (double[P4EST_DIM]).
     * If disregarded, it is the center of the computational domain
     * If NULL, it is the current centroid point (i.e. no effect at all)
     * An exception is thrown in debug mode if the molecule is already scaled and if its bounding
     * box is not entirely in the domain.
     */
    void                      translate();
    void                      translate(const double *xyz_c);
    /*!
     * \brief rotate: rotates the entire molecule, around its centroid, and translates it to
     * the new desired centroid point if provided.
     * \param angles: pointer to an array of angles (in radians) defining the rotation.
     * --> In 2D, the right-handed rotation angle (one value)
     * --> in 3D, the angles psi, theta_n, phi_n (three values), representing a right-handed
     * angle of rotation psi around the axis n pointed by polar and azimuthal angles theta_n
     * and phi_n.
     * \param xyz_c (optional): pointer to the new desired centroid location, the centroid of
     * the molecule is unchanged if NULL or disregarded
     * An exception is thrown in debug mode if the molecule is already scaled and if its
     * new bounding box is not entirely in the domain.
     * The side_length_of_bounding_cube is recalculated (on-the-fly).
     */
    void                      rotate(double* angles, const double *xyz_c = NULL);
    /*!
     * \brief scale_and_translate: modifies the value of the private angstrom_to_domain
     * variable and updates the corresponding dimensional variables that depend on that
     * factor.
     * \param angstrom_to_domain_ (optional): pointer to the new scaling factor, based on which the
     * multiplicator s is computed. s = 1 if the pointer if NULL (or disregarded);
     * \param xyz_c(optional): pointer to the new desired centroid location (an array of double[P4EST_DIM]).
     * If disregarded or NULL, the centroid is simply multiplied by the appropriated scaling factor
     * An exception is thrown in debug mode if the molecule is already scaled and if its bounding
     * box is not entirely in the domain.
     */
    void                      scale_and_translate(const double* angstrom_to_domain_, const double* xyz_c = NULL);
    /*!
     * \brief reduce_to_single_atom: keeps the first atom in the list and delete all other ones
     */
    void                      reduce_to_single_atom();
    /*!
     * \brief is_bounding_box_in_domain: checks if the bounding box of the molecule is
     * in the domain.
     * \param box_c (optional): box centroid location, current centroid of the molecule
     * if disregarded
     * \return true if the bounding box is entirely in the domain
     */
    bool                      is_bounding_box_in_domain(const double* box_c = NULL) const;
    /*!
     * \brief operator (): calculates the signed distance to the vdW surface of the molecule
     * (negative outside, positive inside)
     * \param x,y,z: coordinates of the point where the vdW level set function is evaluated
     * \return max over all atoms of radius of atom - distance from atom center to xyz
     */
    double                    operator()(const double x, const double y
                                     #ifdef P4_TO_P8
                                         , const double z
                                     #endif
                                         ) const;
    inline double             get_largest_radius() const{return largest_radius;}
    inline double             get_angstrom_to_domain_factor() const{return angstrom_to_domain;}
    inline int                get_number_of_atoms() const{return atoms.size();}
    inline int                get_number_of_charged_atoms() const{return n_charged_atoms;}
    inline const Atom*        get_atom(int k) const{return &atoms.at(k);}
    inline const Atom*        get_charged_atom(int k) const{return &atoms.at(index_of_charged_atom.at(k));}
    inline const double*      get_centroid() const{return molecule_centroid;}
    inline double             get_side_length_of_bounding_cube() const{return side_length_of_bounding_cube;}
    inline bool               is_scaled() const {return scale_is_set;}
    ~molecule() // delete dynamically allocated memory
    {}
  };

  class SAS_creator
  {
  protected:
    const int           mpi_rank; // because I'm (very) lazy
    const int           mpi_size; // because I'm (very) lazy
    const MPI_Comm      mpi_comm; // because I'm (very) lazy
    const double        phi_sas_lower_bound;
    PetscErrorCode      ierr;
    int                 mpiret;
    vector<int>         global_indices_of_known_values; // for scattering known values to new layout for refined grids
    parStopWatch*       sas_timer;
    parStopWatch*       sub_timer;

    // the refine_fns and reinitialization_weight_fn are not virtual, it's independent of the method
    static p4est_bool_t refine_for_reinitialization_fn(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad);
    static p4est_bool_t refine_for_exact_calculation_fn(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad);
    static int          reinitialization_weight_fn(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant);
    void                scatter_locally(p4est_t*& park);
    void                scatter_to_new_layout(p4est_t*& park, const bool ghost_flag = false);
    void                partition_forest_and_update_sas(p4est_t*& park);
    void                ghost_creation_and_final_partitioning(p4est_t*& park);
    void                refine_and_partition(p4est_t* & park, const int& step_idx);
    void                refine_the_p4est(p4est_t*& park);
    // implementation-dependent refinement and update subroutines
    virtual void        weighted_partition(p4est_t*& park) = 0;
    virtual void        specific_refinement(p4est_t*& park) = 0;
    virtual void        initialization_routine(p4est_t*& park) = 0;
    virtual void        update_phi_sas_and_quadrant_data(p4est_t*& park) = 0;
  public:
    SAS_creator(p4est_t*& park, const bool timing_flag, const bool subtiming_flag);
    void                construct_SAS(p4est_t* & park);
    virtual             ~SAS_creator();
  };

  class SAS_creator_brute_force:public SAS_creator
  {
  private:
    enum {
      query_tag = 159951,
      reply_tag
    };
    struct  receiver_data
    {
      int recv_rank;
      int recv_count;
    };
    struct  query_buffer
    {
      vector<double>          node_coordinates;   // contiguous:    x[n_0]y[n_0]z[n_0]x[n_1]y[n_1]z[n_1]...
      vector<p4est_locidx_t>  node_local_indices; // corr. indices: n_0n_1...
    };
    void        initialization_routine(p4est_t*& park);
    static int  weight_fn(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant);
    void        weighted_partition(p4est_t*& park);
    void        specific_refinement(p4est_t*& park);
    void        update_phi_sas_and_quadrant_data(p4est_t*& park);
  public:
    SAS_creator_brute_force(p4est_t*& park, const bool timing_flag = false, const bool subtiming_flag = false)
      :SAS_creator(park, timing_flag, subtiming_flag)
    {
      if(sas_timer != NULL)
        sas_timer->start("    step 0: initialization ");
      partition_forest_and_update_sas(park);
      if (sas_timer != NULL)
      {
        sas_timer->stop(); sas_timer->read_duration();
      }
    }
    ~SAS_creator_brute_force()
    {} // no dynamically allocated data, but compiler complains otherwise...
  };

  class SAS_creator_list_reduction:public SAS_creator
  {
  private:
    enum {
      query_tag = 951159,
      reply_tag
    };
    struct  receiver_data
    {
      int recv_rank;
      int recv_count;
    };
    struct  query_buffer
    {
      vector<p4est_locidx_t>  off_proc_list_idx;  // map keys of the reduced lists in the proc that owns them...
      vector<p4est_locidx_t>  new_list_idx;       // map keys of the reduced lists when brought back to this proc
      vector<p4est_locidx_t>  local_quad_idx;
    };
    const bool  get_exact_phi;

    void        initialization_routine(p4est_t*& park);
    static int  weight_fn(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant);
    void        weighted_partition(p4est_t*& park);
    static void replace_fn(p4est_t * park, p4est_topidx_t which_tree,
                           int num_outgoing,p4est_quadrant_t * outgoing[],
                           int num_incoming, p4est_quadrant_t * incoming[]);
    void        specific_refinement(p4est_t*& park);
    void        update_phi_sas_and_quadrant_data(p4est_t*& park);
  public:
    SAS_creator_list_reduction(p4est_t*& park, const bool exact_calculations = false, const bool timing_flag = false, const bool subtiming_flag = false)
      :SAS_creator(park, timing_flag, subtiming_flag), get_exact_phi(exact_calculations)
    {
      if (sas_timer != NULL)
        sas_timer->start("    step 0: initialization ");
      my_p4est_biomolecules_t* biomol = (my_p4est_biomolecules_t*) park->user_pointer;
      biomol->update_last_current_level_only = false;
      partition_forest_and_update_sas(park);
      biomol->update_last_current_level_only = true;
      if (sas_timer != NULL)
      {
        sas_timer->stop(); sas_timer->read_duration();
      }
    }
    ~SAS_creator_list_reduction()
    {} // no dynamically allocated data, but compiler complains otherwise...
  };
private:
#ifdef CASL_THROWS
  const par_error_manager   err_manager;
#endif
  // relevant parameters
  parStopWatch*             timer = NULL;
  p4est_t*                  p4est;
  const vector<double>      domain_dim;
  const vector<double>      root_cells_dim;
  const int                 rank_encoding;
  const ulong               max_quad_loc_idx;
  const string              no_vtk = "null";
  int                       global_max_level;         // max level of refinement of the forest in the entire domain
  vector<molecule>          bio_molecules;            // the vector of molecules
  vector<int>               atom_index_offset;        // atom index offset when atoms are serialized
  int                       total_nb_atoms;           // self-explanatory
  int                       index_of_biggest_mol;     // self-explanatory
  double                    box_size_of_biggest_mol;  // self-explanatory
public : double                    angstrom_to_domain;       // angstrom-to-domain conversion factor
  biomol_grid_parameters    parameters;
  // what will be buit
  map<p4est_locidx_t, reduced_list_ptr> old_reduced_lists;  // used for the list reduction method (only)
  vector<reduced_list_ptr>  reduced_lists;                  // used for the list reduction method (only)
  bool                      update_last_current_level_only; // for balanced calculations in list reduction method (only)
  SAS_creator*              sas_creator;
  p4est_nodes_t*            nodes;                    // grid nodes
  p4est_ghost_t*            ghost;                    // ghost cells
  my_p4est_brick_t*           brick;
  my_p4est_hierarchy_t*       hierarchy;
  my_p4est_node_neighbors_t*  neighbors;
  my_p4est_level_set_t*       ls;
  Vec                       phi;                      // node-sampled values of level-set function
  const double*             phi_read_only_p;          // pointer to local data of phi (read only)
  double*                   phi_p;                    // pointer to local data of phi
  Vec                       inner_domain;
  // private methods
  vector<double>      calculate_dimensions_of_root_cells(p4est_t* p4est_);
  vector<double>      calculate_domain_dimensions(p4est_t* p4est_);
  /*!
   * \brief check_if_directory: checks if the folder path is a valid path toward a directory.
   * An exception is thrown if it is not.
   * \param folder_path: self-explanatory
   */
  void                check_if_directory(const string& folder_path) const;
  /*!
   * \brief check_validity_of_vector_of_mol: [activated only in debug] checks if the vector of molecules and
   * other associated class member variables are valid and consistent.
   * An exception is thrown in case of inconsistency.
   */
  void                check_validity_of_vector_of_mol() const;
  /*!
   * \brief are_all_molecules_scaled_consistently: self-explanatory
   * \return true is scaling is consistent (at least one molecule needed, of course)
   */
  bool                are_all_molecules_scaled_consistently() const;
  /*!
   * \brief is_no_molecule_scaled: checks if no scaling at all has been applied yet
   * \return true if all molecules are yet to be scaled
   */
  bool                is_no_molecule_scaled() const;
  /*!
   * \brief get_vector_of_current_centroids: self-explanatory
   * \param current_centroids: reference of the vector of centroid coordinates to be created
   */
  void                get_vector_of_current_centroids(vector<double>& current_centroids);
  /*!
   * \brief rescale_all_molecules: apply the universal scaling factor angstrom_to_domain to all
   * molecules in the vector of molecules.
   * \param new_centroids [optional]: array of new centroid coordinates (array of length
   * P4EST_DIM*nmol()).
   * If disregarded, the current centroids are kept unchanged.
   * If NULL, the centroids are scaled like all other coordinates.
   */
  void                rescale_all_molecules();
  void                rescale_all_molecules(const double* new_centroids);
  /*!
   * \brief add_single_molecule: adds a molecule to the vector of molecules cuurently considered
   * \param file_path: valid path to the pqr file of the molecule to be added;
   * \param centroid [optional]: pointer to the desired centroid of the molecule (double [P4EST_DIM]).
   * If diregarded or NULL, the (possibly scaled) centroid of the molecule is the same as read from
   * the pqr file.
   * \param angles [optional]: pointer to an array of angles (in radians) defining the rotation
   * --> In 2D, the right-handed rotation angle (one value).
   * --> in 3D, the angles psi, theta_n, phi_n (three values), representing a right-handed angle
   * of rotation psi around the axis n pointed by polar and azimuthal angles theta_n and phi_n.
   * If disregarded or NULL, no rotation is applied.
   * \param angstrom_to_domain_ [optional]: pointer to the scaling factor to be used when reading the
   * pqr file.
   * In order to keep and ensure the consistency of the applied (universal) scaling factor,
   * - If this argument is disregarded or NULL,
   *   /\ if no scaling at all has been applied yet, the molecule is added without scaling too;
   *   /\ if a universal scaling holds, on the other hand, it is applied to the added molecule too;
   *  [/\ if none of the above holds, this is a logic error, it is fixed by rescaling all molecules
   *       in release mode, a logic_error is thrown in debug mode;]
   * - if a valid pointer is provided, its value is applied to scale the newly added molecule, of
   *   course, and
   *   /\ all molecules are rescaled accordingly if the value is different from the current universal
   *      scaling factor or if no scaling at all has been applied yet;
   *  [/\ if inconsistent scaling has been detected, this is a logic error, it is fixed by rescaling all
   *      molecules accordingly in release mode, but a logic_error is thrown in debug mode;]
   */
  void                add_single_molecule(const string& file_path, const double* centroid = NULL, double* angles = NULL, const double* angstrom_to_domain_ = NULL);

  int                 find_mol_index(const int& global_atom_index, const int& guess) const;
  const Atom*         get_atom(const int& global_atom_index, int& guess) const;
  p4est_t*            reset_p4est();
  void                update_max_level();
  void                add_reduced_list(p4est_topidx_t which_tree, p4est_quadrant_t* quad, reduced_list_ptr parent_list, const bool &need_exact_phi);
  void                remove_internal_cavities_poisson(const bool export_cavities);
  bool                is_point_in_outer_domain_and_updated(p4est_locidx_t k, quad_neighbor_nodes_of_node_t& qnnn, const my_p4est_node_neighbors_t* ngbd, double*& inner_domain_p) const;
  void                remove_internal_cavities_region_growing(const bool export_cavities);
  struct              inner_box_identifier : public
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
    const my_p4est_biomolecules_t* biomol_pointer;
    double operator()(double x, double y
                  #ifdef P4_TO_P8
                      , double z
                  #endif
                      ) const ;
  } is_point_in_a_bounding_box ;
public:
  static const int    nangle_per_mol; // 3 in 3D, 1 in 2D, the values is set in the .cpp file
  static FILE*        log_file;
  static FILE*        timing_file;
  static FILE*        error_file;
  /*!
   * \brief my_p4est_biomolecules_t: constructor. Reads molecule(s) from a list of files, rotates, translates
   * and scales them if desired.
   * \param p4est_        [required]: a valid pointer to a valid p4est_t;
   * \param mpi_          [required]: a valid pointer to a valid mpi_environment_t
   *                             /!\: the mpi communicator should match the mpicomm of the p4est
   * \param pqr_names     [optional]: pointer to a vector of string(s) representing either
   * 1) the full path to the molecule pqr file(s) (in such a case set input_folder to NULL) or
   * 2) the name of the file(s) to be read in the input folder;
   * (If disregarded or NULL, no molecule is read.)
   * \param input_folder  [optional]: pointer to a string representing the directory where the
   * pqr files are stored.
   * Disregarded if NULL, --> should be NULL if pqr_names points to a vector of FULL path(s) to
   * the pqr files!
   * \param angles        [optional]: pointer to the vector of angle(s) defining the rotation(s)
   * to be applied to the molecule. This vector should be of size nangle_per_mol*nmol() or
   * nangle_per_mol (same rotation applied to all molecules).
   * Definition of the angle(s) (per molecule):
   * In 2D, the right-handed rotation angle (one value).
   * in 3D, the angles psi, theta_n, phi_n (three values), representing a right-handed angle
   * of rotation psi around the axis n pointed by polar and azimuthal angles theta_n and phi_n.
   * If disregarded or NULL, no rotation is applied.
   * \param centroids     [optional]: pointer to the vector of desired centroid(s) where the
   * molecule(s) should be placed. This vector should be of size P4EST_DIM*nmol().
   * If diregarded or NULL, the (possibly scaled) centroid of the molecule is the same as read
   * from the pqr file.
   * \param rel_side_length_biggest_box [optional]: pointer to a parameter representing the
   * desired ratio of the side length of the biggest centroid-centered molecule-bounding cube
   * to the minimal domain dimension, based on which the angstrom_to_domain scaling factor will
   * be calculated and applied. The pointed value must be in )0.0, 1.0(.
   * If disregarded or NULL, no universal scaling is applied.
   */
  my_p4est_biomolecules_t(p4est_t* p4est_, mpi_environment_t* mpi_,
                          const vector<string>* pqr_names = NULL, const string* input_folder = NULL,
                          vector<double>* angles = NULL,
                          const vector<double>* centroids = NULL,
                          const double* rel_side_length_biggest_box = NULL);
  /* overloads the constructor, allows to skip the input_folder argument */
  my_p4est_biomolecules_t(p4est_t* p4est_, mpi_environment_t* mpi_,
                          const vector<string>* pqr_names = NULL,
                          vector<double>* angles = NULL,
                          const vector<double>* centroids = NULL,
                          const double* rel_side_length_biggest_box = NULL) :
    my_p4est_biomolecules_t(p4est_,mpi_, pqr_names, NULL, angles, centroids, rel_side_length_biggest_box){}
  /* overloading the private method for public use, enabling sanity checks */
  void                add_single_molecule(const string& file_path, const vector<double>* centroid = NULL, vector<double>* angles = NULL, const double* angstrom_to_domain = NULL);
  /*!
   * \brief rescale_all_molecules: apply a new desired scaling factor angstrom_to_domain to all
   * molecules in the vector of molecules.
   * \param new_scaling_factor: new desired angstrom_to_domain scaling factor.
   * \param centroids [optional]: vector new centroid coordinates (vector of size P4EST_DIM*nmol()).
   * If disregarded, the current centroids are kept unchanged.
   * If NULL, the centroids are scaled like all other coordinates.
   */
  void                rescale_all_molecules(const double& new_scaling_factor);
  void                rescale_all_molecules(const double& new_scaling_factor, const vector<double>* centroids);
  /*!
   * \brief set_biggest_bounding_box
   * \param biggest_cube_side_length_to_min_domain_size: parameter representing the desired
   * ratio of the side length of the biggest centroid-centered molecule-bounding cube to the
   * minimal domain dimension, based on which the angstrom_to_domain scaling factor will be
   * calculated and applied. The value must be in )0.0, 1.0(.
   * The centroids are unchanged
   */
  void                set_biggest_bounding_box(const double& biggest_cube_side_length_to_min_domain_size);
  /*!
   * \brief print_summary: self-explanatory, writes in the log file
   */
  void                print_summary() const;
  void                set_grid_and_surface_parameters(const int& lmin, const int& lmax, const double& lip_, const double& rp_, const int ooa_);
  void                set_splitting_criterion(const int& lmin, const int& lmax, const double& lip_);
  void                set_probe_radius(const double& rp);
  void                set_order_of_accuracy(const int& ooa);
  double              get_largest_radius_of_all() const;
  /*!
   * \brief operator (): calculates the signed distance to the SAS of the molecule(s)
   * (negative outside, positive inside)
   * \param x,y,z: coordinates of the point where the SAS level set function is evaluated
   * \return max over all atoms of probe_radius + radius of atom - distance from atom center to xyz
   */
  double              operator()(const double x, const double y
                               #ifdef P4_TO_P8
                                 , const double z
                               #endif
                                 ) const;
  double              reduced_operator(const double* xyz, const int& reduced_list_idx, const bool need_exact_value, const bool last_stage) const;
  double              better_distance(const double *xyz, const int& reduced_list_idx, double* kink_point) const;
  void                build_brick();
  void                partition_uniformly(const bool export_cavities, const bool build_ghost = true);
  void                enforce_min_level(const bool export_cavities);
  static int          partition_weight_for_enforcing_min_level(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant);
  static p4est_bool_t refine_fn_min_level(p4est_t *park, p4est_topidx_t which_tree, p4est_quadrant_t *quad);
  static void         replace_fn_min_level(p4est_t *park, p4est_topidx_t which_tree, int num_outgoing, p4est_quadrant_t *outgoing[], int num_incoming, p4est_quadrant_t *incoming[]);

  bool                coarsening_step(int& step_idx, bool export_acceleration);
  static p4est_bool_t coarsen_fn(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad[]);
  static void         set_quad_weight(p4est_quadrant_t* &quad, const p4est_nodes_t* & nodes, const double* const& phi_fct, const double& lower_bound);
  static int          weight_for_coarsening(p4est_t * park, p4est_topidx_t which_tree, p4est_quadrant_t * quadrant);
  void                remove_internal_cavities(const cavity_removal_method& method_to_use = region_growing, const bool export_cavities = false)
  {
    switch (method_to_use) {
    case poisson:
      remove_internal_cavities_poisson(export_cavities);
      break;
    case region_growing:
      remove_internal_cavities_region_growing(export_cavities);
      break;
    default:
#ifdef CASL_THROWSb
      string err_msg = "my_p4est_biomolecules_t::remove_internal_cavities(const my_p4est_node_neighbors_t& ngbd, const cavity_removal_method& method_to_use = region_growing): unknown method...";
      err_manager.print_message_and_abort(err_msg, 314159);
#else
      MPI_Abort(p4est->mpicomm, 314159);
#endif
      break;
    }
  }
  p4est_t*            construct_SES(const sas_generation_method& method_to_use = list_reduction, const bool SAS_timing_flag = false, const bool SAS_subtiming_flag = false, string vtk_folder = "null");
  void                expand_ghost();
  Vec                 return_phi_vector();
  p4est_nodes_t*      return_nodes();
  p4est_ghost_t*      return_ghost();
  void                return_phi_vector_nodes_and_ghost(Vec& phi_out, p4est_nodes_t*& nodes_out, p4est_ghost_t*& ghost_out);
  inline int          nmol() const {return bio_molecules.size();}
  inline int          natoms() const {return total_nb_atoms;}
  ~my_p4est_biomolecules_t();
};


class my_p4est_biomolecules_solver_t{
  const my_p4est_biomolecules_t * const biomolecules;

  typedef enum {
    linearPB,
    nonlinearPB
  } solver_type;
public:
  double        mol_rel_permittivity;                     // relative permittivity of the molecule
  double        elec_rel_permittivity;                    // relative permittivity of the electrolyte
  double        temperature;                              // the temperature (in K)
  double        far_field_ion_density;                    // the far-field ion density (in m^{-3})
  int           ion_charge;                               // 'z' for a z:z symmetrical electrolyte, >=1
  double        solvation_free_energy;                    // the holy grail
  // physical constant
public:
  const double  eps_0             = 8.854187817*1e-12;    // = 1/(mu_0*c^2), exact value, vacuum permittivity in F/m
  const double  kB                = 1.38064853*1e-23;     // (rounded-up) Boltzmann constant in J/K
  const double  electron          = 1.6021766209*1e-19;   // elementary electric charge in C
  const double  meter_to_angstrom = 1e10;
  const double  avogadro_number   = 6.022140858*1e23;     // (rounded-up) Avogadro constant in mol^{-1}

  PetscErrorCode ierr;

  my_p4est_cell_neighbors_t*              cell_neighbors = NULL;
  //my_p4est_poisson_jump_nodes_voronoi_t*  jump_solver = NULL;
  my_p4est_general_poisson_nodes_mls_solver_t* jump_solver= NULL;
  my_p4est_general_poisson_nodes_mls_solver_t* jump_solver1= NULL;
  my_p4est_poisson_nodes_t*               node_solver = NULL;

  enum discretization_scheme_t
  {
    UNDEFINED,
    NO_DISCRETIZATION,
    WALL_DIRICHLET,
    WALL_NEUMANN,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    IMMERSED_INTERFACE,
  };

  Vec           psi_star, psi_naught, psi_bar, validation_error;
  bool          psi_star_psi_naught_and_psi_bar_are_set;
  Vec           psi_hat;
  bool          psi_hat_is_set;
  //Vec eps_grad_n_psi_hat_jump, DIM(eps_grad_n_psi_hat_jump_xx_,eps_grad_n_psi_hat_jump_yy_,eps_grad_n_psi_hat_jump_zz_);

  inline bool   is_molecular_permittivity_set() const {return (mol_rel_permittivity > 1.0-EPS);}
  inline bool   is_electrolyte_permittivity_set() const {return (elec_rel_permittivity > 1.0-EPS);}
  inline bool   are_permittivities_set() const {return (is_electrolyte_permittivity_set() && is_molecular_permittivity_set());}
  inline bool   is_temperature_set() const { return (temperature > EPS);}
  inline bool   is_far_field_ion_density_set() const { return (far_field_ion_density > EPS || far_field_ion_density==0.0);}
  inline bool   is_ion_charge_set() const { return (ion_charge > 0);}
  inline bool   are_all_debye_parameters_set() const {return (is_temperature_set() && is_far_field_ion_density_set() && is_ion_charge_set());}
  inline double length_scale_in_meter() const {return (1.0/(meter_to_angstrom*biomolecules->angstrom_to_domain));}
  bool          are_all_parameters_set() const {return (are_all_debye_parameters_set() && are_permittivities_set());}
  void          make_sure_is_node_sampled(Vec& vector);
  void          solve_singular_part();

  // compute singular charges' contributions
  double        non_dimensional_coulomb_in_mol(double x, double y
                                             #ifdef P4_TO_P8
                                               , double z
                                             #endif
                                               )
  {
    double psi_star_value = 0;
    for (int mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
      {
        const Atom* a = mol.get_charged_atom(charged_atom_idx);
//        std::cout << sqrt(SQR(x - a->x) + SQR(y - a->y) + SQR(z - a->z))/biomolecules->angstrom_to_domain << std::endl;
#ifdef P4_TO_P8
        psi_star_value += (a->q*SQR(electron)*((double) ion_charge))/
            (length_scale_in_meter()*4.0*PI*eps_0*mol_rel_permittivity*kB*temperature*sqrt(SQR(x - a->x) + SQR(y - a->y) + SQR(z - a->z))); // constant = 0
#else
        // there is no real 2D equivalent in terms of electrostatics,
        // in the 2d case, let's consider q a linear (partial) charge density,
        // q is considered to be in electron per nanometer (TOTALLY arbitrary)...
        // psi_star_value += (a->q*0.1*meter_to_angstrom*SQR(electron)*((double) ion_charge)*log(sqrt(SQR(x - a->x) + SQR(y - a->y))))/(2.0*PI*eps_0*mol_rel_permittivity*kB*temperature); // constant = 0
#endif
      }
    }
    return psi_star_value;
  }
  double        non_dimensional_coulomb_in_elec(double x, double y
                                             #ifdef P4_TO_P8
                                               , double z
                                             #endif
                                               )
  {
    double psi_star_value = 0;
    for (int mol_idx = 0; mol_idx < biomolecules->nmol(); ++mol_idx)
    {
      const my_p4est_biomolecules_t::molecule& mol = biomolecules->bio_molecules.at(mol_idx);
      for (int charged_atom_idx = 0; charged_atom_idx < mol.get_number_of_charged_atoms(); ++charged_atom_idx)
      {
        const Atom* a = mol.get_charged_atom(charged_atom_idx);
#ifdef P4_TO_P8
       psi_star_value += ((a->q*SQR(electron)*((double) ion_charge))/
            (length_scale_in_meter()*4.0*PI*eps_0*kB*temperature*2*biomolecules->angstrom_to_domain))*(-1/elec_rel_permittivity);
        //std::cout << "psi_star correction = "<< psi_star_value <<"\n";
#else
        // there is no real 2D equivalent in terms of electrostatics,
        // in the 2d case, let's consider q a linear (partial) charge density,
        // q is considered to be in electron per nanometer (TOTALLY arbitrary)...
        //psi_star_value -= (a->q*0.1*meter_to_angstrom*SQR(electron)*((double) ion_charge)*log(sqrt(SQR(x - a->x) + SQR(y - a->y))))/(2.0*PI*eps_0*elec_rel_permittivity*kB*temperature); // constant = 0

#endif
      }
    }
    return psi_star_value;
  }

  void          return_psi_hat(Vec& psi_hat_out);
  void          return_psi_star_psi_naught_and_psi_bar(Vec& psi_star_out, Vec& psi_naught_out, Vec& psi_bar_out);
  void          calculate_jumps_in_normal_gradient(Vec& eps_grad_n_psi_hat_jump, bool validation_flag);
  void          get_rhs_and_add_plus(Vec& rhs_plus, Vec& add_plus);
  void          get_residual_at_voronoi_points_and_set_as_rhs(const Vec& psi_hat_on_voronoi);
  void          get_linear_diagonal_terms(Vec& pristine_diagonal_terms);
  void          clean_matrix_diagonal(const Vec& pristine_diagonal_terms);

  struct:
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
    double operator()(double, double
                  #ifdef P4_TO_P8
                      , double
                  #endif
                      ) const { return 0.0; }
  } homogeneous_dirichlet_bc_wall_value;

  struct far_field_boundary_cond:
    #ifdef P4_TO_P8
      CF_3
    #else
      CF_2
    #endif
  {
     my_p4est_biomolecules_solver_t*  biomol_solver;
    far_field_boundary_cond( my_p4est_biomolecules_solver_t* biomol_solver):biomol_solver(biomol_solver){}
    double operator()(double x, double y
                  #ifdef P4_TO_P8
                      , double z
                  #endif
                      ) const
    {
        return (biomol_solver->non_dimensional_coulomb_in_mol(DIM(x,y,z)))*biomol_solver->mol_rel_permittivity/biomol_solver->elec_rel_permittivity /*+ biomol_solver->non_dimensional_coulomb_in_elec(DIM(x,y,z))*/;
    }
  };

  struct:
    #ifdef P4_TO_P8
      WallBC3D
    #else
      WallBC2D
    #endif
  {
    BoundaryConditionType operator()(double, double
                                 #ifdef P4_TO_P8
                                     , double
                                 #endif
                                     ) const { return DIRICHLET; }
  } dirichlet_bc_wall_type;

#ifdef P4_TO_P8
  BoundaryConditions3D dirichlet_bc;
#else
  BoundaryConditions2D dirichlet_bc;
#endif

public:
  my_p4est_biomolecules_solver_t(const my_p4est_biomolecules_t* biomolecules_);
  // relative permittivities: coefficients in the poisson jump solver
  void          set_molecular_relative_permittivity(double epsilon_molecule);
  void          set_electrolyte_relative_permittivity(double epsilon_electrolyte);
  void          set_relative_permittivities(double epsilon_molecule, double epsilon_electrolyte);
  void          set_temperature_in_kelvin(double temperature_in_K = 300.0);
  inline void   set_temperature_in_celsius(double temperature_in_C) { set_temperature_in_kelvin(temperature_in_C+273.15);}
  void          set_far_field_ion_density(double n_0);
  //inline void   set_molar_concentration_of_electrolyte_in_mol_per_liter(double conc) { set_far_field_ion_density(1000.0*avogadro_number*conc);}
  inline void   set_molar_concentration_of_electrolyte_in_mol_per_liter(double conc) { set_far_field_ion_density(avogadro_number*conc*1000);}
  void          set_ion_charge(int z = 1);
  void          set_inverse_debye_length_in_meters_inverse(double inverse_debye_length_in_m_inverse);
  inline void   set_inverse_debye_length_in_angstrom_inverse(double inverse_debye_length_in_A_inverse) {set_inverse_debye_length_in_meters_inverse(inverse_debye_length_in_A_inverse*meter_to_angstrom);}
  double        get_inverse_debye_length_in_meters_inverse() const;
  inline double get_inverse_debye_length_in_angstrom_inverse() const {return get_inverse_debye_length_in_meters_inverse()/meter_to_angstrom;}
  inline double get_inverse_debye_length_in_domain() const {return (get_inverse_debye_length_in_angstrom_inverse()/(biomolecules->angstrom_to_domain));}
  inline double get_temperature_in_kelvin() const {return temperature;}
  inline double get_temperature_in_celsius() const {return (get_temperature_in_kelvin() - 273.15);}
  inline double get_far_field_ion_density() const {return far_field_ion_density;}
  inline int    get_ion_charge() const {return ion_charge;}

  void          solve_linear(int n_iter_psi_bar_extension = 20) {(void) solve_nonlinear(1e-8, 1, n_iter_psi_bar_extension);} // equivalent to ONE iteration of the nonlinear solver
  int           solve_nonlinear(double upper_bound_residual = 1e-8, int it_max = 10000, bool validation_flag = false);
  void          get_solvation_free_energy(bool validation_flag = false);
  Vec           get_psi(double max_absolute_psi = DBL_MAX, bool validation_flag = false);

  Vec           return_validation_error();
  Vec           return_residual();
  void          return_all_psi_vectors(Vec& psi_star_out, Vec& psi_naught_out, Vec& psi_bar_out, Vec& psi_hat_out, bool validation_flag = false);
  ~my_p4est_biomolecules_solver_t();
};

#endif // MY_P4EST_BIOMOLECULES_H
