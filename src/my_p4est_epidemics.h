#ifndef MY_P4EST_EPIDEMICS_H
#define MY_P4EST_EPIDEMICS_H
#include <vector>
#include <src/types.h>
#include <src/math.h>

#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
//#include "nearpt3/nearpt3.cc"
#include </home/pouria/Documents/ann_1.1.2/include/ANN/ANN.h>

struct Tract {
  int id;
  double x;
  double y;
  double density;
  double pop;
  double area;
};


class my_p4est_epidemics_t
{
private:

  PetscErrorCode ierr;

  class circle_t : public CF_2
  {
  private:
    double xc, yc, r;
    my_p4est_epidemics_t *prnt;
  public:
    circle_t(double xc, double yc, double r, my_p4est_epidemics_t *prnt) : xc(xc), yc(yc), r(r), prnt(prnt)
    {
      lip = 1.2;
    }
    double operator()(double x, double y) const
    {
      double d = -4*prnt->L;
      for(int i=-1; i<2; ++i)
        for(int j=-1; j<2; ++j)
        {
          d = MAX(d, r-sqrt( SQR(x+i*prnt->L-xc) + SQR(y+j*prnt->L-yc) ) );
        }
      return d;
    }
  };

  class zero_t : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 1;
    }
  } zero;
  /* grid */
  my_p4est_brick_t *brick;
  p4est_t *p4est;
  p4est_connectivity_t *connectivity;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;


  Vec phi_g;
  std::vector<Vec> phi;
  std::vector<Vec> v[2];

  BoundaryConditionType bc_type;

  double dxyz[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double L;

  /* solutions */
  Vec U_n, U_np1;
  Vec V_n, V_np1;
  Vec W_n, W_np1;


  /* physical parameters */
  double dt_n;
  double R_A;
  double R_B;
  double Xi_A;
  double Xi_B;
  double D_A, D_B, D_AB;

  /* ANN stuff  */
  int k_neighs = 5;             // number of nearest neighbors to draw
  ANNpointArray dataPts;        // data points
  ANNkd_tree* kdTree;
  double R_eff = 0.02;          // effective radius for neighborhood

  std::vector<Tract> tracts;
  std::vector<double> densities;
  double xc_, yc_;
  double Lx_max, Lx_min, Ly_max, Ly_min;
public:

  my_p4est_epidemics_t(my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_epidemics_t();

  void read(const std::string& census);
  void translate(double xc, double yc);
  void unit_scaling();
  void set_density();
  double interp_density(double x, double y);

  void compute_phi_g();

  void set_parameters(double R_A,
                      double R_B,
                      double Xi_A,
                      double Xi_B);



  void set_D(double D_A, double D_B, double D_AB);

  inline double get_dt() { return dt_n; }

  inline p4est_t* get_p4est() { return p4est; }

  inline p4est_nodes_t* get_nodes() { return nodes; }

  void compute_velocity();

  void solve(int iter);

  void compute_dt();

  void update_grid();

  void initialize_infections();

  void save_vtk(int iter);
};


#endif /* MY_P4EST_EPIDEMICS_H */
