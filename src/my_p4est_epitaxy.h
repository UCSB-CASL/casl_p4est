#ifndef MY_P4EST_EPITAXY_H
#define MY_P4EST_EPITAXY_H

#include <vector>

#include <src/types.h>
#include <src/math.h>

#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>

class my_p4est_epitaxy_t
{
private:
  PetscErrorCode ierr;

  class circle_t : public CF_2
  {
  private:
    double xc, yc, r;
    my_p4est_epitaxy_t *prnt;
  public:
    circle_t(double xc, double yc, double r, my_p4est_epitaxy_t *prnt) : xc(xc), yc(yc), r(r), prnt(prnt)
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

  class ZERO : public CF_2
  {
  public:
    double operator()(double, double) const
    {
      return 0;
    }
  } zero;

  my_p4est_brick_t *brick;
  p4est_t *p4est;
  p4est_connectivity_t *connectivity;
  p4est_ghost_t *ghost;
  p4est_nodes_t *nodes;
  my_p4est_hierarchy_t *hierarchy;
  my_p4est_node_neighbors_t *ngbd;

  Vec rho_g;

  std::vector<Vec> phi;
  std::vector<Vec> rho;
  std::vector<Vec> rho_np1;
  std::vector<Vec> v[2];

  double dt_n;
  double D;
  double F;
  double sigma1;
  double sigma1_np1;
  double Nuc;
  double Nuc_np1;
  double L;
  int new_island;
  double island_nucleation_scaling;

  double alpha;
  double rho_avg;
  double rho_avg_np1;
  double rho_sqr_avg;
  double rho_sqr_avg_np1;

  double dxyz[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  void fill_island(const double *phi_p, std::vector<int> &color, int col, size_t n);

public:
  my_p4est_epitaxy_t(my_p4est_node_neighbors_t *ngbd);

  ~my_p4est_epitaxy_t();

  void set_parameters(double D, double F, double alpha);

  inline double get_dt() { return dt_n; }

  inline p4est_t* get_p4est() { return p4est; }

  void compute_velocity();

  void compute_average_islands_velocity();

  void compute_dt();

  void update_grid();

  void solve_rho();

  void update_nucleation();

  void nucleate_new_island();

  /* return true if the time step is fine, false otherwise */
  bool check_time_step();

  void save_vtk(int iter);
};


#endif /* MY_P4EST_EPITAXY_H */
