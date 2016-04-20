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

  Vec phi_g;
  Vec rho_g;

  std::vector<Vec> phi;
  std::vector<Vec> rho;
  std::vector<Vec> rho_np1;
  std::vector<Vec> v[2];
  std::vector<Vec> island_number;
  std::vector<int> nb_islands_per_level;

  int nb_levels_deleted;

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

  bool one_level_only;
  double lattice_spacing;
  double alpha;
  double rho_avg;
  double rho_avg_np1;
  double rho_sqr_avg;
  double rho_sqr_avg_np1;

  double dxyz[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];

  void compute_phi_g();

  void fill_island(const double *phi_p, double *island_number_p, int col, p4est_locidx_t n);

  void find_connected_ghost_islands(const double *phi_p, double *island_number_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited);

  template<typename T>
  bool contains(const std::vector<T> &vec, const T& elem)
  {
    return find(vec.begin(), vec.end(), elem)!=vec.end();
  }

public:
  my_p4est_epitaxy_t(my_p4est_node_neighbors_t *ngbd);

  ~my_p4est_epitaxy_t();

  void set_parameters(double D, double F, double alpha, double lattice_spacing);

  void set_one_level_only(bool val) { one_level_only = val; }

  inline double get_dt() { return dt_n; }

  inline p4est_t* get_p4est() { return p4est; }

  inline p4est_nodes_t* get_nodes() { return nodes; }

  inline double get_Nuc() { return Nuc; }

  void compute_velocity();

  void compute_average_islands_velocity();

  void compute_dt();

  void update_grid();

  void solve_rho();

  void update_nucleation();

  void nucleate_new_island();

  void compute_islands_numbers();

  /*!
   * \brief check_time_step
   * \return true if the time step is fine, false otherwise
   *    criteria for "time step is fine" are
   *      - rho_g is positive everywhere
   *      - no more than 1 island is to be nucleated
   */
  bool check_time_step();

  /*!
   * \brief compute_coverate
   * \return the fraction of the domain covered by islands
   */
  double compute_coverage();

  void compute_statistics();

  void save_vtk(int iter);
};


#endif /* MY_P4EST_EPITAXY_H */
