#ifndef MY_P4EST_INTERPOLATION_NODES
#define MY_P4EST_INTERPOLATION_NODES

#ifdef P4_TO_P8
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_interpolation.h>
#else
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_interpolation.h>
#endif

using std::vector;

class my_p4est_interpolation_nodes_t : public my_p4est_interpolation_t
{
private:
  p4est_nodes_t *nodes;

  vector<Vec> Fxx, Fyy;
#ifdef P4_TO_P8
  vector<Vec> Fzz;
#endif

  interpolation_method method;

  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_nodes_t(const my_p4est_interpolation_nodes_t& other);
  my_p4est_interpolation_nodes_t& operator=(const my_p4est_interpolation_nodes_t& other);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n);

  void update_neighbors(const my_p4est_node_neighbors_t* ngbd_n_);

  using my_p4est_interpolation_t::set_input;
  void set_input(Vec F, interpolation_method method) {set_input(&F, method, 1);}
  void set_input(vector<Vec> Fs, interpolation_method method) { set_input(Fs.data(), method, Fs.size());}
  void set_input(Vec *F, interpolation_method method, unsigned int n_vecs_);

#ifdef P4_TO_P8
  void set_input(Vec F, Vec Fxx_, Vec Fyy_, Vec Fzz_, interpolation_method method) {set_input(&F, &Fxx_, &Fyy_, &Fzz_, method, 1);}
  void set_input(vector<Vec> Fs, vector<Vec> Fxxs, vector<Vec> Fyys, vector<Vec> Fzzs, interpolation_method method)
  {
    P4EST_ASSERT((Fs.size() == Fxxs.size()) && (Fs.size() == Fyys.size()) && (Fs.size() == Fzzs.size()));
    set_input(Fs.data(), Fxxs.data(), Fyys.data(), Fzzs.data(), method, Fs.size());
  }
  void set_input(Vec* F, Vec *Fxx_, Vec *Fyy_, Vec *Fzz_, interpolation_method method, unsigned int n_vecs_);
#else
  void set_input(Vec F, Vec Fxx_, Vec Fyy_, interpolation_method method) {set_input(&F, &Fxx_, &Fyy_, method, 1);}
  void set_input(vector<Vec> Fs, vector<Vec> Fxxs, vector<Vec> Fyys, interpolation_method method)
  {
    P4EST_ASSERT((Fs.size() == Fxxs.size()) && (Fs.size() == Fyys.size()));
    set_input(Fs.data(), Fxxs.data(), Fyys.data(), method, Fs.size());
  }
  void set_input(Vec *F, Vec *Fxx_, Vec *Fyy_, interpolation_method method, unsigned int n_vecs_);
#endif

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
#ifdef P4_TO_P8
  void operator()(double x, double y, double z, double *results) const;
#else
  void operator()(double x, double y, double *results) const;
#endif
  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results) const;

};

#endif /* MY_P4EST_INTERPOLATION_NODES */
