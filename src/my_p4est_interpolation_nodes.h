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

  vector<Vec> Fxxyyzz[P4EST_DIM];
  vector<Vec> Fxxyyzz_block;

  interpolation_method method;


  // rule of three -- disable copy ctr and assignment if not useful
  my_p4est_interpolation_nodes_t(const my_p4est_interpolation_nodes_t& other);
  my_p4est_interpolation_nodes_t& operator=(const my_p4est_interpolation_nodes_t& other);

  void set_input(Vec *F, Vec *Fxxyyzz_block_, DIM(Vec *Fxx_, Vec *Fyy_, Vec *Fzz_), interpolation_method method, const unsigned int &n_vecs_, const unsigned int &block_size_f);

public:
  using my_p4est_interpolation_t::interpolate;

  my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n);

  void update_neighbors(const my_p4est_node_neighbors_t* ngbd_n_);

  using my_p4est_interpolation_t::set_input;
  // -------------- setting inputs without giving second derivatives -------------
  void set_input(Vec *F, interpolation_method method, const unsigned int &n_vecs_, const unsigned int &block_size_f=1);
  inline void set_input(Vec F, interpolation_method method, const unsigned int &block_size_f=1) {set_input(&F, method, 1, block_size_f);}
  inline void set_input(vector<Vec> Fs, interpolation_method method, const unsigned int &block_size_f=1) { set_input(Fs.data(), method, Fs.size(), block_size_f);}
  // -----------------------------------------------------------------------------

  // ------ setting inputs with second derivatives, component by component -------
  inline void set_input(Vec F, DIM(Vec Fxx_, Vec Fyy_, Vec Fzz_), interpolation_method method, const unsigned int &block_size_f=1)
  {
    set_input(&F, NULL, DIM(&Fxx_, &Fyy_, &Fzz_), method, 1, block_size_f);
  }
  inline void set_input(vector<Vec> Fs, DIM(vector<Vec> Fxxs, vector<Vec> Fyys, vector<Vec> Fzzs), interpolation_method method, const unsigned int &block_size_f=1)
  {
    P4EST_ASSERT(ANDD(Fs.size() == Fxxs.size(), Fs.size() == Fyys.size(), Fs.size() == Fzzs.size()));
    set_input(Fs.data(), NULL, DIM(Fxxs.data(), Fyys.data(), Fzzs.data()), method, Fs.size(), block_size_f);
  }
  inline void set_input(Vec *F, DIM(Vec *Fxx_, Vec *Fyy_, Vec *Fzz_), interpolation_method method, unsigned int n_vecs_, const unsigned int &block_size_f=1)
  {
    set_input(F, NULL, DIM(Fxx_, Fyy_, Fzz_), method, n_vecs_, block_size_f);
  }
  // -----------------------------------------------------------------------------

  // ----- setting inputs with P4EST_DIM-block structured second derivatives -----
  // (blocksize of Fxxyyzz_block_ must be P4EST_DIM*block_size_f)
  inline void set_input(Vec F, Vec Fxxyyzz_block_, interpolation_method method, const unsigned int &block_size_f=1)
  {
    set_input(&F, &Fxxyyzz_block_, DIM(NULL, NULL, NULL), method, 1, block_size_f);
  }
  inline void set_input(vector<Vec> Fs, vector<Vec> Fxxyyzz_blocks, interpolation_method method, const unsigned int &block_size_f=1)
  {
    P4EST_ASSERT(Fs.size() == Fxxyyzz_blocks.size());
    set_input(Fs.data(), Fxxyyzz_blocks.data(), DIM(NULL, NULL, NULL), method, Fs.size(), block_size_f);
  }
  inline void set_input(Vec *F, Vec *Fxxyyzz_blocks_, interpolation_method method, unsigned int n_vecs_, const unsigned int &block_size_f=1)
  {
    set_input(F, Fxxyyzz_blocks_, DIM(NULL, NULL, NULL), method, n_vecs_, block_size_f);
  }
  // -----------------------------------------------------------------------------

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
  void operator()(DIM(double x, double y, double z), double *results) const;
  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &comp) const;

};

#endif /* MY_P4EST_INTERPOLATION_NODES */
