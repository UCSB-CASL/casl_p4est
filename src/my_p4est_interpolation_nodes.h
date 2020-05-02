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

  void set_input(Vec *F, Vec *Fxxyyzz_block_, Vec *Fxx_, Vec *Fyy_,
               #ifdef P4_TO_P8
                 Vec *Fzz_,
               #endif
                 interpolation_method method, const unsigned int &n_vecs_, const unsigned int &block_size_f);

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
  inline void set_input(Vec F, Vec Fxx_, Vec Fyy_,
               #ifdef P4_TO_P8
                 Vec Fzz_,
               #endif
                 interpolation_method method, const unsigned int &block_size_f=1)
  {
    set_input(&F, NULL, &Fxx_, &Fyy_,
          #ifdef P4_TO_P8
              &Fzz_,
          #endif
              method, 1, block_size_f);
  }
  inline void set_input(vector<Vec> Fs, vector<Vec> Fxxs, vector<Vec> Fyys,
               #ifdef P4_TO_P8
                 vector<Vec> Fzzs,
               #endif
                 interpolation_method method, const unsigned int &block_size_f=1)
  {
    P4EST_ASSERT((Fs.size() == Fxxs.size()) && (Fs.size() == Fyys.size())
             #ifdef P4_TO_P8
                 && (Fs.size() == Fzzs.size())
             #endif
                 );
    set_input(Fs.data(), NULL, Fxxs.data(), Fyys.data(),
          #ifdef P4_TO_P8
              Fzzs.data(),
          #endif
              method, Fs.size(), block_size_f);
  }
  inline void set_input(Vec *F, Vec *Fxx_, Vec *Fyy_,
               #ifdef P4_TO_P8
                 Vec *Fzz_,
               #endif
                 interpolation_method method, unsigned int n_vecs_, const unsigned int &block_size_f=1)
  {
    set_input(F, NULL, Fxx_, Fyy_,
          #ifdef P4_TO_P8
              Fzz_,
          #endif
              method, n_vecs_, block_size_f);
  }
  // -----------------------------------------------------------------------------

  // ----- setting inputs with P4EST_DIM-block structured second derivatives -----
  // (blocksize of Fxxyyzz_block_ must be P4EST_DIM*block_size_f)
  inline void set_input(Vec F, Vec Fxxyyzz_block_, interpolation_method method, const unsigned int &block_size_f=1)
  {
    set_input(&F, &Fxxyyzz_block_, NULL, NULL,
#ifdef P4_TO_P8
              NULL,
#endif
              method, 1, block_size_f);
  }
  inline void set_input(vector<Vec> Fs, vector<Vec> Fxxyyzz_blocks, interpolation_method method, const unsigned int &block_size_f=1)
  {
    P4EST_ASSERT(Fs.size() == Fxxyyzz_blocks.size());
    set_input(Fs.data(), Fxxyyzz_blocks.data(), NULL, NULL,
          #ifdef P4_TO_P8
              NULL,
          #endif
              method, Fs.size(), block_size_f);
  }
  inline void set_input(Vec *F, Vec *Fxxyyzz_blocks_, interpolation_method method, unsigned int n_vecs_, const unsigned int &block_size_f=1)
  {
    set_input(F, Fxxyyzz_blocks_, NULL, NULL,
          #ifdef P4_TO_P8
              NULL,
          #endif
              method, n_vecs_, block_size_f);
  }
  // -----------------------------------------------------------------------------

  // definition of abstract interpolation methods
  using my_p4est_interpolation_t::operator();
#ifdef P4_TO_P8
  void operator()(double x, double y, double z, double *results) const;
#else
  void operator()(double x, double y, double *results) const;
#endif
  void interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &comp) const;

  inline void add_all_nodes(p4est_t *p4est, p4est_nodes_t *nodes)
  {
    double xyz[P4EST_DIM];
    for (p4est_locidx_t n = 0; n < p4est_locidx_t (nodes->indep_nodes.elem_count); ++n)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);
      this->add_point(n, xyz);
    }
  }

  inline void add_all_local_nodes(p4est_t *p4est, p4est_nodes_t *nodes)
  {
    double xyz[P4EST_DIM];
    for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      node_xyz_fr_n(n, p4est, nodes, xyz);
      this->add_point(n, xyz);
    }
  }

};

#endif /* MY_P4EST_INTERPOLATION_NODES */
