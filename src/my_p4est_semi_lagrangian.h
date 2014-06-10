#ifndef PARALLEL_SEMI_LAGRANGIAN_H
#define PARALLEL_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <iostream>

#ifdef P4_TO_P8
#include <p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <p4est.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#endif

class SemiLagrangian
{
  p4est_t **p_p4est_, *p4est_;
  p4est_nodes_t **p_nodes_, *nodes_;
  p4est_ghost_t **p_ghost_, *ghost_;
  my_p4est_brick_t *myb_;
  my_p4est_node_neighbors_t *ngbd_;
  my_p4est_hierarchy_t *hierarchy_;
  double cfl;

  std::string partition_name_, topology_name_;

  struct splitting_criteria_update_t : splitting_criteria_t
  {
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    double *phi_tmp;
    my_p4est_hierarchy_t *hierarchy;
    splitting_criteria_update_t( double lip, int min_lvl, int max_lvl,
                                 double *phi,  my_p4est_brick_t *myb,
                                 p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes)
    {
      this->lip = lip;
      this->min_lvl = min_lvl;
      this->max_lvl = max_lvl;
      this->myb = myb;
      this->p4est_tmp = p4est;
      this->ghost_tmp = ghost;
      this->nodes_tmp = nodes;
      this->phi_tmp = phi;
#ifndef P4EST_POINT_LOOKUP
      hierarchy = new my_p4est_hierarchy_t(p4est, ghost, myb);
#endif
    }
    ~splitting_criteria_update_t()
    {
#ifndef P4EST_POINT_LOOKUP
        delete hierarchy;
#endif
    }

    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xyz) const
    {
      return linear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xyz);
    }
  };

  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];

  std::vector<double> local_xyz_departure_dep, non_local_xyz_departure_dep;   //Buffers to hold local and non-local departure points
  PetscErrorCode ierr;

  void advect_from_n_to_np1(double dt,
                          #ifdef P4_TO_P8
                            const CF_3& vx, const CF_3& vy, const CF_3& vz,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                          #else
                            const CF_2& vx, const CF_2& vy,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                          #endif
                            double *phi_np1,p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, bool save_topology = false);

  void advect_from_n_to_np1_CFL(const std::vector<p4est_locidx_t> &map, double dt,
                          #ifdef P4_TO_P8
                            const CF_3& vx, const CF_3& vy, const CF_3& vz,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                          #else
                            const CF_2& vx, const CF_2& vy,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                          #endif
                            double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1);

  void advect_from_n_to_np1(double dt,
                          #ifdef P4_TO_P8
                            Vec vx, Vec vx_xx, Vec vx_yy, Vec vx_zz,
                            Vec vy, Vec vy_xx, Vec vy_yy, Vec vy_zz,
                            Vec vz, Vec vz_xx, Vec vz_yy, Vec vz_zz,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                          #else
                            Vec vx, Vec vx_xx, Vec vx_yy,
                            Vec vy, Vec vy_xx, Vec vy_yy,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                          #endif
                            double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, bool save_topology = false);

  void advect_from_n_to_np1_test(double dt,
                               #ifdef P4_TO_P8
                               Vec vx_nm1, Vec vx_xx_nm1, Vec vx_yy_nm1, Vec vx_zz_nm1,
                               Vec vy_nm1, Vec vy_xx_nm1, Vec vy_yy_nm1, Vec vy_zz_nm1,
                               Vec vz_nm1, Vec vz_xx_nm1, Vec vz_yy_nm1, Vec vz_zz_nm1,
                               Vec vx_n, Vec vx_xx_n, Vec vx_yy_n, Vec vx_zz_n,
                               Vec vy_n, Vec vy_xx_n, Vec vy_yy_n, Vec vy_zz_n,
                               Vec vz_n, Vec vz_xx_n, Vec vz_yy_n, Vec vz_zz_n,
                               Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                               #else
                               Vec vx_nm1, Vec vx_xx_nm1, Vec vx_yy_nm1,
                               Vec vy_nm1, Vec vy_xx_nm1, Vec vy_yy_nm1,
                               Vec vx_n, Vec vx_xx_n, Vec vx_yy_n,
                               Vec vy_n, Vec vy_xx_n, Vec vy_yy_n,
                               Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                               #endif
                               double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, bool save_topology = false);

  void advect_from_n_to_np1_CFL(const std::vector<double> &map, double dt,
                          #ifdef P4_TO_P8
                            Vec vx, Vec vx_xx, Vec vx_yy, Vec vx_zz,
                            Vec vy, Vec vy_xx, Vec vy_yy, Vec vy_zz,
                            Vec vz, Vec vz_xx, Vec vz_yy, Vec vz_zz,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n, Vec phi_zz_n,
                          #else
                            Vec vx, Vec vx_xx, Vec vx_yy,
                            Vec vy, Vec vy_xx, Vec vy_yy,
                            Vec phi_n, Vec phi_xx_n, Vec phi_yy_n,
                          #endif
                            double *phi_np1, p4est_t *p4est_np1, p4est_nodes_t *nodes_np1);

  static p4est_bool_t refine_criteria_sl(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
  {
    splitting_criteria_update_t *data = (splitting_criteria_update_t*) p4est->user_pointer;

    if (quad->level < data->min_lvl)
      return P4EST_TRUE;
    else if (quad->level >= data->max_lvl)
      return P4EST_FALSE;
    else
    {      
      double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
      double d = sqrt(P4EST_DIM)*dx;
      double lip = data->lip;

      /* find the quadrant in p4est_tmp */
      p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
    #ifdef P4_TO_P8
      double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
    #endif

      double xyz [] =
      {
        quad_x_fr_i(quad) + tree_xmin + dx/2.0,
        quad_y_fr_j(quad) + tree_ymin + dx/2.0
  #ifdef P4_TO_P8
        ,
        quad_z_fr_k(quad) + tree_zmin + dx/2.0
  #endif
      };

      p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
      sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
      my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
      sc_array_destroy(remote_matches);
#else
     std::vector<p4est_quadrant_t> remote_matches;
     data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

      p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
      p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
      p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

      double *phi_tmp;
      phi_tmp = data->phi_tmp;

      double f[P4EST_CHILDREN];
      for(short j=0; j<P4EST_CHILDREN; ++j)
      {
        f[j] = phi_tmp[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
        if (fabs(f[j]) <= 0.5*lip*d)
          return P4EST_TRUE;
      }
#ifdef P4_TO_P8
      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
          f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
        return P4EST_TRUE;

      return P4EST_FALSE;
    }
  }


  static p4est_bool_t refine_criteria_with_ghost_sl(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
  {
    splitting_criteria_update_t *data = (splitting_criteria_update_t*) p4est->user_pointer;

    if (quad->level < data->min_lvl)
      return P4EST_TRUE;
    else if (quad->level >= data->max_lvl)
      return P4EST_FALSE;
    else
    {
      double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
      double d = sqrt(P4EST_DIM)*dx;
      double lip = data->lip;

      /* find the quadrant in p4est_tmp */
      p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
    #ifdef P4_TO_P8
      double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
    #endif

      double xyz [] =
      {
        quad_x_fr_i(quad) + tree_xmin + dx/2.0,
        quad_y_fr_j(quad) + tree_ymin + dx/2.0
  #ifdef P4_TO_P8
        ,
        quad_z_fr_k(quad) + tree_zmin + dx/2.0
  #endif
      };
      p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
      sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
      int rank_found = my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
      sc_array_destroy(remote_matches);
#else
     std::vector<p4est_quadrant_t> remote_matches;
     int rank_found = data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

      p4est_locidx_t quad_tmp_idx;
      if(rank_found == p4est->mpirank)
      {
        p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
        quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;
      }
      else if(rank_found != -1)
      {
        quad_tmp_idx = quad_tmp.p.piggy3.local_num + data->p4est_tmp->local_num_quadrants;
      }
      else
      {
        throw std::runtime_error("[ERROR]: semi_lagrangian: this quadrant is not local ...");
      }

      double *phi_tmp;
      phi_tmp = data->phi_tmp;

      double f[P4EST_CHILDREN];
      p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
      for(short j=0; j<P4EST_CHILDREN; ++j)
      {
        f[j] = phi_tmp[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
        if (fabs(f[j]) <= 0.5*lip*d)
          return P4EST_TRUE;
      }

#ifdef P4_TO_P8
      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
          f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
        return P4EST_TRUE;

      return P4EST_FALSE;
    }
  }

  static p4est_bool_t coarsen_criteria_with_ghost_sl(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t **quad)
  {
    splitting_criteria_update_t *data = (splitting_criteria_update_t*)p4est->user_pointer;

    if (quad[0]->level <= data->min_lvl)
      return P4EST_FALSE;
    else if (quad[0]->level > data->max_lvl)
      return P4EST_TRUE;
    else
    {
      p4est_locidx_t quad_tmp_idx[P4EST_CHILDREN];

      double dx = (double)P4EST_QUADRANT_LEN(quad[0]->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
      double d = sqrt(P4EST_DIM)*dx;
      double lip = data->lip;

      /* find the quadrant in p4est_tmp */
      p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];
    #ifdef P4_TO_P8
      double tree_zmin = p4est->connectivity->vertices[3*v_mmm + 2];
    #endif

      double xyz [] =
      {
        quad_x_fr_i(quad[0]) + tree_xmin + dx/2.0,
        quad_y_fr_j(quad[0]) + tree_ymin + dx/2.0
  #ifdef P4_TO_P8
        ,
        quad_z_fr_k(quad[0]) + tree_zmin + dx/2.0
  #endif
      };
#ifdef P4_TO_P8
      for(short k=0; k<2; ++k)
#endif
      for(short j=0; j<2; ++j)
        for(short i=0; i<2; ++i)
        {
#ifdef P4_TO_P8
          short n = 4*k+2*j+i;
          double xyz_tmp [] = { xyz[0] + i*dx, xyz[1] + j*dx, xyz[2] + k*dx };
#else
          short n = 2*j+i;
          double xyz_tmp [] = { xyz[0] + i*dx, xyz[1] + j*dx };
#endif

          p4est_quadrant_t quad_tmp;
    #ifdef P4EST_POINT_LOOKUP
          sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
          int rank_found = my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz_tmp, &quad_tmp, remote_matches);
          sc_array_destroy(remote_matches);
    #else
         std::vector<p4est_quadrant_t> remote_matches;
         int rank_found = data->hierarchy->find_smallest_quadrant_containing_point(xyz_tmp, quad_tmp, remote_matches);
    #endif

          if(rank_found == data->p4est_tmp->mpirank)
          {
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            quad_tmp_idx[n] = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;
          }
          else if(rank_found != -1)
          {
            quad_tmp_idx[n] = quad_tmp.p.piggy3.local_num + data->p4est_tmp->local_num_quadrants;
          }
          else
          {
            throw std::runtime_error("[ERROR]: semi_lagrangian: this quadrant is not local ...");
          }
        }

      double *phi_tmp = data->phi_tmp;

      double f[P4EST_CHILDREN];
      p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
#ifdef P4_TO_P8
      for(short k=0; k<2; ++k)
#endif
      for(short j=0; j<2; ++j)
        for(short i=0; i<2; ++i)
        {
#ifdef P4_TO_P8
          short n = 4*k+2*j+i;
#else
          short n = 2*j+i;
#endif
          f[n] = phi_tmp[ q2n[ quad_tmp_idx[n]*P4EST_CHILDREN + n ] ];
          if (fabs(f[n]) <= 0.5*lip*d)
            return P4EST_FALSE;
        }

#ifdef P4_TO_P8
      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
          f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
        return P4EST_FALSE;

      return P4EST_TRUE;
    }
  }


  /* compute the ghost layer for the intermediate trees, so that the refine operation can be applied */
  /* this does not work due to a bug in p4est ! cf. the coarsen with 4 quadrants owned by different mpiranks */
  double update_p4est_intermediate_trees_with_ghost(const CF_2& vx, const CF_2& vy, Vec &phi);

public:
  SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb, my_p4est_node_neighbors_t *ngbd);

  inline void set_CFL(double cfl) {this->cfl = cfl;}

#ifdef P4_TO_P8
  double compute_dt(const CF_3& vx, const CF_3& vy, const CF_3& vz);
  double compute_dt(Vec vx, Vec vy, Vec vz);
#else
  double compute_dt(const CF_2& vx, const CF_2& vy);
  double compute_dt(Vec vx, Vec vy);
#endif

  inline void set_comm_topology_filenames(const std::string& partition_name, const std::string& topology_name) {
    partition_name_ = partition_name;
    topology_name_  = topology_name;
  }

  /* start from a root tree and successively refine intermediate trees until tree n+1 is built */
#ifdef P4_TO_P8
  void update_p4est_second_order(Vec vx, Vec vy, Vec vz, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
  void update_p4est_second_order_test(Vec vx_nm1, Vec vy_nm1, Vec vz_nm1, Vec vx_n, Vec vy_n, Vec vz_n, double dt, Vec &phi, Vec phi_xx=NULL, Vec phi_yy=NULL, Vec phi_zz=NULL);
  void update_p4est_second_order(const CF_3& vx, const CF_3& vy, const CF_3& vz, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
  double update_p4est_second_order_CFL(Vec vx, Vec vy, Vec vz, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
  double update_p4est_second_order_CFL(const CF_3& vx, const CF_3& vy, const CF_3& vz, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void update_p4est_second_order(Vec vx, Vec vy, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
  void update_p4est_second_order_test(Vec vx_nm1, Vec vy_nm1, Vec vx_n, Vec vy_n, double dt, Vec &phi, Vec phi_xx=NULL, Vec phi_yy=NULL);
  void update_p4est_second_order(const CF_2& vx, const CF_2& vy, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
  double update_p4est_second_order_CFL(Vec vx, Vec vy, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
  double update_p4est_second_order_CFL(const CF_2& vx, const CF_2& vy, double dt, Vec &phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
};

#endif // PARALLEL_SEMI_LAGRANGIAN_H
