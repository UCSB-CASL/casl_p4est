#ifndef PARALLEL_SEMI_LAGRANGIAN_H
#define PARALLEL_SEMI_LAGRANGIAN_H

#include <vector>
#include <algorithm>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <iostream>

#include <p4est.h>
#include <src/refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>

class SemiLagrangian
{
  p4est_t **p_p4est_, *p4est_;
  p4est_nodes_t **p_nodes_, *nodes_;
  p4est_ghost_t **p_ghost_, *ghost_;
  my_p4est_brick_t *myb_;

  struct splitting_criteria_update_t : splitting_criteria_t
  {
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    std::vector<double> *phi_tmp;
    splitting_criteria_update_t( double lip, int min_lvl, int max_lvl,
                                 std::vector<double> *phi,  my_p4est_brick_t *myb,
                                 p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes )
    {
      this->lip = lip;
      this->min_lvl = min_lvl;
      this->max_lvl = max_lvl;
      this->myb = myb;
      this->p4est_tmp = p4est;
      this->ghost_tmp = ghost;
      this->nodes_tmp = nodes;
      this->phi_tmp = phi;
    }
    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xy) const
    {
      return bilinear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xy);
    }
  };

  double xmin, xmax, ymin, ymax;

  std::vector<double> local_xy_departure_dep, non_local_xy_departure_dep;   //Buffers to hold local and non-local departure points
  PetscErrorCode ierr;

  double compute_dt(const CF_2& vx, const CF_2& vy);

  void advect_from_n_to_np1(const CF_2& vx, const CF_2& vy, double dt, Vec phi_n, double *phi_np1,
                            p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, my_p4est_node_neighbors_t &qnnn);

  void advect_from_n_to_np1(Vec vx, Vec vy, double dt, Vec phi_n, double *phi_np1,
                            p4est_t *p4est_np1, p4est_nodes_t *nodes_np1, my_p4est_node_neighbors_t &qnnn);

  static p4est_bool_t refine_criteria_sl(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
  {
    splitting_criteria_update_t *data = (splitting_criteria_update_t*) p4est->user_pointer;

    if (quad->level < data->min_lvl)
      return P4EST_TRUE;
    else if (quad->level >= data->max_lvl)
      return P4EST_FALSE;
    else
    {
      double dx, dy;
      dx_dy_dz_quadrant(p4est, which_tree, quad, &dx, &dy, NULL);
      double d = sqrt(dx*dx + dy*dy);
      double lip = data->lip;

      /* find the quadrant in p4est_tmp */
      double xy [] = { (double)quad->x/(double)P4EST_ROOT_LEN,
                       (double)quad->y/(double)P4EST_ROOT_LEN };

      c2p_coordinate_transform(p4est, which_tree, &xy[0], &xy[1], NULL);
      xy[0] += dx/2;
      xy[1] += dy/2;

      p4est_quadrant_t quad_tmp;
      sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
      my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xy, &quad_tmp, remote_matches);
      sc_array_destroy(remote_matches);

      p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
      p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
      p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

      double *phi_tmp;
      phi_tmp = data->phi_tmp->data();

      double f[4];
      for(short j=0; j<P4EST_CHILDREN; ++j)
      {
        f[j] = phi_tmp[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
        if (fabs(f[j]) <= 0.5*lip*d)
          return P4EST_TRUE;
      }

      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
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
      double dx, dy;
      dx_dy_dz_quadrant(p4est, which_tree, quad, &dx, &dy, NULL);
      double d = sqrt(dx*dx + dy*dy);
      double lip = data->lip;

      /* find the quadrant in p4est_tmp */
      double xy [] = { (double)quad->x/(double)P4EST_ROOT_LEN,
                       (double)quad->y/(double)P4EST_ROOT_LEN };

      c2p_coordinate_transform(p4est, which_tree, &xy[0], &xy[1], NULL);
      xy[0] += dx/2;
      xy[1] += dy/2;

      p4est_quadrant_t quad_tmp;
      int rank_found = my_p4est_brick_point_lookup(data->p4est_tmp, data->ghost_tmp, data->myb, xy, &quad_tmp, NULL);

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
        throw std::runtime_error("semi_lagrangian: this quadrant is not local ...");
      }

      double *phi_tmp;
      phi_tmp = data->phi_tmp->data();

      double f[4];
      p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
      for(short j=0; j<P4EST_CHILDREN; ++j)
      {
        f[j] = phi_tmp[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
        if (fabs(f[j]) <= 0.5*lip*d)
          return P4EST_TRUE;
      }

      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
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
      double dx, dy;
      dx_dy_dz_quadrant(p4est, which_tree, quad[0], &dx, &dy, NULL);
      double d = 2*sqrt(dx*dx + dy*dy);
      double lip = data->lip;

      double xy [] = { (double)quad[0]->x/(double)P4EST_ROOT_LEN,
                       (double)quad[0]->y/(double)P4EST_ROOT_LEN };
      c2p_coordinate_transform(p4est, which_tree, &xy[0], &xy[1], NULL);

      p4est_locidx_t quad_tmp_idx[P4EST_CHILDREN];

      xy[0] += dx/2;
      xy[1] += dy/2;

      for(short j=0; j<2; ++j)
        for(short i=0; i<2; ++i)
        {
          short n = 2*j+i;
          double xy_tmp [] = { xy[0] + i*dx, xy[1] + j*dy };

          p4est_quadrant_t quad_tmp;
          int rank_found = my_p4est_brick_point_lookup(data->p4est_tmp, data->ghost_tmp, data->myb, xy_tmp, &quad_tmp, NULL);

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
            throw std::runtime_error("semi_lagrangian: this quadrant is not local ...");
          }
        }

      double *phi_tmp = data->phi_tmp->data();

      double f[4];
      p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
      for(short j=0; j<2; ++j)
        for(short i=0; i<2; ++i)
        {
          short n = 2*j+i;
          f[n] = phi_tmp[ q2n[ quad_tmp_idx[n]*P4EST_CHILDREN + 2*j + i ] ];
          if (fabs(f[n]) <= 0.5*lip*d)
            return P4EST_FALSE;
        }

      if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0)
        return P4EST_FALSE;

      return P4EST_TRUE;
    }
  }


public:
  SemiLagrangian(p4est_t **p4est, p4est_nodes_t **nodes, p4est_ghost_t **ghost, my_p4est_brick_t *myb);

  void update_p4est_second_order(Vec vx, Vec vy, Vec &phi, double dt);

  /* start from a root tree and successively refine intermediate trees until tree n+1 is built */
  void update_p4est_second_order(const CF_2& vx, const CF_2& vy, Vec &phi, double dt);

  /* compute the ghost layer for the intermediate trees, so that the refine operation can be applied */
  /* this does not work due to a bug in p4est ! cf. the coarsen with 4 quadrants owned by different mpiranks */
  double update_p4est_intermediate_trees_with_ghost(const CF_2& vx, const CF_2& vy, Vec &phi);
};

#endif // PARALLEL_SEMI_LAGRANGIAN_H
