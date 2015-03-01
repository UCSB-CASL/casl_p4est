#include "my_p4est_faces.h"


my_p4est_faces_t::my_p4est_faces_t(p4est_t *p4est, p4est_ghost_t *ghost, my_p4est_brick_t *myb, my_p4est_cell_neighbors_t *ngbd_c)
{
  this->p4est = p4est;
  this->ghost = ghost;
  this->myb = myb;
  this->ngbd_c = ngbd_c;

  init_faces();
}

void my_p4est_faces_t::init_faces()
{
  q2u_.resize(P4EST_FACES);
  for(int d=0; d<P4EST_FACES; ++d)
    q2u_[d].resize(p4est->local_num_quadrants + ghost->ghosts.elem_count, NO_VELOCITY);

  unsigned int nb_u = 0;
  unsigned int nb_v = 0;

  vector<p4est_quadrant_t> ngbd;

  /* first process local velocities */
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;

      if(is_quad_xmWall(p4est, tree_idx, quad))
        q2u_[dir::f_m00][quad_idx] = nb_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, -1, 0);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level || /* the neighbor is a bigger cell */
             (q2u_[dir::f_p00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY && /* the shared face has not been indexed yet */
              ( ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants || /* ngbd is local */
                ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants > ghost->proc_offsets[p4est->mpirank] ) ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_m00][quad_idx] = nb_u++;
          else
            q2u_[dir::f_m00][quad_idx] = q2u_[dir::f_p00][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_xpWall(p4est, tree_idx, quad))
        q2u_[dir::f_p00][quad_idx] = nb_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 1, 0);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level || /* the neighbor is a bigger cell */
             (q2u_[dir::f_m00][ngbd[0].p.piggy3.local_num]==NO_VELOCITY && /* the shared face has not been indexed yet */
              ( ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants || /* ngbd is local */
                ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants > ghost->proc_offsets[p4est->mpirank] ) ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_p00][quad_idx] = nb_u++;
          else
            q2u_[dir::f_p00][quad_idx] = q2u_[dir::f_m00][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_ymWall(p4est, tree_idx, quad))
        q2u_[dir::f_0m0][quad_idx] = nb_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, -1);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level || /* the neighbor is a bigger cell */
             (q2u_[dir::f_0p0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY && /* the shared face has not been indexed yet */
              ( ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants || /* ngbd is local */
                ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants > ghost->proc_offsets[p4est->mpirank] ) ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_0m0][quad_idx] = nb_v++;
          else
            q2u_[dir::f_0m0][quad_idx] = q2u_[dir::f_0p0][ngbd[0].p.piggy3.local_num];
        }
      }

      if(is_quad_ypWall(p4est, tree_idx, quad))
        q2u_[dir::f_0p0][quad_idx] = nb_u++;
      else
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 0, 1);
        if(ngbd.size()==1)
        {
          if(ngbd[0].level>quad->level || /* the neighbor is a bigger cell */
             (q2u_[dir::f_0m0][ngbd[0].p.piggy3.local_num]==NO_VELOCITY && /* the shared face has not been indexed yet */
              ( ngbd[0].p.piggy3.local_num < p4est->local_num_quadrants || /* ngbd is local */
                ngbd[0].p.piggy3.local_num-p4est->local_num_quadrants > ghost->proc_offsets[p4est->mpirank] ) ) ) /* ngbd is on process with larger index */
            q2u_[dir::f_0p0][quad_idx] = nb_v++;
          else
            q2u_[dir::f_0p0][quad_idx] = q2u_[dir::f_0m0][ngbd[0].p.piggy3.local_num];
        }
      }
    }
  }

  u2q_.resize(P4EST_DIM);
  u2q_[0].resize(nb_u);
  u2q_[1].resize(nb_v);

  for(p4est_locidx_t quad_idx=0; quad_idx<p4est->local_num_quadrants; ++quad_idx)
  {
    if(q2u_[dir::f_m00][quad_idx] != NO_VELOCITY)
      u2q_[0][q2u_[dir::f_m00][quad_idx]] = quad_idx;

    if(q2u_[dir::f_p00][quad_idx] != NO_VELOCITY)
      u2q_[0][q2u_[dir::f_p00][quad_idx]] = quad_idx;

    if(q2u_[dir::f_0m0][quad_idx] != NO_VELOCITY)
      u2q_[1][q2u_[dir::f_0m0][quad_idx]] = quad_idx;

    if(q2u_[dir::f_0p0][quad_idx] != NO_VELOCITY)
      u2q_[1][q2u_[dir::f_0p0][quad_idx]] = quad_idx;
  }
}

