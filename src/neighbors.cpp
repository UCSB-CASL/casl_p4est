#include "neighbors.h"

void CellNeighbors::Init(){

  // loop over different faces
  for (short face_id = 0; face_id<P4EST_FACES; ++face_id){
    vector<quad_array>& face_ngbd = ngbd[face_id];

    // loop over all trees
    for (p4est_topidx_t tr = p4est->first_local_tree; tr<=p4est->last_local_tree; ++tr){

      // Get a referrence to the current tree
      p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
      p4est_locidx_t tree_offset = tree->quadrants_offset;

      // loop over all local quadrants
      for (size_t qu = 0; qu<tree->quadrants.elem_count; ++qu){
        p4est_locidx_t qu_locidx = qu + tree_offset;

        // Get a referrence to neighbors for this current quadrant
        vector<p4est_quadrant_t*> &qu_ngbd = face_ngbd[qu_locidx];

        // First check for the type of the neighbors
        p4est_locidx_t idx = P4EST_FACES*qu_locidx + face_id;
        if( q2f[idx] >= -8 && q2f[idx] < 0){ // two small cells
          p4est_locidx_t *q2h_ = (p4est_locidx_t*)sc_array_index(q2h, q2q[idx]);

          qu_ngbd.resize(2);

          // loop over indecies
          for (short i=0; i<2; i++){
            p4est_locidx_t &q = q2h_[i];
            if (q >= 0 && q < n_local){

              p4est_topidx_t tree_id = -1;
              qu_ngbd[i] = p4est_mesh_quadrant_cumulative(p4est, q, &tree_id, NULL);

              qu_ngbd[i]->p.piggy3.local_num = q;
              qu_ngbd[i]->p.piggy3.which_tree = tree_id;

            } else if (q >= n_local && q < n_local + n_ghost){

              qu_ngbd[i] = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q-n_local);
              qu_ngbd[i]->p.piggy3.local_num = q;
            } else {
#ifdef CASL_THROWS
              std::ostringstream oss;
              oss << "[CASL_ERROR]: quadrant '" << q << "' is neither local nor ghost";
              throw std::runtime_error(oss.str());
#endif
            }
          }

        } else if ( q2f[idx] >= 0 && q2f[idx] < 24) { // one same size || one bigger
          // check to see if the quadrant is local or ghost
          p4est_locidx_t &q = q2q[idx];

          qu_ngbd.resize(1);

          if (q >= 0 && q < n_local){

            p4est_topidx_t tree_id = -1;
            qu_ngbd[0] = p4est_mesh_quadrant_cumulative(p4est, q, &tree_id, NULL);

            qu_ngbd[0]->p.piggy3.which_tree = tree_id;
            qu_ngbd[0]->p.piggy3.local_num  = q;

          } else if (q >= n_local && q < n_local + n_ghost){
            qu_ngbd[0] = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q-n_local);
            qu_ngbd[0]->p.piggy3.local_num = q;
          } else {
#ifdef CASL_THROWS
            std::ostringstream oss;
            oss << "[CASL_ERROR]: quadrant '" << q << "' is neither local nor ghost";
            throw std::runtime_error(oss.str());
#endif
          }

        } else {
#ifdef CASL_THROWS
          std::ostringstream oss;
          oss << "[CASL_ERROR]: configuration '" << q2f[idx] << "' is not defined";
          throw std::runtime_error(oss.str());
#endif
        }
      }
    }
  }

  initialized = true;
}


void NodeNeighbors::Init(){

  p4est_t *p4est = cell_ngbds->p4est;
  double rl = static_cast<double>(P4EST_ROOT_LEN);

  p4est_gloidx_t cum_sum = 0;
  for (int r=0; r<p4est->mpirank; ++r)
    cum_sum += nodes->global_owned_indeps[r];

  cout << "offset =  " << cum_sum << endl;
  cout << "nodes->shared_offsets = " << nodes->shared_offsets << endl;
  for (size_t i=0; i<nodes->shared_indeps.elem_count; ++i){
    sc_recycle_array_t *shared_nodes = (sc_recycle_array_t*)sc_array_index(&nodes->shared_indeps, i);
    cout << "shared_nodes->elem_count = " << shared_nodes->elem_count << endl;
    cout << "shared_nodes->f.elem_count = " << shared_nodes->a.elem_count << endl;

    for (size_t j=0; j<shared_nodes->elem_count; ++j){
      p4est_indep_t *shared_node = (p4est_indep_t*)sc_array_index(&shared_nodes->a, j);
      cout << "shared_node->x = " << (double)shared_node->y/rl << " ";
    }
    cout << endl;
  }
}
