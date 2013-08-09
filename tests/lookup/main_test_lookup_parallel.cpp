// p4est Library
#include <p4est.h>
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <list>

// casl_p4est
#include <src/utilities.h>
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/semi_lagrangian.h>
#include <src/refine_coarsen.h>

#include <mpi/mpi.h>

// function prototype
void run_test(p4est_t *p4est, const my_p4est_brick_t *brick);

struct Point{
  double x,y;
  p4est_locidx_t owner;
};

struct Cell{
  p4est_quadrant_t *quad;
  p4est_gloidx_t idx;
};

using namespace std;
int main (int argc, char* argv[]){

  mpi_context_t       mpi_context, *mpi = &mpi_context;
  mpi->mpicomm = MPI_COMM_WORLD;
  p4est_connectivity_t *connectivity;
  p4est_t              *p4est;

  //  srand(time(0));
  rand_grid_data_t data = {9, 1, 5000, 100};

  Session::init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  // Create the connectivity object
  w2.start("connectivity");
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);
  w2.stop(); w2.read_duration();

  // Now create the forest
  w2.start("p4est generation");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();
  p4est->user_pointer = (void*)(&data);

  w2.start("refinement/coarsening");
  for (int i=0; i<data.max_lvl; i++){
    data.counter = 0;
    p4est_refine (p4est, P4EST_FALSE, refine_random , NULL);
  }
  w2.stop(); w2.read_duration();

  w2.start("grid partitioning");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  my_p4est_vtk_write_all(p4est, NULL, 1.0,
                         0, 0, "grid");

  // run tests
  try {
    run_test(p4est, &brick);
  } catch (const exception& e) {
    cerr << e.what() << endl;
  }

  // destroy the p4est and its connectivity structure
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

/*!
 * \brief run_test this is similar to test_3 in the serial version, except it is run in parallel
 * \param p4est [in] the forest object
 * \param brick [in] brick connectivity macro-mesh
 */
void run_test(p4est_t *p4est, const my_p4est_brick_t* brick)
{

  //  {
  //    int i = 0;
  //    char hostname[256];
  //    gethostname(hostname, sizeof(hostname));
  //    printf("[%d]: PID %d on %s ready for attach\n", p4est->mpirank, getpid(), hostname);
  //    fflush(stdout);
  //    sleep(10);
  //  }

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex; // gives corners of each tree
  double         *v2q = p4est->connectivity->vertices;       // physical coordinates of each corner
  p4est_nodes_t  *nodes = my_p4est_nodes_new(p4est);
  p4est_locidx_t *q2n = nodes->local_nodes;

  p4est_locidx_t num_indep = nodes->indep_nodes.elem_count, num_hanging = nodes->face_hangings.elem_count;
  p4est_locidx_t num_nodes = num_indep + num_hanging;
  vector<list<Cell> > smallest_cells(num_nodes);

  // loop over all local trees
  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it<=p4est->last_local_tree; ++tr_it){
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);

    // loop over all local quadrants
    for(p4est_locidx_t qu_it = 0; qu_it<tree->quadrants.elem_count; ++qu_it){
      p4est_locidx_t qu_locidx = qu_it + tree->quadrants_offset;
      p4est_gloidx_t qu_gloidx = qu_locidx + p4est->global_first_quadrant[p4est->mpirank];
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

      // loop over all local nodes of current quadrant
      for (ushort i=0; i<P4EST_CHILDREN; i++){
        p4est_locidx_t ni = q2n[qu_locidx*P4EST_CHILDREN+i];

        list<Cell> &cells = smallest_cells[ni];
        Cell c; c.quad = quad; c.idx = qu_gloidx;
        if (cells.empty() || cells.back().quad->level == c.quad->level){
          cells.push_back(c);
          continue;
        } else {
          for (list<Cell>::iterator it = cells.begin(); it != cells.end(); ++it){
            if (it->quad->level < c.quad->level){
              cells.clear();
              cells.push_back(c);
              break;
            }
          }
        }
      }
    }
  }

  // loop over trees
  double xy[2];
  p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
  sc_array_t *remote_match = sc_array_new(sizeof(p4est_quadrant_t));

  p4est_indep_t *node_indep   = NULL;
  p4est_hang2_t *node_hanging = NULL;
  for (p4est_locidx_t ni = 0; ni<num_nodes; ++ni){
    p4est_topidx_t tr = 0;

    bool is_indep   = ni >= 0 && ni < num_indep;
    bool is_hanging = ni >= num_indep && ni < num_nodes;

    if (is_indep){
      node_indep   = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, ni);
      tr = node_indep->p.piggy3.which_tree;
    } else if (is_hanging) {
      node_hanging = (p4est_hang2_t*)sc_array_index(&nodes->face_hangings, ni-num_indep);
      tr = node_hanging->p.piggy.which_tree;
    } else {
      P4EST_ASSERT(false);
    }

    P4EST_ASSERT(tr>=0 && tr<=p4est->connectivity->num_trees);

    double tr_xmin = v2q[3*t2v[P4EST_CHILDREN*tr + 0] + 0];
    double tr_ymin = v2q[3*t2v[P4EST_CHILDREN*tr + 0] + 1];

    if (is_indep){
      xy[0] = int2double_coordinate_transform(node_indep->x) + tr_xmin;
      xy[1] = int2double_coordinate_transform(node_indep->y) + tr_ymin;
    } else if (is_hanging) {
      xy[0] = int2double_coordinate_transform(node_hanging->x) + tr_xmin;
      xy[1] = int2double_coordinate_transform(node_hanging->y) + tr_ymin;
    }

    // find the quadrant
    p4est_quadrant_t qu_match;
    int rank_matched = my_p4est_brick_point_lookup(p4est, ghost, brick, xy, &qu_match, remote_match);

    // We must chech to see if we need to search remote processors
    // notify receivers that we will be sending them info and figure out who is going to send us stuff!
    int *recvers = new int[remote_match->elem_count], *senders = new int[p4est->mpisize];
    int num_recvers = remote_match->elem_count, num_senders;

    for (int i=0; i<num_recvers; i++){
      p4est_quadrant_t *remote_quad = (p4est_quadrant_t*)sc_array_index(remote_match, i);
      recvers[i] = remote_quad->p.piggy1.owner_rank;
    }

    sc_notify(recvers, num_recvers, senders, &num_senders, p4est->mpicomm);

    // Great! now we know who we are sending to (recvers) and receiving from (senders)
    vector<double> xy_recvd(2*num_senders);      // store xy location of points recvd from other processors
    vector<int> tr_recvd(num_senders);           // store tr index to be search for recvd points
    vector<int> qu_gloidx_sendbuf(num_senders);  // send buffer for the quadrants that are found on this processor and should be shipped back
    vector<int> qu_gloidx_recvbuf(num_recvers);  // recv buffer for the quadrants that are found on other processors and must be recvd
    vector<int> qu_level_sendbuf (num_senders);  // send buffer to store the level of the quadrant for comparison
    vector<int> qu_level_recvbuf (num_recvers);  // recv buffer to store the level of the quadrant for comparison

    // send processors the xy location of a point to look for
    for (int i=0; i<num_recvers; i++){
      MPI_Send(xy, 2, MPI_DOUBLE, recvers[i], 0, p4est->mpicomm);
      MPI_Send(&tr, 1, MPI_INT, recvers[i], 1, p4est->mpicomm);
    }

    // recieve the data from other processors
    for (int i=0; i<num_senders; i++){
      MPI_Recv(&xy_recvd[2*i], 2, MPI_DOUBLE, senders[i], 0, p4est->mpicomm, NULL);
      MPI_Recv(&tr_recvd[i], 1, MPI_INT, senders[i], 1, p4est->mpicomm, NULL);
    }

    // Now every processor knows what poinst to look up further for others
    sc_array *remote_dummy = sc_array_new(sizeof(p4est_quadrant_t));
    for (int i=0; i<num_senders; i++){
      double xy_tmp [] = {xy_recvd[2*i + 0], xy_recvd[2*i + 1]};
      p4est_quadrant_t qu;
      my_p4est_brick_point_lookup(p4est, ghost, brick, xy_tmp, &qu, remote_dummy);

      p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, qu.p.piggy3.which_tree);
      qu_gloidx_sendbuf[i] = qu.p.piggy3.local_num + tree->quadrants_offset + p4est->global_first_quadrant[p4est->mpirank];
      qu_level_sendbuf[i]  = qu.level;
    }
    sc_array_destroy(remote_dummy);

    // So now every procssor has performed the requests made by the other processor,
    // they should start sending information back to the owners.
    for (int i=0; i<num_senders; i++){
      MPI_Send(&qu_gloidx_sendbuf[i], 1, MPI_INT, senders[i], 0, p4est->mpicomm);
      MPI_Send(&qu_level_sendbuf[i],  1, MPI_INT, senders[i], 1, p4est->mpicomm);
    }

    // now recieve data from processors
    for (int i=0; i<num_recvers; i++){
      MPI_Recv(&qu_gloidx_recvbuf[i], 1, MPI_INT, recvers[i], 0, p4est->mpicomm, NULL);
      MPI_Recv(&qu_level_recvbuf[i],  1, MPI_INT, recvers[i], 1, p4est->mpicomm, NULL);
    }

    // Ok now every processor has asked about all the remote quadrants and has recieved
    // all the answers it requires. We now need to compare these results against the local
    // stuff to make a final decision
    p4est_gloidx_t qu_gloidx;
    if (rank_matched != -1){
      qu_gloidx = qu_match.p.piggy3.local_num;
      tr = qu_match.p.piggy3.which_tree;
      p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
      qu_gloidx += tree->quadrants_offset + p4est->global_first_quadrant[p4est->mpirank];
    }

    int imin = 0;
    for (int i=0; i<num_recvers; i++){
      if (qu_level_recvbuf[i] > qu_level_recvbuf[imin]){
        imin = i;
      }
    }

    if (rank_matched == -1 || (num_recvers != 0 && qu_level_recvbuf[imin] > qu_match.level) ) { //smallest remote quad is smaller than the local one
      qu_gloidx = qu_gloidx_recvbuf[imin];
    }

    delete [] recvers;
    delete [] senders;

    qu_level_sendbuf.clear();
    qu_level_recvbuf.clear();
    qu_gloidx_sendbuf.clear();
    qu_gloidx_recvbuf.clear();
    xy_recvd.clear();
    tr_recvd.clear();

    list<Cell> &cells = smallest_cells[ni];
    bool pass = false;
    for (list<Cell>::const_iterator it = cells.begin(); it != cells.end(); ++it){
      if (it->idx == qu_gloidx) {
        pass = true;
        break;
      }
    }

    if (!pass){
      ostringstream oss;
      oss << "[ERROR, " << p4est->mpirank << "]: In function '" << __FUNCTION__ << "()', node (" << xy[0] << ", " << xy[1] << ") could belong to any of the following quadrants:\n ";
      for (list<Cell>::const_iterator it = cells.begin(); it != cells.end(); ++it){
        oss << it->idx << ",";
      }
      oss << endl;
      oss << "Lookup function returned quadrant " << qu_gloidx << endl;

      cerr << oss.str() << endl;

//      p4est_ghost_destroy(ghost);
//      sc_array_destroy(remote_match);
//      throw runtime_error(oss.str());
    }

  }
  sc_array_destroy(remote_match);
  p4est_ghost_destroy(ghost);

  cout << "\n ***************************\n";
  cout << " Test finished successfully!";
  cout << "\n ***************************\n" << endl;
}

