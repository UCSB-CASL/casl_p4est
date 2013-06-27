// p4est Library
#include <p4est.h>
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// My files for this project

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

  {
      int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("[%d]: PID %d on %s ready for attach\n", p4est->mpirank, getpid(), hostname);
      fflush(stdout);
      sleep(10);
  }

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
    P4EST_ASSERT(rank_matched != -1);

    p4est_gloidx_t qu = qu_match.p.piggy3.local_num;
    tr = qu_match.p.piggy3.which_tree;
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
    qu += tree->quadrants_offset + p4est->global_first_quadrant[rank_matched];

    list<Cell> &cells = smallest_cells[ni];
    bool pass = false;
    for (list<Cell>::const_iterator it = cells.begin(); it != cells.end(); ++it)
    {
      if (it->idx == qu) {
        pass = true;
        break;
      }
    }

    if (!pass){
      ostringstream oss;
      oss << "[ERROR, " << p4est->mpirank << "]: In function '" << __FUNCTION__ << "()', node (" << xy[0] << ", " << xy[1] << ") could belong to any of the following quadrants:\n ";
      for (list<Cell>::const_iterator it = cells.begin(); it != cells.end(); ++it)
      {
        oss << it->idx << ",";
      }
      oss << endl;
      oss << "Lookup function returned quadrant " << qu << endl;

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
