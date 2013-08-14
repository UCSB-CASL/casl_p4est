// p4est Library
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
void run_test_1(p4est_t *p4est, const my_p4est_brick_t *brick);
void run_test_2(p4est_t *p4est, const my_p4est_brick_t *brick);
void run_test_3(p4est_t *p4est, const my_p4est_brick_t *brick);

struct Point{
  double x,y;
  p4est_locidx_t owner;
};

struct Cell{
  p4est_quadrant_t *quad;
  p4est_locidx_t idx;
};

using namespace std;
int main (int argc, char* argv[]){

  mpi_context_t       mpi_context, *mpi = &mpi_context;
  mpi->mpicomm = MPI_COMM_WORLD;
  p4est_connectivity_t *connectivity;
  p4est_t              *p4est;

  //    srand(time(0));
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
    run_test_1(p4est, &brick);
    run_test_2(p4est, &brick);
    run_test_3(p4est, &brick);
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
 * \brief run_test_1 tests correctness of the my_p4est_brick_point_lookup_smallest() function
 * \param p4est [in] pointer to the forest object
 * \param brick [in] pointer to the brick connectivity object
 */
void run_test_1(p4est_t *p4est, const my_p4est_brick_t* brick)
{
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex; // gives corners of each tree
  double         *v2q = p4est->connectivity->vertices;       // physical coordinates of each corner

  // generate a set of random points. num_per_quad inside each quadrant
  const int num_per_quad = 10;
  vector<Point> points(p4est->local_num_quadrants * num_per_quad);

  // loop over trees
  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it<= p4est->last_local_tree; ++tr_it){
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);

    p4est_topidx_t tr_mm = t2v[tr_it*P4EST_CHILDREN];
    double tr_xmin = v2q[3*tr_mm + 0];
    double tr_ymin = v2q[3*tr_mm + 1];

    // loop over quadrants
    for (p4est_locidx_t qu_it = 0; qu_it<tree->quadrants.elem_count; ++qu_it){
      p4est_locidx_t qu_locidx = qu_it + tree->quadrants_offset;
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

      // get coordinate and length of the current quadrant
      double ql = int2double_coordinate_transform(P4EST_QUADRANT_LEN(quad->level));
      double qx = int2double_coordinate_transform(quad->x) + tr_xmin;
      double qy = int2double_coordinate_transform(quad->y) + tr_ymin;

      // generate the random points
      for (int i=0; i<num_per_quad; i++){
        Point &p = points[qu_locidx*num_per_quad + i];
        p.x = ranged_rand(qx, qx+ql);
        p.y = ranged_rand(qy, qy+ql);
        p.owner = qu_locidx;
      }
    }
  }

  p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
  // now check if we can find the correct owners
  for (int i=0; i<points.size(); i++){
    p4est_topidx_t which_tree = 0;
    p4est_locidx_t which_quad;

    double xy[] = {points[i].x, points[i].y};
    my_p4est_brick_point_lookup_smallest(p4est, ghost, brick, xy, &which_tree, &which_quad, NULL);

    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, which_tree);
    p4est_locidx_t lookup_match = which_quad + tree->quadrants_offset;
    if (points[i].owner != lookup_match){
      ostringstream oss;
      oss << "[ERROR]: In function '" << __FUNCTION__ << "()', point (" << xy[0] << ", " << xy[1] << ") belongs to quadrant " << points[i].owner << ". Lookup function returned quadrant " << lookup_match << endl;

      p4est_ghost_destroy(ghost);
      throw runtime_error(oss.str());
    }
  }
  p4est_ghost_destroy(ghost);

  cout << "\n ***************************\n";
  cout << " Test1 finished successfully!";
  cout << "\n ***************************\n" << endl;
}

/*!
 * \brief run_test_2 second test is similar to the first except it test points along the edges
 * \param p4est [in] the forest object
 * \param brick [in] brick connectivity macro-mesh
 */
void run_test_2(p4est_t *p4est, const my_p4est_brick_t* brick)
{
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex; // gives corners of each tree
  double         *v2q = p4est->connectivity->vertices;       // physical coordinates of each corner
  p4est_nodes_t  *nodes = my_p4est_nodes_new(p4est);
  p4est_locidx_t *q2n = nodes->local_nodes;

  vector<list<Cell> > smallest_cells(nodes->num_owned_indeps);
  // loop over all local trees
  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it<=p4est->last_local_tree; ++tr_it){
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);

    // loop over all local quadrants
    for(p4est_locidx_t qu_it = 0; qu_it<tree->quadrants.elem_count; ++qu_it){
      p4est_locidx_t qu_locidx = qu_it + tree->quadrants_offset;
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

      // loop over all local nodes of current quadrant
      for (ushort i=0; i<P4EST_CHILDREN; i++){
        p4est_locidx_t ni = q2n[qu_locidx*P4EST_CHILDREN+i] - nodes->offset_owned_indeps;

        list<Cell> &cells = smallest_cells[ni];
        Cell c; c.quad = quad; c.idx = qu_locidx;
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
  for (p4est_locidx_t ni = 0; ni<nodes->num_owned_indeps; ++ni){
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, ni+nodes->offset_owned_indeps);
    p4est_topidx_t tr   = node->p.piggy3.which_tree;

    double tr_xmin = v2q[3*t2v[P4EST_CHILDREN*tr + 0] + 0];
    double tr_ymin = v2q[3*t2v[P4EST_CHILDREN*tr + 0] + 1];

    xy[0] = int2double_coordinate_transform(node->x) + tr_xmin;
    xy[1] = int2double_coordinate_transform(node->y) + tr_ymin;

    // find the quadrant
    p4est_locidx_t qu;
    my_p4est_brick_point_lookup_smallest(p4est, ghost, brick, xy, &tr, &qu, NULL);
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
    qu += tree->quadrants_offset;

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
      oss << "[ERROR]: In function '" << __FUNCTION__ << "()', node (" << xy[0] << ", " << xy[1] << ") could belong to any of the following quadrants:\n ";
      for (list<Cell>::const_iterator it = cells.begin(); it != cells.end(); ++it)
      {
        oss << it->idx << ",";
      }
      oss << endl;
      oss << "Lookup function returned quadrant " << qu << endl;

      p4est_ghost_destroy(ghost);
      throw runtime_error(oss.str());
    }

  }
  p4est_ghost_destroy(ghost);


  cout << "\n ***************************\n";
  cout << " Test2 finished successfully!";
  cout << "\n ***************************\n" << endl;
}


/*!
 * \brief run_test_3 this test is exactly the second test except it directly uses my_p4est_brick_point_lookup function
 * \param p4est [in] the forest object
 * \param brick [in] brick connectivity macro-mesh
 */
void run_test_3(p4est_t *p4est, const my_p4est_brick_t* brick)
{
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex; // gives corners of each tree
  double         *v2q = p4est->connectivity->vertices;       // physical coordinates of each corner
  p4est_nodes_t  *nodes = my_p4est_nodes_new(p4est);
  p4est_locidx_t *q2n = nodes->local_nodes;

  vector<list<Cell> > smallest_cells(nodes->num_owned_indeps);
  // loop over all local trees
  for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it<=p4est->last_local_tree; ++tr_it){
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);

    // loop over all local quadrants
    for(p4est_locidx_t qu_it = 0; qu_it<tree->quadrants.elem_count; ++qu_it){
      p4est_locidx_t qu_locidx = qu_it + tree->quadrants_offset;
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu_it);

      // loop over all local nodes of current quadrant
      for (ushort i=0; i<P4EST_CHILDREN; i++){
        p4est_locidx_t ni = q2n[qu_locidx*P4EST_CHILDREN+i] - nodes->offset_owned_indeps;

        list<Cell> &cells = smallest_cells.at(ni);
        Cell c; c.quad = quad; c.idx = qu_locidx;
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
  for (p4est_locidx_t ni = 0; ni<nodes->num_owned_indeps; ++ni){
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, ni+nodes->offset_owned_indeps);
    p4est_topidx_t tr   = node->p.piggy3.which_tree;

    double tr_xmin = v2q[3*t2v[P4EST_CHILDREN*tr + 0] + 0];
    double tr_ymin = v2q[3*t2v[P4EST_CHILDREN*tr + 0] + 1];

    xy[0] = int2double_coordinate_transform(node->x) + tr_xmin;
    xy[1] = int2double_coordinate_transform(node->y) + tr_ymin;

    // find the quadrant
    p4est_quadrant_t qu_match;
    my_p4est_brick_point_lookup(p4est, ghost, brick, xy, &qu_match, remote_match);
    p4est_locidx_t qu = qu_match.p.piggy3.local_num;
    tr = qu_match.p.piggy3.which_tree;

    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);
    qu += tree->quadrants_offset;

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
      oss << "[ERROR]: In function '" << __FUNCTION__ << "()', node (" << xy[0] << ", " << xy[1] << ") could belong to any of the following quadrants:\n ";
      for (list<Cell>::const_iterator it = cells.begin(); it != cells.end(); ++it)
      {
        oss << it->idx << ",";
      }
      oss << endl;
      oss << "Lookup function returned quadrant " << qu << endl;

      p4est_ghost_destroy(ghost);
      sc_array_destroy(remote_match);

      throw runtime_error(oss.str());
    }

  }
  sc_array_destroy(remote_match);
  p4est_ghost_destroy(ghost);

  cout << "\n ***************************\n";
  cout << " Test3 finished successfully!";
  cout << "\n ***************************\n" << endl;
}
