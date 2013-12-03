// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/point3.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/point2.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>

using namespace std;

#ifdef P4_TO_P8
struct circle:CF_3{
  circle(double x0_, double y0_, double z0_, double r_)
    : x0(x0_), y0(y0_), z0(z0_), r(r_)
  {}
  void update(double x0_, double y0_, double z0_, double r_)
  {
    x0 = x0_;
    y0 = y0_;
    z0 = z0_;
    r  = r_;
  }

  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
private:
  double x0, y0, z0, r;
};

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);
  }
} uex;

#else
struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update(double x0_, double y0_, double r_)
  {
    x0 = x0_;
    y0 = y0_;
    r  = r_;
  }

  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double x0, y0, r;
};

static struct:CF_2{
  double operator()(double x, double y) const {
    return sin(2*M_PI*x)*sin(2*M_PI*y);
  }
} uex;

#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

struct stat_info_t{
  int local_node_size, local_quad_size, ghost_node_size, ghost_quad_size;
};

#ifdef P4_TO_P8
    typedef Point3 point_t;
#else
    typedef Point2 point_t;
#endif

void generate_random_points(const p4est_t* p4est, const my_p4est_hierarchy_t& hierarchy, p4est_locidx_t num_local, p4est_locidx_t num_remote, std::vector<point_t>& points);
void gather_remote_cells(const p4est_t *p4est, const my_p4est_hierarchy_t& hierarchy, std::vector<const HierarchyCell *> &remotes, std::vector<p4est_topidx_t> &r_trs, p4est_topidx_t tr, p4est_locidx_t q = 0);

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    PetscErrorCode      ierr;

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("qmin", "min number of quadrants");
    cmd.add_option("qmax", "max number of quadrants");
    cmd.add_option("mode", "interpolation mode 0 = linear, 1 = quadratic, 2 = non-oscilatory quadratic");
    cmd.add_option("splits", "number of splits");
    cmd.add_option("alpha", "fraction of total points to be remote (must be in [0,1]). Ignored if -scaled is given");
    cmd.add_option("scaled", "choose a number of remote points that is proportional to number of ghost cells");
    cmd.add_option("write-vtk", "if this flag is set, vtk files will be written to the disk");\
    cmd.add_option("output-dir", "address of the output directory for all I/O");
    cmd.parse(argc, argv);

    const int lmin = cmd.get("lmin", 2);
    const int lmax = cmd.get("lmax", 10);
    const int qmin = cmd.get("qmin", 100);
#ifdef P4_TO_P8
    const int qmax = cmd.get<int>("qmax");
#else
    const int qmax = cmd.get<int>("qmax");
#endif
    const int splits = cmd.get("splits", 0);
    const int mode   = cmd.get("mode", 2);
    const double alpha = cmd.get("alpha", 0.005);
    const bool scaled = cmd.contains("scaled");
    const bool write_vtk = cmd.contains("write-vtk");
    const std::string output_dir = cmd.get<std::string>("output-dir");

    splitting_criteria_random_t data(lmin, lmax, qmin, qmax);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Print the SHA1 of the current commit
    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

    // print basic information
    PetscPrintf(mpi->mpicomm, "mpisize = %d\n"
                "lmin = %d lmax = %d qmin = %d qmax = %d splits = %d mode = %d alpha = %f scaled = %d\n",
                mpi->mpisize, lmin, lmax, qmin, qmax, splits, mode, alpha, scaled);

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t my_brick, *brick = &my_brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2, brick);
#else
    connectivity = my_p4est_brick_new(2, 2, brick);
#endif
    w2.stop(); w2.read_duration();

#ifdef P4_TO_P8
    PetscPrintf(mpi->mpicomm, "brick_size = %dx%dx%d\n", brick->nxyztrees[0], brick->nxyztrees[1], brick->nxyztrees[2]);
#else
    PetscPrintf(mpi->mpicomm, "brick_size = %dx%d\n", brick->nxyztrees[0], brick->nxyztrees[1]);
#endif

    // Now create the forest
    w2.start("p4est generation");
    p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    p4est->user_pointer = (void*)(&data);
    w2.start("refine");
    srand(p4est->mpirank);
    while (p4est->local_num_quadrants < data.max_quads){      
      my_p4est_refine(p4est, P4EST_FALSE, refine_random, NULL);
      my_p4est_partition(p4est, NULL);
    }
    for (int n=0; n<splits; n++)
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    my_p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating node structure");
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    w2.start("gather statistics");
    {
      p4est_gloidx_t num_nodes = 0;
      for (int r =0; r<p4est->mpisize; r++)
        num_nodes += nodes->global_owned_indeps[r];

      std::vector<stat_info_t> stats(p4est->mpisize);

      // make sure all variables are of the size MPI_INT
      stat_info_t my_stat =
      {
        nodes->num_owned_indeps,
        p4est->local_num_quadrants,
        nodes->indep_nodes.elem_count - nodes->num_owned_indeps,
        ghost->ghosts.elem_count
      };
      MPI_Gather(&my_stat, sizeof(stat_info_t), MPI_BYTE, &stats[p4est->mpirank], sizeof(stat_info_t), MPI_BYTE, 0, p4est->mpicomm);

      PetscPrintf(p4est->mpicomm, "%% global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);
      PetscPrintf(p4est->mpicomm, "%% mpi_rank local_node_size local_quad_size ghost_node_size ghost_quad_size\n");
      for (int r=0; r<p4est->mpisize; r++)
        PetscPrintf(p4est->mpicomm, "%4d, %7d, %7d, %5d, %5d\n", r, stats[r].local_node_size, stats[r].local_quad_size, stats[r].ghost_node_size, stats[r].ghost_quad_size);
    }
    w2.stop(); w2.read_duration();

    if (write_vtk){
      std::ostringstream grid_name; grid_name << output_dir << "/" << P4EST_DIM << "d_grid_np_" << p4est->mpisize << "_sp_" << splits;
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             0, 0, grid_name.str().c_str());  
    }    

    // generate the hierarch yand node_neighbors
    w2.start("hierarchy and node neighbors");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    w2.stop(); w2.read_duration();

    // generate a bunch of random points
    w2.start("computing random points");
    std::vector<point_t> points;
    if (p4est->mpisize == 1)
      generate_random_points(p4est, hierarchy, 10*p4est->local_num_quadrants, 0, points);
    else if (scaled)
      generate_random_points(p4est, hierarchy, 10*p4est->local_num_quadrants, 10*ghost->ghosts.elem_count, points);
    else
      generate_random_points(p4est, hierarchy, 10*(1-alpha)*p4est->local_num_quadrants, 10*alpha*p4est->local_num_quadrants, points);
    w2.stop(); w2.read_duration();

    // construct the interpolating function
    Vec u;
    ierr = VecCreateGhostNodes(p4est, nodes, &u); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, uex, u);

    // PETSc logging variables
    PetscLogEvent log_interpolation_all;
    PetscLogEvent log_interpolation_construction;
    PetscLogEvent log_interpolation_add_points;
#ifdef CASL_LOG_EVENTS
    ierr = PetscLogEventRegister("log_interpolation_all                                   ", 0, &log_interpolation_all); CHKERRXX(ierr);
    ierr = PetscLogEventRegister("log_interpolation_construction                          ", 0, &log_interpolation_construction); CHKERRXX(ierr);
    ierr = PetscLogEventRegister("log_interpolation_add_points                            ", 0, &log_interpolation_add_points); CHKERRXX(ierr);    
#endif

    ierr = PetscLogEventBegin(log_interpolation_all, 0, 0, 0, 0); CHKERRXX(ierr);
    ierr = PetscLogEventBegin(log_interpolation_construction, 0, 0, 0, 0); CHKERRXX(ierr);
    parStopWatch w3;

    w3.start("interpolation all");
    w2.start("constructing interpolation");
    InterpolatingFunctionNodeBase interp(p4est, nodes, ghost, brick, &node_neighbors);
    switch (mode){
    case 0:
      interp.set_input_parameters(u, linear);
      break;
    case 1:
      interp.set_input_parameters(u, quadratic);
      break;
    case 2:
      interp.set_input_parameters(u, quadratic_non_oscillatory);
      break;
    default:
      throw std::runtime_error("[ERROR]: invalid interpolation method");
    }
    w2.stop(); w2.read_duration();
    ierr = PetscLogEventEnd(log_interpolation_construction, 0, 0, 0, 0); CHKERRXX(ierr);

    w2.start("adding points");
    ierr = PetscLogEventBegin(log_interpolation_add_points, 0, 0, 0, 0); CHKERRXX(ierr);
    std::vector<double> f(points.size());
    for (size_t i=0; i<points.size(); i++){
#ifdef P4_TO_P8
      double xyz [] = {points[i].x, points[i].y, points[i].z};
#else
      double xyz [] = {points[i].x, points[i].y};
#endif
      interp.add_point_to_buffer(i, xyz);
    }
    w2.stop(); w2.read_duration();
    ierr = PetscLogEventEnd(log_interpolation_add_points, 0, 0, 0, 0); CHKERRXX(ierr);


    w2.start("interpolating");
    interp.interpolate(&f[0]);
    w2.stop(); w2.read_duration();
    w3.stop(); w3.read_duration();
    ierr = PetscLogEventEnd(log_interpolation_all, 0, 0, 0, 0); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    ierr = VecDestroy(u); CHKERRXX(ierr);
    p4est_destroy (p4est);
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

void gather_remote_cells(const p4est_t *p4est, const my_p4est_hierarchy_t& hierarchy,
                         std::vector<const HierarchyCell*>& remotes, std::vector<p4est_topidx_t>& r_trs, p4est_topidx_t tr, p4est_locidx_t q)
{
  const HierarchyCell *c = hierarchy.get_cell(tr, q);
  if (c->child == CELL_LEAF){
    if (c->owner_rank != p4est->mpirank && c->quad == NOT_A_P4EST_QUADRANT){
      remotes.push_back(c);
      r_trs.push_back(tr);
    }
    return;
  } else {
    for (short i=0; i<P4EST_CHILDREN; i++)
      gather_remote_cells(p4est, hierarchy, remotes, r_trs, tr, c->child+i);
  }
}

void generate_random_points(const p4est_t *p4est, const my_p4est_hierarchy_t &hierarchy, p4est_locidx_t num_local, p4est_locidx_t num_remote, std::vector<point_t> &points)
{
  points.resize(num_local + num_remote);

  /* first generate local points: */
  for (p4est_locidx_t i = 0; i<num_local; i++){
    // randomly select a local tree
    p4est_topidx_t tr = ranged_rand_inclusive(p4est->first_local_tree, p4est->last_local_tree);
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);

    // randomly select a quadrant
    p4est_locidx_t q = ranged_rand(0, tree->quadrants.elem_count);
    const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tr];
    points[i].x = p4est->connectivity->vertices[3*v + 0] + ranged_rand(0.01, 0.99)*qh + quad_x_fr_i(quad);
    points[i].y = p4est->connectivity->vertices[3*v + 1] + ranged_rand(0.01, 0.99)*qh + quad_y_fr_j(quad);
#ifdef P4_TO_P8
    points[i].z = p4est->connectivity->vertices[3*v + 2] + ranged_rand(0.01, 0.99)*qh + quad_z_fr_k(quad);
#endif
  }

  /* now generate random remote points. to do this we fist need a list of all remote quadrants
   * that are stored in the hierarchy */
  std::vector<const HierarchyCell*> remotes; remotes.reserve(num_remote);
  std::vector<p4est_topidx_t> r_trs; r_trs.reserve(num_remote);
  for (p4est_topidx_t tr = 0; tr <p4est->connectivity->num_trees; tr++)
    gather_remote_cells(p4est, hierarchy, remotes, r_trs, tr);

  for (p4est_locidx_t i = num_local; i<num_remote+num_local; i++){
    // select a random remote quadrant
    int q = ranged_rand(0, remotes.size());
    const HierarchyCell *c = remotes[q];

    // generate the random point
    double qh = (double)P4EST_QUADRANT_LEN(c->level)/(double)P4EST_ROOT_LEN;
    p4est_topidx_t v = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*r_trs[q]];

    points[i].x = p4est->connectivity->vertices[3*v + 0] + ranged_rand(0.01, 0.99)*qh + (double)(c->imin)/(double)P4EST_ROOT_LEN;
    points[i].y = p4est->connectivity->vertices[3*v + 1] + ranged_rand(0.01, 0.99)*qh + (double)(c->jmin)/(double)P4EST_ROOT_LEN;
#ifdef P4_TO_P8
    points[i].z = p4est->connectivity->vertices[3*v + 2] + ranged_rand(0.01, 0.99)*qh + (double)(c->kmin)/(double)P4EST_ROOT_LEN;
#endif
  }

#ifdef WRITE_VTK_FILES
  std::ostringstream oss; oss << P4EST_DIM << "d_points_" << p4est->mpirank << ".csv";
  FILE *pf = fopen(oss.str().c_str(), "w");
  fprintf(pf, "x, y, z\n");
  for (size_t i=0; i<points.size(); i++){
#ifdef P4_TO_P8
    fprintf(pf, "%lf, %lf, %lf\n", points[i].x, points[i].y, points[i].z);
#else
    fprintf(pf, "%lf, %lf, 0\n", points[i].x, points[i].y);
#endif
  }
  fclose(pf);
#endif
}
