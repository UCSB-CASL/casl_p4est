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
#include <p8est_communication.h>
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
#include <p4est_communication.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/point2.h>
#endif

#include <src/ipm_logging.h>
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

struct RandomCircles:CF_3{
  RandomCircles(int n_): n(n_), x0(n), y0(n), z0(n), r(n)
  {
    for (int i=0; i<n; i++){
      x0[i] = ranged_rand(0.0, 2.0);
      y0[i] = ranged_rand(0.0, 2.0);
      z0[i] = ranged_rand(0.0, 2.0);
      r[i]  = ranged_rand(0.1, 0.3);
    }
  }

  double operator ()(double x, double y, double z) const {
    double f = -DBL_MAX;
    for (int i=0; i<n; i++)
      f = MAX(r[i] - sqrt(SQR(x-x0[i]) + SQR(y-y0[i]) + SQR(z-z0[i])), f);
    return f;
  }

private:
  int n;
  std::vector<double> x0, y0, z0, r;
};

struct RandomPoints:CF_3{
  RandomPoints(int n_): n(n_), x0(n), y0(n), z0(n)
  {
    for (int i=0; i<n; i++){
      x0[i] = ranged_rand(0.0, 2.0);
      y0[i] = ranged_rand(0.0, 2.0);
      z0[i] = ranged_rand(0.0, 2.0);
    }
  }

  double operator ()(double x, double y, double z) const {
    double f = -DBL_MAX;
    for (int i=0; i<n; i++)
      f = MAX(-sqrt(SQR(x-x0[i]) + SQR(y-y0[i]) + SQR(z-z0[i])), f);
    return f;
  }

private:
  int n;
  std::vector<double> x0, y0, z0;
};

#else
struct RandomCircles:CF_2{
  RandomCircles(int n_): n(n_), x0(n), y0(n), r(n)
  {
    for (int i=0; i<n; i++){
      x0[i] = ranged_rand(0.0, 2.0);
      y0[i] = ranged_rand(0.0, 2.0);
      r[i]  = ranged_rand(0.1, 0.3);
    }
  }

  double operator ()(double x, double y) const {
    double f = -DBL_MAX;
    for (int i=0; i<n; i++)
      f = MAX(r[i] - sqrt(SQR(x-x0[i]) + SQR(y-y0[i])), f);
    return f;
  }

private:
  int n;
  std::vector<double> x0, y0, r;
};

struct RandomPoints:CF_2{
  RandomPoints(int n_): n(n_), x0(n), y0(n)
  {
    for (int i=0; i<n; i++){
      x0[i] = ranged_rand(0.0, 2.0);
      y0[i] = ranged_rand(0.0, 2.0);
    }
  }

  double operator ()(double x, double y) const {
    double f = -DBL_MAX;
    for (int i=0; i<n; i++)
      f = MAX(-sqrt(SQR(x-x0[i]) + SQR(y-y0[i])), f);
    return f;
  }

private:
  int n;
  std::vector<double> x0, y0;
};

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

void generate_random_points(const p4est_t* p4est, const my_p4est_hierarchy_t& hierarchy, p4est_locidx_t num_local, p4est_locidx_t num_remote, std::vector<point_t>& points, bool write_points = false);
void generate_random_points(const p4est_t *p4est, p4est_ghost_t *ghost, p4est_locidx_t num_local, p4est_locidx_t num_remote, std::vector<point_t> &points, bool write_points = false);
void gather_remote_cells(const p4est_t *p4est, const my_p4est_hierarchy_t& hierarchy, std::vector<const HierarchyCell *> &remotes, std::vector<p4est_topidx_t> &r_trs, p4est_topidx_t tr, p4est_locidx_t q = 0);

void mark_random_quadrants(p4est_t *p4est, splitting_criteria_marker_t& markers){

  int lmax = markers.max_lvl;
  int lmin = markers.min_lvl;

  std::vector<double> s(lmax - lmin + 1);
  double sum = 0;
  for (int l=0; l<lmax-lmin+1; l++) {
    s[l] = 1.0/sqrt(l+1.0);
    sum += s[l];
  }

  for (int l=0; l<lmax-lmin+1; l++)
    s[l] /= sum;

  /* using 'volatile' keyword will prevent compiler to optimize the loop, ensuring that all processors
   * will generate the same sequence of random numbers, thus ensuring that the final random forrest is
   * independent of the number of processors
   */
  volatile p4est_bool_t refine;
  for (p4est_gloidx_t i = 0; i<p4est->global_first_quadrant[p4est->mpirank]; i++)
    refine = ranged_rand(0.,1.) < 0.5;
  for (p4est_topidx_t tr = p4est->first_local_tree; tr <= p4est->last_local_tree; tr++){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
    for (size_t qu = 0; qu < tree->quadrants.elem_count; qu++){
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
      p4est_locidx_t q = qu + tree->quadrants_offset;

      markers[q] = ranged_rand(0.,1.) < s[quad->level - lmin];
    }
  }
  for (p4est_gloidx_t i = p4est->global_first_quadrant[p4est->mpirank+1]; i<p4est->global_num_quadrants; i++)
    refine = ranged_rand(0.,1.) < 0.5;
}

std::string output_dir;

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    PetscErrorCode      ierr;

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    int petsc_version_length = 1000;
    char petsc_version[petsc_version_length];
    ierr = PetscGetVersion(petsc_version, petsc_version_length); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi->mpicomm, "Petsc version %s\n", petsc_version); CHKERRXX(ierr);

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("qmin", "min number of quadrants");
    cmd.add_option("qmax", "max number of quadrants");
    cmd.add_option("itmax", "maximum number of iterations when creating random tree for strong scaling");
    cmd.add_option("mode", "interpolation mode 0 = linear, 1 = quadratic, 2 = non-oscilatory quadratic");
    cmd.add_option("splits", "number of splits");
    cmd.add_option("scale-with-nodes", "scale number of random points with nodes rather than quadrants");
    cmd.add_option("alpha", "fraction of total points to be remote (must be in [0,1]). Ignored if -scaled is given");
    cmd.add_option("scaled", "choose a number of remote points that is proportional to number of ghost cells");
    cmd.add_option("write-vtk", "if this flag is set, vtk files will be written to the disk");
    cmd.add_option("output-dir", "address of the output directory for all I/O");
    cmd.add_option("prefactor", "generate this number times number of local/ghost quadrants random points");
    cmd.add_option("repeat", "repeat the experiment this many times");
    cmd.add_option("write-points", "write csv information for the random points");
    cmd.add_option("test", "type of test (weak = 0 and strong = 1, 2)");
    cmd.add_option("nc", "number of randomly placed circles");
    cmd.parse(argc, argv);
    cmd.print();

    output_dir                  = cmd.get<std::string>("output-dir");
    const int lmin              = cmd.get("lmin", 2);
    const int lmax              = cmd.get("lmax", 10);
    const int qmin              = cmd.get("qmin", 100);    
    const int splits            = cmd.get("splits", 0);
    const int mode              = cmd.get("mode", 2);    
    const int prefactor         = cmd.get("prefactor", 50);
    const int repeat            = cmd.get("repeat",1);
    const int test              = cmd.get<int>("test");
    const double alpha          = cmd.get("alpha", 0.005);
    const bool scaled           = cmd.contains("scaled");
    const bool scale_with_nodes = cmd.contains("scale-with-nodes");
    const bool write_vtk        = cmd.contains("write-vtk");    
    const bool write_points     = cmd.contains("write-points");    

    parStopWatch w1;//(parStopWatch::all_timings);
    parStopWatch w2;//(parStopWatch::all_timings);
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    // Print the SHA1 of the current commit
    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

    // print basic information
    PetscPrintf(mpi->mpicomm, "mpisize = %d\n", mpi->mpisize);

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
    w2.start("refine");
    if (test == 0){ // weak scaling test
      const int qmax              = cmd.get<int>("qmax");
      splitting_criteria_random_t random_data(lmin, lmax, qmin, qmax);

      p4est->user_pointer = (void*)(&random_data);
      srand(p4est->mpirank);
      while (p4est->local_num_quadrants < random_data.max_quads){

        my_p4est_refine(p4est, P4EST_FALSE, refine_random, NULL);
        my_p4est_partition(p4est, P4EST_FALSE, NULL);
      }
    } else if (test == 1) { // strong scaling
      while(p4est->global_num_quadrants < qmin){
        // define a globally unique random refinement
        splitting_criteria_marker_t markers(p4est, lmin, lmax, 1.2);
        mark_random_quadrants(p4est, markers);

        p4est->user_pointer = &markers;
        my_p4est_refine(p4est, P4EST_FALSE, refine_marked_quadrants, NULL);
        
        my_p4est_partition(p4est, P4EST_FALSE, NULL);
        
      }
    }
    for (int n=0; n<splits; n++)
      my_p4est_refine(p4est, P4EST_FALSE, refine_every_cell, NULL);
    w2.stop(); w2.read_duration();


    // Finally re-partition
    w2.start("partition");
    
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating node structure");
    
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
    
    w2.stop(); w2.read_duration();

//    char name[1024];
//    sprintf(name, "test_%d", p4est->mpisize);
//    std::vector<double> levels(p4est->local_num_quadrants + ghost->ghosts.elem_count);

//    p4est_locidx_t count = 0;
//    for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it <= p4est->last_local_tree; tr_it++)
//    {
//      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr_it);
//      for(size_t q = 0; q<tree->quadrants.elem_count; q++){
//        const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
//        levels[count++] = quad->level;
//      }
//    }
//    for (size_t q = 0 ; q<ghost->ghosts.elem_count; q++){
//      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
//      levels[count++] = quad->level;
//    }

//    my_p4est_vtk_write_all(p4est, nodes, ghost,
//                           P4EST_TRUE, P4EST_TRUE,
//                           0, 1, name,
//                           VTK_CELL_DATA, "level", &levels[0]);

    w2.start("gather statistics");
    {
      p4est_gloidx_t num_nodes = 0;
      for (int r =0; r<p4est->mpisize; r++)
        num_nodes += nodes->global_owned_indeps[r];

      PetscPrintf(p4est->mpicomm, "%% global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);
      PetscPrintf(p4est->mpicomm, "%% mpi_rank local_node_size local_quad_size ghost_node_size ghost_quad_size\n");
      PetscSynchronizedPrintf(p4est->mpicomm, "%4d, %7d, %7d, %5d, %5d\n", 
        p4est->mpirank, nodes->num_owned_indeps, p4est->local_num_quadrants, nodes->indep_nodes.elem_count-nodes->num_owned_indeps, ghost->ghosts.elem_count);
      PetscSynchronizedFlush(p4est->mpicomm, stdout);
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
//    node_neighbors.init_neighbors(); /* decide if you want to initialize and cache the neighborhood information */
    w2.stop(); w2.read_duration();

    // generate a bunch of random points
    w2.start("computing random points");
    std::vector<point_t> points;    
    int num_local, num_remote;
    if (scale_with_nodes){
      num_local = nodes->num_owned_indeps;
      num_remote = nodes->indep_nodes.elem_count - nodes->num_owned_indeps;
    } else {
      num_local = p4est->local_num_quadrants;
      num_remote = ghost->ghosts.elem_count;
    }

    
#ifdef GHOST_REMOTE_INTERPOLATION
    if (p4est->mpisize == 1)
      generate_random_points(p4est, ghost, prefactor*num_local, 0, points, write_points);
    else if (scaled)
      generate_random_points(p4est, ghost, prefactor*num_local, prefactor*num_remote, points, write_points);
    else
      generate_random_points(p4est, ghost, prefactor*(1-alpha)*num_local, prefactor*alpha*num_local, points, write_points);
#else
    if (p4est->mpisize == 1)
      generate_random_points(p4est, hierarchy, prefactor*num_local, 0, points, write_points);
    else if (scaled)
      generate_random_points(p4est, hierarchy, prefactor*num_local, prefactor*num_remote, points, write_points);
    else
      generate_random_points(p4est, hierarchy, prefactor*(1-alpha)*num_local, prefactor*alpha*num_local, points, write_points);
#endif    
    
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
    parStopWatch w3;//(parStopWatch::all_timings);
    parStopWatch w4;//(parStopWatch::all_timings);
    w4.start("interpolation test");
    for (int i=0; i<repeat; i++){
      w3.start("interpolation all");
      w2.start("constructing interpolation");
      ierr = PetscLogEventBegin(log_interpolation_all, 0, 0, 0, 0); CHKERRXX(ierr);
      ierr = PetscLogEventBegin(log_interpolation_construction, 0, 0, 0, 0); CHKERRXX(ierr);      
      
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
      ierr = PetscLogEventEnd(log_interpolation_add_points, 0, 0, 0, 0); CHKERRXX(ierr);
      w2.stop(); w2.read_duration();      

      w2.start("interpolating");
      interp.interpolate(&f[0]);
      ierr = PetscLogEventEnd(log_interpolation_all, 0, 0, 0, 0); CHKERRXX(ierr);
      w2.stop(); w2.read_duration();
      w3.stop(); w3.read_duration();      
    }
    w4.stop(); w4.read_duration();

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

void generate_random_points(const p4est_t *p4est, const my_p4est_hierarchy_t &hierarchy, p4est_locidx_t num_local, p4est_locidx_t num_remote, std::vector<point_t> &points, bool write_points)
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

  if (write_points){
    std::ostringstream oss; oss << output_dir << "/" << P4EST_DIM << "d_points_" << p4est->mpirank << ".csv";
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
  }
}

void generate_random_points(const p4est_t *p4est, p4est_ghost_t *ghost, p4est_locidx_t num_local, p4est_locidx_t num_remote, std::vector<point_t> &points, bool write_points)
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

  /* now generate remote points: */
  for (p4est_locidx_t i = num_local; i<num_local+num_remote; i++){
    // randomly select a ghost quadrant
    p4est_topidx_t q = ranged_rand(0, ghost->ghosts.elem_count);
    const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t tr = quad->p.piggy3.which_tree;
    p4est_topidx_t v = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tr];
    points[i].x = p4est->connectivity->vertices[3*v + 0] + ranged_rand(0.01, 0.99)*qh + quad_x_fr_i(quad);
    points[i].y = p4est->connectivity->vertices[3*v + 1] + ranged_rand(0.01, 0.99)*qh + quad_y_fr_j(quad);
#ifdef P4_TO_P8
    points[i].z = p4est->connectivity->vertices[3*v + 2] + ranged_rand(0.01, 0.99)*qh + quad_z_fr_k(quad);
#endif
  }

  if (write_points){
    std::ostringstream oss; oss << output_dir << "/" << P4EST_DIM << "d_points_" << p4est->mpirank << ".csv";
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
  }
}
