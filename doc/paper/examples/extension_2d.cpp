// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>


#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_levelset.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_log_wrappers.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

using namespace std;

#ifdef P4_TO_P8
struct circle:CF_3{
  circle() {}
  double operator()(double x, double y, double z) const {
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    return - r + sqrt(SQR(x-0.5) + SQR(y-0.5) + SQR(z-0.5));
  }
private:
  const static double r = 0.05;
};

struct circles:CF_3{
  circles(int n): x0(n), y0(n), z0(n), r(n)
  {
    for (int i=0; i<n; i++) {
      x0[i] = ranged_rand(0.0, 3.0);
      y0[i] = ranged_rand(0.0, 3.0);
      z0[i] = ranged_rand(0.0, 3.0);
      r[i]  = ranged_rand(0.0, 0.3);
    }
  }

  double operator()(double x, double y, double z) const {
    double f = DBL_MAX;
    for (size_t i=0; i<r.size(); i++)
      f = MIN(f, - r[i] + sqrt(SQR(x - x0[i]) + SQR(y - y0[i]) + SQR(z - z0[i])));
    return f;
  }
private:
  std::vector<double> x0, y0, z0, r;
};
#else
struct circle:CF_2{
  circle() {}
  double operator()(double x, double y) const {
    x -= floor(x);
    y -= floor(y);

    return - r + sqrt(SQR(x-0.5) + SQR(y-0.5));
  }
private:
  const static double r = 0.05;
};

struct circles:CF_2{
  circles(int n): x0(n), y0(n), r(n)
  {
    for (int i=0; i<n; i++) {
      x0[i] = ranged_rand(0.0, 3.0);
      y0[i] = ranged_rand(0.0, 3.0);
      r[i]  = ranged_rand(0.0, 0.3);
    }
  }

  double operator()(double x, double y) const {
//    return -.8 + sqrt(SQR(x-1.5) + SQR(y-1.5));
//    double f = -DBL_MAX;
    double f = DBL_MAX;
    for (size_t i=0; i<r.size(); i++)
//      f = MAX(f, r[i] - sqrt(SQR(x - x0[i]) + SQR(y - y0[i])));
      f = MIN(f, - r[i] + sqrt(SQR(x - x0[i]) + SQR(y - y0[i])));
    return f;
  }
private:
  std::vector<double> x0, y0, r;
};
#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

#ifdef P4_TO_P8

struct function_exact_t : CF_3 {
  const CF_3 &func;
  function_exact_t(CF_3 &ls) : func(ls){}
  double operator() (double x, double y, double z) const
  {
    return exp(func(x,y,z));
//    return x*x + y*y;
//    return x + y;
//    return sqrt(ABS(func(x,y)));
  }
};

struct function_to_extend_t : CF_3 {
  const CF_3 &ls;
  const CF_3 &f_ex;
  function_to_extend_t(CF_3 &ls_, CF_3 &f_ex_) : ls(ls_), f_ex(f_ex_){}
  double operator() (double x, double y, double z) const
  {
    return f_ex(x,y,z);
  }
};

#else

struct function_exact_t : CF_2 {
  const CF_2 &func;
  function_exact_t(CF_2 &ls) : func(ls){}
  double operator() (double x, double y) const
  {
    return exp(func(x,y));
//    return x*x + y*y;
//    return x + y;
//    return sqrt(ABS(func(x,y)));
  }
};

struct function_to_extend_t : CF_2 {
  const CF_2 &ls;
  const CF_2 &f_ex;
  function_to_extend_t(CF_2 &ls_, CF_2 &f_ex_) : ls(ls_), f_ex(f_ex_){}
  double operator() (double x, double y) const
  {
    return f_ex(x,y);
  }
};

#endif

#ifdef P4_TO_P8
void check_accuracy(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec q, CF_3 *q_ex, Vec err)
#else
void check_accuracy(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec q, CF_2 *q_ex, Vec err)
#endif
{
  PetscErrorCode ierr;
  double *phi_ptr, *q_ptr, *err_ptr;
  int band = 10;

  /* find dx and dy smallest */
  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data->max_lvl);
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(q  , &q_ptr  ); CHKERRXX(ierr);
  ierr = VecGetArray(err, &err_ptr); CHKERRXX(ierr);

  double err_max = 0;
  double xm=0, ym=0;
#ifdef P4_TO_P8
  double zm=0;
#endif
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(phi_ptr[n]>0 && phi_ptr[n]<diag*band)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
      double x = node_x_fr_i(node) + tree_xmin;
      double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
      double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
      double z = node_z_fr_k(node) + tree_zmin;
      err_ptr[n] = fabs(q_ptr[n] - (*q_ex)(x,y,z));
#else
      err_ptr[n] = fabs(q_ptr[n] - (*q_ex)(x,y));
#endif

      err_max = max(err_max, err_ptr[n]);
    }
    else
      err_ptr[n] = 0;
  }

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(q  , &q_ptr  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(err, &err_ptr); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(err, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (err, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  double err_max_global;
  MPI_Allreduce(&err_max, &err_max_global, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
  if(p4est->mpirank==0)
  {
    PetscPrintf(p4est->mpicomm, "Level : %d / %d\n", data->min_lvl, data->max_lvl);
    PetscPrintf(p4est->mpicomm, "global error extension : %e\n",err_max_global);
#ifdef P4_TO_P8
    PetscPrintf(p4est->mpicomm, "at point (%.10e , %.10e , %.10e)\n", xm, ym, zm);
#else
    PetscPrintf(p4est->mpicomm, "at point (%.10e , %.10e)\n", xm, ym);
#endif
  }
}




int main (int argc, char* argv[])
{
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);
  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  try {
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    p4est_ghost_t      *ghost;
    PetscErrorCode ierr;
    cmdParser cmd;
    cmd.add_option("lmin", "min level");
    cmd.add_option("lmax", "max level");
    cmd.add_option("test", "weak or strong scaling test");
    cmd.add_option("write-vtk", "pass this flag if interested in the vtk files");
    cmd.add_option("output-dir", "parent folder to save everythiong in");
    cmd.add_option("iter", "number of iterations for the reinitialization process");
    cmd.add_option("enable-qnnn-buffer", "if this flag is set qnnns are internally buffered");
    cmd.add_option("order", "order of the extrapolation, 0, 1 or 2");
    cmd.add_option("compute_accuracy", "compute the accuracy of the solution");
    cmd.parse(argc, argv);
    cmd.print();

    const std::string foldername = cmd.get<std::string>("output-dir");
    const int lmin = cmd.get("lmin", 0);
    const int lmax = cmd.get("lmax", 7);
    const bool write_vtk = cmd.contains("write-vtk");
    const string test = cmd.get<string>("test");

    mkdir(foldername.c_str(), 0777);

    PetscPrintf(mpi->mpicomm, "git commit hash value = %s (%s)\n", GIT_COMMIT_HASH_SHORT, GIT_COMMIT_HASH_LONG);

#ifdef P4_TO_P8
    CF_3 *cf = NULL;
#else
    CF_2 *cf = NULL;
#endif

    circle w_cf;
    circles s_cf(100);
#ifdef P4_TO_P8
    int allowed_sizes [] = {16, 128, 432, 1024, 2000, 3456}; // 16*[1:6]^3
#else
//    int allowed_sizes [] = {16, 64, 144, 256, 400, 576, 784, 1024, 1296, 1600, 1936, 2304, 2704, 3136, 3600, 4096}; // 16*[1:16]^2
    int allowed_sizes [] = {1, 2, 4, 8, 16}; // 16*[1:16]^2
#endif
    int nb = -1;

    if (test == "weak"){
      for (unsigned int i=0; i<sizeof(allowed_sizes)/sizeof(allowed_sizes[0]); i++){
        if (mpi->mpisize == allowed_sizes[i]){
          nb = i+1;
          break;
        }
      }
      if (nb == -1)
        throw invalid_argument("[ERROR]: not a valid mpi size");
      cf = &w_cf;
    } else if (test == "strong"){
      nb = 3;
      cf = &s_cf;
    } else {
      throw invalid_argument("test can only be 'weak' or 'strong'");
    }

    function_exact_t function_exact(*cf);
    function_to_extend_t function_to_extend(*cf, function_exact);

    splitting_criteria_cf_t data(lmin, lmax, cf, 1.2);

    parStopWatch w1, w2;
    w1.start("total time");

    // Create the connectivity object
    w2.start("connectivity");
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(nb, nb, nb, &brick);
#else
    connectivity = my_p4est_brick_new(nb, nb, &brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    // Note that we do non-recursive refine + partitioning to ensure that :
    // 1. the work is load-balanced (although this should not be a big deal here ...)
    // 2. and more importantly, we have enough memory for the global grid. If we do not do it this way, the code usually break around level 13 or so.
    w2.start("refine");
    p4est->user_pointer = (void*)(&data);
    for (int l=0; l<lmax; l++){
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, NULL);
    }
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    my_p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // create the ghost layer
    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    // generate the node data structure
    nodes = my_p4est_nodes_new(p4est, ghost);

    // Initialize the level-set function
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *cf, phi);

    // Initialize the function to extend
    Vec q;
    ierr = VecDuplicate(phi, &q); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, function_to_extend, q);

    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
    if(cmd.contains("enable-qnnn-buffer"))
      node_neighbors.init_neighbors();

    my_p4est_level_set level_set(&node_neighbors);

    w2.start("Extending the quantity over interface");
    level_set.extend_Over_Interface_TVD(phi, q, cmd.get("iter", 10), cmd.get("order", 2));
    w2.stop(); w2.read_duration();

    Vec err;
    if(cmd.contains("compute_accuracy"))
    {
      ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
      check_accuracy(p4est, nodes, phi, q, &function_exact, err);
    }

    if (write_vtk){
      w2.start("Saving vtk");
      std::ostringstream grid_name;
      grid_name << foldername << "/" << P4EST_DIM << "d_" << p4est->mpisize << "_"
                << brick.nxyztrees[0] << "x"
                << brick.nxyztrees[1]
             #ifdef P4_TO_P8
                << "x" << brick.nxyztrees[2]
             #endif
                   ;
      cout << grid_name.str() << endl;

      double *phi_p, *q_p;
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
      ierr = VecGetArray(q, &q_p); CHKERRXX(ierr);
      if(cmd.contains("compute_accuracy"))
      {
        double *err_p;
        ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               3, 0, grid_name.str().c_str(),
                               VTK_POINT_DATA, "phi", phi_p,
                               VTK_POINT_DATA, "q", q_p,
                               VTK_POINT_DATA, "error", err_p);
        ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
      }
      else
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               2, 0, grid_name.str().c_str(),
                               VTK_POINT_DATA, "phi", phi_p,
                               VTK_POINT_DATA, "q", q_p);
      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);
      w2.stop(); w2.read_duration();
    }

    if(cmd.contains("compute_accuracy")) ierr = VecDestroy(err); CHKERRXX(ierr);
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(q); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    p4est_ghost_destroy(ghost);
    p4est_nodes_destroy(nodes);
    p4est_destroy(p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();
  } catch (const std::exception& e) {
    PetscSynchronizedFPrintf(mpi->mpicomm, stderr, "[%d] %s\n", mpi->mpirank, e.what());
    PetscSynchronizedFlush(mpi->mpicomm);
    MPI_Abort(mpi->mpicomm, MPI_ERR_UNKNOWN);
  }

  return 0;
}
