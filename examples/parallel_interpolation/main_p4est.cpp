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
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolating_function.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

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
    return x*x + y*y + z*z;
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
    return x*x + y*y;
  }
} uex;

#endif

int main (int argc, char* argv[]){

  try{
    mpi_context_t mpi_context, *mpi = &mpi_context;
    mpi->mpicomm  = MPI_COMM_WORLD;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("lmin", "min level of the tree");
    cmd.add_option("lmax", "max level of the tree");
    cmd.add_option("mode", "interpolation mode 0 = linear, 1 = quadratic, 2 = non-oscilatory quadratic");
    cmd.parse(argc, argv);


#ifdef P4_TO_P8
    circle circ(1, 1, 1, .3);
#else
    circle circ(1, 1, .3);
#endif
    splitting_criteria_cf_t cf_data(cmd.get("lmin", 0), cmd.get("lmax",5), &circ, 1);

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

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

    // Now create the forest
    w2.start("p4est generation");
    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine");
    p4est->user_pointer = (void*)(&cf_data);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // Create the ghost structure
    w2.start("ghost");
    p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    w2.stop(); w2.read_duration();

    // generate the node data structure
    w2.start("creating node structure");
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    w2.start("computing phi");
    Vec phi;
    ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
    Vec u;
    ierr = VecDuplicate(phi, &u); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, circ, phi);
    sample_cf_on_nodes(p4est, nodes, uex, u);
    w2.stop(); w2.read_duration();

    std::ostringstream grid_name; grid_name << P4EST_DIM << "d_grid";
    my_p4est_vtk_write_all(p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           0, 0, grid_name.str().c_str());


    // set up the qnnn neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    std::ostringstream hierarchy_name; hierarchy_name << P4EST_DIM << "d_hierrchy";
    hierarchy.write_vtk(hierarchy_name.str().c_str());
    my_p4est_node_neighbors_t qnnn(&hierarchy, nodes);

    grid_name.str(""); grid_name << P4EST_DIM << "d_grid_qnnn_" << p4est->mpirank << "_" << p4est->mpisize;
    FILE *qFile = fopen(grid_name.str().c_str(), "w");
    for (size_t n=0; n<nodes->num_owned_indeps; n++)
      qnnn[n].print_debug(qFile);
    fclose(qFile);

    Vec u_xx, u_yy;
    double *u_xx_p, *u_yy_p;
    ierr = VecCreateGhost(p4est, nodes, &u_xx); CHKERRXX(ierr);
    ierr = VecCreateGhost(p4est, nodes, &u_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec u_zz;
    double *u_zz_p;
    ierr = VecCreateGhost(p4est, nodes, &u_zz); CHKERRXX(ierr);
#endif

    double u_xx_min, u_xx_max;
    double u_yy_min, u_yy_max;
#ifdef P4_TO_P8
    double u_zz_min, u_zz_max;
//    qnnn.second_derivatives_central(u, u_xx, u_yy, u_zz);
    ierr = VecGetArray(u_zz, &u_zz_p); CHKERRXX(ierr);
#else
    qnnn.second_derivatives_central(u, u_xx, u_yy);
#endif
    ierr = VecGetArray(u_xx, &u_xx_p); CHKERRXX(ierr);
    ierr = VecGetArray(u_yy, &u_yy_p); CHKERRXX(ierr);

    ierr = VecMax(u_xx, NULL, &u_xx_max); CHKERRXX(ierr);
    ierr = VecMin(u_xx, NULL, &u_xx_min); CHKERRXX(ierr);
    ierr = VecMax(u_yy, NULL, &u_yy_max); CHKERRXX(ierr);
    ierr = VecMin(u_yy, NULL, &u_yy_min); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecMax(u_zz, NULL, &u_zz_max); CHKERRXX(ierr);
    ierr = VecMin(u_zz, NULL, &u_zz_min); CHKERRXX(ierr);
#endif

    PetscPrintf(p4est->mpicomm, "d_u_xx = %1.12E\n", u_xx_max - u_xx_min);
    PetscPrintf(p4est->mpicomm, "d_u_yy = %1.12E\n", u_yy_max - u_yy_min);
#ifdef P4_TO_P8
    PetscPrintf(p4est->mpicomm, "d_u_zz = %1.12E\n", u_zz_max - u_zz_min);
#endif

    std::ostringstream oss; oss << P4EST_DIM << "d_phi_" << mpi->mpisize;
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           1 + P4EST_DIM, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "u_xx", u_xx_p,
                           VTK_POINT_DATA, "u_yy", u_yy_p
                       #ifdef P4_TO_P8
                           , VTK_POINT_DATA, "u_zz", u_zz_p
                       #endif
                           );
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(u_xx, &u_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(u_yy, &u_yy_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(u_zz, &u_zz_p); CHKERRXX(ierr);
#else
    ierr = VecRestoreArray(u_xx, &u_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(u_yy, &u_yy_p); CHKERRXX(ierr);
#endif


    // move the circle to create another grid
    cf_data.max_lvl -= 2;
    cf_data.min_lvl += 1;
#ifdef P4_TO_P8
    circ.update(.75, 1.15, .57, .2);
#else
    circ.update(.75, 1.15, .2);
#endif

    // Create a new grid
    w2.start("creating/refining/partitioning new p4est");
    p4est_t *p4est_np1 = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    p4est_np1->user_pointer = (void*)&cf_data;
    p4est_refine(p4est_np1, P4EST_TRUE, refine_levelset_cf, NULL);
    p4est_partition(p4est_np1, NULL);
    w2.stop(); w2.read_duration();

    /*
     * Here we create a new nodes structure. Note that in general if what you
     * want is the same procedure as before. This means if the previous grid
     * included ghost cells in the ghost node struture, usually the new one
     * should also include a NEW ghost structure.
     * Here, however, we do not care about this and simply pass NULL to for the
     * new node structure
     */
    w2.start("creating new node data structure");
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
    w2.stop(); w2.read_duration();

    // Create an interpolating function
    InterpolatingFunction phi_func(p4est, nodes, ghost, brick, &qnnn);

    for (p4est_locidx_t i=0; i<nodes_np1->num_owned_indeps; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, i+nodes_np1->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
      double tree_zmin = connectivity->vertices[3*v_mm + 2];
#endif
      double xyz [P4EST_DIM] =
      {
        node->x == P4EST_ROOT_LEN-1 ? 1.0 + tree_xmin:(double)node->x/(double)P4EST_ROOT_LEN + tree_xmin,
        node->y == P4EST_ROOT_LEN-1 ? 1.0 + tree_ymin:(double)node->y/(double)P4EST_ROOT_LEN + tree_ymin
#ifdef P4_TO_P8
        ,
        node->z == P4EST_ROOT_LEN-1 ? 1.0 + tree_zmin:(double)node->z/(double)P4EST_ROOT_LEN + tree_zmin
#endif
      };

      // buffer the point
      phi_func.add_point_to_buffer(i, xyz);
    }
    PetscSynchronizedFlush(p4est->mpicomm);

    // interpolate on to the new vector
    Vec phi_np1;
    ierr = VecCreateGhost(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    // interpolate
    switch (cmd.get("mode",0)){
    case 0:
      phi_func.set_input_parameters(phi, linear);
      break;
    case 1:
      phi_func.set_input_parameters(phi, quadratic);
      break;
    case 2:
      phi_func.set_input_parameters(phi, quadratic_non_oscillatory);
      break;
    default:
      throw std::invalid_argument("[Error]: Interpolation mode can only be 0, 1, or 2");
    }
    phi_func.interpolate(phi_np1);

    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    oss.str(""); oss << P4EST_DIM << "d_phi_np1_" << mpi->mpisize;

    double *phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    
    my_p4est_vtk_write_all(p4est_np1, nodes_np1, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi_np1", phi_np1_p);

    ierr = VecRestoreArray(phi_np1,  &phi_np1_p); CHKERRXX(ierr);

    // finally, delete PETSc Vecs by calling 'VecDestroy' function
    ierr = VecDestroy(phi);          CHKERRXX(ierr);
    ierr = VecDestroy(phi_np1);  CHKERRXX(ierr);

    ierr = VecDestroy(u);    CHKERRXX(ierr);
    ierr = VecDestroy(u_xx); CHKERRXX(ierr);
    ierr = VecDestroy(u_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(u_zz); CHKERRXX(ierr);
#endif

    // destroy the p4est and its connectivity structure
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy (p4est);

    p4est_nodes_destroy (nodes_np1);
    p4est_destroy (p4est_np1);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

