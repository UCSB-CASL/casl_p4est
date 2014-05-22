// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/my_p8est_levelset.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/my_p4est_levelset.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

#undef MIN
#undef MAX

using namespace std;

#define POW3(x) (x*x*x)

#ifdef P4_TO_P8
struct:CF_3{
  double operator()(double x, double y, double z) const {
    return 0.45 - sqrt(SQR(x - 1.0) + SQR(y - 1.0) + SQR(z - 1.0));
  }
} circle;

struct:CF_3{
  double operator()(double x, double y, double z) const {
    return POW3(x - 1.0) + POW3(y - 1.0) + POW3(z - 1.0);
  }
} plus_cf;

struct:CF_3{
  double operator()(double x, double y, double z) const {
    return -2.0 - (POW3(x - 1.0) + POW3(y - 1.0) + POW3(z - 1.0));
  }
} minus_cf;
#else
struct:CF_2{
  double operator()(double x, double y) const {
    return 0.45 - sqrt(SQR(x - 1.0) + SQR(y - 1.0));
  }
} circle;

struct:CF_2{
  double operator()(double x, double y) const {
    return POW3(x - 1.0) + POW3(y - 1.0);
  }
} plus_cf;

struct:CF_2{
  double operator()(double x, double y) const {
    return -2.0 - (POW3(x - 1.0) + POW3(y - 1.0));
  }
} minus_cf;
#endif

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  try{
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    cmdParser cmd;
    cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
    cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
    cmd.add_option("lmin", "the min level of the tree");
    cmd.add_option("lmax", "the max level of the tree");
    cmd.add_option("sp", "number of splits to apply to the min and max level");
    cmd.parse(argc, argv);

    // decide on the type and value of the boundary conditions
    BoundaryConditionType bc_wall_type, bc_interface_type;
    int nb_splits, min_level, max_level;
    bc_wall_type      = cmd.get("bc_wtype"  , DIRICHLET);
    bc_interface_type = cmd.get("bc_itype"  , DIRICHLET);
    nb_splits         = cmd.get("sp" , 0);
    min_level         = cmd.get("lmin"      , 3);
    max_level         = cmd.get("lmax"      , 8);

    splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, 1);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    w2.start("initializing the grid");

    /* create the macro mesh */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2, &brick);
#else
    connectivity = my_p4est_brick_new(2, 2, &brick);
#endif

    /* create the p4est */
    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    p4est->user_pointer = (void*)(&data);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    /* partition the p4est */
    p4est_partition(p4est, NULL);

    /* create the ghost layer */
    p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    /* initialize the vectors */
    struct solution_t {
      double minus, plus;
    };

    Vec sol, phi;
#ifdef P4_TO_P8
    const CF_3* sol_cf [] = {&minus_cf, &plus_cf};
#else
    const CF_2* sol_cf [] = {&minus_cf, &plus_cf};
#endif
    ierr = VecCreateGhostNodes(p4est, nodes, &phi);
    ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &sol); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, sol_cf, sol);
    sample_cf_on_nodes(p4est, nodes, circle, phi);

    // copy to local buffers to save as vtk
    std::vector<double> sol_plus(nodes->indep_nodes.elem_count), sol_minus(nodes->indep_nodes.elem_count);

    solution_t *sol_p;
    ierr = VecGetArray(sol, (double**)&sol_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      sol_minus[i] = sol_p[i].minus;
      sol_plus[i]  = sol_p[i].plus;
    }

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_FALSE, P4EST_FALSE,
                           3, 0, "test",
                           VTK_POINT_DATA, "phi",   phi_p,
                           VTK_POINT_DATA, "plus",  &sol_plus[0],
                           VTK_POINT_DATA, "minus", &sol_minus[0]);

    /* destroy p4est objects */
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, (double**)&sol_p); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);

    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy (ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    std::cout << "[" << mpi->mpirank << "]: " << e.what() << std::endl;
  }

  return 0;
}
