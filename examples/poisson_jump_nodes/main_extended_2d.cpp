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
#include <src/my_p8est_poisson_jump_nodes_extended.h>
#include <src/my_p8est_level_set.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_jump_nodes_extended.h>
#include <src/my_p4est_level_set.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>

#undef MIN
#undef MAX

using namespace std;

#define CUBIC_TEST 0
#define COS_TEST 1

#define TEST COS_TEST

#define POW3(x) (x)*(x)*(x)
const static double mue_p = 5.0;
const static double mue_m = 1.0;

#ifdef P4_TO_P8
static struct:WallBC3D{
  BoundaryConditionType operator()(double x, double y, double z) const {
    (void) x;
    (void) y;
    (void) z;

    return DIRICHLET;
  }
} bc_wall_type;

struct circle_t:CF_3{
  double x0, y0, z0, r0;
  circle_t(double x, double y, double z, double r): x0(x), y0(y), z0(z), r0(r) {}
  double operator()(double x, double y, double z) const {
    return r0 - sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
  }
};

static circle_t circle(0.56, 1.23, 0.68, 0.35);

static struct:CF_3{
  // make sure to change dn and laplacian if you changed this
  double operator()(double x, double y, double z) const {
#if TEST == CUBIC_TEST
    return 5*POW3(x - 1.0) - POW3(y - 1.0) + 3*POW3(z - 1.0);
#elif TEST == COS_TEST
    return cos(M_PI*x)*cos(M_PI*y)*cos(M_PI*z);
#endif
  }

  double dn(double x, double y, double z) const {
    double nx = x - circle.x0;
    double ny = y - circle.y0;
    double nz = z - circle.z0;
    double abs = MAX(EPS, sqrt(nx*nx + ny*ny + nz*nz));
    nx /= abs; ny /= abs; nz /= abs;
#if TEST == CUBIC_TEST
    return mue_p*(  5*3*SQR(x - 1.0)*nx
                  - 3*SQR(y - 1.0)*ny
                  + 3*3*SQR(z - 1.0)*nz);
#elif TEST == COS_TEST
    return mue_p*(-M_PI*sin(M_PI*x)*cos(M_PI*y)*cos(M_PI*z)*nx
                  -M_PI*cos(M_PI*x)*sin(M_PI*y)*cos(M_PI*z)*ny
                  -M_PI*cos(M_PI*x)*cos(M_PI*y)*sin(M_PI*z)*nz);
#endif
  }
} plus_cf;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
#if TEST == CUBIC_TEST
    return -mue_p*(5*3*2*(x - 1.0) - 3*2*(y - 1.0) + 3*3*2*(z - 1.0));
#elif TEST == COS_TEST
    return  mue_p*3.0*M_PI*M_PI*cos(M_PI*x)*cos(M_PI*y)*cos(M_PI*z);
#endif
  }
} rhs_plus_cf;

static struct:CF_3{
  // make sure to change dn if you changed this
  double operator()(double x, double y, double z) const {
#if TEST == CUBIC_TEST
    return 1.0 - (2*POW3(x - 1.0) + POW3(y - 1.0) + POW3(z - 1.0));
#elif TEST == COS_TEST
    return 1.0 - sin(x*y*z*M_PI);
#endif
  }
  double dn(double x, double y, double z) const {
    double nx = x - circle.x0;
    double ny = y - circle.y0;
    double nz = z - circle.z0;
    double abs = MAX(EPS, sqrt(nx*nx + ny*ny + nz*nz));
    nx /= abs; ny /= abs; nz /= abs;

#if TEST == CUBIC_TEST
    return -mue_m*( 2*3*SQR(x - 1.0)*nx
                  + 3*SQR(y - 1.0)*ny
                  + 3*SQR(z - 1.0)*nz);
#elif TEST == COS_TEST
    return -mue_m*M_PI*cos(x*y*z*M_PI)*(y*z*nx + x*z*ny + x*y*nz);
#endif
  }
} minus_cf;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    if (circle(x,y,z) > 0)
      return plus_cf(x,y,z);
    else
      return minus_cf(x,y,z);
  }
} bc_wall_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
#if TEST == CUBIC_TEST
    return mue_m*(2*3*2*(x - 1.0) + 3*2*(y - 1.0) + 3*2*(z - 1.0));
#elif TEST == COS_TEST
    return -mue_m*M_PI*M_PI*(SQR(x*y)+SQR(x*z)+SQR(y*z))*sin(x*y*z*M_PI);
#endif
  }
} rhs_minus_cf;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return plus_cf(x,y,z) - minus_cf(x,y,z);
  }
} jump_sol;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return plus_cf.dn(x,y,z) - minus_cf.dn(x,y,z);
  }
} jump_dn_sol;

#else
static struct:WallBC2D{
  BoundaryConditionType operator()(double x, double y) const {
    (void) x;
    (void) y;
    return DIRICHLET;
  }
} bc_wall_type;

struct circle_t:CF_2{
  double x0, y0, r0;
  circle_t(double x, double y, double r): x0(x), y0(y), r0(r) {}
  double operator()(double x, double y) const {
    return r0 - sqrt(SQR(x - x0) + SQR(y - y0));
  }
};

static circle_t circle(1.38, 0.61, 0.35);

static struct:CF_2{

  // make sure to change dn if you changed this
  double operator()(double x, double y) const {
#if TEST == CUBIC_TEST
    return 5*POW3(x - 1.0) - POW3(y - 1.0);
#elif TEST == COS_TEST
    return cos(M_PI*x)*cos(M_PI*y);
#endif
  }

  double dn(double x, double y) const {
    double nx = x - circle.x0;
    double ny = y - circle.y0;
    double abs = MAX(EPS, sqrt(nx*nx + ny*ny));
    nx /= abs; ny /= abs;

#if TEST == CUBIC_TEST
    return mue_p*(  5*3*SQR(x - 1.0)*nx
                  - 3*SQR(y - 1.0)*ny);
#elif TEST == COS_TEST
    return mue_p*(-M_PI*sin(M_PI*x)*cos(M_PI*y) * nx
                  -M_PI*cos(M_PI*x)*sin(M_PI*y) * ny);
#endif
  }
} plus_cf;

static struct:CF_2{
  double operator()(double x, double y) const {
#if TEST == CUBIC_TEST
    return -mue_p*(5*3*2*(x - 1.0) - 3*2*(y - 1.0));
#elif TEST == COS_TEST
    return  mue_p*2.0*M_PI*M_PI*cos(M_PI*x)*cos(M_PI*y);
#endif
  }
} rhs_plus_cf;

static struct:CF_2{
  // make sure to change dn if you changed this
  double operator()(double x, double y) const {
#if TEST == CUBIC_TEST
    return 1.0 - (2*POW3(x - 1.0) + POW3(y - 1.0));
#elif TEST == COS_TEST
    return 1.0 - sin(x*y*M_PI);
#endif
  }

  double dn(double x, double y) const {
    double nx = x - circle.x0;
    double ny = y - circle.y0;
    double abs = MAX(EPS, sqrt(nx*nx + ny*ny));
    nx /= abs; ny /= abs;

#if TEST == CUBIC_TEST
    return -mue_m*( 2*3*SQR(x - 1.0)*nx
                  + 3*SQR(y - 1.0)*ny);
#elif TEST == COS_TEST
    return -mue_m*M_PI*cos(x*y*M_PI)*(y*nx + x*ny);
#endif
  }
} minus_cf;

static struct:CF_2{
  double operator()(double x, double y) const {
    if (circle(x,y) > 0)
      return plus_cf(x,y);
    else
      return minus_cf(x,y);
  }
} bc_wall_value;

static struct:CF_2{
  double operator()(double x, double y) const {
#if TEST == CUBIC_TEST
    return mue_m*(2*3*2*(x - 1.0) + 3*2*(y - 1.0));
#elif TEST == COS_TEST
    return -mue_m*M_PI*M_PI*(SQR(y)+SQR(x))*sin(x*y*M_PI);
#endif
  }
} rhs_minus_cf;

static struct:CF_2{
  double operator()(double x, double y) const {
    return plus_cf(x,y) - minus_cf(x,y);
  }
} jump_sol;

static struct:CF_2{
  double operator()(double x, double y) const {
    return plus_cf.dn(x,y) - minus_cf.dn(x,y);
  }
} jump_dn_sol;
#endif

int main (int argc, char* argv[]){

  mpi_enviroment_t mpi;
  mpi.init(argc, argv);
  try{
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("lip", "Lipchitz constant of level-set for grid generation");
    cmd.add_option("lmin", "the min level of the tree");
    cmd.add_option("lmax", "the max level of the tree");
    cmd.add_option("sp", "number of splits to apply to the min and max level");
    cmd.parse(argc, argv);

    // decide on the type and value of the boundary conditions
    int nb_splits, min_level, max_level;
    nb_splits         = cmd.get("sp" , 0);
    min_level         = cmd.get("lmin", 3);
    max_level         = cmd.get("lmax", 8);
    double lip        = cmd.get("lip", 1.5);

    splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, lip);

    parStopWatch w1, w2;
    w1.start("total time");

    w2.start("initializing the grid");

    /* create the macro mesh */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;

    int n_xyz [] = {1, 1, 1};
    double xyz_min [] = {0, 0, 0};
    double xyz_max [] = {2, 2, 2};

    connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max,  &brick);

    /* create the p4est and partition it iteratively */
    p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est->user_pointer = (void*)(&data);
    for (int l=0; l<max_level+nb_splits; l++){
      p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      p4est_partition(p4est, P4EST_TRUE, NULL);
    }

    /* create the ghost layer */
    p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    p4est_gloidx_t global_num_quadrants = p4est->global_num_quadrants;
    p4est_gloidx_t global_num_nodes = 0;
    for (int i = 0; i<p4est->mpisize; i++){
      global_num_nodes += nodes->global_owned_indeps[i];
    }

    PetscPrintf(p4est->mpicomm, "global number of nodes     = %7ld \n"
                                "global number of quadrants = %7ld \n", global_num_nodes, global_num_quadrants);

    /* initialize the vectors */
    struct solution_t {
      double minus, plus;
    };

    Vec sol, phi, rhs;
#ifdef P4_TO_P8
    const CF_3* sol_cf [] = {&minus_cf, &plus_cf};
    const CF_3* rhs_cf [] = {&rhs_minus_cf, &rhs_plus_cf};
#else
    const CF_2* sol_cf [] = {&minus_cf, &plus_cf};
    const CF_2* rhs_cf [] = {&rhs_minus_cf, &rhs_plus_cf};
#endif
    ierr = VecCreateGhostNodes(p4est, nodes, &phi);
    ierr = VecCreateGhostNodesBlock(p4est, nodes, 2, &sol); CHKERRXX(ierr);
    ierr = VecDuplicate(sol, &rhs); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, sol_cf, sol);
    sample_cf_on_nodes(p4est, nodes, rhs_cf, rhs);
    sample_cf_on_nodes(p4est, nodes, circle, phi);

    // copy to local buffers to save as vtk
    std::vector<double> exact_plus(nodes->indep_nodes.elem_count), exact_minus(nodes->indep_nodes.elem_count);
    Vec sol_plus, sol_minus;
    ierr = VecDuplicate(phi, &sol_plus); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &sol_minus); CHKERRXX(ierr);

    solution_t *sol_p;
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, (double**)&sol_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      exact_minus[i] = sol_p[i].minus;
      exact_plus[i]  = sol_p[i].plus;
    }

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           3, 0, "exact",
                           VTK_POINT_DATA, "phi",   phi_p,
                           VTK_POINT_DATA, "exact plus",  &exact_plus[0],
                           VTK_POINT_DATA, "exact minus", &exact_minus[0]);

    // set up the boundary conditions
#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif
    bc.setWallTypes(bc_wall_type);
    bc.setWallValues(bc_wall_value);

    w2.start("solving jump problem");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    my_p4est_node_neighbors_t neighbors(&hierarchy, nodes);
    my_p4est_poisson_jump_nodes_extended_t solver(&neighbors);
    solver.set_bc(bc);
    solver.set_jump(jump_sol, jump_dn_sol);
    solver.set_phi(phi);
    solver.set_rhs(rhs);
    solver.set_mue(mue_p, mue_m);
    solver.solve(sol);
    w2.stop(); w2.read_duration();

    // save the result
    double *sol_plus_p, *sol_minus_p;
    ierr = VecGetArray(sol_plus, &sol_plus_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol_minus, &sol_minus_p); CHKERRXX(ierr);

    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      sol_minus_p[i] = sol_p[i].minus;
      sol_plus_p[i]  = sol_p[i].plus;
    }

    // extend solutions over interface
    w2.start("extending solution");
    my_p4est_level_set_t ls(&neighbors);
    ls.extend_Over_Interface_TVD(phi, sol_minus, 50);
    // reverse sign
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      phi_p[i] = -phi_p[i];
    }
    ls.extend_Over_Interface_TVD(phi, sol_plus, 50);
    // reverse sign to its normal value
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      phi_p[i] = -phi_p[i];
    }
    w2.stop(); w2.read_duration();

    // compute the error -- overwritting to save on space
    double err_max [] = {0, 0}; // {minus, plus}
    std::vector<double> err(nodes->indep_nodes.elem_count, 0);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
      exact_minus[i] = fabs(exact_minus[i] - sol_minus_p[i]);
      exact_plus[i]  = fabs(exact_plus[i] - sol_plus_p[i]);
      if (phi_p[i] < 0){
        err_max[0] = MAX(err_max[0], exact_minus[i]);
        err[i] = exact_minus[i];
      } else {
        err_max[1] = MAX(err_max[1], exact_plus[i]);
        err[i] = exact_plus[i];
      }
    }

    // compute the globally maximum error. Note that MPI_IN_PLACE should only be used at the root
    if (p4est->mpirank == 0)
      MPI_Reduce(MPI_IN_PLACE, err_max, 2, MPI_DOUBLE, MPI_MAX, 0, p4est->mpicomm);
    else
      MPI_Reduce(err_max, err_max, 2, MPI_DOUBLE, MPI_MAX, 0, p4est->mpicomm);

    PetscPrintf(p4est->mpicomm, "Max err: minus = %1.5e \t plus = %1.5e\n", err_max[0], err_max[1]);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           6, 0, "sol",
                           VTK_POINT_DATA, "phi",   phi_p,
                           VTK_POINT_DATA, "plus",  sol_plus_p,
                           VTK_POINT_DATA, "minus", sol_minus_p,
                           VTK_POINT_DATA, "err_plus",  &exact_plus[0],
                           VTK_POINT_DATA, "err_minus", &exact_minus[0],
                           VTK_POINT_DATA, "err", &err[0]);


    /* destroy p4est objects */
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol, (double**)&sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_plus, &sol_plus_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_minus, &sol_minus_p); CHKERRXX(ierr);

    ierr = VecDestroy(sol); CHKERRXX(ierr);
    ierr = VecDestroy(sol_plus); CHKERRXX(ierr);
    ierr = VecDestroy(sol_minus); CHKERRXX(ierr);

    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy (ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    std::cout << "[" << mpi.rank() << "]: " << e.what() << std::endl;
  }

  return 0;
}
