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
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/point3.h>
#include "charging_linear_3d.h"
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
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/point2.h>
#include "charging_linear_2d.h"
#endif

#include <src/ipm_logging.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>

using namespace std;

#ifdef P4_TO_P8
class Interface:public CF_3{
public:
  double operator ()(double x, double y, double z) const {
    return 0.15 - sqrt(SQR(x-0.5) + SQR(y - 0.5) + SQR(z - 0.5));
  }
} sphere;
#else
class Interface:public CF_2{
public:
  double operator ()(double x, double y) const {
    return 0.15 - sqrt(SQR(x-0.5) + SQR(y - 0.5));
  }
} sphere;
#endif

#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

std::string output_dir;

#ifdef P4_TO_P8
class constant_cf: public CF_3{
  double c;
public:
  constant_cf(double c_): c(c_) {}
  inline void set(double c_) { c = c_; }
  double operator ()(double /* x */, double /* y */, double /* z */) const {
    return c;
  }
};
#else
class constant_cf: public CF_2{
  double c;
public:
  constant_cf(double c_): c(c_) {}
  inline void set(double c_) { c = c_; }
  double operator ()(double /* x */, double /* y */) const {
    return c;
  }
};
#endif


//class Solver{
//  p4est_t* p4est;
//  p4est_ghost_t* ghost;
//  p4est_nodes_t *nodes;
//  my_p4est_brick_t *brick;
//  my_p4est_node_neighbors_t *ngbd;

//  Vec phi, phi_xx, phi_yy;
//#ifdef P4_TO_P8
//  Vec phi_zz;
//#endif
//  bool local_phi_dd;

//  Vec psi, rhs, sol;
//  PetscErrorCode ierr;

//  Vec G, alpha;
//  InterpolatingFunctionNodeBase G_interp;
//  PoissonSolverNodeBase psi_solver;

//#ifdef P4_TO_P8
//  BoundaryConditions3D psi_bc;
//  class: public WallBC3D {
//  public:
//    BoundaryConditionType operator ()(double x, double /* y */, double /* z */) const {
//      if (fabs(x) < EPS)
//        return DIRICHLET;
//      else
//        return NEUMANN;
//    }
//  } wall_bc;

//  class:public CF_3{
//  public:
//    double operator()(double /* x */, double /* y */, double /* z */) const {
//        return 0.0;
//    };
//  } wall_psi_value;
//#else
//  BoundaryConditions2D psi_bc;

//  class: public WallBC2D {
//  public:
//    BoundaryConditionType operator ()(double x, double /* y */) const {
//      if (fabs(x) < EPS)
//        return DIRICHLET;
//      else
//        return NEUMANN;
//    }
//  } wall_bc;

//  class:public CF_2{
//  public:
//    double operator()(double /* x */, double /* y */) const {
//        return 0.0;
//    }
//  } wall_psi_value;
//#endif
//  double lambda, dt;

//  void solve_potential();
//  void solve_concentration();

//  Solver(const Solver& other);
//  Solver& operator=(const Solver& other);

//public:
//  Solver(p4est_t* p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, my_p4est_brick_t *brick_, my_p4est_node_neighbors_t *ngbd_)
//    : p4est(p4est_), ghost(ghost_), nodes(nodes_), brick(brick_), ngbd(ngbd_),
//      local_phi_dd(false),
//      G_interp(p4est, nodes, ghost, brick, ngbd),
//      psi_solver(ngbd)
//  {
//    ierr = VecCreateGhostNodes(p4est, nodes, &psi); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est, nodes, &G); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est, nodes, &alpha); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);
//  }

//  ~Solver(){
//    ierr = VecDestroy(psi); CHKERRXX(ierr);
//    ierr = VecDestroy(G); CHKERRXX(ierr);
//    ierr = VecDestroy(alpha); CHKERRXX(ierr);
//    ierr = VecDestroy(sol); CHKERRXX(ierr);
//    ierr = VecDestroy(rhs); CHKERRXX(ierr);

//    if(local_phi_dd){
//      ierr = VecDestroy(phi_xx); CHKERRXX(ierr);
//      ierr = VecDestroy(phi_yy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ierr = VecDestroy(phi_zz); CHKERRXX(ierr);
//#endif
//    }
//  }

//#ifdef P4_TO_P8
//  inline void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL)
//#else
//  inline void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL)
//#endif
//  {
//    this->phi = phi;
//#ifdef P4_TO_P8
//    if (phi_xx != NULL && phi_yy != NULL && phi_zz != NULL)
//#else
//    if (phi_xx != NULL && phi_yy != NULL)
//#endif
//    {
//      this->phi_xx = phi_xx;
//      this->phi_yy = phi_yy;
//#ifdef P4_TO_P8
//      this->phi_zz = phi_zz;
//#endif
//      local_phi_dd = false;
//    } else {
//      ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_xx); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_yy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_zz); CHKERRXX(ierr);
//      ngbd->second_derivatives_central(phi, this->phi_xx, this->phi_yy, this->phi_zz);
//#else
//      ngbd->second_derivatives_central(phi, this->phi_xx, this->phi_yy);
//#endif
//      local_phi_dd = true;
//    }

//  }

//  inline void set_parameters(double dt, double lambda){
//    this->dt     = dt;
//    this->lambda = lambda;
//  }

//  inline void init(){
//    psi_bc.setInterfaceType(ROBIN);
//    psi_bc.setInterfaceValue(G_interp);
//    psi_bc.setWallTypes(wall_bc);
//    psi_bc.setWallValues(wall_psi_value);

//    double *alpha_p, *psi_p;
//    ierr = VecGetArray(alpha, &alpha_p); CHKERRXX(ierr);
//    ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);////  struct:CF_2{
////    double operator()(double x, double y) const { return 100; }
////  } g1;

////  struct:CF_2{
////    double operator()(double x, double y) const { return 0; }
////  } g2;
//    for (size_t n=0; n<nodes->indep_nodes.elem_count; n++){
//      alpha_p[n] = lambda / dt;
//      psi_p[n]   = 1.0;
//    }
//    ierr = VecRestoreArray(alpha, &alpha_p); CHKERRXX(ierr);
//    ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);

//    psi_solver.set_robin_coef(alpha);

//    psi_solver.set_bc(psi_bc);
//#ifdef P4_TO_P8
//    psi_solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
//#else
//    psi_solver.set_phi(phi, phi_xx, phi_yy);
//#endif
//  }

//  void solve();
//  void write_vtk(const std::string& filename);
//};

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
    cmd.add_option("output-dir", "address of the output directory for all I/O");
    cmd.parse(argc, argv);
    cmd.print();

    output_dir       = cmd.get<std::string>("output-dir", ".");
    const int lmin   = cmd.get("lmin", 4);
    const int lmax   = cmd.get("lmax", 10);

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
    connectivity = my_p4est_brick_new(4, 1, 1, brick);
#else
    connectivity = my_p4est_brick_new(4, 1, brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    splitting_criteria_cf_t sp(lmin, lmax, &sphere, 1.2);
    p4est->user_pointer = &sp;
    for (int l=0; l<lmax; l++){
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, NULL);
    }
    w2.stop(); w2.read_duration();

    w2.start("nodes and ghost construction");
    p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    // make the level-set signed distance
    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, sphere, phi);
    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, "grid",
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    ngbd.init_neighbors();

    proposal::Solver solver(p4est, ghost, nodes, brick, &ngbd);
    solver.set_parameters(1e-3, 0.1);
    solver.set_phi(phi);
    solver.init();

    for (int i=0; i<50; i++){
      ostringstream oss;
      oss << "solving iteration " << i;
      w2.start(oss.str());

      solver.solve();
      oss.str(""); oss << output_dir + "/solution." << i;
      solver.write_vtk(oss.str());

      w2.stop(); w2.read_duration();
    }

    // free memory
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    p4est_destroy(p4est);
    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    my_p4est_brick_destroy(connectivity, brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }

  return 0;
}

//void Solver::solve()
//{
//  // compute value of the boundary condition
//  double *psi_p, *G_p;
//  ierr = VecGetArray(psi, &psi_p);
//  ierr = VecGetArray(G, &G_p);
//  for (size_t n=0; n<nodes->indep_nodes.elem_count; n++){
//    G_p[n] = lambda/dt * psi_p[n];
//  }
//  ierr = VecRestoreArray(psi, &psi_p);
//  ierr = VecRestoreArray(G, &G_p);

//  // construct an interpolating function
//  G_interp.set_input_parameters(G, linear);

//  psi_bc.setInterfaceType(ROBIN);
//  psi_bc.setInterfaceValue(G_interp);

//  double *rhs_p;
//  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
//  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
//    rhs_p[i] = 0;
//  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

//  psi_solver.set_rhs(rhs);
//  psi_solver.solve(psi);

//  my_p4est_level_set ls(ngbd);
//  ls.extend_Over_Interface(phi, psi, 2, 5);
//}

//void Solver::write_vtk(const std::string& filename){
//  double *phi_p, *psi_p;

//  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);

//  my_p4est_vtk_write_all(p4est, nodes, ghost,
//                         P4EST_TRUE, P4EST_TRUE,
//                         2, 0, filename.c_str(),
//                         VTK_POINT_DATA, "phi", phi_p,
//                         VTK_POINT_DATA, "psi", psi_p);

//  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
//}

