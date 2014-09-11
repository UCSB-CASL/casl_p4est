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

class Solver{
  p4est_t* p4est;
  p4est_ghost_t* ghost;
  p4est_nodes_t *nodes;
  my_p4est_brick_t *brick;
  my_p4est_node_neighbors_t *ngbd;

  Vec phi, phi_xx, phi_yy;
#ifdef P4_TO_P8
  Vec phi_zz;
#endif
  bool local_phi_dd;

  Vec con[3], psi[3];
  Vec w,q,jw,jq,Gw,Gq;
  Vec rhs;
  PetscErrorCode ierr;

  constant_cf wall_psi_value;

  InterpolatingFunctionNodeBase Gq_interp, Gw_interp;

#ifdef P4_TO_P8
  class: public WallBC3D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */, double /* z */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } wall_bc;

  class:public CF_3{
  public:
    double operator()(double x, double /* y */, double /* z */) const {
      if (fabs(x) < EPS)
        return 1.0;
      else
        return 0.0;
    };
  } wall_con_value;
#else
  class: public WallBC2D {
  public:
    BoundaryConditionType operator ()(double x, double /* y */) const {
      if (fabs(x) < EPS)
        return DIRICHLET;
      else
        return NEUMANN;
    }
  } wall_bc;

  class:public CF_2{
  public:
    double operator()(double x, double /* y */) const {
      if (fabs(x) < EPS)
        return 1.0;
      else
        return 0.0;
    }
  } wall_con_value;
#endif
  double lambda, zeta, dt;

  inline double compute_q(double c, double psi) { /* dpsi = psi - psi_w */
    return 2.0*sqrt(c)*sinh((psi-zeta)/2.0);
  }
  inline double compute_jq(double c, double psi) {
    return sqrt(c)*cosh((psi-zeta)/2.0);
  }
  inline double compute_w(double c, double psi) {
    return 4.0*sqrt(c)*SQR(sinh((psi-zeta)/4.0));
  }
  inline double compute_jw(double c, double psi) {
    return 2.0/sqrt(c)*SQR(sinh((psi-zeta)/4.0));
  }

  void solve_potential();
  void solve_concentration();

  Solver(const Solver& other);
  Solver& operator=(const Solver& other);

public:
  Solver(p4est_t* p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, my_p4est_brick_t *brick_, my_p4est_node_neighbors_t *ngbd_)
    : p4est(p4est_), ghost(ghost_), nodes(nodes_), brick(brick_), ngbd(ngbd_),
      local_phi_dd(false), wall_psi_value(0.0),
      Gq_interp(p4est, nodes, ghost, brick, ngbd),
      Gw_interp(p4est, nodes, ghost, brick, ngbd)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &con[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &con[1]); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &con[2]); CHKERRXX(ierr);

    ierr = VecDuplicate(con[0], &psi[0]); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &psi[1]); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &psi[2]); CHKERRXX(ierr);

    ierr = VecDuplicate(con[0], &w); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &q); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &jw); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &jq); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &Gw); CHKERRXX(ierr);
    ierr = VecDuplicate(con[0], &Gq); CHKERRXX(ierr);

    ierr = VecDuplicate(con[0], &rhs); CHKERRXX(ierr);
  }

  ~Solver(){
    ierr = VecDestroy(con[0]); CHKERRXX(ierr);
    ierr = VecDestroy(con[1]); CHKERRXX(ierr);
    ierr = VecDestroy(con[2]); CHKERRXX(ierr);

    ierr = VecDestroy(psi[0]); CHKERRXX(ierr);
    ierr = VecDestroy(psi[1]); CHKERRXX(ierr);
    ierr = VecDestroy(psi[2]); CHKERRXX(ierr);

    ierr = VecDestroy(w); CHKERRXX(ierr);
    ierr = VecDestroy(q); CHKERRXX(ierr);
    ierr = VecDestroy(jw); CHKERRXX(ierr);
    ierr = VecDestroy(jq); CHKERRXX(ierr);
    ierr = VecDestroy(Gw); CHKERRXX(ierr);
    ierr = VecDestroy(Gq); CHKERRXX(ierr);

    ierr = VecDestroy(rhs); CHKERRXX(ierr);

    if(local_phi_dd){
      ierr = VecDestroy(phi_xx); CHKERRXX(ierr);
      ierr = VecDestroy(phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecDestroy(phi_zz); CHKERRXX(ierr);
#endif
    }
  }

#ifdef P4_TO_P8
  inline void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL)
#else
  inline void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL)
#endif
  {
    this->phi = phi;
#ifdef P4_TO_P8
    if (phi_xx != NULL && phi_yy != NULL && phi_zz != NULL)
#else
    if (phi_xx != NULL && phi_yy != NULL)
#endif
    {
      this->phi_xx = phi_xx;
      this->phi_yy = phi_yy;
#ifdef P4_TO_P8
      this->phi_zz = phi_zz;
#endif
      local_phi_dd = false;
    } else {
      ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_xx); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_zz); CHKERRXX(ierr);
      ngbd->second_derivatives_central(phi, this->phi_xx, this->phi_xx, this->phi_zz);
#else
      ngbd->second_derivatives_central(phi, this->phi_xx, this->phi_yy);
#endif
      local_phi_dd = true;
    }

  }

  inline void set_parameters(double dt, double lambda, double zeta){
    this->dt     = dt;
    this->lambda = lambda;
    this->zeta   = zeta;

    double *psi_p, *con_p;
    for (int j = 0; j<3; j++){
      ierr = VecGetArray(psi[j], &psi_p); CHKERRXX(ierr);
      ierr = VecGetArray(con[j], &con_p); CHKERRXX(ierr);
      for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
        psi_p[i] = zeta;
        con_p[i] = 1.0;
      }
      ierr = VecRestoreArray(psi[j], &psi_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(con[j], &con_p); CHKERRXX(ierr);
    }
  }

  void solve(int itmax = 5, double tol = 1e-6);
  void write_vtk(const std::string& filename);
};

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
    const int lmin   = cmd.get("lmin", 3);
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
    connectivity = my_p4est_brick_new(10, 1, 1, brick);
#else
    connectivity = my_p4est_brick_new(10, 1, brick);
#endif
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    splitting_criteria_cf_t sp(lmin, lmax, &sphere, 1.8);
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

    Solver solver(p4est, ghost, nodes, brick, &ngbd);
    solver.set_parameters(1e-3, 0.1, 3);
    solver.set_phi(phi);

    solver.write_vtk(output_dir + "/solution.0");
    for (int i=0; i<1000; i++){
      ostringstream oss;
      oss << "solving iteration " << i;
      w2.start(oss.str());

      solver.solve(INT_MAX, 1e-3);
      oss.str(""); oss << output_dir + "/solution." << i+1;
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


void Solver::write_vtk(const std::string& filename){
  double *phi_p, *con_p, *psi_p, *q_p, *w_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi[0], &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(con[0], &con_p); CHKERRXX(ierr);
  ierr = VecGetArray(w, &w_p); CHKERRXX(ierr);
  ierr = VecGetArray(q, &q_p); CHKERRXX(ierr);

  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
    w_p[i] = compute_w(con_p[i], psi_p[i]);
    q_p[i] = compute_q(con_p[i], psi_p[i]);
  }

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         5, 0, filename.c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "psi", psi_p,
                         VTK_POINT_DATA, "con", con_p,
                         VTK_POINT_DATA, "q", q_p,
                         VTK_POINT_DATA, "w", w_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi[0], &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(con[0], &con_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(w, &w_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);
}

void Solver::solve(int itmax, double tol)
{
  my_p4est_level_set ls(ngbd);

  double *psi_p[2], *con_p[2], *phi_p;
  ierr = VecGetArray(phi,    &phi_p); CHKERRXX(ierr);

  ierr = VecGetArray(psi[1], &psi_p[0]); CHKERRXX(ierr);
  ierr = VecGetArray(psi[2], &psi_p[1]); CHKERRXX(ierr);
  ierr = VecGetArray(con[1], &con_p[0]); CHKERRXX(ierr);
  ierr = VecGetArray(con[2], &con_p[1]); CHKERRXX(ierr);

  // copy data: 0 --> 1
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
    psi_p[1][i] = psi_p[0][i];
    con_p[1][i] = con_p[0][i];
  }

  for (int it = 0; it < itmax; it++) {
    solve_concentration();
    ls.extend_Over_Interface(phi, con[2], 2, 5);

    solve_potential();
    ls.extend_Over_Interface(phi, psi[2], 2, 5);

    // compute error
    double err_g[2];
    double err_l[2] = {0, 0};
    for (size_t i=0; i<nodes->indep_nodes.elem_count; i++){
      if (phi_p[i] <= 0.){
        err_l[0] = MAX(err_l[0], fabs(con_p[0][i] - con_p[1][i]));
        err_l[1] = MAX(err_l[1], fabs(psi_p[0][i] - psi_p[1][i]));
      }
    }
    MPI_Allreduce(err_l, err_g, 2, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
    PetscPrintf(p4est->mpicomm, "   it = %d err_con = %e err_psi = %e\n", it, err_g[0], err_g[1]);

    // swap pointers
    Vec tmp;
    tmp    = con[1];
    con[1] = con[2];
    con[2] = tmp;

    tmp    = psi[1];
    psi[1] = psi[2];
    psi[2] = tmp;

    if (err_g[0] < tol && err_g[1] < tol)
      break;
  }

  // swap pointers
  Vec tmp;
  tmp    = con[0];
  con[0] = con[1];
  con[1] = tmp;

  tmp    = psi[0];
  psi[0] = psi[1];
  psi[1] = tmp;
}

void Solver::solve_concentration(){
  // compute value of the boundary condition
  double *psi_p[2], *con_p[2], *jw_p, *Gw_p;
  ierr = VecGetArray(psi[0], &psi_p[0]);
  ierr = VecGetArray(psi[1], &psi_p[1]); // get the last iteration for potential
  ierr = VecGetArray(con[0], &con_p[0]);
  ierr = VecGetArray(con[1], &con_p[1]);

  ierr = VecGetArray(jw, &jw_p); CHKERRXX(ierr);
  ierr = VecGetArray(Gw, &Gw_p); CHKERRXX(ierr);

  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    jw_p[i] = compute_jw(con_p[1][i], psi_p[1][i]) * lambda/dt;
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    Gw_p[i] = con_p[1][i]*jw_p[i] - lambda/dt *(compute_w(con_p[1][i], psi_p[1][i]) - compute_w(con_p[0][i], psi_p[0][i]));

  ierr = VecRestoreArray(jw, &jw_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(Gw, &Gw_p); CHKERRXX(ierr);

  // construct an interpolating function
  Gw_interp.set_input_parameters(Gw, linear);

  // compute the RHS
  double *rhs_p;

  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    rhs_p[i] = con_p[0][i]/dt;
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  // set the boundary condition
#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setInterfaceType(ROBIN);
  bc.setInterfaceValue(Gw_interp);
  bc.setWallTypes(wall_bc);
  bc.setWallValues(wall_con_value);

  PoissonSolverNodeBase solver(ngbd);
  solver.set_bc(bc);
#ifdef P4_TO_P8
  solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
#else
  solver.set_phi(phi, phi_xx, phi_yy);
#endif
  solver.set_diagonal(1.0/dt);
  solver.set_rhs(rhs);
  solver.set_robin_coef(jw);

  solver.solve(con[2], true);

  ierr = VecRestoreArray(psi[0], &psi_p[0]);
  ierr = VecRestoreArray(psi[1], &psi_p[1]);
  ierr = VecRestoreArray(con[0], &con_p[0]);
  ierr = VecRestoreArray(con[1], &con_p[1]);
}

void Solver::solve_potential(){
  // compute value of the boundary condition
  double *psi_p[2], *con_p[2], *jq_p, *Gq_p;
  ierr = VecGetArray(psi[0], &psi_p[0]);
  ierr = VecGetArray(psi[1], &psi_p[1]);
  ierr = VecGetArray(con[0], &con_p[0]);
  ierr = VecGetArray(con[1], &con_p[1]); // get the last iteration for concentration

  ierr = VecGetArray(jq, &jq_p); CHKERRXX(ierr);
  ierr = VecGetArray(Gq, &Gq_p); CHKERRXX(ierr);

  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    jq_p[i] = compute_jq(con_p[1][i], psi_p[1][i]) * lambda/dt;
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    Gq_p[i] = psi_p[1][i]*jq_p[i] - lambda/dt *(compute_q(con_p[1][i], psi_p[1][i]) - compute_q(con_p[0][i], psi_p[0][i]));

  ierr = VecRestoreArray(jq, &jq_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(Gq, &Gq_p); CHKERRXX(ierr);

  // construct an interpolating function
  Gq_interp.set_input_parameters(Gq, linear);

  // compute the RHS
  double *rhs_p;

  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    rhs_p[i] = 0;
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  // set the boundary condition
#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setInterfaceType(ROBIN);
  bc.setInterfaceValue(Gq_interp);
  bc.setWallTypes(wall_bc);
  bc.setWallValues(wall_psi_value);

  PoissonSolverNodeBase solver(ngbd);
  solver.set_bc(bc);
#ifdef P4_TO_P8
  solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
#else
  solver.set_phi(phi, phi_xx, phi_yy);
#endif
  solver.set_mu(con[2]);
  solver.set_rhs(rhs);
  solver.set_robin_coef(jq);

  solver.solve(psi[2], true);

  ierr = VecRestoreArray(psi[0], &psi_p[0]);
  ierr = VecRestoreArray(psi[1], &psi_p[1]);
  ierr = VecRestoreArray(con[0], &con_p[0]);
  ierr = VecRestoreArray(con[1], &con_p[1]);
}

