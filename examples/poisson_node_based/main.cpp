// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// casl_p4est
#include <src/utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/refine_coarsen.h>
#include <src/petsc_compatibility.h>
#include <src/poisson_solver_node_base.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

static struct:CF_2{
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
  double x0, y0, r;
} circle;

// Exact solution
static struct:CF_2{
    double operator()(double x, double y) const {
        return cos(2*M_PI*x)*cos(2*M_PI*y);
    }
} u_ex;

static struct:CF_2{
    double operator()(double x, double y) const {
        return 8*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y);
    }
} f_ex;

// Boundary condition. Impose Dirichlet on the interface and the walls
static struct:WallBC{
    BoundaryConditionType operator()(double x, double y) const {
        (void)x, (void)y;
        return NEUMANN;
    }
} bc_wall_neumann_type;

static struct:WallBC{
    BoundaryConditionType operator()(double x, double y) const {
        (void)x, (void)y;
        return DIRICHLET;
    }
} bc_wall_dirichlet_type;

static struct:WallBC{
  const static double eps = 1e-1;
    BoundaryConditionType operator()(double x, double y) const {
      if (x<eps && y>=0 && y<=1)
        return DIRICHLET;
      else if (x<eps   && y>1  && y<=2)
        return NEUMANN;
      else if (y<eps   && x>=0 && x<=1)
        return NEUMANN;
      else if (y<eps   && x>1  && x<=2)
        return DIRICHLET;
      else if (x>2-eps && y>=0 && y<=1)
        return NEUMANN;
      else if (x>2-eps && y>1  && y<=2)
        return DIRICHLET;
      else if (y>2-eps && x>=0 && x<=1)
        return DIRICHLET;
      else if (y>2-eps && x>1  && x<=2)
        return NEUMANN;
      else // in the domain, so it does not matter
        return DIRICHLET;
    }
} bc_wall_mixed_type;

static struct:CF_2{
    double operator()(double x, double y) const {
        return 0;
    }
} bc_wall_neumann_value;

static struct:CF_2{
    double operator()(double x, double y) const {
      return u_ex(x,y);
    }
} bc_wall_dirichlet_value;

static struct:CF_2{
  const static double eps = 1e-1;
    double operator()(double x, double y) const {
      if (x<eps && y>=0 && y<=1)
        return bc_wall_dirichlet_value(x,y);
      else if (x<eps   && y>1  && y<=2)
        return bc_wall_neumann_value(x,y);
      else if (y<eps   && x>=0 && x<=1)
        return bc_wall_neumann_value(x,y);
      else if (y<eps   && x>1  && x<=2)
        return bc_wall_dirichlet_value(x,y);
      else if (x>2-eps && y>=0 && y<=1)
        return bc_wall_neumann_value(x,y);
      else if (x>2-eps && y>1  && y<=2)
        return bc_wall_dirichlet_value(x,y);
      else if (y>2-eps && x>=0 && x<=1)
        return bc_wall_dirichlet_value(x,y);
      else if (y>2-eps && x>1  && x<=2)
        return bc_wall_neumann_value(x,y);
      else // in the domain, so it does not matter
        return bc_wall_dirichlet_value(x,y);
    }
} bc_wall_mixed_value;

static struct:CF_2{
    double operator()(double x, double y) const {
      return u_ex(x,y);
    }
} bc_interface_dirichlet_value;

static struct:CF_2{
    double operator()(double x, double y) const {
      return (2*M_PI*x*sin(2*M_PI*x)*cos(2*M_PI*y) + 2*M_PI*y*cos(2*M_PI*x)*sin(2*M_PI*y))/circle.r;
    }
} bc_interface_neumann_value;

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  cmdParser cmd;
  cmd.add_option("bc_wall_type", "type of boundary condition to use on the wall");
  cmd.add_option("bc_interface_type", "type of boundary condition to use on the interface");
  cmd.parse(argc, argv);

  // decide on the type and value of the boundary conditions
  BoundaryConditionType bc_wall_type, bc_interface_type;
  bc_wall_type      = cmd.get<BoundaryConditionType>("bc_wall_type",      DIRICHLET);
  bc_interface_type = cmd.get<BoundaryConditionType>("bc_interface_type", DIRICHLET);

  CF_2 *bc_wall_value, *bc_interface_value;
  WallBC *wall_bc;

  switch(bc_interface_type){
  case DIRICHLET:
    bc_interface_value = &bc_interface_dirichlet_value;
    break;
  case NEUMANN:
    bc_interface_value = &bc_interface_neumann_value;
    break;
  default:
    throw std::invalid_argument("[ERROR]: Interface bc type can only be 'Dirichlet' or 'Neumann' type");
  }

  switch(bc_wall_type){
  case DIRICHLET:
    bc_wall_value = &bc_wall_dirichlet_value;
    wall_bc       = &bc_wall_dirichlet_type;
    break;
  case NEUMANN:
    bc_wall_value = &bc_wall_neumann_value;
    wall_bc       = &bc_wall_neumann_type;
    break;
  case MIXED:
    bc_wall_value = &bc_wall_mixed_value;
    wall_bc       = &bc_wall_mixed_type;
    break;
  default:
    throw std::invalid_argument("[ERROR]: Wall bc type can only be 'Dirichlet', 'Neumann', or 'Mixed' type");
  }

  circle.update(1, 1, .3);
  splitting_criteria_cf_t data(5, 10, &circle, 1);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  w2.start("creating connectivity");
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);
  w2.stop(); w2.read_duration();

  w2.start("creating p4est");
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
  w2.stop(); w2.read_duration();

  w2.start("refining p4est");
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  w2.stop(); w2.read_duration();

  w2.start("partitioning p4est object");
  p4est_partition(p4est, NULL);
  w2.stop(); w2.read_duration();

  w2.start("generating ghost data structure");
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
  w2.stop(); w2.read_duration();

  w2.start("creating nodes data structure");
  nodes = my_p4est_nodes_new(p4est, ghost);
  w2.stop(); w2.read_duration();

  w2.start("creating ghosted vectors");
  Vec phi, rhs, uex, sol;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &uex); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);
  w2.stop(); w2.read_duration();

  w2.start("initializing vectors");
  sample_cf_on_nodes(p4est, nodes, circle, phi);
  sample_cf_on_nodes(p4est, nodes, u_ex, uex);
  sample_cf_on_nodes(p4est, nodes, f_ex, rhs);
  w2.stop(); w2.read_duration();

  w2.start("constructing p4est hierarchy");
  my_p4est_hierarchy_t hierarchy(p4est, ghost);
  w2.stop(); w2.read_duration();

  w2.start("constructing node neighboring information");
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
  w2.stop(); w2.read_duration();

  w2.start("building PoissonSolver");
  BoundaryConditions2D bc;
  bc.setInterfaceType(bc_interface_type);
  bc.setInterfaceValue(*bc_interface_value);
  bc.setWallTypes(*wall_bc);
  bc.setWallValues(*bc_wall_value);

  PoissonSolverNodeBase solver(node_neighbors, &brick);
  solver.set_phi(phi);
  solver.set_rhs(rhs);
  solver.set_bc(bc);
  w2.stop(); w2.read_duration();

  w2.start("setting up and solving the linear system");
  solver.solve(sol);
  ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(sol, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
  w2.stop(); w2.read_duration();

  // done. lets write levelset and solutions
  double *sol_p, *phi_p, *uex_p;
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(uex, &uex_p); CHKERRXX(ierr);

  std::ostringstream oss; oss << "solution_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         3, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "uex", uex_p);

  // restore pointers
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(uex, &uex_p); CHKERRXX(ierr);

  // finally, delete PETSc Vecs by calling 'VecDestroy' function
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  ierr = VecDestroy(uex); CHKERRXX(ierr);
  ierr = VecDestroy(sol); CHKERRXX(ierr);
  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();
  return 0;
}
