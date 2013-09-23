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

//#define DEBUG_TIMINGS

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
        (void) x; (void) y;
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
      double nx = (x-circle.x0) / sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );//circle.r;
      double ny = (y-circle.y0) / sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );//circle.r;
      double norm = sqrt( nx*nx + ny*ny);
      nx /= norm; ny /= norm;
      return 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y) * nx + 2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y) * ny;
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
  cmd.add_option("min_level", "the min level of the tree");
  cmd.add_option("max_level", "the max level of the tree");
  cmd.add_option("nb_splits", "number of splits to apply to the min and max level");
  cmd.parse(argc, argv);

  // decide on the type and value of the boundary conditions
  BoundaryConditionType bc_wall_type, bc_interface_type;
  int nb_splits, min_level, max_level;
  bc_wall_type      = cmd.get<BoundaryConditionType>("bc_wall_type"     , DIRICHLET);
  bc_interface_type = cmd.get<BoundaryConditionType>("bc_interface_type", DIRICHLET);
  nb_splits         = cmd.get<int>                  ("nb_splits"        , 0);
  min_level         = cmd.get<int>                  ("min_level"        , 5);
  max_level         = cmd.get<int>                  ("max_level"        , 10);

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
  splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, 1);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

#ifdef DEBUG_TIMINGS
  w2.start("creating connectivity");
#endif
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("creating p4est");
#endif
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("refining p4est");
#endif
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("partitioning p4est object");
#endif
  p4est_partition(p4est, NULL);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("generating ghost data structure");
#endif
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("creating nodes data structure");
#endif
  nodes = my_p4est_nodes_new(p4est, ghost);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("creating ghosted vectors");
#endif
  Vec phi, rhs, uex, sol;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &uex); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("initializing vectors");
#endif
  sample_cf_on_nodes(p4est, nodes, circle, phi);
  sample_cf_on_nodes(p4est, nodes, u_ex, uex);
  sample_cf_on_nodes(p4est, nodes, f_ex, rhs);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("constructing p4est hierarchy");
#endif
  my_p4est_hierarchy_t hierarchy(p4est, ghost);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("constructing node neighboring information");
#endif
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

#ifdef DEBUG_TIMINGS
  w2.start("building PoissonSolver");
#endif
  BoundaryConditions2D bc;
  bc.setInterfaceType(bc_interface_type);
  bc.setInterfaceValue(*bc_interface_value);
  bc.setWallTypes(*wall_bc);
  bc.setWallValues(*bc_wall_value);

  PoissonSolverNodeBase solver(&node_neighbors, &brick);
  solver.set_phi(phi);
  solver.set_rhs(rhs);
  solver.set_bc(bc);
#ifdef DEBUG_TIMINGS
  w2.stop(); w2.read_duration();
#endif

  w2.start("setting up and solving the linear system");
  solver.solve(sol);
  ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (sol, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
  w2.stop(); w2.read_duration();

  // done. lets write levelset and solutions
  double *sol_p, *phi_p, *uex_p;
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(uex, &uex_p); CHKERRXX(ierr);

  /* compute the error */
  double err_max = 0;
  double err[nodes->indep_nodes.elem_count];
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    if(phi_p[n]<0)
    {
      err[n] = fabs(sol_p[n] - uex_p[n]);
      err_max = max(err_max, err[n]);
    }
    else
      err[n] = 0;
  }
  double glob_err_max;
  MPI_Allreduce(&err_max, &glob_err_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
  if(p4est->mpirank==0)
    printf("lvl : %d / %d, L_inf error : %e\n",min_level+nb_splits, max_level+nb_splits, glob_err_max);

  std::ostringstream oss; oss << "solution_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         4, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "uex", uex_p,
                         VTK_POINT_DATA, "err", err );

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
