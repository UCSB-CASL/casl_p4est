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
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#endif

#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

#ifdef P4_TO_P8
static struct:CF_3{
  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
  double  x0, y0, z0, r;
} circle ;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
  }
} u_ex;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  12*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
  }
} f_ex;

static struct:WallBC3D{
  BoundaryConditionType operator()(double x, double y, double z) const {
    (void)x;
    (void)y;
    (void)z;
    return NEUMANN;
  }
} bc_wall_neumann_type;

static struct:WallBC3D{
  BoundaryConditionType operator()(double x, double y, double z) const {
    (void)x;
    (void)y;
    (void)z;
    return DIRICHLET;
  }
} bc_wall_dirichlet_type;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    (void) x;
    (void) y;
    (void) z;
    return 0;
  }
} bc_wall_neumann_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return u_ex(x,y,z);
  }
} bc_wall_dirichlet_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return u_ex(x,y,z);
  }
} bc_interface_dirichlet_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    double nx = (x-circle.x0) / sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
    double ny = (y-circle.y0) / sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
    double nz = (z-circle.z0) / sqrt( SQR(z-circle.z0) + SQR(z-circle.z0) );
    double norm = sqrt( nx*nx + ny*ny + nz*nz);
    nx /= norm; ny /= norm; nz /= norm;
    return ( 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z) * nx +
             2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z) * ny +
             2*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z) * nz );
  }
} bc_interface_neumann_value;
#else
static struct:CF_2{
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
  double  x0, y0, r;
} circle;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  cos(2*M_PI*x)*cos(2*M_PI*y);
  }
} u_ex;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  8*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y);
  }
} f_ex;

static struct:WallBC2D{
  BoundaryConditionType operator()(double x, double y) const {
    (void)x;
    (void)y;
    return NEUMANN;
  }
} bc_wall_neumann_type;

static struct:WallBC2D{
  BoundaryConditionType operator()(double x, double y) const {
    (void)x;
    (void)y;
    return DIRICHLET;
  }
} bc_wall_dirichlet_type;

static struct:CF_2{
  double operator()(double x, double y) const {
    (void) x;
    (void) y;
    return 0;
  }
} bc_wall_neumann_value;

static struct:CF_2{
  double operator()(double x, double y) const {
    return u_ex(x,y);
  }
} bc_wall_dirichlet_value;

static struct:CF_2{
  double operator()(double x, double y) const {
    return u_ex(x,y);
  }
} bc_interface_dirichlet_value;

static struct:CF_2{
  double operator()(double x, double y) const {
    double nx = (x-circle.x0) / sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
    double ny = (y-circle.y0) / sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
    double norm = sqrt( nx*nx + ny*ny);
    nx /= norm; ny /= norm;
    return 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y) * nx + 2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y) * ny;
  }
} bc_interface_neumann_value;
#endif

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode      ierr;

  cmdParser cmd;
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
  cmd.add_option("lmin", "the min level of the tree");
  cmd.add_option("lmax", "the max level of the tree");
  cmd.add_option("nb_splits", "number of splits to apply to the min and max level");
  cmd.parse(argc, argv);

  // decide on the type and value of the boundary conditions
  BoundaryConditionType bc_wall_type, bc_interface_type;
  int nb_splits, min_level, max_level;
  bc_wall_type      = cmd.get("bc_wtype"  , DIRICHLET);
  bc_interface_type = cmd.get("bc_itype"  , DIRICHLET);
  nb_splits         = cmd.get("nb_splits" , 0);
  min_level         = cmd.get("lmin"      , 0);
  max_level         = cmd.get("lmax"      , 10);

#ifdef P4_TO_P8
  CF_3 *bc_wall_value, *bc_interface_value;
  WallBC3D *wall_bc;
#else
  CF_2 *bc_wall_value, *bc_interface_value;
  WallBC2D *wall_bc;
#endif

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
  default:
    throw std::invalid_argument("[ERROR]: Wall bc type can only be 'Dirichlet' or 'Neumann' type");
  }

#ifdef P4_TO_P8
  circle.update(1, 1, 1, .3);
#else
  circle.update(1, 1, .3);
#endif
  splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, 1);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

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
  p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);

  /* generate unique node indices */
  nodes = my_p4est_nodes_new(p4est, ghost);
  w2.stop(); w2.read_duration();

  /* initialize the vectors */
  Vec phi, rhs, uex, sol;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &uex); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, circle, phi);
  sample_cf_on_nodes(p4est, nodes, u_ex, uex);
  sample_cf_on_nodes(p4est, nodes, f_ex, rhs);

  /* create the hierarchy structure */
  w2.start("construct the hierachy information");
  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  w2.stop(); w2.read_duration();

  /* generate the neighborhood information */
  w2.start("construct the neighborhood information");
  my_p4est_node_neighbors_t node_neighbors(&hierarchy, nodes);
  w2.stop(); w2.read_duration();

  /* initalize the bc information */
  Vec interface_value_Vec, wall_value_Vec;
  ierr = VecDuplicate(phi, &interface_value_Vec); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &wall_value_Vec); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, *bc_interface_value, interface_value_Vec);
  sample_cf_on_nodes(p4est, nodes, *bc_wall_value, wall_value_Vec);

  InterpolatingFunction interface_interp(p4est, nodes, ghost, &brick, &node_neighbors), wall_interp(p4est, nodes, ghost, &brick, &node_neighbors);
  interface_interp.set_input_parameters(interface_value_Vec, linear);
  wall_interp.set_input_parameters(wall_value_Vec, linear);

  bc_interface_value = &interface_interp;
  bc_wall_value = &wall_interp;

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif
  bc.setInterfaceType(bc_interface_type);
  bc.setInterfaceValue(*bc_interface_value);
  bc.setWallTypes(*wall_bc);
  bc.setWallValues(*bc_wall_value);

  /* initialize the poisson solver */
  w2.start("solve the poisson equation");
  PoissonSolverNodeBase solver(&node_neighbors);
  solver.set_phi(phi);
  solver.set_rhs(rhs);
  solver.set_bc(bc);
  w2.stop(); w2.read_duration();

  /* solve the system */
  solver.solve(sol);
  ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (sol, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

  /* prepare for output */
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
  PetscPrintf(p4est->mpicomm, "lvl : %d / %d, L_inf error : %e\n",min_level+nb_splits, max_level+nb_splits, glob_err_max);

  /* save the vtk file */
  std::ostringstream oss; oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         4, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "uex", uex_p,
                         VTK_POINT_DATA, "err", err );

  /* restore internal pointers */
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(uex, &uex_p); CHKERRXX(ierr);

  /* destroy allocated vectors */
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  ierr = VecDestroy(uex); CHKERRXX(ierr);
  ierr = VecDestroy(sol); CHKERRXX(ierr);
  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDestroy(wall_value_Vec); CHKERRXX(ierr);
  ierr = VecDestroy(interface_value_Vec); CHKERRXX(ierr);

  /* destroy p4est objects */
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy (ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();
  return 0;
}
