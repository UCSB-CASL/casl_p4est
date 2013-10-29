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
#include <src/semi_lagrangian.h>
#include <src/my_p4est_log_wrappers.h>

#undef MIN
#undef MAX
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#include <src/poisson_solver_node_base.h>

#define MIN_LEVEL 3
#define MAX_LEVEL 9

// logging variables
PetscLogEvent log_compute_curvature;
#ifndef CASL_LOG_EVENTS
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#endif
#ifndef CASL_LOG_FLOPS
#define PetscLogFlops(n) 0
#endif

double D = 1;
double tf = 1;
double Tmax = 1;
double Tmin = 0.7;
double epsilon_c = .1;
int save_every_n_iteration = 1;
int iter_max = 300;

using namespace std;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
private:
  double  x0, y0, r;
};

struct BCWALLTYPE : WallBC2D{
  BoundaryConditionType operator()( double x, double y ) const
  {
    (void) x; (void) y;
    return DIRICHLET;
  }
} bc_wall_type;

struct BCWALLVALUE : CF_2 {
  double operator() (double x, double y) const
  {
    (void) x; (void) y;
    return Tmin;
  }
} bc_wall_value;

struct BCInterfaceValue : CF_2 {
private:
  my_p4est_brick_t *brick;
  my_p4est_node_neighbors_t *ngbd;
  InterpolatingFunction interp;
public:
  BCInterfaceValue( my_p4est_brick_t *brick_, p4est_t *p4est_,
                    p4est_nodes_t *nodes_, p4est_ghost_t *ghost_,
                    my_p4est_node_neighbors_t *ngbd_, Vec kappa_)
    : brick(brick_), ngbd(ngbd_), interp(p4est_, nodes_, ghost_, brick_, ngbd_)
  {
    interp.set_input_parameters(kappa_, quadratic);
  }

  double operator() (double x, double y) const
  {
    (void) x; (void) y;
    return Tmax;
//    return Tmax - epsilon_c * interp(x,y);
    /* T = -eps_c kappa - eps_v V */
  }
};

void save_VTK(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_brick_t *brick, Vec phi, Vec T,
              Vec vx, Vec vy, Vec vx_ext, Vec vy_ext, Vec kappa, int compt)
{
  std::ostringstream oss; oss << "stefan_" << p4est->mpisize << "_"
                              << brick->nxytrees[0] << "x"
                              << brick->nxytrees[1] << "." << compt;

  PetscErrorCode ierr;

  double *phi_ptr, *T_ptr, *vx_ptr, *vy_ptr, *vx_ext_ptr, *vy_ext_ptr, *kappa_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(T, &T_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(vx, &vx_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(vy, &vy_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(vx_ext, &vx_ext_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(vy_ext, &vy_ext_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(kappa , &kappa_ptr ); CHKERRXX(ierr);

  my_p4est_vtk_write_all(  p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           7, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_ptr,
                           VTK_POINT_DATA, "temp", T_ptr,
                           VTK_POINT_DATA, "vx", vx_ptr,
                           VTK_POINT_DATA, "vy", vy_ptr,
                           VTK_POINT_DATA, "vx_ext", vx_ext_ptr,
                           VTK_POINT_DATA, "vy_ext", vy_ext_ptr,
                           VTK_POINT_DATA, "kappa", kappa_ptr);

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(T, &T_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(vx, &vx_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(vy, &vy_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(vx_ext, &vx_ext_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(vy_ext, &vy_ext_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa , &kappa_ptr ); CHKERRXX(ierr);
}


void compute_curvature(p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd, Vec phi, Vec kappa)
{
  PetscErrorCode ierr;  
  Vec dx;

  ierr = VecDuplicate(phi, &dx); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_compute_curvature, phi, kappa, dx, 0); CHKERRXX(ierr);

  double *phi_ptr, *kappa_ptr, *dx_ptr;
  ierr = VecGetArray(phi  , &phi_ptr  ); CHKERRXX(ierr);
  ierr = VecGetArray(dx   , &dx_ptr   ); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    dx_ptr[n] = (*ngbd)[n].dx_central (phi_ptr);

  ierr = VecRestoreArray(dx, &dx_ptr); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(dx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (dx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGetArray(dx   , &dx_ptr   ); CHKERRXX(ierr);
  ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double dy  = (*ngbd)[n].dy_central (phi_ptr);
    double dx  = dx_ptr[n];
    double dxx = (*ngbd)[n].dxx_central(phi_ptr);
    double dyy = (*ngbd)[n].dyy_central(phi_ptr);
    double dxy = (*ngbd)[n].dy_central (dx_ptr);
    if(sqrt(dx*dx + dy*dy) < 1e-1) kappa_ptr[n] = 0;
    else kappa_ptr[n] = ( dy*dy*dxx - 2*dx*dy*dxy + dx*dx*dyy) / ((dx*dx + dy*dy) * sqrt(dx*dx + dy*dy));
  }

  ierr = VecRestoreArray(phi  , &phi_ptr  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(dx, &dx_ptr); CHKERRXX(ierr);

  ierr = VecDestroy(dx); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(kappa, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (kappa, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_compute_curvature, phi, kappa, dx, 0); CHKERRXX(ierr);
}


int main (int argc, char* argv[])
{

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode ierr;

  circle circ(1.0, 1.0, 0.2);
  splitting_criteria_cf_t data(MIN_LEVEL, MAX_LEVEL, &circ, 1.2);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);
#ifdef CASL_LOGS
  ierr = PetscLogEventRegister("compute_curvature                              " , 0, &log_compute_curvature); CHKERRXX(ierr);
#endif

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  // Create the connectivity object
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);

  // Now create the forest
  p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

  // Now refine the tree
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // Finally re-partition
  my_p4est_partition(p4est, NULL);

  /* Create the ghost structure */
  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // Initialize the level-set function
  Vec phi;
  Vec Tn;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecDuplicate(phi,&Tn); CHKERRXX(ierr);

  double *phi_ptr, *Tn_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(Tn, &Tn_ptr); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    phi_ptr[i] = circ(x,y);
    Tn_ptr [i] = Tmax;
  }

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(Tn, &Tn_ptr); CHKERRXX(ierr);

  // loop over time
  int tc = 0;
  double t=0;
  double dt_n = 1. / pow(2.,(double) MAX_LEVEL);
  double dt_np1;

  for (t=0; t<tf && tc < iter_max; tc++)
  {
    if(p4est->mpirank==0) printf("Iteration %d, time %e\n",tc,t);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd(&hierarchy,nodes);

    my_p4est_level_set ls(&ngbd);
//    ls.reinitialize_1st_order( phi, 100 );
    ls.reinitialize_2nd_order( phi, 100 );

    /* compute the curvature for boundary conditions */
    Vec kappa;
    ierr = VecDuplicate(phi, &kappa); CHKERRXX(ierr);
    compute_curvature(nodes, &ngbd, phi, kappa);

    /* solve for the temperature */
    BoundaryConditions2D bc;
    bc.setInterfaceType(DIRICHLET);
    BCInterfaceValue bc_interface_value(&brick, p4est, nodes, ghost, &ngbd, kappa);
    bc.setInterfaceValue(bc_interface_value);
    bc.setWallTypes(bc_wall_type);
    bc.setWallValues(bc_wall_value);

    PoissonSolverNodeBase solver(&ngbd);
    solver.set_phi(phi);
    solver.set_mu(D*dt_n);
    solver.set_diagonal(1.);
    solver.set_bc(bc);
    solver.set_rhs(Tn);

    solver.solve(Tn);

    ierr = VecGhostUpdateBegin(Tn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (Tn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the velocity field */
    ls.extend_Over_Interface(phi, Tn, bc, 2, 10);

    /* compute grad(T) dot n */
    Vec vx, vy;
    double *vx_ptr, *vy_ptr;
    ierr = VecDuplicate(phi, &vx); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &vy); CHKERRXX(ierr);

    ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(Tn , &Tn_ptr ); CHKERRXX(ierr);
    ierr = VecGetArray(vx , &vx_ptr ); CHKERRXX(ierr);
    ierr = VecGetArray(vy , &vy_ptr ); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double px = ngbd[n].dx_central(phi_ptr);
      double py = ngbd[n].dy_central(phi_ptr);

      double norm = sqrt(px*px + py*py);
      if(norm > EPS) { px /= norm; py /= norm; }
      else           { px = 0; py = 0; }

      vx_ptr[n] = (px>0 ? -1 : 1) * px * ngbd[n].dx_central(Tn_ptr);
      vy_ptr[n] = (py>0 ? -1 : 1) * py * ngbd[n].dy_central(Tn_ptr);
    }

    ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(Tn , &Tn_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vx , &vx_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(vy , &vy_ptr); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    Vec vx_extended, vy_extended;
    ierr = VecDuplicate(phi,&vx_extended); CHKERRXX(ierr);
    ierr = VecDuplicate(phi,&vy_extended); CHKERRXX(ierr);

    ls.extend_from_interface_to_whole_domain(phi, vx, vx_extended, 10);
    ls.extend_from_interface_to_whole_domain(phi, vy, vy_extended, 10);

    if (tc % save_every_n_iteration == 0)
      save_VTK(p4est, nodes, &brick, phi, Tn, vx, vy, vx_extended, vy_extended, kappa, tc/save_every_n_iteration);

    ierr = VecDestroy(vx); CHKERRXX(ierr);
    ierr = VecDestroy(vy); CHKERRXX(ierr);

    /* compute the time step for the next iteration */
    ierr = VecGetArray(vx_extended, &vx_ptr ); CHKERRXX(ierr);
    ierr = VecGetArray(vy_extended, &vy_ptr ); CHKERRXX(ierr);

    double max_norm_u_loc = 0;
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      max_norm_u_loc = max(max_norm_u_loc, sqrt( vx_ptr[n]*vx_ptr[n] + vy_ptr[n]*vy_ptr[n] ) );

    ierr = VecRestoreArray(vx_extended, &vx_ptr ); CHKERRXX(ierr);
    ierr = VecRestoreArray(vy_extended, &vy_ptr ); CHKERRXX(ierr);
    double max_norm_u;
    ierr = MPI_Allreduce(&max_norm_u_loc, &max_norm_u, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm); CHKERRXX(ierr);

    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + 3];

    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];

    splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
    double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

    dt_np1 = min(1.,1./max_norm_u) * 1. * min(dx, dy);

    /* advect the function in time */
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    SemiLagrangian sl(&p4est_np1, &nodes_np1, &ghost_np1, &brick);
    sl.update_p4est_second_order(vx_extended, vy_extended, dt_n, phi);

    ierr = VecDestroy(vx_extended); CHKERRXX(ierr);
    ierr = VecDestroy(vy_extended); CHKERRXX(ierr);

    /* interpolate Tn on the new grid */
    Vec Tnp1;
    ierr = VecDuplicate(phi, &Tnp1); CHKERRXX(ierr);
    InterpolatingFunction interp(p4est, nodes, ghost, &brick, &ngbd);

    p4est_topidx_t *t2v = p4est_np1->connectivity->tree_to_vertex; // tree to vertex list
    double *t2c = p4est_np1->connectivity->vertices; // coordinates of the vertices of a tree
    for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n + nodes_np1->offset_owned_indeps);
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;

      p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree
      double tr_xmin = t2c[3 * tr_mm + 0];
      double tr_ymin = t2c[3 * tr_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tr_xmin;
      double y = int2double_coordinate_transform(node->y) + tr_ymin;

      interp.add_point_to_buffer(n, x, y);
    }

    interp.set_input_parameters(Tn, quadratic);
    interp.interpolate(Tnp1);

    ierr = VecDestroy(Tn); CHKERRXX(ierr);
    Tn = Tnp1;

    ierr = VecGhostUpdateBegin(Tn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    p4est_destroy(p4est);       p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;

    ierr = VecDestroy(kappa); CHKERRXX(ierr);

    ierr = VecGhostUpdateEnd  (Tn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    t += dt_n;
    dt_n = dt_np1;
  }

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  ierr = VecDestroy(Tn); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}
