#include <src/my_p4est_to_p8est.h>

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

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

#include <src/ipm_logging.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include <mpi.h>

using namespace std;

int nx, ny, nz;

class tube: public CF_3 {
  int n;
  double r;
public:
  tube(int n_, double r_)
    : n(n_), r(r_)
  {lip = 1.2;}
  double operator ()(double x, double y, double z) const {

    x -= floor(x);
    y -= floor(y);

    double f = -DBL_MAX;
    double l = 1.0/(n);
    for (int i=1; i<=n; i++){
      for (int j=1; j<=n; j++){
        f = MAX(f, r - sqrt(SQR(x - i*l + l/2.0) + SQR(y - j*l + l/2.0)));
      }
    }

    f = -MAX(-f, z - 1.0);
    f =  MAX(f, 0.05 - z);

    return f;
  }
};


#ifndef GIT_COMMIT_HASH_SHORT
#define GIT_COMMIT_HASH_SHORT "unknown"
#endif

#ifndef GIT_COMMIT_HASH_LONG
#define GIT_COMMIT_HASH_LONG "unknown"
#endif

std::string output_dir;
double alpha, beta;

void motion_under_curvature3(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax);
void construct_grid_with_reinitializatrion1(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi);

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
    cmd.add_option("itmax", "maximum number of iterations when creating random tree for strong scaling");
    cmd.add_option("output-dir", "address of the output directory for all I/O");
    cmd.add_option("nx", "# of blocks in x direction");
    cmd.add_option("ny", "# of blocks in y direction");
    cmd.add_option("nz", "# of blocks in z direction");
    cmd.add_option("count", "# of tubes in one direction per tree");
    cmd.add_option("beta", "strength of curvature");
    cmd.parse(argc, argv);
    cmd.print();

    output_dir       = cmd.get<std::string>("output-dir");
    beta  = cmd.get("beta", 0.02);
    const int lmin   = cmd.get("lmin", 3);
    const int lmax   = cmd.get("lmax", 10);
    const int itmax  = cmd.get("itmax", 3);
    const int count  = cmd.get("count", 2);
    nx = cmd.get("nx", 3);
    ny = cmd.get("ny", 3);
    nz = cmd.get("nz", 5);

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
    connectivity = my_p4est_brick_new(nx, ny, nz, brick);
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est_t *p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

    double r = 1.0/(4.*count);
    tube tubes(count, r);
    splitting_criteria_cf_t sp(lmin, lmax, &tubes, 1.2);
    p4est->user_pointer = &sp;
    w2.stop(); w2.read_duration();

    p4est_ghost_t *ghost = NULL;
    p4est_nodes_t *nodes = NULL;
    Vec phi = NULL;

    // make the level-set signed distance
    w2.start("grid construction");
    construct_grid_with_reinitializatrion1(p4est, ghost, nodes, brick, phi);
    w2.stop(); w2.read_duration();

    double *phi_p;
    VecGetArray(phi, &phi_p);
    my_p4est_vtk_write_all(p4est, nodes, ghost, 0, 0, 1, 0, "aa", VTK_POINT_DATA, "phi", phi_p);
    VecRestoreArray(phi, &phi_p);

    p4est_gloidx_t num_nodes = 0;
    for (int r =0; r<p4est->mpisize; r++)
      num_nodes += nodes->global_owned_indeps[r];

    PetscPrintf(p4est->mpicomm, "%% Initial grid info:\n global_quads = %ld \t global_nodes = %ld\n", p4est->global_num_quadrants, num_nodes);

    w2.start("smoothing things");
    motion_under_curvature3(p4est, ghost, nodes, brick, phi, itmax);
    w2.stop(); w2.read_duration();

    // write some statistics
    Vec ones, ones_l;
    ierr = VecDuplicate(phi, &ones); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(ones, &ones_l); CHKERRXX(ierr);
    ierr = VecSet(ones_l, 1.0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(ones, &ones_l); CHKERRXX(ierr);

    double v_p =integrate_over_negative_domain(p4est, nodes, phi, ones);
    double a_p =integrate_over_interface(p4est, nodes, phi, ones);
    double h_p = v_p/a_p;
    double porosity = v_p/nx/ny/nz;
    PetscPrintf(p4est->mpicomm, "Porosity = %% %2.2f \t h_p = %e\n", 100.*porosity, h_p);
    ierr = VecDestroy(ones); CHKERRXX(ierr);

    std::ostringstream parname, topname, ngbname;
    parname << output_dir + "/" + "partition" << ".dat";
    topname << output_dir + "/" + "topology" << ".dat";
    ngbname << output_dir + "/" + "neighbors" << ".dat";

    write_comm_stats(p4est, ghost, nodes, parname.str().c_str(), topname.str().c_str(), ngbname.str().c_str());

    // solve the system
    my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    my_p4est_level_set ls(&ngbd);
    ngbd.init_neighbors();

    PoissonSolverNodeBase solver(&ngbd);
    struct:WallBC3D{
      BoundaryConditionType operator()(double x, double y, double z) const {
        (void)x;
        (void)y;

        if (fabs(z) > nz - EPS)
          return DIRICHLET;
        else
          return NEUMANN;
      }
    } wall_bc;

    struct:CF_3{
      double operator()(double x, double y, double z) const {
        (void)x;
        (void)y;
        if (fabs(z) > nz - EPS)
          return 1.0;
        else
          return 0.0;
      }
    } wall_bc_value;

    struct:CF_3{
      double operator()(double x, double y, double z) const {
        (void)x;
        (void)y;
        (void)z;
        return 0;
      }
    } interface_value;

    BoundaryConditions3D bc;
    bc.setInterfaceType(ROBIN);
    bc.setInterfaceValue(interface_value);
    bc.setWallTypes(wall_bc);
    bc.setWallValues(wall_bc_value);

    Vec alpha;
    double Da = 1;
    ierr = VecCreateGhostNodes(p4est, nodes, &alpha); CHKERRXX(ierr);
    double *alpha_p;
    ierr = VecGetArray(alpha, &alpha_p); CHKERRXX(ierr);
    for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
      alpha_p[i] = Da;
    ierr = VecRestoreArray(alpha, &alpha_p); CHKERRXX(ierr);

    solver.set_robin_coef(alpha);

    solver.set_bc(bc);
    solver.set_phi(phi);
    Vec sol, rhs;
    ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);

    double dt = 1e-2;
//    solver.set_diagonal(1.0/dt);
    double *sol_p, *rhs_p;

    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

    for (int i=0; i<1; i++){
      for (size_t n=0;n<nodes->indep_nodes.elem_count; n++)
        rhs_p[n] = 0;//sol_p[n] / dt;

      w2.start("solving linear system");
      solver.set_rhs(rhs);
      solver.solve(sol);
      w2.stop(); w2.read_duration();

      w2.start("extrapolation");
      ls.extend_Over_Interface(phi, sol);
      w2.stop(); w2.read_duration();

      ostringstream oss; oss << output_dir + "/solution." << i;
      my_p4est_vtk_write_all(p4est, nodes, NULL,
                             P4EST_TRUE, P4EST_FALSE,
                             2, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "sol", sol_p);
    }

    ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

    // free memory
    ierr = VecDestroy(alpha); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
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

void motion_under_curvature3(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* myb, Vec& phi, int itmax)
{
  PetscErrorCode ierr;
  const splitting_criteria_cf_t *sp = (const splitting_criteria_cf_t*)p4est->user_pointer;
  parStopWatch w;

  double dx = (double)P4EST_QUADRANT_LEN(sp->max_lvl)/(double)P4EST_ROOT_LEN;
  double d_tau = dx;

  Vec phi_x, phi_y, norm_grad_phi;
  double *phi_x_p, *phi_y_p, *norm_grad_phi_p;
#ifdef P4_TO_P8
  Vec phi_z;
  double *phi_z_p;
#endif
  Vec rhs;
  Vec phi_np1;

  double *phi_p, *rhs_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  std::ostringstream oss; oss << output_dir + "/curvature.0";

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_FALSE,
                         1, 0, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  struct:WallBC3D{
    BoundaryConditionType operator()(double /* x */, double /* y */, double /* z */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_3{
    double operator ()(double /* x */, double /* y */, double /* z */) const {return 0;}
  } zero_func;
#else
  struct:WallBC2D{
    BoundaryConditionType operator()(double /* x */, double /* y */ ) const {return NEUMANN;}
  } wall_bc_neumann;

  struct:CF_2{
    double operator ()(double /* x */, double /* y */) const {return 0;}
  } zero_func;
#endif

  for(int iter = 0; iter < itmax; iter++)
  {
    w.start("advect in normal direction");
    // create hierarchy and node neighbors
    my_p4est_hierarchy_t hierarchy(p4est, ghost, myb);
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    my_p4est_level_set ls(&ngbd);
    w.stop(); w.read_duration();

    w.start("preparing rhs");
    ierr = VecDuplicate(phi, &phi_x); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_y); CHKERRXX(ierr);
  #ifdef P4_TO_P8
    ierr = VecDuplicate(phi, &phi_z); CHKERRXX(ierr);
  #endif
    ierr = VecDuplicate(phi, &norm_grad_phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &phi_np1); CHKERRXX(ierr);

    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecGetArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    /* first compute grad phi */
    // 1- layer nodes
    for(size_t ni = 0; ni<ngbd.get_layer_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_layer_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 2- begin nonblocking update
    ierr = VecGhostUpdateBegin(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // 3- local nodes
    for(size_t ni = 0; ni<ngbd.get_local_size(); ++ni)
    {
      p4est_locidx_t n = ngbd.get_local_node(ni);
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      phi_x_p[n] = qnnn.dx_central(phi_p);
      phi_y_p[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      phi_z_p[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]) + SQR(phi_z_p[n]));
#else
      double norm = sqrt(SQR(phi_x_p[n]) + SQR(phi_y_p[n]));
#endif

      norm_grad_phi_p[n] = norm > EPS ? norm : 0.;
    }

    // 4- finish nonblocking update
    ierr = VecGhostUpdateEnd(norm_grad_phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* prepare right hand side */
    for(p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n)
    {
      quad_neighbor_nodes_of_node_t qnnn;
      ngbd.get_neighbors(n, qnnn);

      rhs_p[n] = phi_p[n];
      if(norm_grad_phi_p[n]>EPS){
#ifdef P4_TO_P8
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) + phi_z_p[n]*qnnn.dz_central(norm_grad_phi_p) );
#else
        rhs_p[n] -= d_tau*beta/norm_grad_phi_p[n] * (phi_x_p[n]*qnnn.dx_central(norm_grad_phi_p) + phi_y_p[n]*qnnn.dy_central(norm_grad_phi_p) );
#endif
      }
    }

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_x, &phi_x_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_y, &phi_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(phi_z, &phi_z_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(norm_grad_phi, &norm_grad_phi_p); CHKERRXX(ierr);

    // remove unecessary arrays
    ierr = VecDestroy(phi_x); CHKERRXX(ierr);
    ierr = VecDestroy(phi_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_z); CHKERRXX(ierr);
#endif
    ierr = VecDestroy(norm_grad_phi); CHKERRXX(ierr);
    w.stop(); w.read_duration();

    /* solve the system */
    w.start("solving the system");
    {
#ifdef P4_TO_P8
      BoundaryConditions3D bc;
#else
      BoundaryConditions2D bc;
#endif
      bc.setWallTypes(wall_bc_neumann);
      bc.setWallValues(zero_func);

      VecSet(phi, -1);

      PoissonSolverNodeBase solver(&ngbd);
      solver.set_bc(bc);
      solver.set_rhs(rhs);
      solver.set_phi(phi);
      solver.set_diagonal(1.0);
      solver.set_mu(d_tau*beta);
      solver.set_tolerances(1e-6);
      solver.solve(phi_np1);
    }
    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD);
    w.stop(); w.read_duration();

    ls.reinitialize_1st_order_time_2nd_order_space(phi_np1, 10);

//    /* construct a new grid */
    w.start("update grid");
    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_np1->connectivity = p4est->connectivity;
    InterpolatingFunctionNodeBase phi_interp(p4est, nodes, ghost, myb, &ngbd);

    phi_interp.set_input_parameters(phi_np1, quadratic);

    splitting_criteria_cf_t sp_np1(sp->min_lvl, sp->max_lvl, &phi_interp, 0.5*sp->lip);
    p4est_np1->user_pointer = &sp_np1;

    my_p4est_coarsen(p4est_np1, P4EST_FALSE, coarsen_levelset_cf, NULL);
    my_p4est_refine(p4est_np1, P4EST_FALSE, refine_levelset_cf, NULL);

    // partition the new forest and create new nodes and ghost structures
    my_p4est_partition(p4est_np1, NULL);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    // update the level-set value by interpolating from the old grid
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    for (size_t n = 0; n<nodes_np1->indep_nodes.elem_count; n++) {
      p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_np1->indep_nodes, n);
      p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

      p4est_topidx_t* t2v = p4est->connectivity->tree_to_vertex;
      double *t2c = p4est->connectivity->vertices;
      p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree

      double xyz [] =
      {
        node_x_fr_i(indep_node) + t2c[3 * tr_mm + 0],
        node_y_fr_j(indep_node) + t2c[3 * tr_mm + 1]
  #ifdef P4_TO_P8
        ,
        node_z_fr_k(indep_node) + t2c[3 * tr_mm + 2]
  #endif
      };

      phi_interp.add_point_to_buffer(n, xyz);
    }
    phi_interp.interpolate(phi_p);

    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_destroy(p4est); p4est = p4est_np1;

    std::ostringstream oss; oss << output_dir + "/curvature." << iter+1;

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_FALSE,
                           1, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    w.stop(); w.read_duration();
  }

  p4est->user_pointer = (void*)(sp);
}


void construct_grid_with_reinitializatrion1(p4est_t* &p4est, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes, my_p4est_brick_t* brick, Vec& phi)
{
  splitting_criteria_cf_t *sp = (splitting_criteria_cf_t*)p4est->user_pointer;
  PetscErrorCode ierr;
  parStopWatch w;

  // Now refine the tree
  w.start("initial grid");
  for (int n=0; n<sp->max_lvl; n++){
    my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, NULL);
  }

  // Create the ghost structure
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  nodes = my_p4est_nodes_new(p4est, ghost);

  // create level-set
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, *sp->phi, phi);

  my_p4est_hierarchy_t hierarchy(p4est, ghost, brick);
  my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
  ngbd.init_neighbors();

  my_p4est_level_set ls(&ngbd);
  ls.reinitialize_1st_order_time_2nd_order_space(phi);
  w.stop(); w.read_duration();

  // recreate the grid
  Vec phi_xx, phi_yy;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec phi_zz;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz); CHKERRXX(ierr);
  ngbd.second_derivatives_central(phi, phi_xx, phi_yy, phi_zz);
#else
  ngbd.second_derivatives_central(phi, phi_xx, phi_yy);
#endif

  p4est_t *p4est_tmp = my_p4est_new(p4est->mpicomm, p4est->connectivity, 0, NULL, NULL);
  p4est_ghost_t *ghost_tmp = NULL;
  p4est_nodes_t *nodes_tmp = NULL;

  Vec phi_tmp;
  for (int l=0; l<=sp->max_lvl; l++){
    my_p4est_partition(p4est_tmp, NULL);

    std::ostringstream oss;
    oss << "partial refinement of " << l << "/" << sp->max_lvl;
    w.start(oss.str());

    ghost_tmp = my_p4est_ghost_new(p4est_tmp, P4EST_CONNECT_FULL);
    nodes_tmp = my_p4est_nodes_new(p4est_tmp, ghost_tmp);

    InterpolatingFunctionNodeBase interp(p4est, nodes, ghost, brick, &ngbd);
#ifdef P4_TO_P8
    interp.set_input_parameters(phi, quadratic_non_oscillatory, phi_xx, phi_yy, phi_zz);
#else
    interp.set_input_parameters(phi, quadratic_non_oscillatory, phi_xx, phi_yy);
#endif
    double *phi_tmp_p;
    ierr = VecCreateGhostNodes(p4est_tmp, nodes_tmp, &phi_tmp); CHKERRXX(ierr);
    ierr = VecGetArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);

    // interpolate form old grid
    for (size_t n = 0; n<nodes_tmp->indep_nodes.elem_count; n++){
      p4est_indep_t *indep_node = (p4est_indep_t*)sc_array_index(&nodes_tmp->indep_nodes, n);
      p4est_topidx_t tree_idx = indep_node->p.piggy3.which_tree;

      p4est_topidx_t* t2v = p4est_tmp->connectivity->tree_to_vertex;
      double *t2c = p4est_tmp->connectivity->vertices;
      p4est_topidx_t tr_mm = t2v[P4EST_CHILDREN*tree_idx + 0];  //mm vertex of tree

      double xyz [] =
      {
        node_x_fr_i(indep_node) + t2c[3 * tr_mm + 0],
        node_y_fr_j(indep_node) + t2c[3 * tr_mm + 1]
  #ifdef P4_TO_P8
        ,
        node_z_fr_k(indep_node) + t2c[3 * tr_mm + 2]
  #endif
      };

      interp.add_point_to_buffer(n, xyz);
    }
    interp.interpolate(phi_tmp_p);

    if(l == sp->max_lvl)
      break;

    // mark the cells for refinement
    splitting_criteria_marker_t markers(p4est_tmp, sp->min_lvl, sp->max_lvl, 1.2);
    p4est_locidx_t *q2n = nodes_tmp->local_nodes;

    for (p4est_topidx_t tr_it = p4est_tmp->first_local_tree; tr_it<= p4est_tmp->last_local_tree; tr_it++){
      p4est_tree_t *tree = (p4est_tree_t *)sc_array_index(p4est_tmp->trees, tr_it);
      for (size_t q = 0; q<tree->quadrants.elem_count; q++){
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
        p4est_locidx_t qu_idx = q + tree->quadrants_offset;
        double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

        double f[P4EST_CHILDREN];
        for (short i = 0; i<P4EST_CHILDREN; i++){
          f[i] = phi_tmp_p[q2n[P4EST_CHILDREN*qu_idx + i]];
          if (fabs(f[i]) <= 0.5*markers.lip*dx){
            markers[qu_idx] = P4EST_TRUE;
            continue;
          }
        }

#ifdef P4_TO_P8
        if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
            f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
        if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
        {
          markers[qu_idx] = P4EST_TRUE;
          continue;
        }
      }
    }

    // refine p4est
    p4est_tmp->user_pointer = &markers;
    my_p4est_refine(p4est_tmp, P4EST_FALSE, refine_marked_quadrants, NULL);

    p4est_nodes_destroy(nodes_tmp);
    p4est_ghost_destroy(ghost_tmp);
    ierr = VecRestoreArray(phi_tmp, &phi_tmp_p); CHKERRXX(ierr);
    ierr = VecDestroy(phi_tmp); CHKERRXX(ierr);

    w.stop(); w.read_duration();
  }

  p4est_destroy(p4est);
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);

  p4est = p4est_tmp; p4est->user_pointer = sp;
  ghost = ghost_tmp;
  nodes = nodes_tmp;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_tmp;
}
