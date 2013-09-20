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

#define MAX_LEVEL 11


#undef MIN
#undef MAX
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_reinitialize.h>

using namespace std;

struct circle:CF_2{
  circle(double x0_, double y0_, double r_): x0(x0_), y0(y0_), r(r_) {}
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return sqrt(SQR(x-x0) + SQR(y-y0)) - r;
  }
private:
  double  x0, y0, r;
};


class BCInterfaceDirichlet : public CF_2 {
public:
  double operator() (double x, double y) const
  {
    return x*x + y*y;
//    return x+y;
//    return 1.;
  }
} bc_interface_dirichlet;

class BCInterfaceNeumann : public CF_2
{
public:
    double operator() ( double x, double y) const
    {
        double nx = (x-1.)/sqrt( SQR(x-1.) + SQR(y-1.) );
        double ny = (y-1.)/sqrt( SQR(x-1.) + SQR(y-1.) );
        double norm = nx*nx + ny*ny;
        nx /= norm;
        ny /= norm;

        return 2*x*nx + 2*y*ny;
//        return nx + ny;
//        return 0;
    }
} bc_interface_neumann;


int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode ierr;

  circle circ(1, 1, .5);
  splitting_criteria_cf_t data(0, MAX_LEVEL, &circ, 1.2);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1, w2;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  // Create the connectivity object
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(2, 2, &brick);

  // Now create the forest
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

  // Now refine the tree
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  // Finally re-partition
  p4est_partition(p4est, NULL);

  /* Create the ghost structure */
  p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);

  // generate the node data structure
  nodes = my_p4est_nodes_new(p4est, ghost);

  // Initialize the level-set function
  Vec phi, f;
  ierr = VecCreateGhost(p4est, nodes, &phi); CHKERRXX(ierr);
  ierr = VecCreateGhost(p4est, nodes, &f); CHKERRXX(ierr);

  double *phi_ptr, *f_ptr;
  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(f  , &f_ptr); CHKERRXX(ierr);

  for (size_t i = 0; i<nodes->indep_nodes.elem_count; ++i)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];

    double x = int2double_coordinate_transform(node->x) + tree_xmin;
    double y = int2double_coordinate_transform(node->y) + tree_ymin;

    phi_ptr[p4est2petsc_local_numbering(nodes, i)] = circ(x,y);
    if(circ(x,y)<0)
      f_ptr[p4est2petsc_local_numbering(nodes, i)] = bc_interface_dirichlet(x,y);
    else
      f_ptr[p4est2petsc_local_numbering(nodes, i)] = 0;
  }

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(f  , &f_ptr); CHKERRXX(ierr);

  my_p4est_hierarchy_t hierarchy(p4est, ghost);
  my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
  my_p4est_level_set ls(&brick, p4est, nodes, ghost, &ngbd);

  BoundaryConditions2D bc;
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_dirichlet);
//  bc.setInterfaceType(NEUMANN);
//  bc.setInterfaceValue(bc_interface_neumann);
  int band = 10;

  ls.extend_Over_Interface(phi, f, bc, 2, band);

  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(f  , &f_ptr); CHKERRXX(ierr);

  /* check the error */
  double dx = 2. / pow(2.,(double) data.max_lvl);
  double dy = 2. / pow(2.,(double) data.max_lvl);
  double diag = sqrt(dx*dx + dy*dy);

  double err_max = 0;

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(phi_ptr[n]>0 && phi_ptr[n]<diag*band)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tree_xmin;
      double y = int2double_coordinate_transform(node->y) + tree_ymin;

      err_max = max(err_max, fabs(f_ptr[n] - bc_interface_dirichlet(x,y)));
    }
  }
  double err_max_global;
  MPI_Allreduce(&err_max, &err_max_global, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
  if(p4est->mpirank==0)
    printf("global error extension : %e\n",err_max_global);


  double err[nodes->indep_nodes.elem_count];
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    p4est_locidx_t n_petsc = p4est2petsc_local_numbering(nodes,n);
    if(phi_ptr[n_petsc]>0 && phi_ptr[n_petsc]<diag*band)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tree_xmin;
      double y = int2double_coordinate_transform(node->y) + tree_ymin;

      err[n_petsc] = max(err_max, fabs(f_ptr[n_petsc] - bc_interface_dirichlet(x,y)));
    }
    else
      err[n_petsc] = 0;
  }

  // write the intial data to disk
  my_p4est_vtk_write_all(p4est, nodes, NULL,
                         P4EST_TRUE, P4EST_TRUE,
                         3, 0, "extension_0",
                         VTK_POINT_DATA, "phi", phi_ptr,
                         VTK_POINT_DATA, "f", f_ptr,
                         VTK_POINT_DATA, "error", err);

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(f  , &f_ptr); CHKERRXX(ierr);

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  ierr = VecDestroy(f  ); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

