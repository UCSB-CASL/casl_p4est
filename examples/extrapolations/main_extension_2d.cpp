// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>


#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_levelset.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_levelset.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

int min_level = 5;
int max_level = 7;

int nb_splits = 0;

#define EXPONENTIAL
//#define QUADRATIC
//#define LINEAR
//#define CONSTANT

int band = 10;
int order = 2;

//int brick_nx = 1;
//int brick_ny = 1;
//double x_center = .5;
//double y_center = .5;
//double r = .25112;

int brick_nx = 2;
int brick_ny = 2;
double x_center = (double) brick_nx / 2.;
double y_center = (double) brick_ny / 2.;
double r = .512092;

#ifdef P4_TO_P8
int brick_nz = 2;
double z_center = (double) brick_nz / 2.;
#endif

//int brick_nx = 3;
//int brick_ny = 3;
//double x_center = 1.5;
//double y_center = 1.5;
//double r = .5192827;

BoundaryConditionType bc_interface_type = DIRICHLET;
//BoundaryConditionType bc_interface_type = NEUMANN;

#undef MIN
#undef MAX

using namespace std;

#ifdef P4_TO_P8

class circle : public CF_3{
  double operator()(double x, double y, double z) const {
    return sqrt(SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center)) - r;
  }
};


class BCInterfaceDirichlet : public CF_3 {
public:
  double operator() (double x, double y, double z) const
  {
#ifdef EXPONENTIAL
    return exp(SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center));
#endif

#ifdef QUADRATIC
    return SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center);
#endif

#ifdef LINEAR
    return 1000*((x-x_center) + (y-y_center) + (z-z_center));
#endif

#ifdef CONSTANT
    (void) x; (void) y; (void) z;
    return 1.;
#endif
  }
} bc_interface_dirichlet;

class BCInterfaceNeumann : public CF_3
{
public:
    double operator() ( double x, double y, double z) const
    {
        double nx = (x-x_center)/sqrt( SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center) );
        double ny = (y-y_center)/sqrt( SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center) );
        double nz = (z-z_center)/sqrt( SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center) );
        double norm = nx*nx + ny*ny + nz*nz;
        nx /= norm;
        ny /= norm;
        nz /= norm;

#ifdef EXPONENTIAL
        return (2*(x-x_center)*nx + 2*(y-y_center)*ny + 2*(z-z_center)*nz) * exp(SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center));
#endif

#ifdef QUADRATIC
        return 2*(x-x_center)*nx + 2*(y-y_center)*ny + 2*(z-z_center)*nz;
#endif

#ifdef LINEAR
        return nx + ny + nz;
#endif

#ifdef CONSTANT
        return 0;
#endif
    }
} bc_interface_neumann;

#else

class circle : public CF_2{
  double operator()(double x, double y) const {
    return sqrt(SQR(x-x_center) + SQR(y-y_center)) - r;
  }
};


class BCInterfaceDirichlet : public CF_2 {
public:
  double operator() (double x, double y) const
  {
#ifdef EXPONENTIAL
    return exp(SQR(x-x_center) + SQR(y-y_center));
#endif

#ifdef QUADRATIC
    return SQR(x-x_center) + SQR(y-y_center);
#endif

#ifdef LINEAR
    return 1000*((x-x_center) + (y-y_center));
#endif

#ifdef CONSTANT
    (void) x; (void) y;
    return 1.;
#endif
  }
} bc_interface_dirichlet;

class BCInterfaceNeumann : public CF_2
{
public:
    double operator() ( double x, double y) const
    {
        double nx = (x-x_center)/sqrt( SQR(x-x_center) + SQR(y-y_center) );
        double ny = (y-y_center)/sqrt( SQR(x-x_center) + SQR(y-y_center) );
        double norm = nx*nx + ny*ny;
        nx /= norm;
        ny /= norm;

#ifdef EXPONENTIAL
        return (2*(x-x_center)*nx + 2*(y-y_center)*ny) * exp(SQR(x-x_center) + SQR(y-y_center));
#endif

#ifdef QUADRATIC
        return 2*(x-x_center)*nx + 2*(y-y_center)*ny;
#endif

#ifdef LINEAR
        return nx + ny;
#endif

#ifdef CONSTANT
        return 0;
#endif
    }
} bc_interface_neumann;

#endif


int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode ierr;
  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of additional levels");
  cmd.add_option("order", "order of the extrapolating polynomial");
  cmd.parse(argc, argv);

  nb_splits = cmd.get("nb_splits", 0);
  order = cmd.get("order", 2);
  min_level = cmd.get("lmin", 2);
  max_level = cmd.get("lmax", 5);

  circle circ;
  splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circ, 1.2);

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  parStopWatch w1;
  w1.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

  /* Create the connectivity object */
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  connectivity = my_p4est_brick_new(brick_nx, brick_ny, brick_nz, &brick);
#else
  connectivity = my_p4est_brick_new(brick_nx, brick_ny, &brick);
#endif

  /* Now create the forest */
  p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

  /* Now refine the tree */
  p4est->user_pointer = (void*)(&data);
  p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  /* Finally re-partition */
  p4est_partition(p4est, NULL);

  /* Create the ghost structure */
  p4est_ghost_t *ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

  /* generate the node data structure */
  nodes = my_p4est_nodes_new(p4est, ghost);

  /* Initialize the level-set function */
  Vec phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, circ, phi);

  my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
  my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
  my_p4est_level_set ls(&ngbd);

  /* find dx and dy smallest */
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  /* perturb the level-set function */
#ifdef P4_TO_P8
  ls.perturb_level_set_function(phi, MIN(dx, dy, dz)*1e-3);
#else
  ls.perturb_level_set_function(phi, MIN(dx, dy)*1e-3);
#endif

  Vec f;
  ierr = VecDuplicate(phi, &f); CHKERRXX(ierr);

  Vec bc_vec;
  ierr = VecDuplicate(phi, &bc_vec); CHKERRXX(ierr);

  double *phi_ptr, *f_ptr, *bc_vec_ptr;
  ierr = VecGetArray(phi   , &phi_ptr   ); CHKERRXX(ierr);
  ierr = VecGetArray(f     , &f_ptr     ); CHKERRXX(ierr);
  ierr = VecGetArray(bc_vec, &bc_vec_ptr); CHKERRXX(ierr);

  for (size_t n = 0; n<nodes->indep_nodes.elem_count; ++n)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = connectivity->vertices[3*v_mm + 0];
    double tree_ymin = connectivity->vertices[3*v_mm + 1];
    double x = node_x_fr_i(node) + tree_xmin;
    double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
    double tree_zmin = connectivity->vertices[3*v_mm + 2];
    double z = node_z_fr_k(node) + tree_zmin;
#endif

#ifdef P4_TO_P8
    bc_vec_ptr[n] = bc_interface_type==DIRICHLET ? bc_interface_dirichlet(x,y,z) : bc_interface_neumann(x,y,z) ;
#else
    bc_vec_ptr[n] = bc_interface_type==DIRICHLET ? bc_interface_dirichlet(x,y) : bc_interface_neumann(x,y) ;
#endif

    if(phi_ptr[n]<0)
#ifdef P4_TO_P8
      f_ptr[n] = bc_interface_dirichlet(x,y,z);
#else
      f_ptr[n] = bc_interface_dirichlet(x,y);
#endif
    else
      f_ptr[n] = 0;
  }

  ierr = VecRestoreArray(phi   , &phi_ptr   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(f     , &f_ptr     ); CHKERRXX(ierr);
  ierr = VecRestoreArray(bc_vec, &bc_vec_ptr); CHKERRXX(ierr);

#ifdef P4_TO_P8
  BoundaryConditions3D bc;
#else
  BoundaryConditions2D bc;
#endif

  bc.setInterfaceType(bc_interface_type);
  if(bc_interface_type==DIRICHLET)
    bc.setInterfaceValue(bc_interface_dirichlet);
  else
    bc.setInterfaceValue(bc_interface_neumann);

//  ls.extend_Over_Interface(phi, f, bc, order, band);
  ls.extend_Over_Interface(phi, f, bc_interface_type, bc_vec, order, band);

  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(f  , &f_ptr); CHKERRXX(ierr);

  /* check the error */
  double err_max = 0;
  double xm=0, ym=0;
#ifdef P4_TO_P8
  double zm=0;
#endif
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(phi_ptr[n]>0 && phi_ptr[n]<diag*band)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
      double x = node_x_fr_i(node) + tree_xmin;
      double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
      double tree_zmin = connectivity->vertices[3*v_mm + 2];
      double z = node_z_fr_k(node) + tree_zmin;
      if(fabs(f_ptr[n] - bc_interface_dirichlet(x,y,z)) > err_max)
#else
      if(fabs(f_ptr[n] - bc_interface_dirichlet(x,y)) > err_max)
#endif
      {
        xm = x;
        ym = y;
#ifdef P4_TO_P8
        zm = z;
#endif
      }

#ifdef P4_TO_P8
      err_max = max(err_max, fabs(f_ptr[n] - bc_interface_dirichlet(x,y,z)));
#else
      err_max = max(err_max, fabs(f_ptr[n] - bc_interface_dirichlet(x,y)));
#endif
    }
  }
  double err_max_global;
  MPI_Allreduce(&err_max, &err_max_global, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
  if(p4est->mpirank==0)
  {
    printf("Level : %d / %d\n", min_level+nb_splits, max_level+nb_splits);
    printf("global error extension : %e\n",err_max_global);
#ifdef P4_TO_P8
    printf("at point (%.10e , %.10e , %.10e)\n", xm, ym, zm);
#else
    printf("at point (%.10e , %.10e)\n", xm, ym);
#endif
  }



  double err[nodes->indep_nodes.elem_count];
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    if(phi_ptr[n]>0 && phi_ptr[n]<diag*band)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
      double x = node_x_fr_i(node) + tree_xmin;
      double y = node_y_fr_j(node) + tree_ymin;

#ifdef P4_TO_P8
      double tree_zmin = connectivity->vertices[3*v_mm + 2];
      double z = node_z_fr_k(node) + tree_zmin;
      err[n] = fabs(f_ptr[n] - bc_interface_dirichlet(x,y,z));
#else
      err[n] = fabs(f_ptr[n] - bc_interface_dirichlet(x,y));
#endif

    }
    else
      err[n] = 0;
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

  ierr = VecDestroy(phi   ); CHKERRXX(ierr);
  ierr = VecDestroy(f     ); CHKERRXX(ierr);
  ierr = VecDestroy(bc_vec); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

