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
#include <src/my_p8est_level_set.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_refine_coarsen.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

int min_level = 5;
int max_level = 7;

int nb_splits = 0;

//#define EXPONENTIAL
//#define QUADRATIC
#define LINEAR
//#define CONSTANT

int n_xyz [] = {2, 2, 2};
double xyz_min [] = {0, 0, 0};
double xyz_max [] = {2, 2, 2};

double x_center = (xyz_max[0]-xyz_min[0]) / 2.;
double y_center = (xyz_max[1]-xyz_min[1]) / 2.;

#ifdef P4_TO_P8
double z_center = (xyz_max[2]-xyz_min[2]) / 2.;
#endif

double r = .512092;
#undef MIN
#undef MAX

using namespace std;

#ifdef P4_TO_P8

class circle : public CF_3{
  double operator()(double x, double y, double z) const {
    return sqrt(SQR(x-x_center) + SQR(y-y_center) + SQR(z-z_center)) - r;
  }
};


class F_EXACT : public CF_3 {
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
    return (x-x_center) + (y-y_center) + (z-z_center);
#endif

#ifdef CONSTANT
    (void) x; (void) y; (void) z;
    return 1.;
#endif
  }
} f_exact;

#else

class circle : public CF_2{
  double operator()(double x, double y) const {
    return sqrt(SQR(x-x_center) + SQR(y-y_center)) - r;
  }
};


class F_EXACT : public CF_2 {
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
    return (x-x_center) + (y-y_center);
#endif

#ifdef CONSTANT
    (void) x; (void) y;
    return 1.;
#endif
  }
} f_exact;

#endif


#ifdef P4_TO_P8
void check_accuracy(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec q, CF_3 *q_ex, Vec err)
#else
void check_accuracy(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec q, CF_2 *q_ex, Vec err)
#endif
{
  PetscErrorCode ierr;
  double *phi_ptr, *q_ptr, *err_ptr;

  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecGetArray(q  , &q_ptr  ); CHKERRXX(ierr);
  ierr = VecGetArray(err, &err_ptr); CHKERRXX(ierr);

  double err_max = 0;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    if(phi_ptr[n]>0)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
      double x = node_x_fr_n(node) + tree_xmin;
      double y = node_y_fr_n(node) + tree_ymin;
#ifdef P4_TO_P8
      double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
      double z = node_z_fr_n(node) + tree_zmin;
#endif

      double nx = x-x_center;
      double ny = y-y_center;
#ifdef P4_TO_P8
      double nz = z-z_center;
#endif
      double norm = sqrt(SQR(nx) + SQR(ny)
                   #ifdef P4_TO_P8
                         + SQR(nz)
                   #endif
                         );
      nx /= norm;
      ny /= norm;
#ifdef P4_TO_P8
      nz /= norm;
#endif

      double x_p = x_center + r * nx;
      double y_p = y_center + r * ny;
#ifdef P4_TO_P8
      double z_p = z_center + r * nz;
#endif

#ifdef P4_TO_P8
      err_ptr[n] = fabs(q_ptr[n] - (*q_ex)(x_p, y_p, z_p));
#else
      err_ptr[n] = fabs(q_ptr[n] - (*q_ex)(x_p, y_p));
#endif

      err_max = max(err_max, err_ptr[n]);
    }
    else
      err_ptr[n] = 0;
  }

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
  ierr = VecRestoreArray(q  , &q_ptr  ); CHKERRXX(ierr);
  ierr = VecRestoreArray(err, &err_ptr); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(err, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (err, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  double err_max_global;
  MPI_Allreduce(&err_max, &err_max_global, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);

  splitting_criteria_cf_t *data = (splitting_criteria_cf_t*) p4est->user_pointer;
  PetscPrintf(p4est->mpicomm, "Level : %d / %d\n", data->min_lvl, data->max_lvl);
  PetscPrintf(p4est->mpicomm, "global error extension : %e\n",err_max_global);
}


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  p4est_t            *p4est;
  p4est_nodes_t      *nodes;
  PetscErrorCode ierr;
  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of additional levels");
  cmd.add_option("iter", "number of iterations");
  cmd.add_option("compute_accuracy", "perform an accuracy check");
  cmd.parse(argc, argv);

  nb_splits = cmd.get("nb_splits", 0);
  min_level = cmd.get("lmin", 2);
  max_level = cmd.get("lmax", 5);

  circle circ;
  splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circ, 1.2);

  parStopWatch w1;
  w1.start("total time");

  /* Create the connectivity object */
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick);

  /* Now create the forest */
  p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  /* Now refine the tree */
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

  /* Finally re-partition */
  my_p4est_partition(p4est, P4EST_TRUE, NULL);

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
  my_p4est_level_set_t ls(&ngbd);

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
#endif

  /* perturb the level-set function */
#ifdef P4_TO_P8
  ls.perturb_level_set_function(phi, MIN(dx, dy, dz)*1e-3);
#else
  ls.perturb_level_set_function(phi, MIN(dx, dy)*1e-3);
#endif

  Vec f, f_ext;
  ierr = VecDuplicate(phi, &f); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &f_ext); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, f_exact, f);

  double *phi_p, *f_p, *f_ext_p;

  ls.extend_from_interface_to_whole_domain_TVD(phi, f, f_ext, cmd.get("iter", 10));

  Vec err;
  if(cmd.contains("compute_accuracy"))
  {
    ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);
    check_accuracy(p4est, nodes, phi, f_ext, &f_exact, err);
  }

  ierr = VecGetArray(phi  , &phi_p  ); CHKERRXX(ierr);
  ierr = VecGetArray(f    , &f_p    ); CHKERRXX(ierr);
  ierr = VecGetArray(f_ext, &f_ext_p); CHKERRXX(ierr);


  /* write the data to disk */
  char file_name[1000];
  const char* out_dir = getenv("OUT_DIR");
  if (out_dir)
    sprintf(file_name, "%s/misc", out_dir);
  else {
    mkdir("misc", 0755);
    sprintf(file_name, "misc/constant_extension_1");
  }
  if(cmd.contains("compute_accuracy"))
  {
    double *err_p;
    ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           4, 0, file_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "f", f_p,
                           VTK_POINT_DATA, "f_extended", f_ext_p,
                           VTK_POINT_DATA, "error", err_p);
    ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
  }
  else
    my_p4est_vtk_write_all(p4est, nodes, NULL,
                           P4EST_TRUE, P4EST_TRUE,
                           3, 0, file_name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "f", f_p,
                           VTK_POINT_DATA, "f_extended", f_ext_p);

  char *path = getenv("PWD");

  PetscPrintf(mpi.comm(), "file saved in ... %s/%s\n", path, file_name);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(f  , &f_p); CHKERRXX(ierr);

  if(cmd.contains("compute_accuracy")) ierr = VecDestroy(err); CHKERRXX(ierr);
  ierr = VecDestroy(phi  ); CHKERRXX(ierr);
  ierr = VecDestroy(f    ); CHKERRXX(ierr);
  ierr = VecDestroy(f_ext); CHKERRXX(ierr);

  // destroy the p4est and its connectivity structure
  p4est_ghost_destroy(ghost);
  p4est_nodes_destroy (nodes);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  w1.stop(); w1.read_duration();

  return 0;
}

