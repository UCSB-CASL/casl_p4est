// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// My files for this project

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// casl_p4est
#include <src/utilities.h>
#include <src/my_p4est_vtk.h>
#include <src/poisson_solver.h>
#include <src/utils.h>
#include <src/my_p4est_nodes.h>
#include <src/semi_lagrangian.h>
#include <src/my_p4est_tools.h>

#define SQR(x) (x)*(x)
#define PHI(x,y) 0.5 - sqrt(SQR(x) + SQR(y))
#define P4EST_TRUE  1
#define P4EST_FALSE 0
typedef int p4est_bool_t;

#if (PETSC_VERSION_MINOR <= 1)
#undef CHKERRXX
#define CHKERRXX
#endif

typedef struct
{
    MPI_Comm            mpicomm;
    int                 mpisize;
    int                 mpirank;
}
mpi_context_t;

using namespace std;


typedef struct {
  double phi[4];
  int max_lvl, min_lvl;
  double lip;
} refine_user_data_t;


static int refine_circle (p4est_t * p4est, p4est_topidx_t which_tree,
                          p4est_quadrant_t * quadrant)
{
    refine_user_data_t *data = (refine_user_data_t*)p4est->user_pointer;

    const p4est_qcoord_t qh = P4EST_QUADRANT_LEN (quadrant->level);

    double dx_smallest = 1.0/P4EST_ROOT_LEN;
    double dy_smallest = 1.0/P4EST_ROOT_LEN;

    double x_c, y_c;
    x_c = (quadrant->x + 0.5*qh)*dx_smallest;
    y_c = (quadrant->y + 0.5*qh)*dy_smallest;

    c2p_coordinate_transform(p4est, which_tree, &x_c, &y_c, NULL);

    double x_ver[P4EST_CHILDREN], y_ver[P4EST_CHILDREN];
    x_ver[0] = x_ver[2] = quadrant->x * dx_smallest;
    x_ver[1] = x_ver[3] = x_ver[0] + qh*dx_smallest;
    y_ver[0] = y_ver[1] = quadrant->y * dy_smallest;
    y_ver[2] = y_ver[3] = y_ver[0] + qh*dy_smallest;

    for (int i=0; i<P4EST_CHILDREN; i++)
        c2p_coordinate_transform(p4est, which_tree, &x_ver[i], &y_ver[i], NULL);

    double d1  = sqrt(SQR(x_ver[0]-x_ver[3]) + SQR(y_ver[0]-y_ver[3]));
    double d2  = sqrt(SQR(x_ver[1]-x_ver[2]) + SQR(y_ver[1]-y_ver[2]));
    double d = 0.5 * (d1+d2);

    double phi = PHI(x_c-1.0, y_c-1.0);

    // refinement rule
    if (quadrant->level <data->min_lvl)
        return P4EST_TRUE;
    if (quadrant->level>=data->max_lvl)
        return P4EST_FALSE;
    if (fabs(phi)<d*data->lip)
        return P4EST_TRUE;

    return P4EST_FALSE;
}

static int refine_dummy (p4est_t * p4est, p4est_topidx_t which_tree,
                         p4est_quadrant_t * quadrant)
{
  if (quadrant->level < 1 && quadrant->x == 0 && quadrant->y == 0 && (which_tree == 0 || which_tree == 3) )
    return P4EST_TRUE;
  else
    return P4EST_FALSE;
}

class Session{
    int argc;
    char** argv;
    PetscErrorCode ierr;

public:
    Session(int argc_, char* argv_[])
        : argc(argc_), argv(argv_)
    {}
    ~Session(){
        sc_finalize ();
        ierr = PetscFinalize(); CHKERRXX(ierr);
    }

    void init(MPI_Comm mpicomm){
        ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
        sc_init (mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
        p4est_init (NULL, SC_LP_DEFAULT);
    }
};

int main (int argc, char* argv[]){

    mpi_context_t       mpi_context, *mpi = &mpi_context;
    mpi->mpicomm = MPI_COMM_WORLD;
    p4est_connectivity_t *connectivity;
    p4est_t            *p4est;
    my_p4est_nodes_t   *nodes;
    p4est_locidx_t     *e2n;

    PetscErrorCode ierr;    

    refine_user_data_t data = {6, 0, 1.0};


    Session session(argc, argv);
    session.init(mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);


    // First we need to create a connectivity object
    // that describes the macro-mesh
    w2.start("connectivity");
    connectivity = p4est_connectivity_new_brick (2, 2, 0, 0);
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = p4est_new_ext (mpi->mpicomm, connectivity, 0, data.min_lvl,
                           P4EST_TRUE, 0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine");
    p4est->user_pointer = (void*)(&data);
    p4est_refine(p4est, P4EST_TRUE, refine_circle, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    nodes = my_p4est_nodes_new(p4est);
    e2n = nodes->local_nodes;

    Vec phi, vx, vy;

    ierr = VecGhostCreate_p4est(p4est, nodes, &phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &vx); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &vy); CHKERRXX(ierr);

    // Initialize level-set function
    double *phi_val, *vx_val, *vy_val;
    ierr = VecGetArray(phi, &phi_val); CHKERRXX(ierr);
    ierr = VecGetArray(vx , &vx_val ); CHKERRXX(ierr);
    ierr = VecGetArray(vy , &vy_val ); CHKERRXX(ierr);

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; ++i)
    {
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
      p4est_topidx_t v_pp = connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 3];

      double tree_xmin = connectivity->vertices[3*v_mm + 0];
      double tree_xmax = connectivity->vertices[3*v_pp + 0];
      double tree_ymin = connectivity->vertices[3*v_mm + 1];
      double tree_ymax = connectivity->vertices[3*v_pp + 1];

      double x = (double)node->x / (double)P4EST_ROOT_LEN; x = x*(tree_xmax-tree_xmin) + tree_xmin;
      double y = (double)node->y / (double)P4EST_ROOT_LEN; y = y*(tree_ymax-tree_ymin) + tree_ymin;

      phi_val[i] = PHI(x-1.0,y-1.0);

      double vx   = 0.25 * (x - 1.0);
      double vy   = 0.25 * (y - 1.0);

      vx_val[i]  = vx;
      vy_val[i]  = vy;
    }

    ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    Vec phi_cell;
    ierr = VecCreate(p4est->mpicomm, &phi_cell); CHKERRXX(ierr);
    ierr = VecSetSizes(phi_cell, p4est->local_num_quadrants, p4est->global_num_quadrants); CHKERRXX(ierr);
    ierr = VecSetFromOptions(phi_cell); CHKERRXX(ierr);

    double *v_phi_cell;
    ierr = VecGetArray(phi_cell, &v_phi_cell); CHKERRXX(ierr);

    for (p4est_topidx_t tr_it = p4est->first_local_tree; tr_it <= p4est->last_local_tree; ++tr_it)
    {
      p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_it);
      for (p4est_locidx_t qu = 0; qu < tree->quadrants.elem_count; ++qu)
      {
        p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, qu);

        double qh = (double)P4EST_QUADRANT_LEN(quad->level);

        double xy [] =
        {
          ((double)quad->x + 0.5*qh)/(double)P4EST_ROOT_LEN,
          ((double)quad->y + 0.5*qh)/(double)P4EST_ROOT_LEN
        };

        c2p_coordinate_transform(p4est, tr_it, &xy[0], &xy[1], NULL);

        p4est_topidx_t which_tree = tr_it;
        p4est_locidx_t quad_locidx;
        p4est_quadrant_t *quadrant;

        my_p4est_brick_point_lookup(p4est, xy, &which_tree, &quad_locidx, &quadrant);
        p4est_tree_t *tmp_tr = p4est_tree_array_index(p4est->trees, which_tree);
        quad_locidx += tmp_tr->quadrants_offset;

        p4est_locidx_t nodes_locidx [] =
        {
          e2n[quad_locidx*P4EST_CHILDREN + 0],
          e2n[quad_locidx*P4EST_CHILDREN + 1],
          e2n[quad_locidx*P4EST_CHILDREN + 2],
          e2n[quad_locidx*P4EST_CHILDREN + 3]
        };

        for (int i = 0 ; i<4; ++i)
          nodes_locidx[i] = p4est2petsc_local_numbering(nodes, nodes_locidx[i]);


        double F [] =
        {
          phi_val[nodes_locidx[0]],
          phi_val[nodes_locidx[1]],
          phi_val[nodes_locidx[2]],
          phi_val[nodes_locidx[3]]
        };

        v_phi_cell[quad_locidx] = bilinear_interpolation(p4est, which_tree, quadrant, F, xy[0], xy[1]);
      }
    }

    my_p4est_vtk_write_all(p4est, NULL, 1.0,
                           3, 1, "levelset_init",
                           VTK_POINT_DATA, "phi", phi_val,
                           VTK_CELL_DATA,  "phi_c", v_phi_cell,
                           VTK_POINT_DATA, "vx", vx_val,
                           VTK_POINT_DATA, "vy", vy_val);

    ierr = VecRestoreArray(phi_cell, &v_phi_cell); CHKERRXX(ierr);
    ierr = VecDestroy(&phi_cell); CHKERRXX(ierr);

    // Restore temporary objects
    ierr = VecRestoreArray(phi, &phi_val); CHKERRXX(ierr);
    ierr = VecRestoreArray(vx,  &vx_val ); CHKERRXX(ierr);
    ierr = VecRestoreArray(vy,  &vy_val ); CHKERRXX(ierr);

    semi_lagrangian SL(p4est, nodes);

    double tf  = 1.0;
    double dt  = 0.1;
    int tc     = 0;
    for (double t = 0; t<tf; t += dt, ++tc)
    {
      SL.advect(vx, vy, dt, phi);

      // Save stuff
      std::ostringstream oss; oss << "levelset." << tc;

      ierr = VecGetArray(phi, &phi_val); CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est, NULL, 1.0,
                             1, 0, oss.str().c_str(),
                             VTK_POINT_DATA, "phi", phi_val);
      ierr = VecRestoreArray(phi, &phi_val); CHKERRXX(ierr);
    }

    // Destroy PETSc objects
    ierr = VecDestroy(&phi); CHKERRXX(ierr);
    ierr = VecDestroy(&vx ); CHKERRXX(ierr);
    ierr = VecDestroy(&vy ); CHKERRXX(ierr);

    // destroy the p4est and its connectivity structure
    my_p4est_nodes_destroy (nodes);
    p4est_destroy (p4est);
    p4est_connectivity_destroy (connectivity);

    w1.stop(); w1.read_duration();

    return 0;
}
