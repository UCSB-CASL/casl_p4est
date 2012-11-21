// p4est Library
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

// My files for this project

#include "src/my_p4est_nodes.h"
#include "src/my_p4est_vtk.h"
#include "src/poisson_solver.h"
#include <src/utils.h>

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// CASL
#include <src/utilities.h>

#define SQR(x) (x)*(x)
#define PHI(x,y,r) (r) - sqrt(x*x + y*y)
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

    double phi = PHI(x_c, y_c, 0.5);

    // refinement rule
    if (quadrant->level <data->min_lvl)
        return P4EST_TRUE;
    if (quadrant->level>=data->max_lvl)
        return P4EST_FALSE;
    if (fabs(phi)<d*data->lip)
        return P4EST_TRUE;

    return P4EST_FALSE;
}

static int refine_fn_2(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant){

    refine_user_data_t *data = (refine_user_data_t*)p4est->user_pointer;
    bool condition = true;

    if (quadrant->level < data->min_lvl)
        return P4EST_TRUE;
    else if (quadrant->level >= data->max_lvl)
        return P4EST_FALSE;
    else
        if (condition)
            return P4EST_TRUE;

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
    p4est_gloidx_t     *cumulative_owned_nodes, N;
    int i;
    PetscErrorCode ierr;    

    refine_user_data_t data = {4, 1, 1.2};


    Session session(argc, argv);
    session.init(mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);


    // First we need to create a connectivity object
    // that describes the macro-mesh
    w2.start("connectivity");
    connectivity = p4est_connectivity_new_brick (1, 1, 0, 0);
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

    p4est_vtk_write_file (p4est, NULL, "partitioned");

    // Compute node numbers
    w2.start("nodes");
    nodes = my_p4est_nodes_new (p4est);
    cumulative_owned_nodes = P4EST_ALLOC (p4est_gloidx_t, p4est->mpisize + 1);
    cumulative_owned_nodes[0] = 0;
    for (i = 0; i < p4est->mpisize; ++i) {
        cumulative_owned_nodes[i + 1] =
            cumulative_owned_nodes[i] + nodes->global_owned_indeps[i];
    }
    N = cumulative_owned_nodes[p4est->mpisize];
    w2.stop(); w2.read_duration();


    // destroy the p4est and its connectivity structure
    P4EST_FREE (cumulative_owned_nodes);
    my_p4est_nodes_destroy (nodes);
    p4est_destroy (p4est);
    p4est_connectivity_destroy (connectivity);

    w1.stop(); w1.read_duration();

    return 0;
}
