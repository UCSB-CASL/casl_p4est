// p4est Library
#include <p4est_extended.h>
#include <p4est_bits.h>
#include <p4est_nodes.h>

// My files for this project
#include "srcs/p4est_vtk.h"
#include "srcs/poisson_solver.h"

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>

// CASL
#include <parlib/utilities/utilities.h>

#define SQR(x) (x)*(x)
#define PHI(x,y,r) (r) - sqrt(x*x + y*y)
#define P4EST_TRUE  1
#define P4EST_FALSE 0
typedef int p4est_bool_t;

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

p4est_connectivity_t* my_connectivity(void){
    // Number of vertices in the macro-mesh
    const p4est_topidx_t num_vertices = 9;

    // Number of trees in the macro-mesh
    const p4est_topidx_t num_trees = 4;

    // TODO: What is this for?
    const p4est_topidx_t num_ctt = 0;

    // Coordinates for the vertices of the macro-mesh; could be in any order
    const double vertices[] = {
        -1, -1,  0,
         0, -1,  0,
         1, -1,  0,
        -1,  0,  0,
         0,  0,  0,
         1,  0,  0,
        -1,  1,  0,
         0,  1,  0,
         1,  1,  0
    };

    // What are 4 corners of the trees? This should be z-ordered
    const p4est_topidx_t tree_to_vertex[] = {
        0, 1, 3, 4,
        1, 2, 4, 5,
        3, 4, 6, 7,
        4, 5, 7, 8
    };

    // What are the neibors of this tree?
    const p4est_topidx_t tree_to_tree[] = {
        0, 1, 0, 2,
        0, 1, 1, 3,
        2, 3, 0, 2,
        2, 3, 1, 3
    };

    // What are the corresponding faces that this tree is connected to?
    const int8_t tree_to_face[] = {
        0, 0, 2, 2,
        1, 1, 2, 2,
        0, 0, 3, 3,
        1, 1, 3, 3
    };

    return p4est_connectivity_new_copy (num_vertices, num_trees, 0,
                                        vertices, tree_to_vertex,
                                        tree_to_tree, tree_to_face,
                                        NULL, &num_ctt, NULL, NULL);
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
    p4est_t            *p4est;
    p4est_connectivity_t *connectivity;
    PetscErrorCode ierr;    

    refine_user_data_t data = {8, 4, 1.2};


    Session session(argc, argv);
    session.init(mpi->mpicomm);

    parStopWatch w1, w2;
    w1.start("total time");

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);


    // First we need to create a connectivity object that describes the macro-mesh
    w2.start("connectivity");
    connectivity = my_connectivity();
    w2.stop(); w2.read_duration();

    // Now create the forest
    w2.start("p4est generation");
    p4est = p4est_new_ext (mpi->mpicomm, connectivity, 0, data.min_lvl, P4EST_FALSE,
                           0, NULL, NULL);
    w2.stop(); w2.read_duration();

    // Now refine the tree
    w2.start("refine");
    p4est->user_pointer = (void*)(&data);
    p4est_refine(p4est, P4EST_TRUE, refine_circle, NULL);
    w2.stop(); w2.read_duration();

    // Now balence
    w2.start("balance");
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    w2.stop(); w2.read_duration();

    // Finally re-partition
    w2.start("partition");
    p4est_partition(p4est, NULL);
    w2.stop(); w2.read_duration();

    // save the new forest
    /*
   * You could also use the following function:
   *  p4est_vtk_write_file(p4est, NULL, "mine");
   * Problem is it scales the cells for some reason and you end up with
   * disconnected cells!
   */

    // Loop over quadrants and compute the cell-center phi
    double *phi_cell = new double[p4est->local_num_quadrants];
    double *tree_id  = new double[p4est->local_num_quadrants];
    double *proc_id  = new double[p4est->local_num_quadrants];

    w2.start("phi_cell");
    p4est_topidx_t quad_count = 0;
    for (p4est_topidx_t tr = p4est->first_local_tree; tr<=p4est->last_local_tree; ++tr){
        p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr);

        for (p4est_topidx_t qu = 0; qu<tree->quadrants.elem_count; ++qu){
            p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, qu);

            double qh = P4EST_QUADRANT_LEN(quad->level);

            double x = (quad->x + 0.5*qh)/P4EST_ROOT_LEN;
            double y = (quad->y + 0.5*qh)/P4EST_ROOT_LEN;

            c2p_coordinate_transform(p4est, tr, &x, &y, NULL);

            phi_cell[quad_count] = PHI(x, y, 0.5);
            tree_id[quad_count]  = tr;
            proc_id[quad_count]  = mpi->mpirank;

            quad_count++;
        }
    }
    w2.stop(); w2.read_duration();


    // Now lets loop over the nodes and compute a node-version
    w2.start("phi_node");
    p4est_nodes_t *nodes = p4est_nodes_new(p4est, NULL);
    sc_array_t *indeps   = &nodes->indep_nodes;
    double *phi_node = new double[indeps->elem_count];

    for (p4est_topidx_t ni=0; ni<indeps->elem_count; ++ni){
        p4est_indep_t *in = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, ni);

        double x = in->x; x /= P4EST_ROOT_LEN;
        double y = in->y; y /= P4EST_ROOT_LEN;

        c2p_coordinate_transform(p4est, in->p.which_tree, &x, &y, NULL);

        phi_node[ni] = PHI(x, y, 0.5);
    }
    w2.stop(); w2.read_duration();

    // Now lets solve a poisson equation
    struct:CF_2{
        double operator()(double x, double y) const {
            return sin(2*M_PI*x)*sin(2*M_PI*y);
        }
    } uex;

    struct:CF_2{
        double operator()(double x, double y) const {
            return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);
        }
    } f;

    w2.start("petsc");
    Vec sol, sol_ex;
    PoissonSolver solver(p4est, uex, f);
    solver.setUpNegativeLaplaceSystem();
    solver.solve(sol, sol_ex);
    w2.stop(); w2.read_duration();

    // Get a pointer to the solution
    double *sol_ptr, *sol_ex_ptr;

    ierr = VecGetArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);


    w2.start("vtk");
    my_p4est_vtk_write_all(p4est, NULL, 1.0,
                           1, 5, "mine",
                           VTK_CELL_DATA, "phi_cell", phi_cell,
                           VTK_POINT_DATA, "phi_node", phi_node,
                           VTK_CELL_DATA, "proc_id", proc_id,
                           VTK_CELL_DATA, "tree_id", tree_id,
                           VTK_CELL_DATA, "sol", sol_ptr,
                           VTK_CELL_DATA, "sol_ex", sol_ex_ptr);

    w2.stop(); w2.read_duration();

    ierr = VecRestoreArray(sol, &sol_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_ex, &sol_ex_ptr); CHKERRXX(ierr);

    // compute the maximum error and check convergence
    ierr = VecAXPY(sol, -1.0, sol_ex); CHKERRXX(ierr);
    double err_max;
    ierr = VecNorm(sol, NORM_INFINITY, &err_max); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi->mpicomm, "Maximum err = %1.4e\n", err_max); CHKERRXX(ierr);

    delete [] phi_cell;
    delete [] phi_node;
    delete [] tree_id;
    delete [] proc_id;

    // destroy the p4est and its connectivity structure
    p4est_destroy (p4est);
    p4est_connectivity_destroy (connectivity);
    p4est_nodes_destroy(nodes);

    w1.stop(); w1.read_duration();

    return 0;
}
