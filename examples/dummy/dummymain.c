
#include <p4est_vtk.h>
#include "../../src/my_p4est_nodes.h"
#include "../../src/my_p4est_tools.h"

static int
simple_refine (p4est_t * p4est, p4est_topidx_t which_tree,
        p4est_quadrant_t * quadrant)
{
        return which_tree == 0 && quadrant->level < 2;
}

int
main (int argc, char ** argv)
{
	int mpiret;
	int mpirank;
	int lp;
        int owner;
	int i, j;
        double xy[2];
	MPI_Comm mpicomm;
        p4est_topidx_t which_tree;
	p4est_connectivity_t * conn;
	p4est_t * p4est;
        p4est_ghost_t * ghost;
        my_p4est_brick_t thebrick, *brick = &thebrick;
        my_p4est_nodes_t * nodes;

	mpiret = MPI_Init (&argc, &argv);
	SC_CHECK_MPI (mpiret);

	mpicomm = MPI_COMM_WORLD;
	mpiret = MPI_Comm_rank (mpicomm, &mpirank);
	SC_CHECK_MPI (mpiret);
	printf ("Hello world %d\n", mpirank);

	/* make sure to use a barrier before anything is timed */
	mpiret = MPI_Barrier (mpicomm);
	SC_CHECK_MPI (mpiret);

	/* set log level */
	/* lp = SC_LP_ESSENTIAL; */
	lp = SC_LP_DEFAULT;
	sc_init (mpicomm, 1, 1, NULL, lp);
	p4est_init (NULL, lp);

        /* create some p4est refinement */
        conn = my_p4est_brick_new (1, 2, brick);
	p4est = p4est_new (mpicomm, conn, 0, NULL, NULL);
        p4est_refine (p4est, 1, simple_refine, NULL);
        p4est_partition (p4est, NULL);
        p4est_vtk_write_file (p4est, NULL, "twobrick");

        /* create ghost and node information */
        ghost = p4est_ghost_new (p4est, P4EST_CONNECT_FULL);
        nodes = my_p4est_nodes_new (p4est);

        /* look up a point */
	for (j = 0; j <= 2 * brick->nxytrees[1]; ++j) {
            for (i = 0; i <= 2 * brick->nxytrees[0]; ++i) {
                xy[0] = i * .5;
        	xy[1] = j * .5;
	        which_tree = conn->num_trees - 1;
        	owner = my_p4est_brick_point_lookup (p4est, ghost, brick,
                	                             xy, &which_tree,
                        	                     NULL, NULL);
	        P4EST_INFOF ("Owner of point %g %g is %d\n",
			     xy[0], xy[1], owner);
	    }
	}

        /* clean up */
        my_p4est_nodes_destroy (nodes);
        p4est_ghost_destroy (ghost);
	p4est_destroy (p4est);
	my_p4est_brick_destroy (conn, brick);

	/* make sure internally used memory has been freed */
	sc_finalize ();

	mpiret = MPI_Finalize ();
	SC_CHECK_MPI (mpiret);

	return 0;
}
