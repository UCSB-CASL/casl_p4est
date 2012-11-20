
#include <p4est.h>

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
	MPI_Comm mpicomm;
	p4est_connectivity_t * conn;
	p4est_t * p4est;

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
	conn = p4est_connectivity_new_brick (1, 2, 0, 0);
	p4est = p4est_new (mpicomm, conn, 0, NULL, NULL);
        p4est_refine (p4est, 2, simple_refine, NULL);
        p4est_partition (p4est, NULL);
        p4est_vtk_write_file (p4est, NULL, "twobrick");

        /* clean up */
	p4est_destroy (p4est);
	p4est_connectivity_destroy (conn);

	/* make sure internally used memory has been freed */
	sc_finalize ();

	mpiret = MPI_Finalize ();
	SC_CHECK_MPI (mpiret);

	return 0;
}
