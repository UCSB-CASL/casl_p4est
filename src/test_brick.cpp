#include "test_brick.h"

test_brick::test_brick(int argc, char **argv)
{
    int                 i, j;
      int                 l, m;
      MPI_Comm            mpicomm;
      int                 mpiret;
      int                 size, rank;
      p4est_connectivity_t *conn;
    #ifdef P4_TO_P8
      int                 k, n;
    #endif

      mpiret = MPI_Init (&argc, &argv);
      SC_CHECK_MPI (mpiret);
      mpicomm = MPI_COMM_WORLD;
      mpiret = MPI_Comm_size (mpicomm, &size);
      SC_CHECK_MPI (mpiret);
      mpiret = MPI_Comm_rank (mpicomm, &rank);
      SC_CHECK_MPI (mpiret);

      sc_init (mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
      p4est_init (NULL, SC_LP_DEFAULT);

      for (i = 1; i <= 5; i++) {
        for (j = 1; j <= 5; j++) {
    #ifdef P4_TO_P8
          for (k = 1; k <= 5; k++) {
    #endif
            for (l = 0; l < 2; l++) {
              for (m = 0; m < 2; m++) {
    #ifdef P4_TO_P8
                for (n = 0; n < 2; n++) {
    #endif
    #ifndef P4_TO_P8
                  conn = p4est_connectivity_new_brick (i, j, l, m);
                  check_brick (conn, i, j, l, m);
    #else
                  conn = p4est_connectivity_new_brick (i, j, k, l, m, n);
                  check_brick (conn, i, j, k, l, m, n);
    #endif
                  p4est_connectivity_destroy (conn);
    #ifdef P4_TO_P8
                }
    #endif
              }
            }
    #ifdef P4_TO_P8
          }
    #endif
        }
      }

      /* clean up and exit */
      sc_finalize ();

      mpiret = MPI_Finalize ();
      SC_CHECK_MPI (mpiret);



}
