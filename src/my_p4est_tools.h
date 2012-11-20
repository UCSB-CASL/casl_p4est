
#ifndef MY_P4EST_TOOLS_H
#define MY_P4EST_TOOLS_H

#include <p4est.h>
#include <p4est_ghost.h>

#ifdef __cplusplus
extern "C" {
#if 0
}
#endif
#endif

typedef struct my_p4est_nodes
{
        p4est_locidx_t  num_local_cells;        /**< elements on this proc */
        p4est_locidx_t  num_owned_nodes;        /**< owned local nodes */
        p4est_locidx_t  num_ghost_nodes;        /**< non-owned local nodes */
        p4est_locidx_t  num_local_nodes;        /**< owned + ghost nodes */
        p4est_gloidx_t  *global_node_offsets;   /**< mpisize + 1 many */

        p4est_locidx_t  *cell_to_node;          /**< four indices per cell */
}
my_p4est_nodes_t;

my_p4est_nodes_t *
my_p4est_nodes_new (p4est_t * p4est, p4est_ghost_t * ghost);

void
my_p4est_nodes_destroy (my_p4est_nodes_t * nodes);

#ifdef __cplusplus
#if 0
{
#endif
}
#endif

#endif /* !MY_P4EST_TOOLS_H */
