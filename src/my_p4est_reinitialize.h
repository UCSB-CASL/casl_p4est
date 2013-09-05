#include <src/CASL_math.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>

void reinitialize_One_Iteration( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *node_neighbors,
                                         double *p0, double *pn, double *pnp1, double limit );

void reinitialize( p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *node_neighbors,
                   Vec& phi_petsc, int number_of_iteration=100, double limit=DBL_MAX );


