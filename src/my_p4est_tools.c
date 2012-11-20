
#include "my_p4est_tools.h"

my_p4est_nodes_t *
my_p4est_nodes_new (p4est_t * p4est, p4est_ghost_t * ghost)
{
	p4est_locidx_t nlc;
	p4est_locidx_t * ctn;
	p4est_gloidx_t * gno;
	my_p4est_nodes_t * nodes;

	nodes = P4EST_ALLOC_ZERO (my_p4est_nodes_t, 1);

	nodes->num_local_cells = nlc = p4est->local_num_quadrants;
	nodes->global_node_offsets = gno =
		P4EST_ALLOC (p4est_gloidx_t, p4est->mpisize + 1);
	nodes->cell_to_node = ctn =
		P4EST_ALLOC (p4est_locidx_t, 4 * nlc);

	return nodes;
}

void
my_p4est_nodes_destroy (my_p4est_nodes_t * nodes)
{
	P4EST_FREE (nodes->cell_to_node);
	P4EST_FREE (nodes->global_node_offsets);
	P4EST_FREE (nodes);
}
