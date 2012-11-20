
#include <p4est_iterate.h>
#include "my_p4est_tools.h"

static void
iter_corner (p4est_iter_corner_info_t * info, void * user_data)
{
	const size_t num_sides = info->sides.elem_count;
	my_p4est_nodes_t * nodes = (my_p4est_nodes_t *) user_data;

	int owner;
	size_t zz;
	p4est_iter_corner_side_t * side;

	for (zz = 0; zz < num_sides; ++zz) {
		side = sc_array_index (&info->sides, zz);
		
		P4EST_LDEBUGF ("ci %lld %lld %lld %d %d txyl %d %x %x %d\n",
			(long long) nodes->num_local_nodes,
			(long long) num_sides, (long long) zz,
			side->corner, side->is_ghost,
			side->treeid,
			side->quad ? side->quad->x : -1,
			side->quad ? side->quad->y : -1,
			side->quad ? side->quad->level : -1);

		if (zz == 0) {
		}
	}
	++nodes->num_local_nodes;
}

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
		P4EST_ALLOC (p4est_locidx_t, P4EST_CHILDREN * nlc);
#ifdef P4EST_DEBUG
	memset (ctn, -1, P4EST_CHILDREN * nlc * sizeof (p4est_locidx_t));
#endif

	p4est_iterate (p4est, ghost, nodes, NULL, NULL, iter_corner);
	
	return nodes;
}

void
my_p4est_nodes_destroy (my_p4est_nodes_t * nodes)
{
	P4EST_FREE (nodes->cell_to_node);
	P4EST_FREE (nodes->global_node_offsets);
	P4EST_FREE (nodes);
}
