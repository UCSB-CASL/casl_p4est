//
// Created by youngmin on 10/17/19.
//

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/casl_math.h>

int main( int argc, char** argv )
{
	// Prepare parallel enviroment.
	mpi_environment_t mpi{};
	mpi.init( argc, argv );

	// Stopwatch
	parStopWatch w;
	w.start( "Running example: curvature" );

	// p4est variables
	p4est_t *p4est;
	p4est_nodes_t *nodes;
	p4est_ghost_t *ghost;
	p4est_connectivity_t *conn;
	my_p4est_brick_t brick;

	// Domain size information
	const int N_CELLS = 128;					// We need this number of cells per dimension, and the number of grid points = N_CELLS + 1.
	const double MIN_D = 0;						// Min and max value for any dimension.
	const double MAX_D = 1.0;
	const double H = 1. / ( N_CELLS - 1. );		// Spatial step size.
	const int ITER = 20;						// Number of iterations for level set reinitialization.

	const int n_xyz[] = { N_CELLS, N_CELLS,  N_CELLS };
	const double xyz_min[] = { MIN_D - H, MIN_D - H, MIN_D - H };
	const double xyz_max[] = { MAX_D, MAX_D, MAX_D };
	const int periodic[] = { 0, 0, 0 };
	conn = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	/*!
	 * Circular *non* signed distance function.
 	 */
	struct circle: CF_2
	{
		/*!
		 * Circle constructor.
		 * @param x0_ Center x-coordinate.
		 * @param y0_ Center y-coordinate.
		 * @param r_ Circle radius.
		 */
		circle( double x0_, double y0_, double r_ ) : x0( x0_ ), y0( y0_ ), r( r_ )
		{}

		/*!
		 * Non-signed distance function evaluation at a point.
		 * @param x Point x-coordinate.
		 * @param y Point y-coordinate.
		 * @return 0 if point lies on circunference, < 0 if inside circle, > 0 if outside.
		 */
		double operator()( double x, double y ) const override
		{
			return SQR( x - x0 ) + SQR( y - y0 ) - SQR( r );
		}

	private:
		double x0, y0, r;		// Center coordinates and circle radius.
	};

	char filename[FILENAME_MAX];

	// Create the forest.
	p4est = my_p4est_new( mpi.comm(), conn, 0, nullptr, nullptr );

	// Re-partition the forest.
	my_p4est_partition( p4est, P4EST_TRUE, nullptr );

	// Create ghost layer.
	ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );

	// Create node structure.
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize neighbor node structure.
	my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
	my_p4est_node_neighbors_t neighbors( &hierarchy, nodes );
	neighbors.init_neighbors();

	Vec phi, kappa, normal[P4EST_DIM];
	VecCreateGhostNodes( p4est, nodes, &phi );
	foreach_dimension( dim ) VecCreateGhostNodes( p4est, nodes, &normal[dim] );
	VecDuplicate( phi, &kappa );

	// Compute levelset.
	double *phi_p;
	VecGetArray( phi, &phi_p );

	// Collect phi values.
	double c[2] = {0.5 + ranged_rand( -H/2.0, +H/2.0 ),			// Center coords are randomly chosen around the center of the grid.
				   0.5 + ranged_rand( -H/2.0, +H/2.0 )};
	double r = 0.5 - 2.0 * H;
	circle interface( c[0], c[1], r );							// Non-signed distance function with circular interface.
	sample_cf_on_nodes( p4est, nodes, interface, phi );

	sprintf( filename, "before_%d", P4EST_DIM );
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							1, 0, filename,
							VTK_POINT_DATA, "phi", phi_p );

	// Reinitialize.
	my_p4est_level_set_t ls( &neighbors );
	ls.reinitialize_2nd_order( phi, ITER );

	// Compute normals (returns scaled normal).
	compute_normals( neighbors, phi, normal );

	sprintf( filename, "after_%d", P4EST_DIM );
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							1, 0, filename,
							VTK_POINT_DATA, "phi", phi_p );


	// Compute curvature using div(normal) expression (normal MUST be scaled).
	compute_mean_curvature( neighbors, normal, kappa );

	if( mpi.rank() == 0 )
	{
		PetscPrintf( mpi.comm(), "Done %d\n", 1 );
		PetscPrintf( mpi.comm(), "\n" );
	}

	// destroy vectors
	VecDestroy( phi );
	VecDestroy( kappa );
	foreach_dimension( dim ) VecDestroy( normal[dim] );

	// destroy the structures
	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );

	my_p4est_brick_destroy( conn, &brick );
	w.stop();
	w.read_duration();
}

