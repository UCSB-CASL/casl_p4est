/**
 * Title: machine_learning_extrapolation
 * Description:
 * Author: Luis Ángel (임 영민)
 * Date Created: 09-29-2020
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#endif

#include <src/casl_geometry.h>
#include <random>

using namespace std;

/**
 * Scalar field to extend over the interface and into Omega+.
 * Notice that it has the same structure as the level-set function because we want to evaluate at the nodes.
 */
class Field: public CF_2
{
public:
	/**
	 * The scalar function to extend: f(x,y).
	 * @param [in] x Query point x-coordinate.
	 * @param [in] y Query point y-coordinate.
	 * @return f(x,y).
	 */
	double operator()( double x, double y ) const override
	{
		return sin( M_PI * x ) * cos( M_PI * y );
	}
};

int main(int argc, char** argv)
{
	const double MIN_D = -1;				// Minimum value for domain (in x, y, and z).  Domain is symmetric.
	const int NUM_TREES_PER_DIM = 2;		// Number of trees per dimension.
	const int REFINEMENT_MAX_LEVEL = 6;		// Maximum level of refinement.
	const int REFINEMENT_BAND_WIDTH = 5;	// Band around interface for grid refinement.
	const int REINIT_NUM_ITER = 10;			// Number of iterations to solve PDE for reinitialization.
	const int EXTENSION_NUM_ITER = 50;		// Number of iterations to solve PDE for extrapolation.
	const int EXTENSION_ORDER = 2;			// Order of extrapolation (0: constant, 1: linear, 2: quadratic).
	const int EXTENSION_BAND_WIDTH = 2;		// Band around interface (in diagonal lengths) to check for extension accuracy.
	const int SAMPLING_BAND_WIDTH = 4;		// Number of grid points to sample in Omega- and along Gamma for learning.

	// Prepare parallel enviroment, although we enforce just a single processor to avoid race conditions when generating
	// datasets.
	mpi_environment_t mpi{};
	mpi.init( argc, argv );
	if( mpi.rank() > 1 )
		throw std::runtime_error( "Only a single process is allowed!" );

	// Random-number generator.
//	std::random_device rd;  				// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( 5489u );
	std::uniform_real_distribution<double> uniformDistribution;

	// Stopwatch.
	parStopWatch watch;
	watch.start();

	// p4est variables.
	p4est_t *p4est;
	p4est_nodes_t *nodes;
	p4est_ghost_t *ghost;
	p4est_connectivity_t *connectivity;
	my_p4est_brick_t brick;

	// Domain information.
	const int n_xyz[] = { NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM };
	const double xyz_min[] = { MIN_D, MIN_D, MIN_D };
	const double xyz_max[] = { -MIN_D, -MIN_D, -MIN_D };
	const int periodic[] = {0, 0, 0};
	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Definining the non-signed distance level-set function to be reinitialized.
	const double THETA = -M_PI_4;				// To vary between [-pi/2, pi/2).
	const double R_THETA = M_PI_2 + THETA;		// Defines the rotation of the local coordinate system for plane.
	const Point2 N( cos( R_THETA ), sin( R_THETA ) );	// Plane normal and center (i.e. point on plane).
	Point2 C( 0.001, 0.001 );					// A point on plane defining the center of local coodinate system.
	geom::Plane plane( N, C );
	splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, REFINEMENT_MAX_LEVEL, &plane, REFINEMENT_BAND_WIDTH );

	// Scalar field to extend.
	Field field;

	// Create the forest using a level set as refinement criterion.
	p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
	p4est->user_pointer = (void *)( &levelSetSC );

	// Partition and refine forest.
	my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf_and_uniform_band, nullptr );
	my_p4est_partition( p4est, P4EST_TRUE, nullptr );

	// Create the ghost (cell) and node structures.
	ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize the neighbor nodes structure.
	my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
	my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
	nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
										// how much we'll need the node neighbors.

	// Smallest quadrant features.
	double dxyz[P4EST_DIM]; 			// Dimensions.
	double H;        					// Minimum side length of the smallest quadrant (i.e. H).
	double diagMin;        				// Diagonal length of the smallest quadrant.
	get_dxyz_min( p4est, dxyz, H, diagMin );

	// A ghosted parallel PETSc vector to store level-set function values.
	Vec phi;
	PetscErrorCode ierr = VecCreateGhostNodes( p4est, nodes, &phi );
	CHKERRXX( ierr );

	// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
	sample_cf_on_nodes( p4est, nodes, plane, phi );

	// Reinitialize level-set using PDE-based approach.
	my_p4est_level_set_t ls( &nodeNeighbors );
	ls.reinitialize_2nd_order( phi, REINIT_NUM_ITER );

	const double *phiReadPtr;
	ierr = VecGetArrayRead( phi, &phiReadPtr );
	CHKERRXX( ierr );

	// Vectors to store the scalar function exactly and numerically extrapolated.
	Vec extField, exactField;
	ierr = VecDuplicate( phi, &extField );
	CHKERRXX( ierr );
	ierr = VecDuplicate( phi, &exactField );
	CHKERRXX( ierr );

	sample_cf_on_nodes( p4est, nodes, field, exactField );		// Now we have the field evaluated exactly at each node.
	ierr = VecCopyGhost( exactField, extField );
	CHKERRXX( ierr );

	// Reset field for phi > 0 (i.e. for nodes in Omega+).
	double *extFieldPtr;
	ierr = VecGetArray( extField, &extFieldPtr );
	CHKERRXX( ierr );

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
	{
		if( phiReadPtr[n] > 0 )
			extFieldPtr[n] = 0;
	}

	// Perform extrapolation using all derivatives (from Daniil's paper).
	ls.extend_Over_Interface_TVD_Full( phi, extField, EXTENSION_NUM_ITER, EXTENSION_ORDER );

	// Calculate extrapolation errors in a band around Gamma, in Omega+.
	Vec errorExtField;			// Extension error parallel vector.
	ierr = VecDuplicate( phi, &errorExtField );
	CHKERRXX( ierr );

	double *errorExtFieldPtr;
	ierr = VecGetArray( errorExtField, &errorExtFieldPtr );
	CHKERRXX( ierr );

	const double *exactFieldReadPtr;
	ierr = VecGetArrayRead( exactField, &exactFieldReadPtr );
	CHKERRXX( ierr );

	for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
	{
		if( phiReadPtr[n] > 0 && phiReadPtr[n] < EXTENSION_BAND_WIDTH * diagMin )
			errorExtFieldPtr[n] = ABS( exactFieldReadPtr[n] - extFieldPtr[n] );
	}

	// Prepare bilinear interpolation of scalar field.
	my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
	interpolation.set_input( extField, linear );

	// When we created the plane level-set, we defined a local coordinate system centered at the point C and with a
	// rotation angle R_THETA.  This effectively creates a coordinate system where the plane's normal vector coincides
	// with the local coordinate system x-axis.
	// Next, we need to collect points that make up samples in the learning process.  For this, consider the following:
	//                    p *                       p = query point (having local x-coordinate > 0).
	//                      ^ n                     n = normal to interface (plane).
	//  k3             y_c  ║ x_c              k2   x_c = x-axis in local coordinate system.
	//   ┌─┬─┬─┬─┬─┬─<══════o.......─┬─┬─┬─┬─┬─┐    y_c = y-axis in local coordinate system.
	//   ├─┼─┼─┼─┼─┼─...............─┼─┼─┼─┼─┼─┤    o = reference point in local coordinates.
	//   ├─┼─┼─┼─┼─┼─...............─┼─┼─┼─┼─┼─┤    ki = corner points in local coordinate system.
	//   ├─┼─┼─┼─┼─┼─...............─┼─┼─┼─┼─┼─┤
	//   ├─┼─┼─┼─┼─┼─...............─┼─┼─┼─┼─┼─┤
	//   └─┴─┴─┴─┴─┴─...............─┴─┴─┴─┴─┴─┘
	//  k1                                     k0
	Point2 p_c( 0.1, MIN_D + 2 * ABS( MIN_D ) * uniformDistribution( gen ) );		// Random point generated in local coordinates w.r.t. plane.
	Point2 projP_c( 0, p_c.y );						// This is basically the 'o' reference point in the above diagram.
	const double STENCIL_WIDTH = SAMPLING_BAND_WIDTH * H;
	Point2 corners_c[4] = {							// The four corners above determine if we have a well defined
		projP_c + Point2( 0, -STENCIL_WIDTH ),		// stencil for sampling.
		projP_c + Point2( 0, +STENCIL_WIDTH ),
		projP_c + Point2( -STENCIL_WIDTH, -STENCIL_WIDTH ),
		projP_c + Point2( -STENCIL_WIDTH, +STENCIL_WIDTH )
	};
	for( const auto& corner_c : corners_c )			// Check that the stencil is fully contained in Omega.
	{
		double x_w = cos( R_THETA ) * corner_c.x - sin( R_THETA ) * corner_c.y + C.x;	// Get x and y components of
		double y_w = sin( R_THETA ) * corner_c.x + cos( R_THETA ) * corner_c.y + C.y;	// corner in world coordinates.
		assert( x_w >= MIN_D && x_w <= -MIN_D && MIN_D <= y_w && y_w <= -MIN_D );
	}

	// Collect sample for query point p by using interpolation of exact values of field at the nodes in Omega-, and of
	// the extended field values values at the adjacent nodes to Gamma in Omega+.
	// Features gro from k0 to k3 above, where x varies slower than y.
	std::vector<double> sample;
	sample.reserve( (SAMPLING_BAND_WIDTH + 1) * (2 * (SAMPLING_BAND_WIDTH + 1) - 1) );
	for( int i = -SAMPLING_BAND_WIDTH; i <= 0; i++ )
	{
		double x_c = projP_c.x + i * H;
		for( int j = -SAMPLING_BAND_WIDTH; j <= SAMPLING_BAND_WIDTH; j++ )
		{
			// Getting coordinates in world coordinate system.
			double y_c = projP_c.y + j * H;
			double x_w = cos( R_THETA ) * x_c - sin( R_THETA ) * y_c + C.x;
			double y_w = sin( R_THETA ) * x_c + cos( R_THETA ) * y_c + C.y;

			// Interpolating scalar field.
			sample.push_back( interpolation( x_w, y_w ) );

			// Some stats.
			std::cout << "plot(" << x_w << "," << y_w << ", 'b.'); "
					  << sample.back() << "; "
					  << ABS( sample.back() - field( x_w, y_w ) ) << ";" << std::endl;
		}
	}

	// Save the grid into vtk
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							4, 0, "machine_learning_extrapolation",
							VTK_POINT_DATA, "phi", phiReadPtr,
							VTK_POINT_DATA, "exactField", exactFieldReadPtr,
							VTK_POINT_DATA, "extField", extFieldPtr,
							VTK_POINT_DATA, "errorExtField", errorExtFieldPtr );

	// Cleaning up.
	ierr = VecRestoreArrayRead( phi, &phiReadPtr );					// Restoring vector pointers.
	CHKERRXX( ierr );
	ierr = VecRestoreArrayRead( exactField, &exactFieldReadPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArray( extField, &extFieldPtr );
	CHKERRXX( ierr );
	ierr = VecRestoreArray( errorExtField, &errorExtFieldPtr );
	CHKERRXX( ierr );

	ierr = VecDestroy( phi );										// Freeing memory.
	CHKERRXX( ierr );
	ierr = VecDestroy( exactField );
	CHKERRXX( ierr );
	ierr = VecDestroy( extField );
	CHKERRXX( ierr );
	ierr = VecDestroy( errorExtField );
	CHKERRXX( ierr );

	// Destroy the structures.
	p4est_nodes_destroy( nodes );
	p4est_ghost_destroy( ghost );
	p4est_destroy( p4est );
	my_p4est_brick_destroy( connectivity, &brick );

	watch.stop();
	watch.read_duration();
}

