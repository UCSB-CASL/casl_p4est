//
// Created by Im YoungMin on 6/2/20.
//

/**
 * Testing the sine wave level-set function that uses arc-length parameterization.
 *
 * Developer: Luis Ángel.
 * Date: June 2, 2020.
 */

// System.
#include <stdexcept>
#include <iostream>

#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_fast_sweeping.h>
#include <src/my_p4est_nodes_along_interface.h>
//#include <src/my_p4est_level_set.h>

#include <src/petsc_compatibility.h>
#include <random>
#include "arclength_parameterized_sine_2d.h"


/**
 * Generate the sample row of level-set function values and target h\kappa for a node that has been found next to the
 * sine wave interface.  We assume that this query node is effectively adjacent to \Gamma.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] neighbors Pointer to neighbors data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] sine The level-set function with a sinusoidal interface.
 * @param [in] gen Random number generator.
 * @param [in] normalDistribution A standard normal random distribution generator.
 * @param [out] distances True normal distances from full neighborhood to sine wave using Newton-Raphson's root-finding.
 * @return Vector with sampled phi values and target dimensionless curvature.
 * @throws runtime exception if distance between original projected point on interface and point found by Newton-Raphson
 * are farther than H and if Newton-Raphson's method converged to a local minimum (didn't get to zero).
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const my_p4est_node_neighbors_t *neighbors, const double *phiReadPtr, const ArcLengthParameterizedSine& sine,
	std::mt19937& gen, std::normal_distribution<double>& normalDistribution, std::vector<double>& distances )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// (Reinitialized) level-set function values and target h\kappa.
	distances.clear();
	distances.reserve( NUM_COLUMNS );					// Include h\kappa as well.

	int s;												// Index to fill in the sample vector.
	double grad[P4EST_DIM];
	double gradNorm;
	double xyz[P4EST_DIM];
	double pOnInterfaceX, pOnInterfaceY;
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	double u, v, valOfDerivative, centerU;
	double dx, dy, newDistance;
	int iterations;
	for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
	{
		sample[s] = phiReadPtr[stencil[s]];				// This is the distance obtained after reinitialization.

		// To find the true distance we need the neighborhood of each stencil node.
		neighbors->get_neighbors( stencil[s], qnnnPtr );
		qnnnPtr->gradient( phiReadPtr, grad );
		gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );	// Get the unit gradient.

		// Approximate position of point projected on interface.
		node_xyz_fr_n( stencil[s], p4est, nodes, xyz );
		pOnInterfaceX = xyz[0] - grad[0] / gradNorm * sample[s];
		pOnInterfaceY = xyz[1] - grad[1] / gradNorm * sample[s];

		// Transform point on interface to sine-wave canonical coordinates.
		sine.toCanonicalCoordinates( xyz[0], xyz[1] );
		sine.toCanonicalCoordinates( pOnInterfaceX, pOnInterfaceY );
		pOnInterfaceY = sine.getA() * sin( sine.getOmega() * pOnInterfaceX );	// Better approximation to y on \Gamma.

		// Compute current distance to \Gamma using the improved y.
		dx = xyz[0] - pOnInterfaceX;
		dy = xyz[1] - pOnInterfaceY;
		distances.push_back( sqrt( SQR( dx ) + SQR( dy ) ) );

		// Find parameter u that yields "a" minimum distance between point and sine-wave using Newton-Raphson's method.
		u = distThetaDerivative( stencil[s], xyz[0], xyz[1], sine, gen, normalDistribution, valOfDerivative, newDistance );
		v = sine.getA() * sin( sine.getOmega() * u );			// Recalculating point on interface (still in canonical coords).

		if( newDistance > distances[s] )
			throw std::runtime_error( "Failure with node " + std::to_string( stencil[s] ) + ": " + std::to_string( valOfDerivative ) );

		distances[s] = newDistance;					// Root finding was successful: keep minimum distance.

		if( sample[s] < 0 )							// Fix sign.
			distances[s] *= -1;

		if( s == 4 )								// For center node we need the parameter u to yield curvature.
			centerU = u;
	}

	sample[s] = H * sine.curvature( centerU );		// Last column holds h\kappa.
	distances.push_back( sample[s] );

	return sample;
}


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -0.5, MAX_D = -MIN_D;								// The canonical space is [-1/2, +1/2]^2.
	const double HALF_D = ( MAX_D - MIN_D ) / 2;							// Half domain.
	const int MAX_REFINEMENT_LEVEL = 7;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes per dimension.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Highest spatial resolution in x/y directions.
	const double MIN_A = 1.5 * H;				// An almost flat wave.
	const double MAX_A = 0.25;					// Tallest wave.
	const double MAX_HKAPPA_LB = 1.0 / 6.0;		// Lower and upper bounds for maximum h\kappa (used for discriminating
	const double MAX_HKAPPA_UB = 2.0 / 3.0;		// which samples to keep -- see below for details).

	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 1;	// Number of columns in resulting dataset.

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen; 					// Standard mersenne_twister_engine seeded with rd().
	std::uniform_real_distribution<double> uniformDistributionH_2( -H / 2, +H / 2 );
	std::normal_distribution<double> normalDistribution;

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To test we don't admit more than a single process.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		std::cout << "Testing sine wave level-set function..." << std::endl;

		// Domain information, applicable to all sinusoidal interfaces.
		int n_xyz[] = {1, 1, 1};							// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

		double maxRE = 0;									// Maximum relative error.

		const double T[] = {
			( MIN_D + MAX_D ) / 2 + uniformDistributionH_2( gen ),		// Translate center coords by a randomly chosen
			( MIN_D + MAX_D ) / 2 + uniformDistributionH_2( gen )		// perturbation from the grid's midpoint.
		};

		// p4est variables and data structures: these change with every sine wave because we must refine the
		// trees according to the new waves's origin and amplitude.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the level-set function to be reinitialized.
		const double HALF_AXIS_LEN = ( MAX_D - MIN_D ) * M_SQRT2 / 2 + 2 * H;		// Adding some padding of 2H.
		const double A = MAX_A;
		const double MIN_OMEGA = sqrt( MAX_HKAPPA_LB / ( H * A ) );
		const double MAX_OMEGA = sqrt( MAX_HKAPPA_UB / ( H * A ) );
		ArcLengthParameterizedSine sine( A, MAX_OMEGA, T[0], T[1], 0, H, HALF_AXIS_LEN );
		splitting_criteria_cf_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &sine );

		// Create the forest using a level-set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = ( void * ) ( &levelSetSC );

		// Refine and recursively partition forest.
		my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf, nullptr );
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
		nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
											// how much we'll neeed the node neighbors.

		// A ghosted parallel PETSc vector to store level-set function values.
		Vec phi;
		ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );

		// Calculate the level-set function values for each independent node (i.e. locally owned and ghost nodes).
		sample_cf_on_nodes( p4est, nodes, sine, phi );

		// Reinitialize level-set function using the fast sweeping method.
		FastSweeping fsm;
		fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
		fsm.reinitializeLevelSetFunction( &phi, 8 );
//		my_p4est_level_set_t ls( &nodeNeighbors );
//		ls.reinitialize_2nd_order( phi, 5 );

		// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are
		// the points we'll use to create our sample files.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );

		// An interface flag vector to distinguish nodes along the interface with full uniform neighborhoods.
		Vec interfaceFlag;
		ierr = VecDuplicate( phi, &interfaceFlag );
		CHKERRXX( ierr );

		double *interfaceFlagPtr;
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );
		for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
			interfaceFlagPtr[i] = 0;		// Init to zero and set flag of (valid) nodes along interface to 1.

		// A vector to store the dimensionless curvature at nodes along the interface.
		Vec hCurvature;
		ierr = VecDuplicate( interfaceFlag, &hCurvature );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, hCurvature );
		CHKERRXX( ierr );

		double *hCurvaturePtr;
		ierr = VecGetArray( hCurvature, &hCurvaturePtr );
		CHKERRXX( ierr );

		// A vector to store the error of full neighborhoods along the interface.
		Vec vError;
		ierr = VecDuplicate( interfaceFlag, &vError );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, vError );
		CHKERRXX( ierr );

		double *vErrorPtr;
		ierr = VecGetArray( vError, &vErrorPtr );
		CHKERRXX( ierr );

		// Getting the full uniform stencils of interface points.
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Now, collect samples with reinitialized level-set function values and target h\kappa.
		// Avoid nodes that are close to physical domain boundary as they are less accurate.
		for( auto n : indices )
		{
			double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
			node_xyz_fr_n( n, p4est, nodes, xyz );
			if( ABS( xyz[0] - MIN_D ) <= 4 * H || ABS( xyz[0] - MAX_D ) <= 4 * H ||
				ABS( xyz[1] - MIN_D ) <= 4 * H || ABS( xyz[1] - MAX_D ) <= 4 * H )
				continue;

			std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.
			try
			{
				if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
				{
					std::vector<double> distances;
					std::vector<double> sample = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil, p4est,
						nodes, &nodeNeighbors, phiReadPtr, sine, gen, normalDistribution, distances );
					hCurvaturePtr[n] = sample[NUM_COLUMNS-1];
					interfaceFlagPtr[n] = 1;

					// Error metric for validation.
					for( int i = 0; i < NUM_COLUMNS - 1; i++ )
					{
						vErrorPtr[stencil[i]] = ( distances[i] - sample[i] ) / H;
						maxRE = MAX( maxRE, ABS( vErrorPtr[stencil[i]] ) );
					}

					std::cout << n << ", " << xyz[0] << ", " << xyz[1] << ", " << sample[NUM_COLUMNS-1] << ";" << std::endl;
				}
			}
			catch( std::exception &e )
			{
					std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
			}
		}

		// The error.
		std::cout << "Maximum relative error: " << maxRE << std::endl;

		std::ostringstream oss;
		oss << "sine_wave_" << mpi.size() << "_" << P4EST_DIM;
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								4, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr,
								VTK_POINT_DATA, "hKappa", hCurvaturePtr,
								VTK_POINT_DATA, "error", vErrorPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		ierr = VecRestoreArray( vError, &vErrorPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( hCurvature, &hCurvaturePtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( vError );
		CHKERRXX( ierr );

		ierr = VecDestroy( hCurvature );
		CHKERRXX( ierr );

		ierr = VecDestroy( interfaceFlag );
		CHKERRXX( ierr );

		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}