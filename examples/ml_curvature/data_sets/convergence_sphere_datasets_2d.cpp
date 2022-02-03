/**
 * Generating circular samples for a convergence test, similar to the test found in section 3.1 in
 * "A hybrid particle volume-of-fluid method for curvature estimation in multiphase flows", by P. Karnakov, S. Litvinov,
 * and P. Koumoutsakos.
 *
 * @note Comparison with the above paper suggested in first revision of the paper.
 *
 * Only reinitialized samples are collected for radii in [1.5h, 16h], for h=1/128.
 *
 * Developer: Luis Ángel.
 * Created: February 2, 2022.
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
#include <src/my_p4est_nodes_along_interface.h>
#include <src/my_p4est_level_set.h>
#include <src/casl_geometry.h>
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include "local_utils.h"


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////
	const int MAX_REFINEMENT_LEVEL = 7;						// Always take H from unit squares with this maximum
	const double H = 1. / pow( 2, MAX_REFINEMENT_LEVEL );	// level of refinement.
	const int NUM_REINIT_ITERS = 10;						// Number of iterations for PDE reintialization.
	const std::vector<double> RADII = {1.5, 2, 4, 6, 8, 10, 12, 14, 16};	// As multiples of H.
	const double DIM = ceil( (RADII.back() + 4) * H );		// Symmetric units around origin: [-DIM, +DIM]^{P4EST_DIM}.
	const int NUM_CENTERS_PER_RADIUS = 100;

	// Destination folder.
	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde-1120/data/" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/convergence_circles/";
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 2;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];					// Column headers following the x-y truth table of
	generateColumnHeaders( COLUMN_NAMES );					// 3-state variables.

	// Random-number generator for centers.
	std::mt19937 gen{}; 			// NOLINT.
	std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		parStopWatch watch;
		printf( ">> Began to generate convergence circle datasets for maximum level of refinement = %i and finest h = %g\n", MAX_REFINEMENT_LEVEL, H );
		watch.start();

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		// Domain information, applicable to all spherical interfaces.
		int n_xyz[] = { 2 * (int)DIM, 2 * (int)DIM, 2 * (int)DIM };		// Symmetric num. of trees in +ve and -ve axes.
		double xyz_min[] = { -DIM, -DIM, -DIM };			// Squared domain.
		double xyz_max[] = { DIM, DIM, DIM };
		int periodic[] = { 0, 0, 0 };						// Non-periodic domain.

		int nSamples = 0;
		int nc = 0;							// Keeps track of number of circles whose samples have been collected.
		while( nc < RADII.size() )
		{
			const double RADIUS = RADII[nc] * H;
			const double KAPPA = 1. / RADIUS;
			const double H_KAPPA = H * KAPPA;				// Expected dimensionless curvature: h*kappa = h/r.
			std::vector<std::vector<double>> rlsSamples;

			// Preparing file where to write samples: its name is of the form sphere_rls_x.y.csv, where x.y is the radius as a multiple of H.
			char rStr[100];
			sprintf( rStr, "%.1f", RADII[nc] );
			std::ofstream rlsFile;
			std::string rlsFileName = DATA_PATH + "sphere_rls_" + std::string( rStr ) +  ".csv";
			rlsFile.open( rlsFileName, std::ofstream::trunc );
			if( !rlsFile.is_open() )
				throw std::runtime_error( "Output file " + rlsFileName + " couldn't be opened!" );

			// Write column headers: enforcing strings by adding quotes around them.
			std::ostringstream headerStream;
			for( int i = 0; i < NUM_COLUMNS - 1; i++ )
				headerStream << "\"" << COLUMN_NAMES[i] << "\",";
			headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
			rlsFile << headerStream.str() << std::endl;
			rlsFile.precision( 15 );

			for( int r = 0; r < NUM_CENTERS_PER_RADIUS; r++ )
			{
				const double C[] = {
					DIM( uniformDistribution( gen ),		// Center coords are randomly chosen around the origin.
						 uniformDistribution( gen ),
						 uniformDistribution( gen ) )
				};

				// p4est variables and data structures: these change with every single circle because we must refine the
				// trees according to the new circle's center and radius.
				p4est_t *p4est;
				p4est_nodes_t *nodes;
				my_p4est_brick_t brick;
				p4est_ghost_t *ghost;
				p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

				// Defining the level-sets.
				geom::SphereNSD sphereNsd( DIM( C[0], C[1], C[2] ), RADIUS );
				geom::Sphere sphere( DIM( C[0], C[1], C[2] ), RADIUS );		// This one is the exact signed distance used only to refine the grid.
				splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &sphere, 6.0 );

				// Create the forest using a level set as refinement criterion.
				p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
				p4est->user_pointer = (void *)( &levelSetSC );

				// Refine and recursively partition forest.
				my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf_and_uniform_band, nullptr );
				my_p4est_partition( p4est, P4EST_TRUE, nullptr );

				// Create the ghost (cell) and node structures.
				ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
				nodes = my_p4est_nodes_new( p4est, ghost );

				// Initialize the neighbor nodes structure.
				my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
				my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
				nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
													// how much we'll neeed the node neighbors.

				// Validation.
				double dxyz[P4EST_DIM]; 			// Dimensions of the smallest quadrants.
				double dxyz_min;        			// Minimum side length of the smallest quadrants.
				double diag_min;        			// Diagonal length of the smallest quadrants.
				get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );
				assert( ABS( dxyz_min - H ) <= EPS );

				// Ghosted parallel PETSc vectors to store level-set function values.
				Vec rlsPhi;
				ierr = VecCreateGhostNodes( p4est, nodes, &rlsPhi );
				CHKERRXX( ierr );

				Vec curvature, normal[P4EST_DIM];
				ierr = VecDuplicate( rlsPhi, &curvature );
				CHKERRXX( ierr );
				for( auto& dim : normal )
				{
					VecCreateGhostNodes( p4est, nodes, &dim );
					CHKERRXX( ierr );
				}

				// Calculate the level-set function values for all independent nodes.
				sample_cf_on_nodes( p4est, nodes, sphereNsd, rlsPhi );

				// Reinitialize level-set function using PDE equation.
				my_p4est_level_set_t ls( &nodeNeighbors );
				ls.reinitialize_2nd_order( rlsPhi, NUM_REINIT_ITERS );

				// Compute curvature with reinitialized data, which will be interpolated at the interface.
				compute_normals( nodeNeighbors, rlsPhi, normal );
				compute_mean_curvature( nodeNeighbors, rlsPhi, normal, curvature );

				// Prepare interpolation.
				my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
				interpolation.set_input( curvature, linear );

				// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these
				// are the points we'll use to create our sample files and compare with the signed distance function.
				NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );
				std::vector<p4est_locidx_t> indices;
				nodesAlongInterface.getIndices( &rlsPhi, indices );

				// Getting the full uniform stencils of interface points.
				const double *rlsPhiReadPtr;
				ierr = VecGetArrayRead( rlsPhi, &rlsPhiReadPtr );
				CHKERRXX( ierr );

				for( auto n : indices )
				{
					std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D, 27 values in 3D.
					std::vector<double> rlsDataNve;			// Phi and h*kappa results in negative form only.
					rlsDataNve.reserve( NUM_COLUMNS );		// Efficientize containers.
					try
					{
						if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
						{
							for( auto s : stencil )
								rlsDataNve.push_back( -rlsPhiReadPtr[s] );

							rlsDataNve.push_back( -H_KAPPA );		// Appending the target h*kappa.

							// Appending the interpolated h*kappa.
							double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
							node_xyz_fr_n( n, p4est, nodes, xyz );
							double grad[P4EST_DIM];					// Getting its gradient (i.e. normal).
							const quad_neighbor_nodes_of_node_t *qnnnPtr;
							nodeNeighbors.get_neighbors( n, qnnnPtr );
							qnnnPtr->gradient( rlsPhiReadPtr, grad );
							double gradNorm = sqrt( SUMD( SQR( grad[0] ), SQR( grad[1] ), SQR( grad[2] ) ) );	// Get the unit gradient.

							for( int i = 0; i < P4EST_DIM; i++ )	// Translation: this is where we need to interpolate
								xyz[i] -= grad[i] / gradNorm * rlsPhiReadPtr[n];	// the numerical curvature.

							double iHKappa = H * interpolation( DIM( xyz[0], xyz[1], xyz[2] ) );
							rlsDataNve.push_back( -iHKappa );		// Attach interpolated h*kappa.

							rlsSamples.push_back( rlsDataNve );		// Accumulating samples.
						}
					}
					catch( std::exception &e )
					{
						std::cerr << e.what() << std::endl;
					}
				}

				// Cleaning up.
				ierr = VecRestoreArrayRead( rlsPhi, &rlsPhiReadPtr );
				CHKERRXX( ierr );

				ierr = VecDestroy( rlsPhi );
				CHKERRXX( ierr );

				ierr = VecDestroy( curvature );
				CHKERRXX( ierr );

				for( auto& dim : normal )
				{
					ierr = VecDestroy( dim );
					CHKERRXX( ierr );
				}

				// Destroy the p4est and its connectivity structure.
				p4est_nodes_destroy( nodes );
				p4est_ghost_destroy( ghost );
				p4est_destroy( p4est );
				p4est_connectivity_destroy( connectivity );
			}

			// Write all samples collected for all circles with the same radius but randomized center content to file.
			for( const auto& row : rlsSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( rlsFile, "," ) );	// Inner elements.
				rlsFile << row.back() << std::endl;
			}

			nc++;

			std::cout << "     (" << nc << ") Done with radius = " << RADIUS
					  << ".  Samples = " << rlsSamples.size() << ";  time = " << watch.get_duration_current() << std::endl;

			rlsFile.close();
		}

		printf( "<< Finished generating %i circles and %i samples in %f secs.\n", nc, nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}