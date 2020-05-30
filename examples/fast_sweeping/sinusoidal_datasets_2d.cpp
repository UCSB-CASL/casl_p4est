/**
 * Generate datasets for training a feedforward neural network on sinusoidal interfaces using samples from a
 * reinitialized level-set function with the fast sweeping method.
 * The level-set function is implemented as an arc-length parameterized sine wave function that is transformed with an
 * affine transformation to allow for pattern variations.  See arclength_parameterized_sin_2d.h for more details.
 *
 * Developer: Luis Ángel.
 * Date: May 28, 2020.
 */

// System.
#include <stdexcept>
#include <iostream>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_fast_sweeping.h>
#include <src/my_p8est_nodes_along_interface.h>
//#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_fast_sweeping.h>
#include <src/my_p4est_nodes_along_interface.h>
//#include <src/my_p4est_level_set.h>
#endif

#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iterator>
#include <fstream>
#include "arclength_parameterized_sine_2d.h"

/**
 * Generate the column headers following the truth-table order with x changing slowly, then y changing faster than x,
 * and finally z changing faster than y.  Each dimension has three states: m, 0, and p (minus, center, plus).  For
 * example, in 2D, the columns that are generated are:
 * 	   Acronym      Meaning
 *		"mm"  =>  (i-1, j-1)
 *		"m0"  =>  (i-1, j  )
 *		"mp"  =>  (i-1, j+1)
 *		"0m"  =>  (  i, j-1)
 *		"00"  =>  (  i,   j)
 *		"0p"  =>  (  i, j+1)
 *		"pm"  =>  (i+1, j-1)
 *		"p0"  =>  (i+1,   j)
 *		"pp"  =>  (i+1, j+1)
 *		"hk"  =>  h * \kappa
 * @param [out] header Array of column headers to be filled up.  Must be backed by a correctly allocated array.
 */
void generateColumnHeaders( std::string header[] )
{
	const int STEPS = 3;
	std::string states[] = {"m", "0", "p"};			// States for x, y, and z directions.
	int i = 0;
	for( int x = 0; x < STEPS; x++ )
		for( int y = 0; y < STEPS; y++ )
#ifdef P4_TO_P8
			for( int z = 0; z < STEPS; z++ )
#endif
		{
			i = SUMD( x * (int)pow( STEPS, P4EST_DIM - 1 ), y * (int)pow( STEPS, P4EST_DIM - 2 ), z );
			header[i] = SUMD( states[x], states[y], states[z] );
		}
	header[i+1] = "hk";								// Don't forget the h*\kappa column!
}


/**
 * Generate the sample row of level-set function values and target h\kappa for a node that has been found next to the
 * sine wave interface.  We assume that this query node is effectively adjacent to \Gamma.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] sine The level-set function with a sinusoidal interface.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const double *phiReadPtr, const ArcLengthParameterizedSine& sine )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// (Reinitialized) level-set function values and target h\kappa.

	int s;												// Index to fill in the sample vector.
	double leftPhi, rightPhi, topPhi, bottomPhi;		// To compute grad(\phi_{i,j}).
	double centerPhi;									// \phi{i,j}.
	for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
	{
		sample[s] = phiReadPtr[stencil[s]];

		switch( s )
		{
			case 5: topPhi = sample[s]; break;			// \phi_{i,j+1}.
			case 1: leftPhi = sample[s]; break;			// \phi_{i-1,j}.
			case 4: centerPhi = sample[s]; break;		// Point's phi value.
			case 7: rightPhi = sample[s]; break;		// \phi_{i+1,j}.
			case 3: bottomPhi = sample[s]; break;		// \phi_{i,j-1}.
			default: ;
		}
	}

	// Computing the target curvature.
	double xyz[P4EST_DIM];								// Position of node at the center of the stencil in world coords.
	node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
	double grad[] = {									// Gradient at center point.
		( rightPhi - leftPhi ) / ( 2 * H ),				// Using central differences.
		( topPhi - bottomPhi ) / ( 2 * H )
	};
	double gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );
	double pOnInterfaceX = xyz[0] - grad[0] / gradNorm * centerPhi,		// Coordinates of projection of grid point on
		   pOnInterfaceY = xyz[1] - grad[1] / gradNorm * centerPhi;		// interface in world coords.

   // Transform node position and approximated point on the interface to sine wave canonical coordinate system to
   // simplify computations with minimization process.
   sine.toCanonicalCoordinates( xyz[0], xyz[1] );
   sine.toCanonicalCoordinates( pOnInterfaceX, pOnInterfaceY );

	double u = pOnInterfaceX;							// Initial parameter guess for root finding method.
	double valOfDerivative;
	double newU = distThetaDerivative( nodeIdx, xyz[0], xyz[1], sine, valOfDerivative, u, u - H, u + H );

	pOnInterfaceX = newU;								// Recalculating point on interface (still in canonical coords).
	pOnInterfaceY = sine.getA() * sin( sine.getOmega() * newU );

	double dx = xyz[0] - pOnInterfaceX,					// Verify that point on interface is not far from grid point.
		   dy = xyz[1] - pOnInterfaceY;
	if( dx * dx + dy * dy <= H * H && ABS( valOfDerivative ) <= 1e-8 )
		u = newU;										// Parameter is OK.
	else
	{
		sine.toWorldCoordinates( xyz[0], xyz[1] );
		sine.toWorldCoordinates( pOnInterfaceX, pOnInterfaceY );
		std::cerr << "Node " << nodeIdx << ": Minimization placed node on interface too far.  Reverting back to point "
				  << "on interface calculated with phi values.  \n     "
				  << valOfDerivative << "; plot([" << xyz[0] << "], [" << xyz[1] << "], 'b.', ["
				  << pOnInterfaceX << "], [" << pOnInterfaceY << "], 'ko');" << std::endl;
	}

	sample[s] = H * sine.curvature( u );				// Last column holds h\kappa.
	std::cout << u << ", " << sample[s] << ";" << std::endl;

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
	const int NUM_AMPLITUDES = (int)pow( 2, MAX_REFINEMENT_LEVEL ) - 5;		// Number different sine wave amplitudes.
																			// and ensures at least 2 circles per finest quad/oct.
	const double MIN_A = 1.5 * H;				// An almost flat wave.
	const double MAX_A = HALF_D - 2 * H;		// Very tall wave.
	const double MIN_OMEGA = M_PI / HALF_D;		// Minimum frequency to ensure that at least 2\pi lies in the domain.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/fsm/data/";			// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 1;	// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];		// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( rd() ); 					// Standard mersenne_twister_engine seeded with rd().
	std::uniform_real_distribution<double> uniformDistribution( -H / 2, +H / 2 );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing datasets
		// to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		std::cout << "Collecting samples from a sinusoidal wave function..." << std::endl;

		//////////////////////////////////////////// Generating the datasets ///////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for %i distinct amplitudes with maximum refinement level of %i and finest h = %f\n",
			NUM_AMPLITUDES, MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Prepare samples file: rls_X.csv for reinitialized level-set, sdf_X.csv for signed-distance function values.
		std::ofstream outputFile;
		std::string fileName = DATA_PATH + "sine_" + std::to_string( MAX_REFINEMENT_LEVEL ) +  ".csv";
		outputFile.open( fileName, std::ofstream::trunc );
		if( !outputFile.is_open() )
			throw std::runtime_error( "Output file " + fileName + " couldn't be opened!" );

		// Write column headers: enforcing strings by adding quotes around them.
		std::ostringstream headerStream;
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		outputFile << headerStream.str() << std::endl;

		outputFile.precision( 15 );							// Precision for floating point numbers.

		// Variables to control the spread of sine waves' amplitudes, which must vary uniformly from MIN_A to MAX_A.
		double A_DIST = MAX_A - MIN_A;						// Amplitudes are in [1.5H, 0.5-2H], inclusive.
		double linspace[NUM_AMPLITUDES];
		for( int i = 0; i < NUM_AMPLITUDES; i++ )			// Uniform linear space from 0 to 1, with NUM_AMPLITUDES steps.
			linspace[i] = (double)( i ) / ( NUM_AMPLITUDES - 1.0 );

		// Domain information, applicable to all sinusoidal interfaces.
		int n_xyz[] = {1, 1, 1};							// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

//		int nSamples = 0;
		int nc = 0;								// Keeps track of number of circles whose samples has been collected.
//		while( nc < NUM_CIRCLES )
//		{
			const double A = MIN_A + linspace[nc] * A_DIST;		// Amplitude to be evaluated.
//			std::vector<std::vector<double>> samples;
			double maxRE = 0;									// Maximum relative error.

			const double T[] = {
				( MIN_D + MAX_D ) / 2 + uniformDistribution( gen ),		// Translate center coords by a randomly chosen
				( MIN_D + MAX_D ) / 2 + uniformDistribution( gen )		// perturbation from the grid's midpoint.
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
			ArcLengthParameterizedSine sine( MIN_A, 2 / (3 * H), T[0], T[1], M_PI_4, H, HALF_AXIS_LEN );
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
			double *phiPtr;
			ierr = VecGetArray( phi, &phiPtr );
			CHKERRXX( ierr );
			for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( i, p4est, nodes, xyz );	// Use arclength parameterization to approximate level-set sine
				phiPtr[i] = sine( xyz[0], xyz[1] );		// function.
			}
			ierr = VecRestoreArray( phi, &phiPtr );
			CHKERRXX( ierr );

			// Reinitialize level-set function using the fast sweeping method.
			FastSweeping fsm;
			fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
			fsm.reinitializeLevelSetFunction( &phi, 8 );
//			my_p4est_level_set_t ls( &nodeNeighbors );
//			ls.reinitialize_2nd_order( phi, 5 );

			// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are
			// the points we'll use to create our sample files.
			NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );

			Vec interfaceFlag;
			ierr = VecDuplicate( phi, &interfaceFlag );
			CHKERRXX( ierr );

			double *interfaceFlagPtr;
			ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr );
			CHKERRXX( ierr );
			for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
				interfaceFlagPtr[i] = 0;		// Init to zero and set flag of (valid) nodes along interface to 1.

			std::vector<p4est_locidx_t> indices;
			nodesAlongInterface.getIndices( &phi, indices );

			// Getting the full uniform stencils of interface points.
			const double *phiReadPtr;
			ierr = VecGetArrayRead( phi, &phiReadPtr );
			CHKERRXX( ierr );

			// Now, collect samples with reinitialized level-set function values and target h\kappa.
			int nSamples = 0;
			std::vector<std::vector<double>> samples;
			for( auto n : indices )
			{
				std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.
				try
				{
					if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
					{
						std::vector<double> sample = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil, p4est,
																					nodes, phiReadPtr, sine );
						samples.push_back( sample );
						nSamples++;
						interfaceFlagPtr[n] = 1;
					}
				}
				catch( std::exception &e )
				{
//					double xyz[P4EST_DIM];				// Position of node at the center of the stencil.
//					node_xyz_fr_n( n, p4est, nodes, xyz );
//					std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
				}
			}

			std::ostringstream oss;
			oss << "sine_" << mpi.size() << "_" << P4EST_DIM;
			my_p4est_vtk_write_all( p4est, nodes, ghost,
									P4EST_TRUE, P4EST_TRUE,
									2, 0, oss.str().c_str(),
									VTK_POINT_DATA, "phi", phiReadPtr,
									VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr );
			my_p4est_vtk_write_ghost_layer( p4est, ghost );

/*			for( auto n : indices )
			{
				std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D, 27 values in 3D.
				std::vector<double> dataPve;			// Phi and h*kappa results in positive and negative form.
				std::vector<double> dataNve;
				dataPve.reserve( NUM_COLUMNS );			// Efficientize containers.
				dataNve.reserve( NUM_COLUMNS );
				try
				{
					if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
					{
						for( auto s : stencil )
						{
							dataPve.push_back( +phiReadPtr[s] );		// Positive curvature phi values.
							dataNve.push_back( -phiReadPtr[s] );		// Negative curvature phi values.

							double xyz[P4EST_DIM];						// Error checking.
							node_xyz_fr_n( s, p4est, nodes, xyz );
							double error = ABS( sphere( DIM( xyz[0], xyz[1], xyz[2] ) ) - phiReadPtr[s] ) / H;
							maxRE = MAX( maxRE, error );
						}

						// Appending the target h*\kappa.
						dataPve.push_back( +H_KAPPA );
						dataNve.push_back( -H_KAPPA );

						// Accumulating samples.
//							samples.push_back( dataPve );
//							samples.push_back( dataNve );
					}
				}
				catch( std::exception &e )
				{
					std::cerr << e.what() << std::endl;
				}
			}
*/
			ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr );
			CHKERRXX( ierr );

			ierr = VecRestoreArrayRead( phi, &phiReadPtr );
			CHKERRXX( ierr );

			// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
			ierr = VecDestroy( interfaceFlag );
			CHKERRXX( ierr );

			ierr = VecDestroy( phi );
			CHKERRXX( ierr );

			// Destroy the p4est and its connectivity structure.
			p4est_nodes_destroy( nodes );
			p4est_ghost_destroy( ghost );
			p4est_destroy( p4est );
			p4est_connectivity_destroy( connectivity );
/*
			// Write all samples collected for all circles with the same radius but randomized center content to file.
			for( const auto& row : samples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
				outputFile << row.back() << std::endl;
			}

			nc++;
			nSamples += samples.size();

			std::cout << "     (" << nc << ") Done with radius = " << A
					  << ".  Maximum relative error = " << maxRE
					  << ".  Samples = " << samples.size() << ";" << std::endl;

			if( nc % 10 == 0 )
				printf( "   [%i radii evaluated after %f secs.]\n", nc, watch.get_duration_current() );
//		}

		printf( "<< Finished generating %i circles and %i samples in %f secs.\n", nc, nSamples, watch.get_duration_current() );
		watch.stop();*/
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}