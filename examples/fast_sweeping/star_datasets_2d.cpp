/**
 * Generate star-shaped interface data sets for evaluating our feedforward neural network, which was trained on
 * spherical interfaces using samples from a reinitialized level-set function with the fast sweeping method.
 *
 * Developer: Luis Ángel.
 * Date: May 20, 2020.
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

#include "DistThetaRootFinding.h"
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>

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
 * star interface.  We assume that this query node is effectively adjacent to \Gamma.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] H Spacing (smallest quad/oct side-length).
 * @param [in] stencil The full uniform stencil of indices centered at the query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] star The level-set function with a star-shaped interface.
 * @param [in/out] pointsFile File object where to write coordinates of nodes adjacent to \Gamma.
 * @param [in/out] anglesFile File object where to write angles of normal projected points on \Gamma.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const double* phiReadPtr, const geom::Star& star, std::ofstream& pointsFile, std::ofstream& anglesFile )
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
	double xyz[P4EST_DIM];								// Position of node at the center of the stencil.
	node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
	double grad[] = {									// Gradient at center point.
		( rightPhi - leftPhi ) / ( 2 * H ),				// Using central differences.
		( topPhi - bottomPhi ) / ( 2 * H )
	};
	double gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );
	double pOnInterfaceX = xyz[0] - grad[0] / gradNorm * centerPhi,		// Coordinates of projection of grid point on interface.
		   pOnInterfaceY = xyz[1] - grad[1] / gradNorm * centerPhi;

	double thetaOnInterface = atan2( pOnInterfaceY, pOnInterfaceX );	// Initial guess for root finding method.
	thetaOnInterface = ( thetaOnInterface < 0 )? thetaOnInterface + 2 * M_PI : thetaOnInterface;
	double valOfDerivative;
	double newThetaOnInterface = distThetaDerivative( xyz[0], xyz[1], star, valOfDerivative, thetaOnInterface, thetaOnInterface - H / 2, thetaOnInterface + H / 2 );
	double r = star.r( newThetaOnInterface );							// Recalculating closest point on interface.
	pOnInterfaceX = r * cos( newThetaOnInterface );
	pOnInterfaceY = r * sin( newThetaOnInterface );

	double dx = xyz[0] - pOnInterfaceX,						// Verify that point on interface is not far from corresponding grid point.
		   dy = xyz[1] - pOnInterfaceY;
	if( dx * dx + dy * dy <= H * H && ABS( valOfDerivative ) <= 1e-8 )
		thetaOnInterface = newThetaOnInterface;				// Theta is OK.
	else
		std::cerr << "Minimization placed point on interface too far.  Reverting back to point on interface calculated with phi values" << std::endl;
//	std::cout << nodeIdx << "; " << valOfDerivative
//			  << "; plot([" << xyz[0] << "], [" << xyz[1] << "], 'mo', ["
//			  << pOnInterfaceX << "], [" << pOnInterfaceY << "], 'ko');" << std::endl;

	sample[s] = H * star.curvature( thetaOnInterface );		// Last column holds h\kappa.

	// Write center sample point coordinates.
	pointsFile << xyz[0] << "," << xyz[1] << std::endl;

	// Write angle parameter for projected point on interface.
	anglesFile << ( thetaOnInterface < 0 ? 2 * M_PI + thetaOnInterface : thetaOnInterface ) << std::endl;

	return sample;
}

int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -0.5, MAX_D = +0.5;								// The canonical space is [0,1]^{P4EST_DIM}.
	const double HALF_D = ( MAX_D - MIN_D ) / 2;							// Half domain.
	const int MAX_REFINEMENT_LEVEL = 7;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes per dimension.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Highest spatial resolution in x/y directions.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/fsm/data/star_" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";	// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 1;					// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];									// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// Check again.  To generate datasets we don't admit more than a single process to avoid race conditions when
		// writing datasets to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		// Setting up the star-shaped interface.
		// Using a=0.075 and b=0.35 for smooth star, and a=0.12 and b=0.305 for sharp star.
		geom::Star star( 0.075, 0.35, 5 );
		const double STAR_SIDE_LENGTH = star.getInscribingSquareSideLength();
		if( STAR_SIDE_LENGTH > MAX_D - MIN_D - 4 * H )						// Consider also 2H padding on each side.
			throw std::runtime_error( "Star exceed allowed space in domain!" );

		//////////////////////////////////////////// Prepare output files //////////////////////////////////////////////

		// Prepare file where to write sample level-set function values.
		std::ofstream outputFile;
		std::string fileName = DATA_PATH + "phi.csv";
		outputFile.open( fileName, std::ofstream::trunc );
		if( !outputFile.is_open() )
			throw std::runtime_error( "Phi values output file " + fileName + " couldn't be opened!" );

		std::ostringstream headerStream;									// Write output file header.
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		outputFile << headerStream.str() << std::endl;

		outputFile.precision( 15 );											// Precision for floating point numbers.

		// Prepare file where to write sampled nodes' cartesian coordinates.
		std::ofstream pointsFile;
		std::string pointsFileName = DATA_PATH + "points.csv";
		pointsFile.open( pointsFileName, std::ofstream::trunc );
		if( !pointsFile.is_open() )
			throw std::runtime_error( "Point cartesian coordinates file " + pointsFileName + " couldn't be opened!" );

		pointsFile.precision( 15 );
		pointsFile << R"("x","y")" << std::endl;

		// Prepare file where to write the params.
		std::ofstream paramsFile;
		std::string paramsFileName = DATA_PATH + "params.csv";
		paramsFile.open( paramsFileName, std::ofstream::trunc );
		if( !paramsFile.is_open() )
			throw std::runtime_error( "Params file " + paramsFileName + " couldn't be opened!" );

		std::ostringstream headerParamsStream;
		headerParamsStream << R"("minD","maxD","sideLength","a","b","p")";
		paramsFile << headerParamsStream.str() << std::endl;

		paramsFile.precision( 15 );
		paramsFile << MIN_D << "," << MAX_D << "," << STAR_SIDE_LENGTH << "," << star.getA() << "," << star.getB() << "," << star.getP() << std::endl;

		// Prepare file where to write the angle parameter for corresponding points on interface.
		std::ofstream anglesFile;
		std::string anglesFileName = DATA_PATH + "angles.csv";
		anglesFile.open( anglesFileName, std::ofstream::trunc );
		if( !anglesFile.is_open() )
			throw std::runtime_error( "Angles file " + anglesFileName + " couldn't be opened!" );

		anglesFile.precision( 15 );
		anglesFile << R"("theta")" << std::endl;							// Write header.

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		parStopWatch watch;
		printf( ">> Began to generate datasets for a star-shaped interface with maximum refinement level of %i and H = %f\n",
			MAX_REFINEMENT_LEVEL, H );
		watch.start();

		// Domain information.
		int n_xyz[] = {1, 1, 1};									// One tree per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};					// Square domain.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};									// Non-periodic.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the non-signed distance level-set function to be reinitialized.
		splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &star, 2.0 );

		// Create the forest using a level set as refinement criterion.
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
		sample_cf_on_nodes( p4est, nodes, star, phi );

		// Reinitialize level-set function (using the fast sweeping method).
		FastSweeping fsm;
		fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
		fsm.reinitializeLevelSetFunction( &phi, 8 );
//		my_p4est_level_set_t ls( &nodeNeighbors );
//		ls.reinitialize_2nd_order( phi, 5 );

		// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are the
		// points we'll use to create our sample files.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		// Getting the full uniform stencils of interface points.
		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Also, collect indicators for those nodes that are adjacent to the interface.
		Vec interfaceFlag;
		ierr = VecDuplicate( phi, &interfaceFlag );
		CHKERRXX( ierr );

		double *interfaceFlagPtr;
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );
		for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
			interfaceFlagPtr[i] = 0;				// Init to zero and set flag of nodes along interface equal to 1.

		// Now, collect samples with level-set function values and target h\kappa.
		int nSamples = 0;
		std::vector<std::vector<double>> samples;
		for( auto n : indices )
		{
			std::vector<p4est_locidx_t> stencil;	// Contains 9 nodal indices in 2D.
			try
			{
				if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
				{
					std::vector<double> sample = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil, p4est,
						nodes, phiReadPtr, star, pointsFile, anglesFile );
					samples.push_back( sample );
					interfaceFlagPtr[n] = 1;		// Set the node-along-interface indicator.
					nSamples++;
				}
			}
			catch( std::exception &e )
			{
				std::cerr << e.what() << std::endl;
			}
		}

		// Write all samples collected for all circles with the same radius but randomized center content to file.
		for( const auto& row : samples )
		{
			std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
			outputFile << row.back() << std::endl;
		}

		nSamples += samples.size();

		printf( "<< Finished generating %i samples in %f secs.\n", nSamples, watch.get_duration_current() );
		watch.stop();

		// Write paraview file to visualize the star interface and nodes following along.
		std::ostringstream oss;
		oss << "star_" << mpi.size() << "_" << P4EST_DIM;
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								2, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		// Cleaning up.
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( interfaceFlag );
		CHKERRXX( ierr );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );

		// Closing file objects.
		outputFile.close();
		pointsFile.close();
		anglesFile.close();
		paramsFile.close();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}