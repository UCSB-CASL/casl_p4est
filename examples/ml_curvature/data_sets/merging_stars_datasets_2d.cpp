/**
 * Generate merging star-shaped interfaces data sets for evaluating our feedforward neural network, which was trained on
 * circular, sinusoidal, and merging circular interfaces using samples from reinitialized level-set functions.
 * Collect samples from PDE reinitialization using 5, 10, and 20 iterations for accuracy comparison, and from FSM.
 * For each reinitialization scheme, generate files with reinitialized level-set function values, exact distances, point
 * coordinates and the gradient error (i.e. |1 - |grad_phi(x)||).
 * The exact distances are obtained by first using bisection for bracketing, and then Newton-Raphson's method to refine
 * the zero of the derivative of the minimum distance function betwen a point (i.e. node along Gamma) and the compound
 * star-shaped interface.
 *
 * We have a couple of merging stars examples that are used for comparing the neural network accuracy against the
 * numerical method.
 * - Example 1 (merging_stars_example_1.h): Two 5-armed stars with distinct origins merge at the tip of two of their
 *   arms.  See repository ML_Curvature_3.0 and its Jupyther notebook MergingStarsExample1.ipynb.
 *
 * The output files generated are placed in the data/star_L/merging/example_E, where L is the maximum level of
 * refinement, and E is the example number.  At least the data/star_L directory must exist.
 * The rest of subdirectories are created if they don't exist.  Files are overwritten ([R] = fsm|iter5|iter10|iter20):
 * a) [R]_points.csv Cartesian coordinates of center nodes sampled along the compound interface and their discontinuous
 *    flag (0 smooth, 1 discontinuous).
 * b) [R]_phi.csv Reinitialized level-set function values of the 9-point stencils along the compound interface.
 * c) [R]_phi_sdf.csv Exact distances from the 9-point stencils along the compound interface to Gamma.
 * e) [R]_grad_errors.csv The gradient error with respect to unity.
 * f) params.csv Contains the parameters to recreate each individual star; params are columns, stars are rows.
 *
 * Developer: Luis Ángel.
 * Date: August 21, 2020.
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
#include <src/my_p4est_level_set.h>

#include "merging_stars_example_1.h"
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include "local_utils.h"


/**
 * Generate the sample row of level-set function values and target h*kappa for a node that has been found next to the
 * compound interface.  We assume that this query node is effectively adjacent to Gamma since all of its 9-point stencil
 * nodes are evaluated agains the merging stars interface.
 * @param [in] nodeIdx Query node adjancent or on the interface.
 * @param [in] NUM_COLUMNS Number of columns in output file.
 * @param [in] stencil The full uniform stencil of node indices centered at the standing query node.
 * @param [in] p4est Pointer to p4est data structure.
 * @param [in] nodes Pointer to nodes data structure.
 * @param [in] neighbors Pointer to neighbors data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] mergingStars The level-set function with the compound star-shaped interface.
 * @param [in,out] whoMap Memoization hash map to identidy to whom a node owes its shortest distance and h*kappa.
 * @param [in,out] distanceMap Memoization hash map to store exact signed distances.
 * @param [in,out] hkMap Memoization hash map to store dimensionless curvatures.
 * @param [in/out] pointsFile Reference to file object where to write coordinates of nodes adjacent to Gamma.
 * @param [out] distances A vector of "true" distances from all of 9 stencil points to the compound interface.
 * @param [out] xOnGamma x-coordinate of normal projection of grid node onto compound interface.
 * @param [out] yOnGamma y-coordinate of normal projection of grid node onto compound interface.
 * @param [out] troubling Whether the stencil is near to a discontinuous region.
 * @return Vector of reinitialized level-set function values for the stencil centered at the nodeIdx node.
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const my_p4est_node_neighbors_t *neighbors, const double *phiReadPtr, const MergingStarsExample1& mergingStarsExample1,
	std::unordered_map<p4est_locidx_t, short>& whoMap, std::unordered_map<p4est_locidx_t, double>& distanceMap,
	std::unordered_map<p4est_locidx_t, double>& hkMap,
	std::ofstream& pointsFile, std::vector<double>& distances, double& xOnGamma, double& yOnGamma, bool& troubling )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// Level-set function values and target h*kappa (to be returned).
	distances.clear();
	distances.reserve( NUM_COLUMNS );					// True distances and target h/kappa as well.

	int s;												// Index to fill in the sample vector.
	double grad[P4EST_DIM];
	double gradNorm;
	double xyz[P4EST_DIM];
	double pOnInterfaceX, pOnInterfaceY;
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	double hk, centerHk;
	short who;
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

		if( stencil[s] == nodeIdx )	// Send out rough estimation of point on interface, where curvature will be interpolated.
		{
			xOnGamma = pOnInterfaceX;
			yOnGamma = pOnInterfaceY;
		}

//		if( nodeIdx == 3515 )	// This node ends up with target h*kappa = -1 because it's quite close to p2.
//		{
//			std::cout << std::setprecision( 15 )
//					  << "plot(" << xyz[0] << ", " << xyz[1] << ", 'b.', " << pOnInterfaceX << ", " << pOnInterfaceY
//					  << ", 'mo');" << std::endl;
//		}

		// Find the shortest distance and the corresponding h*kappa to merging stars using Newton-Raphson's method.
		try
		{
			distances.push_back( distanceMap.at( stencil[s] ) );	// Check caches to avoid duplicate computations.
			hk = hkMap.at( stencil[s] );
			who = whoMap.at( stencil[s] );
		}
		catch( std::out_of_range& e )
		{
			distances.push_back( mergingStarsExample1.getSignedDistance( stencil[s], xyz[0], xyz[1],
																		 pOnInterfaceX, pOnInterfaceY, hk, who ) );
			distanceMap[stencil[s]] = distances.back();				// Update caches.
			hkMap[stencil[s]] = hk;
			whoMap[stencil[s]] = who;

//			if( nodeIdx == 3515 )
//			{
//				std::cout << std::setprecision( 15 )
//						  << "plot(" << pOnInterfaceX << ", " << pOnInterfaceY << ", 'ko');" << std::endl;
//			}
		}

		if( stencil[s] == nodeIdx )
			centerHk = hk;
	}

	sample[s] = centerHk;								// Last column holds h*kappa.
	distances.push_back( sample[s] );

	// Determine if it's a troubling stencil: where either the center node lies in a disconinuity region, or one of its
	// 8 neighbors owes its signed distance to a star other than center's or lies also in a discontinuity region.
	troubling = whoMap[nodeIdx] == 0;
	for( s = 0; !troubling && s < 9; s++ )
		troubling = whoMap[stencil[s]] != whoMap[nodeIdx];

	// Write center sample node index and coordinates.
	node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
	pointsFile << nodeIdx << "," << xyz[0] << "," << xyz[1] << "," << troubling << std::endl;

	return sample;
}


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -1, MAX_D = -MIN_D;					// Domain space is [-1, +1]^2.
	const int MAX_REFINEMENT_LEVEL = 7;							// Maximum level of refinement.
	const double H = 1 / pow( 2, MAX_REFINEMENT_LEVEL );		// Highest spatial resolution in x/y directions.

	std::string DATA_PATH = "/Volumes/YoungMinEXT/pde/data-merging/star_" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";	// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 2;		// Number of columns in phi dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];						// Column headers following the x-y truth table of
	generateColumnHeaders( COLUMN_NAMES );						// three-state variables: m, 0, p.

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( rd() ); 					// Standard mersenne_twister_engine seeded with rd().
	std::normal_distribution<double> normalDistribution;		// Used for bracketing and root finding.

	try
	{
		// Initializing parallel environment (although we enforce working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing data
		// sets to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		//////////////////////////////////// Star-shaped interface parameter setup /////////////////////////////////////

		// Example 1.
		MergingStarsExample1 mergingStarsExample1( gen, normalDistribution, H );

		//////////////////////////////////////////// Prepare output files //////////////////////////////////////////////

		// Check for output directory, e.g. "data/star_7/merging/example_1/".  If it doesn't exist, create it.
		DATA_PATH += "merging/";
		checkOrCreateDirectory( DATA_PATH );								// First part of path: merging.
		DATA_PATH += "example_1/";
		checkOrCreateDirectory( DATA_PATH );								// Second part of path: example_1.

		// Prepare files where to write sample level-set function values reinitialized with fast sweeping and with pde-
		// based equation using 5, 10, and 20 iterations.
		std::string phiKeys[4] = { "fsm", "iter5", "iter10", "iter20" };	// The 4 types of reinitialized phi values.
		std::unordered_map<std::string, std::ofstream> phiFilesMap;
		std::unordered_map<std::string, std::ofstream> sdfPhiFilesMap;
		std::unordered_map<std::string, double> maxREMap;					// Track the (relative) max. absolute error.
		phiFilesMap.reserve( 4 );
		sdfPhiFilesMap.reserve( 4 );
		maxREMap.reserve( 4 );
		for( const auto& key : phiKeys )
		{
			phiFilesMap[key] = std::ofstream();
			std::string fileName = DATA_PATH + key + "_phi.csv";
			phiFilesMap[key].open( fileName, std::ofstream::trunc );
			if( !phiFilesMap[key].is_open() )
				throw std::runtime_error( "Phi values output file " + fileName + " couldn't be opened!" );

			std::ostringstream headerStream;							// Write output file header.
			for( int i = 0; i < NUM_COLUMNS - 1; i++ )
				headerStream << "\"" << COLUMN_NAMES[i] << "\",";
			headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
			phiFilesMap[key] << headerStream.str() << std::endl;
			phiFilesMap[key].precision( 15 );							// Precision for floating point numbers.

			maxREMap[key] = 0;

			// Signed-distance files.
			sdfPhiFilesMap[key] = std::ofstream();
			fileName = DATA_PATH + key + "_phi_sdf.csv";
			sdfPhiFilesMap[key].open( fileName, std::ofstream::trunc );
			if( !sdfPhiFilesMap[key].is_open() )
				throw std::runtime_error( "Exact signed-distance phi values output file " + fileName + " couldn't be opened!" );
			sdfPhiFilesMap[key] << headerStream.str() << std::endl;
			sdfPhiFilesMap[key].precision( 15 );
		}

		// Prepare files where to write sampled nodes' cartesian coordinates for each reinitialization scheme and their
		// discontinuity flag value.
		std::unordered_map<std::string, std::ofstream> pointsFilesMap;
		pointsFilesMap.reserve( 4 );
		for( const auto& key : phiKeys )
		{
			pointsFilesMap[key] = std::ofstream();
			std::string pointsFileName = DATA_PATH + key + "_points.csv";
			pointsFilesMap[key].open( pointsFileName, std::ofstream::trunc );
			if( !pointsFilesMap[key].is_open() )
				throw std::runtime_error( "Point coordinates file " + pointsFileName + " couldn't be opened!" );

			pointsFilesMap[key].precision( 15 );
			pointsFilesMap[key] << R"("i","x","y","f")" << std::endl;	// Write header: node index, x and y coords, and discontinuity flag.
		}

		// Now prepare files to write the quality metric for each grid node along Gamma: Q(x) = |1 - |grad_phi(x)||.
		std::unordered_map<std::string, std::ofstream> gradErrorFilesMap;
		gradErrorFilesMap.reserve( 4 );
		for( const auto& key : phiKeys )
		{
			gradErrorFilesMap[key] = std::ofstream();
			std::string gradErrorFileName = DATA_PATH + key + "_grad_errors.csv";
			gradErrorFilesMap[key].open( gradErrorFileName, std::ofstream::trunc );
			if( !gradErrorFilesMap[key].is_open() )
				throw std::runtime_error( "Grad error file " + gradErrorFileName + " couldn't be opened!" );

			gradErrorFilesMap[key].precision( 15 );
			gradErrorFilesMap[key] << R"("error")" << std::endl;		// Write header: "error".
		}

		// Prepare file where to write the params.  These are used in python to draw each star.  The format is:
		//         c.x  |  c.y  |  a  |  b  |  d  |  p  |
		// -------------+-------+-----+-----+-----+-----+
		// Row 1:  cx1  |  cy1  | a1  | b1  | d1  |  p1 |
		// Row 2:  cx1  |  cy1  | a1  | b1  | d1  |  p1 |
		// Where c is the center coordinates, a the arm amplitude, b the base radius, d the angular phase, and p the
		// number of arms in the star.
		std::ofstream paramsFile;
		std::string paramsFileName = DATA_PATH + "params.csv";
		paramsFile.open( paramsFileName, std::ofstream::trunc );
		if( !paramsFile.is_open() )
			throw std::runtime_error( "Params file " + paramsFileName + " couldn't be opened!" );

		std::ostringstream headerParamsStream;
		headerParamsStream << R"("c.x","c.y","a","b","d","p")";
		paramsFile << headerParamsStream.str() << std::endl;

		paramsFile.precision( 15 );
		for( short i = 0; i < 2; i++ )
		{
			const Point2& c = mergingStarsExample1.getCenter( i );
			const geom::Star& star = mergingStarsExample1.getStar( i );
			paramsFile << c.x << "," << c.y << ","
					   << star.getA() << "," << star.getB() << "," << star.getD() << "," << star.getP() << std::endl;
		}

		/////////////////////////////////////////// Generating the datasets ////////////////////////////////////////////

		printf( ">> Generating data sets for merging stars example 1 with maximum refinement level of %i and H = %f\n",
			MAX_REFINEMENT_LEVEL, H );

		// Domain information.
		int n_xyz[] = {2, 2, 2};									// One tree per unit-dimension.
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
		splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &mergingStarsExample1, 4 );

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

		// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
		sample_cf_on_nodes( p4est, nodes, mergingStarsExample1, phi );

		// Prepare scaffold to collect nodal indices along the interface.  Also, collect indicators for those nodes that
		// are adjacent to the interface and have a full 9-point uniform neighborhood.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );

		Vec interfaceFlag;
		ierr = VecDuplicate( phi, &interfaceFlag );
		CHKERRXX( ierr );

		Vec gradIndicator;
		ierr = VecDuplicate( phi, &gradIndicator );
		CHKERRXX( ierr );

		// Parallel vectors required for interpolated curvature onto the interface.
		Vec curvature, normal[P4EST_DIM];
		ierr = VecDuplicate( phi, &curvature );
		CHKERRXX( ierr );
		for( auto& dim : normal )
		{
			VecCreateGhostNodes( p4est, nodes, &dim );
			CHKERRXX( ierr );
		}

		double *interfaceFlagPtr, *gradIndicatorPtr;
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );
		ierr = VecGetArray( gradIndicator, &gradIndicatorPtr );
		CHKERRXX( ierr );
		for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
		{
			interfaceFlagPtr[i] = 0;		// Init to zero and set flag of (valid) nodes along interface to 1 below.
			gradIndicatorPtr[i] = 0;		// Init to zero and then store the absolute difference between 1 and grad(f).
		}

		// Parallel vector to store troubling flag of interface nodes.
		Vec vTroubling;
		ierr = VecDuplicate( interfaceFlag, &vTroubling );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, vTroubling );
		CHKERRXX( ierr );

		double *vTroublingPtr;
		ierr = VecGetArray( vTroubling, &vTroublingPtr );
		CHKERRXX( ierr );

		// Parallel vector to store target h*kappa for nodes along the interface.
		Vec vHKappa;
		ierr = VecDuplicate( interfaceFlag, &vHKappa );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, vHKappa );
		CHKERRXX( ierr );

		double *vHKappaPtr;
		ierr = VecGetArray( vHKappa, &vHKappaPtr );
		CHKERRXX( ierr );

		// Reinitialize level-set function (using the fast sweeping method and the pde-based equation with 5, 10, and 20
		// iterations).  We go through the hash map of output files to write the corresponding samples.
		std::unordered_map<std::string, Vec> reinitPhis;
		std::unordered_map<std::string, const double*> reinitPhiReadPtrs;
		reinitPhis.reserve( 4 );
		reinitPhiReadPtrs.reserve( 4 );
		for( const auto& key : phiKeys )
		{
			parStopWatch watch;
			watch.start();

			std::cout << "   :: Collecting samples for [" << key << "]" << std::endl;
			reinitPhis[key] = Vec{};		// Save reinitialized level-set function values for each scheme separately.
			ierr = VecDuplicate( phi, &reinitPhis[key] );
			CHKERRXX( ierr );
			ierr = VecCopy( phi, reinitPhis[key] );
			CHKERRXX( ierr );

			// Reinitialization.
			if( key == "fsm" )				// Fast sweeping?
			{
				FastSweeping fsm;
				fsm.prepare( p4est, nodes, &nodeNeighbors, xyz_min, xyz_max );
				fsm.reinitializeLevelSetFunction( &reinitPhis[key], 8 );
			}
			else							// PDE-based?
			{
				my_p4est_level_set_t ls( &nodeNeighbors );
				ls.reinitialize_2nd_order( reinitPhis[key], std::stoi( key.substr( 4 ) ) );		// 5, 10, 15 iterations.
			}

			// Compute curvature with reinitialized data, which will be interpolated at the interface.
			compute_normals( nodeNeighbors, reinitPhis[key], normal );
			compute_mean_curvature( nodeNeighbors, reinitPhis[key], normal, curvature );

			// Prepare linear interpolation.
			my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
			interpolation.set_input( curvature, linear );

			// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are
			// the points we'll use to create our sample files.  We repeatedly do this for each reinitialization scheme
			// because the interface could move for the pde-based method.
			std::vector<p4est_locidx_t> indices;
			nodesAlongInterface.getIndices( &reinitPhis[key], indices );

			// Getting the full uniform stencils of interface points.
			reinitPhiReadPtrs[key] = nullptr;
			ierr = VecGetArrayRead( reinitPhis[key], &reinitPhiReadPtrs[key] );
			CHKERRXX( ierr );

			// Memoization hash maps.
			std::unordered_map<p4est_locidx_t, short> whoMap;		// Stores to whom a node owes its minimum distance
																	// and h*kappa.  Used to identify troubling nodes.
			std::unordered_map<p4est_locidx_t, double> distanceMap;	// Stores the exact signed distance to Gamma.
			std::unordered_map<p4est_locidx_t, double> hkMap;		// Stores the hk for visited nodes.

			// Now, collect samples with reinitialized and exact signed-distance level-set function values and
			// target h*kappa.
			int nSamples = 0;
			std::vector<std::vector<double>> samples;
			std::vector<std::vector<double>> sdfSamples;
			const quad_neighbor_nodes_of_node_t *qnnnPtr;
			double grad[P4EST_DIM], xyz[P4EST_DIM];
			double gradError;
			for( auto n : indices )
			{
				std::vector<p4est_locidx_t> stencil;	// Will contain 9 nodal indices in 2D.
				try
				{
					if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
					{
						double xOnGamma, yOnGamma;
						bool troubling;
						std::vector<double> distances;	// Holds the signed distances for error measuring.
						node_xyz_fr_n( n, p4est, nodes, xyz );
						std::vector<double> sample = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, stencil, p4est,
																  nodes, &nodeNeighbors, reinitPhiReadPtrs[key],
																  mergingStarsExample1, whoMap, distanceMap, hkMap,
																  pointsFilesMap[key], distances, xOnGamma, yOnGamma,
																  troubling );
						sample[NUM_COLUMNS - 1] = H * interpolation( xOnGamma, yOnGamma );	// Attach interpolated hk.
						distances.push_back( 0 );											// Dummy column.
						samples.push_back( sample );
						sdfSamples.push_back( distances );
						nSamples++;

						nodeNeighbors.get_neighbors( n, qnnnPtr );		// Write the grad error to a file.
						qnnnPtr->gradient_without_correction( reinitPhiReadPtrs[key], grad );
						gradError = ABS( compute_L2_norm( grad, P4EST_DIM ) - 1.0 );
						gradErrorFilesMap[key] << gradError << std::endl;

						if( key == "iter10" )	// Set valid node-along-interface and other indicators just in iter10.
						{
							interfaceFlagPtr[n] = 1;
							gradIndicatorPtr[n] = gradError;
							vTroublingPtr[n] = troubling;
							vHKappaPtr[n] = sample[NUM_COLUMNS - 2];
						}

						// Also, check error for each reinitialization method using the "true" distances from above.
						for( int i = 0; i < NUM_COLUMNS - 2; i++ )
						{
							double error = ( distances[i] - sample[i] ) / H;
							maxREMap[key] = MAX( maxREMap[key], ABS( error ) );
						}
					}
				}
				catch( std::exception &e )
				{
					std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
				}
			}

			// Write all reinitialized samples collected.
			for( const auto& row : samples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( phiFilesMap[key], "," ) );		// Inner elements.
				phiFilesMap[key] << row.back() << std::endl;
			}

			// Also write all exact signed-distance function values.
			for( const auto& row : sdfSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( sdfPhiFilesMap[key], "," ) );		// Inner elements.
				sdfPhiFilesMap[key] << row.back() << std::endl;
			}

			std::cout << "   Generated " << nSamples << " samples for reinitialization " << key << std::endl;
			std::cout << "   Max (relative) absolute error for " << key << " was " << maxREMap[key] << std::endl;
			std::cout << "   Timing: " << watch.get_duration_current() << std::endl;
			watch.stop();
		}
		std::cout << "<< Done!" << std::endl;

		// Write paraview file to visualize the star interface and nodes following it along.
		std::ostringstream oss;
		oss << "merging_stars_example_1";
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								8, 0, oss.str().c_str(),
								VTK_POINT_DATA, "fsm_phi", reinitPhiReadPtrs["fsm"],
								VTK_POINT_DATA, "iter5_phi", reinitPhiReadPtrs["iter5"],
								VTK_POINT_DATA, "iter10_phi", reinitPhiReadPtrs["iter10"],
								VTK_POINT_DATA, "iter20_phi", reinitPhiReadPtrs["iter20"],
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr,
								VTK_POINT_DATA, "gradIndicator", gradIndicatorPtr,
								VTK_POINT_DATA, "troublingFlag", vTroublingPtr,
								VTK_POINT_DATA, "hKappa", vHKappaPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		// Cleaning up.
		for( const auto& key : phiKeys )
		{
			ierr = VecRestoreArrayRead( reinitPhis[key], &reinitPhiReadPtrs[key] );
			CHKERRXX( ierr );
		}

		ierr = VecRestoreArray( vHKappa, &vHKappaPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( vTroubling, &vTroublingPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( gradIndicator, &gradIndicatorPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( vHKappa );
		CHKERRXX( ierr );

		ierr = VecDestroy( vTroubling );
		CHKERRXX( ierr );

		ierr = VecDestroy( interfaceFlag );
		CHKERRXX( ierr );

		ierr = VecDestroy( gradIndicator );
		CHKERRXX( ierr );

		ierr = VecDestroy( curvature );
		CHKERRXX( ierr );

		for( auto& dim : normal )
		{
			ierr = VecDestroy( dim );
			CHKERRXX( ierr );
		}

		for( const auto& key : phiKeys )
		{
			ierr = VecDestroy( reinitPhis[key] );
			CHKERRXX( ierr );
		}

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );

		// Closing file objects that were open for phi values, points, angles, and grad errors.
		for( const auto& key : phiKeys )
		{
			phiFilesMap[key].close();
			pointsFilesMap[key].close();
			gradErrorFilesMap[key].close();
			sdfPhiFilesMap[key].close();
		}
		paramsFile.close();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}