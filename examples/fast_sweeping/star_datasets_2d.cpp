/**
 * Generate star-shaped interface data sets for evaluating our feedforward neural network, which was trained on
 * circular and sinusoidal interfaces using samples from a reinitialized level-set function with fast sweeping method.
 * Also, collect samples from PDE reinitialization using 5, 10, and 20 iterations for accuracy comparison.
 * For each reinitialization scheme, generate files with level-set function values, point coordinates, and projections
 * angles of nodes detected along the interface.  We do this because some reinitialization schemes are not always
 * successful at finding the exact distances from stencils to the star-shaped interface when using Newton-Raphson's root
 * finding.
 *
 * Seek the [CHANGE] label to locate where to set the parameters for distinct star-shape level-set function
 * configurations.
 *
 * The output files generated are placed in the data/star_L/a_X.XXX/b_Y.YYY, where L is the maximum level of refinement,
 * X.XXX is the star arm amplitude, and Y.YYY is the star base radius.  At least the data/star_L directory must exist.
 * The rest of subdirectories are created if they don't exist.  Files are overwritten ([R] = fsm|iter5|iter10|iter20):
 * a) params.csv Star parameters.
 * b) [R]_angles.csv Angles of normal-projected points of center nodes of stencils onto the interface.
 * c) [R]_points.csv Cartesian coordinates of center nodes of stencils sampled along the interface.
 * d) [R]_phi.csv Actual level-set function values of the 9-point stencils along the interface.
 *
 * Developer: Luis Ángel.
 * Date: June 6, 2020.
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
#include <src/my_p8est_level_set.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_fast_sweeping.h>
#include <src/my_p4est_nodes_along_interface.h>
#include <src/my_p4est_level_set.h>
#endif

#include "star_theta_root_finding.h"
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include <sys/stat.h>

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
 * @param [in] neighbors Pointer to neighbors data structure.
 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
 * @param [in] star The level-set function with a star-shaped interface.
 * @param [in] gen Random-number generator device.
 * @param [in] uniformDistribution A uniform random-variable distribution.
 * @param [in/out] pointsFile Reference to file object where to write coordinates of nodes adjacent to \Gamma.
 * @param [in/out] anglesFile Reference to file object where to write angles of normal projected points on \Gamma.
 * @param [out] distances A vector of "true" distances from all of 9 stencil points to the star-shaped level-set.
 * @return Vector of sampled, reinitialized level-set function values for the stencil centered at the nodeIdx node.
 * @throws runtime exception if distance between original projected point on interface and point found by Newton-Raphson
 * are farther than H and if Newton-Raphson's method converged to a local minimum (didn't get to zero).
 */
[[nodiscard]] std::vector<double> sampleNodeAdjacentToInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
	const double H, const std::vector<p4est_locidx_t>& stencil, const p4est_t *p4est, const p4est_nodes_t *nodes,
	const my_p4est_node_neighbors_t *neighbors, const double *phiReadPtr, const geom::Star& star, std::mt19937& gen,
	std::uniform_real_distribution<double>& uniformDistribution,
	std::ofstream& pointsFile, std::ofstream& anglesFile, std::vector<double>& distances )
{
	std::vector<double> sample( NUM_COLUMNS, 0 );		// (Reinitialized) level-set function values and target h\kappa.
	distances.clear();
	distances.reserve( NUM_COLUMNS - 1 );				// Only true distances.

	int s;												// Index to fill in the sample vector.
	double grad[P4EST_DIM];
	double gradNorm;
	double xyz[P4EST_DIM];
	double pOnInterfaceX, pOnInterfaceY, newPOnInterfaceX, newPOnInterfaceY;
	const quad_neighbor_nodes_of_node_t *qnnnPtr;
	double theta, r, newTheta, valOfDerivative, centerTheta;
	double dx, dy, newDistance, distanceBtwnPOnInterface;
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

		// Get initial angle for polar approximation to point on star interface.
		theta = atan2( pOnInterfaceY, pOnInterfaceX );
		theta = ( theta < 0 )? theta + 2 * M_PI : theta;
		r = star.r( theta );
		pOnInterfaceX = r * cos( theta );
		pOnInterfaceY = r * sin( theta );				// Better approximation of projection of stencil point onto star.

		// Compute current distance to \Gamma using the improved point on interface.
		dx = xyz[0] - pOnInterfaceX;
		dy = xyz[1] - pOnInterfaceY;
		distances.push_back( sqrt( SQR( dx ) + SQR( dy ) ) );

		// Find theta that yields "a" minimum distance between stencil point and star using Newton-Raphson's method.
		valOfDerivative = 1;
		iterations = 0;
		while( ABS( valOfDerivative ) > 1e-8 && iterations < 20 )	// Avoiding local minima with random exploration.
		{
			double h = ( ( iterations % 2 )? H : -H ) * uniformDistribution( gen );	// Flip sign to explore a wide range.
			newTheta = theta + h;									// Initial parametric guess for root finding.
			newTheta = distThetaDerivative( stencil[s], xyz[0], xyz[1], star, valOfDerivative, newTheta, newTheta - H,
				newTheta + H );
			iterations++;
		}
		r = star.r( newTheta );										// Recalculating closest point on interface.
		newPOnInterfaceX = r * cos( newTheta );
		newPOnInterfaceY = r * sin( newTheta );

		// Compute new distance with "normal projection" of query point onto sine wave.
		dx = xyz[0] - newPOnInterfaceX;
		dy = xyz[1] - newPOnInterfaceY;
		newDistance = sqrt( SQR( dx ) + SQR( dy ) );

		// Compute distance between previous projection and new projection.
		dx = newPOnInterfaceX - pOnInterfaceX;
		dy = newPOnInterfaceY - pOnInterfaceY;
		distanceBtwnPOnInterface = sqrt( SQR( dx ) + SQR( dy ) );
		if( distanceBtwnPOnInterface > H && ABS( valOfDerivative ) > 1e-8 )
			throw std::runtime_error( "Failure with node " + std::to_string( stencil[s] ) );

		if( newDistance < distances[s] )
			distances[s] = newDistance;					// Root finding was successful: keep minimum distance.
		else
			newTheta = theta;

		if( sample[s] < 0 )								// Fix sign.
			distances[s] *= -1;

		if( s == 4 )									// For center node we need theta to yield curvature.
			centerTheta = newTheta;
	}

	sample[s] = H * star.curvature( centerTheta );		// Last column holds h\kappa.

	// Write center sample node index and coordinates.
	node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
	pointsFile << nodeIdx << "," << xyz[0] << "," << xyz[1] << std::endl;

	// Write angle parameter for projected point on interface.
	anglesFile << ( centerTheta < 0 ? 2 * M_PI + centerTheta : centerTheta ) << std::endl;

	return sample;
}

/**
 * Verify if a directory exists.  If not, create it.
 * @param [in] path Directory valid path.
 * @throws Runtime error if directory can't be created or if the path exists and is not a directory.
 */
void checkOrCreateDirectory( const std::string& path )
{
	struct stat info{};
	char aux[10];
	if( stat( path.c_str(), &info ) != 0 )							// Directory doesn't exist?
	{
		if( mkdir( path.c_str(), 0777 ) == -1 )						// Try to create it.
			throw std::runtime_error( "Cannot create " + path + " directory: " + strerror(errno) + "!" );
	}
	else if( !( info.st_mode & (unsigned)S_IFDIR ) )
		throw std::runtime_error( path + " is not a directory!" );
}

int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -0.5, MAX_D = -MIN_D;								// The canonical space is [0,1]^{P4EST_DIM}.
	const double HALF_D = ( MAX_D - MIN_D ) / 2;							// Half domain.
	const int MAX_REFINEMENT_LEVEL = 7;										// Maximum level of refinement.
	const int NUM_UNIFORM_NODES_PER_DIM = (int)pow( 2, MAX_REFINEMENT_LEVEL ) + 1;		// Number of uniform nodes per dimension.
	const double H = ( MAX_D - MIN_D ) / (double)( NUM_UNIFORM_NODES_PER_DIM - 1 );		// Highest spatial resolution in x/y directions.

	std::string DATA_PATH = "/Volumes/YoungMinEXT/fsm/data/star_" + std::to_string( MAX_REFINEMENT_LEVEL ) + "/";	// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 1;					// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS];									// Column headers following the x-y truth table of 3-state variables.
	generateColumnHeaders( COLUMN_NAMES );

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen( rd() ); 					// Standard mersenne_twister_engine seeded with rd().
	std::uniform_real_distribution<double> uniformDistribution;

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To generate datasets we don't admit more than a single process to avoid race conditions when writing data
		// sets to files.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		//////////////////////////////////// Star-shaped interface parameter setup /////////////////////////////////////

		// [CHANGE] Change these values to modify the shape of star interface.
		// Using a=0.075 and b=0.35 for smooth star, and a=0.12 and b=0.305 for sharp star.
		const double A = 0.075;
		const double B = 0.350;
		const int P = 5;
		geom::Star star( A, B, P );
		const double STAR_SIDE_LENGTH = star.getInscribingSquareSideLength();
		if( STAR_SIDE_LENGTH > MAX_D - MIN_D - 4 * H )						// Consider also 2H padding on each side.
			throw std::runtime_error( "Star exceeds allowed space in domain!" );

		//////////////////////////////////////////// Prepare output files //////////////////////////////////////////////

		// Check for output directory, e.g. "data/star_7/a_0.075/b_0.350/".  If it doesn't exist, create it.
		char auxA[10], auxB[10];
		sprintf( auxA, "%.3f", A );
		DATA_PATH += "a_" + std::string( auxA ) + "/";
		checkOrCreateDirectory( DATA_PATH );								// First part of path: a.
		sprintf( auxB, "%.3f", B );
		DATA_PATH += "b_" + std::string( auxB ) + "/";
		checkOrCreateDirectory( DATA_PATH );								// Second part of path: b.

		// Prepare files where to write sample level-set function values reinitialized with fast sweeping and with pde-
		// based equation using 5, 10, and 20 iterations.
		std::string phiKeys[4] = { "fsm", "iter5", "iter10", "iter20" };	// The 4 types of reinitialized phi values.
		std::unordered_map<std::string, std::ofstream> phiFilesMap;
		std::unordered_map<std::string, double> maxREMap;					// Track the (relative) maximum absolute error.
		phiFilesMap.reserve( 4 );
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
		}

		// Prepare files where to write sampled nodes' cartesian coordinates for each reinitialization scheme.
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
			pointsFilesMap[key] << R"("i","x","y")" << std::endl;		// Write header: node index, x, and y coords.
		}

		// Prepare files where to write the angle parameters for corresponding points on interface.
		std::unordered_map<std::string, std::ofstream>anglesFilesMap;
		anglesFilesMap.reserve( 4 );
		for( const auto& key : phiKeys )
		{
			anglesFilesMap[key] = std::ofstream();
			std::string anglesFileName = DATA_PATH + key + "_angles.csv";
			anglesFilesMap[key].open( anglesFileName, std::ofstream::trunc );
			if( !anglesFilesMap[key].is_open() )
				throw std::runtime_error( "Angles file " + anglesFileName + " couldn't be opened!" );

			anglesFilesMap[key].precision( 15 );
			anglesFilesMap[key] << R"("theta")" << std::endl;			// Write header: the theta polar coord.
		}

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
		splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, MAX_REFINEMENT_LEVEL, &star, 2 );

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
		sample_cf_on_nodes( p4est, nodes, star, phi );

		// Prepare scaffold to collect nodal indices along the interface.  Also, collect indicators for those nodes that
		// are adjacent to the interface and have a full 9-point uniform neighborhood.  We'll collect these for fast
		// sweeping renitialization.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );

		Vec interfaceFlag;
		ierr = VecDuplicate( phi, &interfaceFlag );
		CHKERRXX( ierr );

		double *interfaceFlagPtr;
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );
		for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
			interfaceFlagPtr[i] = 0;		// Init to zero and set flag of (valid) nodes along interface to 1.

		// Reinitialize level-set function (using the fast sweeping method and the pde-based equation with 5, 10, and 20
		// iterations).  We go through the hash map of output files to write the corresponding samples.
		std::unordered_map<std::string, Vec> reinitPhis;
		std::unordered_map<std::string, const double*> reinitPhiReadPtrs;
		reinitPhis.reserve( 4 );
		reinitPhiReadPtrs.reserve( 4 );
		for( const auto& key : phiKeys )
		{
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

			// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are the
			// points we'll use to create our sample files.  We repeatedly do this for each reinitialization scheme
			// because the interface could move for the pde-based method.
			std::vector<p4est_locidx_t> indices;
			nodesAlongInterface.getIndices( &reinitPhis[key], indices );

			// Getting the full uniform stencils of interface points.
			reinitPhiReadPtrs[key] = nullptr;
			ierr = VecGetArrayRead( reinitPhis[key], &reinitPhiReadPtrs[key] );
			CHKERRXX( ierr );

			// Now, collect samples with reinitialized level-set function values and target h\kappa.
			int nSamples = 0;
			std::vector<std::vector<double>> samples;
			for( auto n : indices )
			{
				std::vector<p4est_locidx_t> stencil;	// Contains 9 nodal indices in 2D.
				try
				{
					if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
					{
						std::vector<double> distances;	// Holds the signed distances for error measuring.
						std::vector<double> sample = sampleNodeAdjacentToInterface( n, NUM_COLUMNS, H, stencil, p4est,
							nodes, &nodeNeighbors, reinitPhiReadPtrs[key], star, gen, uniformDistribution,
							pointsFilesMap[key], anglesFilesMap[key], distances );
						samples.push_back( sample );
						nSamples++;

						if( key == "fsm" )				// Set valid node-along-interface indicator (just once, in fsm).
							interfaceFlagPtr[n] = 1;

						// Also, check error for each reinitialization method using the "true" distances from above.
						for( int i = 0; i < NUM_COLUMNS - 1; i++ )
						{
							double error = ( distances[i] - sample[i] ) / H;
							maxREMap[key] = MAX( maxREMap[key], ABS( error ) );
						}
					}
				}
				catch( std::exception &e )
				{
					double xyz[P4EST_DIM];				// Position of node at the center of the stencil.
					node_xyz_fr_n( n, p4est, nodes, xyz );
					std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
				}
			}

			// Write all samples collected.
			for( const auto& row : samples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( phiFilesMap[key], "," ) );		// Inner elements.
				phiFilesMap[key] << row.back() << std::endl;
			}

			std::cout << "   Generated " << nSamples << " samples for reinitialization " << key << std::endl;
			std::cout << "   Max (relative) absolute error for " << key << " was " << maxREMap[key] << std::endl;
		}
		std::cout << "<< Done!" << std::endl;
		watch.stop();

		// Write paraview file to visualize the star interface and nodes following it along.
		std::ostringstream oss;
		oss << "star_" << mpi.size() << "_" << P4EST_DIM;
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								5, 0, oss.str().c_str(),
								VTK_POINT_DATA, "fsm_phi", reinitPhiReadPtrs["fsm"],
								VTK_POINT_DATA, "iter5_phi", reinitPhiReadPtrs["iter5"],
								VTK_POINT_DATA, "iter10_phi", reinitPhiReadPtrs["iter10"],
								VTK_POINT_DATA, "iter20_phi", reinitPhiReadPtrs["iter20"],
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		// Cleaning up.
		for( const auto& key : phiKeys )
		{
			ierr = VecRestoreArrayRead( reinitPhis[key], &reinitPhiReadPtrs[key] );
			CHKERRXX( ierr );
		}

		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( interfaceFlag );
		CHKERRXX( ierr );

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

		// Closing file objects that were open for phi values, points, and angles.
		for( const auto& key : phiKeys )
		{
			phiFilesMap[key].close();
			pointsFilesMap[key].close();
			anglesFilesMap[key].close();
		}
		paramsFile.close();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}