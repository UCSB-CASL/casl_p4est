// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <fstream>

// casl_p4est
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>
#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/casl_math.h>

#include "PointsOnCurves.h"

// Root finding on flower dataset.
#include "DistThetaRootFinding.h"

/**
 * Global variables.
 */
const std::string DATA_PATH = "/Users/youngmin/Documents/CS/CASL/LSCurvatureML/data/adaptiveFlower/";
const std::string COLUMN_NAMES[] = {
		"(i-1,j+1)", "(i,j+1)", "(i+1,j+1)", "(i-1,j)", "(i,j)", "(i+1,j)", "(i-1,j-1)", "(i,j-1)", "(i+1,j-1)",
		"h*kappa", "h"
};
const int NUM_COLUMNS = 11;

/**
 * Revalidate previously identified node near the interface (i.e. one of its four edges is crossed by the interface).
 * @param nodeIdx [in]: Currently evaluated node index.
 * @param nodeNeighbors [in]: Neighborhood structure of currently evaluated node.
 * @param phiPtr [in]: Phi values for nodes in neighborhood.
 * @param interfaceFlagPtr [in,out]: Flag that indicates whether a node should be part of interface.
 * @param h [in]: The minimum spatial step size expected for quad neighbors.
 */
void revalidateNodeNearInterface( const p4est_locidx_t nodeIdx, const my_p4est_node_neighbors_t& nodeNeighbors,
		const double* phiPtr, double* interfaceFlagPtr, const double h )
{
	if( interfaceFlagPtr[nodeIdx] != 0 )									// Consider only potential interface points.
	{
		if( phiPtr[nodeIdx] == 0 )											// Case of node exactly on interface.
			return;

		const quad_neighbor_nodes_of_node_t& qnnn = nodeNeighbors.get_neighbors( nodeIdx );		// Local neighborhood of currently evaluated node.
		double directions[] = { qnnn.f_p00_linear( phiPtr ), qnnn.f_m00_linear( phiPtr ),		// Horizontal directions.
								qnnn.f_0p0_linear( phiPtr ), qnnn.f_0m0_linear( phiPtr ) 		// Vertical directions.
		};

		bool revalidated = false;
		for( double direction : directions )								// Check when there's a sign change in phi values.
		{
			if( direction * phiPtr[nodeIdx] < 0 )
			{
				revalidated = true;											// Current node has been proven to be 'on' the interface.

				if( qnnn.d_p00 > 1.01 * h || qnnn.d_m00 > 1.01 * h || qnnn.d_0p0 > 1.01 * h || qnnn.d_0m0 > 1.01 * h ||	// Check distances are up to h: E, W, R, S.
					qnnn.d_p00_p0 > 1.01 * h || qnnn.d_m00_p0 > 1.01 * h ||												// NE, NW.
					qnnn.d_p00_m0 > 1.01 * h || qnnn.d_m00_m0 > 1.01 * h ||												// SE, SW.
					qnnn.d_0p0_p0 > 1.01 * h || qnnn.d_0p0_m0 > 1.01 * h ||												// NE, NW.
					qnnn.d_0m0_p0 > 1.01 * h || qnnn.d_0m0_m0 > 1.01 * h )												// SE, SW.
				{
					double xyz[P4EST_DIM];
					node_xyz_fr_n( nodeIdx, nodeNeighbors.get_p4est(), nodeNeighbors.get_nodes(), xyz );
					throw std::runtime_error( "Node " + std::to_string( nodeIdx ) + " at "
																	 "(" + std::to_string( xyz[0] ) + ", " + std::to_string( xyz[1] ) + ") "
																	 "has neighbors at more than a distance h from it" );
				}
				break;
			}
		}
		if( !revalidated )
			interfaceFlagPtr[nodeIdx] = 0;									// Revert interface flag to zero: point not "connected" to interface.
	}
}

/**
 * Generate the sample values for a node (revalidated to be) near the interface.
 * @param nodeIdx [in]: Node index.
 * @param nodeNeighbors [in]: Node neighbors.
 * @param phiPtr [in]: Level set function values.
 * @param interfaceFlagPtr [in]: Flagged nodes near interface.
 * @param h [in]: Minimum spatial step size.
 * @param flower [in]: Reference to flower object initialized with appropriate parameters.
 * @param pointsFile [in|out]: File where to write nodes close to interface.
 * @param anglesFile [in|out]: File where to write angle parameter for projection of nodes on interface.
 * @return A 9+1+1 element sample with the stencil phi values, the expected curvature, and h.
 */
std::vector<double> sampleNodeNearInterface( const p4est_locidx_t nodeIdx, const my_p4est_node_neighbors_t& nodeNeighbors,
		const double* phiPtr, const double* interfaceFlagPtr, const double h, const Flower& flower, std::ofstream& pointsFile, std::ofstream& anglesFile )
{
	if( interfaceFlagPtr[nodeIdx] != 1 )
		throw std::runtime_error( "Node " + std::to_string( nodeIdx ) + " is not near interface!" );

	// Create the stencil of phi values.
	const quad_neighbor_nodes_of_node_t& qnnn = nodeNeighbors.get_neighbors( nodeIdx );		// Quad neighborhood at currently evaluated node.
	double fNE, fSE, fNW, fSW;																// The four diagonal phi values we need for stencil.
	fNE = fSE = fNW = fSW = std::numeric_limits<double>::infinity();

	// Diagonal neighbor nodes indices (-1 if they are undefined).
	p4est_locidx_t eastNeighborIdx = qnnn.neighbor_p00();
	p4est_locidx_t northNeighborIdx = qnnn.neighbor_0p0();
	p4est_locidx_t westNeighborIdx = qnnn.neighbor_m00();
	p4est_locidx_t southNeighborIdx = qnnn.neighbor_0m0();

	// Diagonal neighbor nodes (if they don't exist, they are initialized to current's node quad neighbors).
	const quad_neighbor_nodes_of_node_t& eastQnnn = (eastNeighborIdx != -1)? nodeNeighbors.get_neighbors( eastNeighborIdx ) : qnnn;
	const quad_neighbor_nodes_of_node_t& northQnnn = (northNeighborIdx != -1)? nodeNeighbors.get_neighbors( northNeighborIdx ) : qnnn;
	const quad_neighbor_nodes_of_node_t& westQnnn = (westNeighborIdx != -1 )? nodeNeighbors.get_neighbors( westNeighborIdx ) : qnnn;
	const quad_neighbor_nodes_of_node_t& southQnnn = (southNeighborIdx != -1 )? nodeNeighbors.get_neighbors( southNeighborIdx ) : qnnn;

	// For NE:
	if( eastNeighborIdx != -1 )										// From east node.
	{
		fNE = eastQnnn.f_0p0_linear( phiPtr );
		fSE = eastQnnn.f_0m0_linear( phiPtr );						// Found fSE as well.
	}
	else
	{
		if( northNeighborIdx != -1 )								// From north node.
		{
			fNE = northQnnn.f_p00_linear( phiPtr );
			fNW = northQnnn.f_m00_linear( phiPtr );					// Found fNW as well.
		}
		else
			throw std::domain_error( "Can't define NE location for node " + std::to_string( nodeIdx ) );
	}

	// For NW:
	if( fNW == std::numeric_limits<double>::infinity() )
	{
		if( northNeighborIdx != -1 )								// From north node.
			fNW = northQnnn.f_m00_linear( phiPtr );
		else
		{
			if( westNeighborIdx != -1 )
			{
				fNW = westQnnn.f_0p0_linear( phiPtr );
				fSW = westQnnn.f_0m0_linear( phiPtr );				// Found fSW as well.
			}
			else
				throw std::domain_error( "Can't define NW location for node " + std::to_string( nodeIdx ) );
		}
	}

	// For SW:
	if( fSW == std::numeric_limits<double>::infinity() )
	{
		if( westNeighborIdx != -1 )									// From west node.
			fSW = westQnnn.f_0m0_linear( phiPtr );
		else
		{
			if( southNeighborIdx != -1 )							// From south node.
			{
				fSW = southQnnn.f_m00_linear( phiPtr );
				fSE = southQnnn.f_p00_linear( phiPtr );				// Found fSE as well.
			}
			else
				throw std::domain_error( "Cant' define SW location for node " + std::to_string( nodeIdx ) );
		}
	}

	// For SE:
	if( fSE == std::numeric_limits<double>::infinity() )
	{
		if( southNeighborIdx != -1 )								// From south node.
			fSE = southQnnn.f_p00_linear( phiPtr );
		else
			throw std::domain_error( "Can't define SE location for node " + std::to_string( nodeIdx ) );
	}

	// Build stencil with NE, NW, SW, SE.
	double stencil[] = {
									fNW, qnnn.f_0p0_linear( phiPtr ), 						  fNE,
			qnnn.f_m00_linear( phiPtr ), 			 phiPtr[nodeIdx], qnnn.f_p00_linear( phiPtr ),
									fSW, qnnn.f_0m0_linear( phiPtr ), 						  fSE
	};

	std::vector<double> sample( NUM_COLUMNS, 0 );		// Phi, h and h*kappa results.

	int s;
	double leftPhi, rightPhi, topPhi, bottomPhi;		// To compute grad(\phi_{i,j}).
	double centerPhi;									// \phi{i,j}.
	for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
	{
		sample[s] = stencil[s];

		switch( s )
		{
			case 1: topPhi = sample[s]; break;			// \phi_{i,j+1}.
			case 3: leftPhi = sample[s]; break;			// \phi_{i-1,j}.
			case 4: centerPhi = sample[s]; break;		// Point's phi value.
			case 5: rightPhi = sample[s]; break;		// \phi_{i+1,j}.
			case 7: bottomPhi = sample[s]; break;		// \phi_{i,j-1}.
			default: ;
		}
	}

	// Computing the target curvature.
	double xyz[P4EST_DIM];
	node_xyz_fr_n( nodeIdx, nodeNeighbors.get_p4est(), nodeNeighbors.get_nodes(), xyz );
	double grad[] = {
			( rightPhi - leftPhi ) / ( 2 * h ),							// Using central differences.
			( topPhi - bottomPhi ) / ( 2 * h )
	};
	double gradNorm = sqrt( grad[0] * grad[0] + grad[1] * grad[1] );
	double pOnInterfaceX = xyz[0] - grad[0] / gradNorm * centerPhi,		// Initial cartesian coordinates of projection of grid point on interface.
		   pOnInterfaceY = xyz[1] - grad[1] / gradNorm * centerPhi;

	double thetaOnInterface = atan2( pOnInterfaceY, pOnInterfaceX );	// Initial guess for root finding method.
	thetaOnInterface = ( thetaOnInterface < 0 )? thetaOnInterface + 2 * M_PI : thetaOnInterface;
	double valOfDerivative;
	double newThetaOnInterface = distThetaDerivative( xyz[0], xyz[1], flower, valOfDerivative, thetaOnInterface, thetaOnInterface - 0.0001, thetaOnInterface + 0.0001 );
	double r = flower.r( thetaOnInterface );							// Recalculating closest point on interface.
	pOnInterfaceX = r * cos( newThetaOnInterface );
	pOnInterfaceY = r * sin( newThetaOnInterface );

//	std::cout << valOfDerivative << "; plot([" << xyz[0] << "], [" << xyz[1] << "], 'b.', [" << pOnInterfaceX << "], [" << pOnInterfaceY << "], 'm.');" << std::endl;

	double dx = xyz[0] - pOnInterfaceX,						// Verify that point on interface is not far from corresponding grid point.
		   dy = xyz[1] - pOnInterfaceY;
	if( dx * dx + dy * dy <= h * h )
		thetaOnInterface = newThetaOnInterface;				// Theta is OK.
	else
		std::cerr << "Minimization placed point on interface too far.  Reverting back to point on interface calculated with phi values" << std::endl;

	sample[s] = h * flower.curvature( thetaOnInterface );	// Second to last column holds h*\kappa.
	s++;
	sample[s] = h;											// Last column holds spatial scale h.

	// Write sample point cartesian coordinates.
	pointsFile << xyz[0] << "," << xyz[1] << std::endl;

	// Write angle parameter for projected point on interface.
	anglesFile << ( thetaOnInterface < 0 ? 2 * M_PI + thetaOnInterface : thetaOnInterface ) << std::endl;

	return sample;
}

int main (int argc, char* argv[])
{
	try
	{
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		p4est_t *p4est;
		p4est_nodes_t *nodes;
		PetscErrorCode ierr;

		const int NUMBER_OF_ITERATIONS = 5;
		double minVal = -1;										// Domain variables.
		double h = 0.1;

		// Setting up the flower.
		const int N_GRID_POINTS = 129;
		Flower flower( 0.05, 0.15, 3 );							// For flower with a=0.05: M = 129, padding = 12, N_CELLS = 1, yields 261 grid points per unit.
		flower.getHAndMinVal( N_GRID_POINTS, h, minVal, 5 );	// For flower with a=0.075: M = 129, padding = 5, N_CELLS = 1, yields 263.222 grid points per unit.
		const int N_CELLS = 1;
		const int MAX_LEVEL_OF_REFINEMENT = 7;

		splitting_criteria_cf_and_uniform_band_t cfFlower( 1, MAX_LEVEL_OF_REFINEMENT, &flower, 2.0 );

		parStopWatch w;
		w.start( "total time" );

		// Create the connectivity object.
		p4est_connectivity_t *connectivity;
		my_p4est_brick_t brick;
		int n_xyz[] = {N_CELLS, N_CELLS, N_CELLS};
		double xyz_min[] = {minVal, minVal, minVal};
		double xyz_max[] = {-minVal, -minVal, -minVal};
		int periodic[] = {0, 0, 0};

		connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Now create the forest.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );

		// Refine the forest using a refinement criteria.
		p4est->user_pointer = ( void * ) (&cfFlower);
		my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf_and_uniform_band, nullptr );

		// Finally re-partition.
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create the ghost structure.
		p4est_ghost_t *ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );

		// Generate the node data structure.
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t node_neighbors( &hierarchy, nodes );
		node_neighbors.init_neighbors();

		//////////////////////////////////////////// Prepare output files //////////////////////////////////////////////
		/// Note: We assume this is run in only one process to avoid file writing synchronization problems.

		// Prepare file where to write sample phi values.
		std::ofstream outputFile;
		std::string fileName = DATA_PATH + "phi_iter" + std::to_string( NUMBER_OF_ITERATIONS ) + ".csv";
		outputFile.open( fileName, std::ofstream::trunc );
		if( !outputFile.is_open() )
			throw std::runtime_error( "Phi values output file " + fileName + " couldn't be opened!" );

		std::ostringstream headerStream;									// Write output file header.
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			headerStream << "\"" << COLUMN_NAMES[i] << "\",";
		headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
		outputFile << headerStream.str() << std::endl;

		outputFile.precision( 10 );											// Precision for floating point numbers.

		// Prepare file where to write sample grid point cartesian coordinates.
		std::ofstream pointsFile;
		std::string pointsFileName = DATA_PATH + "points_iter" + std::to_string( NUMBER_OF_ITERATIONS ) + ".csv";
		pointsFile.open( pointsFileName, std::ofstream::trunc );
		if( !pointsFile.is_open() )
			throw std::runtime_error( "Point cartesian coordinates file " + pointsFileName + " couldn't be opened!" );

		pointsFile.precision( 10 );
		pointsFile << R"("x","y")" << std::endl;

		// Prepare file where to write the params.
		std::ofstream paramsFile;
		std::string paramsFileName = DATA_PATH + "params_iter" + std::to_string( NUMBER_OF_ITERATIONS ) + ".csv";
		paramsFile.open( paramsFileName, std::ofstream::trunc );
		if( !paramsFile.is_open() )
			throw std::runtime_error( "Params file " + paramsFileName + " couldn't be opened!" );

		std::ostringstream headerParamsStream;
		headerParamsStream << R"("nGridPoints","h","minVal","a","b","p")";
		paramsFile << headerParamsStream.str() << std::endl;

		paramsFile.precision( 10 );
		paramsFile << N_GRID_POINTS << "," << h << "," << minVal << "," << flower.getA() << "," << flower.getB() << "," << flower.getP() << std::endl;

		// Prepare file where to write the angle parameter for corresponding points on interface.
		std::ofstream anglesFile;
		std::string anglesFileName = DATA_PATH + "angles_iter" + std::to_string( NUMBER_OF_ITERATIONS ) + ".csv";
		anglesFile.open( anglesFileName, std::ofstream::trunc );
		if( !anglesFile.is_open() )
			throw std::runtime_error( "Angles file " + anglesFileName + " couldn't be opened!" );

		anglesFile.precision( 10 );
		anglesFile << R"("theta")" << std::endl;							// Write header.

		/////////////////////////////////////////// Generating the dataset /////////////////////////////////////////////

		Vec phi, interfaceFlag;
		ierr = VecCreateGhostNodes( p4est, nodes, &phi ); CHKERRXX( ierr );
		ierr = VecDuplicate( phi, &interfaceFlag ); CHKERRQ( ierr );

		sample_cf_on_nodes( p4est, nodes, flower, phi );

		my_p4est_level_set_t ls( &node_neighbors );
		ls.reinitialize_2nd_order( phi, NUMBER_OF_ITERATIONS );	// Level set reinitialization.

		double hxyz[P4EST_DIM];
		p4est_dxyz_min( node_neighbors.get_p4est(), hxyz );		// The minimum value of h = -2*minVal / (NUM_CELLS * 2^lmax; and nGridPoints = N_CELLS * 2^lmax + 1.

		// Finding the nodes next to the interface.  These are those whose (phi) value (distance) is smaller than the
		// diagonal of the smallest cell.
		const double *readPhiPtr;
		double *interfaceFlagPtr;
		double diag_min = p4est_diag_min( node_neighbors.get_p4est() );
		double xyz[P4EST_DIM];

		ierr = VecGetArrayRead( phi, &readPhiPtr ); CHKERRQ( ierr );
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr ); CHKERRQ( ierr );

		for( size_t n = 0; n < node_neighbors.get_nodes()->indep_nodes.elem_count; n++ )
		{
			interfaceFlagPtr[n] = 0;							// Mark with 0 a non interface grid node, and 1 for an interface node.
			if( fabs( readPhiPtr[n] ) < diag_min )
			{
				node_xyz_fr_n( n, node_neighbors.get_p4est(), node_neighbors.get_nodes(), xyz );
				interfaceFlagPtr[n] = 1;
//				PetscPrintf( mpi.comm(), "(%f, %f)\n", xyz[0], xyz[1] );
			}
		}

		ierr = VecRestoreArrayRead( phi, &readPhiPtr ); CHKERRQ( ierr );
		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr ); CHKERRQ( ierr );

		/// Keep interface nodes whose any of their four edges is crossed by the interface: write samples at the same time.

		ierr = VecGetArrayRead( phi, &readPhiPtr ); CHKERRXX( ierr );				// Need to load reference data.
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr ); CHKERRXX( ierr );

		std::vector<std::vector<double>> samples;									// Samples we need to export to a CSV.

		// Boundary nodes first (i.e. those in the layer).
		for( size_t i=0; i < node_neighbors.get_layer_size(); i++ )
		{
			p4est_locidx_t nodeIdx = node_neighbors.get_layer_node( i );
			revalidateNodeNearInterface( nodeIdx, node_neighbors, readPhiPtr, interfaceFlagPtr, h );
			if( interfaceFlagPtr[nodeIdx] != 0 )									// Node "on" interface?
				samples.push_back( sampleNodeNearInterface( nodeIdx, node_neighbors, readPhiPtr, interfaceFlagPtr, h, flower, pointsFile, anglesFile ) );
		}

		// Start updating ghost values.
		ierr = VecGhostUpdateBegin( interfaceFlag, INSERT_VALUES, SCATTER_FORWARD ); CHKERRXX( ierr );

		// Continue with the internal nodes.
		for( size_t i = 0; i < node_neighbors.get_local_size(); i++ )
		{
			p4est_locidx_t nodeIdx = node_neighbors.get_local_node( i );
			revalidateNodeNearInterface( nodeIdx, node_neighbors, readPhiPtr, interfaceFlagPtr, h );
			if( interfaceFlagPtr[nodeIdx] != 0 )									// Node "on" interface?
				samples.push_back( sampleNodeNearInterface( nodeIdx, node_neighbors, readPhiPtr, interfaceFlagPtr, h, flower, pointsFile, anglesFile ) );
		}

		// Restore internal data.
		ierr = VecRestoreArrayRead( phi, &readPhiPtr ); CHKERRXX( ierr );

		// Finish the ghost update process to ensure all values are updated.
		ierr = VecGhostUpdateEnd( interfaceFlag, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX( ierr );
		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr ); CHKERRXX( ierr );

		////////////////////////////////////////////// Writing results /////////////////////////////////////////////////

		// Write to file the samples content.
		for( const std::vector<double>& row : samples )
		{
			std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( outputFile, "," ) );		// Inner elements.
			outputFile << row.back() << std::endl;
		}

		double *phi_p;
		std::ostringstream oss;
		oss << "flower_" << mpi.size() << "_" << P4EST_DIM;
		ierr = VecGetArray( phi, &phi_p ); CHKERRXX( ierr );
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr ); CHKERRQ( ierr );
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								2, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phi_p,
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );
		ierr = VecRestoreArray( phi, &phi_p ); CHKERRXX( ierr );
		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr ); CHKERRQ( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( phi ); CHKERRXX( ierr );
		ierr = VecDestroy( interfaceFlag ); CHKERRQ( ierr );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );

		outputFile.close();
		pointsFile.close();
		paramsFile.close();

		w.stop();
		w.read_duration();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}

