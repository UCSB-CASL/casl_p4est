/**
 * Extrapolation across the interface in 2D, using augmented radial basis functions.
 * Author: Luis Ángel (임 영민)
 * Date Created: October 21, 2020.
 */

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

#include <Accelerate/Accelerate.h>
#include <src/casl_geometry.h>
#include <set>
#include <map>
#include <random>
#include "radial_basis_functions.h"

/**
 * Scalar fields to extend over the interface and into Omega+.
 * Notice that they have the same structure as the level-set function because we want to evaluate at the nodes.
 */
class Field: public CF_2
{
private:
	int _choice;			// Choose which function to extend.

public:
	/**
	 * Constructor sets choice to first scalar function.
	 * See the () operator for function options.
	 */
	Field()
	: _choice( 0 ) {}

	/**
	 * The scalar function to extend: f(x,y).
	 * @param [in] x Query point x-coordinate.
	 * @param [in] y Query point y-coordinate.
	 * @return f(x,y).
	 */
	double operator()( double x, double y ) const override
	{
		switch( _choice )
		{
			case 0: return sin( M_PI * x ) * cos( M_PI * y );
			case 1: return sin( M_PI * x ) + cos( M_PI * y );
			default: throw std::invalid_argument( "Invalid scalar function choice!" );
		}
	}

	/**
	 * Choose function to extend.
	 * @param choice
	 */
	void setChoice( int choice )
	{
		_choice = choice;
	}

	/**
	 * Get a string description of selected scalar field.
	 * @return
	 */
	[[nodiscard]] std::string toString() const
	{
		switch( _choice )
		{
			case 0: return "sin(πx)cos(πy)";
			case 1: return "sin(πx) + cos(πy)";
			default: throw std::invalid_argument( "Invalid scalar function choice!" );
		}
	}
};


/**
 * Extrapolate the scalar field using radial basis functions and the information coming from visited nodes.
 * @param [in] atNodeIdx Target node index.
 * @param [in] window Neighboring nodes to be considered in the extrapolation.
 * @param [in] rbf Radial basis function.
 * @param [in] p4est Pointer to the p4est struct.
 * @param [in] nodes Pointer to nodes struct.
 * @param [in,out] rbfFieldPtr A parallel vector with the extrapolation information to also be updated.
 * @param [in] usingPolynomial Whether or not augment the radial basis function with a first-degree polynomial.
 */
void extrapolate( p4est_locidx_t atNodeIdx, std::vector<p4est_locidx_t>& window, const RBF& rbf,
				  const p4est_t *p4est, const p4est_nodes_t *nodes, double *rbfFieldPtr, bool usingPolynomial = true )
{
	// Retrieve spatial information for standing node.
	double xyz[P4EST_DIM];
	node_xyz_fr_n( atNodeIdx, p4est, nodes, xyz );
	Point2 p( xyz[0], xyz[1] );

	// Retrieve spatial information for neighborhood.
	const int N = window.size();
	std::vector<Point2> windowPositions;
	windowPositions.reserve( N );
	for( const auto& n : window )
	{
		node_xyz_fr_n( n, p4est, nodes, xyz );
		windowPositions.emplace_back( xyz[0], xyz[1] );
	}

	// Preparing the matrix and vectors we need.
	const int NUM_COEFFS = (usingPolynomial)? 1 : 0;	// If needed, we use degree polynomial p(x,y) = c0 + c1*x + c2*y
														// to augment the RBF.
														// For MQ, this polynomial is p(x,y) = c because MQ RBFs are
														// conditionally strictly positive definite functions of order 1;
														// this implies the augmenting polynomial should be of degree 0.
	int DIM = N + NUM_COEFFS;		// Effective matrix and vector dimension.
	int LDA = DIM;					// Leading dimension of A matrix.
	int LDB = DIM;					// Leading dimension of b vector.
	int N_RHS = 1;					// Number of right-hand sides.
	char TRANSPOSE = 'T';			// We'll provide the matrix A in row-major order.

	// Building the A matrix to solve the system Ax = b.
	//     | Phi   |  v1   vx   vy |
	//     | ----------------------|
	// A = | v1^T  |  0    0    0  |,  where Phi is an NxN matrix, and v1, vx, and vy are a Nx1 vectors if p(x,y) is degree 1.
	//     | vx^T  |  0    0    0  |   v1 has just 1's, and vx and vy hold the x and y coordinates of the nodes.
	//     | vy^T  |  0    0    0  |
	double A[DIM * DIM];						// In this case, A is symmetric.
	for( int i = 0; i < N; i++ )
	{
		A[i * DIM + i] = rbf( 0 );				// Diagonal elements for Phi.
		for( int j = i + 1; j < N; j++ )
		{
			double d = Point2::norm_L2( windowPositions[i], windowPositions[j] );
			A[i * DIM + j] = A[j * DIM + i] = rbf( d );		// Phi off-diagonal elements.
		}

		if( usingPolynomial )
			A[i * DIM + N] = A[N * DIM + i] = 1;			// v1 and v1^T.
	}

	if( usingPolynomial )
		A[N * DIM + N] = 0;						// Complete the zero diagonal element.

	// Debugging: Matrix A.
//	if( atNodeIdx == 2859 )
//	{
//		printf( "[" );
//		for( int i = 0; i < DIM; i++ )
//		{
//			printf( "[" );
//			for( int j = 0; j < DIM; j++ )
//				printf( "%.15g, ", A[i * DIM + j] );
//			printf( "],\n" );
//		}
//		printf( "]\n" );
//	}

	// Bulding the right-hand side vector b.
	//     |   f0   |
	//     |   f1   |, where fi is known scalar field at ith node.  The remaining DIM - N components of the vector are
	//     |    :   |  zeroed out to meet the miminimization conditions.
	// b = | f{N-1} |
	//     |    0   |
	//     |    0   |
	//     |    0   |
	double b[DIM];
	for( int i = 0; i < N; i++ )	// Scalar field known values from visited nodes.
		b[i] = rbfFieldPtr[window[i]];
	if( usingPolynomial )
		b[N] = 0;					// Complete the zeros in b.

	// Debugging: printing vector b.
//	if( atNodeIdx == 2859 )
//	{
//		printf( "\n[" );
//		for( int i = 0; i < DIM; i++ )
//			printf( "%.15g, ", b[i] );
//		printf( "]\n" );
//	}

	// Solving Ax = b with LU factorization from LAPACK.
	// https://stackoverflow.com/questions/10112135/understanding-lapack-calls-in-c-with-a-simple-example
	// http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga5ee879032a8365897c3ba91e3dc8d512.html
	int info;
	int ipiv[DIM];
	dgetrf_( &DIM, &DIM, A, &LDA, ipiv, &info );							// Solution is placed in b.
	dgetrs_( &TRANSPOSE, &DIM, &N_RHS, A, &LDA, ipiv, b, &LDB, &info );		// See dgesv_ for column order, one-shot solution.

	assert( info == 0 );

	// Debugging: printing solution vector.
//	printf( "\n[" );
//	for( int i = 0; i < DIM; i++ )
//		printf( "%.15g, ", b[i] );
//	printf( "]\n" );

	// Using computed weights to extrapolate to target node.
	double s = 0;
	for( int i = 0; i < N; i++ )		// RBF contributions.
	{
		double d = Point2::norm_L2( windowPositions[i], p );
		s += b[i] * rbf( d );
	}

	if( usingPolynomial )				// Polynomial contributions.
		s += b[N];
	rbfFieldPtr[atNodeIdx] = s;
}


int main(int argc, char** argv)
{
	const double MIN_D = -1;					// Minimum value for domain (in x and y).  Domain is symmetric.
	const int NUM_TREES_PER_DIM = 2;			// Number of trees per dimension: each with same width and height.
	const int REFINEMENT_MAX_LEVELS[] = { 6 };	// Maximum levels of refinement.
	const int REFINEMENT_BAND_WIDTH = 5;		// Band around interface for grid refinement.
	const int EXTENSION_NUM_ITER = 25;			// Number of iterations to solve PDE for extrapolation.
	const int EXTENSION_ORDER = 2;				// Order of extrapolation (0: constant, 1: linear, 2: quadratic).

	// Prepare parallel enviroment, although we enforce just a single processor for testing.
	mpi_environment_t mpi{};
	mpi.init( argc, argv );
	if( mpi.rank() > 1 )
		throw std::runtime_error( "Only a single process is allowed!" );

	// Stopwatch.
	parStopWatch watch;
	watch.start();

	// Domain information.
	const int n_xyz[] = { NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM };
	const double xyz_min[] = { MIN_D, MIN_D, MIN_D };
	const double xyz_max[] = { -MIN_D, -MIN_D, -MIN_D };
	const int periodic[] = { 0, 0, 0 };

	// Sampling randomness.
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen{}; 						// Standard mersenne_twister_engine seeded with rd().
	std::normal_distribution<double> normalDistribution( 0., sqrt( REFINEMENT_BAND_WIDTH ) );

	// Scalar field to extend (defaults to f(x,y) = sin(πx)cos(πy).
	Field field;

	std::cout << "###### Extending scalar function '" << field.toString() << "' in 2D ######" << std::endl;

	// We iterate over each scalar field and collect samples from it.
	for( const auto& REFINEMENT_MAX_LEVEL : REFINEMENT_MAX_LEVELS )
	{
		const double H = 1. / pow( 2., REFINEMENT_MAX_LEVEL );		// Mesh size.
		std::cout << "## MAX LEVEL OF REFINEMENT = " << REFINEMENT_MAX_LEVEL << ".  H = " << H << " ##" << std::endl;

		// p4est variables.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		p4est_ghost_t *ghost;
		my_p4est_brick_t brick;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining a signed distance level-set function.
		geom::Sphere circle( 0, 0, 0.501 );							// Circle used in Daniel's paper.
		splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, REFINEMENT_MAX_LEVEL, &circle, REFINEMENT_BAND_WIDTH );

		// Error L∞ norm from extrapolation at both methods.
		double rbfMaxError = 0;
		double pdeMaxError = 0;

		// Radial basis function.
		MultiquadricRBF rbf( 70 * 0.32 * H );

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
		nodeNeighbors.init_neighbors();

		// Smallest quadrant features.
		double dxyz[P4EST_DIM]; 						// Dimensions.
		double dxyzMin;									// Minimum side length of the smallest quadrant (i.e. H).
		double diagMin;        							// Diagonal length of the smallest quadrant.
		get_dxyz_min( p4est, dxyz, dxyzMin, diagMin );
		assert( ABS( H - dxyzMin ) < EPS );				// Right mesh size?
		const double RBF_LOCAL_RADIUS = REFINEMENT_BAND_WIDTH * H + EPS;	// When doing RBF extrapolation, we look at this maximum radial distance.
		const double EXTENSION_DISTANCE = 2 * diagMin;	// We are interested in extending and measuring error up to this distance.

		// A ghosted parallel PETSc vector to store level-set function values.
		Vec phi;
		PetscErrorCode ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );

		// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
		sample_cf_on_nodes( p4est, nodes, circle, phi );

		// Level-set object.
		my_p4est_level_set_t levelSet( &nodeNeighbors );
//		levelSet.reinitialize_2nd_order( phi, REINIT_NUM_ITER );

		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Allocating vectors to store the field: rfb extrapolation, pde extrapolation, and exact value.
		Vec rbfField, pdeField, exactField;
		ierr = VecDuplicate( phi, &rbfField );
		CHKERRXX( ierr );
		ierr = VecDuplicate( phi, &pdeField );
		CHKERRXX( ierr );
		ierr = VecDuplicate( phi, &exactField );
		CHKERRXX( ierr );

		// Allocating vectors to store extrapolation errors.
		Vec rbfError, pdeError;
		ierr = VecDuplicate( phi, &rbfError );
		CHKERRXX( ierr );
		ierr = VecDuplicate( phi, &pdeError );
		CHKERRXX( ierr );

		double *rbfErrorPtr, *pdeErrorPtr;
		ierr = VecGetArray( rbfError, &rbfErrorPtr );
		CHKERRXX( ierr );
		ierr = VecGetArray( pdeError, &pdeErrorPtr );
		CHKERRXX( ierr );

		// Multiset datastructure for pending nodes.
		auto nodeComparator = [phiReadPtr](const p4est_locidx_t& lhs, const p4est_locidx_t& rhs) -> bool{
			return phiReadPtr[lhs] < phiReadPtr[rhs];
		};
		std::multiset<p4est_locidx_t, decltype( nodeComparator )> pendingNodesSet( nodeComparator );

		// A vector of booleans to keep track of visited nodes.
		bool visitedNodes[nodes->num_owned_indeps];

		// Evaluate exact field everywhere and copy only the known region (i.e. inside the interface) to the RBF and PDE versions.
		sample_cf_on_nodes( p4est, nodes, field, exactField );
		ierr = VecCopyGhost( exactField, rbfField );
		CHKERRXX( ierr );
		ierr = VecCopyGhost( exactField, pdeField );
		CHKERRXX( ierr );

		double *rbfFieldPtr, *pdeFieldPtr;
		ierr = VecGetArray( rbfField, &rbfFieldPtr );
		CHKERRXX( ierr );
		ierr = VecGetArray( pdeField, &pdeFieldPtr );
		CHKERRXX( ierr );

		const double *exactFieldReadPtr;
		ierr = VecGetArrayRead( exactField, &exactFieldReadPtr );
		CHKERRXX( ierr );

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			visitedNodes[n] = false;

			// For those nodes outside the interface, reset their extrapolation value and add those we are interested in
			// computing their extrapolation to a binary tree for efficient access.
			if( phiReadPtr[n] > 0 )
			{
				pdeFieldPtr[n] = rbfFieldPtr[n] = 0;
				if( phiReadPtr[n] <= EXTENSION_DISTANCE )		// Don't attempt extension beyond it's needed.
					pendingNodesSet.insert( n );
			}
			else if( ABS( phiReadPtr[n] ) <= RBF_LOCAL_RADIUS )
			{
				// Mark those nodes that lie inside the interface within a band of RBF_LOCAL_RADIUS as visited.
				visitedNodes[n] = true;
			}
		}

		// Perform extrapolation using all derivatives (from Daniil's paper).
		levelSet.extend_Over_Interface_TVD_Full( phi, pdeField, EXTENSION_NUM_ITER, EXTENSION_ORDER );

		// Perform extrapolation using a radial basis function network augmented with a polynomial p(x,y) of first degree.
		double nodesStatus[nodes->num_owned_indeps];
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
			nodesStatus[n] = (visitedNodes[n])? -1 : 0;

		double xyz[P4EST_DIM];
		char distCStr[16];
		auto pendingNodeIt = pendingNodesSet.begin();
		while( pendingNodeIt != pendingNodesSet.end() )
		{
			// Pop next closest node to interface.
			p4est_locidx_t pendingNodeIdx = *pendingNodeIt;
			pendingNodesSet.erase( pendingNodeIt );
			node_xyz_fr_n( pendingNodeIdx, p4est, nodes, xyz );
			Point2 p( xyz[0], xyz[1] );

			// Retrieved visited nodes that lie within a given radius from the current node.
			std::map<std::string, std::vector<p4est_locidx_t>> bins;	// Build a histogram by placing the node IDs at each bin.
			int totalPointsInRange = 0;
			for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
			{
				if( visitedNodes[n] )		// We get information only from visited/fixed nodes.
				{
					node_xyz_fr_n( n, p4est, nodes, xyz );
					Point2 q( xyz[0], xyz[1] );
					double d = (p - q).norm_L2();
					if( d <= RBF_LOCAL_RADIUS )				// Within range?  Build histogram.
					{
						sprintf( distCStr, "%.6f", d / H );	// Find the bin according to its multiplicity of H.
						std::string binKey = std::string( distCStr );
						std::vector<p4est_locidx_t>& nodeIds = bins[binKey];
						nodeIds.push_back( n );
						totalPointsInRange++;
					}
				}
			}

			// Place nodal index vectors from each bin in an array for sampling.
			std::vector<std::vector<p4est_locidx_t>*> binsVectors;
			binsVectors.reserve( bins.size() );
			for( auto& bin : bins )
				binsVectors.push_back( &(bin.second) );

			// Use a normal distribution to select samples.
			const int N_SAMPLES_PER_WINDOW = 25;					// Number of samples per interpolation window.
			std::vector<p4est_locidx_t> window;						// Nodal indices.
			window.reserve( N_SAMPLES_PER_WINDOW );
			const double INTERVAL_WIDTH = REFINEMENT_BAND_WIDTH / (double)bins.size();
			int s = 0;
			while( s < N_SAMPLES_PER_WINDOW )
			{
				double r = normalDistribution( gen );
				int idx = MIN( (unsigned long)floor( ABS( r ) / INTERVAL_WIDTH ), bins.size() - 1 );
				if( !binsVectors[idx]->empty() )
				{
					window.push_back( binsVectors[idx]->back() );	// Remove them from the hash map and add them
					binsVectors[idx]->pop_back();					// to the sliding window.
					s++;

					nodesStatus[window.back()] = -2;
				}
			}

			// Extrapolation.
			extrapolate( pendingNodeIdx, window, rbf, p4est, nodes, rbfFieldPtr, true );

			// Post-evaluation tasks.
			nodesStatus[pendingNodeIdx] = +2;
			visitedNodes[pendingNodeIdx] = true;
			pendingNodeIt = pendingNodesSet.begin();				// Next pending node closest to the interface.
		}

		// Storing errors at nodes within a band from the interface.
		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
			if( phiReadPtr[n] > 0 && phiReadPtr[n] <= EXTENSION_DISTANCE )
			{
				rbfErrorPtr[n] = ABS( exactFieldReadPtr[n] - rbfFieldPtr[n] );
				pdeErrorPtr[n] = ABS( exactFieldReadPtr[n] - pdeFieldPtr[n] );

				// Keep track of max error.
				rbfMaxError = MAX( rbfMaxError, rbfErrorPtr[n] );
				pdeMaxError = MAX( pdeMaxError, pdeErrorPtr[n] );
			}
		}

		// Outputting stats.
		printf( "RBF max error: %.8g\n", rbfMaxError );
		printf( "PDE max error: %.8g\n", pdeMaxError );

		std::string vtkOutput = "machineLearningExtrapolation_" + std::to_string( REFINEMENT_MAX_LEVEL );
		my_p4est_vtk_write_all( p4est, nodes, ghost,
						  P4EST_TRUE, P4EST_TRUE,
						  7, 0, vtkOutput.c_str(),
						  VTK_POINT_DATA, "phi", phiReadPtr,
						  VTK_POINT_DATA, "nodes_status", nodesStatus,
						  VTK_POINT_DATA, "f_exact", exactFieldReadPtr,
						  VTK_POINT_DATA, "f_pde", pdeFieldPtr,
						  VTK_POINT_DATA, "f_rbf", rbfFieldPtr,
						  VTK_POINT_DATA, "error_rbf", rbfErrorPtr,
						  VTK_POINT_DATA, "error_pde", pdeErrorPtr );

		// Cleaning up.
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );			// Restoring vector pointers.
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( exactField, &exactFieldReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( rbfField, &rbfFieldPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( pdeField, &pdeFieldPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( rbfError, &rbfErrorPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( pdeError, &pdeErrorPtr );
		CHKERRXX( ierr );

		ierr = VecDestroy( phi );								// Freeing memory.
		CHKERRXX( ierr );
		ierr = VecDestroy( exactField );
		CHKERRXX( ierr );
		ierr = VecDestroy( rbfField );
		CHKERRXX( ierr );
		ierr = VecDestroy( pdeField );
		CHKERRXX( ierr );
		ierr = VecDestroy( rbfError );
		CHKERRXX( ierr );
		ierr = VecDestroy( pdeError );
		CHKERRXX( ierr );

		// Destroy the structures.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );
	}

	std::cout << "## Done in " <<  watch.get_duration_current() << " secs. " << std::endl;
	watch.stop();
}

