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

#include <src/casl_geometry.h>
#include <set>

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
 * Multiquadric RBF given by rbf(r) = sqrt( r^2 + a^2 ), where r >= 0, and a > 0 is shape parameter.
 */
class MultiquadricRBF
{
private:
	double _a;		// Shape parameter.

public:
	/**
	 * Constructor.
	 * @param [in] a Positive shape parameter.
	 */
	explicit MultiquadricRBF( double a = 0.001 )
	{
		assert( a > 0 );
		_a = a;
	}

	/**
	 * Evaluate radial basis function.
	 * @param [in] r Distance parameter.
	 * @return rbf(r).
	 */
	[[nodiscard]] double operator()( double r ) const
	{
		return sqrt( SQR( r ) + SQR( _a ) );
	}
};


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

	// Scalar field to extend (defaults to f(x,y) = sin(pi*x) * cos(pi*x)).
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

		// Multiset datastructures for visited and pending nodes.
		auto nodeComparator = [phiReadPtr](const p4est_locidx_t& lhs, const p4est_locidx_t& rhs) -> bool{
			return phiReadPtr[lhs] < phiReadPtr[rhs];
		};
		std::multiset<p4est_locidx_t, decltype( nodeComparator )> visitedNodesSet( nodeComparator );
		std::multiset<p4est_locidx_t, decltype( nodeComparator )> pendingNodesSet( nodeComparator );

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

		for( p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++ )
		{
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
				// Adding those nodes that lie inside the interface within a band of RBF_LOCAL_RADIUS to efficient set
				// of known/fixed nodes.
				visitedNodesSet.insert( n );
			}
		}

		// Perform extrapolation using all derivatives (from Daniil's paper).
		levelSet.extend_Over_Interface_TVD_Full( phi, pdeField, EXTENSION_NUM_ITER, EXTENSION_ORDER );

		// Testing that nodes are sorted by their distance to the interface using the multiset above.
		double nodesStatus[nodes->num_owned_indeps];
		for( auto& n : nodesStatus )
			n = 0;

		std::cout << "** Visited nodes **" << std::endl;
		for( const auto& n : visitedNodesSet )
		{
			std::cout << "[" << n << "] " << phiReadPtr[n] << std::endl;
			nodesStatus[n] = -1;
		}

		std::cout<< "** Pending nodes **" << std::endl;
		for( const auto& n : pendingNodesSet )
		{
			std::cout << "[" << n << "] " << phiReadPtr[n] << std::endl;
			nodesStatus[n] = +1;
		}

		const double *exactFieldReadPtr;
		ierr = VecGetArrayRead( exactField, &exactFieldReadPtr );
		CHKERRXX( ierr );

		std::string vtkOutput = "machineLearningExtrapolation_" + std::to_string( REFINEMENT_MAX_LEVEL );
		my_p4est_vtk_write_all( p4est, nodes, ghost,
						  P4EST_TRUE, P4EST_TRUE,
						  5, 0, vtkOutput.c_str(),
						  VTK_POINT_DATA, "phi", phiReadPtr,
						  VTK_POINT_DATA, "nodes_status", nodesStatus,
						  VTK_POINT_DATA, "f_exact", exactFieldReadPtr,
						  VTK_POINT_DATA, "f_pde", pdeFieldPtr,
						  VTK_POINT_DATA, "f_rbf", rbfFieldPtr );

		// Cleaning up.
		ierr = VecRestoreArrayRead( phi, &phiReadPtr );			// Restoring vector pointers.
		CHKERRXX( ierr );
		ierr = VecRestoreArrayRead( exactField, &exactFieldReadPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( rbfField, &rbfFieldPtr );
		CHKERRXX( ierr );
		ierr = VecRestoreArray( pdeField, &pdeFieldPtr );
		CHKERRXX( ierr );

		ierr = VecDestroy( phi );								// Freeing memory.
		CHKERRXX( ierr );
		ierr = VecDestroy( exactField );
		CHKERRXX( ierr );
		ierr = VecDestroy( rbfField );
		CHKERRXX( ierr );
		ierr = VecDestroy( pdeField );
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

