/**
 * Title: machine_learning_extrapolation
 * Description:
 * Author: Luis Ángel (임 영민)
 * Date Created: 09-29-2020
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
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
			case 0: return cos( M_PI * x ) * sin( M_PI * y );
			case 1: return cos( M_PI * x ) + sin( M_PI * y );
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
			case 0: return "cos( M_PI * x ) * sin( M_PI * y )";
			case 1: return "cos( M_PI * x ) + sin( M_PI * y )";
			default: throw std::invalid_argument( "Invalid scalar function choice!" );
		}
	}
};

int main(int argc, char** argv)
{
	const double MIN_D = -1;				// Minimum value for domain (in x, y, and z).  Domain is symmetric.
	const int NUM_TREES_PER_DIM = 2;		// Number of trees per dimension: each with same width and height.
	const int REFINEMENT_MAX_LEVEL = 6;		// Maximum level of refinement.
	const double H = 1 / pow( 2, REFINEMENT_MAX_LEVEL );	// Minimum cell width.
	const int REFINEMENT_BAND_WIDTH = 5;	// Band around interface for grid refinement.
	const int REINIT_NUM_ITER = 10;			// Number of iterations to solve PDE for reinitialization.
	const int EXTENSION_NUM_ITER = 25;		// Number of iterations to solve PDE for extrapolation.
	const int EXTENSION_ORDER = 2;			// Order of extrapolation (0: constant, 1: linear, 2: quadratic).
	const int SAMPLING_BAND_WIDTH = 4;		// Number of grid points to sample in Omega- and along Gamma for learning.
	const int SAMPLING_MIN_P_BAND = 5;		// When sampling, a query point P is considered, and this has to lie in this
	const int SAMPLING_MAX_P_BAND = 8;		// range (in multiples of H) into Omega+.

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

	//////////////////////////////////////// Preparing files to save data sets /////////////////////////////////////////

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/extrapolation/data/";						// Destination folder.
	const int NUM_COLUMNS = (SAMPLING_BAND_WIDTH + 1) * (2 * (SAMPLING_BAND_WIDTH + 1) - 1) + 3;	// Number of columns.
	std::vector<std::string> COLUMN_NAMES;	// Column headers following the x-y truth table with x changing slower than y.
	COLUMN_NAMES.reserve( NUM_COLUMNS );

	// Columns format:
	// "x-4y-4", "x-4y-3", ..., "x-4y+3", "x-4y+4",    These columns contain the interpolated scalar field on the the
	// "x-3y-4", "x-3y-3", ..., "x-3y+3", "x-3y+4",    local grid lying on the negative x-axis.
	//     :         :              :         :
	//  "x0y-4",  "x0y-3", ...,  "x0y+3",  "x0y+4",
	//     :         :              :         :
	// "x+3y-4", "x+3y-3", ..., "x+3y+3", "x+3y+4",
	// "x+4y-4", "x+4y-3", ..., "x+4y+3", "x+4y+4",
	// "dh",											This is the distance (in Hs) along the local positive x-axis.
	// "f",												Target scalar field value at query point.
	// "ief"											Numerical approximation of extended field value at query point.
	for( int i = -SAMPLING_BAND_WIDTH; i <= 0; i++ )
	{
		for( int j = -SAMPLING_BAND_WIDTH; j <= SAMPLING_BAND_WIDTH; j++ )
		{
			COLUMN_NAMES.emplace_back( "x" + std::string( i > 0? "+" : "" ) + std::to_string( i ) +
									   "y" + std::string( j > 0? "+" : "" ) + std::to_string( j ) );
		}
	}
	COLUMN_NAMES.emplace_back( "dh" );
	COLUMN_NAMES.emplace_back( "f" );
	COLUMN_NAMES.emplace_back( "ief" );

	printf( ">> Began to generate datasets for planar level-set with max refinement level of %i and finest h = %.8f\n",
			REFINEMENT_MAX_LEVEL, H );

	// Prepare samples file: f_X.csv.
	std::ofstream fFile;
	std::string fFileName = DATA_PATH + "f_" + std::to_string( REFINEMENT_MAX_LEVEL ) +  ".csv";
	fFile.open( fFileName, std::ofstream::trunc );
	if( !fFile.is_open() )
		throw std::runtime_error( "Output file '" + fFileName + "' couldn't be opened!" );

	// Write column headers: enforcing strings by adding quotes around them.
	std::ostringstream headerStream;
	for( int i = 0; i < NUM_COLUMNS - 1; i++ )
		headerStream << "\"" << COLUMN_NAMES[i] << "\",";
	headerStream << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"";
	fFile << headerStream.str() << std::endl;

	fFile.precision( 15 );					// Precision for floating point numbers.

	/////////////////////////////////////////////// Generating data sets ///////////////////////////////////////////////

	// Domain information.
	const int n_xyz[] = { NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM };
	const double xyz_min[] = { MIN_D, MIN_D, MIN_D };
	const double xyz_max[] = { -MIN_D, -MIN_D, -MIN_D };
	const int periodic[] = { 0, 0, 0 };

	const double THETA = 0;						// Plane tilt.  In a broader application, to vary between [-pi/2, pi/2).
	const double R_THETA = M_PI_2 + THETA;		// Defines the rotation of the local coordinate system for plane.
	const Point2 N( cos( R_THETA ), sin( R_THETA ) );	// Plane normal vector in world coordinates.
	const int NUM_RANDOM_CENTERS = 10;			// Number of random plane 'centers' to be processed for current tilt.

	// Basically, we want X random query points per H^2, lying in a band in Omega+.
	const double TRUE_MIN_D = MIN_D + SAMPLING_BAND_WIDTH * H;	// Min bound where points more likely have full stencils.
	const int NUM_RANDOM_SAMPLES_PER_PLANE = (int)(2 * ABS( TRUE_MIN_D ) / H) *
		(SAMPLING_MAX_P_BAND - SAMPLING_MIN_P_BAND) * 10;

	// Scalar field to extend.
	Field field;
	int fieldChoices[] = { 0, 1 };

	int totalSamples = 0;

	// We iterate over each scalar field and collect samples from it.
	for( const auto& fieldChoice : fieldChoices )
	{
		field.setChoice( fieldChoice );
		std::cout << "   # Now collecting samples for scalar field '" << field.toString() << "':" << std::endl;

		// For current plane tilt, generate NUM_RANDOM_CENTERS variations of plane locations or displacements randomly.
		for( int nc = 0; nc < NUM_RANDOM_CENTERS; nc++ )
		{
			// p4est variables.
			p4est_t *p4est;
			p4est_nodes_t *nodes;
			p4est_ghost_t *ghost;
			my_p4est_brick_t brick;
			p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

			// Definining the non-signed distance level-set function to be reinitialized.
			Point2 C( -H / 2 + H * uniformDistribution( gen ),		// A point in the plane defining the center of local
					  -H / 2 + H * uniformDistribution( gen ) );	// coordinate in some random location within a square
					  												// of side length H, whose center is the global
					  												// origin.
			geom::Plane plane( N, C );
			splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, REFINEMENT_MAX_LEVEL, &plane, REFINEMENT_BAND_WIDTH );

			// Running max error from interpolation.
			double maxRelError = 0;

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
			double dxyz[P4EST_DIM]; 			// Dimensions.
			double dxyzMin;						// Minimum side length of the smallest quadrant (i.e. H).
			double diagMin;        				// Diagonal length of the smallest quadrant.
			get_dxyz_min( p4est, dxyz, dxyzMin, diagMin );
			assert( ABS( H - dxyzMin ) < EPS );	// Verifying we are dealing with the right domain spacing.

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

			// Vector to store the scalar function numerically extrapolated.
			Vec extField, exactField;
			ierr = VecDuplicate( phi, &extField );
			CHKERRXX( ierr );
			ierr = VecDuplicate( phi, &exactField );
			CHKERRXX( ierr );

			sample_cf_on_nodes( p4est, nodes, field, exactField );	// Field is now evaluated exactly at each node.
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

			// Prepare bilinear interpolation of scalar field (which is exact in Omega- and extrapolated in Omega+).
			my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
			interpolation.set_input( extField, linear );

			// When we created the plane level-set, we defined a local coordinate system centered at the point C and
			// with a rotation angle R_THETA.  This creates a coordinate system where the plane's normal vector
			// coincides with the local coordinate system x-axis.
			// Next, we need to collect points that make up samples in the learning process.  For this, consider:
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

			// Collecting samples for standing plane with current tilt and center.
			int ns = 0;
			std::vector<std::vector<double>> samples;
			while( ns < NUM_RANDOM_SAMPLES_PER_PLANE )
			{
				// Random point generated in local coordinates w.r.t. plane.
				double X_H = SAMPLING_MIN_P_BAND + (SAMPLING_MAX_P_BAND - SAMPLING_MIN_P_BAND) * uniformDistribution( gen );
				Point2 p_c( H * X_H, TRUE_MIN_D + 2 * ABS( TRUE_MIN_D ) * uniformDistribution( gen ) );
				Point2 p_w( cos( R_THETA ) * p_c.x - sin( R_THETA ) * p_c.y + C.x,	// Query point in world coordinates.
							sin( R_THETA ) * p_c.x + cos( R_THETA ) * p_c.y + C.y );

				if( p_w.x < MIN_D || p_w.x > -MIN_D || p_w.y < MIN_D || p_w.y > -MIN_D )
					continue;									// Check query point is within boundaries.

				Point2 projP_c( 0, p_c.y );						// This is the 'o' reference point in the above diagram.
//				Point2 projP_w( cos( R_THETA ) * projP_c.x - sin( R_THETA ) * projP_c.y + C.x,		// Reference 'o' point
//				   				sin( R_THETA ) * projP_c.x + cos( R_THETA ) * projP_c.y + C.y );	// in world coordinates.
				const double STENCIL_WIDTH = SAMPLING_BAND_WIDTH * H;
				Point2 corners_c[4] = {							// The four corners above determine if we have a well
					projP_c + Point2( 0, -STENCIL_WIDTH ),		// defined stencil for sampling.
					projP_c + Point2( 0, +STENCIL_WIDTH ),
					projP_c + Point2( -STENCIL_WIDTH, -STENCIL_WIDTH ),
					projP_c + Point2( -STENCIL_WIDTH, +STENCIL_WIDTH )
				};

				bool validStencil = true;
				for( const auto& corner_c : corners_c )			// Check that the stencil is fully contained in Omega.
				{
					double x_w = cos( R_THETA ) * corner_c.x - sin( R_THETA ) * corner_c.y + C.x;	// Get corner's world
					double y_w = sin( R_THETA ) * corner_c.x + cos( R_THETA ) * corner_c.y + C.y;	// coordinates.
					if( x_w < MIN_D || -MIN_D < x_w || y_w < MIN_D || -MIN_D < y_w )
					{
						validStencil = false;
						break;
					}
				}

				if( !validStencil )		// Invalid stencil?  Try again.
					continue;

				// Collect sample for query point p by using interpolation of exact field values at the nodes in Omega-,
				// and of the extended field values values at the immediately adjacent nodes to Gamma in Omega+.
				// Features go from k0 to k3 above, where x varies slower than y.
				std::vector<double> sample;
				sample.reserve( (SAMPLING_BAND_WIDTH + 1) * (2 * (SAMPLING_BAND_WIDTH + 1) - 1) + 3 );
//				std::cout << "plot(" << p_w.x << ", " << p_w.y << ", 'm.', "
//					  << projP_w.x << ", " << projP_w.y << ", 'c.');" << std::endl;
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

						// Stats.
						double relError = ABS( (sample.back() - field( x_w, y_w )) / field( x_w, y_w ) );
						maxRelError = MAX( maxRelError, relError );
//					std::cout << "plot(" << x_w << "," << y_w << ", 'b.'); " << relError << ";" << std::endl;
					}
				}

				// Completing the sample.
				sample.push_back( X_H );				// Value in range [SAMPLING_MIN_P_BAND, SAMPLING_MAX_P_BAND].
				sample.push_back( field( p_w.x, p_w.y ) );			// Target extrapolation value.
				sample.push_back( interpolation( p_w.x, p_w.y ) );	// Numerically interpolating the extrapolated field.

				samples.push_back( std::move( sample ) );

				ns++;
			}

			// Cleaning up.
			ierr = VecRestoreArrayRead( phi, &phiReadPtr );			// Restoring vector pointers.
			CHKERRXX( ierr );
			ierr = VecRestoreArray( extField, &extFieldPtr );
			CHKERRXX( ierr );

			ierr = VecDestroy( phi );								// Freeing memory.
			CHKERRXX( ierr );
			ierr = VecDestroy( exactField );
			CHKERRXX( ierr );
			ierr = VecDestroy( extField );
			CHKERRXX( ierr );

			// Destroy the structures.
			p4est_nodes_destroy( nodes );
			p4est_ghost_destroy( ghost );
			p4est_destroy( p4est );
			my_p4est_brick_destroy( connectivity, &brick );

			// Write all samples collected.
			for( const auto& row : samples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( fFile, "," ) );	// Inner elements.
				fFile << row.back() << std::endl;
			}

			std::cout << "     [" << nc << "] Done with plane centered at (" << C.x << ", " << C.y << ")"
					  << ".  Maximum relative error = " << maxRelError
					  << ".  Samples = " << samples.size()
					  << ".  Timing = " << watch.get_duration_current() << std::endl;

			totalSamples += samples.size();
		}
	}

	printf( "<< Finished generating %i samples in %f secs.\n", totalSamples, watch.get_duration_current() );
	watch.stop();
	fFile.close();
}

