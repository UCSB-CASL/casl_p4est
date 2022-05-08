/**
 * Generate samples from a hyperbolic paraboloid surface for *offline inference* (i.e., using Python).  The canonical surface is a Monge
 * patch represented by
 *
 *                                                    Q(u,v) = a*u^2 - b*v^2,
 *
 * where a > 0 and b > 0.  As for the level-set function whose Gamma = Q(u,v) = 0, phi < 0 for points above Q(u,v), and phi > 0 for points
 * below Q(u,v).  We simplify calculations by expressing query points in terms of the canonical frame, which can be affected by rigid-body
 * transformations (i.e., a translation and rotation).
 *
 * Theoretically, the hyperbolic paraboloid Gaussian curvature is always negative (never 0).  Thus, its data set contains samples only
 * saddle regions (i.e., h2kg < 0).  If requested, we can apply negative-mean-curvature normalization selectively for each numerical
 * non-saddle sample (say we found some point for which ih2kg >= nonSaddleMinIH2KG).  In any case, every sample is reoriented by rotating
 * the stencil so that the gradient at the center node has all its components non-negative.  Finally, we reflect the data packet about the
 * y-x = 0 plane, and we produce two samples for each interface point.  At inference time, both outputs are averaged to improve accuracy.
 *
 * The sample file is of the form "#/hyp_paraboloid/$/iter%_data.csv", and the params file is "#/hyp_paraboloid/$/iter%_params.csv", where
 * # is the unit-octree max level of refinement, $ is the experiment id, and % is the number of redistancing steps.  The data file contains
 * as many rows as twice the number of collected samples with all data-packet info.  The params file stores the values for "a", "b", and
 * "hk", where "hk" is the true maximum dimensionless mean curvature.  The latter can occur at two places if 1 <= r < 3, and a=rb or b=ra.
 * If the user picks an r factor that yields maxima whose critical points are less than 1.5h apart, we abort the program as curvature
 * becomes "under-resolved".  In addition, we export VTK data for visualization and validation.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: May 4, 2022.
 * Updated: May 8, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include "hyp_paraboloid_3d.h"
#include <src/parameter_list.h>
#include <cassert>


void writeParamsFile( const mpi_environment_t& mpi, const std::string& path, const std::string& fileName, const double& a, const double& b,
					  const double& hk );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<u_char>      experimentId( pl,     0, "experimentId"	 , "Experiment Id (default: 0)" );
	param_t<double>             maxHK( pl,  0.35, "maxHK"		 	 , "Desired max (absolute) dimensionless mean curvature at the critical"
																	   " points.  Must be in the interval of [1/15, 2/3] (default: 0.35)" );
	param_t<u_char>             maxRL( pl,     6, "maxRL"		 	 , "Max level of refinement per unit-cube octree (default: 6)" );
	param_t<int>          reinitIters( pl,    10, "reinitIters"	 	 , "Number of iterations for reinitialization (default: 10)" );
	param_t<u_short>    minSamRadiusH( pl,    16, "minSamRadiusH"	 , "Min sampling radius in h units on the uv plane.  Must be in the "
																	   "range of [16, 32] (default 16)" );
	param_t<double>                 r( pl,     3, "r"		     	 , "The ratio a/b in the range of [-10, -1) union [1, 10].  If it's "
																	   "negative, then b=|r|a; otherwise, a=rb (default: 3)" );
	param_t<bool>       perturbOrigin( pl,  true, "perturbFrame"	 , "Whether to perturb the surface's frame randomly in [-h/2,+h/2]^3 "
																	   "(default: true)" );
	param_t<bool>      randomRotation( pl,  true, "randomRotation"	 , "Whether to apply a rotation with a random angle about a random unit"
																	   " axis (default: true)" );
	param_t<u_int>        randomState( pl,    11, "randomState"	 	 , "Seed for random perturbations of canonical frame (default: 11)" );
	param_t<std::string>       outDir( pl,   ".", "outDir"		 	 , "Path where files will be written to (default: build folder)" );
	param_t<bool>      useNegCurvNorm( pl,  true, "useNegCurvNorm"	 , "Whether to apply negative-mean-curvature normalization for non-"
																	   "saddle samples (default: true)" );
	param_t<double>       randomNoise( pl,  1e-4, "randomNoise"		 , "How much random noise to add to phi(x) as [+/-]h*randomNoise.  Use "
																	   "a negative value or 0 to disable this feature (default: 1e-4)" );

	try
	{
		///////////////////////////////////////////////////////// Parameter setup //////////////////////////////////////////////////////////

		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating hyperbolic paraboloid data set for offline error-correcting neural network evaluation" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		if( maxHK() < 1./15 || maxHK() > 2./3 )
			throw std::invalid_argument( "[CASL_ERROR] Desired max hk must be in the range of [1/15, 2/3]." );

		if( r() < -10 || (r() >= -1 && r() < 1) || r() > 10 )
			throw std::invalid_argument( "[CASL_ERROR] The ratio a/b must be in the range of [-10, -1) union [1, 10]." );

		if( minSamRadiusH() < 16 || minSamRadiusH() > 32 )
			throw std::invalid_argument( "[CASL_ERROR] Desired sampling radius in h units must be in the range of [16, 32]." );

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.
		std::mt19937 gen( randomState() );					// Engine used for random perturbations.
		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );

		double MAX_K = maxHK() / h;
		double A, B, samRadius;
		HypParaboloid::findParamsAndSamRadius( r(), MAX_K, A, B, samRadius, h, minSamRadiusH() );
		HypParaboloid hypParaboloid( A, B );				// The surface in canonical space.

		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;	// Buffer of accumulated (normalized and augmented) samples.
		double origin[P4EST_DIM] = {0, 0, 0};				// Hyp-paraboloid's frame origin (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};				// Rotation axis for possible random rotation.
		double rotAngle = 0;

		double trackedMinHK = DBL_MAX;						// We want to track min and max mean |hk*| for debugging.
		double trackedMaxHK = 0;
		if( mpi.rank() == 0 )			// Only rank 0 controls the buffer and perturbs hyb-paraboloid's frame to create an affine-
		{								// transformed level-set function.
			buffer.reserve( 1e5 );

			if( perturbOrigin() )
			{
				std::uniform_real_distribution<double> uniformDistributionH_2( -h/2, +h/2 );
				for( auto& dim : origin )
					dim = uniformDistributionH_2( gen );
			}

			if( randomRotation() )		// Generate a random unit axis and its rotation angle.
			{
				std::uniform_real_distribution<double> uniformDist;
				rotAngle = 2 * M_PI * uniformDist( gen );
				double azimuthAngle = uniformDist( gen ) * 2 * M_PI;
				double polarAngle = acos( 2 * uniformDist( gen ) - 1 );
				rotAxis[0] = cos( azimuthAngle ) * sin( polarAngle );
				rotAxis[1] = sin( azimuthAngle ) * sin( polarAngle );
				rotAxis[2] = cos( polarAngle );
			}
		}
		SC_CHECK_MPI( MPI_Bcast( origin, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same random shift and rotation.
		SC_CHECK_MPI( MPI_Bcast( &rotAngle, 1, MPI_DOUBLE, 0, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Bcast( rotAxis, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );

		// Prepping the params.
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() ) + "/hyp_paraboloid/" + std::to_string( experimentId() );
		writeParamsFile( mpi, DATA_PATH, "iter" + std::to_string( reinitIters() ) + "_params.csv", A, B, maxHK() );

		// Prepping the samples file.  Only rank 0 writes the samples to a file.
		std::ofstream file;
		std::string fileName = "iter" + std::to_string( reinitIters() ) + "_data.csv";
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

		///////////////////////////////////////////////////////// Data production //////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began generating hyperbolic paraboloid data set for offline evaluation with a = %g, b = %g, "
								 "max |hk| = %g, and h = %g (level %i)\n", A, B, maxHK(), h, maxRL() );

		parStopWatch watch;
		watch.start();

		// Domain information.
		u_char octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		int periodic[P4EST_DIM] = {0, 0, 0};
		HypParaboloidLevelSet *pLS = setupDomain( mpi, hypParaboloid, h, origin, rotAngle, rotAxis, maxRL(), octMaxRL, samRadius, n_xyz,
												  xyz_min, xyz_max );

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Create the forest using the hyp-paraboloid level-set as a refinement criterion.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, octMaxRL, pLS, 3.0 );
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		pLS->toggleCache( true );		// Turn on cache to speed up repeated distance computations.
		pLS->reserveCache( (size_t)pow( 0.75 * (xyz_max[0] - xyz_min[0]) / h, 3 ) );	// Reserve space in cache to improve hashing.
		for( int i = 0; i < octMaxRL; i++ )												// queries for grid points.
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
		ngbd->init_neighbors();

		// Verify mesh size.
		double dxyz[P4EST_DIM];
		get_dxyz_min( p4est, dxyz );
		assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == h );

		// A ghosted parallel PETSc vector to store level-set function values and where we computed exact signed distances.
		Vec phi = nullptr, exactFlag = nullptr;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactFlag ) );

		// Populate phi and compute exact distance for vertices within a (linearly estimated) shell around Gamma.  Reinitialization perturbs
		// the otherwise calculated exact distances.
		pLS->evaluate( p4est, nodes, phi, exactFlag, 5 );

		// Add random noise if requested.
		if( randomNoise() > 0 )
			addRandomNoiseToLSFunction( phi, nodes, genNoise, randomNoiseDist );

		my_p4est_level_set_t ls( ngbd );
		ls.reinitialize_2nd_order( phi, reinitIters() );

		// Collect samples.
		Vec sampledFlag;					// This vector distinguishes sampled nodes along the interface.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );

		Vec hkError, ihk, h2kgError, ih2kg;	// Vectors with sampled errors and interpolated dimensionless curvatures at the interface.
		Vec phiError;						// Phi error only for sampled interface nodes.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hkError ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &ihk ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &h2kgError ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &ih2kg ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phiError ) );

		std::vector<std::vector<double>> samples;
		double trackedMaxErrors[P4EST_DIM];
		int nNumericalSaddles;
		pLS->collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, trackedMaxErrors, trackedMinHK, trackedMaxHK, samples,
							 nNumericalSaddles, exactFlag, sampledFlag, hkError, ihk, h2kgError, ih2kg, phiError,
							 SQR(samRadius), SQR(samRadius), nonSaddleMinIH2KG() );
		pLS->clearCache();
		pLS->toggleCache( false );
		delete pLS;

		// Accumulate samples in the buffer; normalize phi by h, apply negative-mean-curvature normalization to non-saddle samples only if
		// requested, but always reorient data packets.  Also, augment samples by reflecting about plane y - x = 0.
		int bufferSize = kml::utils::processSamplesAndAccumulate( mpi, samples, buffer, h, useNegCurvNorm()? 2 : 0, nonSaddleMinIH2KG() );
		int nSamples = saveSamples( mpi, buffer, bufferSize, file );

		// Export visual data.
		const double *phiReadPtr, *phiErrorReadPtr, *sampledFlagReadPtr, *exactFlagReadPtr;
		const double *hkErrorReadPtr, *ihkReadPtr, *h2kgErrorReadPtr, *ih2kgReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( hkError, &hkErrorReadPtr ) );
		CHKERRXX( VecGetArrayRead( ihk, &ihkReadPtr ) );
		CHKERRXX( VecGetArrayRead( h2kgError, &h2kgErrorReadPtr ) );
		CHKERRXX( VecGetArrayRead( ih2kg, &ih2kgReadPtr ) );
		CHKERRXX( VecGetArrayRead( phiError, &phiErrorReadPtr ) );
		CHKERRXX( VecGetArrayRead( exactFlag, &exactFlagReadPtr ) );

		std::ostringstream oss;
		oss << "hyp_paraboloid_dataset_id" << (int)experimentId() << "_lvl" << (int)maxRL();
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								8, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "sampledFlag", sampledFlagReadPtr,
								VTK_POINT_DATA, "hkError", hkErrorReadPtr,
								VTK_POINT_DATA, "ihk", ihkReadPtr,
								VTK_POINT_DATA, "h2kgError", h2kgErrorReadPtr,
								VTK_POINT_DATA, "ih2kg", ih2kgReadPtr,
								VTK_POINT_DATA, "phiError", phiErrorReadPtr,
								VTK_POINT_DATA, "exactFlag", exactFlagReadPtr );

		CHKERRXX( VecRestoreArrayRead( exactFlag, &exactFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( phiError, &phiErrorReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( ih2kg, &ih2kgReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( h2kgError, &h2kgErrorReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( ihk, &ihkReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( hkError, &hkErrorReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

		// Clean up.
		CHKERRXX( VecDestroy( phiError ) );
		CHKERRXX( VecDestroy( ih2kg ) );
		CHKERRXX( VecDestroy( h2kgError ) );
		CHKERRXX( VecDestroy( ihk ) );
		CHKERRXX( VecDestroy( hkError ) );
		CHKERRXX( VecDestroy( sampledFlag ) );
		CHKERRXX( VecDestroy( exactFlag ) );
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		delete ngbd;
		delete hierarchy;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );

		CHKERRXX( PetscPrintf( mpi.comm(), "   Collected and saved %i samples (incl. standard and reflected) with the following stats:\n", nSamples ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of saddle points   = %i (or %i samples)\n", nNumericalSaddles, 2 * nNumericalSaddles ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Tracked mean |hk*| in the range of [%.6g, %.6g]\n", trackedMinHK, trackedMaxHK ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Tracked max hk error      = %.6g\n", trackedMaxErrors[0] ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Tracked max h^2kg error   = %.6g\n", trackedMaxErrors[1] ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Tracked max |phi error|/h = %.6g\n", trackedMaxErrors[2] / h ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "<< Finished after %.2f secs.\n", watch.get_duration_current() ) );
		watch.stop();
		if( mpi.rank() == 0 )
			file.close();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}

/**
 * Create and write the hyp-paraboloid's params file (for Q(u,v) = a*u^2 - b*v^2).
 * @param [in] mpi MPI environment.
 * @param [in] path Full path where to place the file.
 * @param [in] fileName File name.
 * @param [in] a The a parameter along the u direction.
 * @param [in] b The b parameter along the v direction.
 * @param [in] hk Highest absolute dimensionless mean curvature attained at critical point(s).
 * @throws runtime_error if path or file can't be created and or written.
 */
void writeParamsFile( const mpi_environment_t& mpi, const std::string& path, const std::string& fileName, const double& a, const double& b,
					  const double& hk )
{
	std::string errorPrefix = "[CASL_ERROR] writeParamsFile: ";
	std::string fullFileName = path + "/" + fileName;

	if( create_directory( path, mpi.rank(), mpi.comm() ) )
		throw std::runtime_error( errorPrefix + "Couldn't create directory: " + path );

	if( mpi.rank() == 0 )
	{
		std::ofstream file;
		file.open( fullFileName, std::ofstream::trunc );
		if( !file.is_open() )
			throw std::runtime_error( errorPrefix + "Output file " + fullFileName + " couldn't be opened!" );

		file << R"("a","b","hk")" << std::endl;	// The header.
		file.precision( 15 );
		file << a << "," << b << "," << hk << std::endl;
		file.close();
	}

	CHKERRXX( PetscPrintf( mpi.comm(), "Rank %d successfully created params file '%s'\n", mpi.rank(), fullFileName.c_str() ) );
	SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );				// Wait here until rank 0 is done.
}
