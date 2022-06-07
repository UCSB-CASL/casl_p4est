/**
 * Generate samples from an ellipsoid for *offline inference* (i.e., using Python).  The canonical surface is represented implicitly by
 *
 *                                           phi(x,y,z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1,
 *
 * where phi<0 for points inside the ellipsoid, phi>0 for points outside, and phi=0 for Gamma.  We simplify calculations by expressing
 * query points in terms of the canonical frame, which can be affected by a rigid transformation (i.e., translation and rotation).
 *
 * Theoretically, the Gaussian curvature should be positive everywhere on the surface, but because of numerical inaccuracies, we can get
 * negative Gaussian curvature samples.  In any case, if a point belongs to non-saddle region (i.e., ih2kg >= C), it'll be negative-mean-
 * curvature normalized by taking the sign of ihk ONLY if requested.  On the other hand, if the point belongs to a (numerical) saddle
 * region, its sample won't be negative-mean-curvature normalized.  Here, we extract six samples per interface node by applying a sequence
 * of reorientations and reflections.  These make the center node' gradient components non-negative.  At inference time, all six outputs are
 * averaged to improve accuracy.
 *
 * Negative-mean-curvature normalization depends on the sign of the linearly interpolated mean ihk at the interface.  As for the Gaussian
 * curvature, we normalize it by scaling it with h^2 ---which leads to the true h2kg and the linearly interpolated ih2kg values in the
 * collected data packets.
 *
 * The sample file is of the form "#/ellipsoid/$/iter%_data.csv", and the params file is "#/ellipsoid/$/iter%_params.csv", where # is the
 * unit-octree max level of refinement, $ is the experiment id, and % is the number of redistancing steps.  The data file contains as many
 * rows as six times the number of interface points with all data-packet info.  The params file stores the values for "a", "b", and "c" and
 * its corresponding max hk values.  In addition, we export VTK data for visualization.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: April 5, 2022.
 * Updated: June 7, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include "ellipsoid_3d.h"
#include <src/parameter_list.h>
#include <src/my_p8est_vtk.h>


void writeParamsFile( const mpi_environment_t& mpi, const std::string& path, const std::string& fileName,
					  const double& a, const double& b, const double& c, const double& hka, const double& hkb, const double& hkc );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<u_short>     experimentId( pl,     0, "experimentId"	 , "Experiment Id (default: 0)" );
	param_t<bool>   useExactSignedLSF( pl,  true, "useExactSignedLSF", "Whether to use an exact signed level-set function to populate the "
																	   "nodal phi vector (default: true" );
	param_t<u_short>            maxRL( pl,     6, "maxRL"			 , "Maximum level of refinement per unit-cube octree (default: 6)" );
	param_t<u_short>      reinitIters( pl,    10, "reinitIters"		 , "Number of iterations for reinitialization (default: 10)" );
	param_t<double>             maxHK( pl,  2./3, "maxHK"			 , "Expected maximum dimensionless mean curvature (default: 2/3)" );
	param_t<double>                 a( pl,  1.65, "a"				 , "Ellipsoid's x-semiaxis (default: 1.65)" );
	param_t<double>                 b( pl,  0.75, "b"				 , "Ellipsoid's y-semiaxis (default: 0.75)" );
	param_t<double>                 c( pl,   0.2, "c"				 , "Ellipsoid's z-semiaxis (default: 0.2)" );
	param_t<bool>       perturbCenter( pl,  true, "perturbCenter"	 , "Whether to perturb the ellipsoid's center randomly in [-h/2,+h/2]^3"
															   		   " (default: true)" );
	param_t<bool>      randomRotation( pl,  true, "randomRotation"	 , "Whether to apply a rotation with a random angle about a random unit"
																	   " axis (default: true)" );
	param_t<u_int>        randomState( pl,    11, "randomState"		 , "Seed for random perturbations of the canonical frame (default: 11)" );
	param_t<std::string>       outDir( pl,   ".", "outDir"			 , "Path where data files will be written to (default: build folder)" );
	param_t<bool>      useNegCurvNorm( pl,  true, "useNegCurvNorm"	 , "Whether we want to apply negative-mean-curvature normalization for "
															   		   "numerical non-saddle samples (default: true)" );
	param_t<double>       randomNoise( pl,  1e-4, "randomNoise"		 , "How much random noise to add to phi(x) as [+/-]h*randomNoise.  Use "
																	   "a negative value or 0 to disable this feature (default: 1e-4)" );

	try
	{
		////////////////////////////////////////////////////////// Parameter setup /////////////////////////////////////////////////////////

		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating ellipsoidal data set for offline evaluation of a trained error-correcting neural network" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		const double h = 1. / (1 << maxRL());					// Highest spatial resolution in x/y directions.

		if( a() < 1.5 * h || b() < 1.5 * h || c() < 1.5 * h )
			throw std::invalid_argument( "[CASL_ERROR] Any semiaxis must be larger than 1.5h" );

		std::mt19937 gen( randomState() );						// Engine used for random perturbations of canonical frame: rotation and shift.
		std::mt19937 genNoise( mpi.rank() );					// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );

		Ellipsoid ellipsoid( a(), b(), c() );					// An ellipsoid implicit function in canonical coords (i.e., untransformed).
		double maxK[P4EST_DIM];
		ellipsoid.getMaxMeanCurvatures( maxK );
		for( const auto& k : maxK )
		{
			if( h * k > maxHK() )
				throw std::invalid_argument( "[CASL_ERROR] One of the ellipsoid's max hk exceeds the expected max hk." );
		}

		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;		// Buffer of accumulated (normalized and augmented) samples.
		double center[P4EST_DIM] = {0, 0, 0};					// Ellipsoidal center (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};					// Rotation axis for possible random rotation.
		double rotAngle = 0;
		double trackedMinHK = DBL_MAX;							// We want to track min and max mean |hk*| for debugging.
		double trackedMaxHK = 0;
		if( mpi.rank() == 0 )			// Only rank 0 controls the buffer and perturbs the ellipsoid's center to create an affine-trans-
		{								// formed level-set function.
			buffer.reserve( 1e5 );

			if( perturbCenter() )		// Should we apply a random shift to ellipsoid's canonical frame?
			{
				std::uniform_real_distribution<double> uniformDistributionH_2( -h/2, +h/2 );
				for( auto& dim : center )
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
		SC_CHECK_MPI( MPI_Bcast( center, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same random shift.
		SC_CHECK_MPI( MPI_Bcast( &rotAngle, 1, MPI_DOUBLE, 0, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Bcast( rotAxis, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );

		const double maxHK_a = maxK[0] * h;
		const double maxHK_b = maxK[1] * h;
		const double maxHK_c = maxK[2] * h;

		// Prepping the params.
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() ) + "/ellipsoid/" + std::to_string( experimentId() );
		writeParamsFile( mpi, DATA_PATH, "iter" + std::to_string( reinitIters() ) + "_params.csv", a(), b(), c(), maxHK_a, maxHK_b, maxHK_c );

		// Prepping the samples file.  Only rank 0 writes the samples to a file.
		std::ofstream file;
		std::string fileName = "iter" + std::to_string( reinitIters() ) + "_data.csv";
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

		///////////////////////////////////////////////////////// Data production //////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began generating ellipsoid data set for offline evaluation with hk_a = %g (a = %g), hk_b = %g (b = %g),"
								 " hk_c = %g (c = %g), and h = %g (level %i)\n", maxHK_a, a(), maxHK_b, b(), maxHK_c, c(), h, maxRL() );

		parStopWatch watch;
		watch.start();

		// Domain information.
		u_short octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		int periodic[P4EST_DIM] = {0, 0, 0};
		setupDomain( mpi, center, a(), b(), c(), h, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max );

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Define a level-set function with ellipsoidal interface.
		EllipsoidalLevelSet levelSet( &mpi, Point3( center ), Point3( rotAxis ), rotAngle, &ellipsoid, h );
		splitting_criteria_cf_and_uniform_band_t splittingCriterion( 0, octMaxRL, &levelSet, 3.0 );

		// Enable and reserve space for caching.
		levelSet.toggleCache( true );
		auto discreteVolumeDiff = (size_t)(3. * 4./3 * M_PI * ((a() + 2*h)*(b() + 2*h)*(c() + 2*h) - (a() - 2*h)*(b() - 2*h)*(c() - 2*h))
			/ CUBE( h ) / mpi.size());
		levelSet.reserveCache( discreteVolumeDiff );

		// Create the forest using ellipsoidal level-set function as a refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &splittingCriterion );

		// Refine and partition forest.
		for( int i = 0; i < octMaxRL; i++ )
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		my_p4est_ghost_expand( p4est, ghost );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
		ngbd->init_neighbors();

		// Verify mesh size.
		double dxyz[P4EST_DIM];
		get_dxyz_min( p4est, dxyz );
		assert( dxyz[0] == dxyz[1] && dxyz[1] == dxyz[2] && dxyz[2] == h );

		// Ghosted parallel PETSc vector to store level-set values.
		Vec phi;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

		// Evaluate level-set function, reinitialize it, and add noise if requested.
		sample_cf_on_nodes( p4est, nodes, levelSet, phi );	// This computes exact-signed distances and populates nearest points (needed to know true hk).
		if( !useExactSignedLSF() )
			levelSet.evaluateNS( p4est, nodes, phi );		// This overwrites exact signed distances found before.

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
		levelSet.collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, trackedMaxErrors, trackedMinHK, trackedMaxHK, samples,
								 nNumericalSaddles, sampledFlag, hkError, ihk, h2kgError, ih2kg, phiError, nonSaddleMinIH2KG() );
		levelSet.clearCache();
		levelSet.toggleCache( false );

		// Accumulate samples in the buffer; normalize phi by h, apply negative-mean-curvature normalization to non-saddle samples only if
		// requested, and reorient data packets.  The last 2 parameter avoids flipping the signs of hk, ihk, h2kg, and ih2kg.  Then, augment
		// samples by reflecting about y - x = 0.
		int bufferSize = kml::utils::processSamplesAndAccumulate( mpi, samples, buffer, h, useNegCurvNorm()? 2 : 0, nonSaddleMinIH2KG() );
		int nSamples = saveSamples( mpi, buffer, bufferSize, file );

		// Export visual data.
		const double *phiReadPtr, *phiErrorReadPtr, *sampledFlagReadPtr;
		const double *hkErrorReadPtr, *ihkReadPtr, *h2kgErrorReadPtr, *ih2kgReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( hkError, &hkErrorReadPtr ) );
		CHKERRXX( VecGetArrayRead( ihk, &ihkReadPtr ) );
		CHKERRXX( VecGetArrayRead( h2kgError, &h2kgErrorReadPtr ) );
		CHKERRXX( VecGetArrayRead( ih2kg, &ih2kgReadPtr ) );
		CHKERRXX( VecGetArrayRead( phiError, &phiErrorReadPtr ) );

		std::ostringstream oss;
		oss << "ellipsoid_dataset_id" << (int)experimentId() << "_lvl" << (int)maxRL();
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								7, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "sampledFlag", sampledFlagReadPtr,
								VTK_POINT_DATA, "hkError", hkErrorReadPtr,
								VTK_POINT_DATA, "ihk", ihkReadPtr,
								VTK_POINT_DATA, "h2kgError", h2kgErrorReadPtr,
								VTK_POINT_DATA, "ih2kg", ih2kgReadPtr,
								VTK_POINT_DATA, "phiError", phiErrorReadPtr );

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
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		delete ngbd;
		delete hierarchy;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );

		CHKERRXX( PetscPrintf( mpi.comm(), "   Collected and saved %i samples (incl. the six permutations) with the following stats:\n", nSamples ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of saddle points   = %i\n", nNumericalSaddles ) );
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
 * Create and write the ellipsoid's params file.
 * @param [in] mpi MPI environment.
 * @param [in] path Full path where to place the file.
 * @param [in] fileName File name.
 * @param [in] a Ellipsoid's x-semiaxis.
 * @param [in] b Ellipsoid's y-semiaxis.
 * @param [in] c Ellipsoid's z-semiaxis.
 * @param [in] hka Expected dimensionless mean curvature at the x-intercepts.
 * @param [in] hkb Expected dimensionless mean curvature at the y-intercepts.
 * @param [in] hkc Expected dimensionless mean curvature at the z-intercepts.
 * @throws runtime_error if path or file can't be created and or written.
 */
void writeParamsFile( const mpi_environment_t& mpi, const std::string& path, const std::string& fileName,
					  const double& a, const double& b, const double& c, const double& hka, const double& hkb, const double& hkc )
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

		file << R"("a","b","c","hka","hkb","hkc")" << std::endl;	// The header.
		file.precision( 15 );
		file << a << "," << b << "," << c << "," << hka << "," << hkb << "," << hkc << std::endl;
		file.close();
	}

	CHKERRXX( PetscPrintf( mpi.comm(), "Rank %d successfully created params file '%s'\n", mpi.rank(), fullFileName.c_str() ) );
	SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );				// Wait here until rank 0 is done.
}