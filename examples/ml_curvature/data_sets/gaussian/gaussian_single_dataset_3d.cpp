/**
 * Generate samples from a Gaussian surface for *offline inference* (i.e., using Python).  The canonical surface is a Monge patch represen-
 * ted by
 *
 *                                           Q(u,v) = a*exp(-0.5*(u^2/su^2 + v^2/sv^2)),
 *
 * with zero means mu=mv=0, variances su^2 and sv^2, and height a.  As for the level-set function whose Gamma=Q(u,v)=0, phi < 0 for points
 * below Q(u,v), and phi > 0 for points above the Gaussian.  We simplify calculations by expressing query points in terms of the canonical
 * frame, which can be affected by a rigid-body transformation (i.e., translation and rotation).
 *
 * Theoretically, the Gaussian curvature can be positive and negative.  Thus, this data set can be used to test the neural network ability
 * on saddle samples.  In any case, if a point belongs to a non-saddle region (i.e., ih2kg>=C), it'll be negative-mean-curvature normalized
 * by taking the sign of ihk ONLY if requested.  On the other hand, if the point belongs to a (numerical) saddle region, its sample will
 * never be negative-mean-curvature-normalized.  In any case, we apply sample reorientation by rotating the stencil so that the (possibly
 * updated) gradient has all its components non-negative.  Finally, we reflect the data packet about the y-x = 0 plane, and thus we produce
 * two samples for each interface point.  At inference time, both outputs are averaged to improve accuracy.
 *
 * Negative-mean-curvature normalization, when applicable, depends on the sign of the linearly interpolated mean ihk at the interface.  As
 * for the Gaussian curvature, we normalize it by scaling it with h^2 ---which leads to the true h2kg and the linearly interpolated ih2kg
 * values in the collected data packets.
 *
 * The sample file is of the form "#/gaussian/$/iter%_data.csv", and the params file is "#/gaussian/$/iter%_params.csv", where # is the
 * unit-octree max level of refinement, $ is the experiment id, and % is the number of redistancing steps.  The data file contains as many
 * rows as twice the number of collected samples with all data-packet info.  The params file stores the values for "a", "su^2", and "sv^2"
 * and the maximum (true) "hk" at the peak.  In addition, we export VTK data for visualization and validation.
 *
 * @note Here and across related files to machine-learning computation of mean curvature use the geometrical definition of mean curvature;
 * that is, H = 0.5(k1 + k2), where k1 and k2 are principal curvatures.
 *
 * Developer: Luis Ángel.
 * Created: February 5, 2022.
 * Updated: May 7, 2022.
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
#include "gaussian_3d.h"
#include <src/parameter_list.h>
#include <cassert>


void writeParamsFile( const mpi_environment_t& mpi, const std::string& path, const std::string& fileName,
					  const double& a, const double& su2, const double& sv2, const double& hk );

GaussianLevelSet *setupDomain( const mpi_environment_t& mpi, const Gaussian& gaussian, const double& h, const double origin[P4EST_DIM],
							   const double& rotAngle, const double rotAxis[P4EST_DIM], const u_char& maxRL, u_char& octMaxRL,
							   int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM], double xyz_max[P4EST_DIM] );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<u_char>      experimentId( pl,     0, "experimentId"	 , "Experiment Id (default: 0)" );
	param_t<double> nonSaddleMinIH2KG( pl, -7e-6, "nonSaddleMinIH2KG", "Min numerical dimensionless Gaussian curvature (at Gamma) for "
																	   "numerical non-saddle samples (default: -7e-6)" );
	param_t<double>             maxHK( pl,   0.6, "maxHK"			 , "Desired max (absolute) dimensionless mean curvature at the peak. "
																	   "Must be in the open interval of (1/3, 2/3) (default: 0.6)" );
	param_t<u_char>             maxRL( pl,     6, "maxRL"			 , "Maximum level of refinement per unit-cube octree (default: 6)" );
	param_t<int>          reinitIters( pl,    10, "reinitIters"		 , "Number of iterations for reinitialization (default: 10)" );
	param_t<double>                 a( pl,   1.0, "a"				 , "Gaussian height (i.e., Q(0,0)) in the range of [16h, 64h] "
																	  "(default 1)" );
	param_t<double>         susvRatio( pl,   3.0, "susvRatio"	 	 , "The ratio su/sv in the range of [1, 3] (default: 3)" );
	param_t<bool>  		perturbOrigin( pl,  true, "perturbOrigin"	 , "Whether to perturb the Gaussian's frame randomly in [-h/2,+h/2]^3 "
															   		   "(default: true)" );
	param_t<bool>      randomRotation( pl,  true, "randomRotation"	 , "Whether to apply a rotation with a random angle about a random unit"
																	   " axis (default: true)" );
	param_t<u_int>        randomState( pl,     7, "randomState"	 	 , "Seed for canonical frame's random perturbations (default: 7)" );
	param_t<std::string>       outDir( pl,   ".", "outDir"		 	 , "Path where data files will be written to (default: build folder)" );
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
		if( cmd.parse( argc, argv, "Generating Gaussian data set for offline evaluation of a trained error-correcting neural network" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		// Parameter validation.
		if( reinitIters() <= 0 )
			throw std::invalid_argument( "[CASL_ERROR] Number of reinitializating iterations must be strictly positive." );

		if( maxHK() <= 1./3 || maxHK() >= 2./3 )
			throw std::invalid_argument( "[CASL_ERROR] Desired max hk must be in the range of (1/3, 2/3)." );

		if( susvRatio() < 1 || susvRatio() > 3 )
			throw std::invalid_argument( "[CASL_ERROR] The ratio su/sv must be in the range of [1, 3]." );

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.

		if( a() < 16 * h || a() > 64 * h )
			throw std::invalid_argument( "[CASL_ERROR] Gaussian amplitude must be in the range of [16h, 64h] "
										 "= [" + std::to_string( 16 * h ) + ", " + std::to_string( 64 * h ) + "]." );

		std::mt19937 gen( randomState() );					// Engine used for random perturbations of the canonical frame: rotation and shift.
		std::mt19937 genNoise( mpi.rank() );				// Engine for random noise on phi(x) if requested (and different for each rank).
		const double RAND_NOISE = randomNoise() > 0? randomNoise() : 1;
		std::uniform_real_distribution<double> randomNoiseDist( -h * RAND_NOISE, +h * RAND_NOISE );
		const double MAX_K = maxHK() / h;					// Now that we know the parameters are valid, find max hk and variances.
		const double SV2 = a() * (1 + SQR( susvRatio() )) / (2 * SQR( susvRatio() ) * MAX_K);
		const double SU2 = SQR( susvRatio() ) * SV2;

		Gaussian gaussian( a(), SU2, SV2 );

		std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;	// Buffer of accumulated (normalized and augmented) samples.
		double origin[P4EST_DIM] = {0, 0, 0};				// Gaussian's frame origin (possible perturbed).
		double rotAxis[P4EST_DIM] = {0, 0, 1};				// Rotation axis for possible random rotation.
		double rotAngle = 0;

		double trackedMinHK = DBL_MAX;						// We want to track min and max mean |hk*| for debugging.
		double trackedMaxHK = 0;
		if( mpi.rank() == 0 )			// Only rank 0 controls the buffer and perturbs the Gaussian's frame to create an affine-trans-
		{								// formed level-set function.
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
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() ) + "/gaussian/" + std::to_string( experimentId() );
		writeParamsFile( mpi, DATA_PATH, "iter" + std::to_string( reinitIters() ) + "_params.csv", a(), SU2, SV2, maxHK() );

		// Prepping the samples file.  Only rank 0 writes the samples to a file.
		std::ofstream file;
		std::string fileName = "iter" + std::to_string( reinitIters() ) + "_data.csv";
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

		///////////////////////////////////////////////////////// Data production //////////////////////////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began generating Gaussian data set for offline evaluation with a = %g, su^2 = %g, sv^2 = %g, max "
								 "|hk| = %g, and h = %g (level %i)\n", a(), SU2, SV2, maxHK(), h, maxRL() );

		parStopWatch watch;
		watch.start();

		// Domain information.
		u_char octMaxRL;
		int n_xyz[P4EST_DIM];
		double xyz_min[P4EST_DIM];
		double xyz_max[P4EST_DIM];
		int periodic[P4EST_DIM] = {0, 0, 0};
		GaussianLevelSet *gLS = setupDomain( mpi, gaussian, h, origin, rotAngle, rotAxis, maxRL(), octMaxRL, n_xyz, xyz_min, xyz_max );

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Create the forest using the Gaussian level-set as a refinement criterion.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, octMaxRL, gLS, 3.0 );
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		gLS->toggleCache( true );		// Turn on cache to speed up repeated distance computations.
		gLS->reserveCache( (size_t)pow( 0.75 * (xyz_max[0] - xyz_min[0]) / h, 3 ) );	// Reserve space in cache to improve hashing.
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
		// the otherwise calculated exact distances.  Add noise if requested.
		gLS->evaluate( p4est, nodes, phi, exactFlag );

		if( randomNoise() > 0 )
		{
			double *phiPtr;
			CHKERRXX( VecGetArray( phi, &phiPtr ) );
			foreach_node( n, nodes )
				phiPtr[n] += randomNoiseDist( genNoise );
			CHKERRXX( VecRestoreArray( phi, &phiPtr ) );
		}

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
		gLS->collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, trackedMaxErrors, trackedMinHK, trackedMaxHK, samples,
							 nNumericalSaddles, exactFlag, sampledFlag, hkError, ihk, h2kgError, ih2kg, phiError, NAN, NAN,
							 nonSaddleMinIH2KG() );
		gLS->clearCache();
		gLS->toggleCache( false );
		delete gLS;

		// Accumulate samples in the buffer; normalize phi by h, apply negative-mean-curvature normalization to non-saddle samples only if
		// requested, and reorient data packets.  The last 2 parameter avoids flipping the signs of hk, ihk, h2kg, and ih2kg.  Then, augment
		// samples by reflecting about y - x = 0.
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
		oss << "gaussian_dataset_id" << (int)experimentId() << "_lvl" << (int)maxRL();
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
 * Create and write the Gaussian's params file.
 * @param [in] mpi MPI environment.
 * @param [in] path Full path where to place the file.
 * @param [in] fileName File name.
 * @param [in] a Amplitude.
 * @param [in] su2 Variance along the u direction.
 * @param [in] sv2 Variance along the v direction.
 * @param [in] hk Absolute dimensionless mean curvature attained at the peak (i.e., at u=v=0).
 * @throws runtime_error if path or file can't be created and or written.
 */
void writeParamsFile( const mpi_environment_t& mpi, const std::string& path, const std::string& fileName,
					  const double& a, const double& su2, const double& sv2, const double& hk )
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

		file << R"("a","su2","sv2","hk")" << std::endl;	// The header.
		file.precision( 15 );
		file << a << "," << su2 << "," << sv2 << "," << hk << std::endl;
		file.close();
	}

	CHKERRXX( PetscPrintf( mpi.comm(), "Rank %d successfully created params file '%s'\n", mpi.rank(), fullFileName.c_str() ) );
	SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );				// Wait here until rank 0 is done.
}

/**
 * Set up domain and create a Gaussian level-set function.
 * @param [in] mpi MPI environment object.
 * @param [in] gaussian Gaussian function in canonical space.
 * @param [in] h Mesh size.
 * @param [in] origin Gaussian's local frame origin with respect to world coordinates.
 * @param [in] rotAngle Gaussian's local frame angle of rotation about a unit axis.
 * @param [in] rotAxis The unit axis to rotate the frame about.
 * @param [in] maxRL Maximum level of refinement for the whole domain.
 * @param [out] octMaxRL Effective maximum refinement for each octree to achieve desired h.
 * @param [out] n_xyz Number of octrees in each direction.
 * @param [out] xyz_min Mininum coordinates of computational domain.
 * @param [out] xyz_max Maximum coordinates of computational domain.
 * @return Dynamically allocated Gaussian level-set object.  You must delete it in caller function.
 */
GaussianLevelSet *setupDomain( const mpi_environment_t& mpi, const Gaussian& gaussian, const double& h, const double origin[P4EST_DIM],
							   const double& rotAngle, const double rotAxis[P4EST_DIM], const u_char& maxRL, u_char& octMaxRL,
							   int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM], double xyz_max[P4EST_DIM] )
{
	// Finding how far to go in the limiting ellipse half-axes.  We'll tringulate surface only within this region.
	const double U_ZERO = gaussian.findKappaZero( h, dir::x );
	const double V_ZERO = gaussian.findKappaZero( h, dir::y );
	const double ULIM = U_ZERO + gaussian.su();		// Limiting ellipse semi-axes for triangulation.
	const double VLIM = V_ZERO + gaussian.sv();
	const double QTOP = gaussian.a() + 4 * h;		// Adding some padding so that we can sample points correctly at the tip.
	double qAtULim = gaussian( ULIM, 0 );			// Let's find the lowest Q.
	double qAtVLim = gaussian( 0, VLIM );
	const double QBOT = MIN( qAtULim, qAtVLim ) - 4 * h;
	const size_t HALF_U_H = ceil(ULIM / h);			// Half u axis in h units.
	const size_t HALF_V_H = ceil(VLIM / h);			// Half v axis in h units.

	const double QCylCCoords[8][P4EST_DIM] = {		// Cylinder in canonical coords containing the Gaussian surface.
		{-ULIM, 0, QTOP}, {+ULIM, 0, QTOP}, 		// Top coords (the four points lying on the same QTOP found above).
		{0, -VLIM, QTOP}, {0, +VLIM, QTOP},
		{-ULIM, 0, QBOT}, {+ULIM, 0, QBOT},			// Base coords (the four points lying on the same QBOT found above).
		{0, -VLIM, QBOT}, {0, +VLIM, QBOT}
	};

	// Defining the transformed Gaussian level-set function.  This also discretizes the surface using a balltree to speed up queries
	// during grid refinment.
	auto *gLS = new GaussianLevelSet( &mpi, Point3( origin ), Point3( rotAxis ), rotAngle, HALF_U_H, HALF_V_H, maxRL, &gaussian, SQR(ULIM), SQR(VLIM), 5 );

	// Finding the world coords of (canonical) cylinder containing Q(u,v).
	double minQCylWCoords[P4EST_DIM] = {+DBL_MAX, +DBL_MAX, +DBL_MAX};	// Min and max cylinder world coords.
	double maxQCylWCoords[P4EST_DIM] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
	for( const auto& cylCPoint : QCylCCoords )
	{
		Point3 cylWPoint = gLS->toWorldCoordinates( cylCPoint[0], cylCPoint[1], cylCPoint[2] );
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			minQCylWCoords[i] = MIN( minQCylWCoords[i], cylWPoint.xyz( i ) );
			maxQCylWCoords[i] = MAX( maxQCylWCoords[i], cylWPoint.xyz( i ) );
		}
	}

	// Use the x,y,z ranges to find the domain's length in each direction.
	double QCylWRange[P4EST_DIM];
	double WCentroid[P4EST_DIM];
	for( int i = 0; i < P4EST_DIM; i++ )
	{
		QCylWRange[i] = maxQCylWCoords[i] - minQCylWCoords[i];
		WCentroid[i] = (minQCylWCoords[i] + maxQCylWCoords[i]) / 2;	// Raw centroid.
		WCentroid[i] = round( WCentroid[i] / h ) * h;				// Centroid as a multiple of h.
	}

	const double CUBE_SIDE_LEN = MAX( QCylWRange[0], MAX( QCylWRange[1], QCylWRange[2] ) );
	const unsigned char OCTREE_RL_FOR_LEN = MAX( 0, maxRL - 5 );	// Defines the log2 of octree's len (i.e., octree's len is a power of two).
	const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
	octMaxRL = maxRL - OCTREE_RL_FOR_LEN;							// Effective max refinement level to achieve desired h.
	const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );			// Number of trees in each dimension.
	const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;			// Adjusted domain cube len as a multiple of *both* h and octree len.
	const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

	// Defining a symmetric cubic domain whose dimensions are multiples of h and contain Gaussian and its limiting ellipse.
	for( int i = 0; i < P4EST_DIM; i++ )
	{
		n_xyz[i] = N_TREES;
		xyz_min[i] = WCentroid[i] - HALF_D_CUBE_SIDE_LEN;
		xyz_max[i] = WCentroid[i] + HALF_D_CUBE_SIDE_LEN;
	}

	return gLS;
}
