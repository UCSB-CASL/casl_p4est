/**
 * Generate data set for training a neural network on spherical interfaces using samples from a reinitialized non-signed
 * level-set function function.
 *
 * The collected samples include level-set, gradient, and mean and Gaussian curvature data.  Unlike sinusoidal surfaces,
 * all data belons to non-saddle regions.  For this reason, we can use the same techniques from 2d and normalize the
 * samples to the negative-mean-curvature spectrum followed by gradient reorientation and augmentaion by reflecting ab
 * out the plane y - x = 0.
 *
 * The file generated is named sphere.csv and is specific for a domain mesh size and number of iterations for reinitia-
 * lization.
 *
 * Developer: Luis Ángel.
 * Created: March 19, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

// System.
#include <stdexcept>
#include <iostream>

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include <src/parameter_list.h>
#include <src/casl_geometry.h>
#include <src/my_p8est_nodes_along_interface.h>
#include <src/my_p8est_curvature_ml.h>
#include <src/my_p8est_interpolation_nodes.h>


std::pair<double,double> collectSamples( const int& keepEveryXSamples, const double& radius, const double& h,
										 const mpi_environment_t& mpi, const p4est_t *p4est, const p4est_nodes_t *nodes,
										 const my_p4est_node_neighbors_t *ngbd, const Vec& phi, const u_char& octreeMaxRL,
										 std::mt19937& gen, std::vector<std::vector<double>>& samples );

int saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>>& buffer, int& bufferSize,
				 std::ofstream& file, const std::string& fileName, const size_t& bufferMinSize,
				 const int& samplesLeftToSave );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<double>          minHK( pl, 0.005, "minHK"				, "Minimum mean dimensionless curvature for non-saddle points (default: 0.005)" );
	param_t<double>          maxHK( pl,  2./3, "maxHK"				, "Maximum mean dimensionless curvature (default: 2/3)" );
	param_t<u_char>          maxRL( pl,     6, "maxRL"				, "Maximum level of refinement per unit-square quadtree (default: 6)" );
	param_t<u_short>   reinitIters( pl,    10, "reinitIters"		, "Number of iterations for reinitialization (default: 10)" );
	param_t<int>       spheresPerH( pl,     2, "spheresPerH"		, "How many sphere radii to fit in a cell (default: 2)" );
	param_t<double>   samplesPerH3( pl,   0.1, "samplesPerH3"		, "Samples per H^3 based on the average radius (default: 0.1)" );
	param_t<int> keepEveryXSamples( pl,    10, "keepEveryXSamples"	, "Keep record every x samples next to Gamma randomly (default: 10)" );
	param_t<size_t>  bufferMinSize( pl,   1e4, "bufferMinSize"		, "Buffer minimum overflow size to trigger storage (default: 10K)" );
	param_t<std::string>    outDir( pl,   ".", "outDir"				, "Path where files will be written to (default: build folder)" );

	std::mt19937 genProb{};		// NOLINT Random engine for probability when choosing candidate nodes (it's OK that it's not in sync among processes).
	std::mt19937 genTrans{};	// NOLINT This engine is used for the random shift of the sphere.

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Generating the sphere data set for three-dimensional mean curvature" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		CHKERRXX( PetscPrintf( mpi.comm(), "\n******************* Generating a sphere data set in 3D *******************\n" ) );

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.
		const double MIN_K = minHK() / h;					// Target mean curvature bounds.
		const double MAX_K = maxHK() / h;
		const double MIN_RADIUS = 1. / MAX_K;
		const double MAX_RADIUS = 1. / MIN_K;
		const double D_DIM = ceil( MAX_RADIUS + 4 * h );	// Symmetric units around origin: [-DIM, +DIM]^{P4EST_DIM}.
		const int NUM_SPHERES = ceil( spheresPerH() * ((MAX_RADIUS - MIN_RADIUS) / h + 1) );	// Number of circles is proportional to radii difference and H.

		// Expected number of samples per distinct radius.
		// First, we allow to generate a tentative number of samples.  Then, we randomly subsample.  This allows varying
		// the origin of the spheres, and then pick a smaller subset with samples from several configurations.
		// Number of samples per radius is approximated by samplesPerH3 samples per h^3, which comes from the volume
		// difference of the average sphere and the next sphere.
		// Then, use a ramp from 0.75*SAMPLES_PER_RADIUS to 1.25*SAMPLES_PER_RADIUS for MIN_RADIUS and MAX_RADIUS.  This
		// ensures sufficient samples for the small radii.
		// If user wants it, keep every xth sample randomly to reduce data set size.
		const double AVG_RADIUS = (MAX_RADIUS + MIN_RADIUS) / 2.;
		const int AVG_SAMPLES_PER_RADIUS = (int)ceil( samplesPerH3() * 4./3 * M_PI / CUBE( h ) * (CUBE( AVG_RADIUS ) - CUBE( AVG_RADIUS - h )) ) / keepEveryXSamples();

		std::uniform_real_distribution<double> uniformDistribution( -h/2, +h/2 );	// Random translation.

		/////////////////////////////////////////// Preparing data set files ///////////////////////////////////////////

		parStopWatch watch;
		PetscPrintf( mpi.comm(), ">> Began to generate datasets for %i spheres with maximum level of refinement = %i "
								 "and finest h = %g\n", NUM_SPHERES, maxRL(), h );
		watch.start();

		// Prepping the samples' files.  Notice we are no longer interested on exact-signed distance functions, only
		// reinitialized data.  Only rank 0 writes the samples to a file.
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
		std::ofstream file;
		std::string fileName = "sphere.csv";
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

		/////////////////////////////////////////// Data production loop ////////////////////////////////////////////

		// Variables to control the spread of spheres' radii.  These must vary depending on the uniform spread of mean curvature.
		double meanKDistance = MIN_K - MAX_K;				// Radii are in [1/MAX_KAPPA, 1/MIN_KAPPA].
		double rLinspace[NUM_SPHERES];
		for( int i = 0; i < NUM_SPHERES; i++ )				// Uniform linear space from 0 to 1, with NUM_SPHERES steps.
			rLinspace[i] = (double)(i) / (NUM_SPHERES - 1);

		std::vector<double> samplesPerRadius;
		linspace( 0.75 * AVG_SAMPLES_PER_RADIUS, 1.25 * AVG_SAMPLES_PER_RADIUS, NUM_SPHERES, samplesPerRadius );

		// Domain information, applicable to all spherical interfaces.
		int n_xyz[P4EST_DIM] = { 2 * (int)D_DIM, 2 * (int)D_DIM, 2 * (int)D_DIM };	// Sym num of trees in +ve and -ve axes.
		double xyz_min[P4EST_DIM] = { -D_DIM, -D_DIM, -D_DIM };						// Squared domain.
		double xyz_max[P4EST_DIM] = { D_DIM, D_DIM, D_DIM };
		int periodic[P4EST_DIM] = { 0, 0, 0 };										// Non-periodic domain.

		int nSamples = 0;
		int nc = 0;							// Keeps track of number of spheres whose samples have been collected.
		while( nc < NUM_SPHERES )
		{
			const double KAPPA = MAX_K + rLinspace[nc] * meanKDistance;
			const double R = 1 / KAPPA;						// Radius to be evaluated and its dimensionless mean kappa.

			std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;	// Cumulative buffer of (normalized and augmented) samples.
			if( mpi.rank() == 0 )								// Only rank 0 controls the buffer.
				buffer.reserve( bufferMinSize() );
			int bufferSize = 0;									// Everyone knows current buffer's state to keep them in sync.
			SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );

			// Generate a given number of randomly centered spheres with the same radius and accumulate samples until we
			// reach a quota.
			double maxRE = 0;								// Maximum relative error.
			int nSamplesLeftForSameRadius = 2 * (int)round( samplesPerRadius[nc] );		// Twice because of the augmentation.
			while( nSamplesLeftForSameRadius > 0  )
			{
				double C[] = {
					DIM( uniformDistribution( genTrans ),	// Center coords are randomly chosen around the origin.
						 uniformDistribution( genTrans ),
						 uniformDistribution( genTrans ) )
				};
				SC_CHECK_MPI( MPI_Bcast( C, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same random shift.

				// p4est variables and data structures: these change with every single sphere because we must refine the
				// trees according to the center and radius.
				p4est_t *p4est;
				p4est_nodes_t *nodes;
				my_p4est_brick_t brick;
				p4est_ghost_t *ghost;
				p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

				// Definining the non-signed distance level-set function to be reinitialized.
				geom::SphereNSD sphereNsd( DIM( C[0], C[1], C[2] ), R );
				geom::Sphere sphere( DIM( C[0], C[1], C[2] ), R );
				splitting_criteria_cf_and_uniform_band_t levelSetSC( 1, maxRL(), &sphere, 3.0 );

				// Create the forest using a level set as refinement criterion.
				p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
				p4est->user_pointer = (void *)( &levelSetSC );

				// Refine and partition forest.
				for( int i = 0; i < maxRL(); i++ )
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

				// Validation.
				double dxyz[P4EST_DIM]; 			// Dimensions of the smallest quadrants.
				double dxyz_min;        			// Minimum side length of the smallest quadrants.
				double diag_min;        			// Diagonal length of the smallest quadrants.
				get_dxyz_min( p4est, dxyz, &dxyz_min, &diag_min );
				assert( ABS( dxyz_min - h ) <= EPS );

				// Ghosted parallel PETSc vectors to store level-set values.
				Vec phi;
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

				// Calculate the level-set function values for all independent nodes.
				sample_cf_on_nodes( p4est, nodes, sphereNsd, phi );

				// Reinitialize level-set function.
				my_p4est_level_set_t ls( ngbd );
				ls.reinitialize_2nd_order( phi, reinitIters() );

				std::pair<double, double> maxErrors;
				std::vector<std::vector<double>> samples;
				maxErrors = collectSamples( keepEveryXSamples(), R, h, mpi, p4est, nodes, ngbd, phi, maxRL(), genProb, samples );
				maxRE = MAX( maxRE, maxErrors.first );

				int batchSize = kml::utils::processSamplesAndAccumulate( mpi, samples, buffer, h, true );
				bufferSize += batchSize;
				int savedSamples = saveSamples( mpi, buffer, bufferSize, file, fileName, bufferMinSize(), nSamplesLeftForSameRadius );
				nSamples += savedSamples;
				nSamplesLeftForSameRadius -= savedSamples;
#ifdef DEBUG
				PetscPrintf( mpi.comm(), ":: Number of samples left: %d\n", nSamplesLeftForSameRadius );
#endif

				// Destroy the p4est and its connectivity structure.
				delete ngbd;
				delete hierarchy;
				p4est_nodes_destroy( nodes );
				p4est_ghost_destroy( ghost );
				p4est_destroy( p4est );
				my_p4est_brick_destroy( connectivity, &brick );

				// Synchronize.
				SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );
			}

			PetscPrintf( mpi.comm(), "     (%d) Done with radius = %f.  Maximum relative error = %f.  Samples = %d;\n",
						 nc, R, maxRE, 2 * (int)round( samplesPerRadius[nc] ) );
			nc++;

			if( nc % 10 == 0 )
				PetscPrintf( mpi.comm(), "   [%i radii evaluated after %f secs.]\n", nc, watch.get_duration_current() );
		}

		file.close();

		PetscPrintf( mpi.comm(), "<< Finished generating %i circles and %i samples in %f secs.\n",
					 nc, nSamples, watch.get_duration_current() );
		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}

/**
 * Collect samples using a subsampling approach
 * @note Samples are not normalized in any way: not negative-mean-curvature nor gradient-reoriented to first octant.
 * @param [in] keepEveryXSamples Keep record every x samples next to Gamma, randomly.
 * @param [in] radius Sphere radius.
 * @param [in] h Mesh size.
 * @param [in] mpi MPI environment.
 * @param [in] p4est P4est data structure.
 * @param [in] nodes Nodes data structure.
 * @param [in] ngbd Nodes' neighborhood data structure.
 * @param [in] phi Parallel vector with level-set values.
 * @param [in] octreeMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
 * @param [in,out] gen Random-number generator device to decide whether to take a sample or not.
 * @param [out] samples Array of collected samples.
 * @return Maximum errors in dimensionless mean and Gaussian curvatures (reduced across processes).
 * @throws runtime_error or invalid_argument if phi is not given, if the radius is non positive, or if a saddle point was found.
 */
std::pair<double,double> collectSamples( const int& keepEveryXSamples, const double& radius, const double& h,
										 const mpi_environment_t& mpi, const p4est_t *p4est, const p4est_nodes_t *nodes,
										 const my_p4est_node_neighbors_t *ngbd, const Vec& phi, const u_char& octreeMaxRL,
										 std::mt19937& gen, std::vector<std::vector<double>>& samples )
{
	std::string errorPrefix = "collectSamples: ";

	if( !phi )
		throw std::invalid_argument( errorPrefix + "phi vector can't be null!" );

	std::uniform_real_distribution<double> skipDist;				// Random candidate selection.

	// Get indices for locally owned candidate nodes next to Gamma.
	NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, (char)octreeMaxRL );
	std::vector<p4est_locidx_t> indices;
	nodesAlongInterface.getIndices( &phi, indices );

	// Compute normal vectors and mean/Gaussian/principal curvatures.
	Vec normals[P4EST_DIM],	kappaMG[2], kappa12[2];
	for( auto& component : normals )
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &component ) );
	CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaMG[0] ) );	// This is mean curvature, and
	CHKERRXX( VecCreateGhostNodes( p4est, nodes, &kappaMG[1] ) );	// this is Gaussian curvature.
	for( auto& pk : kappa12 )
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &pk ) );

	compute_normals_and_curvatures( *ngbd, phi, normals, kappaMG[0], kappaMG[1], kappa12 );

	const double *phiReadPtr;								// We need access to phi to project points onto Gamma.
	CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );

	const double *normalsReadPtr[P4EST_DIM];
	for( int i = 0; i < P4EST_DIM; i++ )
		CHKERRXX( VecGetArrayRead( normals[i], &normalsReadPtr[i] ) );

	samples.clear();										// We'll get (possibly) as many as points next to Gamma
	samples.reserve( indices.size() );						// and within limiting sampling circle.

	// Prepare mean and Gaussian curvatures interpolation.
	my_p4est_interpolation_nodes_t kappaMGInterp( ngbd );
	kappaMGInterp.set_input( kappaMG, interpolation_method::linear, 2 );

	std::uniform_real_distribution<double> pDistribution;
	double trackedMaxHKError = 0;				// Track the min and max mean |hk*| and Gaussian curvature errors.
	double trackedMaxH2KGError = 0;

#ifdef DEBUG
	std::cout << "Rank " << mpi.rank() << " reports " << indices.size() << " candidate nodes for sampling." << std::endl;
#endif

	for( const auto& n : indices )
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( n, p4est, nodes, xyz );

		std::vector<p4est_locidx_t> stencil;
		try
		{
			if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	// Does it have a valid stencil?
				continue;

			// Starting at this point, we collect samples.  We shouldn't get any negative-Gaussian sample because there
			// are no saddles in a sphere.
			double hk = h / radius;							// Mean hk at the *exact* projection onto Gamma.
			for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
				xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
			double kappaMGValues[2];
			kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
			double ihkVal = h * kappaMGValues[0];
			double ih2kgVal = SQR( h ) * kappaMGValues[1];
			if( ih2kgVal > 0 )								// Not a saddle?  We're OK.
			{
				if( keepEveryXSamples > 1 && skipDist( gen ) >= 1. / keepEveryXSamples )	// Probabilistic subsampling.
					continue;
			}
			else
				throw std::runtime_error( errorPrefix + "Found a sample with negative Gaussian curvature!" );

			// Up to this point, we got a good sample.  Populate its features.
			std::vector<double> *sample;					// Points to new sample in the appropriate array.
			samples.emplace_back();
			sample = &samples.back();
			sample->reserve( K_INPUT_SIZE_LEARN );			// phi + normals + hk* + ihk + h2kg* + ih2kg = 112 fields.

			for( const auto& idx : stencil )				// First, phi values.
				sample->push_back( phiReadPtr[idx] );

#ifdef DEBUG
			// Verify that phi(center)'s sign differs with any of its irradiating neighbors.
			if( !NodesAlongInterface::isInterfaceStencil( *sample ) )
				throw std::runtime_error( errorPrefix + "Detected a non-interface stencil!" );
#endif

			for( const auto &component : normalsReadPtr)	// Next, normal components (First x group, then y, then z).
			{
				for( const auto& idx: stencil )
					sample->push_back( component[idx] );
			}

			sample->push_back( hk );						// Then, attach target mean hk* and numerical ihk.
			sample->push_back( ihkVal );
			double h2kg = SQR( h / radius );
			sample->push_back( h2kg );						// And, attach true Gaussian h^2*kg and ih2kg.
			sample->push_back( ih2kgVal );

			// Update stats.
			double errorHK = ABS( (*sample)[K_INPUT_SIZE_LEARN - 4] - (*sample)[K_INPUT_SIZE_LEARN - 3] );
			double errorH2KG = ABS( (*sample)[K_INPUT_SIZE_LEARN - 2] - (*sample)[K_INPUT_SIZE_LEARN - 1] );

			trackedMaxHKError = MAX( trackedMaxHKError, errorHK );
			trackedMaxH2KGError = MAX( trackedMaxH2KGError, errorH2KG );
		}
		catch( std::runtime_error &e )
		{
			std::cerr << e.what() << std::endl;
		}
	}

#ifdef DEBUG
	std::cout << "Rank " << mpi.rank() << " collected " << samples.size() + samples.size() << " *unique* samples." << std::endl;
#endif
	kappaMGInterp.clear();

	SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHKError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxH2KGError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );

#ifdef DEBUG	// Printing the errors.
	CHKERRXX( PetscPrintf( mpi.comm(), "Tracked max mean hk error = %f\n", trackedMaxHKError ) );
	CHKERRXX( PetscPrintf( mpi.comm(), "Tracked max Gaussian h^2k error = %f\n", trackedMaxH2KGError ) );
#endif

	// Clean up.
	for( int i = 0; i < P4EST_DIM; i++ )
		CHKERRXX( VecRestoreArrayRead( normals[i], &normalsReadPtr[i] ) );
	CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );

	for( auto& pk : kappa12 )
		CHKERRXX( VecDestroy( pk ) );
	CHKERRXX( VecDestroy( kappaMG[0] ) );
	CHKERRXX( VecDestroy( kappaMG[1] ) );
	for( auto& component : normals )
		CHKERRXX( VecDestroy( component ) );

	return std::make_pair( trackedMaxHKError, trackedMaxH2KGError );
}

/**
 * Save samples in buffer if it has overflowed the user-defined min size, or if user wants to bypass the condition beca-
 * use the
 * Upon exiting, the buffer will be emptied and re-reserved, and the buffer size will be reset if we saved to a file.
 * @param [in] mpi MPI environment.
 * @param [in,out] buffer Sample buffer.
 * @param [in,out] bufferSize Current buffer's size.
 * @param [in,out] file File where to write samples.
 * @param [in] fileName File names array.
 * @param [in] bufferMinSize Predefined minimum size to trigger file saving (same value for non-saddles and saddles).
 * @param [in] samplesLeftToSave How many samples do we still need to collect/save to meet the quota.
 * @return Number of samples stored to a file if we did that (already shared across processes).
 */
int saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>>& buffer, int& bufferSize,
				  std::ofstream& file, const std::string& fileName, const size_t& bufferMinSize,
				  const int& samplesLeftToSave )
{
	int savedSamples = 0;

	if( bufferSize > 0 && samplesLeftToSave > 0 && (samplesLeftToSave <= bufferSize || bufferSize >= bufferMinSize) )	// Check if it's time to save samples.
	{
		if( mpi.rank() == 0 )
		{
			int i;
			size_t numSamplesToSave = MIN( bufferSize, samplesLeftToSave );
			for( i = 0; i < numSamplesToSave; i++ )
			{
				int j;
				for( j = 0; j < K_INPUT_SIZE_LEARN - 1; j++ )
					file << buffer[i][j] << ",";		// Inner elements.
				file << buffer[i][j] << std::endl;		// Last element is ihk in 2D or ih2kg in 3D.
			}
			savedSamples = i;

			CHKERRXX( PetscPrintf( mpi.comm(), "[*] Saved %d out of %d samples to output file %s.\n", savedSamples, bufferSize, fileName.c_str() ) );

			buffer.clear();							// Reset control variables.
			buffer.reserve( bufferMinSize );
		}
		bufferSize = 0;
	}

	// Communicate to everyone the total number of saved samples.
	SC_CHECK_MPI( MPI_Bcast( &savedSamples, 1, MPI_INT, 0, mpi.comm() ) );	// Acts as an MPI_Barrier, too.
	return savedSamples;
}