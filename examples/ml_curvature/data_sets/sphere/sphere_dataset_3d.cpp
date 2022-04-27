/**
 * Generate data set for training a neural network on spherical interfaces using samples from reinitialized non-exact signed distance
 * level-set functions.
 *
 * We use an approach similar to
 * @cite H.V. Patel, A. Panda, J.A.M. Kuipers, and E.A.J.F. Peters.  Computing interface curvature from volume fractions: A machine learning
 * approach.  Comput. & Fluids, 193:104263, 2019,
 * where we a uniform distribution draws the target mean curvatures.  Given one hk*, we collect M samples, and proceed to draw another hk*.
 * Note this process produces a uniform distribution for hk* but not for radii (unlike Patel et al., who chose random radii instead of
 * random hk* values).
 *
 * The collected samples include level-set, gradient, and mean and Gaussian curvature data.  Unlike sinusoidal surfaces, all data belongs to
 * non-saddle regions.  For this reason, we can choose to either keep them as is or flip the stencil signs randomly.  In any case, we
 * perform gradient reorientation and augmentation by reflecting about the plane y - x = 0.
 *
 * The file generated is named sphere_rand.csv and is specific for a domain mesh size and number of iterations for reinitialization.
 *
 * Developer: Luis Ángel.
 * Created: March 31, 2022.
 * Updated: April 24, 2022.
 */
#include <src/my_p4est_to_p8est.h>		// Defines the P4_TO_P8 macro.

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


void collectSamples( const double& radius, const double& h, const mpi_environment_t& mpi, const p4est_t *p4est, const p4est_nodes_t *nodes,
					 const my_p4est_node_neighbors_t *ngbd, const Vec& phi, const u_char& octreeMaxRL, const double xyzMin[P4EST_DIM],
					 const double xyzMax[P4EST_DIM], std::vector<std::vector<double>>& samples, const bool& flipSign, std::mt19937& gen );

int saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>>& buffer, int& bufferSize, std::ofstream& file,
				 const u_int& numSamplesToSave, std::mt19937& gen, double& trackedMaxHKError, double& trackedMaxH2KGError );

void setupDomain( const mpi_environment_t& mpi, const double C[P4EST_DIM], const double& R, const double& h, const u_char& MAX_RL,
				  std::mt19937& gen, u_char& octreeMaxRL, int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM], double xyz_max[P4EST_DIM] );


int main ( int argc, char* argv[] )
{
	// Setting up parameters from command line.
	param_list_t pl;
	param_t<bool>     flipSignRandomly( pl,  true, "flipSignRandomly"		, "Whether to generate convex and concave samples by 'flipping "
																			  "a coin' (default: true)" );
	param_t<double>              minHK( pl, 0.004, "minHK"					, "Min dimensionless mean curvature for non-saddle points "
																			  "(default: 0.004)" );
	param_t<double>              maxHK( pl,  2./3, "maxHK"					, "Max dimensionless mean curvature (default: 2/3)" );
	param_t<u_char>              maxRL( pl,     6, "maxRL"					, "Max level of refinement per unit-cube octree (default: 6)" );
	param_t<u_short>       reinitIters( pl,    10, "reinitIters"			, "Number of iterations for reinitialization (default: 10)" );
	param_t<u_int>          numSpheres( pl,   2e5, "numSpheres"				, "Number of spheres (with distinct radii) to evaluate "
																			  "(default: 200K)" );
	param_t<u_int> numSamplesPerSphere( pl,     5, "numSamplesPerSphere"	, "Number of samples to collect randomly for each sphere "
																			  "(default: 5)" );
	param_t<std::string>        outDir( pl,   ".", "outDir"					, "Where to write data set (default: build folder)" );
	param_t<bool> useSignedDistanceFun( pl,  true, "useSignedDistanceFun"	, "If true, use phi(x)=|x-x0| - r; otherwise, use "
																			  "phi(x)=|x-x0|^2 - r^2 (default: true)" );
	param_t<double>        randomNoise( pl,  1e-4, "randomNoise"			, "Amount of uniform random noise to add to phi(x) as "
																			  "[+/-]h*randomNoise (default: 1e-4)" );

	std::mt19937 gen{};	// NOLINT Used for the random shift of the sphere and to choose a mean curvature uniformly (only on rank 0).

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

		CHKERRXX( PetscPrintf( mpi.comm(), "\n************************* Generating a sphere data set in 3D *************************\n" ) );

		///////////////////////////////////////////////////////// Parameter setup //////////////////////////////////////////////////////////

		const double h = 1. / (1 << maxRL());				// Highest spatial resolution in x/y directions.
		const double MIN_K = minHK() / h;					// Target mean curvature bounds.
		const double MAX_K = maxHK() / h;

		std::uniform_real_distribution<double> uniformDistTrans( -h/2, +h/2 );	// Random translation.
		std::uniform_real_distribution<double> randomNoiseDist( -h * randomNoise(), +h * randomNoise() );
		std::mt19937 genNoise( mpi.rank() );				// A separate seed for each rank: to be used only for noise, if requested.
		std::mt19937 genSign( mpi.rank() + 7 );				// Also for each rank but to be used to flip stencil signs as requested.

		///////////////////////////////////////////////////// Preparing data set files /////////////////////////////////////////////////////

		parStopWatch watch;
		PetscPrintf( mpi.comm(), ">> Began to generate datasets for %i spheres with %i samples each, with max level of refinement = %i and "
								 "finest h = %g\n", numSpheres(), numSamplesPerSphere(), maxRL(), h );
		watch.start();

		// Prepping the samples file.  Notice we are no longer interested on exact-signed distance functions, only reinitialized data.
		// Only rank 0 writes the samples to a file.
		const std::string DATA_PATH = outDir() + "/" + std::to_string( maxRL() );
		std::ofstream file;
		std::string fileName = "sphere_rand.csv";
		kml::utils::prepareSamplesFile( mpi, DATA_PATH, fileName, file );

		/////////////////////////////////////////////////////// Data production loop ///////////////////////////////////////////////////////

		// Variables to control the spread of spheres' radii.  These must vary depending on the uniform spread of mean curvature.
		double meanKDistance = MIN_K - MAX_K;				// Radii are in [1/MAX_KAPPA, 1/MIN_KAPPA].
		double rLinspace[numSpheres()];
		if( mpi.rank() == 0 )
		{
			std::uniform_real_distribution<double> uniformDist;
			for( int i = 0; i < numSpheres(); i++ )			// Uniform random dist in (0, 1] with numSpheres() steps to be shared among
				rLinspace[i] = uniformDist( gen );			// processes.
			rLinspace[0] = 0;
			rLinspace[numSpheres() - 1] = 1;
			std::sort( rLinspace, rLinspace + numSpheres() );
		}
		SC_CHECK_MPI( MPI_Bcast( rLinspace, numSpheres(), MPI_DOUBLE, 0, mpi.comm() ) );

		int nSamples = 0;
		int nc = 0;												// Keeps track of number of spheres whose samples have been collected.
		while( nc < numSpheres() )
		{
			const double KAPPA = MAX_K + rLinspace[nc] * meanKDistance;
			const double R = 1 / KAPPA;							// Radius to be evaluated and its dimensionless mean kappa.

			std::vector<std::vector<FDEEP_FLOAT_TYPE>> buffer;	// Cumulative buffer of (normalized and augmented) samples.
			if( mpi.rank() == 0 )								// Only rank 0 controls the buffer.
				buffer.reserve( 5e4 );
			SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );

			// Generate a randomly centered sphere with chosen radius and save M samples randomly.
			double C[] = {
				DIM( uniformDistTrans( gen ),					// Random center coords.
					 uniformDistTrans( gen ),
					 uniformDistTrans( gen ) )
			};
			SC_CHECK_MPI( MPI_Bcast( C, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );	// All processes use the same random shift.

			// Domain information.  To avoid discretizing the whole sphere (too expensive!), find a random region on the sphere and set a 3d
			// window around it.
			u_char octreeMaxRL;
			int n_xyz[P4EST_DIM];
			double xyz_min[P4EST_DIM];
			double xyz_max[P4EST_DIM];
			int periodic[P4EST_DIM] = {0, 0, 0};
			setupDomain( mpi, C, R, h, maxRL(), gen, octreeMaxRL, n_xyz, xyz_min, xyz_max );

			// p4est variables and data structures: these change with every single sphere because we must refine the trees according to the
			// center and radius.
			p4est_t *p4est;
			p4est_nodes_t *nodes;
			my_p4est_brick_t brick;
			p4est_ghost_t *ghost;
			p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

			// Definining the signed distance function to use as refinement criterion and possibly to be reinitialized.
			geom::Sphere sphere( C[0], C[1], C[2], R );
			splitting_criteria_cf_and_uniform_band_t levelSetSC( 0, octreeMaxRL, &sphere, 3.0 );

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
			double *phiPtr;
			CHKERRXX( VecGetArray( phi, &phiPtr ) );
			geom::SphereNSD *sphereNsd = (useSignedDistanceFun()? nullptr : new geom::SphereNSD( C[0], C[1], C[2], R ));
			foreach_node( n, nodes )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( n, p4est, nodes, xyz );
				phiPtr[n] = (useSignedDistanceFun()? sphere( xyz[0], xyz[1], xyz[2] ) : (*sphereNsd)( xyz[0], xyz[1], xyz[2] ));
				if( randomNoise() > 0 )
					phiPtr[n] += randomNoiseDist( genNoise );

			}
			CHKERRXX( VecRestoreArray( phi, &phiPtr ) );
			delete sphereNsd;

			// Reinitialize level-set function.
			my_p4est_level_set_t ls( ngbd );
			ls.reinitialize_2nd_order( phi, reinitIters() );

			// Collect and save samples.
			std::vector<std::vector<double>> samples;
			collectSamples( R, h, mpi, p4est, nodes, ngbd, phi, octreeMaxRL, xyz_min, xyz_max, samples, flipSignRandomly(), genSign );

			double maxErrors[2];
			int bufferSize = kml::utils::processSamplesAndAccumulate( mpi, samples, buffer, h, 0 );	// Don't neg-curvature normalize!  We already did so for each sample randomly.
			int savedSamples = saveSamples( mpi, buffer, bufferSize, file, numSamplesPerSphere(), gen, maxErrors[0], maxErrors[1] );
			nSamples += savedSamples;

			// Clean up.
			CHKERRXX( VecDestroy( phi ) );

			// Destroy the p4est and its connectivity structure.
			delete ngbd;
			delete hierarchy;
			p4est_nodes_destroy( nodes );
			p4est_ghost_destroy( ghost );
			p4est_destroy( p4est );
			my_p4est_brick_destroy( connectivity, &brick );

			// Synchronize.
			SC_CHECK_MPI( MPI_Barrier( mpi.comm() ) );

			PetscPrintf( mpi.comm(), "     (%d) Done with radius = %.8g.  Maximum hk error = %.8g.  Samples so far = %d;\n",
						 nc, R, maxErrors[0], nSamples );
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
 * Set up the dimensions of a relatively small domain around a random region on the surface of the sphere.
 * @param [in] mpi MPI environment.
 * @param [in] C Sphere's center.
 * @param [in] R Sphere's radius.
 * @param [in] h Mesh size.
 * @param [in] MAX_RL Maximum level of refinement per unit octant (i.e., h = 2^{-MAX_RL}).
 * @param [in,out] gen Random engine.
 * @param [out] octreeMaxRL Effective individual octree maximum level of refinement to achieve the desired h.
 * @param [out] n_xyz Number of octrees in each direction with maximum level of refinement octreeMaxRL.
 * @param [out] xyz_min Omega minimum dimensions.
 * @param [out] xyz_max Omega maximum dimensions.
 */
void setupDomain( const mpi_environment_t& mpi, const double C[P4EST_DIM], const double& R, const double& h, const u_char& MAX_RL,
				  std::mt19937& gen, u_char& octreeMaxRL, int n_xyz[P4EST_DIM], double xyz_min[P4EST_DIM], double xyz_max[P4EST_DIM] )
{
	if( mpi.rank() == 0 )
	{
		std::uniform_real_distribution<double> uniformDist;			// Use the technique by Patel et al. to find the azimuth and polar angles randomly.
		double azimuthAngle = uniformDist( gen ) * 2 * M_PI;
		double polarAngle = acos( 2 * uniformDist( gen ) - 1 );
		double pointOnSphere[P4EST_DIM] = { 						// Let's locate the random point on the sphere.
			C[0] + R * cos( azimuthAngle ) * sin( polarAngle ),
			C[1] + R * sin( azimuthAngle ) * sin( polarAngle ),
			C[2] + R * cos( polarAngle )
		};

		double COmega[P4EST_DIM];
		for( int i = 0; i < P4EST_DIM; i++ )						// Define the nearest discrete point (multiple of h) to point
			COmega[i] = round( pointOnSphere[i] / h ) * h;			// on sphere.

		double samRadius = 16 * h;									// At least we want this distance around COmega.
		const double CUBE_SIDE_LEN = 2 * samRadius;					// We want a cubic domain with an effective, yet small size.
		const u_char OCTREE_RL_FOR_LEN = MAX( 0, MAX_RL - 3 );		// Defines the log2 of octree's len (i.e., octree's len is a power of two).
		const double OCTREE_LEN = 1. / (1 << OCTREE_RL_FOR_LEN);
		octreeMaxRL = MAX_RL - OCTREE_RL_FOR_LEN;					// Effective max refinement level to achieve desired h.
		const int N_TREES = ceil( CUBE_SIDE_LEN / OCTREE_LEN );		// Number of trees in each dimension.
		const double D_CUBE_SIDE_LEN = N_TREES * OCTREE_LEN;		// Adjusted domain cube len as a multiple of h and octree len.
		const double HALF_D_CUBE_SIDE_LEN = D_CUBE_SIDE_LEN / 2;

		// Defining a symmetric cubic domain whose dimensions are multiples of h.
		for( int i = 0; i < P4EST_DIM; i++ )
		{
			n_xyz[i] = N_TREES;
			xyz_min[i] = COmega[i] - HALF_D_CUBE_SIDE_LEN;
			xyz_max[i] = COmega[i] + HALF_D_CUBE_SIDE_LEN;
		}
	}

	SC_CHECK_MPI( MPI_Bcast( &octreeMaxRL, 1, MPI_UNSIGNED_CHAR, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( n_xyz, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( xyz_min, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( xyz_max, P4EST_DIM, MPI_DOUBLE, 0, mpi.comm() ) );
}

/**
 * Collect samples for nodes next to Gamma.
 * @note Samples are not normalized in any way: not negative-mean-curvature nor gradient-reoriented to first octant.
 * @param [in] radius Sphere radius.
 * @param [in] h Mesh size.
 * @param [in] mpi MPI environment.
 * @param [in] p4est P4est data structure.
 * @param [in] nodes Nodes data structure.
 * @param [in] ngbd Nodes' neighborhood data structure.
 * @param [in] phi Parallel vector with level-set values.
 * @param [in] octreeMaxRL Effective octree maximum level of refinement (octree's side length must be a multiple of h).
 * @param [in] xyzMin Domain min bounds.
 * @param [in] xyzMax Domain max bounds.
 * @param [out] samples Array of collected samples.
 * @param [in] flipSign Should we flip stencil signs randomly?
 * @param [in,out] gen Random-number-generating engine for flipping signs as requested.
 * @throws invalid_argument if phi is not given, or runtime_error if a saddle point is found, or if we find a node where ihk * hk < 0.
 */
void collectSamples( const double& radius, const double& h, const mpi_environment_t& mpi, const p4est_t *p4est, const p4est_nodes_t *nodes,
					 const my_p4est_node_neighbors_t *ngbd, const Vec& phi, const u_char& octreeMaxRL, const double xyzMin[P4EST_DIM],
					 const double xyzMax[P4EST_DIM], std::vector<std::vector<double>>& samples, const bool& flipSign, std::mt19937& gen )
{
	std::string errorPrefix = "collectSamples: ";
	std::uniform_real_distribution<double> dist;		// The "coin" deciding when to flip stencil's sign.

	if( !phi )
		throw std::invalid_argument( errorPrefix + "phi vector can't be null!" );

	// Get indices for locally owned nodes next to Gamma (we need all of them).
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

	samples.clear();										// We'll get as many as points are next to Gamma.
	samples.reserve( indices.size() );

	// Prepare mean and Gaussian curvatures interpolation.
	my_p4est_interpolation_nodes_t kappaMGInterp( ngbd );
	kappaMGInterp.set_input( kappaMG, interpolation_method::linear, 2 );

	std::uniform_real_distribution<double> pDistribution;

#ifdef DEBUG
	std::cout << "Rank " << mpi.rank() << " reports " << indices.size() << " sample nodes." << std::endl;
#endif

	for( const auto& n : indices )
	{
		double xyz[P4EST_DIM];
		node_xyz_fr_n( n, p4est, nodes, xyz );

		std::vector<p4est_locidx_t> stencil;
		try
		{
			if( ABS( xyz[0] - xyzMin[0] ) <= 4 * h || ABS( xyz[0] - xyzMax[0] ) <= 4 * h ||	// Skip nodes too close
				ABS( xyz[1] - xyzMin[1] ) <= 4 * h || ABS( xyz[1] - xyzMax[1] ) <= 4 * h ||	// to domain boundary.
				ABS( xyz[2] - xyzMin[2] ) <= 4 * h || ABS( xyz[2] - xyzMax[2] ) <= 4 * h )
				continue;

			if( !nodesAlongInterface.getFullStencilOfNode( n , stencil ) )	// Does it have a valid stencil?
				continue;

			// Starting at this point, we collect samples.  We shouldn't get any negative-Gaussian sample because there
			// are no saddles in a sphere.
			double hk = h / radius;							// Target mean hk.
			for( int c = 0; c < P4EST_DIM; c++ )			// Find the location where to (linearly) interpolate curvatures.
				xyz[c] -= phiReadPtr[n] * normalsReadPtr[c][n];
			double kappaMGValues[2];
			kappaMGInterp( xyz, kappaMGValues );			// Get linearly interpolated mean and Gaussian curvature in one shot.
			double ihkVal = h * kappaMGValues[0];
			double ih2kgVal = SQR( h ) * kappaMGValues[1];
			if( ih2kgVal < 0 )								// Skip *numerical* saddles (which often appear for large radii).
				throw std::runtime_error( "collectSamples: Negative, invalid Gaussian curvature detected!" );

			if( ihkVal * hk < 0 )							// Well, this shouldn't happen!
				throw std::runtime_error( "collectSamples: Sign discrepancy between ihk and hk!" );

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

			if( flipSign && dist( gen ) <= 0.5 )
				kml::utils::normalizeToNegativeCurvature( *sample, hk, true );	// Change everything up to ihk; leave h2kg and ih2kg intact.
		}
		catch( std::runtime_error &e )
		{
			std::cerr << e.what() << std::endl;
		}
	}

#ifdef DEBUG
	std::cout << "Rank " << mpi.rank() << " collected " << samples.size() << " *unique* samples." << std::endl;
#endif
	kappaMGInterp.clear();

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
}

/**
 * Save buffered samples to a file by choosing numSamplesToSave of them randomly.
 * Upon exiting, the buffer will be emptied.
 * @param [in] mpi MPI environment.
 * @param [in,out] buffer Sample buffer.
 * @param [in,out] bufferSize Current buffer's size.
 * @param [in,out] file File where to write samples.
 * @param [in] numSamplesToSave How many samples do we want to store in file.
 * @param [in,out] gen Random engine for shuffling samples.
 * @return number of saved samples (already shared among processes).
 * @throws invalid_argument exception if buffer is empty or there are less samples than the number we intend to save.
 */
int saveSamples( const mpi_environment_t& mpi, vector<vector<FDEEP_FLOAT_TYPE>>& buffer, int& bufferSize, std::ofstream& file,
				 const u_int& numSamplesToSave, std::mt19937& gen, double& trackedMaxHKError, double& trackedMaxH2KGError )
{
	int savedSamples = 0;
	trackedMaxHKError = 0;
	trackedMaxH2KGError = 0;

	if( bufferSize > 0 && numSamplesToSave > 0 && numSamplesToSave <= bufferSize )	// We must have sufficient samples to save.
	{
		if( mpi.rank() == 0 )
		{
			std::vector<int> idxs( bufferSize );
			for( int k = 0; k < bufferSize; k++ )
				idxs[k] = k;
			std::shuffle( idxs.begin(), idxs.end(), gen );		// Shuffle indices.

			int i;
			for( i = 0; i < numSamplesToSave; i++ )
			{
				int idx = idxs[i];
				double errorHK = ABS( buffer[idx][K_INPUT_SIZE_LEARN - 4] - buffer[idx][K_INPUT_SIZE_LEARN - 3] );
				double errorH2KG = ABS( buffer[idx][K_INPUT_SIZE_LEARN - 2] - buffer[idx][K_INPUT_SIZE_LEARN - 1] );

				trackedMaxHKError = MAX( trackedMaxHKError, errorHK );
				trackedMaxH2KGError = MAX( trackedMaxH2KGError, errorH2KG );

				int j;
				for( j = 0; j < K_INPUT_SIZE_LEARN - 1; j++ )
					file << buffer[idx][j] << ",";				// Inner elements.
				file << buffer[idx][j] << std::endl;			// Last element is ihk in 2D or ih2kg in 3D.
			}
			savedSamples = i;

			buffer.clear();
		}
		bufferSize = 0;
	}
	else
		throw std::invalid_argument( "saveSamples: buffer is empty or there are less samples than intended number to save(?)!" );

	// Communicate to everyone the total number of saved samples and the errors.
	SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxHKError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &trackedMaxH2KGError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
	SC_CHECK_MPI( MPI_Bcast( &savedSamples, 1, MPI_INT, 0, mpi.comm() ) );

	return savedSamples;
}