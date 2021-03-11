/**
 * Title: ml_mass_conservation
 *
 * Description: Data set generation for training a neural network that corrects the semi-Lagrangian scheme for simple
 * advection.  We assume that all considered velocity fields are divergence-free.  To generate these velocity fields, we
 * obtain the skew gradient of random Gaussians.
 * @note Not yet tested on 3D.
 *
 * Author: Luis Ángel (임 영민)
 * Date Created: 01-20-2021
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#endif

#include <iostream>
#include <src/petsc_compatibility.h>
#include <src/casl_geometry.h>
#include "CoarseGrid.h"
#include "VelocityField.h"
#include "Utils.h"
#include <random>

/**
 * Main function.
 * @param argc Number of input arguments.
 * @param argv Actual arguments.
 * @return 0 if process finished successfully, nonzero otherwise.
 */
int main( int argc, char** argv )
{
	// Main global variables.
	const double DURATION = 0.5;		// Max duration of the simulation (unless a backtracked interface point fall outside the domain).
	const int COARSE_MAX_RL = 6;		// Maximum refinement levels for coarse and fine grids.
	const int FINE_MAX_RL = 8;
	const int REINIT_NUM_ITER = 10;		// Number of iterations for level-set renitialization.

	const double CFL = 1.0;				// Courant-Friedrichs-Lewy condition.
	const auto PHI_INTERP_MTHD = interpolation_method::quadratic;		// Phi interpolation method.
	const auto VEL_INTERP_MTHD = interpolation_method::quadratic;		// Velocity interpolation method.

	const double MIN_D = -1;			// Square domain minimum and maximum values for each dimension.  Omega is
	const double MAX_D = -MIN_D;		// centered at the origin of global coordinate system.
	const int NUM_TREES_PER_DIM = 2;	// Number of macro cells per dimension.

	const int NUM_ITER_VTK = 8;			// Save VTK files every NUM_ITER_VTK iterations (break line too).

	const int NUM_VEL_FIELDS = 7;		// Number of different random velocity fields to try out.
	const int NUM_CENTERS = 4;			// Number of different center locations to try out per circle radius.

	const double BAND_C = 2; 			// Minimum number of cells around interface in COARSE (C) and FINE (F) grids.
	const double BAND_F = BAND_C * (1u << (FINE_MAX_RL - COARSE_MAX_RL));

	char msg[1024];						// Some string to write messages to standard ouput.

	// Destination folder and file.
	const std::string DATA_PATH = "/Volumes/YoungMinEXT/massLoss/";
	const std::string FILE_PATH = DATA_PATH + "data_" + std::to_string( COARSE_MAX_RL ) + "_"
													  + std::to_string( FINE_MAX_RL ) + "_k_minus.csv";
	const int NUM_COLUMNS = 21;			// Number of columns in resulting dataset.
	std::string COLUMN_NAMES[NUM_COLUMNS] = {"phi_a",				// Level-set value at arrival point.
											 "u_a", "v_a", 			// Velocity components at arrival point.
											 "d",					// Scaled distance (by 1/dx_coarse).
											 "x_d", "y_d",			// Scaled departure coords with respect to quad's lower corner.
											 "phi_d_mm", "phi_d_mp", "phi_d_pm", "phi_d_pp",	// Level-set values at departure quad's children.
											 "u_d_mm", "u_d_mp", "u_d_pm", "u_d_pp",			// Velocity component u at departure quad's children.
											 "v_d_mm", "v_d_mp", "v_d_pm", "v_d_pp",			// Velocity component v at departure quad's children.
											 "target_phi_d",		// Expected/target phi value at departure point.
											 "numerical_phi_d",		// Numerically computed phi value at departure point.
											 "numerical_kappa"		// Mean curvature at closest point to node on Gamma.
	};

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;			// PETSc error flag code.

		// To generate data sets we don't admit more than a single process to avoid race conditions.
		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		parStopWatch watch;
		watch.start();

		sprintf( msg, ">> Began to generate data sets for MAX_RL_COARSE = %d and MAX_RL_FINE = %d\n",
		   		 COARSE_MAX_RL, FINE_MAX_RL );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );

		// Prepare output file.
		std::ofstream file;
		utils::openFile( FILE_PATH, 15, file );

		// Write header columns.
		for( int i = 0; i < NUM_COLUMNS - 1; i++ )
			file << "\"" << COLUMN_NAMES[i] << "\",";
		file << "\"" << COLUMN_NAMES[NUM_COLUMNS - 1] << "\"" << std::endl;

		// Domain information: a square with the same number of trees per dimension.
		const int n_xyz[] = {NUM_TREES_PER_DIM, NUM_TREES_PER_DIM, NUM_TREES_PER_DIM};
		const double xyz_min[] = {MIN_D, MIN_D, MIN_D};
		const double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		const double mesh_len[] = {MAX_D - MIN_D, MAX_D - MIN_D, MAX_D - MIN_D};
		const int periodic[] = {1, 1, 1};
		const double dxyz_min_c = (mesh_len[0] / NUM_TREES_PER_DIM) / double(1 << COARSE_MAX_RL);	// Min coarse cell width.

		// Let's choose radii between 5 COARSE min cell width and an eighth of domain side length.
		const double MIN_RADIUS = 5 * dxyz_min_c;
		const double MAX_RADIUS = MAX_D / 4;
		const int NUM_RADII = (int)ceil( 3 * (MAX_RADIUS - MIN_RADIUS) / dxyz_min_c ) + 1;
		std::vector<double> radii;
		linspace( MIN_RADIUS, MAX_RADIUS, NUM_RADII, radii );

		std::mt19937 gen; 			// NOLINT Standard mersenne_twister_engine with default seed for repeatability.
		std::uniform_real_distribution<double> uniformDistributionAroundCenter( MIN_D / 2.0, MAX_D / 2.0 );

		// Subsampling to reduce number of observations along flat regions.
		std::mt19937 gen2( 37 );	// NOLINT Standard Mersenne Twister engine for subsampling.
		auto subSampling = []( double kappa ){
			kappa = ABS( kappa );
			if( kappa <= 5 )		// Choose how manu "data augmentations" do we need depending on curvature.
				return 1;
			if( kappa <= 7.5 )
				return 2;
			if( kappa <= 10 )
				return 3;
			return 4;
		};

		// Looping through random velocity fields.
		unsigned long nSamplesInTotal = 0;
		for( int nvel = 0; nvel < NUM_VEL_FIELDS; nvel++ )
		{
			sprintf( msg, "\n---------------------------- Velocity Field %03d ----------------------------\n", nvel );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );

			// Defining a random velocity field, normalized to unit length (and dump velocity field files for analysis).
			RandomVelocityField randomVelocityField( gen );
			randomVelocityField.normalize( xyz_min, mesh_len, 1 << FINE_MAX_RL, true, nvel );

			unsigned long nSamplesPerVelocityField = 0;

			// Looping through the radii.
			for( const auto& radius : radii )
			{
				sprintf( msg, "++++++++++++++++++++++++++++ Radius %g ++++++++++++++++++++++++++++\n", radius );
				ierr = PetscPrintf( mpi.comm(), msg );
				CHKERRXX( ierr );

				unsigned long nSamplesPerRadius = 0;

				// Looping thorugh the random centers for circular-interface initial level-set functions.
				for( int ncenter = 0; ncenter < NUM_CENTERS; ncenter++ )
				{
					// Define the initial interface (valid for COARSE and FINE grids).
					double x0 = uniformDistributionAroundCenter( gen );
					double y0 = uniformDistributionAroundCenter( gen );
					geom::SphereNSD sphere( DIM( x0, y0, 0 ), radius );

					// Declaration of the FINE macromesh via the brick and connectivity objects.
					my_p4est_brick_t brick_f;
					p4est_connectivity_t *connectivity_f;
					connectivity_f = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick_f, periodic );

					// Pointers to FINE p4est variables.
					p4est_t *p4est_f;
					p4est_ghost_t *ghost_f;
					p4est_nodes_t *nodes_f;

					// Create the FINE forest using a level-set as refinement criterion.
					p4est_f = my_p4est_new( mpi.comm(), connectivity_f, 0, nullptr, nullptr );
					splitting_criteria_cf_and_uniform_band_t lsSplittingCriterion_f( 1, FINE_MAX_RL, &sphere, BAND_F );
					p4est_f->user_pointer = &lsSplittingCriterion_f;

					// Refine and partition the FINE forest.
					for( int i = 0; i < FINE_MAX_RL; i++ )
					{
						my_p4est_refine( p4est_f, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
						my_p4est_partition( p4est_f, P4EST_FALSE, nullptr );
					}

					// Create the FINE ghost (cell) and node structures.
					ghost_f = my_p4est_ghost_new( p4est_f, P4EST_CONNECT_FULL );
					nodes_f = my_p4est_nodes_new( p4est_f, ghost_f );

					// Initialize the FINE neighbor node structure.
					auto *hierarchy_f = new my_p4est_hierarchy_t( p4est_f, ghost_f, &brick_f );
					auto *nodeNeighbors_f = new my_p4est_node_neighbors_t( hierarchy_f, nodes_f );
					nodeNeighbors_f->init_neighbors();

					// Create a coarse grid.
					CoarseGrid coarseGrid( mpi, n_xyz, xyz_min, xyz_max, periodic, BAND_C, COARSE_MAX_RL, &sphere );

					// Retrieve FINE grid size data.
					double dxyz_f[P4EST_DIM];
					double dxyz_min_f;
					double diag_min_f;
					get_dxyz_min( p4est_f, dxyz_f, dxyz_min_f, diag_min_f );

					// Declare data vectors for FINE grid.
					Vec phi_f;
					Vec vel_f[P4EST_DIM];

					// Allocate memory for FINE phi and vel Vecs.
					ierr = VecCreateGhostNodes( p4est_f, nodes_f, &phi_f );
					CHKERRXX( ierr );
					for( auto& dir : vel_f )
					{
						ierr = VecCreateGhostNodes( p4est_f, nodes_f, &dir );
						CHKERRXX( ierr );
					}

					// Sample the level-set function at t = 0 at all independent nodes of FINE grid.
					sample_cf_on_nodes( p4est_f, nodes_f, sphere, phi_f );

					// Sample the velocity field at t = 0 at all independent nodes of FINE and COARSE grids.
					randomVelocityField.evaluate( mesh_len, p4est_f, nodes_f, vel_f );
					randomVelocityField.evaluate( mesh_len, coarseGrid.p4est, coarseGrid.nodes, coarseGrid.vel );

					// Reinitialize both FINE and COARSE grids before we start advections as we are using a
					// non-signed distance level-set function.
					my_p4est_level_set_t levelSet_f( nodeNeighbors_f );				// FINE grid.
					levelSet_f.reinitialize_2nd_order( phi_f, REINIT_NUM_ITER );
					my_p4est_level_set_t levelSet_c( coarseGrid.nodeNeighbors );	// Coarse grid.
					levelSet_c.reinitialize_2nd_order( coarseGrid.phi, REINIT_NUM_ITER );

					// Define time stepping variables.
					double tn_c = 0;					// Current time for COARSE grid.
					double tn_f = 0;					// Current time for FINE grid.
					int iter = 0;
					const double MAX_VEL_NORM = 1.0; 	// Maximum velocity length known after normalizing random field.
					double dt_c = CFL * coarseGrid.minCellWidth / MAX_VEL_NORM;	// deltaT for COARSE grid.
					double dt_f = CFL * dxyz_min_f / MAX_VEL_NORM;				// FINE deltaT knowing that the CFL
																				// cond. is (c * deltaT)/deltaX <= CFLN.
					bool allInside = true;				// Turns false if at least one interface backtracked point in
														// the coarse grid falls outside the computational domain.

					// Advection loop.
					// For each COARSE step, there are 2^(FINE_MAX_RL - COARSE_MAX_RL) FINE steps.
					const int N_FINE_STEPS_PER_COARSE_STEP = 1u << (FINE_MAX_RL - COARSE_MAX_RL);
					unsigned long nSamplesPerLoop = 0;	// Count how many samples we collect for a simulation loop.
					double maxRelError = 0;				// Maximum relative error (w.r.t. COARSE cell width) for loop.

					ierr = PetscPrintf( mpi.comm(), " * Receiving: " );			// Labeling reception of packets.
					CHKERRXX( ierr );
					while( allInside && tn_c + 0.1 * dt_c < DURATION )
					{
						// Clip leading COARSE time step if it's going to go over the final time.
						if( tn_c + dt_c > DURATION )
							dt_c = DURATION - tn_c;

						// Display current iteration.  After this, we show packets collected along Gamma.
						sprintf( msg, "[%03d]: ", iter );
						ierr = PetscPrintf( mpi.comm(), msg );
						CHKERRXX( ierr );

						// Update up to N_FINE_STEPS_PER_COARSE_STEP times the FINE grid.
						for( int step = 0; step < N_FINE_STEPS_PER_COARSE_STEP; step++ )
						{
							if( step == N_FINE_STEPS_PER_COARSE_STEP - 1 )	// In last step, stretch the FINE step so
							{												// that tn_f matches tn_c + dt_c.
								dt_f = tn_c + dt_c - tn_f;
							}
							else if( tn_f + dt_f > tn_c + dt_c )			// Is FINE step going over COARSE step?
							{
								dt_f = tn_c + dt_c - tn_f;
								step = N_FINE_STEPS_PER_COARSE_STEP - 1;
							}

							// Declare auxiliary FINE p4est objects; they will be updated during the semi-Lagrangian
							// advection step.
							p4est_t *p4est_f1 = p4est_copy( p4est_f, P4EST_FALSE );
							p4est_ghost_t *ghost_f1 = my_p4est_ghost_new( p4est_f1, P4EST_CONNECT_FULL );
							p4est_nodes_t *nodes_f1 = my_p4est_nodes_new( p4est_f1, ghost_f1 );

							// Create FINE semi-lagrangian object.
							my_p4est_semi_lagrangian_t semiLagrangian_f( &p4est_f1, &nodes_f1, &ghost_f1, nodeNeighbors_f );
							semiLagrangian_f.set_phi_interpolation( PHI_INTERP_MTHD );
							semiLagrangian_f.set_velo_interpolation( VEL_INTERP_MTHD );

							// Advect the FINE level-set function one step, then update the grid.
							semiLagrangian_f.update_p4est_one_vel_step( vel_f, dt_f, phi_f, BAND_F );

							// Destroy old FINE forest and create new structures.
							p4est_destroy( p4est_f );
							p4est_f = p4est_f1;
							p4est_ghost_destroy( ghost_f );
							ghost_f = ghost_f1;
							p4est_nodes_destroy( nodes_f );
							nodes_f = nodes_f1;

							delete hierarchy_f;
							delete nodeNeighbors_f;
							hierarchy_f = new my_p4est_hierarchy_t( p4est_f, ghost_f, &brick_f );
							nodeNeighbors_f = new my_p4est_node_neighbors_t( hierarchy_f, nodes_f );
							nodeNeighbors_f->init_neighbors();

							// Reinitialize FINE level-set function.
							my_p4est_level_set_t levelSet_f1( nodeNeighbors_f );
							levelSet_f1.reinitialize_2nd_order( phi_f, REINIT_NUM_ITER );

							// Advance FINE time.
							tn_f += dt_f;

							// Reconstruct and resample the random velocity field on new FINE grid.
							for( auto& dir : vel_f )
							{
								ierr = VecDestroy( dir );
								CHKERRXX( ierr );
								ierr = VecCreateGhostNodes( p4est_f, nodes_f, &dir );
								CHKERRXX( ierr );
							}
							randomVelocityField.evaluate( mesh_len, p4est_f, nodes_f, vel_f );
						}

						// Restore FINE time step size.
						dt_f = CFL * dxyz_min_f / MAX_VEL_NORM;

						// Collect samples from COARSE grid: must be done before "advecting" COARSE grid.
						// Also allocates flagged nodes along Gamma.
						std::vector<slml::DataPacket *> dataPackets;
						double relError = 0;
						allInside = coarseGrid.collectSamples( nodeNeighbors_f, phi_f, dt_c, dataPackets, relError );

						if( allInside )	// Continue processing samples if all backtracked interface points lied inside
						{				// the domain.
							// Data augmentation and writing to output file.
							// Accumulate all samples first, and then write file.
							std::vector<std::vector<double>> samples;
							samples.reserve( dataPackets.size() * 4 );
							for( auto dataPacket : dataPackets )
							{
								// As a way to simplify the resulting nnet architecture, I'll make everything match
								// negative-curvature samples by flipping the sign of level-set values in data packet.
								if( dataPacket->numK > 0 )
								{
									dataPacket->phi_a *= -1;
									for( auto& phi_d : dataPacket->phi_d )
										phi_d *= -1;
									dataPacket->targetPhi_d *= -1;
									dataPacket->numBacktrackedPhi_d *= -1;
									dataPacket->numK *= -1;
								}

								// I'll record samples based on their curvature.  Anything below a threshold will be
								// subsampled, and anything above will be given green light (and augmented four times).
								int takeNSamples = subSampling( dataPacket->numK );
								std::unordered_set<int> rotations;
								if( takeNSamples < 4 )
									utils::generateRandomSetOfNumbers( 0, 3, takeNSamples, rotations, gen2 );

								for( int i = 0; i < 4; i++ )	// For each received data packet, rotate it 4 times.
								{
									if( takeNSamples == 4 || rotations.find( i ) != rotations.end() )
									{
										samples.emplace_back( std::vector<double>());	// Serialize data.
										samples.back().reserve( NUM_COLUMNS );
										dataPacket->serialize( samples.back());
									}

									dataPacket->rotate90();								// Data augmentation.
								}
							}

							// Write samples vector to file.
							for( const auto& row : samples )
							{
								// Write inner elements separted by commas and leave the last without trailing comma.
								std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( file, "," ) );
								file << row.back() << std::endl;
							}

							// As a way to verify things are working well, we track the maximum relative error.
							maxRelError = MAX( maxRelError, relError );
							nSamplesPerLoop += samples.size();

							// Break data packet reception output every now and then.
							if( (iter + 1) % NUM_ITER_VTK == 0 )
							{
								ierr = PetscPrintf( mpi.comm(), "\n              " );
								CHKERRXX( ierr );
							}

							// "Advecting" COARSE grid by using the FINE grid as reference: updates internal phi vector
							// with NO reinitialization afterwards.
							coarseGrid.fitToFineGrid( nodeNeighbors_f, phi_f );

							// Resample the random velocity field on new COARSE grid.
							randomVelocityField.evaluate( mesh_len, coarseGrid.p4est, coarseGrid.nodes, coarseGrid.vel );

							// Advance COARSE time step.
							tn_c += dt_c;
							tn_f = tn_c;						// Synchronize COARSE and FINE times.
							iter++;

							// Don't forget to destroy dynamic objects.  If not all points were backtracked inside the
							// domain, CoarseGrid::collectSamples has called freed packets already.
							slml::SemiLagrangian::freeDataPacketArray( dataPackets );
						}
					}

					// Signal how we stopped the simulation loop.
					if( !allInside )
						ierr = PetscPrintf( mpi.comm(), "///\n" );	// At least one point along Gamma was backtracked
					else											// outside Omega.
						ierr = PetscPrintf( mpi.comm(), "###\n" );	// All points along Gamma were backtracked inside
					CHKERRXX( ierr );								// Omega.

					// Status message.
					sprintf( msg, "= %lu samples collected, maxRelError = %g, after %g seconds.\n",
							 nSamplesPerLoop, maxRelError, watch.get_duration_current() );
					ierr = PetscPrintf( mpi.comm(), msg );
					CHKERRXX( ierr );

					// Destroy the dynamically allocated Vecs for FINE grid.
					ierr = VecDestroy( phi_f );
					CHKERRXX( ierr );
					for( auto& dir : vel_f )
					{
						ierr = VecDestroy( dir );
						CHKERRXX( ierr );
					}

					// Destroy the dynamically allocated FINE p4est and my_p4est structures.
					delete hierarchy_f;
					delete nodeNeighbors_f;
					p4est_nodes_destroy( nodes_f );
					p4est_ghost_destroy( ghost_f );
					p4est_destroy( p4est_f );

					// Destroy the dynamically allocated FINE brick and connectivity structures.
					// Connectivity and Brick objects are the only ones that are not re-created in every iteration of
					// semi-Lagrangian advection.
					my_p4est_brick_destroy( connectivity_f, &brick_f );

					coarseGrid.destroy();

					nSamplesPerRadius += nSamplesPerLoop;
				}

				nSamplesPerVelocityField += nSamplesPerRadius;

				// Status message for radius.
				sprintf( msg, "+ %lu samples collected for RADIUS %g after %g seconds\n",
			 			 nSamplesPerRadius, radius, watch.get_duration_current() );
				ierr = PetscPrintf( mpi.comm(), msg );
				CHKERRXX( ierr );
			}

			nSamplesInTotal += nSamplesPerVelocityField;

			// Status message for velocity field.
			sprintf( msg, "- %lu samples collected for VELOCITY FIELD %3d after %g seconds\n",
					 nSamplesPerVelocityField, nvel, watch.get_duration_current() );
			ierr = PetscPrintf( mpi.comm(), msg );
			CHKERRXX( ierr );
		}

		file.close();
		watch.stop();

		// Status message for whole data collection.
		sprintf( msg, "\n<< Finished data collecting %lu samples after %g seconds",
		   		 nSamplesInTotal, watch.get_duration_current() );
		ierr = PetscPrintf( mpi.comm(), msg );
		CHKERRXX( ierr );
	}
	catch( const std::exception &exception )
	{
		std::cerr << exception.what() << std::endl;
	}
}