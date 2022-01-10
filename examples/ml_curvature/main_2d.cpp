/**
 * Testing the hybrid curvature inference system on star-shaped interfaces.  The procedure here is an online specializa-
 * tion of the star_datasets_2d.cpp code base.  It requires the validation.json file in the same location as the neural
 * network.  To generate validation.json, execute the Evaluating.py module in the Hybrid_DL_Curvature python project.
 *
 * Tested on one and multiple processes.
 *
 * Developer: Luis Ángel.
 * Created: January 4, 2021.
 */

// System.
#include <stdexcept>
#include <iostream>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_curvature_ml.h>

#include <src/petsc_compatibility.h>
#include <src/casl_geometry.h>
#include <unordered_map>
#include <src/parameter_list.h>


int main ( int argc, char* argv[] )
{
	using json = nlohmann::json;

	// Setting up parameters from command line.
	param_list_t pl;
	param_t<unsigned short> maxRL( pl, 7, "maxRL", "Maximum level of refinement per unit-square quadtree (default: 7)" );
	param_t<unsigned int> reinitNumIters( pl, 10, "reinitNumIters", "Number of iterations for reinitialization (default: 10)" );
	param_t<std::string> nnetsDir( pl, "/Users/youngmin/k_nnets", "nnetsDir", "Folder where nnets are found (default: same folder as the executable)" );
	param_t<bool> exportVTK( pl, true, "exportVTK", "Export VTK file (default: 1)" );

	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		// Loading parameters from command line.
		cmdParser cmd;
		pl.initialize_parser( cmd );
		if( cmd.parse( argc, argv, "Testing error-correcting neural networks for curvature computation with star-shaped interfaces" ) )
			return 0;
		pl.set_from_cmd_all( cmd );

		/////////////////////////////////////////////// Parameter setup ////////////////////////////////////////////////

		const std::string ROOT = nnetsDir() + (P4EST_DIM > 2? "/3d/" : "/2d/") + std::to_string( maxRL() );	// NOLINT.

		const double H = 1. / (1 << maxRL());

		/////////////////////////////////////// Star-shaped interface parameters ///////////////////////////////////////

		// Steep curvature parameters: load these from the validation json file.  To generate the validation file, visit
		// the Evaluating module in the python project Hybrid_DL_Curvature.
		std::ifstream in( ROOT + "/validation.json" );
		json validation;
		in >> validation;
		const auto A = validation["a"].get<double>();
		const auto B = validation["b"].get<double>();
		const auto ARMS = validation["p"].get<int>();

		if( maxRL() != validation["maxRL"].get<int>() )
			throw std::runtime_error( "Validation maxRL and application maxRL don't match!" );

		geom::Star star( A, B, ARMS );			// Define the star interface and determine domain bound based on it.
		const double STAR_SIDE_LENGTH = star.getInscribingSquareSideLength() + 4 * H;
		const double MIN_D = -ceil( STAR_SIDE_LENGTH / 0.5 ) * 0.5;
		const double MAX_D = -MIN_D;			// The canonical space is square and has integral side lengths.

		// Load a hashmap with validation information entries of the form ("xcoord,ycoord,": [nodeIdx, ihk, hybhk, tgthk]).
		// We'll use it to check that the kml::Curvature class is computing curvature accurately per python project's
		// output.  For this to work, star_datasets_2d.cpp and this file must be in synchrony.
		std::unordered_map<std::string, std::vector<double>> validationMap;
		for( const auto& entry : validation["data"].get<std::vector<std::vector<double>>>() )
		{
			std::string key;
			int i;
			for( i = 0; i < P4EST_DIM; i++ )
				key += std::to_string( (int)entry[1+i] ) + ",";
			validationMap[key] = {entry[0], entry[i+1], entry[i+2], entry[i+3]};
		}

		///////////////// First, checking that we can load the neural network and scalers appropriately ////////////////

		const int N_SAMPLES = 2;
		double samples[N_SAMPLES][K_INPUT_SIZE] = {
			{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.9, -0.8},
			{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.1}
		};
		double outputs[N_SAMPLES];
		kml::NeuralNetwork nnet( ROOT, H, false );
		nnet.predict( samples, outputs, N_SAMPLES, false );		// If there's an error, it'll be thrown here.

		//////////////////////////////////// Test the sample star-shaped interface /////////////////////////////////////

		PetscPrintf( mpi.comm(), ">> Began test for a star-shaped interface with MAX_RL = %i and H = %f\n", maxRL(), H );

		// Domain information.
		const int N_TREES = (int)(MAX_D - MIN_D);
		int n_xyz[] = {N_TREES, N_TREES, N_TREES};			// Trees per dimension.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Square domain bounds.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic.

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the non-signed distance level-set function to be reinitialized.
		splitting_criteria_cf_and_uniform_band_t levelSetSC( MAX( 1, maxRL() - 5 ), maxRL(), &star, 7.0 );

		// Create the forest using a level set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)(&levelSetSC);

		// Refine and recursively partition forest.
		for( int i = 0; i < maxRL(); i++ )
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
		nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
											// how much we'll neeed the node neighbors.

		// A ghosted parallel PETSc vector to store level-set function values to be reinitialized.
		Vec phi;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

		// Calculate the level-set function values for independent nodes (i.e. locally owned and ghost nodes).
		sample_cf_on_nodes( p4est, nodes, star, phi );

		// Reinitialize level-set function.
		parStopWatch watch;
		watch.start();
		my_p4est_level_set_t ls( &nodeNeighbors );
		ls.reinitialize_2nd_order( phi, (int)reinitNumIters() );

		// Compute normal unit vectors: we'll use them to compute the numerical curvature.
		Vec numCurvature, hybHK, hybFlag, normal[P4EST_DIM];
		CHKERRXX( VecDuplicate( phi, &numCurvature ) );
		CHKERRXX( VecDuplicate( phi, &hybHK ) );
		CHKERRXX( VecDuplicate( phi, &hybFlag ) );
		for( auto& dim : normal )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

		compute_normals( nodeNeighbors, phi, normal );
		double prepTime = watch.get_duration_current();

		// Compute hybrid (dimensionless) curvature.
		kml::Curvature mlCurvature( &nnet, H );
		std::pair<double, double> durations = mlCurvature.compute( nodeNeighbors, phi, normal, numCurvature, hybHK, hybFlag, true, &watch );
		watch.stop();

		const double *hybHKReadPtr, *hybFlagReadPtr, *phiReadPtr, *normalReadPtr[P4EST_DIM];
		CHKERRXX( VecGetArrayRead( hybHK, &hybHKReadPtr ) );
		CHKERRXX( VecGetArrayRead( hybFlag, &hybFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecGetArrayRead( normal[i], &normalReadPtr[i] ) );

		// Perform validation against data from the off-line inference to check correctness of online inference.
		double xyz[P4EST_DIM];
		const FDEEP_FLOAT_TYPE FLOAT_32_EPS = 1. / (1 << 23);
		char msg[256];
		size_t totalNodesEvaluated = 0;
		size_t matchingNodes = 0;
		for( p4est_locidx_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
		{
			if( hybFlagReadPtr[n] == 1 )			// Used hybrid approach on the nth node?
			{
				std::string key;
				totalNodesEvaluated++;
				node_xyz_fr_n( n, p4est, nodes, xyz );
				for( const double& coord : xyz )
					key += std::to_string( (int)(coord / H) ) + ",";

				auto got = validationMap.find( key );
				if( got != validationMap.end() )
				{
					auto offlineHK = (FDEEP_FLOAT_TYPE)got->second[2];							// Validation hybrid hk.
					auto onlineHK = (FDEEP_FLOAT_TYPE)hybHKReadPtr[n];
					if( ABS( onlineHK - offlineHK ) < FLOAT_32_EPS )							// A match?
					{
						matchingNodes++;
						validationMap.erase( got );	// If everything goes well, the validation map should end up empty.
					}
					else
					{
						const char *coords = key.c_str();
						sprintf( msg, "xxxx Rank %i: Node %i's (%s) hybHK %.7f doesn't match with the offline value %.7f!",
								 mpi.rank(), n, coords, onlineHK, offlineHK );
						std::cerr << msg << std::endl;
					}
				}
				else
				{
					sprintf( msg, "---- Rank %i: Node %i is not in the validation map!", mpi.rank(), n );
					std::cerr << msg << std::endl;
				}
			}
		}

		if( mpi.size() == 1 && !validationMap.empty() )
			std::cerr << "\nValidation map is not empty, it has " << validationMap.size() << " elements!" << std::endl;

		printf( "\n<< Rank %i: Done!  Matching nodes = %zu/%zu."
				"\n   It took %f seconds to compute the numerical curvature."
				"\n   And %f seconds for the hybrid one.",
				mpi.rank(), matchingNodes, totalNodesEvaluated, durations.first + prepTime, durations.second + prepTime );

		// Write paraview file to visualize the star interface and nodes following it along -- mainly for debugging.
		if( exportVTK() )
		{
			char auxA[10], auxB[10];
			sprintf( auxA, "%.3f", A );
			sprintf( auxB, "%.3f", B );

			std::ostringstream oss;
			oss << "star_a_" << std::string( auxA ) << "_b_" << std::string( auxB );
			my_p4est_vtk_write_all( p4est, nodes, ghost,
									P4EST_TRUE, P4EST_TRUE,
									3 + P4EST_DIM, 0, oss.str().c_str(),
									VTK_POINT_DATA, "phi", phiReadPtr,
									VTK_POINT_DATA, "flag", hybFlagReadPtr,
									VTK_POINT_DATA, "hybHK", hybHKReadPtr,
									VTK_POINT_DATA, "nx", normalReadPtr[0],
									VTK_POINT_DATA, "ny", normalReadPtr[1]
#ifdef P4_TO_P8
									, VTK_POINT_DATA, "nz", normalReadPtr[2]
#endif
									);
		}

		// Cleaning up.
		for( int i = 0; i < P4EST_DIM; i++ )
			CHKERRXX( VecRestoreArrayRead( normal[i], &normalReadPtr[i] ) );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( hybHK, &hybHKReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( hybFlag, &hybFlagReadPtr ) );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		CHKERRXX( VecDestroy( phi ) );
		CHKERRXX( VecDestroy( numCurvature ) );
		CHKERRXX( VecDestroy( hybHK ) );
		CHKERRXX( VecDestroy( hybFlag ) );
		for( auto& dim : normal )
			CHKERRXX( VecDestroy( dim ) );
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}