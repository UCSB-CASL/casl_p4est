/**
 * Testing the merging spheres in 2D.
 *
 * Developer: Luis Ángel.
 * Date: August 5, 2020.
 */

// System.
#include <stdexcept>
#include <iostream>

#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_nodes_along_interface.h>
#include <src/my_p4est_level_set.h>

#include <src/petsc_compatibility.h>
#include <random>
#include "two_spheres_levelset_2d.h"


int main ( int argc, char* argv[] )
{
	///////////////////////////////////////////////////// Metadata /////////////////////////////////////////////////////

	const double MIN_D = -1, MAX_D = -MIN_D;							// The canonical space is [-1, +1]^2.
	const int MAX_REFINEMENT_LEVEL = 7;									// Maximum and minimum levels of refinement.
	const int MIN_REFINEMENT_LEVEL = 1;
	const double H = 1 / pow( 2, MAX_REFINEMENT_LEVEL );				// Highest spatial resolution in x/y directions.
	const int N_NODES_PER_DIM = (int)((MAX_D - MIN_D) / H + 1);
	double M[N_NODES_PER_DIM][N_NODES_PER_DIM];							// Matrix to organize output for Matlab.

	const int STENCIL_SIZE = (int)pow( 3, P4EST_DIM );					// Stencil size.

	// Random-number generator (https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).
	std::random_device rd;  					// Will be used to obtain a seed for the random number engine.
	std::mt19937 gen; 							// Standard mersenne_twister_engine seeded with rd().
	std::uniform_real_distribution<double> uniformDistributionH_2( -H / 2, +H / 2 );
	std::normal_distribution<double> normalDistribution;

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );
		PetscErrorCode ierr;

		// To test we don't admit more than a single process.
		if( mpi.rank() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		std::cout << "Testing two-sphere level-set function..." << std::endl;

		// Domain information.
		int n_xyz[] = {2, 2, 2};							// Two trees per dimension, each occupying a unit cube.
		double xyz_min[] = {MIN_D, MIN_D, MIN_D};			// Domain bounds.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};
		int periodic[] = {0, 0, 0};							// Non-periodic domain.

		double maxRE = 0;									// Maximum relative error.

		const double T[] = {
			uniformDistributionH_2( gen ),		// Translate center coords by a randomly chosen
			uniformDistributionH_2( gen )		// perturbation from the grid's midpoint.
		};

		// p4est variables and data structures: these change with every sine wave because we must refine the
		// trees according to the new waves's origin and amplitude.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Definining the level-set function to be reinitialized.
		const double HALF_AXIS_LEN = 0.5;
		const double TILT = -0.39269908169872414;
		const double X1 = -( HALF_AXIS_LEN * cos( -TILT ) + -0.0037645302580557806 );		// Center coordinates of reference circle.
		const double Y1 = HALF_AXIS_LEN * sin( -TILT ) + -0.0030244556919302671;
		const double R1 = 0.48437500000000044;											// Radii.
		const double R2 = 0.062724820143884932;
		const double D = 0.54702488009592376;
		TwoSpheres twoSpheres( X1, Y1, R1, R2, D, TILT );
		splitting_criteria_cf_t levelSetSC( MIN_REFINEMENT_LEVEL, MAX_REFINEMENT_LEVEL, &twoSpheres );

		// Create the forest using a level-set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *) ( &levelSetSC );

		// Refine and recursively partition forest.
		my_p4est_refine( p4est, P4EST_TRUE, refine_levelset_cf, nullptr );
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t nodeNeighbors( &hierarchy, nodes );
		nodeNeighbors.init_neighbors(); 	// This is not mandatory, but it can only help performance given
											// how much we'll neeed the node neighbors.

		// A ghosted parallel PETSc vector to store level-set function values.
		Vec phi;
		ierr = VecCreateGhostNodes( p4est, nodes, &phi );
		CHKERRXX( ierr );

		// Calculate the level-set function values for each independent node (i.e. locally owned and ghost nodes).
		sample_cf_on_nodes( p4est, nodes, twoSpheres, phi );

		// Vectors for curvature and normals.
		Vec curvature, normal[P4EST_DIM];
		ierr = VecDuplicate( phi, &curvature );
		CHKERRXX( ierr );
		for( auto& dim : normal )
		{
			VecCreateGhostNodes( p4est, nodes, &dim );
			CHKERRXX( ierr );
		}

		// Reinitialize level-set function.
		my_p4est_level_set_t ls( &nodeNeighbors );
		ls.reinitialize_2nd_order( phi, 10 );

		// Compute numerical curvature with reinitialized data, which will be interpolated at the interface.
		compute_normals( nodeNeighbors, phi, normal );
		compute_mean_curvature( nodeNeighbors, phi, normal, curvature );

		// Prepare interpolation.
		my_p4est_interpolation_nodes_t interpolation( &nodeNeighbors );
		interpolation.set_input( curvature, linear );

		// Once the level-set function is reinitialized, collect nodes on or adjacent to the interface; these are
		// the points we'll use to create our sample files.
		NodesAlongInterface nodesAlongInterface( p4est, nodes, &nodeNeighbors, MAX_REFINEMENT_LEVEL );

		// An interface flag vector to distinguish nodes along the interface with full uniform neighborhoods.
		Vec interfaceFlag;
		ierr = VecDuplicate( phi, &interfaceFlag );
		CHKERRXX( ierr );

		double *interfaceFlagPtr;
		ierr = VecGetArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );
		for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
			interfaceFlagPtr[i] = 0;		// Init to zero and set flag of (valid) nodes along interface to 1 below.

		// A vector to store the signed distance function for all nodes.
		Vec distPhi;
		ierr = VecDuplicate( phi, &distPhi );
		CHKERRXX( ierr );

		double *distPhiPtr;
		ierr = VecGetArray( distPhi, &distPhiPtr );
		CHKERRXX( ierr );

		// A vector to store which circle produces the closest distance to each node.  This is later used to find
		// troubling nodes.
		Vec who;
		ierr = VecDuplicate( phi, &who );
		CHKERRXX( ierr );

		double *whoPtr;
		ierr = VecGetArray( who, &whoPtr );
		CHKERRXX( ierr );

		// A vector to store the dimensionless curvature at nodes along the interface.
		Vec hKappa;
		ierr = VecDuplicate( interfaceFlag, &hKappa );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, hKappa );
		CHKERRXX( ierr );

		double *hKappaPtr;
		ierr = VecGetArray( hKappa, &hKappaPtr );
		CHKERRXX( ierr );

		for( p4est_locidx_t i = 0; i < nodes->indep_nodes.elem_count; i++ )		// Calculating the signed-distance and
		{																		// dimensionless curvature.
			double xyz[P4EST_DIM], xOnGamma, yOnGamma;							// Last two are dummy vars to avoid error.
			node_xyz_fr_n( i, p4est, nodes, xyz );
			distPhiPtr[i] = twoSpheres.getSignedDistance( xyz[0], xyz[1], H, hKappaPtr[i], whoPtr[i], xOnGamma, yOnGamma );
			if( MIN_REFINEMENT_LEVEL == MAX_REFINEMENT_LEVEL )
			{
				int I = ( int ) round((xyz[0] - MIN_D) / H );
				int J = ( int ) round((xyz[1] - MIN_D) / H );
				M[I][J] = distPhiPtr[i];
			}
		}

		// A vector to store the maximum absolute error at interface nodes.
		Vec nError;
		ierr = VecDuplicate( interfaceFlag, &nError );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, nError );
		CHKERRXX( ierr );

		double *nErrorPtr;
		ierr = VecGetArray( nError, &nErrorPtr );
		CHKERRXX( ierr );

		// A vector to store the h*kappa error at the interface nodes.
		Vec hkError;
		ierr = VecDuplicate( interfaceFlag, &hkError );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, hkError );
		CHKERRXX( ierr );

		double *hkErrorPtr;
		ierr = VecGetArray( hkError, &hkErrorPtr );
		CHKERRXX( ierr );

		// A vector to store the gradient quality metric: Q(phi) = |1-|grad(phi)||.
		Vec gradQuality;
		ierr = VecDuplicate( interfaceFlag, &gradQuality );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, gradQuality );
		CHKERRXX( ierr );

		double *gradQualityPtr;
		ierr = VecGetArray( gradQuality, &gradQualityPtr );
		CHKERRXX( ierr );

		// A vector to store the node trouble indicator.
		Vec troubling;
		ierr = VecDuplicate( interfaceFlag, &troubling );
		CHKERRXX( ierr );

		ierr = VecCopy( interfaceFlag, troubling );
		CHKERRXX( ierr );

		double *troublingPtr;
		ierr = VecGetArray( troubling, &troublingPtr );
		CHKERRXX( ierr );

		// Getting the full uniform stencils of interface points.
		std::vector<p4est_locidx_t> indices;
		nodesAlongInterface.getIndices( &phi, indices );

		const double *phiReadPtr;
		ierr = VecGetArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Now, collect samples with reinitialized level-set function values and target h\kappa.
		for( auto n : indices )
		{
			double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
			node_xyz_fr_n( n, p4est, nodes, xyz );

			std::vector<p4est_locidx_t> stencil;	// Contains 9 values in 2D.
			try
			{
				if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
				{
					std::vector<double> sdfData;	// Signed-distance function values.
					std::vector<double> rlsData;	// Reinitialized level-set function values.
					bool troublingFlag = whoPtr[n] == 0;	// A troubling node is that who lies on a discontinuity or
															// if at least one of its 9-point stencil neighbors lies on
															// a discontinuity or its phi value is due to a circle
															// different to n's.

					for( auto s : stencil )
					{
						sdfData.push_back( distPhiPtr[s] );
						rlsData.push_back( phiReadPtr[s] );

						// Error for standing stencil.
						double error = ABS( sdfData.back() - rlsData.back() ) / H;
						nErrorPtr[n] = MAX( nErrorPtr[n], error );

						// Checking for trouble.
						if( whoPtr[s] != whoPtr[n] )
							troublingFlag = true;
					}

					maxRE = MAX( maxRE, nErrorPtr[n] );
					interfaceFlagPtr[n] = 1;

					// Computing the gradient and the quality metric, using centered differences to approx. derivatives.
//					double grad[P4EST_DIM] = { (rlsData[7] - rlsData[1]) / (2 * H), (rlsData[5] - rlsData[3]) / (2 * H) };
//					double gradNorm = sqrt( SQR( grad[0] ) + SQR( grad[1] ) );
					double grad[P4EST_DIM];					// Getting its gradient (i.e. normal).
					const quad_neighbor_nodes_of_node_t *qnnnPtr;
					nodeNeighbors.get_neighbors( n, qnnnPtr );
					qnnnPtr->gradient( phiReadPtr, grad );
					double gradNorm = sqrt( SUMD( SQR( grad[0] ), SQR( grad[1] ), SQR( grad[2] ) ) );

					gradQualityPtr[n] = ABS( 1 - gradNorm );

					// Normalize gradient and translate center node to interface for interpolation.
					for( int i = 0; i < P4EST_DIM; i++ )				// Translation: this is the location where
						xyz[i] -= grad[i] / gradNorm * phiReadPtr[n];	// we need to interpolate numerical curvature.

					double iHKappa = H * interpolation( DIM( xyz[0], xyz[1], xyz[2] ) );
					hkErrorPtr[n] = ABS( iHKappa - hKappaPtr[n] );		// Dimensionless curvature absolute error.

					// Storing the troubling indicator.
					if( troublingFlag )
						troublingPtr[n] = 1;
				}
			}
			catch( std::exception &e )
			{
					std::cerr << "Node " << n << " (" << xyz[0] << ", " << xyz[1] << "): " << e.what() << std::endl;
			}
		}

		// The error.
		std::cout << "Maximum relative error: " << maxRE << std::endl;

		// Auxiliary output for Matlab.
		if( MIN_REFINEMENT_LEVEL == MAX_REFINEMENT_LEVEL )
		{
			std::cout << "Writing signed-distance function file for Matlab... ";
			std::ofstream distPhiFile;
			std::string distPhiFileName = "distPhi.csv";
			distPhiFile.open( distPhiFileName, std::ofstream::trunc );
			if( !distPhiFile.is_open() )
				throw std::runtime_error( "Output file " + distPhiFileName + " couldn't be opened!" );
			distPhiFile.precision( 15 );

			for( int i = 0; i < N_NODES_PER_DIM; i++ )
			{
				int j = 0;
				for( ; j < N_NODES_PER_DIM - 1; j++ )
					distPhiFile << M[i][j] << ",";
				distPhiFile << M[i][j] << std::endl;
			}

			distPhiFile.close();
			std::cout << "Done!" << std::endl;
		}

		std::ostringstream oss;
		oss << "merging_circles_" << P4EST_DIM << "d";
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								8, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr,
								VTK_POINT_DATA, "interfaceFlag", interfaceFlagPtr,
								VTK_POINT_DATA, "hKappa", hKappaPtr,
								VTK_POINT_DATA, "distPhi", distPhiPtr,
								VTK_POINT_DATA, "nError", nErrorPtr,
								VTK_POINT_DATA, "gradQuality", gradQualityPtr,
								VTK_POINT_DATA, "hkError", hkErrorPtr,
								VTK_POINT_DATA, "troubling", troublingPtr );
		my_p4est_vtk_write_ghost_layer( p4est, ghost );

		ierr = VecRestoreArray( troubling, &troublingPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( who, &whoPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( hkError, &hkErrorPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( gradQuality, &gradQualityPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( nError, &nErrorPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( hKappa, &hKappaPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( distPhi, &distPhiPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArray( interfaceFlag, &interfaceFlagPtr );
		CHKERRXX( ierr );

		ierr = VecRestoreArrayRead( phi, &phiReadPtr );
		CHKERRXX( ierr );

		// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
		ierr = VecDestroy( troubling );
		CHKERRXX( ierr );

		ierr = VecDestroy( who );
		CHKERRXX( ierr );

		ierr = VecDestroy( hkError );
		CHKERRXX( ierr );

		ierr = VecDestroy( gradQuality );
		CHKERRXX( ierr );

		ierr = VecDestroy( nError );
		CHKERRXX( ierr );

		ierr = VecDestroy( hKappa );
		CHKERRXX( ierr );

		ierr = VecDestroy( distPhi );
		CHKERRXX( ierr );

		ierr = VecDestroy( interfaceFlag );
		CHKERRXX( ierr );

		ierr = VecDestroy( phi );
		CHKERRXX( ierr );

		ierr = VecDestroy( curvature );
		CHKERRXX( ierr );

		for( auto& dim : normal )
		{
			ierr = VecDestroy( dim );
			CHKERRXX( ierr );
		}

		// Destroy the p4est and its connectivity structure.
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}