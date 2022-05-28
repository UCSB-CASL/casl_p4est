//
// Created by Im YoungMin on 5/27/22.
//

#ifndef ML_CURVATURE_TEST_UTILS_H
#define ML_CURVATURE_TEST_UTILS_H

#ifdef P4_TO_P8

#include <src/my_p8est_utils.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_level_set.h>
#include <random>
#include "../data_sets/level_set_patch_3d.h"

namespace test_utils
{
	/**
	 * Perform an online evaluation of the neural networks on a surface for which we can extract samples with known true mean curvature.
	 * @param [in] mpi MPI environment.
	 * @param [in] h Mesh size.
	 * @param [in] maxRL Max level of refinement of a unit-cube octree that leads to h.
	 * @param [in] octMaxRL Actual max level of refinement of the octrees that make up the domain.
	 * @param [in] n_xyz Number of effective octrees in each direction.
	 * @param [in] xyz_min Domain min coordinates.
	 * @param [in] xyz_max Domain max coordinates.
	 * @param [in] reinitIters Number of iterations for redistancing.
	 * @param [in,out] sdLS The surface as a discretized level-set function.
	 * @param [in] nonSaddleMinIH2KG Min ih2kg to identify a sample as a non-saddle sample.
	 * @param [in] ru2 Limiting ellipse semi-axis on the u-direction for sampling.
	 * @param [in] rv2 Limiting ellipse semi-axis on the v-direction for sampling.
	 * @param [in] nnetNS Non-saddle neural network.
	 * @param [in] nnetSD Saddle neural network.
	 * @param [in,out] genNoise Noise random generating engine.
	 * @param [in] randomNoiseDist Noise distribution (null to not perturb phi).
	 * @param [in] exportVTK Whether to export results to VTK for visualization.
	 * @param [in] surfaceName If exporting VTK, prefix file with this name.
	 */
	void evaluatePerformance( const mpi_environment_t& mpi, const double& h, const int& maxRL, const u_char& octMaxRL,
							  const int n_xyz[P4EST_DIM], const double xyz_min[P4EST_DIM], const double xyz_max[P4EST_DIM],
							  const int& reinitIters, SignedDistanceLevelSet *sdLS, const double& nonSaddleMinIH2KG,
							  const double& ru2, const double& rv2, const kml::NeuralNetwork *nnetNS, const kml::NeuralNetwork *nnetSD,
							  std::mt19937& genNoise, std::uniform_real_distribution<double> *randomNoiseDist,
							  const bool& exportVTK=false, const std::string& surfaceName="" )
	{
		parStopWatch watch;
		watch.start();

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		int periodic[P4EST_DIM] = {0, 0, 0};
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Create the forest using the surface as a refinement criterion.
		splitting_criteria_cf_and_uniform_band_t levelSetSplittingCriterion( 0, octMaxRL, sdLS, 3.0 );
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSplittingCriterion );

		// Refine and partition forest.
		sdLS->toggleCache( true );		// Turn on cache to speed up repeated distance computations.
		sdLS->reserveCache( (size_t)pow( 0.75 * (xyz_max[0] - xyz_min[0]) / h, 3 ) );	// Reserve space in cache to improve hashing.
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
		sdLS->evaluate( p4est, nodes, phi, exactFlag );

		if( randomNoiseDist != nullptr )
			addRandomNoiseToLSFunction( phi, nodes, genNoise, *randomNoiseDist );

		// Reinitialization.
		double reinitStartTime = watch.get_duration_current();
		CHKERRXX( PetscPrintf( mpi.comm(), "* Reinitializing...  " ) );
		my_p4est_level_set_t ls( ngbd );
		ls.reinitialize_2nd_order( phi, reinitIters );
		double reinitTime = watch.get_duration_current() - reinitStartTime;
		CHKERRXX( PetscPrintf( mpi.comm(), "done after %.6f secs.\n", reinitTime ) );

		Vec sampledFlag, trueHK, hkError;	// These vectors distinguish sampled nodes along the interface and store the true hk on Gamma and its error.
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &sampledFlag ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &trueHK ) );
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hkError ) );

		// We won't use these samples -- we really just want to get the sampledFlag to filter nodes in the hybrid curvature computation.
		std::vector<std::vector<double>> samples;
		double trackedMaxErrors[P4EST_DIM];
		int nNumericalSaddles;
		double trackedMinHK = DBL_MAX;
		double trackedMaxHK = 0;
		sdLS->collectSamples( p4est, nodes, ngbd, phi, octMaxRL, xyz_min, xyz_max, trackedMaxErrors, trackedMinHK, trackedMaxHK, samples,
							  nNumericalSaddles, exactFlag, sampledFlag, nullptr, nullptr, nullptr, nullptr, nullptr, ru2, rv2,
							  nonSaddleMinIH2KG, trueHK );
		sdLS->clearCache();
		sdLS->toggleCache( false );
		samples.clear();

		// Collecting error stats and performance metrics from numerical baseline curvature computation.
		double numTime, numMaxAbsError, numMeanAbsError;
		int numGridPoints;
		CHKERRXX( PetscPrintf( mpi.comm(), "* Evaluating numerical baseline...  " ) );
		numTime = numericalBaselineComputation( mpi, *ngbd, h, phi, trueHK, &watch, sampledFlag, numMaxAbsError, numMeanAbsError, numGridPoints );
		CHKERRXX( PetscPrintf( mpi.comm(), "done with the following stats:\n" ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Time (in secs)            = %.6f\n", numTime + reinitTime ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of grid points     = %i\n", numGridPoints ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Mean absolute error       = %.6e\n", numMeanAbsError ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Maximum absolute error    = %.6e\n", numMaxAbsError ) );

		// Hybrid curvature vectors.
		Vec numCurvature, hybHK, hybFlag, normal[P4EST_DIM];
		CHKERRXX( VecDuplicate( phi, &numCurvature ) );	// Numerical mean curvature at the nodes.
		CHKERRXX( VecDuplicate( phi, &hybHK ) );		// Hybrid curvature at normal projections of interface nodes (masked with sampledFlag).
		CHKERRXX( VecDuplicate( phi, &hybFlag ) );		// Where we used the hybrid approach (should match sampledFlag).
		for( auto& dim : normal )
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );

		// Compute hybrid (dimensionless) mean curvature.
		CHKERRXX( PetscPrintf( mpi.comm(), "* Computing hybrid mean curvature... " ) );
		kml::Curvature mlCurvature( nnetNS, nnetSD, h, 0.004, 0.007, nonSaddleMinIH2KG );
		std::pair<double, double> durations = mlCurvature.compute( *ngbd, phi, normal, numCurvature, hybHK, hybFlag, true, &watch, sampledFlag );

		// Compute statistics.
		int nHybNodes = 0;								// In how many nodes did we use the hybrid approach?
		double maxAbsError = 0;
		double meanAbsError = 0;
		double *hkErrorPtr;
		CHKERRXX( VecGetArray( hkError, &hkErrorPtr ) );
		const double *hybHKReadPtr, *trueHKReadPtr, *sampledFlagReadPtr, *hybFlagReadPtr;
		CHKERRXX( VecGetArrayRead( hybHK, &hybHKReadPtr ) );
		CHKERRXX( VecGetArrayRead( trueHK, &trueHKReadPtr ) );
		CHKERRXX( VecGetArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecGetArrayRead( hybFlag, &hybFlagReadPtr ) );
		foreach_local_node( n, nodes )
		{
			if( hybFlagReadPtr[n] == 1 )
			{
				if( sampledFlagReadPtr[n] != 0 )		// Filter did work.
				{
					hkErrorPtr[n] = ABS( trueHKReadPtr[n] - hybHKReadPtr[n] );
					meanAbsError += hkErrorPtr[n];
					maxAbsError = MAX( maxAbsError, hkErrorPtr[n] );
					nHybNodes++;
				}
				else
					std::cerr << "Error!!! Did you just compute the hybrid curvature for non-sampled node " << n << "?!" << std::endl;
			}
			else
			{
				if( sampledFlagReadPtr[n] != 0 )
					std::cerr << "Error!!! Node " << n << " was supposed to be considered for hybrid computation!" << std::endl;
			}
		}

		// Reduce stats across processes.
		CHKERRXX( VecGhostUpdateBegin( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
		CHKERRXX( VecGhostUpdateEnd( hkError, INSERT_VALUES, SCATTER_FORWARD ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &nHybNodes, 1, MPI_INT, MPI_SUM, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &maxAbsError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm() ) );
		SC_CHECK_MPI( MPI_Allreduce( MPI_IN_PLACE, &meanAbsError, 1, MPI_DOUBLE, MPI_SUM, mpi.comm() ) );
		meanAbsError /= nHybNodes;

		CHKERRXX( PetscPrintf( mpi.comm(), "done with the following stats:\n" ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Time (in secs)            = %.6f\n", durations.second + reinitTime ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Number of grid points     = %i (%i saddles)\n", nHybNodes, nNumericalSaddles ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Mean absolute error       = %.6e\n", meanAbsError ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "   - Maximum absolute error    = %.6e\n", maxAbsError ) );

		// Export visual data.
		if( exportVTK )
		{
			const double *phiReadPtr;
			CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ));
			std::ostringstream oss;
			oss << surfaceName << "_online_test_lvl" << ( int ) maxRL;
			my_p4est_vtk_write_all( p4est, nodes, ghost,
									P4EST_TRUE, P4EST_TRUE,
									5, 0, oss.str().c_str(),
									VTK_POINT_DATA, "phi", phiReadPtr,
									VTK_POINT_DATA, "hybHK", hybHKReadPtr,
									VTK_POINT_DATA, "trueHK", trueHKReadPtr,
									VTK_POINT_DATA, "hybFlag", hybFlagReadPtr,
									VTK_POINT_DATA, "hkError", hkErrorPtr );
			CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ));
		}

		CHKERRXX( VecRestoreArrayRead( hybHK, &hybHKReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( trueHK, &trueHKReadPtr ) );
		CHKERRXX( VecRestoreArray( hkError, &hkErrorPtr ) );
		CHKERRXX( VecRestoreArrayRead( sampledFlag, &sampledFlagReadPtr ) );
		CHKERRXX( VecRestoreArrayRead( hybFlag, &hybFlagReadPtr ) );

		// Clean up.
		CHKERRXX( VecDestroy( numCurvature ) );
		CHKERRXX( VecDestroy( hybHK ) );
		CHKERRXX( VecDestroy( hybFlag ) );
		for( auto& dim : normal )
			CHKERRXX( VecDestroy( dim ) );
		CHKERRXX( VecDestroy( hkError ) );
		CHKERRXX( VecDestroy( trueHK ) );
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

		CHKERRXX( PetscPrintf( mpi.comm(), "<< Done after %.2f secs.\n", watch.get_duration_current() ) );
		watch.stop();
	}


	/**
	 * Retrieve the shift and rotation parameters to define an affine transformation of the local coordinate frame.
	 * @param [in] mpi MPI environment.
	 * @param [out] origin Local frame's shift.
	 * @param [out] rotAxis Rotation axis.
	 * @param [out] rotAngle Rotation angle.
	 * @param [in] perturbOrigin Whether to apply a random shift to local coordinate system.
	 * @param [in] randomRotation Whether to apply a random rotation to local coordinate system.
	 * @param [in] h Mesh size.
	 * @param [in,out] gen Random engine.
	 */
	void computeAffineTransformationParameters( const mpi_environment_t& mpi, double origin[P4EST_DIM], double rotAxis[P4EST_DIM],
												double& rotAngle, const bool& perturbOrigin, const bool& randomRotation, const double& h,
												std::mt19937& gen )
	{
		for( int i = 0; i < P4EST_DIM; i++ )
			origin[i] = 0;									// Local frame origin (possible perturbed).
		rotAxis[0] = 0; rotAxis[1] = 0; rotAxis[2] = 1;		// Rotation axis for possible random rotation.
		rotAngle = 0;

		if( mpi.rank() == 0 )			// Only rank 0 perturbs the local frame to create an affine-transformed level-set function.
		{
			if( perturbOrigin )
			{
				std::uniform_real_distribution<double> uniformDistributionH_2( -h/2, +h/2 );
				for( int i = 0; i < P4EST_DIM; i++ )
					origin[i] = uniformDistributionH_2( gen );
			}

			if( randomRotation )		// Generate a random unit axis and its rotation angle.
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
	}
}
#endif

#endif //ML_CURVATURE_TEST_UTILS_H
