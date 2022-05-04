/**
 * Generating circular samples for a vortex-shear test, similar to the test found in section 5.5 in "HFES: A height function method with
 * explicit input and signed output for high-order estimations of curvature and unit vectors of planar curves", by Q. Zhang.
 *
 * @note Comparison with the above paper suggested in first and second revisions of the paper.
 *
 * Only reinitialized samples are collected for a circle placed at (0.5, 0.75) of radius = 0.15 after it completes a revolution within the
 * unit box [0,1]^2, where the stream function is
 *                           psi(x,y) = -1/pi * sin^2(pi*x) * sin^2(pi*y) * cos(pi*t/T),
 * with T = 2.  We do this for maximum levels of refinement 7, 8, 9, 10.  To determine the "target" or best attainable accuracy, we use a
 * second level-set function with the same terminal interface at t=T but at a much higher resolution.  The "true" curvature is then
 * extracted via interpolation from this second level-set function.
 *
 * Developer: Luis Ángel.
 * Created: May 2, 2022.
 * Updated: May 4, 2022.
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
#include <src/casl_geometry.h>
#include <src/petsc_compatibility.h>
#include <string>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <unordered_map>
#include "local_utils.h"

/////////////////////////////////////////////////// Divergence-free vortex velocity field //////////////////////////////////////////////////

class UComponent : public CF_3
{
private:
	double T;	// Half period.

public:
	explicit UComponent( const double& halfPeriod=2.0 ) : T( halfPeriod ) {}

	double operator()( double x, double y, double t ) const override
	{
		return -SQR( sin( M_PI * x ) ) * sin( 2 * M_PI * y ) * cos( M_PI * t / T );
	}
};

class VComponent : public CF_3
{
private:
	double T;	// Half period.

public:
	explicit VComponent( const double& halfPeriod=2.0 ) : T( halfPeriod ) {}

	double operator()( double x, double y, double t ) const override
	{
		return SQR( sin( M_PI * y ) ) * sin( 2 * M_PI * x ) * cos( M_PI * t / T );
	}
};

//////////////////////////////////////////////////////////// Auxiliary functions ///////////////////////////////////////////////////////////

void prepareFile( std::ofstream& rlsFile, const std::string& rlsFileName, const u_short& numColumns, const std::string* columnNames );
void sampleVelocityField( Vec vel[P4EST_DIM], const p4est_t *p4est, const p4est_nodes_t *nodes, const CF_3 *velocityField[P4EST_DIM], const double& tn=0 );
void writeVTK( const u_char& MRL, const int& vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, Vec phiExact,
			   Vec hk, Vec ihk, Vec hkError );
void setUpP4estStructs( const mpi_environment_t& mpi, p4est_t *& p4est, p4est_nodes_t *& nodes, my_p4est_brick_t& brick,
						p4est_connectivity_t *& connectivity, splitting_criteria_cf_and_uniform_band_t *& spl, p4est_ghost_t *& ghost,
						my_p4est_hierarchy_t *& hierarchy, my_p4est_node_neighbors_t *& ngbd, const int n_xyz[P4EST_DIM],
						const double xyz_min[P4EST_DIM], const double xyz_max[P4EST_DIM], const int periodic[P4EST_DIM], const u_char& mrl,
						CF_2& levelSetFunction, const double& halfBandWidth );

/////////////////////////////////////////////////////////////// Main function //////////////////////////////////////////////////////////////

int main ( int argc, char* argv[] )
{
	const double RADIUS = 0.15;
	const double CENTER[P4EST_DIM] = {0.5, 0.75};
	const u_char MAX_REFINEMENT_LEVEL[] = {7};
	const int NUM_REINIT_ITERS = 10;						// Number of iterations for PDE reintialization.
	const double CFL = 1.0;									// Courant-Friedrichs-Lewy condition.
	const double DURATION = 2.0;							// The velocity flips direction at half this duration.
	const double HALF_BAND_WIDTH = 6.0;
	const int REF_FACTOR = 4;								// Reference grid will be 2^{REF_FACTOR} times finer.

	const std::string DATA_PATH = "/Volumes/YoungMinEXT/pde-1120/data/vortex/";	// Destination folder.
	const int NUM_COLUMNS = (int)pow( 3, P4EST_DIM ) + 2;	// Number of columns in resulting data sets.
	std::string COLUMN_NAMES[NUM_COLUMNS];					// Column headers following the x-y truth table of
	generateColumnHeaders( COLUMN_NAMES );					// 3-state variables.

	// Domain information, applicable to all cases.
	int n_xyz[] = { 1, 1, 1 };
	double xyz_min[] = { 0, 0, 0 };							// Squared domain [0, 1]^2.
	double xyz_max[] = { 1, 1, 1 };
	int periodic[] = { 0, 0, 0 };							// Non-periodic domain.

	u_char minMRL = CHAR_MAX;
	for( const auto& mrl : MAX_REFINEMENT_LEVEL )			// Find the minimum MRL among those we need (used below for VTK exportations).
		minMRL = MIN( minMRL, mrl );

	try
	{
		// Initializing parallel environment (although in reality we're working on a single process).
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		if( mpi.size() > 1 )
			throw std::runtime_error( "Only a single process is allowed!" );

		///////////////////////////////////////////////////// Generating the data sets /////////////////////////////////////////////////////

		for( const auto& MRL : MAX_REFINEMENT_LEVEL )
		{
			const double H = 1. / pow( 2, MRL );					// Mesh size.
			const int NUM_ITER_VTK = 16 * (1 << (MRL - minMRL));	// Save VTK files every NUM_ITER_VTK iterations.
			std::vector<std::vector<double>> rlsSamples;

			parStopWatch watch;
			CHKERRXX( PetscPrintf( mpi.comm(), ">> Began 2D vortex test with MAX_RL = %d...\n", MRL ) );
			watch.start();

			// Preparing file where to write samples: its name is of the form sphere_rls_x.csv, where x is the maximum level of refinement.
			std::ofstream rlsFile;
			prepareFile( rlsFile, DATA_PATH + "sphere_rls_" + std::to_string( MRL ) +  ".csv", NUM_COLUMNS, COLUMN_NAMES );

			// Define the velocity field.
			UComponent uComponent( DURATION );
			VComponent vComponent( DURATION );
			const CF_3 *velocityField[P4EST_DIM] = {&uComponent, &vComponent};

			// Defining the level-sets.
			geom::SphereNSD sphereNsd( CENTER[0], CENTER[1], RADIUS );
			geom::Sphere sphere( CENTER[0], CENTER[1], RADIUS );	// This is the exact dist func we'll use for discretizing coarse/finer grids.

			// Coarse p4est variables and data structures and set up coarse grid.
			p4est_t *p4est;
			p4est_nodes_t *nodes;
			my_p4est_brick_t brick;
			p4est_ghost_t *ghost;
			p4est_connectivity_t *connectivity;
			splitting_criteria_cf_and_uniform_band_t *spl;
			my_p4est_hierarchy_t *hierarchy;
			my_p4est_node_neighbors_t *ngbd;
			setUpP4estStructs( mpi, p4est, nodes, brick, connectivity, spl, ghost, hierarchy, ngbd, n_xyz, xyz_min, xyz_max, periodic, MRL,
							   sphere, HALF_BAND_WIDTH );

			// Validation.
			double dxyz[P4EST_DIM]; 			// Dimensions of the smallest quadrants.
			double dxyz_min;        			// Minimum side length of the smallest quadrants.
			double diag_min;        			// Diagonal length of the smallest quadrants.
			get_dxyz_min( p4est, dxyz, dxyz_min, diag_min );
			assert( ABS( dxyz_min - H ) <= EPS );

			// Declare data vectors and pointers for read/write.
			Vec phi;							// Level-set function values (subject to reinitialization).
			Vec vel[P4EST_DIM];					// Veloctiy field.
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );
			for( auto& dir : vel )
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dir ) );

			// Sample the level-set function values and nodal velocities at tn=0.
			sample_cf_on_nodes( p4est, nodes, sphereNsd, phi );
			sampleVelocityField( vel, p4est, nodes, velocityField );

			// Reinitialize coarse level-set function.
			my_p4est_level_set_t ls0( ngbd );
			ls0.reinitialize_2nd_order( phi, NUM_REINIT_ITERS );

			//////////////////////////////////////////////////////// Advection step ////////////////////////////////////////////////////////

			Vec exactPhi;								// Compute exact signed distance function to visualize results for debugging.
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactPhi ) );
			sample_cf_on_nodes( p4est, nodes, sphere, exactPhi );

			Vec hk, ihk, hkError;						// Let's keep track of the curvature and its error, although we populate these until the end.
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hk ) );
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &ihk ) );
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hkError ) );

			double tn = 0;								// Current time.
			double tn_vel = 0;
			bool hasVelSwitched = false;
			int iter = 0;
			const double MAX_VEL_NORM = 1.0; 			// Maximum velocity norm is known analitically.
			double dt = CFL * dxyz_min / MAX_VEL_NORM;	// deltaT knowing that the CFL condition is c*dt/dx <= CFL.
			int vtkIdx = 1;
			writeVTK( MRL, 0, p4est, nodes, ghost, phi, exactPhi, hk, ihk, hkError );
			while( tn < DURATION )
			{
				if( tn + dt > DURATION )				// Clip step if it's going to go over the final time.
					dt = DURATION - tn;

				if( tn + dt >= DURATION / 2.0 && !hasVelSwitched )		// Clip time step if it's going to go over half time.
				{
					dt = (DURATION / 2.0) - tn;
					hasVelSwitched = true;
					CHKERRXX( PetscPrintf( mpi.comm(), "*** Switching velocity direction at the end of this iteration ***\n" ) );
					tn_vel += dt;						// To skip the stall point at tn = T/2.
				}

				// Coarse p4est objects at time tnp1; they will be updated during the semi-Lagrangian advection step.
				p4est_t *p4est_np1 = p4est_copy( p4est, P4EST_FALSE );
				p4est_ghost_t *ghost_np1 = my_p4est_ghost_new( p4est_np1, P4EST_CONNECT_FULL );
				p4est_nodes_t *nodes_np1 = my_p4est_nodes_new( p4est_np1, ghost_np1 );

				// Create semi-Lagrangian object and advect.
				my_p4est_semi_lagrangian_t semiLagrangian( &p4est_np1, &nodes_np1, &ghost_np1, ngbd );
				semiLagrangian.set_phi_interpolation( interpolation_method::quadratic_non_oscillatory_continuous_v1 );
				semiLagrangian.set_velo_interpolation( interpolation_method::quadratic_non_oscillatory_continuous_v1 );
				semiLagrangian.update_p4est( vel, dt, phi );

				// Destroy old forest and create new structures.
				p4est_destroy( p4est );
				p4est = p4est_np1;
				p4est_ghost_destroy( ghost );
				ghost = ghost_np1;
				p4est_nodes_destroy( nodes );
				nodes = nodes_np1;

				delete hierarchy;
				delete ngbd;
				hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
				ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
				ngbd->init_neighbors();

				// Reinitialize level-set function.
				my_p4est_level_set_t ls( ngbd );
				ls.reinitialize_2nd_order( phi, NUM_REINIT_ITERS );

				// Update stepping variables and velocities.
				tn += dt;
				tn_vel += dt;
				dt = CFL * dxyz_min / MAX_VEL_NORM;				// Restore time step size.
				iter++;

				// Rellocate velocity vectors and resample velocity field.
				for( auto& dir : vel)
				{
					CHKERRXX( VecDestroy( dir ) );
					CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dir ) );
				}
				sampleVelocityField( vel, p4est, nodes, velocityField, tn_vel );

				// Resample exact solution.
				CHKERRXX( VecDestroy( exactPhi ) );
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &exactPhi ) );
				sample_cf_on_nodes( p4est, nodes, sphere, exactPhi );

				// Reallocate curvature vectors.
				CHKERRXX( VecDestroy( hk ) ); CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hk ) );
				CHKERRXX( VecDestroy( ihk ) ); CHKERRXX( VecCreateGhostNodes( p4est, nodes, &ihk ) );
				CHKERRXX( VecDestroy( hkError ) ); CHKERRXX( VecCreateGhostNodes( p4est, nodes, &hkError ) );

				// Display iteration message.
				CHKERRXX( PetscPrintf( mpi.comm(), "\tIteration %04d: t = %1.4f \n", iter, tn ) );

				// Save to vtk format (mid file is always written).  The last state is saved after we collect samples and record errors.
				if( ABS( tn - DURATION / 2.0 ) <=PETSC_MACHINE_EPSILON || (iter % NUM_ITER_VTK == 0 && ABS( tn - DURATION ) > PETSC_MACHINE_EPSILON) )
				{
					writeVTK( MRL, vtkIdx, p4est, nodes, ghost, phi, exactPhi, hk, ihk, hkError );
					vtkIdx++;
				}
			}

			///////////////////////////// Sample-collection step: use coarse interface to define a finer grid //////////////////////////////

			// Reference (finer) p4est variables and data structures.  Then, we'll define a finer grid to supply the reference hk.
			p4est_t *ref_p4est;
			p4est_nodes_t *ref_nodes;
			my_p4est_brick_t ref_brick;
			p4est_ghost_t *ref_ghost;
			p4est_connectivity_t *ref_connectivity;
			splitting_criteria_cf_and_uniform_band_t *ref_spl;
			my_p4est_hierarchy_t *ref_hierarchy;
			my_p4est_node_neighbors_t *ref_ngbd;
			const int REF_MRL = MRL + REF_FACTOR;
			setUpP4estStructs( mpi, ref_p4est, ref_nodes, ref_brick, ref_connectivity, ref_spl, ref_ghost, ref_hierarchy, ref_ngbd, n_xyz,
							   xyz_min, xyz_max, periodic, REF_MRL, sphere, HALF_BAND_WIDTH * (1 << REF_FACTOR) );

			// Get the reference level-set function values interpolated from coarse grid and perfected via reinitialization.
			Vec refPhi;
			CHKERRXX( VecCreateGhostNodes( ref_p4est, ref_nodes, &refPhi ) );
			double *refPhiPtr;
			CHKERRXX( VecGetArray( refPhi, &refPhiPtr ) );

			Vec phi_xx[P4EST_DIM];	// Second derivatives of final coarse level-set function.
			for( auto& dim : phi_xx )
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );
			ngbd->second_derivatives_central( phi, phi_xx[0], phi_xx[1] );
			my_p4est_interpolation_nodes_t phiInterp( ngbd );
			phiInterp.set_input( phi, phi_xx[0], phi_xx[1], interpolation_method::quadratic_non_oscillatory_continuous_v1 );
			foreach_node( n, ref_nodes )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( n, ref_p4est, ref_nodes, xyz );
				phiInterp.add_point( n, xyz );
			}
			phiInterp.interpolate( refPhiPtr );
			phiInterp.clear();
			CHKERRXX( VecRestoreArray( refPhi, &refPhiPtr ) );

			my_p4est_level_set_t refLS( ref_ngbd );
			refLS.reinitialize_2nd_order( refPhi, NUM_REINIT_ITERS * 15 );	// Notice how many more iterations we use.

			// Prepare queries from coarse grid for curvature.
			Vec refNormals[P4EST_DIM];
			Vec refCurvature;
			CHKERRXX( VecCreateGhostNodes( ref_p4est, ref_nodes, &refCurvature ) );
			for( auto& dim : refNormals )
				CHKERRXX( VecCreateGhostNodes( ref_p4est, ref_nodes, &dim ) );
			compute_normals( *ref_ngbd, refPhi, refNormals );
			compute_mean_curvature( *ref_ngbd, refNormals, refCurvature );	// Notice I'm using the most accurate way of computing kappa.

			Vec refCurvature_xx[P4EST_DIM];
			for( auto& dim : refCurvature_xx )
				CHKERRXX( VecCreateGhostNodes( ref_p4est, ref_nodes, &dim ) );
			ref_ngbd->second_derivatives_central( refCurvature, refCurvature_xx[0], refCurvature_xx[1] );
			my_p4est_interpolation_nodes_t refCurvatureInterp( ref_ngbd );
			refCurvatureInterp.set_input( refCurvature, refCurvature_xx[0], refCurvature_xx[1], interpolation_method::quadratic_non_oscillatory );

			my_p4est_interpolation_nodes_t refPhiInterp( ref_ngbd );	// We also need to interpolate the phi and normal values from the reference grid.
			refPhiInterp.set_input( refPhi, interpolation_method::linear );

			my_p4est_interpolation_nodes_t refNormalsXInterp( ref_ngbd );
			refNormalsXInterp.set_input( refNormals[0], interpolation_method::linear );
			my_p4est_interpolation_nodes_t refNormalsYInterp( ref_ngbd );
			refNormalsYInterp.set_input( refNormals[1], interpolation_method::linear );

			//////////////////////////////// Sample-collection step: use finer grid to find "true" curvature ///////////////////////////////

			// Compute coarse curvature with reinitialized data, which will be linearly interpolated at the interface.
			Vec normals[P4EST_DIM];
			Vec curvature;
			CHKERRXX( VecCreateGhostNodes( p4est, nodes, &curvature ) );
			for( auto& dim : normals )
				CHKERRXX( VecCreateGhostNodes( p4est, nodes, &dim ) );
			compute_normals( *ngbd, phi, normals );
			compute_mean_curvature( *ngbd, phi, normals, curvature );

			my_p4est_interpolation_nodes_t interpolation( ngbd );
			interpolation.set_input( curvature, interpolation_method::linear );

			// Collect nodes on or adjacent to the interface with valid stencils.
			NodesAlongInterface nodesAlongInterface( p4est, nodes, ngbd, (char)MRL );
			std::vector<p4est_locidx_t> indices;
			nodesAlongInterface.getIndices( &phi, indices );

			const double *phiReadPtr, *normalsReadPtr[P4EST_DIM];
			CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
			for( int i = 0; i < P4EST_DIM; i++ )
				CHKERRXX( VecGetArrayRead( normals[i], &normalsReadPtr[i] ) );

			double maxHKError = 0;						// And this one is the max difference between ref hk and coarse hk.
			double *hkPtr, *ihkPtr, *hkErrorPtr;
			CHKERRXX( VecGetArray( hk, &hkPtr ) );		// Record curvature and its error in these vectors.
			CHKERRXX( VecGetArray( ihk, &ihkPtr ) );
			CHKERRXX( VecGetArray( hkError, &hkErrorPtr ) );
			for( auto n : indices )
			{
				std::vector<p4est_locidx_t> stencil;
				std::vector<double> sample;				// Level-set values and h*kappa results in negative form only.
				sample.reserve( NUM_COLUMNS );
				try
				{
					if( nodesAlongInterface.getFullStencilOfNode( n , stencil ) )
					{
						for( auto s : stencil )
							sample.push_back( -phiReadPtr[s] );

						// Finding projection on to interface.
						double xyz[P4EST_DIM];					// Position of node at the center of the stencil.
						node_xyz_fr_n( n, p4est, nodes, xyz );

						double refXYZ[P4EST_DIM] = {xyz[0], xyz[1]};
						double refPhiVal = refPhiInterp( refXYZ[0], refXYZ[1] );
						double refGrad[P4EST_DIM] = {refNormalsXInterp( refXYZ[0], refXYZ[1] ), refNormalsYInterp( refXYZ[0], refXYZ[1] )};
						double refGradNorm = sqrt( SQR( refGrad[0] ) + SQR( refGrad[1] ) );

						for( int i = 0; i < P4EST_DIM; i++ )
						{
							xyz[i] -= normalsReadPtr[i][n] * phiReadPtr[n];					// Coarse location where to interpolate curvature.
							refXYZ[i] -= refPhiVal * refGrad[i] / refGradNorm;				// Reference projection.
						}

						hkPtr[n] = refCurvatureInterp( refXYZ[0], refXYZ[1] );				// Record curvature and error.
						ihkPtr[n] = interpolation( xyz[0], xyz[1] );
						hkErrorPtr[n] = ABS( hkPtr[n] - ihkPtr[n] );

						sample.push_back( -H * hkPtr[n] );									// Appending the reference "target" h*kappa.
						sample.push_back( -H * ihkPtr[n] );									// Attach interpolated h*kappa.
						maxHKError = MAX( maxHKError, H * hkErrorPtr[n] );

						rlsSamples.push_back( sample );			// Accumulating samples.
					}
				}
				catch( std::exception &e )
				{
					std::cerr << e.what() << std::endl;
				}
			}

			// Let's find the error in coarse phi.
			double maxPhiError = 0;						// This help us check how much the coarse level-set deviates from expected solution.
			foreach_node( n, nodes )
			{
				double xyz[P4EST_DIM];
				node_xyz_fr_n( n, p4est, nodes, xyz );
				if( ABS( phiReadPtr[n] ) <= diag_min )
					maxPhiError = MAX( maxPhiError, ABS( phiReadPtr[n] - sphere( xyz[0], xyz[1] ) ) / H );
			}

			// Save visualization for ref grid.
			char name[1024];
			const double *refPhiReadPtr;
			sprintf( name, "vtu_%d/reference", MRL );
			CHKERRXX( VecGetArrayRead( refPhi, &refPhiReadPtr ) );
			my_p4est_vtk_write_all( ref_p4est, ref_nodes, ref_ghost,
									P4EST_TRUE, P4EST_TRUE,
									1, 0, name,
									VTK_POINT_DATA, "ref_phi", refPhiReadPtr );
			CHKERRXX( VecRestoreArrayRead( refPhi, &refPhiReadPtr ) );
			CHKERRXX( PetscPrintf( p4est->mpicomm, ":: Saved vtk file for reference grid ::\n" ) );

			// Save las visualization for coarse grid with its errors.
			writeVTK( MRL, vtkIdx, p4est, nodes, ghost, phi, exactPhi, hk, ihk, hkError );

			// Clearing interpolators.
			interpolation.clear();
			refNormalsYInterp.clear();
			refNormalsXInterp.clear();
			refPhiInterp.clear();
			refCurvatureInterp.clear();
			phiInterp.clear();

			// Cleaning up coarse data.
			CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
			for( int i = 0; i < P4EST_DIM; i++ )
				CHKERRXX( VecGetArrayRead( normals[i], &normalsReadPtr[i] ) );
			CHKERRXX( VecRestoreArray( hk, &hkPtr ) );
			CHKERRXX( VecRestoreArray( ihk, &ihkPtr ) );
			CHKERRXX( VecRestoreArray( hkError, &hkErrorPtr ) );

			CHKERRXX( VecDestroy( exactPhi ) );
			CHKERRXX( VecDestroy( phi ) );
			CHKERRXX( VecDestroy( curvature ) );
			for( auto& dim : normals )
				CHKERRXX( VecDestroy( dim ) );
			for( auto& dim : phi_xx )
				CHKERRXX( VecDestroy( dim ) );
			for( auto& dim : vel )
				CHKERRXX( VecDestroy( dim ) );
			CHKERRXX( VecDestroy( hk ) );
			CHKERRXX( VecDestroy( ihk ) );
			CHKERRXX( VecDestroy( hkError ) );

			// Cleaning up reference data.
			CHKERRXX( VecDestroy( refPhi ) );
			CHKERRXX( VecDestroy( refCurvature ) );
			for( auto& dim : refNormals )
				CHKERRXX( VecDestroy( dim ) );
			for( auto& dim : refCurvature_xx )
				CHKERRXX( VecDestroy( dim ) );

			// Destroy reference grid structs.
			delete ref_spl;
			delete ref_hierarchy;
			delete ref_ngbd;
			p4est_nodes_destroy( ref_nodes );
			p4est_ghost_destroy( ref_ghost );
			p4est_destroy( ref_p4est );
			my_p4est_brick_destroy( ref_connectivity, &ref_brick );

			// Destroy the coarse p4est and its connectivity structure.
			delete spl;
			delete hierarchy;
			delete ngbd;
			p4est_nodes_destroy( nodes );
			p4est_ghost_destroy( ghost );
			p4est_destroy( p4est );
			my_p4est_brick_destroy( connectivity, &brick );

			// Write all samples collected for all circles with the same radius but randomized center content to file.
			for( const auto& row : rlsSamples )
			{
				std::copy( row.begin(), row.end() - 1, std::ostream_iterator<double>( rlsFile, "," ) );	// Inner elements.
				rlsFile << row.back() << std::endl;
			}

			std::cout << "\tDone with MRL = " << int(MRL)
					  << ".  Samples = " << rlsSamples.size() << ";  time = " << watch.get_duration_current() << std::endl
					  << "\tMax h-relative phi error = " << maxPhiError << ";  max hk-error w.r.t. ref hk = " << maxHKError << std::endl;

			rlsFile.close();
			watch.stop();
		}
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}

void setUpP4estStructs( const mpi_environment_t& mpi, p4est_t *& p4est, p4est_nodes_t *& nodes, my_p4est_brick_t& brick,
						p4est_connectivity_t *& connectivity, splitting_criteria_cf_and_uniform_band_t *& spl, p4est_ghost_t *& ghost,
						my_p4est_hierarchy_t *& hierarchy, my_p4est_node_neighbors_t *& ngbd, const int n_xyz[P4EST_DIM],
						const double xyz_min[P4EST_DIM], const double xyz_max[P4EST_DIM], const int periodic[P4EST_DIM], const u_char& mrl,
						CF_2& levelSetFunction, const double& halfBandWidth )
{
	connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

	// Create the forest using exact signed-distance level set as refinement criterion.
	p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
	spl = new splitting_criteria_cf_and_uniform_band_t( 1, mrl, &levelSetFunction, halfBandWidth, 3 );
	p4est->user_pointer = (void *)(spl);

	// Refine and recursively partition forest.
	for( int i = 0; i < mrl; i++ )
	{
		my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
		my_p4est_partition( p4est, P4EST_FALSE, nullptr );
	}

	// Create the ghost cells and node structures.
	ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
	nodes = my_p4est_nodes_new( p4est, ghost );

	// Initialize the neighbor nodes structure.
	hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
	ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );
	ngbd->init_neighbors();
}

void writeVTK( const u_char& MRL, const int& vtkIdx, p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost, Vec phi, Vec phiExact,
			   Vec hk, Vec ihk, Vec hkError )
{
	char name[1024];

	const double *phiReadPtr, *phiExactReadPtr, *hkReadPtr, *ihkReadPtr, *hkErrorReadPtr;	// Pointers to vector contents.

	sprintf( name, "vtu_%d/vortex_%d", MRL, vtkIdx );
	CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
	CHKERRXX( VecGetArrayRead( phiExact, &phiExactReadPtr ) );
	CHKERRXX( VecGetArrayRead( hk, &hkReadPtr ) );
	CHKERRXX( VecGetArrayRead( ihk, &ihkReadPtr ) );
	CHKERRXX( VecGetArrayRead( hkError, &hkErrorReadPtr ) );
	my_p4est_vtk_write_all( p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							5, 0, name,
							VTK_POINT_DATA, "phi", phiReadPtr,
							VTK_POINT_DATA, "phiExact", phiExactReadPtr,
							VTK_POINT_DATA, "hk", hkReadPtr,
							VTK_POINT_DATA, "ihk", ihkReadPtr,
							VTK_POINT_DATA, "hkError", hkErrorReadPtr );
	CHKERRXX( VecRestoreArrayRead( hkError, &hkErrorReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( ihk, &ihkReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( hk, &hkReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
	CHKERRXX( VecRestoreArrayRead( phiExact, &phiExactReadPtr ) );

	CHKERRXX( PetscPrintf( p4est->mpicomm, ":: Saved vtk files with index %02d ::\n", vtkIdx ) );
}

void sampleVelocityField( Vec vel[P4EST_DIM], const p4est_t *p4est, const p4est_nodes_t *nodes, const CF_3 *velocityField[P4EST_DIM], const double& tn )
{
	double *velPtr[P4EST_DIM];
	for( int dir = 0; dir < P4EST_DIM; dir++ )
	{
		CHKERRXX( VecGetArray( vel[dir], &velPtr[dir] ) );
		foreach_node( n, nodes )
		{
			double xyz[P4EST_DIM];
			node_xyz_fr_n( n, p4est, nodes, xyz );
			velPtr[dir][n] = (*velocityField[dir])( xyz[0], xyz[1], tn );	// Sample velocity at time tn.
		}
		CHKERRXX( VecRestoreArray( vel[dir], &velPtr[dir] ) );
	}
}

void prepareFile( std::ofstream& rlsFile, const std::string& rlsFileName, const u_short& numColumns, const std::string* columnNames )
{
	rlsFile.open( rlsFileName, std::ofstream::trunc );
	if( !rlsFile.is_open() )
		throw std::runtime_error( "Output file " + rlsFileName + " couldn't be opened!" );

	// Write column headers: enforcing strings by adding quotes around them.
	std::ostringstream headerStream;
	for( int i = 0; i < numColumns - 1; i++ )
		headerStream << "\"" << columnNames[i] << "\",";
	headerStream << "\"" << columnNames[numColumns - 1] << "\"";
	rlsFile << headerStream.str() << std::endl;
	rlsFile.precision( 15 );
}