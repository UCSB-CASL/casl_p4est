/**
 * Testing many random spheres.
 *
 * Developer: Luis Ángel.
 * Created: May 12, 2022.
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
#include <random>
#include <src/my_p8est_curvature_ml.h>
#include <src/my_p8est_vtk.h>

#define NUM_SPHERES 30

class ManySpheres : public CF_DIM
{
	double _x0[NUM_SPHERES], _y0[NUM_SPHERES], _z0[NUM_SPHERES];
	double _r[NUM_SPHERES];

public:
	explicit ManySpheres( std::mt19937& gen )
	{
		std::uniform_real_distribution<double> radius(0.125, 0.25);
		std::uniform_real_distribution<double> coords(-1 + 0.125, 1 - 0.125);
		for( int i = 0; i < NUM_SPHERES; i++ )
		{
			_r[i] = radius(gen);
			_x0[i] = coords(gen);
			_y0[i] = coords(gen);
			_z0[i] = coords(gen);
		}
	}
	double operator()( double x, double y, double z ) const override
	{
		double minD = DBL_MAX;		// For a union of level-sets, we use the minimum value.
		for( int i = 0; i < NUM_SPHERES; i++ )
		{
			double d = sqrt( SQR(x - _x0[i]) + SQR(y -_y0[i]) + SQR(z - _z0[i]) ) - _r[i];
			minD = MIN( d, minD );
		}
		return minD;
	}
};


int main ( int argc, char* argv[] )
{
	try
	{
		// Initializing parallel environment.
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		const u_short maxRL = 10;
		const u_short minRL = 4;

		// Domain information.
		int n_xyz[P4EST_DIM] = {2, 2, 2};
		double xyz_min[P4EST_DIM] = {-1, -1, -1};
		double xyz_max[P4EST_DIM] = {+1, +1, +1};
		int periodic[P4EST_DIM] = {0, 0, 0};

		// p4est variables and data structures.
		p4est_t *p4est;
		p4est_nodes_t *nodes;
		my_p4est_brick_t brick;
		p4est_ghost_t *ghost;
		p4est_connectivity_t *connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Use the exact signed distance spherical functions.
		std::mt19937 gen{};		// NOLINT.
		ManySpheres manySpheres( gen );
		splitting_criteria_cf_and_uniform_band_t levelSetSC( minRL, maxRL, &manySpheres, 3.0 );

		// Create the forest using a level set as refinement criterion.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );
		p4est->user_pointer = (void *)( &levelSetSC );

		// Refine and partition forest.
		for( int i = 0; i < maxRL; i++ )
		{
			my_p4est_refine( p4est, P4EST_FALSE, refine_levelset_cf_and_uniform_band, nullptr );
			my_p4est_partition( p4est, P4EST_FALSE, nullptr );
		}
		CHKERRXX( PetscPrintf( mpi.comm(), "Done refining.\n" ) );

		// Create the ghost (cell) and node structures.
		ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		auto *hierarchy = new my_p4est_hierarchy_t( p4est, ghost, &brick );
		auto *ngbd = new my_p4est_node_neighbors_t( hierarchy, nodes );

		// Ghosted parallel PETSc vectors to store level-set values.
		Vec phi;
		CHKERRXX( VecCreateGhostNodes( p4est, nodes, &phi ) );

		// Calculate the level-set function values for all independent nodes.
		sample_cf_on_nodes( p4est, nodes, manySpheres, phi );
		CHKERRXX( PetscPrintf( mpi.comm(), "Done sampling level-sets.\n" ) );

		// Export to VTK.
		const double *phiReadPtr;
		CHKERRXX( VecGetArrayRead( phi, &phiReadPtr ) );
		std::ostringstream oss;
		oss << "many_spheres";
		my_p4est_vtk_write_all( p4est, nodes, ghost,
								P4EST_TRUE, P4EST_TRUE,
								1, 0, oss.str().c_str(),
								VTK_POINT_DATA, "phi", phiReadPtr );
		CHKERRXX( VecRestoreArrayRead( phi, &phiReadPtr ) );
		CHKERRXX( PetscPrintf( mpi.comm(), "Done exporting data to VTK for visualization.\n" ) );

		// Clean up.
		CHKERRXX( VecDestroy( phi ) );

		// Destroy the p4est and its connectivity structure.
		delete ngbd;
		delete hierarchy;
		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		my_p4est_brick_destroy( connectivity, &brick );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}