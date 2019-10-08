/*
 * Title: machine_learning_ls_datasets
 * Description: Create datasets for training a neural network on a non-signed distance function with circular level set.
 * Author: Luis Ángel (임 영민)
 * Date Created: 10-05-2019
 */

// System.
#include <stdexcept>
#include <iostream>

// casl_p4est
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_level_set.h>

#include <src/petsc_compatibility.h>
#include <string>
#include "PointsOnCurves.h"

/*!
 * Circular *non* signed distance function.
 */
struct circle: CF_2
{
	/*!
	 * Circle constructor.
	 * @param x0_ Center x-coordinate.
	 * @param y0_ Center y-coordinate.
	 * @param r_ Circle radius.
	 */
	circle( double x0_, double y0_, double r_ ) : x0( x0_ ), y0( y0_ ), r( r_ )
	{}

	/*!
	 * Non-signed distance function evaluation at a point.
	 * @param x Point x-coordinate.
	 * @param y Point y-coordinate.
	 * @return 0 if point lies on circunference, < 0 if inside circle, > 0 if outside.
	 */
	double operator()( double x, double y ) const override
	{
		return SQR( x - x0 ) + SQR( y - y0 ) - SQR( r );
	}

private:
	double x0, y0, r;		// Center coordinates and circle radius.
};


/*!
 * Global variables.
 */
const std::string DATA_PATH = "/Volumes/YoungMinEXT/LSCurvatureML/data/CASL Non Distance Function/";
const std::string COLUMN_NAMES[] = {
		"(i-1,j+1)", "(i,j+1)", "(i+1,j+1)", "(i-1,j)", "(i,j)", "(i+1,j)", "(i-1,j-1)", "(i,j-1)", "(i+1,j-1)",
		"h*kappa", "h"
};
const int NUM_COLUMNS = 11;
const int NUM_SAMPLES = 1u << 10u;					// Number of samples for each circle radius for any grid resolution.
const int N_CIRCLES = 60;							// Number of circles for a given radius per grid resolution.
const int MIN_D = 0,								// Min and max dimension values.
		  MAX_D = 1;

/*!
 * Generate and save training datasets with a very large number or rows and 9+1+1 columns containing the renitialized phi
 * values for a non-distance function with circular interface.
 * @param nGridPoints: Number of grid points per axis (x and y).
 * @param iter: Number of iterations for level set reinitialization.
 * @param mpi: MPI object.
 */
void saveReinitializedDataset( int nGridPoints, int iter, const mpi_environment_t& mpi );


int main (int argc, char* argv[])
{
	try
	{
		mpi_environment_t mpi{};
		mpi.init( argc, argv );

		int numberOfIterations = 5;												// Change it to 5, 10, 15, 20.
		for( int resolution = 16; resolution <= 512; resolution += 8 )			// From 16x16 to 512x512 grid resolutions.
			saveReinitializedDataset( resolution, numberOfIterations, mpi );
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}


void saveReinitializedDataset( int nGridPoints, int iter, const mpi_environment_t& mpi )
{
	try
	{
		double h = 1. / ( nGridPoints - 1. );						// Spatial step size in both x and y directions.

		////////////////////////////////////////// Setting up the p4est structs ////////////////////////////////////////

		p4est_t *p4est;
		p4est_nodes_t *nodes;
		PetscErrorCode ierr;

		p4est_connectivity_t *connectivity;							// Create the connectivity object.  Our domain is [0,1]^2, and we want a cartesian grid.
		my_p4est_brick_t brick;
		int n_xyz[] = {nGridPoints, nGridPoints, nGridPoints};		// Number of root cells in the macromesh along x, y, z
		double xyz_min[] = {MIN_D - h, MIN_D - h, MIN_D - h};		// Coordinates of the lower-left-back point: We want to skip first column and row.
		double xyz_max[] = {MAX_D, MAX_D, MAX_D};					// Coordinates of the front-right-top point.
		int periodic[] = {0, 0, 0};									// Whether the domain is periodic or not.

		connectivity = my_p4est_brick_new( n_xyz, xyz_min, xyz_max, &brick, periodic );

		// Now create the forest.
		p4est = my_p4est_new( mpi.comm(), connectivity, 0, nullptr, nullptr );

		// Finally, re-partition.
		my_p4est_partition( p4est, P4EST_TRUE, nullptr );

		// Create the ghost structure.
		p4est_ghost_t *ghost = my_p4est_ghost_new( p4est, P4EST_CONNECT_FULL );

		// Generate the node data structure.
		nodes = my_p4est_nodes_new( p4est, ghost );

		// Initialize the neighbor nodes structure.
		my_p4est_hierarchy_t hierarchy( p4est, ghost, &brick );
		my_p4est_node_neighbors_t node_neighbors( &hierarchy, nodes );

		///////////////////////////////////////// Generating the datasets //////////////////////////////////////////////

		parStopWatch watch;

		printf( ">> Beginning to generate dataset for %i circles, %i grid points, and h = %f, in a [0,1]x[0,1] domain\n", N_CIRCLES, nGridPoints, h );
		std::string fileName = DATA_PATH + "iter" + std::to_string( iter ) + "/reinitDataset_m" + std::to_string( nGridPoints ) +  ".csv";
		watch.start();

		// TODO: Open file in write mode.  Write header row.

		int nc = 0;											// Keeps track of number of circles whose dataset has been generated.
		double minRadius = 2.0 * ( h + 0.000001 * ranged_rand( 0, 1 ) );
		double maxRadius = 0.5 - 2.0 * h;
		double distance = maxRadius - minRadius;			// Circles' radii are in [2*(h+\eps), 0.5-2h].
		double spread[N_CIRCLES];
		for( int i = 0; i < N_CIRCLES; i++ )
			spread[i] = static_cast<double>( i ) / ( N_CIRCLES - 1.0 );		// Uniform distribution from 0 to 1, with N_CIRCLES steps, inclusive, to spread distances.

		while( nc < N_CIRCLES )
		{
			double r = minRadius + spread[nc] * distance;					// Circle radius to be evaluated.
			std::vector<std::vector<double>> samples;
			while( samples.size() < NUM_SAMPLES )							// Generate samples until we reach at least the expected value.
			{
				double c[2] = {0.5 + ranged_rand( -h/2.0, +h/2.0 ),			// Center coords are randomly chosen around the center of the grid.
				   			   0.5 + ranged_rand( -h/2.0, +h/2.0 )};
				double hkappa = h / r;										// Target dimensionless curvature: h\kappa = h/r.
				circle circ( c[0], c[1], r );								// Non-signed distance function with circular interface.

				/////// Generate phi values ///////
				Vec phi;
				ierr = VecCreateGhostNodes( p4est, nodes, &phi ); CHKERRXX( ierr );
				double *phi_p;
				double nodeCoords[nodes->indep_nodes.elem_count][P4EST_DIM];			// Nodes' coordinates in XY space.
				int nodeCoordsIdxs[nodes->indep_nodes.elem_count][P4EST_DIM];			// Nodes' grid index coords.
				ierr = VecGetArray( phi, &phi_p ); CHKERRXX( ierr );
				for( size_t i = 0; i < nodes->indep_nodes.elem_count; i++ )
				{
					node_xyz_fr_n( i, p4est, nodes, nodeCoords[i] );
					phi_p[i] = circ( nodeCoords[i][0], nodeCoords[i][1] );
					nodeCoordsIdxs[i][0] = static_cast<int>( nodeCoords[i][0] / h );	// Indices from -1 to (nGridPoints - 1)
					nodeCoordsIdxs[i][1] = static_cast<int>( nodeCoords[i][1] / h );	// in both x- and y- directions.
				}

				ierr = VecRestoreArray( phi, &phi_p ); CHKERRXX( ierr );

				////// Reinitialize levelset and put values in a cartesian grid of easy access ///////
				my_p4est_level_set_t ls( &node_neighbors );
				ls.reinitialize_2nd_order( phi, iter );

				ierr = VecGetArray( phi, &phi_p ); CHKERRXX( ierr );				// Get reinitialized values.

				// Create a regular grid (i.e. matrix) of reinitialized values.  Notice that we skip negative indices since we
				// needed to "shift" the interface one cell up and to the right so that the number of grid points along x and y
				// would be even (and odd cells).  Thus, there's a center cell whose midpoint coincides with [0.5, 0.5].
				double Grid[nodes->indep_nodes.elem_count - 1][nodes->indep_nodes.elem_count - 1];
				for( size_t n = 0; n < nodes->indep_nodes.elem_count; n++ )
				{
					int i = nodeCoordsIdxs[n][0],
							j = nodeCoordsIdxs[n][1];
					if( i < 0 || j < 0 )
						continue;

					Grid[i][j] = phi_p[n];	// Stores information in transposed format (i.e. ith row contains all y values for same x).
				}

				ierr = VecRestoreArray( phi, &phi_p ); CHKERRXX( ierr );

				// Finally, delete PETSc Vecs by calling 'VecDestroy' function.
				ierr = VecDestroy( phi ); CHKERRXX( ierr );

				/////// Process grid points' along the circular level set for which we want to compute curvature ///////
				std::set<std::tuple<int, int>> points = PointsOnCurves::getPointsAlongCircle( c, r, h );

				for( const auto& p : points )
					std::cout << std::get<0>( p ) << ", " << std::get<1>( p ) << "; " << std::endl;

				for( const auto& p : points )
				{
					int i = std::get<0>( p ),							// Returned indices.
						j = std::get<1>( p );
					int subgrid[9][P4EST_DIM] = {						// The nine grid points including the point of interest right in the middle.
							{i - 1, j + 1}, {i, j + 1}, {i + 1, j + 1},
							{i - 1,     j}, {i,     j}, {i + 1,     j},
							{i - 1, j - 1}, {i, j - 1}, {i + 1, j - 1}
					};

					std::vector<double> dataPve( NUM_COLUMNS, 0 ),		// Phi, h and h*kappa results in positive and negative form.
										dataNve( NUM_COLUMNS, 0 );

					int s;
					for( s = 0; s < 9; s++ )							// Collect phi(x) for each of the 9 grid points.
					{
						const int* q = &subgrid[s][0];
						if( q[0] < 0 || q[0] >= nGridPoints || q[1] < 0 || q[1] >= nGridPoints )
							throw std::out_of_range( "Grid point out of range: (" + std::to_string( q[0] ) + ", " + std::to_string( q[1] ) + ")" );

						dataPve[s] = Grid[q[0]][q[1]];					// Store both positive and negative phi values.
						dataNve[s] = -dataPve[s];
					}

					dataPve[s] = hkappa;								// Second to last column holds h*\kappa.
					dataNve[s] = -hkappa;
					s++;
					dataPve[s] = h;										// Last column holds spatial scale h.
					dataNve[s] = h;
					samples.push_back( dataPve );						// Build nx(9+1+1) matrix of phi, h*kappa, and h values.
					samples.push_back( dataNve );
				}
			}

			// We have generated at least NUM_SAMPLES samples for circles of the same given radius.
			// This increases the number of samples for very small circles by multiplicity but varying their centers.
			// Now remove any excess.
			// TODO: Write function to randomly sample.
//			if( samples.size() > NUM_SAMPLES )
//				samples = datasample( samples, NUM_SAMPLES, "Replace", false );		// Sample randomly.

			nc++;

			if( nc % 10 == 0 )
				printf( "   %i circle groups evaluated after %f secs.\n", nc, watch.get_duration_current() );

			// TODO: Write to file the samples content.
		}

		printf( "<< Finished generating %i circles in %f secs.\n", nc - 1, watch.get_duration_current() );

		//////////////////////////////// Destroy the p4est and its connectivity structure //////////////////////////////

		p4est_nodes_destroy( nodes );
		p4est_ghost_destroy( ghost );
		p4est_destroy( p4est );
		p4est_connectivity_destroy( connectivity );

		watch.stop();
	}
	catch( const std::exception &e )
	{
		std::cerr << e.what() << std::endl;
	}
}