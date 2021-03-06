//
// Created by Im YoungMin on 2/26/21.
//

#ifndef ML_MASS_CONSERVATION_VELOCITYFIELD_H
#define ML_MASS_CONSERVATION_VELOCITYFIELD_H

#include <src/my_p4est_utils.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>

#include <vector>
#include <random>

#include "Utils.h"

/**
 * Auxiliary class to create random divergence-free velocity fields.
 * Based on the paper "Learned discretizations for passive scalar advection in a 2-D turbulent flow" whose code base is
 * at https://github.com/google-research/data-driven-pdes/blob/master/datadrivenpdes/advection/velocity_fields.py.  The
 * idea also implements the method in "Comment on 'Diffusion by a random velocity field' [Phys. Fluids 13, 22 (1970)]."
 * @note Tested only on 2D.
 */
class RandomVelocityField
{
private:
	const int MAX_PERIOD;				// Maximum wavelength (or period) in each Cartesian direction.
	const double POWER_LAW;				// Constant to scale amplitudes.

	std::mt19937& _gen;					// Random number generator.
	std::vector<double> _xWaveVectors;	// Unscaled wave vectors' components.
	std::vector<double> _yWaveVectors;
	std::vector<double> _amplitudes;	// Wave amplitudes.
	std::vector<double> _phaseShifts;	// Phase shift for waves.

public:

	/**
	 * Constructor.
	 * @param [in] gen Standard mersenne_twister_engine for random number generation.
	 * @param [in] powerLaw Power law for decay (must be negative).
	 * @param [in] maxPeriod Maximum positive wavelength (at least 1).
	 */
	explicit RandomVelocityField( std::mt19937& gen, double powerLaw=-3, int maxPeriod=4 )
							   	  : _gen( gen ), POWER_LAW( -ABS( powerLaw ) ), MAX_PERIOD( MAX( 1, ABS( maxPeriod ) ) )
	{
		std::uniform_real_distribution<double> uniformDistribution;

		// Create the wave vectors as a combination of expected periods from -MAX_PERIOD to MAX_PERIOD, in steps of 1.
		const int N_LIN_ELEMENTS = SQR( MAX_PERIOD * 2 + 1 );
		_xWaveVectors.reserve( N_LIN_ELEMENTS );
		_yWaveVectors.reserve( N_LIN_ELEMENTS );
		for( int kx = -MAX_PERIOD; kx <= MAX_PERIOD; kx++ )
		{
			for( int ky = -MAX_PERIOD; ky <= MAX_PERIOD; ky++ )
			{
				_xWaveVectors.push_back( kx );	// We have here the sequence: -4 -4 -4 ... -1  0  0  0  0  0  1 ... 4 4 4
				_yWaveVectors.push_back( ky );	// We have here the sequence: -4 -3 -2 ...  0 -4 -3 -2 -1  0 -4 ... 2 3 4
			}
		}

		// Use a power-law distribution with a hard cutoff, namely with amplitudes scaled by (k+1)^n where
		// k = sqrt(kx^2 + ky^2) and n is some (negative) constant.
		_amplitudes.reserve( N_LIN_ELEMENTS );
		_phaseShifts.reserve( N_LIN_ELEMENTS );
		for( int i = 0; i < N_LIN_ELEMENTS; i++ )
		{
			double scale = pow( sqrt( SQR( _xWaveVectors[i] ) + SQR( _yWaveVectors[i] ) ) + 1, POWER_LAW );
			_amplitudes.push_back( scale * uniformDistribution( _gen ) );
			_phaseShifts.push_back( uniformDistribution( _gen ) * 2 * M_PI );	// Creating all wave features at once.
		}
	}

	/**
	 * Evaluate velocity field on an adaptive grid, backed by parallel PETSc vectors.
	 * @param [in] meshLen Mesh length in each Cartesian direction.
	 * @param [in] nodes Pointer to nodes struct.
	 * @param [out] vel Velocity parallel PETSc vectors.
	 */
	void evaluate( const double meshLen[P4EST_DIM], const p4est_t *p4est, const p4est_nodes_t *nodes,
				   Vec vel[P4EST_DIM] ) const
	{
		PetscErrorCode ierr;

		// Prepare access to velocity vectors that we'll populate.
		double *velPtr[P4EST_DIM];
		for( int dim = 0; dim < P4EST_DIM; dim++ )
		{
			ierr = VecGetArray( vel[dim], &velPtr[dim] );
			CHKERRXX( ierr );
		}

		// Wave vector components.
		std::vector<double> kx;
		std::vector<double> ky;
		kx.reserve( _xWaveVectors.size() );
		ky.reserve( _yWaveVectors.size() );
		for( int i = 0; i < _xWaveVectors.size(); i++ )
		{
			kx.push_back( 2 * M_PI * _xWaveVectors[i] / meshLen[0] );
			ky.push_back( 2 * M_PI * _yWaveVectors[i] / meshLen[1] );
		}

		// Populate velocity vectors at each independent (owned and ghost) node.
		double xyz[P4EST_DIM];
		foreach_local_node( n, nodes )
		{
			node_xyz_fr_n( n, p4est, nodes, xyz );
			velPtr[0][n] = velPtr[1][n] = 0;
			for( int k = 0; k < _amplitudes.size(); k++ )
			{
				double phase = (kx[k] * xyz[0] + ky[k] * xyz[1]) + _phaseShifts[k];	// The operand: (x·k) + φ.

				// Velocity components: accumulate contribution of each wave.
				for( int dim = 0; dim < P4EST_DIM; dim++ )
				{
					const std::vector<double>& scale = (!dim)? _yWaveVectors : _xWaveVectors;
					velPtr[dim][n] += (!dim? 1 : -1) * scale[k] * _amplitudes[k] * sin( phase );
				}
			}
		}

		// Finish up.
		for( int dim = 0; dim < P4EST_DIM; dim++ )
		{
			ierr = VecRestoreArray( vel[dim], &velPtr[dim] );
			CHKERRXX( ierr );
		}
	}

	/**
	 * Normalize velocity field to approximately maximum unit length by scaling the wave amplitudes.
	 * To scale the velocity field, we create an ancillary grid with the same number of uniform cells along each
	 * Cartesian direction.  The user must supply the mesh side length and the number of cells per linear unit length.
	 * If needed, for debug purposes, the function dumps the grid and the velocity field components to output files.
	 * @param [in] xyzMin Mesh minimum corner (i.e., the minimum coordinates for any point in the domain).
	 * @param [in] meshLen Mesh length in each cartesian direction.
	 * @param [in] cellsPerUnit Cells per linear unit side length.
	 * @param [in] dumpFiles Whether to dump or not the velocity components and the grid ranges along x and y directions.
	 * @param [in] dumpIdx If dumping files, use this as file suffix.
	 * @throws Runtime exception if writing dump files fails.
	 */
	void normalize( const double xyzMin[P4EST_DIM], const double meshLen[P4EST_DIM], int cellsPerUnit=128,
				    const bool& dumpFiles=false, const unsigned int& dumpIdx=0 )
	{
		std::vector<double> grid[P4EST_DIM];				// Grid coordinates on each Cartesian direction.
		for( int dim = 0; dim < P4EST_DIM; dim++ )
			linspace( xyzMin[dim], xyzMin[dim] +  meshLen[dim], (int)round( cellsPerUnit * meshLen[dim] ) + 1, grid[dim] );

		// Prepare wave vector components.
		std::vector<double> kx;
		std::vector<double> ky;
		kx.reserve( _xWaveVectors.size() );
		ky.reserve( _yWaveVectors.size() );
		for( int i = 0; i < _xWaveVectors.size(); i++ )
		{
			kx.push_back( 2 * M_PI * _xWaveVectors[i] / meshLen[0] );
			ky.push_back( 2 * M_PI * _yWaveVectors[i] / meshLen[1] );
		}

		// Evaluating velocity field on ancillary uniform grid.
		double U[grid[0].size()][grid[1].size()];			// Notice the indexing: i is for x, not y.
		double V[grid[0].size()][grid[1].size()];

		double vMax = 0;									// Finding also the maximum velocity magnitude.
		for( int i = 0; i < grid[0].size(); i++ )
		{
			for( int j = 0; j < grid[1].size(); j++ )
			{
				U[i][j] = V[i][j] = 0;						// Initialization.

				for( int k = 0; k < _amplitudes.size(); k++ )
				{
					double phase = (kx[k] * grid[0][i] + ky[k] * grid[1][j]) + _phaseShifts[k];	// Sine operand: (x·k) + φ.

					// Velocity components: accumulate contribution of each wave.
					U[i][j] += _yWaveVectors[k] * _amplitudes[k] * sin( phase );
					V[i][j] += -_xWaveVectors[k] * _amplitudes[k] * sin( phase );
				}

				double v = sqrt( SQR( U[i][j] ) + SQR( V[i][j] ) );
				vMax = MAX( vMax, v );
			}
		}

		// After finding the maximum velocity magnitude, scale amplitudes so that, approximately, the velocity field
		// has, roughly, max norm 1, at least in the considered computational domain.
		for( double& amplitude : _amplitudes )
			amplitude /= vMax;

		// Write data to CSVs for debugging, if needed.
		if( dumpFiles )
		{
			const int PRECISION = 15;

			// Dumping the grid coords on each direction.
			for( int dim = 0; dim < P4EST_DIM; dim++ )
			{
				std::ofstream coordsFile;
				utils::openFile( "coords" + std::to_string( dim ) + "_" + std::to_string( dumpIdx ) + ".csv",
					 			 PRECISION, coordsFile );

				// Writting a coords in one shot (almost).
				std::copy( grid[dim].begin(), grid[dim].end() - 1, std::ostream_iterator<double>( coordsFile, "," ) );
				coordsFile << grid[dim].back() << std::endl;

				coordsFile.close();
			}

			// Dumping velocity field components in separate files.
			std::ofstream uFile, vFile;
			utils::openFile( "u_" + std::to_string( dumpIdx ) + ".csv", PRECISION, uFile );
			utils::openFile( "v_" + std::to_string( dumpIdx ) + ".csv", PRECISION, vFile );

			for( int i = 0; i < grid[0].size(); i++ )
			{
				int j;
				for( j = 0; j < grid[1].size() - 1; j++ )
				{
					uFile << U[i][j] << ",";
					vFile << V[i][j] << ",";
				}
				uFile << U[i][j] << std::endl;
				vFile << V[i][j] << std::endl;
			}

			uFile.close();
			vFile.close();
		}
	}

};


#endif //ML_MASS_CONSERVATION_VELOCITYFIELD_H
