#ifndef ML_CURVATURE_LOCAL_UTILS_H
#define ML_CURVATURE_LOCAL_UTILS_H

#include <sys/stat.h>
#include <vector>
#include <string>
#include <src/my_p4est_utils.h>
#include "data_sets/star_theta_root_finding.h"

namespace kutils
{
	/**
	 * Generate the column headers following the truth-table order with x changing slowly, then y changing faster than x,
	 * and finally z changing faster than y.  Each dimension has three states: m, 0, and p (minus, center, plus).  For
	 * example, in 2D, the columns that are generated are:
	 *		"mm", "m0", "mp"           =>  phi(i-1, j-1), phi(i-1, j), phi(i-1, j+1) |
	 *		"0m", "00", "0p"           =>  phi(  i, j-1), phi(  i, j), phi(  i, j+1) |  First 9 entries are level-set values.
	 *		"pm", "p0", "pp"           =>  phi(i+1, j-1), phi(i+1, j), phi(i+1, j+1) |
	 *		"nx_mm", "nx_m0", "nx_mp"  =>   nx(i-1, j-1),  nx(i-1, j),  nx(i-1, j+1) +
	 *		"nx_0m", "nx_00", "nx_0p"  =>   nx(  i, j-1),  nx(  i, j),  nx(  i, j+1) +  Second 9 entries are x-components of normal unit vectors.
	 *		"nx_pm", "nx_p0", "nx_pp"  =>   nx(i+1, j-1),  nx(i+1, j),  nx(i+1, j+1) +
	 *		"ny_mm", "ny_m0", "ny_mp"  =>   ny(i-1, j-1),  ny(i-1, j),  ny(i-1, j+1) -
	 *		"ny_0m", "ny_00", "ny_0p"  =>   ny(  i, j-1),  ny(  i, j),  ny(  i, j+1) -  Third 9 entries are y-components of normal unit vectors.
	 *		"ny_pm", "ny_p0", "ny_pp"  =>   ny(i+1, j-1),  ny(i+1, j),  ny(i+1, j+1) -
	 *		"hk"  =>  Exact target h * kappa (optional)
	 *		"ihk" =>  Interpolated h * kappa
 	* @param [out] header Array of column headers to be filled up.  Must be backed by a correctly allocated array.
	 * @param [in] includeTargetHK Whether to include or not the "hk" column.
	 */
	void generateColumnHeaders( std::string header[], bool includeTargetHK=true )
	{
		const int STEPS = 3;
		std::string states[] = {"m", "0", "p"};			// States for x, y, and z directions.
		int i = 0;
		for( int x = 0; x < STEPS; x++ )
			for( int y = 0; y < STEPS; y++ )
#ifdef P4_TO_P8
				for( int z = 0; z < STEPS; z++ )
#endif
			{
				i = SUMD( x * (int)pow( STEPS, P4EST_DIM - 1 ), y * (int)pow( STEPS, P4EST_DIM - 2 ), z );
				header[i] = SUMD( states[x], states[y], states[z] );

				header[i+1*num_neighbors_cube] = "nx_" + header[i];
				header[i+2*num_neighbors_cube] = "ny_" + header[i];
			}
		i += 2*num_neighbors_cube;
		if( includeTargetHK )
			header[++i] = "hk";							// Don't forget the h*kappa column if requested!
		header[++i] = "ihk";
	}

	/**
	 * Rotate stencil of level-set function values in a sample vector by 90 degrees counter or clockwise.
	 * @param [in,out] stencil Array of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 * @param [in] dir Rotation direction: > 0 for counterclockwise, <= 0 for clockwise.
	 */
	void rotateStencil90( double stencil[], const int& dir=1 )
	{
		// Lambda function to perform rotation of a 2D vector.
		auto _rotate90 = (dir >= 1)? []( double& x, double& y ){
			std::swap( x, (y *= -1) );			// Rotate by positive 90 degrees (counterclockwise).
		} : []( double& x, double& y ){
			std::swap( (x *= -1), y );			// Rotate by negative 90 degrees (clockwise).
		};

		// Lambda function to rotate features.
		auto _rotateFeatures = (dir >= 1)? []( double f[] ){
			double c[] = {f[2], f[5], f[8], f[1], f[4], f[7], f[0], f[3], f[6]};
			for( int i = 0; i < num_neighbors_cube; i++ )	// Rotate by positive 90 degrees (counterclokwise).
				f[i] = c[i];
		} : []( double f[] ){
			double c[] = {f[6], f[3], f[0], f[7], f[4], f[1], f[8], f[5], f[2]};
			for( int i = 0; i < num_neighbors_cube; i++ )		// Rotate by negative 90 degrees.
				f[i] = c[i];
		};

		// Rotate features.
		_rotateFeatures( &stencil[0] );						// Level-set values.
		_rotateFeatures( &stencil[num_neighbors_cube] );	// Normal x-components.
		_rotateFeatures( &stencil[2*num_neighbors_cube] );	// Normal y-components.

		// Rotate actual vectors.
		for( int i = num_neighbors_cube; i < 2 * num_neighbors_cube; i++ )
			_rotate90( stencil[i], stencil[i + num_neighbors_cube] );
	}

	/**
	 * Rotate stencil of level-set function values in a sample vector by 90 degrees counter or clockwise.
	 * @param [in,out] stencil Vector of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 * @param [in] dir Rotation direction: > 0 for counterclockwise, <= 0 for clockwise.
	 */
	void rotateStencil90( std::vector<double>& stencil, const int& dir=1 )
	{
		rotateStencil90( stencil.data(), dir );
	}

	/**
	 * Reflect stencil of level-set values along line y = x.
	 * @note Useful for data augmentation assuming that we are using normalization to first quadrant of a local
	 * coordinate system whose origin is at the center of the stencil.  Exploits fact that curvature is invariant to
	 * reflections and rotations.
	 * @param [in,out] stencil Array of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 */
	void reflectStencil_yEqx( double stencil[] )
	{
		// First the swap all features from one vertex to another.
		for( int i = 0; i < 3; i++ )
		{
			int offset = num_neighbors_cube*i;
			std::swap( stencil[1 + offset], stencil[3 + offset] );
			std::swap( stencil[2 + offset], stencil[6 + offset] );
			std::swap( stencil[5 + offset], stencil[7 + offset] );
		}

		// Then, swap normal components: x<->y.
		for( int i = 0; i < num_neighbors_cube; i++ )
			std::swap( stencil[num_neighbors_cube + i], stencil[2*num_neighbors_cube + i] );
	}

	/**
	 * Reflect stencil of level-set values along line y = x.
	 * @note Useful for data augmentation assuming that we are using normalization to first quadrant of a local
	 * coordinate system whose origin is at the center of the stencil.  Exploits fact that curvature is invariant to
	 * reflections and rotations.
	 * @param [in,out] stencil Vector of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 */
	void reflectStencil_yEqx( std::vector<double>& stencil )
	{
		reflectStencil_yEqx( stencil.data() );
	}

	/**
	 * Rotate stencil in such a way that the gradient computed at center node 00 has an with respect to the horizontal
	 * in the range of [0, pi/2].
	 * @note Exploits the fact that curvature is invariant to rotation.  Prior to calling this function you must have
	 * flipped the sign of the stencil (and gradient) so that the curvature is negative.
	 * @param [in,out] stencil Array of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 * @param [in] gradient Gradient at the center node.
	 */
	void rotateStencilToFirstQuadrant( double stencil[], const double gradient[P4EST_DIM] )
	{
		double theta = atan2( gradient[1], gradient[0] );
		const double TWO_PI = 2. * M_PI;
		theta = (theta < 0)? TWO_PI + theta : theta;		// Make sure current angle lies in [0, 2pi].

		// Rotate only if theta not in [0, pi/2].
		if( theta > M_PI_2 )
		{
			if( theta <= M_PI )								// Quadrant II?
			{
				rotateStencil90( stencil, -1 );				// Rotate by -pi/2.
			}
			else if( theta < M_PI_2 * 3 )					// Quadrant III?
			{
				rotateStencil90( stencil );					// Rotate by pi.
				rotateStencil90( stencil );
			}
			else											// Quadrant IV?
			{
				rotateStencil90( stencil );					// Rotate by pi/2.
			}
		}
	}

	/**
	 * Rotate stencil in such a way that the gradient computed at center node 00 has an angle with respect to the
	 * horizontal in the range of [0, pi/2].
	 * @param [in,out] stencil Vector of feature values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 * @param [in] gradient Gradient at the center node.
	 */
	void rotateStencilToFirstQuadrant( std::vector<double>& stencil, const double gradient[P4EST_DIM] )
	{
		rotateStencilToFirstQuadrant( stencil.data(), gradient );
	}

	/**
	 * Generate the sample row of level-set function values and target hk for a node next to the star interface.
	 * @param [in] nodeIdx Query node adjancent or on the interface.
	 * @param [in] NUM_COLUMNS Number of columns in output file.
	 * @param [in] H Spacing (smallest quad/oct side-length).
	 * @param [in] stencil The full uniform stencil of indices centered at the query node.
	 * @param [in] p4est Pointer to p4est data structure.
	 * @param [in] nodes Pointer to nodes data structure.
	 * @param [in] phiReadPtr Pointer to level-set function values, backed by a parallel PETSc ghosted vector.
	 * @param [in] star The level-set function with a star-shaped interface.
	 * @param [in] gen Random-number generator device.
	 * @param [in] normalDistribution A normal random variable distribution.
	 * @param [in/out] pointsFile Pointer to optional file object to write coordinates of nodes adjacent to Gamma.
	 * @param [in/out] anglesFile Pointer to optional file object to write angles of normal projected points on Gamma.
	 * @param [out] distances A vector of "true" distances from all of 9 stencil points to the star-shaped level-set.
	 * @param [out] pOnGamma Normal projection onto interface.
	 * @param [in,out] visitedNodes Hash map functioning as a memoization mechanism to speed up access to visited nodes.
	 * @param [in] normalReadPtr Pointer to normal vector components backed by parallel PETSc vectors.
	 * @return Vector of sampled, reinitialized level-set function values for the stencil centered at the nodeIdx node.
	 */
	std::vector<double> sampleNodeNextToStarInterface( const p4est_locidx_t nodeIdx, const int NUM_COLUMNS,
													   const double H, const std::vector<p4est_locidx_t>& stencil,
													   const p4est_t *p4est, const p4est_nodes_t *nodes,
													   const double *phiReadPtr, const geom::Star& star,
													   std::mt19937& gen, std::normal_distribution<double>& normalDistribution,
													   std::ofstream *pointsFile, std::ofstream *anglesFile,
													   std::vector<double>& distances, double pOnGamma[P4EST_DIM],
													   std::unordered_map<p4est_locidx_t, Point2>& visitedNodes,
													   const double *normalReadPtr[P4EST_DIM] )
	{
		std::vector<double> sample( NUM_COLUMNS, 0 );		// Level-set function values and target hk.
		distances.clear();
		distances.reserve( NUM_COLUMNS );					// True distances and target hk.

		int s;												// Index to fill in the sample vector.
		double xyz[P4EST_DIM];
		double pOnInterface[P4EST_DIM];
		double theta, r, valOfDerivative, centerTheta;
		double dx, dy, newDistance;
		for( s = 0; s < num_neighbors_cube; s++ )			// Collect phi(x) for each of the 9 grid points.
		{
			sample[s] = phiReadPtr[stencil[s]];				// This is the distance obtained after reinitialization.

			// Approximate position of point projected on interface.
			const double grad[P4EST_DIM] = {DIM( normalReadPtr[0][stencil[s]], normalReadPtr[1][stencil[s]], normalReadPtr[2][stencil[s]] )};
			node_xyz_fr_n( stencil[s], p4est, nodes, xyz );
			for( int dim = 0; dim < P4EST_DIM; dim++ )
				pOnInterface[dim] = xyz[dim] - grad[dim] * sample[s];

			if( s == 4 )	// Rough estimation of point on interface, where curvature will be interpolated.
			{
				for( int dim = 0; dim < P4EST_DIM; dim++ )
					pOnGamma[dim] = pOnInterface[dim];
			}

			// Get initial angle for polar approximation to point on star interface.
			theta = atan2( pOnInterface[1], pOnInterface[0] );
			theta = ( theta < 0 )? theta + 2 * M_PI : theta;
			r = star.r( theta );
			pOnInterface[0] = r * cos( theta );
			pOnInterface[1] = r * sin( theta );				// Better approximation of projection of stencil point onto star.

//		if( s == 4 )
//		{
//			std::cout << std::setprecision( 15 )
//					  << "plot(" << xyz[0] << ", " << xyz[1] << ", 'b.', " << pOnInterfaceX << ", " << pOnInterfaceY
//					  << ", 'mo');" << std::endl;
//		}

			// Compute current distance to Gamma using the improved point on interface.
			dx = xyz[0] - pOnInterface[0];
			dy = xyz[1] - pOnInterface[1];
			distances.push_back( sqrt( SQR( dx ) + SQR( dy ) ) );

			// Find theta that yields "a" minimum distance between stencil point and star using Newton-Raphson's method.
			if( distances.back() > EPS )
			{
				if( visitedNodes.find( stencil[s] ) != visitedNodes.end() )		// Speed up queries.
				{
					theta = visitedNodes[stencil[s]].x;			// First component is the angular parameter.
					newDistance = visitedNodes[stencil[s]].y;	// Second component is the distance to Gamma.
				}
				else
				{
					valOfDerivative = 1;
					theta = distThetaDerivative( stencil[s], xyz[0], xyz[1], star, theta, H, gen, normalDistribution,
												 valOfDerivative, newDistance );

//				if( s == 4 )
//				{
//					r = star.r( theta );					// Recalculating closest point on interface.
//					xOnGamma = r * cos( theta );
//					yOnGamma = r * sin( theta );
//					std::cout << std::setprecision( 15 )
//							  << "plot(" << xOnGamma << ", " << yOnGamma << ", 'ko');" << std::endl;
//				}

					double relDist = (newDistance - distances[s]) / distances[s];
					if( relDist > 1e-8  )					// Verify that new point is closer than previous approximation.
					{
						std::ostringstream stream;
						stream << "Failure with node " << stencil[s] << ".  Val. of Der: " << std::scientific << valOfDerivative
							   << std::scientific << std::setprecision( 15 ) << ".  New dist: " << newDistance
							   << ".  Old dist: " << distances[s]
							   << ".  Rel dist: " << relDist;
						throw std::runtime_error( stream.str() );
					}

					visitedNodes[stencil[s]] = Point2( theta, newDistance );		// Memorize information for visited node.
				}

				distances[s] = newDistance;					// Root finding was successful: keep minimum distance.
			}

			if( star( xyz[0], xyz[1] ) < 0 )				// Fix sign.
				distances[s] *= -1;

			if( s == 4 )									// For center node we need theta to yield curvature.
				centerTheta = theta;
		}

		sample[s] = H * star.curvature( centerTheta );		// Last column holds h*kappa.
		distances.push_back( sample[s] );

		// Write center sample node index and coordinates.
		if( pointsFile )
		{
			node_xyz_fr_n( nodeIdx, p4est, nodes, xyz );
			*pointsFile << nodeIdx << "," << xyz[0] << "," << xyz[1] << std::endl;
		}

		// Write angle parameter for projected point on interface.
		if( anglesFile )
		{
			*anglesFile << ( centerTheta < 0 ? 2 * M_PI + centerTheta : centerTheta ) << std::endl;
		}

		return sample;
	}
}

#endif // ML_CURVATURE_LOCAL_UTILS_H
