#ifndef ML_CURVATURE_LOCAL_UTILS_H
#define ML_CURVATURE_LOCAL_UTILS_H

#include <sys/stat.h>
#include <vector>
#include <string>
#include <src/my_p4est_utils.h>

namespace kutils
{
	/**
	 * Generate the column headers following the truth-table order with x changing slowly, then y changing faster than x,
	 * and finally z changing faster than y.  Each dimension has three states: m, 0, and p (minus, center, plus).  For
	 * example, in 2D, the columns that are generated are:
	 * 	   Acronym      Meaning
	 *		"mm"  =>  (i-1, j-1)
	 *		"m0"  =>  (i-1, j  )
	 *		"mp"  =>  (i-1, j+1)
	 *		"0m"  =>  (  i, j-1)
	 *		"00"  =>  (  i,   j)
	 *		"0p"  =>  (  i, j+1)
	 *		"pm"  =>  (i+1, j-1)
	 *		"p0"  =>  (i+1,   j)
	 *		"pp"  =>  (i+1, j+1)
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
			}
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
		double phiVals[num_neighbors_cube];
		if( dir >= 1 )									// Counterclockwise rotation?
		{
			phiVals[0] = stencil[2];
			phiVals[1] = stencil[5];
			phiVals[2] = stencil[8];
			phiVals[3] = stencil[1];
			phiVals[4] = stencil[4];
			phiVals[5] = stencil[7];
			phiVals[6] = stencil[0];
			phiVals[7] = stencil[3];
			phiVals[8] = stencil[6];
		}
		else											// Clockwise rotation?
		{
			phiVals[0] = stencil[6];
			phiVals[1] = stencil[3];
			phiVals[2] = stencil[0];
			phiVals[3] = stencil[7];
			phiVals[4] = stencil[4];
			phiVals[5] = stencil[1];
			phiVals[6] = stencil[8];
			phiVals[7] = stencil[5];
			phiVals[8] = stencil[2];
		}

		for( int i = 0; i < num_neighbors_cube; i++ )
			stencil[i] = phiVals[i];
	}

	/**
	 * Rotate stencil of level-set function values in a sample vector by 90 degrees counter or clockwise.
	 * @param [in,out] stencil Vector of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
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
	 * @param [in,out] stencil Array of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 */
	void reflectStencil_yEqx( double stencil[] )
	{
		std::swap( stencil[1], stencil[3] );
		std::swap( stencil[2], stencil[6] );
		std::swap( stencil[5], stencil[7] );
	}

	/**
	 * Reflect stencil of level-set values along line y = x.
	 * @note Useful for data augmentation assuming that we are using normalization to first quadrant of a local
	 * coordinate system whose origin is at the center of the stencil.  Exploits fact that curvature is invariant to
	 * reflections and rotations.
	 * @param [in,out] stencil Vector of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 */
	void reflectStencil_yEqx( std::vector<double>& stencil )
	{
		reflectStencil_yEqx( stencil.data() );
	}

	/**
	 * Rotate stencil in such a way that the gradient computed at center node 00 has an with respect to the (local)
	 * horizontal in the range of [0, pi/2].
	 * @note Exploits the fact that curvature is invariant to rotation.  Prior to calling this function you must have
	 * flipped the sign of the stencil (and gradient) so that the curvature is negative.
	 * @param [in,out] stencil Array of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
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
	 * Rotate stencil in such a way that the gradient computed at center node 00 has an with respect to the (local)
	 * horizontal in the range of [0, pi/2].
	 * @param [in,out] stencil Vector of level-set function values in standard order (e.g., mm, m0, mp, 0m,..., p0, pp).
	 * @param [in] gradient Gradient at the center node.
	 */
	void rotateStencilToFirstQuadrant( std::vector<double>& stencil, const double gradient[P4EST_DIM] )
	{
		rotateStencilToFirstQuadrant( stencil.data(), gradient );
	}
}

#endif // ML_CURVATURE_LOCAL_UTILS_H
