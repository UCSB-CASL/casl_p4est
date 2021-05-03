//
// Created by Im YoungMin on 7/13/20.
//

#ifndef FAST_SWEEPING_LOCAL_UTILS_H
#define FAST_SWEEPING_LOCAL_UTILS_H

#include <sys/stat.h>
#include <vector>
#include <string>
#include <src/my_p4est_utils.h>


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
 *		"hk"  =>  Exact target h * kappa
 *		"ihk" =>  Interpolated h * kappa
 * @param [out] header Array of column headers to be filled up.  Must be backed by a correctly allocated array.
 */
void generateColumnHeaders( std::string header[] )
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
	header[i+1] = "hk";								// Don't forget the h*kappa column!
	header[i+2] = "ihk";
}

/**
 * Rotate the level-set function values in a sample vector by 90 degrees counter-clockwise.
 * This is used to augment data sets.  Input samples are modified in place.  Dimensionless curvature remains the same.
 * @param [in|out] sample The sample vector with level-set function values in the standard order (e.g. mm, m0, mp, ...)
 * @param [in] NUM_COLUMNS Number of columns in full sample.
 */
void rotatePhiValues90( std::vector<double>& sample, const int NUM_COLUMNS )
{
	double phiVals[] = {
		sample[2], sample[5], sample[8], sample[1], sample[4], sample[7], sample[0], sample[3], sample[6]
	};

	for( int i = 0; i < NUM_COLUMNS - 2; i++ )
		sample[i] = phiVals[i];
}

#endif //FAST_SWEEPING_LOCAL_UTILS_H
