//
// Created by Im YoungMin on 7/13/20.
//

#ifndef FAST_SWEEPING_LOCAL_UTILS_H
#define FAST_SWEEPING_LOCAL_UTILS_H

#include <sys/stat.h>


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
 * Verify if a directory exists.  If not, create it.
 * @param [in] path Directory valid path.
 * @throws Runtime error if directory can't be created or if the path exists and is not a directory.
 */
void checkOrCreateDirectory( const std::string& path )
{
	struct stat info{};
	if( stat( path.c_str(), &info ) != 0 )							// Directory doesn't exist?
	{
		if( mkdir( path.c_str(), 0777 ) == -1 )						// Try to create it.
			throw std::runtime_error( "Cannot create " + path + " directory: " + strerror(errno) + "!" );
	}
	else if( !( info.st_mode & (unsigned)S_IFDIR ) )
		throw std::runtime_error( path + " is not a directory!" );
}


/**
 * Compute the numerical curvature of the input 9-point stencil.  Use the following facts:
 * Idx | xy |  Meaning
 *  0  | mm | (i-1, j-1)
 *  1  | m0 | (i-1,   j)
 *  2  | mp | (i-1, j+1)
 *  3  | 0m | (  i, j-1)
 *  4  | 00 | (  i,   j)
 *  5  | 0p | (  i, j+1)
 *  6  | pm | (i+1, j-1)
 *  7  | p0 | (i+1,   j)
 *  8  | pp | (i+1, j+1)
 * @param [in] p Vector with level-set function values.
 * @param [in] H Spacing.
 * @return kappa.
 */
double computeNumericalCurvature( const std::vector<double>& p, const double H )
{
	double phi_x = ( p[7] - p[1] ) / ( 2 * H );
	double phi_y = ( p[5] - p[3] ) / ( 2 * H );
	double phi_xx = ( p[7] - 2 * p[4] + p[1] ) / SQR( H );
	double phi_yy = ( p[5] - 2 * p[4] + p[3] ) / SQR( H );
	double phi_x_ijP1 = ( p[8] - p[2] ) / ( 2 * H );
	double phi_x_ijM1 = ( p[6] - p[0] ) / ( 2 * H );
	double phi_xy = ( phi_x_ijP1 - phi_x_ijM1 ) / ( 2 * H );
	return (SQR( phi_x ) * phi_yy - 2 * phi_x * phi_y * phi_xy + SQR( phi_y ) * phi_xx) / pow( SQR( phi_x ) + SQR( phi_y ), 1.5 );
}

#endif //FAST_SWEEPING_LOCAL_UTILS_H
