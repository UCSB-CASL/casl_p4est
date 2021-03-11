//
// Created by Im YoungMin on 3/4/21.
//

#ifndef ML_MASS_CONSERVATION_UTILS_H
#define ML_MASS_CONSERVATION_UTILS_H

#include <unordered_set>
#include <vector>
#include <random>

namespace utils
{
	/**
	 * Utility function to open a file.
	 * @param [in] fileName File name.
	 * @param [in] precision Numerical precision.
	 * @param [out] file Output file handler.
	 * @param [in] mode Opening mode.
	 * @throws runtime exception if opening file fails.
	 */
	void openFile( const std::string& fileName, int precision, std::ofstream& file,
				   unsigned int mode=std::ofstream::trunc )
	{
		file.open( fileName, mode );
		if( !file.is_open() )
			throw std::runtime_error( "Output file " + fileName + " couldn't be opened!" );
		file.precision( precision );
	}

	/**
	 * Generate a set of n random, non-repeating, integers in the range [lowerBound, upperBound].
	 * @param [in] lowerBound Lower bound for random integers.
	 * @param [in] upperBound Upper bound for random integers.
	 * @param [in] n Number of random integers to return.
	 * @param [out] numbers Set of returned random integers.
	 * @param [in] gen Random number generator.
	 * @throws runtime exception if n <= 0 or > total number of integers in upperBound - lowerBound + 1.
	 * @throws runtime exception if upperBound <= lowerBound.
	 */
	void generateRandomSetOfNumbers( const int& lowerBound, const int& upperBound, const int& n,
								  	 std::unordered_set<int>& numbers, std::mt19937& gen )
	{
		const int N_NUMS = upperBound - lowerBound + 1;

		if( upperBound < lowerBound )
			throw std::runtime_error( "upperBound must be strictly larger than lowerBound!" );
		if( n <= 0 || n > upperBound - lowerBound + 1 )
			throw std::runtime_error( "Wrong value for n!" );

		// Create a range of numbers in [lowerBound, upperBound] and shuffle them.
		std::vector<int> range;
		range.reserve( N_NUMS );
		for( int i = lowerBound; i <= upperBound; i++ )
			range.push_back( i );
		std::shuffle( range.begin(), range.end(), gen );

		// Dump n random integers into output set.
		numbers.clear();
		numbers.reserve( n );
		for( int i = 0; i < n; i++ )
			numbers.insert( range[i] );
	}
}

#endif //ML_MASS_CONSERVATION_UTILS_H
