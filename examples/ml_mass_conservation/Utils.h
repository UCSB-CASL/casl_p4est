//
// Created by Im YoungMin on 3/4/21.
//

#ifndef ML_MASS_CONSERVATION_UTILS_H
#define ML_MASS_CONSERVATION_UTILS_H

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
}

#endif //ML_MASS_CONSERVATION_UTILS_H
