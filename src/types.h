#ifndef MY_P4EST_TYPES_H
#define MY_P4EST_TYPES_H

#include <stdlib.h>
#include <ostream>

#ifdef P4_TO_P8
#include "my_p8est_utils.h"
#else
#include "my_p4est_utils.h"
#endif

#ifdef _WIN32
#include <stdint.h>
#endif

struct QuadValue {
  double val[4];

  QuadValue()
  {
    for (unsigned char kk = 0; kk < 4; ++kk)
      val[kk] = 0.0;
  }

  QuadValue(double v00, double v01, double v10, double v11)
  {
    val[0] = v00;
    val[1] = v01;
    val[2] = v10;
    val[3] = v11;
  }

  /*!
     * \brief overload the << operator for QuadValue
     */
  friend std::ostream& operator<<(std::ostream& os, const QuadValue& q)
  {
    os << "val01 : " << q.val[1] << ",\t" << "val11 : " << q.val[3] << std::endl;
    os << "val00 : " << q.val[0] << ",\t" << "val10 : " << q.val[2] << std::endl;
    return os;
  }
};

/**
 * An extension of QuadValue class to also store the corresponding independent node indices for a given quad.
 * A quad is characterized as:
 *          v10      v11
 *           *--------*
 *           |        |
 *           |        |
 *           *--------*
 *          v00      v01
 */
struct QuadValueExtended : public QuadValue
{
	p4est_locidx_t indices[4] = { -1, -1, -1, -1 };		// Nodal indices for matching quad vertex values.

	/**
	 * Default constructor.
	 */
	QuadValueExtended() : QuadValue(){}

	/**
	 * Detailed constructor.
	 * @param [in] v00 Nodal value at corner #00.
	 * @param [in] v01 Nodal value at corner #01.
	 * @param [in] v10 Nodal value at corner #10.
	 * @param [in] v11 Nodal value at corner #11.
	 * @param [in] idx00 Nodal index at corner #00.
	 * @param [in] idx01 Nodal index at corner #01.
	 * @param [in] idx10 Nodal index at corner #10.
	 * @param [in] idx11 Nodal index at corner #11.
	 */
	QuadValueExtended( double v00, double v01, double v10, double v11,
			p4est_locidx_t idx00, p4est_locidx_t idx01, p4est_locidx_t idx10, p4est_locidx_t idx11 )
			: QuadValue( v00, v01, v10, v11 )
	{
		indices[0] = idx00;
		indices[1] = idx01;
		indices[2] = idx10;
		indices[3] = idx11;
	}

	/**
	 * Output stream operator to stringify the object.
	 * @param [in] os Ostream object.
	 * @param [in] q QuadValueExtended object to stringify.
	 * @return Stringified input object.
	 */
	friend std::ostream& operator<<( std::ostream& os, const QuadValueExtended& q )
	{
		os << "#00 [" << q.indices[0] << "]: " << q.val[0] << ",\t" << "#01 [" << q.indices[1] << "]:" << q.val[1] << std::endl;
		os << "#10 [" << q.indices[2] << "]: " << q.val[2] << ",\t" << "#11 [" << q.indices[3] << "]:" << q.val[3] << std::endl;
		return os;
	}
};

struct OctValue {
  double val[8];

  OctValue()
  {
    for (unsigned char kk = 0; kk < 8; ++kk)
      val[kk] = 0.0;
  }

  OctValue(double v000, double v001, double v010, double v011, double v100, double v101, double v110, double v111)
  {
    val[0] = v000;
    val[1] = v001;
    val[2] = v010;
    val[3] = v011;
    val[4] = v100;
    val[5] = v101;
    val[6] = v110;
    val[7] = v111;
  }

  /*!
     * \brief overload the << operator for QuadValue
     */
  friend std::ostream& operator<<(std::ostream& os, const OctValue& q)
  {
    os << "val001 : " << q.val[1] << ",\tval011 : " << q.val[3] << ",\tval101 : " << q.val[5] << ",\tval111 : " << q.val[7] << std::endl;
    os << "val000 : " << q.val[0] << ",\tval010 : " << q.val[2] << ",\tval100 : " << q.val[4] << ",\tval110 : " << q.val[6] << std::endl;
    return os;
  }
};

/**
 * An extension of OctValue class to also store the corresponding independent node indices for a given octant.
 * An octant is characterized as:
 *            110      111
 *             *--------*
 *            /.   011 /|
 *       010 *--------* |
 *           | *......|.*
 *           |· 100   |/ 101
 *           *--------*
 *          000      001
 */
struct OctValueExtended : public OctValue
{
	p4est_locidx_t indices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };		// Nodal indices for matching quad vertex values.

	/**
	 * Default constructor.
	 */
	OctValueExtended() : OctValue(){}

	/**
	 * Detailed constructor.
	 * @param [in] v000 Nodal value at corner #000.
	 * @param [in] v001 Nodal value at corner #001.
	 * @param [in] v010 Nodal value at corner #010.
	 * @param [in] v011 Nodal value at corner #011.
	 * @param [in] v100 Nodal value at corner #100.
	 * @param [in] v101 Nodal value at corner #101.
	 * @param [in] v110 Nodal value at corner #110.
	 * @param [in] v111 Nodal value at corner #111.
	 * @param [in] idx000 Nodal index at corner #000.
	 * @param [in] idx001 Nodal index at corner #001.
	 * @param [in] idx010 Nodal index at corner #010.
	 * @param [in] idx011 Nodal index at corner #011.
	 * @param [in] idx100 Nodal index at corner #100.
	 * @param [in] idx101 Nodal index at corner #101.
	 * @param [in] idx110 Nodal index at corner #110.
	 * @param [in] idx111 Nodal index at corner #111.
	 */
	OctValueExtended( double v000, double v001, double v010, double v011, double v100, double v101, double v110, double v111,
					   p4est_locidx_t idx000, p4est_locidx_t idx001, p4est_locidx_t idx010, p4est_locidx_t idx011,
					   p4est_locidx_t idx100, p4est_locidx_t idx101, p4est_locidx_t idx110, p4est_locidx_t idx111 )
			: OctValue( v000, v001, v010, v011, v100, v101, v110, v111 )
	{
		indices[0] = idx000;
		indices[1] = idx001;
		indices[2] = idx010;
		indices[3] = idx011;
		indices[4] = idx100;
		indices[5] = idx101;
		indices[6] = idx110;
		indices[7] = idx111;
	}

	/**
	 * Output stream operator to stringify the object.
	 * @param [in] os Ostream object.
	 * @param [in] q QuadValueExtended object to stringify.
	 * @return Stringified input object.
	 */
	friend std::ostream& operator<<( std::ostream& os, const OctValueExtended& q )
	{
		os << "#000 [" << q.indices[0] << "]: " << q.val[0] << ",\t" << "#001 [" << q.indices[1] << "]:" << q.val[1] << std::endl;
		os << "#010 [" << q.indices[2] << "]: " << q.val[2] << ",\t" << "#011 [" << q.indices[3] << "]:" << q.val[3] << std::endl;
		os << "#100 [" << q.indices[4] << "]: " << q.val[4] << ",\t" << "#101 [" << q.indices[5] << "]:" << q.val[5] << std::endl;
		os << "#110 [" << q.indices[6] << "]: " << q.val[6] << ",\t" << "#111 [" << q.indices[7] << "]:" << q.val[7] << std::endl;
		return os;
	}
};

#endif // MY_P4EST_TYPES_H
