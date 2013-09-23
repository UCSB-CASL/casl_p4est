#ifndef __CASL_TYPES_H__
#define __CASL_TYPES_H__

#include <stdlib.h>
#include <ostream>

#include "utils.h"

#ifdef _WIN32
#include <stdint.h>
#endif

struct QuadValue {
  double val00;
  double val01;
  double val10;
  double val11;

  QuadValue()
  {
    val00 = val10 = val01 = val11 = 0;
  }

  QuadValue(double v00, double v01, double v10, double v11)
  {
    val00 = v00;
    val10 = v10;
    val01 = v01;
    val11 = v11;
  }

  /*!
     * \brief overload the << operator for QuadValue
     */
  friend std::ostream& operator<<(std::ostream& os, const QuadValue& q)
  {
    os << "val01 : " << q.val01 << ",\t" << "val11 : " << q.val11 << std::endl;
    os << "val00 : " << q.val00 << ",\t" << "val10 : " << q.val10 << std::endl;
    return os;
  }
};

struct OctValue {
  double val000;
  double val010;
  double val100;
  double val110;
  double val001;
  double val011;
  double val101;
  double val111;

  OctValue()
  {
    val000 = val100 = val010 = val110 = 0;
    val001 = val101 = val011 = val111 = 0;
  }

  OctValue(double v000, double v001, double v010, double v011, double v100, double v101, double v110, double v111)
  {
    val000 = v000;
    val100 = v100;
    val010 = v010;
    val110 = v110;
    val001 = v001;
    val101 = v101;
    val011 = v011;
    val111 = v111;
  }

  /*!
     * \brief overload the << operator for QuadValue
     */
  friend std::ostream& operator<<(std::ostream& os, const OctValue& q)
  {
    os << "val001 : " << q.val001 << ",\tval011 : " << q.val011 << ",\tval101 : " << q.val101 << ",\tval111 : " << q.val111 << std::endl;
    os << "val000 : " << q.val000 << ",\tval010 : " << q.val010 << ",\tval100 : " << q.val100 << ",\tval110 : " << q.val110 << std::endl;
    return os;
  }
};

#endif // __CASL_TYPES_H__
