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

#endif // MY_P4EST_TYPES_H
