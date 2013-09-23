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


typedef enum {
  DIRICHLET,
  NEUMANN,
  NOINTERFACE
} BoundaryConditionType;

class WallBC
{
public:
  virtual BoundaryConditionType operator()( double x, double y ) const=0 ;
};

class WallBC3D
{
public:
  virtual BoundaryConditionType operator()( double x, double y, double z ) const=0 ;
};


class BoundaryConditions2D
{
private:
  const WallBC* WallType_;
  BoundaryConditionType InterfaceType_;

  const CF_2 *p_WallValue;
  const CF_2 *p_InterfaceValue;

public:
  BoundaryConditions2D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
    p_InterfaceValue = NULL;
  }


  inline void setWallTypes( const WallBC& w )
  {
    WallType_ = &w;
  }

  inline const WallBC& getWallType() const
  {
    return *WallType_;
  }


  inline void setWallValues( const CF_2& v ){
    p_WallValue = &v;
  }

  inline void setInterfaceType(BoundaryConditionType bc){
    InterfaceType_ = bc;
  }

  inline void setInterfaceValue(const CF_2& in){
    p_InterfaceValue = &in;
  }

  inline BoundaryConditionType wallType( double x, double y ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(x,y);
  }

  inline BoundaryConditionType interfaceType() const{ return InterfaceType_;}

  inline double wallValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y);
  }

  inline double interfaceValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y);
  }
};


class BoundaryConditions3D
{
private:
  const WallBC3D* WallType_;
  BoundaryConditionType InterfaceType_;

  const CF_3 *p_WallValue;
  const CF_3 *p_InterfaceValue;

public:
  BoundaryConditions3D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
    p_InterfaceValue = NULL;
  }


  inline void setWallTypes( const WallBC3D& w )
  {
    WallType_ = &w;
  }

  inline const WallBC3D& getWallType() const
  {
    return *WallType_;
  }


  inline void setWallValues( const CF_3& v ){
    p_WallValue = &v;
  }

  inline void setInterfaceType(BoundaryConditionType bc){
    InterfaceType_ = bc;
  }

  inline void setInterfaceValue(const CF_3& in){
    p_InterfaceValue = &in;
  }

  inline BoundaryConditionType wallType( double x, double y, double z ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(x,y,z);
  }

  inline BoundaryConditionType interfaceType() const{ return InterfaceType_;}

  inline double wallValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y,z);
  }

  inline double interfaceValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y,z);
  }
};


#endif // __CASL_TYPES_H__
