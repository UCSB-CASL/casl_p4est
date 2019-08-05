#ifndef UTILS_H
#define UTILS_H

#include <src/casl_math.h>
#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <src/my_p8est_nodes.h>
#include <p8est_ghost.h>
#include <p8est_bits.h>
#include <src/my_p8est_refine_coarsen.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_nodes.h>
#include <p4est_ghost.h>
#include <p4est_bits.h>
#include <src/my_p4est_refine_coarsen.h>
#endif
#include <src/petsc_logging.h>
#include "petsc_compatibility.h"

#include <src/mls_integration/cube2_mls.h>
#include <src/mls_integration/cube3_mls.h>

#include <petsc.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <fstream>

// forward declaration
class my_p4est_node_neighbors_t;
struct quad_neighbor_nodes_of_node_t;

#define COMMA ,
#define P4(a) a

#ifdef P4_TO_P8
#define OCOMP(a) a
#define XCOMP(a) a
#define YCOMP(a) a
#define ZCOMP(a) a

#define _CODE(a) a
#define XCODE(a) a
#define YCODE(a) a
#define ZCODE(a) a

#define CODE2D(a)
#define CODE3D(a) a

#define EXECD(a,b,c) a; b; c;

#define CODE2D(a)
#define CODE3D(a) a

#define P8(a) a
#define P8C(a) COMMA a
#define P8EST(a) a
#define ONLY3D(a) a
#define DIM(a,b,c) a COMMA b COMMA c

#define  SUMD(a,b,c) ( (a) +  (b) +  (c) )
#define MULTD(a,b,c) ( (a) *  (b) *  (c) )
#define  ANDD(a,b,c) ( (a) && (b) && (c) )
#define   ORD(a,b,c) ( (a) || (b) || (c) )

#define CODEDIM(a,b) b

#define XFOR(a) for (a)
#define YFOR(a) for (a)
#define ZFOR(a) for (a)
#else
#define OCOMP(a) a
#define XCOMP(a) a
#define YCOMP(a) a
#define ZCOMP(a)

#define _CODE(a) a
#define XCODE(a) a
#define YCODE(a) a
#define ZCODE(a)

#define CODE2D(a) a
#define CODE3D(a)

#define EXECD(a,b,c) a; b;

#define CODE2D(a) a
#define CODE3D(a)

#define CODEDIM(a,b) a

#define P8(a)
#define P8C(a)
#define P8EST(a)
#define ONLY3D(a)
#define DIM(a,b,c) a COMMA b

#define  SUMD(a,b,c) ( (a) +  (b) )
#define MULTD(a,b,c) ( (a) *  (b) )
#define  ANDD(a,b,c) ( (a) && (b) )
#define   ORD(a,b,c) ( (a) || (b) )

#define XFOR(a) for (a)
#define YFOR(a) for (a)
#define ZFOR(a)
#endif

enum cf_value_type_t { VAL, DDX, DDY, DDZ, LAP, CUR };

enum mls_opn_t { MLS_INTERSECTION = 0, MLS_ADDITION = 1, MLS_COLORATION = 2, MLS_INT = MLS_INTERSECTION, MLS_ADD = MLS_ADDITION };

namespace dir {
/* vertices directions */
enum {
  v_mmm = 0,
  v_pmm,
  v_mpm,
  v_ppm
#ifdef P4_TO_P8
  ,v_mmp,
  v_pmp,
  v_mpp,
  v_ppp
#endif
};
/* faces directions */
enum {
  f_m00 = 0,
  f_p00,
  f_0m0,
  f_0p0
#ifdef P4_TO_P8
  ,f_00m,
  f_00p
#endif
};
/* cartesian direction */
enum {
  x = 0,
  y
#ifdef P4_TO_P8
  ,z
#endif
};
}

enum node_neighbor_cube_t
{
#ifdef P4_TO_P8
  // zm plane
  nn_mmm = 0, nn_0mm, nn_pmm,
  nn_m0m,     nn_00m, nn_p0m,
  nn_mpm,     nn_0pm, nn_ppm,

  // z0 plane
  nn_mm0, nn_0m0, nn_pm0,
  nn_m00, nn_000, nn_p00,
  nn_mp0, nn_0p0, nn_pp0,

  // zp plane
  nn_mmp, nn_0mp, nn_pmp,
  nn_m0p, nn_00p, nn_p0p,
  nn_mpp, nn_0pp, nn_ppp

#else
  nn_mm0 = 0, nn_0m0, nn_pm0,
  nn_m00,     nn_000, nn_p00,
  nn_mp0,     nn_0p0, nn_pp0
#endif
};

enum node_neighbor_face_t
{
#ifdef P4_TO_P8
  nnf_mm = 0, nnf_0m, nnf_pm,
  nnf_m0,     nnf_00, nnf_p0,
  nnf_mp,     nnf_0p, nnf_pp,
#else
  nnf_m0 = 0, nnf_00, nnf_p0,
#endif
};

#ifdef P4_TO_P8
const unsigned short num_neighbors_cube = 27;
const unsigned short num_neighbors_face = 9;
const unsigned short num_neighbors_cube_ = 27;
const unsigned short num_neighbors_face_ = 9;

const unsigned short f2c_m[P4EST_FACES][num_neighbors_face_] = { { nn_0mm, nn_00m, nn_0pm,
                                                                   nn_0m0, nn_000, nn_0p0,
                                                                   nn_0mp, nn_00p, nn_0pp },

                                                                 { nn_0mm, nn_00m, nn_0pm,
                                                                   nn_0m0, nn_000, nn_0p0,
                                                                   nn_0mp, nn_00p, nn_0pp },

                                                                 { nn_m0m, nn_00m, nn_p0m,
                                                                   nn_m00, nn_000, nn_p00,
                                                                   nn_m0p, nn_00p, nn_p0p },

                                                                 { nn_m0m, nn_00m, nn_p0m,
                                                                   nn_m00, nn_000, nn_p00,
                                                                   nn_m0p, nn_00p, nn_p0p },

                                                                 { nn_mm0, nn_0m0, nn_pm0,
                                                                   nn_m00, nn_000, nn_p00,
                                                                   nn_mp0, nn_0p0, nn_pp0 },

                                                                 { nn_mm0, nn_0m0, nn_pm0,
                                                                   nn_m00, nn_000, nn_p00,
                                                                   nn_mp0, nn_0p0, nn_pp0 }};

const unsigned short f2c_p[P4EST_FACES][num_neighbors_face_] = { { nn_mmm, nn_m0m, nn_mpm,
                                                                   nn_mm0, nn_m00, nn_mp0,
                                                                   nn_mmp, nn_m0p, nn_mpp },

                                                                 { nn_pmm, nn_p0m, nn_ppm,
                                                                   nn_pm0, nn_p00, nn_pp0,
                                                                   nn_pmp, nn_p0p, nn_ppp },

                                                                 { nn_mmm, nn_0mm, nn_pmm,
                                                                   nn_mm0, nn_0m0, nn_pm0,
                                                                   nn_mmp, nn_0mp, nn_pmp },

                                                                 { nn_mpm, nn_0pm, nn_ppm,
                                                                   nn_mp0, nn_0p0, nn_pp0,
                                                                   nn_mpp, nn_0pp, nn_ppp },

                                                                 { nn_mmm, nn_0mm, nn_pmm,
                                                                   nn_m0m, nn_00m, nn_p0m,
                                                                   nn_mpm, nn_0pm, nn_ppm },

                                                                 { nn_mmp, nn_0mp, nn_pmp,
                                                                   nn_m0p, nn_00p, nn_p0p,
                                                                   nn_mpp, nn_0pp, nn_ppp }};
//const unsigned short i_idx[] = { 0, 1, 2 };
//const unsigned short j_idx[] = { 1, 2, 0 };
//const unsigned short k_idx[] = { 2, 0, 1 };
const unsigned short i_idx[] = { 0, 1, 2 };
const unsigned short j_idx[] = { 1, 0, 0 };
const unsigned short k_idx[] = { 2, 2, 1 };
#else
const unsigned short num_neighbors_cube = 9;
const unsigned short num_neighbors_face = 3;
const unsigned short num_neighbors_cube_ = 9;
const unsigned short num_neighbors_face_ = 3;

const unsigned short f2c_m[P4EST_FACES][num_neighbors_face_] = { { nn_0m0, nn_000, nn_0p0 },
                                                                 { nn_0m0, nn_000, nn_0p0 },
                                                                 { nn_m00, nn_000, nn_p00 },
                                                                 { nn_m00, nn_000, nn_p00 }};

const unsigned short f2c_p[P4EST_FACES][num_neighbors_face_] = { { nn_mm0, nn_m00, nn_mp0 },
                                                                 { nn_pm0, nn_p00, nn_pp0 },
                                                                 { nn_mm0, nn_0m0, nn_pm0 },
                                                                 { nn_mp0, nn_0p0, nn_pp0 }};
const unsigned short i_idx[] = { 0, 1 };
const unsigned short j_idx[] = { 1, 0 };
#endif


//#ifdef P4_TO_P8
//const unsigned short q2c[P4EST_CHILDREN][P4EST_CHILDREN] = { { nn_mmm, nn_0mm, nn_m0m, nn_00m,
//                                                               nn_mm0, nn_0m0, nn_m00, nn_000 },

//                                                             { nn_0mm, nn_pmm, nn_00m, nn_p0m,
//                                                               nn_0m0, nn_pm0, nn_000, nn_p00 },

//                                                             { nn_m0m, nn_00m, nn_mpm, nn_0pm,
//                                                               nn_m00, nn_000, nn_mp0, nn_0p0 },

//                                                             { nn_00m, nn_p0m, nn_0pm, nn_ppm,
//                                                               nn_000, nn_p00, nn_0p0, nn_pp0 },

//                                                             { nn_mm0, nn_0m0, nn_m00, nn_000,
//                                                               nn_mmp, nn_0mp, nn_m0p, nn_00p },

//                                                             { nn_0m0, nn_pm0, nn_000, nn_p00,
//                                                               nn_0mp, nn_pmp, nn_00p, nn_p0p },

//                                                             { nn_m00, nn_000, nn_mp0, nn_0p0,
//                                                               nn_m0p, nn_00p, nn_mpp, nn_0pp },

//                                                             { nn_000, nn_p00, nn_0p0, nn_pp0,
//                                                               nn_00p, nn_p0p, nn_0pp, nn_ppp }};
//#else
//const unsigned short q2c[P4EST_CHILDREN][P4EST_CHILDREN] = { { nn_mm0, nn_0m0, nn_m00, nn_000 },
//                                                             { nn_0m0, nn_pm0, nn_000, nn_p00 },
//                                                             { nn_m00, nn_000, nn_mp0, nn_0p0 },
//                                                             { nn_000, nn_p00, nn_0p0, nn_pp0 } };
//#endif
const unsigned short q2c_num_pts = P4EST_CHILDREN;
const unsigned short t2c_num_pts = P4EST_DIM+1;

#ifdef P4_TO_P8
const unsigned short q2c[P4EST_CHILDREN][q2c_num_pts] = { { nn_000, nn_m00, nn_0m0, nn_mm0, nn_00m, nn_m0m, nn_0mm, nn_mmm },
                                                          { nn_000, nn_p00, nn_0m0, nn_pm0, nn_00m, nn_p0m, nn_0mm, nn_pmm },
                                                          { nn_000, nn_m00, nn_0p0, nn_mp0, nn_00m, nn_m0m, nn_0pm, nn_mpm },
                                                          { nn_000, nn_p00, nn_0p0, nn_pp0, nn_00m, nn_p0m, nn_0pm, nn_ppm },
                                                          { nn_000, nn_m00, nn_0m0, nn_mm0, nn_00p, nn_m0p, nn_0mp, nn_mmp },
                                                          { nn_000, nn_p00, nn_0m0, nn_pm0, nn_00p, nn_p0p, nn_0mp, nn_pmp },
                                                          { nn_000, nn_m00, nn_0p0, nn_mp0, nn_00p, nn_m0p, nn_0pp, nn_mpp },
                                                          { nn_000, nn_p00, nn_0p0, nn_pp0, nn_00p, nn_p0p, nn_0pp, nn_ppp } };

const unsigned short t2c[P4EST_CHILDREN][t2c_num_pts] = { { nn_000, nn_m00, nn_0m0, nn_00m },
                                                          { nn_000, nn_p00, nn_0m0, nn_00m },
                                                          { nn_000, nn_m00, nn_0p0, nn_00m },
                                                          { nn_000, nn_p00, nn_0p0, nn_00m },
                                                          { nn_000, nn_m00, nn_0m0, nn_00p },
                                                          { nn_000, nn_p00, nn_0m0, nn_00p },
                                                          { nn_000, nn_m00, nn_0p0, nn_00p },
                                                          { nn_000, nn_p00, nn_0p0, nn_00p },};

#else
const unsigned short q2c[P4EST_CHILDREN][q2c_num_pts] = { { nn_000, nn_m00, nn_0m0, nn_mm0 },
                                                          { nn_000, nn_p00, nn_0m0, nn_pm0 },
                                                          { nn_000, nn_m00, nn_0p0, nn_mp0 },
                                                          { nn_000, nn_p00, nn_0p0, nn_pp0 },};

const unsigned short t2c[P4EST_CHILDREN][t2c_num_pts] = { { nn_000, nn_m00, nn_0m0 },
                                                          { nn_000, nn_p00, nn_0m0 },
                                                          { nn_000, nn_m00, nn_0p0 },
                                                          { nn_000, nn_p00, nn_0p0 } };
#endif

enum interpolation_method{
  linear,
  quadratic,
  quadratic_non_oscillatory,
  quadratic_non_oscillatory_continuous_v1,
  quadratic_non_oscillatory_continuous_v2
};

class CF_1
{
public:
  double lip, t;
  virtual double operator()(double x) const=0 ;
  virtual ~CF_1() {}
};


class CF_2
{
public:
  double lip, t;
  double value(double *xyz) const {return this->operator ()(xyz[0], xyz[1]);}
  virtual double operator()(double x, double y) const=0 ;
  virtual ~CF_2() {}
};

class CF_3
{
public:
  double lip, t;
  double value(double *xyz) const {return this->operator ()(xyz[0], xyz[1], xyz[2]);}
  virtual double operator()(double x, double y, double z) const=0 ;
  virtual ~CF_3() {}
};

#ifdef P4_TO_P8
#define CF_DIM CF_3
#else
#define CF_DIM CF_2
#endif

enum {
  WALL_m00 = -1,
  WALL_p00 = -2,
  WALL_0m0 = -3,
  WALL_0p0 = -4,
  WALL_00m = -5,
  WALL_00p = -6,
  INTERFACE = -7,
  WALL_parallel_to_face = -8 // to allow for Dirichlet wall boundary conditions on the face_solver even with rectangular grids
};

typedef enum {
  DIRICHLET,
  NEUMANN,
  ROBIN,
  NOINTERFACE,
  MIXED,
  IGNORE
} BoundaryConditionType;

class mixed_interface
{
public:
  virtual BoundaryConditionType mixed_type(const double xyz_[]) const=0;
  virtual ~mixed_interface() {}
};

std::ostream& operator << (std::ostream& os, BoundaryConditionType  type);
std::istream& operator >> (std::istream& is, BoundaryConditionType& type);

class WallBC2D
{
public:
  virtual BoundaryConditionType operator()( double x, double y ) const=0 ;
  double value(double *xyz) const {return this->operator ()(xyz[0], xyz[1]);}
  virtual ~WallBC2D() = 0;
};

class WallBC3D
{
public:
  virtual BoundaryConditionType operator()( double x, double y, double z ) const=0 ;
  double value(double *xyz) const {return this->operator ()(xyz[0], xyz[1], xyz[2]);}
  virtual ~WallBC3D() = 0;
};

#ifdef P4_TO_P8
#define WallBCDIM WallBC3D
#define BoundaryConditionsDIM BoundaryConditions3D
#else
#define WallBCDIM WallBC2D
#define BoundaryConditionsDIM BoundaryConditions2D
#endif


class BoundaryConditions2D
{
private:
  const WallBC2D* WallType_;
  BoundaryConditionType InterfaceType_;
  const mixed_interface* MixedInterface;

  const CF_2 *p_WallValue;
  const CF_2 *p_InterfaceValue;
  const CF_2 *p_RobinCoef;

public:
  BoundaryConditions2D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
    MixedInterface = NULL;
    p_InterfaceValue = NULL;
    p_RobinCoef = NULL;
  }

  inline void setWallTypes( const WallBC2D& w )
  {
    WallType_ = &w;
  }

  inline const WallBC2D& getWallType() const
  {
    return *WallType_;
  }

  inline void setWallValues( const CF_2& v ){
    p_WallValue = &v;
  }

  inline void setInterfaceType(BoundaryConditionType bc, const mixed_interface* obj_= NULL){
    InterfaceType_ = bc;
    if(InterfaceType_ == MIXED)
    {
      if(obj_ == NULL)
        throw std::invalid_argument("BoundaryConditions2D::setInterfaceType(): if the interface type is set to MIXED, a pointer to a class of abstract type mixed_interface MUST be provided as well!");
      MixedInterface = obj_;
    }
  }

  inline void setInterfaceValue(const CF_2& in){
    p_InterfaceValue = &in;
  }

  inline void setRobinCoef(const CF_2& in){
    p_RobinCoef = &in;
  }

  inline const CF_2& getInterfaceValue(){
    return *p_InterfaceValue;
  }

  inline const CF_2& getWallValue(){
    return *p_WallValue;
  }

  inline const CF_2& getRobinCoef(){
    return *p_RobinCoef;
  }

  inline BoundaryConditionType wallType( double x, double y ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(x,y);
  }

  inline BoundaryConditionType wallType(const double xyz_[]) const
  {
    return wallType(xyz_[0], xyz_[1]);
  }

  inline BoundaryConditionType interfaceType() const
  {
    return InterfaceType_;
  }

  inline BoundaryConditionType interfaceType(const double* xyz) const
  {
    if(InterfaceType_ != MIXED)
      return interfaceType();
    return MixedInterface->mixed_type(xyz);
  }

  inline double wallValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y);
  }

  inline double wallValue(const double xyz_[]) const
  {
    return wallValue(xyz_[0], xyz_[1]);
  }

  inline double interfaceValue(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y);
  }
  inline double  interfaceValue(double xyz_[]) const
  {
    return interfaceValue(xyz_[0], xyz_[1]);
  }


  inline double robinCoef(double x, double y) const
  {
#ifdef CASL_THROWS
    if(p_RobinCoef == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the Robin coef has not been set on the interface.");
#endif
    return p_RobinCoef->operator ()(x,y);
  }

  // using double *xyz
  inline BoundaryConditionType wallType( double *xyz ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(xyz[0],xyz[1]);
  }

  inline double wallValue( double *xyz) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(xyz[0],xyz[1]);
  }

  inline double interfaceValue( double *xyz) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(xyz[0],xyz[1]);
  }

  inline double robinCoef( double *xyz) const
  {
#ifdef CASL_THROWS
    if(p_RobinCoef == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the Robin coef has not been set on the interface.");
#endif
    return p_RobinCoef->operator ()(xyz[0],xyz[1]);
  }
};

class BoundaryConditions3D
{
private:
  const WallBC3D* WallType_;
  BoundaryConditionType InterfaceType_;
  const mixed_interface* MixedInterface;

  const CF_3 *p_WallValue;
  const CF_3 *p_InterfaceValue;
  const CF_3 *p_RobinCoef;

public:
  BoundaryConditions3D()
  {
    WallType_ = NULL;
    p_WallValue = NULL;
    InterfaceType_ = NOINTERFACE;
    MixedInterface = NULL;
    p_InterfaceValue = NULL;
    p_RobinCoef = NULL;
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

  inline void setInterfaceType(BoundaryConditionType bc, const mixed_interface* obj_= NULL){
    InterfaceType_ = bc;
    if(InterfaceType_ == MIXED)
    {
      if(obj_ == NULL)
        throw std::invalid_argument("BoundaryConditions3D::setInterfaceType(): if the interface type is set to MIXED, a pointer to a class of abstract type mixed_interface MUST be provided as well");
      MixedInterface = obj_;
    }
  }

  inline void setInterfaceValue(const CF_3& in){
    p_InterfaceValue = &in;
  }

  inline void setRobinCoef(const CF_3& in){
    p_RobinCoef = &in;
  }

  inline const CF_3& getWallValue(){
    return *p_WallValue;
  }

  inline const CF_3& getInterfaceValue(){
    return *p_InterfaceValue;
  }

  inline const CF_3& getRobinCoef(){
    return *p_RobinCoef;
  }

  inline BoundaryConditionType wallType( double x, double y, double z ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(x,y,z);
  }

  inline BoundaryConditionType wallType(const double xyz_[]) const
  {
    return wallType(xyz_[0],xyz_[1],xyz_[2]);
  }

  inline BoundaryConditionType interfaceType() const{ return InterfaceType_;}

  inline BoundaryConditionType interfaceType(const double* xyz) const
  {
    if(InterfaceType_ != MIXED)
      return interfaceType();
    return MixedInterface->mixed_type(xyz);
  }

  inline double wallValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(x,y,z);
  }

  inline double wallValue(const double xyz_[]) const
  {
    return p_WallValue->operator ()(xyz_[0],xyz_[1],xyz_[2]);
  }

  inline double interfaceValue(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(x,y,z);
  }

  inline double  interfaceValue(double xyz_[]) const
  {
    return interfaceValue(xyz_[0], xyz_[1], xyz_[2]);
  }


  inline double robinCoef(double x, double y, double z) const
  {
#ifdef CASL_THROWS
    if(p_RobinCoef == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the Robin coef has not been set on the interface.");
#endif
    return p_RobinCoef->operator ()(x,y,z);
  }

  // using double *xyz
  inline BoundaryConditionType wallType( double *xyz ) const
  {
#ifdef CASL_THROWS
    if(WallType_ == NULL) throw std::invalid_argument("[CASL_ERROR]: The type of boundary conditions has not been set on the walls.");
#endif
    return (*WallType_)(xyz[0],xyz[1],xyz[2]);
  }

  inline double wallValue(double *xyz) const
  {
#ifdef CASL_THROWS
    if(p_WallValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the walls.");
#endif
    return p_WallValue->operator ()(xyz[0],xyz[1],xyz[2]);
  }

  inline double interfaceValue(double *xyz) const
  {
#ifdef CASL_THROWS
    if(p_InterfaceValue == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the boundary conditions has not been set on the interface.");
#endif
    return p_InterfaceValue->operator ()(xyz[0],xyz[1],xyz[2]);
  }

  inline double robinCoef( double *xyz) const
  {
#ifdef CASL_THROWS
    if(p_RobinCoef == NULL) throw std::invalid_argument("[CASL_ERROR]: The value of the Robin coef has not been set on the interface.");
#endif
    return p_RobinCoef->operator ()(xyz[0],xyz[1],xyz[2]);
  }
};

/*!
 * \brief index_of_node finds the (local) index of a node as defined within p4est, i.e. as a pest_quadrant_t structure whose level is P4EST_MAXLEVEL!
 *        The method uses a binary search through the provided nodes: its complexity is O(log(N_nodes)).
 *        The given node MUST MANDATORILY be canonicalized before being passed to this function to ensure consistency with the provided nodes: use
 *        p4est_node_canonicalize beforehand!
 * \param [in]    n node whose local index is queried!
 * \param [in]    nodes the nodes data structure
 * \param [inout] idx the local index of the node on output if found, undefined if not found (i.e. if the returned value is false)
 * \return true if the queried node exists and was found in the nodes (i.e. if the idx is valid), false otherwise.
 */
bool index_of_node(const p4est_quadrant_t *n, p4est_nodes_t* nodes, p4est_locidx_t& idx);

/*!
 * \brief linear_interpolation performs linear interpolation for a point
 * \param [in]    p4est the forest
 * \param [in]    tree_id the current tree that owns the quadrant
 * \param [in]    quad the current quarant
 * \param [in]    F a simple C-style array of size n_results*P4EST_CHILDREN, containing the values of the n_vecs function(s) at the vertices of the quadrant. __MUST__ be z-ordered
 *                F[k*P4EST_CHILDREN+i] = value of he kth function at quadrant's node i (in z-order), 0 <= i < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    xyz_global global coordinates of the point
 * \param [inout] simple C-style array of size n_results containing the results of the quadratic_interpolation of the n_results different functions at the node of interest (located at xyz_global)
 * \param [in]    n_results number of functions to be interpolated
 */
void linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global, double *results, const unsigned int n_results);
double linear_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xyz_global);

/*!
 * \brief quadratic_non_oscillatory_interpolation performs non-oscilatory quadratic interpolation for a point
 * \param [in]    p4est the forest
 * \param [in]    tree_id the current tree that owns the quadrant
 * \param [in]    quad the current quarant
 * \param [in]    F a simple C-style array of size n_results*P4EST_CHILDREN, containing the values of the n_vecs function(s) at the vertices of the quadrant. __MUST__ be z-ordered
 *                F[k*P4EST_CHILDREN+i] = value of he kth function at quadrant's node i (in z-order), 0 <= i < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    Fdd a simple C-style array of size n_results*P4EST_CHILDREN*P4EST_DIM, containing the values of the second derivatives of the function(s) at the vertices of the quadrant
 *                Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM+i] = value of the second derivative along dimension i, at quadrant's node j, of the kth function,
 *                0 <= i < P4EST_DIM, 0<= j < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    xyz_global global coordinates of the point
 * \param [inout] simple C-style array of size n_results containing the results of the quadratic_interpolation of the n_results different functions at the node of interest (located at xyz_global)
 * \param [in]    n_results number of functions to be interpolated
 */
void quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results);
double quadratic_non_oscillatory_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);
double quadratic_non_oscillatory_continuous_v1_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);
double quadratic_non_oscillatory_continuous_v2_interpolation(const p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);

/*!
 * \brief quadratic_interpolation performs quadratic interpolation for a point
 * \param [in]    p4est the forest
 * \param [in]    tree_id the current tree that owns the quadrant
 * \param [in]    quad the current quarant
 * \param [in]    F a simple C-style array of size n_results*P4EST_CHILDREN, containing the values of the n_vecs function(s) at the vertices of the quadrant. __MUST__ be z-ordered
 *                F[k*P4EST_CHILDREN+i] = value of he kth function at quadrant's node i (in z-order), 0 <= i < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    Fdd a simple C-style array of size n_results*P4EST_CHILDREN*P4EST_DIM, containing the values of the second derivatives of the function(s) at the vertices of the quadrant
 *                Fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM+i] = value of the second derivative along dimension i, at quadrant's node j, of the kth function,
 *                0 <= i < P4EST_DIM, 0<= j < P4EST_CHILDREN, 0 <= k < n_results
 * \param [in]    xyz_global global coordinates of the point
 * \param [inout] simple C-style array of size n_results containing the results of the quadratic_interpolation of the n_results different functions at the node of interest (located at xyz_global)
 * \param [in]    n_results number of functions to be interpolated
 */
void quadratic_interpolation(const p4est_t* p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global, double *results, unsigned int n_results);
double quadratic_interpolation(const p4est_t* p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fdd, const double *xyz_global);

p4est_bool_t nodes_are_equal(int mpi_size, p4est_nodes_t* nodes_1, p4est_nodes_t* nodes_2);

/*!
 * \brief VecCreateGhostNodes Creates a ghosted PETSc parallel vector on the nodes based on p4est node ordering
 * \param p4est [in]  the forest
 * \param nodes [in]  the nodes numbering data structure
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateGhostNodes(const p4est_t *p4est, p4est_nodes_t *nodes, Vec* v);

/*!
 * \brief VecCreateGhostNodesBlock Creates a ghosted block PETSc parallel vector on the nodes
 * \param p4est      [in]  p4est object
 * \param nodes      [in]  the nodes object
 * \param block_size [in]  block size of the vector
 * \param v          [out] PETSc vector
 * \return
 */
PetscErrorCode VecCreateGhostNodesBlock(const p4est_t *p4est, p4est_nodes_t *nodes, PetscInt block_size, Vec* v);

p4est_bool_t ghosts_are_equal(p4est_ghost_t* ghost_1, p4est_ghost_t* ghost_2);

/*!
 * \brief VecCreateGhostNodes Creates a ghosted PETSc parallel vector on the cells
 * \param p4est [in]  the forest
 * \param ghost [in]  the ghost cells
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateGhostCells(const p4est_t *p4est, p4est_ghost_t *ghost, Vec* v);

/*!
 * \brief VecCreateCellsNoGhost Creates a PETSc parallel vector on the cells
 * \param p4est [in]  the forest
 * \param v     [out] PETSc vector type
 */
PetscErrorCode VecCreateCellsNoGhost(const p4est_t *p4est, Vec* v);

/*!
 * \brief VecCreateGhostNodesBlock Creates a ghosted block PETSc parallel vector
 * \param p4est      [in]  p4est object
 * \param ghost      [in]  the ghost cells
 * \param block_size [in]  block size of the vector
 * \param v          [out] PETSc vector
 * \return
 */
PetscErrorCode VecCreateGhostCellsBlock(const p4est_t *p4est, p4est_ghost_t *ghost, PetscInt block_size, Vec* v);

/*!
 * \brief VecScatterCreateChangeLayout Create a VecScatter context useful for changing the parallel layout of a vector
 * \param comm  [in]  MPI_Comm to which parallel vectors belong
 * \param from  [in]  input vector layout
 * \param to    [in]  output vector layout
 * \param ctx   [out] the created VecScatter context
 * \return
 */
PetscErrorCode VecScatterCreateChangeLayout(MPI_Comm comm, Vec from, Vec to, VecScatter *ctx);

/*!
 * \brief VecGhostChangeLayoutBegin Start changing the layout of a parallel vector. This potentially involves
 *  sending and receiving messages in a non-blocking mode
 * \param ctx   [in]  VecScatter context to initiate the transfer
 * \param from  [in]  input vector to the change the parallel layout
 * \param to    [out] output vector with the same global values but with a different parallel layout
 * \return
 */
PetscErrorCode VecGhostChangeLayoutBegin(VecScatter ctx, Vec from, Vec to);

/*!
 * \brief VecGhostChangeLayoutEnd Finish changing the layout of a parallel vector. This potentially involves
 *  sending and receiving messages in a non-blocking mode
 * \param ctx   [in]  VecScatter context to initiate the transfer
 * \param from  [in]  input vector to the change the parallel layout
 * \param to    [out] output vector with the same global values but with a different parallel layout
 * \return
 */
PetscErrorCode VecGhostChangeLayoutEnd(VecScatter ctx, Vec from, Vec to);

/*!
 * \brief is_folder returns true if the path points to an existing folder
 * does not use boost nor c++17 standard to maximize portability
 * \param path: path to be checked
 * \return true if the path points to a folder
 * [throws std::runtime_error if the path cannot be accessed]
 */
bool is_folder(const char* path);

/*!
 * \brief file_exists returns true if the path points to an existing file
 * does not use boost nor c++17 standard to maximize portability
 * \param path:path to be checked
 * \return true if there exists a file corresponding to the given path
 */
bool file_exists(const char* path);

/*!
 * \brief create_directory creates a folder indicated by the given path, permission rights: 755
 * does not use boost nor c++17 standard to maximize portability (parents are created as well)
 * \param path: path to the folder to be created
 * \param mpi_rank: rank of the calling process
 * \param comm: communicator
 * \return 0 if the creation was successful, non-0 otherwise
 * [the root process creates the folder, the operation is collective by MPI_Bcast on the result]
 */
int create_directory(const char* path, int mpi_rank, MPI_Comm comm=MPI_COMM_WORLD);

/*!
 * \brief delete_directory_recursive explores a directory then
 * - it deletes all regular files in the directrory;
 * - it goes through subdirectories and calls the same function on them;
 * - after recursive call returns the subdirectory is removed;
 * does not use boost nor c++17 standard to maximize portability
 * \param root_path: path to the root directory to be entirely deleted
 * \param mpi_rank: rank of the calling process
 * \param comm: communicator
 * \param non_collective: flag skipping the (collective) steps (for recursive calls on root process)
 * \return 0 if the deletion was successful, non-0 otherwise
 * [the root process deletes the content, the operation is collective by MPI_Bcast on the final result]
 * [throws std::invalid_argument if the root_path is NOT a directory]
 */
int delete_directory(const char* root_path, int mpi_rank, MPI_Comm comm=MPI_COMM_WORLD, bool non_collective=false);

int get_subdirectories_in(const char* root_path, std::vector<std::string>& subdirectories);

inline double int2double_coordinate_transform(p4est_qcoord_t a){
  return static_cast<double>(a)/static_cast<double>(P4EST_ROOT_LEN);
}

void dxyz_min(const p4est_t *p4est, double *dxyz);

void get_dxyz_min(const p4est_t *p4est, double *dxyz, double &dxyz_min);
void get_dxyz_min(const p4est_t *p4est, double *dxyz, double &dxyz_min, double &diag_min);

void dxyz_quad(const p4est_t *p4est, const p4est_quadrant_t *quad, double *dxyz);

void xyz_min(const p4est_t *p4est, double *xyz_min_);

void xyz_max(const p4est_t *p4est, double *xyz_max_);

inline void xyz_min_max(const p4est_t *p4est, double *xyz_min_, double *xyz_max_){
  xyz_min(p4est, xyz_min_);
  xyz_max(p4est, xyz_max_);
}

inline double node_x_fr_n(const p4est_indep_t *ni){
  return ni->x == P4EST_ROOT_LEN-1 ? 1.0:static_cast<double>(ni->x)/static_cast<double>(P4EST_ROOT_LEN);
}

inline double node_y_fr_n(const p4est_indep_t *ni){
  return ni->y == P4EST_ROOT_LEN-1 ? 1.0:static_cast<double>(ni->y)/static_cast<double>(P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
inline double node_z_fr_n(const p4est_indep_t *ni){
  return ni->z == P4EST_ROOT_LEN-1 ? 1.0:static_cast<double>(ni->z)/static_cast<double>(P4EST_ROOT_LEN);
}
#endif

inline double node_x_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&const_cast<p4est_nodes_t*>(nodes)->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  return (tree_xmax-tree_xmin)*node_x_fr_n(node) + tree_xmin;
}

inline double node_y_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&const_cast<p4est_nodes_t*>(nodes)->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  return (tree_ymax-tree_ymin)*node_y_fr_n(node) + tree_ymin;
}

#ifdef P4_TO_P8
inline double node_z_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes)
{
  p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&const_cast<p4est_nodes_t*>(nodes)->indep_nodes, n);
  p4est_topidx_t tree_id = node->p.piggy3.which_tree;

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  return (tree_zmax-tree_zmin)*node_z_fr_n(node) + tree_zmin;
}
#endif

inline void node_xyz_fr_n(p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes, double *xyz)
{
  xyz[0] = node_x_fr_n(n,p4est,nodes);
  xyz[1] = node_y_fr_n(n,p4est,nodes);
#ifdef P4_TO_P8
  xyz[2] = node_z_fr_n(n,p4est,nodes);
#endif
}

inline void p4est_dxyz_min(const p4est_t* p4est, double *dxyz)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN-1];
  const double *vert = p4est->connectivity->vertices;

  double h = 1.0 / (double) (1 << data->max_lvl);
  for (short i=0; i<P4EST_DIM; ++i)
    dxyz[i] = (vert[3*vp + i] - vert[3*vm + i]) * h;
}

inline void p4est_dxyz_max(const p4est_t* p4est, double *dxyz)
{
  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;
  p4est_topidx_t tr_idx = p4est->trees->elem_count - 1;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[tr_idx * P4EST_CHILDREN + P4EST_CHILDREN-1];
  const double *vert = p4est->connectivity->vertices;

  double h = 1.0 / (double) (1 << data->min_lvl);
  for (short i=0; i<P4EST_DIM; ++i)
    dxyz[i] = (vert[3*vp + i] - vert[3*vm + i]) * h;
}

inline double p4est_diag_min(const p4est_t* p4est) {
  double dx[P4EST_DIM];
  p4est_dxyz_min(p4est, dx);
#ifdef P4_TO_P8
  return sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
#else
  return sqrt(SQR(dx[0]) + SQR(dx[1]));
#endif
}

inline double p4est_diag_max(const p4est_t* p4est) {
  double dx[P4EST_DIM];
  p4est_dxyz_max(p4est, dx);
#ifdef P4_TO_P8
  return sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
#else
  return sqrt(SQR(dx[0]) + SQR(dx[1]));
#endif
}

/*!
 * \brief get the z-coordinate of the bottom left corner of a quadrant in the local tree coordinate system
 */
inline double quad_x_fr_i(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->x)/static_cast<double>(P4EST_ROOT_LEN);
}

/*!
 * \brief get the y-coordinate of the bottom left corner of a quadrant in the local tree coordinate system
 */
inline double quad_y_fr_j(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->y)/static_cast<double>(P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
/*!
 * \brief get the x-coordinate of the bottom left corner of a quadrant in the local tree coordinate system
 */
inline double quad_z_fr_k(const p4est_quadrant_t *qi){
  return static_cast<double>(qi->z)/static_cast<double>(P4EST_ROOT_LEN);
}
#endif

inline p4est_tree_t* get_tree(p4est_topidx_t tr, p4est_t* p4est)
{
#ifdef CASL_THROWS
  if(tr < p4est->first_local_tree || tr > p4est->last_local_tree) {
    std::ostringstream oss;
    oss << "Tree with index " << tr << " is outside range. Processor " << p4est->mpirank
        << " inclusive range is [" << p4est->first_local_tree << ", " << p4est->last_local_tree << "]" << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_tree_t*)sc_array_index(p4est->trees, tr);
}

inline p4est_quadrant_t* get_quad(p4est_locidx_t q, p4est_tree_t* tree)
{
#ifdef CASL_THROWS
  if(q < 0 || q >= (p4est_locidx_t) tree->quadrants.elem_count) {
    std::ostringstream oss;
    oss << "Quad with index " << q << " is outside range of current tree. "
        << "Number of quadrants on this tree is " << tree->quadrants.elem_count << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
}

inline p4est_quadrant_t* get_quad(p4est_locidx_t q, p4est_ghost_t* ghost)
{
#ifdef CASL_THROWS
  if(q < 0 || q >= (p4est_locidx_t) ghost->ghosts.elem_count) {
    std::ostringstream oss;
    oss << "Quad with index " << q << " is outside range of ghost layer. "
        << "Size of ghost layer is " << ghost->ghosts.elem_count << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
}

inline p4est_indep_t* get_node(p4est_locidx_t n, p4est_nodes_t* nodes)
{
#ifdef CASL_THROWS
  if(n < 0 || n >= (p4est_locidx_t) nodes->indep_nodes.elem_count) {
    std::ostringstream oss;
    oss << "Node with index " << n << " is outside range of nodes." << std::endl;
    throw std::out_of_range(oss.str());
  }
#endif

  return (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
}

/*!
 * \brief get the x-coordinate of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline double quad_x_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost)
{
  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  return (tree_xmax-tree_xmin)*(quad_x_fr_i(quad) + .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_xmin;
}

/*!
 * \brief quad_x        compute the x-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  the x-coordinate
 */
inline double quad_x(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];
  return (tree_xmax-tree_xmin)*(quad_x_fr_i(quad) + 0.5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_xmin;
}

/*!
 * \brief quad_dx     compute the dx size of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  dx
 */
inline double quad_dx(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_xmin = p4est->connectivity->vertices[3*v_m + 0];
  double tree_xmax = p4est->connectivity->vertices[3*v_p + 0];

  return (tree_xmax-tree_xmin)*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
}

/*!
 * \brief get the y-coordinate of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline double quad_y_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost)
{
  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  return (tree_ymax-tree_ymin)*(quad_y_fr_j(quad) + .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_ymin;
}

/*!
 * \brief quad_y        compute the y-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  the y-coordinate
 */
inline double quad_y(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];
  return (tree_ymax-tree_ymin)*(quad_y_fr_j(quad) + 0.5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_ymin;
}

/*!
 * \brief quad_dy     compute the dy size of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  dy
 */
inline double quad_dy(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_ymin = p4est->connectivity->vertices[3*v_m + 1];
  double tree_ymax = p4est->connectivity->vertices[3*v_p + 1];

  return (tree_ymax-tree_ymin)*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
}

#ifdef P4_TO_P8
/*!
 * \brief get the z-coordinate of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline double quad_z_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost)
{
  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  return (tree_zmax-tree_zmin)*(quad_z_fr_k(quad) + .5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_zmin;
}

/*!
 * \brief quad_z        compute the y-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  the z-coordinate
 */
inline double quad_z(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];
  return (tree_zmax-tree_zmin)*(quad_z_fr_k(quad) + 0.5*(double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN) + tree_zmin;
}

/*!
 * \brief quad_dz     compute the dz size of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \return  dz
 */
inline double quad_dz(const p4est_t *p4est, const p4est_quadrant_t *quad)
{
  p4est_locidx_t tree_idx = quad->p.piggy3.which_tree;
  p4est_topidx_t v_m = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
  p4est_topidx_t v_p = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
  double tree_zmin = p4est->connectivity->vertices[3*v_m + 2];
  double tree_zmax = p4est->connectivity->vertices[3*v_p + 2];

  return (tree_zmax-tree_zmin)*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
}
#endif


/*!
 * \brief get the xyz-coordinates of the center of a quadrant
 * \param quad_idx the index of the quadrant in the local forest, NOT in the tree tree_idx !!
 */
inline void quad_xyz_fr_q(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const p4est_t *p4est, p4est_ghost_t *ghost, double *xyz)
{
  xyz[0] = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
  xyz[1] = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
  xyz[2] = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif
}

/*!
 * \brief quad_z_fr_q   compute the y-coordinate of the center of a quadrant
 * \param p4est [in]    const pointer to the p4est structure
 * \param quad  [in]    const pointer to the quadrant.
 *        NOTE: Assumes that the piggy3 member if filled
 * \param xyz   [out]   pointer to array of size P4EST_DIM to store xyz
 * \return  the z-coordinate
 */
inline void quad_xyz(const p4est_t *p4est, const p4est_quadrant_t *quad, double *xyz)
{
  xyz[0] = quad_x(p4est, quad);
  xyz[1] = quad_y(p4est, quad);
#ifdef P4_TO_P8
  xyz[2] = quad_z(p4est, quad);
#endif
}

/*!
 * \brief quad_dxyz   compute the dxyz sizes of the a quadrant
 * \param p4est [in]  const pointer to the p4est structure
 * \param quad  [in]  const pointer to the quadrant structure
 *        NOTE: Assumes that the piggy3 member if filled
 * \param dxyz  [out]   pointer to array of size P4EST_DIM to store dxyz
 * \return  dy
 */
inline void quad_dxyz(const p4est_t *p4est, const p4est_quadrant_t *quad, double *dxyz)
{
  dxyz[0] = quad_dx(p4est, quad);
  dxyz[1] = quad_dy(p4est, quad);
#ifdef P4_TO_P8
  dxyz[2] = quad_dz(p4est, quad);
#endif
}

/*!
 * \brief compute the xyz_min of a given tree index
 * \param p4est the forest object
 * \param tr_idx index of the tree to find the xyz_min
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void tree_xyz_min(p4est_t* p4est, p4est_topidx_t tr_idx, double *xyz)
{
  p4est_topidx_t vtx = p4est->connectivity->tree_to_vertex[tr_idx*P4EST_CHILDREN];
  xyz[0] = p4est->connectivity->vertices[3*vtx + 0];
  xyz[1] = p4est->connectivity->vertices[3*vtx + 1];
#ifdef P4_TO_P8
  xyz[2] = p4est->connectivity->vertices[3*vtx + 2];
#endif
}

/*!
 * \brief compute the xyz_max of a given tree index
 * \param p4est the forest object
 * \param tr_idx index of the tree to find the xyz_max
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void tree_xyz_max(p4est_t* p4est, p4est_topidx_t tr_idx, double *xyz)
{
  p4est_topidx_t vtx = p4est->connectivity->tree_to_vertex[tr_idx*P4EST_CHILDREN + P4EST_CHILDREN - 1];
  xyz[0] = p4est->connectivity->vertices[3*vtx + 0];
  xyz[1] = p4est->connectivity->vertices[3*vtx + 1];
#ifdef P4_TO_P8
  xyz[2] = p4est->connectivity->vertices[3*vtx + 2];
#endif
}

/*!
 * \brief computes the xyz_min of the entire forest
 * \param p4est teh forest object
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void p4est_xyz_min(p4est_t* p4est, double *xyz)
{
  tree_xyz_min(p4est, 0, xyz);
}

/*!
 * \brief computes the xyz_max of the entire forest
 * \param p4est teh forest object
 * \param xyz pointer to an array of double[P4EST_DIM]
 */
inline void p4est_xyz_max(p4est_t* p4est, double *xyz)
{
  tree_xyz_max(p4est, p4est->trees->elem_count - 1, xyz);
}

/*!
 * \brief integrate_over_negative_domain_in_one_quadrant
 */
double integrate_over_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

/*!
 * \brief integrate_over_negative_domain integrate a quantity f over the negative domain defined by phi
 *        note: second order convergence
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi
 * \param f the scalar to integrate
 * \return the integral of f over the phi<0 domain, \int_{\phi<0} f
 */
double integrate_over_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f);

/*!
 * \brief area_in_negative_domain_in_one_quadrant
 */
double area_in_negative_domain_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

/*!
 * \brief area_in_negative_domain compute the area of the negative domain defined by phi
 *        note: second order convergence
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \return the area in the negative phi domain, i.e. \int_{phi<0} 1
 */
double area_in_negative_domain(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi);

/*!
 * \brief integrate_over_interface_in_one_quadrant
 */
double integrate_over_interface_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

/*!
 * \brief integrate_over_interface integrate a scalar f over the 0-contour of the level-set function phi.
 *        note: first order convergence only
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \param f the scalar to integrate
 * \return the integral of f over the contour defined by phi, i.e. \int_{phi=0} f
 */
double integrate_over_interface(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi, Vec f);

/*!
 * \brief compute_mean_curvature computes the mean curvature using compact stencil k = -div(n)
 * \param qnnn neighborhood information for the point
 * \param phi pointer to the level set function
 * \param phi_x pointer to an array of size P4EST_DIM for the first derivatives of levelset. CANNOT be NULL.
 * \return mean curvature at a single point
 */
double compute_mean_curvature(const quad_neighbor_nodes_of_node_t& qnnn, double* phi, double* phi_x[P4EST_DIM]);

/*!
 * \brief compute_mean_curvature computes the mean curvature using divergence of normal k = -div(n)
 * \param qnnn neighborhood information for the point
 * \param normals pointer to an array of size P4EST_DIM of the normals. CANNOT be NULL.
 * \return mean curvature at a single point
 */
double compute_mean_curvature(const quad_neighbor_nodes_of_node_t& qnnn, double* normals[P4EST_DIM]);

/*!
 * \brief compute_mean_curvature computes the mean curvature in the entire domain k = -div(n)
 * \param neighbors the node neighborhood information
 * \param phi levelset function
 * \param phi_x an array of size P4EST_DIM representing the first derivative of levelset in the entire domain. CANNOT be NULL.
 * \param kappa curvature function in the entire domain
 */
void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec phi, Vec phi_x[P4EST_DIM], Vec kappa);

/*!
 * \brief compute_mean_curvature computes the mean curvature in the entire domain k = -div(n)
 * \param neighbors the node neighborhood information
 * \param normals pointer to an array of size P4EST_DIM for the normals. CANNOT be NULL.
 * \param kappa curvature function in the entire domain
 */
void compute_mean_curvature(const my_p4est_node_neighbors_t &neighbors, Vec normals[P4EST_DIM], Vec kappa);

/*!
 * \brief compute_normals computes the (scaled) normal to the surface
 * \param [in]  qnnn    neighborhood information for the point
 * \param [in]  phi     pointer to the levelset function
 * \param [out] normals array of size P4EST_DIM for the normals
 */
void compute_normals(const quad_neighbor_nodes_of_node_t& qnnn, double *phi, double normals[P4EST_DIM]);

/*!
 * \brief compute_normals computes the (scaled) normal to the surface for the entire grid
 * \param [in]  neighbors the neighborhood information
 * \param [in]  phi       PETSc vector of the levelset function
 * \param phi the level-set function
 * \param [out] normals   array of size P4EST_DIM of PETSc vectors to store the normal in the entire doamin
 */
void compute_normals(const my_p4est_node_neighbors_t& neighbors, Vec phi, Vec normals[P4EST_DIM]);

/*!
 * \brief interface_length_in_one_quadrant
 */
double interface_length_in_one_quadrant(const p4est_t *p4est, const p4est_nodes_t *nodes, const p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

/*!
 * \brief interface_length
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \return the length (or area in 3D) of the contour defined by phi
 */
double interface_length(const p4est_t *p4est, const p4est_nodes_t *nodes, Vec phi);

/*!
 * \brief is_node_xmWall checks if a node is on x^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the left domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_xpWall checks if a node is on x^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the right domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_xpWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_ymWall checks if a node is on y^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain bottom boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_ymWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_ymWall checks if a node is on y^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain top boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_ypWall(const p4est_t *p4est, const p4est_indep_t *ni);

#ifdef P4_TO_P8
/*!
 * \brief is_node_zmWall checks if a node is on z^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain back boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_zmWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_zpWall checks if a node is on z^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain front boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_zpWall(const p4est_t *p4est, const p4est_indep_t *ni);
#endif

/*!
 * \brief is_node_Wall checks if a node is on any of domain boundaries
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_Wall  (const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_Wall checks if a node is on any of domain boundaries
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_node_Wall  (const p4est_t *p4est, const p4est_indep_t *ni, bool is_wall[]);

/*!
 * \brief is_quad_xmWall checks if a quad is on x^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the left domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_xmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_xpWall checks if a quad is on x^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the right domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_xpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_ymWall checks if a quad is on y^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the bottom domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_ymWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_ypWall checks if a quad is on y^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the top domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_ypWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_zmWall checks if a quad is on z^- domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the back domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_zmWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_zpWall checks if a quad is on z^+ domain boundary
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the front domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_zpWall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_quad_Wall checks if a quad is on the domain boundary in a given direction
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \param dir   [in] the direction to check, dir::f_m00, dir::f_p00, dir::f_0m0 ...
 * \return true if the quad is on the domain boundary in the direction dir and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_Wall(const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi, int dir);

/*!
 * \brief is_quad_Wall checks if a quad is on any of domain boundaries
 * \param p4est [in] p4est
 * \param qi    [in] pointer to the quadrant
 * \return true if the quad is on the domain boundary and p4est is _NOT_ periodic
 * \note: periodicity is not implemented
 */
bool is_quad_Wall  (const p4est_t *p4est, p4est_topidx_t tr_it, const p4est_quadrant_t *qi);

/*!
 * \brief is_periodic checks if the forest is periodic in direction dir
 * \param p4est [in] the forest
 * \param dir   [in] the direction to check, 0 (x), 1 (y) or 2 (z, only in 3D)
 * \return true if the forest is periodic in direction dir, false otherwise
 */
inline bool is_periodic(const p4est_t *p4est, int dir)
{
  /* check whether there is not a boundary on the left side of first tree */
  P4EST_ASSERT (0 <= dir && dir < P4EST_DIM);

  const int face = 2 * dir;
  const p4est_topidx_t tfindex = 0 * P4EST_FACES + face;

  return !(p4est->connectivity->tree_to_tree[tfindex] == 0 &&
           p4est->connectivity->tree_to_face[tfindex] == face);
}

/*!
 * \brief is_periodic checks if the forest is periodic in any direction
 * \param p4est [in] the forest
 * \return true if the forest is periodic, false otherwise
 */
inline bool is_periodic(const p4est_t *p4est)
{
#ifdef P4_TO_P8
  return is_periodic(p4est, 0) || is_periodic(p4est, 1) || is_periodic(p4est, 2);
#else
  return is_periodic(p4est, 0) || is_periodic(p4est, 1);
#endif
}

/*!
 * \brief find the owner rank of a ghost quadrant
 * \param ghost the ghost structure
 * \param ghost_idx the index of the ghost quadrant (between 0 and the number of ghost quadrants)
 * \return the rank who owns the ghost quadrant
 */
int quad_find_ghost_owner(const p4est_ghost_t *ghost, p4est_locidx_t ghost_idx);

/*!
 * \brief sample_cf_on_nodes samples a cf function on the nodes. both local and ghost poinst are considered
 * \param p4est [in] the p4est object
 * \param nodes [in] the nodes data structure
 * \param cf    [in] the cf function. It is assumed that the function can be evaluated at _ANY_ point, whether local or remote
 * \param f     [in, out] a PETSc Vec object to store the result. It is assumed that the vector is allocated. A check
 * is performed to ensure enough memory is available in the Vec object.
 */
#ifdef P4_TO_P8
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, Vec f);
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_3& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3* cf_array[], Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_3& cf, std::vector<double>& f);
#else
void sample_cf_on_local_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);
void sample_cf_on_cells(const p4est_t *p4est, p4est_ghost_t *ghost, const CF_2& cf, Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2* cf_array[], Vec f);
void sample_cf_on_nodes(const p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, std::vector<double>& f);
#endif

void write_comm_stats(const p4est_t *p4est, const p4est_ghost_t* ghost, const p4est_nodes_t *nodes,
                 const char* partition_name = NULL, const char* topology_name = NULL, const char* neighbors_name = NULL);

inline double ranged_rand(double a, double b, int seed = 0){
  if (seed) srand(seed);
  return (static_cast<double>(rand())/static_cast<double>(RAND_MAX) * (b-a) + a);
}

inline int ranged_rand(int a, int b, int seed = 0){
  if (seed) srand(seed);
  return (rand()%(b-a) + a);
}

inline int ranged_rand_inclusive(int a, int b, int seed = 0){
  if (seed) srand(seed);
  return (rand()%(b-a+1) + a);
}

// A Logger for interpolation function
struct InterpolatingFunctionLogEntry{
  int num_local_points, num_send_points, num_send_procs, num_recv_points, num_recv_procs;
};

class InterpolatingFunctionLogger{
  InterpolatingFunctionLogger() {}
  InterpolatingFunctionLogger(const InterpolatingFunctionLogger& ) {}
  static std::vector<InterpolatingFunctionLogEntry> entries;

public:
  inline static InterpolatingFunctionLogger& get_instance() {
    static InterpolatingFunctionLogger instance;
    return instance;
  }

  inline void log(const InterpolatingFunctionLogEntry& entry) {
    entries.push_back(entry);
  }

  inline void write(const std::string& filename) {
    for (size_t i = 0; i<entries.size();i++) {
      FILE *fp;
      std::ostringstream oss; oss << filename << "_" << i << ".dat";
      PetscFOpen(PETSC_COMM_WORLD, oss.str().c_str(), "w", &fp);
      PetscFPrintf(PETSC_COMM_WORLD, fp, "%% num_local_points | num_send_points | num_send_procs | num_recv_points | num_recv_procs \n");
      PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%7d \t %7d \t %4d \t %7d \t %4d \n", entries[i].num_local_points,
                                                                                           entries[i].num_send_points,
                                                                                           entries[i].num_send_procs,
                                                                                           entries[i].num_recv_points,
                                                                                           entries[i].num_recv_procs);
      PetscSynchronizedFlush(PETSC_COMM_WORLD, fp);
      PetscFClose(PETSC_COMM_WORLD, fp);
    }
    entries.clear();
  }
};

/*!
 * \brief prepares MPI, PETSc, p4est, and sc libraries
 */
class mpi_environment_t{
  PetscErrorCode ierr;
  MPI_Comm mpicomm;
  int mpirank;
  int mpisize;

public:
  ~mpi_environment_t(){
    ierr = PetscFinalize(); CHKERRXX(ierr);
    sc_finalize();
    MPI_Finalize();
  }

  void init(int argc, char **argv){
    mpicomm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(mpicomm, &mpisize);
    MPI_Comm_rank(mpicomm, &mpirank);

    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);

#ifdef DEBUG
    sc_init (mpicomm, P4EST_FALSE, P4EST_FALSE, NULL, SC_LP_DEFAULT); // to allow easy debugging --> backtracks the P4EST_ASSERTs!
#else
    sc_init (mpicomm, P4EST_FALSE, P4EST_FALSE, NULL, SC_LP_SILENT);
#endif
    p4est_init (NULL, SC_LP_SILENT);
#ifdef CASL_LOG_EVENTS
    register_petsc_logs();
#endif
  }

  inline const MPI_Comm& comm() const {return mpicomm;}
  inline const int& rank() const {return mpirank;}
  inline const int& size() const {return mpisize;}

};

class parStopWatch{
public:
  typedef enum{
    root_timings,
    all_timings
  } stopwatch_timing;

private:
  double ts, tf;
  MPI_Comm comm_;
  int mpirank;
  int mpisize;
  std::string msg_;
  stopwatch_timing timing_;
  std::vector<double> t;
  FILE *f_;

public:

  parStopWatch(stopwatch_timing timing = root_timings, FILE *f = stdout, MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm), timing_(timing), f_(f)
  {
    MPI_Comm_rank(comm_, &mpirank);
    MPI_Comm_size(comm_, &mpisize);
    t.resize(mpisize,0);
  }

  void start(const std::string& msg){
    msg_ = msg;
    if(msg_.length() > 0)
      PetscFPrintf(comm_, f_, "%s ... \n", msg.c_str());
    ts = MPI_Wtime();
  }

  void stop(){
    tf = MPI_Wtime();
  }

  double read_duration(){
    double elap = tf - ts;
    if (timing_ == all_timings)
      MPI_Gather(&elap, 1, MPI_DOUBLE, &t[0], 1, MPI_DOUBLE, 0, comm_);
    return elap;
  }

  void print_stats_only(){
    if(timing_ != all_timings)
    {
      PetscFPrintf(comm_, stderr, "parStopWatch::print_stats_only() can be called only in 'all_timing' mode.");
      return;
    }
    print_duration(true);
  }

  double print_duration(bool print_stats_only_ = false){
    double elap = read_duration();
    PetscPrintf(comm_, "%s ... done in \n", msg_.c_str());
    if (timing_ == all_timings){
      double tmax, tmin, tavg, tdev;
      tmax = tmin = elap;
      tavg = tdev = 0;
      if (mpirank == 0){
        if(!print_stats_only_)
          PetscFPrintf(comm_, f_, "t = [");
        for (size_t i=0; i<t.size()-1; i++){
          if(!print_stats_only_)
            PetscFPrintf(comm_, f_, "%.5lf, ", t[i]);
          tavg += t[i];
          tmax = MAX(tmax, t[i]);
          tmin = MIN(tmin, t[i]);
        }
        if(!print_stats_only_)
          PetscFPrintf(comm_, f_, "%.5lf];\n", t.back());

        tavg += t.back();
        tmax = MAX(tmax, t.back());
        tmin = MIN(tmin, t.back());

        tavg /= mpisize;

        for (size_t i=0; i<t.size(); i++){
          tdev += (t[i]-tavg)*(t[i]-tavg);
        }
        tdev = sqrt(tdev/mpisize);
      }

      PetscFPrintf(comm_, f_, " t_max = %.5lf (s), t_max/t_min = %.2lf, t_avg = %.5lf (s), t_dev/t_avg = %% %2.1lf, t_dev/(t_max-t_min) = %% %2.1lf\n\n", tmax, tmax/tmin, tavg, tdev/tavg*100, tdev/(tmax-tmin)*100);
    } else {
      PetscFPrintf(comm_, f_, " %.5lf secs. on process %d [Note: only showing root's timings]\n\n", elap, mpirank);
    }
    return elap;
  }

  double read_duration_current(){
    double elap = MPI_Wtime() - ts;

    PetscPrintf(comm_, "%s ... done in \n", msg_.c_str());
    if (timing_ == all_timings){
      MPI_Gather(&elap, 1, MPI_DOUBLE, &t[0], 1, MPI_DOUBLE, 0, comm_);
      double tmax, tmin, tavg, tdev;
      tmax = tmin = elap;
      tavg = tdev = 0;
      if (mpirank == 0){
        PetscFPrintf(comm_, f_, "t = [");
        for (size_t i=0; i<t.size()-1; i++)
          PetscFPrintf(comm_, f_, "%.5lf, ", t[i]);
        PetscFPrintf(comm_, f_, "%.5lf];\n", t.back());

        for (size_t i=0; i<t.size(); i++){
          tavg += t[i];
          tmax = MAX(tmax, t[i]);
          tmin = MIN(tmin, t[i]);
        }
        tavg /= mpisize;

        for (size_t i=0; i<t.size(); i++){
          tdev += (t[i]-tavg)*(t[i]-tavg);
        }
        tdev = sqrt(tdev/mpisize);
      }

      PetscFPrintf(comm_, f_, " t_max = %.5lf (s), t_max/t_min = %.2lf, t_avg = %.5lf (s), t_dev/t_avg = %% %2.1lf, t_dev/(t_max-t_min) = %% %2.1lf\n\n", tmax, tmax/tmin, tavg, tdev/tavg*100, tdev/(tmax-tmin)*100);
    } else {
      PetscFPrintf(comm_, f_, " %.5lf secs. on process %d [Note: only showing root's timings]\n\n", elap, mpirank);
    }
    return elap;
  }
};

/*!
 * \brief prodives a CF_2/CF_3 interface to interpolation on quadrants
 */

#ifdef P4_TO_P8
class quadrant_interp_t : public CF_3
#else
class quadrant_interp_t : public CF_2
#endif
{
  p4est_t *p4est_;
  p4est_topidx_t tree_idx_;
  const p4est_quadrant_t *quad_;
  interpolation_method method_;
  std::vector<double> *F_;
  std::vector<double> *Fdd_;

public:
  quadrant_interp_t(p4est_t *p4est, p4est_topidx_t tree_idx, const p4est_quadrant_t *quad, interpolation_method method, std::vector<double> *F, std::vector<double> *Fdd = NULL)
    : p4est_(p4est), tree_idx_(tree_idx), quad_(quad), method_(method), F_(F), Fdd_(Fdd) {}

  void reinit(p4est_t *p4est, p4est_topidx_t tree_idx, p4est_quadrant_t *quad, interpolation_method method, std::vector<double> *F, std::vector<double> *Fdd = NULL)
  {
    p4est_ = p4est;
    tree_idx_ = tree_idx;
    quad_ = quad;
    method_ = method;
    F_ = F;
    Fdd_ = Fdd;
  }

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const;
#else
  double operator()(double x, double y) const;
#endif

//#ifdef P4_TO_P8
//  double operator()(double x, double y, double z) const
//  {
//    double xyz_node[P4EST_DIM] = { x, y, z};
//#else
//  double operator()(double x, double y) const
//  {
//    double xyz_node[P4EST_DIM] = { x, y };
//#endif

//#ifdef CASL_THROWS
//    if (F_ == NULL) throw std::invalid_argument("[CASL_ERROR]: Values are not provided for interpolation.");
//    if (Fdd_ == NULL && (method_ == quadratic || method_ == quadratic_non_oscillatory) ) throw std::invalid_argument("[CASL_ERROR]: Second order derivatives are not provided for quadratic interpolation.");
//#endif

//    switch (method_)
//    {
//      case linear:                    return linear_interpolation                   (p4est_, tree_idx_, *quad_, F_, xyz_node); break;
//      case quadratic:                 return quadratic_interpolation                (p4est_, tree_idx_, *quad_, F_, Fdd_, xyz_node); break;
//      case quadratic_non_oscillatory: return quadratic_non_oscillatory_interpolation(p4est_, tree_idx_, *quad_, F_, Fdd_, xyz_node); break;
//    }
//  }
};

void copy_ghosted_vec(Vec input, Vec output);
void set_ghosted_vec(Vec vec, double scalar);
void shift_ghosted_vec(Vec vec, double scalar);
void scale_ghosted_vec(Vec vec, double scalar);

void invert_phi(p4est_nodes_t *nodes, Vec phi);

PetscErrorCode VecCopyGhost(Vec input, Vec output);
PetscErrorCode VecSetGhost(Vec vec, PetscScalar scalar);
PetscErrorCode VecShiftGhost(Vec vec, PetscScalar scalar);
PetscErrorCode VecScaleGhost(Vec vec, PetscScalar scalar);
PetscErrorCode VecPointwiseMultGhost(Vec output, Vec input1, Vec input2);
PetscErrorCode VecPointwiseMinGhost(Vec output, Vec input1, Vec input2);
PetscErrorCode VecPointwiseMaxGhost(Vec output, Vec input1, Vec input2);
PetscErrorCode VecAXPBYGhost(Vec y, PetscScalar alpha, PetscScalar beta, Vec x);
PetscErrorCode VecReciprocalGhost(Vec input);

struct vec_and_ptr_t
{
  static PetscErrorCode ierr;

  Vec     vec;
  double *ptr;

  vec_and_ptr_t() : vec(NULL), ptr(NULL) {}

  vec_and_ptr_t(Vec parent) : vec(NULL), ptr(NULL) { create(parent); }

  vec_and_ptr_t(p4est_t *p4est, p4est_nodes_t *nodes) : vec(NULL), ptr(NULL) { create(p4est, nodes); }

  inline void create(Vec parent)
  {
    ierr = VecDuplicate(parent, &vec); CHKERRXX(ierr);
  }

  inline void create(p4est_t *p4est, p4est_nodes_t *nodes)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &vec); CHKERRXX(ierr);
  }

  inline void destroy()
  {
    if (vec != NULL) { ierr = VecDestroy(vec); CHKERRXX(ierr); }
  }

  inline void get_array()
  {
    ierr = VecGetArray(vec, &ptr); CHKERRXX(ierr);
  }

  inline void restore_array()
  {
    ierr = VecRestoreArray(vec, &ptr); CHKERRXX(ierr);
  }

  inline void set(Vec input)
  {
    vec = input;
  }
};


struct vec_and_ptr_dim_t
{
  static PetscErrorCode ierr;

  Vec     vec[P4EST_DIM];
  double *ptr[P4EST_DIM];

  vec_and_ptr_dim_t()
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      vec[dim] = NULL;
      ptr[dim] = NULL;
    }
  }

  vec_and_ptr_dim_t(Vec parent[])
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      vec[dim] = NULL;
      ptr[dim] = NULL;
    }
    create(parent);
  }

  vec_and_ptr_dim_t(p4est_t *p4est, p4est_nodes_t *nodes)
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      vec[dim] = NULL;
      ptr[dim] = NULL;
    }
    create(p4est, nodes);
  }

  inline void create(Vec parent[])
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecDuplicate(parent[dim], &vec[dim]); CHKERRXX(ierr);
    }
  }

  inline void create(p4est_t *p4est, p4est_nodes_t *nodes)
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &vec[dim]); CHKERRXX(ierr);
    }
  }

  inline void destroy()
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      if (vec[dim] != NULL) { ierr = VecDestroy(vec[dim]); CHKERRXX(ierr); }
    }
  }

  inline void get_array()
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecGetArray(vec[dim], &ptr[dim]); CHKERRXX(ierr);
    }
  }

  inline void restore_array()
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecRestoreArray(vec[dim], &ptr[dim]); CHKERRXX(ierr);
    }
  }

  inline void set(Vec input[])
  {
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      vec[dim] = input[dim];
    }
  }
};

struct vec_and_ptr_array_t
{
  static PetscErrorCode ierr;

  int i, size;
  std::vector<Vec>      vec;
  std::vector<double *> ptr;

  vec_and_ptr_array_t() : size(0) {}

  vec_and_ptr_array_t(int size) : size(size), vec(size, NULL), ptr(size, NULL) {}

  vec_and_ptr_array_t(int size, Vec parent) : size(size), vec(size, NULL), ptr(size, NULL) { create(parent); }

  vec_and_ptr_array_t(int size, Vec parent[]) : size(size), vec(size, NULL), ptr(size, NULL) { create(parent); }

  vec_and_ptr_array_t(int size, p4est_t *p4est, p4est_nodes_t *nodes) : size(size), vec(size, NULL), ptr(size, NULL) { create(p4est, nodes); }

  inline void resize(int size)
  {
    this->size = size;
    vec.assign(size, NULL);
    ptr.assign(size, NULL);
  }

  inline void create(Vec parent)
  {
    for (i = 0; i < size; ++i)
    {
      ierr = VecDuplicate(parent, &vec[i]); CHKERRXX(ierr);
    }
  }

  inline void create(Vec parent[])
  {
    for (i = 0; i < size; ++i)
    {
      ierr = VecDuplicate(parent[i], &vec[i]); CHKERRXX(ierr);
    }
  }

  inline void create(p4est_t *p4est, p4est_nodes_t *nodes)
  {
    for (i = 0; i < size; ++i)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &vec[i]); CHKERRXX(ierr);
    }
  }

  inline void destroy()
  {
    for (i = 0; i < size; ++i)
    {
      if (vec[i] != NULL) { ierr = VecDestroy(vec[i]); CHKERRXX(ierr); }
    }
  }

  inline void get_array()
  {
    for (i = 0; i < size; ++i)
    {
      ierr = VecGetArray(vec[i], &ptr[i]); CHKERRXX(ierr);
    }
  }

  inline void restore_array()
  {
    for (i = 0; i < size; ++i)
    {
      ierr = VecRestoreArray(vec[i], &ptr[i]); CHKERRXX(ierr);
    }
  }

  inline void set(Vec input[])
  {
    for (int i = 0; i < size; ++i)
    {
      vec[i] = input[i];
    }
  }
};

void compute_normals_and_mean_curvature(const my_p4est_node_neighbors_t &neighbors, const Vec phi, Vec normals[], Vec kappa);

void save_vector(const char *filename, const std::vector<double> &data, std::ios_base::openmode mode = std::ios_base::out, char delim = ',');

template<typename T>
bool contains(const std::vector<T> &vec, const T& elem)
{
  return find(vec.begin(), vec.end(), elem)!=vec.end();
}

void fill_island(const my_p4est_node_neighbors_t &ngbd, const double *phi_p, double *island_number_p, int number, p4est_locidx_t n);
void find_connected_ghost_islands(const my_p4est_node_neighbors_t &ngbd, const double *phi_p, double *island_number_p, p4est_locidx_t n, std::vector<double> &connected, std::vector<bool> &visited);
void compute_islands_numbers(const my_p4est_node_neighbors_t &ngbd, const Vec phi, int &nb_islands_total, Vec island_number);

//void get_all_neighbors(const p4est_locidx_t n, const p4est_t *p4est, const p4est_nodes_t *nodes, const my_p4est_node_neighbors_t *ngbd, p4est_locidx_t *neighbors, bool *neighbor_exists);

void compute_phi_eff(Vec phi_eff, p4est_nodes_t* nodes, std::vector<Vec>& phi, std::vector<mls_opn_t>& opn);

void compute_phi_eff(Vec phi_eff, p4est_nodes_t *nodes, int num_phi, ...);

class zero_cf_t : public CF_DIM
{
public:
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
};

static zero_cf_t zero_cf;

inline double smooth_max(double a, double b, double e)
{
  return .5*(a+b+sqrt(SQR(a-b)+e*e));
}

inline double smooth_min(double a, double b, double e)
{
  return .5*(a+b-(sqrt(SQR(a-b)+e*e)));
}

inline double smooth_min2(double a, double b, double e)
{
  return .5*(a+b-(sqrt(SQR(a-b)+e*e)-e/sqrt(SQR(a-b)+e)));
}


class mls_eff_cf_t : public CF_DIM
{
  std::vector<CF_DIM *>  phi_cf;
  std::vector<mls_opn_t> action;
public:
  mls_eff_cf_t() {}
  mls_eff_cf_t(std::vector<CF_DIM *> &phi_cf, std::vector<mls_opn_t> &action) { set(phi_cf, action); }

  inline void set(std::vector<CF_DIM *> &phi_cf, std::vector<mls_opn_t> &action)
  {
    this->phi_cf = phi_cf;
    this->action = action;
  }

  inline void clear()
  {
    this->phi_cf.clear();
    this->action.clear();
  }

  inline void add_domain(CF_DIM &phi_cf, mls_opn_t opn)
  {
    this->phi_cf.push_back(&phi_cf);
    this->action.push_back(opn);
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    double phi_eff = -10;
    double phi_cur = -10;
    for (int i=0; i<phi_cf.size(); ++i)
    {
      phi_cur = (*phi_cf[i])( DIM(x,y,z) );
      switch (action[i]) {
        case MLS_INTERSECTION: if (phi_cur > phi_eff) phi_eff = phi_cur; break;
        case MLS_ADDITION:     if (phi_cur < phi_eff) phi_eff = phi_cur; break;
      }
    }

    return phi_eff;
  }
};

class mls_smooth_cf_t : public CF_DIM
{
  std::vector<CF_DIM *>  phi_cf;
  std::vector<mls_opn_t> action;
  double epsilon;
public:
  mls_smooth_cf_t() {}
  mls_smooth_cf_t(std::vector<CF_DIM *> &phi_cf, std::vector<mls_opn_t> &action, double epsilon=0) { set(phi_cf, action, epsilon); }

  inline void set(std::vector<CF_DIM *> &phi_cf, std::vector<mls_opn_t> &action, double epsilon=0)
  {
    this->phi_cf  = phi_cf;
    this->action  = action;
    this->epsilon = epsilon;
  }

  inline void set_smoothing(double epsilon)
  {
    this->epsilon = epsilon;
  }

  inline void add_domain(CF_DIM &phi_cf, mls_opn_t opn)
  {
    this->phi_cf.push_back(&phi_cf);
    this->action.push_back(opn);
  }

  double operator()( DIM(double x, double y, double z) ) const
  {
    double phi_eff = -10;
    double phi_cur = -10;
    for (unsigned short i = 0; i < phi_cf.size(); ++i)
    {
      phi_cur = (*phi_cf[i])( DIM(x,y,z) );
      switch (action[i]) {
        case MLS_INTERSECTION: phi_eff = 0.5*(phi_eff+phi_cur+sqrt(SQR(phi_eff-phi_cur)+epsilon)); break;
        case MLS_ADDITION:     phi_eff = 0.5*(phi_eff+phi_cur-sqrt(SQR(phi_eff-phi_cur)+epsilon)+(epsilon)/sqrt(SQR(phi_eff-phi_cur)+epsilon)); break;
      }
    }
    return phi_eff;
  }
};

//class mls_t
//{
//public:
//  unsigned int size;

//  std::vector<mls_opn_t> opn;
//  std::vector<Vec *>     phi;
//  std::vector<int>       clr;

//  std::vector<mls_opn_t>& get_opn() { return opn; }
//  std::vector<Vec *>&     get_phi() { return phi; }
//  std::vector<int>&       get_clr() { return clr; }

//  mls_t() : size(0) {}

//  inline void add_level_set(Vec &phi, mls_opn_t opn, int clr)
//  {
//    this->phi.push_back(&phi);
//    this->opn.push_back(opn);
//    this->clr.push_back(clr);
//    ++size;
//  }

//  inline void add_level_set(Vec &phi, mls_opn_t opn)
//  {
//    this->phi.push_back(&phi);
//    this->opn.push_back(opn);
//    this->clr.push_back(size);
//    ++size;
//  }

//  inline void add_level_set(std::vector<Vec *> phi, std::vector<mls_opn_t> opn, std::vector<int> clr)
//  {
//    if (phi.size() != opn.size() || phi.size() != clr.size())
//      throw std::invalid_argument("Number of elements in arrays phi, acn and clr does not coincide\n");

//    for (unsigned int i = 0; i < phi.size(); ++i) {
//      this->phi.push_back(phi[i]);
//      this->opn.push_back(opn[i]);
//      this->clr.push_back(clr[i]);
//      ++size;
//    }
//  }

//  inline void add_level_set(std::vector<Vec *> phi, std::vector<mls_opn_t> opn)
//  {
//    if (phi.size() != opn.size() || phi.size() != clr.size())
//      throw std::invalid_argument("Number of elements in arrays phi, acn and clr does not coincide\n");

//    for (unsigned int i = 0; i < phi.size(); ++i) {
//      this->phi.push_back(phi[i]);
//      this->opn.push_back(opn[i]);
//      this->clr.push_back(size);
//      ++size;
//    }
//  }
//};

inline void reconstruct_cube(cube2_mls_t &cube, std::vector<double> &phi, std::vector<mls_opn_t> &opn, std::vector<int> &clr)
{
  std::vector<action_t> acn;

  for (unsigned int i = 0; i < opn.size(); ++i)
  {
    switch (opn[i]) {
      case MLS_INTERSECTION: acn.push_back(CUBE_MLS_INTERSECTION); break;
      case MLS_ADDITION:     acn.push_back(CUBE_MLS_ADDITION); break;
      case MLS_COLORATION:   acn.push_back(CUBE_MLS_COLORATION); break;
      default: throw;
    }
  }
  cube.reconstruct(phi, acn, clr);
}

inline void reconstruct_cube(cube3_mls_t &cube, std::vector<double> &phi, std::vector<mls_opn_t> &opn, std::vector<int> &clr)
{
  std::vector<action_t> acn;

  for (unsigned int i = 0; i < opn.size(); ++i)
  {
    switch (opn[i]) {
      case MLS_INTERSECTION: acn.push_back(CUBE_MLS_INTERSECTION); break;
      case MLS_ADDITION:     acn.push_back(CUBE_MLS_ADDITION); break;
      case MLS_COLORATION:   acn.push_back(CUBE_MLS_COLORATION); break;
      default: throw;
    }
  }
  cube.reconstruct(phi, acn, clr);
}

//void find_interface_points(p4est_locidx_t n, const my_p4est_node_neighbors_t *ngbd,
//                           std::vector<mls_opn_t> opn,
//                           std::vector<double *> phi_ptr, DIM( std::vector<double *> phi_xx_ptr,
//                                                               std::vector<double *> phi_yy_ptr,
//                                                               std::vector<double *> phi_zz_ptr ),
//                           int phi_idx[], double dist[]);

void find_closest_interface_location(int &phi_idx, double &dist, double d, std::vector<mls_opn_t> opn,
                                     std::vector<double> &phi_a,
                                     std::vector<double> &phi_b,
                                     std::vector<double> &phi_a_xx,
                                     std::vector<double> &phi_b_xx);

struct interface_point_t
{
  double xyz[P4EST_DIM];

  interface_point_t() { set(DIM(0,0,0)); }
  interface_point_t(double xyz[]) { set(xyz); }
  interface_point_t(DIM(double x, double y, double z)){ set(DIM(x,y,z)); }

  inline double x() { return xyz[0]; }
  inline double y() { return xyz[1]; }
  inline double z() { return xyz[2]; }
  inline void get_xyz(double xyz[])
  {
    XCODE( xyz[0] = this->xyz[0] );
    YCODE( xyz[1] = this->xyz[1] );
    ZCODE( xyz[2] = this->xyz[2] );
  }
  inline void set(double xyz[]) { set(DIM(xyz[0], xyz[1], xyz[2])); }
  inline void set(DIM(double x, double y, double z))
  {
    XCODE( this->xyz[0] = x );
    YCODE( this->xyz[1] = y );
    ZCODE( this->xyz[2] = z );
  }
};

struct interface_point_cartesian_t
{
  p4est_locidx_t n;
  short          dir;
  double         dist;
  double         xyz[P4EST_DIM];
  interface_point_cartesian_t (p4est_locidx_t n=-1, int dir=0, double dist=0, double *xyz=NULL)
    : n(n), dir(dir), dist(dist)
  {
    if (xyz != NULL)
    {
      XCODE( this->xyz[0] = xyz[0] );
      YCODE( this->xyz[1] = xyz[1] );
      ZCODE( this->xyz[2] = xyz[2] );
    }
  }

  inline void get_xyz(double *xyz)
  {
    XCODE( xyz[0] = this->xyz[0] );
    YCODE( xyz[1] = this->xyz[1] );
    ZCODE( xyz[2] = this->xyz[2] );
  }

  // linear interpolation of a Vec at an interface point (assumes locally uniform grid!)
  double interpolate(const my_p4est_node_neighbors_t *ngbd, double *ptr);

  // quadratic interpolation of a Vec at an interface point (assumes locally uniform grid!)
  double interpolate(const my_p4est_node_neighbors_t *ngbd, double *ptr, double *ptr_dd[P4EST_DIM]);
};

struct interface_info_t
{
  int    id;
  double area;
  double centroid[P4EST_DIM];
};

struct my_p4est_finite_volume_t
{
  double full_cell_volume;
  double volume;

  std::vector<interface_info_t> interfaces;

  _CODE( double full_face_area [P4EST_FACES]; )
  _CODE( double face_area      [P4EST_FACES]; )
  XCODE( double face_centroid_x[P4EST_FACES]; )
  YCODE( double face_centroid_y[P4EST_FACES]; )
  ZCODE( double face_centroid_z[P4EST_FACES]; )

  my_p4est_finite_volume_t() { interfaces.reserve(1); }
};

void construct_finite_volume(my_p4est_finite_volume_t& fv, p4est_locidx_t n, p4est_t *p4est, p4est_nodes_t *nodes, std::vector<CF_DIM *> phi, std::vector<mls_opn_t> opn, int order=1, int cube_refinement=0, bool compute_centroids=false, double perturb=1.0e-12);

void compute_wall_normal(const int &dir, double normal[]);

struct points_around_node_map_t
{
  int count;
  std::vector<int> size;   // N*sizeof(int)
  std::vector<int> offset; // N*sizeof(int)

  points_around_node_map_t(int num_nodes=0)
    : size(num_nodes,0), offset(num_nodes+1, 0), count(0) {}

  inline void reinitialize(int num_nodes) { size.assign(num_nodes, 0); offset.assign(num_nodes+1, 0); count = 0; }

  inline void add_point(p4est_locidx_t n) { ++size[n]; count++; }
  inline int  get_idx(p4est_locidx_t n, int i) { return offset[n] + i; }
  inline int  get_num_points_node(p4est_locidx_t n) { return size[n]; }
  inline int  get_num_points_total() { return count; }
  inline void compute_offsets()
  {
    offset[0] = 0;
    for (int i=1; i<size.size(); ++i)
      offset[i] = offset[i-1] + size[i-1];
  }
};

struct cartesian_intersections_map_t
{
  int idx[P4EST_FACES];
};


// advanced boundary conditions structure with pointwise application features
struct boundary_conditions_t
{
  // N = num_owned_indeps
  // M = number of boundary nodes (nodes at which we impose boundary conditions)
  // K = number of cartesian intersections

  BoundaryConditionType  type;
  bool                   pointwise;
  CF_DIM                *value_cf; // either solution value for Dirichlet or flux for Robin/Neumann
  CF_DIM                *coeff_cf; // coefficient in Robin b.c.
  std::vector<double>   *value_pw; // values for imposing Dirichlet (size K) or Neumann bc (size M)
  std::vector<double>   *value_pw_robin; // for imposing Robin bc (size M) (_in addition_ to value_pw)
  std::vector<double>   *coeff_pw_robin; // for imposing Robin bc (size M)

  std::vector<int>       node_map; // N -> dirichlet_local_map or areas, neumann_pts and robin_pts

  std::vector< std::vector<int> >            dirichlet_local_map; // M -> dirichlet_points and dirichlet_weights
  std::vector<double>                        dirichlet_weights;   // K
  std::vector<interface_point_cartesian_t>   dirichlet_pts;       // K

  std::vector<double>            areas;       // M
  std::vector<interface_point_t> neumann_pts; // M
  std::vector<interface_point_t> robin_pts;   // M

  boundary_conditions_t()
    : type(NOINTERFACE), pointwise(false),
      value_cf(NULL), coeff_cf(NULL),
      value_pw(NULL), value_pw_robin(NULL), coeff_pw_robin(NULL) {}

  inline void set(BoundaryConditionType type, CF_DIM &value_cf, CF_DIM &coeff_cf)
  {
    this->type      = type;
    this->pointwise = false;
    this->value_cf  = &value_cf;
    this->coeff_cf  = &coeff_cf;
  }

  inline void set(BoundaryConditionType type, CF_DIM &value_cf)
  {
    if (type == ROBIN) throw;
    this->type      = type;
    this->pointwise = false;
    this->value_cf  = &value_cf;
    this->coeff_cf  = NULL;
  }

  inline void set(BoundaryConditionType type, std::vector<double> &value_pw, std::vector<double> &value_pw_robin, std::vector<double> &coeff_pw_robin)
  {
    this->type      = type;
    this->pointwise = true;
    this->value_pw  = &value_pw;
    this->value_pw_robin = &value_pw_robin;
    this->coeff_pw_robin = &coeff_pw_robin;
  }

  inline void set(BoundaryConditionType type, std::vector<double> &value_pw)
  {
    if (type == ROBIN) throw;
    this->type      = type;
    this->pointwise = true;
    this->value_pw  = &value_pw;
    this->value_pw_robin = NULL;
    this->coeff_pw_robin = NULL;
  }

  inline void add_fv_pt(p4est_locidx_t n, double &area, interface_point_t &neumann, interface_point_t &robin)
  {
#ifdef CASL_THROWS
    if (type == DIRICHLET) throw;
    if (node_map[n] != -1) throw;
#endif

    node_map[n] = areas.size();

    areas      .push_back(area);
    robin_pts  .push_back(robin);
    neumann_pts.push_back(neumann);
  }

  inline void add_fd_pt(p4est_locidx_t n, int dir, double dist, double *xyz, double weight)
  {
    if (node_map[n] == -1)
    {
      node_map[n] = dirichlet_local_map.size();
      dirichlet_local_map.push_back(std::vector<int>());
    }
#ifdef CASL_THROWS
    else
    {
      if (node_map[n] > dirichlet_local_map.size()-1) throw;
    }

    if (type == ROBIN || type == NEUMANN) throw;
#endif

    dirichlet_local_map[node_map[n]].push_back(dirichlet_weights.size());

    dirichlet_weights.push_back(weight);
    dirichlet_pts    .push_back(interface_point_cartesian_t(n, dir, dist, xyz));
  }

  inline bool is_boundary_node(p4est_locidx_t n) { return node_map[n] != -1; }

  inline int num_value_pts()
  {
    switch (type)
    {
      case DIRICHLET: return dirichlet_weights.size();
      case NEUMANN:   return areas.size();
      case ROBIN:     return areas.size();
      default: throw;
    }
  }

  inline int num_robin_pts()
  {
    switch (type)
    {
      case DIRICHLET: return 0;
      case NEUMANN:   return 0;
      case ROBIN:     return areas.size();
      default: throw;
    }
  }

  inline void xyz_value_pt(int idx, double xyz[])
  {
    switch (type)
    {
      case DIRICHLET: dirichlet_pts[idx].get_xyz(xyz); break;
      case NEUMANN:   neumann_pts  [idx].get_xyz(xyz); break;
      case ROBIN:     neumann_pts  [idx].get_xyz(xyz); break;
      default: throw;
    }
  }

  inline void xyz_robin_pt(int idx, double xyz[])
  {
    switch (type)
    {
      case DIRICHLET: throw;
      case NEUMANN:   throw;
      case ROBIN:     robin_pts[idx].get_xyz(xyz); break;
      default: throw;
    }
  }

  inline int num_value_pts(p4est_locidx_t n)
  {
    if (node_map[n] == -1) return 0;
    else
    {
      switch (type)
      {
        case DIRICHLET: return dirichlet_local_map[node_map[n]].size();
        case NEUMANN:   return 1;
        case ROBIN:     return 1;
        default: throw;
      }
    }
  }

  inline int num_robin_pts(p4est_locidx_t n)
  {
    if (node_map[n] == -1) return 0;
    else
    {
      switch (type)
      {
        case DIRICHLET: return 0;
        case NEUMANN:   return 0;
        case ROBIN:     return 1;
        default: throw;
      }
    }
  }

  inline int idx_value_pt(p4est_locidx_t n, int k)
  {
    if (node_map[n] == -1) throw;

    switch (type)
    {
      case DIRICHLET: return dirichlet_local_map[node_map[n]][k];
      case NEUMANN:   return node_map[n];
      case ROBIN:     return node_map[n];
      default: throw;
    }
  }

  inline int idx_robin_pt(p4est_locidx_t n, int k)
  {
    if (node_map[n] == -1) throw;

    switch (type)
    {
      case DIRICHLET: throw;
      case NEUMANN:   throw;
      case ROBIN:     return node_map[n];
      default: throw;
    }
  }

  inline void reset(int num_nodes)
  {
    node_map.assign(num_nodes, -1);

    areas      .clear();
    neumann_pts.clear();
    robin_pts  .clear();

    dirichlet_local_map.clear();
    dirichlet_weights  .clear();
    dirichlet_pts      .clear();

    pointwise      = NULL;
    value_pw       = NULL;
    value_pw_robin = NULL;
    coeff_pw_robin = NULL;
  }

  inline double get_value_pw(p4est_locidx_t n, int i)
  {
    if (node_map[n] == -1) throw;

    switch (type)
    {
      case DIRICHLET: return (*value_pw)[dirichlet_local_map[node_map[n]][i]];
      case NEUMANN:   return (*value_pw)[node_map[n]];
      case ROBIN:     return (*value_pw)[node_map[n]];
    }

  }

  inline double get_robin_pw_value(p4est_locidx_t n)
  {
    if (node_map[n] == -1) throw;

    switch (type)
    {
      case DIRICHLET: throw;
      case NEUMANN:   throw;
      case ROBIN:     return (*value_pw_robin)[node_map[n]];
    }

  }

  inline double get_robin_pw_coeff(p4est_locidx_t n)
  {
    if (node_map[n] == -1) throw;

    switch (type)
    {
      case DIRICHLET: throw;
      case NEUMANN:   throw;
      case ROBIN:     return (*coeff_pw_robin)[node_map[n]];
    }
  }

  inline double get_value_cf(double xyz[]) { return value_cf->value(xyz); }
  inline double get_coeff_cf(double xyz[]) { return coeff_cf->value(xyz); }
};

// interface conditions
struct interface_conditions_t
{
  // N = num_owned_indeps
  // M = number of boundary nodes (nodes at which we impose boundary conditions)
  bool                   pointwise;
  CF_DIM                *sol_jump_cf;
  CF_DIM                *flx_jump_cf;
  std::vector<double>   *sol_jump_pw_taylor; // M, values used in Taylor expansion (usually at projection points)
  std::vector<double>   *flx_jump_pw_taylor; // M, values used in Taylor expansion (usually at projection points)
  std::vector<double>   *flx_jump_pw_integr; // M, values used for integration of "surface generation" term (usually at centroids)

  std::vector<int>               node_map;   // N
  std::vector<double>            areas;      // M
  std::vector<interface_point_t> integr_pts; // M, points for integration (usually centroids)
  std::vector<interface_point_t> taylor_pts; // M, points for Taylor expansion (usually projections)

  // total number of sampling points
  inline int num_integr_pts() { return integr_pts.size(); }
  inline int num_taylor_pts() { return taylor_pts.size(); }

  // coordinates of a given sampling point
  inline void xyz_integr_pt(int idx, double xyz[]) { return integr_pts[idx].get_xyz(xyz); }
  inline void xyz_taylor_pt(int idx, double xyz[]) { return taylor_pts[idx].get_xyz(xyz); }

  // number of sampling points for a given grid node
  inline int num_integr_pts(p4est_locidx_t n) { return node_map[n] == -1 ? 0 : 1; }
  inline int num_taylor_pts(p4est_locidx_t n) { return node_map[n] == -1 ? 0 : 1; }

  // index of local point in common list
  inline int idx_integr_pt(p4est_locidx_t n, int j) { if (j != 0) throw; return node_map[n]; }
  inline int idx_taylor_pt(p4est_locidx_t n, int j) { if (j != 0) throw; return node_map[n]; }

  interface_conditions_t()
    : pointwise(false), sol_jump_cf(NULL), flx_jump_cf(NULL),
      sol_jump_pw_taylor(NULL), flx_jump_pw_taylor(NULL), flx_jump_pw_integr(NULL) {}

  inline void set(CF_DIM &sol_jump_cf, CF_DIM &flx_jump_cf)
  {
    this->pointwise   = false;
    this->sol_jump_cf = &sol_jump_cf;
    this->flx_jump_cf = &flx_jump_cf;
  }

  inline bool is_interface_node(p4est_locidx_t n) { return node_map[n] != -1; }

  inline void add_pt(p4est_locidx_t n, double &area, interface_point_t &taylor_pt, interface_point_t &integr_pt)
  {
#ifdef CASL_THROWS
    if (is_interface_node(n)) throw;
#endif
    node_map[n] = areas.size();

    areas     .push_back(area);
    taylor_pts.push_back(taylor_pt);
    integr_pts.push_back(integr_pt);
  }

  inline void get_pt(p4est_locidx_t n, double &area, interface_point_t &taylor_pt, interface_point_t &integr_pt)
  {
#ifdef CASL_THROWS
    if (!is_interface_node(n)) throw;
#endif
    area      = areas     [node_map[n]];
    taylor_pt = taylor_pts[node_map[n]];
    integr_pt = integr_pts[node_map[n]];
  }

  inline void set(std::vector<double> &sol_jump_pw_taylor, std::vector<double> &flx_jump_pw_taylor, std::vector<double> &flx_jump_pw_integr)
  {
    this->pointwise          = true;
    this->sol_jump_pw_taylor = &sol_jump_pw_taylor;
    this->flx_jump_pw_taylor = &flx_jump_pw_taylor;
    this->flx_jump_pw_integr = &flx_jump_pw_integr;
  }

  inline void reset(int num_nodes)
  {
    node_map.assign(num_nodes, -1);

    areas     .clear();
    integr_pts.clear();
    taylor_pts.clear();

    pointwise          = false;
    sol_jump_pw_taylor = NULL;
    flx_jump_pw_taylor = NULL;
    flx_jump_pw_integr = NULL;
  }

  inline double get_sol_jump_pw_taylor(p4est_locidx_t n) { if (node_map[n] == -1) throw; return (*sol_jump_pw_taylor)[node_map[n]]; }
  inline double get_flx_jump_pw_taylor(p4est_locidx_t n) { if (node_map[n] == -1) throw; return (*flx_jump_pw_taylor)[node_map[n]]; }
  inline double get_flx_jump_pw_integr(p4est_locidx_t n) { if (node_map[n] == -1) throw; return (*flx_jump_pw_integr)[node_map[n]]; }

  inline double get_sol_jump_cf(double xyz[]) { return sol_jump_cf->value(xyz); }
  inline double get_flx_jump_cf(double xyz[]) { return flx_jump_cf->value(xyz); }
};

double smoothstep(int N, double x);

void variable_step_BDF_implicit(const int order, std::vector<double> &dt, std::vector<double> &coeffs);
#endif // UTILS_H
