#ifndef _MACROS_BY_CHOHONG
#define _MACROS_BY_CHOHONG

#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <stdint.h>
#include <map>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_QUADTREE_LEVEL 25
#if (MAX_QUADTREE_LEVEL<15)
typedef int16_t IndexInt;
#else
typedef int32_t IndexInt;
#endif

//#define WITH_64BIT_INT
#ifdef WITH_64BIT_INT
typedef int64_t CaslInt;
#else
typedef int32_t CaslInt;
#endif

const static CaslInt MAX_NUMBER_OF_NODES_IN_ONE_DIRECTION = (1<<MAX_QUADTREE_LEVEL);


static inline CaslInt my_isnan(double x) { return x != x; }
static inline CaslInt my_isinf(double x) { return !my_isnan(x) && my_isnan(x - x); }
static inline double my_sgn(double x) { return (x>0 ? 1.0:-1.0); }
static inline uint32_t mylog2(const uint32_t x) {
    assert(x != 0);
    uint32_t y;
    asm ( "\tbsr %1, %0\n"
    : "=r"(y)
            : "r" (x)
            );
    return y;
}
typedef enum {
    POINT_DATA,
    CELL_DATA
} vtk_data_format;

typedef enum {
    DIRICHLET,
    NEUMANN,
    ROBIN,
    INTERIOR,
    PERIODIC,
    ONE_SIDED,
    JUMP,
    OUTSIDE,
    NOINTERFACE,
    CELL_DIRICHLET
} BoundaryConditionType;

std::istream& operator >> (std::istream& is, BoundaryConditionType& bc);
std::ostream& operator << (std::ostream& os, const BoundaryConditionType& bc);
BoundaryConditionType string_to_bc(const std::string& str);
std::string bc_to_string(const BoundaryConditionType& bc);


class CF_1
{
public:
    double lip;
    virtual double operator()(double x) const=0 ;
};


class CF_2
{
public:
    double lip;
    virtual double operator()(double x, double y) const=0 ;
};

class CF_3
{
public:
    double lip;
    virtual double operator()(double x, double y,double z) const=0 ;
};

class BoundaryConditions3D  {
protected:
    BoundaryConditionType xpWall;
    BoundaryConditionType xmWall;
    BoundaryConditionType ypWall;
    BoundaryConditionType ymWall;
    BoundaryConditionType zpWall;
    BoundaryConditionType zmWall;
    BoundaryConditionType Interface;
public:

    BoundaryConditions3D()
        : xpWall(ONE_SIDED)
        , xmWall(ONE_SIDED)
        , ypWall(ONE_SIDED)
        , ymWall(ONE_SIDED)
        , zpWall(ONE_SIDED)
        , zmWall(ONE_SIDED)
        , Interface(NOINTERFACE)
    {}

    inline void setWalls(BoundaryConditionType xm,
                         BoundaryConditionType xp,
                         BoundaryConditionType ym,
                         BoundaryConditionType yp,
                         BoundaryConditionType zm,
                         BoundaryConditionType zp){
        xmWall = xm;
        xpWall = xp;
        assert((xmWall == PERIODIC && xpWall == PERIODIC) || (xmWall != PERIODIC && xpWall != PERIODIC));
        ymWall = ym;
        ypWall = yp;
        assert((ymWall == PERIODIC && ypWall == PERIODIC) || (ymWall != PERIODIC && ypWall != PERIODIC));
        zmWall = zm;
        zpWall = zp;
        assert((zmWall == PERIODIC && zpWall == PERIODIC) || (zmWall != PERIODIC && zpWall != PERIODIC));
    }
    inline void setInterface(BoundaryConditionType bc){
        Interface = bc;
    }

    inline void operator=(const BoundaryConditions3D &bc){
        xpWall = bc.xpWall;
        xmWall = bc.xmWall;
        ypWall = bc.ypWall;
        ymWall = bc.ymWall;
        zpWall = bc.zpWall;
        zmWall = bc.zmWall;
        Interface = bc.Interface;
    }
    inline BoundaryConditionType get_xmWall()   { return xmWall;}
    inline BoundaryConditionType get_xpWall()   { return xpWall;}
    inline BoundaryConditionType get_ymWall()   { return ymWall;}
    inline BoundaryConditionType get_ypWall()   { return ypWall;}
    inline BoundaryConditionType get_zmWall()   { return zmWall;}
    inline BoundaryConditionType get_zpWall()   { return zpWall;}
    inline BoundaryConditionType get_Interface(){ return Interface;}
} ;

class BoundaryConditions2D  {

    BoundaryConditionType xpWallType_;
    BoundaryConditionType xmWallType_;
    BoundaryConditionType ypWallType_;
    BoundaryConditionType ymWallType_;
    BoundaryConditionType InterfaceType_;

    const CF_2 *p_xpWallValue;
    const CF_2 *p_xmWallValue;
    const CF_2 *p_ypWallValue;
    const CF_2 *p_ymWallValue;
    const CF_2 *p_InterfaceValue;
    const CF_2 *p_robin_alpha;
    const CF_2 *p_robin_beta;


public:
    BoundaryConditions2D()
        : xpWallType_(DIRICHLET)
        , xmWallType_(DIRICHLET)
        , ypWallType_(DIRICHLET)
        , ymWallType_(DIRICHLET)
        , InterfaceType_(NOINTERFACE)
        , p_xpWallValue(NULL)
        , p_xmWallValue(NULL)
        , p_ypWallValue(NULL)
        , p_ymWallValue(NULL)
        , p_InterfaceValue(NULL)
        , p_robin_alpha(NULL)
        , p_robin_beta(NULL)
    { }


    void setWallTypes(BoundaryConditionType xm,
                             BoundaryConditionType xp,
                             BoundaryConditionType ym,
                             BoundaryConditionType yp)
    {
        xmWallType_ = xm;
        xpWallType_ = xp;
        assert((xmWallType_ == PERIODIC && xpWallType_ == PERIODIC) || (xmWallType_ != PERIODIC && xpWallType_ != PERIODIC));
        ymWallType_ = ym;
        ypWallType_ = yp;
        assert((ymWallType_ == PERIODIC && ypWallType_ == PERIODIC) || (ymWallType_ != PERIODIC && ypWallType_ != PERIODIC));
    }

    void setWallValues(const CF_2& xm, const CF_2& xp, const CF_2& ym, const CF_2& yp){
        p_xmWallValue = &xm;
        p_xpWallValue = &xp;
        p_ymWallValue = &ym;
        p_ypWallValue = &yp;
    }

    void setRobin(const CF_2& alpha, const CF_2& beta){
        p_robin_alpha = &alpha;
        p_robin_beta  = &beta;
    }

    void setInterfaceType(BoundaryConditionType bc){
        InterfaceType_ = bc;
    }

    void setInterfaceValue(const CF_2& in){
        p_InterfaceValue = &in;
    }

    BoundaryConditionType xmWallType() const{ return xmWallType_;}
    BoundaryConditionType xpWallType() const{ return xpWallType_;}
    BoundaryConditionType ymWallType() const{ return ymWallType_;}
    BoundaryConditionType ypWallType() const{ return ypWallType_;}
    BoundaryConditionType InterfaceType() const{ return InterfaceType_;}

    double xmWallValue(double x, double y) const { return (*p_xmWallValue)(x,y);}
    double xpWallValue(double x, double y) const { return (*p_xpWallValue)(x,y);}
    double ymWallValue(double x, double y) const { return (*p_ymWallValue)(x,y);}
    double ypWallValue(double x, double y)  const { return (*p_ypWallValue)(x,y);}
    double InterfaceValue(double x, double y) const { return (*p_InterfaceValue)(x,y);}
    double robinAlpha(double x, double y) const {return (*p_robin_alpha)(x,y);}
    double robinBeta(double x, double y) const {return (*p_robin_beta)(x,y);}
} ;

enum Choice_of_Accuracy
{
    FIRST_ORDER_ACCURACY,
    SECOND_ORDER_ACCURACY,
    THIRD_ORDER_ACCURACY
};

enum FLUID_BOUNDARY_CONDITION
{
    FLUID_BC_SOLID  , // on the boundary, Dirichlet on all the components of the velocity field
    FLUID_BC_INFLOW , // on the boundary, Dirichlet on the normal component of the velocity field
    FLUID_BC_OUTFLOW, // on the boundary, Neumann   on the normal component of the velocity field
    FLUID_BC_INSIDE   // inside the domain
};



#ifndef EPS
#define EPS 0.00000001
#endif

#ifndef MAX_IT
#define MAX_IT 10000
#endif

#ifndef ABS
#define ABS(a) ((a)>0 ? (a) : -(a))
#endif

#ifndef POSSQR
#define POSSQR(a) ((a)>0 ? (a)*(a) : 0 )
#endif

#ifndef NEGSQR

#define NEGSQR(a) ((a)<0 ? (a)*(a) : 0 )
#endif

#ifndef BIG
#define BIG 100000000
#endif

#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif

#ifndef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#endif

#ifndef SQR
#define SQR(a) (a)*(a)
#endif

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef GAUSS
#define GAUSS(a) ((a)>=0 ? CaslInt(a) : CaslInt(a-1))
#endif

#ifndef POS
#define POS(a) ((a)>0 ? (a) : 0)
#endif

#ifndef NEG
#define NEG(a) ((a)<0 ? (a) : 0)
#endif

#ifndef _DEBUG_
#define _DEBUG_ cout<<"This is cout form: "<<__FILE__<<" line # "<<__LINE__<<"\n press any key to continue ..."<<endl;getchar()
#endif

#ifndef DROP_TOL
#define DROP_TOL 1E-10
#endif

#ifndef BELOW_DROP_TOL
#define BELOW_DROP_TOL(a,b) (ABS((a))/ABS((b)) < DROP_TOL)
#endif

#define MAX_CHAR_LENGTH 4096


//---------------------------------------------------------------------
// Stopwatch
//---------------------------------------------------------------------

#if !defined (_OPENMP)
class StopWatch
{
private:
    clock_t startTime; // time the stop watch was started
    clock_t  stopTime; // stop the stop watch
    CaslInt total_ticks;

public:
    StopWatch() {total_ticks=0;}
    void stop (){ stopTime =clock(); } // stop the stopwatch
    void start(){ startTime=clock(); } // start the stopwatch
    void start(const char *name_of_work){ printf("%s ... \n",name_of_work);startTime=clock(); }
    void read_duration( const char* name_Of_work)
    {
        double duration = (double)(stopTime - startTime) / CLOCKS_PER_SEC;
        printf( "%s in %2.3f seconds \n", name_Of_work,duration );
    }
    void read_duration()
    {
        double duration = (double)(stopTime - startTime) / CLOCKS_PER_SEC;
        printf( " ... done in %.2f seconds \n", duration );
    }
    void read_ticks( const char* name_Of_work)
    {
        CaslInt number_of_ticks = (CaslInt)(stopTime - startTime);
        printf( "%d ticks in %s\n", number_of_ticks, name_Of_work );
    }
    void add_into_total_ticks()
    {
        total_ticks += (CaslInt)(stopTime - startTime);
    }
    void read_total_ticks( const char* name_Of_work)
    {
        printf( "%d ticks in %s\n", total_ticks, name_Of_work );
    }
};
#endif

//---------------------------------------------------------------------
// math library
//---------------------------------------------------------------------
inline double SQRT( double sqr )
{
    double x = 1;
    const double err_tolerance = 1E-19;
    double err = 1.;

    while( err>err_tolerance )
    {
        err = x;
        x = 0.5*(x+sqr/x);
        err = x-err; if(err<0) err=-err;
    }

    return x;
}


inline double DELTA( double x, double h )
{
    if( x > h ) return 0;
    if( x <-h ) return 0;
    else	    return (1+cos(PI*x/h))*0.5/h;
}

inline double HVY( double x, double h )
{
    if( x > h ) return 1;
    if( x <-h ) return 0;
    else	    return (1+x/h+sin(PI*x/h)/PI)*0.5;
}

inline double HVY( double x, double x0, double h )
{
    if( x - x0 > h ) return 1;
    if( x - x0 <-h ) return 0;
    else	    return (1+(x-x0)/h+sin(PI*(x-x0)/h)/PI)*0.5;
}

inline double SGN( double x, double h )
{
    if( x > h ) return  1;
    if( x <-h ) return -1;
    else	    return x/h+sin(PI*x/h)/PI;
}

inline double MINMOD( double a, double b )
{
    if(a*b<=0) return 0;
    else
    {
        if((ABS(a))<(ABS(b))) return a;
        else                  return b;
    }
}

inline double HARMOD( double a, double b )
{
    if(a*b<=0) return 0;
    else
    {
        if(a<0) a=-a;
        if(b<0) b=-b;

        return 2*a*b/(a+b);

        //double f =(a-b)/(a+b); if(f<0) f=-f;
        //return .5*(a+b)*(1-f*f*f);
    }
}

inline double ENO2( double a, double b )
{
    if(a*b<=0) return 0;
    else
    {
        if((ABS(a))<(ABS(b))) return a;
        else                  return b;
    }
}

inline double SUPERBEE( double a, double b )
{
    if(a*b<=0) return 0;
    else
    {
        double theta = b/a;
        if(theta<0.5) return 2*b;
        if(theta<1.0) return a;
        if(theta<2.0) return b;
        else          return 2*a;
    }
}

double interface_location( double   a, double   b,
                           double  fa, double  fb );

double interface_location_between_b_and_c( double   a, double   b, double  c, double  d,
                                           double  fa, double  fb, double fc, double fd );

double interface_location_between_b_and_c_minmod( double   a, double   b, double  c, double  d,
                                                  double  fa, double  fb, double fc, double fd );

double interface_location_with_first_order_derivative(	double   a, double   b,
                                                        double  fa, double  fb,
                                                        double fxa, double fxb );

//---------------------------------------------------------------------
// find the interface location on interval [a,b]
//---------------------------------------------------------------------
double interface_location_with_second_order_derivative(double    a, double    b,
                                                       double   fa, double   fb,
                                                       double fxxa, double fxxb );
double Mean_Curvature(  double Fx,
                        double Fy,
                        double Fz, double Fxx,
                        double Fyy,
                        double Fzz, double Fxy,
                        double Fyz,
                        double Fxz);

double Gaussian_Curvature(  double Fx,
                            double Fy,
                            double Fz, double Fxx,
                            double Fyy,
                            double Fzz, double Fxy,
                            double Fyz,
                            double Fxz);

void Pricipal_Curvatures(  double     mean_curvature,
                           double Gaussian_curvature, double& K1,
                           double& K2);
#endif
