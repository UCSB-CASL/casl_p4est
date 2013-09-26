#ifndef UTILS_H
#define UTILS_H

// casl_p4est
#include <p4est_nodes.h>

// p4est
#include <p4est.h>

// PETSc
#include <petsc.h>

// System
#include <stdexcept>
#include <sstream>

class CF_1
{
public:
  double lip;
  virtual double operator()(double x) const=0 ;
  virtual ~CF_1() {}
};


class CF_2
{
public:
  double lip;
  virtual double operator()(double x, double y) const=0 ;
  virtual ~CF_2() {}
};

class CF_3
{
public:
  double lip;
  virtual double operator()(double x, double y,double z) const=0 ;
  virtual ~CF_3() {}
};

typedef enum {
  DIRICHLET,
  NEUMANN,
  NOINTERFACE,
  MIXED
} BoundaryConditionType;

std::ostream& operator << (std::ostream& os, BoundaryConditionType  type);
std::istream& operator >> (std::istream& is, BoundaryConditionType& type);

class WallBC2D
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
  const WallBC2D* WallType_;
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

// p4est boolean type
typedef int p4est_bool_t;
#define P4EST_TRUE  1
#define P4EST_FALSE 0

// Some Macros
#define EPS 1e-13
#ifndef ABS
#define ABS(a) ((a)>0 ? (a) : -(a))
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

inline double DELTA( double x, double h )
{
  if( x > h ) return 0;
  if( x <-h ) return 0;
  else      return (1+cos(M_PI*x/h))*0.5/h;
}

inline double HVY( double x, double h )
{
  if( x > h ) return 1;
  if( x <-h ) return 0;
  else      return (1+x/h+sin(M_PI*x/h)/M_PI)*0.5;
}

inline double SIGN(double a)
{
  return (a>0) ? 1:-1;
}

inline double MINMOD( double a, double b )
{
  if(a*b<=0) return 0;
  else
  {
    if((fabs(a))<(fabs(b))) return a;
    else                    return b;
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

/*!
 * \brief c2p_coordinate_transform Converts local (within tree [0,1]) coordinates into global coordinates
 * \param p4est the forest
 * \param tree_id the current tree in which the point is located
 * \param x will be ignored if set to NULL
 * \param y will be ignored if set to NULL
 * \param z will be ignored if set to NULL
 */
void c2p_coordinate_transform(p4est_t *p4est, p4est_topidx_t tree_id, double *x, double *y, double *z);

/*!
 * \brief dx_dy_dz_quadrant finds the actual dx_dy_dz of a quadrant
 * \param p4est the forest
 * \param tree_id the current tree in which quadrant is located
 * \param quad the current quadrant
 * \param dx will be ignored if set to NULL
 * \param dy will be ignored if set to NULL
 * \param dz will be ignored if set to NULL
 */
void dx_dy_dz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *dx, double *dy, double *dz);

/*!
 * \brief xyz_quadrant finds the global x_y_z of a quadrant
 * \param p4est the forest
 * \param tree_id the current tree that owns the quadrant
 * \param quad the current quadrant
 * \param x will be ignored if set to NULL
 * \param y will be ignored if set to NULL
 * \param z will be ignored if set to NULL
 */
void xyz_quadrant(p4est_t *p4est, p4est_topidx_t& tree_id, p4est_quadrant_t* quad, double *x, double *y, double *z);

/*!
 * \brief bilinear_interpolation performs bilinear interpolation for a point
 * \param p4est the forest
 * \param tree_id the current tree that owns the quadrant
 * \param quad the current quarant
 * \param F a simple C-style array of size 4, containing the values of the function at the vertices of the quadrant. __MUST__ be z-ordered
 * \param x_global global x-coordinate of the point
 * \param y_global global y-coordinate of the point
 * \return interpolated value
 */
double bilinear_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *xy_global);

/*!
 * \brief non_oscilatory_quadratic_interpolation performs non-oscilatory quadratic interpolation for a point
 * \param p4est the forest
 * \param tree_id the current tree that owns the quadrant
 * \param quad the current quarant
 * \param F a simple C-style array of size 4, containing the values of the function at the vertices of the quadrant. __MUST__ be z-ordered
 * \param Fxx a simple C-style array of size 4, containing the values of the xx derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param Fyy a simple C-style array of size 4, containing the values of the yy derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param x_global global x-coordinate ointerface_location_with_second_order_derivativef the point
 * \param y_global global y-coordinate of the point
 * \return interpolated value
 */
double quadratic_non_oscillatory_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fxx, const double *Fyy, const double *xy_global);

/*!
 * \brief quadratic_interpolation performs quadratic interpolation for a point
 * \param p4est the forest
 * \param tree_id the current tree that owns the quadrant
 * \param quad the current quarant
 * \param F a simple C-style array of size 4, containing the values of the function at the vertices of the quadrant. __MUST__ be z-ordered
 * \param Fxx a simple C-style array of size 4, containing the values of the xx derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param Fyy a simple C-style array of size 4, containing the values of the yy derivative of function at the vertices of the quadrant. does not need to be z-ordered
 * \param x_global global x-coordinate of the point
 * \param y_global global y-coordinate of the point
 * \return interpolated value
 */
double quadratic_interpolation(p4est_t *interface_location_with_second_order_derivativep4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, const double *F, const double *Fxx, const double *Fyy, const double *xy_global);

/*!
 * \brief p4est_VecCreate Creates a normal PETSc parallel vector based on p4est node ordering
 * \param p4est the forest
 * \param nodes the nodes numbering data structure
 * \param v PETSc vector type
 */
PetscErrorCode VecCreateGhost(p4est_t *p4est, p4est_nodes_t *nodes, Vec* v);

/*!
 * \brief p4est2petsc_local_numbering converts p4est local node numbering convention to petsc local numbering convention
 * \param nodes the nodes numbering structure
 * \param p4est_node_locidx local numbering in p4est convention
 * \return local numbering in petsc convention
 */
p4est_locidx_t p4est2petsc_local_numbering(p4est_nodes_t *nodes, p4est_locidx_t p4est_node_locidx);

inline double int2double_coordinate_transform(p4est_qcoord_t a){
  return static_cast<double>(a)/static_cast<double>(P4EST_ROOT_LEN);
}

/*!
 * \brief integrate_over_negative_domain_in_one_quadrant
 */
double integrate_over_negative_domain_in_one_quadrant(p4est_t *p4est, p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

/*!
 * \brief integrate_over_negative_domain integrate a quantity f over the negative domain defined by phi
 *        note: second order convergence
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi
 * \param f the scalar to integrate
 * \return the integral of f over the phi<0 domain, \int_{\phi<0} f
 */
double integrate_over_negative_domain(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec f);

/*!
 * \brief area_in_negative_domain_in_one_quadrant
 */
double area_in_negative_domain_in_one_quadrant(p4est_t *p4est, p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

/*!
 * \brief area_in_negative_domain compute the area of the negative domain defined by phi
 *        note: second order convergence
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \return the area in the negative phi domain, i.e. \int_{phi<0} 1
 */
double area_in_negative_domain(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi);

/*!
 * \brief integrate_over_interface_in_one_quadrant
 */
double integrate_over_interface_in_one_quadrant(p4est_t *p4est, p4est_nodes_t *nodes, p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

/*!
 * \brief integrate_over_interface integrate a scalar f over the 0-contour of the level-set function phi.
 *        note: first order convergence only
 * \param p4est the p4est
 * \param nodes the nodes structure associated to p4est
 * \param phi the level-set function
 * \param f the scalar to integrate
 * \return the integral of f over the contour defined by phi, i.e. \int_{phi=0} f
 */
double integrate_over_interface(p4est_t *p4est, p4est_nodes_t *nodes, Vec phi, Vec f);

/*!
 * \brief is_node_xmWall checks if a node is on x^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the left domain boundary and p4est is _NOT_ periodic
 */
bool is_node_xmWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_xpWall checks if a node is on x^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the right domain boundary and p4est is _NOT_ periodic
 */
bool is_node_xpWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_ymWall checks if a node is on y^- domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain bottom boundary and p4est is _NOT_ periodic
 */
bool is_node_ymWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_ymWall checks if a node is on y^+ domain boundary
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain top boundary and p4est is _NOT_ periodic
 */
bool is_node_ypWall(const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief is_node_Wall checks if a node is on any of domain boundaries
 * \param p4est [in] p4est
 * \param ni    [in] pointer to the node structure
 * \return true if the point is on the domain boundary and p4est is _NOT_ periodic
 */
bool is_node_Wall  (const p4est_t *p4est, const p4est_indep_t *ni);

/*!
 * \brief sample_cf_on_nodes samples a cf function on the nodes. both local and ghost poinst are considered
 * \param p4est [in] the p4est object
 * \param nodes [in] the nodes data structure
 * \param cf    [in] the cf function. It is assumed that the function can be evaluated at _ANY_ point, whether local or remote
 * \param f     [in, out] a PETSc Vec object to store the result. It is assumed that the vector is allocated. A check
 * is performed to ensure enough memory is available in the Vec object.
 */
void sample_cf_on_nodes(p4est_t *p4est, p4est_nodes_t *nodes, const CF_2& cf, Vec f);

template<typename T>
T ranged_rand(T a, T b, int seed = 0){
  if (seed) srand(seed);
  return static_cast<T>(static_cast<double>(rand())/static_cast<double>(RAND_MAX) * (b-a) + a);
}

/*!
 * \brief prepares MPI, PETSc, p4est, and sc libraries
 */
class Session{   
public:
  ~Session(){
    PetscErrorCode ierr = PetscFinalize(); CHKERRXX(ierr);
  }

  void init(int argc, char **argv, MPI_Comm mpicomm = MPI_COMM_WORLD){
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
    sc_init (mpicomm, 1, 1, NULL, SC_LP_SILENT);
    p4est_init (NULL, SC_LP_SILENT);
  }
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
  std::string msg_;
  stopwatch_timing timing_;

public:   

  parStopWatch(stopwatch_timing timing = root_timings, MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm), timing_(timing)
  {
    MPI_Comm_rank(comm_, &mpirank);
  }

  void start(const std::string& msg){
    msg_ = msg;
    PetscPrintf(comm_, "%s ... \n", msg.c_str());
    ts = MPI_Wtime();
  }

  void stop(){
    tf = MPI_Wtime();
  }

  double read_duration(){
    double elap = tf - ts;

    PetscPrintf(comm_, "%s ... done in ", msg_.c_str());
    if (timing_ == all_timings){
      PetscSynchronizedPrintf(comm_, "\n   %.4lf secs. on process %2d",elap, mpirank);
      PetscSynchronizedFlush(comm_);
      PetscPrintf(comm_, "\n");
    } else {
      PetscPrintf(comm_, " %.4lf secs. on process %d [Note: only showing root's timings]\n", elap, mpirank);
    }
    return elap;
  }
};

typedef struct
{
  MPI_Comm            mpicomm;
  int                 mpisize;
  int                 mpirank;
}
mpi_context_t;

#endif // UTILS_H
