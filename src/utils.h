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

// p4est boolean type
typedef int p4est_bool_t;
#define P4EST_TRUE  1
#define P4EST_FALSE 0

// Some Macros
#define EPS 1e-12
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

inline double HVY( double x, double x0, double h )
{
  if( x - x0 > h ) return 1;
  if( x - x0 <-h ) return 0;
  else      return (1+(x-x0)/h+sin(M_PI*(x-x0)/h)/M_PI)*0.5;
}

inline double SGN( double x, double h )
{
  if( x > h ) return  1;
  if( x <-h ) return -1;
  else      return x/h+sin(M_PI*x/h)/M_PI;
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
double bilinear_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, const p4est_quadrant_t &quad, double *F, const double *xy_global);

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
