#ifndef UTILS_H
#define UTILS_H

// casl_p4est
#include <src/my_p4est_nodes.h>
#include <src/ArrayV.h>

// p4est
#include <p4est.h>

// PETSc
#include <petsc.h>

// System
#include <stdexcept>
#include <sstream>

// Some Macros
#define EPS 1e-12
#define P4EST_TRUE  1
#define P4EST_FALSE 0

#if (PETSC_VERSION_MINOR <= 1)
#undef CHKERRXX
#define CHKERRXX
#endif


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
double bilinear_interpolation(p4est_t *p4est, p4est_topidx_t tree_id, p4est_quadrant_t *quad, double *F, double x_global, double y_global);

/*!
 * \brief p4est_VecCreate Creates a normal PETSc parallel vector based on p4est node ordering
 * \param p4est the forest
 * \param nodes the nodes numbering data structure
 * \param v PETSc vector type
 */
PetscErrorCode VecGhostCreate_p4est(p4est_t *p4est, my_p4est_nodes_t *nodes, Vec* v);

/*!
 * \brief p4est2petsc_local_numbering converts p4est local node numbering convention to petsc local numbering convention
 * \param nodes the nodes numbering structure
 * \param p4est_node_locidx local numbering in p4est convention
 * \return local numbering in petsc convention
 */
p4est_locidx_t p4est2petsc_local_numbering(my_p4est_nodes_t *nodes, p4est_locidx_t p4est_node_locidx);

/*!
 * \brief The Session class
 */
class Session{
  int argc;
  char** argv;
  PetscErrorCode ierr;

public:
  Session(int argc_, char* argv_[])
    : argc(argc_), argv(argv_)
  {}
  ~Session(){
    sc_finalize ();
    ierr = PetscFinalize(); CHKERRXX(ierr);
  }

  void init(MPI_Comm mpicomm){
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
    sc_init (mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
    p4est_init (NULL, SC_LP_DEFAULT);
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
