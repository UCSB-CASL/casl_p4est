// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_level_set.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_level_set.h>
#endif

#undef MIN
#undef MAX

#define const_mue

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

#ifdef P4_TO_P8
static struct:CF_3{
  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
  double operator()(double x, double y, double z) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
  }
  double  x0, y0, z0, r;
} circle ;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
  }
} u_ex;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  -2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
  }
} u_ex_x;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  -2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z);
  }
} u_ex_y;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  -2*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z);
  }
} u_ex_z;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
#ifdef const_mue
    (void) x;
    (void) y;
    (void) z;

    return 1;
#else
    return  1 + x*x + y*y + z*z;
#endif
  }
} mue_ex;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
#ifdef const_mue
    return  (12*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z));
#else
    return  (12*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z)*(1+x*x+y*y+z*z) +
              4*M_PI*x*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z) +
              4*M_PI*y*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z) +
              4*M_PI*z*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z) );
#endif
  }
} f_ex;

static struct:WallBC3D{
  BoundaryConditionType operator()(double x, double y, double z) const {
    (void)x;
    (void)y;
    (void)z;
    return NEUMANN;
  }
} bc_wall_neumann_type;

static struct:WallBC3D{
  BoundaryConditionType operator()(double x, double y, double z) const {
    (void)x;
    (void)y;
    (void)z;
    return DIRICHLET;
  }
} bc_wall_dirichlet_type;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    (void) x;
    (void) y;
    (void) z;
    return 0;
  }
} bc_wall_neumann_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return u_ex(x,y,z);
  }
} bc_wall_dirichlet_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    return u_ex(x,y,z);
  }
} bc_interface_dirichlet_value;

static struct:CF_3{
  double operator()(double x, double y, double z) const {
    double r  = sqrt(SQR(x-circle.x0) + SQR(y-circle.y0) + SQR(z-circle.z0));
    double nx = (x-circle.x0) / r;
    double ny = (y-circle.y0) / r;
    double nz = (z-circle.z0) / r;
    double norm = sqrt( nx*nx + ny*ny + nz*nz);
    nx /= norm; ny /= norm; nz /= norm;
    return mue_ex(x,y,z)*( 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z) * nx +
                           2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z) * ny +
                           2*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z) * nz );
  }
} bc_interface_neumann_value;
#else
static struct:CF_2{
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const {
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
  }
  double  x0, y0, r;
} circle;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  cos(2*M_PI*x)*cos(2*M_PI*y);
  }
} u_ex;


static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  -2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y);
  }
} u_ex_x;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  -2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y);
  }
} u_ex_y;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
#ifdef const_mue
    (void) x;
    (void) y;
    return  1 ; // + x*x + y*y;
#else
    return  1 + x*x + y*y;
#endif
  }
} mue_ex;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
#ifdef const_mue
    return  ( 8*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y) );
#else
    return  ( 8*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*(1+x*x+y*y) +
             4*M_PI*x*sin(2*M_PI*x)*cos(2*M_PI*y) +
             4*M_PI*y*cos(2*M_PI*x)*sin(2*M_PI*y) );
#endif

  }
} f_ex;


static struct:WallBC2D{
  BoundaryConditionType operator()(double x, double y) const {
    (void)x;
    (void)y;
    return NEUMANN;
  }
} bc_wall_neumann_type;

static struct:WallBC2D{
  BoundaryConditionType operator()(double x, double y) const {
    (void)x;
    (void)y;
    return DIRICHLET;
  }
} bc_wall_dirichlet_type;

static struct:CF_2{
  double operator()(double x, double y) const {
    (void) x;
    (void) y;
    return 0;
  }
} bc_wall_neumann_value;

static struct:CF_2{
  double operator()(double x, double y) const {
    return u_ex(x,y);
  }
} bc_wall_dirichlet_value;

static struct:CF_2{
  double operator()(double x, double y) const {
    return u_ex(x,y);
  }
} bc_interface_dirichlet_value;

static struct:CF_2{
  double operator()(double x, double y) const {
    double r = sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
    double nx = (x-circle.x0) / r;
    double ny = (y-circle.y0) / r;
    double norm = sqrt( nx*nx + ny*ny);
    nx /= norm; ny /= norm;
    return mue_ex(x,y)*(2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y) * nx +
                        2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y) * ny );
  }
} bc_interface_neumann_value;
#endif

int main (int argc, char* argv[]){

  mpi_enviroment_t mpi;
  mpi.init(argc, argv);
  try{
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    cmdParser cmd;
    cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
    cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
    cmd.add_option("lmin", "the min level of the tree");
    cmd.add_option("lmax", "the max level of the tree");
    cmd.add_option("sp", "number of splits to apply to the min and max level");
    cmd.parse(argc, argv);

    // decide on the type and value of the boundary conditions
    BoundaryConditionType bc_wall_type, bc_interface_type;
    int nb_splits, min_level, max_level;
    bc_wall_type      = cmd.get("bc_wtype"  , DIRICHLET);
    bc_interface_type = cmd.get("bc_itype"  , DIRICHLET);
    nb_splits         = cmd.get("sp", 0);
    min_level         = cmd.get("lmin"      , 3);
    max_level         = cmd.get("lmax"      , 8);

#ifdef P4_TO_P8
    CF_3 *bc_wall_value, *bc_interface_value;
    WallBC3D *wall_bc;
#else
    CF_2 *bc_wall_value, *bc_interface_value;
    WallBC2D *wall_bc;
#endif

    switch(bc_interface_type){
    case DIRICHLET:
      bc_interface_value = &bc_interface_dirichlet_value;
      break;
    case NEUMANN:
      bc_interface_value = &bc_interface_neumann_value;
      break;
    default:
      throw std::invalid_argument("[ERROR]: Interface bc type can only be 'Dirichlet' or 'Neumann' type");
    }

    switch(bc_wall_type){
    case DIRICHLET:
      bc_wall_value = &bc_wall_dirichlet_value;
      wall_bc       = &bc_wall_dirichlet_type;
      break;
    case NEUMANN:
      bc_wall_value = &bc_wall_neumann_value;
      wall_bc       = &bc_wall_neumann_type;
      break;
    default:
      throw std::invalid_argument("[ERROR]: Wall bc type can only be 'Dirichlet' or 'Neumann' type");
    }

#ifdef P4_TO_P8
    circle.update(1, 1, 1, .3);
#else
    circle.update(1, 1, .3);
#endif
    splitting_criteria_cf_t data(min_level+nb_splits, max_level+nb_splits, &circle, 1);

    parStopWatch w1, w2;
    w1.start("total time");
    w2.start("initializing the grid");

    /* create the macro mesh */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2,
                                      0, 2, 0, 2, 0, 2, &brick);
#else
    connectivity = my_p4est_brick_new(2, 2, 0, 2, 0, 2, &brick);
#endif

    /* create the p4est */
    p4est = p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
    p4est->user_pointer = (void*)(&data);
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    /* partition the p4est */
    my_p4est_partition(p4est, P4EST_TRUE, NULL);

    /* create the ghost layer */
    p4est_ghost_t* ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    /* initialize the vectors */
    Vec phi, rhs, uex, sol, mue;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &uex); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &mue); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, circle, phi);
    sample_cf_on_nodes(p4est, nodes, u_ex, uex);
    sample_cf_on_nodes(p4est, nodes, f_ex, rhs);
    sample_cf_on_nodes(p4est, nodes, mue_ex, mue);

    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);

    w2.stop(); w2.read_duration();

    /* initalize the bc information */
    Vec interface_value_Vec, wall_value_Vec;
    ierr = VecDuplicate(phi, &interface_value_Vec); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &wall_value_Vec); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, *bc_interface_value, interface_value_Vec);
    sample_cf_on_nodes(p4est, nodes, *bc_wall_value, wall_value_Vec);

    my_p4est_interpolation_nodes_t interface_interp(&ngbd), wall_interp(&ngbd);
    interface_interp.set_input(interface_value_Vec, linear);
    wall_interp.set_input(wall_value_Vec, linear);

    bc_interface_value = &interface_interp;
    bc_wall_value = &wall_interp;

#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif
    bc.setInterfaceType(bc_interface_type);
    bc.setInterfaceValue(*bc_interface_value);
    bc.setWallTypes(*wall_bc);
    bc.setWallValues(*bc_wall_value);

    /* initialize the poisson solver */
    w2.start("solve the poisson equation");
    my_p4est_poisson_nodes_t solver(&ngbd);
    solver.set_phi(phi);    
    solver.set_rhs(rhs);
    solver.set_mu(mue);
    solver.set_bc(bc);

    /* solve the system */
    solver.solve(sol);    

    w2.stop(); w2.read_duration();

    /* prepare for output */
    double *sol_p, *phi_p, *uex_p;
    ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecGetArray(uex, &uex_p); CHKERRXX(ierr);


    /* compute the error */
#ifdef P4_TO_P8
    double err_max[4] = {0, 0, 0, 0};
#else
    double err_max[3] = {0, 0, 0};
#endif
    std::vector<double> err(nodes->indep_nodes.elem_count);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
      if(phi_p[n]<0)
      {
        err[n] = fabs(sol_p[n] - uex_p[n]);
        err_max[0] = max( err_max[0], err[n] );
      }
      else
        err[n] = 0;
    }

    Vec uex_x, uex_y;
    ierr = VecDuplicate(phi, &uex_x); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &uex_y); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, u_ex_x, uex_x);
    sample_cf_on_nodes(p4est, nodes, u_ex_y, uex_y);
    double *uex_x_ptr, *uex_y_ptr;
    ierr = VecGetArray(uex_x, &uex_x_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(uex_y, &uex_y_ptr); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec uex_z;
    ierr = VecDuplicate(phi, &uex_z); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, u_ex_z, uex_z);
    double *uex_z_ptr;
    ierr = VecGetArray(uex_z, &uex_z_ptr); CHKERRXX(ierr);
#endif
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      const quad_neighbor_nodes_of_node_t qnnn = ngbd.get_neighbors(n);
      if(phi_p[n]<0)
      {
        err_max[1] = max( err_max[1], ABS(qnnn.dx_central(sol_p) - uex_x_ptr[n]) );
        err_max[2] = max( err_max[2], ABS(qnnn.dy_central(sol_p) - uex_y_ptr[n]) );
#ifdef P4_TO_P8
        err_max[3] = max( err_max[3], ABS(qnnn.dz_central(sol_p) - uex_z_ptr[n]) );
#endif
      }
    }
    ierr = VecRestoreArray(uex_x, &uex_x_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(uex_y, &uex_y_ptr); CHKERRXX(ierr);
    ierr = VecDestroy(uex_x); CHKERRXX(ierr);
    ierr = VecDestroy(uex_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(uex_z, &uex_z_ptr); CHKERRXX(ierr);
    ierr = VecDestroy(uex_z); CHKERRXX(ierr);
#endif


#ifdef P4_TO_P8
    double glob_err_max[4];
    MPI_Allreduce(&err_max, &glob_err_max, 4, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
#else
    double glob_err_max[3];
    MPI_Allreduce(&err_max, &glob_err_max, 3, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
#endif

    PetscPrintf(p4est->mpicomm, "lvl : %d / %d, L_inf error : %e\n", min_level+nb_splits, max_level+nb_splits, glob_err_max[0]);
#ifdef P4_TO_P8
    PetscPrintf(p4est->mpicomm, "L_inf error f / dx / dy / dz : %e / %e / %e / %e\n", glob_err_max[0], glob_err_max[1], glob_err_max[2], glob_err_max[3]);
#else
    PetscPrintf(p4est->mpicomm, "L_inf error f / dx / dy : %e / %e / %e\n", glob_err_max[0], glob_err_max[1], glob_err_max[2]);
#endif

    /* save the vtk file */
    std::ostringstream oss; oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           4, 0, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "sol", sol_p,
                           VTK_POINT_DATA, "uex", uex_p,
                           VTK_POINT_DATA, "err", &err[0]);

    /* restore internal pointers */
    ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(uex, &uex_p); CHKERRXX(ierr);

    /* destroy allocated vectors */
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    ierr = VecDestroy(uex); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);
    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    ierr = VecDestroy(wall_value_Vec); CHKERRXX(ierr);
    ierr = VecDestroy(interface_value_Vec); CHKERRXX(ierr);

    /* destroy p4est objects */
    p4est_nodes_destroy (nodes);
    p4est_ghost_destroy (ghost);
    p4est_destroy (p4est);
    my_p4est_brick_destroy(connectivity, &brick);

    w1.stop(); w1.read_duration();

  } catch (const std::exception& e) {
    std::cout << "[" << mpi.rank() << " -- ERROR]: " << e.what() << std::endl;
  }

  return 0;
}

