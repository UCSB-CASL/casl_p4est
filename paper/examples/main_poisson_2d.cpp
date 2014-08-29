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
#include <src/my_p8est_poisson_node_base.h>
#include <src/my_p8est_levelset.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/my_p4est_levelset.h>
#endif

#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

//#define NO_INTERFACE
//#define PLAN
#define CIRCLE

double c = -.124;//-.1;

using namespace std;

#ifdef P4_TO_P8
static struct:CF_3{
  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
  double operator()(double x, double y, double z) const {
#ifdef NO_INTERFACE
    return -1;
#endif
#ifdef PLAN
    return -x + 1.212;
#endif
#ifdef CIRCLE
    return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
#endif
  }
  double  x0, y0, z0, r;
} circle ;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  cos(2*M_PI*(x+c))*cos(2*M_PI*(y+c))*cos(2*M_PI*(z+c));
  }
} u_ex;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  -2*M_PI*sin(2*M_PI*(x+c))*cos(2*M_PI*(y+c))*cos(2*M_PI*(z+c));
  }
} u_ex_x;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  -2*M_PI*cos(2*M_PI*(x+c))*sin(2*M_PI*(y+c))*cos(2*M_PI*(z+c));
  }
} u_ex_y;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  -2*M_PI*cos(2*M_PI*(x+c))*cos(2*M_PI*(y+c))*sin(2*M_PI*(z+c));
  }
} u_ex_z;

static class: public CF_3
{
public:
  double operator()(double x, double y, double z) const {
    return  12*M_PI*M_PI*cos(2*M_PI*(x+c))*cos(2*M_PI*(y+c))*cos(2*M_PI*(z+c));
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
    if(ABS(x  )<EPS && ABS(y  )<EPS && ABS(z  )<EPS) return 1./3.*(-u_ex_x(x,y,z) -u_ex_y(x,y,z) -u_ex_z(x,y,z));
    if(ABS(x  )<EPS && ABS(y  )<EPS && ABS(z-2)<EPS) return 1./3.*(-u_ex_x(x,y,z) -u_ex_y(x,y,z) +u_ex_z(x,y,z));
    if(ABS(x  )<EPS && ABS(y-2)<EPS && ABS(z  )<EPS) return 1./3.*(-u_ex_x(x,y,z) +u_ex_y(x,y,z) -u_ex_z(x,y,z));
    if(ABS(x  )<EPS && ABS(y-2)<EPS && ABS(z-2)<EPS) return 1./3.*(-u_ex_x(x,y,z) +u_ex_y(x,y,z) +u_ex_z(x,y,z));

    if(ABS(x-2)<EPS && ABS(y  )<EPS && ABS(z  )<EPS) return 1./3.*( u_ex_x(x,y,z) -u_ex_y(x,y,z) -u_ex_z(x,y,z));
    if(ABS(x-2)<EPS && ABS(y  )<EPS && ABS(z-2)<EPS) return 1./3.*( u_ex_x(x,y,z) -u_ex_y(x,y,z) +u_ex_z(x,y,z));
    if(ABS(x-2)<EPS && ABS(y-2)<EPS && ABS(z  )<EPS) return 1./3.*( u_ex_x(x,y,z) +u_ex_y(x,y,z) -u_ex_z(x,y,z));
    if(ABS(x-2)<EPS && ABS(y-2)<EPS && ABS(z-2)<EPS) return 1./3.*( u_ex_x(x,y,z) +u_ex_y(x,y,z) +u_ex_z(x,y,z));

    if(ABS(x  )<EPS && ABS(y  )<EPS) return .5*(-u_ex_x(x,y,z) -u_ex_y(x,y,z));
    if(ABS(x  )<EPS && ABS(y-2)<EPS) return .5*(-u_ex_x(x,y,z) +u_ex_y(x,y,z));
    if(ABS(x-2)<EPS && ABS(y  )<EPS) return .5*( u_ex_x(x,y,z) -u_ex_y(x,y,z));
    if(ABS(x-2)<EPS && ABS(y-2)<EPS) return .5*( u_ex_x(x,y,z) +u_ex_y(x,y,z));

    if(ABS(x  )<EPS && ABS(z  )<EPS) return .5*(-u_ex_x(x,y,z) -u_ex_z(x,y,z));
    if(ABS(x  )<EPS && ABS(z-2)<EPS) return .5*(-u_ex_x(x,y,z) +u_ex_z(x,y,z));
    if(ABS(x-2)<EPS && ABS(z  )<EPS) return .5*( u_ex_x(x,y,z) -u_ex_z(x,y,z));
    if(ABS(x-2)<EPS && ABS(z-2)<EPS) return .5*( u_ex_x(x,y,z) +u_ex_z(x,y,z));

    if(ABS(y  )<EPS && ABS(z  )<EPS) return .5*(-u_ex_y(x,y,z) -u_ex_z(x,y,z));
    if(ABS(y  )<EPS && ABS(z-2)<EPS) return .5*(-u_ex_y(x,y,z) +u_ex_z(x,y,z));
    if(ABS(y-2)<EPS && ABS(z  )<EPS) return .5*( u_ex_y(x,y,z) -u_ex_z(x,y,z));
    if(ABS(y-2)<EPS && ABS(z-2)<EPS) return .5*( u_ex_y(x,y,z) +u_ex_z(x,y,z));

    if(ABS(x)<EPS)   return -u_ex_x(x,y,z);
    if(ABS(x-2)<EPS) return  u_ex_x(x,y,z);
    if(ABS(y)<EPS)   return -u_ex_y(x,y,z);
    if(ABS(y-2)<EPS) return  u_ex_y(x,y,z);
    if(ABS(z)<EPS)   return -u_ex_z(x,y,z);
//    if(ABS(z-2)<EPS)
      return  u_ex_z(x,y,z);
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
#ifdef PLAN
    return -u_ex_x(x,y,z);
#endif

#ifdef CIRCLE
    double r  = sqrt(SQR(x-circle.x0) + SQR(y-circle.y0) + SQR(z-circle.z0));
    double nx = (x-circle.x0) / r;
    double ny = (y-circle.y0) / r;
    double nz = (z-circle.z0) / r;
    double norm = sqrt( nx*nx + ny*ny + nz*nz);
    nx /= norm; ny /= norm; nz /= norm;
    return ( 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z) * nx +
             2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z) * ny +
             2*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z) * nz );
#endif
  }
} bc_interface_neumann_value;
#else
static struct:CF_2{
  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
  double operator()(double x, double y) const{
#ifdef NO_INTERFACE
    return -1;
#endif
#ifdef PLAN
    return x - 1.212;
#endif
#ifdef CIRCLE
    return r - sqrt(SQR(x-x0) + SQR(y-y0));
#endif
  }
  double  x0, y0, r;
} circle;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  cos(2*M_PI*(x+c))*cos(2*M_PI*(y+c));
  }
} u_ex;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  -2*M_PI*sin(2*M_PI*(x+c))*cos(2*M_PI*(y+c));
  }
} u_ex_x;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  -2*M_PI*cos(2*M_PI*(x+c))*sin(2*M_PI*(y+c));
  }
} u_ex_y;

static class: public CF_2
{
public:
  double operator()(double x, double y) const {
    return  8*M_PI*M_PI*cos(2*M_PI*(x+c))*cos(2*M_PI*(y+c));
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

//static
static struct:CF_2{
  double operator()(double x, double y) const {
//    (void) x;
//    (void) y;
    if(ABS(x  )<EPS && ABS(y  )<EPS) return .5*(-u_ex_x(x,y)-u_ex_y(x,y));
    if(ABS(x  )<EPS && ABS(y-2)<EPS) return .5*(-u_ex_x(x,y)+u_ex_y(x,y));
    if(ABS(x-2)<EPS && ABS(y  )<EPS) return .5*( u_ex_x(x,y)-u_ex_y(x,y));
    if(ABS(x-2)<EPS && ABS(y-2)<EPS) return .5*( u_ex_x(x,y)+u_ex_y(x,y));
    if(ABS(x)<EPS)   return -u_ex_x(x,y);
    if(ABS(x-2)<EPS) return  u_ex_x(x,y);
    if(ABS(y)<EPS)   return -u_ex_y(x,y);
//    if(ABS(y-2)<EPS)
      return  u_ex_y(x,y);
//    return 0;
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
#ifdef PLAN
    return u_ex_x(x,y);
#endif

#ifdef CIRCLE
    double r = sqrt( SQR(x-circle.x0) + SQR(y-circle.y0) );
    double nx = (x-circle.x0) / r;
    double ny = (y-circle.y0) / r;
    double norm = sqrt( nx*nx + ny*ny);
    nx /= norm; ny /= norm;
    return 2*M_PI*sin(2*M_PI*(x+c))*cos(2*M_PI*(y+c)) * nx + 2*M_PI*cos(2*M_PI*(x+c))*sin(2*M_PI*(y+c)) * ny;
#endif
  }
} bc_interface_neumann_value;
#endif

int main (int argc, char* argv[]){

  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;
  try{
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;

    Session mpi_session;
    mpi_session.init(argc, argv, mpi->mpicomm);

    cmdParser cmd;
    cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
    cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
    cmd.add_option("lmin", "the min level of the tree");
    cmd.add_option("lmax", "the max level of the tree");
    cmd.add_option("sp", "number of splits to apply to the min and max level");
    cmd.add_option("save_vtk", "give this flag to save a vtk file of the solution and error");
		cmd.add_option("repeat", "number of repeat to perform");
		cmd.add_option("compute_error", "compute the error on the computed solution");
    cmd.parse(argc, argv);
    cmd.print();

    // decide on the type and value of the boundary conditions
    BoundaryConditionType bc_wall_type, bc_interface_type;
    int nb_splits, min_level, max_level;
    bc_wall_type      = cmd.get("bc_wtype"  , DIRICHLET);
    bc_interface_type = cmd.get("bc_itype"  , DIRICHLET);
    nb_splits         = cmd.get("sp" , 0);
    min_level         = cmd.get("lmin"      , 3);
    max_level         = cmd.get("lmax"      , 8);
		int repeat        = cmd.get("repeat"    , 1);

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

    MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
    MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

    w2.start("initializing the grid");

    /* create the macro mesh */
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
#ifdef P4_TO_P8
    connectivity = my_p4est_brick_new(2, 2, 2, &brick);
#else
    connectivity = my_p4est_brick_new(2, 2, &brick);
#endif

    /* create the p4est */
    p4est = p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);
    p4est->user_pointer = (void*)(&data);
    p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

    /* partition the p4est */
    p4est_partition(p4est, NULL);

    /* create the ghost layer */
    p4est_ghost_t* ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FULL);

    /* generate unique node indices */
    nodes = my_p4est_nodes_new(p4est, ghost);
    w2.stop(); w2.read_duration();

    /* initialize the vectors */
    Vec phi, rhs, uex, sol;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &uex); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &sol); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, circle, phi);
    sample_cf_on_nodes(p4est, nodes, u_ex, uex);
    sample_cf_on_nodes(p4est, nodes, f_ex, rhs);

    /* create the hierarchy structure */
    w2.start("construct the hierachy information");
    my_p4est_hierarchy_t hierarchy(p4est, ghost, &brick);
    w2.stop(); w2.read_duration();

    /* generate the neighborhood information */
    w2.start("construct the neighborhood information");
    my_p4est_node_neighbors_t ngbd(&hierarchy, nodes);
    ngbd.init_neighbors();
    w2.stop(); w2.read_duration();

    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
    double dx = (xmax-xmin) / pow(2.,(double) data.max_lvl);
    double dy = (ymax-ymin) / pow(2.,(double) data.max_lvl);

  #ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
    double dz = (zmax-zmin) / pow(2.,(double) data.max_lvl);
  #endif

    my_p4est_level_set ls(&ngbd);
    ls.perturb_level_set_function(phi, SQR(MIN(dx, dy
                                           #ifdef P4_TO_P8
                                               , dz
                                           #endif
                                               ))*1e-3);

    /* initalize the bc information */
    Vec interface_value_Vec, wall_value_Vec;
    ierr = VecDuplicate(phi, &interface_value_Vec); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &wall_value_Vec); CHKERRXX(ierr);

    sample_cf_on_nodes(p4est, nodes, *bc_interface_value, interface_value_Vec);
    sample_cf_on_nodes(p4est, nodes, *bc_wall_value, wall_value_Vec);

    InterpolatingFunctionNodeBase interface_interp(p4est, nodes, ghost, &brick, &ngbd), wall_interp(p4est, nodes, ghost, &brick, &ngbd);
    interface_interp.set_input_parameters(interface_value_Vec, linear);
    wall_interp.set_input_parameters(wall_value_Vec, linear);

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

		Vec err;
		ierr = VecDuplicate(phi, &err); CHKERRXX(ierr);

    /* initialize the poisson solver */
		for(int i=0; i<repeat; ++i)
		{
			w2.start("solve the poisson equation");
			PoissonSolverNodeBase solver(&ngbd);
			solver.set_phi(phi);
			solver.set_rhs(rhs);
			solver.set_bc(bc);

			/* solve the system */
			MPI_Barrier(p4est->mpicomm);
			solver.solve(sol);

			if(cmd.contains("compute_error"))
			{
				ierr = VecGhostUpdateBegin(sol, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
				ierr = VecGhostUpdateEnd  (sol, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
				w2.stop(); w2.read_duration();

				if(bc_interface_type==NEUMANN && bc_wall_type==NEUMANN)
				{
					PetscPrintf(p4est->mpicomm, "Neumann BC only! Shifting solution\n\n");
					solver.shift_to_exact_solution(sol, uex);
				}

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

				double *err_p;
				ierr = VecGetArray(err, &err_p); CHKERRXX(ierr);
				for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
				{
					if(phi_p[n]<0)
					{
						err_p[n] = fabs(sol_p[n] - uex_p[n]);
						err_max[0] = max( err_max[0], err_p[n] );
					}
					else
						err_p[n] = 0;
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
					if(phi_p[n]<0)
					{
						err_max[1] = max( err_max[1], ABS(ngbd[n].dx_central(sol_p) - uex_x_ptr[n]) );
						err_max[2] = max( err_max[2], ABS(ngbd[n].dy_central(sol_p) - uex_y_ptr[n]) );
#ifdef P4_TO_P8
						err_max[3] = max( err_max[3], ABS(ngbd[n].dz_central(sol_p) - uex_z_ptr[n]) );
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
				PetscPrintf(p4est->mpicomm, "L_inf error dx / dy / dz : %e / %e / %e\n", glob_err_max[1], glob_err_max[2], glob_err_max[3]);
#else
				PetscPrintf(p4est->mpicomm, "L_inf error dx / dy : %e / %e\n", glob_err_max[1], glob_err_max[2]);
#endif

				/* save the vtk file */
				if(cmd.contains("save_vtk"))
				{
					std::ostringstream oss; oss << P4EST_DIM << "d_solution_" << p4est->mpisize;
					my_p4est_vtk_write_all(p4est, nodes, ghost,
							P4EST_TRUE, P4EST_TRUE,
							4, 0, oss.str().c_str(),
							VTK_POINT_DATA, "phi", phi_p,
							VTK_POINT_DATA, "sol", sol_p,
							VTK_POINT_DATA, "uex", uex_p,
							VTK_POINT_DATA, "err", err_p );
					PetscPrintf(mpi->mpicomm, "Results saved in %s\n", oss.str().c_str());
				}

				/* restore internal pointers */
				ierr = VecRestoreArray(err, &err_p); CHKERRXX(ierr);
				ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
				ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
				ierr = VecRestoreArray(uex, &uex_p); CHKERRXX(ierr);
			}
		}

    /* destroy allocated vectors */
		ierr = VecDestroy(err); CHKERRXX(ierr);
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
    std::cout << "[" << mpi->mpirank << " -- ERROR]: " << e.what() << std::endl;
  }

  return 0;
}
