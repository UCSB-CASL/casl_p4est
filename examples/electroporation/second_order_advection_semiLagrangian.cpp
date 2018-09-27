#define P4_TO_P8
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p4est_to_p8est.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/point3.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/Parser.h>
#include <src/math.h>


using namespace std;
int test_number = 3;  //0 - rotation, 1 - vortex, 2- flow of an inviscid fluid past a sphere, U=5 m/s
/* test_number=3: is the rotation of a cosine bell around the sphere, 
 * after one full rotation the difference to initial distribution 
 * is the amount of error!
 * see reference: http://gfs.sourceforge.net/tests/tests/cosine.html
 * Williamson 1992: https://www.sciencedirect.com/science/article/pii/S0021999105800166?via%3Dihub
*/
double xmin = 0;
double xmax = 1;
double ymin = 0;
double ymax = 1;
double zmin = 0;
double zmax = 1;
double xyz_min_ [3];
double xyz_max_ [3];
int lmin = 6;
int lmax = 7;
int nx = 2;
int ny = 2;
int nz = 2;
double tf = 2*PI; // or test=0: 1, test=1: 2*PI
int save_vtk = 1;
int save_every_n = 50;
double cfl = 1;
int nb_splits = 1;


double R1 = .25*MIN(xmax-xmin, ymax-ymin, zmax-zmin);
double R2 = 0.5*MIN(xmax-xmin, ymax-ymin, zmax-zmin);
struct level_set_t : public CF_3
{
    level_set_t() {lip = 1.2; }
    double operator()(double x, double y, double z) const
    {
	double xc = (xmax - xmin)/2.0;
	double yc = (ymax - ymin)/2.0;
	double zc = (zmax - zmin)/2.0;
        switch(test_number)
        {
        case -1: return 1;
        case 0: return sqrt(SQR(x-xc) + SQR(y-yc) + SQR(z-zc)) - R1;
        case 1: return sqrt(SQR(x-xc) + SQR(y-yc) + SQR(z-zc)) - R1;
	case 2: return sqrt(SQR(x-xc) + SQR(y-yc) + SQR(z-zc)) - R1;
	case 3: return MIN(sqrt(SQR(x-xc) + SQR(y-yc) + SQR(z-zc)) - R1, R2 - sqrt(SQR(x-xc) + SQR(y-yc) + SQR(z-zc)));
	}
     }
}level_set;

struct BCWALLTYPE : WallBC3D
{
    BoundaryConditionType operator()(double x, double y, double z) const
    {
        return NEUMANN;
    }
} bc_wall_type_p;

struct BCWALLVALUE : CF_3
{
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} bc_wall_value_p;

struct u_t : CF_3
{
  double operator()(double x, double y, double z) const {
    double xc = (xmax - xmin)/2.0;
    double yc = (ymax - ymin)/2.0;
    double zc = (zmax - zmin)/2.0;
    double xn = x-xc;
    double yn = y-yc;
    double zn = z-zc;
    switch(test_number)
    {
    case 0: return -yn;
    case 1: return -SQR(sin(PI*xn))*sin(2*PI*yn);
    case 2: 
    {
        double r = sqrt(xn*xn + yn*yn + zn*zn);
        double theta = atan2(sqrt(xn*xn+yn*yn),zn);
	double PHI = atan2(yn,xn);

	double u_r = -5*cos(theta)*(1-1.5*R1/r+R1*R1*R1/2/r/r/r);
	double u_th = 5*sin(theta)*(1-0.75*R1/r-0.25*R1*R1*R1/r/r/r);
	double u_ph = 0;

	double ux = u_r*sin(theta)*cos(PHI) + r*u_th*cos(theta)*cos(PHI)-r*u_ph*sin(theta)*sin(PHI);
 	double uy = u_r*sin(theta)*sin(PHI)+r*u_th*cos(theta)*sin(PHI)+r*u_ph*sin(theta)*cos(PHI);
	double uz = u_r*cos(theta) - r*u_th*sin(theta);
    	return ux;
    }
    case 3: return -yn;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} u;

struct v_t : CF_3
{
  double operator()(double x, double y, double z) const {
    double xc = (xmax - xmin)/2.0;
    double yc = (ymax - ymin)/2.0;
    double zc = (zmax - zmin)/2.0;
    double xn = x-xc;
    double yn = y-yc;
    double zn = z-zc;
    switch(test_number)
    {
    case 0: return xn;
    case 1: return SQR(sin(PI*yn))*sin(2*PI*xn);
    case 2: 
    {
        double r = sqrt(xn*xn + yn*yn + zn*zn);
        double theta = atan2(sqrt(xn*xn+yn*yn),zn);
	double PHI = atan2(yn,xn);

	double u_r = -5*cos(theta)*(1-1.5*R1/r+R1*R1*R1/2/r/r/r);
	double u_th = 5*sin(theta)*(1-0.75*R1/r-0.25*R1*R1*R1/r/r/r);
	double u_ph = 0;

	double ux = u_r*sin(theta)*cos(PHI) + r*u_th*cos(theta)*cos(PHI)-r*u_ph*sin(theta)*sin(PHI);
 	double uy = u_r*sin(theta)*sin(PHI)+r*u_th*cos(theta)*sin(PHI)+r*u_ph*sin(theta)*cos(PHI);
	double uz = u_r*cos(theta) - r*u_th*sin(theta);
    	return uy;
    }
    case 3: return xn;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} v;

struct w_t : CF_3
{
    double operator()(double x, double y, double z) const {
    double xc = (xmax - xmin)/2.0;
    double yc = (ymax - ymin)/2.0;
    double zc = (zmax - zmin)/2.0;
    double xn = x-xc;
    double yn = y-yc;
    double zn = z-zc;
    switch(test_number)
    {
    case 0: return 0;
    case 1: return 0;
    case 2: 
    {
        double r = sqrt(xn*xn + yn*yn + zn*zn);
        double theta = atan2(sqrt(xn*xn+yn*yn),zn);
	double PHI = atan2(yn,xn);

	double u_r = -5*cos(theta)*(1-1.5*R1/r+R1*R1*R1/2/r/r/r);
	double u_th = 5*sin(theta)*(1-0.75*R1/r-0.25*R1*R1*R1/r/r/r);
	double u_ph = 0;

	double ux = u_r*sin(theta)*cos(PHI) + r*u_th*cos(theta)*cos(PHI)-r*u_ph*sin(theta)*sin(PHI);
 	double uy = u_r*sin(theta)*sin(PHI)+r*u_th*cos(theta)*sin(PHI)+r*u_ph*sin(theta)*cos(PHI);
	double uz = u_r*cos(theta) - r*u_th*sin(theta);
    	return uz;
    }
    case 3: return 0;
    default: throw std::invalid_argument("[ERROR]: choose a valid test.");
    }
  }
} w;

class Initial_M : public CF_3
{
public:
    double operator()(double x, double y, double z) const
    {
	if(test_number==0 || test_number==1 || test_number==2)
	{
        	if(level_set(x,y,z)>0)
		{
            		return (z - zmin)/(zmax-zmin) + 1;
		}
        	else
           		 return 0;
	} else if (test_number==3)
    	{
		double r = level_set(x,y,z);
		if(level_set(x,y,z)>0) return 0.5*(1 + cos(PI*r/R1));
		else return 0;
	}
    }
}initial_M;

struct INTERFACE : CF_3
{
    double operator()(double x, double y, double z) const
    {
        return 0;
    }
} bc_interface_value_p;

void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec M, Vec velo_n[3], int compt)
{
    PetscErrorCode ierr;
    char *out_dir = NULL;
    out_dir = getenv("OUT_DIR");
    if(out_dir==NULL)
    {
        ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save visuals\n"); CHKERRXX(ierr);
        return;
    }

    std::ostringstream oss;

    oss << out_dir << "/jump_"
        << p4est->mpisize << "_"
        << brick->nxyztrees[0] << "x"
        << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
           "x" << brick->nxyztrees[2] <<
       #endif
           "." << compt;

    double *phi_p, *M_p, *velo_p[3];
    ierr = VecGetArray(M, &M_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
    for(int dir=0;dir<3;++dir)
	VecGetArray(velo_n[dir], &velo_p[dir]);
    /* save the size of the leaves */
    Vec leaf_level;
    ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
    double *l_p;
    ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for( size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
            const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
            l_p[tree->quadrants_offset+q] = quad->level;
        }
    }

    for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
    {
        const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
        l_p[p4est->local_num_quadrants+q] = quad->level;
    }

    my_p4est_vtk_write_all(p4est, nodes, ghost,
                           P4EST_TRUE, P4EST_TRUE,
                           5, 1, oss.str().c_str(),
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "M", M_p,
			   VTK_POINT_DATA, "vx", velo_p[0],
			   VTK_POINT_DATA, "vy", velo_p[1],
			   VTK_POINT_DATA, "vz", velo_p[2],
            		   VTK_CELL_DATA , "leaf_level", l_p);

    ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
    ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

    ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(M, &M_p); CHKERRXX(ierr);
    for(int dir=0;dir<3;++dir)
	VecRestoreArray(velo_n[dir], &velo_p[dir]);

    PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}


void advect_field_semi_lagrangian(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd_n, double dt_nm1, double dt_n,  Vec vnm1[P4EST_DIM], Vec **vxx_nm1, Vec v[P4EST_DIM], Vec **vxx, Vec M_n, Vec *M_xx_n, double *M_np1_p, bool periodic_adv)
{
    PetscErrorCode ierr;
    my_p4est_interpolation_nodes_t interp_nm1(ngbd_n);
    my_p4est_interpolation_nodes_t interp(ngbd_n);

    std::vector<double> v_tmp_nm1[P4EST_DIM];
    std::vector<double> v_tmp[P4EST_DIM];


    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz);
        interp.add_point(n, xyz);
        // phi_interp.add_point(n,xyz);
    }

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        v_tmp[dir].resize(nodes->indep_nodes.elem_count);
#ifdef P4_TO_P8
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2], quadratic);
#else
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], quadratic);
#endif
        interp.interpolate(v_tmp[dir].data());
    }
    interp.clear();
    // now find v_star
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        // Find initial xy points
        double xyz_star[] =
        {
            node_x_fr_n(n, p4est, nodes) - 0.5*dt_n*v_tmp[0][n],
            node_y_fr_n(n, p4est, nodes) - 0.5*dt_n*v_tmp[1][n]
    #ifdef P4_TO_P8
            , node_z_fr_n(n, p4est, nodes) - 0.5*dt_n*v_tmp[2][n]
    #endif
        };

        for(int dir=0; dir<P4EST_DIM; ++dir)
        {
            if        (periodic_adv && xyz_star[dir]<xyz_min_[dir]) xyz_star[dir] += xyz_max_[dir]-xyz_min_[dir];
            else if (periodic_adv && xyz_star[dir]>xyz_max_[dir]) xyz_star[dir] -= xyz_max_[dir]-xyz_min_[dir];
            else     xyz_star[dir] = MAX(xyz_min_[dir], MIN(xyz_max_[dir], xyz_star[dir]));
        }
        interp.add_point(n, xyz_star);
        interp_nm1.add_point(n, xyz_star);
    }

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        v_tmp_nm1[dir].resize(nodes->indep_nodes.elem_count);

#ifdef P4_TO_P8
        interp_nm1.set_input(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], vxx_nm1[dir][2], quadratic);
#else
        interp_nm1.set_input(vnm1[dir], vxx_nm1[dir][0], vxx_nm1[dir][1], quadratic);
#endif
        interp_nm1.interpolate(v_tmp_nm1[dir].data());


#ifdef P4_TO_P8
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2], quadratic);
#else
        interp.set_input(v[dir], vxx[dir][0], vxx[dir][1], quadratic);
#endif
        interp.interpolate(v_tmp[dir].data());
    }
    interp_nm1.clear();
    interp.clear();
    // finally, find the backtracing value
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
        double vx_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp[0][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[0][n];
        double vy_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp[1][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[1][n];
#ifdef P4_TO_P8
        double vz_star = (1 + 0.5*dt_n/dt_nm1)*v_tmp[2][n] - 0.5*dt_n/dt_nm1 * v_tmp_nm1[2][n];
#endif

        double xyz_d[] =
        {
            node_x_fr_n(n, p4est, nodes) - dt_n*vx_star,
            node_y_fr_n(n, p4est, nodes) - dt_n*vy_star
    #ifdef P4_TO_P8
            ,
            node_z_fr_n(n, p4est, nodes) - dt_n*vz_star
    #endif
        };

        for(int dir=0; dir<P4EST_DIM; ++dir)
        {
            if      (periodic_adv && xyz_d[dir]<xyz_min_[dir]) xyz_d[dir] += xyz_max_[dir]-xyz_min_[dir];
            else if (periodic_adv && xyz_d[dir]>xyz_max_[dir]) xyz_d[dir] -= xyz_max_[dir]-xyz_min_[dir];
            else                                               xyz_d[dir] = MAX(xyz_min_[dir], MIN(xyz_max_[dir], xyz_d[dir]));
        }

        interp.add_point(n, xyz_d);
    }

#ifdef P4_TO_P8
    interp.set_input(M_n, M_xx_n[0], M_xx_n[1], M_xx_n[2], quadratic_non_oscillatory);
#else
    interp.set_input(M_n, M_xx_n[0], M_xx_n[1], quadratic_non_oscillatory);
#endif
    interp.interpolate(M_np1_p);
}

void advect(p4est_t *p4est, p4est_nodes_t *nodes, my_p4est_node_neighbors_t *ngbd_n, Vec v_nm1[3], Vec v[3], double dt_nm1, double dt_n, Vec &M, bool periodic_adv)
{
    PetscErrorCode ierr;
    Vec *M_xx;
    // compute vx_xx, vx_yy
    Vec *vxx[P4EST_DIM];
    Vec *vnm1_xx[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        vxx[dir] = new Vec[P4EST_DIM];
        vnm1_xx[dir] = new Vec[P4EST_DIM];
        if(dir==0)
        {
            for(int dd=0; dd<P4EST_DIM; ++dd)
            {
                ierr = VecCreateGhostNodes(p4est, nodes, &vxx[dir][dd]); CHKERRXX(ierr);
                ierr = VecCreateGhostNodes(p4est, nodes, &vnm1_xx[dir][dd]); CHKERRXX(ierr);
            }
        }
        else
        {
            for(int dd=0; dd<P4EST_DIM; ++dd)
            {
                ierr = VecDuplicate(vxx[0][dd], &vxx[dir][dd]); CHKERRXX(ierr);
                ierr = VecDuplicate(vnm1_xx[0][dd], &vnm1_xx[dir][dd]); CHKERRXX(ierr);
            }
        }
#ifdef P4_TO_P8
        ngbd_n->second_derivatives_central(v[dir], vxx[dir][0], vxx[dir][1], vxx[dir][2]);
#else
        ngbd_n->second_derivatives_central(v[dir], vxx[dir][0], vxx[dir][1]);
#endif
#ifdef P4_TO_P8
        ngbd_n->second_derivatives_central(v_nm1[dir], vnm1_xx[dir][0], vnm1_xx[dir][1], vnm1_xx[dir][2]);
#else
        ngbd_n->second_derivatives_central(v_nm1[dir], vnm1_xx[dir][0], vnm1_xx[dir][1]);
#endif
    }
    // compute M_xx and M_yy
    M_xx = new Vec[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecDuplicate(vxx[0][dir], &M_xx[dir]); CHKERRXX(ierr);
    }
#ifdef P4_TO_P8
    ngbd_n->second_derivatives_central(M, M_xx[0], M_xx[1], M_xx[2]);
#else
    ngbd_n->second_derivatives_central(M, M_xx[0], M_xx[1]);
#endif

    double *M_np1_p;
    Vec M_np1;
    ierr = VecCreateGhostNodes(p4est, nodes, &M_np1); CHKERRXX(ierr);
    VecGetArray(M_np1, &M_np1_p);
    advect_field_semi_lagrangian(p4est, nodes, ngbd_n, dt_nm1, dt_n, v_nm1, vnm1_xx, v, vxx, M, M_xx, M_np1_p, periodic_adv);
    VecRestoreArray(M_np1, &M_np1_p);
    ierr = VecDestroy(M); CHKERRXX(ierr);
    M = M_np1;
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        for(int dd=0; dd<P4EST_DIM; ++dd)
        {
            ierr = VecDestroy(vxx[dir][dd]); CHKERRXX(ierr);
            ierr = VecDestroy(vnm1_xx[dir][dd]); CHKERRXX(ierr);
        }
        delete[] vxx[dir];
        delete[] vnm1_xx[dir];
    }
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
        ierr = VecDestroy(M_xx[dir]); CHKERRXX(ierr);
    }
    delete[] M_xx;
}

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;

  mpi_environment_t mpi;
  mpi.init(argc, argv);

  splitting_criteria_cf_t data(lmin, lmax, &level_set, 1.2);


  const int n_xyz [] = {nx, ny, nz};
  const double xyz_min [] = {xmin, ymin, zmin};
  const double xyz_max [] = {xmax, ymax, zmax};
  xyz_min_ [0] = xmin; xyz_min_ [1] = ymin; xyz_min_ [2] = zmin;
  xyz_max_ [0] = xmax; xyz_max_ [1] = ymax; xyz_max_ [2] = zmax; 
  const int periodic [] = {0, 0, 0};
  

  // Create the connectivity object
  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  double dxyz_min[P4EST_DIM];
  dxyz_min[0] = (xmax-xmin)/nx/(1<<lmax);
  dxyz_min[1] = (ymax-ymin)/ny/(1<<lmax);
  dxyz_min[2] = (zmax-zmin)/nz/(1<<lmax);
  double dt = cfl*MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
  double dt_nm1 = dt;
  PetscPrintf(mpi.comm(), "test= %d, time-step= %g, dx=%g, cfl=%g\n", test_number, dt, dxyz_min[0], cfl);


  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);
  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  /* Initialize the level-set function */
  double tn = 0;
  Vec phi_n;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_n); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est, nodes, level_set, phi_n);
  my_p4est_level_set_t ls(ngbd);
  ls.reinitialize_2nd_order(phi_n);
  ls.perturb_level_set_function(phi_n, EPS);

  /* initialize the velocity field */
  const CF_3 *velo_cf[3] = { &u, &v, &w };
  Vec velo_nm1[3], velo_n[3];
  Vec M;
  VecDuplicate(phi_n, &M);
  sample_cf_on_nodes(p4est, nodes, initial_M, M);
  for(int dir=0; dir<3; ++dir)
  {
    ierr = VecDuplicate(phi_n, &velo_n[dir]); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *velo_cf[dir], velo_n[dir]);
    ierr = VecDuplicate(phi_n, &velo_nm1[dir]); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, *velo_cf[dir], velo_nm1[dir]);
  }


  int iter = 0;

  if(save_vtk==1)
    save_VTK(p4est, ghost, nodes, &brick, phi_n, M, velo_n, iter/save_every_n);

  BoundaryConditions3D bc_interface;
  bc_interface.setInterfaceType(NEUMANN);
  bc_interface.setInterfaceValue(bc_interface_value_p);

  double *phi_p;
  ierr = VecGetArray(phi_n, &phi_p); CHKERRXX(ierr);
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
      phi_p[i] = -phi_p[i];
  for(int dir=0;dir<3; ++dir)
  {
      ls.extend_Over_Interface(phi_n, velo_nm1[dir], bc_interface, 2, 20);
      ls.extend_Over_Interface(phi_n, velo_n[dir], bc_interface, 2, 20);
  }
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
      phi_p[i] = -phi_p[i];

  double *velo_p[3], *velo_nm1_p[3];
  for(int dir=0;dir<3; ++dir)
  {
	VecGetArray(velo_n[dir], &velo_p[dir]);
	VecGetArray(velo_nm1[dir], &velo_nm1_p[dir]);
  	for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++)
  	{
      		if(phi_p[i]<0)
		{      
			velo_p[dir][i] = 0;
			velo_nm1_p[dir][i] = 0;
		}
  	}
	VecRestoreArray(velo_n[dir], &velo_p[dir]);
	VecRestoreArray(velo_nm1[dir], &velo_nm1_p[dir]);
  }

  ierr = VecRestoreArray(phi_n, &phi_p); CHKERRXX(ierr);

  Vec domain;
  ierr = VecDuplicate(phi_n, &domain); CHKERRXX(ierr);
  Vec lc;
  ierr = VecGhostGetLocalForm(domain, &lc); CHKERRXX(ierr);
  ierr = VecSet(lc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(domain, &lc); CHKERRXX(ierr);
  
  while(tn<tf)
  {
    PetscPrintf(mpi.comm(), "Iteration=%d, time=%g (s) out of %g (s)\n", iter, tn, tf);
    
    advect(p4est, nodes, ngbd, velo_nm1, velo_n, dt_nm1, dt, M, true);

    double total_mass = integrate_over_negative_domain(p4est, nodes, domain, M);
    PetscPrintf(mpi.comm(), "TOTAL mass is %g\n", total_mass);
    if(save_vtk && iter % save_every_n == 0)
      save_VTK(p4est, ghost, nodes, &brick, phi_n, M, velo_n, iter/save_every_n);

    char *out_dir = NULL;
    out_dir = getenv("OUT_DIR");
    if(out_dir==NULL)
    {
       ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR before running the code to save stats\n"); CHKERRXX(ierr);
    }
    else
    {
       char out_path_Z[1000];
       sprintf(out_path_Z, "%s/transport.dat", out_dir);
       if(p4est->mpirank==0)
       {
       	  if(iter ==0){
             FILE *f = fopen(out_path_Z, "w");
             fprintf(f, "Simulation with test number %d, (lmin, lmax)=(%d,%d), (nx,ny,nz)=(%d, %d, %d), dt=%g, dx=%g, cfl=%g \n", test_number, lmin, lmax, nx, ny, nz, dt, dt/cfl, cfl);
             fprintf(f, "time [s]    | total mass [mol^3]   \n");
             fprintf(f, "%g \t %g \n", tn, total_mass);
             fclose(f);
          }
          else{
             FILE *f = fopen(out_path_Z, "a");
             fprintf(f, "%g \t %g\n", tn, total_mass);
             fclose(f);
          }
       }
    }
    tn += dt;
    iter++;
  }
  ierr = PetscPrintf(mpi.comm(), "Final time: tf=%g\n", tn); CHKERRXX(ierr);

  /* compute the error */
  if(test_number==3)
  {
  	double err = 0;
  	double *M_p;
  	VecGetArray(M, &M_p);
  	for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  	{
    		double x = node_x_fr_n(n, p4est, nodes);
    		double y = node_y_fr_n(n, p4est, nodes);
    		double z = node_z_fr_n(n, p4est, nodes);
    		err = max(err, (M_p[n]-initial_M(x,y,z)));
  	}
  	VecRestoreArray(M, &M_p);
  
  	int mpiret;
  	mpiret = MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
  	ierr = PetscPrintf(mpi.comm(), "Maximum error (L_inf) in density is: %g\n", err); CHKERRXX(ierr);
  }


  ierr = VecDestroy(phi_n);   CHKERRXX(ierr);
  ierr = VecDestroy(M);   CHKERRXX(ierr);
  ierr = VecDestroy(domain);   CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(velo_n  [dir]); CHKERRXX(ierr);
  }

   /* destroy the p4est and its connectivity structure */
  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy (nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}


