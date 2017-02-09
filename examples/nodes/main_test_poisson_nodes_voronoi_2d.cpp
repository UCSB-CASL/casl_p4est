
/*
 * Test the cell based p4est.
 * 1 - solve a poisson equation on an irregular domain (circle)
 * 2 - interpolate from faces to nodes
 * 3 - extrapolate faces over interface
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/my_p8est_interpolation_nodes.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_poisson_nodes_voronoi.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

double xmin = -1;
double xmax =  1;
double ymin = -1;
double ymax =  1;
#ifdef P4_TO_P8
double zmin = -1;
double zmax =  1;
#endif

using namespace std;

int lmin = 4;
int lmax = 7;
int nb_splits = 5;

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

/*
 * 0 - circle
 * 1 - plane (horizontal)
 * 2 - plane (vertical)
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 - x+y
 * 1 - x*x + y*y
 * 2 - sin(x)*cos(y)
 * 3 - sin(x) + cos(y)
 * 4 - cos(r)
 * 5 - sin(2*PI*x/(xmax-xmin))*cos(2*PI*y/(ymax-ymin))
 */
int test_number = 5;

int px = 0;
int py = 0;
#ifdef P4_TO_P8
int pz = 0;
#endif

bool save_vtk = true;

double mu = 3e-5;
//double mu = 1.1;
double add_diagonal = 0.0;

BoundaryConditionType bc_itype = ROBIN;
//BoundaryConditionType bc_itype = NEUMANN;
//BoundaryConditionType bc_itype = DIRICHLET;

BoundaryConditionType bc_wtype = DIRICHLET;
//BoundaryConditionType bc_wtype = NEUMANN;

double diag_add = 0;

double xc = (xmax+xmin)/2;
double yc = (ymax+ymin)/2;
#ifdef P4_TO_P8
double zc = (zmax+zmin)/2;
double r0 = (double) MIN(xmax-xmin, ymax-ymin, zmax-zmin) / 4;
#else
double r0 = (double) MIN(xmax-xmin, ymax-ymin) / 4;
#endif



#ifdef P4_TO_P8

class LEVEL_SET: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(interface_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - xc) + SQR(y - yc) + SQR(z - zc));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class NO_INTERFACE_CF : public CF_3
{
public:
  double operator()(double, double, double) const
  {
    return -1;
  }
} no_interface_cf;

double u_exact(double x, double y, double z)
{
  switch(test_number)
  {
  case 0:
    return x+y+z;
  case 1:
    return x*x + y*y + z*z;
  case 2:
    return sin(x)*cos(y)*exp(z)+2;
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

class ROBIN_COEFF : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    return x+y+z;
  }
} robin_coef;

class BCINTERFACEVAL : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if(bc_itype==DIRICHLET)
      return u_exact(x,y,z);
    else if(bc_itype==NEUMANN || bc_itype==ROBIN)
    {
      double dx, dy, dz;
      switch(interface_type)
      {
      case 0:
        if(fabs(x-xc)<EPS && fabs(y-yc)<EPS && fabs(z-zc)<EPS)
        {
          dx = 0;
          dy = 0;
          dz = 0;
        }
        else
        {
          dx = -(x - xc)/sqrt(SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
          dy = -(y - yc)/sqrt(SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
          dz = -(z - zc)/sqrt(SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
        }
        break;
      default:
        throw std::invalid_argument("choose a valid interface type.");
      }

      double alpha = (bc_itype==ROBIN ? robin_coef(x,y,z) : 0);
      switch(test_number)
      {
      case 0:
        return dx + dy + dz + alpha*u_exact(x,y,z);
      case 1:
        return 2*x*dx + 2*y*dy + 2*z*dz + alpha*u_exact(x,y,z);
      case 2:
        return cos(x)*cos(y)*exp(z)*dx - sin(x)*sin(y)*exp(z)*dy + sin(x)*cos(y)*exp(z)*dz + alpha*u_exact(x,y,z);
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
    else
    {
      throw std::invalid_argument("unknown boundary condition type.");
    }
  }
} bc_interface_val;

class BCWALLTYPE : public WallBC3D
{
public:
  BoundaryConditionType operator()(double, double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;

class BCWALLVAL : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if(bc_wall_type(x,y,z)==DIRICHLET)
      return u_exact(x,y,z);
    else if(bc_wall_type(x,y,z)==NEUMANN)
    {
      double dx = 0; dx = fabs(x-xmin)<EPS ? -1 : (fabs(x-xmax)<EPS  ? 1 : 0);
      double dy = 0; dy = fabs(y-ymin)<EPS ? -1 : (fabs(y-ymax)<EPS  ? 1 : 0);
      double dz = 0; dz = fabs(z-zmin)<EPS ? -1 : (fabs(z-zmax)<EPS  ? 1 : 0);

      switch(test_number)
      {
      case 0:
        return dx + dy + dz;
      case 1:
        return 2*x*dx + 2*y*dy + 2*z*dz;
      case 2:
        return cos(x)*cos(y)*exp(z)*dx - sin(x)*sin(y)*exp(z)*dy + sin(x)*cos(y)*exp(z)*dz;
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
    else
    {
      throw std::invalid_argument("unknown boundary condition type.");
    }
  }
} bc_wall_val;

#else

class LEVEL_SET: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(interface_type)
    {
    case 0:
      return r0 - sqrt(SQR(x - xc) + SQR(y - yc));
    case 1:
      return -y+ymin+(ymax-ymin)/11;
    case 2:
      return -x+xmin+(xmax-xmin)/11;
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class NO_INTERFACE_CF : public CF_2
{
public:
  double operator()(double, double) const
  {
    return -1;
  }
} no_interface_cf;

double u_exact(double x, double y)
{
  double r = sqrt(SQR(x-xc) + SQR(y-yc));
  switch(test_number)
  {
  case 0:
    return x+y;
  case 1:
    return x*x + y*y;
  case 2:
    return sin(x)*cos(y);
  case 3:
    return sin(x)+cos(y);
  case 4:
    return cos(r);
  case 5:
    return sin(2*PI*x/(xmax-xmin))*cos(2*PI*y/(ymax-ymin));
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

class ROBIN_COEFF : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return -5.0;
//    return 3e5*SQR((4+x+y));
  }
} robin_coef;

class BCINTERFACEVAL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(bc_itype==DIRICHLET)
      return u_exact(x,y);
    else if(bc_itype==NEUMANN || bc_itype==ROBIN)
    {
      double dx, dy;
      switch(interface_type)
      {
      case 0:
        if(fabs(x-xc)<EPS && fabs(y-yc)<EPS)
        {
          dx = 0;
          dy = 0;
        }
        else
        {
          dx = -(x - xc)/sqrt(SQR(x - xc) + SQR(y - yc));
          dy = -(y - yc)/sqrt(SQR(x - xc) + SQR(y - yc));
        }
        break;
      case 1:
        dx =  0;
        dy = -1;
        break;
      case 2:
        dx = -1;
        dy =  0;
        break;
      default:
        throw std::invalid_argument("choose a valid interface type.");
      }

      double alpha = (bc_itype==ROBIN ? robin_coef(x,y) : 0);
      switch(test_number)
      {
      case 0:
        return dx + dy + alpha*u_exact(x,y);
      case 1:
        return 2*x*dx + 2*y*dy + alpha*u_exact(x,y);
      case 2:
        return cos(x)*cos(y)*dx - sin(x)*sin(y)*dy + alpha*u_exact(x,y);
      case 3:
        return cos(x)*dx - sin(y)*dy + alpha*u_exact(x,y);
      case 4:
        return sin(r0) + alpha*u_exact(x,y);
      case 5:
        return 2*PI/(xmax-xmin)*cos(2*PI*x/(xmax-xmin))*cos(2*PI*y/(ymax-ymin))*dx - 2*PI/(ymax-ymin)*sin(2*PI*x/(xmax-xmin))*sin(2*PI*y/(ymax-ymin))*dy + alpha*u_exact(x,y);
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
    else
    {
      throw std::invalid_argument("unknown boundary condition type.");
    }
  }
} bc_interface_val;

class BCWALLTYPE : public WallBC2D
{
public:
  BoundaryConditionType operator()(double, double) const
  {
    return bc_wtype;
  }
} bc_wall_type;

class BCWALLVAL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(bc_wall_type(x,y)==DIRICHLET)
      return u_exact(x,y);
    else if(bc_wall_type(x,y)==NEUMANN)
    {
      double dx = 0; dx = fabs(x-xmin)<EPS ? -1 : (fabs(x-xmax)<EPS  ? 1 : 0);
      double dy = 0; dy = fabs(y-ymin)<EPS ? -1 : (fabs(y-ymax)<EPS  ? 1 : 0);
      double r = sqrt(SQR(x-xc)+SQR(y-yc));
      switch(test_number)
      {
      case 0:
        return dx + dy;
      case 1:
        return 2*x*dx + 2*y*dy;
      case 2:
        return cos(x)*cos(y)*dx - sin(x)*sin(y)*dy;
      case 3:
        return cos(x)*dx - sin(y)*dy;
      case 4:
        return -(x-xc)*sin(r)/r *dx - (y-yc)*sin(r)/r *dy;
      case 5:
        return 2*PI/(xmax-xmin)*cos(2*PI*x/(xmax-xmin))*cos(2*PI*y/(ymax-ymin))*dx - 2*PI/(ymax-ymin)*sin(2*PI*x/(xmax-xmin))*sin(2*PI*y/(ymax-ymin))*dy;
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
    else
    {
      throw std::invalid_argument("unknown boundary condition type.");
    }
  }
} bc_wall_val;

#endif



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,
              my_p4est_node_neighbors_t *ngbd, my_p4est_brick_t *brick,
              Vec phi, Vec sol, Vec err,
              int compt)
{
  PetscErrorCode ierr;
  char *out_dir = NULL;
  out_dir = getenv("OUT_DIR");
  if(out_dir==NULL)
  {
    ierr = PetscPrintf(p4est->mpicomm, "You need to set the environment variable OUT_DIR to save visuals.\n"); CHKERRXX(ierr);
    return;
  }

  std::ostringstream oss;
  oss << out_dir << "/vtu/poisson_voronoi_" << compt;

  /* if periodic, create non periodic forest for visualization */
  my_p4est_brick_t brick_vis;
  p4est_connectivity_t *connectivity_vis = NULL;
  p4est_t *p4est_vis;
  p4est_ghost_t *ghost_vis;
  p4est_nodes_t *nodes_vis;
  Vec phi_vis;
  Vec sol_vis;
  Vec err_vis;

  if(is_periodic(p4est))
  {
    bool is_grid_changing = true;
    splitting_criteria_t* sp_old = (splitting_criteria_t*)p4est->user_pointer;
    my_p4est_interpolation_nodes_t interp(ngbd);

//    double xyz_min[P4EST_DIM];
//    double xyz_max[P4EST_DIM];
//    xyz_min_max(p4est, xyz_min, xyz_max);

#ifdef P4_TO_P8
    double  xyz_min [] = {xmin, ymin, zmin};
    double  xyz_max [] = {xmax, ymax, zmax};
    int     n_xyz   [] = {nx, ny, nz};
    int     periodic[] = {0, 0, 0};
#else
    double xyz_min [] = {xmin, ymin};
    double xyz_max [] = {xmax, ymax};
    int    n_xyz   [] = {nx, ny};
    int    periodic[] = {0, 0};
#endif

    p4est_connectivity_t *connectivity_vis = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick_vis, periodic);

    p4est_vis = my_p4est_new(p4est->mpicomm, connectivity_vis, 0, NULL, NULL);
    ghost_vis = my_p4est_ghost_new(p4est_vis, P4EST_CONNECT_FULL);
    nodes_vis = my_p4est_nodes_new(p4est_vis, ghost_vis);
    ierr = VecCreateGhostNodes(p4est_vis, nodes_vis, &phi_vis); CHKERRXX(ierr);

    for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
      interp.add_point(n, xyz);
    }
    interp.set_input(phi, linear);
    interp.interpolate(phi_vis);
    double *phi_vis_p;

    while(is_grid_changing)
    {
      ierr = VecGetArray(phi_vis, &phi_vis_p); CHKERRXX(ierr);
      splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
      is_grid_changing = sp.refine_and_coarsen(p4est_vis, nodes_vis, phi_vis_p);
      ierr = VecRestoreArray(phi_vis, &phi_vis_p); CHKERRXX(ierr);

      if(is_grid_changing)
      {
        my_p4est_partition(p4est_vis, P4EST_TRUE, NULL);
        p4est_ghost_destroy(ghost_vis); ghost_vis = my_p4est_ghost_new(p4est_vis, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_vis); nodes_vis = my_p4est_nodes_new(p4est_vis, ghost_vis);
        ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_vis, nodes_vis, &phi_vis); CHKERRXX(ierr);

        interp.clear();
        for(size_t n=0; n<nodes_vis->indep_nodes.elem_count; ++n)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est_vis, nodes_vis, xyz);
          interp.add_point(n, xyz);
        }
        interp.set_input(phi, linear);
        interp.interpolate(phi_vis);
      }
    }

    ierr = VecDuplicate(phi_vis, &sol_vis); CHKERRXX(ierr);
    interp.set_input(sol, linear);
    interp.interpolate(sol_vis);

    ierr = VecDuplicate(phi_vis, &err_vis); CHKERRXX(ierr);
    interp.set_input(err, linear);
    interp.interpolate(err_vis);
  }
  else
  {
    p4est_vis = p4est;
    ghost_vis = ghost;
    nodes_vis = nodes;
    phi_vis = phi;
    sol_vis = sol;
    err_vis = err;
  }

  double *phi_p, *sol_p, *err_p;
  ierr = VecGetArray(phi_vis, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol_vis, &sol_p); CHKERRXX(ierr);
  ierr = VecGetArray(err_vis, &err_p); CHKERRXX(ierr);

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est_vis, ghost_vis, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est_vis->first_local_tree; tree_idx <= p4est_vis->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_vis->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost_vis->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost_vis->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  my_p4est_vtk_write_all(p4est_vis, nodes_vis, ghost_vis,
                         P4EST_TRUE, P4EST_TRUE,
                         3, 1, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol", sol_p,
                         VTK_POINT_DATA, "err", err_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_vis, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol_vis, &sol_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_vis, &err_p); CHKERRXX(ierr);

  if(is_periodic(p4est))
  {
    ierr = VecDestroy(err_vis); CHKERRXX(ierr);
    ierr = VecDestroy(sol_vis); CHKERRXX(ierr);
    ierr = VecDestroy(phi_vis); CHKERRXX(ierr);
    p4est_nodes_destroy(nodes_vis);
    p4est_ghost_destroy(ghost_vis);
    p4est_destroy(p4est_vis);
    my_p4est_brick_destroy(connectivity_vis, &brick_vis);
  }

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("check_extrapolations", "verify second order convergence for extrapolations, default = 0");
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
                 0 - x+y+z\n\
                 1 - x*x + y*y + z*z\n\
                 2 - sin(x)*cos(y)*exp(z)");
#else
  cmd.add_option("test", "choose a test.\n\
                 0 - x+y\n\
                 1 - x*x + y*y\n\
                 2 - sin(x)*cos(y)\n\
                 3 - sin(x) + cos(y)\n\
                 4 - cos(r)\n\
                 5 - sin(2*PI*x/(xmax-xmin))*cos(y)");
#endif
  cmd.parse(argc, argv);

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  test_number = cmd.get("test", test_number);

  bc_wtype = cmd.get("bc_wtype", bc_wtype);
  bc_itype = cmd.get("bc_itype", bc_itype);

  save_vtk = cmd.get("save_vtk", save_vtk);

  bool check_extrapolations = cmd.get("check_extrapolations", 1);

  parStopWatch w;
  w.start("total time");


  if(0)
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }


  /* create the p4est */
  my_p4est_brick_t brick;

#ifdef P4_TO_P8
  double  xyz_min [] = {xmin, ymin, zmin};
  double  xyz_max [] = {xmax, ymax, zmax};
  int     n_xyz   [] = {nx, ny, nz};
  int     periodic[] = {px, py, pz};
#else
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
  int    n_xyz   [] = {nx, ny};
  int    periodic[] = {px, py};
#endif

  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  double err_n;
  double err_nm1;

  double err_ex_n;
  double err_ex_nm1;

  double avg_exa = 0;
  if((bc_itype==NOINTERFACE || bc_itype==NEUMANN) && bc_wtype==NEUMANN)
  {
    if(bc_itype==NOINTERFACE)
    {
      switch(test_number)
      {
#ifdef P4_TO_P8
      case 0: avg_exa = .5*xmax*ymax*zmax*(xmax+ymax+zmax) - .5*xmin*ymin*zmin*(xmin+ymin+zmin); break;
      case 1: avg_exa = 1./3.*xmax*ymax*zmax*(xmax*xmax + ymax*ymax + zmax*zmax) - 1./3.*xmin*ymin*zmin*(xmin*xmin + ymin*ymin + zmin*zmin); break;
      case 2: avg_exa = sin(ymax)*(1-cos(xmax))*(exp(zmax)-1) + 2.*xmax*ymax*zmax - sin(ymin)*(1-cos(xmin))*(exp(zmin)-1) + 2.*xmin*ymin*zmin; break;
#else
      case 0: avg_exa = .5*xmax*ymax*(xmax+ymax) - .5*xmin*ymin*(xmin+ymin); break;
      case 1: avg_exa = 1./3.*xmax*ymax*(xmax*xmax + ymax*ymax) - 1./3.*xmin*ymin*(xmin*xmin + ymin*ymin); break;
      case 2: avg_exa = sin(ymax)*(1-cos(xmax)) - sin(ymin)*(1-cos(xmin)); break;
#endif
      default: throw std::invalid_argument("invalid test number.");
      }

#ifdef P4_TO_P8
      avg_exa /= (double)(xmax-xmin)*(ymax-ymin)*(zmax-zmin);
#else
      avg_exa /= (double)(xmax-xmin)*(ymax-ymin);
#endif
    }
    else
    {
      switch(test_number)
      {
#ifndef P4_TO_P8
      case 0: avg_exa = .5*xmax*ymax*(xmax+ymax) - .5*xmin*ymin*(xmin+ymin) - 2*PI*r0*r0; break;
      case 1: avg_exa = 1./3.*xmax*ymax*(xmax*xmax + ymax*ymax) - 1./3.*xmin*ymin*(xmin*xmin + ymin*ymin) - PI*r0*r0*(SQR((xmax+xmin)/2) + SQR((ymax+ymin)/2) + r0*r0/2); break;
#endif
      default: throw std::invalid_argument("all neumann bc not implemented for this case.");
      }

      avg_exa /= (double)(xmax-xmin)*(ymax-ymin) - PI*r0*r0;
    }
  }

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

//    srand(1);
//    splitting_criteria_random_t data(2, 7, 1000, 10000);
    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set, 1.2);
    p4est->user_pointer = (void*)(&data);

//    my_p4est_refine(p4est, P4EST_TRUE, refine_random, NULL);
    for(int i=0; i<lmax+iter; ++i)
    {
      my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
//    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
//    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
//    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    Vec phi;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
    if(bc_itype==NOINTERFACE)
      sample_cf_on_nodes(p4est, nodes, no_interface_cf, phi);
    else
      sample_cf_on_nodes(p4est, nodes, level_set, phi);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.perturb_level_set_function(phi, EPS);

    /* find dx and dy smallest */
    double dxyz[P4EST_DIM];
    dxyz_min(p4est, dxyz);

#ifdef P4_TO_P8
    double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
#else
    double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
#endif

    /* TEST THE NODES FUNCTIONS */
#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif

    bc.setWallTypes(bc_wall_type);
    bc.setWallValues(bc_wall_val);
    bc.setInterfaceType(bc_itype);
    bc.setInterfaceValue(bc_interface_val);

    Vec rhs;
    ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
    double *rhs_p;
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double x = node_x_fr_n(n, p4est, nodes);
      double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, p4est, nodes);
#endif
      double r = sqrt(SQR(x-xc)+SQR(y-yc));
      switch(test_number)
      {
#ifdef P4_TO_P8
      case 0:
        rhs_p[n] = mu*0 + add_diagonal*u_exact(x,y,z);
        break;
      case 1:
        rhs_p[n] = -6*mu + add_diagonal*u_exact(x,y,z);
        break;
      case 2:
        rhs_p[n] = mu*sin(x)*cos(y)*exp(z) + add_diagonal*u_exact(x,y,z);
        break;
#else
      case 0:
        rhs_p[n] = mu*0 + add_diagonal*u_exact(x,y);
        break;
      case 1:
        rhs_p[n] = -4*mu + add_diagonal*u_exact(x,y);
        break;
      case 2:
        rhs_p[n] = 2*mu*sin(x)*cos(y) + add_diagonal*u_exact(x,y);
        break;
      case 3:
        rhs_p[n] = mu*u_exact(x,y) + add_diagonal*u_exact(x,y);
        break;
      case 4:
        rhs_p[n] = fabs(r)<EPS ? 2*mu : mu*(sin(r)/r + cos(r)) + add_diagonal*u_exact(x,y);
        break;
      case 5:
        rhs_p[n] = mu*( SQR(2*PI/(xmax-xmin))*sin(2*PI*x/(xmax-xmin))*cos(2*PI*y/(ymax-ymin)) + SQR(2*PI/(ymax-ymin))*sin(2*PI*x/(xmax-xmin))*cos(2*PI*y/(ymax-ymin)) ) + add_diagonal*u_exact(x,y);
        break;
#endif
      default:
        throw std::invalid_argument("set rhs : unknown test number.");
      }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

//    my_p4est_poisson_nodes_voronoi_t solver(&ngbd_n);
    my_p4est_poisson_nodes_t solver(&ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(add_diagonal);
    solver.set_mu(mu);
    solver.set_bc(bc);
    solver.set_rhs(rhs);

    Vec robin;
    if(bc_itype==ROBIN || bc_wtype==ROBIN)
    {
      ierr = VecDuplicate(phi, &robin); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, robin_coef, robin);
      solver.set_robin_coef(robin);
    }

    Vec sol;
    ierr = VecDuplicate(rhs, &sol); CHKERRXX(ierr);

    solver.solve(sol);

    if(bc_itype==ROBIN || bc_wtype==ROBIN)
    {
      ierr = VecDestroy(robin);
    }

    /* if all NEUMANN boundary conditions, shift solution */
    if(solver.get_matrix_has_nullspace())
    {
      double avg_sol = integrate_over_interface(p4est, nodes, phi, sol)/area_in_negative_domain(p4est, nodes, phi);

      double *sol_p;
      ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
        sol_p[n] = sol_p[n] - avg_sol + avg_exa;

      ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
    }

    /* check the error */
    err_nm1 = err_n;
    err_n = 0;

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    const double *sol_p;
    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);

    Vec err_nodes;
    ierr = VecDuplicate(sol, &err_nodes); CHKERRXX(ierr);
    double *err_p;
    ierr = VecGetArray(err_nodes, &err_p); CHKERRXX(ierr);

    double err_bc = 0;
    ls.extend_Over_Interface_TVD(phi, sol);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(bc_itype==NOINTERFACE || phi_p[n]<0)
      {
        double x = node_x_fr_n(n, p4est, nodes);
        double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double z = node_z_fr_n(n, p4est, nodes);
        err_p[n] = fabs(sol_p[n] - u_exact(x,y,z));
#else
        err_p[n] = fabs(sol_p[n] - u_exact(x,y));
#endif
        err_n = MAX(err_n, err_p[n]);

        /* check that boundary condition is satisfied */
#ifdef P4_TO_P8
        if(phi_p[n]>-MIN(dxyz[0],dxyz[1],dxyz[2]))
#else
        if(phi_p[n]>-MIN(dxyz[0],dxyz[1]))
#endif
        {
          const quad_neighbor_nodes_of_node_t &qnnn = ngbd_n.get_neighbors(n);
          double nx = qnnn.dx_central(phi_p);
          double ny = qnnn.dy_central(phi_p);
          double norm = sqrt(nx*nx + ny*ny);
          if(norm>EPS) { nx /= norm; ny /= norm; }
          else         { nx = 0; ny = 0; }
          double du_dn = qnnn.dx_central(sol_p)*nx + qnnn.dy_central(sol_p)*ny;
          switch(bc_itype)
          {
          case DIRICHLET:
            err_bc = MAX(err_bc, fabs(sol_p[n] - bc_interface_val(x,y)));
            break;
          case NEUMANN:
            err_bc = MAX(err_bc, fabs(du_dn - bc_interface_val(x,y)));
            break;
          case ROBIN:
            err_bc = MAX(err_bc, fabs(du_dn + robin_coef(x,y)*sol_p[n] - bc_interface_val(x,y)));
            break;
          default:
            throw std::invalid_argument("Unknown interface boundary condition.");
          }

        }
      }
      else
        err_p[n] = 0;
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_nodes, &err_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_bc, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(mpi.comm(), "Error on the boundary condition : %e\n", err_bc); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(mpi.comm(), "Error on nodes : %e, order = %g\n", err_n, log(err_nm1/err_n)/log(2)); CHKERRXX(ierr);


    /* extrapolate the solution and check accuracy */
    double band = 4;

    if(check_extrapolations)
    {
      Vec mask = solver.get_mask();

      if(bc_itype!=NOINTERFACE)
//        ls.extend_Over_Interface_TVD(phi, mask, sol, 100, 2);
        ls.extend_Over_Interface_TVD(phi, sol, 100, 2);

      ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);
      ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

      Vec err_ex;
      ierr = VecDuplicate(sol, &err_ex); CHKERRXX(ierr);
      double *err_ex_p;
      ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

      err_ex_nm1 = err_ex_n;
      err_ex_n = 0;

      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if(phi_p[n]>0)
        {
          double x = node_x_fr_n(n, p4est, nodes);
          double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
          double z = node_z_fr_n(n, p4est, nodes);
          if(phi_p[n]<band*diag)
            err_ex_p[n] = fabs(sol_p[n] - u_exact(x,y,z));
#else
          if(phi_p[n]<band*diag)
            err_ex_p[n] = fabs(sol_p[n] - u_exact(x,y));
#endif
          else
            err_ex_p[n] = 0;

          err_ex_n = MAX(err_ex_n, err_ex_p[n]);
        }
        else
          err_ex_p[n] = 0;
      }

      ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_n, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(mpi.comm(), "Error extrapolation : %g, order = %g\n", err_ex_n, log(err_ex_nm1/err_ex_n)/log(2)); CHKERRXX(ierr);

      ierr = VecDestroy(err_ex); CHKERRXX(ierr);
    }


    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &ngbd_n, &brick, phi, sol, err_nodes, iter);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);

    ierr = VecDestroy(err_nodes); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
