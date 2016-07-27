
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
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_poisson_cells.h>
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
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

double xmin =  0;
double xmax =  1;
double ymin =  0;
double ymax =  1;
#ifdef P4_TO_P8
double zmin =  0;
double zmax =  1;
#endif

using namespace std;

int lmin = 3;
int lmax = 3;
int nb_splits = 4;

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

double mu = 1;
double add_diagonal = 0;

/*
 * 0 - circle
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 - x+y
 * 1 - x*x + y*y
 * 2 - sin(x)*cos(y)
 * 3 - cos(r-r0)  so that homogeneous neumann on interface
 *
 *  ********* 3D *********
 * 0 - x+y+z
 * 1 - x*x + y*y + z*z
 * 2 - sin(x)*cos(y)*exp(z)
 */
int test_number = 2;

//BoundaryConditionType bc_itype = NEUMANN;
BoundaryConditionType bc_itype = NOINTERFACE;
//BoundaryConditionType bc_wtype = DIRICHLET;
BoundaryConditionType bc_wtype = NEUMANN;

double diag_add = 0;

#ifdef P4_TO_P8
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
    case 0: return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
    default: throw std::invalid_argument("Choose a valid level set.");
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
  case 0: return x+y+z;
  case 1: return x*x + y*y + z*z;
  case 2: return sin(x)*cos(y)*exp(z);
  case 3: return cos(sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2)) - r0);
  default: throw std::invalid_argument("Choose a valid test.");
  }
}

class BCINTERFACEVAL : public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    if(bc_itype==DIRICHLET)
      return u_exact(x,y,z);
    else
    {
      double dx, dy, dz;
      switch(interface_type)
      {
      case 0:
        if(fabs(x-(xmax+xmin)/2)<EPS && fabs(y-(ymax+ymin)/2)<EPS && fabs(z-(zmax+zmin)/2)<EPS)
        {
          dx = 0;
          dy = 0;
          dz = 0;
        }
        else
        {
          dx = -(x - (xmax+xmin)/2)/sqrt(SQR(x -(xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
          dy = -(y - (ymax+ymin)/2)/sqrt(SQR(x -(xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
          dz = -(z - (zmax+zmin)/2)/sqrt(SQR(x -(xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
        }
        break;
      default:
        throw std::invalid_argument("choose a valid interface type.");
      }

      switch(test_number)
      {
      case 0: return dx + dy + dz;
      case 1: return 2*x*dx + 2*y*dy + 2*z*dz;
      case 2: return cos(x)*cos(y)*exp(z)*dx - sin(x)*sin(y)*exp(z)*dy + sin(x)*cos(y)*exp(z)*dz;
      case 3: return sin(sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2)) - r0);
      default: throw std::invalid_argument("Choose a valid test.");
      }
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
    else
    {
      double dx = 0; dx = fabs(x-xmin)<EPS ? -1 : (fabs(x-xmax)<EPS  ? 1 : 0);
      double dy = 0; dy = fabs(y-ymin)<EPS ? -1 : (fabs(y-ymax)<EPS  ? 1 : 0);
      double dz = 0; dz = fabs(z-zmin)<EPS ? -1 : (fabs(z-zmax)<EPS  ? 1 : 0);
      double r = sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
      switch(test_number)
      {
      case 0: return dx + dy + dz;
      case 1: return 2*x*dx + 2*y*dy + 2*z*dz;
      case 2: return cos(x)*cos(y)*exp(z)*dx - sin(x)*sin(y)*exp(z)*dy + sin(x)*cos(y)*exp(z)*dz;
      case 3: return -sin(r-r0) * (dx*x/r + dy*y/r + dz*z/r);
      default: throw std::invalid_argument("Choose a valid test.");
      }
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
      return r0 - sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2));
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
  switch(test_number)
  {
  case 0:
    return x+y;
  case 1:
    return x*x + y*y;
  case 2:
    return sin(x)*cos(y);
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

class BCINTERFACEVAL : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    if(bc_itype==DIRICHLET)
      return u_exact(x,y);
    else
    {
      double dx, dy;
      switch(interface_type)
      {
      case 0:
        if(fabs(x-(xmax+xmin)/2)<EPS && fabs(y-(ymax+ymin)/2)<EPS)
        {
          dx = 0;
          dy = 0;
        }
        else
        {
          dx = -(x - (xmax+xmin)/2)/sqrt(SQR(x -(xmax+xmin)/2) + SQR(y - (ymax+ymin)/2));
          dy = -(y - (ymax+ymin)/2)/sqrt(SQR(x -(xmax+xmin)/2) + SQR(y - (ymax+ymin)/2));
        }
        break;
      default:
        throw std::invalid_argument("choose a valid interface type.");
      }

      switch(test_number)
      {
      case 0:
        return dx + dy;
      case 1:
        return 2*x*dx + 2*y*dy;
      case 2:
        return cos(x)*cos(y)*dx - sin(x)*sin(y)*dy;
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
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
    else
    {
      double dx = 0; dx = fabs(x-xmin)<EPS ? -1 : (fabs(x-xmax)<EPS  ? 1 : 0);
      double dy = 0; dy = fabs(y-ymin)<EPS ? -1 : (fabs(y-ymax)<EPS  ? 1 : 0);
      switch(test_number)
      {
      case 0:
        return dx + dy;
      case 1:
        return 2*x*dx + 2*y*dy;
      case 2:
        return cos(x)*cos(y)*dx - sin(x)*sin(y)*dy;
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
  }
} bc_wall_val;

#endif



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec sol_cells, Vec err_cells, Vec err_ex,
              int compt)
{
  PetscErrorCode ierr;
  std::ostringstream oss;
  const char *out_dir = getenv("OUT_DIR");

  if (out_dir)
    oss << out_dir << "/vtu";
  else;
    oss << "./out_dir/vtu";
  ostringstream command;
  command << "mkdir -p " << oss.str();
  if (p4est->mpirank == 0) cout << "Creating a folder in " << oss.str() << endl;
  system(command.str().c_str());

  oss << "/cells_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p, *sol_cells_p, *err_cells_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol_cells, &sol_cells_p); CHKERRXX(ierr);
  ierr = VecGetArray(err_cells, &err_cells_p); CHKERRXX(ierr);

  double *err_ex_p;
  ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

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
                         1, 4, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_CELL_DATA, "sol_cells", sol_cells_p,
                         VTK_CELL_DATA, "err_cells", err_cells_p,
                         VTK_CELL_DATA, "err_ex", err_ex_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol_cells, &sol_cells_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(err_cells, &err_cells_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
#ifdef P4_TO_P8
  cmd.add_option("test", "choose a test.\n\
                 0 - x+y+z\n\
                 1 - x*x + y*y + z*z\n\
                 2 - sin(x)*cos(y)*exp(z)");
#else
  cmd.add_option("test", "choose a test.\n\
                 0 - x+y\n\
                 1 - x*x + y*y\n\
                 2 - sin(x)*cos(y)");
#endif
  cmd.parse(argc, argv);
  cmd.print();

  lmin = cmd.get("lmin", lmin);
  lmax = cmd.get("lmax", lmax);
  nb_splits = cmd.get("nb_splits", nb_splits);
  test_number = cmd.get("test", test_number);

  bc_wtype = cmd.get("bc_wtype", bc_wtype);
  bc_itype = cmd.get("bc_itype", bc_itype);

  bool save_vtk = cmd.contains("save_vtk");

  parStopWatch w;
  w.start("total time");

  // FIXME: What is this for? Attaching debugger?
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

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
#ifdef P4_TO_P8
  int n_xyz [] = {nx, ny, nz};
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
#else
  int n_xyz [] = {nx, ny};
  double xyz_min [] = {xmin, ymin};
  double xyz_max [] = {xmax, ymax};
#endif
  const int periodic []   = {0, 0, 0};
  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

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
    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_cell_neighbors_t ngbd_c(&hierarchy);
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
    double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double diag = sqrt(dx*dx + dy*dy);
#endif

    /* TEST THE CELLS FUNCTIONS */
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
    ierr = VecCreateGhostCells(p4est, ghost, &rhs); CHKERRXX(ierr);
    double *rhs_p;
    ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

    for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      for(size_t quad_idx=0; quad_idx<tree->quadrants.elem_count; ++quad_idx)
      {
        p4est_locidx_t q_idx = quad_idx + tree->quadrants_offset;
        double x = quad_x_fr_q(q_idx, tree_idx, p4est, ghost);
        double y = quad_y_fr_q(q_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
        double z = quad_z_fr_q(q_idx, tree_idx, p4est, ghost);
        double r = sqrt(SQR(x - (xmax+xmin)/2) + SQR(y - (ymax+ymin)/2) + SQR(z - (zmax+zmin)/2));
#endif
        switch(test_number)
        {
#ifdef P4_TO_P8
        case 0:
          rhs_p[q_idx] = mu*0 + add_diagonal*u_exact(x,y,z);
          break;
        case 1:
          rhs_p[q_idx] = -6*mu + add_diagonal*u_exact(x,y,z);
          break;
        case 2:
          rhs_p[q_idx] = mu*sin(x)*cos(y)*exp(z) + add_diagonal*u_exact(x,y,z);
          break;
        case 3:
          rhs_p[q_idx] = mu*(2/r * sin(r-r0) + cos(r-r0)) + add_diagonal*u_exact(x,y,z);
          break;
#else
        case 0:
          rhs_p[q_idx] = mu*0 + add_diagonal*u_exact(x,y);
          break;
        case 1:
          rhs_p[q_idx] = -4*mu + add_diagonal*u_exact(x,y);
          break;
        case 2:
          rhs_p[q_idx] = 2*mu*sin(x)*cos(y) + add_diagonal*u_exact(x,y);
          break;
#endif
        default:
          throw std::invalid_argument("set rhs : unknown test number.");
        }
      }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

    my_p4est_poisson_cells_t solver(&ngbd_c, &ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(add_diagonal);
    solver.set_mu(mu);
    solver.set_bc(bc);
    solver.set_rhs(rhs);

    Vec sol;
    ierr = VecDuplicate(rhs, &sol); CHKERRXX(ierr);

    solver.solve(sol);

    /* if all NEUMANN boundary conditions, shift solution */
    if(solver.get_matrix_has_nullspace())
    {
      my_p4est_level_set_cells_t lsc(&ngbd_c, &ngbd_n);
      double avg_sol = lsc.integrate(phi, sol)/area_in_negative_domain(p4est, nodes, phi);

      double *sol_p;
      ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

      for(p4est_locidx_t quad_idx=0; quad_idx<p4est->local_num_quadrants; ++quad_idx)
        sol_p[quad_idx] = sol_p[quad_idx] - avg_sol + avg_exa;

      for(size_t quad_idx=0; quad_idx<ghost->ghosts.elem_count; ++quad_idx)
        sol_p[quad_idx+p4est->local_num_quadrants] = sol_p[quad_idx+p4est->local_num_quadrants] - avg_sol + avg_exa;

      ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
    }

    /* check the error */
    err_nm1 = err_n;
    err_n = 0;

    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    const double *sol_p;
    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);

    Vec err_cells;
    ierr = VecDuplicate(sol, &err_cells); CHKERRXX(ierr);
    double *err_p;
    ierr = VecGetArray(err_cells, &err_p); CHKERRXX(ierr);

    for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      for(size_t quad_idx=0; quad_idx<tree->quadrants.elem_count; ++quad_idx)
      {
        p4est_locidx_t q_idx = quad_idx + tree->quadrants_offset;

        /* check if quadrant is well defined */
        double phi_q = 0;
        bool is_neg = false;
        for(int i=0; i<P4EST_CHILDREN; ++i)
        {
          double tmp = phi_p[nodes->local_nodes[P4EST_CHILDREN*q_idx + i]];
          phi_q += tmp;
          is_neg = is_neg || (tmp<0);
        }
        phi_q /= (double) P4EST_CHILDREN;

        if(bc_itype==NOINTERFACE || (bc_itype==DIRICHLET && phi_q<0) || (bc_itype==NEUMANN && is_neg))
        {
          double x = quad_x_fr_q(q_idx, tree_idx, p4est, ghost);
          double y = quad_y_fr_q(q_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
          double z = quad_z_fr_q(q_idx, tree_idx, p4est, ghost);
          err_p[q_idx] = fabs(sol_p[q_idx] - u_exact(x,y,z));
#else
          err_p[q_idx] = fabs(sol_p[q_idx] - u_exact(x,y));
#endif
          err_n = MAX(err_n, err_p[q_idx]);
        }
        else
          err_p[q_idx] = 0;
      }
    }
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_cells, &err_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Error on cells : %.5e, order = %g\n", err_n, log(err_nm1/err_n)/log(2)); CHKERRXX(ierr);


    /* extrapolate the solution and check accuracy */
    my_p4est_level_set_cells_t ls_c(&ngbd_c, &ngbd_n);
    double band = 4;

    if(bc_itype!=NOINTERFACE)
      ls_c.extend_Over_Interface(phi, sol, &bc, 2, band);

    ierr = VecGetArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    Vec err_ex;
    ierr = VecDuplicate(sol, &err_ex); CHKERRXX(ierr);
    double *err_ex_p;
    ierr = VecGetArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    err_ex_nm1 = err_ex_n;
    err_ex_n = 0;

    for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      for(size_t quad_idx=0; quad_idx<tree->quadrants.elem_count; ++quad_idx)
      {
        p4est_locidx_t q_idx = quad_idx + tree->quadrants_offset;

        /* check if quadrant is well defined */
        double phi_q = 0;
        bool all_pos = true;
        for(int i=0; i<P4EST_CHILDREN; ++i)
        {
          double tmp = phi_p[nodes->local_nodes[P4EST_CHILDREN*q_idx + i]];
          phi_q += tmp;
          all_pos = all_pos && (tmp>0);
        }
        phi_q /= (double) P4EST_CHILDREN;

        if(bc_itype==NOINTERFACE || (bc_itype==DIRICHLET && phi_q>0) || (bc_itype==NEUMANN && all_pos))
        {
          double x = quad_x_fr_q(q_idx, tree_idx, p4est, ghost);
          double y = quad_y_fr_q(q_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
          double z = quad_z_fr_q(q_idx, tree_idx, p4est, ghost);
          if(phi_q<band*diag)
            err_ex_p[q_idx] = fabs(sol_p[q_idx] - u_exact(x,y,z));
#else
          if(phi_q<band*diag)
            err_ex_p[q_idx] = fabs(sol_p[q_idx] - u_exact(x,y));
#endif
          else
            err_ex_p[q_idx] = 0;

          err_ex_n = MAX(err_ex_n, err_ex_p[q_idx]);
        }
        else
          err_ex_p[q_idx] = 0;
      }
    }

    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol, &sol_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_ex, &err_ex_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (err_ex, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_n, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = PetscPrintf(p4est->mpicomm, "Error extrapolation : %.5e, order = %g\n", err_ex_n, log(err_ex_nm1/err_ex_n)/log(2)); CHKERRXX(ierr);


    if(save_vtk)
    {
      save_VTK(p4est, ghost, nodes, &brick, phi, sol, err_cells, err_ex, iter);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    ierr = VecDestroy(rhs); CHKERRXX(ierr);
    ierr = VecDestroy(sol); CHKERRXX(ierr);

    ierr = VecDestroy(err_cells); CHKERRXX(ierr);
    ierr = VecDestroy(err_ex); CHKERRXX(ierr);

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
