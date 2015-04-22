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
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_poisson_faces.h>
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
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

int lmin = 2;
int lmax = 4;
int nb_splits = 1;

int nx = 2;
int ny = 2;
#ifdef P4_TO_P8
int nz = 2;
#endif

double mu = 1.5;
double add_diagonal = 2.3;

/*
 * 0 - circle
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 - u_m=1+log(r/r0), u_p=1, mu_m=mu_p=1, diag_add=0
 */
int test_number = 0;

BoundaryConditionType bc_itype = DIRICHLET;
BoundaryConditionType bc_wtype = DIRICHLET;

double diag_add = 0;

#ifdef P4_TO_P8
double r0 = (double) MIN(nx,ny,nz) / 4;
#else
double r0 = (double) MIN(nx,ny) / 4;
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
      return r0 - sqrt(SQR(x - (double) nx/2) + SQR(y - (double) ny/2) + SQR(z - (double) nz/2));
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
        dx = -(x - (double) nx/2)/sqrt(SQR(x -(double) nx/2) + SQR(y - (double) ny/2) + SQR(z - (double) nz/2));
        dy = -(y - (double) ny/2)/sqrt(SQR(x -(double) nx/2) + SQR(y - (double) ny/2) + SQR(z - (double) nz/2));
        dz = -(z - (double) nz/2)/sqrt(SQR(x -(double) nx/2) + SQR(y - (double) ny/2) + SQR(z - (double) nz/2));
        break;
      default:
        throw std::invalid_argument("choose a valid interface type.");
      }

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
      double dx = 0; dx = fabs(x)<EPS ? -1 : (fabs(x-nx)<EPS  ? 1 : 0);
      double dy = 0; dy = fabs(y)<EPS ? -1 : (fabs(y-ny)<EPS  ? 1 : 0);
      double dz = 0; dz = fabs(z)<EPS ? -1 : (fabs(z-nz)<EPS  ? 1 : 0);
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
      return r0 - sqrt(SQR(x - (double) nx/2) + SQR(y - (double) ny/2));
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
        dx = -(x - (double) nx/2)/sqrt(SQR(x -(double) nx/2) + SQR(y - (double) ny/2));
        dy = -(y - (double) ny/2)/sqrt(SQR(x -(double) nx/2) + SQR(y - (double) ny/2));
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
      double dx = 0; dx = fabs(x)<EPS ? -1 : (fabs(x-nx)<EPS  ? 1 : 0);
      double dy = 0; dy = fabs(y)<EPS ? -1 : (fabs(y-ny)<EPS  ? 1 : 0);
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
              Vec phi, Vec u_idx, Vec v_idx, Vec *sol_nodes, Vec *err_nodes,
              int compt)
{
  PetscErrorCode ierr;
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, "/home/guittet/code/Output/p4est_navier_stokes");
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/faces_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p, *u_idx_p, *v_idx_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(u_idx, &u_idx_p); CHKERRXX(ierr);
  ierr = VecGetArray(v_idx, &v_idx_p); CHKERRXX(ierr);

  double *sol_nodes_p[P4EST_DIM];
  double *err_nodes_p[P4EST_DIM];
  for(int d=0; d<P4EST_DIM; ++d)
  {
    ierr = VecGetArray(sol_nodes[d], &sol_nodes_p[d]); CHKERRXX(ierr);
    ierr = VecGetArray(err_nodes[d], &err_nodes_p[d]); CHKERRXX(ierr);
  }

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
                         1+2*P4EST_DIM, 3, oss.str().c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "sol_u", sol_nodes_p[0],
      VTK_POINT_DATA, "sol_v", sol_nodes_p[1],
    #ifdef P4_TO_P8
      VTK_POINT_DATA, "sol_w", sol_nodes_p[2],
    #endif
                         VTK_POINT_DATA, "err_u", err_nodes_p[0],
      VTK_POINT_DATA, "err_v", err_nodes_p[1],
    #ifdef P4_TO_P8
      VTK_POINT_DATA, "err_w", err_nodes_p[2],
    #endif
                         VTK_CELL_DATA, "u_idx", u_idx_p,
                         VTK_CELL_DATA, "v_idx", v_idx_p,
                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(u_idx, &u_idx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(v_idx, &v_idx_p); CHKERRXX(ierr);

  for(int d=0; d<P4EST_DIM; ++d)
  {
    ierr = VecRestoreArray(sol_nodes[d], &sol_nodes_p[d]); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_nodes[d], &err_nodes_p[d]); CHKERRXX(ierr);
  }

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}



int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_context_t mpi_context, *mpi = &mpi_context;
  mpi->mpicomm  = MPI_COMM_WORLD;

  Session mpi_session;
  mpi_session.init(argc, argv, mpi->mpicomm);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the tree");
  cmd.add_option("lmax", "max level of the tree");
  cmd.add_option("nb_splits", "number of recursive splits");
  cmd.add_option("bc_wtype", "type of boundary condition to use on the wall");
  cmd.add_option("bc_itype", "type of boundary condition to use on the interface");
  cmd.add_option("save_voro", "save the voronoi partition in vtk format");
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

  bool save_voro = cmd.get("save_voro", false);
  bool save_vtk = cmd.get("save_vtk", false);

  parStopWatch w;
  w.start("total time");

  MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);

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
  connectivity = my_p4est_brick_new(nx, ny, nz, &brick);
#else
  connectivity = my_p4est_brick_new(nx, ny, &brick);
#endif

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  vector<double> err_n  (P4EST_DIM, 0);
  vector<double> err_nm1(P4EST_DIM, 0);

  vector<double> err_nodes_n  (P4EST_DIM, 0);
  vector<double> err_nodes_nm1(P4EST_DIM, 0);

  vector<double> err_ex_f_n  (P4EST_DIM, 0);
  vector<double> err_ex_f_nm1(P4EST_DIM, 0);

  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi->mpicomm, "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi->mpicomm, connectivity, 0, NULL, NULL);

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
  #endif

    /* TEST THE FACES FUNCTIONS */
    my_p4est_faces_t faces(p4est, ghost, &brick, &ngbd_c);

#ifdef P4_TO_P8
    BoundaryConditions3D bc[P4EST_DIM];
#else
    BoundaryConditions2D bc[P4EST_DIM];
#endif

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      bc[dir].setWallTypes(bc_wall_type);
      bc[dir].setWallValues(bc_wall_val);
      bc[dir].setInterfaceType(bc_itype);
      bc[dir].setInterfaceValue(bc_interface_val);
    }

    Vec rhs[P4EST_DIM];
    Vec face_is_well_defined[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostFaces(p4est, &faces, &rhs[dir], dir); CHKERRXX(ierr);
      double *rhs_p;
      ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

      ierr = VecDuplicate(rhs[dir], &face_is_well_defined[dir]); CHKERRXX(ierr);
      check_if_faces_are_well_defined(p4est, &ngbd_n, &faces, dir, phi, bc_itype, face_is_well_defined[dir]);

      for(p4est_locidx_t f_idx=0; f_idx<faces.num_local[dir]; ++f_idx)
      {
        double x = faces.x_fr_f(f_idx, dir);
        double y = faces.y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
        double z = faces.z_fr_f(f_idx, dir);
#endif
        switch(test_number)
        {
#ifdef P4_TO_P8
        case 0:
          rhs_p[f_idx] = mu*0 + add_diagonal*u_exact(x,y,z);
          break;
        case 1:
          rhs_p[f_idx] = -6*mu + add_diagonal*u_exact(x,y,z);
          break;
        case 2:
          rhs_p[f_idx] = mu*sin(x)*cos(y)*exp(z) + add_diagonal*u_exact(x,y,z);
          break;
#else
        case 0:
          rhs_p[f_idx] = mu*0 + add_diagonal*u_exact(x,y);
          break;
        case 1:
          rhs_p[f_idx] = -4*mu + add_diagonal*u_exact(x,y);
          break;
        case 2:
          rhs_p[f_idx] = 2*mu*sin(x)*cos(y) + add_diagonal*u_exact(x,y);
          break;
#endif
        default:
          throw std::invalid_argument("set rhs : unknown test number.");
        }
      }

      ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
    }

    PoissonSolverFaces solver(&faces, &ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(add_diagonal);
    solver.set_mu(mu);
    solver.set_bc(bc);
    solver.set_rhs(rhs);
    solver.set_compute_partition_on_the_fly(false);

    Vec sol[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
      ierr = VecDuplicate(rhs[dir], &sol[dir]); CHKERRXX(ierr);

    solver.solve(sol);

    if(save_voro)
    {
      char name[1000];
      sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/voro_%d.vtk", p4est->mpirank);
      solver.print_partition_VTK(name);
    }

    /* check the error */
    my_p4est_interpolation_nodes_t interp_n(&ngbd_n);
    interp_n.set_input(phi, linear);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      double *sol_p;
      ierr = VecGetArray(sol[dir], &sol_p); CHKERRXX(ierr);

      err_nm1[dir] = err_n[dir];
      err_n[dir] = 0;

      for(p4est_locidx_t f_idx=0; f_idx<faces.num_local[dir]; ++f_idx)
      {
        double x = faces.x_fr_f(f_idx, dir);
        double y = faces.y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
        double z = faces.z_fr_f(f_idx, dir);
        if(interp_n(x,y,z)<0)
          err_n[dir] = MAX(err_n[dir], fabs(sol_p[f_idx] - u_exact(x,y,z)));
#else
        if(interp_n(x,y)<0)
          err_n[dir] = MAX(err_n[dir], fabs(sol_p[f_idx] - u_exact(x,y)));
#endif
      }

      int mpiret;
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(p4est->mpicomm, "Error for direction %d : %g, order = %g\n", dir, err_n[dir], log(err_nm1[dir]/err_n[dir])/log(2)); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol[dir], &sol_p); CHKERRXX(ierr);
    }


    /* interpolate the solution on the nodes */
    Vec sol_nodes[P4EST_DIM];
    Vec err_nodes[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDuplicate(phi, &sol_nodes[dir]); CHKERRXX(ierr);
      double *sol_nodes_p;
      ierr = VecGetArray(sol_nodes[dir], &sol_nodes_p); CHKERRXX(ierr);

      ierr = VecDuplicate(phi, &err_nodes[dir]); CHKERRXX(ierr);
      double *err_p;
      ierr = VecGetArray(err_nodes[dir], &err_p); CHKERRXX(ierr);

      double *phi_p;
      ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

      err_nodes_nm1[dir] = err_nodes_n[dir];
      err_nodes_n[dir] = 0;
      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if(phi_p[n]<0)
        {
          sol_nodes_p[n] = interpolate_f_at_node_n(p4est, ghost, nodes, &faces,
                                                   &ngbd_c, &ngbd_n, sol[dir], dir, n,
                                                   face_is_well_defined[dir], bc);


          double x = node_x_fr_n(n, p4est, nodes);
          double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
          double z = node_z_fr_n(n, p4est, nodes);
          err_p[n] = fabs(sol_nodes_p[n] - u_exact(x,y,z));
          err_nodes_n[dir] = max(err_nodes_n[dir], fabs(u_exact(x,y,z) - sol_nodes_p[n]));
#else
          err_p[n] = fabs(sol_nodes_p[n] - u_exact(x,y));
          err_nodes_n[dir] = max(err_nodes_n[dir], fabs(u_exact(x,y) - sol_nodes_p[n]));
#endif
        }
      }

      ierr = VecRestoreArray(sol_nodes[dir], &sol_nodes_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(err_nodes[dir], &err_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(sol_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (sol_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);\

      ierr = VecGhostUpdateBegin(err_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (err_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      int mpiret;
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_nodes_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(p4est->mpicomm, "Error on nodes for direction %d : %g, order = %g\n", dir, err_nodes_n[dir], log(err_nodes_nm1[dir]/err_nodes_n[dir])/log(2)); CHKERRXX(ierr);
    }


    /* extrapolate the solution and check accuracy */
    my_p4est_level_set_faces_t ls_f(&ngbd_n, &faces);
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      double band = 4;
      ls_f.extend_Over_Interface(phi, sol[dir], bc[dir], dir, face_is_well_defined[dir], 2, band);

      double *sol_p;
      ierr = VecGetArray(sol[dir], &sol_p); CHKERRXX(ierr);

      err_ex_f_nm1[dir] = err_ex_f_n[dir];
      err_ex_f_n[dir] = 0;

      const PetscScalar *face_is_well_defined_p;
      ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

      for(p4est_locidx_t f_idx=0; f_idx<faces.num_local[dir]+faces.num_ghost[dir]; ++f_idx)
      {
        double x = faces.x_fr_f(f_idx, dir);
        double y = faces.y_fr_f(f_idx, dir);
#ifdef P4_TO_P8
        double z = faces.z_fr_f(f_idx, dir);
        if(!face_is_well_defined_p[f_idx] && interp_n(x,y,z)<band*MIN(dx,dy,dz))
          err_ex_f_n[dir] = MAX(err_ex_f_n[dir], fabs(sol_p[f_idx] - u_exact(x,y,z)));
#else
        if(!face_is_well_defined_p[f_idx] && interp_n(x,y)<band*MIN(dx,dy))
          err_ex_f_n[dir] = MAX(err_ex_f_n[dir], fabs(sol_p[f_idx] - u_exact(x,y)));
#endif
      }

      ierr = VecRestoreArray(sol[dir], &sol_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

      int mpiret;
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_f_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(p4est->mpicomm, "Error extrapolation for direction %d : %g, order = %g\n", dir, err_ex_f_n[dir], log(err_ex_f_nm1[dir]/err_ex_f_n[dir])/log(2)); CHKERRXX(ierr);

    }


    if(save_vtk)
    {
      double *sol_u_p;
      ierr = VecGetArray(sol[0], &sol_u_p); CHKERRXX(ierr);

      Vec um_cells, up_cells;
      ierr = VecCreateGhostCells(p4est, ghost, &um_cells); CHKERRXX(ierr);
      ierr = VecDuplicate(um_cells, &up_cells);
      double *um_cells_p;
      ierr = VecGetArray(um_cells, &um_cells_p); CHKERRXX(ierr);
      double *up_cells_p;
      ierr = VecGetArray(up_cells, &up_cells_p); CHKERRXX(ierr);
      for(p4est_locidx_t q=0; q<p4est->local_num_quadrants; ++q)
      {
        p4est_locidx_t um = faces.q2f(q, dir::f_m00);
        if(um!=-1)
        {
          double x = faces.x_fr_f(um, dir::x);
          double y = faces.y_fr_f(um, dir::x);
#ifdef P4_TO_P8
          double z = faces.z_fr_f(um, dir::x);
          if(interp_n(x,y,z)<0)
#else
          if(interp_n(x,y)<0)
#endif
          {
            um_cells_p[q] = sol_u_p[um];
#ifdef P4_TO_P8
            up_cells_p[q] = fabs(sol_u_p[um] - u_exact(x,y,z));
#else
            up_cells_p[q] = fabs(sol_u_p[um] - u_exact(x,y));
#endif
          }
        }
        else
        {
          um_cells_p[q] = 0;
          up_cells_p[q] = 0;
        }
      }
      ierr = VecGhostUpdateBegin(um_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (um_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(up_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (up_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol[0], &sol_u_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(um_cells, &um_cells_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(up_cells, &up_cells_p); CHKERRXX(ierr);

      /* END OF TESTS */

      save_VTK(p4est, ghost, nodes, &brick, phi, um_cells, up_cells, sol_nodes, err_nodes, iter);

      ierr = VecDestroy(um_cells); CHKERRXX(ierr);
      ierr = VecDestroy(up_cells); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(sol[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(sol_nodes[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(err_nodes[dir]); CHKERRXX(ierr);
    }

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}
