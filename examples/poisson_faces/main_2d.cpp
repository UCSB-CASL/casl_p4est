
/*
 * Test the face based p4est.
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
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_faces.h>
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
#include <src/my_p4est_interpolation_faces.h>
#endif

#include <src/point3.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX


const double xmin = -1.0;
const double xmax =  1.0;
const double ymin = -1.0;
const double ymax =  1.0;
#ifdef P4_TO_P8
const double zmin = -1.0;
const double zmax =  1.0;
#endif

using namespace std;

int lmin = 2;
int lmax = 4;
int nb_splits = 4;

int nx = 1;
int ny = 1;
#ifdef P4_TO_P8
int nz = 1;
#endif

double mu = 20.;
double add_diagonal = 0.0;

/*
 * 0 - circle
 */
int interface_type = 0;

/*
 *  ********* 2D *********
 * 0 - x+y
 * 1 - x*x + y*y
 * 2 - sin(x)*cos(y)
 * 3 - sin(2.0*M_PI*x/(xmax - xmin))*cos(2.0*M_PI*y*(ymax - ymin)) to check periodic boundary conditions
 */
int test_number = 2;

BoundaryConditionType bc_wtype = NEUMANN; // DIRICHLET;
BoundaryConditionType bc_itype = DIRICHLET; // NEUMANN;

double r0 = MIN(DIM(xmax - xmin, ymax - ymin, zmax - zmin)) / 4.0;


#ifdef P4_TO_P8
static const string test_description = "choose a test.\n\
    0 - x + y + z\n\
    1 - x*x + y*y + z*z\n\
    2 - 2.0 + sin(x)*cos(y)*exp(z)\n\
    3 - sin(2.0*M_PI*x/(xmax - xmin))*cos(2.0*M_PI*y/(ymax - ymin))*sin(2.0*M_PI*z/(zmax - zmin))";
#else
static const string test_description = "choose a test.\n\
    0 - x + y\n\
    1 - x*x + y*y\n\
    2 - sin(x)*cos(y)\n\
    3 - sin(2.0*M_PI*x/(xmax - xmin))*cos(2.0*M_PI*y/(ymax - ymin))";
#endif

class LEVEL_SET: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    switch(interface_type)
    {
    case 0:
      return r0 - sqrt(SUMD(SQR(x - 0.5*(xmax + xmin)), SQR(y - 0.5*(ymax + ymin)), SQR(z - 0.5*(zmax + zmin))));
    default:
      throw std::invalid_argument("Choose a valid level set.");
    }
  }
} level_set;

class NO_INTERFACE_CF : public CF_DIM
{
public:
  double operator()(DIM(double, double, double)) const
  {
    return -1;
  }
} no_interface_cf;

double u_exact(DIM(double x, double y, double z))
{
  switch(test_number)
  {
  case 0:
    return SUMD(x, y, z);
  case 1:
    return SUMD(x*x, y*y, z*z);
  case 2:
    return MULTD(sin(x), cos(y), exp(z)) ONLY3D(+2.0);
  case 3:
    return MULTD(sin(2.0*M_PI*x/(xmax- xmin)), cos(2.0*M_PI*y/(ymax - ymin)), sin(2.0*M_PI*z/(zmax - zmin)));
  default:
    throw std::invalid_argument("Choose a valid test.");
  }
}

class BCINTERFACEVAL : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    if(bc_itype == DIRICHLET)
      return u_exact(DIM(x, y, z));
    else
    {
      double DIM(dx, dy, dz);
      switch(interface_type)
      {
      case 0:
        if(ANDD(fabs(x - 0.5*(xmax + xmin)) < EPS*(xmax - xmin), fabs(y - 0.5*(ymax + ymin)) < EPS*(ymax - ymin), fabs(z - 0.5*(zmax + zmin)) < EPS*(zmax - zmin)))
        {
          dx = dy ONLY3D( = dz) = 0.0;
        }
        else
        {
          dx = -(x - 0.5*(xmax + xmin))/sqrt(SUMD(SQR(x - 0.5*(xmax + xmin)), SQR(y - 0.5*(ymax + ymin)), SQR(z - 0.5*(zmax + zmin))));
          dy = -(y - 0.5*(ymax + ymin))/sqrt(SUMD(SQR(x - 0.5*(xmax + xmin)), SQR(y - 0.5*(ymax + ymin)), SQR(z - 0.5*(zmax + zmin))));
#ifdef P4_TO_P8
          dz = -(z - 0.5*(zmax + zmin))/sqrt(SUMD(SQR(x - 0.5*(xmax + xmin)), SQR(y - 0.5*(ymax + ymin)), SQR(z - 0.5*(zmax + zmin))));
#endif
        }
        break;
      default:
        throw std::invalid_argument("choose a valid interface type.");
      }

      switch(test_number)
      {
      case 0:
        return SUMD(dx, dy, dz);
      case 1:
        return SUMD(2.0*x*dx, 2.0*y*dy, 2.0*z*dz);
      case 2:
        return MULTD(cos(x), cos(y), exp(z))*dx - MULTD(sin(x), sin(y), exp(z))*dy ONLY3D(+ sin(x)*cos(y)*exp(z)*dz);
      case 3:
        return MULTD((2.0*M_PI/(xmax - xmin))*cos(2.0*M_PI*x/(xmax- xmin)), cos(2.0*M_PI*y/(ymax - ymin)), sin(2.0*M_PI*z/(zmax - zmin)))*dx
            + MULTD(sin(2.0*M_PI*x/(xmax- xmin)), (-2.0*M_PI/(ymax - ymin))*sin(2.0*M_PI*y/(ymax - ymin)), sin(2.0*M_PI*z/(zmax - zmin)))*dy
            ONLY3D(+ sin(2.0*M_PI*x/(xmax- xmin))*cos(2.0*M_PI*y/(ymax - ymin))*(2.0*M_PI/(zmax - zmin))*cos(2.0*M_PI*z/(zmax - zmin))*dz);
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
  }
} bc_interface_val;

class BCWALLTYPE : public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return bc_wtype;
  }
} bc_wall_type;

class BCWALLVAL : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    if(bc_wall_type(DIM(x, y, z)) == DIRICHLET)
      return u_exact(DIM(x, y, z));
    else
    {
      double dx = 0; dx = fabs(x - xmin) < (xmax - xmin)*EPS ? -1.0 : (fabs(x - xmax) < (xmax - xmin)*EPS  ? 1.0 : 0.0);
      double dy = 0; dy = fabs(y - ymin) < (ymax - ymin)*EPS ? -1.0 : (fabs(y - ymax) < (ymax - ymin)*EPS  ? 1.0 : 0.0);
#ifdef P4_TO_P8
      double dz = 0; dz = fabs(z - zmin) < (zmax - zmin)*EPS ? -1.0 : (fabs(z - zmax) < (zmax - zmin)*EPS  ? 1.0 : 0.0);
#endif
      switch(test_number)
      {
      case 0:
        return SUMD(dx, dy, dz);
      case 1:
        return SUMD(2*x*dx, 2*y*dy, 2*z*dz);
      case 2:
        return MULTD(cos(x), cos(y), exp(z))*dx - MULTD(sin(x), sin(y), exp(z))*dy ONLY3D(+ sin(x)*cos(y)*exp(z)*dz);
      case 3:
        return MULTD((2.0*M_PI/(xmax - xmin))*cos(2.0*M_PI*x/(xmax- xmin)), cos(2.0*M_PI*y/(ymax - ymin)), sin(2.0*M_PI*z/(zmax - zmin)))*dx
            + MULTD(sin(2.0*M_PI*x/(xmax- xmin)), (-2.0*M_PI/(ymax - ymin))*sin(2.0*M_PI*y/(ymax - ymin)), sin(2.0*M_PI*z/(zmax - zmin)))*dy
            ONLY3D(+ sin(2.0*M_PI*x/(xmax- xmin))*cos(2.0*M_PI*y/(ymax - ymin))*(2.0*M_PI/(zmax - zmin))*cos(2.0*M_PI*z/(zmax - zmin))*dz);
      default:
        throw std::invalid_argument("Choose a valid test.");
      }
    }
  }
} bc_wall_val;

void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              Vec phi, Vec error_cells, Vec *sol_nodes, Vec *err_nodes,
              int compt)
{
  PetscErrorCode ierr;
  const char *out_dir = getenv("OUT_DIR");
  if(!out_dir){
    out_dir = "./out_dir";
  }
  ostringstream command;
  command << "mkdir -p " << out_dir << "/vtu";
  int sys_return = system(command.str().c_str()); (void) sys_return;
  std::ostringstream oss;
  oss << out_dir << "/vtu/faces_" << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x" << brick->nxyztrees[1] << ONLY3D("x" << brick->nxyztrees[2] <<) "." << compt;

  double *phi_p, *error_cells_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(error_cells, &error_cells_p); CHKERRXX(ierr);

  double *sol_nodes_p[P4EST_DIM];
  double *err_nodes_p[P4EST_DIM];
  for(unsigned char d = 0; d < P4EST_DIM; ++d)
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
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q = 0; q < ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = p4est_quadrant_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  my_p4est_vtk_write_all_general(p4est, nodes, ghost,
                                 P4EST_TRUE, P4EST_TRUE,
                                 1, 2, 0, 1, 0, 1, oss.str().c_str(),
                                 VTK_NODE_SCALAR, "phi", phi_p,
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "solution", DIM(sol_nodes_p[0], sol_nodes_p[1], sol_nodes_p[2]),
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "errors", DIM(err_nodes_p[0], err_nodes_p[1], err_nodes_p[2]),
                                 VTK_CELL_VECTOR_BLOCK, "error_faces", error_cells_p,
                                 VTK_CELL_SCALAR, "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(error_cells, &error_cells_p); CHKERRXX(ierr);

  for(unsigned char d = 0; d < P4EST_DIM; ++d)
  {
    ierr = VecRestoreArray(sol_nodes[d], &sol_nodes_p[d]); CHKERRXX(ierr);
    ierr = VecRestoreArray(err_nodes[d], &err_nodes_p[d]); CHKERRXX(ierr);
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
  cmd.add_option("save_voro", "save the voronoi partition in vtk format");
  cmd.add_option("save_vtk", "save the p4est in vtk format");
  cmd.add_option("px", "activates periodicity along x if present (only ok with test 3)");
  cmd.add_option("py", "activates periodicity along y if present (only ok with test 3)");
#ifdef P4_TO_P8
  cmd.add_option("pz", "activates periodicity along z if present (only ok with test 3)");
#endif
  cmd.add_option("test", test_description);

  if (cmd.parse(argc, argv))
    return 0;

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
  const int     n_xyz   [P4EST_DIM] = {DIM(nx, ny, nz)};
  const double  xyz_min [P4EST_DIM] = {DIM(xmin, ymin, zmin)};
  const double  xyz_max [P4EST_DIM] = {DIM(xmax, ymax, zmax)};
  const int     periodic[P4EST_DIM] = {DIM(cmd.contains("px"), cmd.contains("py"), cmd.contains("pz"))};

  if(ORD(periodic[0], periodic[1], periodic[2]) && test_number != 3)
    throw std::invalid_argument("Periodicity can be activated only with test case 3!");

  if(ANDD(periodic[0], periodic[1], periodic[2]) && bc_itype == NEUMANN)
    add_diagonal = MAX(add_diagonal, 1.0); // to avoid nullspace...

  connectivity = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

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
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &level_set, 1.6);
    p4est->user_pointer = (void*)(&data);

    for(int l=0; l<lmax+iter; ++l)
    {
      my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
    }
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
    if(bc_itype == NOINTERFACE)
      sample_cf_on_nodes(p4est, nodes, no_interface_cf, phi);
    else
      sample_cf_on_nodes(p4est, nodes, level_set, phi);

    my_p4est_level_set_t ls(&ngbd_n);
    ls.perturb_level_set_function(phi, EPS);

    /* find dx and dy smallest */
    const double dx = (xmax - xmin)/(nx*pow(2., data.max_lvl));
    const double dy = (ymax - ymin)/(ny*pow(2., data.max_lvl));
#ifdef P4_TO_P8
    const double dz = (zmax - zmin)/(nz*pow(2., data.max_lvl));
#endif

    /* TEST THE FACES FUNCTIONS */
    my_p4est_faces_t faces(p4est, ghost, &brick, &ngbd_c);

    BoundaryConditionsDIM bc[P4EST_DIM];

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      bc[dir].setWallTypes(bc_wall_type);
      bc[dir].setWallValues(bc_wall_val);
      bc[dir].setInterfaceType(bc_itype);
      bc[dir].setInterfaceValue(bc_interface_val);
    }

    Vec rhs[P4EST_DIM];
    Vec face_is_well_defined[P4EST_DIM];
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostFaces(p4est, &faces, &rhs[dir], dir); CHKERRXX(ierr);
      double *rhs_p;
      ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

      ierr = VecDuplicate(rhs[dir], &face_is_well_defined[dir]); CHKERRXX(ierr);
      check_if_faces_are_well_defined(&ngbd_n, &faces, dir, phi, bc_itype, face_is_well_defined[dir]);

      for(p4est_locidx_t f_idx = 0; f_idx < faces.num_local[dir]; ++f_idx)
      {
        double xyz[P4EST_DIM];
        faces.xyz_fr_f(f_idx, dir, xyz);
        switch(test_number)
        {
        case 0:
          rhs_p[f_idx] = mu*0 + add_diagonal*u_exact(DIM(xyz[0], xyz[1], xyz[2]));
          break;
        case 1:
          rhs_p[f_idx] = -2.0*((double) P4EST_DIM)*mu + add_diagonal*u_exact(DIM(xyz[0], xyz[1], xyz[2]));
          break;
        case 2:
          rhs_p[f_idx] = mu*SUMD(1.0, 1.0, -1.0)*MULTD(sin(xyz[0]), cos(xyz[1]), exp(xyz[2])) + add_diagonal*u_exact(DIM(xyz[0], xyz[1], xyz[2]));
          break;
        case 3:
          rhs_p[f_idx] = mu*SUMD(SQR(2.0*M_PI/(xmax - xmin)), SQR(2.0*M_PI/(ymax - ymin)), SQR(2.0*M_PI/(zmax - zmin)))*MULTD(sin(2.0*M_PI*xyz[0]/(xmax- xmin)), cos(2.0*M_PI*xyz[1]/(ymax - ymin)), sin(2.0*M_PI*xyz[2]/(zmax - zmin))) + add_diagonal*u_exact(DIM(xyz[0], xyz[1], xyz[2]));
          break;
        default:
          throw std::invalid_argument("set rhs : unknown test number.");
        }
      }

      ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
    }

    Vec dxyz_hodge[P4EST_DIM];
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecDuplicate(rhs[dir], &dxyz_hodge[dir]); CHKERRXX(ierr);
      Vec loc;
      ierr = VecGhostGetLocalForm(dxyz_hodge[dir], &loc); CHKERRXX(ierr);
      ierr = VecSet(loc, 0); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(dxyz_hodge[dir], &loc); CHKERRXX(ierr);
    }

    my_p4est_poisson_faces_t solver(&faces, &ngbd_n);
    solver.set_phi(phi);
    solver.set_diagonal(add_diagonal);
    solver.set_mu(mu);
    solver.set_bc(bc, dxyz_hodge, face_is_well_defined);
    solver.set_rhs(rhs);
    solver.set_compute_partition_on_the_fly(false);

    Vec sol[P4EST_DIM];
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
      ierr = VecDuplicate(rhs[dir], &sol[dir]); CHKERRXX(ierr); }

    solver.solve(sol);

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
      ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr); }

    const int *matrix_has_nullspace = solver.get_matrix_has_nullspace();
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      if(matrix_has_nullspace[dir]){
        ierr = PetscPrintf(p4est->mpicomm, "Warning !! all neumann not tested for solver on faces ... missing integrations on faces.\n"); }

    if(save_voro)
    {
      char name[PATH_MAX];
      const char *out_dir = getenv("OUT_DIR");
      if(!out_dir){
        out_dir = "./out_dir";
        mkdir(out_dir, 0755);
      }
      ostringstream command;
      command << "mkdir -p " << out_dir << "/voro_grid";
      int sys_return = system(command.str().c_str()); (void) sys_return;

      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        sprintf(name, "%s/voro_grid/voro_%s_face_%d.vtk", out_dir, (dir == dir::x ? "x" : (dir == dir::y ? "y" : "z")), p4est->mpirank);
        solver.print_partition_VTK(name, dir);
      }
    }

    /* check the error */
    my_p4est_interpolation_nodes_t interp_n(&ngbd_n);
    interp_n.set_input(phi, linear);

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      double *sol_p;
      ierr = VecGetArray(sol[dir], &sol_p); CHKERRXX(ierr);

      err_nm1[dir] = err_n[dir];
      err_n[dir] = 0;

      for(p4est_locidx_t f_idx=0; f_idx<faces.num_local[dir]; ++f_idx)
      {
        double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
        if(interp_n(DIM(xyz[0], xyz[1], xyz[2]))<0)
          err_n[dir] = MAX(err_n[dir], fabs(sol_p[f_idx] - u_exact(DIM(xyz[0], xyz[1], xyz[2]))));
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      ierr = PetscPrintf(p4est->mpicomm, "Error for direction %d : %g, order = %g\n", dir, err_n[dir], log(err_nm1[dir]/err_n[dir])/log(2)); CHKERRXX(ierr);

      ierr = VecRestoreArray(sol[dir], &sol_p); CHKERRXX(ierr);
    }


    /* interpolate the solution on the nodes */
    Vec sol_nodes[P4EST_DIM];
    Vec err_nodes[P4EST_DIM];

    my_p4est_interpolation_faces_t interp_f(&ngbd_n, &faces);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM]; node_xyz_fr_n(n, p4est, nodes, xyz);
      interp_f.add_point(n, xyz);
    }

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecDuplicate(phi, &sol_nodes[dir]); CHKERRXX(ierr);
      interp_f.set_input(sol[dir], dir, 2, face_is_well_defined[dir], &bc[dir]);
      interp_f.interpolate(sol_nodes[dir]);
    }
    interp_f.clear();

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
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
        if(phi_p[n]<0)
        {
          double xyz[P4EST_DIM]; node_xyz_fr_n(n, p4est, nodes, xyz);
          err_p[n] = fabs(sol_nodes_p[n] - u_exact(DIM(xyz[0], xyz[1], xyz[2])));
          err_nodes_n[dir] = max(err_nodes_n[dir], fabs(u_exact(DIM(xyz[0], xyz[1], xyz[2])) - sol_nodes_p[n]));
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
    if(bc_itype != NOINTERFACE)
    {
      my_p4est_level_set_faces_t ls_f(&ngbd_n, &faces);
      for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      {
        double band = 4;
        ls_f.extend_Over_Interface(phi, sol[dir], bc[dir], dir, face_is_well_defined[dir], NULL, 2, band);

        double *sol_p;
        ierr = VecGetArray(sol[dir], &sol_p); CHKERRXX(ierr);

        err_ex_f_nm1[dir] = err_ex_f_n[dir];
        err_ex_f_n[dir] = 0;

        const PetscScalar *face_is_well_defined_p;
        ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

        for(p4est_locidx_t f_idx=0; f_idx<faces.num_local[dir]+faces.num_ghost[dir]; ++f_idx)
        {
          double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
          if(!face_is_well_defined_p[f_idx] && interp_n(DIM(xyz[0], xyz[1], xyz[2])) < band*MIN(DIM(dx,dy,dz)))
            err_ex_f_n[dir] = MAX(err_ex_f_n[dir], fabs(sol_p[f_idx] - u_exact(DIM(xyz[0], xyz[1], xyz[2]))));
        }

        ierr = VecRestoreArray(sol[dir], &sol_p); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

        int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_ex_f_n[dir], 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
        ierr = PetscPrintf(p4est->mpicomm, "Error extrapolation for direction %d : %g, order = %g\n", dir, err_ex_f_n[dir], log(err_ex_f_nm1[dir]/err_ex_f_n[dir])/log(2)); CHKERRXX(ierr);
      }
    }


    if(save_vtk)
    {

      Vec error_cells;
      double *error_cells_p;
      ierr = VecCreateGhostCellsBlock(p4est, ghost, P4EST_DIM, &error_cells); CHKERRXX(ierr);
      ierr = VecGetArray(error_cells, &error_cells_p); CHKERRXX(ierr);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        const double *sol_p;
        ierr = VecGetArrayRead(sol[dir], &sol_p); CHKERRXX(ierr);
        for (p4est_topidx_t tr_idx = p4est->first_local_tree; tr_idx <= p4est->last_local_tree ; ++tr_idx) {
          const p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tr_idx);
          for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
            p4est_locidx_t quad_idx = q + tree->quadrants_offset;
            error_cells_p[P4EST_DIM*q + dir] = 0.0;
            // negative direction
            p4est_locidx_t f_idx = faces.q2f(quad_idx, 2*dir);
            if(f_idx!=-1)
            {
              double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
              if(interp_n(DIM(xyz[0], xyz[1], xyz[2])) < 0.0)
                error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - u_exact(DIM(xyz[0], xyz[1], xyz[2]))));
            }
            else
            {
              set_of_neighboring_quadrants ngbd; ngbd.clear();
              ngbd_c.find_neighbor_cells_of_cell(ngbd, quad_idx, tr_idx, DIM(dir == dir::x ? -1 : 0, dir == dir::y ? -1 : 0, dir == dir::z ? -1 : 0));
              for (set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end() ; ++it) {
                f_idx = faces.q2f(it->p.piggy3.local_num, 2*dir+1);
                double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
                if(interp_n(DIM(xyz[0], xyz[1], xyz[2])) < 0.0)
                  error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - u_exact(DIM(xyz[0], xyz[1], xyz[2]))));
              }
            }
            // positive direction
            f_idx = faces.q2f(quad_idx, 2*dir+1);
            if(f_idx!=-1)
            {
              double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
              if(interp_n(DIM(xyz[0], xyz[1], xyz[2])) < 0.0)
                error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - u_exact(DIM(xyz[0], xyz[1], xyz[2]))));
            }
            else
            {
              set_of_neighboring_quadrants ngbd; ngbd.clear();
              ngbd_c.find_neighbor_cells_of_cell(ngbd, quad_idx, tr_idx, DIM(dir == dir::x ? +1 : 0, dir == dir::y ? +1 : 0, dir == dir::z ? +1 : 0));
              for (set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end() ; ++it) {
                f_idx = faces.q2f(it->p.piggy3.local_num, 2*dir);
                double xyz[P4EST_DIM]; faces.xyz_fr_f(f_idx, dir, xyz);
                if(interp_n(DIM(xyz[0], xyz[1], xyz[2])) < 0.0)
                  error_cells_p[P4EST_DIM*q + dir] = MAX(error_cells_p[P4EST_DIM*q + dir], fabs(sol_p[f_idx] - u_exact(DIM(xyz[0], xyz[1], xyz[2]))));
              }
            }
          }
        }
        ierr = VecRestoreArrayRead(sol[dir], &sol_p); CHKERRXX(ierr);
      }
      ierr = VecGhostUpdateBegin(error_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (error_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(error_cells, &error_cells_p); CHKERRXX(ierr);

      /* END OF TESTS */

      save_VTK(p4est, ghost, nodes, &brick, phi, error_cells, sol_nodes, err_nodes, iter);

      ierr = VecDestroy(error_cells); CHKERRXX(ierr);
    }

    ierr = VecDestroy(phi); CHKERRXX(ierr);

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
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
