/*
 * Sanity and accuracy check for the navier stokes solver
 *
 * run the program with the -help flag to see the available options
 */
#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_vtk.h>
#endif


#include <src/Parser.h>

#undef MIN
#undef MAX

// --> extra info to be printed when -help is invoked
static const std::string extra_info = "\
    This program provides a general setup for testing the accuracy in inifninty norm of the \n\
    Navier-Stokes solver. \n\
    In 3D, the simulation runs with appropriate forcing terms to simulate the vortex \n\
    u = cos(x)sin(y)sin(z)  *(1 in absence of time variations, cos(t) with time variations)\n\
    v = sin(x)sin(y)sin(z)  *(1 in absence of time variations, cos(t) with time variations)\n\
    w = -2sin(x)sin(y)cos(z)*(1 in absence of time variations, cos(t) with time variations)\n\
    in the inner domain of the levelset {(x,y,z) : cos(x)*cos(y)*cos(z) + .4 < 0.0} in the domain \n\
    [PI/2, 3PI/2]^3 with Dirichlet boundary conditions for velocities and homogeneous Neumann on \n\
    the Hodge variable on the interface. \n\
    In 2D, the simulation runs with appropriate forcing terms to simulate the vortex \n\
    u =  sin(x)cos(y)*(1 in absence of time variations, cos(t) with time variations)\n\
    v = -cos(x)sin(y)*(1 in absence of time variations, cos(t) with time variations)\n\
    in the inner domain of the levelset {(x,y) : -sin(x)*sin(y) + .2 < 0.0} in the domain \n\
    [0 PI]^2 with Dirichlet boundary conditions for velocities and homogeneous Neumann on \n\
    the Hodge variable on the interface. \n\
    The program is designed to run on static grids only, and the computational grid is constructed \n\
    by refining every cell of a base grid as many times as desired (input argument 'ksplit')\n\
    Developers: cleaned up and restructured to new features' and coding standards by Raphael Egan \n\
    (raphaelegan@ucsb.edu) based on a general main file by Arthur Guittet";

#ifdef P4_TO_P8
const double xyz_m[P4EST_DIM] = {M_PI_2, M_PI_2, M_PI_2};
const double xyz_M[P4EST_DIM] = {3.0*M_PI_2, 3.0*M_PI_2, 3.0*M_PI_2};
#else
const double xyz_m[P4EST_DIM] = {0.0, 0.0};
const double xyz_M[P4EST_DIM] = {M_PI, M_PI};
#endif
const int periodic[P4EST_DIM] = {DIM(0, 0, 0)};

const int default_nx = 1;
const int default_ny = 1;
#ifdef P4_TO_P8
const int default_nz = 1;
#endif

const unsigned int default_lmin             = 3;
const unsigned int default_lmax             = 3;
const unsigned int default_ksplit           = 0;
const unsigned int default_niter_hodge_max  = 10;
const unsigned int default_sl_order         = 2;
const double default_utol                   = 1.0e-4;

const double default_rho            = 1.0;
const double default_mu             = 1.0;
const double default_tf             = M_PI/3.0;
const double default_cfl            = 1.0;
const double default_thresh         = 0.1;
const double default_uniform_band   = 5.0;
const double default_smoke_thresh   = 0.5;
const unsigned int default_vtk_n    = 1;
const std::string default_root_vtk_dir = "/home/regan/workspace/projects/ns_accuracy_check/" + std::to_string(P4EST_DIM) + "d";

class INIT_SMOKE : public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
#ifdef P4_TO_P8
    return (sqrt(SQR(x - M_PI) + SQR(y - M_PI) + SQR(z - M_PI)) < .2 ? 1.0 : 0.0);
#else
    return (sqrt(SQR(x - M_PI_2) + SQR(y - 3.0*M_PI_4)) < .2 ? 1.0 : 0.0);
#endif
  }
} init_smoke;

class BC_SMOKE : public CF_DIM
{
public:
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_smoke;

class INTERFACE_LEVEL_SET: public CF_DIM
{
public:
  INTERFACE_LEVEL_SET() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
#ifdef P4_TO_P8
    return cos(x)*cos(y)*cos(z) + .4;
#else
    return -sin(x)*sin(y) + .2;
#endif
  }
} interface_level_set;

struct BCWALLTYPE_P : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_wall_value_p;

struct BCINTERFACEVALUE_P : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} bc_interface_value_p;

struct VELOCITY_COMPONENT : CF_DIM
{
  const unsigned char dir;
  VELOCITY_COMPONENT(const unsigned char& dir_) : dir(dir_)
  {
    P4EST_ASSERT(dir < P4EST_DIM);
  }

  double vortex(DIM(double x, double y, double z)) const
  {
    switch (dir) {
    case dir::x: // u velocity
#ifdef P4_TO_P8
      return cos(x)*sin(y)*sin(z);
#else
      return sin(x)*cos(y);
#endif
    case dir::y: // v velocity
#ifdef P4_TO_P8
      return sin(x)*cos(y)*sin(z);
#else
      return -cos(x)*sin(y);
#endif
#ifdef P4_TO_P8
    case dir::z:
      return -2.0*sin(x)*sin(y)*cos(z);
#endif
    default:
      throw std::runtime_error("ANALYTICAL_SOLUTION_VELOCITY::vortex: unknown Cartesian direction");
      break;
    }
  }

  double dvortex_d(const unsigned char& der, DIM(double x, double y, double z)) const
  {
    switch (dir) {
    case dir::x: // partial derivatives of u velocity
      switch (der) {
      case dir::x:
#ifdef P4_TO_P8
        return -sin(x)*sin(y)*sin(z);
#else
        return cos(x)*cos(y);
#endif
      case dir::y:
#ifdef P4_TO_P8
        return cos(x)*cos(y)*sin(z);
#else
        return -sin(x)*sin(y);
#endif
#ifdef P4_TO_P8
      case dir::z:
        return cos(x)*sin(y)*cos(z);
#endif
      default:
        throw std::runtime_error("ANALYTICAL_SOLUTION_VELOCITY::dvortex_d: unknown Cartesian direction for partial derivative of u");
        break;
      }
    case dir::y: // partial derivatives of v velocity
      switch (der) {
      case dir::x:
#ifdef P4_TO_P8
        return cos(x)*cos(y)*sin(z);
#else
        return sin(x)*sin(y);
#endif
      case dir::y:
#ifdef P4_TO_P8
        return -sin(x)*sin(y)*sin(z);
#else
        return -cos(x)*cos(y);
#endif
#ifdef P4_TO_P8
      case dir::z:
        return sin(x)*cos(y)*cos(z);
#endif
      default:
        throw std::runtime_error("ANALYTICAL_SOLUTION_VELOCITY::dvortex_d: unknown Cartesian direction for partial derivative of v");
        break;
      }
#ifdef P4_TO_P8
    case dir::z: // partial derivatives of w velocity
      switch (der) {
      case dir::x:
        return -2.0*cos(x)*sin(y)*cos(z);
      case dir::y:
        return -2.0*sin(x)*cos(y)*cos(z);
      case dir::z:
        return +2.0*sin(x)*sin(y)*sin(z);
      default:
        throw std::runtime_error("ANALYTICAL_SOLUTION_VELOCITY::dvortex_d: unknown Cartesian direction for partial derivative of w");
        break;
      }
#endif
    default:
      throw std::runtime_error("ANALYTICAL_SOLUTION_VELOCITY::dvortex_d: unknown Cartesian direction for velocity component");
      break;
    }
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    return cos(t)*vortex(DIM(x, y, z));
  }

  double _dt(DIM(double x, double y, double z)) const
  {
    return -sin(t)*vortex(DIM(x, y, z));
  }
  double _d(const unsigned char& der, DIM(double x, double y, double z)) const
  {
    return cos(t)*dvortex_d(der, DIM(x, y, z));
  }

  double laplace(DIM(double x, double y, double z)) const
  {
    return -P4EST_DIM*cos(t)*vortex(DIM(x, y, z));
  }
};

struct EXTERNAL_FORCE_PER_UNIT_VOLUME_COMPONENT : CF_DIM
{
  const unsigned char dir;
  const double rho, mu;
  VELOCITY_COMPONENT** velocity_field;
  EXTERNAL_FORCE_PER_UNIT_VOLUME_COMPONENT(const unsigned char& dir_, const double &rho_, const double &mu_, VELOCITY_COMPONENT** analytical_solution) :
    dir(dir_), rho(rho_), mu(mu_), velocity_field(analytical_solution)
  {
    P4EST_ASSERT(dir < P4EST_DIM);
    P4EST_ASSERT(rho > 0.0 && mu > 0.0);
  }
  double operator()(DIM(double x, double y, double z)) const
  {
    return (rho*(velocity_field[dir]->_dt(DIM(x, y, z)) + SUMD((*velocity_field[0])(DIM(x, y, z))*velocity_field[dir]->_d(dir::x, DIM(x, y, z)), (*velocity_field[1])(DIM(x, y, z))*velocity_field[dir]->_d(dir::y, DIM(x, y, z)), (*velocity_field[2])(DIM(x, y, z))*velocity_field[dir]->_d(dir::z, DIM(x, y, z)))) - mu*velocity_field[dir]->laplace(DIM(x, y, z)));
  }
};

struct BCWALLTYPE_VELOCITY : WallBCDIM
{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_velocity;


void check_error_analytic_vortex(const mpi_environment_t& mpi, my_p4est_navier_stokes_t *ns, VELOCITY_COMPONENT* velocity_field[P4EST_DIM])
{
  PetscErrorCode ierr;
  int mpiret;

  p4est_t *p4est        = ns->get_p4est();
  p4est_ghost_t *ghost  = ns->get_ghost();
  p4est_nodes_t *nodes  = ns->get_nodes();
  Vec *v                = ns->get_velocity_np1();
  Vec phi_ns            = ns->get_phi();
  const double *phi_p;
  ierr = VecGetArrayRead(phi_ns, &phi_p); CHKERRXX(ierr);

  const double *v_p[P4EST_DIM];
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
    ierr = VecGetArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  double err_v[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)};
  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM]; node_xyz_fr_n(n, p4est, nodes, xyz);
    if(phi_p[n]< 0.0) // level_set(DIM(xyz[0], xyz[1], xyz[2])) < 0.0)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        err_v[dir] = MAX(err_v[dir], fabs(v_p[dir][n] - (*velocity_field[dir])(DIM(xyz[0], xyz[1], xyz[2]))));
  }

  mpiret = MPI_Allreduce(MPI_IN_PLACE, err_v, P4EST_DIM, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(v[dir], &v_p[dir]); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), "Error on velocity (at nodes) in direction %s : %.5e\n", (dir == dir::x ? "x" : ONLY3D(OPEN_PARENTHESIS dir == dir::y ?) "y" ONLY3D(: "z" CLOSE_PARENTHESIS)), err_v[dir]); CHKERRXX(ierr);
  }

  // error on hodge variable:
  double err_h = 0;
  Vec hodge = ns->get_hodge(); const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
  my_p4est_level_set_cells_t lsc(ns->get_ngbd_c(), ns->get_ngbd_n());
  const double average = lsc.integrate(phi_ns, hodge)/area_in_negative_domain(p4est, nodes, phi_ns);
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      double xyz[P4EST_DIM]; quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost ,xyz);
      if(quadrant_value_is_well_defined(ns->get_bc_hodge(), p4est, ghost, nodes, quad_idx, tree_idx, phi_p))
        err_h = MAX(err_h, fabs(hodge_p[quad_idx] - average));
    }
  }
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_h, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);

  ierr = PetscPrintf(mpi.comm(), "Error on hodge : %.5e\n", err_h); CHKERRXX(ierr);
}

void save_vtk_with_errors(const std::string vtk_file_name, my_p4est_navier_stokes_t* ns, VELOCITY_COMPONENT* velocity_field[P4EST_DIM])
{
  PetscErrorCode ierr;
  Vec phi, *vn_nodes, hodge, error_v[P4EST_DIM];
  const double *phi_p, *vn_nodes_p[P4EST_DIM], *hodge_p;
  double *error_v_p[P4EST_DIM];
  phi       = ns->get_phi();
  hodge     = ns->get_hodge();
  vn_nodes  = ns->get_velocity_np1();
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostNodes(ns->get_p4est(), ns->get_nodes(), &error_v[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vn_nodes[dir], &vn_nodes_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(error_v[dir], &error_v_p[dir]); CHKERRXX(ierr);
  }

  for (size_t k = 0; k < ns->get_nodes()->indep_nodes.elem_count; ++k) {
    double xyz_node[P4EST_DIM]; node_xyz_fr_n(k, ns->get_p4est(), ns->get_nodes(), xyz_node);
    error_v_p[0][k] = error_v_p[1][k] ONLY3D(= error_v_p[2][k]) = NAN;
    if(phi_p[k]< 0.0) // level_set(DIM(xyz[0], xyz[1], xyz[2])) < 0.0)level_set(DIM(xyz_node[0], xyz_node[1], xyz_node[2])) < 0.0)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        error_v_p[dir][k] = fabs(vn_nodes_p[dir][k] - (*velocity_field[dir])(DIM(xyz_node[0], xyz_node[1], xyz_node[2])));
  }

  my_p4est_vtk_write_all_general(ns->get_p4est(), ns->get_nodes(), ns->get_ghost(),
                                 P4EST_TRUE, P4EST_TRUE,
                                 1, /* number of VTK_NODE_SCALAR */
                                 2, /* number of VTK_NODE_VECTOR_BY_COMPONENTS */
                                 0, /* number of VTK_NODE_VECTOR_BY_BLOCK */
                                 1, /* number of VTK_CELL_SCALAR */
                                 0, /* number of VTK_CELL_VECTOR_BY_COMPONENTS */
                                 0, /* number of VTK_CELL_VECTOR_BY_BLOCK */
                                 vtk_file_name.c_str(),
                                 VTK_NODE_SCALAR, "phi", phi_p,
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "velocity", DIM(vn_nodes_p[0], vn_nodes_p[1], vn_nodes_p[2]),
      VTK_NODE_VECTOR_BY_COMPONENTS, "error_on_v", DIM(error_v_p[0], error_v_p[1], error_v_p[2]),
      VTK_CELL_SCALAR, "hodge", hodge_p);

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArray(error_v[dir], &error_v_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_nodes_p[dir]); CHKERRXX(ierr);
    ierr = VecDestroy(error_v[dir]); CHKERRXX(ierr);
  }
  ierr = PetscPrintf(ns->get_mpicomm(), "Saved visual data in ... %s\n", vtk_file_name.c_str()); CHKERRXX(ierr);
}


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("lmin", "min level of the initial tree(s) in the accuracy analysis, default is " + std::to_string(default_lmin));
  cmd.add_option("lmax", "max level of the initial tree(s) in the accuracy analysis, default is " + std::to_string(default_lmax));
  cmd.add_option("ksplit", "split step in the accuracy analysis max level of the tree, default is " + std::to_string(default_ksplit) + "\n\t (The effective levels are lmin + ksplit and lmax + ksplit)");
  cmd.add_option("nhodge", "max number of subiterations for the Hodge variable, default is " + std::to_string(default_niter_hodge_max));
  cmd.add_option("utol", "tolerance on any velocity component to reach for subiterations on Hodge variable, default is " + std::to_string(default_utol));
  cmd.add_option("nx", "the number of trees in the x direction, default is " + std::to_string(default_nx));
  cmd.add_option("ny", "the number of trees in the y direction, default is " + std::to_string(default_ny));
#ifdef P4_TO_P8
  cmd.add_option("nz", "the number of trees in the z direction, default is " + std::to_string(default_nz));
#endif
  cmd.add_option("tf", "the final time, default is " + std::to_string(default_tf));
  cmd.add_option("rho", "the mass density, default is " + std::to_string(default_rho));
  cmd.add_option("mu", "the shear viscosity, default is " + std::to_string(default_mu));
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is " + std::to_string(default_sl_order));
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx, default is " + std::to_string(default_uniform_band));
  cmd.add_option("steady", "deactivates time variations in the analytical solution if present");
  cmd.add_option("no_interface", "deactivates the existence of the interface (to check handling of wall boundary conditions)");
  cmd.add_option("cfl", "upper bound on CFL number, default is " + std::to_string(default_cfl));
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is " + std::to_string(default_thresh));
  cmd.add_option("no_print", "deactives intermediary prints (hodge convergence and progress) if present");
  cmd.add_option("save_vtk", "save the p4est in vtk format if present");
  cmd.add_option("vtk_n", "export vtk snapshots every vtk_n iterations, default is " + std::to_string(default_vtk_n));
  cmd.add_option("smoke", "activates advection of a passive scalar ('smoke') if present");
  cmd.add_option("smoke_thresh", "threshold for refinement with passive scalar (at grid construction, only), default is " + std::to_string(default_smoke_thresh));
  cmd.add_option("refine_with_smoke", "refine the grid with the smoke density and threshold smoke_thresh if present (at grid construction, only)");
  cmd.add_option("root_vtk_dir", "root directory for exportation of vtk files if not defined otherwise in the environment variable OUT_DIR,\n\
                 subfolders will be created, default is " + default_root_vtk_dir);
  if(cmd.parse(argc, argv, extra_info))
    return 0;

  const unsigned int sl_order         = cmd.get<unsigned int>("sl_order", default_sl_order); P4EST_ASSERT(sl_order == 1 || sl_order == 2);
  const unsigned int lmin             = cmd.get<unsigned int>("lmin", default_lmin);
  const unsigned int lmax             = cmd.get<unsigned int>("lmax", default_lmax); P4EST_ASSERT(lmax >= lmin);
  const unsigned int ksplit           = cmd.get<unsigned int>("ksplit", default_ksplit);
  const unsigned int niter_hodge_max  = cmd.get<unsigned int>("nhodge", default_niter_hodge_max); P4EST_ASSERT(niter_hodge_max > 0);
  const double utol                   = cmd.get<double>("utol", default_utol); P4EST_ASSERT(utol > 0.0);
  const double cfl                    = cmd.get<double>("cfl", default_cfl); P4EST_ASSERT(cfl > 0.0);
  const double thresh                 = cmd.get<double>("thresh", default_thresh); P4EST_ASSERT(thresh > 0.0);
  const bool no_print                 = cmd.contains("no_print");
  const bool save_vtk                 = cmd.contains("save_vtk");
  const unsigned int vtk_n            = cmd.get<unsigned int>("vtk_n", default_vtk_n); P4EST_ASSERT(vtk_n > 0);
  const bool with_smoke               = cmd.contains("smoke");
  const bool ref_with_smoke           = cmd.contains("refine_with_smoke"); P4EST_ASSERT(!ref_with_smoke || (ref_with_smoke && with_smoke));
  const double smoke_thresh           = cmd.get<double>("smoke_thresh", default_smoke_thresh); P4EST_ASSERT(smoke_thresh > 0.0);
  const double tf                     = cmd.get("tf", default_tf); P4EST_ASSERT(tf > 0.0);
  const int n_xyz[P4EST_DIM]          = {DIM(cmd.get<int>("nx", default_nx), cmd.get<int>("ny", default_ny), cmd.get<int>("nz", default_nz))};
  const double uniform_band           = cmd.get("uniform_band", default_uniform_band); P4EST_ASSERT(uniform_band > 0.0);
  const double mu                     = cmd.get<double>("mu", default_mu); P4EST_ASSERT(mu > 0.0);
  const double rho                    = cmd.get<double>("rho", default_rho); P4EST_ASSERT(rho > 0.0);
  const bool steady                   = cmd.contains("steady");
  const bool no_interface             = cmd.contains("no_interface");
  const std::string root_vtk_dir      = cmd.get<std::string>("root_vtk_dir", (getenv("OUT_DIR") == NULL ? default_root_vtk_dir : getenv("OUT_DIR")));

  const std::string vtk_folder = root_vtk_dir + "/vtu/analytic_vortex/" + (steady ? "without" : "with")
      + "_time_variation/macromesh_" + std::to_string(n_xyz[0]) + "_" + std::to_string(n_xyz[1]) + ONLY3D("_" + std::to_string(n_xyz[2]))
      + (no_interface ? "/no_interface" : "")
      + "/lmin_" + std::to_string(lmin) + "_lmax_" + std::to_string(lmax) + "_split_" + std::to_string(ksplit);

  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(), (std::string("Parameters : mu = %g, rho = %g, macromesh is %d x %d") + ONLY3D(std::string(" x %d") +) std::string("\n")).c_str(), mu, rho, DIM(n_xyz[0], n_xyz[1], n_xyz[2])); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi.comm(), "cfl = %g, uniform_band = %g\n", cfl, uniform_band);

  if(save_vtk && create_directory(vtk_folder, mpi.rank(), mpi.comm()))
  {
    throw std::runtime_error("main_ns_analytic_vortex: could not create exportation directory" + vtk_folder);
    return 1;
  }

  parStopWatch watch;
  watch.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(n_xyz, xyz_m, xyz_M, &brick, periodic);
  const double tree_dimensions[P4EST_DIM] = {DIM((xyz_M[0] - xyz_m[0])/n_xyz[0], (xyz_M[1] - xyz_m[1])/n_xyz[1], (xyz_M[2] - xyz_m[2])/n_xyz[2])};


  p4est_t *p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  splitting_criteria_cf_t data(lmin, lmax, &interface_level_set, 1.2);

  p4est_nm1->user_pointer = (void*)&data;
  data.min_lvl = lmin;
  data.max_lvl = lmax;
  for(unsigned int l = 0; l < lmax; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  for(unsigned l = 0; l < ksplit; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_every_cell, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  data.min_lvl += ksplit;
  data.max_lvl += ksplit;

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);
  if(third_degree_ghost_are_required(tree_dimensions))
    my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, &brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);

  if(ref_with_smoke)
  {
    splitting_criteria_thresh_t crit_thresh(lmin + ksplit, lmax + ksplit, &init_smoke, smoke_thresh);
    p4est_n->user_pointer = (void*)&crit_thresh;
    my_p4est_refine(p4est_n, P4EST_TRUE, refine_levelset_thresh, NULL);
    p4est_balance(p4est_n, P4EST_CONNECT_FULL, NULL);
  }

  p4est_n->user_pointer = (void*)&data;
  my_p4est_partition(p4est_n, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);
  if(third_degree_ghost_are_required(tree_dimensions))
    my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, &brick, ngbd_c);

  Vec phi;
  if(!no_interface)
  {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, interface_level_set, phi);

    my_p4est_level_set_t lsn(ngbd_n);
    lsn.reinitialize_1st_order_time_2nd_order_space(phi);
    lsn.perturb_level_set_function(phi, EPS);
  }

  VELOCITY_COMPONENT* velocity_field[P4EST_DIM];
  EXTERNAL_FORCE_PER_UNIT_VOLUME_COMPONENT * external_force_per_unit_volume[P4EST_DIM];
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    velocity_field[dir] = new VELOCITY_COMPONENT(dir);
    bc_v[dir].setWallTypes(bc_wall_type_velocity);  bc_v[dir].setWallValues(*velocity_field[dir]);
    if(!no_interface){
      bc_v[dir].setInterfaceType(DIRICHLET);        bc_v[dir].setInterfaceValue(*velocity_field[dir]);
    }
  }
  bc_p.setWallTypes(bc_wall_type_p);  bc_p.setWallValues(bc_wall_value_p);
  if(!no_interface){
    bc_p.setInterfaceType(NEUMANN);   bc_p.setInterfaceValue(bc_interface_value_p); }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    external_force_per_unit_volume[dir] = new EXTERNAL_FORCE_PER_UNIT_VOLUME_COMPONENT(dir, rho, mu, velocity_field);

  Vec vn_nodes_n[P4EST_DIM], vn_nodes_nm1[P4EST_DIM];
  double *vn_nodes_n_p[P4EST_DIM], *vn_nodes_nm1_p[P4EST_DIM];
  double max_mag_velocity = 0.0;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_nodes_n[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vn_nodes_nm1[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes_n[dir], &vn_nodes_n_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes_nm1[dir], &vn_nodes_nm1_p[dir]); CHKERRXX(ierr);
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    velocity_field[dir]->t = 0.0;
  for (size_t node_idx = 0; node_idx < nodes_n->indep_nodes.elem_count; ++node_idx) {
    double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
    double mag_velocity = 0.0;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      vn_nodes_n_p[dir][node_idx] = (*velocity_field[dir])(DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
      mag_velocity += SQR(vn_nodes_n_p[dir][node_idx]);
    }
    max_mag_velocity = MAX(max_mag_velocity, sqrt(mag_velocity));
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_mag_velocity, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
  double dt = (MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))/(1 << (lmax + ksplit)))*cfl/max_mag_velocity;
  if(!steady)
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      velocity_field[dir]->t = -dt;
  for (size_t node_idx = 0; node_idx < nodes_nm1->indep_nodes.elem_count; ++node_idx) {
    double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_nm1, nodes_nm1, xyz_node);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      vn_nodes_nm1_p[dir][node_idx] = (*velocity_field[dir])(DIM(xyz_node[0], xyz_node[1], xyz_node[2]));
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArray(vn_nodes_n[dir], &vn_nodes_n_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_nodes_nm1[dir], &vn_nodes_nm1_p[dir]); CHKERRXX(ierr);
  }

  my_p4est_navier_stokes_t ns(ngbd_nm1, ngbd_n, faces_n);
  if(!no_interface)
    ns.set_phi(phi);
  ns.set_parameters(mu, rho, sl_order, uniform_band, thresh, cfl);
  ns.set_dt(dt, dt);
  ns.set_velocities(vn_nodes_nm1, vn_nodes_n);
  ns.set_bc(bc_v, &bc_p);

  if(with_smoke)
  {
    Vec smoke;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    ns.set_smoke(smoke, &bc_smoke, ref_with_smoke, smoke_thresh);
  }

  double tn = 0;
  int iter = 0;

  my_p4est_poisson_cells_t* cell_solver         = NULL;
  my_p4est_poisson_faces_t* face_solver         = NULL;
  Vec dxyz_hodge_old[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateNoGhostFaces(p4est_n, faces_n, &dxyz_hodge_old[dir], dir); CHKERRXX(ierr);
  }

  while(tn + 0.01*dt < tf)
  {
    if(iter > 0)
    {
      ns.compute_dt();
      dt = ns.get_dt();

      if(tn + dt > tf)
      {
        dt = tf - tn;
        ns.set_dt(dt);
      }

      bool solvers_can_be_reused = ns.update_from_tn_to_tnp1(NULL, true, false);
      P4EST_ASSERT(solvers_can_be_reused);
      if (cell_solver != NULL && !solvers_can_be_reused){
        delete cell_solver; cell_solver = NULL; }
      if (face_solver != NULL && !solvers_can_be_reused){
        delete face_solver; face_solver = NULL; }
    }

    if(!steady)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        velocity_field[dir]->t = tn + dt;

    CF_DIM *external_forces[P4EST_DIM] = {DIM(external_force_per_unit_volume[0], external_force_per_unit_volume[1], external_force_per_unit_volume[2])};
    ns.set_external_forces(external_forces);

    double convergence_check_on_dxyz_hodge = DBL_MAX;
    unsigned int iter_hodge = 0;
    while(iter_hodge < niter_hodge_max && convergence_check_on_dxyz_hodge > utol)
    {
      ns.copy_dxyz_hodge(dxyz_hodge_old);
      ns.solve_viscosity(face_solver, (face_solver != NULL));
      convergence_check_on_dxyz_hodge = ns.solve_projection(cell_solver, (cell_solver != NULL), KSPCG, PCHYPRE, false, NULL, dxyz_hodge_old, uvw_components);
      if(!no_print){
        ierr = PetscPrintf(mpi.comm(), "hodge iteration #%d, max correction in \\nabla Hodge = %e\n", iter_hodge, convergence_check_on_dxyz_hodge); CHKERRXX(ierr);
      }

      Vec hodge_new = ns.get_hodge();
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);

      iter_hodge++;
    }

    ns.compute_velocity_at_nodes(true);
    ns.compute_pressure();

    tn += dt;

    if(!no_print){
      ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, MIN(100*tn/tf, 100.0), ns.get_max_L2_norm_u(), ns.get_p4est()->global_num_quadrants); CHKERRXX(ierr);
    }

    if(ns.get_max_L2_norm_u() > 10.0)
      break;

    if(save_vtk && iter%vtk_n == 0)
      save_vtk_with_errors((vtk_folder + "/snapshot_" + std::to_string(iter/vtk_n)).c_str(), &ns, velocity_field);

    iter++;
  }
  check_error_analytic_vortex(mpi, &ns, velocity_field);

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(dxyz_hodge_old[dir]); CHKERRXX(ierr);
    delete  velocity_field[dir];
    delete  external_force_per_unit_volume[dir];
  }

  watch.stop();
  watch.read_duration();

  return 0;
}
