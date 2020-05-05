/*
 * Application of the navier-stokes solver for 2D cavity flow
 *
 * run the program with the -help flag to see the available options
 */
#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_vtk.h>
#endif

#include <iomanip>
#include <src/Parser.h>

#undef MIN
#undef MAX

#ifndef P4_TO_P8
// --> extra info to be printed when -help is invoked
static const std::string extra_info = "\
    This program provides a general setup for running the Navier-Stokes solver for the canonical \n\
    cavity flow, in 2D only. \n\
    A solid interface ('hole') can be added in the center of the computational domain. \n\
    Dirichlet boundary conditions for velocities and homogeneous Neumann on the Hodge variable are used \n\
    for all boundary conditions. \n\
    Developer: cleaned up and restructured to new features' and coding standards by Raphael Egan \n\
    (raphaelegan@ucsb.edu) based on a general main file by Arthur Guittet";

#if defined(LAPTOP)
const std::string default_export_dir  = "/home/raphael/workspace/projects/cavity_flow";
#else
const std::string default_export_dir  = "/home/regan/workspace/projects/cavity_flow";
#endif

const double default_side_length  = 1.0;
const double default_top_velocity = 1.0;
const int default_nx = 2;
const int default_ny = 2;

const unsigned int default_lmin             = 3;
const unsigned int default_lmax             = 3;
const unsigned int default_niter_hodge_max  = 10;
const hodge_control def_hodge_control       = uvw_components;
const unsigned int default_sl_order         = 2;
const double default_hodge_tol              = 1.0e-4;
const double default_hole_radius            = 0.25;
const double default_tf                     = 37.0;
const double default_Re                     = 1000.0;
const double default_rho                    = 1.0;

const std::string default_pc_cell             = "sor";
const std::string default_cell_solver         = "bicgstab";
const std::string default_pc_face             = "sor";

const double default_cfl            = 1.0;
const double default_thresh         = 0.1;
const double default_uniform_band   = 5.0;
const double default_smoke_thresh   = 0.5;
const double default_vtk_dt         = 0.2;
const std::string default_root_vtk_dir = "/home/regan/workspace/projects/cavity_flow/" + std::to_string(P4EST_DIM) + "d";

class INIT_SMOKE : public CF_2
{
  const my_p4est_brick_t *macromesh;
public:
  INIT_SMOKE(const my_p4est_brick_t *macromesh_) : macromesh(macromesh_) {}
  double operator()(double x, double y) const
  {
    return (sqrt(SQR(x - (0.5*macromesh->xyz_min[0] + 0.5*macromesh->xyz_max[0])) + SQR(y - (0.85*macromesh->xyz_max[1] + 0.15*macromesh->xyz_min[0]))) < .05 ? 1.0 : 0.0);
  }
};

class BC_SMOKE : public CF_2
{
public:
  double operator()(double, double) const
  {
    return 0.0;
  }
} bc_smoke;

class INTERFACE_LEVEL_SET: public CF_2
{
  const double hole_radius;
  const my_p4est_brick_t *macromesh;
public:
  INTERFACE_LEVEL_SET(const double &hole_radius_, const my_p4est_brick_t *macromesh_) : hole_radius(hole_radius_), macromesh(macromesh_)
  {
    lip = 1.2;
    if(hole_radius <= 0.0)
      throw std::invalid_argument("INTERFACE_LEVEL_SET::INTERFACE_LEVEL_SET() : the radius of the hole must be strictly positive");
  }

  double operator()(double x, double y) const
  {
    return hole_radius - sqrt(SQR(x - (0.5*macromesh->xyz_min[0] + 0.5*macromesh->xyz_max[0])) + SQR(y - (0.5*macromesh->xyz_min[1] + 0.5*macromesh->xyz_max[1])));
  }
};

class NOINTERFACE_LEVEL_SET: public CF_2
{
public:
  double operator()(double, double) const
  {
    return -1.0;
  }
};

struct BCWALLTYPE_P : WallBC2D
{
  BoundaryConditionType operator()(double, double) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLTYPE_VELOCITY : WallBC2D
{
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_velocity;

struct BCWALLVALUE_U : CF_2
{
  const double top_wall_velocity;
  const my_p4est_brick_t *macromesh;
  BCWALLVALUE_U(const double &top_wall_velocity_, const my_p4est_brick_t *macromesh_) : top_wall_velocity(top_wall_velocity_), macromesh(macromesh_) {}

  double operator()(double, double y) const
  {
    return (fabs(y - macromesh->xyz_max[1]) < EPS*(macromesh->xyz_max[1] - macromesh->xyz_min[1]) ? top_wall_velocity : 0.0);
  }
};

double Reynolds(const BCWALLVALUE_U &bc_wall_u, const my_p4est_navier_stokes_t *ns)
{
  return bc_wall_u.top_wall_velocity*ns->get_length_of_domain()/ns->get_nu();
}

void export_velocity_cavity(const std::string &export_dir, my_p4est_navier_stokes_t *ns, const BCWALLVALUE_U &bc_wall_u)
{
  PetscErrorCode ierr;

  const my_p4est_node_neighbors_t *ngbd_n = ns->get_ngbd_n();
  const Vec *vn = ns->get_velocity_np1();
  const p4est_t *p4est = ns->get_p4est();
  Vec phi = ns->get_phi();

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  my_p4est_interpolation_nodes_t interp0(ngbd_n);
  my_p4est_interpolation_nodes_t interp1(ngbd_n);
  my_p4est_interpolation_nodes_t interp0_phi(ngbd_n);
  my_p4est_interpolation_nodes_t interp1_phi(ngbd_n);
  const my_p4est_brick_t *brick = ns->get_brick();
  int N = 200;
  for(int i = 0; i <= N; ++i)
  {
    double xyz0[] = { brick->xyz_min[0] + (brick->xyz_max[0] - brick->xyz_min[0])*(double)i/(double)N, .5*brick->xyz_min[1] + .5*brick->xyz_max[1] };
    interp0.add_point(i, xyz0);
    interp0_phi.add_point(i, xyz0);

    double xyz1[] = { .5*brick->xyz_min[0] + .5*brick->xyz_max[0], brick->xyz_min[0] + (brick->xyz_max[0] - brick->xyz_min[0])*(double)i/(double)N  };
    interp1.add_point(i, xyz1);
    interp1_phi.add_point(i, xyz1);
  }

  std::vector<double> v0(N + 1);
  interp0.set_input(vn[1], quadratic);
  interp0.interpolate(v0.data());

  std::vector<double> v1(N + 1);
  interp1.set_input(vn[0], quadratic);
  interp1.interpolate(v1.data());

  std::vector<double> phi0(N + 1);
  interp1.set_input(phi, quadratic);
  interp1.interpolate(phi0.data());

  std::vector<double> phi1(N + 1);
  interp1.set_input(phi, quadratic);
  interp1.interpolate(phi1.data());

  if(!ns->get_mpirank())
  {
    FILE* fp;
    std::ostringstream filename;
    filename << std::fixed << std::setprecision(2);
    filename << "velocity_" << data->min_lvl << "-" << data->max_lvl << "_" << brick->nxyztrees[0] << "x" << brick->nxyztrees[1]
        << "_Re_" << Reynolds(bc_wall_u, ns) << "_cfl_" << ns->get_cfl() << ".dat";
    fp = fopen((export_dir + "/" + filename.str()).c_str(), "w");

    if(fp == NULL)
      throw std::invalid_argument("export_velocity_cavity: could not open file.");

    ierr = PetscFPrintf(ns->get_mpicomm(), fp, "%% normalized x/y \t vx \t vy\n"); CHKERRXX(ierr);
    for(int i = 0; i <= N; ++i)
    {
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "%g, %g, %g\n", (double)i/(double)N, phi0[i] < 0.0 ? v0[i] : 0, phi1[i]<0 ? v1[i] : 0); CHKERRXX(ierr);
    }

    fclose(fp);


    const std::string data_file = filename.str();
    filename.str("");
    filename << "velocity_profiles.gnu";
    if(!file_exists(export_dir + "/" + filename.str()))
    {
      FILE* gnuplot_script_fp = fopen((export_dir + "/" + filename.str()).c_str(), "w");

      if(gnuplot_script_fp == NULL)
        throw std::invalid_argument("export_velocity_cavity: could not open gnuplot script file.");

      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set term wxt noraise 0\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set key top right Left font \"Arial,14\"\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set xlabel \"x\" font \"Arial,14\"\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set ylabel \"v\" font \"Arial,14\"\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set xrange [0.0 : 1.0]\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set yrange [%g : %g]\n", (Reynolds(bc_wall_u, ns) <= 1500.0 ? -0.6 : -0.8), (Reynolds(bc_wall_u, ns) < 1500.0 ? 0.4 : 0.6)); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "plot\t \"%s\" using 1:2 with lines lw 3 notitle\n\n", data_file.c_str()); CHKERRXX(ierr);

      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set term wxt noraise 1\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set key top right Left font \"Arial,14\"\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set xlabel \"u\" font \"Arial,14\"\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set ylabel \"y\" font \"Arial,14\"\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set xrange [%g : %g]\n", (Reynolds(bc_wall_u, ns) <= 1500.0 ? -0.4 : -0.5), 1.0); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "set yrange [0.0 : 1.0]\n"); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "plot\t \"%s\" using 3:1 with lines lw 3 notitle\n\n", data_file.c_str()); CHKERRXX(ierr);

      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "pause 2 \n", filename.str().c_str()); CHKERRXX(ierr);
      ierr = PetscFPrintf(ns->get_mpicomm(), fp, "reread", filename.str().c_str()); CHKERRXX(ierr);

      fclose(gnuplot_script_fp);
    }
  }
}
#endif

void create_macromesh(const cmdParser &cmd, my_p4est_brick_t *&brick, p4est_connectivity_t *&conn)
{
  // build the macromesh first
  const double side_length = cmd.get<double>("length", default_side_length);
  if(cmd.contains("hole_radius") && cmd.get<double>("hole_radius", default_hole_radius) > side_length/2.0)
    throw std::invalid_argument("create_macromesh: the radius of the hole is larger than half the length of the domain, please correct that.");
  const double xyz_min_[2] = {0.0, 0.0};
  const double xyz_max_[2] = {side_length, side_length};
  const int n_tree_x = cmd.get<int>("nx", default_nx);
  const int n_tree_y = cmd.get<int>("ny", default_ny);
  const int n_tree_xyz[2] = {n_tree_x, n_tree_y};
  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid); brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;
  const int periodic[2] = {0, 0};
  if(conn != NULL)
    p4est_connectivity_destroy(conn);
  conn = my_p4est_brick_new(n_tree_xyz, xyz_min_, xyz_max_, brick, periodic);
  return;
}

my_p4est_navier_stokes_t* create_ns_solver(const mpi_environment_t &mpi, const cmdParser &cmd, my_p4est_brick_t *brick, p4est_connectivity_t *connectivity,
                                           const splitting_criteria_cf_and_uniform_band_t *data, BoundaryConditions2D bc_v[2], BoundaryConditions2D &bc_p)
{
  p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(int l = 0; l < data->max_lvl; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = (void*) data;
  const bool with_smoke         = cmd.contains("smoke");
  const bool refine_with_smoke  = cmd.contains("refine_with_smoke");
  const double smoke_thresh     = cmd.get<double>("smoke_thresh", default_smoke_thresh);
  INIT_SMOKE init_smoke(brick);
  if(with_smoke && refine_with_smoke)
  {
    splitting_criteria_thresh_t crit_thresh(data->min_lvl, data->max_lvl, &init_smoke, smoke_thresh);
    p4est_n->user_pointer = (void*) &crit_thresh;
    my_p4est_refine(p4est_n, P4EST_TRUE, refine_levelset_thresh, NULL);
    p4est_balance(p4est_n, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
    p4est_n->user_pointer = (void*) data;
  }

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c);

  Vec node_sampled_phi;
  PetscErrorCode ierr = VecCreateGhostNodes(p4est_n, nodes_n, &node_sampled_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *data->phi, node_sampled_phi);
  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_1st_order_time_2nd_order_space(node_sampled_phi);
  lsn.perturb_level_set_function(node_sampled_phi, EPS);


  my_p4est_navier_stokes_t *ns = new my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n);
  ns->set_phi(node_sampled_phi);
  if(with_smoke)
  {
    Vec smoke;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    ns->set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }

  const double rho            = cmd.get("mass_density", default_rho);
  const double u_top          = bc_v[0].wallValue((0.5*brick->xyz_min[0] + 0.5*brick->xyz_max[0]), brick->xyz_max[1]);
  const double cavity_length  = ns->get_length_of_domain();
  ns->set_parameters(rho*u_top*cavity_length/cmd.get<double>("Re", default_Re),
                     rho,
                     cmd.get<int>("sl_order", default_sl_order),
                     data->uniform_band,
                     cmd.get<double>("thresh", default_thresh),
                     cmd.get<double>("cfl", default_cfl));

  CF_2 *initial_velocity[2];
  for (unsigned char dim = 0; dim < 2; ++dim)
    initial_velocity[dim] = &zero_cf;
  ns->set_velocities(initial_velocity, initial_velocity);

  ns->set_bc(bc_v, &bc_p);
  return ns;
}

void get_solver_and_pc_type(const mpi_environment_t &mpi, const cmdParser &cmd, KSPType &cell_solver_type, PCType &pc_cell, PCType &pc_face)
{
  const std::string des_pc_cell = cmd.get<std::string>("pc_cell", default_pc_cell);
  const std::string des_solver_cell = cmd.get<std::string>("cell_solver", default_cell_solver);
  const std::string des_pc_face = cmd.get<std::string>("pc_face", default_pc_face);
  if (des_pc_cell.compare("hypre") == 0)
    pc_cell = PCHYPRE;
  else if (des_pc_cell.compare("jacobi'") == 0)
    pc_cell = PCJACOBI;
  else
  {
    if (des_pc_cell.compare("sor") != 0 && mpi.rank() == 0)
      std::cerr << "The desired preconditioner for the cell-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
    pc_cell = PCSOR;
  }
  if (des_solver_cell.compare("cg") == 0)
    cell_solver_type = KSPCG;
  else
  {
    if (des_solver_cell.compare("bicgstab") != 0 && mpi.rank() == 0)
      std::cerr << "The desired Krylov solver for the cell-solver was either not allowed or not correctly understood. BiCGStab is used instead" << std::endl;
    cell_solver_type = KSPBCGS;
  }
  if (des_pc_face.compare("hypre") == 0)
    pc_face = PCHYPRE;
  else if (des_pc_face.compare("jacobi'") == 0)
    pc_face = PCJACOBI;
  else
  {
    if (des_pc_face.compare("sor") != 0 && mpi.rank() == 0)
      std::cerr << "The desired preconditioner for the face-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
    pc_face = PCSOR;
  }
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

#ifdef P4_TO_P8
  std::cerr << "This program is not developed for 3D application, implemented for 2D run and validation only..." << std::endl;
  return 1;
#else

  cmdParser cmd;
  // computational grid parameters
  cmd.add_option("lmin",                  "min level of the trees, default is " + std::to_string(default_lmin));
  cmd.add_option("lmax",                  "max level of the trees, default is " + std::to_string(default_lmax));
  cmd.add_option("thresh",                "the threshold used for the refinement criteria, default is " + std::to_string(default_thresh));
  cmd.add_option("uniform_band",          "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is " + std::to_string(default_uniform_band));
  cmd.add_option("nx",                    "number of trees in the macromesh along x. The default value is " + std::to_string(default_nx));
  cmd.add_option("ny",                    "number of trees in the macromesh along y. The default value is " + std::to_string(default_ny));
  cmd.add_option("smoke_thresh",          "threshold for smoke refinement, default is " + std::to_string(default_smoke_thresh));
  cmd.add_option("refine_with_smoke",     "refine the grid with the smoke density and threshold smoke_thresh if present");
  // physical parameters for the simulations
  cmd.add_option("duration",              "the duration of the simulation. tstart = 0.0, default duration is " + std::to_string(default_tf));
  cmd.add_option("Re",                    "the Reynolds number = rho*u_top*L/mu, default is " + std::to_string(default_Re));
  cmd.add_option("smoke",                 "no smoke if option not present, with smoke if option present");
  cmd.add_option("length",                "side length of the cavity, default is " + std::to_string(default_side_length));
  cmd.add_option("hole_radius",           "if defined, adds a hole (solid, non-penetrable interface) in the center of the domain. The radius of the hole can be specified by the user. Default value is " + std::to_string(default_hole_radius));
  cmd.add_option("top_velocity",          "velocity of the top wall of the cavity. Default value is "  + std::to_string(default_top_velocity));
  cmd.add_option("mass_density",          "mass density of the fluid. Default value is "  + std::to_string(default_rho));
  // method-related parameters
  cmd.add_option("sl_order",              "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is " + std::to_string(default_sl_order));
  cmd.add_option("cfl",                   "dt = cfl * dx/vmax, default is " + std::to_string(default_cfl));
  cmd.add_option("hodge_tol",             "numerical tolerance used for the convergence criterion on the Hodge variable (or its gradient), at all time steps.\n\
                 Default is " + std::to_string(default_hodge_tol) + ": w.r.t absolute value of hodge if checking against 'value'; relative to top wall velocity otherwise)");
  cmd.add_option("niter_hodge",           "max number of iterations for convergence of the Hodge variable, at all time steps, default is " + std::to_string(default_niter_hodge_max));
  cmd.add_option("hodge_control",         "type of convergence check used for inner loops, i.e. convergence criterion on the Hodge variable. \n\
                 Possible values are 'u', 'v' (, 'w'), 'uvw' (for gradient components) or 'value' (for local values of Hodge), default is " + convert_to_string(def_hodge_control));
  cmd.add_option("pc_cell",               "preconditioner for cell-solver: jacobi, sor or hypre, default is " + default_pc_cell);
  cmd.add_option("cell_solver",           "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is " + default_cell_solver);
  cmd.add_option("pc_face",               "preconditioner for face-solver: jacobi, sor or hypre, default is " + default_pc_face);
  // output-control parameters
  cmd.add_option("export_folder",         "exportation_folder if not defined otherwise in the environment variable OUT_DIR,\n\
                 subfolder(s) will be created, default is " + default_export_dir);
  cmd.add_option("save_vtk",              "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt",                "export vtk files every vtk_dt time lapse. If not specified, default is " + std::to_string(default_vtk_dt));
  cmd.add_option("timing",                "if defined, activates the internal timer and prints final information .");

  if(cmd.parse(argc, argv, extra_info))
    return 0;

  PetscErrorCode ierr;

  // create macromesh
  my_p4est_brick_t *brick = NULL;
  p4est_connectivity_t *connectivity = NULL;
  create_macromesh(cmd, brick, connectivity);

  // create (possible) interface levelset
  CF_DIM *levelset = NULL;
  const bool with_hole = cmd.contains("hole_radius");
  if(with_hole)
    levelset = new INTERFACE_LEVEL_SET(cmd.get<double>("hole_radius", default_hole_radius), brick);
  else
    levelset = new NOINTERFACE_LEVEL_SET;

  // Create boundary conditions for velocity
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BCWALLVALUE_U bc_wall_value_u(cmd.get<double>("top_velocity", default_top_velocity), brick);
  bc_v[0].setWallTypes(bc_wall_type_velocity);  bc_v[1].setWallTypes(bc_wall_type_velocity);
  bc_v[0].setWallValues(bc_wall_value_u);       bc_v[1].setWallValues(zero_cf);
  if(with_hole)
  {
    bc_v[0].setInterfaceType(DIRICHLET);        bc_v[1].setInterfaceType(DIRICHLET);
    bc_v[0].setInterfaceValue(zero_cf);         bc_v[1].setInterfaceValue(zero_cf);
  }
  // Create boundary condition for pressure (homogeneous Neumann everywhere...)
  BoundaryConditionsDIM bc_p;
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(zero_cf);
  if(with_hole)
  {
    bc_p.setInterfaceType(NEUMANN); bc_p.setInterfaceValue(zero_cf);
  }

  // create refinement criteria
  splitting_criteria_cf_and_uniform_band_t data(cmd.get<int>("lmin", default_lmin), cmd.get<int>("lmax", default_lmax),
                                                levelset, cmd.get<double>("uniform_band", default_uniform_band));
  // create N-S solver and refinement criteria
  my_p4est_navier_stokes_t *ns = create_ns_solver(mpi, cmd, brick, connectivity, &data, bc_v, bc_p);

  // intialize simulation and exportation(s)
  double tn = 0.0; // we start from 0.0
  const double duration = cmd.get<double>("duration", default_tf);
  double vtk_dt = DBL_MAX;
  const bool save_vtk = cmd.contains("save_vtk");

  if (save_vtk)
  {
    vtk_dt = cmd.get<double>("vtk_dt", default_vtk_dt);
    if (vtk_dt <= 0.0)
      throw std::invalid_argument("cavity_flow::main: the value of vtk_dt must be strictly positive.");
  }
  const double dxmin = MIN(ns->get_length_of_domain()/brick->nxyztrees[0], ns->get_height_of_domain()/brick->nxyztrees[1])/((double) (1 << data.max_lvl));
  double dt = MIN(dxmin*ns->get_cfl()/bc_wall_value_u.top_wall_velocity, vtk_dt, duration);
  ns->set_dt(dt);

  std::map<ns_task, double> global_computational_times;
  if(cmd.contains("timing"))
  {
    ns->activate_timer();
    global_computational_times[grid_update] = global_computational_times[viscous_step] = global_computational_times[projection_step] = global_computational_times[velocity_interpolation] = 0.0;
  }

  const std::string export_root = cmd.get<std::string>("export_folder", (getenv("OUT_DIR") == NULL ? default_export_dir : getenv("OUT_DIR")));
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << export_root << "/" << (with_hole ? "with" : "without") << "_hole/Re_" << Reynolds(bc_wall_value_u, ns) << "_cfl_" << ns->get_cfl() << "/nx_" << brick->nxyztrees[0]
      << "_ny_" << brick->nxyztrees[1] << "_lmin_" << data.min_lvl << "_lmax_" << data.max_lvl;
  const std::string export_dir  = oss.str();
  const std::string vtk_path    = export_dir + "/vtu";
  if(create_directory(export_dir, ns->get_mpirank(), ns->get_mpicomm()))
    throw std::runtime_error("cavity_flow::main: could not create exportation directory " + export_dir);
  if(save_vtk)
  {
    if(save_vtk && create_directory(vtk_path, ns->get_mpirank(), ns->get_mpicomm()))
      throw std::runtime_error("cavity_flow::main: could not create exportation directory for vtk files " + vtk_path);
  }

  oss.str("");
  oss << std::scientific << std::setprecision(2);
  oss << "Parameters : Re = " << Reynolds(bc_wall_value_u, ns) << ", mu = " << ns->get_mu() << ", rho = " << ns->get_rho() << ", macromesh is "
      << ns->get_brick()->nxyztrees[0] << " X " << ns->get_brick()->nxyztrees[1] << "\n";
  ierr = PetscPrintf(ns->get_mpicomm(), oss.str().c_str()); CHKERRXX(ierr);
  ierr = PetscPrintf(ns->get_mpicomm(), "cfl = %g, uniform_band = %g\n", ns->get_cfl(), ns->get_uniform_band());


  my_p4est_poisson_cells_t* cell_solver = NULL;
  my_p4est_poisson_faces_t* face_solver = NULL;
  Vec *dxyz_hodge_old = NULL, hold_old = NULL;
  const KSPType face_solver_type = KSPBCGS; // the only valid one
  KSPType cell_solver_type;
  PCType pc_cell, pc_face;
  get_solver_and_pc_type(mpi, cmd, cell_solver_type, pc_cell, pc_face);

  const hodge_control control_hodge = cmd.get<hodge_control>("hodge_control", def_hodge_control);
  const unsigned int niter_hodge_max = cmd.get<unsigned int>("niter_hodge", default_niter_hodge_max);
  const double hodge_tol = cmd.get<double>("hodge_tol", default_hodge_tol)*(control_hodge == hodge_value ? 1.0 : bc_wall_value_u.top_wall_velocity);
  if(control_hodge != hodge_value)
    dxyz_hodge_old = new Vec[2];
  unsigned int iter = 0;
  int vtk_iter = -1;
  while(tn + 0.01*dt < duration)
  {
    if(iter > 0)
    {
      ns->compute_dt();
      dt = ns->get_dt();

      if(tn + dt > duration)
      {
        dt = duration - tn;
        ns->set_dt(dt);
      }

      if(save_vtk && dt > vtk_dt)
      {
        dt = vtk_dt; // so that we don't miss snapshots...
        ns->set_dt(dt);
      }

      bool solvers_can_be_reused = ns->update_from_tn_to_tnp1(levelset);

      if(cell_solver != NULL && !solvers_can_be_reused){
        delete  cell_solver; cell_solver = NULL; }
      if(face_solver != NULL && !solvers_can_be_reused){
        delete  face_solver; face_solver = NULL; }

    }

    double convergence_check_on_hodge = DBL_MAX;
    if(control_hodge == hodge_value) {
      ierr = VecCreateNoGhostCells(ns->get_p4est(), &hold_old); CHKERRXX(ierr); }
    else
      for (unsigned char dir = 0; dir < 2; ++dir){
        ierr = VecCreateNoGhostFaces(ns->get_p4est(), ns->get_faces(), &dxyz_hodge_old[dir], dir); CHKERRXX(ierr); }

    unsigned int iter_hodge = 0;
    while(iter_hodge < niter_hodge_max && convergence_check_on_hodge > hodge_tol)
    {
      if(control_hodge == hodge_value)
        ns->copy_hodge(hold_old, false);
      else
        ns->copy_dxyz_hodge(dxyz_hodge_old);

      ns->solve_viscosity(face_solver, (face_solver != NULL), face_solver_type, pc_face);
      convergence_check_on_hodge = ns->solve_projection(cell_solver, (cell_solver != NULL), cell_solver_type, pc_cell, true, hold_old, dxyz_hodge_old, control_hodge);

      if(control_hodge == hodge_value){
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in Hodge = %e\n", iter_hodge, convergence_check_on_hodge); CHKERRXX(ierr);
      } else if(control_hodge == uvw_components){
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in \\nabla Hodge = %e\n", iter_hodge, convergence_check_on_hodge); CHKERRXX(ierr);
      } else {
        ierr = PetscPrintf(ns->get_mpicomm(), "hodge iteration #%d, max correction in d(Hodge)/d%s = %e\n", iter_hodge, (control_hodge == u_component ? "x" : "y"), convergence_check_on_hodge); CHKERRXX(ierr);
      }

      iter_hodge++;
    }
    if(control_hodge == hodge_value) {
      ierr = VecDestroy(hold_old); CHKERRXX(ierr); }
    else
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir){
        ierr = VecDestroy(dxyz_hodge_old[dir]); CHKERRXX(ierr); }

    ns->compute_velocity_at_nodes();
    ns->compute_pressure();

    tn += dt;

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*tn/duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(cmd.contains("timing"))
    {
      const std::map<ns_task, execution_time_accumulator>& timings = ns->get_timings();
      P4EST_ASSERT(timings.find(projection_step) != timings.end() && timings.find(viscous_step) != timings.end() && timings.find(velocity_interpolation) != timings.end()); // should *always* find those
      P4EST_ASSERT(timings.at(projection_step).read_counter() == timings.at(viscous_step).read_counter()); // the number of subiterations should match

      if(timings.find(grid_update) != timings.end())
        global_computational_times[grid_update]           += timings.at(grid_update).read_total_time();
      global_computational_times[viscous_step]            += timings.at(viscous_step).read_total_time();
      global_computational_times[projection_step]         += timings.at(projection_step).read_total_time();
      global_computational_times[velocity_interpolation]  += timings.at(velocity_interpolation).read_total_time();
    }


    if(ns->get_max_L2_norm_u() > 100.0)
    {
      if(save_vtk)
        ns->save_vtk((vtk_path + "/snapshot_" + std::to_string(vtk_iter + 1)).c_str());
      std::cerr << "The simulation blew up..." << std::endl;
      break;
    }

    if(save_vtk && int(tn/vtk_dt) != vtk_iter)
    {
      vtk_iter = int(tn/vtk_dt);
      ns->save_vtk((vtk_path + "/snapshot_" + std::to_string(vtk_iter)).c_str());
      export_velocity_cavity(export_dir, ns, bc_wall_value_u);
    }
    iter++;
  }

  if(cmd.contains("timing"))
  {
    ierr = PetscPrintf(mpi.comm(), "Mean computational time spent on \n"); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " viscosity step: %.5e\n", global_computational_times.at(viscous_step)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " projection step: %.5e\n", global_computational_times.at(projection_step)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " computing velocities at nodes: %.5e\n", global_computational_times.at(velocity_interpolation)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " grid update: %.5e\n", global_computational_times.at(grid_update)/iter); CHKERRXX(ierr);
    ierr = PetscPrintf(mpi.comm(), " full iteration (total): %.5e\n", (global_computational_times.at(viscous_step) + global_computational_times.at(projection_step) + global_computational_times.at(velocity_interpolation) + global_computational_times.at(grid_update))/iter); CHKERRXX(ierr);
  }

  if(dxyz_hodge_old != NULL)
    delete [] dxyz_hodge_old;

  if(cell_solver != NULL)
    delete cell_solver;
  if(face_solver != NULL)
    delete face_solver;

  delete ns;
  // the brick and the connectivity are deleted within the above destructor...
  delete levelset;

  return 0;

#endif
}
