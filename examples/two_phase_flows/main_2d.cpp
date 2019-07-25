
/*
 * The navier stokes solver applied to the flow past a static sphere
 *
 * run the program with the -help flag to see the available options
 */

// System
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// p4est Library
#ifdef P4_TO_P8
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_vtk.h>
#endif


#include <src/Parser.h>

#undef MIN
#undef MAX

using namespace std;

const double xmin = 0.0;
const double xmax = 1.0;
const double ymin = 0.0;
const double ymax = 1.0;
#ifdef P4_TO_P8
const double zmin = 0.0;
const double zmax = 1.0;
#endif


const double r0   = 0.35;

#ifdef P4_TO_P8
class LEVEL_SET: public CF_3 {
#else
class LEVEL_SET: public CF_2 {
#endif
public:
  LEVEL_SET() { lip = 1.2; }
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    return r0 - sqrt(SQR(x-(xmax+xmin)/2) + SQR(y-(ymax+ymin)/2)
                 #ifdef P4_TO_P8
                     + SQR(z-(zmax+zmin)/2)
                 #endif
                     );
  }
} level_set;

#ifdef P4_TO_P8
struct BCWALLTYPE_P : WallBC3D {
  BoundaryConditionType operator()(double, double, double) const
#else
struct BCWALLTYPE_P : WallBC2D{
  BoundaryConditionType operator()(double, double) const
#endif
  {
    return NEUMANN;
  }
} bc_wall_type_p;

#ifdef P4_TO_P8
struct BCWALLVALUE_P : CF_3{
  double operator()(double, double, double) const
#else
struct BCWALLVALUE_P : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0;
  }
} bc_wall_value_p;

#ifdef P4_TO_P8
struct BCWALLTYPE_U : WallBC3D{
  BoundaryConditionType operator()(double, double, double) const
#else
struct BCWALLTYPE_U : WallBC2D{
  BoundaryConditionType operator()(double, double) const
#endif
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

#ifdef P4_TO_P8
struct BCWALLTYPE_V : WallBC3D{
  BoundaryConditionType operator()(double, double, double) const
#else
struct BCWALLTYPE_V : WallBC2D{
  BoundaryConditionType operator()(double, double) const
#endif
  {
    return DIRICHLET;
  }
} bc_wall_type_v;


#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D{
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_w;
#endif

#ifdef P4_TO_P8
struct BCWALLVALUE_U : CF_3{
  double operator()(double, double, double) const
#else
struct BCWALLVALUE_U : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0;
  }
} bc_wall_value_u;

#ifdef P4_TO_P8
struct BCWALLVALUE_V : CF_3{
  double operator()(double, double, double) const
#else
struct BCWALLVALUE_V : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0;
  }
} bc_wall_value_v;

#ifdef P4_TO_P8
struct BCWALLVALUE_W : CF_3{
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_w;
#endif

#ifdef P4_TO_P8
struct initial_velocity_unm1_t : CF_3{
  double operator()(double, double, double) const
#else
struct initial_velocity_unm1_t : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_unm1;

#ifdef P4_TO_P8
struct initial_velocity_vnm1_t : CF_3{
  double operator()(double, double, double) const
#else
struct initial_velocity_vnm1_t : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_vnm1;

#ifdef P4_TO_P8
struct initial_velocity_wnm1_t : CF_3{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} initial_velocity_wnm1;
#endif


#ifdef P4_TO_P8
struct initial_velocity_un_t : CF_3{
  double operator()(double, double, double) const
#else
struct initial_velocity_un_t : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_un;

#ifdef P4_TO_P8
struct initial_velocity_vn_t : CF_3{
  double operator()(double, double, double) const
#else
struct initial_velocity_vn_t : CF_2{
  double operator()(double, double) const
#endif
  {
    return 0.0;
  }
} initial_velocity_vn;

#ifdef P4_TO_P8
struct initial_velocity_wn_t : CF_3{
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} initial_velocity_wn;
#endif

#ifdef P4_TO_P8
struct external_force_u_t : CF_3
{
  double operator()(double, double, double) const
#else
struct external_force_u_t : CF_2
{
  double operator()(double, double) const
#endif
  {
    return 0;
  }
};

#ifdef P4_TO_P8
struct external_force_v_t : CF_3
{
  double operator()(double, double, double) const
#else
struct external_force_v_t : CF_2
{
  double operator()(double, double) const
#endif
  {
    return 0;
  }
};

#ifdef P4_TO_P8
struct external_force_w_t : CF_3
{
  double operator()(double, double, double) const
#else
struct external_force_w_t : CF_2
{
  double operator()(double, double) const
#endif
  {
    return 0;
  }
};

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;
  cmd.add_option("restart", "if defined, this restarts the simulation from a saved state on disk (the value must be a valid path to a directory in which the solver state was saved)");
  // computational grid parameters
  cmd.add_option("lmin", "min level of the trees, default is 2");
  cmd.add_option("lmax", "max level of the trees, default is 5");
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is 0.1");
  cmd.add_option("uniform_band", "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced)");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is 1 (length of domain is 1)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is 1 (height of domain is 1)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is 1 (width of domain is 1)");
#endif
  // physical parameters for the simulations
  cmd.add_option("duration", "the duration of the simulation (tfinal-tstart). If not restarted, tstart = 0.0, default duration is 20.");
  // method-related parameters
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is 2");
  cmd.add_option("cfl", "dt = cfl * dx/vmax, default is 1.");
  cmd.add_option("hodge_tol", "numerical tolerance on the Hodge variable, at all time steps, default is 1e-3");
  cmd.add_option("niter_hodge", "max number of iterations for convergence of the Hodge variable, at all time steps, default is 10");
  cmd.add_option("pc_cell", "preconditioner for cell-solver: jacobi, sor or hypre, default is sor.");
  cmd.add_option("cell_solver", "Krylov solver used for cell-poisson problem, i.e. hodge variable: cg or bicgstab, default is bicgstab.");
  cmd.add_option("pc_face", "preconditioner for face-solver: jacobi, sor or hypre, default is sor.");
  // output-control parameters
  cmd.add_option("export_folder", "exportation_folder");
  cmd.add_option("save_vtk", "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt", "export vtk files every vtk_dt time lapse (REQUIRED if save_vtk is activated)");
  cmd.add_option("save_state_dt", "if defined, this activates the 'save-state' feature. The solver state is saved every save_state_dt time steps in backup_ subfolders.");
  cmd.add_option("save_nstates", "determines how many solver states must be memorized in backup_ folders (default is 1).");

  string extra_info = "";
  cmd.parse(argc, argv, extra_info);

  double tstart;
  double dt;
  int lmin, lmax;
  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
  my_p4est_brick_t* brick                           = NULL;
  splitting_criteria_cf_and_uniform_band_t* data    = NULL;
  LEVEL_SET level_set;

  int sl_order;
  double threshold_split_cell, uniform_band_m, uniform_band_p, cfl;
  int n_tree_xyz [P4EST_DIM];

  const double hodge_tolerance          = cmd.get<double>("hodge_tol", 1e-3);
  const unsigned int niter_hodge_max    = cmd.get<unsigned int>("niter_hodge", 10);
  const double duration                 = cmd.get<double>("duration", 200.0);
#if defined(POD_CLUSTER)
  const string export_dir               = cmd.get<string>("export_folder", "/home/regan/two_phase_flow");
#elif defined(STAMPEDE)
  const string export_dir               = cmd.get<string>("export_folder", "/work/04965/tg842642/stampede2/two_phase_flow");
#elif defined(LAPTOP)
  const string export_dir               = cmd.get<string>("export_folder", "/home/raphael/workspace/projects/two_phase_flow");
#else
  const string export_dir               = cmd.get<string>("export_folder", "/home/regan/workspace/projects/two_phase_flow");
#endif
  const bool save_vtk                   = cmd.contains("save_vtk");
  double vtk_dt                         = -1.0;
  if(save_vtk)
  {
    if(!cmd.contains("vtk_dt"))
#ifdef P4_TO_P8
      throw std::runtime_error("main_two_phase_flow_3d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
#else
      throw std::runtime_error("main_two_phase_flow_2d.cpp: the argument vtk_dt MUST be provided by the user if vtk exportation is desired.");
#endif
    vtk_dt = cmd.get<double>("vtk_dt", -1.0);
    if(vtk_dt <= 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_two_phase_flow_3d.cpp: the value of vtk_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_two_phase_flow_2d.cpp: the value of vtk_dt must be strictly positive.");
#endif
  }
  const bool save_state                 = cmd.contains("save_state_dt"); double dt_save_data = -1.0;
  const unsigned int n_states           = cmd.get<unsigned int>("save_nstates", 1);
  if(save_state)
  {
    dt_save_data                        = cmd.get<double>("save_state_dt", -1.0);
    if(dt_save_data < 0.0)
#ifdef P4_TO_P8
      throw std::invalid_argument("main_two_phase_flow_3d.cpp: the value of save_state_dt must be strictly positive.");
#else
      throw std::invalid_argument("main_two_phase_flow_2d.cpp: the value of save_state_dt must be strictly positive.");
#endif
  }
  const bool use_adapted_dt             = cmd.contains("adapted_dt");

  const string des_pc_cell              = cmd.get<string>("pc_cell", "hypre");
  const string des_solver_cell          = cmd.get<string>("cell_solver", "cg");
  const string des_pc_face              = cmd.get<string>("pc_face", "sor");
  KSPType cell_solver_type;
  PCType pc_cell, pc_face;
  if (des_pc_cell.compare("sor")==0)
    pc_cell = PCSOR;
  else if (des_pc_cell.compare("jacobi'")==0)
    pc_cell = PCJACOBI;
  else
  {
    if(des_pc_cell.compare("hypre")!=0 && !mpi.rank())
      std::cerr << "The desired preconditioner for the cell-solver was either not allowed or not correctly understood. HYPRE is used instead" << std::endl;
    pc_cell = PCHYPRE;
  }
  if(des_solver_cell.compare("bicgstab")==0)
    cell_solver_type = KSPBCGS;
  else
  {
    if(des_solver_cell.compare("cg")!=0 && !mpi.rank())
      std::cerr << "The desired Krylov solver for the cell-solver was either not allowed or not correctly understood. CG is used instead" << std::endl;
    cell_solver_type = KSPCG;
  }
  if (des_pc_face.compare("hypre")==0)
    pc_face = PCHYPRE;
  else if (des_pc_face.compare("jacobi'")==0)
    pc_face = PCJACOBI;
  else
  {
    if(des_pc_face.compare("sor")!=0 && !mpi.rank())
      std::cerr << "The desired preconditioner for the face-solver was either not allowed or not correctly understood. Successive over-relaxation is used instead" << std::endl;
    pc_face = PCSOR;
  }


  PetscErrorCode ierr;
  const double rho_m  = 1.0;
  const double rho_p  = 1.0;
  const double mu_m   = 0.001;
  const double mu_p   = 0.00001;
  const double surface_tension = 0.0073;
#ifdef P4_TO_P8
  const double xyz_min [] = {xmin, ymin, zmin};
  const double xyz_max [] = {xmax, ymax, zmax};
  const int periodic[] = {0, 0, 0};
#else
  const double xyz_min [] = {xmin, ymin};
  const double xyz_max [] = {xmax, ymax};
  const int periodic[] = {0, 0};
#endif

#ifdef P4_TO_P8
  BoundaryConditions3D bc_v[P4EST_DIM];
  BoundaryConditions3D bc_p;
#else
  BoundaryConditions2D bc_v[P4EST_DIM];
  BoundaryConditions2D bc_p;
#endif

  bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
  bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
  bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

  external_force_u_t external_force_u;
  external_force_v_t external_force_v;
#ifdef P4_TO_P8
  external_force_w_t external_force_w;
  CF_3 *external_forces[P4EST_DIM] = { &external_force_u, &external_force_v, &external_force_w };
#else
  CF_2 *external_forces[P4EST_DIM] = { &external_force_u, &external_force_v };
#endif

  lmin                    = cmd.get<int>("lmin", 4);
  lmax                    = cmd.get<int>("lmax", 6);
  threshold_split_cell    = cmd.get<double>("thresh", 0.05);
  n_tree_xyz[0]           = cmd.get<int>("nx", 1);
  n_tree_xyz[1]           = cmd.get<int>("ny", 1);
#ifdef P4_TO_P8
  n_tree_xyz[2]           = cmd.get<int>("nz", 1);
#endif

  uniform_band_m = uniform_band_p = .15*r0;
#ifdef P4_TO_P8
  const double dxmin      = MAX((xmax-xmin)/(double)n_tree_xyz[0], (ymax-ymin)/(double)n_tree_xyz[1], (zmax-zmin)/(double)n_tree_xyz[2]) / (1<<lmax);
#else
  const double dxmin      = MAX((xmax-xmin)/(double)n_tree_xyz[0], (ymax-ymin)/(double)n_tree_xyz[1]) / (1<<lmax);
#endif
  uniform_band_m         /= dxmin;
  uniform_band_p         /= dxmin;
  uniform_band_m          = cmd.get<double>("uniform_band", uniform_band_m);
  uniform_band_p          = cmd.get<double>("uniform_band", uniform_band_p);
  sl_order                = cmd.get<int>("sl_order", 2);
  cfl                     = cmd.get<double>("cfl", 1.0);


  p4est_connectivity_t *connectivity;
  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid);brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;

  connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min, xyz_max, brick, periodic);
  if(data != NULL){
    delete data; data = NULL; }
  P4EST_ASSERT(data == NULL);
  data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band_m);

  p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(int l=0; l<lmax; ++l)
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
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1); ngbd_nm1->init_neighbors();


  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = (void*) data;

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n); ngbd_n->init_neighbors();
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, brick, ngbd_c, true);
  P4EST_ASSERT(faces_n->finest_face_neighborhoods_are_valid());

  /* build the interface-capturing grid, its expanded ghost, its nodes, its hierarchy, its node neighborhoods
   * the REINITIALIZED levelset on the interface-capturing grid
   */
  splitting_criteria_cf_t* data_fine = new splitting_criteria_cf_t(lmin, lmax+1, &level_set);
  p4est_t* p4est_fine = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_fine->user_pointer = (void*) data_fine;
  p4est_refine(p4est_fine, P4EST_FALSE, refine_levelset_cf, NULL);
  p4est_ghost_t* ghost_fine = my_p4est_ghost_new(p4est_fine, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_fine, ghost_fine);
  my_p4est_hierarchy_t* hierarchy_fine = new my_p4est_hierarchy_t(p4est_fine, ghost_fine, brick);
  p4est_nodes_t* nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);
  my_p4est_node_neighbors_t* ngbd_n_fine = new my_p4est_node_neighbors_t(hierarchy_fine, nodes_fine); ngbd_n_fine->init_neighbors();

  Vec fine_phi;
  ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &fine_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_fine, nodes_fine, level_set, fine_phi);

#ifdef P4_TO_P8
  CF_3 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1 };
  CF_3 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn   };
#else
  CF_2 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1 };
  CF_2 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn   };
#endif

  two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces_n, ngbd_n_fine);
  bool second_order_phi = true;
  two_phase_flow_solver->set_phi(fine_phi, second_order_phi);
  two_phase_flow_solver->set_dynamic_viscosities(mu_m, mu_p);
  two_phase_flow_solver->set_densities(rho_m, rho_p);
  two_phase_flow_solver->set_surface_tension(surface_tension);
  two_phase_flow_solver->set_uniform_bands(uniform_band_m, uniform_band_p);
  two_phase_flow_solver->set_vorticity_split_threshold(threshold_split_cell);
  two_phase_flow_solver->set_cfl(cfl);

  two_phase_flow_solver->set_velocities(vnm1, vn, vnm1, vn);

  tstart = 0.0; // no restart so we assume we start from 0.0
  dt = 0.001;
  if(save_vtk)
    dt = MIN(dt, vtk_dt);
  two_phase_flow_solver->set_dt(dt, dt);
  two_phase_flow_solver->set_bc(bc_v, &bc_p);
  two_phase_flow_solver->set_external_forces(external_forces);

//#ifdef P4_TO_P8
//  ierr = PetscPrintf(mpi.comm(), "Parameters : Re = %g, mu = %g, rho = %g, grid is %dx%dx%d\n", Re, mu, rho, ntree_x, ntree_y, ntree_z); CHKERRXX(ierr);
//#else
//  ierr = PetscPrintf(mpi.comm(), "Parameters : Re = %g, mu = %g, rho = %g, grid is %dx%d\n", Re, mu, rho, ntree_x, ntree_y); CHKERRXX(ierr);
//#endif
//  ierr = PetscPrintf(mpi.comm(), "cfl = %g, uniform_band = %g\n", cfl, uniform_band);

  char out_dir[PATH_MAX], vtk_path[PATH_MAX], vtk_name[PATH_MAX];
  sprintf(out_dir, "%s/%dD/lmin_%d_lmax_%d", export_dir.c_str(), P4EST_DIM, lmin, lmax);
  sprintf(vtk_path, "%s/vtu", out_dir);
  if(save_vtk)
  {
    if(create_directory(out_dir, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_two_phase_flow_3d: could not create exportation directory %s", out_dir);
#else
      sprintf(error_msg, "main_two_phase_flow_2d: could not create exportation directory %s", out_dir);
#endif
      throw std::runtime_error(error_msg);
    }
    if(save_vtk && create_directory(vtk_path, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
#ifdef P4_TO_P8
      sprintf(error_msg, "main_two_phase_flow_3d: could not create exportation directory for vtk files %s", vtk_path);
#else
      sprintf(error_msg, "main_two_phase_flow_2d: could not create exportation directory for vtk files %s", vtk_path);
#endif
      throw std::runtime_error(error_msg);
    }
  }

  int iter = 0;
  int export_vtk = -1;
  int save_data_idx = (int) floor(tstart/dt_save_data); // so that we don't save the very first one which was either already read from file, or the known initial condition...

  double tn = tstart;

//  while(tn+0.01*dt<tstart+duration)
//  {
//    if(iter>0)
//    {
////      if(use_adapted_dt)
////        two_phase_flow_solver->compute_adapted_dt();
////      else
////        two_phase_flow_solver->compute_dt();
//      dt = two_phase_flow_solver->get_dt();

//      if(tn+dt>tstart+duration)
//      {
//        dt = tstart+duration-tn;
//        two_phase_flow_solver->set_dt(dt);
//      }
//      if(save_vtk && dt > vtk_dt)
//      {
//        dt = vtk_dt; // so that we don't miss snapshots...
//        two_phase_flow_solver->set_dt(dt);
//      }
//    }
////    if(save_state && ((int) floor(tn/dt_save_data)) != save_data_idx)
////    {
////      save_data_idx = ((int) floor(tn/dt_save_data));
////      ns->save_state(out_dir, tn, n_states);
////    }

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, two_phase_flow_solver->get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double corr_hodge = 1.0;
    unsigned int iter_hodge = 0;
//    while(iter_hodge<niter_hodge_max && corr_hodge>hodge_tolerance)
//    {
      hodge_new = two_phase_flow_solver->get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      two_phase_flow_solver->compute_jump_mu_grad_v();
      two_phase_flow_solver->compute_jumps_hodge();
      two_phase_flow_solver->solve_viscosity_explicit();
      two_phase_flow_solver->solve_projection(true);

//      hodge_new = two_phase_flow_solver->get_hodge();
//      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
//      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
//      corr_hodge = 0.0;
//      p4est_t *p4est = two_phase_flow_solver->get_p4est();
//      my_p4est_interpolation_nodes_t *interp_phi = two_phase_flow_solver->get_interp_phi();
//      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
//      {
//        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
//        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
//        {
//          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
//          corr_hodge = max(corr_hodge, fabs(ho[quad_idx]-hn[quad_idx]));
//        }
//      }
//      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &corr_hodge, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
//      ierr = VecRestoreArrayRead(hodge_old, &ho); CHKERRXX(ierr);
//      ierr = VecRestoreArrayRead(hodge_new, &hn); CHKERRXX(ierr);

//      ierr = PetscPrintf(mpi.comm(), "hodge iteration #%d, error = %e\n", iter_hodge, corr_hodge); CHKERRXX(ierr);
//      iter_hodge++;
//    }
//    ierr = VecDestroy(hodge_old); CHKERRXX(ierr);
      two_phase_flow_solver->extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(10);
//      two_phase_flow_solver->set_sign_of_phi_faces();
    two_phase_flow_solver->compute_velocity_at_nodes();
//    ns->compute_pressure();

//    tn += dt;

//    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*(tn-tstart)/duration, ns->get_max_L2_norm_u(), ns->get_p4est()->global_num_quadrants); CHKERRXX(ierr);

//    if(ns->get_max_L2_norm_u()>200.0)
//    {
//      if(save_vtk)
//      {
//        sprintf(vtk_name, "%s/snapshot_%d", vtk_path, export_vtk+1);
//        ns->save_vtk(vtk_name);
//      }
//      std::cerr << "The simulation blew up..." << std::endl;
//      break;
//    }

//    if(save_vtk && ((int) floor(tn/vtk_dt)) != export_vtk)
//    {
//      export_vtk = ((int) floor(tn/vtk_dt));
//      sprintf(vtk_name, "%s/snapshot_%d", vtk_path, export_vtk);
//      ns->save_vtk(vtk_name);
//    }

//    iter++;
//  }




  const double *hodge_p;
  const double *v_nodes_omega_plus_p[P4EST_DIM], *v_nodes_omega_minus_p[P4EST_DIM];
  ierr = VecGetArrayRead(two_phase_flow_solver->get_hodge(), &hodge_p); CHKERRXX(ierr);
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(two_phase_flow_solver->get_vnp1_nodes_omega_minus()[dir], &v_nodes_omega_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(two_phase_flow_solver->get_vnp1_nodes_omega_plus()[dir], &v_nodes_omega_plus_p[dir]); CHKERRXX(ierr);
  }
  my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                         P4EST_TRUE, P4EST_TRUE,
                         4, /* number of VTK_POINT_DATA */
                         1, /* number of VTK_CELL_DATA  */
                         (export_dir+"/coarse").c_str(),
                         VTK_POINT_DATA, "vn_x_plus", v_nodes_omega_plus_p[0],
                         VTK_POINT_DATA, "vn_y_plus", v_nodes_omega_plus_p[1],
                         VTK_POINT_DATA, "vn_x_minus", v_nodes_omega_minus_p[0],
                         VTK_POINT_DATA, "vn_y_minus", v_nodes_omega_minus_p[1],
      VTK_CELL_DATA, "hodge", hodge_p);
  ierr = VecRestoreArrayRead(two_phase_flow_solver->get_hodge(), &hodge_p); CHKERRXX(ierr);
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(two_phase_flow_solver->get_vnp1_nodes_omega_minus()[dir], &v_nodes_omega_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(two_phase_flow_solver->get_vnp1_nodes_omega_plus()[dir], &v_nodes_omega_plus_p[dir]); CHKERRXX(ierr);
  }

  const double* fine_phi_p, *fine_curvature_p, *fine_normal[P4EST_DIM];
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(two_phase_flow_solver->get_normals()[dir], &fine_normal[dir]); CHKERRXX(ierr);
  }
  ierr = VecGetArrayRead(two_phase_flow_solver->get_curvature(), &fine_curvature_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  my_p4est_vtk_write_all(p4est_fine, nodes_fine, ghost_fine,
                         P4EST_TRUE, P4EST_TRUE,
                         4, /* number of VTK_POINT_DATA */
                         0, /* number of VTK_CELL_DATA  */
                         (export_dir+"/fine").c_str(),
                         VTK_POINT_DATA, "phi", fine_phi_p,
                         VTK_POINT_DATA, "curvature", fine_curvature_p,
                         VTK_POINT_DATA, "nx", fine_normal[0],
      VTK_POINT_DATA, "ny", fine_normal[1]);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(two_phase_flow_solver->get_curvature(), &fine_curvature_p); CHKERRXX(ierr);
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(two_phase_flow_solver->get_normals()[dir], &fine_normal[dir]); CHKERRXX(ierr);
  }


  delete two_phase_flow_solver;
  delete data;
  delete data_fine;

  return 0;
}
