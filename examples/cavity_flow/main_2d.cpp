
/*
 * The navier stokes solver
 *
 * run the program with the -help flag to see the available options
 */

// p4est Library
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

using namespace std;

#if defined(LAPTOP)
const string default_export_dir  = "/home/raphael/workspace/projects/cavity_flow/original_solver";
#else
const string default_export_dir  = "/home/regan/workspace/projects/cavity_flow/original_solver";
#endif

const double default_side_length  = 1.0;
const double default_top_velocity = 1.0;
const int default_nx = 2;
const int default_ny = 2;

const unsigned int default_lmin             = 3;
const unsigned int default_lmax             = 3;
const unsigned int default_niter_hodge_max  = 10;
const unsigned int default_sl_order         = 2;
const double default_hodge_tol              = 1.0e-3;
const double default_hole_radius            = 0.25;
const double default_tf                     = 37.0;
const double default_Re                     = 1000.0;
const double default_rho                    = 1.0;

const double default_cfl                = 1.0;
const double default_vorticity_thresh   = 0.1;
const double default_uniform_band       = 5.0;
const double default_smoke_thresh       = 0.5;
const double default_vtk_dt             = 0.2;

class INIT_SMOKE : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    return (sqrt(SQR(x-.5) + SQR(y-.85))<.05) ? 1 : 0;
  }
} init_smoke;

class BC_SMOKE : public CF_2
{
public:
  double operator()(double, double) const
  {
    return 0;
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
      throw invalid_argument("INTERFACE_LEVEL_SET::INTERFACE_LEVEL_SET() : the radius of the hole must be strictly positive");
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

struct BCWALLVALUE_P : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_wall_value_p;

struct BCWALLTYPE_U : WallBC2D
{
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBC2D
{
  BoundaryConditionType operator()(double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;

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

struct BCWALLVALUE_V : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_wall_value_v;

struct BCINTERFACE_VALUE_U : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_interface_value_u;

struct BCINTERFACE_VALUE_V : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_interface_value_v;

struct BCINTERFACE_VALUE_P : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} bc_interface_value_p;

struct initial_velocity_unm1_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} initial_velocity_unm1;

struct initial_velocity_u_n_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} initial_velocity_un;

struct initial_velocity_vnm1_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} initial_velocity_vnm1;

struct initial_velocity_vn_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} initial_velocity_vn;

struct external_force_u_t : CF_2
{
  double operator()(double, double) const
  {
    return 0;
  }
} external_force_u;

struct external_force_v_t : CF_2
{

  double operator()(double, double) const
  {
    return 0;
  }
};

void check_velocity_cavity(const string& export_dir, mpi_environment_t& mpi, my_p4est_navier_stokes_t *ns, double Re, double n_times_dt)
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
  int N = 200;
  for(int i=0; i<=N; ++i)
  {
    double xyz0[] = { (double)i/(double)N, .5 };
    interp0.add_point(i, xyz0);
    interp0_phi.add_point(i, xyz0);

    double xyz1[] = { .5, (double)i/(double)N };
    interp1.add_point(i, xyz1);
    interp1_phi.add_point(i, xyz1);
  }

  std::vector<double> v0(N+1);
  interp0.set_input(vn[1], quadratic);
  interp0.interpolate(v0.data());

  std::vector<double> v1(N+1);
  interp1.set_input(vn[0], quadratic);
  interp1.interpolate(v1.data());

  std::vector<double> phi0(N+1);
  interp1.set_input(phi, quadratic);
  interp1.interpolate(phi0.data());

  std::vector<double> phi1(N+1);
  interp1.set_input(phi, quadratic);
  interp1.interpolate(phi1.data());

  if(!mpi.rank())
  {
    FILE* fp;
    ostringstream filename;
    filename << fixed << setprecision(2);
    filename << "velocity_" << data->min_lvl << "-" << data->max_lvl << "_" << ngbd_n->get_brick()->nxyztrees[0] << "x" << ngbd_n->get_brick()->nxyztrees[1]
             << "_Re_" << Re << "_cfl_" << n_times_dt << ".dat";
    fp = fopen((export_dir + "/" + filename.str()).c_str(), "w");

    if(fp == NULL)
      throw std::invalid_argument("check_velocity_cavity: could not open file.");

    ierr = PetscFPrintf(mpi.comm(), fp, "%% x/y \t vx \t vy\n"); CHKERRXX(ierr);
    for(int i=0; i<=N; ++i)
    {
      ierr = PetscFPrintf(mpi.comm(), fp, "%g, %g, %g\n", (double)i/(double)N, phi0[i]<0 ? v0[i] : 0, phi1[i]<0 ? v1[i] : 0); CHKERRXX(ierr);
    }

    fclose(fp);
  }
}

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

#ifdef P4_TO_P8
  cerr << "This program is not developed for 3D application, implemented for 2D run and validation only..." << endl;
  return 1;
#endif

  cmdParser cmd;
  cmd.add_option("lmin",                  "min level of the trees, default is " + std::to_string(default_lmin));
  cmd.add_option("lmax",                  "max level of the trees, default is " + std::to_string(default_lmax));
  cmd.add_option("vort_thresh",           "the threshold used for the vorticity-based refinement criterion, default is " + std::to_string(default_vorticity_thresh));
  cmd.add_option("uniform_band",          "size of the uniform band around the interface, in number of dx (a minimum of 2 is strictly enforced), default is " + std::to_string(default_uniform_band));
  cmd.add_option("nx",                    "number of trees in the macromesh along x. The default value is " + std::to_string(default_nx));
  cmd.add_option("ny",                    "number of trees in the macromesh along y. The default value is " + std::to_string(default_ny));
  cmd.add_option("smoke_thresh",          "threshold for smoke refinement, default is " + std::to_string(default_smoke_thresh));
  cmd.add_option("refine_with_smoke",     "refine the grid with the smoke density and threshold smoke_thresh if present");
  // physical parameters for the simulations
  cmd.add_option("duration",              "the duration of the simulation. tstart = 0.0, default duration is " + to_string(default_tf));
  cmd.add_option("Re",                    "the Reynolds number = rho*u_top*L/mu, default is " + to_string(default_Re));
  cmd.add_option("smoke",                 "no smoke if option not present, with smoke if option present");
  cmd.add_option("length",                "side length of the cavity, default is " + to_string(default_side_length));
  cmd.add_option("hole_radius",           "if defined, adds a hole (solid, non-penetrable interface) in the center of the domain. The radius of the hole can be specified by the user. Default value is " + to_string(default_hole_radius));
  cmd.add_option("top_velocity",          "velocity of the top wall of the cavity. Default value is "  + to_string(default_top_velocity));
  cmd.add_option("mass_density",          "mass density of the fluid. Default value is "  + to_string(default_rho));
  // method-related parameters
  cmd.add_option("sl_order",              "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is " + to_string(default_sl_order));

  cmd.add_option("cfl",                   "dt = cfl * dx/vmax, default is " + to_string(default_cfl));
  cmd.add_option("hodge_tol",             "numerical tolerance used for the convergence criterion on the Hodge variable, at all time steps.\n\
                 Default is " + to_string(default_hodge_tol) + ": w.r.t absolute value of hodge.");
  cmd.add_option("niter_hodge",           "max number of iterations for convergence of the Hodge variable, at all time steps, default is " + to_string(default_niter_hodge_max));
  // output-control parameters
  cmd.add_option("export_folder",         "exportation_folder if not defined otherwise in the environment variable OUT_DIR,\n\
                  subfolder(s) will be created, default is " + default_export_dir);
  cmd.add_option("save_vtk",              "activates exportation of results in vtk format");
  cmd.add_option("vtk_dt",                "export vtk files every vtk_dt time lapse. If not specified, default is " + to_string(default_vtk_dt));
  cmd.add_option("timing",                "if defined, activates the internal timer and prints final information .");
  cmd.parse(argc, argv);

  int sl_order = cmd.get("sl_order", default_sl_order);
  int lmin = cmd.get("lmin", default_lmin);
  int lmax = cmd.get("lmax", default_lmax);
  double n_times_dt = cmd.get("n_times_dt", default_cfl);
  double threshold_split_cell = cmd.get("thresh", default_vorticity_thresh);
  bool save_vtk = cmd.contains("save_vtk");
  const double duration = cmd.get<double>("duration", default_tf);
  double vtk_dt = DBL_MAX;
  if (save_vtk)
  {
    vtk_dt = cmd.get<double>("vtk_dt", default_vtk_dt);
    if (vtk_dt <= 0.0)
      throw invalid_argument("cavity_flow::main: the value of vtk_dt must be strictly positive.");
  }


  const bool with_smoke         = cmd.contains("smoke");
  const bool refine_with_smoke  = cmd.contains("refine_with_smoke");
  const double smoke_thresh     = cmd.get<double>("smoke_thresh", default_smoke_thresh);
  INIT_SMOKE init_smoke;

  double Re, rho, top_wall_velocity, side_length;
  Re = cmd.get<double>("Re", default_Re);
  rho = cmd.get<double>("mass_density", default_rho);
  top_wall_velocity = cmd.get<double>("top_velocity", default_top_velocity);
  side_length = cmd.get<double>("length", default_side_length);
  const int n_tree_x = cmd.get<int>("nx", default_nx);
  const int n_tree_y = cmd.get<int>("ny", default_ny);
  double mu = rho*top_wall_velocity*side_length/Re;
  const double xyz_min_[2] = {0.0, 0.0};
  const double xyz_max_[2] = {side_length, side_length};
  int n_xyz [] = {n_tree_x, n_tree_y};
  int periodic[] = {0, 0};

  double uniform_band = cmd.get("uniform_band", default_uniform_band);

  PetscErrorCode ierr;
  ierr = PetscPrintf(mpi.comm(), "Parameters : Re = %g, mu = %g, rho = %g, grid is %dx%d\n", Re, mu, rho, n_tree_x, n_tree_y); CHKERRXX(ierr);
  ierr = PetscPrintf(mpi.comm(), "n_times_dt = %g, uniform_band = %g\n", n_times_dt, uniform_band);

  parStopWatch watch;
  watch.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;
  connectivity = my_p4est_brick_new(n_xyz, xyz_min_, xyz_max_, &brick, periodic);

  // create (possible) interface levelset
  CF_2 *levelset = NULL;
  const bool with_hole = cmd.contains("hole_radius");
  if(with_hole)
    levelset = new INTERFACE_LEVEL_SET(cmd.get<double>("hole_radius", default_hole_radius), &brick);
  else
    levelset = new NOINTERFACE_LEVEL_SET;

  p4est_t *p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  splitting_criteria_cf_t data(lmin, lmax, levelset, 1.2);

  p4est_nm1->user_pointer = (void*)&data;
  for(int l = 0; l < lmax; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }

  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, &brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);

  if(refine_with_smoke==1)
  {
    splitting_criteria_thresh_t crit_thresh(lmin, lmax, &init_smoke, smoke_thresh);
    p4est_n->user_pointer = (void*)&crit_thresh;
    my_p4est_refine(p4est_n, P4EST_TRUE, refine_levelset_thresh, NULL);
    p4est_balance(p4est_n, P4EST_CONNECT_FULL, NULL);
  }

  p4est_n->user_pointer = (void*)&data;
  my_p4est_partition(p4est_n, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);

  p4est_nodes_t *nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  my_p4est_cell_neighbors_t *ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_n);
  my_p4est_faces_t *faces_n = new my_p4est_faces_t(p4est_n, ghost_n, &brick, ngbd_c);

  Vec phi;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *levelset, phi);

  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_1st_order_time_2nd_order_space(phi);
  lsn.perturb_level_set_function(phi, EPS);

  CF_2 *vnm1[P4EST_DIM] = { &initial_velocity_unm1, &initial_velocity_vnm1 };
  CF_2 *vn  [P4EST_DIM] = { &initial_velocity_un  , &initial_velocity_vn   };

  BoundaryConditions2D bc_v[P4EST_DIM];
  BoundaryConditions2D bc_p;

  BCWALLVALUE_U bc_wall_value_u(top_wall_velocity, &brick);
  bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
  bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

#ifndef P4_TO_P8
  if(with_hole)
#endif
  {
    bc_v[0].setInterfaceType(DIRICHLET); bc_v[0].setInterfaceValue(bc_interface_value_u);
    bc_v[1].setInterfaceType(DIRICHLET); bc_v[1].setInterfaceValue(bc_interface_value_v);
#ifdef P4_TO_P8
    bc_v[2].setInterfaceType(DIRICHLET); bc_v[2].setInterfaceValue(bc_interface_value_w);
#endif
    bc_p.setInterfaceType(NEUMANN); bc_p.setInterfaceValue(bc_interface_value_p);
  }


  my_p4est_navier_stokes_t ns(ngbd_nm1, ngbd_n, faces_n);
  ns.set_phi(phi);
  ns.set_parameters(mu, rho, sl_order, uniform_band, threshold_split_cell, n_times_dt);
  const double dxmin = MIN((brick.xyz_max[0] - brick.xyz_min[0])/brick.nxyztrees[0], (brick.xyz_max[1] - brick.xyz_min[1])/brick.nxyztrees[1])/((double) (1 << data.max_lvl));
  double dt = MIN(dxmin*n_times_dt/bc_wall_value_u.top_wall_velocity, vtk_dt, duration);

  ns.set_dt(dt, dt);
  dt = ns.get_dt();
  ns.set_velocities(vnm1, vn);
  ns.set_bc(bc_v, &bc_p);

  if(with_smoke)
  {
    Vec smoke;
    ierr = VecDuplicate(phi, &smoke); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, init_smoke, smoke);
    ns.set_smoke(smoke, &bc_smoke, refine_with_smoke, smoke_thresh);
  }

  const unsigned int niter_hodge_max = cmd.get<unsigned int>("niter_hodge", default_niter_hodge_max);
  const double hodge_tol = cmd.get<double>("hodge_tol", default_hodge_tol);
  double tn = 0;
  unsigned int iter = 0;
  int vtk_iter = -1;

  const string export_root = cmd.get<string>("export_folder", (getenv("OUT_DIR") == NULL ? default_export_dir : getenv("OUT_DIR")));
  ostringstream oss;
  oss.str("");
  oss << fixed << setprecision(2);
  oss << export_root << "/" << (with_hole ? "with" : "without") << "_hole/Re_" << Re  << "_cfl_" << n_times_dt << "/nx_" << brick.nxyztrees[0]
      << "_ny_" << brick.nxyztrees[1] << "_lmin_" << data.min_lvl << "_lmax_" << data.max_lvl;
  const string export_dir  = oss.str();
  const string vtk_path    = export_dir + "/vtu";

  while(tn+0.01*dt < duration)
  {
    if(iter>0)
    {
      ns.compute_dt();

      dt = ns.get_dt();

      if(tn + dt > duration)
      {
        dt = duration - tn;
        ns.set_dt(dt);
      }

      ns.update_from_tn_to_tnp1(levelset);
    }

    Vec hodge_old;
    Vec hodge_new;
    ierr = VecCreateSeq(PETSC_COMM_SELF, ns.get_p4est()->local_num_quadrants, &hodge_old); CHKERRXX(ierr);
    double err_hodge = 1;
    unsigned int iter_hodge = 0;
    while(iter_hodge < niter_hodge_max && err_hodge > hodge_tol)
    {
      hodge_new = ns.get_hodge();
      ierr = VecCopy(hodge_new, hodge_old); CHKERRXX(ierr);

      ns.solve_viscosity();
      ns.solve_projection();

      hodge_new = ns.get_hodge();
      const double *ho; ierr = VecGetArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      const double *hn; ierr = VecGetArrayRead(hodge_new, &hn); CHKERRXX(ierr);
      err_hodge = 0;
      p4est_t *p4est = ns.get_p4est();
      my_p4est_interpolation_nodes_t *interp_phi = ns.get_interp_phi();
      for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
      {
        p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
        for(size_t q=0; q<tree->quadrants.elem_count; ++q)
        {
          p4est_locidx_t quad_idx = tree->quadrants_offset+q;
          double xyz[P4EST_DIM];
          quad_xyz_fr_q(quad_idx, tree_idx, p4est, ns.get_ghost(), xyz);
          if((*interp_phi)(xyz[0],xyz[1])<0)
            err_hodge = max(err_hodge, fabs(ho[quad_idx]-hn[quad_idx]));
        }
      }
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &err_hodge, 1, MPI_DOUBLE, MPI_MAX, mpi.comm()); SC_CHECK_MPI(mpiret);
      ierr = VecRestoreArrayRead(hodge_old, &ho); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(hodge_new, &hn); CHKERRXX(ierr);

      ierr = PetscPrintf(mpi.comm(), "hodge iteration #%d, error = %e\n", iter_hodge, err_hodge); CHKERRXX(ierr);
      iter_hodge++;
    }
    ierr = VecDestroy(hodge_old); CHKERRXX(ierr);
    ns.compute_velocity_at_nodes();
    ns.compute_pressure();

    tn += dt;

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n", iter, tn, 100*tn/duration, ns.get_max_L2_norm_u(), ns.get_p4est()->global_num_quadrants); CHKERRXX(ierr);

    if(ns.get_max_L2_norm_u() > 100.0)
    {
      if(save_vtk)
        ns.save_vtk((vtk_path + "/snapshot_" + to_string(vtk_iter + 1)).c_str());
      cerr << "The simulation blew up..." << endl;
      break;
    }

    if(save_vtk && int(tn/vtk_dt) != vtk_iter)
    {
      vtk_iter = int(tn/vtk_dt);
      ns.save_vtk((vtk_path + "/snapshot_" + to_string(vtk_iter)).c_str());
      check_velocity_cavity(export_dir, mpi, &ns, Re, n_times_dt);
    }
    iter++;
  }

  delete levelset;

  watch.stop();
  watch.read_duration();

  return 0;
}
