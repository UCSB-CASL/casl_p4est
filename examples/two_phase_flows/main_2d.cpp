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

const double xyz_m[P4EST_DIM]         = {DIM(-1.25, -1.25, -1.25)};
const double xyz_M[P4EST_DIM]         = {DIM(+1.25, +1.25, +1.25)};
const unsigned int default_lmin       = 5;
const unsigned int default_lmax       = 5;
const double default_r0               = 0.5;
const double default_thresh           = 1000000000000000000.00;
const double default_unif_m_to_r      = 0.15;
const double default_unif_p_to_r      = 0.15;
const int default_nx                  = 1;
const int default_ny                  = 1;
#ifdef P4_TO_P8
const int default_nz                  = 1;
#endif
const double default_duration         = 250.0;
const unsigned int default_sl_order   = 1;
const double default_cfl              = 1.0;
const double default_vtk_dt           = 0.25;
const std::string default_export_dir_root = "/home/regan/workspace/projects/two_phase_flow/static_bubble_" + std::to_string(P4EST_DIM) + "d";
const bool default_voro_on_the_fly    = false;
const double default_rho_m            = 1.0;
const double default_rho_p            = 1.0;
const double default_mu_m             = 1.0/12000.0;
const double default_mu_p             = 1.0/12000.0;
const double default_surf_tension     = 1.0/12000.0;
const extrapolation_technique default_extrapolation_technique = PSEUDO_TIME;
const unsigned int default_extrapolation_niter  = 20;
const unsigned int default_extrapolation_degree = 1;
const bool default_no_advection = true;
//const bool default_no_advection = false;


class LEVEL_SET: public CF_DIM {
  const double radius;
public:
  LEVEL_SET(const double& rad_) : radius(rad_) { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    return radius - sqrt(SUMD(SQR(x - 0.5*(xyz_m[0] + xyz_M[0])), SQR(y - 0.5*(xyz_m[1] + xyz_M[1])), SQR(z - 0.5*(xyz_m[2] + xyz_M[2]))));
  }
};

struct BCWALLTYPE_P : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return NEUMANN;
  }
} bc_wall_type_p;

struct BCWALLVALUE_P : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
} bc_wall_value_p;

struct BCWALLTYPE_U : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBCDIM {
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;


#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBC3D {
  BoundaryConditionType operator()(double, double, double) const
  {
    return DIRICHLET;
  }
} bc_wall_type_w;
#endif

struct BCWALLVALUE_U : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
} bc_wall_value_u;

struct BCWALLVALUE_V : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
} bc_wall_value_v;

#ifdef P4_TO_P8
struct BCWALLVALUE_W : CF_3 {
  double operator()(double, double, double) const
  {
    return 0;
  }
} bc_wall_value_w;
#endif

struct initial_velocity_unm1_t : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_unm1;

struct initial_velocity_vnm1_t : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_vnm1;

#ifdef P4_TO_P8
struct initial_velocity_wnm1_t : CF_3 {
  double operator()(double, double, double) const
  {
    return 0.0;
  }
} initial_velocity_wnm1;
#endif


struct initial_velocity_un_t : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} initial_velocity_un;

struct initial_velocity_vn_t : CF_DIM {
  double operator()(DIM(double, double, double)) const
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

struct external_force_u_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
};

struct external_force_v_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
};

struct external_force_w_t : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
};

struct zero_cf : CF_DIM
{
  double operator()(DIM(double, double, double)) const
  {
    return 0;
  }
};

int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;// computational grid parameters
  cmd.add_option("lmin",  "min level of the trees, default is " + std::to_string(default_lmin));
  cmd.add_option("lmax",  "max level of the trees, default is " + std::to_string(default_lmax));
  cmd.add_option("r0",    "original radius of the bubble " + std::to_string(default_r0));
  cmd.add_option("thresh", "the threshold used for the refinement criteria, default is " + std::to_string(default_thresh));
  cmd.add_option("uniform_band_m", "size of the uniform band in negative domain around the interface, in number of dx (default ensures a layer of " + std::to_string(default_unif_m_to_r) + " of the initial bubble");
  cmd.add_option("uniform_band_p", "size of the uniform band in positive domain around the interface, in number of dx (default ensures a layer of " + std::to_string(default_unif_p_to_r) + " of the initial bubble");
  cmd.add_option("nx", "number of trees in the x-direction. The default value is " + std::to_string(default_nx) + " (length of domain is 2.5)");
  cmd.add_option("ny", "number of trees in the y-direction. The default value is " + std::to_string(default_ny) + " (height of domain is 2.5)");
#ifdef P4_TO_P8
  cmd.add_option("nz", "number of trees in the z-direction. The default value is " + std::to_string(default_nz) + " (width of domain is 2.5)");
#endif
  // physical parameters for the simulations
  cmd.add_option("duration", "the duration of the simulation. Default duration is " + std::to_string(default_duration));
  cmd.add_option("rho_m", "mass density of fluid in negative domain. Default is " + std::to_string(default_rho_m));
  cmd.add_option("rho_p", "mass density of fluid in positive domain. Default is " + std::to_string(default_rho_p));
  cmd.add_option("mu_m", "shear viscosity of the fluid in the negative domain. Default is " + std::to_string(default_mu_m));
  cmd.add_option("mu_p", "shear viscosity of the fluid in the positive domain. Default is " + std::to_string(default_mu_p));
  cmd.add_option("surf_tension", "the surface tension between the two fluids. Default duration is " + std::to_string(default_surf_tension));
  // method-related parameters
  cmd.add_option("sl_order", "the order for the semi lagrangian, either 1 (stable) or 2 (accurate), default is" + std::to_string(default_sl_order));
  cmd.add_option("cfl", "dt = cfl * dx/vmax, default is " + std::to_string(default_cfl));
  cmd.add_option("extrapolation", "face-based extrapolation technique (0: pseudo-time, 1: explicit iterative), default is " + std::string(default_extrapolation_technique == PSEUDO_TIME ? "pseudo time" : "explicit iterative"));
  cmd.add_option("extrapolation_niter", "number of iterations used for the face-based extrapolation technique, default is " + std::to_string(default_extrapolation_niter));
  cmd.add_option("extrapolation_degree", "degree of the face-based extrapolation technique (0: constant, 1: extrapolate normal derivatives too), default is " + std::to_string(default_extrapolation_degree));
  cmd.add_option("voro_on_the_fly", "activates the calculation of Voronoi cells on the fly (default is " + std::string(default_voro_on_the_fly? "done on the fly)" : "stored in memory)"));
  cmd.add_option("no_advection", "deactivates the advection, the fluid dynamics becomes diffusion problems (default is " + std::string(default_no_advection? "without" : "with") + " advection)");
  // output-control parameters
  cmd.add_option("no_save_vtk", "deactivates exportation of vtk files if present");
  cmd.add_option("vtk_dt", "vtk_dt = time_step between two vtk exportation, default is " + std::to_string(default_vtk_dt));
  cmd.add_option("export_dir_root", "root of the exportation directory, results are stored in subdirectories. Default root is " + default_export_dir_root);

  std::string extra_info = "";
  if(cmd.parse(argc, argv, extra_info))
    return 0;

  double tstart;
  double dt;
  my_p4est_two_phase_flows_t* two_phase_flow_solver = NULL;
  my_p4est_brick_t* brick                           = NULL;
  splitting_criteria_cf_and_uniform_band_t* data    = NULL;
  const unsigned int sl_order         = cmd.get<unsigned int>("sl_order", default_sl_order);
  const double threshold_split_cell   = cmd.get<double>("thresh", default_thresh);;
  const double r0                     = cmd.get<double>("r0", default_r0);
  const unsigned int lmin             = cmd.get<unsigned int>("lmin", default_lmin);
  const unsigned int lmax             = cmd.get<unsigned int>("lmax", default_lmax);
  const int n_tree_xyz [P4EST_DIM]    = {DIM(cmd.get<int>("nx", default_nx), cmd.get<int>("ny", default_ny), cmd.get<int>("nz", default_nz))};
  const double tree_dimensions[P4EST_DIM] = {DIM((xyz_M[0] - xyz_m[0])/n_tree_xyz[0], (xyz_M[1] - xyz_m[1])/n_tree_xyz[1], (xyz_M[2] - xyz_m[2])/n_tree_xyz[2])};
  const double max_tree_dimensions    = MAX(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]));
  const double uniform_band_m         = cmd.get<double>("uniform_band_m", default_unif_m_to_r*r0*(1 << lmax)/max_tree_dimensions);
  const double uniform_band_p         = cmd.get<double>("uniform_band_p", default_unif_p_to_r*r0*(1 << lmax)/max_tree_dimensions);
  const double cfl                    = cmd.get<double>("cfl", default_cfl);
  const extrapolation_technique extrapolation = (cmd.contains("extrapolation") ? (cmd.get<unsigned int>("extrapolation") == 0 ? PSEUDO_TIME : EXPLICIT_ITERATIVE) : default_extrapolation_technique);
  const unsigned int ext_nsteps       = cmd.get<unsigned int>("extrapolation_niter", default_extrapolation_niter);
  const unsigned int ext_degree       = cmd.get<unsigned int>("extrapolation_degree", default_extrapolation_degree);
  const double duration               = cmd.get<double>("duration", default_duration);
  const std::string export_dir_root   = cmd.get<std::string>("export_dir_root", default_export_dir_root);
  const bool save_vtk                 = !cmd.contains("no_save_vtk");
  const bool no_advection             = default_no_advection || cmd.contains("no_advection");
  double vtk_dt                       = -1.0;
  if(save_vtk)
  {
    vtk_dt = cmd.get<double>("vtk_dt", default_vtk_dt);
    if(vtk_dt <= 0.0)
      throw std::invalid_argument("main_" + std::to_string(P4EST_DIM) + "d.cpp for static bubble: the value of vtk_dt must be strictly positive.");
  }
  const double rho_m                  = cmd.get<double>("rho_m", default_rho_m);
  const double rho_p                  = cmd.get<double>("rho_p", default_rho_p);
  const double mu_m                   = cmd.get<double>("mu_m", default_mu_m);
  const double mu_p                   = cmd.get<double>("mu_p", default_mu_p);
  const double surface_tension        = cmd.get<double>("surf_tension", default_surf_tension);
  LEVEL_SET level_set(r0);
  const int periodic[P4EST_DIM]       = {DIM(0, 0, 0)};

  PetscErrorCode ierr;
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;

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
#endif
  CF_DIM *external_forces[P4EST_DIM] = { DIM(&external_force_u, &external_force_v, &external_force_w) };


  p4est_connectivity_t *connectivity;
  if(brick != NULL && brick->nxyz_to_treeid != NULL)
  {
    P4EST_FREE(brick->nxyz_to_treeid);brick->nxyz_to_treeid = NULL;
    delete brick; brick = NULL;
  }
  P4EST_ASSERT(brick == NULL);
  brick = new my_p4est_brick_t;

  connectivity = my_p4est_brick_new(n_tree_xyz, xyz_m, xyz_M, brick, periodic);
  if(data != NULL){
    delete data; data = NULL; }
  P4EST_ASSERT(data == NULL);
  data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band_m);

  p4est_t* p4est_nm1 = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);
  p4est_nm1->user_pointer = (void*) data;

  for(unsigned int l = 0; l < lmax; ++l)
  {
    my_p4est_refine(p4est_nm1, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);
  }
  /* create the initial forest at time nm1 */
  p4est_balance(p4est_nm1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_nm1, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);
  if(third_degree_ghost_are_required(tree_dimensions))
    my_p4est_ghost_expand(p4est_nm1, ghost_nm1);


  p4est_nodes_t *nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  my_p4est_hierarchy_t *hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  my_p4est_node_neighbors_t *ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1); ngbd_nm1->init_neighbors();

  /* create the initial forest at time n */
  p4est_t *p4est_n = my_p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_n->user_pointer = (void*) data;

  p4est_ghost_t *ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_n, ghost_n);
  if(third_degree_ghost_are_required(tree_dimensions))
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
  if(third_degree_ghost_are_required(tree_dimensions))
    my_p4est_ghost_expand(p4est_fine, ghost_fine);
  my_p4est_hierarchy_t* hierarchy_fine = new my_p4est_hierarchy_t(p4est_fine, ghost_fine, brick);
  p4est_nodes_t* nodes_fine = my_p4est_nodes_new(p4est_fine, ghost_fine);
  my_p4est_node_neighbors_t* ngbd_n_fine = new my_p4est_node_neighbors_t(hierarchy_fine, nodes_fine); ngbd_n_fine->init_neighbors();

  Vec fine_phi;
  ierr = VecCreateGhostNodes(p4est_fine, nodes_fine, &fine_phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_fine, nodes_fine, level_set, fine_phi);

  CF_DIM *vnm1[P4EST_DIM] = { DIM(&initial_velocity_unm1, &initial_velocity_vnm1, &initial_velocity_wnm1) };
  CF_DIM *vn  [P4EST_DIM] = { DIM(&initial_velocity_un  , &initial_velocity_vn  , &initial_velocity_wn)   };

  two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces_n, ngbd_n_fine);
  bool second_order_phi = true;
  two_phase_flow_solver->set_phi(fine_phi, second_order_phi);
  two_phase_flow_solver->set_dynamic_viscosities(mu_m, mu_p);
  two_phase_flow_solver->set_densities(rho_m, rho_p);
  two_phase_flow_solver->set_surface_tension(surface_tension);
  two_phase_flow_solver->set_uniform_bands(uniform_band_m, uniform_band_p);
  two_phase_flow_solver->do_voronoi_computations_on_the_fly(default_voro_on_the_fly || cmd.contains("voro_on_the_fly"));
  two_phase_flow_solver->set_vorticity_split_threshold(threshold_split_cell);
  two_phase_flow_solver->set_cfl(cfl);
  two_phase_flow_solver->set_semi_lagrangian_order(sl_order);

  two_phase_flow_solver->set_node_velocities(vnm1, vn, vnm1, vn);
  if(no_advection)
  {
    two_phase_flow_solver->initialize_diffusion_problem_vectors();
    Vec* vnm1_faces = two_phase_flow_solver->get_diffusion_vnm1_faces();
    Vec* vn_faces = two_phase_flow_solver->get_diffusion_vn_faces();
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      double *vnm1_faces_dir_p, *vn_faces_dir_p;
      ierr = VecGetArray(vnm1_faces[dir], &vnm1_faces_dir_p); CHKERRXX(ierr);
      ierr = VecGetArray(vn_faces[dir],   &vn_faces_dir_p); CHKERRXX(ierr);
      for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]; ++face_idx) {
        double xyz_face[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, dir, xyz_face);
        vnm1_faces_dir_p[face_idx]  = (*vnm1[dir])(xyz_face);
        vn_faces_dir_p[face_idx]    = (*vn[dir])(xyz_face);
      }
      ierr = VecRestoreArray(vnm1_faces[dir], &vnm1_faces_dir_p); CHKERRXX(ierr);
    }
  }


  tstart = 0.0; // no restart so we assume we start from 0.0
  dt = sqrt((rho_m + rho_p)*pow(MIN(DIM(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]))/((double) (1 << lmax)), 3.0)/(4.0*M_PI*surface_tension));
  if(save_vtk)
    dt = MIN(dt, vtk_dt);
  two_phase_flow_solver->set_dt(dt, dt);
  two_phase_flow_solver->set_bc(bc_v, &bc_p);
  two_phase_flow_solver->set_external_forces(external_forces);

  const std::string vtk_export_dir = export_dir_root  + "/" + std::to_string(n_tree_xyz[0]) + "x" + std::to_string(n_tree_xyz[1]) ONLY3D(+ "x" + std::to_string(n_tree_xyz[2])) + "/lmin_" + std::to_string(lmin) + "_lmax_" + std::to_string(lmax);
  if(save_vtk && create_directory(vtk_export_dir.c_str(), mpi.rank(), mpi.comm()))
    throw std::runtime_error("main_" + std::to_string(P4EST_DIM) + "d, static bubble: could not create exportation directory " + vtk_export_dir);

  int iter = 0, iter_vtk = -1;
  double tn = tstart;

  double max_velocity = 0.0;

  parStopWatch w;
  w.start("Execution time");

  while(tn + 0.01*dt < tstart + duration)
  {
    if(iter > 0)
    {
      two_phase_flow_solver->compute_dt();
      dt = two_phase_flow_solver->get_dt();
      if(!no_advection)
        two_phase_flow_solver->update_from_tn_to_tnp1();
      else
      {
        two_phase_flow_solver->slide_face_fields(true);
        two_phase_flow_solver->slide_node_velocities();
      }
    }

    two_phase_flow_solver->compute_viscosity_jumps();
    two_phase_flow_solver->compute_jumps_hodge();

    if(!no_advection)
      two_phase_flow_solver->solve_viscosity();
    else
      two_phase_flow_solver->solve_diffusion_viscosity();

    two_phase_flow_solver->solve_projection(false);
    two_phase_flow_solver->extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(extrapolation, ext_nsteps, ext_degree);
    two_phase_flow_solver->compute_velocity_at_nodes();

    if(save_vtk && (int) floor((tn + dt)/vtk_dt) != iter_vtk)
    {
      iter_vtk = (int) floor((tn + dt)/vtk_dt);
      two_phase_flow_solver->save_vtk((vtk_export_dir + "/snapshot_" + std::to_string(iter_vtk)).c_str(), true, (vtk_export_dir + "/fine_snapshot_"+std::to_string(iter_vtk)).c_str());
    }

    tn += dt;
    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_u = %.5e (Omega minus), \t %.5e (Omega plus), \t number of leaves = %d\n",
                       iter, tn, 100*(tn-tstart)/duration, two_phase_flow_solver->get_max_velocity_m(), two_phase_flow_solver->get_max_velocity_p(), two_phase_flow_solver->get_p4est()->global_num_quadrants); CHKERRXX(ierr);
    max_velocity = MAX(max_velocity, two_phase_flow_solver->get_max_velocity());

    iter++;
  }
  ierr = PetscPrintf(mpi.comm(), "Maximum value of parasitic current = %.5e\n", max_velocity);


  w.stop();
  w.read_duration();

  delete two_phase_flow_solver;
  delete data;
  delete data_fine;

  return 0;
}
