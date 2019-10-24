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

const unsigned int test_case = 1;

const double xmin = 0.0;
const double xmax = 1.0;
const double ymin = 0.0;
const double ymax = 1.0;
#ifdef P4_TO_P8
const double zmin = 0.0;
const double zmax = 1.0;
#endif

const double r0   = 0.08;
const double time_reverse = 1.0;


class LEVEL_SET: public CF_DIM {
public:
  LEVEL_SET() { lip = 1.2; }
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return r0 - sqrt(SUMD(SQR(x-(xmax+xmin)/2), SQR(y-(ymax+ymin)/2), SQR(z-(zmax+zmin)/2)));
      break;
    case 1:
      return r0 - sqrt(SUMD(SQR(x-(xmax+xmin)/2), SQR(y-(0.75*ymax+0.25*ymin)), SQR(z-(zmax+zmin)/2)));
    default:
      throw std::runtime_error("LEVEL_SET: unknown case");
      break;
    }
  }
} level_set;

class prescribed_velocity_component_t : public CF_DIM
{
protected:
  double t_;
public:
  prescribed_velocity_component_t(const double & time_) : t_(time_) {}
  inline void set_time(const double &time_) {t_ = time_;}
  virtual double dx(DIM(double x, double y, double z)) const = 0;
  virtual double dy(DIM(double x, double y, double z)) const = 0;
#ifdef P4_TO_P8
  virtual double dz(DIM(double x, double y, double z)) const = 0;
#endif
};

class prescribed_velocity_u_t : public prescribed_velocity_component_t
{
public:
  prescribed_velocity_u_t(const double &time_): prescribed_velocity_component_t(time_) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 1.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*MULTD(SQR(sin(M_PI*x)), sin(2.0*M_PI*y), -2.0*sin(2.0*M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
  double dx(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*MULTD(M_PI*sin(2.0*M_PI*x), sin(2.0*M_PI*y), -2.0*sin(2.0*M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
  double dy(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*MULTD(SQR(sin(M_PI*x)), 2.0*M_PI*cos(2.0*M_PI*y), -2.0*sin(2.0*M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
#ifdef P4_TO_P8
  double dz(double x, double y, double z) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*SQR(sin(M_PI*x))*sin(2.0*M_PI*y)*-2.0*2.0*M_PI*cos(2.0*M_PI*z);
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
#endif
};

class prescribed_velocity_v_t : public prescribed_velocity_component_t
{
public:
  prescribed_velocity_v_t(const double &time_): prescribed_velocity_component_t(time_) {}
  double operator()(DIM(double x, double y , double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?+1.0:-1.0)*MULTD(SQR(sin(2.0*M_PI*x), sin(M_PI*y)), -1.0*sin(2.0*M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_v_t: unknown case");
      break;
    }
  }
  double dx(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?+1.0:-1.0)*MULTD(2.0*M_PI*cos(2.0*M_PI*x), SQR(sin(M_PI*y)), -1.0*sin(2.0*M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
  double dy(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?+1.0:-1.0)*MULTD(sin(2.0*M_PI*x), M_PI*sin(2.0*M_PI*y), -1.0*sin(2.0*M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
#ifdef P4_TO_P8
  double dz(double x, double y, double z) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?+1.0:-1.0)*sin(2.0*M_PI*x)*SQR(sin(M_PI*y))*-1.0*2.0*M_PI*cos(2.0*M_PI*z);
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
#endif
};

#ifdef P4_TO_P8
class prescribed_velocity_w_t : public prescribed_velocity_component_t
{
public:
  prescribed_velocity_w_t(const double &time_): prescribed_velocity_component_t(time_) {}
  double operator()(double x, double y, double z) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*sin(2.0*M_PI*x)*sin(2.0*M_PI*y)*SQR(sin(M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_w_t: unknown case");
      break;
    }
  }
  double dx(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*2.0*M_PI*cos(2.0*M_PI*x)*sin(2.0*M_PI*y)*SQR(sin(M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
  double dy(DIM(double x, double y, double z)) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*sin(2.0*M_PI*x)*2.0*M_PI*cos(2.0*M_PI*y)*SQR(sin(M_PI*z));
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
  double dz(double x, double y, double z) const
  {
    switch (test_case) {
    case 0:
      return 0.0;
      break;
    case 1:
      return ((t_<time_reverse)?-1.0:+1.0)*sin(2.0*M_PI*x)*sin(2.0*M_PI*y)*M_PI*sin(2.0*M_PI*z);
      break;
    default:
      throw std::runtime_error("prescribed_velocity_u_t: unknown case");
      break;
    }
  }
};
#endif

class jump_mu_grad_v : public CF_DIM
{
private:
  prescribed_velocity_component_t *velocity_field_np1[P4EST_DIM];
  const unsigned char dir, der;
  const double jump_mu;
public:
  jump_mu_grad_v(prescribed_velocity_component_t *velocity_field[P4EST_DIM], const unsigned char &dir_, const unsigned char &der_, const double &jump_mu_): dir(dir_), der(der_), jump_mu(jump_mu_)
  {
    update_velocity_field(velocity_field);
  }
  void update_velocity_field(prescribed_velocity_component_t* velocity_field[P4EST_DIM])
  {
    for (unsigned char k = 0; k < P4EST_DIM; ++k)
      velocity_field_np1[k] = velocity_field[k];
  }
  double operator() (DIM(double x, double y, double z)) const
  {
    switch (der) {
    case dir::x:
      return jump_mu*velocity_field_np1[dir]->dx(DIM(x, y, z));
      break;
    case dir::y:
      return jump_mu*velocity_field_np1[dir]->dy(DIM(x, y, z));
      break;
#ifdef P4_TO_P8
    case dir::z:
      return jump_mu*velocity_field_np1[dir]->dz(DIM(x, y, z));
      break;
#endif
    default:
      throw std::runtime_error("jump_mu_grad_v: unknown differentiation direction");
      break;
    }
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


struct BCWALLTYPE_U : WallBCDIM{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_u;

struct BCWALLTYPE_V : WallBCDIM{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_v;


#ifdef P4_TO_P8
struct BCWALLTYPE_W : WallBCDIM{
  BoundaryConditionType operator()(DIM(double, double, double)) const
  {
    return DIRICHLET;
  }
} bc_wall_type_w;
#endif

struct BCWALLVALUE_VELOCITY_COMPONENT : CF_DIM {
  prescribed_velocity_component_t *v_component_field;
  BCWALLVALUE_VELOCITY_COMPONENT(prescribed_velocity_component_t *v_component_field_) : v_component_field(v_component_field_) {}
  double operator()(DIM(double x, double y, double z)) const
  {
    return (*v_component_field)(DIM(x, y, z));
  }
};

struct zero_force_component_t : CF_DIM {
  double operator()(DIM(double, double, double)) const
  {
    return 0.0;
  }
} zero_force_component_t;

void get_extrapolation_error_in_band(const my_p4est_two_phase_flows_t* two_phase_flow_solver, prescribed_velocity_component_t* prescribed_velocity[P4EST_DIM],
                                     double extrapolation_error_v_minus[P4EST_DIM], double extrapolation_error_v_plus[P4EST_DIM])
{
  const p4est_nodes_t *nodes_n = two_phase_flow_solver->get_nodes_n();
  const my_p4est_interpolation_nodes_t *interp_phi = two_phase_flow_solver->get_interp_phi();
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    extrapolation_error_v_minus[dim]  = 0.0;
    extrapolation_error_v_plus[dim]   = 0.0;
  }
  Vec vnp1_nodes_minus  = two_phase_flow_solver->get_vnp1_nodes_omega_minus();
  Vec vnp1_nodes_plus   = two_phase_flow_solver->get_vnp1_nodes_omega_plus();
  const double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p); CHKERRXX(ierr);


//  Vec error_x, error_y;
//  ierr = VecCreateGhostNodes(two_phase_flow_solver->get_p4est_n(), two_phase_flow_solver->get_nodes_n(), &error_x); CHKERRXX(ierr);
//  ierr = VecCreateGhostNodes(two_phase_flow_solver->get_p4est_n(), two_phase_flow_solver->get_nodes_n(), &error_y); CHKERRXX(ierr);
//  double *err_x_p, *err_y_p;
//  ierr = VecGetArray(error_x, &err_x_p); CHKERRXX(ierr);
//  ierr = VecGetArray(error_y, &err_y_p); CHKERRXX(ierr);

  double xyz_node[P4EST_DIM];
  for (p4est_locidx_t n = 0; n < nodes_n->num_owned_indeps; ++n) {
    node_xyz_fr_n(n, two_phase_flow_solver->get_p4est_n(), nodes_n, xyz_node);
    if(fabs((*interp_phi)(xyz_node)) < 2.0*two_phase_flow_solver->get_diag_min())
    {
      if((*interp_phi)(xyz_node) <= 0.0)
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        {
          extrapolation_error_v_plus[dim] = MAX(extrapolation_error_v_plus[dim], fabs((*prescribed_velocity[dim])(xyz_node) - vnp1_nodes_plus_p[P4EST_DIM*n+dim]));
//          if(dim ==0)
//            err_x_p[n] = fabs((*prescribed_velocity[dim])(xyz_node) - vnp1_nodes_plus_p[P4EST_DIM*n+dim]);
//          else
//            err_y_p[n] = fabs((*prescribed_velocity[dim])(xyz_node) - vnp1_nodes_plus_p[P4EST_DIM*n+dim]);
        }
      else
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        {
          extrapolation_error_v_minus[dim] = MAX(extrapolation_error_v_minus[dim], fabs((*prescribed_velocity[dim])(xyz_node) - vnp1_nodes_minus_p[P4EST_DIM*n+dim]));
//          if(dim ==0)
//            err_x_p[n] = fabs((*prescribed_velocity[dim])(xyz_node) - vnp1_nodes_plus_p[P4EST_DIM*n+dim]);
//          else
//            err_y_p[n] = fabs((*prescribed_velocity[dim])(xyz_node) - vnp1_nodes_plus_p[P4EST_DIM*n+dim]);
        }
    }
//    else
//    {
//      err_x_p[n] = 0.0;
//      err_y_p[n] = 0.0;
//    }
  }

//  my_p4est_vtk_write_all(two_phase_flow_solver->get_p4est_n(), two_phase_flow_solver->get_nodes_n(), two_phase_flow_solver->get_ghost_n(),
//                         P4EST_TRUE, P4EST_FALSE,
//                         2, 0,
//                         "/home/regan/workspace/projects/two_phase_flow/sharp_advection_2d/error_extrap",
//                         VTK_NODE_SCALAR, "error_x", err_x_p,
//                         VTK_NODE_SCALAR, "error_y", err_y_p);

//  ierr = VecRestoreArray(error_x, &err_x_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(error_y, &err_y_p); CHKERRXX(ierr);
//  ierr = VecDestroy(error_x); CHKERRXX(ierr);
//  ierr = VecDestroy(error_y); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p); CHKERRXX(ierr);

  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, extrapolation_error_v_minus, 2, MPI_DOUBLE, MPI_MAX, two_phase_flow_solver->get_p4est_n()->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, extrapolation_error_v_plus,  2, MPI_DOUBLE, MPI_MAX, two_phase_flow_solver->get_p4est_n()->mpicomm); SC_CHECK_MPI(mpiret);
  return;
}


int main (int argc, char* argv[])
{
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  cmdParser cmd;// computational grid parameters
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
  cmd.add_option("cfl", "dt = cfl * dx/vmax, default is 1.");
  // output-control parameters

  string extra_info = "";
  cmd.parse(argc, argv, extra_info);

  const double tstart = 0.0;
  int lmin, lmax;
  LEVEL_SET level_set;

  double threshold_split_cell, uniform_band_m, uniform_band_p, cfl;
  int n_tree_xyz [P4EST_DIM];

  const double duration                 = cmd.get<double>("duration", 2.0);
#ifdef P4_TO_P8
  const string export_dir               = "/home/regan/workspace/projects/two_phase_flow/sharp_advection_3d";
#else
  const string export_dir               = "/home/regan/workspace/projects/two_phase_flow/sharp_advection_2d";
#endif
  PetscErrorCode ierr;
  const double rho_m  = 1000.0;
  const double rho_p  = 1.0;
  const double mu_m   = 0.01;
  const double mu_p   = 0.00001;
  const double surface_tension = 0.0073;
  const double xyz_min [P4EST_DIM] = {DIM(xmin, ymin, zmin)};
  const double xyz_max [P4EST_DIM] = {DIM(xmax, ymax, zmax)};
  int periodic[P4EST_DIM];
  switch (test_case) {
  case 0:
    periodic[0] = 1; periodic[1] = 0; ONLY3D(periodic[2] = 0);
    break;
  case 1:
    periodic[0] = 0; periodic[1] = 0; ONLY3D(periodic[2] = 0);
    break;
  default:
    throw std::runtime_error("main_test_sharp_advection_2d: unknown test case");
    break;
  }

  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;

  bc_v[0].setWallTypes(bc_wall_type_u);
  bc_v[1].setWallTypes(bc_wall_type_v);
#ifdef P4_TO_P8
  bc_v[2].setWallTypes(bc_wall_type_w);
#endif
  bc_p.setWallTypes(bc_wall_type_p); bc_p.setWallValues(bc_wall_value_p);

  lmin                    = cmd.get<int>("lmin", 6);
  lmax                    = cmd.get<int>("lmax", 9);
  threshold_split_cell    = cmd.get<double>("thresh", 1.00);
  n_tree_xyz[0]           = cmd.get<int>("nx", 1);
  n_tree_xyz[1]           = cmd.get<int>("ny", 1);
#ifdef P4_TO_P8
  n_tree_xyz[2]           = cmd.get<int>("nz", 1);
#endif

  uniform_band_m = uniform_band_p = .15*r0;
  const double dxmin      = MAX(DIM((xmax-xmin)/(double)n_tree_xyz[0], (ymax-ymin)/(double)n_tree_xyz[1], (zmax-zmin)/(double)n_tree_xyz[2])) / (1<<lmax);
  uniform_band_m         /= dxmin;
  uniform_band_p         /= dxmin;
  uniform_band_m          = cmd.get<double>("uniform_band", uniform_band_m);
  uniform_band_p          = cmd.get<double>("uniform_band", uniform_band_p);
  cfl                     = cmd.get<double>("cfl", 1.0);


  my_p4est_brick_t* brick = new my_p4est_brick_t;
  p4est_connectivity_t *connectivity = my_p4est_brick_new(n_tree_xyz, xyz_min, xyz_max, brick, periodic);
  splitting_criteria_cf_and_uniform_band_t* data  = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, &level_set, uniform_band_m);

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
  splitting_criteria_cf_t* data_fine = new splitting_criteria_cf_t(2, lmax+1, &level_set);
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

  my_p4est_two_phase_flows_t *two_phase_flow_solver = new my_p4est_two_phase_flows_t(ngbd_nm1, ngbd_n, faces_n, ngbd_n_fine);
  bool second_order_phi = true;
  two_phase_flow_solver->set_phi(fine_phi, second_order_phi);
  two_phase_flow_solver->set_dynamic_viscosities(mu_m, mu_p);
  two_phase_flow_solver->set_densities(rho_m, rho_p);
  two_phase_flow_solver->set_surface_tension(surface_tension);
  two_phase_flow_solver->set_uniform_bands(uniform_band_m, uniform_band_p);
  two_phase_flow_solver->set_vorticity_split_threshold(threshold_split_cell);
  two_phase_flow_solver->set_cfl(cfl);

  double dt_n         = two_phase_flow_solver->get_dt();
  const double dt_nm1 = two_phase_flow_solver->get_dtnm1();
  double tn = tstart;

  BCWALLVALUE_VELOCITY_COMPONENT *bc_wall_value_velocity[P4EST_DIM];
  prescribed_velocity_component_t *prescribed_velocity_field_nm1[P4EST_DIM];
  prescribed_velocity_component_t *prescribed_velocity_field_n[P4EST_DIM];
  prescribed_velocity_field_nm1[0] = new prescribed_velocity_u_t(tn-dt_nm1); prescribed_velocity_field_n[0] = new prescribed_velocity_u_t(tn); bc_wall_value_velocity[0] = new BCWALLVALUE_VELOCITY_COMPONENT(prescribed_velocity_field_n[0]);
  prescribed_velocity_field_nm1[1] = new prescribed_velocity_v_t(tn-dt_nm1); prescribed_velocity_field_n[1] = new prescribed_velocity_v_t(tn); bc_wall_value_velocity[1] = new BCWALLVALUE_VELOCITY_COMPONENT(prescribed_velocity_field_n[1]);
  ONLY3D(prescribed_velocity_field_nm1[2] = new prescribed_velocity_w_t(tn-dt_nm1); prescribed_velocity_field_n[2] = new prescribed_velocity_w_t(tn); bc_wall_value_velocity[2] = new BCWALLVALUE_VELOCITY_COMPONENT(prescribed_velocity_field_n[2]);)

  jump_mu_grad_v *my_jump_mu_grad_v[P4EST_DIM][P4EST_DIM];
  CF_DIM *cf_my_jump_mu_grad_v[P4EST_DIM][P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      my_jump_mu_grad_v[dir][der] = new jump_mu_grad_v(prescribed_velocity_field_n, dir, der, mu_p-mu_m);

  CF_DIM *cf_prescribed_velocity_field_nm1[P4EST_DIM];
  CF_DIM *cf_prescribed_velocity_field_n[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    cf_prescribed_velocity_field_n[dir]   = prescribed_velocity_field_n[dir];
    cf_prescribed_velocity_field_nm1[dir] = prescribed_velocity_field_nm1[dir];
    bc_v[dir].setWallValues(*bc_wall_value_velocity[dir]);
  }

  two_phase_flow_solver->set_node_velocities(cf_prescribed_velocity_field_nm1, cf_prescribed_velocity_field_n, cf_prescribed_velocity_field_nm1, cf_prescribed_velocity_field_n);

  two_phase_flow_solver->set_bc(bc_v, &bc_p);

  int iter = 0;
  int iter_export = -1;
  const double dt_export = 0.02;

  double extrapolation_error_v_minus[P4EST_DIM], extrapolation_error_v_plus[P4EST_DIM];
  double max_extrapolation_error_v_minus[P4EST_DIM], max_extrapolation_error_v_plus[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    max_extrapolation_error_v_minus[dim]  = 0.0;
    max_extrapolation_error_v_plus[dim]   = 0.0;
  }

  while(tn+0.01*dt_n<tstart+duration)
  {
//    std::cout << "step A" << std::endl;
    if(iter>0)
    {
      two_phase_flow_solver->compute_dt();
      dt_n = two_phase_flow_solver->get_dt();
      if(tn+dt_n>tstart+duration)
      {
        dt_n = tstart+duration-tn;
        two_phase_flow_solver->set_dt(dt_n);
      }
      two_phase_flow_solver->update_from_tn_to_tnp1(iter);
    }

//    std::cout << "step B" << std::endl;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      prescribed_velocity_field_n[dir]->set_time(tn+dt_n);
      cf_prescribed_velocity_field_n[dir]   = prescribed_velocity_field_n[dir];
      for (unsigned char der = 0; der < P4EST_DIM; ++der) {
        my_jump_mu_grad_v[dir][der]->update_velocity_field(prescribed_velocity_field_n);
        cf_my_jump_mu_grad_v[dir][der] = my_jump_mu_grad_v[dir][der];
      }
    }
    two_phase_flow_solver->set_face_velocities_np1(cf_prescribed_velocity_field_n, cf_prescribed_velocity_field_n);
//    std::cout << "step C" << std::endl;
    two_phase_flow_solver->set_jump_mu_grad_v(cf_my_jump_mu_grad_v);
//    std::cout << "step D" << std::endl;


    two_phase_flow_solver->extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(PSEUDO_TIME, 20);
//    two_phase_flow_solver->extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(EXPLICIT_ITERATIVE, 10);
    two_phase_flow_solver->compute_velocity_at_nodes();

    if(((int)floor((tn+dt_n)/dt_export)) != iter_export)
    {
      iter_export = ((int)floor((tn+dt_n)/dt_export));
      two_phase_flow_solver->save_vtk((export_dir+"/illustration_"+std::to_string(iter_export)).c_str(), true, (export_dir+"/illustration_fine_"+std::to_string(iter_export)).c_str());
    }


    tn += dt_n;

    ierr = PetscPrintf(mpi.comm(), "Iteration #%04d : tn = %.5e, percent done : %.1f%%, \t max_L2_norm_u = %.5e, \t number of leaves = %d\n",
                       iter, tn, 100*(tn-tstart)/duration, two_phase_flow_solver->get_max_velocity(), two_phase_flow_solver->get_p4est()->global_num_quadrants); CHKERRXX(ierr);


    get_extrapolation_error_in_band(two_phase_flow_solver, prescribed_velocity_field_n, extrapolation_error_v_minus, extrapolation_error_v_plus);


    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      max_extrapolation_error_v_minus[dim]  = MAX(max_extrapolation_error_v_minus[dim], extrapolation_error_v_minus[dim]);
      max_extrapolation_error_v_plus[dim]   = MAX(max_extrapolation_error_v_plus[dim], extrapolation_error_v_plus[dim]);
    }

    iter++;
//    if(iter==1)
//      break;
  }


  if(mpi.rank() ==0)
  {
    std::cout << "max extrapolation_error_v_minus = " << max_extrapolation_error_v_minus[0] << " " << max_extrapolation_error_v_minus[1] << std::endl;
    std::cout << "max extrapolation_error_v_plus  = " << max_extrapolation_error_v_plus[0] << " " << max_extrapolation_error_v_plus[1] << std::endl;
  }

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      delete my_jump_mu_grad_v[dir][der];
    delete  prescribed_velocity_field_nm1[dir]; delete prescribed_velocity_field_n[dir]; delete bc_wall_value_velocity[dir];
  }

  delete data;
  delete data_fine;
  delete two_phase_flow_solver;

  return 0;
}
