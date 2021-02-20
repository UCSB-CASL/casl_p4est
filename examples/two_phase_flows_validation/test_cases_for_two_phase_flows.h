#ifndef TWO_PHASE_FLOWS_TESTS_H
#define TWO_PHASE_FLOWS_TESTS_H

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interface_manager.h>
#else
#include <src/my_p4est_utils.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interface_manager.h>
#endif

struct domain_t {
  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM];
  int periodicity[P4EST_DIM];
  inline double length() const { return xyz_max[0] - xyz_min[0]; }
  inline double height() const { return xyz_max[1] - xyz_min[1]; }
#ifdef P4_TO_P8
  inline double width()  const { return xyz_max[2] - xyz_min[2]; }
#endif
};

class test_case_for_two_phase_flows_t
{
  inline double get_advection_dt_0(const double& cfl_advection, const double &dx_min) const
  {
    return cfl_advection*dx_min/max_v_magnitude_0;
  }
  inline double get_visco_capillary_dt_0(const double& cfl_visco_capillary, const double &dx_min) const
  {
    return cfl_visco_capillary*MIN(mu_m, mu_p)*dx_min/max_surface_tension_0;
  }
  inline double get_capillary_dt_0(const double& cfl_capillary, const double &dx_min) const
  {
    return sqrt(cfl_capillary*(rho_m + rho_p)*pow(dx_min, 3)/(4.0*M_PI*max_surface_tension_0));
  }
protected:
  double time;
  double t_end;
  double max_v_magnitude_0;
  double max_surface_tension_0;

  struct wall_pressure_bc_value_t : public CF_DIM
  {
    test_case_for_two_phase_flows_t* owner;
    wall_pressure_bc_value_t(test_case_for_two_phase_flows_t* owner_) : owner(owner_) {}

    inline double operator()(DIM(double x, double y, double z)) const
    {
      double xyz[P4EST_DIM] = {DIM(x, y, z)};
      const char sgn_point = ((owner->levelset_function)(xyz) <= 0.0 ? -1 : +1);
      switch (owner->pressure_wall_bc.getWallType()(xyz)) {
      case DIRICHLET:
        return (sgn_point < 0 ? owner->pressure_minus(xyz) : owner->pressure_plus(xyz));
        break;
      case NEUMANN:
      {
        double wall_normal[P4EST_DIM] = {DIM(0, 0, 0)};
        double mag_wall_normal = 0.0;
        for (u_char dir = 0; dir < P4EST_DIM; ++dir)
        {
          if (xyz[dir] - owner->domain.xyz_max[dir] < EPS*fabs(owner->domain.xyz_max[dir] - owner->domain.xyz_min[dir]))
            wall_normal[dir] = +1.0;
          else if (xyz[dir] - owner->domain.xyz_min[dir] < EPS*fabs(owner->domain.xyz_max[dir] - owner->domain.xyz_min[dir]))
            wall_normal[dir] = -1.0;
          mag_wall_normal += SQR(wall_normal[dir]);
        }
        mag_wall_normal = sqrt(mag_wall_normal);
        for (u_char dir = 0; dir < P4EST_DIM; ++dir)
          wall_normal[dir] = (mag_wall_normal > EPS ? wall_normal[dir]/mag_wall_normal : 0.0);

        double wall_pressure_grad[P4EST_DIM];
        if(sgn_point < 0)
          owner->pressure_gradient_minus(xyz, wall_pressure_grad);
        else
          owner->pressure_gradient_plus(xyz, wall_pressure_grad);
        return SUMD(wall_normal[0]*wall_pressure_grad[0], wall_normal[1]*wall_pressure_grad[1], wall_normal[2]*wall_pressure_grad[2]);
      }
        break;
      default:
        throw std::runtime_error("test_case_for_two_phase_flows_t::wall_pressure_bc_value_t() : unknown wall type");
        break;
      }
      return NAN;
    }

  };

  struct wall_velocity_bc_value_t : public CF_DIM
  {
    test_case_for_two_phase_flows_t* owner;
    const u_char dir;
    wall_velocity_bc_value_t(test_case_for_two_phase_flows_t* owner_, const u_char& dir_) : owner(owner_), dir(dir_) {}

    inline double operator()(DIM(double x, double y, double z)) const
    {
      double xyz[P4EST_DIM] = {DIM(x, y, z)};
      const char sgn_point = ((owner->levelset_function)(xyz) <= 0.0 ? -1 : +1);
      switch (owner->velocity_wall_bc[dir].getWallType()(xyz)) {
      case DIRICHLET:
        return (sgn_point < 0 ? owner->velocity_minus(dir, xyz) : owner->velocity_plus(dir, xyz));
        break;
      case NEUMANN:
      {
        double wall_normal[P4EST_DIM] = {DIM(0, 0, 0)};
        double mag_wall_normal = 0.0;
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
        {
          if (xyz[dim] - owner->domain.xyz_max[dim] < EPS*fabs(owner->domain.xyz_max[dim] - owner->domain.xyz_min[dim]))
            wall_normal[dim] = +1.0;
          else if (xyz[dim] - owner->domain.xyz_min[dim] < EPS*fabs(owner->domain.xyz_max[dim] - owner->domain.xyz_min[dim]))
            wall_normal[dim] = -1.0;
          mag_wall_normal += SQR(wall_normal[dim]);
        }
        mag_wall_normal = sqrt(mag_wall_normal);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          wall_normal[dim] = (mag_wall_normal > EPS ? wall_normal[dim]/mag_wall_normal : 0.0);

        double wall_velocity_grad[P4EST_DIM];
        for (u_char der = 0; der < P4EST_DIM; ++der)
          wall_velocity_grad[der] = (sgn_point < 0 ? owner->first_derivative_velocity_minus(dir, der, xyz) : owner->first_derivative_velocity_plus(dir, der, xyz));
        return SUMD(wall_normal[0]*wall_velocity_grad[0], wall_normal[1]*wall_velocity_grad[1], wall_normal[2]*wall_velocity_grad[2]);

      }
        break;
      default:
        throw std::runtime_error("test_case_for_two_phase_flows_t::wall_velocity_bc_value_t() : unknown wall type");
        break;
      }
      return NAN;
    }

  };

  struct force_per_unit_mass_t : CF_DIM
  {
    test_case_for_two_phase_flows_t* owner;
    const char sgn;
    const u_char dir;
    force_per_unit_mass_t(test_case_for_two_phase_flows_t* owner_, const char& sgn_, const u_char& dir_)
      : owner(owner_), sgn(sgn_), dir(dir_) { }
    inline double operator()(DIM(double x, double y, double z)) const
    {
      const double xyz[P4EST_DIM] = {DIM(x, y, z)};
      double velocity_grad[SQR_P4EST_DIM];
      double velocity[P4EST_DIM];
      double grad_p[P4EST_DIM];
      if(sgn < 0)
      {
        owner->gradient_velocity_minus(xyz, velocity_grad);
        owner->pressure_gradient_minus(xyz, grad_p);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          velocity[dim] = owner->velocity_minus(dim, xyz);
      }
      else
      {
        owner->gradient_velocity_plus(xyz, velocity_grad);
        owner->pressure_gradient_plus(xyz, grad_p);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          velocity[dim] = owner->velocity_plus(dim, xyz);
      }

      double ans = 0.0;

      const double& rho = (sgn < 0 ? owner->rho_m : owner->rho_p);
      const double& mu  = (sgn < 0 ? owner->mu_m  : owner->mu_p);
      ans += rho*(sgn < 0 ? owner->time_derivative_velocity_minus(dir, xyz) : owner->time_derivative_velocity_plus(dir, xyz)); // partial time derivative
      for(u_char der = 0; der < P4EST_DIM; der++)
        ans += rho*velocity[der]*velocity_grad[P4EST_DIM*dir + der]; // advection terms;
      ans += grad_p[dir]; // pressure gradient
      ans -= mu*(sgn < 0 ? owner->laplacian_velocity_minus(dir, xyz) : owner->laplacian_velocity_plus(dir, xyz)); // viscous terms;
      ans /= rho; // "per unit mass"!
      return ans;
    }
  };

  struct levelset_as_cf_dim_t : CF_DIM
  {
    test_case_for_two_phase_flows_t* owner;
    levelset_as_cf_dim_t(test_case_for_two_phase_flows_t* owner_)
      : owner(owner_) { }
    inline double operator()(DIM(double x, double y, double z)) const
    {
      const double xyz[P4EST_DIM] = {DIM(x, y, z)};
      return owner->levelset_function(xyz);
    }
  } levelset_as_cf_dim;

  struct velocity_functor_t : CF_DIM
  {
    test_case_for_two_phase_flows_t* owner;
    const char sgn;
    const u_char dir;
    velocity_functor_t(test_case_for_two_phase_flows_t* owner_, const char& sgn_, const u_char& dir_)
      : owner(owner_), sgn(sgn_), dir(dir_) { }
    inline double operator()(DIM(double x, double y, double z)) const
    {
      const double xyz[P4EST_DIM] = {DIM(x, y, z)};
      return (sgn < 0 ? owner->velocity_minus(dir, xyz) : owner->velocity_plus(dir, xyz));
    }
  };

  domain_t domain;
  std::string description;
  std::string test_name;
  double mu_m, mu_p;
  double rho_m, rho_p;
  double surface_tension;
  bool static_interface, surface_tension_is_constant, nonzero_mass_flux, levelset_cf_is_signed_distance, pressure_is_floating;
  wall_pressure_bc_value_t wall_pressure_bc_value;
  wall_velocity_bc_value_t  DIM(wall_velocity_bc_value_u,   wall_velocity_bc_value_v,   wall_velocity_bc_value_w  );
  force_per_unit_mass_t     DIM(bulk_acceleration_minus_x,  bulk_acceleration_minus_y,  bulk_acceleration_minus_z );
  force_per_unit_mass_t     DIM(bulk_acceleration_plus_x,   bulk_acceleration_plus_y,   bulk_acceleration_plus_z  );
  velocity_functor_t        DIM(functor_u_minus,            functor_v_minus,            functor_w_minus           );
  velocity_functor_t        DIM(functor_u_plus,             functor_v_plus,             functor_w_plus            );

  BoundaryConditionsDIM pressure_wall_bc;
  BoundaryConditionsDIM velocity_wall_bc[P4EST_DIM];
public:
  test_case_for_two_phase_flows_t()
    : time(0.0),
      levelset_as_cf_dim(this),
      wall_pressure_bc_value(this),
      DIM(wall_velocity_bc_value_u(this, dir::x), wall_velocity_bc_value_v(this, dir::y), wall_velocity_bc_value_w(this, dir::z)),
      DIM(bulk_acceleration_minus_x(this, -1, dir::x), bulk_acceleration_minus_y(this, -1, dir::y), bulk_acceleration_minus_z(this, -1, dir::z)),
      DIM(bulk_acceleration_plus_x(this, +1, dir::x), bulk_acceleration_plus_y(this, +1, dir::y), bulk_acceleration_plus_z(this, +1, dir::z)),
      DIM(functor_u_minus(this, -1, dir::x),  functor_v_minus(this, -1, dir::y),  functor_w_minus(this, -1, dir::z)),
      DIM(functor_u_plus(this, +1, dir::x),   functor_v_plus(this, +1, dir::y),   functor_w_plus(this, +1, dir::z))

  {
    pressure_wall_bc.setWallValues(wall_pressure_bc_value);
    velocity_wall_bc[0].setWallValues(wall_velocity_bc_value_u);
    velocity_wall_bc[1].setWallValues(wall_velocity_bc_value_v);
#ifdef P4_TO_P8
    velocity_wall_bc[2].setWallValues(wall_velocity_bc_value_w);
#endif
  }

  inline void sample_variable_surface_tension(const my_p4est_interface_manager_t* interface_manager, Vec variable_surface_tension) const
  {
    const p4est_t* p4est = interface_manager->get_interface_capturing_ngbd_n().get_p4est();
    const p4est_nodes_t* nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
    if(variable_surface_tension == NULL || !VecIsSetForNodes(variable_surface_tension, nodes, p4est->mpicomm, 1))
      throw std::runtime_error("test_case_for_two_phase_flows_t::sample_variable_surface_tension() 'variable_surface_tension' is ill-defined");
    PetscErrorCode ierr;
    double xyz_node[P4EST_DIM];
    double *variable_surface_tension_p;
    ierr = VecGetArray(variable_surface_tension, &variable_surface_tension_p); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_layer_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_layer_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      variable_surface_tension_p[node_idx] = local_surface_tension(xyz_node);
    }
    ierr = VecGhostUpdateBegin(variable_surface_tension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_local_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_local_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      variable_surface_tension_p[node_idx] = local_surface_tension(xyz_node);
    }
    ierr = VecGhostUpdateEnd(variable_surface_tension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(variable_surface_tension, &variable_surface_tension_p); CHKERRXX(ierr);
    return;
  }

  inline void sample_interface_stress_source(my_p4est_interface_manager_t* interface_manager, Vec interfacial_force) const
  {
    const p4est_t* p4est = interface_manager->get_interface_capturing_ngbd_n().get_p4est();
    const p4est_nodes_t* nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
    if(interfacial_force == NULL || !VecIsSetForNodes(interfacial_force, nodes, p4est->mpicomm, P4EST_DIM))
      throw std::runtime_error("test_case_for_two_phase_flows_t::sample_interface_stress_source() 'interfacial_force' is ill-defined");

    PetscErrorCode ierr;
    interface_manager->set_grad_phi();
    interface_manager->set_curvature();
    Vec grad_phi = interface_manager->get_grad_phi();
    Vec curvature = interface_manager->get_curvature();
    const double *grad_phi_p, *curvature_p;
    ierr = VecGetArrayRead(grad_phi, &grad_phi_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(curvature, &curvature_p); CHKERRXX(ierr);
    double xyz_node[P4EST_DIM];
    double normal[P4EST_DIM], grad_phi_mag;
    double grad_v_minus[SQR_P4EST_DIM], grad_v_plus[SQR_P4EST_DIM];
    double grad_surf_tension[P4EST_DIM];
    double *interfacial_force_p;
    ierr = VecGetArray(interfacial_force, &interfacial_force_p); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_layer_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_layer_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      gradient_velocity_minus(xyz_node, grad_v_minus);
      gradient_velocity_plus(xyz_node, grad_v_plus);
      gradient_surface_tension(xyz_node, grad_surf_tension);
      const double mass_flux = local_mass_flux(xyz_node);
      grad_phi_mag = ABSD(grad_phi_p[P4EST_DIM*node_idx + 0], grad_phi_p[P4EST_DIM*node_idx + 1], grad_phi_p[P4EST_DIM*node_idx + 2]);
      for(u_char dir = 0; dir < P4EST_DIM; dir++)
      {
        interfacial_force_p[P4EST_DIM*node_idx + dir] = 0.0; // initialize the components of the interface stress terms
        normal[dir] = (grad_phi_mag > EPS ? grad_phi_p[P4EST_DIM*node_idx + dir]/grad_phi_mag : 0.0);
      }

      // interface_stress term = [(-p*I + mu*(grad_u + grad_u^T))\cdot n] - gamma*kappa*n - SQR(mass_flux)*[1/rho]*n + (I - nn)\cdot grad gamma
      for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
        interfacial_force_p[P4EST_DIM*node_idx + dir] -= (pressure_plus(xyz_node) - pressure_minus(xyz_node))*normal[dir];
        interfacial_force_p[P4EST_DIM*node_idx + dir] -= local_surface_tension(xyz_node)*curvature_p[node_idx]*normal[dir];
        interfacial_force_p[P4EST_DIM*node_idx + dir] -= SQR(mass_flux)*(1.0/rho_p - 1.0/rho_m)*normal[dir];
        for(u_char der = 0; der < P4EST_DIM; der++)
        {
          interfacial_force_p[P4EST_DIM*node_idx + dir] += (mu_p*(grad_v_plus[P4EST_DIM*dir + der] + grad_v_plus[P4EST_DIM*der + dir]) - mu_m*(grad_v_minus[P4EST_DIM*dir + der] + grad_v_minus[P4EST_DIM*der + dir]))*normal[der];
          interfacial_force_p[P4EST_DIM*node_idx + dir] += ((dir == der ? 1.0 : 0.0) - normal[dir]*normal[der])*grad_surf_tension[der];
        }
      }
    }
    ierr = VecGhostUpdateBegin(interfacial_force, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_local_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_local_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      gradient_velocity_minus(xyz_node, grad_v_minus);
      gradient_velocity_plus(xyz_node, grad_v_plus);
      gradient_surface_tension(xyz_node, grad_surf_tension);
      const double mass_flux = local_mass_flux(xyz_node);
      grad_phi_mag = ABSD(grad_phi_p[P4EST_DIM*node_idx + 0], grad_phi_p[P4EST_DIM*node_idx + 1], grad_phi_p[P4EST_DIM*node_idx + 2]);
      for(u_char dir = 0; dir < P4EST_DIM; dir++)
      {
        interfacial_force_p[P4EST_DIM*node_idx + dir] = 0.0; // initialize the components of the interface stress terms
        normal[dir] = (grad_phi_mag > EPS ? grad_phi_p[P4EST_DIM*node_idx + dir]/grad_phi_mag : 0.0);
      }

      // interface_stress term = [(-p*I + mu*(grad_u + grad_u^T))\cdot n] - gamma*kappa*n - SQR(mass_flux)*[1/rho]*n + (I - nn)\cdot grad gamma
      for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
        interfacial_force_p[P4EST_DIM*node_idx + dir] -= (pressure_plus(xyz_node) - pressure_minus(xyz_node))*normal[dir];
        interfacial_force_p[P4EST_DIM*node_idx + dir] -= local_surface_tension(xyz_node)*curvature_p[node_idx]*normal[dir];
        interfacial_force_p[P4EST_DIM*node_idx + dir] -= SQR(mass_flux)*(1.0/rho_p - 1.0/rho_m)*normal[dir];
        for(u_char der = 0; der < P4EST_DIM; der++)
        {
          interfacial_force_p[P4EST_DIM*node_idx + dir] += (mu_p*(grad_v_plus[P4EST_DIM*dir + der] + grad_v_plus[P4EST_DIM*der + dir]) - mu_m*(grad_v_minus[P4EST_DIM*dir + der] + grad_v_minus[P4EST_DIM*der + dir]))*normal[der];
          interfacial_force_p[P4EST_DIM*node_idx + dir] += ((dir == der ? 1.0 : 0.0) - normal[dir]*normal[der])*grad_surf_tension[der];
        }
      }
    }
    ierr = VecGhostUpdateEnd(interfacial_force, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(grad_phi, &grad_phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(curvature, &curvature_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(interfacial_force, &interfacial_force_p); CHKERRXX(ierr);
  }

  inline void sample_mass_flux(const my_p4est_interface_manager_t* interface_manager, Vec mass_flux) const
  {
    const p4est_t* p4est = interface_manager->get_interface_capturing_ngbd_n().get_p4est();
    const p4est_nodes_t* nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
    if(mass_flux == NULL || !VecIsSetForNodes(mass_flux, nodes, p4est->mpicomm, 1))
      throw std::runtime_error("test_case_for_two_phase_flows_t::sample_mass_flux() 'mass_flux' is ill-defined");
    PetscErrorCode ierr;
    double xyz_node[P4EST_DIM];
    double *mass_flux_p;
    ierr = VecGetArray(mass_flux, &mass_flux_p); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_layer_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_layer_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      mass_flux_p[node_idx] = local_mass_flux(xyz_node);
    }
    ierr = VecGhostUpdateBegin(mass_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_local_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_local_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      mass_flux_p[node_idx] = local_mass_flux(xyz_node);
    }
    ierr = VecGhostUpdateEnd(mass_flux, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(mass_flux, &mass_flux_p); CHKERRXX(ierr);
  }

  inline void sample_levelset(const my_p4est_interface_manager_t* interface_manager, Vec phi_np1)
  {
    const p4est_t* p4est = interface_manager->get_interface_capturing_ngbd_n().get_p4est();
    const p4est_nodes_t* nodes = interface_manager->get_interface_capturing_ngbd_n().get_nodes();
    if(phi_np1 == NULL || !VecIsSetForNodes(phi_np1, nodes, p4est->mpicomm, 1))
      throw std::runtime_error("test_case_for_two_phase_flows_t::sample_levelset() 'phi_np1' is ill-defined");
    PetscErrorCode ierr;
    double xyz_node[P4EST_DIM];
    double *phi_np1_p;
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_layer_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_layer_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      phi_np1_p[node_idx] = levelset_function(xyz_node);
    }
    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for(size_t k = 0; k < interface_manager->get_interface_capturing_ngbd_n().get_local_size(); k++)
    {
      p4est_locidx_t node_idx = interface_manager->get_interface_capturing_ngbd_n().get_local_node(k);
      node_xyz_fr_n(node_idx, p4est, nodes, xyz_node);
      phi_np1_p[node_idx] = levelset_function(xyz_node);
    }
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    return;
  }

  // levelset function
  virtual double levelset_function(const double *xyz) const = 0;
  // negative velocity field
  virtual double velocity_minus(const u_char& dir, const double *xyz) const = 0;
  virtual double time_derivative_velocity_minus(const u_char& dir, const double *xyz) const = 0;
  virtual double first_derivative_velocity_minus(const u_char& dir, const u_char& der, const double *xyz) const = 0;
  inline void gradient_velocity_minus(const double *xyz, double grad_v_minus[P4EST_DIM]) const
  {
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      for(u_char der = 0; der < P4EST_DIM; der++)
        grad_v_minus[P4EST_DIM*dir + der] = first_derivative_velocity_minus(dir, der, xyz);
  }
  virtual double laplacian_velocity_minus(const u_char &dir, const double *xyz) const = 0;
  // positive velocity field
  virtual double velocity_plus(const u_char& dir, const double *xyz) const = 0;
  virtual double time_derivative_velocity_plus(const u_char& dir, const double *xyz) const = 0;
  virtual double first_derivative_velocity_plus(const u_char& dir, const u_char& der, const double *xyz) const = 0;
  inline void gradient_velocity_plus(const double *xyz, double grad_v_plus[P4EST_DIM]) const
  {
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      for(u_char der = 0; der < P4EST_DIM; der++)
        grad_v_plus[P4EST_DIM*dir + der] = first_derivative_velocity_plus(dir, der, xyz);
  }
  virtual double laplacian_velocity_plus(const u_char &dir, const double *xyz) const = 0;
  // mass flux
  virtual double local_mass_flux(const double *xyz) const = 0;

  // negative pressure field
  virtual double pressure_minus(const double *xyz) const = 0;
  virtual double first_derivative_pressure_minus(const u_char& der, const double *xyz) const = 0;
  inline void pressure_gradient_minus(const double *xyz, double grad_p_minus[P4EST_DIM]) const
  {
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      grad_p_minus[dir] = first_derivative_pressure_minus(dir, xyz);
    return;
  }

  // positive pressure field
  virtual double pressure_plus(const double *xyz) const = 0;
  virtual double first_derivative_pressure_plus(const u_char& der, const double *xyz) const = 0;
  inline void pressure_gradient_plus(const double *xyz, double grad_p_plus[P4EST_DIM]) const
  {
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      grad_p_plus[dir] = first_derivative_pressure_plus(dir, xyz);
    return;
  }

  // useful if nonconstant surface tension
  virtual double local_surface_tension(const double *xyz) const = 0;
  virtual void gradient_surface_tension(const double *xyz, double grad_surf_tension[P4EST_DIM]) const = 0;

  // some accessors:
  inline const double *get_xyz_min() const            { return domain.xyz_min; }
  inline const double *get_xyz_max() const            { return domain.xyz_max; }
  inline const int *get_periodicity() const           { return domain.periodicity; }
  inline const std::string& get_description() const   { return description; }
  inline const std::string& get_name() const          { return test_name; }
  inline double get_mu_minus() const                  { return mu_m; }
  inline double get_mu_plus() const                   { return mu_p; }
  inline double get_rho_minus() const                 { return rho_m; }
  inline double get_rho_plus() const                  { return rho_p; }
  inline double get_surface_tension() const           { return (surface_tension_is_constant ? surface_tension : NAN); }
  inline const domain_t &get_domain() const           { return domain; }
  inline bool is_interface_static() const             { return static_interface;                  };
  inline bool is_surface_tension_constant() const     { return surface_tension_is_constant;       };
  inline bool is_pressure_floating() const            { return pressure_is_floating;              };
  inline bool with_mass_flux() const                  { return nonzero_mass_flux;                 };
  inline bool is_reinitialization_needed() const      { return !levelset_cf_is_signed_distance ;  };
  inline void set_time(const double& time_)   { time = time_; };
  inline BoundaryConditionsDIM& get_pressure_wall_bc() { return pressure_wall_bc; }
  inline BoundaryConditionsDIM* get_velocity_wall_bc() { return velocity_wall_bc; }
  inline void get_force_per_unit_mass_minus(CF_DIM* force_per_unit_mass_minus[P4EST_DIM])
  {
    force_per_unit_mass_minus[0] = &bulk_acceleration_minus_x;
    force_per_unit_mass_minus[1] = &bulk_acceleration_minus_y;
#ifdef P4_TO_P8
    force_per_unit_mass_minus[2] = &bulk_acceleration_minus_z;
#endif
  }

  inline void get_force_per_unit_mass_plus(CF_DIM* force_per_unit_mass_plus[P4EST_DIM])
  {
    force_per_unit_mass_plus[0] = &bulk_acceleration_plus_x;
    force_per_unit_mass_plus[1] = &bulk_acceleration_plus_y;
#ifdef P4_TO_P8
    force_per_unit_mass_plus[2] = &bulk_acceleration_plus_z;
#endif
  }

  inline const CF_DIM* get_levelset() const { return &levelset_as_cf_dim; }

  inline void get_velocity_functors(const CF_DIM* vminus_functors[P4EST_DIM], const CF_DIM* vplus_functors[P4EST_DIM]) const
  {
    vminus_functors[0] = &functor_u_minus;  vplus_functors[0] = &functor_u_plus;
    vminus_functors[1] = &functor_v_minus;  vplus_functors[1] = &functor_v_plus;
#ifdef P4_TO_P8
    vminus_functors[2] = &functor_w_minus;  vplus_functors[2] = &functor_w_plus;
#endif
    return;
  }

  inline double compute_dt_0(const double& cfl_advection, const double& cfl_visco_capillary, const double& cfl_capillary, const double& dx_min) const
  {
    return MIN(get_advection_dt_0(cfl_advection, dx_min), get_visco_capillary_dt_0(cfl_visco_capillary, dx_min) + sqrt(SQR(get_visco_capillary_dt_0(cfl_visco_capillary, dx_min)) + SQR(get_capillary_dt_0(cfl_capillary, dx_min))));
  }

  inline double get_final_time() const { return t_end; }

};

#ifndef P4_TO_P8
static class test_case_0_t : public test_case_for_two_phase_flows_t
{
  inline double rr(const double* xyz) const { return MAX(ABSD(xyz[0], xyz[1], xyz[2]), 0.00001); }
public:
  test_case_0_t() : test_case_for_two_phase_flows_t()
  {
    mu_m  = 0.0001;
    mu_p  = 0.01;
    rho_m = 0.01;
    rho_p = 1.0;
    surface_tension = 0.0; // no surface tension effect in here but a surface-defined interface stress
    surface_tension_is_constant = true; // no need to bother with the calculation of Marangoni forces
    nonzero_mass_flux = false;
    levelset_cf_is_signed_distance = true;
    pressure_is_floating = true;

    static_interface = true;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim] = -2.0;
      domain.xyz_max[dim] = +2.0;
      domain.periodicity[dim] = 0;
    }
    t_end = M_PI;
    pressure_wall_bc.setWallTypes(neumann_cf);
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      velocity_wall_bc[dir].setWallTypes(dirichlet_cf);

    description =
        std::string("* domain = [-2.0, +2.0] X [-2.0, +2.0] \n")
        + std::string("* no periodicity \n")
        + std::string("* (static) interface = circle of radius 1, centered in (0, 0), negative inside, positive outside \n")
        + std::string("* mu_m = 0.0001; \n")
        + std::string("* mu_p = 0.01; \n")
        + std::string("* rho_m = 0.01; \n")
        + std::string("* rho_p = 1.00; \n")
        + std::string("* surface_tension = 0.00; \n")
        + std::string("* u_m  = y*(x*2 + y*2 - 1)*cos(t); \n")
        + std::string("* v_m  = -x*(x*2 + y*2 - 1)*cos(t); \n")
        + std::string("* u_p  = (y/r - y)*cos(t); \n")
        + std::string("* v_p  = -(x/r -x)*cos(t); \n")
        + std::string("* pressure_minus = cos(x)*cos(y)*cos(t); \n")
        + std::string("* pressure_plus  = 0.0; \n")
        + std::string("* Dirichlet BC on all walls for velocity; Neumann on all walls for pressure (pressure is floating); \n")
        + std::string("* Validation test case 0 in 2D");
    test_name = "test_case_0";
    max_v_magnitude_0 = MAX(2.0*sqrt(3.0)/9.0, 2.0*sqrt(2.0) - 1);
    max_surface_tension_0 = EPS;
  }

  inline double levelset_function(const double *xyz) const
  {
    return rr(xyz) - 1.0;
  }
  // negative velocity field
  inline double velocity_minus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return xyz[1]*(SQR(xyz[0]) + SQR(xyz[1]) - 1.0)*cos(time);
      break;
    case dir::y:
      return -xyz[0]*(SQR(xyz[0]) + SQR(xyz[1]) - 1.0)*cos(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double time_derivative_velocity_minus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return -xyz[1]*(SQR(xyz[0]) + SQR(xyz[1]) - 1.0)*sin(time);
      break;
    case dir::y:
      return +xyz[0]*(SQR(xyz[0]) + SQR(xyz[1]) - 1.0)*sin(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::time_derivative_velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double first_derivative_velocity_minus(const u_char& dir, const u_char& der, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
    {
      switch (der) {
      case dir::x:
        return 2.0*xyz[0]*xyz[1]*cos(time);
        break;
      case dir::y:
        return 3.0*SQR(xyz[1])*cos(time);
        break;
      default:
        throw std::invalid_argument("test_case_0_t::first_derivative_velocity_minus: unknown cartesian direction");
        break;
      }
    }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return -3.0*SQR(xyz[0])*cos(time);
        break;
      case dir::y:
        return -2.0*xyz[0]*xyz[1]*cos(time);
        break;
      default:
        throw std::invalid_argument("test_case_0_t::first_derivative_velocity_minus: unknown cartesian direction");
        break;
      }
      break;
    default:
      throw std::invalid_argument("test_case_0_t::first_derivative_velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double laplacian_velocity_minus(const u_char &dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return 8.0*xyz[1]*cos(time);
      break;
    case dir::y:
      return -8.0*xyz[0]*cos(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::laplacian_velocity_minus: unknown velocity component");
      break;
    }
  }
  // positive velocity field
  inline double velocity_plus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return xyz[1]*(1.0/rr(xyz) - 1.0)*cos(time);
      break;
    case dir::y:
      return -xyz[0]*(1.0/rr(xyz) - 1.0)*cos(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::velocity_plus: unknown velocity component");
      break;
    }
  }
  inline double time_derivative_velocity_plus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return -xyz[1]*(1.0/rr(xyz) - 1.0)*sin(time);
      break;
    case dir::y:
      return +xyz[0]*(1.0/rr(xyz) - 1.0)*sin(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::time_derivative_velocity_plus: unknown velocity component");
      break;
    }
  }
  inline double first_derivative_velocity_plus(const u_char& dir, const u_char& der, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
    {
      switch (der) {
      case dir::x:
        return -(xyz[0]*xyz[1]/pow(rr(xyz), 3.0))*cos(time);
        break;
      case dir::y:
        return (SQR(xyz[0])/pow(rr(xyz), 3.0) - 1.0)*cos(time);
        break;
      default:
        throw std::invalid_argument("test_case_0_t::first_derivative_velocity_plus: unknown cartesian direction");
        break;
      }
    }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return (1.0 - SQR(xyz[1])/pow(rr(xyz), 3.0))*cos(time);
        break;
      case dir::y:
        return (xyz[0]*xyz[1]/pow(rr(xyz), 3.0))*cos(time);
        break;
      default:
        throw std::invalid_argument("test_case_0_t::first_derivative_velocity_plus: unknown cartesian direction");
        break;
      }
      break;
    default:
      throw std::invalid_argument("test_case_0_t::first_derivative_velocity_plus: unknown velocity component");
      break;
    }
  }

  inline double laplacian_velocity_plus(const u_char &dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return (-xyz[1]/pow(rr(xyz), 3.0))*cos(time);
      break;
    case dir::y:
      return (+xyz[0]/pow(rr(xyz), 3.0))*cos(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::laplacian_velocity_plus: unknown velocity component");
      break;
    }
  }
  // mass flux
  inline double local_mass_flux(const double *) const
  {
    return 0.0;
  }

  // negative pressure field
  inline double pressure_minus(const double *xyz) const
  {
    return cos(xyz[0])*cos(xyz[1])*cos(time);
  }
  inline double first_derivative_pressure_minus(const u_char& der, const double *xyz) const
  {
    switch (der) {
    case dir::x:
      return -sin(xyz[0])*cos(xyz[1])*cos(time);
      break;
    case dir::y:
      return -cos(xyz[0])*sin(xyz[1])*cos(time);
      break;
    default:
      throw std::invalid_argument("test_case_0_t::first_derivative_pressure_minus: unknown cartesian direction");
      break;
    }
  }

  // positive pressure field
  inline double pressure_plus(const double *) const
  {
    return 0.0;
  }
  inline double first_derivative_pressure_plus(const u_char&, const double *) const
  {
    return 0.0;
  }

  // useful if nonconstant surface tension
  inline double local_surface_tension(const double *) const
  {
    return surface_tension;
  }
  inline void gradient_surface_tension(const double *, double grad_surf_tension[P4EST_DIM]) const
  {
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      grad_surf_tension[dir] = 0.0;
    return;
  }
} test_case_0;

static class test_case_1_t : public test_case_for_two_phase_flows_t
{
  inline double rr_2(const double* xyz) const { return SQRSUMD(xyz[0] - (time - 0.5), xyz[1] - (time - 0.5), xyz[2] - (time - 0.5)); }
  inline double drr_2_dx(const double* xyz) const { return 2.0*(xyz[0] - (time - 0.5)); }
  inline double drr_2_dy(const double* xyz) const { return 2.0*(xyz[1] - (time - 0.5)); }
  inline double drr_2_dt(const double* xyz) const { return -2.0*(xyz[0] - (time - 0.5)) - 2.0*(xyz[1] - (time - 0.5)); }
  inline double rr(const double* xyz) const { return sqrt(rr_2(xyz)); }
public:
  test_case_1_t() : test_case_for_two_phase_flows_t()
  {
    mu_m  = 0.01;
    rho_m = 0.1;
    mu_p  = 0.1;
    rho_p = 1.0;
    surface_tension = 0.0; // no surface tension effect in here but a surface-defined interface stress
    surface_tension_is_constant = true; // no need to bother with the calculation of Marangoni forces
    nonzero_mass_flux = false;
    levelset_cf_is_signed_distance = true;
    pressure_is_floating = true;

    static_interface = false;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim]     = -2.0;
      domain.xyz_max[dim]     = +2.0;
      domain.periodicity[dim] = 0;
    }
    t_end = 1.0;
    pressure_wall_bc.setWallTypes(neumann_cf); // let's use neumann in this case
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      velocity_wall_bc[dir].setWallTypes(dirichlet_cf);

    description =
        std::string("* domain = [-2.0, 2.0] X [-2.0, 2.0] \n")
        + std::string("* no periodicity \n")
        + std::string("* levelset = sqrt((x - (t - 0.5))^2 + (y - (t - 0.5))^2) - 1\n")
        + std::string("* mu_m = 0.01; \n")
        + std::string("* mu_p = 0.1; \n")
        + std::string("* rho_m = 0.1; \n")
        + std::string("* rho_p = 1.00; \n")
        + std::string("* surface_tension = 0.00; \n")
        + std::string("* u_m  =  (y - (t - 0.5))*(r^2 - 1) + 1; \n")
        + std::string("* v_m  = -(x - (t - 0.5))*(r^2 - 1) + 1; \n")
        + std::string("* u_p  = 1; \n")
        + std::string("* v_p  = 1; \n")
        + std::string("* pressure_minus = 2.0 - r^2; \n")
        + std::string("* pressure_plus  = 0.0; \n")
        + std::string("* (r == sqrt((x - (t - 0.5))^2 + (y - (t - 0.5))^2)); \n")
        + std::string("* Dirichlet BC on all walls for velocity, neumann for pressure; \n")
        + std::string("* Validation test case 1 in 2D, with moving interface");
    test_name = "test_case_1";
    max_v_magnitude_0 = sqrt(1.87601);
    max_surface_tension_0 = EPS;
  }

  inline double levelset_function(const double *xyz) const
  {
    return rr(xyz) - 1.0;
  }
  // negative velocity field
  inline double velocity_minus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return (xyz[1] - time + 0.5)*(rr_2(xyz) - 1.0) + 1;
      break;
    case dir::y:
      return -(xyz[0] - time + 0.5)*(rr_2(xyz) - 1.0) + 1;
      break;
    default:
      throw std::invalid_argument("test_case_1_t::velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double time_derivative_velocity_minus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return -(rr_2(xyz) - 1.0) + (xyz[1] - time + 0.5)*drr_2_dt(xyz);
      break;
    case dir::y:
      return +(rr_2(xyz) - 1.0) - (xyz[0] - time + 0.5)*drr_2_dt(xyz);
      break;
    default:
      throw std::invalid_argument("test_case_1_t::time_derivative_velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double first_derivative_velocity_minus(const u_char& dir, const u_char& der, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
    {
      switch (der) {
      case dir::x:
        return (xyz[1] - time + 0.5)*drr_2_dx(xyz);
        break;
      case dir::y:
        return (rr_2(xyz) - 1.0) + (xyz[1] - time + 0.5)*drr_2_dy(xyz);
        break;
      default:
        throw std::invalid_argument("test_case_1_t::first_derivative_velocity_minus: unknown cartesian direction");
        break;
      }
    }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return -(rr_2(xyz) - 1.0) - (xyz[0] - time + 0.5)*drr_2_dx(xyz);
        break;
      case dir::y:
        return -(xyz[0] - time + 0.5)*drr_2_dy(xyz);
        break;
      default:
        throw std::invalid_argument("test_case_1_t::first_derivative_velocity_minus: unknown cartesian direction");
        break;
      }
      break;
    default:
      throw std::invalid_argument("test_case_1_t::first_derivative_velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double laplacian_velocity_minus(const u_char &dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return 8.0*(xyz[1] - time + 0.5);
      break;
    case dir::y:
      return -8.0*(xyz[0] - time + 0.5);
      break;
    default:
      throw std::invalid_argument("test_case_1_t::laplacian_velocity_minus: unknown velocity component");
      break;
    }
  }
  // positive velocity field
  inline double velocity_plus(const u_char&, const double *) const
  {
    return 1.0;
  }
  inline double time_derivative_velocity_plus(const u_char&, const double *) const
  {
    return 0.0;
  }
  inline double first_derivative_velocity_plus(const u_char& , const u_char& , const double *) const
  {
    return 0.0;
  }

  inline double laplacian_velocity_plus(const u_char &, const double *) const
  {
    return 0.0;
  }
  // mass flux
  inline double local_mass_flux(const double *) const
  {
    return 0.0;
  }

  // negative pressure field
  inline double pressure_minus(const double *xyz) const
  {
    return 2.0 - rr_2(xyz);
  }
  inline double first_derivative_pressure_minus(const u_char& der, const double * xyz) const
  {
    switch (der) {
    case dir::x:
      return -drr_2_dx(xyz);
      break;
    case dir::y:
      return -drr_2_dy(xyz);
      break;
    default:
      throw std::invalid_argument("test_case_1_t::first_derivative_pressure_minus: unknown cartesian direction");
      break;
    };
  }

  // positive pressure field
  inline double pressure_plus(const double *) const
  {
    return 0.0;
  }
  inline double first_derivative_pressure_plus(const u_char&, const double *) const
  {
    return 0.0;
  }

  // useful if nonconstant surface tension
  inline double local_surface_tension(const double *) const
  {
    return surface_tension;
  }
  inline void gradient_surface_tension(const double *, double grad_surf_tension[P4EST_DIM]) const
  {
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      grad_surf_tension[dir] = 0.0;
    return;
  }
} test_case_1;

static class test_case_2_t : public test_case_for_two_phase_flows_t
{
public:
  test_case_2_t() : test_case_for_two_phase_flows_t()
  {
    mu_m  = 0.1;
    rho_m = 0.1;
    mu_p  = 1.0;
    rho_p = 1.0;
    surface_tension = 0.1;
    surface_tension_is_constant = true; // no need to bother with the calculation of Marangoni forces
    nonzero_mass_flux = false;
    levelset_cf_is_signed_distance = false;
    pressure_is_floating = true;

    static_interface = true;
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      domain.xyz_min[dim]     = -M_PI/3.0;
      domain.xyz_max[dim]     = +4.0*M_PI/3.0;
      domain.periodicity[dim] = 0;
    }
    t_end = M_PI;
    pressure_wall_bc.setWallTypes(neumann_cf); // let's use neumann in this case
    for(u_char dir = 0; dir < P4EST_DIM; dir++)
      velocity_wall_bc[dir].setWallTypes(dirichlet_cf);

    description =
        std::string("* domain = [-\\pi/3.0, 4.0\\pi/3.0] X [-\\pi/3.0, 4.0\\pi/3.0] \n")
        + std::string("* no periodicity \n")
        + std::string("* levelset = 0.1 outside of [0, \\pi] x [0, \\pi], 0.1 - sin(x)*sin(y) inside (+ needs reinitialization)\n")
        + std::string("* mu_m = 0.1; \n")
        + std::string("* mu_p = 0.1; \n")
        + std::string("* rho_m = 1.00; \n")
        + std::string("* rho_p = 1.00; \n")
        + std::string("* surface_tension = 0.1; \n")
        + std::string("* u_m  =  sin(x)*cos(y)*sin(t); \n")
        + std::string("* v_m  = -cos(x)*sin(y)*sin(t); \n")
        + std::string("* u_p  = u_m; \n")
        + std::string("* v_p  = v_p; \n")
        + std::string("* pressure_minus = 0.0; \n")
        + std::string("* pressure_plus  = 0.0; \n")
        + std::string("* Dirichlet BC on all walls for velocity, neumann for pressure; \n")
        + std::string("* Validation test case 2 in 2D (from Maxime's paper)");
    test_name = "test_case_2";
    max_v_magnitude_0 = EPS;
    max_surface_tension_0 = surface_tension;
  }

  inline double levelset_function(const double *xyz) const
  {
    return 0.1 - (xyz[0] > 0.0 && xyz[0] < M_PI && xyz[1] > 0.0 && xyz[1] < M_PI ? sin(xyz[0])*sin(xyz[1]) : 0.0);
  }
  // negative velocity field
  inline double velocity_minus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return sin(xyz[0])*cos(xyz[1])*sin(time);
      break;
    case dir::y:
      return -cos(xyz[0])*sin(xyz[1])*sin(time);
      break;
    default:
      throw std::invalid_argument("test_case_2_t::velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double time_derivative_velocity_minus(const u_char& dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return sin(xyz[0])*cos(xyz[1])*cos(time);
      break;
    case dir::y:
      return -cos(xyz[0])*sin(xyz[1])*cos(time);
      break;
    default:
      throw std::invalid_argument("test_case_2_t::time_derivative_velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double first_derivative_velocity_minus(const u_char& dir, const u_char& der, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
    {
      switch (der) {
      case dir::x:
        return cos(xyz[0])*cos(xyz[1])*sin(time);
        break;
      case dir::y:
        return -sin(xyz[0])*sin(xyz[1])*sin(time);
        break;
      default:
        throw std::invalid_argument("test_case_2_t::first_derivative_velocity_minus: unknown cartesian direction");
        break;
      }
    }
      break;
    case dir::y:
      switch (der) {
      case dir::x:
        return sin(xyz[0])*sin(xyz[1])*sin(time);
        break;
      case dir::y:
        return -cos(xyz[0])*cos(xyz[1])*sin(time);
        break;
      default:
        throw std::invalid_argument("test_case_2_t::first_derivative_velocity_minus: unknown cartesian direction");
        break;
      }
      break;
    default:
      throw std::invalid_argument("test_case_2_t::first_derivative_velocity_minus: unknown velocity component");
      break;
    }
  }
  inline double laplacian_velocity_minus(const u_char &dir, const double *xyz) const
  {
    switch (dir) {
    case dir::x:
      return -2.0*sin(xyz[0])*cos(xyz[1])*sin(time);
      break;
    case dir::y:
      return +2.0*cos(xyz[0])*sin(xyz[1])*sin(time);
      break;
    default:
      throw std::invalid_argument("test_case_2_t::laplacian_velocity_minus: unknown velocity component");
      break;
    }
  }
  // positive velocity field
  inline double velocity_plus(const u_char& dir, const double *xyz) const
  {
    return velocity_minus(dir, xyz);
  }
  inline double time_derivative_velocity_plus(const u_char& dir, const double* xyz) const
  {
    return time_derivative_velocity_minus(dir, xyz);
  }
  inline double first_derivative_velocity_plus(const u_char& dir, const u_char& der, const double *xyz) const
  {
    return first_derivative_velocity_minus(dir, der, xyz);
  }

  inline double laplacian_velocity_plus(const u_char &dir, const double *xyz) const
  {
    return laplacian_velocity_minus(dir, xyz);
  }
  // mass flux
  inline double local_mass_flux(const double *) const
  {
    return 0.0;
  }

  // negative pressure field
  inline double pressure_minus(const double *) const
  {
    return 0.0;
  }
  inline double first_derivative_pressure_minus(const u_char&, const double *) const
  {
    return 0.0;
  }

  // positive pressure field
  inline double pressure_plus(const double *) const
  {
    return 0.0;
  }
  inline double first_derivative_pressure_plus(const u_char&, const double *) const
  {
    return 0.0;
  }

  // useful if nonconstant surface tension
  inline double local_surface_tension(const double *) const
  {
    return surface_tension;
  }
  inline void gradient_surface_tension(const double *, double grad_surf_tension[P4EST_DIM]) const
  {
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      grad_surf_tension[dir] = 0.0;
    return;
  }
} test_case_2;
#else

#endif

static class list_of_test_problems_for_two_phase_flows_t
{
private:
  std::vector<test_case_for_two_phase_flows_t*> list_of_test_problems;
public:
  list_of_test_problems_for_two_phase_flows_t()
  {
    list_of_test_problems.clear();
#ifdef P4_TO_P8
#else
    list_of_test_problems.push_back(&test_case_0);
    list_of_test_problems.push_back(&test_case_1);
    list_of_test_problems.push_back(&test_case_2);
#endif
  }

  inline std::string get_description_of_tests() const
  {
    std::string description;
    for (size_t k = 0; k < list_of_test_problems.size(); ++k) {
      description += std::to_string(k) + ":\n" + list_of_test_problems[k]->get_description() + (k < list_of_test_problems.size() - 1 ? "\n" : ".");
    }
    return  description;
  }

  inline test_case_for_two_phase_flows_t* operator[](size_t k) const
  {
    if(k >= list_of_test_problems.size())
      throw std::invalid_argument("list_of_test_problems_for_two_phase_flows_t::operator[]: problem index is too large. Max problem index is " + std::to_string(list_of_test_problems.size() - 1));
    return list_of_test_problems[k];
  }

} list_of_test_problems_for_two_phase_flows;

#endif // TWO_PHASE_FLOWS_TESTS_H
