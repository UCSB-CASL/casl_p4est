#ifndef MY_P4EST_SHS_CHANNEL_H
#define MY_P4EST_SHS_CHANNEL_H

#include <cmath>
#include <stdexcept>
#include <random>

#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#else
#include <src/my_p4est_navier_stokes.h>
#endif

typedef enum
{
  constant_pressure_gradient,
  constant_mass_flow,
  undefined_flow_condition
} flow_setting;

static double my_fmod(const double& num, const double& denom)
{
  return (num - std::floor(num/denom)*denom);
}

class my_p4est_shs_channel_t : public CF_DIM
{
  // parameters defining the channel
  const mpi_environment_t& mpi;
  const my_p4est_brick_t* brick;
  double pitch, gas_frac;
#ifdef P4_TO_P8
  bool spanwise;
#endif
  int8_t max_lvl;
#ifdef P4EST_ENABLE_DEBUG
  bool is_configured;
#endif

  // Boundary condition stuff
  BoundaryConditionsDIM bc_v[P4EST_DIM];
  BoundaryConditionsDIM bc_p;

  zero_cf_t zero_value;

  CF_DIM *bc_wall_value_p_neumann;
  CF_DIM *bc_wall_value_u_neumann, *bc_wall_value_u_dirichlet;
  CF_DIM *bc_wall_value_v_dirichlet;
#ifdef P4_TO_P8
  CF_DIM *bc_wall_value_w_neumann, *bc_wall_value_w_dirichlet;
#endif

  struct BCWALLTYPE_P : WallBCDIM {
    BoundaryConditionType operator()(DIM(double, double, double)) const
    {
      return NEUMANN;
    }
  } bc_wall_type_p;

  struct BCWALLVALUE_P : CF_DIM {
    my_p4est_shs_channel_t * const owner;
    BCWALLVALUE_P(my_p4est_shs_channel_t * env) : owner(env){}
    double operator()(DIM(double x, double y, double z)) const
    {
      return (*owner->bc_wall_value_p_neumann)(DIM(x, y, z));
    }
  } bc_wall_value_p;

  struct BCWALLTYPE_U : WallBCDIM {
    const my_p4est_shs_channel_t *env;
    BCWALLTYPE_U(const my_p4est_shs_channel_t* env_) : env(env_) {}
    BoundaryConditionType operator()(DIM(double x, double y, double z)) const
    {
      return (env->is_ridge(DIM(x, y, z)) ? DIRICHLET : NEUMANN);
    }
  } bc_wall_type_u;

  struct BCWALLVALUE_U : CF_DIM {
    my_p4est_shs_channel_t * const owner;
    BCWALLVALUE_U(my_p4est_shs_channel_t * env) : owner(env){}
    double operator()(DIM(double x, double y, double z)) const
    {
      return (owner->bc_wall_type_u(DIM(x, y, z)) == DIRICHLET ? (*owner->bc_wall_value_u_dirichlet)(DIM(x, y, z)) : (*owner->bc_wall_value_u_neumann)(DIM(x, y, z)));
    }
  } bc_wall_value_u;

  struct BCWALLTYPE_V : WallBCDIM {
    BoundaryConditionType operator()(DIM(double, double, double)) const
    {
      return DIRICHLET;
    }
  } bc_wall_type_v;

  struct BCWALLVALUE_V : CF_DIM {
    my_p4est_shs_channel_t * const owner;
    BCWALLVALUE_V(my_p4est_shs_channel_t * env) : owner(env){}
    double operator()(DIM(double x, double y, double z)) const
    {
      return (*owner->bc_wall_value_v_dirichlet)(DIM(x, y, z));
    }
  } bc_wall_value_v;

#ifdef P4_TO_P8
  struct BCWALLTYPE_W : WallBC3D
  {
    const my_p4est_shs_channel_t* env;
    BCWALLTYPE_W(const my_p4est_shs_channel_t* env_) : env(env_){ }
    BoundaryConditionType operator()(double x, double y, double z) const
    {
      return (env->is_ridge(x, y, z) ? DIRICHLET : NEUMANN);
    }
  } bc_wall_type_w;

  struct BCWALLVALUE_W : CF_DIM {
    my_p4est_shs_channel_t * const owner;
    BCWALLVALUE_W(my_p4est_shs_channel_t * env) : owner(env){}
    double operator()(DIM(double x, double y, double z)) const
    {
      return (owner->bc_wall_type_w(DIM(x, y, z)) == DIRICHLET ? (*owner->bc_wall_value_w_dirichlet)(DIM(x, y, z)) : (*owner->bc_wall_value_w_neumann)(DIM(x, y, z)));
    }
  } bc_wall_value_w;
#endif

  // For the analytical solution
  int num_terms;
  std::vector<double> coeff;
  bool coeff_are_set;
  std::vector<double> vec_of_ks;
  std::vector<double> vec_of_prefactors;
  std::vector<double> vec_of_tanhs;
  bool vecs_are_set;

#ifdef P4_TO_P8
  inline double normalized_z(const double& z) const
  {
    return my_fmod((z - brick->xyz_max[2]), pitch);
  }
#endif

  inline double offset() const
  {
#ifdef P4_TO_P8
    return (!spanwise ? 0.1*width()/(brick->nxyztrees[2]*(1 << max_lvl)) : 0.5*length()/(brick->nxyztrees[0]*(1 << max_lvl)));
#else
    return 0.5*length()/(brick->nxyztrees[0]*(1 << max_lvl));
#endif
  }

  inline double distance_to_ridge(DIM(const double& x, const double& y, const double& z)) const
  {
#ifdef P4_TO_P8
    if(!spanwise)
      return  sqrt(SQR(MIN(brick->xyz_max[1] - y, y - brick->xyz_min[1])) + (is_ridge(x, y, z) ? 0.0 : SQR(MIN(normalized_z(z) - offset(), pitch*gas_frac - offset() - normalized_z(z)))));
#endif
    return    sqrt(SQR(MIN(brick->xyz_max[1] - y, y - brick->xyz_min[1])) + (is_ridge(DIM(x, y, z)) ? 0.0 : SQR(MIN(my_fmod((x - brick->xyz_min[0] - offset()), pitch), gas_frac*pitch - my_fmod((x - brick->xyz_min[0] - offset()), pitch)))));
  }

  inline double k_(const int& n) const
  {
    return 2.0*M_PI*n/pitch_to_delta();
  }

  inline double beta_(const double & k) const
  {
#ifdef P4_TO_P8
    if(!spanwise)
      return k*tanh(k);
#endif
    return 2.0*k*SQR(1.0 - exp(-2.0*k))/(1.0 - 4.0*k*exp(-2.0*k) - exp(-4.0*k));
  }

  inline void compute_vecs()
  {
    if(vecs_are_set)
      return;
#ifdef P4_TO_P8
    if(!spanwise)
    {
      for(int n = 0; n < num_terms; ++n)
      {
        vec_of_ks[n] = k_(n);
        vec_of_prefactors[n] = 1.0/(1.0 + exp(-2.0*vec_of_ks[n]));
      }
      vecs_are_set = true;
      return;
    }
#endif
    for(int n = 0; n < num_terms; ++n)
    {
      vec_of_ks[n] = k_(n);
      vec_of_tanhs[n] = tanh(vec_of_ks[n]);
      if(n == 0)  vec_of_prefactors[0] = 0.0;
      else        vec_of_prefactors[n] = (1.0 + exp(-2.0*vec_of_ks[n]))/(-1.0 + 4.0*vec_of_ks[n]*exp(-2.0*vec_of_ks[n]) + exp(-4.0*vec_of_ks[n]));
    }
    vecs_are_set = true;
    return;
  }

  inline void compute_series_coeff()
  {
    if(coeff_are_set)
      return;
    if(!vecs_are_set)
      compute_vecs();
    P4EST_ASSERT(vecs_are_set);
    if(mpi.rank() == 0)
    {
      double *vec_of_betam1 = new double[num_terms];
      double *vec_of_sines  = new double[3*(num_terms - 1)];
      for(int i = 0; i < num_terms;        ++i) vec_of_betam1[i] = beta_(vec_of_ks[i]) - 1.0;
      for(int i = 0; i < 3*(num_terms - 1);++i) vec_of_sines[i]  = sin((i - (num_terms - 2))*M_PI*gas_frac);

      PetscErrorCode ierr;
      Mat matrix;
      Vec rhs, solution;
      MatScalar *matrix_entries;
      PetscScalar *rhs_entries;
      ierr = MatCreate(MPI_COMM_SELF, &matrix); CHKERRXX(ierr);
      ierr = MatSetSizes(matrix, (PetscInt) num_terms, (PetscInt) num_terms, (PetscInt) num_terms, (PetscInt) num_terms); CHKERRXX(ierr);
      ierr = MatSetType(matrix, MATSEQDENSE); CHKERRXX(ierr);
      // --> COLUMN-MAJOR order (Petsc will use Fortran-based Blas/Lapack under the hood!)
      // the (i, j) element of matrix will be matrix_entries[j + num_terms*i]
      ierr = MatSeqDenseSetPreallocation(matrix, NULL); CHKERRXX(ierr);
      // create rhs and solution
      ierr = VecCreateSeq(PETSC_COMM_SELF, (PetscInt) num_terms, &rhs); CHKERRXX(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, (PetscInt) num_terms, &solution); CHKERRXX(ierr);
      // get accessors
      ierr = MatDenseGetArray(matrix, &matrix_entries); CHKERRXX(ierr);
      ierr = VecGetArray(rhs, &rhs_entries); CHKERRXX(ierr);
      for(PetscInt j = 0; j < (PetscInt) num_terms; ++j) // A_{i, j} // j <--> n
        for(PetscInt i = 0; i < (PetscInt) num_terms; ++i) // i <--> m
        {
          if(j == 0)
          {
            if(i == 0)
            {
              rhs_entries[i] = gas_frac;
              matrix_entries[i + num_terms*j] = 1.0 - gas_frac;
            }
            else
            {
              // n == 0, m != 0
              rhs_entries[i] = vec_of_sines[i + (num_terms - 2)]/(i*M_PI);
              matrix_entries[i + num_terms*j] = -vec_of_sines[i + num_terms - 2]/(i*M_PI); // first column
            }
          }
          else
          {
            if(i ==  0) // m == 0, n != 0
              matrix_entries[i + num_terms*j] = vec_of_betam1[j]*vec_of_sines[j + num_terms - 2]/(j*M_PI);
            else if(i == j) // m == n
              matrix_entries[i + num_terms*j] = 0.5*(1.0 + vec_of_betam1[i]*(gas_frac + vec_of_sines[2*i + num_terms - 2]/(2.0*i*M_PI)));
            else
              matrix_entries[i + num_terms*j] = vec_of_betam1[j]*(vec_of_sines[i - j + num_terms - 2]/(i - j) + vec_of_sines[i + j + num_terms - 2]/(i + j))/(2.0*M_PI);
          }
        }

      ierr = VecRestoreArray(rhs, &rhs_entries); CHKERRXX(ierr);
      ierr = MatDenseRestoreArray(matrix, &matrix_entries); CHKERRXX(ierr);
      ierr = MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
      ierr = MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

      // now solve!
      KSP solver;
      PC pc;
      ierr = KSPCreate(MPI_COMM_SELF, &solver); CHKERRXX(ierr);
      ierr = KSPSetType(solver, KSPPREONLY); CHKERRXX(ierr);
      ierr = KSPGetPC(solver, &pc); CHKERRXX(ierr);
      ierr = PCSetType(pc, PCLU); CHKERRXX(ierr);
      ierr = KSPSetOperators(solver, matrix, matrix, NULL); CHKERRXX(ierr);
      ierr = KSPSolve(solver, rhs, solution); CHKERRXX(ierr);

      // copy solution into coeff
      const PetscScalar *solution_read_p;
      ierr = VecGetArrayRead(solution, &solution_read_p); CHKERRXX(ierr);
      for(PetscInt i = 0; i < (PetscInt) num_terms; ++i)
        coeff[i] = solution_read_p[i];
      ierr = VecRestoreArrayRead(solution, &solution_read_p); CHKERRXX(ierr);

      // clean up
      ierr = VecDestroy(rhs); CHKERRXX(ierr);
      ierr = VecDestroy(solution); CHKERRXX(ierr);
      ierr = MatDestroy(matrix); CHKERRXX(ierr);
      ierr = KSPDestroy(solver); CHKERRXX(ierr);

      delete [] vec_of_betam1;
      delete [] vec_of_sines;

    }
    int mpiret = MPI_Bcast(coeff.data(), num_terms, MPI_DOUBLE, 0, MPI_COMM_WORLD); SC_CHECK_MPI(mpiret);
    coeff_are_set = true;
    return;
  }

#ifdef DEBUG
  inline void check_input_coord(const double xyz_mod[P4EST_DIM]) const
  {
    if(fabs(xyz_mod[1]) > 1.0)
      throw std::invalid_argument("my_p4est_shs_channel_t: the y-coordinate must be contained in [-1, 1].");
#ifdef P4_TO_P8
    if(!spanwise && fabs(xyz_mod[2]) > 0.5*pitch/delta())
      throw std::invalid_argument("my_p4est_shs_channel_t: the z-coordinate must be contained in [-0.5*pitch/delta, 0.5*pitch/delta].");
#endif
    if(fabs(xyz_mod[0]) > 0.5*pitch/delta())
      throw std::invalid_argument("my_p4est_shs_channel_t: the x-coordinate must be contained in [-0.5*pitch/delta, 0.5*pitch/delta].");
  }
#endif

  inline double v_x(const double xyz_mod[P4EST_DIM]) const
  {
#ifdef DEBUG
    check_input_coord(xyz_mod);
#endif
    // The first following conditional return avoids spurious negative values of the exact series solution for v_x close to the ridge-gap transition, due to generalized Gibbs phenomenon at a kink
#ifdef P4_TO_P8
    const bool on_ridge = (!spanwise && fabs(xyz_mod[2]) >= 0.5*gas_frac*pitch) || (spanwise && fabs(xyz_mod[0]) >= 0.5*gas_frac*pitch);
#else
    const bool on_ridge = (fabs(xyz_mod[0]) >= 0.5*gas_frac*pitch);
#endif
    if(fabs(fabs(xyz_mod[1]) - 1.0) < EPS*1.0 && on_ridge)
      return 0.0; // Note h = 1.0

    double v_x_val = 0.5*(1.0 - SQR(xyz_mod[1])) + coeff[0];
#ifdef P4_TO_P8
    if(!spanwise)
    {
      for(int n = 1; n < num_terms; ++n)
        v_x_val += coeff[n]*vec_of_prefactors[n]*(exp(-vec_of_ks[n]*(1.0 - xyz_mod[1])) + exp(-vec_of_ks[n]*(1.0 + xyz_mod[1])))*cos(vec_of_ks[n]*xyz_mod[2]);

      return v_x_val;
    }
#endif
    for(int n = 1; n < num_terms; ++n)
      v_x_val += coeff[n]*vec_of_prefactors[n]*
          ((vec_of_ks[n] - vec_of_tanhs[n])*(exp(-vec_of_ks[n]*(1.0 - xyz_mod[1])) + exp(-vec_of_ks[n]*(1.0 + xyz_mod[1]))) - vec_of_ks[n]*vec_of_tanhs[n]*xyz_mod[1]*(exp(-vec_of_ks[n]*(1.0 - xyz_mod[1])) - exp(-vec_of_ks[n]*(1.0 + xyz_mod[1]))))*
          cos(vec_of_ks[n]*xyz_mod[0]);
    return v_x_val;
  }

  inline double v_y(const double xyz_mod[P4EST_DIM]) const
  {
#ifdef DEBUG
    check_input_coord(xyz_mod);
#endif
    double v_y_val = 0.0;
#ifdef P4_TO_P8
    if(!spanwise)
      return v_y_val;
#endif
    for(int n = 1; n < num_terms; ++n)
      v_y_val += coeff[n]*vec_of_prefactors[n]*vec_of_ks[n]*
          ((exp(-vec_of_ks[n]*(1.0 - xyz_mod[1])) - exp(-vec_of_ks[n]*(1.0 + xyz_mod[1]))) - vec_of_tanhs[n]*xyz_mod[1]*(exp(-vec_of_ks[n]*(1.0 - xyz_mod[1])) + exp(-vec_of_ks[n]*(1.0 + xyz_mod[1]))))*sin(vec_of_ks[n]*xyz_mod[0]);
    return v_y_val;
  }

#ifdef P4_TO_P8
#ifdef DEBUG
  inline double v_z(const double xyz_mod[P4EST_DIM]) const
#else
  inline double v_z(const double*) const
#endif
  {
#ifdef DEBUG
    check_input_coord(xyz_mod);
#endif
    return 0.0;
  }
#endif

  // Modular transfomation used in the following three functions
  // { x_mod = (fmod(x - xmin + 0.5*pitch*(1 - GF), pitch) - 0.5*pitch)/delta --> (maps x \in [xmin, xmax] to [-0.5*pitch/delta, 0.5*pitch/delta] with all gap centers being mapped to zero}
  // { y_mod = (y - 0.5*(ymin + ymax))/delta                                  --> (maps y \in [ymin, ymax] to [-1              , 1]               with all gap centers being mapped to zero}
  // { z_mod = (fmod(z - zmin + 0.5*pitch*(1 - GF), pitch) - 0.5*pitch)/delta --> (maps z \in [zmin, zmax] to [-0.5*pitch/delta, 0.5*pitch/delta] with all gap centers being mapped to zero}
  inline void normalize_coordinates(double xyz_mod[P4EST_DIM], const double xyz[P4EST_DIM]) const
  {
// when we assumed a channel of height 2 centered at the origin, it was:
//    xyz_mod[0] = my_fmod(xyz[0] + 0.5*(length_to_delta() + pitch*(1.0 - gas_frac)), pitch) - 0.5*pitch;
//    xyz_mod[1] = xyz[1];
//    xyz_mod[2] = my_fmod(xyz[2] + 0.5*(width_to_delta()  + pitch*(1.0 - gas_frac)), pitch) - 0.5*pitch;
    // for any general brick, it must be
    xyz_mod[0] = (my_fmod(xyz[0] - brick->xyz_min[0] + 0.5*pitch*(1.0 - gas_frac), pitch) - 0.5*pitch)/delta();
    xyz_mod[1] = (xyz[1] - 0.5*(brick->xyz_min[1] + brick->xyz_max[1]))/delta();
#ifdef P4_TO_P8
    xyz_mod[2] = (my_fmod(xyz[2] - brick->xyz_min[2] + 0.5*pitch*(1.0 - gas_frac), pitch) - 0.5*pitch)/delta();
#endif
    return;
  }

  inline void set_num_terms(const int& nn_)
  {
    if(num_terms == nn_)
      return;
    num_terms = nn_;
    vec_of_ks.resize(num_terms);
    vec_of_prefactors.resize(num_terms);
    vec_of_tanhs.resize(num_terms);
    coeff.resize(num_terms);
    vecs_are_set  = false;
    coeff_are_set = false;
  }

  void check_resolution_pitch_and_gas_fraction() const
  {
    P4EST_ASSERT(is_configured);
#ifdef P4_TO_P8
    const double dimension_tranverse_to_grooves = (spanwise ? length() : width());
    const unsigned char dir_tranverse = (spanwise ? dir::x : dir::z);
#else
    const double dimension_tranverse_to_grooves = length();
    const unsigned char dir_tranverse = dir::x;
#endif
    if(pitch <= 0.0 || pitch > dimension_tranverse_to_grooves)
      throw std::invalid_argument("my_p4est_shs_channel_t::check_resolution_pitch_and_gas_fraction(...): the pitch must be strictly positive and smaller than or equal to the dimension of the domain in the direction tranverse to the grooves.");

    if(fabs(dimension_tranverse_to_grooves/pitch - (int) (dimension_tranverse_to_grooves/pitch)) > 1e-6)
      throw std::invalid_argument("my_p4est_shs_channel_t::check_resolution_pitch_and_gas_fraction(...): the dimension of the domain in the direction tranverse to the grooves MUST be a multiple of the pitch to satisfy periodicity.");

    if(gas_frac < 0.0 || gas_frac > 1.0)
      throw std::invalid_argument("my_p4est_shs_channel_t::check_resolution_pitch_and_gas_fraction(...): the gas fraction must be nonnegative and no greater than 1.0.");

    if(max_lvl < 0)
      throw std::invalid_argument("my_p4est_shs_channel_t::check_resolution_pitch_and_gas_fraction(...): the maximum level of refinement must be nonnegative.");

    double nb_finest_cell_in_groove =  pitch*gas_frac/(dimension_tranverse_to_grooves/(brick->nxyztrees[dir_tranverse]*(1 << max_lvl)));
    double nb_finest_cell_in_ridge  =  pitch*(1.0 - gas_frac)/(dimension_tranverse_to_grooves/(brick->nxyztrees[dir_tranverse]*(1 << max_lvl)));

    if((fabs(nb_finest_cell_in_groove - (int) nb_finest_cell_in_groove) > 1e-6) || (fabs(nb_finest_cell_in_ridge - (int) nb_finest_cell_in_ridge) > 1e-6))
      throw std::invalid_argument("my_p4est_shs_channel_t::check_resolution_pitch_and_gas_fraction(...): the finest grid cells do not capture the groove and/or the ridge (subcell resolution for boundary condition would be required).");
  }

  inline double Re_tau_from_Re_b(const double& Re_b) const
  {
    return sqrt(Re_b/(1.0/3.0 + coeff[0]));
  }

  inline double Re_b_from_Re_tau(const double& Re_tau) const
  {
    return SQR(Re_tau)*(1.0/3.0 + coeff[0]);
  }

  inline double get_corresponding_Re_tau(const flow_setting& flow_setup, const double& Re) const
  {
    double Re_tau;
    switch (flow_setup) {
    case constant_pressure_gradient:
      Re_tau = Re; // if constant_pressure_gradient, Re = Re_tau
      break;
    case constant_mass_flow:
      Re_tau = Re_tau_from_Re_b(Re); // if constant_mass_flow, Re = Re_b
      break;
    default:
      throw std::invalid_argument("my_p4est_shs_channel_t::get_corresponding_Re_tau(): unknown flow setup");
      break;
    }
    return Re_tau;
  }

  inline double get_corresponding_Re_b(const flow_setting& flow_setup, const double& Re) const
  {
    double Re_b;
    switch (flow_setup) {
    case constant_mass_flow:
      Re_b = Re; // if constant_mass_flow, Re = Re_b
      break;
    case constant_pressure_gradient:
      Re_b = Re_b_from_Re_tau(Re); // if constant_pressure_gradient, Re = Re_tau
      break;
    default:
      throw std::invalid_argument("my_p4est_shs_channel_t::get_corresponding_Re_b(): unknown flow setup");
      break;
    }
    return Re_b;
  }

public:
  my_p4est_shs_channel_t(const mpi_environment_t& mpi_) :
    mpi(mpi_),
    bc_wall_value_p(this),
    bc_wall_type_u(this), bc_wall_value_u(this),
    bc_wall_value_v(this)
  ONLY3D(COMMA bc_wall_type_w(this) COMMA bc_wall_value_w(this))
  {
    max_lvl = -1;
    pitch = gas_frac = -1.0;
    num_terms = -1;
    coeff.clear(); coeff_are_set = false;
    vec_of_ks.clear();
    vec_of_prefactors.clear();
    vec_of_tanhs.clear();
    vecs_are_set  = false;
#ifdef P4EST_ENABLE_DEBUG
    is_configured = false;
#endif

    // (note : all boundary condition must always be 0 (whether it is no slip or free slip), EXCEPT FOR validation purposes
    bc_wall_value_p_neumann = &zero_value;
    bc_wall_value_u_dirichlet = &zero_value; bc_wall_value_u_neumann = &zero_value;
    bc_wall_value_v_dirichlet = &zero_value;
#ifdef P4_TO_P8
    bc_wall_value_w_dirichlet = &zero_value; bc_wall_value_w_neumann = &zero_value;
#endif


    bc_p.setWallTypes(bc_wall_type_p);    bc_p.setWallValues(bc_wall_value_p);
    bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(bc_wall_value_u);
    bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(bc_wall_value_v);
#ifdef P4_TO_P8
    bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(bc_wall_value_w);
#endif
  }

  inline void configure(const my_p4est_brick_t *brick_, DIM(const double& pitch_, const double& gas_fraction, const bool& spanwise_), const int8_t& max_level) // unconventional use of DIM(), but makes my life easier
  {
    brick = brick_;
    max_lvl = max_level;
#ifdef P4_TO_P8
    spanwise = spanwise_;
#endif
    pitch = pitch_;
    gas_frac = gas_fraction;

#ifdef P4EST_ENABLE_DEBUG
    is_configured = true;
#endif

    check_resolution_pitch_and_gas_fraction();
    return;
  }


  inline double length()          const { P4EST_ASSERT(is_configured); return brick->xyz_max[0] - brick->xyz_min[0];  }
  inline double height()          const { P4EST_ASSERT(is_configured); return brick->xyz_max[1] - brick->xyz_min[1];  }
#ifdef P4_TO_P8
  inline double width()           const { P4EST_ASSERT(is_configured); return brick->xyz_max[2] - brick->xyz_min[2];  }
#endif
  inline double delta()           const { P4EST_ASSERT(is_configured); return 0.5*height(); }
  inline double pitch_to_delta()  const { P4EST_ASSERT(is_configured); return pitch/delta(); }
  inline double get_pitch()       const { P4EST_ASSERT(is_configured); return pitch; }
  inline double GF()              const { P4EST_ASSERT(is_configured); return gas_frac; }
  inline int lmax()               const { P4EST_ASSERT(is_configured); return max_lvl; }

  inline double operator()(DIM(double x, double y, double z)) const
  {
    P4EST_ASSERT(is_configured);
#ifdef P4_TO_P8
    if(!spanwise)
      return -distance_to_ridge(DIM(x, y, z)) - (height()/brick->nxyztrees[1])/(1 << max_lvl); // negative definite everywhere but as close to 0 as possible close to no-slip walls
#endif
    (void) x; // to avoid "unused parameter" warning
    return -MIN(brick->xyz_max[1] - y, y - brick->xyz_min[1]) - (height()/brick->nxyztrees[1])/(1 << max_lvl); // negative definite everywhere but as close to 0 as possible close to (any) wall
  }

  inline bool is_ridge(DIM(const double& x, const double&, const double& z)) const
  {
    P4EST_ASSERT(is_configured);
#ifdef P4_TO_P8
    if(!spanwise)
      return (offset() >= normalized_z(z) || normalized_z(z) >= pitch*gas_frac - offset());
#endif
    return (my_fmod(x - brick->xyz_min[0] - offset(), pitch)/pitch >= gas_frac);
  }

  inline void solve_for_truncated_series(const int& num_terms_)
  {
    P4EST_ASSERT(is_configured);
    set_num_terms(num_terms_);
    compute_vecs();
    compute_series_coeff();
    P4EST_ASSERT(vecs_are_set && coeff_are_set);
    return;
  }

  inline void check_Reynolds(const flow_setting &flow_setup, const double& Re) const
  {
    double Re_tau;
    switch (flow_setup) {
    case constant_pressure_gradient:
      Re_tau = Re; // if constant_pressure_gradient, Re = Re_tau
      break;
    case constant_mass_flow:
      Re_tau = Re_tau_from_Re_b(Re); // if constant_mass_flow, Re = Re_b
      break;
    default:
      throw std::invalid_argument("my_p4est_shs_channel_t::check_Reynolds(): unknown flow setup");
      break;
    }
    // In laminar canonical channel flows, Re_c = SQR(Re_tau) ...
    bool print_warning_tranverse = SQR(Re_tau) > 0.1;
#ifdef P4_TO_P8
    print_warning_tranverse = print_warning_tranverse && spanwise;
    bool print_warning_long = !spanwise && SQR(Re_tau) > 1000.0;
#endif
    if(print_warning_tranverse)
    {
      PetscErrorCode ierr = PetscPrintf(mpi.comm(), "\n[WARNING]: The exact solution for tranverse grooves is valid only in the limit"
                                                     "of Re << 1, and your current Reynolds number is Re_c = SQR(Re_tau) = %f > 0.1. "
                                                     "Expect discrepancies.\n\n", SQR(Re_tau)); CHKERRXX(ierr);
    }
#ifdef P4_TO_P8
    if(print_warning_long)
    {
      PetscErrorCode ierr = PetscPrintf(mpi.comm(), "\n[WARNING]: The exact solution for longitudinal grooves is valid only for"
                                                     "steady laminar flow, and your current Reynolds number is Re_c = SQR(Re_tau) = %f > 1000."
                                                     "Expect discrepancies.\n\n", SQR(Re_tau)); CHKERRXX(ierr);
    }
#endif
    return;
  }

  inline void v_exact(const flow_setting& flow_setup, const double& Re, const my_p4est_navier_stokes_t* ns, const double xyz[P4EST_DIM], double velocity[P4EST_DIM]) const
  {
    P4EST_ASSERT(is_configured);
    P4EST_ASSERT(vecs_are_set && coeff_are_set);
    double xyz_mod[P4EST_DIM];
    normalize_coordinates(xyz_mod, xyz);

    const double Re_tau = get_corresponding_Re_tau(flow_setup, Re);
    const double velocity_scale =  SQR(Re_tau)*ns->get_nu()/delta();
    velocity[0] = velocity_scale*v_x(xyz_mod);
    velocity[1] = velocity_scale*v_y(xyz_mod);
#ifdef P4_TO_P8
    velocity[2] = velocity_scale*v_z(xyz_mod);
#endif
  }

  inline double v_exact(const unsigned char& dir, const flow_setting& flow_setup, const double& Re, const my_p4est_navier_stokes_t* ns, const double xyz[P4EST_DIM]) const
  {
    P4EST_ASSERT(is_configured);
    P4EST_ASSERT(vecs_are_set && coeff_are_set);
    double xyz_mod[P4EST_DIM];
    normalize_coordinates(xyz_mod, xyz);

    const double Re_tau = get_corresponding_Re_tau(flow_setup, Re);
    const double velocity_scale =  SQR(Re_tau)*ns->get_nu()/delta();
    switch (dir) {
    case dir::x:
      return velocity_scale*v_x(xyz_mod);
    case dir::y:
      return velocity_scale*v_y(xyz_mod);
#ifdef P4_TO_P8
    case dir::z:
      return velocity_scale*v_z(xyz_mod);
#endif
    default:
      throw std::invalid_argument("my_p4est_shs_channel_t::v_exact(): unknown Cartesian direction");
      break;
    }
  }

  inline BoundaryConditionsDIM* get_bc_on_velocity() { return bc_v;   }
  inline BoundaryConditionsDIM* get_bc_on_pressure() { return &bc_p;  }


  inline void set_bc_value_on_velocity(const CF_DIM* bc_value_v)
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      bc_v[dir].setWallValues(bc_value_v[dir]);
  }
  inline void set_bc_value_on_pressure(const CF_DIM* bc_value_p) { bc_p.setWallValues(*bc_value_p); }

  inline double acceleration_for_canonical_u_tau(const double& desired_u_tau) const
  {
    return SQR(desired_u_tau)/delta();
  }

  inline double canonical_u_tau(const double& xforce_per_unit_mass) const
  {
    return sqrt(delta()*xforce_per_unit_mass);
  }

  inline double canonical_Re_tau(const double& xforce_per_unit_mass, const double& kinematic_viscosity) const
  {
    return canonical_u_tau(xforce_per_unit_mass)*delta()/kinematic_viscosity;
  }

  inline double mean_u(const double& mass_flow, const double& mass_density) const
  {
    return mass_flow/MULTD(mass_density, height(), width());
  }

  inline double Re_b(const double& mass_flow, const double& mass_density, const double& kinematic_viscosity) const
  {
    return mean_u(mass_flow, mass_density)*delta()/kinematic_viscosity;
  }

  inline double acceleration_for_constant_mass_flow(const double& desired_U_b, const double& desired_Re_b, const int& nterms)
  {
    // canonical_u_tau/U_b = Re_tau/Re_b
    solve_for_truncated_series(nterms);
    return acceleration_for_canonical_u_tau(desired_U_b*Re_tau_from_Re_b(desired_Re_b)/desired_Re_b);
  }

  inline void create_p4est_ghost_and_nodes( p4est_t* &forest, p4est_ghost_t* &ghost, p4est_nodes_t* &nodes,
										    splitting_criteria_cf_and_uniform_band_shs_t* &sp,
										    p4est_connectivity_t *conn, const mpi_environment_t& mpi_, const int& lmin,
											const unsigned int wall_layer, const double& lmid_delta_percent,
											const double& lip_user )
  {
    P4EST_ASSERT( is_configured );
    delete sp;
    sp = new splitting_criteria_cf_and_uniform_band_shs_t( lmin, max_lvl, this,
														   calculate_uniform_band_for_ns_solver( wall_layer ), delta(),
														   lmid_delta_percent, calculate_lip_for_ns_solver( lip_user ) );

    if( forest != nullptr )
      p4est_destroy( forest );
    forest = my_p4est_new( mpi_.comm(), conn, 0, nullptr, nullptr );
    forest->user_pointer = (void*) sp;

    for( int l = 0; l < sp->max_lvl; ++l )
    {
      my_p4est_refine( forest, P4EST_FALSE, refine_levelset_cf_and_uniform_band_shs, nullptr );
      my_p4est_partition( forest, P4EST_FALSE, nullptr );
    }

	// Create the initial forest at time nm1.
    p4est_balance( forest, P4EST_CONNECT_FULL, nullptr );
    my_p4est_partition( forest, P4EST_FALSE, nullptr );

    if( ghost != nullptr )
      p4est_ghost_destroy( ghost );
    ghost = my_p4est_ghost_new( forest, P4EST_CONNECT_FULL );
    my_p4est_ghost_expand( forest, ghost );
    const double tree_dim[P4EST_DIM] = {DIM( (brick->xyz_max[0] - brick->xyz_min[0]) / brick->nxyztrees[0],
										(brick->xyz_max[1] - brick->xyz_min[1]) / brick->nxyztrees[1],
										(brick->xyz_max[2] - brick->xyz_min[2])/brick->nxyztrees[2] )};
    if( third_degree_ghost_are_required( tree_dim ) )
      my_p4est_ghost_expand( forest, ghost );
    if( nodes != nullptr )
      p4est_nodes_destroy( nodes );
    nodes = my_p4est_nodes_new( forest, ghost );
  }

  // functions linking integer input parameters "wall_layer" (in terms of number of cells) and N-S solver's uniform_band (absoute distance) to each other
  inline unsigned int ncells_layering_walls_from_ns_solver(const double& uniform_band) const
  {
    return (unsigned int) (uniform_band*MAX(DIM(length()/brick->nxyztrees[0], height()/brick->nxyztrees[1], width()/brick->nxyztrees[2]))/(height()/brick->nxyztrees[1]));
  }
  inline double calculate_uniform_band_for_ns_solver(const unsigned int& ncells_layering) const
  {
    return ncells_layering*(height()/brick->nxyztrees[1])/MAX(DIM(length()/brick->nxyztrees[0], height()/brick->nxyztrees[1], width()/brick->nxyztrees[2]));
  }

  // function calculating the value of Lipschitz constant to pass to the N-S solver in order to have a criterion
  // actually considering Lip*dy instead of Lip*diag(cell) inside the N-S solver
  inline double calculate_lip_for_ns_solver(const double& lip_from_user) const
  {
    return lip_from_user*(height()/brick->nxyztrees[1])/sqrt(SUMD(SQR(length()/brick->nxyztrees[0]), SQR(height()/brick->nxyztrees[1]), SQR(width()/brick->nxyztrees[2])));
  }

#ifdef P4_TO_P8
  inline bool spanwise_grooves() const { return spanwise; }
#endif

  inline void initialize_velocity(my_p4est_navier_stokes_t* ns, const int &nterms, const flow_setting& flow_setup, const double &Re, const double &white_noise_rel_rms = 0.0)
  {
    solve_for_truncated_series(nterms);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    const double mean_rms_noise = white_noise_rel_rms*(get_corresponding_Re_b(flow_setup, Re)*ns->get_nu()/delta());

    double max_norm_u_n = 0.0;

    PetscErrorCode ierr;
    Vec vnm1_nodes[P4EST_DIM], vn_nodes[P4EST_DIM];
    double *vnm1_nodes_p[P4EST_DIM], *vn_nodes_p[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostNodes(ns->get_p4est_nm1(), ns->get_nodes_nm1(), &vnm1_nodes[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(ns->get_p4est(), ns->get_nodes(), &vn_nodes[dir]); CHKERRXX(ierr);
      ierr = VecGetArray(vnm1_nodes[dir], &vnm1_nodes_p[dir]); CHKERRXX(ierr);
      ierr = VecGetArray(vn_nodes[dir], &vn_nodes_p[dir]); CHKERRXX(ierr);
    }

    for (size_t k = 0; k < MAX(ns->get_ngbd_n()->get_layer_size(), ns->get_ngbd_nm1()->get_layer_size()); ++k) {
      if(k < ns->get_ngbd_nm1()->get_layer_size())
      {
        p4est_locidx_t node_idx = ns->get_ngbd_nm1()->get_layer_node(k);
        double xyz[P4EST_DIM], velocity[P4EST_DIM];
        node_xyz_fr_n(node_idx, ns->get_ngbd_nm1()->get_p4est(), ns->get_ngbd_nm1()->get_nodes(), xyz);
        v_exact(flow_setup, Re, ns, xyz, velocity);
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          vnm1_nodes_p[dim][node_idx] = velocity[dim] + mean_rms_noise*distribution(generator);
      }

      if(k < ns->get_ngbd_n()->get_layer_size())
      {
        p4est_locidx_t node_idx = ns->get_ngbd_n()->get_layer_node(k);
        double xyz[P4EST_DIM], velocity[P4EST_DIM];
        node_xyz_fr_n(node_idx, ns->get_ngbd_n()->get_p4est(), ns->get_ngbd_n()->get_nodes(), xyz);
        v_exact(flow_setup, Re, ns, xyz, velocity);
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          vn_nodes_p[dim][node_idx] = velocity[dim] + mean_rms_noise*distribution(generator);
        max_norm_u_n = MAX(max_norm_u_n, sqrt(SUMD(SQR(vn_nodes_p[dir::x][node_idx]), SQR(vn_nodes_p[dir::y][node_idx]), SQR(vn_nodes_p[dir::z][node_idx]))));
      }
    }

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateBegin(vnm1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vn_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    for (size_t k = 0; k < MAX(ns->get_ngbd_n()->get_local_size(), ns->get_ngbd_nm1()->get_local_size()); ++k) {
      if(k < ns->get_ngbd_nm1()->get_local_size())
      {
        p4est_locidx_t node_idx = ns->get_ngbd_nm1()->get_local_node(k);
        double xyz[P4EST_DIM], velocity[P4EST_DIM];
        node_xyz_fr_n(node_idx, ns->get_ngbd_nm1()->get_p4est(), ns->get_ngbd_nm1()->get_nodes(), xyz);
        v_exact(flow_setup, Re, ns, xyz, velocity);
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          vnm1_nodes_p[dim][node_idx] = velocity[dim] + mean_rms_noise*distribution(generator);
      }

      if(k < ns->get_ngbd_n()->get_local_size())
      {
        p4est_locidx_t node_idx = ns->get_ngbd_n()->get_local_node(k);
        double xyz[P4EST_DIM], velocity[P4EST_DIM];
        node_xyz_fr_n(node_idx, ns->get_ngbd_n()->get_p4est(), ns->get_ngbd_n()->get_nodes(), xyz);
        v_exact(flow_setup, Re, ns, xyz, velocity);
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          vn_nodes_p[dim][node_idx] = velocity[dim] + mean_rms_noise*distribution(generator);
        max_norm_u_n = MAX(max_norm_u_n, sqrt(SUMD(SQR(vn_nodes_p[dir::x][node_idx]), SQR(vn_nodes_p[dir::y][node_idx]), SQR(vn_nodes_p[dir::z][node_idx]))));
      }
    }

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateEnd(vnm1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vn_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnm1_nodes[dir], &vnm1_nodes_p[dir]); CHKERRXX(ierr);
      ierr = VecRestoreArray(vn_nodes[dir], &vn_nodes_p[dir]); CHKERRXX(ierr);
    }

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_norm_u_n, 1, MPI_DOUBLE, MPI_MAX, ns->get_mpicomm()); SC_CHECK_MPI(mpiret);
    ns->set_velocities(vnm1_nodes, vn_nodes, &max_norm_u_n);
  }

  inline double get_c0() const {
    P4EST_ASSERT(coeff_are_set);
    return coeff[0];
  }

  // The user should NEVER use these for shs simulations
  // the following functions are required for the validation tests only!
  inline void set_dirichlet_value_u(CF_DIM& u_wall)
  {
    bc_wall_value_u_dirichlet = &u_wall;
  }
  inline void set_neumann_value_u(CF_DIM& n_dot_grad_u_wall)
  {
    bc_wall_value_u_neumann = &n_dot_grad_u_wall;
  }
  inline void set_dirichlet_value_v(CF_DIM& v_wall)
  {
    bc_wall_value_v_dirichlet = &v_wall;
  }

#ifdef P4_TO_P8
  inline void set_dirichlet_value_w(CF_DIM& w_wall)
  {
    bc_wall_value_w_dirichlet = &w_wall;
  }
  inline void set_neumann_value_w(CF_DIM& n_dot_grad_w_wall)
  {
    bc_wall_value_w_neumann = &n_dot_grad_w_wall;
  }
#endif

  inline void set_neumann_value_p(CF_DIM& p_wall)
  {
    bc_wall_value_p_neumann = &p_wall;
  }

};

#endif // MY_P4EST_SHS_CHANNEL_H
