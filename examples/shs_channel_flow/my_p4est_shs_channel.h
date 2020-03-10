#ifndef MY_P4EST_SHS_CHANNEL_H
#define MY_P4EST_SHS_CHANNEL_H

#include <cmath>
#include <stdexcept>

#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_utils.h>
#endif

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

  struct BCWALLTYPE_P : WallBCDIM {
    BoundaryConditionType operator()(DIM(double, double, double)) const
    {
      return NEUMANN;
    }
  } bc_wall_type_p;

  struct BCWALLTYPE_U : WallBCDIM {
    const my_p4est_shs_channel_t *env;
    BCWALLTYPE_U(const my_p4est_shs_channel_t* env_) : env(env_) {}
    BoundaryConditionType operator()(DIM(double x, double y, double z)) const
    {
      return (env->is_ridge(DIM(x, y, z)) ? DIRICHLET : NEUMANN);
    }
  } bc_wall_type_u; // (note : always 0 value whether no slip or free slip)

  struct BCWALLTYPE_V : WallBCDIM {
    BoundaryConditionType operator()(DIM(double, double, double)) const
    {
      return DIRICHLET;
    }
  } bc_wall_type_v; // (note : always homogeneous dirichlet : no penetration through the channel wall)

#ifdef P4_TO_P8
  struct BCWALLTYPE_W : WallBC3D
  {
    const my_p4est_shs_channel_t* env;
    BCWALLTYPE_W(const my_p4est_shs_channel_t* env_) : env(env_){ }
    BoundaryConditionType operator()(double x, double y, double z) const
    {
      return (env->is_ridge(x, y, z) ? DIRICHLET : NEUMANN);
    }
  } bc_wall_type_w; // (note : always 0 value whether no slip or free slip)
#endif


  // For the analytical solution
  int num_terms;
  std::vector<double> coeff;
  bool coeff_are_set;
  std::vector<double> vec_of_ks;
  std::vector<double> vec_of_prefactors;
  std::vector<double> vec_of_tanhs;
  bool vecs_are_set;

  inline double length() const  { return brick->xyz_max[0] - brick->xyz_min[0]; }
#ifdef P4_TO_P8
  inline double width() const   { return brick->xyz_max[2] - brick->xyz_min[2]; }
  inline double normalized_z(const double& z) const
  {
    return my_fmod((z + 0.5*width()), pitch);
  }
#endif

  inline double offset() const
  {
#ifdef P4_TO_P8
    return (!spanwise ? 0.1*width()/((double) ((brick->nxyztrees[2]*(1 << max_lvl)))) : 0.5*length()/((double) ((brick->nxyztrees[0]*(1 << max_lvl)))));
#else
    return 0.5*length()/((double) (brick->nxyztrees[0]*(1 << max_lvl)));
#endif
  }

  inline double distance_to_ridge(DIM(const double& x, const double& y, const double& z)) const
  {
#ifdef P4_TO_P8
    if(!spanwise)
      return  sqrt(SQR(MIN(1.0 - y, y + 1.0)) + (is_ridge(x, y, z) ? 0.0 : SQR(MIN(normalized_z(z) - offset(), pitch*gas_frac - offset() - normalized_z(z)))));
#endif
    return    sqrt(SQR(MIN(1.0 - y, y + 1.0)) + (is_ridge(DIM(x, y, z)) ? 0.0 : SQR(MIN(my_fmod((x + 0.5*length() - offset()), pitch), gas_frac*pitch - my_fmod((x + 0.5*length() - offset()), pitch)))));
  }

  inline double k_(const int& n) const
  {
    return 2.0*M_PI*n/pitch;
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

  inline void check_input_coord(const double xyz_mod[P4EST_DIM]) const
  {
    if(fabs(xyz_mod[1]) > 1.0)
      throw std::invalid_argument("my_p4est_shs_channel_t: the y-coordinate must be contained in [-1, 1].");
#ifdef P4_TO_P8
    if(!spanwise && fabs(xyz_mod[2]) > 0.5*pitch)
      throw std::invalid_argument("my_p4est_shs_channel_t: the z-coordinate must be contained in [-pitch/2, pitch/2].");
#endif
    if(fabs(xyz_mod[0]) > 0.5*pitch)
      throw std::invalid_argument("my_p4est_shs_channel_t: the x-coordinate must be contained in [-pitch/2, pitch/2].");
  }


  inline double v_x(const double xyz_mod[P4EST_DIM]) const
  {
    check_input_coord(xyz_mod);
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
    check_input_coord(xyz_mod);
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
  inline double v_z(const double xyz_mod[P4EST_DIM]) const
  {
    check_input_coord(xyz_mod);
    return 0.0;
  }
#endif

  inline void check_Reynolds(const double& Re_tau) const
  {
    // In laminar canonical channel flows, Re_c = SQR(Re_tau) ...
    bool print_warning_tranverse = SQR(Re_tau) > 0.1;
#ifdef P4_TO_P8
    print_warning_tranverse = print_warning_tranverse && spanwise;
    bool print_warning_long = !spanwise && SQR(Re_tau) > 1000.0;
#endif
    if(print_warning_tranverse)
    {
      PetscErrorCode ierr = PetscPrintf(mpi.comm(), "\n[WARNING]: The exact solution for transversal grooves is valid only in the limit"
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

  inline void normalize_coordinates(double xyz_mod[P4EST_DIM], const double xyz[P4EST_DIM]) const
  {
    xyz_mod[0] = my_fmod(xyz[0] + 0.5*(length() + pitch*(1.0 - gas_frac)), pitch) - 0.5*pitch;
    xyz_mod[1] = xyz[1];
#ifdef P4_TO_P8
    xyz_mod[2] = my_fmod(xyz[2] + 0.5*(width()  + pitch*(1.0 - gas_frac)), pitch) - 0.5*pitch;
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


public:
  my_p4est_shs_channel_t(const mpi_environment_t& mpi_) : mpi(mpi_), bc_wall_type_u(this) ONLY3D(COMMA bc_wall_type_w(this))
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
    bc_p.setWallTypes(bc_wall_type_p);    bc_p.setWallValues(zero_value);
    bc_v[0].setWallTypes(bc_wall_type_u); bc_v[0].setWallValues(zero_value);
    bc_v[1].setWallTypes(bc_wall_type_v); bc_v[1].setWallValues(zero_value);
#ifdef P4_TO_P8
    bc_v[2].setWallTypes(bc_wall_type_w); bc_v[2].setWallValues(zero_value);
#endif
  }

  inline void configure(const my_p4est_brick_t *brick_, DIM(const double& pitch_, const double& gas_fraction, const bool& spanwise_), const int8_t& max_level)
  {
    brick = brick_;
    if(max_level < 0)
      throw std::invalid_argument("my_p4est_shs_channel_t::set_channel: the maximum level of refinement must be nonnegative.");
    max_lvl = max_level;
#ifdef P4_TO_P8
    spanwise = spanwise_;
    if(pitch_ <= 0.0 || (!spanwise_ && pitch_ > width()))
      throw std::invalid_argument("my_p4est_shs_channel_t::set_channel: the pitch must be strictly positive and smaller than or equal to the width of the channel.");
    if(pitch_ <= 0.0 || (spanwise_ && pitch_ > length()))
#else
    if(pitch_ <= 0.0 || pitch_ > length())
#endif
      throw std::invalid_argument("my_p4est_shs_channel_t::set_channel: the pitch must be strictly positive and smaller than or equal to the length of the channel.");
    pitch = pitch_;
    if(gas_fraction < 0.0 || gas_fraction >= 1.0)
      throw std::invalid_argument("my_p4est_shs_channel_t::set_channel: the gas fraction must be nonnegative and smaller than 1.0.");
    gas_frac = gas_fraction;

#ifdef P4EST_ENABLE_DEBUG
    is_configured = true;
#endif
  }

  inline double operator()(DIM(double x, double y, double z)) const
  {
    P4EST_ASSERT(is_configured);
    return -distance_to_ridge(DIM(x, y, z)) - pow(2.0, -max_lvl); // negative definite everywhere but as close to 0 as possible close to no-slip walls
  }

  inline bool is_ridge(DIM(const double& x, const double&, const double& z)) const
  {
    P4EST_ASSERT(is_configured);
#ifdef P4_TO_P8
    if(!spanwise)
      return (offset() >= normalized_z(z) || normalized_z(z) >= pitch*gas_frac - offset());
#endif
    return (my_fmod(x + 0.5*length() - offset(), pitch)/pitch >= gas_frac);
  }

  inline void solve_for_truncated_series(const double& num_terms_)
  {
    P4EST_ASSERT(is_configured);
    set_num_terms(num_terms_);
    compute_vecs();
    compute_series_coeff();
    P4EST_ASSERT(vecs_are_set && coeff_are_set);
    return;
  }

  // Modular transfomation used in the following three functions
  // { x_mod = fmod(x + 0.5*(length + pitch*(1 - GF)), pitch) - 0.5*pitch } (maps [-length/2, length/2] to [-pitch/2, pitch/2] with all gap centers being mapped to zero)
  // { z_mod = fmod(z + 0.5*(width  + pitch*(1 - GF)), pitch) - 0.5*pitch } (maps [-width/2,  width/2]  to [-pitch/2, pitch/2] with all gap centers being mapped to zero)
  inline void v_exact(const double &Re_tau, const double xyz[P4EST_DIM], double velocity[P4EST_DIM]) const
  {
    P4EST_ASSERT(is_configured);
    P4EST_ASSERT(vecs_are_set && coeff_are_set);
    double xyz_mod[P4EST_DIM];
    check_Reynolds(Re_tau);
    normalize_coordinates(xyz_mod, xyz);
    velocity[0] = Re_tau*v_x(xyz_mod);
    velocity[1] = Re_tau*v_y(xyz_mod);
#ifdef P4_TO_P8
    velocity[2] = Re_tau*v_z(xyz_mod);
#endif
  }

  inline double v_exact(const unsigned char& dir, const double &Re_tau, const double xyz[P4EST_DIM]) const
  {
    P4EST_ASSERT(is_configured);
    P4EST_ASSERT(vecs_are_set && coeff_are_set);
    double xyz_mod[P4EST_DIM];
    check_Reynolds(Re_tau);
    normalize_coordinates(xyz_mod, xyz);
    switch (dir) {
    case dir::x:
      return Re_tau*v_x(xyz_mod);
    case dir::y:
      return Re_tau*v_y(xyz_mod);
#ifdef P4_TO_P8
    case dir::z:
      return Re_tau*v_z(xyz_mod);
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

};

#endif // MY_P4EST_SHS_CHANNEL_H
