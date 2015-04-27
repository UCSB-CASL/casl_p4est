#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <src/random_generator.h>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/my_p8est_levelset.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#include<src/my_p4est_levelset.h>
#endif

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/stresstensor2.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/stresstensor.h>
#endif



#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>
#include<src/inverse_litography.h>
#include<src/meanfieldplan.h>

//#include </Users/gaddielouaknin/Documents/Install/fftw-3.3.4/include/fftw3-mpi.h>
//#include  </Users/gaddielouaknin/Documents/Install/fftw-3.3.4/include/fftw3.h>

#include <fftw3-mpi.h>
#include  <fftw3.h>

using namespace std;

#ifdef P4_TO_P8
static struct:CF_3{
                  void update (double x0_, double y0_, double z0_, double r_) {x0 = x0_; y0 = y0_; z0 = z0_; r = r_; }
                  double operator()(double x, double y, double z) const {
                  return r - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
                  }
                  double  x0, y0, z0, r;
} circle2 ;

static class: public CF_3
{
                         public:
                         double operator()(double x, double y, double z) const {
                         return  cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
                         }
                         } u_ex2;



class uext:public CF_3
{
public:
    double t;
    double w;
    bool spatial;
    uext(double t,double w,bool spatial)
    {
        this->t=t;
        this->w=w;
        this->spatial=spatial;
    }
    double operator()(double x, double y,double z) const
    {
        if (this->spatial)
            return  cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z)*exp(-12*M_PI*M_PI*M_PI*t-this->w*t);
        else
            return exp(-this->w*t);
    }
};


static class: public CF_3
{
                         public:
                         double operator()(double x, double y, double z) const {
                         return  12*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
                         }
                         } f_ex2;

static struct:WallBC3D{
                  BoundaryConditionType operator()(double x, double y, double z) const {
                  (void)x;
                  (void)y;
                  (void)z;
                  return NEUMANN;
                  }
                  } bc_wall_neumann_type2;

static struct:WallBC3D{
                  BoundaryConditionType operator()(double x, double y, double z) const {
                  (void)x;
                  (void)y;
                  (void)z;
                  return DIRICHLET;
                  }
                  } bc_wall_dirichlet_type2;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  (void) x;
                  (void) y;
                  (void) z;
                  return 0;
                  }
                  } bc_wall_neumann_value2;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  return u_ex2(x,y,z);
                  }
                  } bc_wall_dirichlet_value2;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  return u_ex2(x,y,z);
                  }
                  } bc_interface_dirichlet_value2;








static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  (void) x;
                  (void) y;
                  (void) z;
                  return 0;
                  }
                  } bc_wall_dirichlet_value0;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  (void) x;
                  (void) y;
                  (void) z;
                  return 0;
                  }
                  } bc_interface_dirichlet_value0;

static struct:CF_3{
                  double operator()(double x, double y, double z) const {
                  double r  = sqrt(SQR(x-circle2.x0) + SQR(y-circle2.y0) + SQR(z-circle2.z0));
                  double nx = (x-circle2.x0) / r;
                  double ny = (y-circle2.y0) / r;
                  double nz = (z-circle2.z0) / r;
                  double norm = sqrt( nx*nx + ny*ny + nz*nz);
                  nx /= norm; ny /= norm; nz /= norm;
                  return 0;/*( 2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z) * nx +
                                             2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z) * ny +
                                             2*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z) * nz );*/
                  }
                  } bc_interface_neumann_value2;
#else
static struct:CF_2{
                  void update (double x0_, double y0_, double r_) {x0 = x0_; y0 = y0_; r = r_; }
                  double operator()(double x, double y) const {
                  return r - sqrt(SQR(x-x0) + SQR(y-y0));
                  }
                  double  x0, y0, r;
} circle2;

static class: public CF_2
{
                         public:
                         double operator()(double x, double y) const {
                         return  cos(2*M_PI*x)*cos(2*M_PI*y);
                         }
                         } u_ex2;

class uext:public CF_2
{
public:
    double t;
    double w;
    bool spatial;
    uext(double t,double w,bool spatial)
    {
        this->t=t;
        this->w=w;
        this->spatial=spatial;
    }
    double operator()(double x, double y) const
    {
        if (this->spatial)
            return  cos(2*M_PI*x)*cos(2*M_PI*y)*exp(-8*M_PI*M_PI*t-this->w*t);
        else
            return exp(-this->w*t);
    }
};


static class: public CF_2
{
                         public:
                         double operator()(double x, double y) const {
                         return  8*M_PI*M_PI*cos(2*M_PI*x)*cos(2*M_PI*y);
                         }
                         } f_ex2;

static struct:WallBC2D{
                  BoundaryConditionType operator()(double x, double y) const {
                  (void)x;
                  (void)y;
                  return NEUMANN;
                  }
                  } bc_wall_neumann_type2;

static struct:WallBC2D{
                  BoundaryConditionType operator()(double x, double y) const {
                  (void)x;
                  (void)y;
                  return DIRICHLET;
                  }
                  } bc_wall_dirichlet_type2;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  (void) x;
                  (void) y;
                  return 0;
                  }
                  } bc_wall_neumann_value2;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  return u_ex2(x,y);
                  }
                  } bc_wall_dirichlet_value2;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  return u_ex2(x,y);
                  }
                  } bc_interface_dirichlet_value2;
static struct:CF_2{
                  double operator()(double x, double y) const {
                  (void) x;
                  (void) y;
                  return 0;
                  }
                  } bc_wall_dirichlet_value0;

static struct:CF_2{
                  double operator()(double x, double y) const {
                  (void) x;
                  (void) y;
                  return 0;
                  }
                  } bc_interface_dirichlet_value0;


static struct:CF_2{
                  double operator()(double x, double y) const {
                  double r = sqrt( SQR(x-circle2.x0) + SQR(y-circle2.y0) );
                  double nx = (x-circle2.x0) / r;
                  double ny = (y-circle2.y0) / r;
                  double norm = sqrt( nx*nx + ny*ny);
                  nx /= norm; ny /= norm;
                  return 0;//2*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y) * nx + 2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y) * ny;
                  }
                  } bc_interface_neumann_value2;
#endif



class Diffusion
{
public :

    parStopWatch evolve_probability_equation_watch;
    parStopWatch preconditioner_algo_watch;
    parStopWatch convergence_test_watch;
    parStopWatch interpolation_on_uniform_watch;
    parStopWatch stress_tensor_watch;
    parStopWatch fftw_setup_watch;
    parStopWatch fftw_scaterring_watch;


    double evolve_probability_equation_total_time;
    double preconditioner_algo_watch_total_time;
    double convergence_test_watch_total_time;
    double fftw_scaterring_watch_total_time;


    interpolation_method my_interpolation_method;

PetscReal rtol_in;
enum matrix_solver_algo
{
  richardson_matrix_algo,
  chebyshev_matrix_algo,
  conjugate_gradient_matrix_algo,
  biConjugate_gradient_matrix_algo,
  generalized_minimal_residual_matrix_algo,
  flexibe_generalized_minimal_residual_matrix_algo,
  deflated_generalized_minimal_residual_matrix_algo,
  generalized_conjugate_residual_matrix_algo,
  bcgstab_matrix_algo,
  conjugate_gradient_squared_matrix_algo,
  transpose_free_quasi_minimal_residual_matrix_algo_1,
  transpose_free_quasi_minimal_residual_matrix_algo_2,
  conjugate_residual_matrix_algo,
  least_squares_method_matrix_algo


};

enum preconditioner_algo
{
    jacobi_preconditioner_algo,
    block_jacobi_preconditioner_algo,
    sor_preconditioner_algo,
    sor_eisenstat_preconditioner_algo,
    incomplete_cholesky_preconditioner_algo,
    incomplete_LU_preconditioner_algo,
    additive_schwartz_preconditioner_algo,
    algebraic_multi_grid_preconditioner_algo,
    linear_solver_preconditioner_algo,
    lu_preconditioner_algo,
    cholesky_preconditioner_algo,
    no_preconditionning_preconditioner_algo

};

inline void set_matrix_solver_algorithm(matrix_solver_algo msa){this->my_matrix_solver_algo=msa;}
inline void set_preconditioner_algo(preconditioner_algo pca) {this->my_preconditioner_algo=pca;}



protected:
 matrix_solver_algo    my_matrix_solver_algo;
 preconditioner_algo   my_preconditioner_algo;


public:
    enum phase
    {
        benchmark_constant,
        benchmark_spatial,
        disordered,
        lamellar,
        lamellar_x,
        lamellar_x_from_matlab,
        lamellar_y,
        lamellar_z,
        cylinder,
        gyroid,
        bcc,
        bcc_masked,
        fcc,
        single_ellipse,
        random,
        confined_cubes,
        confined_spheres,
        confined_spheres_helix,
        confine_spheres_3d_l_shape,
        confined_spheres_drop,
        confined_cylinders_helix,
        from_one_text_file,
        from_one_fine_text_file_to_coarse,
         from_two_fine_text_file_to_coarse,
        from_two_text_files,
        coarse_from_text_file,
        from_seed,
        smooth,
        smooth_r,
        smooth_y,
         smooth_x,
        smooth_xy,
        constant_w,
        gaussian_w,
        clock_w,
        cos_w,
        from_mask
    };

    enum numerical_scheme
    {
        finite_difference_explicit,
        finite_difference_implicit,
        splitting_finite_difference_explicit,
        splitting_finite_difference_implicit,
        splitting_finite_difference_implicit_scaled,
        splitting_spectral,
        splitting_spectral_adaptive

    };

    enum casl_diffusion_method
    {
        periodic_crank_nicholson,
        neuman_backward_euler,
        neuman_crank_nicholson,
        dirichlet_backward_euler,
        dirichlet_crank_nicholson,
        robin_backward_euler,
        robin_crank_nicholson

    };

    PetscBool create_plan;


    inline std::string get_my_numerical_scheme_string()
    {
        switch(this->myNumericalScheme)
        {
        case finite_difference_explicit:
            return "finite_difference_explicit";
            break;
        case finite_difference_implicit:
            return "finite_difference_implicit";
            break;
        case splitting_finite_difference_explicit:
            return "splitting_finite_difference_explicit";
            break;
        case splitting_finite_difference_implicit:
            return "splitting_finite_difference_implicit";
            break;
        case splitting_finite_difference_implicit_scaled:
            return "splitting_finite_difference_implicit_scaled";
        case splitting_spectral:
            return "splitting_spectral";
            break;
        case splitting_spectral_adaptive:
            return "splitting_spectral_adaptive";
            break;

        default:
            return " no mode";
            break;
        }
    }


    inline std::string get_my_casl_diffusion_method_string()
    {
        switch(this->my_casl_diffusion_method)
        {
        case periodic_crank_nicholson:
            return "periodic_crank_nicholson";
            break;
        case neuman_backward_euler:
            return "neuman_backward_euler";
            break;
        case neuman_crank_nicholson:
            return "neuman_crank_nicholson";
            break;
        case dirichlet_crank_nicholson:
            return "neuman_crank_nicholson";
            break;

        default:
            return " no mode";
            break;
        }
    }


    inline Diffusion::phase get_my_phase(){return this->myPhase;}

protected:

    //petsc fields

    Mat A;

    Mat M1,M2;
    Mat subM1,subM2;
    IS my_rows_parallel_indexes,my_columns_parallel_indexes;
    PetscBool extractSubDiffusionMatricesBool;
    PetscBool submatrices_extracted,submatrices_destructed;


    KSP myKsp; PC pc;
    PetscScalar    neg_one,one,value[3];
    PetscBool      nonzeroguess;
    PetscInt       n_petsc ,col[3],its;
    PetscReal      norm,tol;  /* norm of solution error */
    Vec sol_t;
    Vec sol_tp1;
    Vec b_t;
    Vec r_t;
    Vec w_t;
       PetscScalar wp_average;
       PetscScalar wp_average_at_saddle_point;



    double xhi_w_a;
    double xhi_w_b;
    double xhi_w_p;
    double xhi_w_m;
    double xhi_w;

    double zeta_n_inverse;



    Vec energy_integrand;
    double *sol_history;
    double *q_forward,*q_backward;

public:
    Vec rhoA; Vec rhoB;
    StressTensor *my_stress_tensor;
    Vec fp,fm;
    PetscBool spatial_xhi_w;
    Vec xhi_w_x;
    MeanFieldPlan *myMeanFieldPlan;
    double wp_average_on_contour,wp_standard_deviation_on_contour;
    double wm_average_on_contour,wm_standard_deviation_on_contour;

protected:

    Vec my_temp_vec; Vec phiTemp;
    double *rhoA_local;
    double *rhoB_local;
    double *drho_t;
    double *drho_x_forward;
    double *drho_x_backward;
    double lambda_plus,lambda_minus;
    double E,Q_forward,Q_backward,V;
    double E_w,E_logQ,E_compressible,E_wall_p,E_wall_m;
    double phi_bar;
    bool forward_stage;
    bool first_stage;


    double fp2,fm2;
    double rho0Average;
    double rhoAAverage;
    double rhoBAverage;
    double rho0Average_surface;
    double rhoAAverage_surface;
    double rhoBAverage_surface;
    double orderRatio;


    int n_local_size_sol;
    int n_global_size_sol;
    int local_size_sol;

    PetscInt *ix_fromLocal2Global;

    int N_iterations;
    int N_t;
    int it;
    int nx_trees;
    double dt;
    double f,X_ab;
    PetscScalar Lx,Lx_physics; PetscScalar D;
    phase myPhase;
    int i_mean_field;



    PetscScalar w_benchmark;
    PetscBool print_diffusion_matrices;
    PetscBool print_remeshed_diffusion_matrices;
    PetscBool spatial;


    PetscBool minimum_IO;


    // p4est fields

    mpi_context_t mpi_context, *mpi;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;

    PetscErrorCode      ierr;
    cmdParser cmd;
    p4est_ghost_t* ghost;
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t brick;
    my_p4est_node_neighbors_t *nodes_neighbours;


#ifdef P4_TO_P8
    BoundaryConditions3D bc;
#else
    BoundaryConditions2D bc;
#endif

#ifdef P4_TO_P8
    CF_3 *bc_wall_value, *bc_interface_value;
    WallBC3D *wall_bc;
#else
    CF_2 *bc_wall_value, *bc_interface_value;
    WallBC2D *wall_bc;
#endif

    BoundaryConditionType bc_wall_type, bc_interface_type;

    int nb_splits, min_level, max_level;

    //    double *sol_p, *phi_p, *uex_p,*rhs_p, *bc_p;
    //    double *sol_p_cell;
    Vec phi;
    //    Vec rhs, uex, sol;
    //    double *err;

    PoissonSolverNodeBase *solver;

    PetscInt rstart,rend,nlocal;

    PetscBool fast_algo;
    PetscBool strang_splitting;
    PetscBool remesh;
    PetscBool mapped;
    PetscBool remeshed;
    PetscBool clean;

PetscBool advance_fields_scft_advance_mask_level_set;
PetscBool setup_finite_difference_solver;
    // casl fields
    //Diffusion::casl_diffusion_method my_casl_diffusion_method=Diffusion::neuman_backward_euler;
    Diffusion::casl_diffusion_method my_casl_diffusion_method;//=Diffusion::periodic_crank_nicholson;
    numerical_scheme myNumericalScheme;//=splitting_finite_difference_implicit;
    Vec phi_polymer_shape;
    PetscBool evolve_free_surface;
    public:
    PoissonSolverNodeBase *myNeumanPoissonSolverNodeBase;
    protected:
    RandomGenerator *my_random_generator;



    int (Diffusion::*my_force_calculator)();//=NULL;
    int (Diffusion::*my_energy_calculator)();//=NULL;




    //--------------FFTW Structures------------------------------//


     //---Computation Structures--------------------------------//
     double *w3D;
     Vec w3dGlobal;


     double *w2D;
     Vec w2dGlobal;
     bool FFTCreated;
     bool mapping_fftw_created;
     bool mapping_fftw_destroyed;
     fftw_complex *input_backward, *output_backward;
     fftw_complex *input_forward, *output_forward;
     double *input_forward_real,*output_backward_real;
     bool real_to_complex_fftw_strategy;
     ptrdiff_t alloc_local, local_n0, local_0_start;
     fftw_plan plan_fftw_forward;
     fftw_plan plan_fftw_backward;


     PetscScalar *fftw_local_petsc_output;
     Vec fftw_global_petsc_output;


     // note that we can use it to do the reverse scattering
     VecScatter scatter_from_fftw_to_p4est;

     /* scatter context */
     IS from_fftw, to_p4est; /* index sets that define the scatter */

     int *idx_from_fftw_to_p4est_fftw_ix;
     int *idx_from_fftw_to_p4est_p4est_ix;

     int *idx_fftw_local;


     //---End of Computation Structures-----------------------//



     //----------End of FFTW structures-----------------------//

public:


     inline Vec* get_phi_is_all_positive()
     {
         return &this->myNeumanPoissonSolverNodeBase->phi_is_all_positive;
     }

     inline void set_f_a(double f_a)
     {
         this->f=f_a;
     }

     inline void set_lambda_plus_and_lambda_minus(double lambda_plus,double lambda_minus)
     {
         this->lambda_plus=lambda_plus;
         this->lambda_minus=lambda_minus;
     }

     inline void setR2c2True()
     {
         this->real_to_complex_fftw_strategy=true;
     }

     inline void resetLxPhysics(double LxPhysics)
     {
         // NOTE: that in the case of a finite difference solver,
         // in addition to rescale the physical length of the domain
         // a remeshing step should be done to properly rebuild the crank nicholson matrices
         this->Lx_physics=LxPhysics;
     }

     inline void set_advance_fields_scft_advance_mask_level_set_2_true()
     {
         this->advance_fields_scft_advance_mask_level_set=PETSC_TRUE;
     }

     inline PetscScalar get_wp_average()
     {
         return this->wp_average;
     }
     inline PetscScalar get_wp_average_at_saddle_point()
     {
         return this->wp_average_at_saddle_point;
     }


     inline void setMinimumIO(PetscBool minimum_io)
     {
         this->minimum_IO=minimum_io;
     }

     inline void set_zeta_n_inverse(double zeta_n_inverse)
     {
         this->zeta_n_inverse=zeta_n_inverse;
     }


     inline double get_phi_bar(){return this->phi_bar;}
     inline double get_e_compressible(){return this->E_compressible;}
     inline double get_e_wall_p(){return this->E_wall_p;}
     inline double get_e_wall_m(){return this->E_wall_m;}


     inline void set_xhi_values(double xhi_w,double xhi_w_a,double xhi_w_b,double xhi_w_m,double xhi_w_p)
     {
         this->xhi_w=xhi_w;
                 this->xhi_w_a=xhi_w_a;
                 this->xhi_w_b=xhi_w_b;
                 this->xhi_w_m=xhi_w_m;
                 this->xhi_w_p=xhi_w_p;
     }


    PetscBool stress_tensor_built;
    PetscBool stress_tensor_cleaned;
    PetscInt stress_tensor_computation_period;
    PetscBool compute_stress_tensor_bool;
    PetscBool periodic_xyz;
    PetscBool printIntemediateSteps;
    PetscBool print_my_current_debugger;
    PetscBool seed_optimization;
    PetscBool substract_average_from_wp;



    //PetscBool

    PetscBool compute_energy_difference_diff;
    PetscScalar analytic_energy_difference_predicted_next_time_step;
    PetscScalar analytic_energy_difference_predicted_current_time_step;



    std::string input_potential_path_w_p;
    std::string input_potential_path_w_m;

    inline Diffusion::casl_diffusion_method get_my_casl_diffusion_method()
    {
        return this->my_casl_diffusion_method;
    }//=Diffusion::periodic_crank_nicholson;
    inline numerical_scheme get_myNumericalScheme()
    {
        return this->myNumericalScheme;
    }//=splitting_finite_difference_implicit;


    inline double getV()
    {
        return this->V;
    }

     inline double getE_logQ(){return this->E_logQ;}
      inline double getE_w(){return this->E_w;}

    inline void check_consistency_methodology(){}

    inline void set_my_casl_diffusion_method(Diffusion::casl_diffusion_method my_casl_diffusion_method)
    {
        this->my_casl_diffusion_method=my_casl_diffusion_method;
    }
    inline void set_my_numerical_scheme(Diffusion::numerical_scheme my_numerical_scheme)
    {
        this->myNumericalScheme=my_numerical_scheme;
    }

    inline void set_polymer_shape(Vec phi_polymer_shape)
    {
        if(this->phi_polymer_shape!=NULL && this->i_mean_field>0)
            VecDestroy(this->phi_polymer_shape);

        VecDuplicate(phi_polymer_shape,&this->phi_polymer_shape);
        VecCopy(phi_polymer_shape,this->phi_polymer_shape);
        this->ierr = VecGhostUpdateBegin(this->phi_polymer_shape, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_polymer_shape, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(this->ierr);
    }

    inline void set_phi_wall(Vec phi_wall)
    {
        if(this->phi_wall!=NULL && this->i_mean_field>0)
            VecDestroy(this->phi_wall);

        VecDuplicate(phi_wall,&this->phi_wall);
        VecCopy(phi_wall,this->phi_wall);
        this->ierr = VecGhostUpdateBegin(this->phi_wall, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(this->ierr);
        this->ierr = VecGhostUpdateEnd  (this->phi_wall, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(this->ierr);
    }


    std::string field_text_file;

    inline void set_i_mean_field(int i_mean_field)
    {
        this->i_mean_field=i_mean_field;
    }

    int scatter_petsc_vector(Vec * v2scatter);

    inline void copy_and_scatter_sol_tp1(Vec *sol_tp12Copy)
    {
        if(!this->periodic_xyz)
        {
            this->ierr=VecGetLocalSize(this->phi_polymer_shape,&this->n_local_size_sol); CHKERRXX(this->ierr);
            this->ierr=VecGetSize(this->phi_polymer_shape,&this->n_global_size_sol); CHKERRXX(this->ierr);
            this->compute_mapping_fromLocal2Global();
            // could use the same filter than in the potential filtering
            // consider making it a member class
            PetscInt *filter_ix=new PetscInt[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
            PetscScalar *filter_zero=new PetscScalar[this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive];
            PetscScalar *phi_is_all_positive;
            this->ierr=VecGetArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);
            int ix=0;
            for(int i=0;i<this->n_local_size_sol;i++)
            {
                if(phi_is_all_positive[i]>0.5)
                {
                    filter_ix[ix]=i;
                    ix++;
                }
            }
            for(int i=0;i<this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive;i++)
            {
                filter_ix[i]=this->ix_fromLocal2Global[filter_ix[i]];
                filter_zero[i]=0.00;
            }

            this->ierr=VecSetValues(this->sol_tp1,this->myNeumanPoissonSolverNodeBase->n_phi_is_all_positive,filter_ix,filter_zero,INSERT_VALUES); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyBegin(this->sol_tp1); CHKERRXX(this->ierr);
            this->ierr=VecAssemblyEnd(this->sol_tp1);   CHKERRXX(this->ierr);
            this->ierr=VecRestoreArray(this->myNeumanPoissonSolverNodeBase->phi_is_all_positive,&phi_is_all_positive); CHKERRXX(this->ierr);
            delete filter_ix;
            delete filter_zero;

        }

        this->ierr=VecDuplicate(this->sol_tp1,sol_tp12Copy); CHKERRXX(this->ierr);
        this->ierr=VecCopy(this->sol_tp1,*sol_tp12Copy); CHKERRXX(this->ierr);
        this->scatter_petsc_vector(sol_tp12Copy);

    }

    Vec wp;
    Vec wm;
    Vec phi_wall;
    Vec phi_wall_p,phi_wall_m;
PetscBool terracing;

double sin_frequency;

inline void setTerraccing2True(){this->terracing=PETSC_TRUE;}
    //------------------properties-----------------------//

    inline double get_E(){return this->E;}
    inline double get_Fp2(){return this->fp2;}
    inline double get_Fm2(){return this->fm2;}

    inline double get_Rho0Average(){return this->rho0Average;}
    inline double get_RhoAAverage(){return this->rhoAAverage;}
    inline double get_RhoBAverage(){return this->rhoBAverage;}

    inline double get_Rho0Average_surface(){return this->rho0Average_surface;}
    inline double get_RhoAAverage_surface(){return this->rhoAAverage_surface;}
    inline double get_RhoBAverage_surface(){return this->rhoBAverage_surface;}

    inline double get_OrderRatio(){return this->orderRatio;   }
    inline int get_n_local_size_sol(){return this->n_local_size_sol;}
    inline int get_n_global_size_sol(){return this->n_global_size_sol;}



    inline double getQForward(){return this->Q_forward;}
    inline double getQBackward(){return this->Q_backward;}


    Session *mpi_session;

    double collapse2Disordered;
    double diverge2Crap;
    StressTensor::computation_mode my_stress_tensor_computation_mode;

    inline void initialyze_default_parameters()
    {


        this->spatial_xhi_w=PETSC_FALSE;
        this->my_stress_tensor_computation_mode=StressTensor::qxqcx_memory_optimized;
        this->rtol_in=1.e-12;
        this->real_to_complex_fftw_strategy=false;
        this->my_matrix_solver_algo=Diffusion::bcgstab_matrix_algo;
        this->my_preconditioner_algo=Diffusion::jacobi_preconditioner_algo;
        this->create_plan=PETSC_FALSE;
        this->neg_one      = -1.0;
        this->one = 1.0;
        this->nonzeroguess = PETSC_FALSE;
        this->n_petsc = 10;
        this->tol=1.e-14;
        this->wp_average=0.00;
        this->xhi_w_a=0.00;
        this->xhi_w_b=0.00;
        this->xhi_w_p=0.00;
        this->xhi_w_m=0.00;
        this->xhi_w=0;
        this->sin_frequency=36;
        this->collapse2Disordered=0.001;
        this->diverge2Crap=2;

        this->extractSubDiffusionMatricesBool= PETSC_FALSE;
        this->submatrices_extracted= PETSC_FALSE,
        this->submatrices_destructed= PETSC_FALSE;;

        this->zeta_n_inverse=0.001;
        this->lambda_minus=0.50;
        this->lambda_plus=0.50;
        this->forward_stage=true;
        this->first_stage=true;
        this->it=0;
        this->dt=0.01;
        this->i_mean_field=0;

        this->w_benchmark=1;
        this->print_diffusion_matrices=PETSC_FALSE;
        this->print_remeshed_diffusion_matrices=PETSC_FALSE;
        this->spatial=PETSC_FALSE;
        this->minimum_IO=PETSC_FALSE;
        this->fast_algo=PETSC_TRUE;
        this->strang_splitting=PETSC_TRUE;
        this->remesh=PETSC_FALSE;
        this->mapped=PETSC_FALSE;
        this->remeshed=PETSC_FALSE;
        this->clean=PETSC_TRUE;
        this->advance_fields_scft_advance_mask_level_set=PETSC_FALSE;
        this->setup_finite_difference_solver=PETSC_TRUE;
        this->evolve_free_surface=PETSC_FALSE;

        this->FFTCreated=false;
        this->mapping_fftw_created=false;
        this->mapping_fftw_destroyed=true;

        this->stress_tensor_built=PETSC_FALSE;
        this->stress_tensor_cleaned=PETSC_FALSE;
        this->stress_tensor_computation_period=100;
        this->compute_stress_tensor_bool=PETSC_FALSE;
        this->periodic_xyz=PETSC_FALSE;
        this->printIntemediateSteps=PETSC_FALSE;
        this->print_my_current_debugger=PETSC_FALSE;
        this->seed_optimization=PETSC_FALSE;
        this->substract_average_from_wp=PETSC_TRUE;

        this->my_interpolation_method=quadratic;



        //PetscBool

        this->compute_energy_difference_diff=PETSC_TRUE;
        this->analytic_energy_difference_predicted_next_time_step=0;
        this->analytic_energy_difference_predicted_current_time_step=0;

        this->input_potential_path_w_p="/Users/gaddielouaknin/Documents/Simulations Output/Confined Spheres in a Sphere/f_05_Xab_40_ellipse/Output/wp_0.txt";
        this->input_potential_path_w_m="/Users/gaddielouaknin/Documents/Simulations Output/Confined Spheres in a Sphere/f_05_Xab_40_ellipse/Output/wm_0.txt";
        this->terracing=PETSC_FALSE;
    }

    Diffusion()
    {

        std::cout<<"Simple Constructor"<<std::endl;
        this->initialyze_default_parameters();
    }

    ~Diffusion();


    int Diffusion_initialyze_petsc(int argc, char* argv[]);
    int createDiffusionMatrices();
    int createDiffusionMatricesNeuman();
    int scaleDiffusionMatricesNeuman();
    int createDiffusionMatricesDirichlet();

    int regenerateDiffusionMatrices();
    int extractSubDiffusionMatrices();
    int createDiffusionMatricesRobin();


    int updateDiffusionMatrices();
    int allocateMemory2History(PetscBool minus_one_iteration);
    int createDiffusionSolution();
    int evolveDiffusionIteration();

    int computeStressTensor();
    int computeShapeDerivative();
    int create_copy_and_scatter_shape_derivative_and_destroy_stress_tensor(Vec *snn);


    int compute_wp_averaged_on_saddle_point();
    int compute_phi_bar();


    //--------------Periodic Algorithms for diffusion equation evolution------------------------//

    int evolve_Equation_Implicit();
    int evolve_Equation_Explicit();
    int evolve_Equation_Finite_Difference_Splitting_Implicit();
    int evolve_Equation_Finite_Difference_Splitting_Implicit_Scaled();
    int evolve_Equation_Finite_Difference_Splitting_Explicit();
    int evolve_Equation_Spectral_Splitting();

    int setup_spectral_solver_amr();
    int setup_spectral_solver_amr_r2c();
    int create_mapping_data_structure_fftw();
    int create_mapping_data_structure_fftw_r2r();



    int setup_spectral_solver_amr_2D();
    int setup_spectral_solver_amr_2D_r2c();
    int create_mapping_data_structure_fftw_2D();
    int create_mapping_data_structure_fftw_2D_r2r();


    int evolve_Equation_Spectral_Splitting_AMR();
     int evolve_Equation_Spectral_Splitting_AMR_r2r();
    int evolve_Equation_Spectral_Splitting_AMR_2D();
    int evolve_Equation_Spectral_Splitting_AMR_2D_r2r();
    int interpolate_w_t_on_uniform_grid();
    int interpolate_w_t_on_uniform_grid_2D();



    int evolve_Equation_Finite_Difference_Splitting_Implicit_Neuman_Backward_Euler_Fast();

    //------------------------------------------------------------------------------------------//

    //-------------Neuman Algorithms for a diffusion equation evolution------------------------//
    int evolve_Equation_Neuman_Backward_Euler();


    //-----------------------------------------------------------------------------------------//



    int sendSol2HitoryDataBase();

     int sendFirstIteration2HitoryDataBase();

    int evolve_probability_equation(int T1,int T2);

    int scatterForwardMyParallelVectors();

    int compute_densities();
    int compute_energy();
     int compute_energy_wall();
      int compute_energy_wall_dirichlet();
     int compute_energy_wall_neuman();
     int compute_energy_wall_with_spatial_selectivity();
     int compute_energy_wall_neuman_with_spatial_selectivity();
     int compute_energy_wall_terrace();
    int computeQ();
    int compute_forces();
    int compute_wall_forces();
    int compute_wall_forces_with_spatial_selectivity();
    int compute_wall_forces_neuman();
    int compute_wall_forces_dirichlet();
    int compute_wall_forces_with_spatial_selectivity_neuman();
    int compute_wall_forces_terrace();
    int evolve_pressure_field();
    int evolve_chemical_exchange_field();
    int compute_time_integral4Densities(int ix);
    int compute_mapping_fromLocal2Global();
    int create_statistical_field();
    int get_fields_from_text_file();
    int get_fine_fields_from_text_file();
    int get_fine_fields_from_two_text_files();

    int get_fields_from_two_text_files();
    int get_coarse_fields_from_text_file();

    int get_coarse_vec_from_text_file(Vec *v2Fill, string file_name, int column);
    int get_vec_from_text_file(Vec *v2Fill, string file_name, int columns2Skip, int columns2Continue);
    int get_coarse_field_from_fine_text_file(Vec *v2Fill, string file_name,int file_level,int columns2Skip,int columns2Continue);


    int generate_disordered_phase();

    int generate_lamellar_phase();

    int generate_lamellar_x_phase();
    int generate_lamellar_x_phase_from_text_file();

    int generate_lamellar_y_phase();
    int generate_lamellar_z_phase();

    int generate_random_phase();
    int generate_bcc_phase();
    int generate_bcc_phase_masked();
    int generate_phase_from_mask(inverse_litography *my_litography);
    int generate_spheres_helixed();
    int generate_spheres_helixed_for_3d_l_shape();
    int generate_cylinders_helixed();
    int generate_ice();

    int generate_single_ellipse();
    int generate_smooth_fields();
    int generate_smooth_fields_xy();
     int generate_smooth_fields_r();
    int generate_smooth_fields_lamellar_y();
    int generate_smooth_fields_lamellar_x();

    int generate_constant_field();
    int generate_gaussian_field();
    int generate_clock_field();
    int generate_cos_field();


    int get_potential_from_text_file_root_processor();
    int scatter_potential_from_root_to_all(){}



    int computeVolume();
    int computeAverageProperties();

    int initialyzeDiffusionFromMeanField(int min_level, int max_level, Vec *phi, Session *mpi_session,
                                         mpi_context_t *mpi_context, mpi_context_t *mpi, p4est_t            *p4est, p4est_nodes_t      *nodes,
                                         p4est_ghost_t* ghost, p4est_connectivity_t *connectivity,
                                         my_p4est_brick_t *brick, my_p4est_hierarchy_t *hierarchy, my_p4est_node_neighbors_t *node_neighbors,
                                         double Lx, double Lx_physics, double f, double Xab, int N_iterations, Diffusion::phase myPhase,
                                         Diffusion::casl_diffusion_method my_casl_diffusion_method, Diffusion::numerical_scheme my_numerical_scheme,
                                         std::string my_io_path, int stress_tensor_computation_period, int nx_trees, double lambda,
                                         PetscBool setup_finite_difference_solver, bool periodic_xyz, string text_file_fields,

                                         string text_file_field_wp, string text_file_field_wm,MeanFieldPlan *myMeanFieldPlan);


    int destruct_initial_data();




    int reInitialyzeDiffusionFromMeanField(p4est_t            *p4est,p4est_nodes_t      *nodes,
                                           p4est_ghost_t* ghost,p4est_connectivity_t *connectivity,
                                           my_p4est_brick_t *brick,my_p4est_hierarchy_t *hierarchy,my_p4est_node_neighbors_t *node_neighbors,double Lx,double Lx_physics,int i_mean_field);


    int compute_energy_difference_analyticaly();

    int filter_irregular_matrices();
    int filter_irregular_matrices_dirichlet();
    int filter_irregular_potential();
    int filter_irregular_forces();
    int filter_petsc_vector(Vec *v2filter);
    int filter_petsc_vector_dirichlet(Vec *v2filter);
    int extend_petsc_vector(Vec *v2extend);
    int get_x_y_z_from_node(int n, double &x, double &y, double &z);


    void evolve_mean_field_step();
    int clean_mean_field_step(bool remesh_next_iteration);
    int evolve_fields_explicit();
    int printStatisticalFields();
    int createKSPContext();
    int Diffusion_finalyze_petsc();
    int printDiffusionMatrices(Mat *M2Print, string file_name_str, int i_mean_field);
    int printDiffusionVector(Vec *V2Print, string file_name_str);
    int printDiffusionArrayFromVector(Vec *v2Print, string file_name_str,PetscBool write_nodes=PETSC_FALSE);
    int printDiffusionArray(double *x2Print, int Nx, string file_name_str);
    int printDiffusionArray(int *x2Print, int Nx, string file_name_str);

    int print_vec_by_cells(Vec *v2_print,std::string file_name);

    void printArrayOnForestOctants2TextFile(double *x2Print_cell,double *x2Print_nodes, int Nx, string file_name_str);
    int printDiffusionIteration();
    int printDiffusionHistory();
    int printDensities();
    int printDensities_by_cells();
    int printForces();
    int printForestNodesNeighborsCells2TextFile();
    int interpolate_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid,std::string str_file);
    int print_uniform_local_array_to_text_file(PetscScalar *array2PrintOnUniformGrid,std::string str_file);
    std::string IO_path;
    inline std::string convert2FullPath(std::string file_name)
    {
        std::stringstream oss;
        std::string mystr;
        oss <<this->IO_path <<file_name;
        mystr=oss.str();
        return mystr;
    }
    void createSparsePetscMatrix(int argc, char *argv[]);
    void printSparsePetscMatrix();
    void printForestNodes2TextFile();
    void printForestOctants2TextFile();
    void printForestQNodes2TextFile();
    void printGhostNodes();
    void printGhostCells();
    void printMacroMesh();
    void setupNeumanSolver(my_p4est_node_neighbors_t *node_neighbors);
     void setupDirichletSolver(my_p4est_node_neighbors_t *node_neighbors);
    void solve_linear_system_slow();
    void solve_linear_system_efficient();

    void setupEnergyAndForces();

    /*!
     * \brief integrate_over_interface_in_one_quadrant
     */
    double integrate_over_interface_in_one_quadrant( p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);

    /*!
     * \brief integrate_over_interface integrate a scalar f over the 0-contour of the level-set function phi.
     *        note: first order convergence only
     * \param p4est the p4est
     * \param nodes the nodes structure associated to p4est
     * \param phi the level-set function
     * \param f the scalar to integrate
     * \return the integral of f over the contour defined by phi, i.e. \int_{phi=0} f
     */
    double integrate_over_interface( Vec phi, Vec f);



    /*!
     * \brief integrate_over_interface_in_one_quadrant
     */
    double integrate_constant_over_interface_in_one_quadrant(p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi);

    /*!
     * \brief integrate_over_interface integrate a scalar f over the 0-contour of the level-set function phi.
     *        note: first order convergence only
     * \param p4est the p4est
     * \param nodes the nodes structure associated to p4est
     * \param phi the level-set function
     * \param f the scalar to integrate
     * \return the integral of f over the contour defined by phi, i.e. \int_{phi=0} f
     */
    double integrate_constant_over_interface( Vec phi);

};

#endif // DIFFUSION_H
