#ifndef STRESSTENSOR_H
#define STRESSTENSOR_H

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>


//#include <mach/vm_statistics.h>
//#include <mach/mach_types.h>
//#include <mach/mach_init.h>
//#include <mach/mach_host.h>

#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_semi_lagrangian.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_levelset.h>
#include<src/my_p4est_semi_lagrangian.h>
#endif

#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>


using namespace std;

class StressTensor
{
public:

    //---------------------loaded variables----------------------------//

    mpi_context_t  *mpi;
    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    PetscErrorCode      ierr;
    p4est_ghost_t* ghost;
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t *brick;
    my_p4est_node_neighbors_t *nodes_neighbours;
    PetscScalar *mask_local;
    Vec *phi_is_all_positive;
    Vec *is_crossed_neumann;
    Vec *phi;
    int *ix_fromLocal2Global;
    int n_local,n_local_hist,N_t,N_iterations;
    PetscScalar dt;

    //------------------------end of loaded variables-----------------------------------------//

    //-----------------------intermediate computation variables------------------------------//

    PetscBool computeOneComponentOnly;

    enum direction
    {
        xx,
        xy,
        xz,
        yy,
        yz,
        zz
    };
    enum computation_mode
    {
        qxqcx,
        qxqcx_memory_optimized,
        qqcxx,
        shape_derivative
    };


    computation_mode my_computation_mode;

    // data structure for periodic domains
    Vec qx_forward,qy_forward,qz_forward;
    Vec qx_backward,qy_backward,qz_backward;
    double *qx_forward_local;
    double *qy_forward_local;
    double *qz_forward_local;
    double *qx_backward_local;
    double *qy_backward_local;
    double *qz_backward_local;


    // data structure for irregular domains
   Vec qxx_backward,qyy_backward,qzz_backward;
    Vec qxy_backward,qxz_backward,qyz_backward;
    double *qxx_backward_local;
    double *qyy_backward_local;
    double *qzz_backward_local;
    double *qxy_backward_local;
    double *qxz_backward_local;
    double *qyz_backward_local;

    double *q_forward_hist,*q_backward_hist;
    double *snn_hist;

    // spatial stress point by point
    double *sxx_local;
    double *sxy_local;
    double *sxz_local;
    double *syy_local;
    double *syz_local;
    double *szz_local;
    double *snn_local;
    double *ds_forward;
    double *ds_backward;
    my_p4est_level_set *LS;

    double Q;
    double V;
    PetscBool periodic_xyz;
    PetscBool minimum_IO;
    PetscBool test;
    PetscScalar dphi_test;
    PetscScalar DH_test;
    double Lx;
    double Lx_physics;
    int max_level,nx_trees;



//------------------------end of intermediate computation variables------------------------------//

//----------------------output variables------------------

    Vec sxx_global,sxy_global,sxz_global,syy_global,syz_global,szz_global,snn_global;
    double Sxx,Sxy,Sxz,Syy,Syz,Szz,Sn;


//--------------------end of output variables-------------




protected:
    int compute_spatial_integrand();
    int compute_spatial_integrand_irregular();
    int fill_snn_hist();
    int compute_shape_derivative();

    int extract_and_process_iteration(int it,double *q_hist_local,double *q_x_local,double *q_y_local,double *q_z_local);

    int extract_and_process_iteration_forward_backward(int it,double *q_hist_local,double *q_x_local,PetscBool forward_solution);


    int extract_and_process_iteration_irregular(int it);
    int extract_and_process_iteration_fill_snn(int it);
    int scatter_petsc_vector(Vec * v2scatter);
    int compute_stress();
    int compute_stress_memory_efficient();
    //int compute_stress_irregular();

    int compute_grad_f1_dot_grad_f2(Vec *f1,Vec *f2,Vec *df1_dot_df2);
    int compute_grad_f1_dot_grad_phi2(Vec *f1, Vec *phi2, Vec *df1_dot_dphi2);


    int create_test_function();
    int compute_test_integral();

    double compute_time_integral4Stresses();
    double compute_time_integral4Shape_derivative();
    int compute_micro_stresses(double *q_f, double *q_b, double *s_ii);



    double integrate_over_interface_in_one_quadrant(  p4est_quadrant_t *quad, p4est_locidx_t quad_idx, Vec phi, Vec f);


    double integrate_over_interface( Vec phi, Vec f);


    int printDiffusionVector(Vec *V2Print, string file_name_str);
    int printDiffusionArrayFromVector(Vec *v2Print, string file_name_str);
    int printDiffusionArray(double *x2Print, int Nx, string file_name_str);
    int interpolate_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid,std::string str_file);

public:


    inline void initialyze_default_parameters()
    {
        this->periodic_xyz=PETSC_FALSE;
        this->minimum_IO=PETSC_FALSE;
        this->test=PETSC_FALSE;
        this->dphi_test=0.01;
        this->DH_test=0;
        this->IO_path="/Users/gaddielouaknin/p4estLocal3/Output/";
        this->computeOneComponentOnly=PETSC_TRUE;
    }

    StressTensor(mpi_context_t *mpi   , p4est_t *p4est, p4est_nodes_t *nodes, p4est_ghost_t *ghost,
                 p4est_connectivity_t *connectivity, my_p4est_brick_t *brick, my_p4est_node_neighbors_t *node_neighbors,
                 Vec *phi_data_structure, double *q_forward_hist, double *q_backward_hist, int n_local, int n_local_hist,
                 int N_t, int N_iterations, double dt, Vec *phi_is_all_positive,Vec *is_crossed_neumann, int *ix_fromLocal2Global, double Q, double V, PetscBool periodic_xyz,
                 computation_mode my_computation_mode, PetscBool minimumIO,std::string IO_path,
                 double Lx,double Lx_physics);


        ~StressTensor();
    int cleanStressTensor();
    int cleanShapeDerivative();
    std::string IO_path;
    inline std::string convert2FullPath(std::string file_name)
    {
        std::stringstream oss;
        std::string mystr;
        oss <<this->IO_path <<file_name;
        mystr=oss.str();
        return mystr;
    }
    inline void set_some_params(int nx_trees,int max_level)
    {
        this->nx_trees=nx_trees;
        this->max_level=max_level;
    }
    int Compute_stress();
  };

#endif // STRESSTENSOR_H
