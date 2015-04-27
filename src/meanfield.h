#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cstdarg>
#include <src/meanfieldplan.h>

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
#include <src/diffusion2.h>
#include <src/my_p8est_levelset.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/stresstensor2.h>
#include <src/FieldProcessor2.h>
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
#include <src/diffusion.h>
#include <src/my_p4est_levelset.h>
#include<src/my_p4est_semi_lagrangian.h>
#include<src/stresstensor.h>
#include <src/FieldProcessor.h>
#endif

#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>


#include <src/potentialgenerator.h>
#include <src/inverse_litography.h>


//#include <src/diffusion.h>
using namespace std;

struct splitting_criteria_update_t4MeanField : splitting_criteria_t
{
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    std::vector<double> *phi_tmp;
    my_p4est_hierarchy_t *hierarchy;
    bool refine_in_minority;
    splitting_criteria_update_t4MeanField( double lip, int min_lvl, int max_lvl,
                                           std::vector<double> *phi,  my_p4est_brick_t *myb,
                                           p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,bool refine_in_minority)
    {
        this->refine_in_minority=refine_in_minority;
        this->lip = lip;
        this->min_lvl = min_lvl;
        this->max_lvl = max_lvl;
        this->myb = myb;
        this->p4est_tmp = p4est;
        this->ghost_tmp = ghost;
        this->nodes_tmp = nodes;
        this->phi_tmp = phi;
#ifndef P4EST_POINT_LOOKUP
        hierarchy = new my_p4est_hierarchy_t(p4est, ghost, myb);
#endif
    }
    ~splitting_criteria_update_t4MeanField()
    {
#ifndef P4EST_POINT_LOOKUP
        delete hierarchy;
#endif
    }

    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xyz) const
    {
        return 0;//linear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xyz);
    }
};

struct splitting_criteria_update_t4MeanFieldWithWidth : splitting_criteria_t
{
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    std::vector<double> *phi_tmp;
    my_p4est_hierarchy_t *hierarchy;
    bool refine_in_minority;
    double width;
    double width_negative;
    double width_positive;
    splitting_criteria_update_t4MeanFieldWithWidth( double lip, int min_lvl, int max_lvl,
                                                    std::vector<double> *phi,  my_p4est_brick_t *myb,
                                                    p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,bool refine_in_minority,
                                                    double width,double width_negative,double width_positive)
    {
        this->width=width;
        this->width_negative=width_negative;
        this->width_positive=width_positive;
        this->refine_in_minority=refine_in_minority;
        this->lip = lip;
        this->min_lvl = min_lvl;
        this->max_lvl = max_lvl;
        this->myb = myb;
        this->p4est_tmp = p4est;
        this->ghost_tmp = ghost;
        this->nodes_tmp = nodes;
        this->phi_tmp = phi;
#ifndef P4EST_POINT_LOOKUP
        hierarchy = new my_p4est_hierarchy_t(p4est, ghost, myb);
#endif
    }
    ~splitting_criteria_update_t4MeanFieldWithWidth()
    {
#ifndef P4EST_POINT_LOOKUP
        delete hierarchy;
#endif
    }

    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xyz) const
    {
        return 0;//linear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xyz);
    }
};


struct splitting_criteria_update_t4MeanFieldComplex : splitting_criteria_t
{
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    std::vector<double> *phi_tmp_m; // exchange_field
    std::vector<double> *phi_tmp_p; // pressure_field
    my_p4est_hierarchy_t *hierarchy;
    bool refine_in_minority;
    splitting_criteria_update_t4MeanFieldComplex( double lip, int min_lvl, int max_lvl,
                                                  std::vector<double> *phi_m, std::vector<double> *phi_p,  my_p4est_brick_t *myb,
                                                  p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,bool refine_in_minority)
    {
        this->refine_in_minority=refine_in_minority;
        this->lip = lip;
        this->min_lvl = min_lvl;
        this->max_lvl = max_lvl;
        this->myb = myb;
        this->p4est_tmp = p4est;
        this->ghost_tmp = ghost;
        this->nodes_tmp = nodes;
        this->phi_tmp_m= phi_m;
        this->phi_tmp_p= phi_p;


#ifndef P4EST_POINT_LOOKUP
        hierarchy = new my_p4est_hierarchy_t(p4est, ghost, myb);
#endif
    }
    ~splitting_criteria_update_t4MeanFieldComplex()
    {
#ifndef P4EST_POINT_LOOKUP
        delete hierarchy;
#endif
    }

    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xyz) const
    {
        return 0;//linear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xyz);
    }
};


struct splitting_criteria_update_t4MeanFieldComplexMasked : splitting_criteria_t
{
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    std::vector<double> *phi_tmp_m;          // exchange_field
    std::vector<double> *phi_tmp_p;          // pressure_field
    std::vector<double> *phi_tmp_mask;       // level set mask field
    my_p4est_hierarchy_t *hierarchy;
    bool refine_in_minority;
    splitting_criteria_update_t4MeanFieldComplexMasked( double lip, int min_lvl, int max_lvl,
                                                        std::vector<double> *phi_m, std::vector<double> *phi_p,
                                                        std::vector<double> *phi_mask,  my_p4est_brick_t *myb,
                                                        p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,bool refine_in_minority)
    {
        this->refine_in_minority=refine_in_minority;
        this->lip = lip;
        this->min_lvl = min_lvl;
        this->max_lvl = max_lvl;
        this->myb = myb;
        this->p4est_tmp = p4est;
        this->ghost_tmp = ghost;
        this->nodes_tmp = nodes;
        this->phi_tmp_m= phi_m;
        this->phi_tmp_p= phi_p;
        this->phi_tmp_mask=phi_mask;


#ifndef P4EST_POINT_LOOKUP
        hierarchy = new my_p4est_hierarchy_t(p4est, ghost, myb);
#endif
    }
    ~splitting_criteria_update_t4MeanFieldComplexMasked()
    {
#ifndef P4EST_POINT_LOOKUP
        delete hierarchy;
#endif
    }

    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xyz) const
    {
        return 0;//linear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xyz);
    }
};


struct splitting_criteria_update_t4MeanFieldComplexMaskedThreeLevels : splitting_criteria_t
{
    double lip;
    my_p4est_brick_t *myb;
    p4est_t *p4est_tmp;
    p4est_ghost_t *ghost_tmp;
    p4est_nodes_t *nodes_tmp;
    std::vector<double> *phi_tmp_m;          // exchange_field
    std::vector<double> *phi_tmp_p;          // pressure_field
    std::vector<double> *phi_tmp_mask;       // level set mask field
    my_p4est_hierarchy_t *hierarchy;
    bool refine_in_minority;
    int bulk_level;
    splitting_criteria_update_t4MeanFieldComplexMaskedThreeLevels( double lip, int min_lvl, int max_lvl,int bulk_level,
                                                                   std::vector<double> *phi_m, std::vector<double> *phi_p,
                                                                   std::vector<double> *phi_mask,  my_p4est_brick_t *myb,
                                                                   p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes,bool refine_in_minority)
    {
        this->refine_in_minority=refine_in_minority;
        this->lip = lip;
        this->min_lvl = min_lvl;
        this->max_lvl = max_lvl;
        this->bulk_level=bulk_level;
        this->myb = myb;
        this->p4est_tmp = p4est;
        this->ghost_tmp = ghost;
        this->nodes_tmp = nodes;
        this->phi_tmp_m= phi_m;
        this->phi_tmp_p= phi_p;
        this->phi_tmp_mask=phi_mask;


#ifndef P4EST_POINT_LOOKUP
        hierarchy = new my_p4est_hierarchy_t(p4est, ghost, myb);
#endif
    }
    ~splitting_criteria_update_t4MeanFieldComplexMaskedThreeLevels()
    {
#ifndef P4EST_POINT_LOOKUP
        delete hierarchy;
#endif
    }

    double operator()(const p4est_quadrant_t &quad, const double *f, const double *xyz) const
    {
        return 0;//linear_interpolation(p4est_tmp, quad.p.piggy3.which_tree, quad, f, xyz);
    }
};


class my_clock
{
public:
    parStopWatch optimization_watch;
    parStopWatch remeshing_watch;
    parStopWatch remapping_watch;
    parStopWatch k_means_watch;
    parStopWatch ls_reinitialyzation_watch;
    parStopWatch setup_watch;
    my_clock(){std::cout<<"initialize clock"<<std::endl;}
};


class DeltaH
{
public:
    double
    dlnQdphi_stress_term,dlnQdphi_simple_term,
    dQ_q_n,dQ_stress_n,
    dHdphi,dEwdphi,dEwmdphi,dEwpdphi,dlnQdphi,
    dEwPredicted,dlnQPredicted,dEPredicted,
    dEwComputed,dlnQComputed,dEComputed,
    dEwError,dlnQError,dEerror,Q_previous,Q_now,dQComputed,dQdPhiSimpleTerm,dQdPhiStressTerm,

    dEwdwp,dEwdwm,
    dlnQdwp,dlnQdwm,
    dHdw,dEwdw,dlnQdw,
    dQPredicted,
    dQ_joker;
    DeltaH(){}
};


class RobinOptimization
{

protected:

      double kappa_a;
      double lambda;

public:
    double kappa_b;

    double E_t;
    double f_t;

    double rho_a_surface;
    double rho_b_surface;
    void compute_energy()
    {
        this->E_t=this->kappa_a*this->rho_a_surface
                 +this->kappa_b*this->rho_b_surface;

    }
    void compute_force()
    {
        this->f_t=(this->E_t-this->kappa_a*this->rho_a_surface)/(this->rho_b_surface);

    }
    void evolve_kappa_b(double rho_a_surface,double rho_b_surface)
    {
        this->rho_a_surface=rho_a_surface;
        this->rho_b_surface=rho_b_surface;
        this->compute_energy();
        this->compute_force();
        this->kappa_b=this->kappa_b-this->lambda*this->E_t;
    }

    RobinOptimization(double kappa_a,double lambda)
    {
        this->kappa_b=0;
        this->kappa_a=kappa_a;
        this->E_t=0;
        this->f_t=0;
        this->lambda=lambda;
        this->rho_a_surface=0.00;
        this->rho_b_surface=0.00;
    }


};


class MeanField
{

public:


    inverse_litography *my_design_plan;
    DeltaH *my_delta_H;
    RobinOptimization *my_robin_optimization;
    MeanFieldPlan *myMeanFieldPlan;
    interpolation_method my_interpolation_method;

    my_clock *scft_clock;

    // This the core algorithm for deciding if to split a cell or not: it checks the Lipshitz criterion
protected:
    static p4est_bool_t refine_criteria_sl(p4est_t *p4est_tmp, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
    {
        splitting_criteria_update_t4MeanField *data = (splitting_criteria_update_t4MeanField*) p4est_tmp->user_pointer;

        if (quad->level < data->min_lvl)
            return P4EST_TRUE;
        else if (quad->level >= data->max_lvl)
            return P4EST_FALSE;
        else
        {
            double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
            double d = sqrt(P4EST_DIM)*dx;
            double lip = data->lip;

            /* find the quadrant in p4est_tmp */
            p4est_topidx_t v_mmm = p4est_tmp->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

            double tree_xmin = p4est_tmp->connectivity->vertices[3*v_mmm + 0];
            double tree_ymin = p4est_tmp->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est_tmp->connectivity->vertices[3*v_mmm + 2];
#endif

            double xyz [] =
            {
                quad_x_fr_i(quad) + tree_xmin + dx/2.0,
                quad_y_fr_j(quad) + tree_ymin + dx/2.0
    #ifdef P4_TO_P8
                ,
                quad_z_fr_k(quad) + tree_zmin + dx/2.0
    #endif
            };

            p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
            sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
            my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
            sc_array_destroy(remote_matches);
#else
            std::vector<p4est_quadrant_t> remote_matches;
            data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

            p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

            double *phi_tmp;
            phi_tmp = data->phi_tmp->data();

            double f[P4EST_CHILDREN];

            //check the distance criterion
            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f[j] = phi_tmp[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (fabs(f[j]) <= 0.5*lip*d ||(f[j]<0 &&data->refine_in_minority))
                    return P4EST_TRUE;
            }

            // check if it does cross the interface
#ifdef P4_TO_P8
            if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
                    f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
            if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
                return P4EST_TRUE;

            return P4EST_FALSE;
        }
    }

    static p4est_bool_t refine_criteria_sl_with_width(p4est_t *p4est_tmp, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
    {
        splitting_criteria_update_t4MeanFieldWithWidth *data = (splitting_criteria_update_t4MeanFieldWithWidth*) p4est_tmp->user_pointer;

        if (quad->level < data->min_lvl)
            return P4EST_TRUE;
        else if (quad->level >= data->max_lvl)
            return P4EST_FALSE;
        else
        {
            double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
            double d = sqrt(P4EST_DIM)*dx;
            double lip = data->lip;

            /* find the quadrant in p4est_tmp */
            p4est_topidx_t v_mmm = p4est_tmp->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

            double tree_xmin = p4est_tmp->connectivity->vertices[3*v_mmm + 0];
            double tree_ymin = p4est_tmp->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est_tmp->connectivity->vertices[3*v_mmm + 2];
#endif

            double xyz [] =
            {
                quad_x_fr_i(quad) + tree_xmin + dx/2.0,
                quad_y_fr_j(quad) + tree_ymin + dx/2.0
    #ifdef P4_TO_P8
                ,
                quad_z_fr_k(quad) + tree_zmin + dx/2.0
    #endif
            };

            p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
            sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
            my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
            sc_array_destroy(remote_matches);
#else
            std::vector<p4est_quadrant_t> remote_matches;
            data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

            p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

            double *phi_tmp;
            phi_tmp = data->phi_tmp->data();

            double f[P4EST_CHILDREN];
            double distance_with_width;

            //check the distance criterion
            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f[j] = phi_tmp[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];

                if(f[j]<0)
                    distance_with_width=min(fabs(f[j]),fabs(f[j])-data->width_negative);
                if(f[j]>0)
                    distance_with_width=min(fabs(f[j]),fabs(f[j])-data->width_positive);



                if (fabs(distance_with_width) <= 0.5*lip*d ||(f[j]<0 &&data->refine_in_minority))
                    return P4EST_TRUE;
            }

            // check if it does cross the interface
#ifdef P4_TO_P8
            if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 ||
                    f[3]*f[4]<0 || f[4]*f[5]<0 || f[5]*f[6]<0 || f[6]*f[7]<0 )
#else
            if (f[0]*f[1]<0 || f[0]*f[2]<0 || f[1]*f[3]<0 || f[2]*f[3]<0 )
#endif
                return P4EST_TRUE;

            return P4EST_FALSE;
        }
    }

    static p4est_bool_t refine_criteria_sl_complex(p4est_t *p4est_tmp, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
    {
        splitting_criteria_update_t4MeanFieldComplex *data = (splitting_criteria_update_t4MeanFieldComplex*) p4est_tmp->user_pointer;

        if (quad->level < data->min_lvl)
            return P4EST_TRUE;
        else if (quad->level >= data->max_lvl)
            return P4EST_FALSE;
        else
        {
            double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
            double d = sqrt(P4EST_DIM)*dx;
            double lip = data->lip;

            /* find the quadrant in p4est_tmp */
            p4est_topidx_t v_mmm = p4est_tmp->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

            double tree_xmin = p4est_tmp->connectivity->vertices[3*v_mmm + 0];
            double tree_ymin = p4est_tmp->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est_tmp->connectivity->vertices[3*v_mmm + 2];
#endif

            double xyz [] =
            {
                quad_x_fr_i(quad) + tree_xmin + dx/2.0,
                quad_y_fr_j(quad) + tree_ymin + dx/2.0
    #ifdef P4_TO_P8
                ,
                quad_z_fr_k(quad) + tree_zmin + dx/2.0
    #endif
            };

            p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
            sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
            my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
            sc_array_destroy(remote_matches);
#else
            std::vector<p4est_quadrant_t> remote_matches;
            data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

            p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

            double *phi_tmp_m;
            phi_tmp_m = data->phi_tmp_m->data();

            double *phi_tmp_p;
            phi_tmp_p = data->phi_tmp_p->data();


            double f_m[P4EST_CHILDREN];
            double f_p[P4EST_CHILDREN];


            //f_m<0 corresponds to minority region

            //check the distance criterion for f_m
            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_m[j] = phi_tmp_m[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_m[j]) <= 0.5*lip*d || (f_m[j]<0 &&data->refine_in_minority ))
                    return P4EST_TRUE;
            }

            // check if it does cross the interface
#ifdef P4_TO_P8
            if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 ||
                    f_m[3]*f_m[4]<0 || f_m[4]*f_m[5]<0 || f_m[5]*f_m[6]<0 || f_m[6]*f_m[7]<0 )
#else
            if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 )
#endif
                return P4EST_TRUE;



            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_p[j] = phi_tmp_p[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_p[j]) <= 0.5*lip*d)
                    return P4EST_TRUE;
            }

            // check if it does cross the interface
#ifdef P4_TO_P8
            if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 ||
                    f_p[3]*f_p[4]<0 || f_p[4]*f_p[5]<0 || f_p[5]*f_p[6]<0 || f_p[6]*f_p[7]<0 )
#else
            if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 )
#endif
                return P4EST_TRUE;

            return P4EST_FALSE;
        }
    }

    static p4est_bool_t refine_criteria_sl_complex_masked(p4est_t *p4est_tmp, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
    {
        splitting_criteria_update_t4MeanFieldComplexMasked *data = (splitting_criteria_update_t4MeanFieldComplexMasked*) p4est_tmp->user_pointer;

        if (quad->level < data->min_lvl)
            return P4EST_TRUE;
        else if (quad->level >= data->max_lvl)
            return P4EST_FALSE;
        else
        {
            double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
            double d = sqrt(P4EST_DIM)*dx;
            double lip = data->lip;

            /* find the quadrant in p4est_tmp */
            p4est_topidx_t v_mmm = p4est_tmp->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

            double tree_xmin = p4est_tmp->connectivity->vertices[3*v_mmm + 0];
            double tree_ymin = p4est_tmp->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est_tmp->connectivity->vertices[3*v_mmm + 2];
#endif

            double xyz [] =
            {
                quad_x_fr_i(quad) + tree_xmin + dx/2.0,
                quad_y_fr_j(quad) + tree_ymin + dx/2.0
    #ifdef P4_TO_P8
                ,
                quad_z_fr_k(quad) + tree_zmin + dx/2.0
    #endif
            };

            p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
            sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
            my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
            sc_array_destroy(remote_matches);
#else
            std::vector<p4est_quadrant_t> remote_matches;
            data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

            p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

            double *phi_tmp_m;
            phi_tmp_m = data->phi_tmp_m->data();

            double *phi_tmp_p;
            phi_tmp_p = data->phi_tmp_p->data();

            double *phi_tmp_mask;
            phi_tmp_mask=data->phi_tmp_mask->data();


            double f_m[P4EST_CHILDREN];
            double f_p[P4EST_CHILDREN];
            double f_mask[P4EST_CHILDREN];

            //       bool IsAllPositive=true;

            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          IsAllPositive =IsAllPositive && (phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ] >0);
            //        }

            //        if(IsAllPositive)
            //           return P4EST_FALSE;

            //               bool IsAllNegative=true;

            //                for(short j=0; j<P4EST_CHILDREN; ++j)
            //                {
            //                  IsAllNegative =IsAllNegative && (phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ] <=0);
            //                }

            //                if(IsAllNegative)
            //                   return P4EST_TRUE;


            //        f_m<0 corresponds to minority region, f_m>0 corresponds to majority region
            //         if one doesnt want to favorize the minority region the second condition has
            //         to be cancelled.

            // check the distance criterion for f_m
            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_m[j] = phi_tmp_m[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_m[j]) <= 0.5*lip*d || (f_m[j]<0 && f_mask[j]<0 &&data-> refine_in_minority))
                    return P4EST_TRUE;
            }

            // check if it does cross the interface for fm
#ifdef P4_TO_P8
            if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 ||
                    f_m[3]*f_m[4]<0 || f_m[4]*f_m[5]<0 || f_m[5]*f_m[6]<0 || f_m[6]*f_m[7]<0 )
#else
            if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 )
#endif
                return P4EST_TRUE;


            // check the distance criterion for f_p
            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_p[j] = phi_tmp_p[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_p[j]) <= 0.5*lip*d)
                    return P4EST_TRUE;
            }

            // check if it does cross the interface for fp
#ifdef P4_TO_P8
            if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 ||
                    f_p[3]*f_p[4]<0 || f_p[4]*f_p[5]<0 || f_p[5]*f_p[6]<0 || f_p[6]*f_p[7]<0 )
#else
            if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 )
#endif
                return P4EST_TRUE;



            // check the distance criterion for the mask
            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_mask[j] = phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_mask[j]) <= 0.5*lip*d)
                    return P4EST_TRUE;
            }

            // check if it does cross the interface for the mask
#ifdef P4_TO_P8
            if (f_mask[0]*f_mask[1]<0 || f_mask[0]*f_mask[2]<0 || f_mask[1]*f_mask[3]<0 || f_mask[2]*f_mask[3]<0 ||
                    f_mask[3]*f_mask[4]<0 || f_mask[4]*f_mask[5]<0 || f_mask[5]*f_mask[6]<0 || f_mask[6]*f_mask[7]<0 )
#else
            if (f_mask[0]*f_mask[1]<0 || f_mask[0]*f_mask[2]<0 || f_mask[1]*f_mask[3]<0 || f_mask[2]*f_mask[3]<0 )
#endif
                return P4EST_TRUE;


            return P4EST_FALSE;
        }
    }

    static p4est_bool_t refine_criteria_sl_complex_masked_uniform(p4est_t *p4est_tmp, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
    {
        splitting_criteria_update_t4MeanFieldComplexMasked *data = (splitting_criteria_update_t4MeanFieldComplexMasked*) p4est_tmp->user_pointer;

        if (quad->level < data->min_lvl)
            return P4EST_TRUE;
        else if (quad->level >= data->max_lvl)
            return P4EST_FALSE;
        else
        {
            double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
            double d = sqrt(P4EST_DIM)*dx;
            double lip = data->lip;

            /* find the quadrant in p4est_tmp */
            p4est_topidx_t v_mmm = p4est_tmp->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

            double tree_xmin = p4est_tmp->connectivity->vertices[3*v_mmm + 0];
            double tree_ymin = p4est_tmp->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est_tmp->connectivity->vertices[3*v_mmm + 2];
#endif

            double xyz [] =
            {
                quad_x_fr_i(quad) + tree_xmin + dx/2.0,
                quad_y_fr_j(quad) + tree_ymin + dx/2.0
    #ifdef P4_TO_P8
                ,
                quad_z_fr_k(quad) + tree_zmin + dx/2.0
    #endif
            };

            p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
            sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
            my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
            sc_array_destroy(remote_matches);
#else
            std::vector<p4est_quadrant_t> remote_matches;
            data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

            p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

            double *phi_tmp_m;
            phi_tmp_m = data->phi_tmp_m->data();

            double *phi_tmp_p;
            phi_tmp_p = data->phi_tmp_p->data();

            double *phi_tmp_mask;
            phi_tmp_mask=data->phi_tmp_mask->data();





            double f_m[P4EST_CHILDREN];
            double f_p[P4EST_CHILDREN];
            double f_mask[P4EST_CHILDREN];

            //       bool IsAllPositive=true;

            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          IsAllPositive =IsAllPositive && (phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ] >0);
            //        }

            //        if(IsAllPositive)
            //           return P4EST_FALSE;



            //f_m<0 corresponds to minority region, f_m>0 corresponds to majority region
            // if one doesnt want to favorize the minority region the second condition has
            // to be cancelled.

            //check the distance criterion for f_m
            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          f_m[j] = phi_tmp_m[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
            //          if (  fabs(f_m[j]) <= 0.5*lip*d )
            //            return P4EST_TRUE;
            //        }

            //        // check if it does cross the interface
            //    #ifdef P4_TO_P8
            //        if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 ||
            //            f_m[3]*f_m[4]<0 || f_m[4]*f_m[5]<0 || f_m[5]*f_m[6]<0 || f_m[6]*f_m[7]<0 )
            //    #else
            //        if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 )
            //    #endif
            //          return P4EST_TRUE;



            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          f_p[j] = phi_tmp_p[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
            //          if (  fabs(f_p[j]) <= 0.5*lip*d)
            //            return P4EST_TRUE;
            //        }

            //        // check if it does cross the interface
            //    #ifdef P4_TO_P8
            //        if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 ||
            //            f_p[3]*f_p[4]<0 || f_p[4]*f_p[5]<0 || f_p[5]*f_p[6]<0 || f_p[6]*f_p[7]<0 )
            //    #else
            //        if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 )
            //    #endif
            //          return P4EST_TRUE;


            //-------UNIFORM INSIDE--------------//

            bool IsAllNegative=true;

            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                IsAllNegative =IsAllNegative && (phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ] <=0);
            }

            if(IsAllNegative)
                return P4EST_TRUE;

            //-------Close to the interface--------------//

            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_mask[j] = phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_mask[j]) <= 0.5*lip*d)
                    return P4EST_TRUE;
            }

            // check if it does cross the interface
#ifdef P4_TO_P8
            if (f_mask[0]*f_mask[1]<0 || f_mask[0]*f_mask[2]<0 || f_mask[1]*f_mask[3]<0 || f_mask[2]*f_mask[3]<0 ||
                    f_mask[3]*f_mask[4]<0 || f_mask[4]*f_mask[5]<0 || f_mask[5]*f_mask[6]<0 || f_mask[6]*f_mask[7]<0 )
#else
            if (f_mask[0]*f_mask[1]<0 || f_mask[0]*f_mask[2]<0 || f_mask[1]*f_mask[3]<0 || f_mask[2]*f_mask[3]<0 )
#endif
                return P4EST_TRUE;


            return P4EST_FALSE;
        }
    }

    static p4est_bool_t refine_criteria_sl_complex_masked_uniform_three_levels(p4est_t *p4est_tmp, p4est_topidx_t which_tree, p4est_quadrant_t *quad)
    {
        splitting_criteria_update_t4MeanFieldComplexMaskedThreeLevels *data =
                (splitting_criteria_update_t4MeanFieldComplexMaskedThreeLevels*) p4est_tmp->user_pointer;

        if (quad->level < data->min_lvl)
            return P4EST_TRUE;
        else if (quad->level >= data->max_lvl)
            return P4EST_FALSE;
        else
        {
            double dx = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN; // dx = dy = dz
            double d = sqrt(P4EST_DIM)*dx;
            double lip = data->lip;

            /* find the quadrant in p4est_tmp */
            p4est_topidx_t v_mmm = p4est_tmp->connectivity->tree_to_vertex[which_tree*P4EST_CHILDREN + 0];

            double tree_xmin = p4est_tmp->connectivity->vertices[3*v_mmm + 0];
            double tree_ymin = p4est_tmp->connectivity->vertices[3*v_mmm + 1];
#ifdef P4_TO_P8
            double tree_zmin = p4est_tmp->connectivity->vertices[3*v_mmm + 2];
#endif

            double xyz [] =
            {
                quad_x_fr_i(quad) + tree_xmin + dx/2.0,
                quad_y_fr_j(quad) + tree_ymin + dx/2.0
    #ifdef P4_TO_P8
                ,
                quad_z_fr_k(quad) + tree_zmin + dx/2.0
    #endif
            };

            p4est_quadrant_t quad_tmp;
#ifdef P4EST_POINT_LOOKUP
            sc_array_t *remote_matches = sc_array_new(sizeof(p4est_quadrant_t));
            my_p4est_brick_point_lookup(data->p4est_tmp, NULL, data->myb, xyz, &quad_tmp, remote_matches);
            sc_array_destroy(remote_matches);
#else
            std::vector<p4est_quadrant_t> remote_matches;
            data->hierarchy->find_smallest_quadrant_containing_point(xyz, quad_tmp, remote_matches);
#endif

            p4est_locidx_t *q2n = data->nodes_tmp->local_nodes;
            p4est_tree_t *tree_tmp = p4est_tree_array_index(data->p4est_tmp->trees, quad_tmp.p.piggy3.which_tree);
            p4est_locidx_t quad_tmp_idx = quad_tmp.p.piggy3.local_num + tree_tmp->quadrants_offset;

            double *phi_tmp_m;
            phi_tmp_m = data->phi_tmp_m->data();

            double *phi_tmp_p;
            phi_tmp_p = data->phi_tmp_p->data();

            double *phi_tmp_mask;
            phi_tmp_mask=data->phi_tmp_mask->data();





            double f_m[P4EST_CHILDREN];
            double f_p[P4EST_CHILDREN];
            double f_mask[P4EST_CHILDREN];

            //       bool IsAllPositive=true;

            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          IsAllPositive =IsAllPositive && (phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ] >0);
            //        }

            //        if(IsAllPositive)
            //           return P4EST_FALSE;



            //f_m<0 corresponds to minority region, f_m>0 corresponds to majority region
            // if one doesnt want to favorize the minority region the second condition has
            // to be cancelled.

            //check the distance criterion for f_m
            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          f_m[j] = phi_tmp_m[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
            //          if (  fabs(f_m[j]) <= 0.5*lip*d )
            //            return P4EST_TRUE;
            //        }

            //        // check if it does cross the interface
            //    #ifdef P4_TO_P8
            //        if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 ||
            //            f_m[3]*f_m[4]<0 || f_m[4]*f_m[5]<0 || f_m[5]*f_m[6]<0 || f_m[6]*f_m[7]<0 )
            //    #else
            //        if (f_m[0]*f_m[1]<0 || f_m[0]*f_m[2]<0 || f_m[1]*f_m[3]<0 || f_m[2]*f_m[3]<0 )
            //    #endif
            //          return P4EST_TRUE;



            //        for(short j=0; j<P4EST_CHILDREN; ++j)
            //        {
            //          f_p[j] = phi_tmp_p[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
            //          if (  fabs(f_p[j]) <= 0.5*lip*d)
            //            return P4EST_TRUE;
            //        }

            //        // check if it does cross the interface
            //    #ifdef P4_TO_P8
            //        if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 ||
            //            f_p[3]*f_p[4]<0 || f_p[4]*f_p[5]<0 || f_p[5]*f_p[6]<0 || f_p[6]*f_p[7]<0 )
            //    #else
            //        if (f_p[0]*f_p[1]<0 || f_p[0]*f_p[2]<0 || f_p[1]*f_p[3]<0 || f_p[2]*f_p[3]<0 )
            //    #endif
            //          return P4EST_TRUE;


            //-------UNIFORM INSIDE--------------//

            bool IsAllNegative=true;

            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                IsAllNegative =IsAllNegative && (phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ] <=0);
            }

            if(IsAllNegative &&  quad->level< data->bulk_level)
                return P4EST_TRUE;

            //-------Close to the interface--------------//

            for(short j=0; j<P4EST_CHILDREN; ++j)
            {
                f_mask[j] = phi_tmp_mask[ q2n[ quad_tmp_idx*P4EST_CHILDREN + j ] ];
                if (  fabs(f_mask[j]) <= 0.5*lip*d)
                    return P4EST_TRUE;
            }

            // check if it does cross the interface
#ifdef P4_TO_P8
            if (f_mask[0]*f_mask[1]<0 || f_mask[0]*f_mask[2]<0 || f_mask[1]*f_mask[3]<0 || f_mask[2]*f_mask[3]<0 ||
                    f_mask[3]*f_mask[4]<0 || f_mask[4]*f_mask[5]<0 || f_mask[5]*f_mask[6]<0 || f_mask[6]*f_mask[7]<0 )
#else
            if (f_mask[0]*f_mask[1]<0 || f_mask[0]*f_mask[2]<0 || f_mask[1]*f_mask[3]<0 || f_mask[2]*f_mask[3]<0 )
#endif
                return P4EST_TRUE;


            return P4EST_FALSE;
        }
    }

    static double my_tanh_x(double x)
    {
        double numerical_cutoff=20;
        double exp_plus_x,exp_minus_x,tanh_x=0;


        if(ABS(x)<=numerical_cutoff)
        {
            exp_plus_x=exp(x);
            exp_minus_x=exp(-x);
            tanh_x=(exp_plus_x-exp_minus_x)/(exp_plus_x+exp_minus_x);

        }

        if(x<-numerical_cutoff)
        {
            tanh_x=-1.00;
        }
        if(x>numerical_cutoff)
        {
            tanh_x=1.00;
        }

        return tanh_x;
    }

    static double distance_of_a_point_from_a_line (double x0,double y0,double z0,double x1,double y1,double z1,double x2,double y2,double z2)
    {

        double distance=0;
        double dx10,dy10,dz10,dx21,dy21,dz21;
        dx10=x1-x0;
        dy10=y1-y0;
        dz10=z1-z0;
        dx21=x2-x1;
        dy21=y2-y1;
        dz21=z2-z1;
        distance=(pow(dx10,2)+pow(dy10,2)+pow(dz10,2))*(pow(dx21,2)+pow(dy21,2)+pow(dz21,2));
        distance=distance-pow(dx10*dx21+dy10*dy21+dz10*dz21,2);
        distance=distance/(pow(dx21,2)+pow(dy21,2)+pow(dz21,2));

        return distance;
    }

protected:



    //mpi fields
    Session *mpi_session;
    // p4est fields
    mpi_context_t mpi_context, *mpi;
    PetscErrorCode      ierr;
    cmdParser cmd;
    splitting_criteria_cf_t *data_p4est_cf;
    splitting_criteria_cf_t *data_p4est_cf_visualization;



    //---------p4est-------------------------------------------//

    p4est_t            *p4est;
    p4est_nodes_t      *nodes;
    p4est_ghost_t* ghost;
    p4est_connectivity_t *connectivity;
    my_p4est_brick_t *brick;
    int nx_trees,ny_trees,nz_trees;
    my_p4est_hierarchy_t *hierarchy;
    my_p4est_node_neighbors_t *node_neighbors;





    //---------p4est after remeshing  --------//

    p4est_t            *p4est_remeshed;
    p4est_nodes_t      *nodes_remeshed;
    p4est_ghost_t* ghost_remeshed;
    p4est_connectivity_t *connectivity_remeshed;
    my_p4est_brick_t *brick_remeshed;
    my_p4est_hierarchy_t *hierarchy_remeshed;
    my_p4est_node_neighbors_t *node_neighbors_remeshed;


    //--------------p4est4visualization-----------------------//

    p4est_t            *p4est_visualization;
    p4est_nodes_t      *nodes_visualization;
    p4est_ghost_t* ghost_visualization;
    p4est_connectivity_t *connectivity_visualization;
    my_p4est_brick_t *brick_visualization;
    my_p4est_hierarchy_t *hierarchy_visualization;
    my_p4est_node_neighbors_t *node_neighbors_visualization;


    //--------------------------------------------------//

    // polymer fields
    double f,X_ab;
    double f_initial;
    double Lx; double Lx_physics;

    Diffusion::phase  myPhase;

    // numerical fields

    double Tf;
    int frequency_length_update;
    bool debug2; bool load_file;
    double my_level;
    double my_T;
    int min_level; int max_level;
    int psdt;
    std::string name;
    PetscScalar lambda_plus,lambda_minus;
    int N_iterations;
    int N_t;
    int N_mean_field_iteration;

    int nb_splits;



    // optimization fields
    double *energy;
    double *energy_w;
    double *energy_logQ;
    double *shape_force;
    double *pressure_force;
    double *exchange_force;
    double  *total_force;
    double   *rhoAAverage;
    double   *rhoBAverage;
    double   *rho0Average;
    double *length_series;
    double *stress_trace;
    double *std_Ordered;
    double *std_Disordered;
    double *order_Ratio;
    double *VRealTime;
    double *wp_average_on_contour;
    double *wp_variance_on_contour;
    //   double *wp_averaged_on_zero_level_set;
    //   double *wp_std_on_zero_level_set;
    //   double *wm_averaged_on_zero_level_set;
    //   double *wm_std_on_zero_level_set;


    int mask_counter;

    double de_predicted_from_level_set_change_current_step_before_lagrange_t;
    double de_predicted_from_level_set_change_current_step_after_lagrange_t;
    double de_predicted_from_level_set_change_current_step_after_cfl_t;
    double de_predicted_from_level_set_change_current_step_after_reinitialyzation_t;
    double de_predicted_from_level_set_change_current_step_after_volume_conservation_t;
    double de_predicted_from_level_set_change_current_step_after_t;
    double prediction_error_level_set_t;
    double de_mask_t;


    PetscBool setup_finite_difference_solver;


    double osher_and_santosa_term;
    double stress_term;
    double simple_term_wm;
    double simple_term_wp;
    double naive_term;
    double correction_term_for_uncounstrained_volume_log_Q;
    double correction_term_for_uncounstrained_volume_E_w;


    //   double *de_predicted_from_level_set_change_current_step_before_lagrange;
    //   double *de_predicted_from_level_set_change_current_step_before_volume_conservation;
    double *de_predicted_from_level_set_change_current_step_after;
    double *prediction_error_level_set;
    double *de_mask;


    double *dphi_step_1;
    double *dphi_step_2;
    double *dphi_step_3;
    double *dphi_step_4;
    double *dphi_step_5;

    double *dphi_step_0_3;
    double *dphi_step_0_4;
    double *dphi_step_0_5;

    //scft fields
    int i_mean_field_iteration;
    double shape_force_t;

    // petsc fields
    Vec phi; Vec phi_vizualization;
    Vec neuman_stride;
    Vec polymer_mask;
    Vec dmask_dx;
    Vec dmask_dy;
    Vec dmask_dz;
    PetscScalar *dmask_dx_local;
    PetscScalar *dmask_dy_local;
    PetscScalar *dmask_dz_local;

    PetscBool pressure_gradient_velocity;


    // Vec polymer_mask_advected;
    Vec polymer_velocity_x;
    Vec polymer_velocity_y;
    Vec polymer_velocity_z;

    double Sxx,Syy,Szz,Sxy,Sxz,Syz;

    Vec snn;

    double stress_tensor_trace;

    // kmeans fields
    Vec Ibin;

    PetscScalar *Ibin_local;

    Vec Ibin_nodes;
    PetscScalar *Ibin_nodes_local;
    PetscScalar *Ibin_local_m;

    Vec Ibin_nodes_m;
    PetscScalar *Ibin_nodes_local_m;

    PetscScalar *Ibin_local_p;

    Vec Ibin_nodes_p;
    PetscScalar *Ibin_nodes_local_p;

    Vec ix1;
    Vec ix2;
    double c1;
    double c2;
    double c1_global;
    double c2_global;
    double A1,A2;
    double A1_global,A2_global;
    double e1; double e2;
    double e1_global,e2_global;
    double e1_global_2,e2_global_2;
    int n_colour1_local,n_colour2_local,n_colour1_global,n_colour2_global;
    int kmeans_iterator;

    int n_change_internal_interfaces;

    double non_conservation_scalar_term;

    inline  string get_my_meshing_startegy()
    {
        switch(this->my_meshing_strategy)
        {
        case one_level_set:
            return "one_level_set";
            break;
        case two_level_set:
            return "two_level_set";
            break;
        case two_level_set_with_interface:
            return "two_level_set_with_interface";
            break;
            //       case complex_level_set:
            //           return "complex_level_set";
            //           break;
            //       case minority_level_set:
            //           return "minority_level_set";
            //           break;
        default:
            return "";
            break;
        }
    }
public:


    PetscBool volumetric_derivative;
    PetscBool velocity_at_interface_only;
    int mask_conservation_period;
    bool reinitialize;

    int vtk_period;
    int mask_period;
    enum strategy_meshing
    {
        one_level_set,
        two_level_set,
        two_level_set_with_interface
        //       complex_level_set,
        //       minority_level_set
    };

    strategy_meshing my_meshing_strategy;
    PetscBool differential_change;

    PetscBool compute_energy_difference_diff;

    double a_radius;//0.5;
    double a_ellipse,b_ellipse,c_ellipse;

    inline void setReinitialise2False(){this->reinitialize=false;}
    inline void setReinitialise2True(){this->reinitialize=true;}

    inline void set_mask_conservation_period(int mask_conservation_period){this->mask_conservation_period=mask_conservation_period;}

    PetscBool reseed;

    double dphi;
    enum mask_strategy
    {
        sphere_mask,
        cube_mask,
        annular_mask,
        ellipse_mask,
        helix_mask,
        drop_mask,
        v_shape,
        l_shape,
        l_shape_3d,
        text_file_mask,
        text_file_mask_from_fine_to_coarse,
        ThreeD_from_TwoD_text_file_mask,
        ThreeDParallel_from_TwoDSerial_text_file_mask,
        ThreeD_from_TwoD_text_file_mask_from_fine_to_coarse,
        from_inverse_design,
        from_inverse_design_with_level_set
    };

    enum seed_strategy
    {
        sphere_seed,
        cube_seed,
        bcc_seed,
        helix_seed,
        from_wp,
        terrace,
        rectangle,
        l_shape_seed,
        text_file_seed,
        text_file_seed_from_fine_to_coarse,
        ThreeD_from_TwoD_text_file_seed,
        ThreeD_from_TwoD_text_file_seed_from_fine_to_coarse
    };



    double xhi_w_a;
    double xhi_w_b;
    double xhi_w_p;
    double xhi_w_m;
    double xhi_w;

    double zeta_n_inverse;
    double interaction_flag;
    double alpha_wall;

    PetscBool terracing;
    PetscScalar y_terrace_floor;

    std::string text_file_mask_str;
    std::string text_file_seed_str;

    seed_strategy my_seed_strategy;
    PetscBool advance_fields_scft;
    PetscBool advance_fields_level_set;
    PetscBool advance_fields_scft_advance_mask_level_set;
    PetscBool conserve_reaction_source_volume;
    PetscBool uniform_normal_velocity;

    PetscBool conserve_shape_volume;

    PetscBool regenerate_potentials_from_mask;

    double alpha_cfl;
    int start_remesh_i;
    int stop_remesh_i;
    PetscBool papers_1_and_2_seed;
    double interface_width;
    double interface_width_positive;
    double interface_width_negative;





    inline void set_source_optimization_parameters(PetscBool conserve_reaction_source_volume,PetscBool uniform_normal_velocity,
                                                   MeanField::seed_strategy my_seed_strategy, double alpha_cfl)
    {
        this->advance_fields_scft=PETSC_FALSE;
        this->advance_fields_level_set=PETSC_TRUE;
        this->uniform_normal_velocity=uniform_normal_velocity;
        this->conserve_reaction_source_volume=conserve_reaction_source_volume;
        this->my_seed_strategy=my_seed_strategy;
        this->alpha_cfl=alpha_cfl;
        this->myDiffusion->seed_optimization=PETSC_TRUE;

    }

    double source_volume_initial;

    enum shape_optimization_strategy
    {
        stress_tensor_optimization,
        level_set_optimization,
        set_velocity_manually
    };

    enum level_set_advection_numerical_scheme
    {
        euler_advection,
        semi_lagrangian_advection,
        gudonov_advection
    };

    level_set_advection_numerical_scheme my_level_set_advection_numerical_scheme;


    enum velocity_strategy
    {
        pressure_velocity,
        shape_derivative_velocity,
        pressure_velocity_with_surface_tension_velocity,
        shape_derivative_with_surface_tension_velocity,
        surface_tension_velocity,
        anti_surface_tension_velocity
    };

    double surface_tension;

    velocity_strategy my_velocity_strategy;

    inline void set_velocity_parameters(double surface_tension, velocity_strategy my_velocity_strategy)
    {
        this->surface_tension=surface_tension;
        this->my_velocity_strategy=my_velocity_strategy;
    }

    Vec kappa;

    double de_predicted_from_level_set_change_next_step;
    double de_predicted_from_level_set_change_current_step;

    double de_predicted_total;
    double de_in_fact_total;
    double de_prediction_error;

    Vec phi_seed_old_on_old_p4est;
    Vec phi_seed_new_on_old_p4est;

    Vec phi_seed_initial;

    Vec phi_seed;
    Vec negative_phi_seed;
    Vec phi_wall;

    double volume_phi_seed;
    double volume_negative_phi_seed;
    double lagrange_multiplier;
    double surface_phi_seed;

    double phi_bar;

    Vec polymer_shape_stored;
    Vec new_polymer_shape_on_old_forest;

    Vec wm_stored,wp_stored;
    Vec rho_a_stored,rho_b_stored;
    Vec fm_stored,fp_stored;
    Vec last_q_stored;
    Vec log_vector_for_plot;

    Vec last_q_stored_normalized_volume;
    Vec last_q_stored_normalized_surface;
    double q_surface_mean_value;

    Vec delta_phi_for_prediction;
    Vec delta_phi_for_advection;
    double ls_tolerance;
    double tol_v_loss;
    double polymer_mask_radius_for_initial_wall;

    shape_optimization_strategy my_shape_optimization_strategy;
    mask_strategy my_mask_strategy;

    double domain_surface,domain_volume,effective_radius;

    PetscBool minimum_io;
    PetscBool uniform_mesh_with_mask;
    PetscBool extend_advect;
    PetscBool change_manually_level_set;
    PetscInt  extension_advection_period;
    PetscInt stress_tensor_computation_period;
    PetscInt extension_advection_counter;
    // PetscBool write_to_vtk=PETSC_TRUE;
    PetscBool do_not_write_to_vtk_in_any_case;
    PetscBool remesh_my_forest;
    PetscInt remesh_period;
    PetscBool write2VtkShapeOptimization;




    Diffusion *myDiffusion;
    std::string IO_path;
    inline std::string convert2FullPath(std::string file_name)
    {
        std::stringstream oss;
        std::string mystr;
        oss <<this->IO_path <<file_name;
        mystr=oss.str();
        return mystr;
    }

    inline std::string get_full_string(std::string file_name_short)
    {
        std::string file_name_long;
        //file_name_long=this->convert2FullPath(file_name_short);

        std::stringstream out;
        out<<file_name_short<<this->myDiffusion->get_my_numerical_scheme_string()<<"_"<<this->myDiffusion->get_my_casl_diffusion_method_string()<<
             "_Lx_"<<this->Lx<<"_lambda_"<<this->lambda_0<<"_minLevel_"<<this->min_level<<"_maxLevel_"<<this->max_level<<"_f_"<<this->f
          <<"_Xhi_"<<this->X_ab;

        file_name_long=out.str();
        return file_name_long;

    }
    inline std::string get_full_string_psdt(std::string file_name_short)
    {
        if(this->myMeanFieldPlan->write2VtkMovie)
        {
        std::string file_name_long;
        //file_name_long=this->convert2FullPath(file_name_short);
        std::stringstream out;
        out<<file_name_short<<this->myDiffusion->get_my_numerical_scheme_string()<<"_"<<this->myDiffusion->get_my_casl_diffusion_method_string()<<
             "_Lx_"<<this->Lx<<"_lambda_"<<this->lambda_0<<"_minLevel_"<<this->min_level<<"_maxLevel_"<<this->max_level<<"_f_"<<this->f
          <<"_Xhi_"<<this->X_ab<<"."<<this->i_mean_field_iteration;
        file_name_long=out.str();
        return file_name_long;
        }
        else
        {
            this->get_full_string(file_name_short);
        }

    }

    PetscBool multi_alpha_cfl;
    PetscBool use_petsc_shift;
    PetscBool reinitialyze_level_set_with_tolerance;
    PetscBool perturb_level_set_after_reinitialyzation;
    PetscBool inverse_design_litography;
    double lambda_0;

    inline void initialyze_default_parameters()
    {

        this->lambda_0=1.00;
        this->inverse_design_litography=PETSC_FALSE;
        this->write2VtkShapeOptimization=PETSC_FALSE;

        this->my_interpolation_method=quadratic;

        this->Lx_physics=4.00;
        this->Tf=1.00;
        this->frequency_length_update=6;
        this->my_level=9;
        this->debug2=true;
        this->load_file=true;
        this->psdt=0.00;
        this->nb_splits=0.00;
        this->mask_counter=0.00;
        this->mask_conservation_period=1;
        this->reinitialize=true;
        this->setup_finite_difference_solver=PETSC_TRUE;
        this->osher_and_santosa_term=0;
        this->stress_term=0;
        this->simple_term_wm=0;this->simple_term_wp=0;
        this->naive_term=0;
        this->pressure_gradient_velocity=PETSC_FALSE;
        this->e1=0; this->e2=0;
        this->n_colour1_local=0;
        this->n_colour2_local=0;
        this->n_colour1_global=0;
        this->n_colour2_global=0;
        this->kmeans_iterator=0;
        this->tol_v_loss=0.1;
        this->n_change_internal_interfaces=1;
        this->volumetric_derivative=PETSC_TRUE;
        this->velocity_at_interface_only=PETSC_FALSE;
        this->vtk_period=1;
        this->mask_period=1;
        this->my_meshing_strategy=MeanField::two_level_set;
        this->differential_change=PETSC_FALSE;
        this->compute_energy_difference_diff=PETSC_TRUE;
        this->a_radius=1.00;//0.5;
        this->polymer_mask_radius_for_initial_wall=this->a_radius*this->Lx/3.00;

        this->reseed=PETSC_FALSE;
        this->dphi=-0.005;
        this->xhi_w_a=0.00;
        this->xhi_w_b=0.00;
        this->xhi_w_p=0.00;
        this->xhi_w_m=0.00;
        this->xhi_w=0;

        this->zeta_n_inverse=0.001;
        this->interaction_flag=0.00;
        this->alpha_wall=4.00;
        this->terracing=PETSC_FALSE;
        this->my_seed_strategy=sphere_seed;
        this->advance_fields_scft=PETSC_TRUE;
        this->advance_fields_level_set=PETSC_FALSE;
        this->advance_fields_scft_advance_mask_level_set=PETSC_FALSE;
        this->conserve_reaction_source_volume=PETSC_FALSE;
        this->uniform_normal_velocity=PETSC_FALSE;
        this->conserve_shape_volume=PETSC_FALSE;
        this->alpha_cfl=1.8;
        this->start_remesh_i=0;
        this->stop_remesh_i=0;

        this->my_level_set_advection_numerical_scheme=MeanField::gudonov_advection;

        this->regenerate_potentials_from_mask=PETSC_TRUE;
        this->multi_alpha_cfl=PETSC_FALSE;

        this->use_petsc_shift=PETSC_FALSE;
        this->reinitialyze_level_set_with_tolerance=PETSC_TRUE;
        this->perturb_level_set_after_reinitialyzation=PETSC_FALSE;
        this->papers_1_and_2_seed=PETSC_FALSE;


        this->de_predicted_from_level_set_change_next_step=0;
        this->de_predicted_from_level_set_change_current_step=0;


        this->ls_tolerance=1.00/10000.00;



        this->my_shape_optimization_strategy=set_velocity_manually;
        this->my_mask_strategy=MeanField::sphere_mask;

        this->domain_surface=0;
        this->domain_volume=0;
        this->effective_radius=0;

        this->minimum_io=PETSC_TRUE;
        this->uniform_mesh_with_mask=PETSC_TRUE;
        this->extend_advect=PETSC_FALSE;
        this->change_manually_level_set=PETSC_FALSE;
        this->extension_advection_period=2000;
        this->stress_tensor_computation_period=100;
        this->extension_advection_counter=0;
        // PetscBool write_to_vtk=PETSC_TRUE;
        this->do_not_write_to_vtk_in_any_case=PETSC_TRUE;
        this->remesh_my_forest=PETSC_TRUE;
        this->remesh_period=10;






    }
    MeanField();
    MeanField(int argc, char* argv[], int nx_trees, int ny_trees, int nz_trees, int min_level, int max_level, int t_iterations, int mean_field_iterations,
              double f, double Xhi_AB, double Lx, double Lx_physics,
              Diffusion::phase myPhase, Diffusion::casl_diffusion_method my_casl_diffusion_method, Diffusion::numerical_scheme my_numerical_scheme, MeanField::mask_strategy my_mask_strategy,
              std::string my_io_path, double ax, double by, double cz, int extension_advection_period, int stress_computation_period, int remesh_period,
              double lambda, strategy_meshing my_meshing_strategy, PetscBool setup_finite_difference_solver,  double polymer_mask_radius,
              std::string text_file_seed_str, std::string text_file_mask_str, std::string text_file_fields,
              std::string text_file_field_wp, std::string text_file_field_wm, MeanFieldPlan *myMeanFieldPlan);

    MeanField(int argc, char* argv[], int nx_trees, int ny_trees, int nz_trees, int min_level, int max_level, int t_iterations, int mean_field_iterations,
              double f, double Xhi_AB, double Lx, double Lx_physics,
              Diffusion::phase myPhase, Diffusion::casl_diffusion_method my_casl_diffusion_method, Diffusion::numerical_scheme my_numerical_scheme, MeanField::mask_strategy my_mask_strategy,
              std::string my_io_path, double ax, double by, double cz, int extension_advection_period, int stress_computation_period, int remesh_period,
              double lambda, strategy_meshing my_meshing_strategy, PetscBool setup_finite_difference_solver,  double polymer_mask_radius,
              std::string text_file_seed_str, std::string text_file_mask_str, std::string text_file_fields,
              std::string text_file_field_wp, std::string text_file_field_wm, MeanFieldPlan *myMeanFieldPlan, inverse_litography *myInverseLitography);


    int compute_advection_velocity();
    int compute_advection_velocity_from_stress_tensor();
    int compute_advection_velocity_from_level_set();
    int set_advection_velocity();
    int extension_advection_algo_semi_lagrangian();
    int extension_advection_algo_euler_or_gudonov();
    int change_level_sets();
    int compute_lagrange_multiplier_delta_phi();
    int compute_lagrange_multiplier_delta_phi_volume_integral();
    int compute_lagrange_multiplier_delta_phi_terracing();
    int compute_force_delta_phi();
    int ensure_volume_conservation();
    int ensure_volume_conservation_terracing();
    double compute_volume_source();
    double compute_volume_source_remeshed();
    double compute_volume_source_terrace();
    double compute_volume_source_remeshed_terrace();
    int compute_center_of_mass(double &xc,double &yc, double &zc);
    int compute_delta_phi(double &delta_phi, double &delta_phi_0);

    double compute_minimum_time_step_for_advection();

    int initialyze_MeanField(int argc, char* argv[], int nx_trees, int ny_trees, int nz_trees, int min_level, int max_level, int t_iterations, int mean_field_iterations,
                             double f, double Xhi_AB, double Lx, double Lx_physics, Diffusion::phase myPhase,
                             Diffusion::casl_diffusion_method my_casl_diffusion_method, Diffusion::numerical_scheme my_numerical_scheme,
                             double ax, double by, double cz, string text_file_fields, string text_file_field_wp, string text_file_field_wm);



    int constructForestOfTrees(int nx_trees,int ny_trees,int nz_trees);
    int remeshForestOfTrees(int nx_trees,int ny_trees,int nz_trees);

    int remeshForestOfTrees_parallel(int nx_trees,int ny_trees,int nz_trees);
    int remeshForestOfTrees_parallel_with_interface(int nx_trees,int ny_trees,int nz_trees);
    int remeshForestOfTrees_parallel_two_level_set(int nx_trees,int ny_trees,int nz_trees);
    int remeshForestOfTrees_parallel_two_level_set_with_interface(int nx_trees,int ny_trees,int nz_trees);
    int constructForestOfTreesVisualization(int nx_trees,int ny_trees,int nz_trees);
    int remeshForestOfTreesVisualization(int nx_trees, int ny_trees, int nz_trees);
    int remeshForestOfTreesVisualization_parallel(int nx_trees, int ny_trees, int nz_trees);
    int remeshForestOfTreesVisualization_parallel_two_level_set(int nx_trees, int ny_trees, int nz_trees);
    int remeshForestOfTreesVisualization_parallel_two_level_set_with_interface(int nx_trees, int ny_trees, int nz_trees);
    int compute_ibin_nodes_on_intermediate_grids( my_p4est_node_neighbors_t &qnnn,Vec phi_n,  double *phi_np1,p4est_t *p4est_np1, p4est_nodes_t *nodes_np1);
    int compute_normal_to_the_interface(Vec *level_set_interface);
    int print_3D_Vector(Vec *nx,Vec *ny,Vec *nz, string file_name_str);
    int print_2D_Vector(Vec *nx,Vec *ny, string file_name_str);
    int print_2D_VectorWithForest(Vec *nx,Vec *ny, string file_name_str);



    int clean_mean_field_step();
    int swap_p4ests();
    int create_polymer_mask();
    int create_polymer_mask_from_inverse_litography();
    int create_polymer_mask_from_inverse_litography_with_level_set();
    int create_polymer_mask_square(Vec *phi_square,double ax,double by,double cz);
    int create_polymer_mask_cube();
    int create_polymer_mask_annular_cube();
    int create_polymer_mask_helix();
    int create_polymer_mask_moving_ellipse(double ax,double by,double cz);
    int create_polymer_mask_drop(double ax, double by, double cz);
    int create_polymer_mask_v_shape(double ax, double by, double cz);
    int create_polymer_mask_l_shape(double ax, double by, double cz);
    int create_polymer_mask_3d_l_shape_old(double ax,double by,double cz);
    int create_polymer_mask_3d_l_shape(double ax,double by,double cz);
    int compute_volume_and_surface_domain();
    int plot_log_vec(Vec *v2plogplot,std::string  file_name,PetscBool write2vtk=PETSC_FALSE);

    int setNumericalParameters(int min_level,int max_level,int t_iterations,int mean_field_iterations);
    int setPolymerParameters(double f, double Xhi_AB, double Lx, Diffusion::phase myPhase);
    int MeanField_initialyze_petsc(int argc, char* argv[]);
    int Optimize();

    int evolve_statistical_fields();
    int seed_advection();
    int advect_internal_interface();
    int change_internal_interface();
    int set_initial_seed_from_level_set();
    int set_initial_wall_from_level_set();
    int set_initial_wall_from_level_set_rectangle();
    int set_initial_wall_from_level_set_terracing();
    int set_initial_seed_from_level_set_wp();
    int set_initial_seed_from_level_set_helix();
    int set_initial_seed_from_level_set_bcc();
    int set_advected_fields_from_level_set();
    int set_advected_wall_from_level_set(PetscBool destroy_phi_wall,PetscBool compute_on_remeshed_p4est);
    int correct_advected_wall_from_level_set_terrace(PetscBool destroy_phi_wall,PetscBool compute_on_remeshed_p4est);
    int regenerate_potentials_from_level_set(Vec *field_seed);



    int predict_energy_change_from_level_set_change();
    int compute_mask_energy_from_level_set(double &mask_energy, Vec *phi_ls);
    int compute_mask_energy_from_level_set_terracing(double &mask_energy, Vec *phi_ls);
    int predict_energy_change_from_polymer_level_set_change();
    int printSCFTShapeOptimizationEvolution();

    int compute_phi_bar();
    int generate_masked_fields();
    int design_seed();

    int extend_petsc_vector(Vec *phi_ls,Vec * v2extend,int number_of_bands_extension=5);
    int extend_petsc_vector_with_stride(Vec *phi_ls,Vec * v2extend,int number_of_bands_extension=5);

    int scatter_petsc_vectors(int num,...);
    int scatter_petsc_vector(Vec * v2scatter);

    int evolve_statistical_fields_explicit();
    int evolve_statistical_fields_implicit();
    int segmentDiffusionPotentials();
    int segmentDiffusionPotentialsW(Vec w2Segment, Vec *IBinSegmented);
    int segmentDiffusionPotentialsWbyNodes(Vec *w2Segment, Vec *IBinSegmented);
    int mapBinaryImageFromCell2Nodes();
    int mapBinaryImageFromCell2NodesW(Vec *Icell2Map,Vec *InodesMapped);
    int my_p4est_vtk_write_all_periodic_adapter(Vec *myNodesVector4Paraview, string file_name,PetscBool write_anyway=PETSC_FALSE);
    int my_p4est_vtk_write_all_periodic_adapter_psdt(Vec *myNodesVector4Paraview, string file_name,PetscBool write_anyway=PETSC_FALSE);
    int reinitialyzeBinaryPotential();
    int createNewForest();
    int remapFromOldForestToNewForest();

    int remapFromOldForestToNewForest2();
    int remapFromNewForestToOldForest(p4est_t *my_p4est_new, p4est_nodes_t *nodes_new, p4est_ghost_t *ghost_new, my_p4est_brick_t *mybrick,Vec *old_data_structure,Vec *vec_data, Vec  *vec2Remap);
    int remapFromOldForestToNewForest(p4est_t *my_p4est_new, p4est_nodes_t *nodes_new, p4est_ghost_t *ghost_new, my_p4est_brick_t *mybrick,
                                      Vec *vec_data_on_old_structure, Vec  *vec2Remap);

    int get_field_from_text_file(Vec *v2Fill,std::string file_name);
    int get_3Dfield_from_2Dtext_file(Vec *v2Fill,std::string file_name,double delta_lz);
    int get_3DParallelField_from_2DSerialtext_file(Vec *v2Fill,std::string file_name,double delta_lz);
    int get_3D_coarse_field_from_2D_fine_text_file(Vec *v2Fill, string file_name, double delta_lz,int file_level);
    int get_coarse_field_from_fine_text_file(Vec *v2Fill, string file_name,int file_level);




    int create_spatial_xhi_wall();



    int printOptimizationEvolution();
    int printShapeOptimizationEvolution();
    int printLevelSetEvolution();
    int printNumericalData();
    int printNumericalDataEvolution();
    int interpolate_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid, string str_file, bool compressed_io);

    int interpolate_to_fine_and_print_vec_to_uniform_grid(Vec *vec2PrintOnUniformGrid, string str_file,int file_level);


    int printStressTensorEvolution();
    int computeSegmentationError(Vec *Icell);
    int computeSegmentationErrorW(Vec *Ibin2,Vec *Icell2);
    int computeSegmentationErrorWbyNodes(Vec *Ibin2,Vec *Icell2);

    void set_fake_vx_vy_vz();

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


    void extend_advect_shape_domain();
    void smooth_seed_for_papers_1_and_2();
    void write2LogEnergyChanges();
    void create_diffuse_mask();
    void advect_diffuse_mask();
    void create_internal_interface();
    void optimize_internal_interface();
    void createPhiWall4Neuman();
    void reCreatePhiWall4Neuman();

    void LaunchMeanField(MeanField::mask_strategy my_mask_strategy, MeanField::strategy_meshing my_meshing_strategy, double polymer_mask_radius, PetscBool setup_finite_difference_solver, string text_file_seed_str, int remesh_period, double f, MeanFieldPlan *myMeanFieldPlan, string my_io_path, int extension_advection_period, double lambda, string text_file_mask_str, int stress_computation_period);
    int set_initial_diffude_mask();
    void write_mean_field_to_txt_and_vtk();
};

#ifdef P4_TO_P8
class numericalLevelSet:public CF_3
{
    double lip;

public:
    InterpolatingFunctionNodeBase *myInterpolatingFunction;
    numericalLevelSet(double lip,Vec *v,p4est_t *p4est,p4est_nodes_t *nodes, p4est_ghost_t *ghost,my_p4est_brick_t *brick,my_p4est_node_neighbors_t *qnnn)
    {
        this->lip=lip;
        this->myInterpolatingFunction=new InterpolatingFunctionNodeBase(p4est, nodes, ghost, brick,qnnn);
        this->myInterpolatingFunction->set_input_parameters(*v,linear);
    }

    double operator()(double x,double y,double z)const
    {
        return this->myInterpolatingFunction->operator ()(x,y,z);
    }
};

#else
class numericalLevelSet:public CF_2
{
    double lip;
    p4est_t *p4est;

public:
    InterpolatingFunctionNodeBase *myInterpolatingFunction;
    numericalLevelSet(double lip,Vec *v,p4est_t *p4est,p4est_nodes_t *nodes, p4est_ghost_t *ghost,my_p4est_brick_t *brick,my_p4est_node_neighbors_t *qnnn)
    {

        this->lip=lip;
        this->p4est=p4est;
        this->myInterpolatingFunction=new InterpolatingFunctionNodeBase(p4est,nodes,ghost,brick,qnnn);
        this->myInterpolatingFunction->set_input_parameters(*v,linear);
    }
    double operator()(double x,double y)const
    {
        std::cout<<"Try to interpolate "<<x<<" "<<y<<" "<<this->p4est->mpirank<<std::endl;
        //        return this->myInterpolatingFunction->operator()(x,y);
        double return_value=(*this->myInterpolatingFunction)(x,y);
        std::cout<<"successfull interpolation "<<x<<" "<<y<<" "<<return_value<<" "<<this->p4est->mpirank<<std::endl;
        return return_value;
        //        double xyz [P4EST_DIM] ={x,y};

        //        // buffer the point
        //        this->myInterpolatingFunction->add_point_to_buffer(0, xyz);
        //        PetscSynchronizedFlush(this->p4est->mpicomm);
        //        double output_vec;
        //        this->myInterpolatingFunction->interpolate(&output_vec);
        //         std::cout<<"successfull interpolation "<<output_vec<<std::endl;
        //        return output_vec;

    }
};
#endif

