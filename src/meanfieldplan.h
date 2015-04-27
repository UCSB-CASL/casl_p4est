#ifndef MEANFIELDPLAN_H
#define MEANFIELDPLAN_H

class MeanFieldPlan
{



public:
    bool first_order_reinitialyzation;
    bool periodic_xyz;
    bool px,py,pz;
    double LipshitzConstant;
    bool neuman_with_mask;
    bool dirichlet_with_mask;
    bool get_stride_from_neuman;
    bool refine_in_minority;
    bool write2Vtk;
    double kappaA;
    double kappaB;
    bool robin_bc;
    bool refine_at_interface_more;
    int bulk_level;
    bool refine_with_width;
    double alpha_width;
    bool writeLastq2TextFile;
    bool optimize_robin_kappa_b;
    int robin_optimization_period;
    double robin_lambda;
    bool regenerate_potentials;
    bool write2VtkMovie;
    double alpha_r_helix;
    bool compressed_io;
    double seed_frequency;
    MeanFieldPlan();
};

#endif // MEANFIELDPLAN_H
