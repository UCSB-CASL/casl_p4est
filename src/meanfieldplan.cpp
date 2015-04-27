#include "meanfieldplan.h"

MeanFieldPlan::MeanFieldPlan()
{

    this->periodic_xyz=false;
    this->px=false;
    this->py=false;
    this->pz=false;
    this->first_order_reinitialyzation=true;
    this->neuman_with_mask=false;
    this->dirichlet_with_mask=false;
    this->get_stride_from_neuman=false;
    this->refine_in_minority=true;
    this->LipshitzConstant=1.2;
    this->write2Vtk=false;
    this->write2VtkMovie=true;
    this->kappaA=0.00;
    this->kappaB=0.00;
    this->robin_bc=false;
    this->refine_at_interface_more=false;
    this->refine_with_width=false;
    this->bulk_level=6;
    this->alpha_width=0.99;
    this->alpha_r_helix=1.00/32.00;
    this->writeLastq2TextFile=false;
    this->optimize_robin_kappa_b=false;
    this->robin_optimization_period=100;
    this->seed_frequency=36.00;
    this->robin_lambda=1.00;
    this->regenerate_potentials=false;
    this->compressed_io=false;

}
