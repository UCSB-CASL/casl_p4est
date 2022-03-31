#include "my_p4est_stefan_with_fluids.h"


#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>

#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>


#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_multialloy.h>
#include <src/my_p4est_macros.h>

#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>

#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif



my_p4est_stefan_with_fluids_t::my_p4est_stefan_with_fluids_t()
{





//  class interfacial_bc_temp: public CF_DIM{
//private:
//    my_p4est_node_neighbors_t* ngbd_bc_temp;

//    // Curvature interp:
//    my_p4est_interpolation_nodes_t* kappa_interp;

//    // Normals interp:
//    my_p4est_interpolation_nodes_t* nx_interp;
//    my_p4est_interpolation_nodes_t* ny_interp;
//    // TO-DO: add 3d case


//public:
//    void set_kappa_interp(my_p4est_node_neighbors_t* ngbd_, Vec &kappa){
//      ngbd_bc_temp = ngbd_;
//      kappa_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
//      kappa_interp->set_input(kappa, linear);

//    }
//    void clear_kappa_interp(){
//      kappa_interp->clear();
//      delete kappa_interp;
//    }
//    void set_normals_interp(my_p4est_node_neighbors_t* ngbd_, Vec &nx, Vec &ny){
//      ngbd_bc_temp = ngbd_;
//      nx_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
//      nx_interp->set_input(nx, linear);

//      ny_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
//      ny_interp->set_input(ny, linear);
//    }
//    void clear_normals_interp(){
//      kappa_interp->clear();
//      delete kappa_interp;
//    }
//    double Gibbs_Thomson(double sigma_, DIM(double x, double y, double z)) const {
//      switch(problem_dimensionalization_type){
//      // Note slight difference in condition bw diff nondim types -- T0 vs Tinf
//      case NONDIM_BY_FLUID_VELOCITY:{
//        return (theta_interface - (sigma_/l_char)*((*kappa_interp)(x,y))*(theta_interface + T0/deltaT));
//      }
//      case NONDIM_BY_SCALAR_DIFFUSIVITY:{
//        return (theta_interface - (sigma_/l_char)*((*kappa_interp)(x,y))*(theta_interface + Tinfty/deltaT));
//      }
//      case DIMENSIONAL:{
//        return (Tinterface*(1 - sigma_*((*kappa_interp)(x,y))));
//      }
//      default:{
//        throw std::runtime_error("Gibbs_thomson: unrecognized problem dimensionalization type \n");
//      }
//      }
//    }

//  };


}
