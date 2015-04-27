#ifndef FIELDPROCESSOR_H
#define FIELDPROCESSOR_H

#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include<math.h>


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

class FieldProcessor
{
    int ierr;



public:

    FieldProcessor(){}
    ~FieldProcessor(){}

    /**
       * It does process the chemical exchange field
       * Input:   a spatial field to process: w2process
       *          a spatial level set from wich the field is processed
       * Output:  the width of the field, the positive level set value, the negative level set value
       */
    int compute_field_interface_width(Vec *w2process,Vec *phi2Process,
                                      double alpha,double Xab,
                                      double &width,double &phi_negative,double &phi_positive );


    /**
       * It does process the pressure field
       * Input:   a spatial field to process: w2process
       *          a spatial level set from wich the field is processed
       * Output:  the coarse gradient inside   the domain (negative level set values)
       *          the coarse gradient outside  the domain (positive level set values)
       *          the max value inside  the domain (negative level set values)
       *          the max value outside the domain (positive level set values)
       *          the min value inside the domain  (negative level set values)
       *          the min value outsie the domain  (positive level set values)
       */
    int compute_coarse_gradients(Vec *w2process,Vec *phi2Process,
                                      double width,
                                      double &coarse_gradient_inside_the_domain,
                                      double &coarse_gradient_outside_the_domain,
                                      double &max_phi_negative,
                                      double &max_phi_positive,
                                      double &min_phi_negative,
                                      double &min_phi_positive);



    /**
     * @brief smooth_level_set, given the neighboring nodes of node data structure and a level set,
     * it computes the second order derivatives and then the curvature. Using the curvartue it performs motion by curvature
     * to smooth the curve (Osher And Sethian 1988).
     * @param nodes_neighbors
     * @param phi
     * @return error code if any
     */
    int smooth_level_set(my_p4est_node_neighbors_t *node_neighbors, Vec *phi, int n_local, int n_smoothies, PetscScalar band2computeKappa);


};

#endif // FIELDPROCESSOR_H
