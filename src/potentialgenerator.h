#ifndef POTENTIALGENERATOR_H
#define POTENTIALGENERATOR_H


#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>


#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_node_base.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_node_base.h>
#endif

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_utils.h>
#endif




#undef MIN
#undef MAX

#include <src/petsc_compatibility.h>
#include <src/Parser.h>
#include <src/CASL_math.h>

using namespace std;

#ifdef P4_TO_P8
class lamellar_level_set:public CF_3
{
    // can be easily extended to multi period later
public:
                        double Lx;
                        double f;
                        lamellar_level_set(double Lx,double f)
                        {
                            this->Lx=Lx;
                            this->f=f;
                        }
                        double operator()(double x, double y,double z) const
                        {
                            // region 1-polymer A
                            if(x<f*this->Lx/4.00)
                            {
                                return this->f-x; //positive distance
                            }
                            //region2-polymer A
                            if(x>this->Lx-this->f*this->Lx/4.00)
                            {
                                return x-this->f*this->Lx/4.00; //positive distance
                            }
                            //region 3-polymer B
                            if(x<this->Lx/2.00+this->f*this->Lx/4.00 && x>this->Lx/2.00-this->f*this->Lx/4.00)
                            {
                                if(x<this->Lx/2.00)
                                    return this->f*this->Lx/4.00-x; // negative distance
                                if(x>this->Lx>=2)
                                    return x-((this->Lx/4.00)*(1-this->f)); //negative distance

                            }
                        }
};



#else
class lamellar_level_set:public CF_2
{
    // can be easily extended to multi period later
public:
                        double Lx;
                        double f;
                        lamellar_level_set(double Lx,double f)
                        {
                            this->Lx=Lx;
                            this->f=f;
                        }
                        double operator()(double x, double y) const
                        {
                            // region 1-polymer A
                            if(x<f*this->Lx/4.00)
                            {
                                return this->f-x; //positive distance
                            }
                            //region2-polymer A
                            if(x>this->Lx-this->f*this->Lx/4.00)
                            {
                                return x-this->f*this->Lx/4.00; //positive distance
                            }
                            //region 3-polymer B
                            if(x<this->Lx/2.00+this->f*this->Lx/4.00 && x>this->Lx/2.00-this->f*this->Lx/4.00)
                            {
                                if(x<this->Lx/2.00)
                                    return this->f*this->Lx/4.00-x; // negative distance
                                if(x>this->Lx>=2)
                                    return x-((this->Lx/4.00)*(1-this->f)); //negative distance

                            }
                        }
};

#endif

class PotentialGenerator
{
public:
    PotentialGenerator();
};

#endif // POTENTIALGENERATOR_H
