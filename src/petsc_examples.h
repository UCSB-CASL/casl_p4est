#ifndef PETSC_EXAMPLES_H
#define PETSC_EXAMPLES_H

// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>



#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_utils.h>
#include <src/my_p4est_to_p8est.h>
#include <src/test_brick8.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/test_brick.h>
#endif

#include <src/CASL_math.h>
#include <src/petsc_compatibility.h>

#include <../src/mat/impls/dense/mpi/mpidense.h>
#include <petsc-private/vecimpl.h>

using namespace std;



class petsc_examples
{
public:
    petsc_examples();
    PetscErrorCode petscVecExample(int argc, char* args[]);
    PetscErrorCode LowRankUpdate(Mat,Mat,Vec,Vec,Vec,Vec,PetscInt);
    void petscVecAssemblyExample(int argc, char* args[]);
    int petscVecRestoreExample(int argc, char* args[]);
    int petscVecGhostExample(int argc, char *args[]);

    static PetscScalar func2(PetscScalar a)
    {
      return 2*a/(1+a*a);
    }

};

#endif // PETSC_EXAMPLES_H
