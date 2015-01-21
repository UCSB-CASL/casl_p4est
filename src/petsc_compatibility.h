#ifndef PETSC_COMPATIBILITY_H
#define PETSC_COMPATIBILITY_H

#include <petsc.h>

#ifndef CHKERRXX
#define CHKERRXX(ierr) CHKERRABORT(PETSC_COMM_WORLD, ierr)
#endif

#ifndef PETSC_VERSION_GT
#define PETSC_VERSION_GT(MAJOR,MINOR,SUBMINOR) \
  (!PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR))
#endif

// Changes in the destructors
#if PETSC_VERSION_LT(3,2,0)
#define MatDestroy(a)                    MatDestroy(a)
#define VecDestroy(a)                    VecDestroy(a)
#define VecScatterDestroy(a)             VecScatterDestroy(a)
#define AODestroy(a)                     AODestroy(a)
#define ISDestroy(a)                     ISDestroy(a)
#define KSPDestroy(a)                    KSPDestroy(a)
#define PCDestroy(a)                     PCDestroy(a)
#define SNESDestroy(a)                   SNESDestroy(a)
#define TSDestroy(a)                     TSDestroy(a)
#define PetscViewerDestroy(a)            PetscViewerDestroy(a)
#define MatNullSpaceDestroy(a)           MatNullSpaceDestroy(a)
#define ISLocalToGlobalMappingDestroy(a) ISLocalToGlobalMappingDestroy(a)
#else
#define MatDestroy(a)                    MatDestroy(&a)
#define VecDestroy(a)                    VecDestroy(&a)
#define VecScatterDestroy(a)             VecScatterDestroy(&a)
#define AODestroy(a)                     AODestroy(&a)
#define ISDestroy(a)                     ISDestroy(&a)
#define KSPDestroy(a)                    KSPDestroy(&a)
#define PCDestroy(a)                     PCDestroy(&a)
#define SNESDestroy(a)                   SNESDestroy(&a)
#define TSDestroy(a)                     TSDestroy(&a)
#define PetscViewerDestroy(a)            PetscViewerDestroy(&a)
#define MatNullSpaceDestroy(a)           MatNullSpaceDestroy(&a)
#define ISLocalToGlobalMappingDestroy(a) ISLocalToGlobalMappingDestroy(&a)
#endif

// Changes in number of arguments
#if PETSC_VERSION_GT(3,5,0)
#define KSPSetOperators(ksp, Amat, Pmat, MatStructure) KSPSetOperators(ksp, Amat, Pmat)
#endif

#if PETSC_VERSION_LE(3,5,0)
#define PetscSynchronizedFlush(comm, fd) PetscSynchronizedFlush(comm)
#endif

#if PETSC_VERSION_GT(3,4,3)
#define MatNullSpaceRemove(a, b, c) MatNullSpaceRemove(a, b)
#endif

#endif // PETSC_COMPATIBILITY_H
