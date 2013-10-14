#ifndef PETSC_COMPATIBILITY_H
#define PETSC_COMPATIBILITY_H

#ifndef CHKERRXX
#define CHKERRXX(ierr) CHKERRABORT(PETSC_COMM_WORLD, ierr)
#endif

#if PETSC_VERSION_LT(3,2,0)
#define MatDestroy(a)                    MatDestroy(a)
#define VecDestroy(a)                    VecDestroy(a)
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

#endif // PETSC_COMPATIBILITY_H
