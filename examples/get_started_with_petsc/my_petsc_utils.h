#ifndef MY_PETSC_UTILS_H
#define MY_PETSC_UTILS_H

#include <petsc.h>
#include <one_dimensional_uniform_grid.h>

/*!
 * Exhaustive comments to be found for the functions here below in the cpp file, alongside their definitions
 */

/*!
 * \brief The cont_function class is a virtual class that is meant to represent any user-defined continuous function
 */
class cont_function {
public:
  virtual double operator()(double) const=0;
  virtual ~cont_function() {}
};

PetscErrorCode vec_create_on_one_dimensional_grid(const one_dimensional_uniform_grid& grid, Vec *vv);

PetscErrorCode sample_vector_on_grid(Vec v, const one_dimensional_uniform_grid &grid, cont_function &f);

PetscErrorCode export_in_binary_format(Vec v, const char *filename);

#endif // MY_PETSC_UTILS_H
