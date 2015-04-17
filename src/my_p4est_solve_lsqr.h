#ifndef MY_P4EST_SOLVE_LSQR_H
#define MY_P4EST_SOLVE_LSQR_H

#include <vector>

#include <src/matrix.h>

using std::vector;

/*!
 * \brief solve a least square system using a cholesky decomposition AtA*y = At*p
 * \param A the lsqr matrix
 * \param p the right hand side
 * \param nb_x the number of points in the x direction
 * \param nb_y the number of points in the y direction
 * \param order the desired order of interpolation, 0 - constant, 1 - linear, 2 - quadratic
 * \return
 */
#ifdef P4_TO_P8
double solve_lsqr_system(matrix_t &A, vector<double> &p, int nb_x, int nb_y, int nb_z, char order=2);
#else
double solve_lsqr_system(matrix_t &A, vector<double> &p, int nb_x, int nb_y, char order=2);
#endif

#endif /* MY_P4EST_SOLVE_LSQR_H */
