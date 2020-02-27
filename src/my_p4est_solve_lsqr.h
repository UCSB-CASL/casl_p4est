#ifndef MY_P4EST_SOLVE_LSQR_H
#define MY_P4EST_SOLVE_LSQR_H

#ifdef P4_TO_P8
#include "my_p8est_macros.h"
#else
#include "my_p4est_macros.h"
#endif

#include <vector>

#include <src/matrix.h>

using std::vector;

/*!
 * \brief solve_lsqr_system solves (possibly constrained) least square system(s) using
 *        cholesky decomposition AtA*y = At*p, with the possibility to return the
 *        interpolation weights for the interpolated value.
 * \param [in] A              : the matrix describing the linear system(s) to be solved in a least-square sense
 * \param [in] p              : pointer to the (array of) right hand side(s)
 * \param [in] n_vectors      : the number of elements in the p array
 * \param [out] solutions     : pointer to the (array of) double solutions
 *                              (i.e. the interpolated values corresponding to each rhs in p
 *                               --> must be correctly preallocated and have a size of n_vectors)
 * \param [in] nb_x,y,z       : numbers of planes of x, y, and z normals containing neighboring points used to
 *                              build the system. These are upper bounds on the number of linearly independent
 *                              geometric neighbors, which in turn limits the degree of the interpolant to be
 *                              considered.
 * \param [in] order          : the degree of the least-square interpolant to be built (0, 1 or 2), default is 2
 * \param [in] nconstraints   : number of contraints on the derivatives of the interpolant (maybe desired for
 *                              taking into account Neumann Boundary Conditions at interpolation steps), 
 *                              default is 0 (i.e. no constraint)
 * \param [out] interp_coeffs : (optional) pointer to a vector of doubles, which will be such that,
 *                                    solutions[i] = sum over k of (interp_coeffs[k]*p[i][k])
 *                              on output. This is required when storing local least-square interpolators
 *                              (--> maybe desired if repeating a lot of similar lsqr interpolation)
 *                              This is not calculated and disregarded if this argument is NULL on input (default).
 * \param [in] cond_thr       : (optional) threshold value for the condition number of the lower triangular
 *                               matrix L obtained in the choleski factorization to consider the system non-singular
 *                              (default value is 1.0e4)
 * [Comments and revisions by Raphael Egan, Feb 26, 2020 (for the long_overdue_merge).]
 */
void solve_lsqr_system(const matrix_t& A, const vector<double> p[], const size_t& n_vectors, double *solutions,
                       DIM(const size_t& nb_x, const size_t& nb_y, const size_t& nb_z),
                       const unsigned char& order = 2, const unsigned char& nconstraints = 0, std::vector<double>* interp_coeffs = NULL, const double& cond_thr = 1.0e4);
inline double solve_lsqr_system(const matrix_t& A, const vector<double>& p,
                                DIM(const size_t& nb_x, const size_t& nb_y, const size_t& nb_z),
                                const unsigned char& order = 2, const unsigned char& nconstraints = 0, std::vector<double>* interp_coeffs = NULL, const double& cond_thr = 1.0e4)
{
  double solution;
  solve_lsqr_system(A, &p, 1, &solution, DIM(nb_x, nb_y, nb_z), order, nconstraints, interp_coeffs, cond_thr);
  return solution;
}

inline double solve_lsqr_system_and_get_coefficients(const matrix_t& A, const vector<double>& p,
                                                     DIM(const size_t &nb_x, const size_t &nb_y, const size_t &nb_z),
                                                     std::vector<double>& interp_coeffs, const double& cond_thr, const unsigned char& order = 2)
{
  return solve_lsqr_system(A, p, DIM(nb_x, nb_y, nb_z), order, 0, &interp_coeffs, cond_thr);
}

#endif /* MY_P4EST_SOLVE_LSQR_H */
