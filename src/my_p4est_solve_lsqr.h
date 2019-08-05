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
void solve_lsqr_system(matrix_t &A, vector<double> p[], unsigned int n_vectors, double *solutions, int nb_x, int nb_y, int nb_z, char order=2, unsigned short nconstraints=0, std::vector<double>* interp_coeffs = NULL);
inline double solve_lsqr_system(matrix_t &A, vector<double> &p, int nb_x, int nb_y, int nb_z, char order=2, unsigned short nconstraints=0, std::vector<double>* interp_coeffs=NULL)
{
  double solution;
  solve_lsqr_system(A, &p, 1, &solution, nb_x, nb_y, nb_z, order, nconstraints, interp_coeffs);
  return solution;
}
#else
void solve_lsqr_system(matrix_t &A, vector<double> p[], unsigned int n_vectors, double *solutions, int nb_x, int nb_y, char order=2, unsigned short nconstraints=0, std::vector<double>* interp_coeffs = NULL);
inline double solve_lsqr_system(matrix_t &A, vector<double> &p, int nb_x, int nb_y, char order=2, unsigned short nconstraints=0, std::vector<double>* interp_coeffs=NULL)
{
  double solution;
  solve_lsqr_system(A, &p, 1, &solution, nb_x, nb_y, order, nconstraints, interp_coeffs);
  return solution;
}
#endif

#ifdef P4_TO_P8
double solve_lsqr_system_and_get_coefficients(matrix_t &A, vector<double> &p, int nb_x, int nb_y, int nb_z, std::vector<double>& interp_coeffs, char order=2);
#else
double solve_lsqr_system_and_get_coefficients(matrix_t &A, vector<double> &p, int nb_x, int nb_y, std::vector<double>& interp_coeffs, char order=2);
#endif

/*!
 * \brief compute the inverse (At.A)^(-1).At of a generic least square system A using a cholesky decomposition
 * \param A the lsqr matrix
 * \param B the solution
 * \return
 */
bool solve_lsqr_system(matrix_t &A, matrix_t &B);

/*!
 * \brief compute the inverse (At.W.A)^(-1).At.W of a generic weighted least square system W.A using a cholesky decomposition
 * \param A the lsqr matrix
 * \param W the weights
 * \param B the solution
 * \return
 */
bool solve_lsqr_system(matrix_t &A, vector<double> &W, matrix_t &B);

#endif /* MY_P4EST_SOLVE_LSQR_H */
