#ifdef P4_TO_P8
#include "my_p8est_solve_lsqr.h"
#include <p8est_connectivity.h>
#else
#include "my_p4est_solve_lsqr.h"
#include <p4est_connectivity.h>
#endif
//#include <lapacke.h>
//#include <cblas.h>
/* [Raphael:] I tried to optimize some of the routines here below with lapacke
 * and cblas functions, but it turns out that they are called only for matrices
 * that are too small to see any improvement. Therefore, I finally decided to
 * leave them as such, since it increases the general portability of the code...
 * --> see commits prior to "long_overdue_merge" for implementation details if interested
 * */
#include "math.h"

#include <iostream>
using namespace std;

size_t tri_idx(size_t i, size_t j)
{
  P4EST_ASSERT(j <= i);
  return i*(i + 1)/2 + j;
}

bool solve_cholesky(const matrix_t &A, const vector<double> b[], vector<double> x[], const size_t& n_vectors, const double& cond_thr, vector<double>* first_line, matrix_t* A_inv)
{
#ifdef CASL_THROWS
  if ((b != NULL && n_vectors == 0) || (b == NULL && n_vectors != 0))
    throw std::invalid_argument("my_p4est_solve_lsqr::solve_cholesky: the number of rhs's must be strictly positive if rhs's are provided or 0 otherwise!");
  if (A.num_cols() != A.num_rows() || !A.is_symmetric())
    throw std::invalid_argument("my_p4est_solve_lsqr::solve_cholesky: solve_cholesky: invalid input matrix A: the matrix must be symmetric)");
  if (b != NULL)
    for (size_t k = 0; k < n_vectors; ++k)
      if (b[k].size() != (size_t) A.num_rows())
        throw std::invalid_argument("my_p4est_solve_lsqr::solve_cholesky: solve_cholesky: invalid input rhs: all rhs vectors must be of the same size (and of the number of rows in A)");
#endif

  // initialize procedure
  size_t n = A.num_rows();
  for (size_t k = 0; k < n_vectors; ++k)
    x[k].resize(n);
  if(first_line != NULL)
    first_line->resize(n, 0.0);
  if(A_inv != NULL)
    A_inv->resize(n, n);
  std::vector<double> Lf(n*(n + 1)/2); // store only the nonzero terms the (i, j) element (j <= i) is Lf[tri_idx(i, j)]
  double max_diag_Lf = 0.0;
  double min_diag_Lf = DBL_MAX;

  // compute cholesky decomposition
  vector<double>* Linv = NULL;
  if(first_line != NULL || A_inv != NULL)
    Linv = new vector<double>(n*(n + 1)/2, 0.0); // inverse of L --> the trick is: make Linv the identity, solve for Linv as right hand-side and store the solution in the same object (there is no conflict)
  for(size_t j = 0; j < n; ++j)
  {
    size_t tri_jj = tri_idx(j, j);
    if(Linv != NULL)
      Linv->at(tri_jj) = 1.0; // make Linv the identity matrix
    if(std::isnan(A.get_value(j,j)))
    {
      if(Linv != NULL)
        delete Linv;
      return false;
    }

    Lf[tri_jj] = A.get_value(j,j);
    for(size_t k = 0; k < j; ++k)
      Lf[tri_jj] -= SQR(Lf[tri_idx(j, k)]);

    if(Lf[tri_jj] < 0.0) // A was probably not SPD...
    {
      if(Linv != NULL)
        delete Linv;
      return false;
    }

    Lf[tri_jj] = sqrt(Lf[tri_jj]);
    max_diag_Lf = MAX(max_diag_Lf, Lf[tri_jj]);
    min_diag_Lf = MIN(min_diag_Lf, Lf[tri_jj]);
    if(Lf[tri_jj] < EPS || std::isnan(Lf[tri_jj]) || std::isinf(Lf[tri_jj]) || max_diag_Lf/min_diag_Lf > cond_thr)
    {
      if(Linv != NULL)
        delete Linv;
      return false;
    }

    for(size_t i = j + 1; i < n; ++i)
    {
      size_t tri_ij = tri_idx(i, j);
      Lf[tri_ij] = A.get_value(i,j);
      for(size_t k = 0; k < j; ++k)
        Lf[tri_ij] -= Lf[tri_idx(i, k)]*Lf[tri_idx(j, k)];
      Lf[tri_ij] /= Lf[tri_jj];
    }
  }

  /* forward solve L*Y=B' */
  std::vector<double *> Y(n);

  for(size_t i = 0; i < n; ++i)
  {
    Y[i] = new double[n_vectors]; // allowed even if n_vectors == 0
    size_t tri_ii = tri_idx(i, i);
    for (size_t k = 0; k < n_vectors; ++k)
      Y[i][k] = b[k][i];
    for(size_t k = 0; k < MAX(n_vectors, (Linv != NULL ? i : 0)); ++k)
      for(size_t j = 0; j < i; ++j)
      {
        size_t tri_ij = tri_idx(i, j);
        if(k < n_vectors)
          Y[i][k] -= Y[j][k]*Lf[tri_ij];
        if(Linv != NULL && k <= j)
          Linv->at(tri_idx(i, k)) -= Linv->at(tri_idx(j, k))*Lf[tri_ij];
      }
    for (size_t k = 0; k < MAX(n_vectors, (Linv != NULL ? i + 1 : 0)); ++k)
    {
      if(k < n_vectors)
        Y[i][k] /= Lf[tri_ii];
      if(Linv != NULL && k  <= i)
        Linv->at(tri_idx(i, k)) /= Lf[tri_ii];
    }
  }

  /* backward solve for Lt*X' = Y */
  double sum = 0.0;
  for(int i = n - 1; i >= 0; --i) // "int" i and not "size_t" i here --> very important (infinite loop otherwise, since size_t is !always! >= 0)
  {
    size_t tri_ii = tri_idx(i, i);
    for (size_t k = 0; k < n_vectors; ++k)
      x[k][i] = Y[i][k];
    if(first_line != NULL)
      first_line->at(i) = Linv->at(tri_idx(i, 0))*Linv->at(tri_idx(i, i)); // initialize
    if(A_inv != NULL)
    {
      // set the diagonal term
      sum = 0.0;
      for (size_t k = i; k < n; ++k)
        sum += SQR(Linv->at(tri_idx(k, i)));
      A_inv->set_value(i, i, sum);
    }
    for(size_t j = i + 1; j < n; ++j)
    {
      if(A_inv != NULL)
        sum = 0.0;
      for (size_t k = 0; k < MAX(n_vectors, (A_inv != NULL ? n - j : 0)); ++k)
      {
        if(k < n_vectors)
          x[k][i] -= x[k][j]*Lf[tri_idx(j, i)];
        if(A_inv != NULL && j + k < n)
          sum += Linv->at(tri_idx(j + k, i))*Linv->at(tri_idx(j + k, j));
      }
      if(A_inv != NULL){
        A_inv->set_value(i, j, sum);
        A_inv->set_value(j, i, sum);
      }
      if(first_line != NULL)
        first_line->at(i) += Linv->at(tri_idx(j, 0))*Linv->at(tri_idx(j, i));
    }
    for (size_t k = 0; k < n_vectors; ++k)
      x[k][i] /= Lf[tri_ii];

    delete[] Y[i];
  }

  if(Linv != NULL)
    delete Linv;

  return true;
}

bool solve_cholesky(const matrix_t &A, const vector<double> &b, vector<double> &x, const double& cond_thr, vector<double>* first_line = NULL, matrix_t* A_inv = NULL)
{
  return solve_cholesky(A, &b, &x, 1, cond_thr, first_line, A_inv);
}

void solve_lsqr_system(const matrix_t& A, const vector<double> p[], const size_t& n_vectors, double *solutions,
                       DIM(const size_t& nb_x, const size_t& nb_y, const size_t& nb_z),
                       const unsigned char& order, const unsigned char& nconstraints, std::vector<double>* interp_coeffs, const double& cond_thr)
{
#ifdef CASL_THROWS
  if(n_vectors == 0) throw std::invalid_argument("my_p4est_solve_lsqr::solve_lsqr_system(...): the number of rhs's must be strictly positive!");
  for (size_t k = 0; k < n_vectors; ++k)
    if((size_t) A.num_rows() != p[k].size())
      throw std::invalid_argument("my_p4est_solve_lsqr::solve_lsqr_system(...): (all of) the right hand side(s) must have the same size as the number of rows in the matrix!");
#endif
  P4EST_ASSERT(nconstraints <= P4EST_DIM);
  const int m = A.num_rows();
  matrix_t *M = new matrix_t();
  vector<double>* Atp     = new std::vector<double>[n_vectors];
  vector<double>* coeffs  = new std::vector<double>[n_vectors];
  vector<double>* my_interp_coeffs = NULL;
  if(interp_coeffs != NULL)
    my_interp_coeffs = new vector<double>(0);
  if(order >= 2 && m >= (1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2 - nconstraints) && ANDD(nb_x >= 3, nb_y >= 3, nb_z >= 3))
  {
    A.tranpose_matvec(p, Atp, n_vectors);
    A.mtm_product(*M);

    if(solve_cholesky(*M, Atp, coeffs, n_vectors, cond_thr, my_interp_coeffs))
    {
      for (unsigned int k = 0; k < n_vectors; ++k)
        solutions[k] = coeffs[k][0];
      delete M;
      if(interp_coeffs)
        A.matvec(*my_interp_coeffs, *interp_coeffs);
      if(my_interp_coeffs != NULL)
        delete  my_interp_coeffs;
      delete [] Atp;
      delete [] coeffs;
      return;
    }
  }

  /* either the system was not invertible - most likely there was a direction with less than 3 points, e.g. in the diagonal !
   * or the number of points along cartesian dimensions is lower than expected, or desired order is smaller than 2 */
  if(my_interp_coeffs != NULL)
    my_interp_coeffs->resize(0);
  if(order >= 1 && m >= (1 + P4EST_DIM - nconstraints) && ANDD(nb_x >= 2, nb_y >= 2, nb_z >= 2))
  {
    const matrix_t* const_trunc_mat = NULL;
    matrix_t *trunc_mat = NULL;
    if(order == 2)
    {
      if(m >= (1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2 - nconstraints) && ANDD(nb_x >= 3, nb_y >= 3, nb_z >= 3))
      {
        if(my_interp_coeffs != NULL)
        {
          trunc_mat = new matrix_t(m, P4EST_DIM + 1 - nconstraints);
          trunc_mat->truncate_matrix(m, P4EST_DIM + 1 - nconstraints, A);
        }
        // the relevant quantities were already calculated but the cholesky_solve failed...
        matrix_t* M_sub= new matrix_t(P4EST_DIM + 1 - nconstraints, P4EST_DIM + 1 - nconstraints);
        M_sub->truncate_matrix(P4EST_DIM + 1 - nconstraints, P4EST_DIM + 1 - nconstraints, *M);
        delete M;
        M = M_sub;
        for (size_t k = 0; k < n_vectors; ++k)
          Atp[k].resize(P4EST_DIM + 1 - nconstraints);
      }
      else
      {
        trunc_mat = new matrix_t(m, P4EST_DIM + 1 - nconstraints);
        trunc_mat->truncate_matrix(m, P4EST_DIM + 1 - nconstraints, A);
        trunc_mat->tranpose_matvec(p, Atp, n_vectors);
        trunc_mat->mtm_product(*M);
      }
      const_trunc_mat = trunc_mat; // will no longer change from here
    }
    else
    {
      const_trunc_mat = &A;
      A.tranpose_matvec(p, Atp, n_vectors);
      A.mtm_product(*M);
    }
    if(solve_cholesky(*M, Atp, coeffs, n_vectors, cond_thr, my_interp_coeffs))
    {
      for (size_t k = 0; k < n_vectors; ++k)
        solutions[k] = coeffs[k][0];
      delete M;
      if(interp_coeffs)
        const_trunc_mat->matvec(*my_interp_coeffs, *interp_coeffs);
      if(my_interp_coeffs != NULL)
        delete my_interp_coeffs;
      if(const_trunc_mat != NULL && const_trunc_mat != &A)
        delete const_trunc_mat;
      delete [] Atp;
      delete [] coeffs;
      return;
    }
    if(const_trunc_mat != NULL && const_trunc_mat != &A)
      delete const_trunc_mat;
  }
  if(my_interp_coeffs != NULL)
    delete my_interp_coeffs;

  /* either the system was not invertible - most likely there was a direction with less than 2 points, e.g. in the diagonal !
   * or the number of points along cartesian dimensions is lower than expected, or desired order is smaller than 1 */
  /* 0-th order polynomial approximation, just compute coeff(0) */
  double denominator = 0.0;
  double numerator = 0.0;
  if(interp_coeffs != NULL)
    interp_coeffs->resize(m);
  for(int i = 0; i < m; ++i)
  {
    denominator += SQR(A.get_value(i,0));
    numerator   += A.get_value(i,0)*p[0][i];
    if(interp_coeffs != NULL)
      interp_coeffs->at(i) = A.get_value(i,0);
  }
  solutions[0] = numerator/denominator;
  if(interp_coeffs != NULL)
    for(int i = 0; i < m; ++i)
      interp_coeffs->at(i) /= denominator;
  for (size_t k = 1; k < n_vectors; ++k) {
    numerator = 0;
    for(int i = 0; i < m; ++i)
      numerator += A.get_value(i,0)*p[k][i];
    solutions[k] = numerator/denominator;
  }
  delete M;
  delete [] Atp;
  delete [] coeffs;
  return;
}

