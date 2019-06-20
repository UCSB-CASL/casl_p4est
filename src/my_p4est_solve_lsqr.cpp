#ifdef P4_TO_P8
#include "my_p8est_solve_lsqr.h"
#else
#include "my_p4est_solve_lsqr.h"
#endif

#include "math.h"

#include <iostream>
using namespace std;

int tri_idx(int i, int j)
{
  return i*(i+1)/2+j;
}

bool solve_cholesky_and_get_first_line(matrix_t &A, vector<double> &b, vector<double> &x, vector<double> &first_line)
{
#ifdef CASL_THROWS
  if(A.num_cols()!=A.num_rows() || A.num_rows()!=(int)b.size() || !A.is_symmetric())
    throw std::invalid_argument("[CASL_ERROR]: solve_cholesky: wrong input parameters");
#endif

  int n = b.size();
  x.resize(n);
  first_line.resize(n, 0.0);

  /* compute cholesky decomposition */
  double Lf[n][n];
  vector<double> Linv; Linv.resize(n*(n+1)/2, 0.0); // inverse of L
  for(int j=0; j<n; ++j)
  {
    Linv.at(tri_idx(j, j)) = 1.0;
    if(std::isnan(A.get_value(j,j))) return false;

    Lf[j][j] = A.get_value(j,j);
    for(int k=0; k<j; ++k)
      Lf[j][j] -= SQR(Lf[j][k]);

    Lf[j][j] = sqrt(Lf[j][j]);
    if(Lf[j][j]<EPS || std::isnan(Lf[j][j]) || std::isinf(Lf[j][j]))
      return false;

    for(int i=j+1; i<n; ++i)
    {
      Lf[i][j] = A.get_value(i,j);
      for(int k=0; k<j; ++k)
        Lf[i][j] -= Lf[i][k]*Lf[j][k];
      Lf[i][j] /= Lf[j][j];
    }
  }

  /* forward solve L*y=b */
  double y[n];

  for(int i=0; i<n; ++i)
  {
    y[i] = b[i];
    for (int k=0; k<i; ++k)
      for(int j=k; j<i; ++j)
        Linv.at(tri_idx(i, k)) -= Linv.at(tri_idx(j, k))*Lf[i][j];
    for (int k=0; k<=i ; ++k)
      Linv.at(tri_idx(i, k)) /= Lf[i][i];
    for(int j=0; j<i; ++j)
      y[i] -= y[j]*Lf[i][j];
    y[i] /= Lf[i][i];
  }

  /* backward solve for Lt*x=y */
  for(int i=n-1; i>=0; --i)
  {
    x[i] = y[i];
    for(int j=i+1; j<n; ++j)
      x[i] -= x[j]*Lf[j][i];
    x[i] /= Lf[i][i];
  }

  for(int i=0; i < n; i++)
    for (int j = i; j < n; ++j)
      first_line[i] += Linv.at(tri_idx(j, 0))*Linv.at(tri_idx(j, i));
  return true;
}

bool solve_cholesky(matrix_t &A, vector<double> b[], vector<double> x[], unsigned int n_vectors)
{
#ifdef CASL_THROWS
  if(n_vectors == 0)
    throw std::invalid_argument("[CASL_ERROR]: solve_cholesky: the number of rhs's must be strictly positive!");
  for (unsigned int k = 0; k < n_vectors; ++k) {
    if(A.num_cols()!=A.num_rows() || A.num_rows()!=(int)b[k].size() || !A.is_symmetric())
      throw std::invalid_argument("[CASL_ERROR]: solve_cholesky: wrong input parameters");
  }
#endif

  int n = b[0].size();
  for (unsigned int k = 0; k < n_vectors; ++k) {
    x[k].resize(n);
  }

  /* compute cholesky decomposition */
  double Lf[n][n];
  for(int j=0; j<n; ++j)
  {
    if(std::isnan(A.get_value(j,j))) return false;

    Lf[j][j] = A.get_value(j,j);
    for(int k=0; k<j; ++k)
      Lf[j][j] -= SQR(Lf[j][k]);

    Lf[j][j] = sqrt(Lf[j][j]);
    if(Lf[j][j]<EPS || std::isnan(Lf[j][j]) || std::isinf(Lf[j][j]))
      return false;

    for(int i=j+1; i<n; ++i)
    {
      Lf[i][j] = A.get_value(i,j);
      for(int k=0; k<j; ++k)
        Lf[i][j] -= Lf[i][k]*Lf[j][k];
      Lf[i][j] /= Lf[j][j];
    }
  }

  /* forward solve L*y=b */
  double y[n_vectors][n];


  for(int i=0; i<n; ++i)
  {
    for (unsigned int k = 0; k < n_vectors; ++k)
      y[k][i] = b[k][i];
    for(int j=0; j<i; ++j)
      for (unsigned int k = 0; k < n_vectors; ++k)
        y[k][i] -= y[k][j]*Lf[i][j];
    for (unsigned int k = 0; k < n_vectors; ++k)
      y[k][i] /= Lf[i][i];
  }

  /* backward solve for Lt*x=y */
  for(int i=n-1; i>=0; --i)
  {
    for (unsigned int k = 0; k < n_vectors; ++k)
      x[k][i] = y[k][i];
    for(int j=i+1; j<n; ++j)
      for (unsigned int k = 0; k < n_vectors; ++k)
        x[k][i] -= x[k][j]*Lf[j][i];
    for (unsigned int k = 0; k < n_vectors; ++k)
      x[k][i] /= Lf[i][i];
  }

  return true;
}

bool solve_cholesky(matrix_t &A, vector<double> &b, vector<double> &x)
{
  return solve_cholesky(A, &b, &x, 1);
}




#ifdef P4_TO_P8
void solve_lsqr_system(matrix_t &A, vector<double> p[], unsigned int n_vectors, double* solutions, int nb_x, int nb_y, int nb_z, char order)
#else
void solve_lsqr_system(matrix_t &A, vector<double> p[], unsigned int n_vectors, double* solutions, int nb_x, int nb_y, char order)
#endif
{
#ifdef CASL_THROWS
  if(n_vectors == 0) throw std::invalid_argument("[CASL_ERROR]: solve_lsqr_system(...): the number of rhs's must be strictly positive!");
  for (unsigned int k = 0; k < n_vectors; ++k)
    if( (unsigned int) A.num_rows() != p[k].size() )
      throw std::invalid_argument("[CASL_ERROR]: solve_lsqr_system(...): the matrix and (one of) the right hand side(s) don't have the same size");
#endif
  unsigned int m = (unsigned int) A.num_rows();
#ifdef P4_TO_P8
  if(order<1 || m<4 || nb_x<2 || nb_y<2 || nb_z<2)
#else
  if(order<1 || m<3 || nb_x<2 || nb_y<2)
#endif
  {
    /* 0-th order polynomial approximation, just compute coeff(0) */
    for (unsigned int k = 0; k < n_vectors; ++k) {
      double sum = 0;
      double rhs = 0;
      for(unsigned int i=0; i<m; ++i)
      {
        sum += SQR(A.get_value(i,0));
        rhs += A.get_value(i,0)*p[k][i];
      }
      solutions[k] = rhs/sum;
    }
    return;
  }

  matrix_t M;
  vector<double> Atp[n_vectors];
  vector<double> coeffs[n_vectors];

#ifdef P4_TO_P8
  if(order<2 || m<10 || nb_x<3 || nb_y<3 || nb_z<3)
#else
  if(order<2 || m<6 || nb_x<3 || nb_y<3)
#endif
  {
    if(order==2)
    {
      matrix_t Asub;
#ifdef P4_TO_P8
      Asub.truncate_matrix(m, 4, A);
#else
      Asub.truncate_matrix(m, 3, A);
#endif

      Asub.tranpose_matvec(p, Atp, n_vectors);
      Asub.mtm_product(M);
    }
    else
    {
      A.tranpose_matvec(p, Atp, n_vectors);
      A.mtm_product(M);
    }

    /* the system was not invertible - most likely there was a direction with less than 2 points, e.g. in the diagonal */
    if(!solve_cholesky(M, Atp, coeffs, n_vectors))
    {
      for (unsigned int k = 0; k < n_vectors; ++k) {
        double sum = 0;
        double rhs = 0;
        for(unsigned int i=0; i<m; ++i)
        {
          sum += SQR(A.get_value(i,0));
          rhs += A.get_value(i,0)*p[k][i];
        }
        solutions[k] = rhs/sum;
      }
      return;
    }
    for (unsigned int k = 0; k < n_vectors; ++k)
      solutions[k] = coeffs[k][0];
    return;
  }

  A.tranpose_matvec(p, Atp, n_vectors);
  A.mtm_product(M);

  /* the system was not invertible - most likely there was a direction with less than 3 points, e.g. in the diagonal ! */
  if(!solve_cholesky(M, Atp, coeffs, n_vectors))
  {
    matrix_t Asub;
#ifdef P4_TO_P8
    Asub.truncate_matrix(m, 4, A);
#else
    Asub.truncate_matrix(m, 3, A);
#endif

    Asub.tranpose_matvec(p, Atp, n_vectors);
    Asub.mtm_product(M);

    solve_cholesky(M, Atp, coeffs, n_vectors);
  }
  for (unsigned int k = 0; k < n_vectors; ++k)
    solutions[k] = coeffs[k][0];
  return;
}


#ifdef P4_TO_P8
double solve_lsqr_system_and_get_coefficients(matrix_t &A, vector<double> &p, int nb_x, int nb_y, int nb_z, vector<double> &interp_coeffs, char order)
#else
double solve_lsqr_system_and_get_coefficients(matrix_t &A, vector<double> &p, int nb_x, int nb_y, vector<double> &interp_coeffs, char order)
#endif
{
#ifdef P4_TO_P8
  if(order<1 || p.size()<4 || nb_x<2 || nb_y<2 || nb_z<2)
#else
  if(order<1 || p.size()<3 || nb_x<2 || nb_y<2)
#endif
  {
    /* 0-th order polynomial approximation, just compute coeff(0) */
    double sum = 0;
    double rhs = 0;
    interp_coeffs.resize(p.size());
    for(unsigned int i=0; i<p.size(); ++i)
    {
      sum += SQR(A.get_value(i,0));
      rhs += A.get_value(i,0)*p[i];
      interp_coeffs[i] = A.get_value(i,0);
    }
    for (unsigned int i = 0; i < p.size(); ++i)
      interp_coeffs[i] /= sum;
    return rhs/sum;
  }

  matrix_t M;
  vector<double> Atp;
  vector<double> coeffs;
  vector<double> my_interp_coeffs;

#ifdef P4_TO_P8
  if(order<2 || p.size()<10 || nb_x<3 || nb_y<3 || nb_z<3)
#else
  if(order<2 || p.size()<6 || nb_x<3 || nb_y<3)
#endif
  {
    matrix_t *trunc_mat;
    matrix_t Asub;
    if(order==2)
    {
#ifdef P4_TO_P8
      Asub.truncate_matrix(A.num_rows(), 4, A);
#else
      Asub.truncate_matrix(A.num_rows(), 3, A);
#endif

      Asub.tranpose_matvec(p, Atp);
      Asub.mtm_product(M);
      trunc_mat = &Asub;
    }
    else
    {
      A.tranpose_matvec(p, Atp);
      A.mtm_product(M);
      trunc_mat = &A;
    }

    /* the system was not invertible - most likely there was a direction with less than 2 points, e.g. in the diagonal */
    if(!solve_cholesky_and_get_first_line(M, Atp, coeffs, my_interp_coeffs))
    {
      double sum = 0;
      double rhs = 0;
      interp_coeffs.resize(p.size());
      for(unsigned int i=0; i<p.size(); ++i)
      {
        sum += SQR(A.get_value(i,0));
        rhs += A.get_value(i,0)*p[i];
        interp_coeffs[i] = A.get_value(i,0);
      }
      for (unsigned int i = 0; i < interp_coeffs.size(); ++i)
        interp_coeffs[i] /= sum;

      return rhs/sum;
    }
    trunc_mat->matvec(my_interp_coeffs, interp_coeffs);

    return coeffs[0];
  }

  A.tranpose_matvec(p, Atp);
  A.mtm_product(M);

  /* the system was not invertible - most likely there was a direction with less than 3 points, e.g. in the diagonal ! */
  if(!solve_cholesky_and_get_first_line(M, Atp, coeffs, my_interp_coeffs))
  {
    matrix_t Asub;
#ifdef P4_TO_P8
      Asub.truncate_matrix(A.num_rows(), 4, A);
#else
      Asub.truncate_matrix(A.num_rows(), 3, A);
#endif

    Asub.tranpose_matvec(p, Atp);
    Asub.mtm_product(M);

    solve_cholesky_and_get_first_line(M, Atp, coeffs, my_interp_coeffs);
    Asub.matvec(my_interp_coeffs, interp_coeffs);
  }
  else
    A.matvec(my_interp_coeffs, interp_coeffs);

  return coeffs[0];
}

bool solve_lsqr_system(matrix_t &A, matrix_t &B)
{
  matrix_t AtA;
  A.mtm_product(AtA);
  bool result = invert_cholesky(AtA, AtA);
  B = A;
  B.tr();
}

bool solve_lsqr_system(matrix_t &A, vector<double> &W, vector<double> &p, vector<double> &y)
{
  matrix_t AtWA;
  vector<double> Wp = p;
  vector<double> AtWp;

  for (int i=0; i<Wp.size(); ++i) Wp[i] *= W[i];

  A.tranpose_matvec(Wp, AtWp);
  A.mtm_product(AtWA, W);
  return solve_cholesky(AtWA, AtWp, y);
}
