#ifdef P4_TO_P8
#include "my_p8est_solve_lsqr.h"
#else
#include "my_p4est_solve_lsqr.h"
#endif

#include "math.h"

#include <iostream>
using namespace std;

bool solve_cholesky(matrix_t &A, vector<double> &b, vector<double> &x)
{
#ifdef CASL_THROWS
  if(A.num_cols()!=A.num_rows() || A.num_rows()!=(int)b.size() || !A.is_symmetric())
    throw std::invalid_argument("[CASL_ERROR]: solve_cholesky: wrong input parameters");
#endif

  int n = b.size();
  x.resize(n);

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
  double y[n];

  for(int i=0; i<n; ++i)
  {
    y[i] = b[i];
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

  return true;
}

bool invert_cholesky(matrix_t &A, matrix_t &Ai)
{
#ifdef CASL_THROWS
  if(A.num_cols()!=A.num_rows() || !A.is_symmetric())
    throw std::invalid_argument("[CASL_ERROR]: solve_cholesky: wrong input parameters");
#endif

  int n = A.num_cols();

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

  /* compute inverse of cholesky decomposition by forward substituion*/
  double Li[n][n];
  for(int i=0; i<n; ++i)
  {
    Li[i][i] = 1./Lf[i][i];
    for (int j=0; j<i; ++j)
    {
      double sum = 0;
      for (int k=0; k<i; ++k)
      {
        sum += Li[k][j]*Lf[j][k];
      }
      Li[i][j] = -sum*Li[i][i];
    }
  }

  Ai.resize(n,n);

  /* compute inverse of the original matrix as (L^-1)^T*(L^-1) */
  for (int i=0; i<n; ++i)
  {
    for (int j=i; j<n; ++j)
    {
      double sum=0;
      for (int k=j; k<n; ++k)
      {
        sum += Li[k][i]*Li[k][j];
      }
      Ai.set_value(i,j,sum);
      if (i != j) Ai.set_value(j,i,sum);
    }
  }

  return true;
}




#ifdef P4_TO_P8
double solve_lsqr_system(matrix_t &A, vector<double> &p, int nb_x, int nb_y, int nb_z, char order)
#else
double solve_lsqr_system(matrix_t &A, vector<double> &p, int nb_x, int nb_y, char order)
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
    for(unsigned int i=0; i<p.size(); ++i)
    {
      sum += SQR(A.get_value(i,0));
      rhs += A.get_value(i,0)*p[i];
    }
    return rhs/sum;
  }

  matrix_t M;
  vector<double> Atp;
  vector<double> coeffs;

#ifdef P4_TO_P8
  if(order<2 || p.size()<10 || nb_x<3 || nb_y<3 || nb_z<3)
#else
  if(order<2 || p.size()<6 || nb_x<3 || nb_y<3)
#endif
  {
    if(order==2)
    {
      matrix_t Asub;
#ifdef P4_TO_P8
      Asub.truncate_matrix(A.num_rows(), 4, A);
#else
      Asub.truncate_matrix(A.num_rows(), 3, A);
#endif

      Asub.tranpose_matvec(p, Atp);
      Asub.mtm_product(M);
    }
    else
    {
      A.tranpose_matvec(p, Atp);
      A.mtm_product(M);
    }

    /* the system was not invertible - most likely there was a direction with less than 2 points, e.g. in the diagonal */
    if(!solve_cholesky(M, Atp, coeffs))
    {
      double sum = 0;
      double rhs = 0;
      for(unsigned int i=0; i<p.size(); ++i)
      {
        sum += SQR(A.get_value(i,0));
        rhs += A.get_value(i,0)*p[i];
      }
      return rhs/sum;
    }

    return coeffs[0];
  }

  A.tranpose_matvec(p, Atp);
  A.mtm_product(M);

  /* the system was not invertible - most likely there was a direction with less than 3 points, e.g. in the diagonal ! */
  if(!solve_cholesky(M, Atp, coeffs))
  {
    matrix_t Asub;
#ifdef P4_TO_P8
      Asub.truncate_matrix(A.num_rows(), 4, A);
#else
      Asub.truncate_matrix(A.num_rows(), 3, A);
#endif

    Asub.tranpose_matvec(p, Atp);
    Asub.mtm_product(M);

    solve_cholesky(M, Atp, coeffs);
  }

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
