#include "solve_lsqr.h"
#include "CASL_math.h"

#include <iostream>
using namespace std;


bool solve_cholesky(double **A, vector<double> &b, vector<double> &x)
{
  int N = b.size();

  /* compute cholesky decomposition */
  double Lf[N][N];
  for(int j=0; j<N; ++j)
  {
    if(isnan(A[j][j])) return false;
    Lf[j][j] = A[j][j];

    for(int k=0; k<j; ++k)
      Lf[j][j] -= SQR(Lf[j][k]);

    Lf[j][j] = sqrt(Lf[j][j]);
    if(Lf[j][j]<EPS || isnan(Lf[j][j]) || isinf(Lf[j][j]))
      return false;

    for(int i=j+1; i<N; ++i)
    {
      Lf[i][j] = A[i][j];
      for(int k=0; k<j; ++k)
        Lf[i][j] -= Lf[i][k]*Lf[j][k];
      Lf[i][j] /= Lf[j][j];
    }
  }

  /* forward solve L*y=b */
  double y[N];

  for(int i=0; i<N; ++i)
  {
    y[i] = b[i];
    for(int j=0; j<i; ++j)
      y[i] -= y[j]*Lf[i][j];
    y[i] /= Lf[i][i];
  }

  /* backward solve for Lt*x=y */
  for(int i=N-1; i>=0; --i)
  {
    x[i] = y[i];
    for(int j=i+1; j<N; ++j)
      x[i] -= x[j]*Lf[j][i];
    x[i] /= Lf[i][i];
  }

  return true;
}




double solve_lsqr_system(double **A, vector<double> &p, int nb_x, int nb_y, char order)
{
  if(order<1 || p.size()<3 || nb_x<2 || nb_y<2)
  {
    /* 0-th order polynomial approximation, just compute coeff(0) */
    double sum = 0;
    double rhs = 0;
    for(unsigned int i=0; i<p.size(); ++i)
    {
      sum += SQR(A[i][0]);
      rhs += A[i][0]*p[i];
    }
    return rhs/sum;
  }

  if(order<2 || p.size()<6 || nb_x<3 || nb_y<3)
  {
    /* truncate the system for 1st order polynomial interpolation */

    /* compute Atp */
    vector<double> Atp(3);
    for(unsigned int n=0; n<3; ++n)
    {
      Atp[n] = 0;
      for(unsigned int m=0; m<p.size(); ++m)
        Atp[n] += A[m][n]*p[m];
    }

    /* compute AtA */
    double **AtA = new double*[3];
    for(int i=0; i<3; ++i) AtA[i] = new double[3];

    for(unsigned int n=0; n<3; ++n)
      for(unsigned int m=n; m<3; ++m)
      {
        AtA[m][n] = 0;
        for(unsigned int k=0; k<p.size(); ++k)
          AtA[m][n] += A[k][m]*A[k][n];
        if(m!=n) AtA[n][m] = AtA[m][n];
      }

    /* now solve system using cholesky decomposition */
    vector<double> coeffs(3);
    if(!solve_cholesky(AtA, Atp, coeffs))
    {
      /* the system was not invertible - most likely there was a direction with less than 2 points, e.g. in the diagonal */
      for(int i=0; i<3; ++i) delete AtA[i];
      delete AtA;

      double sum = 0;
      double rhs = 0;
      for(unsigned int i=0; i<p.size(); ++i)
      {
        sum += SQR(A[i][0]);
        rhs += A[i][0]*p[i];
      }
      return rhs/sum;
    }

    for(int i=0; i<3; ++i) delete AtA[i];
    delete AtA;
    return coeffs[0];
  }




  /* last case, the are enough points for 2nd order polynomial approximation */

  /* compute Atp */
  vector<double> Atp(6);
  for(unsigned int n=0; n<6; ++n)
  {
    Atp[n] = 0;
    for(unsigned int m=0; m<p.size(); ++m)
      Atp[n] += A[m][n]*p[m];
  }

  /* compute AtA */
  double **AtA = new double*[6];
  for(int i=0; i<6; ++i) AtA[i] = new double[6];

  for(unsigned int n=0; n<6; ++n)
    for(unsigned int m=n; m<6; ++m)
    {
      AtA[m][n] = 0;
      for(unsigned int k=0; k<p.size(); ++k)
        AtA[m][n] += A[k][m]*A[k][n];
      if(m!=n) AtA[n][m] = AtA[m][n];
    }

  /* now solve system using cholesky decomposition */
  vector<double> coeffs(6);
  if(!solve_cholesky(AtA, Atp, coeffs))
  {
    /* the system was not invertible - most likely there was a direction with less than 2 points, e.g. in the diagonal */
    for(int i=0; i<6; ++i) delete AtA[i];
    delete AtA;

    /* compute Atp */
    Atp.resize(3);
    for(unsigned int n=0; n<3; ++n)
    {
      Atp[n] = 0;
      for(unsigned int m=0; m<p.size(); ++m)
        Atp[n] += A[m][n]*p[m];
    }

    /* compute AtA */
    AtA = new double*[3];
    for(int i=0; i<3; ++i) AtA[i] = new double[3];

    for(unsigned int n=0; n<3; ++n)
      for(unsigned int m=n; m<3; ++m)
      {
        AtA[m][n] = 0;
        for(unsigned int k=0; k<p.size(); ++k)
          AtA[m][n] += A[k][m]*A[k][n];
        if(m!=n) AtA[n][m] = AtA[m][n];
      }

    /* now solve system using cholesky decomposition */
    coeffs.resize(3);
    solve_cholesky(AtA, Atp, coeffs);

    for(int i=0; i<3; ++i) delete AtA[i];
    delete AtA;
  }
  else
  {
    for(int i=0; i<6; ++i) delete AtA[i];
    delete AtA;
  }

  return coeffs[0];
}
