#include "Cholesky.h"
#include <src/CASL_math.h>

bool Cholesky::compute_Cholesky_Decomposition(const MatrixFull &A )
{
  Lf.resize(A.num_Rows(),A.num_Cols());

  for( size_t j=0; j<A.num_Rows(); ++j )
  {
    double ljj = A.get_Value(j,j);
    if (ISNAN(ljj)){
      return false;
    }

    for( size_t k=0; k<j; ++k)
      ljj -= SQR(Lf.get_Value(j,k));

    if (true)//ljj>1E-15) // ?!
    {
      ljj = sqrt(ljj);

      if(ljj<EPS || ISNAN(ljj) || ISINF(ljj))
        return false;

      Lf.set_Value(j,j,ljj);

      for(size_t i=j+1; i<A.num_Rows(); ++i)
      {
        double lij = A.get_Value(i,j);
        for (size_t k=0; k<j; ++k)
          lij -= Lf.get_Value(i,k)*Lf.get_Value(j,k);
        Lf.set_Value(i,j,lij/ljj);
      }
    }
    else
    {
      Lf.set_Value(j,j,1.0);
      for(size_t i=j+1; i<A.num_Rows(); ++i)
      {
        Lf.set_Value(i,j,0.);
      }
    }
  }
  return true;
}



bool Cholesky::solve(  MatrixFull& A, const std::vector<double>& b, std::vector<double>& x )
{
#ifdef CASL_THROWS
  if(A.num_Cols()!=A.num_Rows() || A.num_Rows() != b.size()) throw std::invalid_argument("[CASL_ERROR]: Cholesky->solve_MatrixFull: A and b have different sizes !");
  if(!A.is_Symmetric()) throw std::invalid_argument("[CASL_ERROR]: Cholesky->solve_MatrixFull: the matrix is not symmetric");
#endif

  if(!compute_Cholesky_Decomposition(A))
    return false;

  size_t n = Lf.num_Rows();

  std::vector<double> y(n);
  x.resize(n);

  // Forward solve L*y=bs
  for(size_t i=0; i<n; ++i)
  {
    y[i] = b[i];
    for (size_t j=0; j<i; ++j)
      y[i] -= y[j]*Lf.get_Value(i,j);
    y[i] /= Lf.get_Value(i,i);
  }

  // Backward solve for Lt*x=y
  for(int i=n-1; i>=0; --i)
  {
    x[i] = y[i];
    for (size_t j=i+1; j<n; ++j)
      x[i] -= x[j]*Lf.get_Value(j,i);
    x[i] /= Lf.get_Value(i,i);
  }
  return true;
}
