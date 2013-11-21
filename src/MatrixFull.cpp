#include "MatrixFull.h"
#include <src/CASL_math.h>
#include <stdexcept>
#include <cstdio>

using std::printf;

void DenseMatrix::resize(size_t M, size_t N)
{
  m_M = M; m_N = N;
  values.resize(M*N);
}

void DenseMatrix::operator=( const DenseMatrix& M )
{
  values = M.values;
  this->m_M = M.m_M;
  this->m_N = M.m_N;
}

void DenseMatrix::set_Value(size_t i,size_t j,double value2Set)
{
  if (i<m_M && j<m_N) values[i*m_N+j]=value2Set;
  else
    if(i>=m_M && j<m_N){
      values.resize((i+1)*m_N);
      m_M = i+1;
      values[i*m_N+j] = value2Set;
    }
#ifdef CASL_THROWS
    else
      throw std::invalid_argument("[ERROR]: MatrixFull->set_Value: j is larger than the number of columns");
#endif
}

bool DenseMatrix::is_Symmetric() const
{
#ifdef CASL_THROWS
  if(m_M != m_N) throw std::runtime_error("[ERROR]: MatrixFull->is_Symmetric: the matrix is not square");
#endif
  for( size_t i=1; i<m_M; i++ )
    for( size_t j=0; j<i; j++ )
      if(ABS(get_Value(i,j) - get_Value(j,i)) > EPS)
        return false;
  return true;
}


void DenseMatrix::matVec( const std::vector<double>& X, std::vector<double>& B )
{
#ifdef CASL_THROWS
  if(m_N != X.size()) throw std::invalid_argument("[ERROR]: MatrixFull->MatVec: the matrix and X must have compatible sizes");
#endif
  B.resize(m_M);
  for( size_t i=0; i<m_M; i++ )
  {
    double b=0;
    for( size_t j=0; j<m_N; j++ )
      b += values[i*m_N+j]*X[j];
    B[i]=b;
  }
}

void DenseMatrix::tranpose_MatVec( const std::vector<double>& X, std::vector<double>& B )
{
#ifdef CASL_THROWS
  if(m_M != X.size()) throw std::invalid_argument("[ERROR]: MatrixFull->transpose_MatVec: the matrix and X must have compatible sizes");
#endif
  B.resize(m_N);
  for( size_t i=0; i<m_N; i++ )
  {
    double b=0;
    for( size_t j=0; j<m_M; j++ )
      b += values[j*m_N+i]*X[j];
    B[i]=b;
  }
}

void DenseMatrix::matrix_Product(  DenseMatrix& B, DenseMatrix& C )
{
#ifdef CASL_THROWS
  if( m_N != B.m_M ) throw std::invalid_argument("[ERROR]: MatrixFull->matrix_Product: the matrix sizes don't match");
#endif
  C.resize(m_M,B.m_N);
#ifdef CASL_OPENMP
#pragma omp parallel for
#endif
  for( size_t i=0; i<  m_M; i++ )
    for( size_t k=0; k<B.m_N; k++ )
    {
      double sum=0;
      for( size_t j=0; j<m_N; j++ )
        sum += values[i*m_N+j]*B.values[j*m_N+k];
      values[i*m_N+k] = sum;
    }
}

void DenseMatrix::MtM_Product(DenseMatrix& M )
{
  M.resize(num_Cols(),num_Cols());
#ifdef CASL_OPENMP
#pragma omp parallel for
#endif
  for( size_t i=0; i<  m_N; i++ )
    for( size_t j=0; j<m_N; j++ )
    {
      double sum=0;
      for( size_t k=0; k<m_M; k++ )
        sum += values[k*m_N + i]*values[k*m_N + j];
      M.set_Value(i,j,sum);
    }
}

void DenseMatrix::scale_by_MaxAbs(std::vector<double>& X)
{
#ifdef CASL_THROWS
  if( m_M != X.size() ) throw std::invalid_argument("[ERROR]: MatrixFull->scale_by_MaxAbs: the matrix and the right hand side don't have the same size");
#endif
  double abs_max=1E-7;
  for( size_t i=0; i<  X.size(); i++ )
    for( size_t j=0; j<m_N; j++ )
      abs_max=MAX(abs_max,ABS(values[i*m_N + j]));

  for( size_t i=0; i<  X.size(); i++ ){
    X[i]/=abs_max;
    for( size_t j=0; j<m_N; j++ )
      values[i*m_N + j]/=abs_max;
  }
}

DenseMatrix DenseMatrix::tr()
{
  DenseMatrix out(m_N,m_M);
  for(size_t i=0;i<m_M;i++)
    for(size_t j=0;j<m_N;j++)
      out.set_Value(j,i,this->get_Value(i,j));
  return out;
}

void DenseMatrix::print()
{
  for( size_t i=0; i<m_M; i++ ){ printf("\n");
    for( size_t j=0; j<m_N; j++ ){
      printf( "% e ", this->get_Value(i,j) ); }} printf("\n\n");
  fflush(stdout);
}

void DenseMatrix::sub( size_t im, size_t jm, size_t iM, size_t jM, DenseMatrix& M )
{
#ifdef CASL_THROWS
  if( im>iM || iM>=m_M || jm>jM || jM>=m_N ) throw std::invalid_argument("[ERROR]: MatrixFull->sub: invalid subsize");
#endif
  M.resize(iM-im+1,jM-jm+1);
  for( size_t i=im; i<=iM; i++ )
    for( size_t j=jm; j<=jM; j++ )
      M.set_Value(i-im, j-jm,this->get_Value(i,j));
}

void DenseMatrix::truncate_Matrix( size_t M, size_t N,  DenseMatrix& Mat )
{
#ifdef CASL_THROWS
  if(M>Mat.m_M || N>Mat.m_N) throw std::invalid_argument("[CASL_ERRO]: MatrixFull->truncate_Matrix: invalid truncation size");
#endif
  resize(M,N);
  for( size_t i=0; i<M; i++ )
    for( size_t j=0; j<N; j++ )
      set_Value(i,j,Mat.get_Value(i, j));
}



void DenseMatrix::create_As_Transpose( DenseMatrix& M )
{
  resize(M.num_Cols(),M.num_Rows());
  for( size_t i=0; i<M.num_Rows(); i++ )
    for( size_t j=0; j<M.num_Cols(); j++ )
      set_Value(j,i,M.get_Value(i,j));
}


void DenseMatrix::operator+=(DenseMatrix& V )
{
#ifdef CASL_THROWS
  if(m_M!=V.m_M || m_N!=V.m_N) throw std::invalid_argument("[CASL_ERRO]: MatrixFull->+=: the matrices have different sizes");
#endif
  for(size_t i=0; i<values.size(); i++)
    this->set_Value(0,i,this->get_Value(0,i)+V.get_Value(0,i));
}

void DenseMatrix::operator-=(DenseMatrix& V )
{
#ifdef CASL_THROWS
  if(m_M!=V.m_M || m_N!=V.m_N) throw std::invalid_argument("[CASL_ERRO]: MatrixFull->-=: the matrices have different sizes");
#endif
  for(size_t i=0; i<values.size(); i++)
  {
    this->set_Value(0,i,this->get_Value(0,i)-V.get_Value(0,i));
  }
}

void DenseMatrix::operator*=(             double s )
{
  for(size_t i=0; i<values.size(); i++)
    values[i] *= s;
}
void DenseMatrix::operator/=(             double s )
{
  for(size_t i=0; i<values.size(); i++)
    values[i] /= s;
}
void DenseMatrix::operator =(             double s )
{
  for(size_t i=0; i<values.size(); i++)
    values[i] = s;
}

