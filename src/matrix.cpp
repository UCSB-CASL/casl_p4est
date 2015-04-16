#include "matrix.h"
#include <stdio.h>

void matrix_t::resize(int m, int n)
{
  this->m = m;
  this->n = n;
  values.resize(m*n);
}


void matrix_t::operator=( const matrix_t& M )
{
  values = M.values;
  m = M.m;
  n = M.n;
}


void matrix_t::set_value(int i, int j, double val)
{
  if (i<m && j<n) values[i*n+j]=val;
  else
    if(i>=m && j<n){
      values.resize((i+1)*n);
      m=i+1;
      values[i*n+j] = val;
    }
#ifdef CASL_THROWS
    else
      throw std::invalid_argument("[CASL_ERROR]: matrix_t->set_value: j is larger than the number of columns");
#endif
}

bool matrix_t::is_symmetric() const
{
#ifdef CASL_THROWS
  if(m!=n) throw std::runtime_error("[CASL_ERROR]: matrix_t->is_symmetric: the matrix is not square");
#endif
  for( int i=1; i<m; i++ )
    for( int j=0; j<i; j++ )
      if(fabs(values[i*n+j] - values[j*n+i]) > EPS)
        return false;
  return true;
}


void matrix_t::matvec( const vector<double>& x, vector<double>& b )
{
#ifdef CASL_THROWS
  if(n != (int) x.size()) throw std::invalid_argument("[CASL_ERROR]: matrix_t->matvec: the matrix and X must have compatible sizes");
#endif
  b.resize(m);
  for( int i=0; i<m; i++ )
  {
    b[i]=0;
    for( int j=0; j<n; j++ )
      b[i] += values[i*n+j]*x[j];
  }
}

void matrix_t::tranpose_matvec( const vector<double>& x, vector<double>& b )
{
#ifdef CASL_THROWS
  if(m != (int) x.size()) throw std::invalid_argument("[CASL_ERROR]: matrix_t->transpose_matvec: the matrix and X must have compatible sizes");
#endif
  b.resize(n);
  for( int i=0; i<n; i++ )
  {
    b[i] = 0;
    for( int j=0; j<m; j++ )
      b[i] += values[j*n+i]*x[j];
  }
}

void matrix_t::matrix_product(  matrix_t& b, matrix_t& c )
{
#ifdef CASL_THROWS
  if( n != b.m ) throw std::invalid_argument("[CASL_ERROR]: matrix_t->matrix_product: the matrix sizes don't match");
#endif
  c.resize(m,b.n);
  for( int i=0; i<m; i++ )
    for( int k=0; k<b.n; k++ )
    {
      double sum=0;
      for( int j=0; j<n; j++ )
        sum += values[i*n+j]*b.values[j*n+k];
      c.values[i*n+k] = sum;
    }
}

void matrix_t::mtm_product(matrix_t& M )
{
  M.resize(num_cols(),num_cols());
  for( int i=0; i<n; i++ )
    for( int j=i; j<n; j++ )
    {
      double sum=0;
      for( int k=0; k<m; k++ )
        sum += values[k*n + i]*values[k*n + j];
      M.values[i*n+j] = sum;
      if(i!=j)
        M.values[j*n+i] = sum;
    }
}

void matrix_t::scale_by_maxabs(vector<double>& x)
{
#ifdef CASL_THROWS
  if( m != (int) x.size() ) throw std::invalid_argument("[CASL_ERROR]: matrix_t->scale_by_maxabs: the matrix and the right hand side don't have the same size");
#endif
  double abs_max = EPS;
  for( unsigned int i=0; i< x.size(); i++ )
    for( int j=0; j<n; j++ )
      abs_max = MAX(abs_max, fabs(values[i*n + j]));

  for( unsigned int i=0; i<x.size(); i++ ){
    x[i] /= abs_max;
    for( int j=0; j<n; j++ )
      values[i*n + j] /= abs_max;
  }
}

matrix_t matrix_t::tr()
{
  matrix_t out(n,m);
  for(int i=0;i<m;i++)
    for(int j=0;j<n;j++)
      out.values[j*n+i] = values[i*n+j];
  return out;
}

void matrix_t::print()
{
  for( int i=0; i<m; i++ )
  {
    printf("\n");
    for( int j=0; j<n; j++ )
      printf( "% e ", values[i*n+j] );
  }
  printf("\n");
}

void matrix_t::sub( int im, int jm, int iM, int jM, matrix_t& M )
{
#ifdef CASL_THROWS
  if( im>iM || im<0 || iM>=m || jm>jM || jm<0 || jM>=n ) throw std::invalid_argument("[CASL_ERROR]: matrix_t->sub: invalid subsize");
#endif
  M.resize(iM-im+1,jM-jm+1);
  for( int i=im; i<=iM; i++ )
    for( int j=jm; j<=jM; j++ )
      M.values[(i-im)*n+(j-jm)] = values[i*n+j];
}

void matrix_t::truncate_matrix( int M, int N,  matrix_t& Mat )
{
#ifdef CASL_THROWS
  if(M>Mat.m || N>Mat.n) throw std::invalid_argument("[CASL_ERRO]: matrix_t->truncate_matrix: invalid truncation size");
#endif
  resize(M,N);
  for( int i=0; i<M; i++ )
    for( int j=0; j<N; j++ )
      values[i*n+j] = Mat.values[i*n+j];
}



void matrix_t::create_as_transpose( matrix_t& M )
{
  resize(M.num_cols(),M.num_rows());
  for( int i=0; i<M.num_rows(); i++ )
    for( int j=0; j<M.num_cols(); j++ )
      values[j*n+i] = M.values[i*n+j];
}


void matrix_t::operator+=(matrix_t& M )
{
#ifdef CASL_THROWS
  if(m!=M.m || n!=M.n) throw std::invalid_argument("[CASL_ERRO]: matrix_t->+=: the matrices have different sizes");
#endif
  for(unsigned int i=0; i<values.size(); i++)
    values[i] += M.values[i];
}

void matrix_t::operator-=(matrix_t& M )
{
#ifdef CASL_THROWS
  if(m!=M.m || n!=M.n) throw std::invalid_argument("[CASL_ERRO]: matrix_t->-=: the matrices have different sizes");
#endif
  for(unsigned int i=0; i<values.size(); i++)
    values[i] -= M.values[i];
}

void matrix_t::operator*=( double s )
{
  for(unsigned int i=0; i<values.size(); i++)
    values[i] *= s;
}

void matrix_t::operator/=( double s )
{
  for(unsigned int i=0; i<values.size(); i++)
    values[i] /= s;
}

void matrix_t::operator =( double s )
{
  for(unsigned int i=0; i<values.size(); i++)
    values[i] = s;
}
