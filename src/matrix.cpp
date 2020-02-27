#include "matrix.h"
#include <stdio.h>

//#include <cblas.h>
/* [Raphael:] I tried to optimize these routines with cblas functions,
 * but it turns out that they are called only for matrices that are too small
 * to see any improvement. Therefore, I decided to leave them as such, as it
 * increases the general portability of the code...
 * */

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


void matrix_t::matvec(const vector<double>& x, vector<double>& b) const
{
#ifdef CASL_THROWS
  if(n != (int) x.size()) throw std::invalid_argument("[CASL_ERROR]: matrix_t->matvec: the matrix and X must have compatible sizes");
#endif
  b.resize(m);
//  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, values.data(), n, x.data(), 1, 0.0, b.data(), 1);
  for(int i = 0; i < m; i++)
  {
    b[i] = 0.0;
    for(int j = 0; j < n; j++)
      b[i] += values[i*n + j]*x[j];
  }
}

void matrix_t::tranpose_matvec(const vector<double> x[], vector<double> b[], unsigned int n_vectors) const
{
#ifdef CASL_THROWS
  if(n_vectors == 0) throw std::invalid_argument("[CASL_ERROR]: matrix_t->transpose_matvec: the number of rhs's must be strictly positive!");
  for (unsigned int k = 0; k < n_vectors; ++k)
    if(m != (int) x[k].size())
      throw std::invalid_argument("[CASL_ERROR]: matrix_t->transpose_matvec: the matrix and (all columns of) X must have compatible sizes");
#endif
  for (unsigned int k = 0; k < n_vectors; ++k) {
    b[k].resize(n);
//    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1.0, values.data(), n, x[k].data(), 1, 0.0, b[k].data(), 1);
    for( int i=0; i<n; i++ )
    {
      b[k][i] = 0;
      for( int j=0; j<m; j++ )
        b[k][i] += values[j*n+i]*x[k][j];
    }
  }
}

void matrix_t::matrix_product(  matrix_t& b, matrix_t& c )
{
#ifdef CASL_THROWS
  if( n != b.m ) throw std::invalid_argument("[CASL_ERROR]: matrix_t->matrix_product: the matrix sizes don't match");
#endif
  c.resize(m,b.n);
//  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, b.n, n, 1.0, this->values.data(), n, b.values.data(), b.n, 0.0, c.values.data(), c.n);
  for( int i=0; i<m; i++ )
    for( int k=0; k<b.n; k++ )
    {
      double sum=0;
      for( int j=0; j<n; j++ )
        sum += values[i*n+j]*b.values[j*n+k];
      c.values[i*n+k] = sum;
    }
}

void matrix_t::mtm_product(matrix_t& M ) const
{
  M.resize(num_cols(),num_cols());
//  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, this->n, this->n, this->m, 1.0, this->values.data(), this->n, this->values.data(), this->n, 0.0, M.values.data(), this->n);
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

void matrix_t::mtm_product(matrix_t& M, vector<double>& W) const
{
  M.resize(num_cols(),num_cols());
  for( int i=0; i<n; i++ )
    for( int j=i; j<n; j++ )
    {
      double sum=0;
      for( int k=0; k<m; k++ )
        sum += values[k*n + i]*values[k*n + j]*W[k];
      M.values[i*n+j] = sum;
      if(i!=j)
        M.values[j*n+i] = sum;
    }
}

double matrix_t::scale_by_maxabs(vector<double> x[], unsigned int n_vectors)
{
#ifdef CASL_THROWS
  if(n_vectors == 0) throw std::invalid_argument("[CASL_ERROR]: matrix_t->scale_by_maxabs: the number of rhs's must be strictly positive!");
  for (unsigned int k = 0; k < n_vectors; ++k)
    if( m != (int) x[k].size() )
      throw std::invalid_argument("[CASL_ERROR]: matrix_t->scale_by_maxabs: the matrix and (one of) the right hand side(s) don't have the same size");
#endif
//  double abs_max = MAX(EPS, fabs(values[cblas_idamax(n*m, this->values.data(), 1)]));
  // these following ones introduce a (very!) slight change of behavior
//  cblas_dscal(n*m, (1.0/abs_max), this->values.data(), 1);
//  for (unsigned int k = 0; k < n_vectors; ++k)
//    cblas_dscal(x[k].size(), (1.0/abs_max), x[k].data(), 1);

  double abs_max = EPS;
  for( unsigned int i = 0; i< (unsigned int) m; i++ )
    for( int j = 0; j < n; j++ )
      abs_max = MAX(abs_max, fabs(values[i*n + j]));

  for( unsigned int i = 0; i < (unsigned int) m; i++ ){
    for( unsigned int k = 0; k < MAX(n_vectors, (unsigned int) n); k++ )
    {
      if(k < n_vectors)
        x[k][i] /= abs_max;
      if(k < (unsigned int) n)
        values[i*n + k] /= abs_max;
    }
  }
  return abs_max;
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
      M.values[(i-im)*n+(j-jm)] = get_value(i,j);
}

void matrix_t::truncate_matrix( int M, int N, const matrix_t& Mat)
{
#ifdef CASL_THROWS
  if(M>Mat.m || N>Mat.n) throw std::invalid_argument("[CASL_ERRO]: matrix_t->truncate_matrix: invalid truncation size");
#endif
  resize(M,N);
  for( int i=0; i<M; i++ )
    for( int j=0; j<N; j++ )
      values[i*n+j] = Mat.values[i*Mat.n+j];
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
//  cblas_daxpy(n*m, 1.0, M.values.data(), 1, this->values.data(), 1);
  for(unsigned int i=0; i<values.size(); i++)
    values[i] += M.values[i];
}

void matrix_t::operator-=(matrix_t& M )
{
#ifdef CASL_THROWS
  if(m!=M.m || n!=M.n) throw std::invalid_argument("[CASL_ERRO]: matrix_t->-=: the matrices have different sizes");
#endif
//  cblas_daxpy(n*m, -1.0, M.values.data(), 1, this->values.data(), 1);
  for(unsigned int i=0; i<values.size(); i++)
    values[i] -= M.values[i];
}

void matrix_t::operator*=( double s )
{
//  cblas_dscal(n*m, s, this->values.data(), 1);
  for(unsigned int i=0; i<values.size(); i++)
    values[i] *= s;
}

void matrix_t::operator/=( double s )
{
//  cblas_dscal(n*m, (1.0/s), this->values.data(), 1);
  for(unsigned int i=0; i<values.size(); i++)
    values[i] /= s;
}

void matrix_t::operator =( double s )
{
  for(unsigned int i=0; i<values.size(); i++)
    values[i] = s;
}
