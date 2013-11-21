#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <vector>
#include <cstddef>

class DenseMatrix
{
  size_t m_M, m_N;
  std::vector<double> values;
  double    at, bt, ct;
  double    maxarg1, maxarg2;

public:

  DenseMatrix( size_t M = 1, size_t N = 1 )
  {
    values.resize(M*N);
    m_M = M; m_N = N;
  }

  void resize(size_t M, size_t N);

  void operator=( const DenseMatrix& M );

  inline size_t num_Rows() const {return m_M;}

  inline size_t num_Cols() const {return m_N;}

  inline double get_Value(size_t i, size_t j) const
  {
    return values[i*m_N+j];
  }

  void set_Value(size_t i,size_t j,double value2Set);

  bool is_Symmetric() const;


  //--------------------------------------------------------------------------
  // b = Ax
  //--------------------------------------------------------------------------
  void matVec( const std::vector<double>& X, std::vector<double>& B );

  void tranpose_MatVec( const std::vector<double>& X, std::vector<double>& B );

  void matrix_Product(  DenseMatrix& B, DenseMatrix& C );

  void scale_by_MaxAbs(std::vector<double>& X);

  void MtM_Product(DenseMatrix& M );


  /*!
     * \brief compute the transpose of the matrix
     * \return the transpose of the matrix
     */
  DenseMatrix tr();

  void print();
  void sub( size_t im, size_t jm, size_t iM, size_t jM, DenseMatrix& M );
  void truncate_Matrix( size_t M, size_t N, DenseMatrix& Mat);
  void create_As_Transpose(  DenseMatrix& M );

  void operator+=(DenseMatrix& V );
  void operator-=(DenseMatrix& V );
  void operator*=(double s );
  void operator/=(double s );
  void operator =(double s );
};

#endif // DENSE_MATRIX_H

