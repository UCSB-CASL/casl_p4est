#ifndef MATRIX_FULL
#define MATRIX_FULL

#include <vector>
#include <cstddef>

class MatrixFull
{
  size_t m_M, m_N;
  std::vector<double> values;
  double    at, bt, ct;
  double    maxarg1, maxarg2;

public:

  MatrixFull( size_t M = 1, size_t N = 1 )
  {
    values.resize(M*N);
    m_M = M; m_N = N;
  }

  void resize(size_t M, size_t N);

  void operator=( const MatrixFull& M );

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

  void matrix_Product(  MatrixFull& B, MatrixFull& C );

  void scale_by_MaxAbs(std::vector<double>& X);

  void MtM_Product(MatrixFull& M );


  /*!
     * \brief compute the transpose of the matrix
     * \return the transpose of the matrix
     */
  MatrixFull tr();

  void print();
  void sub( size_t im, size_t jm, size_t iM, size_t jM, MatrixFull& M );
  void truncate_Matrix( size_t M, size_t N, MatrixFull& Mat);
  void create_As_Transpose(  MatrixFull& M );

  void operator+=(MatrixFull& V );
  void operator-=(MatrixFull& V );
  void operator*=(double s );
  void operator/=(double s );
  void operator =(double s );
};

#endif

