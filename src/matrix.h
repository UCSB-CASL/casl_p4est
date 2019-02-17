#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

#include <src/casl_math.h>

using std::vector;

class matrix_t
{
protected:
  int m, n;
  vector<double> values;

public:

  matrix_t( int m=1, int n=1 )
  {
    this->m=m; this->n=n;
    values.resize(m*n);
  }

  void resize(int m, int n);

  void operator=( const matrix_t& M );

  inline int num_rows() const { return m; }

  inline int num_cols() const { return n; }

  inline double get_value(int i,int j) const
  {
#ifdef CASL_THROWS
    if(i<0 || i>m || j<0 || j>n)
      throw std::invalid_argument("[CASL_ERROR]: matrix_t->get_value: invalid indices.");
#endif
    return values[i*n+j];
  }

  void set_value(int i, int j, double val);

  bool is_symmetric() const;


  //--------------------------------------------------------------------------
  // b = Ax
  //--------------------------------------------------------------------------
  void matvec( const vector<double>& x, vector<double>& b );

  void tranpose_matvec( const vector<double>& x, vector<double>& b ) {tranpose_matvec(&x, &b, 1);}
  void tranpose_matvec( const vector<double> x[], vector<double> b[], unsigned int n_vectors);

  void matrix_product(  matrix_t& b, matrix_t& c );

  double scale_by_maxabs(vector<double>& x) {return scale_by_maxabs(&x, 1); }
  double scale_by_maxabs(vector<double> x[], unsigned int n_vectors);

  void mtm_product(matrix_t& m );


  /*!
     * \brief compute the transpose of the matrix
     * \return the transpose of the matrix
     */
  matrix_t tr();

  void print();
  void sub( int im, int jm, int iM, int jM, matrix_t& M );
  void truncate_matrix( int m, int n, matrix_t& M);
  void create_as_transpose(  matrix_t& M );

  void operator+=(matrix_t& M );
  void operator-=(matrix_t& M );
  void operator*=(double s );
  void operator/=(double s );
  void operator =(double s );
//  const double* read_data() const {return values.data();}
//  double* get_data() {return values.data();}
};



#endif /* MATRIX_H */

