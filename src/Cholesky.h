#ifndef __CASL_CHOLESKY_H__
#define __CASL_CHOLESKY_H__

#include <src/MatrixFull.h>

// Cholesky decomposition and forward/backward solve for SP(D) matrix
class Cholesky
{
    DenseMatrix Lf;

public:
    /*!
     * \brief compute the Cholesky decomposition of the matrix A
     * \param A the matrix
     */
    bool compute_Cholesky_Decomposition( const DenseMatrix &A );

    /*!
     * \brief solve the system A*x = b where A is a symmetric positive definite matrix
     * \param A the matrix
     * \param b the right hand side
     * \param x the solution
     * \return true if the system was successfully solved, false otherwise (i.e. the matrix given is singular)
     */
    bool solve( DenseMatrix &A, const std::vector<double>& b, std::vector<double>& x );
};

#endif  // __CASL_CHOLESKY_H__
