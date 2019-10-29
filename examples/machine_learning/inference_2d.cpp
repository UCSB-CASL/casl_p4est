//
// Created by Im YoungMin on 10/27/19.
// Based on https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex2.c
//


static char help[] = "";

#include <petsc.h>

int main(int argc,char **args)
{
	PetscMPIInt rank;
	const size_t M = 4, N = 2;
	double values[M][M] = {
			{1, 0, 0, 2},
			{3, 4, 0, 0},
			{5, 0, 6, 0},
			{0, 0, 7, 8}
	};

	double values2[M][N] = {
			{1, 2},
			{3, 4},
			{5, 6},
			{7, 8}
	};

	Vec x, y;
	Mat A, B, C;				// y = Ax, C = A*B
	PetscErrorCode ierr;
	PetscScalar s;
	int iStart, iEnd;			// Rows possessed by this processor.

	ierr = PetscInitialize( &argc, &args, ( char * ) nullptr, help );
	if( ierr ) return ierr;

	ierr = MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
	CHKERRQ( ierr );

	// Create parallel matrix, specifying only its global dimensions.  When using MatCreate(), the matrix format can be
	// specified at runtime. Also, the parallel partitioning of the matrix is determined by PETSc at runtime.
	ierr = MatCreate( PETSC_COMM_WORLD, &A );
	CHKERRQ( ierr );
	ierr = MatSetSizes( A, PETSC_DECIDE, PETSC_DECIDE, M, M );
	CHKERRQ( ierr );
	ierr = MatSetFromOptions( A );
	CHKERRQ( ierr );
	ierr = MatMPIAIJSetPreallocation( A, 2, nullptr, 2, nullptr );
	CHKERRQ( ierr );
	ierr = MatSeqAIJSetPreallocation( A, 2, nullptr );
	CHKERRQ( ierr );
	ierr = MatSeqSBAIJSetPreallocation( A, 1, 2, nullptr );
	CHKERRQ( ierr );
	ierr = MatMPISBAIJSetPreallocation( A, 1, 2, nullptr, 2, nullptr );
	CHKERRQ( ierr );
	ierr = MatMPISELLSetPreallocation( A, 2, nullptr, 2, nullptr );
	CHKERRQ( ierr );
	ierr = MatSeqSELLSetPreallocation( A, 2, nullptr );
	CHKERRQ( ierr );

	// Currently, all PETSc parallel matrix formats are partitioned by contiguous chunks of rows across the processors.
	// Determine which rows of the matrix are locally owned.
	ierr = MatGetOwnershipRange( A, &iStart, &iEnd );
	CHKERRQ( ierr );

	// Set matrix elements:
	// - Each processor needs to insert only elements that it owns locally (but any non-local elements will be sent to
	//   the appropriate processor during matrix assembly).
	// - Always specify global rows and columns of matrix entries.
	for( int i = iStart; i < iEnd; i++ )
	{
		for( int j = 0; j < M; j++ )
		{
			double val = values[i][j];
			if( val != 0 )
			{
				ierr = MatSetValues( A, 1, &i, 1, &j, &val, ADD_VALUES );
				CHKERRQ( ierr );
			}
		}
	}

	// Assemble matrix, using the 2-step process: MatAssemblyBegin(), MatAssemblyEnd().
	// Computations can be done while messages are in transition by placing code between these two statements.
	ierr = MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
	CHKERRQ( ierr );
	ierr = MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
	CHKERRQ( ierr );

	// Create parallel vectors.  The vectors and matrices MUST be partitioned accordingly.  PETSc automatically
	// generates appropriately partitioned matrices and vectors when MatCreate and VecCreate() are used with the same
	// communicator.
	ierr = VecCreate( PETSC_COMM_WORLD, &x );
	CHKERRQ( ierr );
	ierr = VecSetSizes( x, PETSC_DECIDE, M );
	CHKERRQ( ierr );
	ierr = VecSetFromOptions( x );
	CHKERRQ( ierr );
	ierr = VecDuplicate( x, &y );
	CHKERRQ( ierr );
	ierr = VecSet( x, 1.0 ); 			// A vector of ones.
	CHKERRQ( ierr );

	// View the result vector y = Ax.
	ierr = MatMult( A, x, y );
	CHKERRQ( ierr );
	ierr = VecView( y, PETSC_VIEWER_STDOUT_WORLD );
	CHKERRQ( ierr );

	// Sum of elements in y.
	ierr = VecSum( y, &s );
	CHKERRQ( ierr );
	PetscPrintf( PETSC_COMM_WORLD, "The sum is %f\n", s );

	// Now, matrix multiplication: sparse by dense matrix = dense matrix.
	// Create the dense matrix B.
	ierr = MatCreateDense( PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, nullptr, &B );
	CHKERRQ( ierr );

	// Determine which rows of the matrix are locally owned.
	ierr = MatGetOwnershipRange( B, &iStart, &iEnd );
	CHKERRQ( ierr );

	// Set matrix elements.
	PetscInt colIdx[] = {0, 1};
	for( int i = iStart; i < iEnd; i++ )
	{
		ierr = MatSetValues( B, 1, &i, N, colIdx, values2[i], INSERT_VALUES );
		CHKERRQ( ierr );
	}
	ierr = MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY );
	CHKERRQ( ierr );
	ierr = MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY );
	CHKERRQ( ierr );
	PetscPrintf( PETSC_COMM_WORLD, "Matrix B = \n" );
	ierr = MatView( B, PETSC_VIEWER_STDOUT_WORLD );
	CHKERRQ( ierr );

	// Multiply A and B, and view the result in C.
	ierr = MatMatMult( A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C );
	CHKERRQ( ierr );
	PetscPrintf( PETSC_COMM_WORLD, "Matrix C = AB = \n" );
	ierr = MatView( C, PETSC_VIEWER_STDOUT_WORLD );
	CHKERRQ( ierr );

	// Now modify C to see which rank each element belongs to.
	ierr = MatGetOwnershipRange( C, &iStart, &iEnd );
	CHKERRQ( ierr );
	for( int i = iStart; i < iEnd; i++ )
	{
		for( int j = 0; j < N; j++ )
		{
			double val = rank;
			ierr = MatSetValues( C, 1, &i, 1, &j, &val, INSERT_VALUES );
			CHKERRQ( ierr );
		}
	}
	ierr = MatAssemblyBegin( C, MAT_FINAL_ASSEMBLY );
	CHKERRQ( ierr );
	ierr = MatAssemblyEnd( C, MAT_FINAL_ASSEMBLY );
	CHKERRQ( ierr );
	PetscPrintf( PETSC_COMM_WORLD, "Ranks of C = \n" );
	ierr = MatView( C, PETSC_VIEWER_STDOUT_WORLD );
	CHKERRQ( ierr );

	ierr = VecDestroy( &x );
	CHKERRQ( ierr );
	ierr = VecDestroy( &y );
	CHKERRQ( ierr );
	ierr = MatDestroy( &A );
	CHKERRQ( ierr );
	ierr = MatDestroy( &B );
	CHKERRQ( ierr );
	ierr = MatDestroy( &C );
	CHKERRQ( ierr );

	// Always call PetscFinalize() before exiting a program.  This routine:
	// - finalizes the PETSc libraries as well as MPI
	// - provides summary and diagnostic information if certain runtime options are chosen (e.g., -log_view).
	ierr = PetscFinalize();
	return ierr;
}