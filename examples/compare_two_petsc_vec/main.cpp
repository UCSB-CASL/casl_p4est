#include <mpi.h>
#include <src/petsc_compatibility.h>

using namespace std;

int main(int argc, char** argv) {

  // prepare parallel enviroment
  MPI_Init(&argc, &argv);
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
  Vec v1, v2, diff;
  Mat A1, A2, A_diff;

  if(argc != 3)
    throw std::invalid_argument("need 2 arguments!");

  const string path_1 = string(argv[1]);
  const string path_2 = string(argv[2]);


  ierr = VecCreate(MPI_COMM_WORLD, &v1);CHKERRQ(ierr);
  ierr = VecCreate(MPI_COMM_WORLD, &v2);CHKERRQ(ierr);
  ierr = MatCreate(MPI_COMM_WORLD, &A1);CHKERRQ(ierr);
  ierr = MatCreate(MPI_COMM_WORLD, &A2);CHKERRQ(ierr);
  ierr = MatCreate(MPI_COMM_WORLD, &A_diff);CHKERRQ(ierr);

  PetscViewer viewer;

  ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, (path_1 + "matrix_p_guess_binary").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = MatLoad(A1, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, (path_2 + "matrix_p_guess_binary").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = MatLoad(A2, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);


  PetscReal A1_norm, A2_norm;
  ierr = MatNorm(A1, NORM_FROBENIUS, &A1_norm); CHKERRXX(ierr);
  ierr = MatNorm(A2, NORM_FROBENIUS, &A2_norm); CHKERRXX(ierr);

  ierr = MatDuplicate(A1, MAT_SHARE_NONZERO_PATTERN, &A_diff); CHKERRXX(ierr);
  ierr = MatCopy(A1, A_diff, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  ierr = MatAXPY(A_diff, -1, A2, SAME_NONZERO_PATTERN); CHKERRXX(ierr);

  PetscReal A_diff_norm;
  ierr = MatNorm(A_diff, NORM_FROBENIUS, &A_diff_norm); CHKERRXX(ierr);

  ierr = PetscPrintf(MPI_COMM_WORLD, "------ matrices ------ \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "-- (frobenius-norm) -- \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "difference = %g,\t A = %g,\t B = %g \n", A_diff_norm, A1_norm, A2_norm); CHKERRXX(ierr);
  ierr = MatDestroy(A1); CHKERRXX(ierr);
  ierr = MatDestroy(A2); CHKERRXX(ierr);
  ierr = MatDestroy(A_diff); CHKERRXX(ierr);


  ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, (path_1 + "rhs_p_guess_binary").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = VecLoad(v1, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, (path_2 + "rhs_p_guess_binary").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = VecLoad(v2, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  PetscReal v1_norm, v2_norm, diff_norm, v1_max_norm, v2_max_norm, max_diff;
  ierr = VecNorm(v1, NORM_2, &v1_norm); CHKERRXX(ierr);
  ierr = VecNorm(v2, NORM_2, &v2_norm); CHKERRXX(ierr);

  ierr = VecNorm(v1, NORM_INFINITY, &v1_max_norm); CHKERRXX(ierr);
  ierr = VecNorm(v2, NORM_INFINITY, &v2_max_norm); CHKERRXX(ierr);


  ierr = VecDuplicate(v1, &diff); CHKERRXX(ierr);
  ierr = VecCopy(v1, diff); CHKERRXX(ierr);
  ierr = VecAXPBY(diff, 1, -1, v2); CHKERRXX(ierr);

  ierr = VecNorm(diff, NORM_2, &diff_norm); CHKERRXX(ierr);
  ierr = VecNorm(diff, NORM_INFINITY, &max_diff); CHKERRXX(ierr);


  ierr = PetscPrintf(MPI_COMM_WORLD, "------ RHS ------ \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "-- (max-norm) -- \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "difference = %g,\t A = %g,\t B = %g \n", max_diff, v1_max_norm, v2_max_norm); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "--- (2-norm) --- \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "difference = %g,\t A = %g,\t B = %g \n", diff_norm, v1_norm, v2_norm); CHKERRXX(ierr);

  ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, (path_1 + "p_guess_binary").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = VecLoad(v1, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, (path_2 + "p_guess_binary").c_str(), FILE_MODE_READ, &viewer); CHKERRXX(ierr);
  ierr = VecLoad(v2, viewer); CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);

  ierr = VecNorm(v1, NORM_2, &v1_norm); CHKERRXX(ierr);
  ierr = VecNorm(v2, NORM_2, &v2_norm); CHKERRXX(ierr);

  ierr = VecNorm(v1, NORM_INFINITY, &v1_max_norm); CHKERRXX(ierr);
  ierr = VecNorm(v2, NORM_INFINITY, &v2_max_norm); CHKERRXX(ierr);


  ierr = VecDuplicate(v1, &diff); CHKERRXX(ierr);
  ierr = VecCopy(v1, diff); CHKERRXX(ierr);
  ierr = VecAXPBY(diff, 1, -1, v2); CHKERRXX(ierr);

  ierr = VecNorm(diff, NORM_2, &diff_norm); CHKERRXX(ierr);
  ierr = VecNorm(diff, NORM_INFINITY, &max_diff); CHKERRXX(ierr);

  ierr = PetscPrintf(MPI_COMM_WORLD, "--- Solution --- \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "-- (max-norm) -- \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "difference = %g,\t A = %g,\t B = %g \n", max_diff, v1_max_norm, v2_max_norm); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "--- (2-norm) --- \n"); CHKERRXX(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "difference = %g,\t A = %g,\t B = %g \n", diff_norm, v1_norm, v2_norm); CHKERRXX(ierr);


  ierr = VecDestroy(v1); CHKERRXX(ierr);
  ierr = VecDestroy(v2); CHKERRXX(ierr);
  ierr = VecDestroy(diff); CHKERRXX(ierr);


  ierr = PetscFinalize(); CHKERRXX(ierr);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

