#ifndef MY_MPI_WORLD_H
#define MY_MPI_WORLD_H

#include <src/petsc_compatibility.h>

class my_mpi_world{
  PetscErrorCode ierr;
  MPI_Comm mpicomm;
  int mpirank;
  int mpisize;

public:
  ~my_mpi_world(){
    ierr = PetscFinalize(); CHKERRXX(ierr);
    MPI_Finalize();
  }

  my_mpi_world(int argc, char **argv){
    mpicomm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(mpicomm, &mpisize);
    MPI_Comm_rank(mpicomm, &mpirank);

    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRXX(ierr);
  }

  inline const MPI_Comm& comm() const {return mpicomm;}
  inline const int& rank() const {return mpirank;}
  inline const int& size() const {return mpisize;}

};

#endif // MY_MPI_WORLD_H
