#ifndef UTILITIES_H
#define UTILITIES_H
#include <mpi.h>
#include <petsc.h>
#include <iostream>
#include <string>
#include "Macros.h"

class PetscSession {
private:
    int rank, size;
    MPI_Comm comm;

public:
    PetscSession(int *argc, char ***argv, const char file[], const char help[])
    {
        PetscInitialize(argc, argv, file, help);

        comm = PETSC_COMM_WORLD;
        MPI_Comm_rank(comm,&rank);
        MPI_Comm_size(comm,&size);
    }

    inline int getRank(){return rank;}
    inline int getSize(){return size;}
    ~PetscSession(){
        PetscFinalize();
    }
};

class parStopWatch{
private:
    double ts, tf;
    MPI_Comm comm_;
    int mpisize;
    std::string msg_;

public:
    parStopWatch(MPI_Comm comm = PETSC_COMM_WORLD)
        : comm_(comm)
    {
      MPI_Comm_size(comm_, &mpisize);
    }

    void start(const std::string& msg){
        msg_ = msg;
        PetscPrintf(comm_, "%s ... \n", msg.c_str());
        ts = MPI_Wtime();
    }

    void stop(){
        tf = MPI_Wtime();
    }

    double read_duration(){
      double elap = tf - ts;
      double max, min, sum;
      MPI_Reduce(&elap, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
      MPI_Reduce(&elap, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
      MPI_Reduce(&elap, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm_);

      PetscPrintf(comm_, "%s ... done in [avg = %.2lf (s), min = %.2lf (s), max/min = %.2lf] on %2d processes\n",msg_.c_str(), sum/mpisize, min, max/min, mpisize);
      return tf-ts;
    }
};


typedef enum {
    LocalIndex,
    PetscGlobalIndex,
    ApplicationGlobalIndex
} AccessIndexType;

typedef enum {
    BeforePartitioning,
    AfterPartitioning
} PartitioningStatus;

typedef enum {
    WholeDomain,
    OmegaMinus
} PartitioningRegion;

#define DPAUSE_MSG(COMM,RANK,MSG) \
    do{ \
    PetscSynchronizedPrintf((COMM), "[%d] Stoped inside:\nFile = %s\nLine = %d\nWith user-provided message: '%s'\n", (RANK), __FILE__, __LINE__, (MSG)); \
    PetscSynchronizedFlush((COMM)); \
    getchar(); \
    MPI_Barrier((COMM)); \
} while(0)

#define DPAUSE(COMM, RANK) \
    do{ \
    PetscSynchronizedPrintf((COMM), "[%d] Stoped inside:\nFile = %s\nLine = %d\n",(RANK), __FILE__, __LINE__); \
    PetscSynchronizedFlush((COMM)); \
    getchar(); \
    MPI_Barrier((COMM)); \
} while(0)

#endif // UTILITIES_H
