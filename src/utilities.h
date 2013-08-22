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
public:
  typedef enum{
    root_timings,
    all_timings
  } stopwatch_timing;

private:
    double ts, tf;
    MPI_Comm comm_;
    int mpirank;
    std::string msg_;
    stopwatch_timing timing_;

public:   

    parStopWatch(stopwatch_timing timing = root_timings, MPI_Comm comm = MPI_COMM_WORLD)
      : comm_(comm), timing_(timing)
    {
      MPI_Comm_rank(comm_, &mpirank);
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

      PetscPrintf(comm_, "%s ... done in ", msg_.c_str());
      if (timing_ == all_timings){
        PetscSynchronizedPrintf(comm_, "\n   %.4lf secs. on process %2d",elap, mpirank);
        PetscSynchronizedFlush(comm_);
        PetscPrintf(comm_, "\n");
      } else {
        PetscPrintf(comm_, " %.4lf secs. on process %d [Note: only showing root's timings]\n", elap, mpirank);
      }
      return elap;
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
