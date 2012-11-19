#ifndef MACROS_H
#define MACROS_H

#include <p4est.h>

#define MPI_SYNCHRONOUS(__comm__, __code__) \
do { \
  int __rank__, __size__; \
  MPI_Comm_rank((__comm__), &__rank__); \
  MPI_Comm_size((__comm__), &__size__); \
  if (__size__ == 1){ \
    __code__ \
  } else { \
    for (int n=0; n<__size__; n++) { \
      if (n == (__rank__)) { __code__ }\
      MPI_Barrier((__comm__)); \
    } \
  } \
} while (0)

#endif // MACROS_H
