#ifndef MY_P4EST_LOG_WRAPPERS_H
#define MY_P4EST_LOG_WRAPPERS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#endif

#include "petsc_compatibility.h"

#ifdef __cplusplus
extern "C"
{
#endif

p4est_t*
my_p4est_new (MPI_Comm mpicomm, p4est_connectivity_t * connectivity,
              size_t data_size, p4est_init_t init_fn, void *user_pointer);

p4est_ghost_t*
my_p4est_ghost_new(p4est_t *p4est, p4est_connect_type_t btype);

void
my_p4est_refine (p4est_t * p4est, int refine_recursive,
                 p4est_refine_t refine_fn, p4est_init_t init_fn);

void
my_p4est_coarsen(p4est_t *p4est, int coarsen_recursive,
                 p4est_coarsen_t coarsen_fn, p4est_init_t init_fn);

void
my_p4est_partition(p4est_t *p4est, p4est_weight_t weight_fn);

void
my_sc_notify(int *receivers, int num_receivers,
             int *senders, int *num_senders,
             MPI_Comm mpicomm);

void
my_sc_notify_allgather(int *receivers, int num_receivers,
             		   int *senders, int *num_senders,
             		   MPI_Comm mpicomm);

#ifdef __cplusplus
}
#endif


#endif // MY_P4EST_LOG_WRAPPERS_H
