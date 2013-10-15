#ifndef MY_P4EST_LOG_WRAPPERS_H
#define MY_P4EST_LOG_WRAPPERS_H

#include <p4est.h>
#include <p4est_ghost.h>

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

#endif // MY_P4EST_LOG_WRAPPERS_H
