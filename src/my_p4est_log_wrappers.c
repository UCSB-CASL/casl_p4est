#ifdef P4_TO_P8
#include <src/my_p8est_log_wrappers.h>
#else
#include <src/my_p4est_log_wrappers.h>
#endif

#include <petsclog.h>
#include <src/p4est_compatibility.h>
#include <sc_notify.h>
#include <src/ipm_logging.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_new;
extern PetscLogEvent log_my_p4est_ghost_new;
extern PetscLogEvent log_my_p4est_ghost_expand;
extern PetscLogEvent log_my_p4est_copy;
extern PetscLogEvent log_my_p4est_refine;
extern PetscLogEvent log_my_p4est_coarsen;
extern PetscLogEvent log_my_p4est_partition;
extern PetscLogEvent log_my_p4est_balance;
extern PetscLogEvent log_my_sc_notify;
extern PetscLogEvent log_my_sc_notify_allgather;
#endif


p4est_t*
my_p4est_new(MPI_Comm mpicomm, p4est_connectivity_t *connectivity, size_t data_size, p4est_init_t init_fn, void *user_pointer)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_new, 0, 0, 0, 0); CHKERRXX(ierr);
	IPMLogRegionBegin("p4est_new");
  p4est_t *p4est = p4est_new(mpicomm, connectivity, data_size, init_fn, user_pointer);
	IPMLogRegionEnd("p4est_new");
  ierr = PetscLogEventEnd(log_my_p4est_new, 0, 0, 0, 0); CHKERRXX(ierr);

  return p4est;
}

p4est_t*
my_p4est_copy(p4est_t* input, int copy_data) {
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_copy, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("p4est_copy");
  p4est_t* p4est = p4est_copy(input, copy_data);
  IPMLogRegionEnd("p4est_copy");
  ierr = PetscLogEventEnd(log_my_p4est_copy, 0, 0, 0, 0); CHKERRXX(ierr);

  return p4est;
}

p4est_ghost_t*
my_p4est_ghost_new(p4est_t *p4est, p4est_connect_type_t btype)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_ghost_new, 0, 0, 0, 0); CHKERRXX(ierr);
	IPMLogRegionBegin("p4est_ghost_new");
  p4est_ghost_t *ghost = p4est_ghost_new(p4est, btype); CHKERRXX(ierr);
	IPMLogRegionEnd("p4est_ghost_new");
  ierr = PetscLogEventEnd(log_my_p4est_ghost_new, 0, 0, 0, 0); CHKERRXX(ierr);
  

  return ghost;
}

void
my_p4est_ghost_expand(p4est_t *p4est, p4est_ghost_t *ghost)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_ghost_expand, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("p4est_ghost_expand");
  p4est_ghost_expand(p4est, ghost);
  IPMLogRegionEnd("p4est_ghost_expand");
  ierr = PetscLogEventEnd(log_my_p4est_ghost_expand, 0, 0, 0, 0); CHKERRXX(ierr);
}

void
my_p4est_refine(p4est_t *p4est, int refine_recursive, p4est_refine_t refine_fn, p4est_init_t init_fn)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_refine, 0, 0, 0, 0); CHKERRXX(ierr);
	IPMLogRegionBegin("p4est_refine");
  p4est_refine(p4est, refine_recursive, refine_fn, init_fn);
	IPMLogRegionEnd("p4est_refine");
  ierr = PetscLogEventEnd(log_my_p4est_refine, 0, 0, 0, 0); CHKERRXX(ierr);
}

void
my_p4est_coarsen(p4est_t *p4est, int coarsen_recursive, p4est_coarsen_t coarsen_fn, p4est_init_t init_fn)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_coarsen, 0, 0, 0, 0); CHKERRXX(ierr);
	IPMLogRegionBegin("p4est_coarsen");
  p4est_coarsen(p4est, coarsen_recursive, coarsen_fn, init_fn);
	IPMLogRegionEnd("p4est_coarsen");
  ierr = PetscLogEventEnd(log_my_p4est_coarsen, 0, 0, 0, 0); CHKERRXX(ierr);
}

void
my_p4est_partition(p4est_t *p4est, int allow_for_coarsening, p4est_weight_t weight_fn)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_partition, 0, 0, 0, 0); CHKERRXX(ierr);
	IPMLogRegionBegin("p4est_partition");  
#if P4EST_VERSION_LT(1,0)
  (void) allow_for_coarsening;
  p4est_partition(p4est, weight_fn);
#else  
  p4est_partition(p4est, allow_for_coarsening, weight_fn);
#endif  
IPMLogRegionEnd("p4est_partition");
  ierr = PetscLogEventEnd(log_my_p4est_partition, 0, 0, 0, 0); CHKERRXX(ierr);
}

void
my_p4est_balance(p4est_t *p4est, p4est_connect_type_t btype, p4est_init_t init_fn)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_balance, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("p4est_balance");
  p4est_balance(p4est, btype, init_fn);
  IPMLogRegionEnd("p4est_balance");
  ierr = PetscLogEventEnd(log_my_p4est_balance, 0, 0, 0, 0); CHKERRXX(ierr);
}

void
my_sc_notify(int *receivers, int num_receivers, int *senders, int *num_senders, MPI_Comm mpicomm)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_sc_notify, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("sc_notify");
  sc_notify(receivers, num_receivers, senders, num_senders, mpicomm);
  IPMLogRegionEnd("sc_notify");
  ierr = PetscLogEventEnd(log_my_sc_notify, 0, 0, 0, 0); CHKERRXX(ierr);
}

void
my_sc_notify_allgather(int *receivers, int num_receivers, int *senders, int *num_senders, MPI_Comm mpicomm)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_sc_notify_allgather, 0, 0, 0, 0); CHKERRXX(ierr);
  IPMLogRegionBegin("sc_notify_allgather");
  sc_notify_allgather(receivers, num_receivers, senders, num_senders, mpicomm);
  IPMLogRegionEnd("sc_notify_allgather");
  ierr = PetscLogEventEnd(log_my_sc_notify_allgather, 0, 0, 0, 0); CHKERRXX(ierr);
}
