#include "poisson_boltzman_solver.h"

#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_poisson_nodes.h>
#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

using namespace std;

PoissonBoltzmanSolver::PoissonBoltzmanSolver(my_p4est_node_neighbors_t& neighbors)
  : neighbors(neighbors) {
  p4est = neighbors.get_p4est();
  nodes = neighbors.get_nodes();
}


void PoissonBoltzmanSolver::set_parameters(double edl, double zeta) {
  this->edl  = edl;
  this->zeta = zeta;
}

void PoissonBoltzmanSolver::set_phi(Vec phi) {
  this->phi = phi;
}


void PoissonBoltzmanSolver::solve_linear(Vec &psi)
{
  PetscErrorCode ierr;
  my_p4est_poisson_nodes_t solver(&neighbors);

  Vec add;
  ierr = VecDuplicate(phi, &add); CHKERRXX(ierr);

  // set the rhs
  double *psi_p, *add_p;
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(add, &add_p); CHKERRXX(ierr);

  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++) {
    // use psi to set the rhs
    psi_p[i] = 0;
    add_p[i] = SQR(1.0/edl);
  }
  ierr = VecRestoreArray(add, &add_p); CHKERRXX(ierr);

  struct:CF_3{
    double operator()(double, double, double) const { return 0; }
  } bc_wall_value;

  struct bc_value_t:CF_3{
    double value;
    double operator()(double, double, double) const { return value; }
  } bc_interface_value; bc_interface_value.value = zeta;

  struct:WallBC3D {
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;

  BoundaryConditions3D bc;
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);

  solver.set_phi(phi);
  solver.set_diagonal(add);
  solver.set_bc(bc);
  solver.set_rhs(psi);
  solver.solve(psi);

  // Destroy unecessary vectors
  ierr = VecDestroy(add); CHKERRXX(ierr);

  // extend solutions
  my_p4est_level_set_t ls(&neighbors);
  ls.extend_Over_Interface_TVD(phi, psi);

  // restore pointers
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
}

void PoissonBoltzmanSolver::solve_nonlinear(Vec &psi, int itmax, double tol)
{
  PetscErrorCode ierr;

  // tmp points to the old solution
  Vec tmp, add;
  ierr = VecDuplicate(phi, &add); CHKERRXX(ierr);
  ierr = VecDuplicate(psi, &tmp); CHKERRXX(ierr);

  double *psi_p, *phi_p, *add_p, *tmp_p;
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(add, &add_p); CHKERRXX(ierr);
  ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

  int it = 0;
  double err = 1 + tol;
  double kappa_sqr = SQR(1.0/edl);

  for (size_t i = 0; i < nodes->num_owned_indeps; i++) {
    tmp_p[i] = 0;
  }

  struct:CF_3{
    double operator()(double, double, double) const { return 0; }
  } bc_wall_value;

  struct:CF_3{
    double value;
    double operator()(double, double, double) const { return value; }
  } bc_interface_value; bc_interface_value.value = zeta;

  struct:WallBC3D {
    BoundaryConditionType operator()(double, double, double) const { return DIRICHLET; }
  } bc_wall_type;

  BoundaryConditions3D bc;
  bc.setWallTypes(bc_wall_type);
  bc.setWallValues(bc_wall_value);
  bc.setInterfaceType(DIRICHLET);
  bc.setInterfaceValue(bc_interface_value);

  my_p4est_level_set_t ls(&neighbors);
  while (it++ < itmax && err > tol) {
    my_p4est_poisson_nodes_t solver(&neighbors);

    for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++) {
      if (phi_p[i] < 0){
        // use psi to set the rhs
        psi_p[i] = -kappa_sqr*sinh(tmp_p[i]) + kappa_sqr*tmp_p[i]*cosh(tmp_p[i]);

        // set the add to diagonal
        add_p[i] = kappa_sqr*cosh(tmp_p[i]);
      } else {
        psi_p[i] = 0;
        add_p[i] = 0;
      }
    }

    solver.set_phi(phi);
    solver.set_diagonal(add);
    solver.set_bc(bc);
    solver.set_rhs(psi);
    solver.solve(psi);

    // extend solutions
    ls.extend_Over_Interface_TVD(phi, psi, 10);

    err = 0;
    for (size_t i = 0; i<nodes->num_owned_indeps; i++) {
      if (phi_p[i] < 0)
        err = MAX(err, fabs(psi_p[i]  - tmp_p[i]));
    }

    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
    for (size_t i = 0; i < nodes->indep_nodes.elem_count; i++){
      tmp_p[i] = psi_p[i];
    }

    PetscPrintf(p4est->mpicomm, "It = %2d \t err = %1.5e\n", it, err);
  }

  // restore pointers
  ierr = VecRestoreArray(add, &add_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

  // destroy temporary solution
  ierr = VecDestroy(add); CHKERRXX(ierr);
  ierr = VecDestroy(tmp); CHKERRXX(ierr);
}
