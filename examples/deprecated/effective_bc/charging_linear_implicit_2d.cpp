#ifdef P4_TO_P8
#include "charging_linear_implicit_3d.h"
#else
#include "charging_linear_implicit_2d.h"
#endif

ImplicitLinearChargingSolver::ImplicitLinearChargingSolver(p4est_t* p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, my_p4est_brick_t *brick_, my_p4est_node_neighbors_t *ngbd_)
  : p4est(p4est_), ghost(ghost_), nodes(nodes_), brick(brick_), ngbd(ngbd_),
    local_phi_dd(false),
    G_interp(p4est, nodes, ghost, brick, ngbd),
    psi_solver(ngbd)
{
  ierr = VecCreateGhostNodes(p4est, nodes, &psi); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &G); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &alpha); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &sol); CHKERRXX(ierr);
}

ImplicitLinearChargingSolver::~ImplicitLinearChargingSolver(){
  ierr = VecDestroy(psi); CHKERRXX(ierr);
  ierr = VecDestroy(G); CHKERRXX(ierr);
  ierr = VecDestroy(alpha); CHKERRXX(ierr);
  ierr = VecDestroy(sol); CHKERRXX(ierr);
  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  if(local_phi_dd){
    ierr = VecDestroy(phi_xx); CHKERRXX(ierr);
    ierr = VecDestroy(phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz); CHKERRXX(ierr);
#endif
  }
}

#ifdef P4_TO_P8
void ImplicitLinearChargingSolver::set_phi(Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void ImplicitLinearChargingSolver::set_phi(Vec phi, Vec phi_xx, Vec phi_yy)
#endif
{
  this->phi = phi;
#ifdef P4_TO_P8
  if (phi_xx != NULL && phi_yy != NULL && phi_zz != NULL)
#else
  if (phi_xx != NULL && phi_yy != NULL)
#endif
  {
    this->phi_xx = phi_xx;
    this->phi_yy = phi_yy;
#ifdef P4_TO_P8
    this->phi_zz = phi_zz;
#endif
    local_phi_dd = false;
  } else {
    ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_xx); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_zz); CHKERRXX(ierr);
    ngbd->second_derivatives_central(phi, this->phi_xx, this->phi_yy, this->phi_zz);
#else
    ngbd->second_derivatives_central(phi, this->phi_xx, this->phi_yy);
#endif
    local_phi_dd = true;
  }
}

void ImplicitLinearChargingSolver::init(){
  psi_bc.setInterfaceType(ROBIN);
  psi_bc.setInterfaceValue(G_interp);
  psi_bc.setWallTypes(wall_bc);
  psi_bc.setWallValues(wall_psi_value);

  double *alpha_p, *psi_p;
  ierr = VecGetArray(alpha, &alpha_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  for (size_t n=0; n<nodes->indep_nodes.elem_count; n++){
    alpha_p[n] = lambda / dt;
    psi_p[n]   = 1.0;
  }
  ierr = VecRestoreArray(alpha, &alpha_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);

  psi_solver.set_robin_coef(alpha);

  psi_solver.set_bc(psi_bc);
#ifdef P4_TO_P8
  psi_solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
#else
  psi_solver.set_phi(phi, phi_xx, phi_yy);
#endif
}

void ImplicitLinearChargingSolver::solve()
{
  parStopWatch w;
  w.start("linear system solve");
  // compute value of the boundary condition
  double *psi_p, *G_p;
  ierr = VecGetArray(psi, &psi_p);
  ierr = VecGetArray(G, &G_p);
  for (size_t n=0; n<nodes->indep_nodes.elem_count; n++){
    G_p[n] = lambda/dt * psi_p[n];
  }
  ierr = VecRestoreArray(psi, &psi_p);
  ierr = VecRestoreArray(G, &G_p);

  // construct an interpolating function
  G_interp.set_input_parameters(G, linear);

  psi_bc.setInterfaceType(ROBIN);
  psi_bc.setInterfaceValue(G_interp);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    rhs_p[i] = 0;
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  psi_solver.set_rhs(rhs);
  psi_solver.solve(psi);
  w.stop(); w.read_duration();

  w.start("extrapolation");
  my_p4est_level_set ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, psi, 20, 2);
//  ls.extend_Over_Interface(phi, psi, 2, 10);
  w.stop(); w.read_duration();
}

void ImplicitLinearChargingSolver::write_vtk(const std::string& filename){
  double *phi_p, *psi_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         2, 0, filename.c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "psi", psi_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
}
