#ifdef P4_TO_P8
#include "charging_nonlinear_explicit_3d.h"
#else
#include "charging_nonlinear_explicit_2d.h"
#endif
#include <src/my_p4est_refine_coarsen.h>

ExplicitNonLinearChargingSolver::ExplicitNonLinearChargingSolver(p4est_t* p4est_, p4est_ghost_t *ghost_, p4est_nodes_t *nodes_, my_p4est_brick_t *brick_, my_p4est_node_neighbors_t *ngbd_)
  : p4est(p4est_), ghost(ghost_), nodes(nodes_), brick(brick_), ngbd(ngbd_),
    local_phi_dd(false),
    psi_interp(p4est, nodes, ghost, brick, ngbd),
    con_interp(p4est, nodes, ghost, brick, ngbd),
    psi_solver(ngbd),
    con_solver(ngbd)
{
  ierr = VecCreateGhostNodes(p4est, nodes, &psi); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &con); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &rhs); CHKERRXX(ierr);
}

ExplicitNonLinearChargingSolver::~ExplicitNonLinearChargingSolver(){
  ierr = VecDestroy(psi); CHKERRXX(ierr);
  ierr = VecDestroy(con); CHKERRXX(ierr);
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
void ExplicitNonLinearChargingSolver::set_phi(Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void ExplicitNonLinearChargingSolver::set_phi(Vec phi, Vec phi_xx, Vec phi_yy)
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

void ExplicitNonLinearChargingSolver::init(){
  // construct an interpolating function
  psi_bc.setInterfaceType(DIRICHLET);
  psi_bc.setInterfaceValue(psi_interp);
  psi_bc.setWallTypes(psi_wall_bc);
  psi_bc.setWallValues(psi_wall_value);

  con_bc.setInterfaceType(DIRICHLET);
  con_bc.setInterfaceValue(con_interp);
  con_bc.setWallTypes(con_wall_bc);
  con_bc.setWallValues(con_wall_value);

  psi_interp.set_input_parameters(psi, linear);
  con_interp.set_input_parameters(con, linear);

  double *psi_p, *con_p;
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(con, &con_p); CHKERRXX(ierr);
  for (size_t n=0; n<nodes->indep_nodes.elem_count; n++){
    psi_p[n] = psi_e;
    con_p[n] = 1.0;
  }
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(con, &con_p); CHKERRXX(ierr);

  psi_solver.set_bc(psi_bc);
  con_solver.set_bc(con_bc);
#ifdef P4_TO_P8
  psi_solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
  con_solver.set_phi(phi, phi_xx, phi_yy, phi_zz);
#else
  psi_solver.set_phi(phi, phi_xx, phi_yy);
  con_solver.set_phi(phi, phi_xx, phi_yy);
#endif

  // initialize the potential field
  parStopWatch w;
  w.start("initializing the potential field");
  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    rhs_p[i] = 0;
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  psi_solver.set_rhs(rhs);
  psi_solver.solve(psi);

  my_p4est_level_set ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, psi, 20, 2);
  w.stop(); w.read_duration();
}

void ExplicitNonLinearChargingSolver::nonlinear_solve(int itmax, double tol) {
  const splitting_criteria_t *data = (const splitting_criteria_t*)p4est->user_pointer;
  double diag = (double)P4EST_QUADRANT_LEN(data->max_lvl) / (double)P4EST_ROOT_LEN * sqrt(2);

  PetscErrorCode ierr;
  double *phi_p, *psi_p, *con_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(con, &con_p);   CHKERRXX(ierr);

  Vec con_np1, psi_np1;
  ierr = VecDuplicate(con, &con_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(psi, &psi_np1); CHKERRXX(ierr);
  double *con_np1_p, *psi_np1_p;
  ierr = VecGetArray(con_np1, &con_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_np1, &psi_np1_p); CHKERRXX(ierr);

  double J[2][2];
  double F[2], dX[2], X[2], tmp[2];
  quad_neighbor_nodes_of_node_t qnnn;

  for (p4est_locidx_t n = 0; n<nodes->num_owned_indeps; n++) {
//    if (fabs(phi_p[n]) > 3*diag) continue; // too far -- we dont need to calculate the values
    if (phi_p[n] > 0) continue; // too far -- we dont need to calculate the values

    // compute the fluxes on the surface from previous time
    ngbd->get_neighbors(n, qnnn);
    double nx = qnnn.dx_central(phi_p), psix = qnnn.dx_central(psi_p), cx = qnnn.dx_central(con_p);
    double ny = qnnn.dy_central(phi_p), psiy = qnnn.dy_central(psi_p), cy = qnnn.dy_central(con_p);
#ifdef P4_TO_P8
    double nz = qnnn.dz_central(phi_p), psiz = qnnn.dz_central(psi_p), cz = qnnn.dz_central(con_p);
#endif

#ifdef P4_TO_P8
    double abs = MAX(sqrt(nx*nx + ny*ny + nz*nz), EPS);
    nx /= abs; ny /= abs; nz /= abs;
    double i_n = -con_p[n]*(nx*psix + ny*psiy + nz*psiz);
    double j_n = -(nx*cx + ny*cy + nz*cz);
#else
    double abs = MAX(sqrt(nx*nx + ny*ny), EPS);
    nx /= abs; ny /= abs;
    double i_n = -con_p[n]*(nx*psix + ny*psiy);
    double j_n = -(nx*cx + ny*cy);
#endif

    // nonlinear solve iteration
    int it = 0;
    double err = 1 + tol;
    tmp[0] = get_q(con_p[n], psi_p[n]) + dt/lambda * i_n;
    tmp[1] = get_w(con_p[n], psi_p[n]) + dt/lambda * j_n;

    // using previous solution as initial guess
    X[0] = con_p[n];
    X[1] = psi_p[n];

    while(it < itmax && err > tol) {
      F[0] = -(get_q(X[0], X[1]) - tmp[0]);
      F[1] = -(get_w(X[0], X[1]) - tmp[1]);

      get_jacobian(X[0], X[1], J);
      double det = J[0][0]*J[1][1] - J[0][1]*J[1][0];
      std::cout << "det = " << det << std::endl;
      if(ISNAN(det))
        std::cout << "WTF" << std::endl;

      if (fabs(det) < 1e-15)
        std::cout << "Warning: Nearly singular Jacobian" << std::endl;

      dX[0] = (J[1][1]*F[0] - J[0][1]*F[1])/det;
      dX[1] = (J[0][0]*F[1] - J[1][0]*F[0])/det;

      err = MAX(fabs(dX[0]), fabs(dX[1])); it++;

      X[0] += dX[0];
      X[1] += dX[1];
      std::cout << "Nonlinear iter = " << it << ", dx = (" << dX[0] << "," << dX[1] << ");" << std::endl;
    }
//    if (err > tol)

    if(ISNAN(X[0]) || ISNAN(X[1]))
      std::cerr << "NAN value for n = " << n << ". c = " << X[0] << ", psi = " << X[1] << std::endl;
    if(ISINF(X[0]) || ISINF(X[1]))
      std::cerr << "INF value for n = " << n << ". c = " << X[0] << ", psi = " << X[1] << std::endl;

    con_np1_p[n] = X[0];
    psi_np1_p[n] = X[1];
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(con, &con_p);   CHKERRXX(ierr);
  ierr = VecRestoreArray(con_np1, &con_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_np1, &psi_np1_p); CHKERRXX(ierr);

  // swap pointers and release memory
  ierr = VecDestroy(psi); CHKERRXX(ierr); psi = psi_np1; psi_interp.set_input_parameters(psi, linear);
  ierr = VecDestroy(con); CHKERRXX(ierr); con = con_np1; con_interp.set_input_parameters(con, linear);

  // Extend over interface
  parStopWatch w;
  w.start("Extending solutions");
  my_p4est_level_set ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, psi, 20, 2);
  ls.extend_Over_Interface_TVD(phi, con, 20, 2);
  w.stop(); w.read_duration();

  // update ghost values -- NOTE: This can be done more efficiently by overlapping
  ierr = VecGhostUpdateBegin(psi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(psi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(con, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(con, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}


void ExplicitNonLinearChargingSolver::nonlinear_solve_decoupled(int itmax, double tol) {
  const splitting_criteria_t *data = (const splitting_criteria_t*)p4est->user_pointer;
  double diag = (double)P4EST_QUADRANT_LEN(data->max_lvl) / (double)P4EST_ROOT_LEN * sqrt(2);

  PetscErrorCode ierr;
  double *phi_p, *psi_p, *con_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(con, &con_p);   CHKERRXX(ierr);

  Vec con_np1, psi_np1;
  ierr = VecDuplicate(con, &con_np1); CHKERRXX(ierr);
  ierr = VecDuplicate(psi, &psi_np1); CHKERRXX(ierr);
  double *con_np1_p, *psi_np1_p;
  ierr = VecGetArray(con_np1, &con_np1_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi_np1, &psi_np1_p); CHKERRXX(ierr);

  double F[2], dX[2], X[2], tmp[2];
  quad_neighbor_nodes_of_node_t qnnn;

  for (p4est_locidx_t n = 0; n<nodes->num_owned_indeps; n++) {
//    if (fabs(phi_p[n]) > 3*diag) continue; // too far -- we dont need to calculate the values
    if (phi_p[n] > 0 || phi_p[n] < -2*diag) continue; // too far -- we dont need to calculate the values

    // compute the fluxes on the surface from previous time
    ngbd->get_neighbors(n, qnnn);
    double nx = qnnn.dx_central(phi_p), psix = qnnn.dx_central(psi_p), cx = qnnn.dx_central(con_p);
    double ny = qnnn.dy_central(phi_p), psiy = qnnn.dy_central(psi_p), cy = qnnn.dy_central(con_p);
#ifdef P4_TO_P8
    double nz = qnnn.dz_central(phi_p), psiz = qnnn.dz_central(psi_p), cz = qnnn.dz_central(con_p);
#endif

#ifdef P4_TO_P8
    double abs = MAX(sqrt(nx*nx + ny*ny + nz*nz), EPS);
    nx /= abs; ny /= abs; nz /= abs;
    double i_n = -con_p[n]*(nx*psix + ny*psiy + nz*psiz);
    double j_n = -(nx*cx + ny*cy + nz*cz);
#else
    double abs = MAX(sqrt(nx*nx + ny*ny), EPS);
    nx /= abs; ny /= abs;
    double i_n = -con_p[n]*(nx*psix + ny*psiy);
    double j_n = -(nx*cx + ny*cy);
#endif

    // nonlinear solve iteration
    int it = 0;
    double err = 1 + tol;
    tmp[0] = get_q(con_p[n], psi_p[n]) + dt/lambda * i_n;
    tmp[1] = get_w(con_p[n], psi_p[n]) + dt/lambda * j_n;

    // using previous solution as initial guess
    X[0] = con_p[n];
    X[1] = psi_p[n];

    while(it < itmax && err > tol) {
      F[0]  = -(get_q(X[0], X[1]) - tmp[0]);
      dX[1] = F[0] / get_dq_dpsi(X[0], X[1]);

      F[1]  = -(get_w(X[0], X[1]) - tmp[1]);
      dX[0] =  F[1] / get_dw_dc(X[0], X[1]);

      X[0] += 0.2*dX[0];
      X[1] += 0.2*dX[1];

      err = MAX(fabs(dX[0]), fabs(dX[1])); it++;
      std::cout << "Nonlinear iter = " << it << ", dx = (" << dX[0] << "," << dX[1] << ");" << std::endl;
    }


    if(ISNAN(X[0]) || ISNAN(X[1]))
      std::cerr << "NAN value for n = " << n << ". c = " << X[0] << ", psi = " << X[1] << std::endl;
    if(ISINF(X[0]) || ISINF(X[1]))
      std::cerr << "INF value for n = " << n << ". c = " << X[0] << ", psi = " << X[1] << std::endl;

    con_np1_p[n] = X[0];
    psi_np1_p[n] = X[1];
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(con, &con_p);   CHKERRXX(ierr);
  ierr = VecRestoreArray(con_np1, &con_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi_np1, &psi_np1_p); CHKERRXX(ierr);

  // swap pointers and release memory
  ierr = VecDestroy(psi); CHKERRXX(ierr); psi = psi_np1; psi_interp.set_input_parameters(psi, linear);
  ierr = VecDestroy(con); CHKERRXX(ierr); con = con_np1; con_interp.set_input_parameters(con, linear);

  // Extend solution
  parStopWatch w;
  w.start("Extending solutions");
  my_p4est_level_set ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, psi, 20, 2);
  ls.extend_Over_Interface_TVD(phi, con, 20, 2);
  w.stop(); w.read_duration();

  // update ghost values -- NOTE: This can be done more efficiently by overlapping
  ierr = VecGhostUpdateBegin(psi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(psi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(con, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(con, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

void ExplicitNonLinearChargingSolver::solve()
{
  parStopWatch w;

  // first solve the nonlinear equations to compute the boundary conditions
  w.start("nonlinear solve");
  nonlinear_solve_decoupled(1);
  w.stop(); w.read_duration();

  write_vtk("internal");

  w.start("Solving for the concentration");
  double *con_p, *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArray(con, &con_p); CHKERRXX(ierr);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    rhs_p[i] = con_p[i] / dt;
  ierr = VecRestoreArray(con, &con_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  con_solver.set_diagonal(1.0/dt);
  con_solver.set_rhs(rhs);
  con_solver.solve(con);
  w.stop(); w.read_duration();

  // solve for the potential
  w.start("Solving for the potential");
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  for (size_t i=0; i<nodes->indep_nodes.elem_count; i++)
    rhs_p[i] = 0;
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  psi_solver.set_rhs(rhs);
  psi_solver.set_mu(con);
  psi_solver.solve(psi);
  w.stop(); w.read_duration();

  w.start("Extending solutions");
  my_p4est_level_set ls(ngbd);
  ls.extend_Over_Interface_TVD(phi, psi, 20, 2);
  ls.extend_Over_Interface_TVD(phi, con, 20, 2);
  w.stop(); w.read_duration();
}

void ExplicitNonLinearChargingSolver::write_vtk(const std::string& filename){
  double *phi_p, *psi_p, *con_p;

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecGetArray(con, &con_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         3, 0, filename.c_str(),
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "psi", psi_p,
                         VTK_POINT_DATA, "con", con_p);

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(psi, &psi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(con, &con_p); CHKERRXX(ierr);
}
